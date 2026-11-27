from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from app.hitl.messages import UNSUPPORTED_REQUEST_ANSWER
from app.llm.semantic_reasoner import is_plain_text_answer


RouteName = Literal[
    "out_of_scope",
    "feedback",
    "retrieve_similar",
    "phase_prediction",
    "explain_state",
    "memory_followup",
    "compound_request",
]
RequestMode = Literal["single", "compound"]
PlanRoute = Literal["execute", "out_of_scope"]
FollowupSource = Literal["current_turn", "session_memory"]
CompoundTask = Literal[
    "retrieve_similar",
    "phase_prediction",
    "explain_state",
    "followup_analysis",
]
FollowupOperation = Literal[
    "closest_window",
    "farthest_window",
    "explain_retrieval",
    "phase_rationale",
    "phase_evidence_strength",
]


class HITLAgentState(TypedDict, total=False):
    question: str
    effective_question: str
    thread_id: str
    feedback_action: str | None
    feedback_label: str
    feedback_note: str
    reviewer: str
    memory: dict[str, Any]
    rewrite: dict[str, Any]
    followup_source: FollowupSource
    request_mode: RequestMode
    plan_route: PlanRoute
    task_plan: list[CompoundTask]
    pending_tasks: list[CompoundTask]
    completed_tasks: list[CompoundTask]
    compound_results: dict[str, dict[str, Any]]
    followup_operations: list[FollowupOperation]
    target_predicted_numeric_window: bool
    target_window_operation: str
    filter_decision: Any
    route: RouteName
    router: dict[str, Any]
    result: dict[str, Any]
    critic_warnings: list[str]
    trace: list[str]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    return value


class HITLSessionMemoryStore:
    """Small JSON-backed session memory keyed by LangGraph thread_id."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def get(self, thread_id: str | None) -> dict[str, Any]:
        if not thread_id:
            return {}
        payload = self._load_all().get(str(thread_id), {})
        return payload if isinstance(payload, dict) else {}

    def save(self, thread_id: str | None, memory: dict[str, Any]) -> None:
        if not thread_id:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_all()
        payload[str(thread_id)] = _json_safe(memory)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class ControlledHITLAgentGraph:
    """Small LangGraph wrapper around deterministic HITL tools.

    The graph is intentionally controlled: LangGraph coordinates fixed tools,
    while the model/retrieval/phase logic remains deterministic and auditable.
    """

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        # The deployment adapter gives each signed session its own feedback directory.
        memory_path = pipeline.feedback_store.path.parent / "session_memory.json"
        self.memory_store = HITLSessionMemoryStore(memory_path)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(HITLAgentState)
        builder.add_node("load_session_memory", self._load_session_memory)
        builder.add_node("rewrite_request", self._rewrite_request)
        builder.add_node("plan_request", self._plan_request)
        builder.add_node("task_controller", self._task_controller)
        builder.add_node("controlled_tool_node", self._controlled_tool_node)
        builder.add_node("synthesize_response", self._synthesize_response)
        builder.add_node("critic_check", self._critic_check)
        builder.add_node("finalize", self._finalize)

        builder.set_entry_point("load_session_memory")
        builder.add_edge("load_session_memory", "rewrite_request")
        builder.add_edge("rewrite_request", "plan_request")
        builder.add_conditional_edges(
            "plan_request",
            self._route_after_plan,
            {
                "execute": "task_controller",
                "out_of_scope": "finalize",
            },
        )
        builder.add_conditional_edges(
            "task_controller",
            self._route_after_task_controller,
            {
                "run_tool": "controlled_tool_node",
                "synthesize": "synthesize_response",
            },
        )
        builder.add_edge("controlled_tool_node", "task_controller")
        builder.add_edge("synthesize_response", "critic_check")
        builder.add_edge("critic_check", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def _append_trace(state: HITLAgentState, item: str) -> list[str]:
        return [*(state.get("trace") or []), item]

    def _base_result(self, state: HITLAgentState) -> dict[str, Any]:
        decision = state.get("filter_decision")
        if hasattr(decision, "to_dict"):
            filter_payload = decision.to_dict()
        elif isinstance(decision, dict):
            filter_payload = decision
        else:
            filter_payload = {
                "intent": state.get("route", "out_of_scope"),
                "confidence": 1.0 if state.get("route") == "memory_followup" else 0.0,
                "allowed": state.get("route") != "out_of_scope",
                "reason": "Session-memory follow-up." if state.get("route") == "memory_followup" else "No route.",
                "matched_terms": [],
            }
        router = state.get("router", {})
        if isinstance(router, dict) and router.get("source") == "llm_fallback":
            filter_payload = {
                **filter_payload,
                "intent": state.get("route", filter_payload.get("intent")),
                "confidence": router.get("confidence", filter_payload.get("confidence")),
                "allowed": state.get("route") != "out_of_scope",
                "reason": f"Hybrid router used LLM fallback: {router.get('reason', '')}".strip(),
                "router_source": "llm_fallback",
            }
        return {
            "filter": filter_payload,
            "run_name": self.pipeline.context.metadata.get("run_name"),
            "window_id": self.pipeline.context.resolved_window_id,
            "rag_enabled": self.pipeline.enable_rag,
            "llm_requested": self.pipeline.enable_llm,
            "llm_available": self.pipeline.llm_available,
            "thread_id": state.get("thread_id"),
            "router": state.get("router", {}),
        }

    def _load_session_memory(self, state: HITLAgentState) -> HITLAgentState:
        thread_id = state.get("thread_id")
        memory = self.memory_store.get(thread_id)
        trace_item = "load_session_memory:hit" if memory.get("last_tool_result") else "load_session_memory:empty"
        return {**state, "memory": memory, "trace": self._append_trace(state, trace_item)}

    def _extract_previous_phase(self, memory: dict[str, Any]) -> dict[str, Any]:
        last_phase = memory.get("last_phase_result", {}) if isinstance(memory, dict) else {}
        last = memory.get("last_tool_result", {}) if isinstance(memory, dict) else {}
        phase = last_phase.get("phase_evidence", {}) if isinstance(last_phase, dict) else {}
        if not phase:
            phase = last.get("phase_evidence", {}) if isinstance(last, dict) else {}
        return phase if isinstance(phase, dict) else {}

    def _rewrite_request(self, state: HITLAgentState) -> HITLAgentState:
        question = state["question"]
        lowered = question.lower()
        memory = state.get("memory") or {}
        phase = self._extract_previous_phase(memory)

        has_memory = bool(memory.get("last_tool_result"))
        has_numeric_memory = False
        reference_terms = {
            "that",
            "this",
            "it",
            "last",
            "previous",
            "prior",
            "earlier",
            "predicted",
            "forecasted",
            "forecast",
            "prediction",
            "result",
            "answer",
            "evidence",
            "predict",
            "selected",
            "chosen",
            "choose",
            "chose",
            "token",
            "phase",
            "candidate",
            "candidates",
            "alternative",
            "alternatives",
            "window",
            "pattern",
        }
        action_terms = {"retrieve", "similar", "history", "historical", "past", "show", "find", "windows", "cases"}
        retrieval_terms = {"retrieve", "similar", "history", "historical", "past", "nearest", "match", "matches", "windows", "analogs", "analogues"}
        predicted_window_terms = {"predicted", "forecasted", "forecast", "future"}
        clarification_terms = {
            "why",
            "confidence",
            "confident",
            "support",
            "probability",
            "explain",
            "mean",
            "meaning",
            "alternative",
            "candidate",
            "why not",
            "summarize",
            "summary",
            "recap",
            "compare",
            "reliable",
            "strong",
        }
        words = set(re.findall(r"[a-z0-9_]+", lowered))
        refers_to_memory = bool(words & reference_terms) or "why not" in lowered
        wants_action = bool(words & action_terms)
        wants_clarification = bool(words & clarification_terms)
        memory_clarification_topic = bool(
            words
            & {
                "probability",
                "support",
                "confidence",
                "candidate",
                "candidates",
                "alternative",
                "alternatives",
                "selected",
                "chosen",
                "token",
                "phase",
                "evidence",
                "summary",
                "summarize",
                "recap",
            }
        )
        wants_retrieval_tool = bool(words & retrieval_terms)
        direct_retrieval_request = bool(words & {"retrieve", "find", "show"}) and bool(
            words & {"similar", "historical", "history", "past", "nearest", "analogs", "analogues", "matches", "windows"}
        )
        direct_current_state_request = bool(words & {"explain", "describe"}) and bool(
            words & {"current", "live"}
        ) and bool(
            words & {"state", "window", "regime", "condition", "conditions", "compressed"}
        )
        direct_phase_request = bool(words & {"predict", "forecast"}) and bool(words & {"next"}) and bool(
            words & {"phase", "token", "regime", "state"}
        )
        direct_feedback_request = bool(words & {"accept", "approve", "flag", "wrong", "incorrect", "reject", "note"})
        direct_core_request = direct_phase_request or direct_current_state_request or direct_retrieval_request or direct_feedback_request
        explicit_previous_context = bool(words & {"previous", "prior", "last", "earlier", "predicted", "forecasted"})
        conversation_relevant = bool(
            refers_to_memory
            or wants_clarification
            or memory_clarification_topic
            or words
            & {
                "wind",
                "speed",
                "maximum",
                "minimum",
                "mean",
                "std",
                "ramp",
                "gust",
                "direction",
                "current",
                "live",
                "state",
                "window",
                "regime",
                "token",
                "phase",
                "candidate",
                "probability",
                "support",
                "evidence",
                "forecast",
                "prediction",
            }
        )
        should_use_memory_rag = (
            has_memory
            and conversation_relevant
            and not direct_core_request
        )

        rewrite = {
            "original_question": question,
            "rewritten_question": question,
            "execution_mode": "normal_tool_routing",
            "reason": "No previous-result rewrite was needed.",
        }
        update: dict[str, Any] = {
            "effective_question": question,
            "followup_source": "current_turn",
            "rewrite": rewrite,
        }

        wants_predicted_numeric_window = (
            has_numeric_memory
            and bool(words & predicted_window_terms)
            and bool(words & {"window", "pattern", "numeric", "values", "speed"})
        )

        if wants_predicted_numeric_window and wants_action:
            rewritten = "Retrieve historical windows similar to the previous predicted numeric wind-speed window."
            rewrite = {
                "original_question": question,
                "rewritten_question": rewritten,
                "execution_mode": "normal_tool_routing",
                "reason": "Resolved follow-up reference to the predicted numeric window sketch.",
                "resolved_artifact": "last_numeric_result.predicted_window_sketch",
            }
            update.update(
                {
                    "effective_question": rewritten,
                    "rewrite": rewrite,
                    "target_predicted_numeric_window": True,
                    "target_window_operation": "retrieve_similar",
                }
            )
        elif wants_predicted_numeric_window and wants_clarification:
            rewritten = "Explain the previous predicted numeric wind-speed window."
            rewrite = {
                "original_question": question,
                "rewritten_question": rewritten,
                "execution_mode": "normal_tool_routing",
                "reason": "Resolved follow-up reference to the predicted numeric window sketch.",
                "resolved_artifact": "last_numeric_result.predicted_window_sketch",
            }
            update.update(
                {
                    "effective_question": rewritten,
                    "rewrite": rewrite,
                    "target_predicted_numeric_window": True,
                    "target_window_operation": "explain_window",
                }
            )
        elif should_use_memory_rag:
            rewritten = "Answer the follow-up using only relevant HITL session-memory evidence."
            rewrite = {
                "original_question": question,
                "rewritten_question": rewritten,
                "execution_mode": "controlled_followup",
                "reason": "Detected a conversation-relevant follow-up over stored HITL evidence.",
            }
            update.update(
                {
                    "effective_question": rewritten,
                    "followup_source": "session_memory",
                    "rewrite": rewrite,
                    "route": "memory_followup",
                }
            )

        return {
            **state,
            **update,
            "trace": self._append_trace(state, f"rewrite_request:{update['followup_source']}"),
        }

    @staticmethod
    def _detect_core_tasks(question: str) -> list[CompoundTask]:
        """Return explicit scientific core tasks in the order requested."""
        patterns: dict[CompoundTask, tuple[str, ...]] = {
            "phase_prediction": (
                r"\b(?:predict|forecast)\s+(?:the\s+)?next\s+(?:wind\s+)?(?:phase|regime|state)\b",
                r"\b(?:predict|forecast)\s+(?:the\s+)?(?:wind\s+)?(?:phase|regime)\b",
                r"\bwhat\s+(?:phase|regime)\s+(?:comes|is)\s+next\b",
            ),
            "explain_state": (
                r"\b(?:explain|describe|summari[sz]e)\s+(?:the\s+)?(?:current|live|present)\s+(?:wind\s+)?(?:state|condition|conditions|window|regime)\b",
                r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:current|live|present)\s+(?:wind\s+)?(?:state|condition|conditions|regime)\b",
                r"\bwhy\s+is\s+(?:the\s+)?(?:current|live|present)\s+(?:wind\s+)?(?:state|condition|regime)\b",
            ),
            "retrieve_similar": (
                r"\b(?:retrieve|find|show|get)\s+(?:the\s+)?(?:most\s+)?(?:similar|nearest|closest|matching)\s+(?:historical\s+|past\s+)?(?:window|windows|conditions|cases)\b",
                r"\b(?:retrieve|find|show|get)\s+(?:the\s+)?(?:historical|past)\s+(?:window|windows|conditions|cases|matches)\b",
                r"\b(?:similar|nearest|closest|matching)\s+(?:historical\s+|past\s+)?(?:window|windows|conditions|cases)\b",
            ),
        }
        lowered = question.lower()
        earliest: dict[CompoundTask, int] = {}
        for task, task_patterns in patterns.items():
            starts = [
                match.start()
                for pattern in task_patterns
                for match in re.finditer(pattern, lowered)
            ]
            if starts:
                earliest[task] = min(starts)
        ordered = [task for task, _ in sorted(earliest.items(), key=lambda item: item[1])]
        return ordered[:3]

    @classmethod
    def _detect_compound_tasks(cls, question: str) -> list[CompoundTask]:
        """Return multiple explicit core tasks; retained for focused tests."""
        tasks = cls._detect_core_tasks(question)
        return tasks if len(tasks) >= 2 else []

    @staticmethod
    def _detect_followup_operations(question: str) -> list[FollowupOperation]:
        """Detect analyses that depend on a forecast or retrieval result."""
        lowered = question.lower()
        operations: list[FollowupOperation] = []
        if re.search(r"\b(?:closest|nearest|most\s+similar)\b", lowered):
            operations.append("closest_window")
        if re.search(r"\b(?:farthest|furthest|least\s+similar|most\s+different)\b", lowered):
            operations.append("farthest_window")
        if re.search(
            r"\b(?:explain|describe|summari[sz]e)\s+(?:the\s+)?(?:retrieved|retrieval|historical|similar)\s+(?:data|windows?|results?)\b",
            lowered,
        ) or re.search(
            r"\b(?:explain|describe|summari[sz]e)\s+(?:the\s+)?data\s+in\s+(?:the\s+)?retrieved\s+(?:historical\s+)?windows?\b",
            lowered,
        ):
            operations.append("explain_retrieval")
        if re.search(r"\bwhy\b", lowered) and re.search(
            r"\b(?:phase|regime|candidate|selected|chosen|chose|alternative)\b",
            lowered,
        ):
            operations.append("phase_rationale")
        if re.search(r"\b(?:how\s+strong|confidence|supporting\s+evidence|evidence\s+strength)\b", lowered) and re.search(
            r"\b(?:phase|forecast|prediction|evidence)\b",
            lowered,
        ):
            operations.append("phase_evidence_strength")
        return operations

    def _plan_request(self, state: HITLAgentState) -> HITLAgentState:
        question = state.get("effective_question") or state["question"]
        if state.get("followup_source") == "session_memory":
            task_plan: list[CompoundTask] = ["followup_analysis"]
            filter_decision = {
                "intent": "memory_followup",
                "confidence": 1.0,
                "allowed": True,
                "reason": "Resolved the request as a follow-up over stored session evidence.",
                "matched_terms": ["session_memory"],
            }
            router = {
                "source": "session_memory_resolver",
                "intent": "memory_followup",
                "task_plan": task_plan,
                "llm_router_attempted": False,
                "llm_router_used": False,
                "reason": filter_decision["reason"],
            }
            return {
                **state,
                "request_mode": "single",
                "plan_route": "execute",
                "task_plan": task_plan,
                "pending_tasks": list(task_plan),
                "completed_tasks": [],
                "compound_results": {},
                "followup_operations": [],
                "filter_decision": filter_decision,
                "route": "memory_followup",
                "router": router,
                "trace": self._append_trace(state, "plan_request:single:followup_analysis"),
            }

        words = set(re.findall(r"[a-z0-9_]+", question.lower()))
        feedback_terms = {"accept", "approve", "flag", "reject", "incorrect", "wrong", "relabel"}
        task_plan: list[CompoundTask] = []
        followup_operations: list[FollowupOperation] = []
        if not state.get("feedback_action") and not words.intersection(feedback_terms):
            core_tasks = self._detect_core_tasks(question)
            detected_followups = self._detect_followup_operations(question)
            retrieval_followups = {
                "closest_window",
                "farthest_window",
                "explain_retrieval",
            }
            phase_followups = {"phase_rationale", "phase_evidence_strength"}
            followup_operations = [
                operation
                for operation in detected_followups
                if (
                    operation in retrieval_followups and "retrieve_similar" in core_tasks
                ) or (
                    operation in phase_followups and "phase_prediction" in core_tasks
                )
            ]
            if len(core_tasks) >= 2 or followup_operations:
                task_plan = list(core_tasks)
                if followup_operations:
                    task_plan.append("followup_analysis")

        if not task_plan:
            routed = self._relevance_check(state)
            route = routed.get("route", "out_of_scope")
            if route in {"feedback", "out_of_scope"}:
                return {
                    **routed,
                    "route": "out_of_scope",
                    "request_mode": "single",
                    "plan_route": "out_of_scope",
                    "task_plan": [],
                    "pending_tasks": [],
                    "completed_tasks": [],
                    "compound_results": {},
                    "followup_operations": [],
                    "trace": self._append_trace(routed, "plan_request:out_of_scope"),
                }
            task_plan = [route]
            return {
                **routed,
                "request_mode": "single",
                "plan_route": "execute",
                "task_plan": list(task_plan),
                "pending_tasks": list(task_plan),
                "completed_tasks": [],
                "compound_results": {},
                "followup_operations": [],
                "trace": self._append_trace(routed, f"plan_request:single:{route}"),
            }

        filter_decision = {
            "intent": "compound_request",
            "confidence": 1.0,
            "allowed": True,
            "reason": "Detected multiple explicit wind-review tasks in one request.",
            "matched_terms": list(task_plan),
        }
        router = {
            "source": "deterministic_compound_planner",
            "intent": "compound_request",
            "task_plan": list(task_plan),
            "llm_router_attempted": False,
            "llm_router_used": False,
            "reason": filter_decision["reason"],
        }
        return {
            **state,
            "request_mode": "compound",
            "plan_route": "execute",
            "task_plan": list(task_plan),
            "pending_tasks": list(task_plan),
            "completed_tasks": [],
            "compound_results": {},
            "followup_operations": followup_operations,
            "filter_decision": filter_decision,
            "route": "compound_request",
            "router": router,
            "trace": self._append_trace(state, f"plan_request:compound:{','.join(task_plan)}"),
        }

    @staticmethod
    def _route_after_plan(state: HITLAgentState) -> PlanRoute:
        return state.get("plan_route", "out_of_scope")

    def _task_controller(self, state: HITLAgentState) -> HITLAgentState:
        pending = list(state.get("pending_tasks") or [])
        next_task = pending[0] if pending else "done"
        return {
            **state,
            "trace": self._append_trace(state, f"task_controller:{next_task}"),
        }

    @staticmethod
    def _route_after_task_controller(state: HITLAgentState) -> Literal["run_tool", "synthesize"]:
        return "run_tool" if state.get("pending_tasks") else "synthesize"

    def _store_task_result(
        self,
        state: HITLAgentState,
        task: CompoundTask,
        result: dict[str, Any],
    ) -> HITLAgentState:
        pending = list(state.get("pending_tasks") or [])
        if not pending or pending[0] != task:
            raise RuntimeError(f"Task controller expected {task}, but pending tasks were {pending}.")
        pending.pop(0)
        compound_results = dict(state.get("compound_results") or {})
        compound_results[task] = result
        completed = [*(state.get("completed_tasks") or []), task]
        return {
            **state,
            "pending_tasks": pending,
            "completed_tasks": completed,
            "compound_results": compound_results,
            "trace": self._append_trace(state, f"controlled_tool_node:{task}"),
        }

    def _controlled_tool_node(self, state: HITLAgentState) -> HITLAgentState:
        pending = list(state.get("pending_tasks") or [])
        if not pending:
            return state
        task = pending[0]
        if task == "followup_analysis":
            if state.get("followup_source") == "session_memory":
                result = self._analyze_session_memory_followup(state)["result"]
            else:
                result = self._analyze_current_turn_followup(state)
            return self._store_task_result(state, task, result)

        tool_registry = {
            "phase_prediction": lambda: self.pipeline.predict_phase("Predict the next wind phase."),
            "retrieve_similar": lambda: self.pipeline.retrieve_similar(
                "Retrieve similar historical wind windows."
            ),
            "explain_state": lambda: self.pipeline.explain_state(
                "Explain the current wind state.",
                window_context="current_window",
            ),
        }
        runner = tool_registry.get(task)
        if runner is None:
            raise RuntimeError(f"No controlled tool is registered for task {task}.")

        # Tools collect structured evidence only. The final synthesis node makes
        # the request's single LLM call for both single and compound requests.
        llm_enabled = self.pipeline.enable_llm
        self.pipeline.enable_llm = False
        try:
            result = runner()
        finally:
            self.pipeline.enable_llm = llm_enabled
        return self._store_task_result(state, task, result)

    @staticmethod
    def _retrieval_rows_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = result.get("similar_windows")
        if not isinstance(rows, list):
            evidence_pack = result.get("evidence_pack")
            if not isinstance(evidence_pack, dict):
                evidence_pack = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
            rows = evidence_pack.get("similar_windows")
        return [row for row in (rows or []) if isinstance(row, dict)]

    @staticmethod
    def _phase_evidence_from_result(result: dict[str, Any]) -> dict[str, Any]:
        phase_payload = result.get("phase_prediction")
        if isinstance(phase_payload, dict) and isinstance(phase_payload.get("evidence"), dict):
            return phase_payload["evidence"]
        evidence_pack = result.get("evidence_pack")
        if not isinstance(evidence_pack, dict):
            evidence_pack = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        phase_evidence = evidence_pack.get("live_phase_prediction") or evidence_pack.get("phase_prediction") or {}
        return phase_evidence if isinstance(phase_evidence, dict) else {}

    @staticmethod
    def _compact_followup_window(row: dict[str, Any]) -> dict[str, Any]:
        return _json_safe({
            "evidence_id": row.get("evidence_id"),
            "window_id": row.get("window_id"),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "regime_name": row.get("regime_name"),
            "retrieval_distance": row.get("retrieval_distance"),
            "wind_speed_mean": row.get("wind_speed_mean"),
        })

    def _analyze_current_turn_followup(self, state: HITLAgentState) -> dict[str, Any]:
        operations = list(state.get("followup_operations") or [])
        results = dict(state.get("compound_results") or {})
        analysis: dict[str, Any] = {}
        sections: list[str] = []

        retrieval_operations = {
            "closest_window",
            "farthest_window",
            "explain_retrieval",
        }
        if retrieval_operations.intersection(operations):
            rows = self._retrieval_rows_from_result(results.get("retrieve_similar", {}))
            ranked: list[tuple[float, dict[str, Any]]] = []
            for row in rows:
                try:
                    ranked.append((float(row.get("retrieval_distance")), row))
                except (TypeError, ValueError):
                    continue
            comparison: dict[str, Any] = {"returned_windows": len(rows)}
            if ranked and "closest_window" in operations:
                closest = min(ranked, key=lambda item: item[0])[1]
                comparison["closest_window"] = self._compact_followup_window(closest)
                sections.append(
                    "The closest retrieved window is "
                    f"{closest.get('window_id') or 'the first ranked match'} "
                    f"(d={float(closest.get('retrieval_distance')):.4f})."
                )
            if ranked and "farthest_window" in operations:
                farthest = max(ranked, key=lambda item: item[0])[1]
                comparison["farthest_window"] = self._compact_followup_window(farthest)
                sections.append(
                    "The farthest window within the retrieved set is "
                    f"{farthest.get('window_id') or 'the last ranked match'} "
                    f"(d={float(farthest.get('retrieval_distance')):.4f})."
                )
            if ranked and {"closest_window", "farthest_window"}.intersection(operations):
                sections.append("A smaller retrieval distance d means a closer match in the stored representation.")
            if "explain_retrieval" in operations:
                sections.append(
                    f"The retrieval returned {len(rows)} historical window(s). Each row is a past observation "
                    "window ranked by its distance from the current wind representation."
                )
            analysis["retrieval_analysis"] = comparison

        phase_operations = {"phase_rationale", "phase_evidence_strength"}
        if phase_operations.intersection(operations):
            phase = self._phase_evidence_from_result(results.get("phase_prediction", {}))
            candidates = [
                row for row in (phase.get("candidate_next_phases") or [])
                if isinstance(row, dict)
            ]
            phase_analysis: dict[str, Any] = {
                "live_token_sequence": phase.get("live_token_sequence") or phase.get("token_sequence"),
                "support": phase.get("support"),
                "minimum_support": phase.get("minimum_support"),
                "candidate_next_phases": candidates[:5],
            }
            if candidates and "phase_rationale" in operations:
                top = candidates[0]
                probability = float(top.get("probability") or 0.0)
                count = top.get("count")
                support = top.get("support") or phase.get("support")
                evidence_text = (
                    f" It appeared in {count} of {support} matching transitions."
                    if count is not None and support
                    else ""
                )
                sections.append(
                    f"The forecast ranked token {top.get('token_id')} ({top.get('regime_name')}) first because "
                    f"it had the highest stored transition probability ({probability:.3f}).{evidence_text}"
                )
                alternatives = candidates[1:3]
                if alternatives:
                    sections.append(
                        "The next alternatives were "
                        + ", ".join(
                            f"token {row.get('token_id')} ({float(row.get('probability') or 0.0):.3f})"
                            for row in alternatives
                        )
                        + "."
                    )
            if "phase_evidence_strength" in operations:
                support = int(phase.get("support") or 0)
                minimum = int(phase.get("minimum_support") or 0)
                if minimum:
                    comparison = "meets" if support >= minimum else "does not meet"
                    sections.append(
                        f"The forecast has support from {support} matching transition(s), which {comparison} "
                        f"the configured minimum of {minimum}."
                    )
                else:
                    sections.append(f"The forecast has support from {support} matching transition(s).")
            analysis["phase_analysis"] = phase_analysis

        answer = " ".join(sections).strip() or UNSUPPORTED_REQUEST_ANSWER
        return {
            "mode": "followup_analysis",
            "explanation_mode": "working_memory_analysis",
            "answer": answer,
            "followup_operations": operations,
            "followup_analysis": analysis,
            "evidence": [f"current_turn:{task}" for task in state.get("completed_tasks", [])],
            "evidence_pack": {"followup_analysis": analysis},
        }

    @staticmethod
    def _compact_compound_result(result: dict[str, Any]) -> dict[str, Any]:
        evidence_pack = result.get("evidence_pack")
        if not isinstance(evidence_pack, dict):
            evidence_pack = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
        phase_payload = result.get("phase_prediction") if isinstance(result.get("phase_prediction"), dict) else {}
        phase_evidence = phase_payload.get("evidence") if isinstance(phase_payload.get("evidence"), dict) else {}
        compact: dict[str, Any] = {
            "mode": result.get("mode"),
            "answer": result.get("answer"),
        }
        if phase_evidence:
            compact["phase_prediction"] = {
                "live_token_sequence": phase_evidence.get("live_token_sequence"),
                "current_live_state": phase_evidence.get("current_live_state"),
                "candidate_next_phases": (phase_evidence.get("candidate_next_phases") or [])[:5],
                "support": phase_evidence.get("support"),
                "minimum_support": phase_evidence.get("minimum_support"),
                "similar_transition_analogs": (phase_evidence.get("similar_transition_analogs") or [])[:5],
            }
        current_state = evidence_pack.get("current_state")
        if isinstance(current_state, dict) and current_state:
            compact["current_state"] = current_state
        similar = result.get("similar_windows") or evidence_pack.get("similar_windows") or []
        if isinstance(similar, list) and similar:
            compact["similar_windows"] = similar[:5]
        followup_analysis = result.get("followup_analysis")
        if isinstance(followup_analysis, dict) and followup_analysis:
            compact["followup_analysis"] = followup_analysis
        return _json_safe(compact)

    def _llm_compound_answer(
        self,
        question: str,
        task_plan: list[CompoundTask],
        compact_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.pipeline.enable_llm or not self.pipeline.llm_available or not self.pipeline.reasoner:
            return None
        evidence_ids = [f"compound:{task}" for task in task_plan]
        prompt = (
            "You are the final answer node in a controlled wind-review workflow.\n"
            "Several approved tools have already run for one user message. Answer every requested part using only "
            "the supplied tool_results. Do not claim that any other tool ran. Do not invent wind values, phases, "
            "probabilities, or historical matches. Keep the answer clear and organized in the same language requested "
            "by the user.\n"
            f"If the evidence cannot answer the request, set answer to exactly: {UNSUPPORTED_REQUEST_ANSWER}\n"
            "Return only valid JSON with keys: answer, evidence, human_review_prompt. answer must be a nonempty string. "
            "evidence must be a list containing only the supplied evidence IDs.\n\n"
            f"User question: {question}\n"
            f"Task order: {json.dumps(task_plan, ensure_ascii=False)}\n"
            f"Evidence IDs: {json.dumps(evidence_ids, ensure_ascii=False)}\n"
            f"Tool results: {json.dumps(_json_safe(compact_results), ensure_ascii=False)}\n"
        )
        if hasattr(self.pipeline.reasoner, "invoke_json_with_raw"):
            payload, raw_text = self.pipeline.reasoner.invoke_json_with_raw(prompt)
        else:
            payload = self.pipeline.reasoner.invoke_json(prompt)
            raw_text = None
        if not isinstance(payload, dict):
            if is_plain_text_answer(raw_text):
                return {
                    "answer": str(raw_text).strip(),
                    "evidence": evidence_ids,
                    "human_review_prompt": "Check that this combined answer matches each tool result.",
                    "llm_output_format": "raw_text",
                }
            return None
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None
        valid_ids = set(evidence_ids)
        evidence = payload.get("evidence")
        if not isinstance(evidence, list):
            evidence = evidence_ids
        evidence = [str(item) for item in evidence if str(item) in valid_ids] or evidence_ids
        return {
            "answer": answer.strip(),
            "evidence": evidence,
            "human_review_prompt": str(
                payload.get("human_review_prompt", "Check that this combined answer matches each tool result.")
            ),
            "llm_output_format": "json",
        }

    @staticmethod
    def _deterministic_compound_answer(
        task_plan: list[CompoundTask],
        results: dict[str, dict[str, Any]],
    ) -> str:
        labels = {
            "phase_prediction": "Next phase forecast",
            "explain_state": "Current state",
            "retrieve_similar": "Similar historical windows",
            "followup_analysis": "Follow-up analysis",
        }
        sections = []
        for task in task_plan:
            answer = results.get(task, {}).get("answer")
            if isinstance(answer, str) and answer.strip():
                sections.append(f"{labels[task]}: {answer.strip()}")
        return "\n\n".join(sections) if sections else UNSUPPORTED_REQUEST_ANSWER

    def _synthesize_response(self, state: HITLAgentState) -> HITLAgentState:
        task_plan = list(state.get("task_plan") or [])
        raw_results = dict(state.get("compound_results") or {})
        if state.get("route") == "memory_followup":
            result = dict(raw_results.get("followup_analysis") or {})
            result.update({
                "task_plan": task_plan,
                "completed_tasks": list(state.get("completed_tasks") or []),
            })
            return {
                **state,
                "result": result,
                "trace": self._append_trace(state, "response:synthesize:memory_followup"),
            }

        compact_results = {
            task: self._compact_compound_result(raw_results.get(task, {}))
            for task in task_plan
        }
        llm_payload = self._llm_compound_answer(state["question"], task_plan, compact_results)
        if llm_payload:
            answer = llm_payload["answer"]
        elif state.get("request_mode") == "single" and task_plan:
            answer = raw_results.get(task_plan[0], {}).get("answer") or UNSUPPORTED_REQUEST_ANSWER
        else:
            answer = self._deterministic_compound_answer(task_plan, raw_results)
        evidence_pack: dict[str, Any] = {
            "compound_tasks": task_plan,
            "compound_results": compact_results,
        }
        phase_result = raw_results.get("phase_prediction", {})
        phase_payload = phase_result.get("phase_prediction") if isinstance(phase_result, dict) else None
        phase_evidence = phase_payload.get("evidence", {}) if isinstance(phase_payload, dict) else {}
        if phase_evidence:
            evidence_pack["live_phase_prediction"] = phase_evidence
        state_result = raw_results.get("explain_state", {})
        state_evidence = state_result.get("evidence_pack") if isinstance(state_result, dict) else {}
        if not isinstance(state_evidence, dict):
            state_evidence = state_result.get("evidence") if isinstance(state_result.get("evidence"), dict) else {}
        if isinstance(state_evidence.get("current_state"), dict):
            evidence_pack["current_state"] = state_evidence["current_state"]
        retrieval_result = raw_results.get("retrieve_similar", {})
        similar_windows = retrieval_result.get("similar_windows", []) if isinstance(retrieval_result, dict) else []
        if similar_windows:
            evidence_pack["similar_windows"] = similar_windows
        followup_result = raw_results.get("followup_analysis", {})
        followup_analysis = (
            followup_result.get("followup_analysis")
            if isinstance(followup_result, dict)
            else None
        )
        if isinstance(followup_analysis, dict) and followup_analysis:
            evidence_pack["followup_analysis"] = followup_analysis

        if state.get("request_mode") == "single" and task_plan:
            task = task_plan[0]
            result = dict(raw_results.get(task, {}))
            if llm_payload:
                result["mode"] = f"llm_{task}"
                result["llm_evidence"] = llm_payload.get("evidence", [])
            result.update({
                "explanation_mode": "llm_rag" if llm_payload else "heuristic",
                "answer": answer,
                "task_plan": task_plan,
                "completed_tasks": list(state.get("completed_tasks") or []),
                "human_review_prompt": (
                    llm_payload.get("human_review_prompt")
                    if llm_payload
                    else result.get("human_review_prompt", "Check this answer against the tool evidence.")
                ),
                "task_debug": {
                    "llm_attempted": bool(
                        self.pipeline.enable_llm and self.pipeline.llm_available and self.pipeline.reasoner
                    ),
                    "llm_used": bool(llm_payload),
                    "llm_output_format": llm_payload.get("llm_output_format") if llm_payload else None,
                },
            })
            return {
                **state,
                "result": result,
                "trace": self._append_trace(state, "response:synthesize:single"),
            }

        result: dict[str, Any] = {
            "mode": "compound_request",
            "explanation_mode": "llm_rag" if llm_payload else "heuristic",
            "answer": answer,
            "compound_tasks": task_plan,
            "compound_results": compact_results,
            "followup_operations": list(state.get("followup_operations") or []),
            "evidence": llm_payload.get("evidence", []) if llm_payload else [f"compound:{task}" for task in task_plan],
            "evidence_pack": evidence_pack,
            "human_review_prompt": (
                llm_payload.get("human_review_prompt")
                if llm_payload
                else "Check each section against the corresponding tool result."
            ),
            "compound_debug": {
                "llm_attempted": bool(
                    self.pipeline.enable_llm and self.pipeline.llm_available and self.pipeline.reasoner
                ),
                "llm_used": bool(llm_payload),
                "llm_output_format": llm_payload.get("llm_output_format") if llm_payload else None,
                "completed_tasks": list(state.get("completed_tasks") or []),
            },
        }
        if isinstance(phase_payload, dict):
            result["phase_prediction"] = phase_payload
        if isinstance(similar_windows, list):
            result["similar_windows"] = similar_windows
        if isinstance(followup_analysis, dict):
            result["followup_analysis"] = followup_analysis
        return {
            **state,
            "result": result,
            "trace": self._append_trace(state, "response:synthesize:compound"),
        }

    def _analyze_session_memory_followup(self, state: HITLAgentState) -> HITLAgentState:
        memory = state.get("memory") or {}
        selected_chunks = self._select_memory_chunks(state["question"], memory)
        payload = {
            "current_question": state["question"],
            "selected_memory": selected_chunks,
            "memory_summary": {
                "stored_turns": len(memory.get("turns") or []) if isinstance(memory, dict) else 0,
                "has_last_phase_result": isinstance(memory.get("last_phase_result"), dict),
                "has_last_retrieval_result": isinstance(memory.get("last_retrieval_result"), dict),
            },
            "grounding_rules": [
                "Use only the selected HITL session-memory evidence.",
                "Do not call forecasting, retrieval, or external tools.",
                "Do not introduce new probabilities, tokens, or weather facts that are not in memory.",
                "If the previous memory is insufficient, say so.",
            ],
        }
        llm_blockers: list[str] = []
        if not selected_chunks:
            llm_blockers.append("no selected memory chunks")
        if not self.pipeline.enable_llm:
            llm_blockers.append("LLM checkbox/request is disabled")
        if not self.pipeline.llm_available:
            llm_blockers.append("LLM client is unavailable")
        if not self.pipeline.reasoner:
            llm_blockers.append("reasoner object is missing")
        llm_attempted = bool(selected_chunks and self.pipeline.enable_llm and self.pipeline.llm_available and self.pipeline.reasoner)
        llm_error = None
        try:
            llm_payload = self._llm_memory_rag(state["question"], payload) if llm_attempted else None
        except Exception:
            llm_payload = None
            # Never put raw provider exceptions (which can contain secrets) in a response.
            llm_error = {"category": "memory_answer_error", "retryable": False}
        if llm_attempted and not llm_payload and llm_error is None:
            llm_error = getattr(self.pipeline.reasoner, "last_error", None)
            if not isinstance(llm_error, dict):
                llm_error = {"category": "invalid_answer", "retryable": False}
        memory_rag_debug = {
            "selected_chunk_ids": [str(chunk.get("evidence_id")) for chunk in selected_chunks],
            "selected_chunk_kinds": [str(chunk.get("kind")) for chunk in selected_chunks],
            "llm_blockers": llm_blockers,
            "llm_attempted": llm_attempted,
            "llm_used": bool(llm_payload),
            "llm_output_format": llm_payload.get("llm_output_format") if isinstance(llm_payload, dict) else None,
            "llm_error": llm_error,
            "llm_fallback_reason": None
            if llm_payload
            else (
                f"LLM was not attempted: {', '.join(llm_blockers)}."
                if not llm_attempted
                else f"LLM explanation unavailable: {(llm_error or {}).get('category', 'invalid_answer')}."
            ),
        }
        if llm_payload:
            result = {
                "mode": "llm_memory_rag_followup",
                **llm_payload,
                "evidence_pack": payload,
                "memory_rag_debug": memory_rag_debug,
            }
        else:
            result = {
                "mode": "memory_rag_unavailable" if selected_chunks else "memory_rag_missing_context",
                "answer": UNSUPPORTED_REQUEST_ANSWER,
                "evidence": [],
                "evidence_pack": payload,
                "memory_rag_debug": memory_rag_debug,
            }
        return {
            **state,
            "route": "memory_followup",
            "result": result,
            "trace": self._append_trace(state, "memory:answer_from_previous_result"),
        }

    @staticmethod
    def _select_memory_result_for_question(question: str, memory: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(memory, dict):
            return {}
        lowered = question.lower()
        phase_terms = {
            "forecast",
            "predict",
            "predicted",
            "phase",
            "token",
            "probability",
            "support",
            "candidate",
            "candidates",
            "alternative",
            "alternatives",
            "confidence",
            "selected",
            "chosen",
            "why",
        }
        words = set(re.findall(r"[a-z0-9_]+", lowered))
        if words & phase_terms and isinstance(memory.get("last_phase_result"), dict):
            return memory["last_phase_result"]
        return memory.get("last_tool_result", {}) if isinstance(memory.get("last_tool_result"), dict) else {}

    @staticmethod
    def _memory_terms(value: Any) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", json.dumps(_json_safe(value), ensure_ascii=False).lower()))

    @staticmethod
    def _candidate_text(row: dict[str, Any]) -> str:
        token_id = row.get("token_id")
        name = row.get("regime_name") or f"token {token_id}"
        probability = row.get("probability")
        count = row.get("count")
        support = row.get("support")
        if count is not None and support:
            evidence = f"count {count} out of support {support}"
        elif count is not None:
            evidence = f"count {count}"
        else:
            evidence = "count unavailable"
        try:
            probability_text = f"{float(probability):.4f}"
        except Exception:
            probability_text = str(probability)
        return f"token {token_id}: {name}; probability {probability_text}; {evidence}"

    def _memory_chunks(self, memory: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(memory, dict):
            return []

        chunks: list[dict[str, Any]] = []

        turns = memory.get("turns") or []
        if isinstance(turns, list) and turns:
            recent_turns = turns[-5:]
            chunks.append(
                {
                    "evidence_id": "memory:recent_turns",
                    "kind": "conversation_turns",
                    "title": "Recent HITL conversation turns",
                    "text": "Recent questions and answers in this HITL thread.",
                    "payload": recent_turns,
                }
            )

        last_review = memory.get("last_review") if isinstance(memory.get("last_review"), dict) else {}
        if last_review:
            action = last_review.get("action")
            if action == "reject":
                action = "flag"
            elif action == "relabel":
                action = "note"
            model_token = last_review.get("model_predicted_token")
            if action == "flag":
                text = (
                    f"Human review flagged the previous forecast for model token {model_token}. "
                    f"Note: {last_review.get('note', '')}"
                )
            elif action == "accept":
                text = (
                    f"Human review accepted the previous forecast for model token {model_token}. "
                    f"Note: {last_review.get('note', '')}"
                )
            else:
                text = f"Human review note for the previous HITL result. Note: {last_review.get('note', '')}"
            chunks.append(
                {
                    "evidence_id": "memory:last_review",
                    "kind": "human_review",
                    "title": "Stored human review state",
                    "text": text,
                    "payload": last_review,
                }
            )

        def add_result_chunks(source_key: str, result: dict[str, Any]) -> None:
            if not isinstance(result, dict):
                return

            answer = result.get("answer")
            if answer:
                chunks.append(
                    {
                        "evidence_id": f"memory:{source_key}:answer",
                        "kind": "stored_answer",
                        "title": f"Stored answer from {source_key}",
                        "text": str(answer),
                        "payload": {"answer": answer, "mode": result.get("mode"), "window_id": result.get("window_id")},
                    }
                )

            phase = result.get("phase_evidence", {}) if isinstance(result.get("phase_evidence"), dict) else {}
            candidates = phase.get("candidate_next_phases", []) if isinstance(phase, dict) else []
            if candidates:
                sequence = phase.get("live_token_sequence") or phase.get("token_sequence") or []
                current = phase.get("current_live_state") or {}
                top = candidates[0]
                total_support = top.get("support") or phase.get("support")
                if not total_support:
                    counts = [row.get("count") for row in candidates if row.get("count") is not None]
                    total_support = sum(int(value) for value in counts) if counts else None
                summary_text = (
                    f"Previous phase forecast used token sequence {sequence}. "
                    f"Top candidate was {self._candidate_text(top)}. "
                    f"Total support: {total_support if total_support is not None else 'unavailable'}. "
                    f"Current state: {current}."
                )
                chunks.append(
                    {
                        "evidence_id": f"memory:{source_key}:phase_forecast",
                        "kind": "phase_forecast",
                        "title": "Stored phase forecast evidence",
                        "text": summary_text,
                        "payload": {
                            "live_token_sequence": sequence,
                            "current_live_state": current,
                            "candidate_next_phases": candidates,
                            "support": total_support,
                        },
                    }
                )
                for row in candidates[:5]:
                    token_id = row.get("token_id")
                    chunks.append(
                        {
                            "evidence_id": f"memory:{source_key}:candidate_token_{token_id}",
                            "kind": "phase_candidate",
                            "title": f"Stored candidate token {token_id}",
                            "text": self._candidate_text(row),
                            "payload": row,
                        }
                    )

            current_state = result.get("current_state")
            if isinstance(current_state, dict) and current_state:
                chunks.append(
                    {
                        "evidence_id": f"memory:{source_key}:current_state",
                        "kind": "current_state",
                        "title": "Stored current semantic state",
                        "text": f"Current semantic state from stored HITL result: {current_state}",
                        "payload": current_state,
                    }
                )

            similar = result.get("similar_windows") or []
            if isinstance(similar, list) and similar:
                chunks.append(
                    {
                        "evidence_id": f"memory:{source_key}:similar_windows",
                        "kind": "similar_windows",
                        "title": "Stored similar historical windows",
                        "text": f"Stored {len(similar)} similar historical window(s) from the previous retrieval.",
                        "payload": similar[:5],
                    }
                )

        for key in ("last_phase_result", "last_retrieval_result", "last_tool_result"):
            add_result_chunks(key, memory.get(key, {}) if isinstance(memory.get(key), dict) else {})

        # Several memory slots can point to the same result. Send each piece of
        # evidence once rather than repeating its table/answer under three IDs.
        if any(chunk.get("kind") == "similar_windows" for chunk in chunks):
            chunks.append({
                "evidence_id": "memory:retrieval_distance_definition",
                "kind": "retrieval_semantics",
                "title": "Meaning of d / retrieval distance in historical windows",
                "text": (
                    "d (retrieval_distance) compares the query window's numerical wind-pattern "
                    "embedding with a historical window's embedding. Smaller means a closer match; "
                    "larger means less similar within the returned set. It is not a distance in meters, "
                    "wind speed, probability, or a difference between token IDs or token sequences. "
                    "The snapshot live-window retriever uses cosine distance between embeddings."
                ),
                "payload": {"field": "retrieval_distance", "smaller_is_closer": True},
            })
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in chunks:
            signature = json.dumps(_json_safe({"kind": chunk.get("kind"), "payload": chunk.get("payload")}),
                                   sort_keys=True, ensure_ascii=False)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(chunk)
        return deduped

    def _select_memory_chunks(self, question: str, memory: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        chunks = self._memory_chunks(memory)
        if not chunks:
            return []

        q_terms = set(re.findall(r"[a-z0-9_]+", question.lower()))
        token_mentions = set(int(value) for value in re.findall(r"token\s*([0-9]+)", question.lower()))
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            terms = self._memory_terms({"title": chunk.get("title"), "text": chunk.get("text"), "payload": chunk.get("payload")})
            score = len(q_terms & terms)
            payload = chunk.get("payload")
            if token_mentions:
                token_values: set[int] = set()
                if isinstance(payload, dict) and payload.get("token_id") is not None:
                    token_values.add(int(payload.get("token_id")))
                if isinstance(payload, dict):
                    for row in payload.get("candidate_next_phases", []) or []:
                        if isinstance(row, dict) and row.get("token_id") is not None:
                            token_values.add(int(row.get("token_id")))
                if token_mentions & token_values:
                    score += 8
            if chunk.get("kind") == "phase_forecast" and q_terms & {"why", "confidence", "probability", "support", "selected", "chosen", "strong", "reliable"}:
                score += 4
            if chunk.get("kind") == "phase_candidate" and q_terms & {"candidate", "alternative", "alternatives", "why", "token"}:
                score += 3
            if chunk.get("kind") == "similar_windows" and q_terms & {"similar", "historical", "history", "windows", "retrieved"}:
                score += 4
            if chunk.get("kind") == "human_review" and q_terms & {
                "review",
                "human",
                "trust",
                "accept",
                "accepted",
                "flag",
                "flagged",
                "reject",
                "correction",
                "corrected",
                "summary",
                "summarize",
                "forecast",
                "prediction",
                "token",
            }:
                score += 5
            scored.append((score, -idx, chunk))

        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected = [chunk for score, _, chunk in scored if score > 0][: max(1, int(limit))]
        if selected:
            return selected
        return chunks[: max(1, int(limit))]

    def _llm_memory_rag(self, question: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.pipeline.enable_llm or not self.pipeline.llm_available or not self.pipeline.reasoner:
            return None
        selected_ids = [str(chunk.get("evidence_id")) for chunk in payload.get("selected_memory", [])]
        review_chunks = [
            chunk
            for chunk in payload.get("selected_memory", [])
            if isinstance(chunk, dict) and chunk.get("kind") == "human_review"
        ]
        prompt = (
            "You are a session-memory RAG agent for a controlled wind HITL system.\n"
            "Answer only using the selected_memory records in the context.\n"
            "Do not run or imply a new forecast, retrieval, or state explanation.\n"
            "Do not invent token IDs, probabilities, counts, weather values, or historical support.\n"
            "Read the user's comparative wording carefully: most/closest means lowest retrieval_distance; "
            "least/farthest/most different means highest retrieval_distance among the selected retrieved windows.\n"
            "If selected_memory contains human_review, distinguish the model prediction from the human review state. "
            "Do not say the model changed; say the session review state was accepted, flagged, or noted. "
            "When a human_review record is present, the answer must include one explicit sentence beginning exactly with "
            "'Human review state:' that summarizes the review action and note.\n"
            f"If the selected memory is insufficient or the request is unsupported, set answer to exactly: {UNSUPPORTED_REQUEST_ANSWER}\n"
            "Return only valid JSON with keys: answer, evidence, human_review_prompt. "
            "answer must be a nonempty string, not a nested object or list.\n"
            "The evidence value must be a list of evidence_id strings from selected_memory.\n\n"
            f"User question: {question}\n"
            f"Context:\n{json.dumps(_json_safe(payload), ensure_ascii=False)}\n"
        )
        if hasattr(self.pipeline.reasoner, "invoke_json_with_raw"):
            result, raw_text = self.pipeline.reasoner.invoke_json_with_raw(prompt)
        else:
            result = self.pipeline.reasoner.invoke_json(prompt)
            raw_text = None
        if not isinstance(result, dict):
            if is_plain_text_answer(raw_text):
                answer = str(raw_text).strip()
                answer = self._ensure_review_sentence(answer, review_chunks)
                return {
                    "answer": answer,
                    "evidence": selected_ids,
                    "human_review_prompt": (
                        "Review whether this LLM memory-RAG answer stays grounded in the selected session evidence."
                    ),
                    "llm_output_format": "raw_text",
                }
            return None
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None
        answer = self._ensure_review_sentence(answer.strip(), review_chunks)
        evidence = result.get("evidence")
        if not isinstance(evidence, list):
            evidence = selected_ids
        valid_ids = {str(chunk.get("evidence_id")) for chunk in payload.get("selected_memory", [])}
        evidence = [str(item) for item in evidence if str(item) in valid_ids]
        if not evidence:
            evidence = selected_ids
        return {
            "answer": answer,
            "evidence": evidence,
            "human_review_prompt": str(
                result.get(
                    "human_review_prompt",
                    "Review whether this memory-grounded clarification matches the stored HITL evidence.",
                )
            ),
            "llm_output_format": "json",
        }

    @staticmethod
    def _ensure_review_sentence(answer: str, review_chunks: list[dict[str, Any]]) -> str:
        if answer.strip() == UNSUPPORTED_REQUEST_ANSWER:
            return UNSUPPORTED_REQUEST_ANSWER
        if not review_chunks or "Human review state:" in answer:
            return answer
        review = review_chunks[0].get("payload", {}) if isinstance(review_chunks[0].get("payload"), dict) else {}
        action = review.get("action")
        if action == "reject":
            action = "flag"
        elif action == "relabel":
            action = "note"
        model_token = review.get("model_predicted_token")
        note = review.get("note")
        if action == "flag":
            sentence = f"Human review state: the model-predicted next token {model_token} was flagged for caution."
        elif action == "accept":
            sentence = f"Human review state: the model-predicted next token {model_token} was accepted."
        else:
            sentence = f"Human review state: a human note was saved for model-predicted next token {model_token}."
        if note:
            sentence = f"{sentence} Note: {note}"
        return f"{answer.rstrip()} {sentence}"

    def _memory_rag_followup_text(self, question: str, last: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
        phase = last.get("phase_evidence", {}) if isinstance(last, dict) else {}
        candidates = phase.get("candidate_next_phases", []) if isinstance(phase, dict) else []
        sequence = phase.get("live_token_sequence") or phase.get("token_sequence") or []
        lowered = question.lower()
        question_terms = set(re.findall(r"[a-z0-9_]+", lowered))
        asks_about_similar_windows = bool(
            question_terms
            & {"similar", "retrieved", "retrieval", "nearest", "closest", "match", "matches", "window", "windows"}
        ) and not bool(question_terms & {"phase", "token", "probability", "support", "candidate", "alternative"})
        if asks_about_similar_windows:
            similar_rows: list[dict[str, Any]] = []
            for chunk in chunks:
                if chunk.get("kind") == "similar_windows" and isinstance(chunk.get("payload"), list):
                    similar_rows = [row for row in chunk["payload"] if isinstance(row, dict)]
                    if similar_rows:
                        break
            if similar_rows:
                asks_distance_meaning = bool(question_terms & {"d", "distance"}) and bool(
                    question_terms & {"mean", "means", "meaning", "stands", "represent", "represents", "explain"}
                ) and not bool(question_terms & {"closest", "nearest", "farthest", "furthest", "least", "most"})
                if asks_distance_meaning:
                    return (
                        "d stands for retrieval distance: a score comparing a past window's numerical "
                        "wind-condition representation with the query window. A smaller d means a closer "
                        "match in that representation. It is not a distance in meters, a wind speed, "
                        "or a forecast probability. This explains the stored retrieval results; no new retrieval was run."
                    )
                if "explain" in question_terms and question_terms & {"data", "rows", "table", "results"}:
                    return (
                        f"The {len(similar_rows)} stored rows are past observation windows returned as "
                        "similar examples. window_id identifies each observation period; regime_name names "
                        "its wind pattern; d is retrieval distance, with smaller values indicating closer "
                        "matches to the query window. Wind-speed fields, when present, are measured in m/s. "
                        "Closest and farthest comparisons apply to these retrieved rows. These historical "
                        "examples are evidence for comparison, not predictions of the next window. "
                        "No new retrieval was run."
                    )
                def _distance(row: dict[str, Any]) -> float:
                    try:
                        return float(row.get("retrieval_distance"))
                    except Exception:
                        return float("inf")

                asks_least_similar = bool(
                    question_terms
                    & {"least", "farthest", "furthest", "worst", "lowest", "dissimilar", "different"}
                )
                finite_rows = [row for row in similar_rows if _distance(row) != float("inf")]
                candidate_rows = finite_rows or similar_rows
                selected = max(candidate_rows, key=_distance) if asks_least_similar else min(candidate_rows, key=_distance)
                window = selected.get("window_id") or "the selected retrieved window"
                distance = selected.get("retrieval_distance")
                regime = selected.get("regime_name")
                mean_speed = selected.get("wind_speed_mean")
                details = []
                if regime:
                    details.append(f"regime {regime}")
                if distance is not None:
                    try:
                        details.append(f"retrieval distance {float(distance):.4f}")
                    except Exception:
                        details.append(f"retrieval distance {distance}")
                if mean_speed is not None:
                    try:
                        details.append(f"mean wind speed {float(mean_speed):.2f} m/s")
                    except Exception:
                        details.append(f"mean wind speed {mean_speed}")
                detail_text = f" ({', '.join(details)})" if details else ""
                comparison = "least similar" if asks_least_similar else "most similar"
                distance_logic = "highest" if asks_least_similar else "lowest"
                return (
                    f"The {comparison} stored historical window is {window}{detail_text}. "
                    f"This is selected by the {distance_logic} retrieval distance among the stored retrieved windows; "
                    "no new retrieval was run."
                )

        review_state = None
        for chunk in chunks:
            if chunk.get("kind") == "human_review" and isinstance(chunk.get("payload"), dict):
                review_state = chunk["payload"]
                break
        asks_about_review = bool(
            question_terms
            & {
                "review",
                "human",
                "trust",
                "accepted",
                "accept",
                "flagged",
                "flag",
                "reject",
                "correction",
                "corrected",
                "summarize",
                "summary",
                "status",
            }
        )

        if not candidates:
            for chunk in chunks:
                if chunk.get("kind") == "phase_forecast" and isinstance(chunk.get("payload"), dict):
                    phase = chunk["payload"]
                    candidates = phase.get("candidate_next_phases", []) or []
                    sequence = phase.get("live_token_sequence") or phase.get("token_sequence") or []
                    break
        current_state = None
        for chunk in chunks:
            if chunk.get("kind") == "current_state" and isinstance(chunk.get("payload"), dict):
                current_state = chunk["payload"]
                break
        if current_state:
            metric_map = [
                ({"maximum", "max", "highest", "peak"}, "wind_speed_max", "maximum wind speed", "m/s"),
                ({"minimum", "min", "lowest"}, "wind_speed_min", "minimum wind speed", "m/s"),
                ({"mean", "average", "avg"}, "wind_speed_mean", "mean wind speed", "m/s"),
                ({"std", "standard", "deviation", "spread"}, "wind_speed_std", "wind-speed standard deviation", "m/s"),
                ({"ramp"}, "ramp_abs_max", "maximum ramp", "m/s"),
                ({"gust", "gusts"}, "gust_factor", "gust factor", ""),
                ({"direction", "turning"}, "direction_abs_change_mean_deg", "average directional change", "degrees"),
            ]
            for triggers, key, label, unit in metric_map:
                if question_terms & triggers and key in current_state and current_state.get(key) is not None:
                    try:
                        value = f"{float(current_state.get(key)):.2f}"
                    except Exception:
                        value = str(current_state.get(key))
                    suffix = f" {unit}" if unit else ""
                    window = current_state.get("window_id") or last.get("window_id") or "the stored current window"
                    regime = current_state.get("regime_name")
                    regime_text = f" for {regime}" if regime else ""
                    return (
                        f"The stored current-state evidence reports {label}{regime_text} as {value}{suffix} "
                        f"in window {window}. This answer uses only HITL session memory; no new state explanation "
                        "or retrieval was run."
                    )
        if review_state and asks_about_review:
            return self._review_followup_text(review_state)
        token_match = re.search(r"token\s*([0-9]+)", lowered)
        requested_token = int(token_match.group(1)) if token_match else None

        if candidates:
            top = candidates[0]
            total_support = top.get("support") or phase.get("support")
            if not total_support:
                counts = [row.get("count") for row in candidates if row.get("count") is not None]
                total_support = sum(int(value) for value in counts) if counts else None

            if requested_token is not None:
                row = next((item for item in candidates if int(item.get("token_id", -1)) == requested_token), None)
                if row:
                    comparison = (
                        f"Token {requested_token} was stored with probability {float(row.get('probability') or 0.0):.4f}"
                    )
                    if row.get("count") is not None and total_support:
                        comparison += f" and count {row.get('count')} out of {total_support}"
                    comparison += f". The top stored candidate was token {top.get('token_id')} with probability {float(top.get('probability') or 0.0):.4f}"
                    if top.get("count") is not None and total_support:
                        comparison += f" and count {top.get('count')} out of {total_support}"
                    review_suffix = f" {self._review_followup_text(review_state)}" if review_state else ""
                    return (
                        f"{comparison}. This answer uses only session-memory evidence from the previous HITL result; "
                        f"no new forecast or retrieval was run.{review_suffix}"
                    )

            candidate_text = "; ".join(self._candidate_text(row) for row in candidates[:3])
            support_text = f"total support {total_support}" if total_support is not None else "stored count support unavailable"
            review_suffix = f" {self._review_followup_text(review_state)}" if review_state else ""
            return (
                f"The stored previous phase forecast used the latest semantic token sequence {sequence}. "
                f"The top stored candidates were: {candidate_text}. The evidence has {support_text}. "
                "This is a memory-grounded clarification only; no new forecast, retrieval, or live-state explanation was run."
                f"{review_suffix}"
            )

        if chunks:
            ids = ", ".join(str(chunk.get("evidence_id")) for chunk in chunks[:4])
            titles = "; ".join(str(chunk.get("title")) for chunk in chunks[:4])
            return (
                f"I found relevant session-memory evidence ({ids}): {titles}. "
                "The stored memory does not contain structured phase-candidate evidence, so I cannot infer new token probabilities."
            )

        return "There is no relevant HITL session memory to answer this follow-up."

    @staticmethod
    def _review_followup_text(review: dict[str, Any] | None) -> str:
        if not review:
            return ""
        action = review.get("action")
        if action == "reject":
            action = "flag"
        elif action == "relabel":
            action = "note"
        model_token = review.get("model_predicted_token")
        note = review.get("note")
        if action == "accept":
            text = f"Human review state: accepted the model-predicted next token {model_token}."
        elif action == "flag":
            text = f"Human review state: flagged the model-predicted next token {model_token}; treat this prediction as needing caution."
        else:
            text = f"Human review state: note saved for the model-predicted next token {model_token}."
        if note:
            text = f"{text} Note: {note}"
        return text

    def _memory_followup_text(self, question: str, last: dict[str, Any]) -> str:
        phase = last.get("phase_evidence", {}) if isinstance(last, dict) else {}
        candidates = phase.get("candidate_next_phases", []) if isinstance(phase, dict) else []
        sequence = phase.get("live_token_sequence") or phase.get("token_sequence") or []
        if candidates:
            top = candidates[0]
            support = int(top.get("support") or phase.get("support") or 0)
            count = int(top.get("count") or 0)
            probability = float(top.get("probability") or 0.0)
            alternatives = candidates[1:3]
            alt_text = ""
            if alternatives:
                alt_text = " The next alternatives were " + ", ".join(
                    f"token {row.get('token_id')} ({float(row.get('probability') or 0.0):.3f})"
                    for row in alternatives
                ) + "."
            return (
                f"The previous forecast selected token {top.get('token_id')} because the stored token sequence "
                f"{sequence} most often transitioned to that phase in the historical evidence. It appeared in "
                f"{count} out of {support} matching transition(s), giving probability {probability:.3f}. "
                f"The selected regime was {top.get('regime_name')}.{alt_text} This answer uses only the previous "
                "HITL result; no new forecast or retrieval was run."
            )
        answer = last.get("answer")
        if answer:
            return (
                "The previous HITL result does not contain structured phase-candidate evidence, so I can only "
                f"refer back to the prior answer: {answer}"
            )
        return "There is no previous HITL result in this thread memory to explain."

    def _relevance_check(self, state: HITLAgentState) -> HITLAgentState:
        decision = self.pipeline.filter.classify(
            state.get("effective_question") or state["question"],
            forced_action=state.get("feedback_action"),
        )
        router = {
            "source": "deterministic",
            "deterministic_intent": decision.intent,
            "deterministic_confidence": decision.confidence,
            "llm_router_attempted": False,
            "llm_router_used": False,
            "reason": decision.reason,
        }

        if state.get("target_predicted_numeric_window"):
            route = "retrieve_similar" if state.get("target_window_operation") == "retrieve_similar" else "explain_state"
            router["source"] = "memory_resolved"
            router["original_intent"] = decision.intent
            router["deterministic_intent"] = route
            router["resolved_context"] = "predicted_numeric_window"
            router["reason"] = "Follow-up reference was resolved to a predicted numeric window before routing."
            decision.intent = route
            decision.confidence = max(decision.confidence, 0.90)
            decision.matched_terms = sorted(set(decision.matched_terms + ["predicted_window"]))
        elif not decision.allowed:
            route = "out_of_scope"
            llm_route = self._llm_route_intent(state, decision)
            if llm_route:
                route = llm_route["intent"]
                router.update(llm_route)
        elif decision.intent == "feedback" or state.get("feedback_action"):
            route = "out_of_scope"
            router["reason"] = "Human feedback actions are currently disabled."
        elif decision.intent == "retrieve_similar":
            route = "retrieve_similar"
        elif decision.intent == "numeric_forecast":
            route = "phase_prediction"
        elif decision.intent == "phase_prediction":
            route = "phase_prediction"
        elif decision.intent == "explain_forecast":
            route = "explain_state"
        else:
            route = "explain_state"

        if decision.allowed and decision.confidence < 0.70 and route != "retrieve_similar":
            llm_route = self._llm_route_intent(state, decision)
            if llm_route:
                route = llm_route["intent"]
                router.update(llm_route)

        return {
            **state,
            "filter_decision": decision,
            "route": route,
            "router": router,
            "trace": self._append_trace(state, f"relevance_check:{route}:{router['source']}"),
        }

    def _llm_route_intent(self, state: HITLAgentState, decision: Any) -> dict[str, Any] | None:
        if not self.pipeline.llm_available:
            return None
        if state.get("feedback_action"):
            return None

        allowed_intents = {
            "explain_state",
            "retrieve_similar",
            "phase_prediction",
            "out_of_scope",
        }
        memory = state.get("memory") or {}
        memory_summary = {
            "has_previous_result": bool(memory.get("last_tool_result")),
            "has_previous_phase_result": bool(memory.get("last_phase_result")),
            "last_route": (memory.get("turns") or [{}])[-1].get("route") if memory.get("turns") else None,
        }
        prompt = (
            "You are a constrained intent router for a wind-forecasting HITL system.\n"
            "Classify the user request into exactly one allowed intent.\n"
            "Allowed intents: explain_state, retrieve_similar, phase_prediction, out_of_scope.\n"
            "Use phase_prediction for forecast or prediction requests, including raw wind-speed wording.\n"
            "Use out_of_scope for unsupported requests, broad what-if simulation, aviation safety advice, "
            "or non-wind topics.\n"
            "Do not answer the user. Return only JSON with keys: intent, confidence, reason.\n\n"
            f"User question: {state.get('effective_question') or state['question']}\n"
            f"Deterministic router: {json.dumps(getattr(decision, 'to_dict', lambda: {})(), ensure_ascii=False)}\n"
            f"Session memory summary: {json.dumps(memory_summary, ensure_ascii=False)}\n"
        )
        payload = self.pipeline.reasoner.invoke_json(prompt) if self.pipeline.reasoner else None
        if not isinstance(payload, dict):
            return None
        intent = str(payload.get("intent", "")).strip()
        if intent not in allowed_intents:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < 0.50:
            return None
        return {
            "source": "llm_fallback",
            "intent": intent,
            "confidence": round(confidence, 3),
            "reason": str(payload.get("reason", "LLM fallback router selected this intent.")),
            "llm_router_attempted": True,
            "llm_router_used": True,
            "deterministic_intent": decision.intent,
            "deterministic_confidence": decision.confidence,
        }

    @staticmethod
    def _route_after_relevance(state: HITLAgentState) -> RouteName:
        return state.get("route", "out_of_scope")

    def _run_retrieval(self, state: HITLAgentState) -> HITLAgentState:
        if state.get("target_predicted_numeric_window"):
            result = self._retrieve_windows_for_predicted_numeric_window(state.get("memory") or {})
        else:
            result = self.pipeline.retrieve_similar(state.get("effective_question") or state["question"])
        return {**state, "result": result, "trace": self._append_trace(state, "tool:retrieve_similar")}

    def _retrieve_windows_for_predicted_numeric_window(self, memory: dict[str, Any]) -> dict[str, Any]:
        numeric = memory.get("last_numeric_result", {}) if isinstance(memory, dict) else {}
        sketch = numeric.get("numeric_forecast", {}).get("predicted_window_sketch", {})
        if not sketch:
            return {
                "mode": "retrieve_similar",
                "retrieval_context": "predicted_numeric_window",
                "explanation_mode": "missing_context",
                "answer": "No predicted numeric window sketch is available in this session memory.",
                "similar_windows": [],
                "evidence": {"predicted_window_error": "missing_sketch"},
                "evidence_pack": {"predicted_window_error": "missing_sketch"},
                "human_review_prompt": "Run a raw numeric forecast first, then ask for similar predicted windows.",
            }

        features_path = self.pipeline.context.metadata["output_paths"].get("features")
        frame = pd.read_csv(features_path)
        candidate_columns = [
            "wind_speed_mean",
            "wind_speed_std",
            "wind_speed_min",
            "wind_speed_max",
            "wind_speed_p10",
            "wind_speed_p90",
            "wind_speed_range",
            "ramp_abs_mean",
            "ramp_abs_max",
        ]
        columns = [column for column in candidate_columns if column in frame.columns and column in sketch]
        if not columns:
            rows: list[dict[str, Any]] = []
        else:
            feature_frame = frame[columns].apply(pd.to_numeric, errors="coerce")
            means = feature_frame.mean(axis=0)
            stds = feature_frame.std(axis=0).replace(0, 1.0).fillna(1.0)
            target = pd.Series({column: float(sketch[column]) for column in columns})
            distances = (((feature_frame - target) / stds) ** 2).sum(axis=1) ** 0.5
            ranked = frame.assign(predicted_window_distance=distances).sort_values(
                "predicted_window_distance",
                ascending=True,
            )
            rows = [
                self._compact_predicted_window_match(row.to_dict(), idx + 1)
                for idx, (_, row) in enumerate(ranked.head(self.pipeline.context.top_k).iterrows())
            ]

        payload = self.pipeline.build_evidence_pack(include_neighbors=False)
        payload["predicted_numeric_window"] = sketch
        payload["similar_windows"] = rows
        payload["matching_features"] = columns
        answer = (
            f"Retrieved {len(rows)} historical window(s) similar to the previous predicted numeric wind pattern. "
            "This compares the predicted raw wind-speed window sketch against historical window-feature rows, "
            "not against the predicted semantic token."
        )
        return {
            "mode": "retrieve_similar",
            "retrieval_context": "predicted_numeric_window",
            "explanation_mode": "feature_distance",
            "answer": answer,
            "similar_windows": rows,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": "Check whether these historical windows are useful analogs for the predicted numeric pattern.",
        }

    @staticmethod
    def _compact_predicted_window_match(row: dict[str, Any], idx: int) -> dict[str, Any]:
        return {
            "evidence_id": f"predicted_numeric_window_match_{idx}",
            "window_id": row.get("window_id"),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "retrieval_distance": row.get("predicted_window_distance"),
            "wind_speed_mean": row.get("wind_speed_mean"),
            "wind_speed_std": row.get("wind_speed_std"),
            "wind_speed_min": row.get("wind_speed_min"),
            "wind_speed_max": row.get("wind_speed_max"),
            "ramp_abs_mean": row.get("ramp_abs_mean"),
            "ramp_abs_max": row.get("ramp_abs_max"),
        }

    def _run_phase_forecast(self, state: HITLAgentState) -> HITLAgentState:
        result = self.pipeline.predict_phase(state.get("effective_question") or state["question"])
        return {**state, "result": result, "trace": self._append_trace(state, "tool:phase_forecast")}

    def _run_state_explanation(self, state: HITLAgentState) -> HITLAgentState:
        if state.get("target_predicted_numeric_window"):
            result = self._explain_predicted_numeric_window(state.get("memory") or {})
        else:
            result = self.pipeline.explain_state(
                state.get("effective_question") or state["question"],
                window_context="current_window",
            )
        return {**state, "result": result, "trace": self._append_trace(state, "tool:explain_state")}

    def _explain_predicted_numeric_window(self, memory: dict[str, Any]) -> dict[str, Any]:
        numeric = memory.get("last_numeric_result", {}) if isinstance(memory, dict) else {}
        forecast = numeric.get("numeric_forecast", {}) if isinstance(numeric, dict) else {}
        sketch = forecast.get("predicted_window_sketch", {}) if isinstance(forecast, dict) else {}
        if not sketch:
            return {
                "mode": "explain_window",
                "explanation_context": "predicted_numeric_window",
                "explanation_mode": "missing_context",
                "answer": "No predicted numeric window sketch is available in this session memory.",
                "evidence": {"predicted_window_error": "missing_sketch"},
                "evidence_pack": {"predicted_window_error": "missing_sketch"},
                "human_review_prompt": "Run a raw numeric forecast first, then ask to explain the forecasted window.",
            }

        payload = self.pipeline.build_evidence_pack(include_neighbors=False)
        payload["predicted_numeric_window"] = sketch
        answer = (
            "The forecasted numeric window is a compact sketch built from recent observed raw wind-speed values "
            "plus the numeric forecast output. "
            f"It has mean speed {sketch.get('wind_speed_mean')} m/s, std {sketch.get('wind_speed_std')}, "
            f"range {sketch.get('wind_speed_range')} m/s, and max ramp {sketch.get('ramp_abs_max')} m/s. "
            "This is not an observed historical semantic window; it is a forecast-derived query window for review."
        )
        return {
            "mode": "explain_window",
            "explanation_context": "predicted_numeric_window",
            "explanation_mode": "numeric_window_sketch",
            "answer": answer,
            "evidence": payload,
            "evidence_pack": payload,
            "human_review_prompt": "Check whether the forecasted numeric window sketch matches the forecast behavior you expected.",
        }

    def _critic_check(self, state: HITLAgentState) -> HITLAgentState:
        result = state.get("result") or {}
        warnings: list[str] = []

        if self.pipeline.enable_llm and not self.pipeline.llm_available:
            warnings.append("LLM was requested but is unavailable; deterministic fallback was used.")
        if not self.pipeline.enable_rag and state.get("route") in {
            "retrieve_similar",
            "phase_prediction",
            "compound_request",
        }:
            warnings.append("RAG/retrieval evidence is disabled for this run.")

        phase_payload = result.get("phase_prediction", {})
        phase_evidence: dict[str, Any] = {}
        if isinstance(phase_payload, dict):
            phase_evidence = phase_payload.get("evidence", {}) or {}
        if not phase_evidence and isinstance(result.get("evidence_pack"), dict):
            phase_evidence = result["evidence_pack"].get("phase_prediction", {}) or result["evidence_pack"].get(
                "live_phase_prediction",
                {},
            )

        if phase_evidence:
            support = int(phase_evidence.get("support", 0) or 0)
            minimum = int(phase_evidence.get("minimum_support", 0) or 0)
            if minimum and support < minimum:
                warnings.append(f"Phase support is low: {support} < minimum {minimum}.")
            if not phase_evidence.get("candidate_next_phases"):
                warnings.append("No candidate next phases were available from the evidence.")

        return {
            **state,
            "critic_warnings": warnings,
            "trace": self._append_trace(state, "critic_check"),
        }

    def _finalize(self, state: HITLAgentState) -> HITLAgentState:
        if state.get("route") == "out_of_scope":
            result = {
                "mode": "out_of_scope",
                "answer": UNSUPPORTED_REQUEST_ANSWER,
            }
        else:
            result = dict(state.get("result") or {})

        # Preserve the scientific tables, but never save a canned explanation as
        # though it were displayed to a public visitor whose LLM call failed.
        needs_llm = self.pipeline.enable_llm or getattr(self.pipeline, "require_llm_explanations", False)
        if needs_llm and state.get("route") in {
            "phase_prediction",
            "retrieve_similar",
            "explain_state",
            "compound_request",
        }:
            mode = str(result.get("mode") or "")
            if (result.get("explanation_mode") == "heuristic" or mode.startswith("deterministic_")
                    or mode in {"heuristic", "hitl_gru_live_phase_prediction"}):
                result["answer"] = UNSUPPORTED_REQUEST_ANSWER

        agent_trace = self._append_trace(state, "finalize")
        result["agent_graph"] = {
            "enabled": True,
            "engine": "langgraph",
            "workflow": "controlled_hitl_agent",
            "thread_id": state.get("thread_id"),
            "route": state.get("route"),
            "router": state.get("router", {}),
            "rewrite": state.get("rewrite"),
            "task_plan": state.get("task_plan", []),
            "completed_tasks": state.get("completed_tasks", []),
            "followup_operations": state.get("followup_operations", []),
            "trace": agent_trace,
            "critic_warnings": state.get("critic_warnings", []),
        }
        if state.get("critic_warnings"):
            result["critic_warnings"] = state["critic_warnings"]

        final_result = {**self._base_result(state), **result}
        phase_payload = final_result.get("phase_prediction", {})
        phase_evidence = phase_payload.get("evidence", {}) if isinstance(phase_payload, dict) else {}
        evidence_pack = final_result.get("evidence_pack", {})
        current_live_state = phase_evidence.get("current_live_state", {}) if isinstance(phase_evidence, dict) else {}
        if not current_live_state and isinstance(evidence_pack, dict):
            current_live_state = evidence_pack.get("current_state", {})
        live_window_id = current_live_state.get("window_id") if isinstance(current_live_state, dict) else None
        if live_window_id and str(live_window_id).startswith("live_"):
            final_result["window_id"] = str(live_window_id)
        self._save_session_memory(state, final_result)
        return {**state, "result": final_result, "trace": agent_trace}

    def _save_session_memory(self, state: HITLAgentState, final_result: dict[str, Any]) -> None:
        thread_id = state.get("thread_id")
        if not thread_id:
            return
        memory = dict(state.get("memory") or {})
        turns = list(memory.get("turns") or [])[-4:]
        turns.append(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "question": state["question"],
                "effective_question": state.get("effective_question") or state["question"],
                "route": state.get("route"),
                "mode": final_result.get("mode"),
                "answer": final_result.get("answer"),
            }
        )
        memory["turns"] = turns[-5:]
        if state.get("route") not in {"memory_followup", "out_of_scope"}:
            compact = self._compact_result_for_memory(final_result)
            memory["last_tool_result"] = compact
            if compact.get("phase_evidence", {}).get("candidate_next_phases"):
                memory["last_phase_result"] = compact
            if state.get("route") == "retrieve_similar" or (
                state.get("route") == "compound_request" and compact.get("similar_windows")
            ):
                memory["last_retrieval_result"] = compact
            if state.get("route") == "feedback" and final_result.get("review_state"):
                reviews = list(memory.get("reviews") or [])[-9:]
                review_state = final_result["review_state"]
                reviews.append(review_state)
                memory["reviews"] = reviews[-10:]
                memory["last_review"] = review_state
        self.memory_store.save(thread_id, memory)

    @staticmethod
    def _compact_result_for_memory(result: dict[str, Any]) -> dict[str, Any]:
        evidence_pack = result.get("evidence_pack") if isinstance(result.get("evidence_pack"), dict) else {}
        phase_payload = result.get("phase_prediction", {})
        phase_evidence = phase_payload.get("evidence", {}) if isinstance(phase_payload, dict) else {}
        if not phase_evidence:
            phase_evidence = evidence_pack.get("live_phase_prediction") or evidence_pack.get("phase_prediction") or {}
        return {
            "intent": result.get("filter", {}).get("intent"),
            "mode": result.get("mode"),
            "window_id": result.get("window_id"),
            "answer": result.get("answer"),
            "phase_evidence": {
                "live_token_sequence": phase_evidence.get("live_token_sequence"),
                "token_sequence": phase_evidence.get("token_sequence"),
                "candidate_next_phases": (phase_evidence.get("candidate_next_phases") or [])[:5],
                "support": phase_evidence.get("support"),
                "minimum_support": phase_evidence.get("minimum_support"),
                "current_live_state": phase_evidence.get("current_live_state"),
            },
            "numeric_forecast": result.get("numeric_forecast") or evidence_pack.get("numeric_forecast"),
            "current_state": evidence_pack.get("current_state"),
            "similar_windows": (result.get("similar_windows") or evidence_pack.get("similar_windows") or [])[:5],
            "review_state": result.get("review_state"),
            "critic_warnings": result.get("critic_warnings", []),
        }

    def invoke(
        self,
        question: str,
        feedback_action: str | None = None,
        feedback_label: str = "",
        feedback_note: str = "",
        reviewer: str = "human",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        thread_id = thread_id or "hitl_default"
        final_state = self.graph.invoke(
            {
                "question": question,
                "effective_question": question,
                "thread_id": thread_id,
                "feedback_action": feedback_action,
                "feedback_label": feedback_label,
                "feedback_note": feedback_note,
                "reviewer": reviewer,
                "trace": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        return final_state["result"]
