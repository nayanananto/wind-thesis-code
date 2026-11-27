from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.hitl.controlled_agent_graph import ControlledHITLAgentGraph


class _Decision:
    def __init__(self, intent: str = "phase_prediction"):
        self.intent = intent
        self.confidence = 0.95
        self.allowed = True
        self.reason = "Unit-test route."
        self.matched_terms = ["predict"]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "allowed": self.allowed,
            "reason": self.reason,
            "matched_terms": self.matched_terms,
        }


class _Filter:
    def classify(self, question: str, forced_action=None) -> _Decision:
        del question, forced_action
        return _Decision()


class _Reasoner:
    last_error = None

    def __init__(self):
        self.prompts: list[str] = []

    def invoke_json_with_raw(self, prompt: str):
        self.prompts.append(prompt)
        return ({
            "answer": "The current state, next-phase forecast, and historical matches are summarized together.",
            "evidence": [
                "compound:explain_state",
                "compound:phase_prediction",
                "compound:retrieve_similar",
            ],
            "human_review_prompt": "Check the combined answer.",
        }, None)


class _Pipeline:
    def __init__(self, root: Path, *, llm: bool = False):
        self.feedback_store = SimpleNamespace(path=root / "feedback.jsonl")
        self.context = SimpleNamespace(
            metadata={"run_name": "compound-unit-test"},
            resolved_window_id="live_test_window",
        )
        self.filter = _Filter()
        self.enable_rag = True
        self.enable_llm = llm
        self.llm_available = llm
        self.reasoner = _Reasoner() if llm else None
        self.calls: list[str] = []
        self.tool_llm_flags: list[bool] = []

    def predict_phase(self, question: str) -> dict:
        self.calls.append("phase_prediction")
        self.tool_llm_flags.append(self.enable_llm)
        return {
            "mode": "phase_forecast",
            "explanation_mode": "heuristic",
            "answer": "Token 2 is the leading next-phase candidate.",
            "phase_prediction": {
                "evidence": {
                    "live_token_sequence": [1, 1, 2],
                    "current_live_state": {
                        "window_id": "live_test_window",
                        "regime_name": "Steady wind",
                    },
                    "candidate_next_phases": [
                        {"token_id": 2, "regime_name": "Gusty wind", "probability": 0.7, "count": 7, "support": 10}
                    ],
                    "support": 10,
                    "minimum_support": 5,
                }
            },
        }

    def explain_state(self, question: str, *, window_context: str) -> dict:
        self.calls.append("explain_state")
        self.tool_llm_flags.append(self.enable_llm)
        return {
            "mode": "explain_state",
            "explanation_mode": "heuristic",
            "answer": "The current window has a steady wind pattern.",
            "evidence_pack": {
                "current_state": {
                    "window_id": "live_test_window",
                    "regime_name": "Steady wind",
                }
            },
        }

    def retrieve_similar(self, question: str) -> dict:
        self.calls.append("retrieve_similar")
        self.tool_llm_flags.append(self.enable_llm)
        rows = [
            {
                "window_id": "historical_close_window",
                "regime_name": "Steady wind",
                "retrieval_distance": 0.12,
            },
            {
                "window_id": "historical_far_window",
                "regime_name": "Gusty wind",
                "retrieval_distance": 0.81,
            },
        ]
        return {
            "mode": "retrieve_similar",
            "explanation_mode": "heuristic",
            "answer": "One similar historical window was retrieved.",
            "similar_windows": rows,
            "evidence_pack": {"similar_windows": rows},
        }


class CompoundRequestTests(unittest.TestCase):
    def test_compiled_graph_uses_one_controller_tool_loop_for_all_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = ControlledHITLAgentGraph(_Pipeline(Path(directory)))
            mermaid = graph.graph.get_graph().draw_mermaid()

        self.assertIn("task_controller", mermaid)
        self.assertIn("controlled_tool_node", mermaid)
        self.assertIn("synthesize_response", mermaid)
        self.assertNotIn("answer_from_memory", mermaid)
        self.assertNotIn("memory_only", mermaid)
        self.assertNotIn("run_feedback", mermaid)
        self.assertNotIn("dispatch_compound_task", mermaid)
        self.assertNotIn("run_compound_task", mermaid)

    def test_multiple_explicit_tasks_run_in_prompt_order(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory))
            result = ControlledHITLAgentGraph(pipeline).invoke(
                "Predict the next phase, explain the current state, and retrieve similar historical windows.",
                thread_id="compound-order",
            )

        expected = ["phase_prediction", "explain_state", "retrieve_similar"]
        self.assertEqual(pipeline.calls, expected)
        self.assertEqual(result["compound_tasks"], expected)
        self.assertEqual(result["agent_graph"]["route"], "compound_request")
        self.assertEqual(result["agent_graph"]["completed_tasks"], expected)
        self.assertEqual(result["filter"]["intent"], "compound_request")
        self.assertEqual(result["explanation_mode"], "heuristic")
        self.assertIn("Next phase forecast:", result["answer"])
        self.assertIn("Current state:", result["answer"])
        self.assertIn("Similar historical windows:", result["answer"])
        self.assertEqual(result["similar_windows"][0]["window_id"], "historical_close_window")

    def test_retrieval_followup_runs_after_retrieval_in_the_same_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory))
            result = ControlledHITLAgentGraph(pipeline).invoke(
                "Find similar historical windows and find the closest and farthest one.",
                thread_id="retrieval-followup",
            )

        self.assertEqual(pipeline.calls, ["retrieve_similar"])
        self.assertEqual(result["compound_tasks"], ["retrieve_similar", "followup_analysis"])
        self.assertEqual(
            result["agent_graph"]["followup_operations"],
            ["closest_window", "farthest_window"],
        )
        comparison = result["followup_analysis"]["retrieval_analysis"]
        self.assertEqual(comparison["closest_window"]["window_id"], "historical_close_window")
        self.assertEqual(comparison["farthest_window"]["window_id"], "historical_far_window")
        self.assertIn("controlled_tool_node:followup_analysis", result["agent_graph"]["trace"])
        self.assertIn("d=0.1200", result["answer"])
        self.assertIn("d=0.8100", result["answer"])

    def test_phase_followup_runs_after_forecast_in_the_same_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory))
            result = ControlledHITLAgentGraph(pipeline).invoke(
                "Predict the next phase and explain why this phase was selected instead of the alternatives.",
                thread_id="phase-followup",
            )

        self.assertEqual(pipeline.calls, ["phase_prediction"])
        self.assertEqual(result["compound_tasks"], ["phase_prediction", "followup_analysis"])
        self.assertEqual(result["followup_operations"], ["phase_rationale"])
        self.assertIn("highest stored transition probability", result["answer"])
        self.assertEqual(
            result["followup_analysis"]["phase_analysis"]["candidate_next_phases"][0]["token_id"],
            2,
        )

    def test_single_request_keeps_the_existing_route(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory))
            result = ControlledHITLAgentGraph(pipeline).invoke(
                "Predict the next phase.",
                thread_id="single-request",
            )

        self.assertEqual(pipeline.calls, ["phase_prediction"])
        self.assertEqual(result["agent_graph"]["route"], "phase_prediction")
        self.assertEqual(result["agent_graph"]["task_plan"], ["phase_prediction"])
        self.assertNotEqual(result["mode"], "compound_request")

    def test_llm_combines_completed_tool_results_once(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory), llm=True)
            result = ControlledHITLAgentGraph(pipeline).invoke(
                "Explain the current state, predict the next phase, and retrieve similar historical windows.",
                thread_id="compound-llm",
            )

        self.assertEqual(pipeline.calls, ["explain_state", "phase_prediction", "retrieve_similar"])
        self.assertEqual(pipeline.tool_llm_flags, [False, False, False])
        self.assertTrue(pipeline.enable_llm)
        self.assertEqual(len(pipeline.reasoner.prompts), 1)
        self.assertEqual(result["explanation_mode"], "llm_rag")
        self.assertTrue(result["compound_debug"]["llm_used"])
        self.assertIn("summarized together", result["answer"])

    def test_style_followup_is_not_mistaken_for_an_extra_tool(self):
        tasks = ControlledHITLAgentGraph._detect_compound_tasks(
            "Predict the next phase and explain it briefly in Bengali."
        )
        self.assertEqual(tasks, [])

    def test_previous_turn_followup_uses_the_controlled_tool_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = _Pipeline(Path(directory), llm=True)
            graph = ControlledHITLAgentGraph(pipeline)
            graph.invoke("Predict the next phase.", thread_id="memory-loop")
            result = graph.invoke("Why was this phase selected?", thread_id="memory-loop")

        self.assertEqual(pipeline.calls, ["phase_prediction"])
        self.assertEqual(result["agent_graph"]["route"], "memory_followup")
        self.assertEqual(result["agent_graph"]["task_plan"], ["followup_analysis"])
        self.assertEqual(result["agent_graph"]["completed_tasks"], ["followup_analysis"])
        self.assertIn("controlled_tool_node:followup_analysis", result["agent_graph"]["trace"])
        self.assertNotIn("memory:answer_from_previous_result", result["agent_graph"]["trace"])
        self.assertEqual(len(pipeline.reasoner.prompts), 2)


if __name__ == "__main__":
    unittest.main()
