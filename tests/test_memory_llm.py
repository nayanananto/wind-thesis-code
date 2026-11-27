"""No-network checks for LLM-only grounded follow-ups and failure handling."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.hitl.controlled_agent_graph import ControlledHITLAgentGraph
from app.hitl.messages import UNSUPPORTED_REQUEST_ANSWER
from app.hitl.standalone_pipeline import StandaloneHITLPipeline


QUESTION = "Explain what d (retrieval distance) means in the retrieved historical windows."
ROWS = [{"window_id": "past_1", "retrieval_distance": 0.04},
        {"window_id": "past_2", "retrieval_distance": 0.11}]


class MemoryLLMTests(unittest.TestCase):
    def graph(self, response=(None, None), *, enabled=True, available=True, error=None):
        reasoner = SimpleNamespace(invoke_json_with_raw=Mock(return_value=response), last_error=error)
        graph = object.__new__(ControlledHITLAgentGraph)
        graph.pipeline = SimpleNamespace(enable_llm=enabled, llm_available=available, reasoner=reasoner)
        return graph

    def state(self):
        result = {"mode": "retrieve_similar", "answer": "Retrieved two windows.", "similar_windows": ROWS}
        return {"question": QUESTION, "trace": [], "memory": {
            "last_tool_result": result, "last_retrieval_result": deepcopy(result),
            "turns": [{"question": "retrieve similar historical windows", "answer": result["answer"]}],
        }}

    def test_json_answer_uses_llm_and_unique_evidence_context(self):
        graph = self.graph(({"answer": "Smaller d means a closer wind-pattern match.", "evidence": []}, None))
        state = self.state()
        original = deepcopy(state)
        result = graph._analyze_session_memory_followup(state)["result"]
        self.assertEqual(result["mode"], "llm_memory_rag_followup")
        self.assertTrue(result["memory_rag_debug"]["llm_used"])
        self.assertEqual(state, original)
        chunks = result["evidence_pack"]["selected_memory"]
        self.assertEqual(sum(c["kind"] == "similar_windows" for c in chunks), 1)
        self.assertTrue(any(c["kind"] == "retrieval_semantics" for c in chunks))
        self.assertNotIn("previous_result", result["evidence_pack"])
        prompt = graph.pipeline.reasoner.invoke_json_with_raw.call_args.args[0]
        self.assertIn("a difference between token IDs or token sequences", prompt)

    def test_plain_prose_is_an_llm_answer_not_a_canned_fallback(self):
        graph = self.graph((None, "A smaller d means the past window is a closer match."))
        result = graph._analyze_session_memory_followup(self.state())["result"]
        self.assertTrue(result["memory_rag_debug"]["llm_used"])
        self.assertEqual(result["llm_output_format"], "raw_text")

    def test_unusable_outputs_never_call_deterministic_memory_writer(self):
        for response in ((None, None), ({"answer": ""}, None), ({"answer": {"text": "nested"}}, None),
                         ({"evidence": []}, None), (None, '{"answer": "truncated'),
                         (None, '[{"answer": "unexpected list"}]'), (None, "null"),
                         (None, "42"), (None, 'Here is the JSON: {"answer":')):
            with self.subTest(response=response):
                graph = self.graph(response)
                with patch.object(graph, "_memory_rag_followup_text") as fallback:
                    result = graph._analyze_session_memory_followup(self.state())["result"]
                fallback.assert_not_called()
                self.assertEqual(result["mode"], "memory_rag_unavailable")
                self.assertEqual(result["answer"], UNSUPPORTED_REQUEST_ANSWER)
                self.assertFalse(result["memory_rag_debug"]["llm_used"])

    def test_failure_diagnostics_distinguish_provider_errors(self):
        for category in ("timeout", "rate_limit", "authentication", "empty_response"):
            graph = self.graph(error={"category": category, "retryable": True})
            result = graph._analyze_session_memory_followup(self.state())["result"]
            self.assertIn(category, result["memory_rag_debug"]["llm_fallback_reason"])
            self.assertNotIn("no valid JSON", result["memory_rag_debug"]["llm_fallback_reason"])

    def test_disabled_or_missing_llm_is_not_invoked(self):
        for enabled, available in ((False, True), (True, False)):
            graph = self.graph(enabled=enabled, available=available)
            result = graph._analyze_session_memory_followup(self.state())["result"]
            graph.pipeline.reasoner.invoke_json_with_raw.assert_not_called()
            self.assertEqual(result["answer"], UNSUPPORTED_REQUEST_ANSWER)

    def test_exception_details_are_not_exposed(self):
        graph = self.graph()
        graph.pipeline.reasoner.invoke_json_with_raw.side_effect = RuntimeError("sk-private-key should not leak")
        result = graph._analyze_session_memory_followup(self.state())["result"]
        self.assertNotIn("sk-private", json.dumps(result))
        self.assertEqual(result["memory_rag_debug"]["llm_error"]["category"], "memory_answer_error")

    def test_unsupported_answer_never_gets_extra_review_text(self):
        for response in (({"answer": UNSUPPORTED_REQUEST_ANSWER}, None),
                         (None, UNSUPPORTED_REQUEST_ANSWER)):
            graph = self.graph(response)
            state = self.state()
            state["memory"]["last_review"] = {"action": "flag", "model_predicted_token": 2, "note": "Check evidence."}
            result = graph._analyze_session_memory_followup(state)["result"]
            self.assertEqual(result["answer"], UNSUPPORTED_REQUEST_ANSWER)

    def test_core_llm_output_cannot_override_execution_fields(self):
        pipeline = object.__new__(StandaloneHITLPipeline)
        pipeline.reasoner = SimpleNamespace(available=True, invoke_json_with_raw=Mock(return_value=({
            "answer": "  " + UNSUPPORTED_REQUEST_ANSWER + "\n", "evidence": [],
            "mode": "pretend_success", "explanation_mode": "pretend_success", "similar_windows": ROWS,
        }, None)))
        result = pipeline._llm_json("retrieve_similar", QUESTION, {})
        self.assertEqual(result["answer"], UNSUPPORTED_REQUEST_ANSWER)
        self.assertEqual(set(result), {"answer", "evidence", "human_review_prompt"})

    def test_failed_core_explanation_stores_only_public_answer_but_keeps_facts(self):
        for enabled in (True, False):
            graph = self.graph(enabled=enabled)
            graph.pipeline.require_llm_explanations = True
            graph.pipeline.enable_rag = True
            graph.pipeline.context = SimpleNamespace(metadata={}, resolved_window_id="current")
            graph.memory_store = SimpleNamespace(save=Mock())
            state = {"question": "retrieve similar historical windows", "thread_id": "test", "trace": [],
                     "route": "retrieve_similar", "memory": {}, "result": {
                         "mode": "retrieve_similar", "explanation_mode": "heuristic",
                         "answer": "Hidden canned explanation", "similar_windows": ROWS,
                     }}
            result = graph._finalize(state)["result"]
            memory = graph.memory_store.save.call_args.args[1]
            self.assertEqual(result["answer"], UNSUPPORTED_REQUEST_ANSWER)
            self.assertEqual(memory["last_tool_result"]["answer"], UNSUPPORTED_REQUEST_ANSWER)
            self.assertEqual(memory["turns"][0]["answer"], UNSUPPORTED_REQUEST_ANSWER)
            self.assertEqual(memory["last_retrieval_result"]["similar_windows"], ROWS)


if __name__ == "__main__":
    unittest.main()
