from benchmark_rag import evaluate_case, summarize_results


class FakeRAG:
    def __init__(self, answer, sources):
        self.answer_text = answer
        self.sources = sources

    def answer(self, question, return_sources=False):
        return self.answer_text, self.sources


def test_evaluation_separates_answer_coverage_from_context_coverage():
    case = {
        "id": "leave-days",
        "question": "How many leave days?",
        "must_have": ["20 days", "annual leave"],
        "expected_sources": ["leave_policy.txt"],
    }
    rag = FakeRAG(
        "Employees receive paid leave.",
        [{"source": "leave_policy.txt", "text": "20 days of annual leave."}],
    )

    result = evaluate_case(rag, case)
    summary = summarize_results(
        [result],
        min_pass_rate=100,
        min_answer_coverage=95,
        min_source_recall=95,
    )

    assert result["answer_coverage"] == 0.0
    assert result["context_coverage"] == 100.0
    assert result["source_recall"] == 100.0
    assert not result["passed"]
    assert not summary["quality_gate_passed"]
    assert any("answer_coverage" in failure for failure in summary["quality_gate_failures"])


def test_quality_gate_passes_when_answer_and_source_match():
    case = {
        "id": "leave-days",
        "question": "How many leave days?",
        "must_have": ["20 days", "annual leave"],
        "expected_sources": ["leave_policy.txt"],
    }
    rag = FakeRAG(
        "Employees receive 20 days of annual leave.",
        [{"source": "leave_policy.txt", "text": "20 days of annual leave."}],
    )

    summary = summarize_results(
        [evaluate_case(rag, case)],
        min_pass_rate=100,
        min_answer_coverage=95,
        min_source_recall=95,
    )

    assert summary["quality_gate_passed"]
    assert summary["answer_coverage"] == 100.0
    assert summary["source_recall_at_k"] == 100.0


def test_semantic_scores_are_reported_and_thresholds_are_enforced():
    case = {
        "id": "leave-days",
        "question": "How many leave days?",
        "must_have": ["20 days"],
        "expected_sources": ["leave_policy.txt"],
    }

    class FakeSemanticEvaluator:
        def evaluate(self, question, answer, retrieved_contexts):
            return {
                "faithfulness": 0.7,
                "answer_relevancy": 0.9,
                "faithfulness_supported_claims": 7,
                "faithfulness_total_claims": 10,
                "faithfulness_claims": [],
                "answer_relevancy_rationale": "It answers the question.",
            }

    result = evaluate_case(
        FakeRAG("Employees receive 20 days of leave.", [{"source": "leave_policy.txt", "text": "20 days"}]),
        case,
        FakeSemanticEvaluator(),
    )
    summary = summarize_results(
        [result],
        min_faithfulness=0.8,
        min_answer_relevancy=0.8,
    )

    assert summary["faithfulness"] == 0.7
    assert summary["answer_relevancy"] == 0.9
    assert not summary["quality_gate_passed"]
    assert any("faithfulness" in failure for failure in summary["quality_gate_failures"])