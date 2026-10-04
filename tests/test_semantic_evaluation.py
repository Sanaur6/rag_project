from types import SimpleNamespace

from src.semantic_evaluation import (
    FaithfulnessAssessment,
    GeminiSemanticEvaluator,
    RelevanceAssessment,
)


class FakeModels:
    def __init__(self):
        self.responses = [
            FaithfulnessAssessment(
                claims=[
                    {"claim": "20 days of annual leave", "supported": True, "evidence": "20 days"},
                    {"claim": "leave is paid", "supported": False, "evidence": ""},
                ]
            ),
            RelevanceAssessment(
                score=0.9,
                rationale="Directly answers the number-of-days question.",
            ),
        ]
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.responses.pop(0), text="")


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_semantic_evaluator_returns_separate_metric_scores():
    client = FakeClient()
    evaluator = GeminiSemanticEvaluator(client=client)

    result = evaluator.evaluate(
        "How many annual leave days?",
        "Employees get 20 days of paid annual leave.",
        ["Full-time employees are entitled to 20 days of annual leave."],
    )

    assert result["faithfulness"] == 0.5
    assert result["faithfulness_supported_claims"] == 1
    assert result["faithfulness_total_claims"] == 2
    assert result["answer_relevancy"] == 0.9
    assert len(client.models.calls) == 2
    assert client.models.calls[0]["config"].response_schema is FaithfulnessAssessment
    assert client.models.calls[1]["config"].response_schema is RelevanceAssessment