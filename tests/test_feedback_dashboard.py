from app import build_benchmark_drafts, summarize_employee_feedback


def test_summarize_employee_feedback_counts_and_normalizes_questions():
    entries = [
        {
            "question": " Remote work policy? ",
            "feedback": 0,
            "feedback_reason": "Outdated",
            "created_at": "2026-01-01T12:00:00+00:00",
        },
        {
            "question": "remote  work policy?",
            "feedback": 1,
            "created_at": "2026-01-02T12:00:00+00:00",
        },
        {
            "question": "Password requirements?",
            "feedback": None,
            "created_at": "2026-01-03T12:00:00+00:00",
        },
    ]

    summary = summarize_employee_feedback(entries)

    assert summary["total_responses"] == 3
    assert summary["rated_responses"] == 2
    assert summary["helpful_responses"] == 1
    assert summary["unhelpful_entries"] == [entries[0]]
    assert summary["reason_counts"] == {"Outdated": 1}
    assert summary["common_questions"] == [
        {"question": "Remote work policy?", "count": 2},
        {"question": "Password requirements?", "count": 1},
    ]


def test_summarize_employee_feedback_handles_no_conversations():
    summary = summarize_employee_feedback([])

    assert summary["total_responses"] == 0
    assert summary["rated_responses"] == 0
    assert summary["helpful_responses"] == 0
    assert summary["unhelpful_entries"] == []
    assert summary["reason_counts"] == {}
    assert summary["common_questions"] == []


def test_build_benchmark_drafts_only_includes_unhelpful_with_review_placeholders():
    entries = [
        {
            "id": 12,
            "question": "What about managers?",
            "feedback": 0,
            "sources": [
                {"source": "company_info.txt"},
                {"source": "company_info.txt"},
                {"source": "leave_policy.txt"},
            ],
        },
        {"id": 13, "question": "Thanks", "feedback": 1, "sources": []},
        {"id": 14, "question": "No rating", "feedback": None, "sources": []},
    ]

    assert build_benchmark_drafts(entries) == [
        {
            "id": "feedback_12",
            "question": "What about managers?",
            "must_have": ["TODO: replace with required answer facts"],
            "expected_sources": ["company_info.txt", "leave_policy.txt"],
        }
    ]