from typing import Any, cast
from types import SimpleNamespace

import pytest

from app import (
    build_chat_markdown,
    delete_managed_document,
    list_managed_documents,
    store_uploaded_document,
)
from src.rag_system import RAGSystem


def test_rag_returns_relevant_sources_for_password_question():
    rag = RAGSystem(docs_folder="data/documents", top_k=3)

    answer, sources = rag.answer("What are the password requirements?", return_sources=True)

    assert answer
    assert any("password" in source["text"].lower() for source in sources)
    assert any("strong" in source["text"].lower() for source in sources)


def test_rag_prefers_remote_work_section_for_remote_work_questions():
    rag = RAGSystem(docs_folder="data/documents", top_k=3)

    answer, sources = rag.answer("What is the remote work arrangement?", return_sources=True)

    assert "remote work" in answer.lower()
    assert "two days per week" in answer.lower()
    assert any("company_info" in source["source"].lower() for source in sources)


def test_rag_handles_unknown_question_without_crashing():
    rag = RAGSystem(docs_folder="data/documents", top_k=3)

    answer, sources = rag.answer("What is the capital of Mars?", return_sources=True)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert sources == [] or isinstance(sources, list)


def test_chat_markdown_export_keeps_answers_and_citations():
    markdown = build_chat_markdown(
        [
            {
                "question": "How many leave days are available?",
                "answer": "Employees receive 20 days of annual leave.",
                "sources": [
                    {
                        "source": "leave_policy.txt",
                        "page": 2,
                        "text": "Full-time employees receive 20 days of paid annual leave.",
                    }
                ],
            }
        ]
    )

    assert "How many leave days are available?" in markdown
    assert "Employees receive 20 days of annual leave." in markdown
    assert "leave_policy.txt (page 2)" in markdown
    assert "Full-time employees receive 20 days of paid annual leave." in markdown


def test_follow_up_question_uses_recent_context_for_retrieval():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    retrieval_questions = []
    rag.query = lambda question, top_k=None: (
        retrieval_questions.append((question, top_k)) or []
    )

    rag.answer(
        "What about managers?",
        conversation_history=[
            {
                "question": "What is the remote work policy?",
                "answer": "Employees may work remotely two days per week.",
            }
        ],
    )

    assert retrieval_questions == [("What is the remote work policy? What about managers?", None)]


def test_standalone_question_does_not_inherit_chat_context():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    retrieval_questions = []
    rag.query = lambda question, top_k=None: (
        retrieval_questions.append((question, top_k)) or []
    )

    rag.answer(
        "How many days of annual leave do employees get?",
        conversation_history=[
            {
                "question": "What are the password requirements?",
                "answer": "Passwords must meet the policy requirements.",
            }
        ],
    )

    assert retrieval_questions == [
        ("How many days of annual leave do employees get?", None)
    ]


def test_follow_up_context_is_included_in_answer_prompt(monkeypatch):
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    rag.api_key = "test-key"
    retrieval_questions = []
    rag.query = lambda question, top_k=None: (
        retrieval_questions.append((question, top_k))
        or [
            {
                "source": "company_info.txt",
                "text": "Managers may work remotely two days per week.",
            }
        ]
    )
    prompts = []
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: (
                prompts.append(kwargs["contents"]) or SimpleNamespace(text="Two days per week.")
            )
        )
    )
    def create_client(api_key):
        assert api_key == "test-key"
        return client

    monkeypatch.setattr("src.rag_system.genai.Client", create_client)

    answer, sources = rag.answer(
        "What about managers?",
        return_sources=True,
        conversation_history=[
            {
                "question": "What is the remote work policy?",
                "answer": "Employees may work remotely two days per week.",
            }
        ],
    )

    assert answer == "Two days per week."
    assert sources[0]["source"] == "company_info.txt"
    assert retrieval_questions == [("What is the remote work policy? What about managers?", None)]
    assert "What is the remote work policy?" in prompts[0]
    assert "do not treat it as factual evidence" in prompts[0]
    assert "Question:\nWhat about managers?" in prompts[0]


def test_fallback_answer_prefers_the_most_relevant_source_chunk():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    sources = [
        {
            "source": "company_info.txt",
            "text": (
                "Employees work eight hours per day. Employees may work remotely two days per week. "
                "Employee benefits include health insurance, paid annual leave, and sick leave."
            ),
        },
        {
            "source": "leave_policy.txt",
            "text": "Annual Leave: Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
        },
    ]

    answer = rag._fallback_answer("How many days of leave do employees get?", sources)

    assert "20 days" in answer


def test_reciprocal_rank_fusion_promotes_candidates_found_by_both_retrievers():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    candidates = rag._fuse_ranked_results(
        [
            [
                {"chunk_index": 0, "score": 10.0},
                {"chunk_index": 1, "score": 9.0},
            ],
            [
                {"chunk_index": 1, "score": 0.9},
                {"chunk_index": 2, "score": 0.8},
            ],
        ]
    )

    assert [candidate["chunk_index"] for candidate in candidates] == [1, 0, 2]


def test_optional_gemini_reranker_reorders_fused_candidates():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    rag.api_key = "test-key"
    rag.use_llm_reranking = True
    rag._gemini_client = cast(Any, SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                parsed={"candidate_indices": [1, 0]},
                text="",
            )
        )
    ))
    candidates = [
        {"chunk_index": 0, "text": "Generic policy overview.", "source": "overview.txt"},
        {"chunk_index": 1, "text": "Annual leave is 20 days.", "source": "leave.txt"},
    ]

    reranked = rag._rerank_candidates("How many days of annual leave?", candidates)

    assert [candidate["source"] for candidate in reranked] == ["leave.txt", "overview.txt"]


def test_uploaded_document_is_searchable_after_reindexing(tmp_path):
    document_text = "Travel reimbursement requires manager approval."
    store_uploaded_document("travel_policy.txt", document_text.encode("utf-8"), tmp_path)

    rag = RAGSystem(docs_folder=str(tmp_path), use_semantic_embeddings=False)
    results = rag.query("travel reimbursement approval")

    assert results
    assert results[0]["source"] == "travel_policy.txt"
    assert "manager approval" in results[0]["text"].lower()


def test_upload_rejects_unsupported_and_empty_files(tmp_path):
    with pytest.raises(ValueError, match="Supported file types"):
        store_uploaded_document("policy.exe", b"content", tmp_path)

    with pytest.raises(ValueError, match="empty"):
        store_uploaded_document("empty.txt", b"", tmp_path)

    with pytest.raises(ValueError, match="could not be parsed"):
        store_uploaded_document("broken.pdf", b"not a PDF file", tmp_path)


def test_action_plan_for_leave_requests():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    plan = rag.build_action_plan(
        "How do I request annual leave?",
        [
            {
                "source": "leave_policy.txt",
                "text": "Employees should submit leave requests through the HR portal at least three working days before the planned leave whenever possible.",
            }
        ],
    )

    assert plan["intent"] == "leave_request"
    assert "HR portal" in plan["next_steps"][0]
    assert plan["confidence"] == "high"


def test_action_plan_for_security_requests():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    plan = rag.build_action_plan(
        "How do I secure my work account?",
        [
            {
                "source": "it_security_policy.txt",
                "text": "All employees must use strong passwords. Multi-factor authentication is mandatory for email, VPN, and cloud infrastructure.",
            }
        ],
    )

    assert plan["intent"] == "security_action"
    assert any("MFA" in step or "multi-factor" in step.lower() for step in plan["next_steps"])
    assert plan["confidence"] == "high"


def test_role_aware_action_plan_tailors_guidance_for_managers():
    rag = RAGSystem(docs_folder="data/documents", use_semantic_embeddings=False)
    plan = rag.build_action_plan(
        "How do I approve leave?",
        [
            {
                "source": "leave_policy.txt",
                "text": "Employees should submit leave requests through the HR portal at least three working days before the planned leave whenever possible.",
            }
        ],
        user_role="manager",
    )

    assert plan["intent"] == "leave_request"
    assert any("manager" in step.lower() for step in plan["next_steps"])
    assert plan["audience"] == "manager"


def test_policy_gap_summary_groups_repeat_issues_by_topic():
    summary = __import__("app").summarize_policy_gaps(
        [
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "What are the VPN password rules?", "feedback": 0, "feedback_reason": "Incorrect"},
        ]
    )

    assert summary["hotspots"][0]["topic"] in {"Remote work", "Security & access"}
    assert summary["hotspots"][0]["count"] >= 2
    assert any(item["topic"] == "Security & access" for item in summary["hotspots"])


def test_weekly_policy_gap_report_summarizes_issues_for_admins():
    report = __import__("app").build_weekly_policy_report(
        [
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "What are the VPN password rules?", "feedback": 0, "feedback_reason": "Incorrect"},
        ]
    )

    assert report["report_title"] == "Weekly policy gap report"
    assert report["total_flagged_questions"] == 3
    assert report["top_topics"][0]["topic"] in {"Remote work", "Security & access"}
    assert report["recommended_actions"]


def test_policy_update_recommendations_point_to_the_right_document():
    recommendations = __import__("app").build_policy_update_recommendations(
        [
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "What are the VPN password rules?", "feedback": 0, "feedback_reason": "Incorrect"},
        ]
    )

    assert recommendations
    assert recommendations[0]["document"] in {"company_info.txt", "it_security_policy.txt"}
    assert recommendations[0]["suggested_update"]


def test_document_freshness_summary_flags_stale_policy_documents():
    summary = __import__("app").build_document_freshness_summary(
        [
            {"filename": "company_info.txt", "modified_at": "2023-01-01T00:00:00+00:00"},
            {"filename": "it_security_policy.txt", "modified_at": "2025-01-01T00:00:00+00:00"},
        ],
        [
            {"question": "How do I approve remote work?", "feedback": 0, "feedback_reason": "Unclear"},
            {"question": "What are the VPN password rules?", "feedback": 0, "feedback_reason": "Incorrect"},
        ],
    )

    assert summary[0]["document"] == "company_info.txt"
    assert summary[0]["risk_level"] in {"high", "medium"}
    assert any(item["document"] == "it_security_policy.txt" for item in summary)


def test_document_management_lists_supported_files_only(tmp_path):
    (tmp_path / "policy.txt").write_text("Policy", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Notes", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    assert [path.name for path in list_managed_documents(tmp_path)] == ["policy.txt"]


def test_document_management_deletes_only_supported_files_in_folder(tmp_path):
    document = tmp_path / "policy.txt"
    document.write_text("Policy", encoding="utf-8")

    assert delete_managed_document("policy.txt", tmp_path) == document
    assert not document.exists()

    with pytest.raises(ValueError, match="supported document"):
        delete_managed_document("../outside.txt", tmp_path)
