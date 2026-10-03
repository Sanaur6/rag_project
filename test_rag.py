import pytest

from app import store_uploaded_document
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
