from src.loader import load_documents


def test_load_documents_reads_company_files():
    documents = load_documents("data/documents")

    assert len(documents) >= 4
    sources = {doc["source"] for doc in documents}
    assert "it_security_policy.txt" in sources
    assert any("Password Requirements" in doc["text"] for doc in documents)
    assert any("leave" in doc["text"].lower() for doc in documents)
