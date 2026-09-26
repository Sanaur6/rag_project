from src.rag_system import RAGSystem


def test_rag_returns_relevant_sources_for_password_question():
    rag = RAGSystem(docs_folder="data/documents", top_k=3)

    answer, sources = rag.answer("What are the password requirements?", return_sources=True)

    assert answer
    assert any("password" in source["text"].lower() for source in sources)
    assert any("strong" in source["text"].lower() for source in sources)


def test_rag_handles_unknown_question_without_crashing():
    rag = RAGSystem(docs_folder="data/documents", top_k=3)

    answer, sources = rag.answer("What is the capital of Mars?", return_sources=True)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert sources == [] or isinstance(sources, list)
