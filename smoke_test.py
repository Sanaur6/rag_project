from src.rag_system import RAGSystem


def main() -> None:
    rag = RAGSystem(docs_folder="data/documents", top_k=3)
    sample_questions = [
        "What are the password requirements?",
        "How many days of leave do employees get?",
        "What is the remote work arrangement?",
    ]

    print("Smoke test: starting RAG initialization")
    print(f"Loaded chunks: {len(rag.chunks)}")
    print(f"Vector dimension: {rag.embeddings.shape[1] if rag.embeddings.size else 0}")

    for question in sample_questions:
        answer, sources = rag.answer(question, return_sources=True)
        print(f"Q: {question}")
        print(f"A: {answer[:120].replace(chr(10), ' ')}")
        print(f"Sources: {[source.get('source') for source in sources[:3]]}")

    print("Smoke test: completed successfully")


if __name__ == "__main__":
    main()
