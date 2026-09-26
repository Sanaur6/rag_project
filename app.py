import os
import sys

from src.rag_system import RAGSystem


def run_cli():
    rag = RAGSystem(docs_folder="data/documents")
    print("Company RAG Assistant is ready.")
    print("Ask questions about company policies. Type 'q' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        answer, sources = rag.answer(question, return_sources=True)
        print("\nAnswer:")
        print(answer)

        if sources:
            print("\nRelevant sources:")
            for idx, source in enumerate(sources[:3], start=1):
                print(f"{idx}. {source.get('source', 'Document')}")


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚")
    st.title("Company Knowledge Assistant")
    st.caption("Ask questions using the company documents in the project folder.")

    rag = RAGSystem(docs_folder="data/documents")
    question = st.text_input("Ask a question")

    if question:
        answer, sources = rag.answer(question, return_sources=True)
        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("### Relevant snippets")
            for idx, source in enumerate(sources, start=1):
                st.markdown(f"**{idx}. {source.get('source', 'Document')}**")
                st.write(source["text"][:500])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit()
    else:
        run_cli()
