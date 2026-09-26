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
    st.caption("Ask questions using the company documents in this project.")

    rag = RAGSystem(docs_folder="data/documents")

    example_questions = [
        "What are the password requirements?",
        "How many days of leave do employees get?",
        "What is the remote work policy?",
    ]

    with st.sidebar:
        st.header("Quick questions")
        for question in example_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.question = question

    question = st.text_input("Ask a question", value=st.session_state.get("question", ""))

    if st.button("Ask") or question:
        answer, sources = rag.answer(question, return_sources=True)

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("### Relevant sources")
            for idx, source in enumerate(sources[:3], start=1):
                with st.container():
                    st.markdown(f"**{idx}. {source.get('source', 'Document')}**")
                    st.write(source["text"][:500])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit()
    elif "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli()
