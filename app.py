import os
import sys
import tempfile
from pathlib import Path

from src.rag_system import RAGSystem
from src.loader import SUPPORTED_EXTENSIONS, load_documents


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def store_uploaded_document(filename: str, content: bytes, docs_folder: str | Path) -> Path:
    safe_name = Path(filename).name
    destination_folder = Path(docs_folder)
    extension = Path(safe_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported file types are TXT, TEXT, PDF, and DOCX.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be 10 MB or smaller.")

    with tempfile.TemporaryDirectory() as temporary_folder:
        staged_file = Path(temporary_folder) / safe_name
        staged_file.write_bytes(content)
        try:
            parsed_documents = load_documents(temporary_folder)
        except Exception as error:
            raise ValueError("The uploaded file could not be parsed.") from error
        if not parsed_documents:
            raise ValueError("No readable text was found in the uploaded file.")

    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / safe_name
    destination.write_bytes(content)
    return destination


def show_sources(sources):
    if not sources:
        return

    print("\nRelevant sources:")
    for idx, source in enumerate(sources[:3], start=1):
        title = source.get("source", "Document")
        text = (source.get("text") or "").strip()
        snippet = " ".join(text.split())
        if len(snippet) > 260:
            snippet = snippet[:260].rstrip() + "..."
        print(f"{idx}. {title}")
        print(snippet)
        print()


def run_cli():
    rag = RAGSystem(docs_folder=os.getenv("DOCS_FOLDER", "data/documents"))
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
        show_sources(sources)


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Company Knowledge Assistant", page_icon="📚")
    st.title("Company Knowledge Assistant")
    st.caption("Ask questions using the company documents in this project.")

    docs_folder = Path(os.getenv("DOCS_FOLDER", "data/documents"))
    if "rag" not in st.session_state:
        st.session_state.rag = RAGSystem(docs_folder=str(docs_folder))
    rag = st.session_state.rag

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

        st.divider()
        st.subheader("Add documents")
        uploads = st.file_uploader(
            "Upload TXT, PDF, or DOCX files",
            type=["txt", "text", "pdf", "docx"],
            accept_multiple_files=True,
            key="document_uploads",
        )
        if st.button("Upload and re-index", disabled=not uploads):
            uploaded_count = 0
            for uploaded_file in uploads or []:
                try:
                    store_uploaded_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        docs_folder,
                    )
                    uploaded_count += 1
                except (OSError, ValueError) as error:
                    st.error(f"{uploaded_file.name}: {error}")

            if uploaded_count:
                st.session_state.rag = RAGSystem(docs_folder=str(docs_folder))
                st.success(f"Added {uploaded_count} document(s); the search index is refreshed.")
                st.rerun()

    question = st.text_input("Ask a question", value=st.session_state.get("question", "")) or ""

    if (st.button("Ask") or question) and question.strip():
        answer, sources = rag.answer(question, return_sources=True)

        st.markdown("### Answer")
        st.info(answer)

        if sources:
            st.markdown("### Source citations")
            for idx, source in enumerate(sources[:3], start=1):
                title = source.get("source", "Document")
                text = (source.get("text") or "").strip()
                snippet = " ".join(text.split())
                if len(snippet) > 300:
                    snippet = snippet[:300].rstrip() + "..."

                with st.container():
                    st.markdown(f"**{idx}. {title}**")
                    st.write(snippet)

                    if source.get("page") is not None:
                        st.caption(f"Page {source.get('page')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit()
    elif "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli()
