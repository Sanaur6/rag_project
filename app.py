import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    import extra_streamlit_components as stx
except ModuleNotFoundError:
    stx = None

from src.loader import SUPPORTED_EXTENSIONS, load_documents
from src.auth import authenticate_user
from src.database import (
    create_login_session,
    list_document_records,
    load_conversation_history,
    revoke_login_session,
    save_conversation,
    save_feedback,
    sync_document_records,
    validate_login_session,
)
from src.rag_system import RAGSystem


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
AUTH_COOKIE_NAME = "company_knowledge_session"


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


def list_managed_documents(docs_folder: str | Path) -> list[Path]:
    folder = Path(docs_folder)
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def delete_managed_document(filename: str, docs_folder: str | Path) -> Path:
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Select a supported document from the documents folder.")

    document = Path(docs_folder) / safe_name
    if not document.is_file():
        raise ValueError("The document no longer exists.")
    document.unlink()
    return document


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

    if stx is None:
        st.error("Persistent sign-in requires extra-streamlit-components in this Python environment.")
        st.code(f'"{sys.executable}" -m pip install -r requirements.txt')
        st.stop()

    st.set_page_config(
        page_title="Company Knowledge Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --ink: #20332c;
            --muted: #66766e;
            --green: #176b52;
            --green-dark: #123d32;
            --mint: #e7f3ec;
            --line: #dce6df;
            --paper: #ffffff;
            --workspace-bg: linear-gradient(128deg, #d9eee2 0%, #e3eff1 52%, #f2e4d3 100%);
        }
        [data-testid="stApp"],
        [data-testid="stMain"] {
            background: var(--workspace-bg) !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #e3eee7;
            background-image:
                url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%20680%20420'%3E%3Cg%20fill='none'%20stroke='%23176b52'%20stroke-opacity='.2'%20stroke-width='1.4'%3E%3Cpath%20d='M80,100L190,70L280,140L390,95L510,160L620,110M190,70L180,190L280,140L300,280L420,260L510,160M80,100L100,250L180,190M390,95L420,260L560,330L620,110'/%3E%3C/g%3E%3Cg%20fill='%23176b52'%20fill-opacity='.28'%3E%3Ccircle%20cx='80'%20cy='100'%20r='3'/%3E%3Ccircle%20cx='190'%20cy='70'%20r='3'/%3E%3Ccircle%20cx='280'%20cy='140'%20r='3'/%3E%3Ccircle%20cx='390'%20cy='95'%20r='3'/%3E%3Ccircle%20cx='510'%20cy='160'%20r='3'/%3E%3Ccircle%20cx='620'%20cy='110'%20r='3'/%3E%3Ccircle%20cx='100'%20cy='250'%20r='3'/%3E%3Ccircle%20cx='180'%20cy='190'%20r='3'/%3E%3Ccircle%20cx='300'%20cy='280'%20r='3'/%3E%3Ccircle%20cx='420'%20cy='260'%20r='3'/%3E%3Ccircle%20cx='560'%20cy='330'%20r='3'/%3E%3C/g%3E%3C/svg%3E"),
                var(--workspace-bg);
            background-position: right top, center;
            background-size: min(760px, 76vw) 460px, cover;
            background-repeat: no-repeat;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1040px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] { background: var(--green-dark); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] label { color: #edf5ef; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #bdcec4; }
        [data-testid="stSidebar"] .stButton > button {
            color: #f4f8f4;
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            color: #ffffff;
            background: rgba(255, 255, 255, 0.16);
            border-color: #a7d3bd;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
        h1 { font-size: 2.25rem; line-height: 1.18; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stTextInput"] input {
            min-height: 3.15rem;
            color: var(--ink);
            background: var(--paper);
            border-color: var(--line);
            border-radius: 8px;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--green);
            box-shadow: 0 0 0 1px var(--green);
        }
        .stButton > button {
            min-height: 2.8rem;
            color: #ffffff;
            background: var(--green);
            border: 1px solid var(--green);
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover {
            color: #ffffff;
            background: #12563f;
            border-color: #12563f;
        }
        [data-testid="stAlert"] { border-radius: 8px; }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.72);
            border-color: var(--line);
            border-radius: 8px;
        }
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 3.9rem;
            margin-bottom: 2rem;
            padding: 0.55rem 0.85rem;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(18, 61, 50, 0.13);
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(27, 58, 45, 0.06);
        }
        .app-brand, .app-context {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .app-mark {
            display: grid;
            width: 2.45rem;
            height: 2.45rem;
            place-items: center;
            color: #ffffff;
            background: var(--green-dark);
            border-radius: 7px;
            font-size: 0.82rem;
            font-weight: 700;
        }
        .app-name {
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 700;
        }
        .app-context {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 700;
        }
        .workspace-dot {
            width: 0.55rem;
            height: 0.55rem;
            border-radius: 50%;
            background: #e39751;
        }
        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1.25rem 1rem 2rem;
            }
            h1 { font-size: 1.8rem; }
            .app-header { margin-bottom: 1.5rem; padding: 0.5rem 0.6rem; }
            .app-context { font-size: 0.6rem; }
            .app-brand, .app-context { gap: 0.4rem; }
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                min-width: 100% !important;
            }
        }
        </style>
        <header class="app-header" aria-label="Company Knowledge Assistant">
            <div class="app-brand"><span class="app-mark">AI</span><span class="app-name">Company Knowledge Assistant</span></div>
            <div class="app-context"><span class="workspace-dot"></span> KNOWLEDGE WORKSPACE</div>
        </header>
        """,
        unsafe_allow_html=True,
    )
    user_username = os.getenv("APP_USERNAME")
    user_password = os.getenv("APP_PASSWORD")
    admin_username = os.getenv("DOCS_ADMIN_USERNAME")
    admin_password = os.getenv("DOCS_ADMIN_PASSWORD")
    if not user_username or not user_password:
        st.error("Web sign-in is disabled because APP_USERNAME and APP_PASSWORD are not configured.")
        st.stop()

    docs_folder = Path(os.getenv("DOCS_FOLDER", "data/documents"))
    database_path = Path(os.getenv("APP_DATABASE_PATH", "data/assistant.sqlite3"))
    cookie_manager = stx.CookieManager(key="auth_cookie_manager")
    session_token = st.session_state.get("auth_session_token") or cookie_manager.get(AUTH_COOKIE_NAME)
    session_role = validate_login_session(session_token, database_path)
    if session_role:
        st.session_state.auth_session_token = session_token
        st.session_state.auth_role = session_role
    else:
        if cookie_manager.get(AUTH_COOKIE_NAME):
            cookie_manager.delete(AUTH_COOKIE_NAME, key="remove_invalid_auth_cookie")
        for key in (
            "auth_role",
            "auth_session_token",
            "conversation_role",
            "active_chat",
            "rag",
        ):
            st.session_state.pop(key, None)

    if not st.session_state.get("auth_role"):
        with st.form("sign_in"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")

        if submitted:
            role = authenticate_user(
                username,
                password,
                user_username,
                user_password,
                admin_username,
                admin_password,
            )
            if role:
                session_token, expires_at = create_login_session(role, database_path)
                cookie_secure = os.getenv("APP_COOKIE_SECURE", "").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                cookie_manager.set(
                    AUTH_COOKIE_NAME,
                    session_token,
                    expires_at=expires_at,
                    secure=cookie_secure,
                    same_site="strict",
                    key="set_auth_session",
                )
                st.session_state.auth_role = role
                st.session_state.auth_session_token = session_token
                st.rerun()
            st.error("Incorrect username or password.")
        st.stop()

    if "rag" not in st.session_state:
        st.session_state.rag = RAGSystem(docs_folder=str(docs_folder))
    role = st.session_state.auth_role
    if (
        st.session_state.get("conversation_role") != role
        or "active_chat" not in st.session_state
    ):
        st.session_state.active_chat = []
        st.session_state.conversation_role = role

    example_questions = [
        "What are the password requirements?",
        "How many days of leave do employees get?",
        "What is the remote work policy?",
    ]

    def render_assistant_page():
        rag = st.session_state.rag
        st.title("What can I help you find?")
        st.caption("Clear answers from your company documents.")

        st.markdown("#### Try a question")
        prompt_columns = st.columns(3)
        for column, question in zip(prompt_columns, example_questions):
            if column.button(question, key=f"sample_{question}", use_container_width=True):
                st.session_state.question = question
                st.session_state.submit_question = True

        question_column, submit_column = st.columns([6, 1], vertical_alignment="bottom")
        question = question_column.text_input(
            "Ask a question",
            key="question",
            placeholder="Ask about a policy, benefit, or workplace process...",
            label_visibility="collapsed",
        ) or ""
        submitted = submit_column.button("Ask", use_container_width=True)
        submit_sample = st.session_state.pop("submit_question", False)

        if (submitted or submit_sample) and question.strip():
            answer, sources = rag.answer(question, return_sources=True)
            conversation_id = save_conversation(
                role,
                question.strip(),
                answer,
                sources[:3],
                database_path,
            )
            st.session_state.active_chat.append(
                {
                    "id": conversation_id,
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources[:3],
                    "feedback": None,
                }
            )

        if st.session_state.active_chat:
            st.markdown("### Current chat")
            if st.button("New chat"):
                st.session_state.active_chat.clear()
                st.rerun()

            for entry in st.session_state.active_chat:
                st.markdown(f"**You:** {entry['question']}")
                st.markdown("**Assistant:**")
                st.info(entry["answer"])
                feedback = st.feedback("thumbs", key=f"answer_feedback_{entry['id']}")
                if feedback is not None and feedback != entry["feedback"]:
                    save_feedback(entry["id"], role, feedback, database_path)
                    entry["feedback"] = feedback
                if entry["feedback"] is not None:
                    st.caption("Thanks for your feedback.")

                if entry["sources"]:
                    st.markdown("Source citations")
                    for idx, source in enumerate(entry["sources"], start=1):
                        title = source.get("source", "Document")
                        text = (source.get("text") or "").strip()
                        snippet = " ".join(text.split())
                        if len(snippet) > 300:
                            snippet = snippet[:300].rstrip() + "..."

                        st.markdown(f"**{idx}. {title}**")
                        st.write(snippet)

                        if source.get("page") is not None:
                            st.caption(f"Page {source.get('page')}")

    def render_history_page():
        st.title("Chat history")
        st.caption("Search and review saved questions and answers.")
        history = load_conversation_history(role, database_path)
        if not history:
            st.info("Saved conversations will appear here.")
            return

        search_text = st.text_input(
            "Search history",
            placeholder="Find a question or answer...",
            key="history_search",
        ).strip().casefold()
        filtered_history = [
            entry
            for entry in history
            if not search_text
            or search_text in entry["question"].casefold()
            or search_text in entry["answer"].casefold()
        ]
        if not filtered_history:
            st.info("No saved questions match that search.")
            return

        rows = [
            {
                "Date": datetime.fromisoformat(entry["created_at"])
                .astimezone()
                .strftime("%Y-%m-%d %H:%M"),
                "Question": entry["question"],
                "Feedback": (
                    "Helpful"
                    if entry["feedback"] == 1
                    else "Not helpful"
                    if entry["feedback"] == 0
                    else "Not rated"
                ),
            }
            for entry in filtered_history
        ]
        table = st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="chat_history_table",
            column_config={
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Feedback": st.column_config.TextColumn("Feedback", width="small"),
            },
        )
        selection = table.get("selection", {})
        selected_rows = selection.get("rows", [])
        if not selected_rows:
            st.caption("Select a saved question to view its answer and sources.")
            return

        entry = filtered_history[selected_rows[0]]
        st.divider()
        st.subheader(entry["question"])
        st.info(entry["answer"])
        rating_text = (
            "Helpful"
            if entry["feedback"] == 1
            else "Not helpful"
            if entry["feedback"] == 0
            else "Not rated"
        )
        st.caption(f"Saved {datetime.fromisoformat(entry['created_at']).astimezone():%Y-%m-%d %H:%M} · Feedback: {rating_text}")

        feedback = st.feedback("thumbs", key=f"history_feedback_{entry['id']}")
        if feedback is not None and feedback != entry["feedback"]:
            save_feedback(entry["id"], role, feedback, database_path)
            st.rerun()

        if entry["sources"]:
            st.markdown("#### Source citations")
            for index, source in enumerate(entry["sources"], start=1):
                st.markdown(f"**{index}. {source.get('source', 'Document')}**")
                st.write(source.get("text", ""))
                if source.get("page") is not None:
                    st.caption(f"Page {source['page']}")

    def render_documents_page():
        st.title("Documents")
        st.caption("Manage the files used by the company knowledge assistant.")

        uploads = st.file_uploader(
            "Add documents",
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

        managed_documents = list_managed_documents(docs_folder)
        sync_document_records(managed_documents, database_path)
        document_records = list_document_records(database_path)
        managed_by_name = {document.name: document for document in managed_documents}
        st.subheader("Document library")
        if not document_records:
            st.info("No documents yet. Add a file above to start building the library.")
            return

        rows = []
        for record in document_records:
            rows.append(
                {
                    "Document": record["filename"],
                    "Type": record["file_type"],
                    "Size": f"{record['size_bytes'] / 1024:.1f} KB",
                    "Modified": datetime.fromisoformat(record["modified_at"])
                    .astimezone()
                    .strftime("%Y-%m-%d %H:%M"),
                }
            )

        table = st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="document_inventory",
            column_config={
                "Document": st.column_config.TextColumn("Document", width="large"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Size": st.column_config.TextColumn("Size", width="small"),
                "Modified": st.column_config.TextColumn("Last modified", width="medium"),
            },
        )

        selection = table.get("selection", {})
        selected_rows = selection.get("rows", [])
        if selected_rows:
            st.session_state.selected_document = document_records[selected_rows[0]]["filename"]
        selected_filename = st.session_state.get("selected_document")
        document = (
            managed_by_name.get(selected_filename)
            if isinstance(selected_filename, str)
            else None
        )
        if document is None:
            st.session_state.pop("selected_document", None)
            st.caption("Select a row to view document details and available actions.")
            return

        record = next(
            item for item in document_records if item["filename"] == document.name
        )
        st.divider()
        st.subheader(document.name)
        detail_column, action_column = st.columns([4, 1])
        with detail_column:
            metadata_columns = st.columns(3)
            metadata_columns[0].metric("File type", record["file_type"])
            metadata_columns[1].metric("File size", f"{record['size_bytes'] / 1024:.1f} KB")
            metadata_columns[2].metric(
                "Added to library",
                datetime.fromisoformat(record["added_at"])
                .astimezone()
                .strftime("%Y-%m-%d %H:%M"),
            )
            indexed_chunks = [
                chunk["text"]
                for chunk in st.session_state.rag.chunks
                if chunk.get("source") == document.name
            ]
            preview = "\n\n".join(indexed_chunks)[:4000]
            st.text_area(
                "Indexed text preview",
                value=preview or "No readable text was indexed for this document.",
                height=220,
                disabled=True,
            )

        with action_column:
            st.download_button(
                "Download",
                data=document.read_bytes(),
                file_name=document.name,
                width="stretch",
            )
            with st.popover("Delete document", use_container_width=True):
                st.warning(f"This will remove {document.name} from the library.")
                if st.button("Confirm delete", key=f"confirm_delete_{document.name}"):
                    try:
                        delete_managed_document(document.name, docs_folder)
                    except (OSError, ValueError) as error:
                        st.error(str(error))
                    else:
                        st.session_state.pop("selected_document", None)
                        st.session_state.rag = RAGSystem(docs_folder=str(docs_folder))
                        st.success("Document deleted and search index refreshed.")
                        st.rerun()

    pages = [
        st.Page(
            render_assistant_page,
            title="Assistant",
            icon=":material/chat_bubble_outline:",
            default=True,
        ),
        st.Page(
            render_history_page,
            title="History",
            icon=":material/history:",
            url_path="history",
        ),
    ]
    if st.session_state.auth_role == "admin":
        pages.append(
            st.Page(
                render_documents_page,
                title="Documents",
                icon=":material/folder_open:",
                url_path="documents",
            )
        )
    navigation = st.navigation(pages, position="sidebar", expanded=True)

    with st.sidebar:
        st.caption(f"Signed in as {st.session_state.auth_role}")
        if st.button("Sign out"):
            revoke_login_session(
                st.session_state.get("auth_session_token"),
                database_path,
            )
            if cookie_manager.get(AUTH_COOKIE_NAME):
                cookie_manager.delete(AUTH_COOKIE_NAME, key="delete_auth_session")
            st.session_state.pop("auth_role", None)
            st.session_state.pop("auth_session_token", None)
            st.session_state.pop("rag", None)
            st.session_state.pop("active_chat", None)
            st.session_state.pop("conversation_role", None)
            st.rerun()
    navigation.run()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit()
    elif "streamlit" in sys.modules:
        run_streamlit()
    else:
        run_cli()
