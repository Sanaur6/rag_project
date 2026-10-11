import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import extra_streamlit_components as stx
except ModuleNotFoundError:
    stx = None

from src.loader import SUPPORTED_EXTENSIONS, load_documents
from src.auth import authenticate_user
from src.database import (
    FEEDBACK_REASONS,
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


def build_benchmark_drafts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"feedback_{entry['id']}",
            "question": entry["question"],
            "must_have": ["TODO: replace with required answer facts"],
            "expected_sources": sorted(
                {
                    source.get("source")
                    for source in entry.get("sources", []) or []
                    if source.get("source")
                }
            ),
        }
        for entry in entries
        if entry["feedback"] == 0
    ]


def build_chat_markdown(entries: list[dict[str, Any]]) -> str:
    lines = ["# Company Knowledge Assistant conversation"]
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                "",
                f"## Question {index}",
                "",
                entry["question"],
                "",
                "## Answer",
                "",
                entry["answer"],
            ]
        )
        sources = entry.get("sources") or []
        if sources:
            lines.extend(["", "### Sources"])
            for source in sources:
                source_name = source.get("source") or "Document"
                page = source.get("page")
                label = f"{source_name} (page {page})" if page is not None else source_name
                lines.append(f"- {label}")
                snippet = " ".join((source.get("text") or "").split())
                if snippet:
                    lines.append(f"  > {snippet}")
    return "\n".join(lines).strip() + "\n"


def summarize_employee_feedback(entries: list[dict[str, Any]]) -> dict[str, Any]:
    question_counts: Counter[str] = Counter()
    question_labels: dict[str, str] = {}
    for entry in entries:
        normalized_question = " ".join(entry["question"].casefold().split())
        question_counts[normalized_question] += 1
        question_labels.setdefault(normalized_question, entry["question"].strip())

    rated_entries = [entry for entry in entries if entry["feedback"] is not None]
    unhelpful_entries = [entry for entry in rated_entries if entry["feedback"] == 0]
    common_questions = [
        {"question": question_labels[question], "count": count}
        for question, count in sorted(
            question_counts.items(),
            key=lambda item: (-item[1], question_labels[item[0]].casefold()),
        )
    ]
    return {
        "total_responses": len(entries),
        "rated_responses": len(rated_entries),
        "helpful_responses": sum(entry["feedback"] == 1 for entry in rated_entries),
        "unhelpful_entries": unhelpful_entries,
        "reason_counts": dict(
            Counter(
                entry["feedback_reason"]
                for entry in unhelpful_entries
                if entry.get("feedback_reason")
            )
        ),
        "common_questions": common_questions,
    }


def summarize_policy_gaps(entries: list[dict[str, Any]]) -> dict[str, Any]:
    topic_counts: Counter[str] = Counter()
    topic_reason_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def infer_topic(question: str) -> str:
        normalized = " ".join(question.casefold().split())
        if any(keyword in normalized for keyword in ("password", "vpn", "mfa", "security", "phishing", "api key", "device")):
            return "Security & access"
        if any(keyword in normalized for keyword in ("leave", "annual leave", "maternity", "paternity", "sick leave", "casual")):
            return "Leave & time off"
        if any(keyword in normalized for keyword in ("remote", "work from home", "hybrid")):
            return "Remote work"
        if any(keyword in normalized for keyword in ("benefit", "salary", "insurance", "compensation")):
            return "Benefits & pay"
        return "General policy guidance"

    for entry in entries:
        if entry.get("feedback") != 0:
            continue
        topic = infer_topic(str(entry.get("question") or ""))
        topic_counts[topic] += 1
        reason = entry.get("feedback_reason") or "Unclear"
        topic_reason_counts[topic][reason] += 1

    hotspots = [
        {
            "topic": topic,
            "count": count,
            "most_common_reason": topic_reason_counts[topic].most_common(1)[0][0],
            "reason_counts": dict(topic_reason_counts[topic]),
        }
        for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "hotspots": hotspots,
        "issue_counts": dict(
            Counter(
                entry.get("feedback_reason")
                for entry in entries
                if entry.get("feedback") == 0 and entry.get("feedback_reason")
            )
        ),
    }


def build_weekly_policy_report(entries: list[dict[str, Any]]) -> dict[str, Any]:
    flagged_entries = [entry for entry in entries if entry.get("feedback") == 0]
    gap_summary = summarize_policy_gaps(entries)
    question_counts: Counter[str] = Counter()
    for entry in flagged_entries:
        normalized = " ".join(str(entry.get("question") or "").casefold().split())
        question_counts[normalized] += 1

    repeated_questions = [
        {"question": question, "count": count}
        for question, count in sorted(
            question_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]

    recommended_actions = []
    for hotspot in gap_summary["hotspots"][:3]:
        topic = hotspot["topic"]
        if topic == "Security & access":
            recommended_actions.append(
                "Review the password, VPN, and MFA guidance to ensure the process is explicit and easy to follow."
            )
        elif topic == "Leave & time off":
            recommended_actions.append(
                "Clarify the leave request process, approval timeline, and emergency exceptions for employees and managers."
            )
        elif topic == "Remote work":
            recommended_actions.append(
                "Refresh the remote-work policy with a clearer approval path and explicit rules for client meetings."
            )
        else:
            recommended_actions.append(
                f"Review the {topic.lower()} guidance and add examples that reduce ambiguity for employees."
            )

    if not recommended_actions:
        recommended_actions.append(
            "Review the most frequent low-confidence questions and add policy examples that answer them directly."
        )

    return {
        "report_title": "Weekly policy gap report",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_flagged_questions": len(flagged_entries),
        "top_topics": gap_summary["hotspots"][:5],
        "repeated_questions": repeated_questions,
        "recommended_actions": recommended_actions,
    }


def build_policy_update_recommendations(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gap_summary = summarize_policy_gaps(entries)
    recommendations: list[dict[str, Any]] = []

    for hotspot in gap_summary["hotspots"][:5]:
        topic = hotspot["topic"]
        if "Security" in topic:
            document = "it_security_policy.txt"
            suggested_update = "Clarify the required password rules, VPN expectations, MFA enforcement, and escalation path for security concerns."
        elif "Leave" in topic:
            document = "leave_policy.txt"
            suggested_update = "Add step-by-step leave submission instructions, emergency exceptions, and manager approval expectations."
        elif "Remote" in topic:
            document = "company_info.txt"
            suggested_update = "Define the remote-work approval flow, daily/weekly limits, and rules for critical meetings more explicitly."
        elif "Benefits" in topic:
            document = "company_info.txt"
            suggested_update = "Add examples for benefit eligibility, reimbursement limits, and approval timelines to reduce confusion."
        else:
            document = "company_info.txt"
            suggested_update = "Add concrete examples and decision points to make the policy easier to interpret and follow."

        recommendations.append(
            {
                "topic": topic,
                "document": document,
                "issue": hotspot["most_common_reason"],
                "suggested_update": suggested_update,
                "flagged_count": hotspot["count"],
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "topic": "General policy guidance",
                "document": "company_info.txt",
                "issue": "Unclear",
                "suggested_update": "Add a short FAQ and a real-world example section to improve interpretation and reduce ambiguity.",
                "flagged_count": 0,
            }
        )
    return recommendations


def build_document_freshness_summary(
    documents: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issue_counts: Counter[str] = Counter()
    issue_reasons: dict[str, Counter[str]] = defaultdict(Counter)

    def infer_document(question: str) -> str:
        normalized = " ".join(question.casefold().split())
        if any(keyword in normalized for keyword in ("password", "vpn", "mfa", "security", "phishing", "api key", "device")):
            return "it_security_policy.txt"
        if any(keyword in normalized for keyword in ("leave", "annual leave", "maternity", "paternity", "sick leave", "casual")):
            return "leave_policy.txt"
        if any(keyword in normalized for keyword in ("remote", "work from home", "hybrid")):
            return "company_info.txt"
        if any(keyword in normalized for keyword in ("benefit", "salary", "insurance", "compensation")):
            return "company_info.txt"
        return "company_info.txt"

    for entry in entries:
        if entry.get("feedback") != 0:
            continue
        document_name = infer_document(str(entry.get("question") or ""))
        issue_counts[document_name] += 1
        issue_reasons[document_name][entry.get("feedback_reason") or "Unclear"] += 1

    latest_documents = []
    for document in documents:
        filename = document.get("filename")
        if not filename:
            continue
        modified_at = document.get("modified_at")
        age_days = None
        if modified_at:
            try:
                modified = datetime.fromisoformat(str(modified_at).replace("Z", "+00:00"))
                age_days = max(0, (datetime.now(timezone.utc) - modified).days)
            except ValueError:
                age_days = None

        flagged_count = issue_counts.get(filename, 0)
        reason = issue_reasons.get(filename, Counter()).most_common(1)[0][0] if issue_reasons.get(filename) else "Unclear"
        if flagged_count >= 2 and age_days is not None and age_days > 180:
            risk_level = "high"
        elif flagged_count >= 1 or (age_days is not None and age_days > 180):
            risk_level = "medium"
        elif age_days is not None and age_days > 365:
            risk_level = "medium"
        else:
            risk_level = "low"

        if flagged_count == 0 and age_days is None:
            continue

        latest_documents.append(
            {
                "document": filename,
                "flagged_count": flagged_count,
                "last_updated_days_ago": age_days,
                "risk_level": risk_level,
                "top_issue": reason,
                "recommendation": (
                    "Update this policy and add examples to reduce repeated ambiguity."
                    if flagged_count
                    else "Review this policy for stale or outdated language and refresh it with current guidance."
                ),
            }
        )

    latest_documents.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}[item["risk_level"]],
            -item["flagged_count"],
            -(item["last_updated_days_ago"] or 0),
            item["document"],
        )
    )
    return latest_documents


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
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stMarkdownContainer"] p {
            line-height: 1.58;
        }
        [data-testid="stSidebar"] { background: var(--green-dark); }
        [data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 4rem);
        }
        [data-testid="stSidebarUserContent"] > div {
            display: flex;
            flex: 1;
            flex-direction: column;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] label { color: #edf5ef; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #bdcec4; }
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] a p,
        [data-testid="stSidebar"] a span,
        [data-testid="stSidebar"] a svg {
            color: #edf5ef !important;
            fill: currentColor;
        }
        [data-testid="stSidebar"] a {
            margin: 0.15rem 0;
            padding: 0.2rem 0.5rem;
            border-radius: 7px;
        }
        [data-testid="stSidebar"] a:hover {
            background: rgba(255, 255, 255, 0.12);
        }
        [data-testid="stSidebar"] a[aria-current="page"] {
            background: rgba(255, 255, 255, 0.18);
        }
        [data-testid="stSidebar"] .st-key-sidebar-footer {
            position: sticky;
            bottom: 0;
            margin-top: auto;
            padding-top: 1rem;
            background: var(--green-dark);
            border-top: 1px solid rgba(255, 255, 255, 0.16);
        }
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
        h1 { margin-bottom: 0.45rem; font-size: 2.25rem; line-height: 1.18; }
        h2 { margin-top: 1.6rem; font-size: 1.45rem; line-height: 1.3; }
        h3 { margin-top: 1.25rem; font-size: 1.12rem; line-height: 1.35; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            min-height: 3.15rem;
            color: var(--ink);
            background: var(--paper);
            border-color: var(--line);
            border-radius: 8px;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--green);
            box-shadow: 0 0 0 1px var(--green);
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 3rem;
            color: var(--ink);
            background: var(--paper);
            border-color: var(--line);
            border-radius: 8px;
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
        .stDownloadButton > button {
            min-height: 2.8rem;
            color: var(--green-dark);
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            font-weight: 600;
        }
        .stDownloadButton > button:hover {
            color: var(--green-dark);
            background: #ffffff;
            border-color: var(--green);
        }
        [data-testid="stAlert"] {
            border: 1px solid rgba(18, 61, 50, 0.12);
            border-radius: 8px;
            background-color: rgba(255, 255, 255, 0.82);
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.72);
            border-color: var(--line);
            border-radius: 8px;
        }
        [data-testid="stMetric"] {
            min-height: 6rem;
            padding: 0.85rem 1rem;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(18, 61, 50, 0.12);
            border-radius: 8px;
            box-shadow: 0 5px 16px rgba(27, 58, 45, 0.04);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        [data-testid="stDataFrame"] {
            overflow: hidden;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(18, 61, 50, 0.14);
            border-radius: 8px;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.82);
            border-color: rgba(18, 61, 50, 0.24);
            border-radius: 8px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--green);
            background: rgba(255, 255, 255, 0.96);
        }
        [data-testid="stForm"] {
            max-width: 460px;
            margin: 1rem auto 0;
            padding: 1.35rem;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(18, 61, 50, 0.14);
            border-radius: 10px;
            box-shadow: 0 12px 32px rgba(27, 58, 45, 0.08);
        }
        [data-testid="stForm"] .stButton > button { width: 100%; }
        .st-key-signin-intro {
            max-width: 460px;
            margin: 3rem auto 0;
        }
        .st-key-history-answer {
            max-width: 960px;
            margin: 1rem auto 0;
            padding: 1.1rem;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(18, 61, 50, 0.13);
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
        .st-key-suggested-questions,
        .st-key-chat-actions {
            max-width: 960px;
            margin: 0 auto;
        }
        .st-key-suggested-questions .stButton > button {
            min-height: 4rem;
            padding: 0.7rem 0.9rem;
            color: var(--ink);
            background: rgba(255, 255, 255, 0.9);
            border-color: rgba(18, 61, 50, 0.18);
            text-align: left;
            white-space: normal;
        }
        .st-key-suggested-questions .stButton > button:hover {
            color: var(--green-dark);
            background: #ffffff;
            border-color: var(--green);
        }
        .st-key-suggested-questions .stButton > button p {
            line-height: 1.4;
            text-align: left;
        }
        .st-key-chat-actions .stButton > button,
        .st-key-chat-actions .stDownloadButton > button {
            color: var(--green-dark);
            background: rgba(255, 255, 255, 0.86);
            border-color: rgba(18, 61, 50, 0.2);
        }
        .st-key-chat-actions .stButton > button:hover,
        .st-key-chat-actions .stDownloadButton > button:hover {
            color: var(--green-dark);
            background: #ffffff;
            border-color: var(--green);
        }
        .st-key-chat-transcript {
            max-width: 960px;
            margin: 0 auto;
        }
        .st-key-chat-transcript [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.88);
            border-color: rgba(18, 61, 50, 0.16);
            border-radius: 10px;
            box-shadow: 0 12px 32px rgba(27, 58, 45, 0.08);
        }
        .st-key-chat-transcript [data-testid="stChatMessage"] {
            border: 1px solid rgba(18, 61, 50, 0.1);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            padding: 0.85rem 1rem;
        }
        .st-key-chat-transcript [data-testid="stChatMessage"] p {
            color: var(--ink);
            line-height: 1.65;
            overflow-wrap: anywhere;
        }
        .st-key-chat-transcript [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }
        [data-testid="stChatInput"] {
            max-width: 960px;
            margin: 0 auto;
        }
        [data-testid="stChatInput"] textarea {
            min-height: 3.5rem;
            color: var(--ink);
            background: var(--paper);
            border-color: var(--line);
            border-radius: 12px;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--green);
            box-shadow: 0 0 0 1px var(--green);
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
            [data-testid="stMainBlockContainer"] [data-testid="stMetric"] {
                min-height: 0;
            }
            [data-testid="stForm"] {
                margin-top: 0.75rem;
                padding: 1rem;
            }
            .st-key-signin-intro { margin-top: 1.5rem; }
            .st-key-chat-transcript [data-testid="stChatMessage"] {
                padding: 0.65rem 0.75rem;
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
        with st.container(key="signin-intro"):
            st.title("Welcome back")
            st.caption("Sign in to your company knowledge workspace.")
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
    role_options = [
        "employee",
        "manager",
        "hr",
        "it",
    ]

    def render_feedback_controls(entry, widget_key_prefix):
        rating = st.feedback("thumbs", key=f"{widget_key_prefix}{entry['id']}")
        selected_rating = rating if rating is not None else entry["feedback"]
        previous_reason = entry.get("feedback_reason")
        selected_reason = previous_reason
        if selected_rating == 0:
            reason_options = ["Choose a reason (optional)", *FEEDBACK_REASONS]
            reason_index = (
                reason_options.index(previous_reason)
                if previous_reason in FEEDBACK_REASONS
                else 0
            )
            reason_choice = st.selectbox(
                "What could be improved?",
                reason_options,
                index=reason_index,
                key=f"feedback_reason_{widget_key_prefix}{entry['id']}",
            )
            selected_reason = (
                reason_choice if reason_choice in FEEDBACK_REASONS else None
            )
        else:
            selected_reason = None

        if selected_rating is not None and (
            selected_rating != entry["feedback"]
            or selected_reason != previous_reason
        ):
            save_feedback(
                entry["id"],
                role,
                selected_rating,
                database_path,
                reason=selected_reason,
            )
            entry["feedback"] = selected_rating
            entry["feedback_reason"] = selected_reason

    def render_assistant_page():
        rag = st.session_state.rag
        if st.session_state.active_chat:
            st.title("Company Knowledge Assistant")
            st.caption("Ask a follow-up to continue the conversation.")
        else:
            st.title("What can I help you find?")
            st.caption("Clear answers from your company documents.")
            st.markdown("#### Try a question")
            with st.container(key="suggested-questions"):
                prompt_columns = st.columns(3)
                for column, sample_question in zip(prompt_columns, example_questions):
                    if column.button(
                        sample_question,
                        key=f"sample_{sample_question}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = sample_question

        if "selected_user_role" not in st.session_state:
            st.session_state.selected_user_role = "employee"
        user_role = st.session_state.selected_user_role
        st.session_state.selected_user_role = st.selectbox(
            "I am asking as",
            role_options,
            index=role_options.index(user_role) if user_role in role_options else 0,
            key="assistant_user_role",
        )
        question = st.chat_input(
            "Ask a follow-up question..."
            if st.session_state.active_chat
            else "Ask about a policy, benefit, or workplace process...",
            key="chat_input",
        )
        pending_question = st.session_state.pop("pending_question", None)
        question = pending_question or question

        if question and question.strip():
            user_role = st.session_state.get("selected_user_role", "employee")
            answer, sources, action_plan = rag.answer(
                question,
                return_sources=True,
                conversation_history=st.session_state.active_chat[-3:],
                include_action_plan=True,
                user_role=user_role,
            )
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
                    "action_plan": action_plan,
                }
            )

        if st.session_state.active_chat:
            st.markdown("### Conversation")
            with st.container(key="chat-actions"):
                action_columns = st.columns(2)
                if action_columns[0].button("New chat", use_container_width=True):
                    st.session_state.active_chat.clear()
                    st.rerun()
                action_columns[1].download_button(
                    "Download conversation",
                    data=build_chat_markdown(st.session_state.active_chat),
                    file_name="company-knowledge-chat.md",
                    mime="text/markdown",
                    width="stretch",
                )

            with st.container(height=560, border=True, key="chat-transcript"):
                for entry in st.session_state.active_chat:
                    with st.chat_message("user"):
                        st.markdown(entry["question"])

                    with st.chat_message("assistant"):
                        st.markdown(entry["answer"])
                        action_plan = entry.get("action_plan") or {}
                        next_steps = action_plan.get("next_steps") or []
                        if next_steps:
                            st.markdown("### Recommended next steps")
                            for idx, step in enumerate(next_steps, start=1):
                                st.write(f"{idx}. {step}")
                            if action_plan.get("requires_human_review"):
                                st.caption("This may require manager, HR, or IT confirmation.")

                        render_feedback_controls(entry, "answer_feedback_")
                        if entry["feedback"] is not None:
                            st.caption("Thanks for your feedback.")

                        if entry["sources"]:
                            with st.expander("Source citations"):
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
        with st.container(key="history-answer"):
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

            render_feedback_controls(entry, "history_feedback_")
            if entry.get("feedback_reason"):
                st.caption(f"Reported issue: {entry['feedback_reason']}")

            if entry["sources"]:
                st.markdown("#### Source citations")
                for index, source in enumerate(entry["sources"], start=1):
                    st.markdown(f"**{index}. {source.get('source', 'Document')}**")
                    st.write(source.get("text", ""))
                    if source.get("page") is not None:
                        st.caption(f"Page {source['page']}")

    def render_feedback_page():
        st.title("Employee feedback")
        st.caption("Review answer ratings and identify recurring questions that need attention.")
        entries = load_conversation_history("user", database_path)
        summary = summarize_employee_feedback(entries)
        gap_summary = summarize_policy_gaps(entries)
        if not summary["total_responses"]:
            st.info("Employee conversations will appear here once questions have been asked.")
            return

        columns = st.columns(4)
        columns[0].metric("Employee questions", summary["total_responses"])
        columns[1].metric("Rated answers", summary["rated_responses"])
        helpful_value = (
            f"{summary['helpful_responses']} "
            f"({summary['helpful_responses'] / summary['rated_responses']:.0%})"
            if summary["rated_responses"]
            else "No ratings"
        )
        columns[2].metric("Helpful", helpful_value)
        columns[3].metric("Needs review", len(summary["unhelpful_entries"]))
        if summary["reason_counts"]:
            reason_summary = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(summary["reason_counts"].items())
            )
            st.caption(f"Reported issues: {reason_summary}")

        if gap_summary["hotspots"]:
            st.subheader("Escalation insights")
            hotspot_rows = [
                {
                    "Policy area": item["topic"],
                    "Flagged": item["count"],
                    "Top issue": item["most_common_reason"],
                }
                for item in gap_summary["hotspots"][:5]
            ]
            st.dataframe(
                hotspot_rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "Policy area": st.column_config.TextColumn("Policy area", width="large"),
                    "Flagged": st.column_config.NumberColumn("Flagged", width="small"),
                    "Top issue": st.column_config.TextColumn("Top issue", width="medium"),
                },
            )

            weekly_report = build_weekly_policy_report(entries)
            update_recommendations = build_policy_update_recommendations(entries)
            st.download_button(
                "Download weekly policy gap report",
                data=json.dumps(weekly_report, indent=2, ensure_ascii=False),
                file_name="weekly_policy_gap_report.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Recommended actions: " + "; ".join(weekly_report["recommended_actions"]))

            st.subheader("Suggested document updates")
            for item in update_recommendations:
                with st.expander(f"{item['topic']} → {item['document']}"):
                    st.write(f"Issue: {item['issue']}")
                    st.write(f"Flagged {item['flagged_count']} times")
                    st.write(item["suggested_update"])

        document_records = list_document_records(database_path)
        freshness_review = build_document_freshness_summary(document_records, entries)
        if freshness_review:
            st.subheader("Policy drift review")
            freshness_rows = [
                {
                    "Document": item["document"],
                    "Flagged": item["flagged_count"],
                    "Days since update": item["last_updated_days_ago"],
                    "Risk": item["risk_level"].title(),
                    "Top issue": item["top_issue"],
                }
                for item in freshness_review[:5]
            ]
            st.dataframe(
                freshness_rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "Document": st.column_config.TextColumn("Document", width="large"),
                    "Flagged": st.column_config.NumberColumn("Flagged", width="small"),
                    "Days since update": st.column_config.NumberColumn("Days since update", width="small"),
                    "Risk": st.column_config.TextColumn("Risk", width="small"),
                    "Top issue": st.column_config.TextColumn("Top issue", width="medium"),
                },
            )

        st.subheader("Most asked questions")
        question_rows = [
            {"Question": item["question"], "Asked": item["count"]}
            for item in summary["common_questions"][:10]
        ]
        st.dataframe(
            question_rows,
            hide_index=True,
            width="stretch",
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Asked": st.column_config.NumberColumn("Asked", width="small"),
            },
        )

        st.subheader("Answers marked not helpful")
        unhelpful_entries = sorted(
            summary["unhelpful_entries"],
            key=lambda entry: entry["created_at"],
            reverse=True,
        )
        if not unhelpful_entries:
            st.info("No employee answers have been marked not helpful.")
            return
        if len(unhelpful_entries) > 20:
            st.caption("Showing the 20 most recent responses.")
        st.download_button(
            "Download benchmark drafts",
            data=json.dumps(
                build_benchmark_drafts(unhelpful_entries[:20]),
                indent=2,
                ensure_ascii=False,
            ),
            file_name="benchmark_drafts.json",
            mime="application/json",
        )
        st.caption(
            "Draft only: replace the required-facts placeholder and verify suggested sources before adding cases to evaluation_cases.json."
        )

        for entry in unhelpful_entries[:20]:
            created_at = datetime.fromisoformat(entry["created_at"]).astimezone()
            label = f"{entry['question']} · {created_at:%Y-%m-%d %H:%M}"
            with st.expander(label):
                if entry.get("feedback_reason"):
                    st.caption(f"Reported issue: {entry['feedback_reason']}")
                st.markdown("**Answer**")
                st.write(entry["answer"])
                if entry["sources"]:
                    st.markdown("**Retrieved sources**")
                    for source in entry["sources"]:
                        st.caption(source.get("source", "Document"))
                        snippet = " ".join((source.get("text") or "").split())
                        if len(snippet) > 300:
                            snippet = snippet[:300].rstrip() + "..."
                        st.write(snippet)

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
        pages.append(
            st.Page(
                render_feedback_page,
                title="Feedback",
                icon=":material/analytics:",
                url_path="feedback",
            )
        )
    navigation = st.navigation(pages, position="hidden")
    with st.sidebar:
        st.markdown("### Knowledge Assistant")
        for page in pages:
            st.page_link(page, width="stretch")
        with st.container(key="sidebar-footer"):
            st.caption(f"Signed in as {st.session_state.auth_role}")
            if st.button("Sign out", use_container_width=True):
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
