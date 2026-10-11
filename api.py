import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.auth import authenticate_user, token_matches
from src.database import (
    VALID_ROLES,
    create_login_session,
    list_document_records,
    load_conversation_history,
    revoke_login_session,
    save_feedback,
    validate_login_session,
)
from src.rag_system import RAGSystem


app = FastAPI(
    title="Company RAG Assistant API",
    version="1.0.0",
    description="HTTP API for querying company policy documents.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
bearer_scheme = HTTPBearer(auto_error=False)
AUTH_COOKIE_NAME = "company_knowledge_session"


def get_database_path() -> str:
    return os.getenv("APP_DATABASE_PATH", "data/assistant.sqlite3")


def get_request_token(request: Request) -> str | None:
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if token:
        return token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1]
    return None


def get_session_role(request: Request) -> str | None:
    token = get_request_token(request)
    if not token:
        return None
    return validate_login_session(token, get_database_path())


def require_access_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    token = credentials.credentials if credentials else get_request_token(request)
    session_role = validate_login_session(token, get_database_path()) if token else None

    if session_role:
        return token

    expected_token = os.getenv("APP_ACCESS_TOKEN")
    if expected_token and token_matches(token, expected_token):
        return token

    if not expected_token and not token:
        raise HTTPException(status_code=503, detail="API access is not configured.")

    raise HTTPException(
        status_code=401,
        detail="A valid session or bearer token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG system.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve.")
    user_role: str | None = Field(
        default=None,
        description="Optional audience for tailor-made policy guidance: employee, manager, hr, or it.",
    )


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    conversation_id: int = Field(..., ge=1)
    role: str = Field(...)
    rating: int = Field(..., ge=0, le=1)
    reason: str | None = None


def summarize_employee_feedback(entries: list[dict[str, Any]]) -> dict[str, Any]:
    question_counts: Counter[str] = Counter()
    question_labels: dict[str, str] = {}
    for entry in entries:
        normalized_question = " ".join(str(entry["question"]).casefold().split())
        question_counts[normalized_question] += 1
        question_labels.setdefault(normalized_question, str(entry["question"]).strip())

    rated_entries = [entry for entry in entries if entry.get("feedback") is not None]
    unhelpful_entries = [entry for entry in rated_entries if entry.get("feedback") == 0]
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
        "helpful_responses": sum(entry.get("feedback") == 1 for entry in rated_entries),
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


def get_rag() -> RAGSystem:
    docs_folder = os.getenv("DOCS_FOLDER", "data/documents")
    return RAGSystem(docs_folder=docs_folder)


@app.get("/", dependencies=[Depends(require_access_token)])
def root() -> dict[str, Any]:
    return {
        "name": "Company RAG Assistant API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", dependencies=[Depends(require_access_token)])
def health() -> dict[str, Any]:
    try:
        rag = get_rag()
        return {
            "status": "ok",
            "documents_loaded": len(rag.chunks),
            "top_k": rag.top_k,
        }
    except Exception as exc:  # pragma: no cover - defensive health check
        return {"status": "error", "detail": str(exc)}


@app.get("/session")
def session(request: Request) -> dict[str, str | None]:
    role = get_session_role(request)
    return {"role": role}


@app.post("/login")
def login(request: LoginRequest, response: Response) -> dict[str, str]:
    user_username = os.getenv("APP_USERNAME")
    user_password = os.getenv("APP_PASSWORD")
    admin_username = os.getenv("DOCS_ADMIN_USERNAME")
    admin_password = os.getenv("DOCS_ADMIN_PASSWORD")

    if not (user_username and user_password):
        raise HTTPException(status_code=503, detail="Browser authentication is not configured.")

    role = authenticate_user(
        request.username,
        request.password,
        user_username,
        user_password,
        admin_username,
        admin_password,
    )
    if not role:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    if role not in VALID_ROLES:
        raise HTTPException(status_code=401, detail="Unsupported role.")

    token, expires_at = create_login_session(role, get_database_path())
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        expires=expires_at,
        httponly=True,
        samesite="lax",
        secure=False,
    )
    return {"status": "ok", "role": role}


@app.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    token = get_request_token(request)
    if token:
        revoke_login_session(token, get_database_path())
    response.delete_cookie(key=AUTH_COOKIE_NAME, httponly=True, samesite="lax")
    return {"status": "ok"}


@app.get("/history")
def history(request: Request, role: str = "user") -> list[dict[str, Any]]:
    session_role = get_session_role(request)
    if session_role not in {role, "admin"}:
        raise HTTPException(status_code=403, detail="Not allowed to access that role history.")
    return load_conversation_history(role, get_database_path())


@app.get("/feedback-summary")
def feedback_summary(request: Request, role: str = "user") -> dict[str, Any]:
    session_role = get_session_role(request)
    if session_role not in {role, "admin"}:
        raise HTTPException(status_code=403, detail="Not allowed to access that feedback summary.")
    entries = load_conversation_history(role, get_database_path())
    return {
        "summary": summarize_employee_feedback(entries),
        "gap_summary": summarize_policy_gaps(entries),
    }


@app.post("/feedback")
def update_feedback(request: FeedbackRequest, http_request: Request) -> dict[str, str]:
    session_role = get_session_role(http_request)
    if session_role not in {request.role, "admin"}:
        raise HTTPException(status_code=403, detail="Not allowed to update that feedback.")
    save_feedback(
        request.conversation_id,
        request.role,
        request.rating,
        get_database_path(),
        reason=request.reason if request.rating == 0 else None,
    )
    return {"status": "ok"}


@app.get("/documents")
def documents(request: Request) -> list[dict[str, Any]]:
    session_role = get_session_role(request)
    if session_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access is required.")
    return list_document_records(get_database_path())


@app.post("/documents/upload")
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    session_role = get_session_role(request)
    if session_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access is required.")

    docs_folder = Path(os.getenv("DOCS_FOLDER", "data/documents"))
    docs_folder.mkdir(parents=True, exist_ok=True)
    destination = docs_folder / Path(file.filename).name
    content = await file.read()
    destination.write_bytes(content)
    return {"status": "ok", "filename": destination.name}


@app.delete("/documents/{filename}")
def delete_document(request: Request, filename: str) -> dict[str, str]:
    session_role = get_session_role(request)
    if session_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access is required.")

    docs_folder = Path(os.getenv("DOCS_FOLDER", "data/documents"))
    target = docs_folder / filename
    if target.exists():
        target.unlink()
    return {"status": "ok"}


@app.post("/ask", dependencies=[Depends(require_access_token)])
def ask(request: AskRequest) -> dict[str, Any]:
    rag = get_rag()
    answer, sources, action_plan = rag.answer(
        request.question,
        top_k=request.top_k,
        return_sources=True,
        include_action_plan=True,
        user_role=request.user_role,
    )
    return {
        "answer": answer,
        "action_plan": action_plan,
        "sources": [
            {
                "source": source.get("source", "unknown"),
                "page": source.get("page"),
                "text": source.get("text", ""),
                "score": source.get("score"),
            }
            for source in sources
        ],
    }
