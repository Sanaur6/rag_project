import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_DATABASE_PATH = "data/assistant.sqlite3"
VALID_ROLES = {"user", "admin"}
SESSION_LIFETIME = timedelta(days=7)

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    filename TEXT PRIMARY KEY,
    file_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at TEXT NOT NULL,
    added_at TEXT NOT NULL,
    synced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    conversation_id INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL CHECK (rating IN (0, 1)),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS login_sessions (
    token_hash TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError("Role must be 'user' or 'admin'.")


def create_login_session(
    role: str,
    database_path: str | Path | None = None,
    lifetime: timedelta = SESSION_LIFETIME,
) -> tuple[str, datetime]:
    _validate_role(role)
    if lifetime.total_seconds() <= 0:
        raise ValueError("Session lifetime must be positive.")

    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + lifetime
    with _connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO login_sessions (token_hash, role, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, role, expires_at.isoformat(), created_at.isoformat()),
        )
    return token, expires_at


def validate_login_session(
    token: str | None, database_path: str | Path | None = None
) -> str | None:
    if not token:
        return None

    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connection(database_path) as connection:
        session = connection.execute(
            "SELECT role, expires_at FROM login_sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if session is None:
            return None

        if datetime.fromisoformat(session["expires_at"]) <= datetime.now(timezone.utc):
            connection.execute(
                "DELETE FROM login_sessions WHERE token_hash = ?", (token_hash,)
            )
            return None
        return str(session["role"])


def revoke_login_session(token: str | None, database_path: str | Path | None = None) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _connection(database_path) as connection:
        connection.execute("DELETE FROM login_sessions WHERE token_hash = ?", (token_hash,))


@contextmanager
def _connection(database_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    configured_path = database_path or os.getenv("APP_DATABASE_PATH", DEFAULT_DATABASE_PATH)
    path = Path(configured_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def sync_document_records(
    documents: list[Path], database_path: str | Path | None = None
) -> None:
    synced_at = _utc_now()
    filenames: set[str] = set()

    with _connection(database_path) as connection:
        for document in documents:
            file_stat = document.stat()
            filenames.add(document.name)
            connection.execute(
                """
                INSERT INTO documents (
                    filename, file_type, size_bytes, modified_at, added_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    file_type = excluded.file_type,
                    size_bytes = excluded.size_bytes,
                    modified_at = excluded.modified_at,
                    synced_at = excluded.synced_at
                """,
                (
                    document.name,
                    document.suffix.lstrip(".").upper(),
                    file_stat.st_size,
                    datetime.fromtimestamp(file_stat.st_mtime, timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    synced_at,
                    synced_at,
                ),
            )

        existing_filenames = {
            row["filename"] for row in connection.execute("SELECT filename FROM documents")
        }
        removed_filenames = existing_filenames - filenames
        connection.executemany(
            "DELETE FROM documents WHERE filename = ?",
            ((filename,) for filename in removed_filenames),
        )


def list_document_records(database_path: str | Path | None = None) -> list[dict[str, Any]]:
    with _connection(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY filename COLLATE NOCASE"
        ).fetchall()
    return [dict(row) for row in rows]


def save_conversation(
    role: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    database_path: str | Path | None = None,
) -> int:
    _validate_role(role)
    with _connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (role, question, answer, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (role, question, answer, json.dumps(sources), _utc_now()),
        )
        return int(cursor.lastrowid)


def load_conversation_history(
    role: str, database_path: str | Path | None = None
) -> list[dict[str, Any]]:
    _validate_role(role)
    with _connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT conversations.*, feedback.rating
            FROM conversations
            LEFT JOIN feedback ON feedback.conversation_id = conversations.id
            WHERE conversations.role = ?
            ORDER BY conversations.id
            """,
            (role,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": json.loads(row["sources_json"]),
            "feedback": row["rating"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_feedback(
    conversation_id: int,
    role: str,
    rating: int,
    database_path: str | Path | None = None,
) -> None:
    _validate_role(role)
    if rating not in (0, 1):
        raise ValueError("Feedback rating must be 0 or 1.")

    with _connection(database_path) as connection:
        conversation = connection.execute(
            "SELECT id FROM conversations WHERE id = ? AND role = ?",
            (conversation_id, role),
        ).fetchone()
        if conversation is None:
            raise ValueError("Conversation was not found for this role.")
        connection.execute(
            """
            INSERT INTO feedback (conversation_id, rating, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                rating = excluded.rating,
                updated_at = excluded.updated_at
            """,
            (conversation_id, rating, _utc_now()),
        )


def clear_conversation_history(role: str, database_path: str | Path | None = None) -> None:
    _validate_role(role)
    with _connection(database_path) as connection:
        connection.execute("DELETE FROM conversations WHERE role = ?", (role,))