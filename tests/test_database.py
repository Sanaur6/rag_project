import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.database import (
    clear_conversation_history,
    create_login_session,
    list_document_records,
    load_conversation_history,
    revoke_login_session,
    save_conversation,
    save_feedback,
    sync_document_records,
    validate_login_session,
)


def test_document_metadata_tracks_files_and_removes_missing_records(tmp_path):
    database_path = tmp_path / "nested" / "assistant.sqlite3"
    document = tmp_path / "policy.txt"
    document.write_text("Policy", encoding="utf-8")

    sync_document_records([document], database_path)
    original_record = list_document_records(database_path)[0]
    assert original_record["filename"] == "policy.txt"
    assert original_record["file_type"] == "TXT"
    assert original_record["size_bytes"] == len("Policy")

    document.write_text("Updated policy", encoding="utf-8")
    sync_document_records([document], database_path)
    updated_record = list_document_records(database_path)[0]
    assert updated_record["size_bytes"] == len("Updated policy")
    assert updated_record["added_at"] == original_record["added_at"]

    sync_document_records([], database_path)
    assert list_document_records(database_path) == []


def test_conversations_and_feedback_persist_by_role(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    user_id = save_conversation(
        "user",
        "What is the leave policy?",
        "Employees receive paid leave.",
        [{"source": "leave_policy.txt", "text": "Paid leave"}],
        database_path,
    )
    save_conversation("admin", "Admin question", "Admin answer", [], database_path)
    save_feedback(user_id, "user", 1, database_path)

    user_history = load_conversation_history("user", database_path)
    assert len(user_history) == 1
    assert user_history[0]["id"] == user_id
    assert user_history[0]["sources"] == [{"source": "leave_policy.txt", "text": "Paid leave"}]
    assert user_history[0]["feedback"] == 1
    assert [entry["question"] for entry in load_conversation_history("admin", database_path)] == [
        "Admin question"
    ]

    clear_conversation_history("user", database_path)
    assert load_conversation_history("user", database_path) == []
    assert len(load_conversation_history("admin", database_path)) == 1


def test_feedback_cannot_be_saved_for_another_role(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    conversation_id = save_conversation("user", "Question", "Answer", [], database_path)

    with pytest.raises(ValueError, match="not found for this role"):
        save_feedback(conversation_id, "admin", 0, database_path)

    with pytest.raises(ValueError, match="must be 0 or 1"):
        save_feedback(conversation_id, "user", 2, database_path)


def test_feedback_reason_is_persisted_and_validated(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    conversation_id = save_conversation("user", "Question", "Answer", [], database_path)

    save_feedback(
        conversation_id,
        "user",
        0,
        database_path,
        reason="Outdated",
    )

    history = load_conversation_history("user", database_path)
    assert history[0]["feedback"] == 0
    assert history[0]["feedback_reason"] == "Outdated"

    with pytest.raises(ValueError, match="not recognized"):
        save_feedback(conversation_id, "user", 0, database_path, reason="Confusing")
    with pytest.raises(ValueError, match="only allowed for not helpful"):
        save_feedback(conversation_id, "user", 1, database_path, reason="Unclear")


def test_existing_feedback_table_is_migrated_without_losing_ratings(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE feedback (
                conversation_id INTEGER PRIMARY KEY,
                rating INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO conversations
                VALUES (1, 'user', 'Question', 'Answer', '[]', '2026-01-01T00:00:00+00:00');
            INSERT INTO feedback VALUES (1, 0, '2026-01-01T00:00:00+00:00');
            """
        )
        connection.commit()
    finally:
        connection.close()

    history = load_conversation_history("user", database_path)
    assert history[0]["feedback"] == 0
    assert history[0]["feedback_reason"] is None

    save_feedback(1, "user", 0, database_path, reason="Incomplete")
    assert load_conversation_history("user", database_path)[0]["feedback_reason"] == "Incomplete"


def test_login_session_token_is_hashed_and_revocable(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    token, expires_at = create_login_session("admin", database_path)

    assert expires_at > datetime.now(timezone.utc)
    assert validate_login_session(token, database_path) == "admin"

    connection = sqlite3.connect(database_path)
    try:
        stored_hash = connection.execute(
            "SELECT token_hash FROM login_sessions"
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored_hash != token

    revoke_login_session(token, database_path)
    assert validate_login_session(token, database_path) is None


def test_expired_login_session_is_rejected_and_removed(tmp_path):
    database_path = tmp_path / "assistant.sqlite3"
    token, _ = create_login_session("user", database_path)
    expired_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE login_sessions SET expires_at = ?",
            (expired_at,),
        )
        connection.commit()
    finally:
        connection.close()

    assert validate_login_session(token, database_path) is None
    connection = sqlite3.connect(database_path)
    try:
        remaining_sessions = connection.execute(
            "SELECT COUNT(*) FROM login_sessions"
        ).fetchone()[0]
    finally:
        connection.close()
    assert remaining_sessions == 0