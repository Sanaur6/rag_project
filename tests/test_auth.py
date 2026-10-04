from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import app
from src.auth import authenticate_user, token_matches
from src.database import (
    create_login_session,
    load_conversation_history,
    save_conversation,
    validate_login_session,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_token_matches_requires_configured_matching_value():
    assert token_matches("secure-value", "secure-value")
    assert not token_matches("incorrect", "secure-value")
    assert not token_matches("anything", None)
    assert not token_matches(None, "secure-value")


def test_authenticate_user_distinguishes_regular_and_admin_roles():
    credentials = ("viewer", "viewer-password", "admin", "admin-password")

    assert authenticate_user("viewer", "viewer-password", *credentials) == "user"
    assert authenticate_user("admin", "admin-password", *credentials) == "admin"
    assert authenticate_user("viewer", "wrong", *credentials) is None
    assert authenticate_user("unknown", "viewer-password", *credentials) is None


def test_api_returns_service_unavailable_when_token_is_unconfigured(client, monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)

    response = client.get("/")

    assert response.status_code == 503


def test_api_rejects_missing_and_incorrect_bearer_tokens(client, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secure-value")

    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer incorrect"}).status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer incorrect"}).status_code == 401
    assert client.post(
        "/ask",
        headers={"Authorization": "Bearer incorrect"},
        json={"question": "policy question"},
    ).status_code == 401


def test_api_accepts_matching_bearer_token(client, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_TOKEN", "secure-value")

    response = client.get("/", headers={"Authorization": "Bearer secure-value"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_streamlit_fails_closed_without_user_credentials(monkeypatch):
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("DOCS_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("DOCS_ADMIN_PASSWORD", raising=False)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=90).run(timeout=90)

    assert any(
        "APP_USERNAME and APP_PASSWORD are not configured" in element.value
        for element in app_test.error
    )


def test_streamlit_restores_cookie_session_and_revokes_it_on_sign_out(
    monkeypatch, tmp_path
):
    docs_folder = tmp_path / "documents"
    docs_folder.mkdir()
    (docs_folder / "policy.txt").write_text("Policy details.", encoding="utf-8")
    database_path = tmp_path / "assistant.sqlite3"
    monkeypatch.setenv("DOCS_FOLDER", str(docs_folder))
    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("APP_USERNAME", "employee")
    monkeypatch.setenv("APP_PASSWORD", "employee-password")
    monkeypatch.setenv("DOCS_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("DOCS_ADMIN_PASSWORD", "admin-password")

    session_token, _ = create_login_session("user", database_path)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=90)
    app_test.session_state["auth_cookie_manager"] = {
        "company_knowledge_session": session_token
    }
    app_test.run(timeout=90)

    assert not app_test.exception
    assert app_test.session_state["auth_role"] == "user"
    assert validate_login_session(session_token, database_path) == "user"

    next(button for button in app_test.button if button.label == "Sign out").click().run(
        timeout=90
    )

    assert validate_login_session(session_token, database_path) is None
    assert "auth_role" not in app_test.session_state


def test_saved_history_does_not_appear_in_the_current_chat(monkeypatch, tmp_path):
    docs_folder = tmp_path / "documents"
    docs_folder.mkdir()
    (docs_folder / "policy.txt").write_text("Policy details.", encoding="utf-8")
    database_path = tmp_path / "assistant.sqlite3"
    monkeypatch.setenv("DOCS_FOLDER", str(docs_folder))
    monkeypatch.setenv("APP_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("APP_USERNAME", "employee")
    monkeypatch.setenv("APP_PASSWORD", "employee-password")
    monkeypatch.setenv("DOCS_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("DOCS_ADMIN_PASSWORD", "admin-password")
    save_conversation("user", "A saved question", "A saved answer", [], database_path)
    session_token, _ = create_login_session("user", database_path)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(app_path, default_timeout=90)
    app_test.session_state["auth_cookie_manager"] = {
        "company_knowledge_session": session_token
    }
    app_test.run(timeout=90)

    assert not app_test.exception
    assert app_test.session_state["active_chat"] == []
    assert any(link.label == "History" for link in app_test.get("page_link"))
    assert [entry["question"] for entry in load_conversation_history("user", database_path)] == [
        "A saved question"
    ]
