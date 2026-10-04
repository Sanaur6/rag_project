# Company RAG Assistant

This project is a small document-based RAG system for company policies and employee information. It loads text documents, chunks them into searchable sections, retrieves relevant matches, and answers user questions using the retrieved context.

## Features

- Document ingestion from TXT, PDF, and DOCX files
- chunk-based retrieval with keyword scoring
- hybrid keyword/vector retrieval fused with reciprocal-rank fusion
- optional Gemini listwise reranking for the strongest final passages
- optional Gemini embedding-based retrieval when a valid API key is configured
- source citation display in the CLI and Streamlit app
- validated TXT, TEXT, PDF, and DOCX uploads with immediate re-indexing in Streamlit
- SQLite persistence for conversation history, answer feedback, and document metadata
- evaluation benchmark for measuring answer quality

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Set your API key if you want Gemini-powered retrieval:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. Configure browser accounts and an API token. Copy `.env.example` to `.env`, then set strong, private passwords for the employee and admin accounts and generate a random API token. `.env` is ignored by Git.

   ```powershell
   Copy-Item .env.example .env
   ```

   Sign into Streamlit with `APP_USERNAME` and `APP_PASSWORD`. The separate `DOCS_ADMIN_USERNAME` and `DOCS_ADMIN_PASSWORD` account has upload and re-index access. `APP_ACCESS_TOKEN` is for API bearer authentication only. The browser sign-in fails closed if the employee account is not configured; API routes fail closed if the API token is missing.

4. Run a smoke check before launching the app:

   ```bash
   python smoke_test.py
   ```

5. Run the CLI app:

   ```bash
   python app.py
   ```

6. Run the Streamlit app:

   ```bash
   streamlit run app.py -- --streamlit
   ```

   On Windows, run `run_app.bat` from the project folder to ensure the app starts with the project's `myenv` environment.

In Streamlit, sign in with the employee account to use the assistant. The **Assistant** page shows the current chat; use **New chat** to clear that view without deleting saved entries. Open **History** to search and review saved questions, answers, and feedback. Admins also have a **Documents** page to upload TXT, TEXT, PDF, or DOCX files, inspect details, download files, and remove outdated documents. Uploads are re-indexed immediately, limited to 10 MB each, and parsed before storage. Files are stored in `data/documents` by default; set `DOCS_FOLDER` to use another folder. SQLite is created at `data/assistant.sqlite3` by default; set `APP_DATABASE_PATH` to use another location. Chat history and feedback are shared by account role, so anyone using the shared employee login can see that role's saved history. Use individual accounts before storing private conversations.

Browser sign-in survives reloads for up to seven days using a revocable session cookie; **Sign out** invalidates it immediately. Set `APP_COOKIE_SECURE=true` when serving over HTTPS. The cookie is managed in the browser and is not HttpOnly, so keep this shared-account app on a trusted network; use an identity provider and per-user accounts for public or sensitive deployments.

The browser uses shared local accounts rather than individual user identities or a full role-management system. The CLI remains local and does not require sign-in. For deployment beyond a trusted environment, add per-user identity/authorization and HTTPS at a trusted reverse proxy; do not expose this development server directly to the internet.

## Tests

```bash
pytest -q
```

## Evaluation benchmark

```bash
python benchmark_rag.py
```

This script loads the cases in `evaluation_cases.json`, runs the RAG on each question, and prints a pass/fail summary with a pass rate.

The report separates answer-term coverage from retrieved-context coverage and expected-source recall@3. Required source files are configured per case in `evaluation_cases.json`. The CLI quality gate defaults to 100% passing cases, 95% answer-term coverage, and 95% expected-source recall; override thresholds to experiment. GitHub Actions runs the tests and a deterministic, no-Gemini quality gate on pushes and pull requests to `main`, then uploads the JSON report.

Retrieval combines keyword and vector rankings with reciprocal-rank fusion. To enable a final Gemini listwise rerank, set `GEMINI_RERANKER_ENABLED=true` and configure `GEMINI_API_KEY`; this makes one additional Gemini request per query. Reranking is disabled by default, and the RRF path works without an API key.

To run the same CI thresholds locally:

```bash
python benchmark_rag.py \
   --min-pass-rate 100 \
   --min-answer-coverage 95 \
   --min-source-recall 100 \
   --output reports/rag-evaluation.json
```

For optional LLM-as-a-judge metrics, set `GEMINI_API_KEY` and run:

```bash
python benchmark_rag.py --semantic \
   --min-faithfulness 0.8 \
   --min-answer-relevancy 0.8 \
   --output reports/rag-semantic-evaluation.json
```

This opt-in mode scores factual claims against retrieved context and separately scores whether each answer addresses its question. It makes two judge requests per case, plus the normal Gemini answer requests; it is intentionally excluded from CI to avoid API costs and keep the default gate deterministic. Set `GEMINI_EVAL_MODEL` to override the default `gemini-2.5-flash` judge model.

To export the benchmark report to a file:

```bash
python benchmark_rag.py --output reports/benchmark.json --format json
python benchmark_rag.py --output reports/benchmark.csv --format csv
```

The exported file can be used to track model quality over time or compare runs across retrieval improvements.

## API server

You can run the project as a FastAPI service:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API's `/`, `/health`, and `/ask` routes require an `APP_ACCESS_TOKEN` bearer token. For example:

Example request:

```bash
curl -X POST http://localhost:8000/ask \
   -H "Authorization: Bearer your_app_access_token" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the password requirements?", "top_k": 3}'
```

The API returns the answer and the source chunks used to generate it.
