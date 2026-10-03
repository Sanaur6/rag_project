# Company RAG Assistant

This project is a small document-based RAG system for company policies and employee information. It loads text documents, chunks them into searchable sections, retrieves relevant matches, and answers user questions using the retrieved context.

## Features

- Document ingestion from TXT, PDF, and DOCX files
- chunk-based retrieval with keyword scoring
- optional Gemini embedding-based retrieval when a valid API key is configured
- source citation display in the CLI and Streamlit app
- validated TXT, TEXT, PDF, and DOCX uploads with immediate re-indexing in Streamlit
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

3. Run a smoke check before launching the app:

   ```bash
   python smoke_test.py
   ```

4. Run the CLI app:

   ```bash
   python app.py
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run app.py -- --streamlit
   ```

In the Streamlit sidebar, upload TXT, TEXT, PDF, or DOCX files and select **Upload and re-index**. Files are stored in `data/documents` by default; set `DOCS_FOLDER` to use another folder. Each upload is size-limited to 10 MB and is parsed before it is stored.

The current Streamlit interface does not provide user authentication. Keep it bound to a trusted local environment and do not expose it publicly or use it for sensitive company documents until authentication and authorization are added.

## Tests

```bash
pytest -q
```

## Evaluation benchmark

```bash
python benchmark_rag.py
```

This script loads the cases in `evaluation_cases.json`, runs the RAG on each question, and prints a pass/fail summary with a pass rate.

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

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the password requirements?", "top_k": 3}'
```

The API returns the answer and the source chunks used to generate it.
