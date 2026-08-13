# RAG Document Q&A API

An AI-powered backend service that ingests documents and answers user questions using Retrieval-Augmented Generation (RAG). Built with FastAPI, sentence-transformers, and ChromaDB.

## Features

- `POST /upload` – Ingest `.txt`, `.md`, `.pdf` documents.
- `POST /query` – Ask natural language questions about uploaded documents.
- `GET /report` – Return mock evaluation metrics (context precision, faithfulness).
- Vector embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- In-memory ChromaDB vector store for semantic search.
- Strong RAG prompt that restricts the LLM to document context only.

## Prerequisites

- Python 3.10
- pip
- Git
- An API key for an OpenAI-compatible LLM

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/rag-document-qa-api.git
cd rag-document-qa-api
```

Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
LLM_API_KEY=sk-your-real-api-key
LLM_PROVIDER=openai
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_IMPL=chroma
TOP_K=3
```

Do **not** commit `.env` to version control. Use `.env.example` as a template.

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive API docs: `http://127.0.0.1:8000/docs`

## API Usage (curl examples)

### Upload a document

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_document.txt"
```

Example success response:

```json
{
  "message": "File uploaded and indexed successfully.",
  "filename": "test_document.txt",
  "num_chunks": 1
}
```

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the primary causes of revenue decline in Q3?"
  }'
```

Example success response (with valid LLM credits):

```json
{
  "answer": "Based on the document, the primary causes of revenue decline in Q3 were severe supply chain disruptions and a decrease in consumer spending.",
  "sources": [
    "Quarterly Financial Report - Q3 ... Revenue dropped by 40% due to severe supply chain disruptions and a decrease in consumer spending ..."
  ]
}
```

If your API key has no credits, the response will be a structured error:

```json
{
  "detail": "LLM API call failed: Error code: 429 - { ... 'insufficient_quota' ... }"
}
```

### Get evaluation report

```bash
curl -X GET "http://127.0.0.1:8000/report" \
  -H "accept: application/json"
```

Example response:

```json
{
  "context_precision": 0.9,
  "faithfulness": 0.85,
  "system_status": "healthy"
}
```

## Sample Document

This repository includes `test_document.txt` as a sample input for the `/upload` endpoint. It contains a simple Q3 financial report scenario to demonstrate RAG behavior.

## Project Structure

```text
rag-document-qa-api/
├─ main.py                # FastAPI app and route definitions
├─ document_processor.py  # File validation, text extraction, chunking
├─ vector_store.py        # ChromaDB wrapper for add/query
├─ llm_service.py         # Embedding model and LLM client
├─ test_document.txt      # Sample document for testing
├─ requirements.txt       # Python dependencies
├─ .env.example           # Example environment configuration
├─ README.md              # Project documentation
└─ venv/                  # Virtual environment (not committed)
```

