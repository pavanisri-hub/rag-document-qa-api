# RAG Document Q&A API

An AI-powered backend service that ingests documents and answers user questions using Retrieval-Augmented Generation (RAG). Built with FastAPI, sentence-transformers, and ChromaDB.

## Prerequisites

- Python 3.10
- pip
- Git
- An API key for an OpenAI-compatible LLM

## Setup

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root:

```env
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=openai
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_IMPL=chroma
TOP_K=3
```

## Running the server

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `POST /upload` – Upload and index a document.
- `POST /query` – Ask a question about uploaded documents.
- `GET /report` – View mock evaluation metrics.