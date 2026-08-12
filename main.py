from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from document_processor import (
    UnsupportedFileTypeError,
    UnreadableDocumentError,
    chunk_text,
    extract_text,
    validate_file_extension,
)
from llm_service import LLMConfigurationError, LLMService
from vector_store import VectorStore, VectorStoreEmptyError


app = FastAPI(
    title="RAG Document Q&A API",
    description="An AI-powered Retrieval-Augmented Generation (RAG) API for document question answering.",
    version="1.0.0",
)

# Optional: allow CORS for local testing/frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services at startup
vector_store = VectorStore(collection_name="documents")
try:
    llm_service = LLMService()
except LLMConfigurationError as exc:
    # Delay hard failure to request-time with clearer error message
    llm_service = None
    init_error = exc
else:
    init_error = None


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Ingest a document:
    - Validate extension
    - Extract raw text
    - Chunk text with overlap
    - Generate embeddings
    - Store in vector DB
    """
    if init_error is not None or llm_service is None:
        raise HTTPException(
            status_code=500,
            detail=f"LLM service not initialized: {init_error}",
        )

    filename = file.filename or "uploaded_file"
    try:
        extension = validate_file_extension(filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    file_bytes = await file.read()

    try:
        text = extract_text(file_bytes, extension)
    except UnreadableDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while extracting text: {exc}",
        )

    # Chunk the text
    chunks: List[str] = chunk_text(text, chunk_size=1000, overlap=200)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Document appears unreadable or empty after chunking.",
        )

    # Generate embeddings
    try:
        embeddings = llm_service.embed_texts(chunks)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {exc}",
        )

    # Store in vector DB
    try:
        num_chunks = vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            filename=filename,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store embeddings in vector database: {exc}",
        )

    return JSONResponse(
        status_code=201,
        content={
            "message": "File uploaded and indexed successfully.",
            "filename": filename,
            "num_chunks": num_chunks,
        },
    )


@app.post("/query")
async def query_documents(payload: QueryRequest) -> Dict[str, Any]:
    """
    Answer a user question using RAG:
    - Validate question
    - Embed question
    - Semantic search over vector DB
    - Build RAG prompt
    - Call LLM
    - Return answer and sources
    """
    if init_error is not None or llm_service is None:
        raise HTTPException(
            status_code=500,
            detail=f"LLM service not initialized: {init_error}",
        )

    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    # Embed question
    try:
        question_embedding = llm_service.embed_question(question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding for question: {exc}",
        )

    # Retrieve similar chunks
    try:
        top_k = 3
        context_chunks = vector_store.query_similar_chunks(
            query_embedding=question_embedding,
            top_k=top_k,
        )
    except VectorStoreEmptyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query vector database: {exc}",
        )

    if not context_chunks:
        raise HTTPException(
            status_code=400,
            detail="No relevant context found. Have you uploaded any documents?",
        )

    # Build prompt
    prompt = llm_service.build_rag_prompt(question=question, context_chunks=context_chunks)

    # Call LLM
    try:
        answer = llm_service.ask_llm(prompt)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while calling LLM: {exc}",
        )

    return {
        "answer": answer,
        "sources": context_chunks,
    }


@app.get("/report")
async def get_report() -> Dict[str, Any]:
    """
    Return mock evaluation metrics for the system.
    """
    return {
        "context_precision": 0.90,
        "faithfulness": 0.85,
        "system_status": "healthy",
    }