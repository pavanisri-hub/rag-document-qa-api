from typing import List, Dict, Any

import chromadb
from chromadb.api.models.Collection import Collection


class VectorStoreEmptyError(Exception):
    pass


class VectorStore:
    """
    Simple wrapper around a ChromaDB collection.
    """

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()  # in-memory by default
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        filename: str,
    ) -> int:
        """
        Add chunks + embeddings to the collection.

        Returns number of chunks added.
        """
        if not chunks:
            return 0

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"filename": filename, "chunk_index": i} for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def has_documents(self) -> bool:
        """
        Check if the collection has any documents.
        """
        count = self.collection.count()
        return count > 0

    def query_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[str]:
        """
        Perform semantic search and return the top_k chunk texts.
        """
        if not self.has_documents():
            raise VectorStoreEmptyError("No documents have been indexed yet.")

        results: Dict[str, Any] = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # results["documents"] is a list of lists (one per query)
        documents = results.get("documents", [[]])[0]
        return documents