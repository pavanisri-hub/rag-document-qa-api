from vector_store import VectorStore, VectorStoreEmptyError


def test_vector_store_add_and_query():
    store = VectorStore(collection_name="test_collection")

    chunks = ["chunk one", "chunk two", "chunk three"]
    embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]  # fake embeddings
    filename = "test_document.txt"

    added = store.add_documents(chunks=chunks, embeddings=embeddings, filename=filename)
    assert added == len(chunks)
    assert store.has_documents() is True

    # query with a dummy embedding; just ensure it doesn't crash
    results = store.query_similar_chunks(query_embedding=[0.1] * 384, top_k=2)
    assert len(results) <= 2
    assert all(isinstance(doc, str) for doc in results)


def test_vector_store_empty_raises():
    store = VectorStore(collection_name="empty_collection")
    assert store.has_documents() is False
    try:
        store.query_similar_chunks(query_embedding=[0.0] * 384, top_k=1)
        assert False, "Expected VectorStoreEmptyError"
    except VectorStoreEmptyError:
        assert True