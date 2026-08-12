import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()


class LLMConfigurationError(Exception):
    pass


class LLMService:
    """
    Handles embedding generation and LLM calls.
    """

    def __init__(
        self,
        embedding_model_name: str = None,
        llm_provider: str = None,
        llm_api_key: str = None,
    ):
        # Load from env if not provided
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")

        if not self.llm_api_key:
            raise LLMConfigurationError(
                "LLM_API_KEY is not set. Please configure it in your .env file."
            )

        # Initialize embedding model once
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize OpenAI-compatible client (for now we only support OpenAI-style)
        if self.llm_provider.lower() == "openai":
            self.client = OpenAI(api_key=self.llm_api_key)
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM_PROVIDER '{self.llm_provider}'. "
                "For this project, use 'openai'."
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of strings.
        """
        if not texts:
            return []
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=False)
        # Ensure list-of-lists of floats
        return [list(map(float, vec)) for vec in embeddings]

    def embed_question(self, question: str) -> List[float]:
        """
        Generate an embedding for a single question string.
        """
        vec = self.embedding_model.encode(question, convert_to_numpy=False)
        return list(map(float, vec))

    @staticmethod
    def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
        """
        Construct the RAG prompt including context and the user question.
        """
        context_block = "\n\n".join(context_chunks)

        prompt = f"""You are a helpful assistant answering questions based ONLY on the provided document context.

Context Information:
---------------------
{context_block}
---------------------

Instructions:
- ONLY use the context information above to answer the question.
- Do NOT use any outside knowledge.
- If the context does not contain the answer, explicitly respond with: "I cannot find the answer in the provided documents."

Question: {question}
Answer:"""
        return prompt

    def ask_llm(self, prompt: str) -> str:
        """
        Send the prompt to the LLM and return the generated answer text.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # you can change this to any available chat model
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            # OpenAI client v1 style
            return response.choices[0].message.content.strip()
        except Exception as exc:
            # Let the API layer translate this into a 500/502
            raise RuntimeError(f"LLM API call failed: {exc}") from exc