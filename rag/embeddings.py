"""Local embeddings with no API key: Chroma's built-in ONNX all-MiniLM-L6-v2 wrapped as a LangChain Embeddings."""
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class LocalMiniLM(Embeddings):
    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
