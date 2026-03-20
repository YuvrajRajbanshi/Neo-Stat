"""
Embeddings module for handling sentence transformers.
Provides embedding model loading and text embedding functions.
"""

from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

# Cache the model globally to avoid reloading
_embedding_model = None


def load_embedding_model() -> SentenceTransformer:
    """
    Load and return the sentence transformer embedding model.
    Uses caching to avoid reloading the model multiple times.

    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model

    try:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        return _embedding_model

    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {str(e)}")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    try:
        model = load_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")


def get_single_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text string to embed

    Returns:
        Embedding vector as a list of floats
    """
    try:
        model = load_embedding_model()
        embedding = model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")


class EmbeddingFunction:
    """
    Wrapper class for embedding function compatible with vector stores.
    """

    def __init__(self):
        self.model = load_embedding_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

    def __call__(self, text: str) -> list[float]:
        """Compatibility shim for vector stores that expect a callable embedding function."""
        return self.embed_query(text)
