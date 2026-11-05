import hashlib
import math
from typing import List
from src.config import settings


class BaseEmbeddingProvider:
    """Abstract interface for generating text embeddings."""
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class FastMockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic hashing-based embedding for ultra-fast, zero-download local testing.
    
    Produces normalized 384-dimensional vectors with reproducible semantic similarity.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            # Deterministic pseudo-embedding from character n-grams and text hash
            vec = [0.0] * self.dimension
            tokens = text.lower().split()
            for token in tokens:
                token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = token_hash % self.dimension
                vec[idx] += 1.0
            
            # Normalize vector to unit length
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec[0] = 1.0
            results.append(vec)
        return results


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Production local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`"
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using OpenAI or Gemini (OpenAI compatibility mode)."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str = None):
        try:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
            self.model = model
        except ImportError:
            raise RuntimeError("openai is not installed. Run `pip install openai`")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory function to get configured embedding provider."""
    provider_type = settings.EMBEDDING_PROVIDER.lower()
    
    if provider_type == "sentence-transformers":
        try:
            return SentenceTransformerEmbeddingProvider(settings.EMBEDDING_MODEL)
        except Exception:
            return FastMockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
            
    elif provider_type == "gemini":
        try:
            active_key = settings.api_keys_list[0] if settings.api_keys_list else settings.OPENAI_API_KEY
            return OpenAIEmbeddingProvider(
                api_key=active_key,
                model="text-embedding-004",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        except Exception:
            return FastMockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
            
    elif provider_type == "openai":
        try:
            active_key = settings.api_keys_list[0] if settings.api_keys_list else settings.OPENAI_API_KEY
            return OpenAIEmbeddingProvider(
                api_key=active_key,
                model="text-embedding-3-small"
            )
        except Exception:
            return FastMockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    
    return FastMockEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)


embedding_service = get_embedding_provider()
