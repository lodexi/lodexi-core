import os
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings
from src.models.query import SearchResultItem


class VectorStoreService:
    """Manages Qdrant vector operations with strict tenant data isolation."""
    
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION
        self.cache_collection_name = f"{self.collection_name}_semantic_cache"
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = self._init_client()
        self._ensure_collection_exists()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client based on configured mode."""
        mode = settings.QDRANT_MODE.lower()
        if mode == "memory":
            return QdrantClient(location=":memory:")
        elif mode == "server":
            return QdrantClient(url=settings.QDRANT_URL)
        else:
            # Embedded disk persistence
            os.makedirs(settings.QDRANT_PATH, exist_ok=True)
            return QdrantClient(path=settings.QDRANT_PATH)

    def _ensure_collection_exists(self):
        """Create the vector collection if it does not already exist."""
        collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        
        if self.cache_collection_name not in collections:
            self.client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            # Create payload index for tenant_id for fast isolated filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(
        self,
        tenant_id: str,
        external_id: str,
        title: str,
        chunks: List[str],
        embeddings: List[List[float]],
        category: Optional[str] = None,
        source_url: Optional[str] = None,
        client_metadata: Optional[Any] = None,
    ) -> List[str]:
        """Index text chunks with immutable tenant_id tags into Qdrant."""
        point_ids: List[str] = []
        points: List[qmodels.PointStruct] = []
        
        for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{external_id}:{idx}"))
            point_ids.append(point_id)
            
            # Support both per-chunk metadata list and global metadata dict
            if isinstance(client_metadata, list) and idx < len(client_metadata):
                chunk_meta = client_metadata[idx] or {}
            elif isinstance(client_metadata, dict):
                chunk_meta = client_metadata
            else:
                chunk_meta = {}
            
            payload = {
                "tenant_id": tenant_id,
                "external_id": external_id,
                "title": title,
                "chunk_index": idx,
                "text": chunk_text,
                "category": category,
                "source_url": source_url,
                "client_metadata": chunk_meta,
            }
            
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return point_ids

    def search_tenant(
        self,
        tenant_id: str,
        query_vector: List[float],
        limit: int = 5,
        category_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResultItem]:
        """Search vectors strictly restricted to the specified tenant_id."""
        
        # Mandatory tenant isolation boundary
        must_conditions: List[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="tenant_id",
                match=qmodels.MatchValue(value=tenant_id),
            )
        ]
        
        if category_filter:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=category_filter),
                )
            )
            
        tenant_filter = qmodels.Filter(must=must_conditions)
        
        # Support qdrant-client search (supports both client.search and client.query_points)
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=tenant_filter,
                limit=limit,
                score_threshold=min_score if min_score > 0.0 else None,
            )
        except AttributeError:
            # Fallback for newer Qdrant client query_points API
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=tenant_filter,
                limit=limit,
                score_threshold=min_score if min_score > 0.0 else None,
            )
            hits = res.points

        results: List[SearchResultItem] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResultItem(
                    chunk_id=str(hit.id),
                    external_id=str(payload.get("external_id", "")),
                    title=str(payload.get("title", "")),
                    snippet=str(payload.get("text", "")),
                    score=round(float(hit.score), 4),
                    category=payload.get("category"),
                    source_url=payload.get("source_url"),
                    client_metadata=payload.get("client_metadata", {}),
                )
            )
        return results

    def delete_document(self, tenant_id: str, external_id: str) -> bool:
        """Delete all chunks belonging to a document within the tenant's partition."""
        del_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tenant_id)),
                qmodels.FieldCondition(key="external_id", match=qmodels.MatchValue(value=external_id)),
            ]
            
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(filter=del_filter),
        )
        return True

    def get_cached_answer(self, tenant_id: str, query_vector: List[float], threshold: float = 0.95) -> Optional[str]:
        """Search the semantic cache for highly similar previous queries."""
        tenant_filter = qmodels.Filter(must=[
            qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tenant_id))
        ])
        
        try:
            hits = self.client.search(
                collection_name=self.cache_collection_name,
                query_vector=query_vector,
                query_filter=tenant_filter,
                limit=1,
                score_threshold=threshold,
            )
        except AttributeError:
            res = self.client.query_points(
                collection_name=self.cache_collection_name,
                query=query_vector,
                query_filter=tenant_filter,
                limit=1,
                score_threshold=threshold,
            )
            hits = res.points

        if hits:
            return str(hits[0].payload.get("answer", ""))
        return None

    def cache_answer(self, tenant_id: str, query_vector: List[float], question: str, answer: str) -> None:
        """Store a generated answer in the semantic cache."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}:{question}"))
        
        self.client.upsert(
            collection_name=self.cache_collection_name,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "tenant_id": tenant_id,
                        "question": question,
                        "answer": answer,
                    },
                )
            ],
            wait=False,
        )


vector_store = VectorStoreService()
