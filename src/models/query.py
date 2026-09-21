from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request for pure semantic retrieval (used by catalog/repository search)."""
    
    query: str = Field(..., min_length=2, description="Natural language search query")
    limit: int = Field(default=5, ge=1, le=50, description="Max number of results to return")
    category_filter: Optional[str] = Field(default=None, description="Optional filter by category")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Similarity threshold score")


class SearchResultItem(BaseModel):
    """A single retrieved and ranked chunk."""
    
    chunk_id: str
    external_id: str
    title: str
    snippet: str
    score: float
    category: Optional[str] = None
    source_url: Optional[str] = None
    client_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response containing ranked semantic search results."""
    
    tenant_id: str
    query: str
    total_found: int
    results: List[SearchResultItem]


class AskRequest(BaseModel):
    """Request for conversational question-answering over documents."""
    
    question: str = Field(..., min_length=3, description="User question")
    limit: int = Field(default=4, ge=1, le=10, description="Number of context chunks to ground the answer")
    category_filter: Optional[str] = Field(default=None, description="Filter knowledge domain")
    system_prompt: Optional[str] = Field(default=None, description="Custom AI persona/instructions")


class Citation(BaseModel):
    """Source reference supporting the generated answer."""
    
    external_id: str
    title: str
    snippet: str
    score: float
    source_url: Optional[str] = None


class AskResponse(BaseModel):
    """Grounded synthesis answer with verifiable citations."""
    
    tenant_id: str
    question: str
    answer: str
    citations: List[Citation]
    grounded: bool = True
    model_used: str = "mock"
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
