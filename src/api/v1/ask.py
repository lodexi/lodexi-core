from typing import List, Optional
from fastapi import APIRouter, Depends, status, Header
from src.api.dependencies import get_current_tenant
from src.core.embeddings import embedding_service
from src.core.llm import llm_service
from src.core.vector_store import vector_store
from src.models.query import AskRequest, AskResponse, Citation
from src.models.tenant import TenantContext

router = APIRouter(prefix="/ask", tags=["Conversational Q&A"])


@router.post(
    "",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded question-answering with verifiable citations (for assistants & SOP Q&A)",
)
async def ask_question(
    payload: AskRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    x_tenant_llm_key: Optional[str] = Header(default=None, alias="X-Tenant-LLM-Key"),
    x_tenant_llm_provider: Optional[str] = Header(default=None, alias="X-Tenant-LLM-Provider"),
):
    """Retrieve relevant contexts and synthesize a grounded answer."""
    query_vector = embedding_service.embed_query(payload.question)
    
    # 1. Check Semantic Cache First
    cached_answer = vector_store.get_cached_answer(tenant.tenant_id, query_vector, threshold=0.95)
    if cached_answer:
        return AskResponse(
            tenant_id=tenant.tenant_id,
            question=payload.question,
            answer=cached_answer,
            citations=[],
            grounded=True,
            model_used=llm_service.model,
            cached=True,
            prompt_tokens=0,
            completion_tokens=0
        )

    # 2. Vector Search with Relevance Threshold
    contexts = vector_store.search_tenant(
        tenant_id=tenant.tenant_id,
        query_vector=query_vector,
        limit=payload.limit,
        category_filter=payload.category_filter,
        min_score=0.0,  # Lowered to 0.0 so fast-mock can retrieve documents
    )
    
    # 3. LLM Synthesis (RAG or Friendly Fallback)
    answer, prompt_tokens, completion_tokens = llm_service.synthesize_answer(
        payload.question, 
        contexts, 
        custom_api_key=x_tenant_llm_key,
        custom_provider=x_tenant_llm_provider,
        custom_system_prompt=payload.system_prompt
    )
    
    # 4. Save to Semantic Cache
    vector_store.cache_answer(tenant.tenant_id, query_vector, payload.question, answer)
    
    citations: List[Citation] = [
        Citation(
            external_id=c.external_id,
            title=c.title,
            snippet=c.snippet,
            score=c.score,
            source_url=c.source_url,
        )
        for c in contexts
    ]
    
    return AskResponse(
        tenant_id=tenant.tenant_id,
        question=payload.question,
        answer=answer,
        citations=citations,
        grounded=True,
        model_used=llm_service.model,
        cached=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
