from fastapi import APIRouter, Request, HTTPException, Query, Depends
from typing import Dict, Any, Optional
from src.core.security import authenticate_api_key
from src.core.embeddings import embedding_service
from src.core.llm import llm_service
from src.core.vector_store import vector_store
from src.models.tenant import TenantContext

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

async def verify_webhook_token(
    token: str = Query(..., description="Lodexi API Key passed as query parameter for webhook auth"),
    project_id: Optional[str] = Query(None, alias="project_id")
) -> TenantContext:
    try:
        # Re-use the existing core logic but using query params
        return authenticate_api_key(token, project_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Webhook authentication failed: {str(e)}")

@router.post("/google-chat")
async def google_chat_webhook(
    request: Request,
    tenant: TenantContext = Depends(verify_webhook_token)
):
    """
    Webhook endpoint for Google Chat.
    Google Chat sends a JSON payload. We extract the message and return a JSON response.
    Requires ?token=LODEXI-API-KEY in the URL.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Google chat ping event
    if payload.get("type") == "ADDED_TO_SPACE":
        return {"text": "Halo! Saya adalah Lodexi Bot. Terima kasih sudah mengundang saya ke ruangan ini. Saya siap menjawab pertanyaan terkait dokumen di Knowledge Base Lodexi."}
    
    # Process actual message
    if payload.get("type") == "MESSAGE":
        # Handle message text (Google Chat sometimes includes the bot mention in the text)
        message_data = payload.get("message", {})
        
        # Sometimes there's argument text (clean text), otherwise fallback to full text
        user_message = message_data.get("argumentText") or message_data.get("text", "")
        user_message = user_message.strip()
        
        if not user_message:
            return {"text": "Pesan kosong atau tidak valid."}
            
        # 1. Embed query
        query_vector = embedding_service.embed_query(user_message)
        
        # 2. Check Semantic Cache
        cached_answer = vector_store.get_cached_answer(tenant.tenant_id, query_vector, threshold=0.95)
        if cached_answer:
            return {"text": f"{cached_answer}\n\n_(Dijawab super cepat dari Cache Lodexi ⚡)_"}
            
        # 3. Vector Search
        contexts = vector_store.search_tenant(
            tenant_id=tenant.tenant_id,
            query_vector=query_vector,
            limit=3,
            min_score=0.5
        )
        
        # 4. LLM Synthesis
        answer, _, _ = llm_service.synthesize_answer(user_message, contexts)
        
        # 5. Save to cache
        vector_store.cache_answer(tenant.tenant_id, query_vector, user_message, answer)
        
        # Format citations if any
        if contexts:
            citations_text = "\n\n**Sumber Info:**\n" + "\n".join([f"- {c.title}" for c in contexts])
            answer += citations_text
            
        return {"text": answer}
        
    return {"text": "Event tidak dikenali oleh Lodexi."}
