import json
import re
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from src.api.dependencies import get_current_tenant
from src.core.embeddings import embedding_service
from src.core.parser import extract_pages_from_file, extract_text_from_file
from src.core.vector_store import vector_store
from src.models.document import DocumentIngestRequest, DocumentIngestResponse
from src.models.tenant import TenantContext

router = APIRouter(prefix="/documents", tags=["Documents Ingestion"])


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into clean, overlapping chunks respecting paragraph and sentence boundaries.
    Avoids cutting words in half and maintains natural reading flow for embedding.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    # Split primarily on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph exceeds chunk size; split into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= chunk_size:
                    current_chunk = f"{current_chunk} {sent}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Carry over overlap tail
                        overlap_tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = f"{overlap_tail} {sent}".strip()
                    else:
                        current_chunk = sent[:chunk_size]
        else:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_tail = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    current_chunk = f"{overlap_tail}\n\n{para}".strip() if overlap_tail else para
                else:
                    current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks or [text.strip()]


@router.post(
    "",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document into the tenant's isolated knowledge base",
)
async def ingest_document(
    payload: DocumentIngestRequest,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Chunk, embed, and store document under the authenticated tenant's namespace."""
    chunks = chunk_text(payload.content)
    embeddings = embedding_service.embed_texts(chunks)
    
    point_ids = vector_store.upsert_chunks(
        tenant_id=tenant.tenant_id,
        external_id=payload.external_id,
        title=payload.title,
        chunks=chunks,
        embeddings=embeddings,
        category=payload.category,
        source_url=payload.source_url,
        client_metadata=payload.metadata,
    )
    
    return DocumentIngestResponse(
        status="indexed",
        tenant_id=tenant.tenant_id,
        external_id=payload.external_id,
        title=payload.title,
        chunks_created=len(chunks),
        point_ids=point_ids,
    )


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a physical file (PDF, DOCX, TXT) with page tracking",
)
async def upload_document(
    file: UploadFile = File(...),
    external_id: str = Form(...),
    category: str = Form(None),
    metadata_json: str = Form(None),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Extract page-aware text from uploaded file and ingest it into the vector store."""
    file_bytes = await file.read()
    
    try:
        pages = extract_pages_from_file(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    client_metadata_base: Dict[str, Any] = {}
    if metadata_json:
        try:
            client_metadata_base = json.loads(metadata_json)
        except Exception:
            pass

    all_chunks: List[str] = []
    all_chunk_metadata: List[Dict[str, Any]] = []

    for page_item in pages:
        page_num = page_item["page"]
        page_text = page_item["text"]
        page_chunks = chunk_text(page_text)
        
        for p_chunk in page_chunks:
            all_chunks.append(p_chunk)
            chunk_meta = dict(client_metadata_base)
            chunk_meta["page"] = page_num
            all_chunk_metadata.append(chunk_meta)

    # Fallback if no text extracted
    if not all_chunks:
        raise HTTPException(
            status_code=400, 
            detail="Tidak dapat mengekstrak teks dari file yang diunggah (dokumen kosong atau berupa gambar scan murni)."
        )

    embeddings = embedding_service.embed_texts(all_chunks)
    
    point_ids = vector_store.upsert_chunks(
        tenant_id=tenant.tenant_id,
        external_id=external_id,
        title=file.filename,
        chunks=all_chunks,
        embeddings=embeddings,
        category=category,
        source_url=None,
        client_metadata=all_chunk_metadata,
    )
    
    return {
        "status": "indexed",
        "tenant_id": tenant.tenant_id,
        "external_id": external_id,
        "filename": file.filename,
        "chunks_created": len(all_chunks),
        "point_ids": point_ids,
    }



@router.delete(
    "/{external_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document from the tenant's partition",
)
async def delete_document(
    external_id: str,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """Remove all vectors belonging to this external_id inside the tenant's partition."""
    vector_store.delete_document(tenant_id=tenant.tenant_id, external_id=external_id)
    return {
        "status": "deleted",
        "tenant_id": tenant.tenant_id,
        "external_id": external_id,
    }
