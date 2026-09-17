from typing import Optional
from fastapi import HTTPException, status, Header
from src.config import settings
from src.models.tenant import TenantContext


def authenticate_api_key(
    api_key: Optional[str] = None,
    x_project_id: Optional[str] = Header(None, alias="X-Project-ID")
) -> TenantContext:
    """Validate an API key and resolve its tenant context.
    
    Raises:
        HTTPException (401): If key is missing or not recognized.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'X-API-Key' header",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    tenant_map = settings.api_key_to_tenant_map
    tenant_id = tenant_map.get(api_key)
    
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
        
    if x_project_id:
        tenant_id = f"{tenant_id}_proj_{x_project_id}"
        
    prefix = api_key[:8] + "..." if len(api_key) >= 8 else "..."
    return TenantContext(
        tenant_id=tenant_id,
        api_key_prefix=prefix,
        is_active=True
    )
