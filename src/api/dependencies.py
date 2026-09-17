from fastapi import Security, Header
from fastapi.security import APIKeyHeader
from src.core.security import authenticate_api_key
from src.models.tenant import TenantContext
from typing import Optional

# Extracts 'X-API-Key' from HTTP request headers
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_tenant(
    api_key: str = Security(api_key_header),
    x_project_id: Optional[str] = Header(None, alias="X-Project-ID")
) -> TenantContext:
    """Dependency that extracts and validates the tenant from the API key."""
    return authenticate_api_key(api_key, x_project_id)
