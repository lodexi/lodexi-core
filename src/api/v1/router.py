from fastapi import APIRouter
from src.api.v1.documents import router as documents_router
from src.api.v1.search import router as search_router
from src.api.v1.ask import router as ask_router
from src.api.v1.webhooks import router as webhooks_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(documents_router)
v1_router.include_router(search_router)
v1_router.include_router(ask_router)
v1_router.include_router(webhooks_router)
