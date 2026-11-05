import json
from typing import Dict, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings with environment variable overrides."""
    
    APP_NAME: str = "LODEX Middleware"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Vector DB (Qdrant)
    QDRANT_MODE: Literal["memory", "embedded", "server"] = "embedded"
    QDRANT_PATH: str = "./data/qdrant"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "lodex_knowledge_base"

    # Embeddings
    EMBEDDING_PROVIDER: Literal["fast-mock", "sentence-transformers", "openai", "gemini"] = "fast-mock"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # LLM
    LLM_PROVIDER: Literal["mock", "openai", "litellm", "gemini"] = "mock"
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # Demo API Keys mapping: api_key -> tenant_id
    DEMO_API_KEYS: str = '{"key_jiulibrary_secret_123": "jiulibrary", "key_staffportal_secret_456": "staff_portal"}'

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def api_keys_list(self) -> list[str]:
        """Parsed list of multiple LLM API keys for rotation."""
        return [k.strip() for k in self.OPENAI_API_KEY.split(",") if k.strip()]

    @property
    def api_key_to_tenant_map(self) -> Dict[str, str]:
        """Parsed mapping of API keys to their corresponding tenant IDs."""
        try:
            return json.loads(self.DEMO_API_KEYS)
        except Exception:
            return {
                "key_jiulibrary_secret_123": "jiulibrary",
                "key_staffportal_secret_456": "staff_portal",
            }


settings = Settings()
