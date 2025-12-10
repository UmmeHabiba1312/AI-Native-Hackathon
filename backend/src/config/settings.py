from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # App configuration
    app_name: str = "AI-Native Textbook API"
    debug: bool = bool(os.getenv("DEBUG", False))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./textbook.db")

    # Qdrant configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")

    # OpenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # JWT configuration
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    # API configuration
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
    api_v1_prefix: str = "/api/v1"

    class Config:
        env_file = ".env"

settings = Settings()