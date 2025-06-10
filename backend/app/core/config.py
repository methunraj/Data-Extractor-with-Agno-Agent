# app/core/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Now supports both Agno and LangChain implementations.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")  # Optional: For Google Custom Search Engine
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "4.0.0-langchain"
    APP_TITLE: str = "Enhanced LangChain AI JSON to XLSX Processing API"
    TEMP_DIR_PREFIX: str = "langchain_xlsx_"
    
    # LangChain specific settings
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")  # Optional: For LangSmith tracing
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()