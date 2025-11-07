"""Configuration settings for the application."""

import os
from typing import Optional


class Settings:
    """Application settings and configuration."""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:latest")
    
    # Processing Configuration
    DEFAULT_OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    
    # LLM Configuration
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: Optional[int] = None
    
    @classmethod
    def get_ollama_url(cls) -> str:
        """Get the Ollama API base URL."""
        return cls.OLLAMA_BASE_URL
    
    @classmethod
    def get_model_name(cls) -> str:
        """Get the Ollama model name."""
        return cls.OLLAMA_MODEL

