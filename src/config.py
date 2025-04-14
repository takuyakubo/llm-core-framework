"""
Global configuration for the LLM Core Framework.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Service Configuration
LANGCHAIN_MAX_CONCURRENCY = int(os.getenv("LANGCHAIN_MAX_CONCURRENCY", "5"))

# Langfuse Tracking Configuration
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
USE_LANGFUSE = bool(os.getenv("USE_LANGFUSE", "False").lower() == "true")

# Debug Configuration
DEBUG_MODE = bool(os.getenv("DEBUG_MODE", "False").lower() == "true")

# Default Model Configuration
DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "default_model": "gpt-4o",
        "default_max_tokens": 4000,
    },
    "anthropic": {
        "default_model": "claude-3-7-sonnet-latest",
        "default_max_tokens": 10000,
    },
    "google": {
        "default_model": "gemini-2.5-pro-preview-03-25",
        "default_max_tokens": 8000,
    },
}

# Media Configuration
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "webp", "gif"]
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

# Template Configuration
TEMPLATE_DIRS = [
    os.path.join(os.path.dirname(__file__), "core", "templates", "default"),
]
