"""
Formatters for preparing media content for different LLM providers.

This module provides utilities for formatting media content (images, audio, etc.)
into the specific formats required by different LLM providers.
"""

from typing import Dict, Any

from src.core.llm.factory import ProviderType


def format_image_for_provider(image_data: str, 
                             provider: str, 
                             image_type: str = "png") -> Dict[str, Any]:
    """
    Format image data for a specific provider.
    
    Args:
        image_data: Base64-encoded image data
        provider: Provider name (from ProviderType)
        image_type: Image type (png, jpeg, etc.)
        
    Returns:
        Formatted image data in the provider-specific format
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider == ProviderType.ANTHROPIC.value:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{image_type}",
                "data": image_data
            }
        }
    elif provider in [ProviderType.OPENAI.value, ProviderType.GOOGLE.value]:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_type};base64,{image_data}"
            }
        }
    else:
        raise ValueError(f"Unsupported provider for image formatting: {provider}")


def format_text_for_provider(text: str, provider: str) -> Dict[str, Any]:
    """
    Format text content for a specific provider.
    
    Currently, most providers use a similar text format, but this function
    allows for provider-specific formatting if needed in the future.
    
    Args:
        text: Text content
        provider: Provider name (from ProviderType)
        
    Returns:
        Formatted text data in the provider-specific format
    """
    # Most providers use a similar format for text
    return {
        "type": "text",
        "text": text
    }
