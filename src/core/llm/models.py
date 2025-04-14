"""
Core abstractions for language models.

This module provides the base classes and interfaces for working with different
language model providers in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class ContentItem:
    """
    Base class for content items that can be sent to language models.
    """
    pass


class TextContent(ContentItem):
    """
    Represents a text content item.
    """
    def __init__(self, text: str):
        self.text = text
        
    def __repr__(self) -> str:
        preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"TextContent('{preview}')"


class ImageContent(ContentItem):
    """
    Represents an image content item with base64-encoded data.
    """
    def __init__(self, image_data: str, image_type: str = "png"):
        self.image_data = image_data
        self.image_type = image_type
        
    def __repr__(self) -> str:
        return f"ImageContent(type='{self.image_type}', len={len(self.image_data)})"


class UnifiedModel(ABC):
    """
    Abstract base class providing a unified interface for different LLM providers.
    
    This class ensures that all model implementations provide consistent methods
    for content formatting, model invocation, and image processing.
    """
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name of the underlying model.
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        """
        pass
    
    @abstractmethod
    def format_prompt(self, content_items: List[ContentItem]) -> Any:
        """
        Formats a list of content items into a provider-specific format.
        
        Args:
            content_items: List of ContentItem objects
            
        Returns:
            Provider-specific formatted prompt structure
        """
        pass
    
    @abstractmethod
    def invoke(self, prompt: Any, **kwargs) -> str:
        """
        Invokes the language model with the given prompt.
        
        Args:
            prompt: Formatted prompt for the model
            **kwargs: Additional arguments for the model
            
        Returns:
            Model response as a string
        """
        pass
    
    def process_with_images(self, text_prompt: str, images: List[str]) -> str:
        """
        Helper method to process a text prompt with multiple images.
        
        Args:
            text_prompt: The text instruction
            images: List of base64-encoded image data
            
        Returns:
            Model response as a string
        """
        content_items = [TextContent(text_prompt)]
        content_items.extend([ImageContent(img_data) for img_data in images])
        formatted_prompt = self.format_prompt(content_items)
        return self.invoke(formatted_prompt)
