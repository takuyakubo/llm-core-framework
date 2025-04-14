"""
Factory for creating language model instances.

This module provides a factory class for creating language model instances from
different providers with a unified interface.
"""

from enum import Enum
from typing import Dict, Type, Any, Optional

from src.config import DEFAULT_MODEL_CONFIGS
from src.core.llm.models import UnifiedModel


class ProviderType(Enum):
    """
    Enumeration of supported LLM providers.
    """
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    

class ModelFactory:
    """
    Factory class for creating language model instances.
    
    This class provides methods for creating model instances from different
    providers with a unified interface.
    """
    
    # Registry of provider-specific model implementations
    _registry: Dict[str, Type[UnifiedModel]] = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, model_class: Type[UnifiedModel]) -> None:
        """
        Register a provider-specific model implementation.
        
        Args:
            provider_name: Name of the provider
            model_class: Model implementation class
        """
        cls._registry[provider_name] = model_class
    
    @classmethod
    def create(cls, model_name: str, **kwargs) -> UnifiedModel:
        """
        Create a model instance for the specified model name.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional arguments for the model
            
        Returns:
            Unified model instance
            
        Raises:
            ValueError: If the provider is not supported or not registered
        """
        provider = cls._get_provider(model_name)
        
        if provider not in cls._registry:
            # Dynamic import if not already registered
            cls._register_provider_dynamically(provider)
            
        if provider not in cls._registry:
            raise ValueError(f"Provider '{provider}' is not registered. "
                           f"Available providers: {list(cls._registry.keys())}")
        
        # Apply default configuration if not explicitly provided
        for key, default_value in cls._get_default_config(provider).items():
            if key not in kwargs:
                kwargs[key] = default_value
                
        model_class = cls._registry[provider]
        return model_class(model_name, **kwargs)
    
    @staticmethod
    def _get_provider(model_name: str) -> str:
        """
        Determine the provider from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Provider name
            
        Raises:
            ValueError: If the provider cannot be determined
        """
        if model_name.startswith("claude-"):
            return ProviderType.ANTHROPIC.value
        elif model_name.startswith("gemini-"):
            return ProviderType.GOOGLE.value
        elif model_name.startswith("gpt-"):
            return ProviderType.OPENAI.value
        else:
            raise ValueError(f"Cannot determine provider for model: {model_name}")
    
    @classmethod
    def _get_default_config(cls, provider: str) -> Dict[str, Any]:
        """
        Get default configuration for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Default configuration
        """
        return DEFAULT_MODEL_CONFIGS.get(provider, {})
    
    @classmethod
    def _register_provider_dynamically(cls, provider: str) -> None:
        """
        Dynamically import and register a provider implementation.
        
        Args:
            provider: Provider name
        """
        try:
            if provider == ProviderType.ANTHROPIC.value:
                from src.core.llm.providers.anthropic import AnthropicModel
                cls.register_provider(provider, AnthropicModel)
            elif provider == ProviderType.OPENAI.value:
                from src.core.llm.providers.openai import OpenAIModel
                cls.register_provider(provider, OpenAIModel)
            elif provider == ProviderType.GOOGLE.value:
                from src.core.llm.providers.google import GoogleModel
                cls.register_provider(provider, GoogleModel)
        except ImportError:
            pass  # Silently fail, will raise error later if needed
