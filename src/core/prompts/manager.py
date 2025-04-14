"""
Prompt management for multi-provider LLM applications.

This module provides classes for managing prompt templates across different
LLM providers. It allows for provider-specific prompt formatting while providing
a consistent interface for the application.
"""

import logging
from copy import deepcopy
from typing import Dict, Any, Callable, List, Optional, Set, Union, Generic, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm.factory import ProviderType
from src.core.prompts.utils import (
    extract_variables_recursive,
    format_variables_recursive,
)


logger = logging.getLogger(__name__)

T = TypeVar("T")


class PromptTemplate:
    """
    A prompt template that supports multiple providers.
    
    This class allows defining different prompt formats for different LLM
    providers while maintaining a consistent interface.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a prompt template.
        
        Args:
            name: Template name
            description: Template description
        """
        self.name = name
        self.description = description
        self.templates: Dict[str, Any] = {}
        self.variables: Set[str] = set()
        self.default_provider: Optional[str] = None
        self.provider_resolver: Optional[Callable[[str], str]] = None
    
    def add_template(self, provider: str, template: Any) -> None:
        """
        Add a template for a specific provider.
        
        Args:
            provider: Provider name (from ProviderType)
            template: Template content
        """
        # Extract variables from the template
        template_variables = extract_variables_recursive(template)
        
        # If this is the first template, set as default and initialize variables
        if not self.templates:
            self.default_provider = provider
            self.variables = template_variables
        else:
            # Check that variables match existing templates
            if self.variables != template_variables:
                logger.warning(
                    f"Template for provider '{provider}' has different variables "
                    f"than other templates: {template_variables} vs {self.variables}"
                )
        
        self.templates[provider] = template
    
    def set_provider_resolver(self, resolver: Callable[[str], str]) -> None:
        """
        Set a function to resolve model names to provider names.
        
        Args:
            resolver: Function that takes a model name and returns a provider name
        """
        self.provider_resolver = resolver
    
    def get_template(self, model_or_provider: str) -> Any:
        """
        Get the template for a specific model or provider.
        
        Args:
            model_or_provider: Model name or provider name
            
        Returns:
            Template content
            
        Raises:
            ValueError: If the template doesn't exist and no default is available
        """
        # Resolve provider from model name if needed
        provider = self._resolve_provider(model_or_provider)
        
        # Get template for provider or use default
        if provider in self.templates:
            return deepcopy(self.templates[provider])
        elif self.default_provider:
            logger.warning(
                f"No template found for provider '{provider}'. "
                f"Using default provider '{self.default_provider}'."
            )
            return deepcopy(self.templates[self.default_provider])
        else:
            raise ValueError(
                f"No template found for provider '{provider}' and no default provider set."
            )
    
    def format(self, model_or_provider: str, **variables) -> Any:
        """
        Format a template with variables for a specific model or provider.
        
        Args:
            model_or_provider: Model name or provider name
            **variables: Variables to format the template with
            
        Returns:
            Formatted template
            
        Raises:
            ValueError: If required variables are missing
        """
        # Check that all required variables are provided
        missing_vars = self.variables - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Get template for provider
        template = self.get_template(model_or_provider)
        
        # Format template with variables
        return format_variables_recursive(template, variables)
    
    def _resolve_provider(self, model_or_provider: str) -> str:
        """
        Resolve a model name or provider name to a provider name.
        
        Args:
            model_or_provider: Model name or provider name
            
        Returns:
            Provider name
        """
        # Check if the input is already a provider name
        if model_or_provider in self.templates:
            return model_or_provider
        
        # Try to resolve using provider resolver
        if self.provider_resolver:
            try:
                return self.provider_resolver(model_or_provider)
            except Exception as e:
                logger.warning(f"Error resolving provider for '{model_or_provider}': {e}")
        
        # Default to input if no resolver or resolution failed
        return model_or_provider


class PromptManager:
    """
    Manager for multiple prompt templates.
    
    This class provides a central registry for prompt templates, making it easy
    to access and manage templates across an application.
    """
    
    def __init__(self):
        """
        Initialize a prompt manager.
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.provider_resolver: Optional[Callable[[str], str]] = None
    
    def register_template(self, template: PromptTemplate) -> None:
        """
        Register a prompt template.
        
        Args:
            template: Prompt template
            
        Raises:
            ValueError: If a template with the same name already exists
        """
        if template.name in self.templates:
            raise ValueError(f"Template '{template.name}' already exists")
        
        # Set provider resolver if available
        if self.provider_resolver and not template.provider_resolver:
            template.set_provider_resolver(self.provider_resolver)
        
        self.templates[template.name] = template
    
    def set_provider_resolver(self, resolver: Callable[[str], str]) -> None:
        """
        Set a function to resolve model names to provider names.
        
        This resolver will be used for all templates registered with this manager.
        
        Args:
            resolver: Function that takes a model name and returns a provider name
        """
        self.provider_resolver = resolver
        
        # Update existing templates
        for template in self.templates.values():
            if not template.provider_resolver:
                template.set_provider_resolver(resolver)
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Prompt template
            
        Raises:
            ValueError: If the template doesn't exist
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        return self.templates[name]
    
    def format(self, name: str, model_or_provider: str, **variables) -> Any:
        """
        Format a template with variables for a specific model or provider.
        
        Args:
            name: Template name
            model_or_provider: Model name or provider name
            **variables: Variables to format the template with
            
        Returns:
            Formatted template
        """
        template = self.get_template(name)
        return template.format(model_or_provider, **variables)
    
    def create_chat_prompt(self, name: str, model_or_provider: str, **variables) -> ChatPromptTemplate:
        """
        Create a ChatPromptTemplate for a template.
        
        Args:
            name: Template name
            model_or_provider: Model name or provider name
            **variables: Variables to format the template with
            
        Returns:
            ChatPromptTemplate
            
        Raises:
            ValueError: If the template content is not a list of BaseMessage objects
        """
        formatted = self.format(name, model_or_provider, **variables)
        
        if not isinstance(formatted, list) or not all(isinstance(m, BaseMessage) for m in formatted):
            raise ValueError(
                f"Template '{name}' content is not a list of BaseMessage objects"
            )
        
        return ChatPromptTemplate(formatted)


# Default provider resolver
def default_provider_resolver(model_name: str) -> str:
    """
    Default function to resolve model names to provider names.
    
    Args:
        model_name: Model name
        
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
