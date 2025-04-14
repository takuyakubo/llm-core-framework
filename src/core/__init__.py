"""
Core framework for building LLM-powered applications.

This package provides the core components for building applications with large
language models (LLMs) including:

- Abstract interfaces for working with different LLM providers
- Media processing utilities for images and other content
- Prompt management for multi-provider applications
- Template management for file-based templates
- Workflow framework for building complex applications
"""

from src.core.llm import (
    ModelFactory,
    UnifiedModel,
    ContentItem,
    TextContent,
    ImageContent,
)

from src.core.media import (
    image_to_base64,
    load_image,
    resize_image,
    format_image_for_provider,
)

from src.core.prompts import (
    PromptTemplate,
    PromptManager,
    extract_variables,
    format_variables,
)

from src.core.templates import (
    TemplateManager,
)

from src.core.workflow import (
    WorkflowNode,
    SequentialWorkflow,
    BaseState,
)

__all__ = [
    # LLM
    "ModelFactory",
    "UnifiedModel",
    "ContentItem",
    "TextContent",
    "ImageContent",
    
    # Media
    "image_to_base64",
    "load_image",
    "resize_image",
    "format_image_for_provider",
    
    # Prompts
    "PromptTemplate",
    "PromptManager",
    "extract_variables",
    "format_variables",
    
    # Templates
    "TemplateManager",
    
    # Workflow
    "WorkflowNode",
    "SequentialWorkflow",
    "BaseState",
]

__version__ = "0.1.0"
