"""
Anthropic Claude model implementation.
"""

from typing import List, Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm.factory import ProviderType
from src.core.llm.models import UnifiedModel, ContentItem, TextContent, ImageContent


class AnthropicModel(UnifiedModel):
    """
    Implementation of the unified model interface for Anthropic Claude models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Anthropic model.
        
        Args:
            model_name: Claude model name
            **kwargs: Additional arguments for the model
        """
        self._model_name = model_name
        self._system_message = kwargs.pop("system_message", None)
        self._client = ChatAnthropic(model=model_name, **kwargs)
    
    @property
    def model_name(self) -> str:
        """
        Returns the name of the underlying model.
        """
        return self._model_name
    
    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        """
        return ProviderType.ANTHROPIC.value
    
    def format_prompt(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """
        Formats content items into Anthropic-specific format.
        
        For Claude models, this converts the content items into a list of message
        dictionaries with the appropriate message type and content format.
        
        Args:
            content_items: List of ContentItem objects
            
        Returns:
            List of formatted message dictionaries
        """
        messages = []
        
        # Add system message if provided
        if self._system_message:
            messages.append(SystemMessage(content=self._system_message))
        
        # Prepare human message content
        human_content = []
        
        for item in content_items:
            if isinstance(item, TextContent):
                human_content.append({
                    "type": "text",
                    "text": item.text
                })
            elif isinstance(item, ImageContent):
                human_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{item.image_type}",
                        "data": item.image_data
                    }
                })
        
        # Add human message
        messages.append(HumanMessage(content=human_content))
        
        return messages
    
    def invoke(self, prompt: Any, **kwargs) -> str:
        """
        Invokes the Claude model with the given prompt.
        
        Args:
            prompt: Formatted prompt messages
            **kwargs: Additional arguments for the model
            
        Returns:
            Model response as a string
        """
        response = self._client.invoke(prompt, **kwargs)
        return response.content
