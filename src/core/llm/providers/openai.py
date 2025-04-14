"""
OpenAI model implementation.
"""

from typing import List, Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm.factory import ProviderType
from src.core.llm.models import UnifiedModel, ContentItem, TextContent, ImageContent


class OpenAIModel(UnifiedModel):
    """
    Implementation of the unified model interface for OpenAI models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: OpenAI model name
            **kwargs: Additional arguments for the model
        """
        self._model_name = model_name
        self._system_message = kwargs.pop("system_message", None)
        self._client = ChatOpenAI(model=model_name, **kwargs)
    
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
        return ProviderType.OPENAI.value
    
    def format_prompt(self, content_items: List[ContentItem]) -> List[Dict[str, Any]]:
        """
        Formats content items into OpenAI-specific format.
        
        For OpenAI models, this converts the content items into a list of message
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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{item.image_type};base64,{item.image_data}"
                    }
                })
        
        # Add human message
        messages.append(HumanMessage(content=human_content))
        
        return messages
    
    def invoke(self, prompt: Any, **kwargs) -> str:
        """
        Invokes the OpenAI model with the given prompt.
        
        Args:
            prompt: Formatted prompt messages
            **kwargs: Additional arguments for the model
            
        Returns:
            Model response as a string
        """
        response = self._client.invoke(prompt, **kwargs)
        return response.content
