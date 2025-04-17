"""
OpenAI model implementation.

This module provides the implementation of the unified model interface for OpenAI models,
including GPT-4 and GPT-3.5 series. It handles the provider-specific details for
initializing models, processing inputs, and formatting responses.

Example usage:
    from core.llm.providers.openai import OpenAIModel
    
    # Create an OpenAI model instance
    model = OpenAIModel("gpt-4o", max_tokens=1000)
    
    # Use the model
    response = model.invoke("Explain quantum computing")
    
    # Process images
    image_path = "path/to/image.jpg"
    image_obj = model.get_image_object(image_path)
    response_with_image = model.invoke(
        [{"type": "text", "text": "What's in this image?"}, image_obj]
    )
"""

from langchain_openai import ChatOpenAI

from core.llm.models import UnifiedModel
from core.llm.utils import image_path_to_image_data

provider_name = "openai"


class OpenAIModel(ChatOpenAI, UnifiedModel):
    """
    Implementation of the unified model interface for OpenAI models.
    
    This class integrates OpenAI's models with the framework's unified interface,
    leveraging the langchain_openai library for the core functionality while
    adding the necessary methods to comply with our UnifiedModel abstraction.
    
    Supported models include:
    - GPT-4o
    - GPT-4 Turbo
    - GPT-4
    - GPT-3.5 Turbo
    
    OpenAIモデルのための統一インターフェースの実装。
    
    このクラスは、OpenAIのモデルをフレームワークの統一インターフェースと統合し、
    langchain_openaiライブラリをコア機能として活用しながら、UnifiedModel抽象化に
    準拠するために必要なメソッドを追加します。
    
    サポートされているモデルには次のものが含まれます：
    - GPT-4o
    - GPT-4 Turbo
    - GPT-4
    - GPT-3.5 Turbo
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI model.
        
        This initializes both the framework's UnifiedModel requirements
        and the underlying LangChain ChatOpenAI implementation.
        
        Args:
            model_name (str): OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            **kwargs: Additional arguments for the model configuration
                      Common parameters include:
                      - max_tokens: Maximum tokens in the response
                      - temperature: Randomness of the output (0.0 to 1.0)
                      - top_p: Alternative to temperature for controlling randomness
                      - presence_penalty: Penalty for new tokens based on presence in text so far
                      - frequency_penalty: Penalty for new tokens based on frequency in text so far
                      
        Note:
            An OPENAI_API_KEY environment variable must be set or passed
            explicitly via the api_key parameter.
            
        OpenAIモデルを初期化します。
        
        これは、フレームワークのUnifiedModelの要件と基礎となるLangChain ChatOpenAI実装の
        両方を初期化します。
        
        注意：
            OPENAI_API_KEY環境変数が設定されているか、api_keyパラメータを通じて
            明示的に渡される必要があります。
        """
        super(ChatOpenAI, self).__init__(model=model_name, **kwargs)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """
        Returns the name of the underlying model.
        
        Returns:
            str: The OpenAI model name (e.g., "gpt-4o", "gpt-3.5-turbo")
            
        基礎となるモデルの名前を返します。
        """
        return self._model_name

    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        
        Returns:
            str: Always returns "openai" for this implementation
            
        モデルプロバイダーの名前を返します。
        この実装では常に「openai」を返します。
        """
        return provider_name

    @staticmethod
    def get_image_object(image_path) -> dict:
        """
        Converts an image file to the OpenAI-specific format for image analysis.
        
        OpenAI models like GPT-4o support image inputs in a specific format.
        This method handles the conversion of an image file to that format.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: A dictionary with the image data formatted for OpenAI models,
                  with the structure:
                  {
                      "type": "image_url",
                      "image_url": {"url": "data:[mime_type];base64,[image_data]"}
                  }
                  
        Example:
            >>> image_obj = model.get_image_object("path/to/image.jpg")
            >>> response = model.invoke([
            ...     {"type": "text", "text": "Describe this image:"},
            ...     image_obj
            ... ])
            
        画像ファイルをOpenAI固有の形式に変換し、画像分析に使用します。
        
        GPT-4oなどのOpenAIモデルは、特定の形式で画像入力をサポートしています。
        このメソッドは、画像ファイルをその形式に変換する処理を行います。
        """
        mime_type, image_data = image_path_to_image_data(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        }
