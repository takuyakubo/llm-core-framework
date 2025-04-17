"""
Core abstractions for language models.

This module provides the base classes and interfaces for working with different
language model providers in a unified way.

The UnifiedModel abstract base class defines the common interface that all provider
implementations must follow, allowing the rest of the framework to work with any
LLM provider through a consistent API.

Example usage:
    model = UnifiedModel()  # Implemented by a specific provider
    result = model.invoke("Tell me about AI")
"""

from abc import ABC, abstractmethod


class UnifiedModel(ABC):
    """
    Abstract base class defining the unified interface for all LLM providers.
    
    This class serves as the contract that all provider-specific implementations
    must fulfill, ensuring that the rest of the framework can interact with any
    LLM service using a consistent interface.
    
    抽象基底クラスとして、すべてのLLMプロバイダーのための統一インターフェースを定義します。
    このクラスは、プロバイダー固有の実装が満たさなければならない契約として機能し、
    フレームワークの残りの部分が一貫したインターフェースを使用して任意のLLMサービスと
    対話できることを保証します。
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name of the underlying model.
        
        This should be the official model identifier used by the provider,
        such as 'gpt-4o', 'claude-3-7-sonnet-latest', or 'gemini-2.5-pro-preview-03-25'.
        
        Returns:
            str: The name of the model
            
        基礎となるモデルの名前を返します。
        これはプロバイダーが使用する公式モデル識別子（'gpt-4o'、'claude-3-7-sonnet-latest'、
        または 'gemini-2.5-pro-preview-03-25'など）であるべきです。
        """
        pass

    @property
    def provider_name(self) -> str:
        """
        Returns the name of the model provider.
        
        This should be the standardized identifier for the provider,
        such as 'openai', 'anthropic', or 'google'.
        
        Returns:
            str: The name of the provider
            
        モデルプロバイダーの名前を返します。
        これはプロバイダーの標準化された識別子（'openai'、'anthropic'、または 'google'など）
        であるべきです。
        """
        pass

    def get_image_object(self, image_path) -> dict:
        """
        Converts an image file to the provider-specific format for image analysis.
        
        Different LLM providers have different APIs for handling images. This method
        abstracts those differences, allowing for consistent image handling across providers.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: A dictionary with the image data formatted for the specific provider
            
        画像ファイルをプロバイダー固有の形式に変換し、画像分析に使用できるようにします。
        LLMプロバイダーごとに画像処理のためのAPIが異なります。このメソッドはそれらの違いを
        抽象化し、プロバイダー間で一貫した画像処理を可能にします。
        """
        pass
    
    def invoke(self, prompt: str, **kwargs):
        """
        Invokes the language model with the given prompt.
        
        This is the primary method for interacting with the language model.
        It should handle sending the request to the model provider and returning
        the response.
        
        Args:
            prompt (str): The text prompt to send to the model
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The model's response (type depends on provider implementation)
            
        Raises:
            Various exceptions depending on the provider implementation
            
        指定されたプロンプトでモデルを呼び出します。
        これはモデルと対話するための主要なメソッドです。モデルプロバイダーにリクエストを
        送信し、レスポンスを返す処理を行います。
        """
        raise NotImplementedError("Each provider must implement this method")
