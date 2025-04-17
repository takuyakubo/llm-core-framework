"""
Factory module for creating language model instances.

This module provides a factory class for creating language model instances
from different providers through a unified interface.

Example usage:
    # Create models from different providers
    claude = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
    gpt = ModelFactory.create("gpt-4o", max_tokens=1000)
    gemini = ModelFactory.create("gemini-2.5-pro-preview-03-25", max_tokens=1000)
    
    # Use them with the same interface
    for model in [claude, gpt, gemini]:
        response = model.invoke("Tell me about AI")
        print(f"{model.model_name}: {response}")
"""

from typing import Dict, Type

from core.llm.models import UnifiedModel
from core.llm.providers import get_provider, model_registory


class ModelFactory:
    """
    Factory class for creating language model instances.

    This class provides methods for creating model instances from different
    providers with a unified interface. It automatically determines the 
    appropriate provider based on the model name and instantiates the 
    corresponding implementation.
    
    The factory pattern allows the rest of the system to work with
    language models without needing to know the specific details of
    each provider's implementation.
    
    言語モデルインスタンスを作成するためのファクトリークラス。
    
    このクラスは、統一されたインターフェースを持つ異なるプロバイダーからの
    モデルインスタンスを作成するためのメソッドを提供します。モデル名に基づいて
    適切なプロバイダーを自動的に決定し、対応する実装をインスタンス化します。
    
    ファクトリーパターンにより、システムの残りの部分は各プロバイダーの実装の
    詳細を知る必要なく、言語モデルを扱うことができます。
    """

    # Registry of provider-specific model implementations
    _registry: Dict[str, Type[UnifiedModel]] = model_registory

    @classmethod
    def create(cls, model_name: str, **kwargs) -> UnifiedModel:
        """
        Create a model instance for the specified model name.
        
        This method determines the appropriate provider for the given model name
        and instantiates the corresponding model implementation with the provided
        parameters.
        
        Commonly used parameters include:
        - max_tokens: Maximum number of tokens to generate
        - temperature: Randomness of the output (0.0 to 1.0)
        - top_p: Alternative to temperature for controlling randomness
        - stop: List of strings that will stop generation when encountered

        Args:
            model_name (str): Name of the model (e.g., "gpt-4o", "claude-3-7-sonnet-latest")
            **kwargs: Additional arguments specific to the model or provider
                      (e.g., max_tokens, temperature)

        Returns:
            UnifiedModel: An instance of the appropriate model implementation

        Raises:
            ValueError: If the provider is not supported or not registered
            
        Example:
            >>> model = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
            >>> response = model.invoke("Explain quantum computing in simple terms")
            
        指定されたモデル名のモデルインスタンスを作成します。
        
        このメソッドは、与えられたモデル名に適切なプロバイダーを決定し、提供された
        パラメータで対応するモデル実装をインスタンス化します。
        
        一般的に使用されるパラメータには以下が含まれます：
        - max_tokens: 生成する最大トークン数
        - temperature: 出力のランダム性（0.0から1.0）
        - top_p: ランダム性を制御するための温度の代替
        - stop: 出現時に生成を停止する文字列のリスト
        """
        provider = get_provider(model_name)
        if provider not in cls._registry:
            raise ValueError(
                f"Provider '{provider}' is not registered. "
                f"Available providers: {list(cls._registry.keys())}"
            )
        model_class = cls._registry[provider]
        return model_class(model_name, **kwargs)
    
    @classmethod
    def register_provider(cls, provider_name: str, model_class: Type[UnifiedModel]) -> None:
        """
        Register a new provider with its model implementation.
        
        This method allows for extending the factory with additional providers
        beyond those included with the framework.
        
        Args:
            provider_name (str): Name of the provider to register
            model_class (Type[UnifiedModel]): The model class implementation for this provider
            
        Example:
            >>> class CustomLLMModel(UnifiedModel):
            ...     # Implementation of the custom model
            ...     pass
            >>> ModelFactory.register_provider("custom_provider", CustomLLMModel)
            >>> model = ModelFactory.create("custom-model-name")
            
        新しいプロバイダーとそのモデル実装を登録します。
        
        このメソッドにより、フレームワークに含まれているもの以外の追加プロバイダーで
        ファクトリーを拡張することができます。
        """
        cls._registry[provider_name] = model_class
