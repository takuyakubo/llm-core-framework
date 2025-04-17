"""
LLM utility functions module.

This module provides utility functions for working with language models,
including image processing and format conversion for different providers.

Example usage:
    from core.llm.utils import image_to_image_data_str, image_path_to_image_data
    from PIL import Image
    
    # Convert an image file to base64
    image_data = image_to_image_data_str("path/to/image.jpg")
    
    # Get mime type and image data for provider formatting
    mime_type, image_data = image_path_to_image_data("path/to/image.jpg")
    
    # Use in provider-specific format
    openai_format = {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
    }
"""

import base64
import io
import mimetypes

from PIL import Image


def image_to_image_data_str(image):
    """
    Convert an image to a base64-encoded string.
    
    This function converts an image (either a file path or a PIL Image object)
    to a base64-encoded string suitable for inclusion in LLM prompts.
    
    Args:
        image (str or PIL.Image.Image): Image file path or PIL Image object
        
    Returns:
        str: Base64-encoded string representation of the image
        
    Raises:
        Exception: If the image format is not supported
        
    Example:
        >>> # From file path
        >>> image_data = image_to_image_data_str("path/to/image.jpg")
        >>> 
        >>> # From PIL Image
        >>> from PIL import Image
        >>> img = Image.open("path/to/image.jpg")
        >>> image_data = image_to_image_data_str(img)
        
    画像をbase64エンコードされた文字列に変換します。
    
    この関数は、画像（ファイルパスまたはPIL Imageオブジェクト）をLLMプロンプトに
    含めるのに適したbase64エンコードされた文字列に変換します。
    """
    # 画像をbase64エンコード
    if isinstance(image, str):  # 画像がパスとして提供された場合
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    elif isinstance(image, Image.Image):  # PILイメージの場合
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise Exception(f"サポートされていない画像形式です (画像 {image})")


def image_path_to_image_data(image_path):
    """
    Get the mime type and base64-encoded data for an image file.
    
    This function takes an image file path, determines its mime type,
    and converts it to a base64-encoded string. This is useful for
    formatting images for different LLM providers.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: A tuple containing (mime_type, base64_encoded_data)
        
    Example:
        >>> mime_type, image_data = image_path_to_image_data("path/to/image.jpg")
        >>> # Use in provider-specific format
        >>> anthropic_format = {
        ...     "type": "image",
        ...     "source": {
        ...         "type": "base64",
        ...         "media_type": mime_type,
        ...         "data": image_data
        ...     }
        ... }
        
    画像ファイルのMIMEタイプとbase64エンコードされたデータを取得します。
    
    この関数は画像ファイルのパスを受け取り、そのMIMEタイプを判断し、
    base64エンコードされた文字列に変換します。これは、異なるLLMプロバイダー向けに
    画像をフォーマットするのに役立ちます。
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    image_data = image_to_image_data_str(image_path)
    return mime_type, image_data
