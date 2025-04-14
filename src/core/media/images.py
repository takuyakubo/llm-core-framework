"""
Image processing utilities.

This module provides utilities for working with images including loading,
converting to base64, resizing, and other common operations.
"""

import base64
import io
import os
from typing import Union, Tuple, Optional

from PIL import Image

from src.config import MAX_IMAGE_SIZE_MB, SUPPORTED_IMAGE_FORMATS


def load_image(source: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Load an image from various source types.
    
    Args:
        source: Image source (file path, bytes, or PIL Image)
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If the image format is unsupported
        IOError: If the image file cannot be loaded
    """
    if isinstance(source, str):
        # Handle file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
            
        image = Image.open(source)
        
        # Verify format is supported
        image_format = image.format.lower() if image.format else None
        if image_format not in [fmt.lower() for fmt in SUPPORTED_IMAGE_FORMATS]:
            raise ValueError(f"Unsupported image format: {image_format}. "
                           f"Supported formats: {SUPPORTED_IMAGE_FORMATS}")
                           
        return image
    elif isinstance(source, bytes):
        # Handle bytes
        return Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        # Already a PIL Image
        return source
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")


def image_to_base64(image: Union[str, bytes, Image.Image], 
                    format: str = "PNG",
                    quality: Optional[int] = None) -> str:
    """
    Convert an image to base64-encoded string.
    
    Args:
        image: Image source (file path, bytes, or PIL Image)
        format: Output image format (default: PNG)
        quality: Image quality for lossy formats (1-100, default: None)
        
    Returns:
        Base64-encoded image data string
        
    Raises:
        TypeError: If the image source type is not supported
    """
    if isinstance(image, str):
        # Handle file path
        with open(image, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    elif isinstance(image, bytes):
        # Handle bytes
        return base64.b64encode(image).decode("utf-8")
    elif isinstance(image, Image.Image):
        # Handle PIL Image
        buffered = io.BytesIO()
        save_kwargs = {}
        
        if quality is not None and format.upper() in ["JPEG", "JPG", "WEBP"]:
            save_kwargs["quality"] = quality
            
        image.save(buffered, format=format, **save_kwargs)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def resize_image(image: Union[str, bytes, Image.Image], 
                  max_size: Tuple[int, int] = (1024, 1024),
                  keep_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize an image while optionally maintaining aspect ratio.
    
    Args:
        image: Image source (file path, bytes, or PIL Image)
        max_size: Maximum width and height
        keep_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    img = load_image(image)
    
    if keep_aspect_ratio:
        img.thumbnail(max_size, Image.LANCZOS)
        return img
    else:
        return img.resize(max_size, Image.LANCZOS)


def validate_image_size(image: Union[str, bytes, Image.Image], 
                         max_size_mb: int = MAX_IMAGE_SIZE_MB) -> bool:
    """
    Check if the image size is within the allowed limit.
    
    Args:
        image: Image source (file path, bytes, or PIL Image)
        max_size_mb: Maximum size in megabytes
        
    Returns:
        True if the image is within the size limit, False otherwise
    """
    if isinstance(image, str):
        # Handle file path
        size_bytes = os.path.getsize(image)
    elif isinstance(image, bytes):
        # Handle bytes
        size_bytes = len(image)
    elif isinstance(image, Image.Image):
        # Handle PIL Image - need to estimate
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        size_bytes = buffer.getbuffer().nbytes
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
        
    size_mb = size_bytes / (1024 * 1024)
    return size_mb <= max_size_mb
