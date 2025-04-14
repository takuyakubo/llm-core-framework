"""
Utility functions for prompt management.

This module provides utility functions for extracting and formatting variables
in prompt templates.
"""

import re
from string import Formatter
from typing import List, Dict, Any, Set, Union


def extract_variables(text: str) -> List[str]:
    """
    Extract variables from a template string.
    
    Variables are identified by the pattern {variable_name}.
    
    Args:
        text: Template string
        
    Returns:
        List of variable names
    """
    formatter = Formatter()
    variables = []
    
    for _, field_name, _, _ in formatter.parse(text):
        if field_name is not None and field_name not in variables:
            variables.append(field_name)
            
    return variables


def extract_variables_recursive(obj: Any) -> Set[str]:
    """
    Recursively extract variables from nested objects.
    
    Args:
        obj: Object to extract variables from (string, list, dict, etc.)
        
    Returns:
        Set of variable names
    """
    variables = set()
    
    if isinstance(obj, str):
        variables.update(extract_variables(obj))
    elif isinstance(obj, list):
        for item in obj:
            variables.update(extract_variables_recursive(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            variables.update(extract_variables_recursive(value))
            
    return variables


def format_variables(template: str, variables: Dict[str, Any]) -> str:
    """
    Format a template string with variables.
    
    Args:
        template: Template string
        variables: Dictionary of variable values
        
    Returns:
        Formatted string
    """
    return template.format(**variables)


def format_variables_recursive(obj: Any, variables: Dict[str, Any]) -> Any:
    """
    Recursively format variables in nested objects.
    
    Args:
        obj: Object to format (string, list, dict, etc.)
        variables: Dictionary of variable values
        
    Returns:
        Formatted object
    """
    if isinstance(obj, str):
        try:
            return format_variables(obj, variables)
        except KeyError:
            # If a variable is missing, return the original string
            return obj
    elif isinstance(obj, list):
        return [format_variables_recursive(item, variables) for item in obj]
    elif isinstance(obj, dict):
        return {key: format_variables_recursive(value, variables) for key, value in obj.items()}
    else:
        return obj


def parse_message_content(content: Union[str, List, Dict]) -> str:
    """
    Parse message content into a string.
    
    This function handles different content formats used by different LLM providers.
    
    Args:
        content: Message content (string, list, dict, etc.)
        
    Returns:
        String representation of the content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle list of content parts (e.g., for multimodal models)
        text_parts = []
        
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                # Skip image parts or other non-text content
                
        return " ".join(text_parts)
    elif isinstance(content, dict):
        # Handle dict content (e.g., for function calling)
        if content.get("type") == "text":
            return content.get("text", "")
        # Handle other formats
        
    return str(content)
