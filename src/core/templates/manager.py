"""
Template management for file-based templates.

This module provides classes for managing file-based templates, such as HTML,
Markdown, or other text-based templates.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Callable, List, Union, Any

from src.config import TEMPLATE_DIRS
from src.core.prompts.utils import (
    extract_variables,
    format_variables,
)


logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manager for file-based templates.
    
    This class provides methods for loading and formatting templates from files.
    """
    
    def __init__(self, base_directories: Optional[List[str]] = None):
        """
        Initialize a template manager.
        
        Args:
            base_directories: List of directories to search for templates
        """
        self.base_directories = base_directories or TEMPLATE_DIRS
        self.templates_cache: Dict[str, str] = {}
        self.template_loaders: Dict[str, Callable[[str], str]] = {}
    
    def register_directory(self, directory: str) -> None:
        """
        Register a directory to search for templates.
        
        Args:
            directory: Directory path
        """
        if os.path.isdir(directory) and directory not in self.base_directories:
            self.base_directories.append(directory)
    
    def register_loader(self, extension: str, loader: Callable[[str], str]) -> None:
        """
        Register a custom loader for a specific file extension.
        
        Args:
            extension: File extension (e.g., '.html', '.md')
            loader: Function that takes a file path and returns the template content
        """
        self.template_loaders[extension] = loader
    
    def get_template(self, template_name: str) -> str:
        """
        Get a template by name.
        
        Args:
            template_name: Template name or path
            
        Returns:
            Template content
            
        Raises:
            FileNotFoundError: If the template file is not found
        """
        # Check cache first
        if template_name in self.templates_cache:
            return self.templates_cache[template_name]
        
        # Normalize path
        template_path = Path(template_name)
        
        # Check if the template name is an absolute path
        if template_path.is_absolute() and template_path.exists():
            return self._load_template(template_path)
        
        # Search in base directories
        for directory in self.base_directories:
            full_path = Path(directory) / template_path
            if full_path.exists():
                return self._load_template(full_path)
        
        # Not found
        raise FileNotFoundError(
            f"Template '{template_name}' not found in directories: {self.base_directories}"
        )
    
    def format_template(self, template_name: str, **variables) -> str:
        """
        Format a template with variables.
        
        Args:
            template_name: Template name or path
            **variables: Variables to format the template with
            
        Returns:
            Formatted template
        """
        template = self.get_template(template_name)
        return format_variables(template, variables)
    
    def get_template_variables(self, template_name: str) -> List[str]:
        """
        Get the variables used in a template.
        
        Args:
            template_name: Template name or path
            
        Returns:
            List of variable names
        """
        template = self.get_template(template_name)
        return extract_variables(template)
    
    def _load_template(self, path: Path) -> str:
        """
        Load a template from a file.
        
        Args:
            path: Template file path
            
        Returns:
            Template content
        """
        # Check if we have a custom loader for this extension
        extension = path.suffix.lower()
        if extension in self.template_loaders:
            content = self.template_loaders[extension](str(path))
        else:
            # Default loader
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Cache the template
        template_key = str(path)
        self.templates_cache[template_key] = content
        
        return content
    
    def clear_cache(self) -> None:
        """
        Clear the template cache.
        """
        self.templates_cache.clear()
    
    def reload_template(self, template_name: str) -> str:
        """
        Reload a template from file.
        
        Args:
            template_name: Template name or path
            
        Returns:
            Updated template content
        """
        # Remove from cache
        if template_name in self.templates_cache:
            del self.templates_cache[template_name]
        
        # Load again
        return self.get_template(template_name)
    
    def list_templates(self, directory: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            directory: Optional directory to list templates from
            
        Returns:
            List of template names
        """
        templates = []
        
        # Directories to search
        dirs_to_search = [directory] if directory else self.base_directories
        
        for base_dir in dirs_to_search:
            if not os.path.isdir(base_dir):
                continue
                
            for root, _, files in os.walk(base_dir):
                for file in files:
                    # Skip non-template files
                    if file.startswith('.'):
                        continue
                        
                    # Get relative path from base directory
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    
                    templates.append(rel_path)
        
        return templates
