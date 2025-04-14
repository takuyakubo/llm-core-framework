"""
Base state classes for workflow management.

This module provides the base state classes and utilities for workflow state management.
"""

from typing import Optional, Dict, Any, ClassVar, List
from pydantic import BaseModel, Field


class BaseState(BaseModel):
    """
    Base state class for workflow nodes.
    
    This class provides the common fields and methods for all workflow states.
    Specific workflow applications should extend this class with application-specific
    fields.
    """
    
    # Error tracking
    error: str = Field(default="", description="Error message if any operation failed")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, 
                                   description="Metadata for tracking and debugging")
    
    # Execution tracking
    executed_nodes: List[str] = Field(default_factory=list, 
                                    description="List of nodes that have been executed")
    
    # Debug mode
    debug_info: Dict[str, Any] = Field(default_factory=dict, 
                                     description="Debug information collected during execution")
    
    # Class variables
    required_fields: ClassVar[List[str]] = []
    
    def track_execution(self, node_name: str) -> None:
        """
        Track that a node has been executed.
        
        Args:
            node_name: Name of the executed node
        """
        if node_name not in self.executed_nodes:
            self.executed_nodes.append(node_name)
    
    def record_error(self, error_message: str, node_name: Optional[str] = None) -> "BaseState":
        """
        Record an error in the state.
        
        Args:
            error_message: Error message
            node_name: Optional name of the node where the error occurred
            
        Returns:
            Updated state with the error recorded
        """
        state_dict = self.model_dump()
        state_dict["error"] = error_message
        
        if node_name:
            if "error_details" not in state_dict["metadata"]:
                state_dict["metadata"]["error_details"] = {}
            
            state_dict["metadata"]["error_details"][node_name] = error_message
            
        return self.__class__(**state_dict)
    
    def add_debug_info(self, key: str, value: Any) -> None:
        """
        Add debug information to the state.
        
        Args:
            key: Debug information key
            value: Debug information value
        """
        self.debug_info[key] = value
    
    def validate_required_fields(self) -> List[str]:
        """
        Validate that all required fields are present.
        
        Returns:
            List of missing required fields
        """
        missing_fields = []
        
        for field in self.required_fields:
            value = getattr(self, field, None)
            
            # Check if the field exists and is not empty
            if value is None:
                missing_fields.append(field)
            elif isinstance(value, str) and not value:
                missing_fields.append(field)
            elif isinstance(value, list) and not value:
                missing_fields.append(field)
                
        return missing_fields
