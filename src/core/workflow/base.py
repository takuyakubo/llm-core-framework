"""
Base classes for building workflow pipelines.

This module provides the base classes for building workflow pipelines with LLMs,
including node and workflow definitions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Tuple, TypeVar, Type, Optional, Dict, Any

from langgraph.graph import END, START, StateGraph

from src.config import DEBUG_MODE
from src.core.workflow.states import BaseState


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseState)


class WorkflowNode(Generic[T], ABC):
    """
    Base class for workflow nodes.
    
    Workflow nodes are the building blocks of a workflow, encapsulating a specific
    processing step. Subclasses should implement the `process` method to define
    the node's behavior.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a workflow node.
        
        Args:
            name: Optional name for the node. If not provided, the class name will be used.
        """
        self._name = name or self.__class__.__name__
    
    @property
    def name(self) -> str:
        """
        Get the node name.
        
        Returns:
            Node name
        """
        return self._name
    
    @property
    def node_id(self) -> str:
        """
        Get the node ID for the graph.
        
        Returns:
            Node ID (name with spaces replaced by underscores)
        """
        return self.name.replace(" ", "_")
    
    def action(self, state: T) -> T:
        """
        Execute the node action.
        
        This method wraps the `process` method with error handling and logging.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        try:
            logger.info(f"Executing node: {self.name}")
            
            # Track node execution
            state.track_execution(self.name)
            
            # Validate state
            self.validate(state)
            
            # Process state
            result_state = self.process(state)
            
            logger.info(f"Node {self.name} completed successfully")
            return result_state
            
        except Exception as e:
            logger.error(f"Error in node {self.name}: {str(e)}", exc_info=DEBUG_MODE)
            
            if DEBUG_MODE:
                raise
                
            return state.record_error(f"Error in {self.name}: {str(e)}", self.name)
    
    def validate(self, state: T) -> None:
        """
        Validate the state before processing.
        
        This method can be overridden by subclasses to perform validation on the
        input state before processing.
        
        Args:
            state: Current workflow state
            
        Raises:
            ValueError: If validation fails
        """
        pass
    
    @abstractmethod
    def process(self, state: T) -> T:
        """
        Process the state.
        
        This method should be implemented by subclasses to define the node's behavior.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass


class SequentialWorkflow(Generic[T]):
    """
    A sequential workflow made up of nodes.
    
    This class builds a graph of sequential nodes, with error handling and state management.
    """
    
    def __init__(self, 
                nodes: List[WorkflowNode[T]], 
                state_class: Type[T],
                name: str = "sequential_workflow"):
        """
        Initialize a sequential workflow.
        
        Args:
            nodes: List of workflow nodes
            state_class: State class
            name: Workflow name
        """
        self.name = name
        self.nodes = nodes
        self.state_class = state_class
        self.graph = StateGraph(state_class)
        self._build_graph()
    
    def _build_graph(self) -> None:
        """
        Build the workflow graph.
        
        This method builds a sequential graph with error handling.
        """
        # Add all nodes to the graph
        for node in self.nodes:
            self.graph.add_node(node.node_id, node.action)
        
        # Connect nodes sequentially
        for i in range(len(self.nodes) - 1):
            self.graph.add_conditional_edges(
                self.nodes[i].node_id,
                self._check_error,
                {
                    "error": END,
                    "continue": self.nodes[i + 1].node_id
                }
            )
        
        # Connect the last node to END
        self.graph.add_edge(self.nodes[-1].node_id, END)
        
        # Set the entry point
        self.graph.set_entry_point(self.nodes[0].node_id)
    
    @staticmethod
    def _check_error(state: BaseState) -> str:
        """
        Check if an error occurred.
        
        Args:
            state: Current workflow state
            
        Returns:
            "error" if an error occurred, "continue" otherwise
        """
        return "error" if state.error else "continue"
    
    def compile(self) -> Any:
        """
        Compile the workflow graph.
        
        Returns:
            Compiled workflow graph
        """
        return self.graph.compile()
    
    def run(self, initial_state: Optional[T] = None, **kwargs) -> T:
        """
        Run the workflow.
        
        Args:
            initial_state: Optional initial state. If not provided, a new state will be created.
            **kwargs: Additional arguments to pass to the initial state
            
        Returns:
            Final workflow state
        """
        # Create initial state if not provided
        if initial_state is None:
            initial_state = self.state_class(**kwargs)
        elif kwargs:
            # Update initial state with kwargs
            state_dict = initial_state.model_dump()
            state_dict.update(kwargs)
            initial_state = self.state_class(**state_dict)
        
        # Compile the graph
        app = self.compile()
        
        # Run the workflow
        result = app.invoke(initial_state)
        
        return result
