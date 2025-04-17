"""
Workflow state management module.

This module provides the base class for state management in workflow graphs.
States represent the data flowing through the workflow, being transformed
by each node in the process.

Example usage:
    from pydantic import Field
    from core.graphs.states import NodeState
    
    class ImageAnalysisState(NodeState):
        image_path: str = Field(description="Path to the image file to analyze")
        prompt: str = Field(description="Prompt to use for the analysis")
        result: str = Field(default="", description="Analysis result")
"""

from pydantic import BaseModel, Field


class NodeState(BaseModel):
    """
    Base class for all workflow state objects.
    
    This class provides the foundation for state management in workflow graphs.
    It includes basic functionality common to all states, such as error handling,
    and serves as the base class that specific workflow states should inherit from.
    
    Workflow states are Pydantic models that represent the data flowing through
    the nodes in a workflow graph. Each node in the graph receives a state object,
    processes it, and passes the updated state to the next node.
    
    Attributes:
        error (str): Error message if an error occurred during processing
        
    ワークフロー状態オブジェクトのベースクラス。
    
    このクラスはワークフローグラフにおける状態管理の基盤を提供します。エラー処理など、
    すべての状態に共通する基本機能を含み、特定のワークフロー状態が継承すべき
    ベースクラスとして機能します。
    
    ワークフロー状態は、ワークフローグラフ内のノード間を流れるデータを表す
    Pydanticモデルです。グラフの各ノードは状態オブジェクトを受け取り、
    処理し、更新された状態を次のノードに渡します。
    """

    error: str = Field(
        default="",
        description="Error message if an error occurred during processing",
    )

    def emit_error(self, error_str: str):
        """
        Create a new state object with an error message.
        
        This method is used to signal an error condition in the workflow.
        When a node encounters an error, it should call this method to
        create a new state with the error message, which will then be
        passed to the workflow's error handling logic.
        
        Args:
            error_str (str): The error message to include in the state
            
        Returns:
            NodeState: A new state object with the error message set
            
        Example:
            >>> state = current_state.emit_error("Failed to process image")
            >>> assert state.error == "Failed to process image"
            
        エラーメッセージを含む新しい状態オブジェクトを作成します。
        
        このメソッドはワークフロー内でエラー状態を示すために使用されます。
        ノードがエラーに遭遇した場合、このメソッドを呼び出して
        エラーメッセージを含む新しい状態を作成し、それをワークフローの
        エラー処理ロジックに渡す必要があります。
        """
        return self.model_copy(update={"error": error_str})
