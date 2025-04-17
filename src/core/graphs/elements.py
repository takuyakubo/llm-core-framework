"""
Workflow graph elements module.

This module provides the core components for building workflow graphs,
including nodes and edges. These elements are the building blocks used
to create complex LLM processing pipelines.

Example usage:
    from typing import List
    from core.graphs.elements import LangGraphNode
    from core.graphs.networks import SequentialWorkflow
    from core.graphs.states import NodeState
    
    # Define custom node states
    class ImageAnalysisState(NodeState):
        image_path: str
        prompt: str
        result: str = ""
    
    # Define custom workflow nodes
    class LoadImageNode(LangGraphNode[ImageAnalysisState]):
        name = "load_image"
        
        def proc(self, state: ImageAnalysisState) -> ImageAnalysisState:
            # Load and process image...
            return state
    
    class AnalyzeImageNode(LangGraphNode[ImageAnalysisState]):
        name = "analyze_image"
        
        def proc(self, state: ImageAnalysisState) -> ImageAnalysisState:
            # Analyze image with LLM...
            result = self.llm.invoke(f"{state.prompt} {image_data}")
            return state.model_copy(update={"result": result})
    
    # Create and run workflow
    nodes = [LoadImageNode(llm), AnalyzeImageNode(llm)]
    workflow = SequentialWorkflow(nodes, ImageAnalysisState)
    app = workflow.get_app()
    
    # Execute workflow
    result = app.invoke({
        "image_path": "path/to/image.jpg",
        "prompt": "Describe what you see in this image"
    })
"""

import logging
from typing import Callable, Generic, Tuple, TypeVar

from langgraph.graph import END

from config import DEBUG_MODE
from core.graphs.states import NodeState

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="NodeState")


class LangGraphNode(Generic[T]):
    """
    Base class for language model workflow nodes.
    
    This class represents a node in a workflow graph that processes a state object
    and returns an updated state. It provides common functionality for all nodes,
    such as error handling, validation, and a standardized interface for the
    workflow framework.
    
    Each node should override the `proc` method to implement its specific processing
    logic. The `validate` method can be overridden to implement validation logic
    for the incoming state.
    
    Attributes:
        name (str): The name of the node, used for logging and graph visualization
        llm: The language model instance to use for this node's processing
    
    言語モデルワークフローノードのベースクラス。
    
    このクラスは、状態オブジェクトを処理して更新された状態を返す、ワークフローグラフ内の
    ノードを表します。エラー処理、検証、およびワークフローフレームワークのための
    標準化されたインターフェースなど、すべてのノードに共通する機能を提供します。
    
    各ノードは、特定の処理ロジックを実装するために「proc」メソッドをオーバーライドする
    必要があります。「validate」メソッドをオーバーライドして、入力状態の検証ロジックを
    実装することもできます。
    """

    name: str = "to be setup"

    def __init__(self, llm) -> None:
        """
        Initialize a workflow node.
        
        Args:
            llm: The language model instance to use for this node
            
        ワークフローノードを初期化します。
        
        引数：
            llm: このノードで使用する言語モデルインスタンス
        """
        self.llm = llm

    def action(self, state: T) -> T:
        """
        Execute the node's action on the given state.
        
        This method wraps the node's processing logic with error handling
        and logging. It first validates the incoming state, then processes
        it, and returns the updated state. If an error occurs during processing,
        it is caught and either re-raised (in debug mode) or returned as an
        error state.
        
        Args:
            state (T): The state object to process
            
        Returns:
            T: The updated state object
            
        Raises:
            Exception: If DEBUG_MODE is True and an error occurs during processing
            
        与えられた状態に対してノードのアクションを実行します。
        
        このメソッドは、エラー処理とロギングを備えたノードの処理ロジックをラップします。
        まず入力状態を検証し、次に処理して、更新された状態を返します。処理中にエラーが
        発生した場合は、キャッチして再度発生させるか（デバッグモードの場合）、
        エラー状態として返します。
        """
        try:
            self.validate(state)
            logger.info(f"{self.name} starts")
            state_ = self.proc(state)
            logger.info(f"{self.name} ends")
            return state_
        except Exception as e:
            if DEBUG_MODE:
                raise e
            return state.emit_error(f"An error occured during {self.name}: {str(e)}")

    def proc(self, state: T) -> T:
        """
        Process the state and return an updated state.
        
        This method should be overridden by subclasses to implement the
        specific processing logic for the node.
        
        Args:
            state (T): The state object to process
            
        Returns:
            T: The updated state object
            
        Raises:
            NotImplementedError: If not overridden by a subclass
            
        状態を処理し、更新された状態を返します。
        
        このメソッドはサブクラスによってオーバーライドされ、ノードの特定の処理ロジックを
        実装する必要があります。
        """
        pass

    def validate(self, state: T) -> None:
        """
        Validate the incoming state before processing.
        
        This method can be overridden by subclasses to implement validation
        logic for the incoming state. If the state is invalid, an exception
        should be raised.
        
        Args:
            state (T): The state object to validate
            
        Raises:
            ValueError: If the state is invalid
            
        処理前に入力状態を検証します。
        
        このメソッドはサブクラスによってオーバーライドされ、入力状態の検証ロジックを
        実装できます。状態が無効な場合は、例外を発生させる必要があります。
        """
        pass

    def generate_node(self) -> Tuple[str, Callable[[T], T]]:
        """
        Generate the node representation for the workflow graph.
        
        Returns:
            Tuple[str, Callable[[T], T]]: A tuple containing the node name
                                         and the action function
                                         
        ワークフローグラフのノード表現を生成します。
        
        戻り値：
            Tuple[str, Callable[[T], T]]: ノード名とアクション関数を含むタプル
        """
        return self.node_name, self.action

    @property
    def node_name(cls) -> str:
        """
        Get the node name for use in the graph.
        
        This property returns a normalized version of the node's name,
        replacing spaces with underscores for compatibility with the graph.
        
        Returns:
            str: The normalized node name
            
        グラフで使用するノード名を取得します。
        
        このプロパティは、グラフとの互換性のために、スペースをアンダースコアに
        置き換えたノード名の正規化されたバージョンを返します。
        """
        return cls.name.replace(" ", "_")


class LangGraphConditionalEdge:
    """
    Conditional edge for connecting nodes in a workflow graph.
    
    This class represents a conditional edge in a workflow graph that
    determines the flow of execution based on the state. It checks for
    errors in the state and directs the flow accordingly.
    
    Attributes:
        source (LangGraphNode): The source node of the edge
        target (LangGraphNode): The target node of the edge
        
    Example:
        >>> edge = LangGraphConditionalEdge(node1, node2)
        >>> source, check_func, targets = edge.args_conditional_edge()
        >>> # Use these in a workflow graph
        
    ワークフローグラフでノードを接続するための条件付きエッジ。
    
    このクラスは、状態に基づいて実行フローを決定するワークフローグラフ内の
    条件付きエッジを表します。状態内のエラーをチェックし、それに応じてフローを
    指示します。
    """

    def __init__(self, src: LangGraphNode, tgt: LangGraphNode):
        """
        Initialize a conditional edge.
        
        Args:
            src (LangGraphNode): The source node of the edge
            tgt (LangGraphNode): The target node of the edge
            
        条件付きエッジを初期化します。
        
        引数：
            src (LangGraphNode): エッジの始点ノード
            tgt (LangGraphNode): エッジの終点ノード
        """
        self.source = src
        self.target = tgt

    @staticmethod
    def check_error(state: NodeState) -> str:
        """
        Check if the state contains an error.
        
        This method checks if the state has an error message and returns
        "error" if one is found, or "continue" otherwise.
        
        Args:
            state (NodeState): The state to check for errors
            
        Returns:
            str: "error" if an error is found, "continue" otherwise
            
        状態にエラーが含まれているかどうかをチェックします。
        
        このメソッドは、状態にエラーメッセージがあるかどうかをチェックし、
        エラーが見つかった場合は「error」を、そうでない場合は「continue」を返します。
        """
        if state.error != "":
            return "error"
        return "continue"

    def args_conditional_edge(self):
        """
        Generate the arguments for defining a conditional edge in a workflow graph.
        
        Returns:
            tuple: A tuple containing the source node name, check function,
                  and a dictionary mapping check results to target nodes
                  
        ワークフローグラフ内の条件付きエッジを定義するための引数を生成します。
        
        戻り値：
            tuple: ソースノード名、チェック関数、およびチェック結果をターゲット
                  ノードにマッピングする辞書を含むタプル
        """
        return (
            self.source.node_name,
            self.check_error,
            {"error": END, "continue": self.target.node_name},
        )
