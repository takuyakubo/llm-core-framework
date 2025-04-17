"""
Workflow network definitions module.

This module provides classes for creating different types of workflow networks,
such as sequential workflows. These networks define how nodes are connected
and how data flows through the workflow.

Example usage:
    from typing import List
    from core.graphs.elements import LangGraphNode
    from core.graphs.networks import SequentialWorkflow
    from core.graphs.states import NodeState
    
    # Define custom state
    class MyWorkflowState(NodeState):
        input_data: str
        result: str = ""
    
    # Define custom nodes
    class ProcessDataNode(LangGraphNode[MyWorkflowState]):
        name = "process_data"
        
        def proc(self, state: MyWorkflowState) -> MyWorkflowState:
            # Process the data
            result = self.llm.invoke(f"Process this: {state.input_data}")
            return state.model_copy(update={"result": result})
    
    class FormatResultNode(LangGraphNode[MyWorkflowState]):
        name = "format_result"
        
        def proc(self, state: MyWorkflowState) -> MyWorkflowState:
            # Format the result
            formatted = f"RESULT: {state.result}"
            return state.model_copy(update={"result": formatted})
    
    # Create a sequential workflow
    nodes = [ProcessDataNode(llm), FormatResultNode(llm)]
    workflow = SequentialWorkflow(nodes, MyWorkflowState)
    app = workflow.get_app()
    
    # Execute the workflow
    result = app.invoke({"input_data": "Hello, world!"})
    print(result.result)  # Prints: "RESULT: [processed data]"
"""

from typing import List

from langgraph.graph import END, START, StateGraph

from core.graphs.elements import LangGraphConditionalEdge, LangGraphNode


class SequentialWorkflow:
    """
    Sequential workflow implementation.
    
    This class creates a linear workflow where nodes are executed in sequence.
    Each node processes the state and passes it to the next node in the chain.
    The workflow handles errors by detecting them in the state and terminating
    the execution if any errors are found.
    
    Attributes:
        workflow (StateGraph): The underlying state graph for the workflow
        
    シーケンシャルワークフローの実装。
    
    このクラスは、ノードが順番に実行される線形ワークフローを作成します。
    各ノードは状態を処理し、それをチェーン内の次のノードに渡します。
    ワークフローは状態内でエラーを検出し、エラーが見つかった場合は
    実行を終了することでエラーを処理します。
    """

    def __init__(self, nodes: List[LangGraphNode], init_state_cls) -> None:
        """
        Initialize a sequential workflow.
        
        Args:
            nodes (List[LangGraphNode]): List of nodes to include in the workflow
            init_state_cls: The class to use for initializing the workflow state
            
        Example:
            >>> nodes = [ProcessDataNode(llm), FormatResultNode(llm)]
            >>> workflow = SequentialWorkflow(nodes, MyWorkflowState)
            >>> app = workflow.get_app()
            >>> result = app.invoke({"input_data": "Hello, world!"})
            
        シーケンシャルワークフローを初期化します。
        
        引数：
            nodes (List[LangGraphNode]): ワークフローに含めるノードのリスト
            init_state_cls: ワークフロー状態の初期化に使用するクラス
        """
        self.workflow = StateGraph(init_state_cls)
        self.setup(nodes)

    def setup(self, nodes: List[LangGraphNode]) -> None:
        """
        Set up the workflow graph with the provided nodes.
        
        This method creates a linear graph where each node is connected to the next
        with conditional edges that check for errors. The first node is connected to
        the START sentinel, and the last node is connected to the END sentinel.
        
        Args:
            nodes (List[LangGraphNode]): List of nodes to include in the workflow
            
        提供されたノードでワークフローグラフをセットアップします。
        
        このメソッドは、各ノードがエラーをチェックする条件付きエッジで次のノードに
        接続される線形グラフを作成します。最初のノードはSTARTセンチネルに接続され、
        最後のノードはENDセンチネルに接続されます。
        """
        nodes_with_sentinels = [START] + nodes + [END]
        edges = [
            LangGraphConditionalEdge(s, t)
            for s, t in zip(nodes_with_sentinels, nodes_with_sentinels[1:])
        ]
        for e in edges:
            if e.target == END:
                self.workflow.add_edge(e.source.node_name, END)
                continue
            self.workflow.add_node(*e.target.generate_node())
            if e.source == START:
                self.workflow.set_entry_point(e.target.node_name)
            else:
                self.workflow.add_conditional_edges(*e.args_conditional_edge())

    def get_app(self):
        """
        Compile the workflow and get the runnable application.
        
        Returns:
            RunnableInterface: A runnable workflow application that can be invoked
            
        Example:
            >>> app = workflow.get_app()
            >>> result = app.invoke({"input_data": "Hello, world!"})
            
        ワークフローをコンパイルして実行可能なアプリケーションを取得します。
        
        戻り値：
            RunnableInterface: 呼び出し可能なワークフローアプリケーション
        """
        return self.workflow.compile()
