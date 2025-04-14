"""
Image analysis example using the LLM core framework.

This example shows how to:
1. Create a prompt template for image analysis
2. Create a workflow for processing images
3. Process images using different LLM providers
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.llm import ModelFactory
from src.core.media import image_to_base64, load_image
from src.core.prompts import PromptTemplate, PromptManager
from src.core.workflow import WorkflowNode, SequentialWorkflow, BaseState
from src.core.llm.factory import ProviderType

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field
from typing import List, Dict, Any, Optional


# Define a state class for image analysis
class ImageAnalysisState(BaseState):
    image_path: str = Field(default="", description="Path to the image to analyze")
    analysis_prompt: str = Field(default="Describe this image in detail", description="Prompt for image analysis")
    image_content: str = Field(default="", description="Base64-encoded image content")
    analysis_result: str = Field(default="", description="Analysis result from the LLM")


# Define workflow nodes
class PrepareImageNode(WorkflowNode[ImageAnalysisState]):
    def __init__(self, name: str = "prepare_image"):
        super().__init__(name)
    
    def validate(self, state: ImageAnalysisState) -> None:
        if not state.image_path:
            raise ValueError("Image path is required")
    
    def process(self, state: ImageAnalysisState) -> ImageAnalysisState:
        # Load and encode the image
        image = load_image(state.image_path)
        image_data = image_to_base64(image)
        
        # Update state
        state.image_content = image_data
        return state


class AnalyzeImageNode(WorkflowNode[ImageAnalysisState]):
    def __init__(self, llm, name: str = "analyze_image"):
        super().__init__(name)
        self.llm = llm
        
        # Create prompt templates for different providers
        self.prompt_manager = PromptManager()
        self.setup_prompts()
    
    def setup_prompts(self):
        # Create a prompt template
        template = PromptTemplate("image_analysis", "Analyze an image")
        
        # Add templates for different providers
        anthropic_template = [
            SystemMessage(content="You are an expert image analyzer. Describe the image in detail."),
            HumanMessage(content=[
                {"type": "text", "text": "{analysis_prompt}"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "{image_content}"}}
            ])
        ]
        
        openai_template = [
            SystemMessage(content="You are an expert image analyzer. Describe the image in detail."),
            HumanMessage(content=[
                {"type": "text", "text": "{analysis_prompt}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_content}"}}
            ])
        ]
        
        # Same format for Google as OpenAI
        google_template = openai_template
        
        # Add templates to the prompt template
        template.add_template(ProviderType.ANTHROPIC.value, anthropic_template)
        template.add_template(ProviderType.OPENAI.value, openai_template)
        template.add_template(ProviderType.GOOGLE.value, google_template)
        
        # Register the template with the manager
        self.prompt_manager.register_template(template)
    
    def validate(self, state: ImageAnalysisState) -> None:
        if not state.image_content:
            raise ValueError("Image content is required")
    
    def process(self, state: ImageAnalysisState) -> ImageAnalysisState:
        # Format the prompt based on the model provider
        prompt = self.prompt_manager.format(
            "image_analysis", 
            self.llm.model_name, 
            analysis_prompt=state.analysis_prompt,
            image_content=state.image_content
        )
        
        # Invoke the model
        response = self.llm.invoke(prompt)
        
        # Update state
        state.analysis_result = response
        return state


def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python image_analysis.py <image_path> [prompt]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail"
    
    # Create initial state
    state = ImageAnalysisState(
        image_path=image_path,
        analysis_prompt=prompt
    )
    
    # Create a model (use environment variables for API keys)
    model = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
    
    # Create workflow nodes
    prepare_node = PrepareImageNode()
    analyze_node = AnalyzeImageNode(model)
    
    # Create and run the workflow
    workflow = SequentialWorkflow(
        nodes=[prepare_node, analyze_node],
        state_class=ImageAnalysisState,
        name="image_analysis_workflow"
    )
    
    # Run the workflow
    result = workflow.run(state)
    
    # Print the result
    if result.error:
        print(f"Error: {result.error}")
    else:
        print("\n--- Image Analysis Result ---\n")
        print(result.analysis_result)


if __name__ == "__main__":
    main()
