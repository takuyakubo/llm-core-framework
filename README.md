# LLM Core Framework

A flexible core framework for building LLM-powered applications with abstracted provider interfaces, media handling, prompt management, and workflow orchestration.

## Features

- **Provider Agnostic**: Unified interface for multiple LLM providers (OpenAI, Anthropic, Google)
- **Media Processing**: Utilities for handling images and other media types
- **Prompt Management**: Standardized system for managing prompts across providers
- **Template Management**: Flexible system for file-based templates
- **Workflow Framework**: Build complex multi-step LLM applications

## Structure

```
src/
├── core/                # Core framework components
│   ├── llm/             # LLM abstraction layer
│   │   ├── factory.py   # Factory for creating models
│   │   ├── models.py    # Base model interfaces
│   │   └── providers/   # Provider implementations
│   ├── media/           # Media processing utilities
│   │   ├── images.py    # Image processing
│   │   └── formatters.py # Provider-specific formatters
│   ├── prompts/         # Prompt management
│   │   ├── manager.py   # Multi-provider prompt management
│   │   └── utils.py     # Prompt utilities
│   ├── templates/       # Template management
│   │   └── manager.py   # Template loading and formatting
│   └── workflow/        # Workflow framework
│       ├── base.py      # Workflow nodes and pipelines
│       └── states.py    # State management
└── config.py            # Global configuration
```

## Installation

```bash
# Clone the repository
git clone https://github.com/takuyakubo/llm-core-framework.git
cd llm-core-framework

# Install requirements
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google
GOOGLE_API_KEY=your_google_api_key

# Framework configuration
LANGCHAIN_MAX_CONCURRENCY=5
DEBUG_MODE=False
```

## Detailed Documentation

### Creating and Using LLM Models

The framework provides a unified interface for working with different LLM providers through the `ModelFactory` class:

```python
from src.core.llm import ModelFactory

# Create models from different providers with the same interface
claude = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
gpt = ModelFactory.create("gpt-4o", max_tokens=1000)
gemini = ModelFactory.create("gemini-2.5-pro-preview-03-25", max_tokens=1000)

# Use them in the same way
for model in [claude, gpt, gemini]:
    response = model.invoke("Tell me about AI")
    print(f"{model.model_name}: {response}")
```

Common parameters for model creation:
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Alternative to temperature for controlling randomness
- `stop`: List of strings that will stop generation when encountered

### Processing Images with LLMs

The framework provides utilities for handling images and sending them to vision-capable LLMs:

```python
from src.core.llm import ModelFactory
from src.core.llm.utils import image_path_to_image_data

# Create a model
model = ModelFactory.create("claude-3-7-sonnet-latest")

# Process an image
image_path = "path/to/image.jpg"
mime_type, image_data = image_path_to_image_data(image_path)

# Get provider-specific image object
image_obj = model.get_image_object(image_path)

# Send to model with a prompt
response = model.invoke([
    {"type": "text", "text": "Describe what you see in this image:"},
    image_obj
])
```

### Managing Prompts Across Providers

The `PromptManager` allows you to define provider-specific prompts with a consistent interface:

```python
from src.core.prompts.managers import PromptManager
from langchain_core.messages import SystemMessage, HumanMessage
from src.core.llm.providers import ProviderType

# Create a prompt manager
analyze_image_prompt = PromptManager("analyze_image_prompt", description="Image analysis prompt")

# Define provider-specific prompts
analyze_image_prompt[ProviderType.ANTHROPIC.value] = [
    SystemMessage(content="You are an image analysis assistant."),
    HumanMessage(content="Analyze the following image and describe what you see: {image_description}")
]

analyze_image_prompt[ProviderType.OPENAI.value] = [
    SystemMessage(content="Analyze images in detail."),
    HumanMessage(content="Please look at this image and provide a detailed analysis: {image_description}")
]

# Add attachment support for images
analyze_image_prompt.append_attach_key("image_data")

# Use the prompt with a specific provider
provider = model.provider_name
formatted_prompt = analyze_image_prompt[provider]({
    "image_description": "This is a photo of a mountain landscape.",
    "_attach_image_data": image_obj
})

# Send to model
response = model.invoke(formatted_prompt)
```

### Building Workflows with Nodes and States

The framework includes a workflow system that allows you to build complex multi-step LLM applications:

```python
from typing import List
from src.core.llm import ModelFactory
from src.core.graphs.elements import LangGraphNode
from src.core.graphs.networks import SequentialWorkflow
from src.core.graphs.states import NodeState
from pydantic import Field

# 1. Define your workflow state
class ImageAnalysisState(NodeState):
    image_path: str = Field(description="Path to the image to analyze")
    prompt: str = Field(description="Prompt to guide the analysis")
    analysis_result: str = Field(default="", description="Result of the analysis")
    summary: str = Field(default="", description="Summary of the analysis")

# 2. Define workflow nodes
class LoadImageNode(LangGraphNode[ImageAnalysisState]):
    name = "load_image"
    
    def proc(self, state: ImageAnalysisState) -> ImageAnalysisState:
        # Process the image - in real code, you would load and prepare the image here
        print(f"Loading image from {state.image_path}")
        return state

class AnalyzeImageNode(LangGraphNode[ImageAnalysisState]):
    name = "analyze_image"
    
    def proc(self, state: ImageAnalysisState) -> ImageAnalysisState:
        # Use the LLM to analyze the image
        image_obj = self.llm.get_image_object(state.image_path)
        result = self.llm.invoke([
            {"type": "text", "text": state.prompt},
            image_obj
        ])
        return state.model_copy(update={"analysis_result": result})

class SummarizeResultNode(LangGraphNode[ImageAnalysisState]):
    name = "summarize_result"
    
    def proc(self, state: ImageAnalysisState) -> ImageAnalysisState:
        # Use the LLM to summarize the analysis
        summary = self.llm.invoke(f"Summarize this analysis: {state.analysis_result}")
        return state.model_copy(update={"summary": summary})

# 3. Create the workflow
def create_image_analysis_workflow():
    # Initialize LLM
    llm = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
    
    # Create nodes
    nodes = [
        LoadImageNode(llm),
        AnalyzeImageNode(llm),
        SummarizeResultNode(llm)
    ]
    
    # Create workflow
    workflow = SequentialWorkflow(nodes, ImageAnalysisState)
    return workflow.get_app()

# 4. Use the workflow
if __name__ == "__main__":
    app = create_image_analysis_workflow()
    result = app.invoke({
        "image_path": "path/to/image.jpg",
        "prompt": "Describe what you see in this image in detail."
    })
    
    print("Analysis result:", result.analysis_result)
    print("Summary:", result.summary)
```

### Using Templates for Output Formatting

The framework includes a template system for formatting output:

```python
from pathlib import Path
from src.core.templates.managers import TemplateManager
from string import Template

# Create a template manager
current_file = Path(__file__)
template_manager = TemplateManager(current_file)

# Load a template
template_manager.check_("example.html")
template_content = template_manager.content

# Format the template with variables
template = Template(template_content)
result = template.substitute(
    title="Image Analysis Report",
    subtitle="Generated by LLM Core Framework",
    date="2025-04-17",
    toc="<li>Image Overview</li><li>Detailed Analysis</li><li>Conclusion</li>",
    description="An automated analysis of the provided image.",
    content_slides="<div class='slide'>...</div>",
    summary="<li>The image contains a mountain landscape</li><li>There are trees in the foreground</li>",
    conclusion="The analysis provides a detailed description of the natural scene.",
    footer="Generated on 2025-04-17"
)

# Write the result to a file
with open("output.html", "w") as f:
    f.write(result)
```

## Error Handling

The framework includes a built-in error handling system. In workflows, errors are caught and propagated through the state:

```python
# Error handling in workflow nodes
def proc(self, state):
    try:
        # Your processing logic here
        return state.model_copy(update={"result": result})
    except Exception as e:
        # Errors are captured in the state
        return state.emit_error(f"Error in {self.name}: {str(e)}")
```

You can control whether exceptions are raised or handled by setting the `DEBUG_MODE` in your configuration:

```python
# In config.py or .env
DEBUG_MODE = True  # Raise exceptions for debugging
```

## Extending the Framework

### Adding a New Provider

To add support for a new LLM provider:

1. Create a new provider module in `src/core/llm/providers/`
2. Implement the `UnifiedModel` interface
3. Register the provider in `src/core/llm/providers/__init__.py`

Example:

```python
# src/core/llm/providers/new_provider.py
from src.core.llm.models import UnifiedModel
from src.core.llm.utils import image_path_to_image_data

provider_name = "new_provider"

class NewProviderModel(UnifiedModel):
    def __init__(self, model_name, **kwargs):
        self._model_name = model_name
        # Initialize the provider's client
        
    @property
    def model_name(self) -> str:
        return self._model_name
        
    @property
    def provider_name(self) -> str:
        return provider_name
        
    def invoke(self, prompt, **kwargs):
        # Implement the provider-specific invocation
        
    @staticmethod
    def get_image_object(image_path) -> dict:
        mime_type, image_data = image_path_to_image_data(image_path)
        # Return provider-specific image format
```

Then register the provider:

```python
# In src/core/llm/providers/__init__.py
from src.core.llm.providers.new_provider import NewProviderModel
from src.core.llm.providers.new_provider import provider_name as npn

class ProviderType(Enum):
    # Add the new provider
    NEW_PROVIDER = npn
    
model_registry = {
    # Add the new provider
    ProviderType.NEW_PROVIDER.value: NewProviderModel,
}

def get_provider(model_name: str) -> str:
    # Add logic to determine the provider from model name
    if model_name.startswith("new-"):
        return ProviderType.NEW_PROVIDER.value
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

The framework is designed to be easily testable. Here's an example of how to test a workflow:

```python
import unittest
from src.core.llm import ModelFactory
from your_module import create_image_analysis_workflow

class TestImageAnalysisWorkflow(unittest.TestCase):
    def setUp(self):
        # Use a mock LLM for testing
        self.mock_llm = ModelFactory.create("mock-llm")
        self.workflow = create_image_analysis_workflow(self.mock_llm)
        
    def test_workflow_execution(self):
        result = self.workflow.invoke({
            "image_path": "test/fixtures/test_image.jpg",
            "prompt": "Test prompt"
        })
        
        self.assertIsNotNone(result.analysis_result)
        self.assertIsNotNone(result.summary)
        self.assertEqual(result.error, "")  # No errors
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
