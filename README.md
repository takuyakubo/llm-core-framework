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

## Usage Examples

### Create a unified model

```python
from src.core.llm import ModelFactory

# Create models from different providers with the same interface
claude = ModelFactory.create("claude-3-7-sonnet-latest", max_tokens=1000)
gpt = ModelFactory.create("gpt-4o", max_tokens=1000)
gemini = ModelFactory.create("gemini-2.5-pro-preview-03-25", max_tokens=1000)

# Use them in the same way
response = claude.invoke("Tell me about AI")
```

### Process images

```python
from src.core.media import load_image, image_to_base64

# Load and encode an image
image = load_image("path/to/image.jpg")
image_data = image_to_base64(image)

# Process with an LLM
model = ModelFactory.create("claude-3-7-sonnet-latest")
result = model.process_with_images("Describe this image:", [image_data])
```

### Create a workflow

See the full example in `examples/image_analysis.py` for a workflow that:
1. Loads and prepares an image
2. Analyzes the image with an LLM
3. Returns the analysis results

Run the example:

```bash
python examples/image_analysis.py path/to/image.jpg "Describe what you see in this image"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
