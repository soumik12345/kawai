# Kawai

A Python ReAct agent framework with tool calling and optional planning capabilities.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **ReAct Pattern**: Implements Reasoning and Acting paradigm with strict one-tool-per-step execution
- **Tool Calling**: OpenAI-compatible function calling via OpenRouter API
- **Optional Planning**: Multi-step planning with configurable re-planning intervals
- **Rich Logging**: Color-coded console output with syntax highlighting
- **Weave Integration**: Automatic experiment tracking and observability

## Installation

```bash
pip install git+https://github.com/soumik12345/kawai
```

## Quick Start

```python
import weave
from kawai import KawaiReactAgent, WebSearchTool, KawaiLoggingCallback

# Initialize Weave for tracking
weave.init(project_name="my-project")

# Create agent
agent = KawaiReactAgent(
    model="openai/gpt-4",
    tools=[WebSearchTool()],
    max_steps=10,
    callbacks=[KawaiLoggingCallback()]
)

# Run task
result = agent.run("What's the latest news on AI?")
print(result["final_answer"])
```

## Environment Setup

Create a `.env` file:

```bash
OPENROUTER_API_KEY=your_openrouter_key
SERPER_API_KEY=your_serper_key  # For web search
```

## Built-in Tools

- **WebSearchTool**: Google search via Serper API
- **FinalAnswerTool**: Task completion (auto-added)
- **UserInputTool**: Interactive user prompts

## Creating Custom Tools

```python
from kawai.tools import KawaiTool, KawaiToolParameter
import weave

class CalculatorTool(KawaiTool):
    tool_name: str = "calculator"
    description: str = "Performs arithmetic operations"
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="expression",
            description="Math expression to evaluate",
            tool_type="string"
        )
    ]

    @weave.op
    def forward(self, expression: str) -> dict:
        return {"result": eval(expression)}
```

## Planning Mode

Enable multi-step planning for complex tasks:

```python
agent = KawaiReactAgent(
    model="openai/gpt-4",
    tools=[WebSearchTool()],
    planning_interval=3,  # Re-plan every 3 steps
    max_steps=15
)
```

## Custom Callbacks

Monitor agent execution with custom callbacks:

```python
from kawai.callback import KawaiCallback

class MyCallback(KawaiCallback):
    def at_reasoning(self, reasoning: str):
        print(f"Thinking: {reasoning}")

    def at_tool_call(self, tool_name: str, tool_arguments: dict):
        print(f"Calling {tool_name}")

agent = KawaiReactAgent(
    model="openai/gpt-4",
    tools=[WebSearchTool()],
    callbacks=[MyCallback()]
)
```

## 🙏 Acknowledgments

Inspired by [HuggingFace smolagents](https://github.com/huggingface/smolagents). ReAct prompts adapted from their toolcalling agent implementation.
