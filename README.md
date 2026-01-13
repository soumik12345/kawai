# Kawai

The Cute agentic framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **ReAct Pattern**: Implements Reasoning and Acting paradigm with strict one-tool-per-step execution
- **Tool Calling**: OpenAI-compatible function calling via OpenRouter API
- **Optional Planning**: Multi-step planning with configurable re-planning intervals
- **Rich Logging**: Color-coded console output with syntax highlighting
- **Weave Integration**: Automatic experiment tracking and observability
- **REST API Deployment**: Deploy agents as REST APIs with streaming support

## Installation

```bash
pip install git+https://github.com/soumik12345/kawai
```

## Quick Start

```python
import weave
from kawai import KawaiReactAgent, WebSearchTool, KawaiLoggingCallback, OpenAIModel

# Initialize Weave for tracking
weave.init(project_name="my-project")

# Create agent
agent = KawaiReactAgent(
    model=OpenAIModel(
        model_id="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
    ),
    tools=[WebSearchTool()],
    max_steps=10,
    callbacks=[KawaiLoggingCallback()]
)

# Run task
result = agent.run("What's the latest news on AI?")
print(result["final_answer"])
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
    model=OpenAIModel(
        model_id="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
    ),
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
    model=OpenAIModel(
        model_id="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
    ),
    tools=[WebSearchTool()],
    callbacks=[MyCallback()]
)
```

## Deploying as REST API

Deploy your agent as a REST API server for integration with other applications:

```python
import weave
from kawai import KawaiReactAgent, WebSearchTool, OpenAIModel

weave.init(project_name="kawai-server")

model = OpenAIModel(
    model_id="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
)

agent = KawaiReactAgent(
    model=model,
    tools=[WebSearchTool()],
    max_steps=10,
)

# Start REST API server
agent.serve(port=8000)
```

The server provides these endpoints:

- **POST /chat** - Non-streaming chat endpoint
- **GET /stream** - Server-Sent Events for real-time streaming
- **GET /health** - Health check with server status
- **GET /sessions/{id}** - Get session information
- **DELETE /sessions/{id}** - Delete a session

### Client Usage

**Python client (non-streaming):**
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What is machine learning?"}
)
print(response.json()["answer"])
```

**Python client (streaming):**
```python
import requests
import json

response = requests.get(
    "http://localhost:8000/stream",
    params={"message": "Search for AI news"},
    stream=True
)

for line in response.iter_lines():
    if line:
        event = json.loads(line.decode().removeprefix("data: "))
        print(f"{event['type']}: {event['data']}")
```

**cURL:**
```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!"}'
```

See the [deployment guide](docs/deployment.md) for production deployment options.

## Examples

* [Memory-augmented web-search agent](./examples/web_search.py)

## 🙏 Acknowledgments

Inspired by [HuggingFace smolagents](https://github.com/huggingface/smolagents). ReAct prompts adapted from their toolcalling agent implementation.
