# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kawai is a Python-based ReAct agent framework inspired by HuggingFace's smolagents. It implements a tool-calling agent system with optional planning capabilities.

**Key Features:**
- ReAct-style agents that use tool calling via OpenRouter API
- Optional planning system that generates and updates execution plans
- Integration with Weave for experiment tracking
- Built-in tools: web search, final answer, user input

## Development Commands

### Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Linting and Formatting
```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Check format without making changes
ruff format --check --diff .

# Auto-fix linting issues
ruff check --fix .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run with verbose output
pytest -v

# Run tests with asyncio support
pytest --asyncio-mode=auto
```

### Documentation
```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy documentation to GitHub Pages
mkdocs gh-deploy
```

### Running Examples
```bash
# Ensure environment variables are set in .env
# Required: OPENROUTER_API_KEY, SERPER_API_KEY (for web search)

python examples/web_search.py
```

## Architecture Overview

### Core Components

#### 1. KawaiReactAgent ([kawai/agents/react.py](kawai/agents/react.py))
The main agent class that orchestrates the ReAct loop:
- **Memory Management**: Maintains conversation history in OpenAI-compatible format with roles (system, user, assistant) and function calls/outputs
- **Tool Execution**: Uses OpenRouter's responses API (compatible with OpenAI format) to call tools
- **Planning System**: Optional multi-step planning with configurable intervals
  - `planning_interval=None`: No planning (default)
  - `planning_interval=N`: Plans initially and re-plans every N steps
  - Uses separate planning prompts to generate/update plans based on task and progress
- **Step Limit**: Configurable `max_steps` to prevent infinite loops

**Key Methods:**
- `run(prompt: str)`: Main execution loop returning dict with `final_answer`, `steps`, `memory`, `completed`, and `plan`
- `execute_tool_from_response_call()`: Handles tool calls from LLM response, executes tools, and manages memory
- `_generate_initial_plan()` / `_generate_updated_plan()`: Creates/updates execution plans using facts survey methodology

#### 2. KawaiTool Base Class ([kawai/tools/tool.py](kawai/tools/tool.py))
Abstract base for all tools with:
- **JSON Schema Generation**: `to_json_schema()` converts tool definition to OpenAI-compatible function calling format
- **Parameter System**: Uses `KawaiToolParameter` for type-safe parameter definitions
- **Type Support**: string, number, boolean, object, array, any (plus nullable variants)
- **Weave Integration**: `@weave.op` decorator on `forward()` for automatic experiment tracking

**To Create Custom Tools:**
1. Subclass `KawaiTool`
2. Define `tool_name`, `description`, and `parameters` as class attributes
3. Implement `forward(**kwargs)` method with your tool logic
4. Return results as dict, string, or any JSON-serializable type

#### 3. Built-in Tools ([kawai/tools/](kawai/tools/))

**FinalAnswerTool** ([answer.py](kawai/tools/answer.py:8-21)):
- Automatically added to every agent if not present
- Required to complete tasks - agent loops until this is called
- Accepts `answer` parameter of type "any" for maximum flexibility
- Returns answer unchanged (passthrough)

**UserInputTool** ([answer.py](kawai/tools/answer.py:24-38)):
- Prompts user for input via command line
- Takes `question` parameter
- Useful for interactive agents requiring clarification

**WebSearchTool** ([web_search.py](kawai/tools/web_search.py)):
- Uses Serper API for Google search results
- Parameters: `query` (required), `filter_year` (optional)
- Returns formatted search results with titles, links, dates, and snippets
- Requires `SERPER_API_KEY` environment variable

#### 4. Prompts System ([kawai/prompts.py](kawai/prompts.py))

**SYSTEM_PROMPT**:
- Derived from smolagents toolcalling agent prompt
- Teaches agent ReAct pattern: Action → Observation → Action cycle
- Includes examples and rules for proper tool usage
- Emphasizes that `final_answer` is the ONLY way to complete tasks

**Planning Prompts**:
- `INITIAL_PLAN_PROMPT`: Creates structured facts survey (given/lookup/derive) and step-by-step plan
- `UPDATE_PLAN_PRE_MESSAGES` + `UPDATE_PLAN_POST_MESSAGES`: Updates plans based on execution history and remaining steps

### Data Flow

1. User provides task → Agent initializes memory with system prompt + task
2. (Optional) Generate initial plan if `planning_interval` is set
3. Loop up to `max_steps`:
   - Call LLM via OpenRouter with current memory + tool schemas
   - Parse response for function calls
   - Execute tools via `forward()` method
   - Append function calls + outputs to memory
   - If `final_answer` tool called, extract answer and break
   - (Optional) Re-plan at specified intervals
4. Return dict with answer, steps taken, full memory, completion status, and plan

### Memory Format

The agent uses OpenAI-compatible message format:
- Standard messages: `{"role": "system|user|assistant", "content": "..."}`
- Function calls: `{"type": "function_call", "call_id": "...", "name": "...", "arguments": "{...}"}`
- Function outputs: `{"type": "function_call_output", "call_id": "...", "output": "..."}`

This format works with OpenRouter's responses API and maintains compatibility with OpenAI's function calling.

## Configuration

### Environment Variables
Required variables (store in `.env` file):
- `OPENROUTER_API_KEY`: API key for OpenRouter (agent's LLM provider)
- `SERPER_API_KEY`: API key for Serper (web search functionality)
- `WEAVE_PROJECT`: Weave project name for experiment tracking (set via `weave.init()`)

### Agent Configuration
```python
agent = KawaiReactAgent(
    model="openai/gpt-4",  # Any OpenRouter-supported model
    tools=[WebSearchTool()],  # List of KawaiTool instances
    system_prompt=SYSTEM_PROMPT,  # Custom system prompt (optional)
    max_steps=5,  # Maximum execution steps
    planning_interval=None,  # None or int for re-planning frequency
)
```

## Code Style and Conventions

### Linting
- Uses Ruff for linting and formatting (configured in [pyproject.toml](pyproject.toml:56-71))
- Target version: Python 3.12
- Import sorting: isort-compatible with first-party package "kawai"
- Selected rules: imports (I), unused imports (F401)

### Type Hints
- Required for public APIs
- Use modern Python type hints (e.g., `list[str]` instead of `List[str]`)
- Use `| None` for optional types instead of `Optional`

### Docstrings
- Use Google-style docstrings (configured for mkdocstrings)
- Document all public classes, methods, and functions

## Testing

Tests use pytest with asyncio support:
- Async tests run automatically with `asyncio_mode = "auto"` (set in pyproject.toml)
- Default fixture loop scope is "function"
- No existing test files found yet - create in `tests/` directory

## CI/CD

GitHub Actions workflow ([.github/workflows/check-lint.yml](.github/workflows/check-lint.yml)):
- Runs on PRs to `main`, `research/*`, `feat/*` branches
- Runs on pushes to `main`
- Checks: Ruff linting and formatting
- No tests are currently run in CI

## Important Notes

### OpenRouter Integration
- Agent uses OpenRouter as LLM provider (not direct OpenAI API)
- Base URL: `https://openrouter.ai/api/v1`
- Uses OpenRouter's `responses.create()` API for tool calling
- Must specify full model name (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet")

### Weave Tracking
- All tool executions are tracked via `@weave.op` decorator
- Agent's `run()` method is tracked as a Weave operation
- Initialize Weave with `weave.init(project_name="...")` before running agents
- Check traces at wandb.ai/weave

### Planning System Design
- Planning is separate from execution - uses dedicated LLM calls
- Initial plan: Surveys facts (given, to lookup, to derive) then creates high-level steps
- Plan updates: Reviews progress, updates fact survey, revises remaining steps
- Plans don't include detailed tool calls, only high-level strategy
- Planning adds overhead but improves complex task performance

### Tool Calling Pattern
- FinalAnswerTool is MANDATORY for task completion
- Agent loops until FinalAnswerTool is called or max_steps reached
- Tool parameters must match JSON schema types exactly
- Tool execution errors are caught and returned as `{"error": "..."}` in memory
