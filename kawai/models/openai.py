import os
from typing import Any

import weave
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


class OpenAIModel(BaseModel):
    """LLM interface that wraps OpenAI-compatible chat completion APIs.

    This class provides a unified interface for interacting with OpenAI-compatible
    APIs, including OpenAI, OpenRouter, and other providers that follow the OpenAI
    Chat Completions API format. It handles conversation memory management, tool
    calling, and integrates with Weave for experiment tracking.

    The model automatically initializes an OpenAI client on instantiation and
    maintains conversation history in the standard OpenAI message format.

    Attributes:
        model_id (str): Model identifier to use for completions. Format depends on
            the provider:
            - OpenAI: "gpt-4", "gpt-3.5-turbo", etc.
            - OpenRouter: "openai/gpt-4", "anthropic/claude-3-sonnet",
              "google/gemini-3-flash-preview", etc.
        system_prompt (str | None): Optional system prompt. Note: This attribute
            exists for backward compatibility but is not currently used. System
            prompts should be added to memory directly. Defaults to None.
        base_url (str | None): Base URL for the API endpoint. If None, uses
            OpenAI's default endpoint. Common values:
            - OpenRouter: "https://openrouter.ai/api/v1"
            - OpenAI: None or "https://api.openai.com/v1"
            Defaults to None.
        api_key_env_var (str): Name of the environment variable containing the
            API key. Defaults to "OPENAI_API_KEY".
        max_tokens (int | None): Maximum number of tokens to generate in the
            completion. If None, uses the model's default limit. Setting this
            can help ensure complete reasoning responses. Defaults to None.
        memory (list[dict[str, Any]]): Conversation history in OpenAI Chat
            Completions format. Each message is a dict with "role" and "content"
            keys, plus optional "tool_calls" or "tool_call_id" for function calling.
            Defaults to empty list.

    !!! example
        ```python
        from kawai import OpenAIModel

        # Using OpenRouter
        model = OpenAIModel(
            model_id="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER_API_KEY",
            max_tokens=4096  # Ensure complete responses
        )

        # Using OpenAI directly
        model = OpenAIModel(
            model_id="gpt-4",
            api_key_env_var="OPENAI_API_KEY"
        )

        # Add messages to memory
        model.update_memory(content="You are a helpful assistant.", role="system")
        model.update_memory(content="What is 2+2?", role="user")

        # Generate completion from memory
        response = model.predict_from_memory()
        print(response.choices[0].message.content)
        ```

    Note:
        - All prediction methods are tracked by Weave via `@weave.op` decorator
        - The API key must be set in the specified environment variable
        - Memory persists across multiple predictions until manually cleared
    """

    model_id: str
    system_prompt: str | None = None
    base_url: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    max_tokens: int | None = None
    memory: list[dict[str, Any]] = []
    _client: OpenAI | None = None

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv(self.api_key_env_var),
        )

    def update_memory(self, content: str, role: str, **kwargs: Any) -> None:
        """Append a message to the conversation memory.

        Args:
            content (str): The message content.
            role (str): The role of the message sender. Common values:
                - "system": System instructions
                - "user": User messages
                - "assistant": Assistant responses
                - "tool": Tool execution results
            **kwargs (Any): Additional fields to include in the message dict, such as:
                - tool_calls: List of tool call objects for assistant messages
                - tool_call_id: ID linking tool responses to their calls

        !!! example
            ```python
            # Add system message
            model.update_memory(content="You are helpful.", role="system")

            # Add user message
            model.update_memory(content="Hello!", role="user")

            # Add assistant message with tool calls
            model.update_memory(
                content="Let me search for that.",
                role="assistant",
                tool_calls=[{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "AI"}'}
                }]
            )

            # Add tool result
            model.update_memory(
                content='{"results": [...]}',
                role="tool",
                tool_call_id="call_123"
            )
            ```
        """
        self.memory.append({"role": role, "content": content, **kwargs})

    @weave.op
    def predict_from_messages(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatCompletion:
        """Generate a chat completion from provided messages.

        This method calls the LLM with explicitly provided messages, independent
        of the internal memory. Useful for one-off predictions or when you need
        full control over the conversation context.

        Args:
            messages (list[dict[str, Any]]): List of message dicts in OpenAI format.
                Each message should have "role" and "content" keys.
            tools (list[dict[str, Any]] | None): Optional list of tool/function
                schemas in OpenAI function calling format. If provided, the model
                can call these tools in its response. Defaults to None.

        Returns:
            ChatCompletion: The model's response in OpenAI ChatCompletion format,
                containing choices with message content and optional tool calls.

        !!! example
            ```python
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]

            response = model.predict_from_messages(messages)
            print(response.choices[0].message.content)

            # With tools
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }]

            response = model.predict_from_messages(messages, tools=tools)
            ```

        Note:
            - This method does NOT update the internal memory
            - Tracked by Weave for observability
            - Tool calls in the response follow OpenAI's function calling format
        """
        completion_kwargs = {}
        if tools is not None:
            completion_kwargs["tools"] = tools
        if self.max_tokens is not None:
            completion_kwargs["max_tokens"] = self.max_tokens
        return self._client.chat.completions.create(
            model=self.model_id, messages=messages, **completion_kwargs
        )

    @weave.op
    def predict_from_memory(
        self, tools: list[dict[str, Any]] | None = None
    ) -> ChatCompletion:
        """Generate a chat completion using the internal conversation memory.

        This method calls the LLM with the current state of the internal memory,
        making it easy to maintain multi-turn conversations. This is the primary
        method used by KawaiReactAgent for generating responses.

        Args:
            tools (list[dict[str, Any]] | None): Optional list of tool/function
                schemas in OpenAI function calling format. If provided, the model
                can call these tools in its response. Defaults to None.

        Returns:
            ChatCompletion: The model's response in OpenAI ChatCompletion format,
                containing choices with message content and optional tool calls.

        !!! example
            ```python
            # Build conversation in memory
            model.update_memory(content="You are helpful.", role="system")
            model.update_memory(content="Hello!", role="user")

            # Generate response from memory
            response = model.predict_from_memory()
            assistant_message = response.choices[0].message.content

            # Add response to memory for next turn
            model.update_memory(content=assistant_message, role="assistant")

            # With tools enabled
            tools = [tool.to_json_schema() for tool in agent_tools]
            response = model.predict_from_memory(tools=tools)
            ```

        Note:
            - Uses the current state of self.memory
            - Does NOT automatically update memory with the response
            - Tracked by Weave for observability
            - This is how KawaiReactAgent generates reasoning and tool calls
        """
        completion_kwargs = {}
        if tools is not None:
            completion_kwargs["tools"] = tools
        if self.max_tokens is not None:
            completion_kwargs["max_tokens"] = self.max_tokens
        return self._client.chat.completions.create(
            model=self.model_id, messages=self.memory, **completion_kwargs
        )
