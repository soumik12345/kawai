from typing import Any

from pydantic import BaseModel


class BaseMemory(BaseModel):
    """Base class for conversation memory storage.

    Provides simple in-memory conversation history storage using OpenAI-compatible
    message format. This class can be subclassed to implement custom memory backends
    with persistent storage, semantic search, or other advanced features.

    Attributes:
        messages (list[dict[str, Any]]): List of conversation messages in OpenAI format.
            Each message is a dictionary with at least 'role' and 'content' keys. Additional keys
            like 'tool_calls' and 'tool_call_id' may be present for function calling.

    !!! example
        ```python
        memory = BaseMemory()
        memory.add("Hello!", role="user")
        memory.add("Hi there!", role="assistant")
        messages = memory.get_messages()
        len(messages)
        ```
    """

    messages: list[dict[str, Any]] = []

    def add(self, content: str, role: str, **kwargs: Any) -> None:
        """Add a message to conversation history.

        Args:
            content (str): The message content text.
            role (str): The message role (e.g., "system", "user", "assistant", "tool").
            **kwargs (Any): Additional message fields (e.g., tool_calls, tool_call_id).
        """
        self.messages.append({"role": role, "content": content, **kwargs})

    def get_messages(self) -> list[dict[str, Any]]:
        """Retrieve all conversation messages.

        Returns:
            List of message dictionaries in OpenAI format.
        """
        return self.messages

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search conversation history.

        Base implementation returns all messages. Subclasses can override to
        implement semantic search or other filtering strategies.

        Args:
            query (str): Search query string.
            **kwargs (Any): Additional search parameters (implementation-specific).

        Returns:
            List of message dictionaries matching the search criteria.
        """
        return self.messages
