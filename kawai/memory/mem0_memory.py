from typing import Any

from mem0 import Memory

from kawai.memory.base import BaseMemory


class Mem0Memory(BaseMemory):
    """Memory implementation with persistent storage and semantic search via Mem0.

    Extends BaseMemory with long-term conversation storage using the Mem0 library.
    Conversations are automatically stored in a vector database for semantic search
    across conversation history. Supports multiple storage backends (Chroma, Qdrant,
    etc.) and custom LLM/embedder configurations.

    Attributes:
        user_id (str): Unique identifier for the user. Used to namespace memories per user.
        mem0_config (dict[str, Any] | None): Configuration dictionary for Mem0. Should include 'llm',
            'embedder', and 'vector_store' sections. If None, uses Mem0 defaults.
        model_config (dict[str, Any]): Pydantic configuration allowing arbitrary types.
        _mem0 (Memory | None): Internal Mem0 Memory instance initialized after construction.

    !!! example
        ```python
        config = {
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": "openrouter/google/gemini-3-flash-preview",
                    "api_key": "your-api-key",
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "vector_store": {
                "provider": "chroma",
                "config": {"path": "./mem0_db"}
            },
        }
        memory = Mem0Memory(user_id="user_123", mem0_config=config)
        memory.add("I like Python", role="user")
        context = memory.get_relevant_context("programming languages")
        ```

    Note:
        Requires installing memory extras: `uv pip install -e ".[memory]"`
    """

    user_id: str
    mem0_config: dict[str, Any] | None = None
    model_config: dict[str, Any] = {"arbitrary_types_allowed": True}
    _mem0: Memory | None = None

    def model_post_init(self, __context: Any, /) -> None:
        self._mem0 = (
            Memory.from_config(self.mem0_config) if self.mem0_config else Memory()
        )

    def add(self, content: str, role: str, **kwargs: Any) -> None:
        """Add a message to both in-memory history and Mem0 persistent storage.

        Args:
            content (str): The message content text.
            role (str): The message role (e.g., "system", "user", "assistant", "tool").
            **kwargs (Any): Additional message fields (e.g., tool_calls, tool_call_id).
        """
        super().add(content=content, role=role, **kwargs)
        self._mem0.add(
            messages=[{"role": role, "content": content}], user_id=self.user_id
        )

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search conversation history using semantic similarity.

        Args:
            query (str): Search query string for semantic matching.
            **kwargs (Any): Additional Mem0 search parameters (e.g., limit, filters).

        Returns:
            List of memory dictionaries from Mem0 with similarity scores and metadata.
        """
        return self._mem0.search(query=query, user_id=self.user_id, **kwargs)

    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """Retrieve semantically relevant past conversations as formatted text.

        Args:
            query (str): Query string to find relevant memories.
            limit (int): Maximum number of memories to retrieve. Defaults to 5.

        Returns:
            Formatted string with bullet points of relevant memories, or empty
            string if no memories found.

        !!! example
            ```python
            memory.add("I prefer TypeScript for web dev", role="user")
            context = memory.get_relevant_context("web development", limit=3)
            print(context)
            ```
        """
        results = self.search(query, limit=limit)
        if not results:
            return ""

        context_parts = []
        for mem in results:
            if isinstance(mem, dict) and "memory" in mem:
                context_parts.append(f"- {mem['memory']}")

        return "\n".join(context_parts)
