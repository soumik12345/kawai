from typing import Any

from mem0 import Memory

from kawai.memory.base import BaseMemory


class Mem0Memory(BaseMemory):
    user_id: str
    mem0_config: dict[str, Any] | None = None
    model_config = {"arbitrary_types_allowed": True}
    _mem0: Memory | None = None

    def model_post_init(self, __context: Any, /) -> None:
        self._mem0 = (
            Memory.from_config(self.mem0_config) if self.mem0_config else Memory()
        )

    def add(self, content: str, role: str, **kwargs: Any) -> None:
        super().add(content=content, role=role, **kwargs)
        self._mem0.add(
            messages=[{"role": role, "content": content}], user_id=self.user_id
        )

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._mem0.search(query=query, user_id=self.user_id, **kwargs)

    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        results = self.search(query, limit=limit)
        if not results:
            return ""

        context_parts = []
        for mem in results:
            if isinstance(mem, dict) and "memory" in mem:
                context_parts.append(f"- {mem['memory']}")

        return "\n".join(context_parts)
