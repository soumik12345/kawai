from typing import Any

from pydantic import BaseModel


class BaseMemory(BaseModel):
    messages: list[dict[str, Any]] = []

    def add(self, content: str, role: str, **kwargs: Any) -> None:
        self.messages.append({"role": role, "content": content, **kwargs})

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages

    def clear(self) -> None:
        self.messages = []

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self.messages
