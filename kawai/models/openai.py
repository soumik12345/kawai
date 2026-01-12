import os
from typing import Any

import weave
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel


class OpenAIModel(BaseModel):
    model_id: str
    system_prompt: str | None = None
    base_url: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    memory: list[dict[str, Any]] = []
    _client: OpenAI | None = None

    def model_post_init(self, __context: Any) -> None:
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv(self.api_key_env_var),
        )

    def update_memory(self, content: str, role: str, **kwargs):
        self.memory.append({"role": role, "content": content, **kwargs})

    @weave.op
    def predict_from_messages(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatCompletion:
        completion_kwargs = {}
        if tools is not None:
            completion_kwargs["tools"] = tools
        return self._client.chat.completions.create(
            model=self.model_id, messages=messages, **completion_kwargs
        )

    @weave.op
    def predict_from_memory(
        self, tools: list[dict[str, Any]] | None = None
    ) -> ChatCompletion:
        completion_kwargs = {}
        if tools is not None:
            completion_kwargs["tools"] = tools
        return self._client.chat.completions.create(
            model=self.model_id, messages=self.memory, **completion_kwargs
        )
