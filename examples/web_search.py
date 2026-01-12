# pip install sentence-transformers>=3.3.1

import os

import weave
from dotenv import load_dotenv

from kawai import (
    KawaiLoggingCallback,
    KawaiReactAgent,
    Mem0Memory,
    OpenAIModel,
    WebSearchTool,
)

load_dotenv()
weave.init(project_name="kawai")

# Configure Mem0Memory with persistent storage
# Note: Requires installing memory extras: `uv pip install -e ".[memory]"`
memory = Mem0Memory(
    user_id="user_123",
    mem0_config={
        # Use LiteLLM to route to OpenRouter (Mem0 doesn't support "openrouter" provider directly)
        "llm": {
            "provider": "openai",
            "config": {
                "model": "google/gemini-3-flash-preview",
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "openai_base_url": "https://openrouter.ai/api/v1",
            },
        },
        # Use HuggingFace embeddings (local, no API key required)
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "Qwen/Qwen3-Embedding-0.6B"},
        },
        # Use Chroma for local vector storage
        "vector_store": {"provider": "chroma", "config": {"path": "./mem0_db"}},
    },
)

model = OpenAIModel(
    model_id="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
    api_key_env_var="OPENROUTER_API_KEY",
    memory=memory,
)

agent = KawaiReactAgent(
    model=model,
    tools=[WebSearchTool()],
    planning_interval=5,
    max_steps=10,
    instructions="Focus on finding sources from 2026.",
    callbacks=[KawaiLoggingCallback()],
)
result = agent.run(
    prompt="Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?"
)
