# pip install sentence-transformers>=3.3.1

import os

import rich
from dotenv import load_dotenv

from kawai import (
    KawaiLoggingCallback,
    KawaiReactAgent,
    Mem0Memory,
    OpenAIModel,
    WebSearchTool,
)

load_dotenv()
# weave.init(project_name="kawai")

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

# Multi-turn conversation demonstrating context retention
rich.print("=" * 80)
rich.print("Turn 1: Initial query about moth genus")
rich.print("=" * 80)
result1 = agent.run(
    prompt="Which genus of moth in the world's seventh-largest country contains only one species?"
)

rich.print("=" * 80)
rich.print("Turn 2: Follow-up question referencing previous context")
rich.print("=" * 80)
result2 = agent.run(prompt="What is the habitat of that moth species?")

rich.print("=" * 80)
rich.print("Turn 3: Another follow-up question")
rich.print("=" * 80)
result3 = agent.run(prompt="Is it endangered or threatened?")

rich.print("=" * 80)
rich.print("Turn 4: New topic to test context switching")
rich.print("=" * 80)
result4 = agent.run(
    prompt="What are the latest developments in quantum computing in 2026?"
)

rich.print("=" * 80)
rich.print("Turn 5: Follow-up on new topic")
rich.print("=" * 80)
result5 = agent.run(prompt="Which companies are leading in this field?")
