"""Session management for the REST API server."""

import asyncio
import copy
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from kawai.agents.react import KawaiReactAgent


class SessionData(BaseModel):
    """Data stored for each session.

    Attributes:
        session_id: Unique identifier for the session.
        created_at: When the session was created.
        last_accessed: When the session was last used.
        agent_config: Configuration to recreate the agent.
        memory_messages: The conversation memory messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    agent_config: dict[str, Any] = Field(default_factory=dict)
    memory_messages: list[dict[str, Any]] = Field(default_factory=list)


class SessionManager:
    """Manages sessions for the REST API server.

    Handles creation, retrieval, and cleanup of agent sessions. Each session
    maintains its own conversation memory for stateful interactions.

    Attributes:
        template_agent: The base agent configuration to clone for new sessions.
        sessions: Dictionary mapping session IDs to SessionData.
        session_timeout: Session expiration time in seconds.
        _lock: Asyncio lock for thread-safe session operations.
        _cleanup_task: Background task for session cleanup.
    """

    def __init__(
        self,
        template_agent: "KawaiReactAgent",
        session_timeout: int = 3600,
    ) -> None:
        """Initialize the session manager.

        Args:
            template_agent: The agent instance to use as template for new sessions.
            session_timeout: Session expiration time in seconds (default: 1 hour).
        """
        self.template_agent = template_agent
        self.sessions: dict[str, SessionData] = {}
        self.session_timeout = session_timeout
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _create_agent_config(self) -> dict[str, Any]:
        """Extract configuration from template agent for session creation."""
        return {
            "model_id": self.template_agent.model.model_id,
            "model_base_url": self.template_agent.model.base_url,
            "model_api_key_env_var": self.template_agent.model.api_key_env_var,
            "model_max_tokens": self.template_agent.model.max_tokens,
            "tools": self.template_agent.tools,
            "system_prompt": self.template_agent.system_prompt,
            "instructions": self.template_agent.instructions,
            "max_steps": self.template_agent.max_steps,
            "planning_interval": self.template_agent.planning_interval,
        }

    def create_agent_for_session(
        self,
        session_data: SessionData,
        callbacks: list | None = None,
    ) -> "KawaiReactAgent":
        """Create an agent instance for a session.

        Args:
            session_data: The session data containing configuration and memory.
            callbacks: Optional list of callbacks to add to the agent.

        Returns:
            A new KawaiReactAgent instance configured for the session.
        """
        # Import here to avoid circular imports
        from kawai.agents.react import KawaiReactAgent
        from kawai.memory.base import BaseMemory
        from kawai.models.openai import OpenAIModel

        config = session_data.agent_config

        # Create fresh memory with existing messages
        memory = BaseMemory(messages=copy.deepcopy(session_data.memory_messages))

        # Create model with fresh memory
        model = OpenAIModel(
            model_id=config["model_id"],
            base_url=config["model_base_url"],
            api_key_env_var=config["model_api_key_env_var"],
            max_tokens=config["model_max_tokens"],
            memory=memory,
        )

        # Create agent with deep-copied tools
        agent = KawaiReactAgent(
            model=model,
            tools=copy.deepcopy(config["tools"]),
            system_prompt=config["system_prompt"],
            instructions=config["instructions"],
            max_steps=config["max_steps"],
            planning_interval=config["planning_interval"],
            callbacks=callbacks or [],
        )

        return agent

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[str, SessionData]:
        """Get an existing session or create a new one.

        Args:
            session_id: Optional session ID. If None, creates a new session.

        Returns:
            Tuple of (session_id, SessionData).
        """
        async with self._lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                session.last_accessed = datetime.now()
                return session_id, session

            # Create new session
            new_session_id = session_id or str(uuid.uuid4())
            agent_config = self._create_agent_config()

            # Get initial memory from template agent's system prompt setup
            # We need to include the system prompt in the initial memory
            initial_messages: list[dict[str, Any]] = []

            session = SessionData(
                session_id=new_session_id,
                agent_config=agent_config,
                memory_messages=initial_messages,
            )
            self.sessions[new_session_id] = session

            return new_session_id, session

    async def update_session_memory(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Update the memory for a session.

        Args:
            session_id: The session ID to update.
            messages: The new memory messages.
        """
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].memory_messages = messages
                self.sessions[session_id].last_accessed = datetime.now()

    async def get_session(self, session_id: str) -> SessionData | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The SessionData if found, None otherwise.
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_accessed = datetime.now()
            return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if the session was deleted, False if it didn't exist.
        """
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    async def get_session_count(self) -> int:
        """Get the number of active sessions.

        Returns:
            The number of active sessions.
        """
        async with self._lock:
            return len(self.sessions)

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.

        Returns:
            The number of sessions removed.
        """
        async with self._lock:
            now = datetime.now()
            expired = [
                sid
                for sid, session in self.sessions.items()
                if (now - session.last_accessed).total_seconds() > self.session_timeout
            ]
            for sid in expired:
                del self.sessions[sid]
            return len(expired)

    async def start_cleanup_task(self, interval: int = 300) -> None:
        """Start the background session cleanup task.

        Args:
            interval: Cleanup interval in seconds (default: 5 minutes).
        """
        if self._cleanup_task is None or self._cleanup_task.done():

            async def cleanup_loop():
                while True:
                    await asyncio.sleep(interval)
                    await self.cleanup_expired_sessions()

            self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Stop the background session cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
