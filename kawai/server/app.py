"""FastAPI application for serving KawaiReactAgent as a REST API."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import anyio
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from kawai.server.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SessionDeleteResponse,
    SessionInfo,
)
from kawai.server.session import SessionManager
from kawai.server.streaming_callback import StreamingCallback, event_to_sse

if TYPE_CHECKING:
    from kawai.agents.react import KawaiReactAgent


def create_app(
    agent: "KawaiReactAgent",
    session_timeout: int = 3600,
    enable_cors: bool = True,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI application for serving the agent.

    Args:
        agent: The KawaiReactAgent instance to serve.
        session_timeout: Session expiration time in seconds (default: 1 hour).
        enable_cors: Whether to enable CORS middleware.
        allowed_origins: List of allowed origins for CORS. If None, allows all.

    Returns:
        A configured FastAPI application.
    """
    # Initialize session manager
    session_manager = SessionManager(
        template_agent=agent,
        session_timeout=session_timeout,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifespan: startup and shutdown."""
        # Startup
        await session_manager.start_cleanup_task()
        yield
        # Shutdown
        await session_manager.stop_cleanup_task()

    app = FastAPI(
        title="Kawai Agent API",
        description="REST API for interacting with KawaiReactAgent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Store startup time for health endpoint
    app.state.start_time = time.time()
    app.state.session_manager = session_manager

    # Configure CORS
    if enable_cors:
        origins = allowed_origins or ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint.

        Returns server health status, active session count, and uptime.
        """
        try:
            # Try to get kawai version
            try:
                from importlib.metadata import version

                kawai_version = version("kawai")
            except Exception:
                kawai_version = None

            session_count = await session_manager.get_session_count()
            uptime = time.time() - app.state.start_time

            return HealthResponse(
                status="healthy",
                sessions=session_count,
                uptime=uptime,
                version=kawai_version,
            )
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                sessions=0,
                uptime=time.time() - app.state.start_time,
                version=None,
            )

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def chat(request: ChatRequest) -> ChatResponse:
        """Non-streaming chat endpoint.

        Send a message to the agent and receive the final response.
        Sessions are maintained for conversation continuity.

        Args:
            request: The chat request containing the message and optional parameters.

        Returns:
            The agent's response including the answer, session ID, and metadata.
        """
        try:
            # Get or create session
            session_id, session_data = await session_manager.get_or_create_session(
                request.session_id
            )

            # Create agent for this session (no streaming callback)
            session_agent = session_manager.create_agent_for_session(
                session_data,
                callbacks=agent.callbacks.copy() if agent.callbacks else [],
            )

            # Override max_steps if provided
            if request.max_steps is not None:
                session_agent.max_steps = request.max_steps

            # Run agent in background thread to not block event loop
            result: dict[str, Any] = await anyio.to_thread.run_sync(
                lambda: session_agent.run(
                    request.message,
                    force_provide_answer=request.force_provide_answer,
                )
            )

            # Update session with new memory state
            await session_manager.update_session_memory(
                session_id,
                session_agent.model.memory.get_messages(),
            )

            return ChatResponse(
                answer=result.get("final_answer"),
                session_id=session_id,
                steps=result.get("steps", 0),
                completed=result.get("completed", False),
                plan=result.get("plan"),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stream")
    async def stream(
        message: str = Query(..., description="The message to send to the agent"),
        session_id: str | None = Query(default=None, description="Optional session ID"),
        max_steps: int | None = Query(
            default=None, description="Optional max steps override"
        ),
        force_provide_answer: bool = Query(
            default=True,
            description="Force answer if max_steps exhausted",
        ),
    ) -> EventSourceResponse:
        """Server-Sent Events streaming endpoint.

        Connect to receive real-time events during agent execution.
        Events include reasoning, tool calls, tool results, and the final answer.

        Args:
            message: The message to send to the agent.
            session_id: Optional session ID for conversation continuity.
            max_steps: Optional override for max_steps.
            force_provide_answer: Whether to force an answer if max_steps is exhausted.

        Returns:
            SSE stream of agent execution events.
        """

        async def event_generator():
            try:
                # Get or create session
                sid, session_data = await session_manager.get_or_create_session(
                    session_id
                )

                # Create streaming callback
                streaming_callback = StreamingCallback()
                streaming_callback.set_event_loop(asyncio.get_event_loop())

                # Get existing callbacks and add streaming callback
                callbacks = agent.callbacks.copy() if agent.callbacks else []
                callbacks.append(streaming_callback)

                # Create agent for this session with streaming callback
                session_agent = session_manager.create_agent_for_session(
                    session_data,
                    callbacks=callbacks,
                )

                # Override max_steps if provided
                if max_steps is not None:
                    session_agent.max_steps = max_steps

                # Start agent in background thread
                async def run_agent():
                    return await anyio.to_thread.run_sync(
                        lambda: session_agent.run(
                            message,
                            force_provide_answer=force_provide_answer,
                        )
                    )

                # Create task for agent execution
                agent_task = asyncio.create_task(run_agent())

                # Stream events as they come
                async for event in streaming_callback.stream_events():
                    yield event_to_sse(event)

                # Wait for agent task to complete
                result = await agent_task

                # Update session with new memory state
                await session_manager.update_session_memory(
                    sid,
                    session_agent.model.memory.get_messages(),
                )

            except Exception as e:
                # Send error event
                from kawai.server.models import StreamEvent

                error_event = StreamEvent(
                    type="error",
                    data={"error": str(e)},
                )
                yield event_to_sse(error_event)

        return EventSourceResponse(event_generator())

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionInfo,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_session(session_id: str) -> SessionInfo:
        """Get information about a session.

        Args:
            session_id: The session ID to look up.

        Returns:
            Session information including creation time and message count.
        """
        session = await session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionInfo(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            last_accessed=session.last_accessed.isoformat(),
            message_count=len(session.memory_messages),
        )

    @app.delete(
        "/sessions/{session_id}",
        response_model=SessionDeleteResponse,
        responses={404: {"model": ErrorResponse}},
    )
    async def delete_session(session_id: str) -> SessionDeleteResponse:
        """Delete a session.

        Args:
            session_id: The session ID to delete.

        Returns:
            Confirmation of deletion.
        """
        deleted = await session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionDeleteResponse(
            session_id=session_id,
            deleted=True,
        )

    return app
