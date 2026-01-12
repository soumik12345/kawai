"""Pydantic models for REST API request and response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for the /chat endpoint.

    Attributes:
        message: The user's message/prompt for the agent.
        session_id: Optional session ID for conversation continuity.
            If not provided, a new session will be created.
        max_steps: Optional override for the agent's max_steps parameter.
        force_provide_answer: Whether to force an answer if max_steps is exhausted.
    """

    message: str = Field(..., description="The user's message/prompt for the agent")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation continuity",
    )
    max_steps: int | None = Field(
        default=None, description="Optional override for max_steps"
    )
    force_provide_answer: bool = Field(
        default=True,
        description="Whether to force an answer if max_steps is exhausted",
    )


class ChatResponse(BaseModel):
    """Response model for the /chat endpoint.

    Attributes:
        answer: The agent's final answer.
        session_id: The session ID for this conversation.
        steps: Number of steps taken to reach the answer.
        completed: Whether the task completed successfully.
        plan: The execution plan if planning was enabled, or None.
    """

    answer: Any = Field(..., description="The agent's final answer")
    session_id: str = Field(..., description="The session ID for this conversation")
    steps: int = Field(..., description="Number of steps taken")
    completed: bool = Field(..., description="Whether the task completed successfully")
    plan: str | None = Field(
        default=None, description="The execution plan if planning was enabled"
    )


class StreamEvent(BaseModel):
    """Model for Server-Sent Events during streaming.

    Attributes:
        type: The type of event (reasoning, tool_call, tool_result, step_start,
            run_start, run_end, warning, planning).
        data: The event data, structure depends on event type.
        timestamp: ISO-8601 timestamp when the event occurred.
    """

    type: str = Field(
        ...,
        description="Event type: reasoning, tool_call, tool_result, step_start, run_start, run_end, warning, planning",
    )
    data: dict[str, Any] = Field(..., description="Event data")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-8601 timestamp",
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint.

    Attributes:
        status: The health status (healthy, degraded, unhealthy).
        sessions: Number of active sessions.
        uptime: Server uptime in seconds.
        version: The kawai package version.
    """

    status: str = Field(..., description="Health status")
    sessions: int = Field(..., description="Number of active sessions")
    uptime: float = Field(..., description="Server uptime in seconds")
    version: str | None = Field(default=None, description="Package version")


class SessionInfo(BaseModel):
    """Response model for session information.

    Attributes:
        session_id: The unique session identifier.
        created_at: ISO-8601 timestamp when the session was created.
        last_accessed: ISO-8601 timestamp when the session was last used.
        message_count: Number of messages in the session memory.
    """

    session_id: str = Field(..., description="The unique session identifier")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")
    last_accessed: str = Field(..., description="ISO-8601 last access timestamp")
    message_count: int = Field(..., description="Number of messages in memory")


class SessionDeleteResponse(BaseModel):
    """Response model for session deletion.

    Attributes:
        session_id: The deleted session ID.
        deleted: Whether the deletion was successful.
    """

    session_id: str = Field(..., description="The deleted session ID")
    deleted: bool = Field(..., description="Whether deletion was successful")


class ErrorResponse(BaseModel):
    """Response model for error responses.

    Attributes:
        error: The error message.
        detail: Additional error details.
    """

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
