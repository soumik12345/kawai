"""Server components for deploying KawaiReactAgent as a REST API."""

from kawai.server.app import create_app
from kawai.server.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SessionDeleteResponse,
    SessionInfo,
    StreamEvent,
)
from kawai.server.session import SessionData, SessionManager
from kawai.server.streaming_callback import StreamingCallback, event_to_sse

__all__ = [
    # App creation
    "create_app",
    # Models
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "SessionDeleteResponse",
    "SessionInfo",
    "StreamEvent",
    # Session management
    "SessionData",
    "SessionManager",
    # Streaming
    "StreamingCallback",
    "event_to_sse",
]
