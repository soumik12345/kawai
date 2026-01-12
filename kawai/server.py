"""OpenAI-compatible API server for KawaiReactAgent.

This module provides a FastAPI-based server that exposes KawaiReactAgent
through an OpenAI-compatible REST API with streaming support and stateful
conversation management.
"""

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from kawai.callback import KawaiCallback
from kawai.memory.base import BaseMemory
from kawai.models.openai import OpenAIModel

# =============================================================================
# Pydantic Models for OpenAI API Compatibility
# =============================================================================


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """Request format for chat completions matching OpenAI's spec."""

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    user: str | None = None


class FunctionCall(BaseModel):
    """Function call information in a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call made by the assistant."""

    id: str
    type: str = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    """Message in a chat completion response."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChoice(BaseModel):
    """A single choice in the completion response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response format for non-streaming chat completions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Response format for streaming chat completion chunks."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kawai"


class ModelListResponse(BaseModel):
    """Response format for listing models."""

    object: str = "list"
    data: list[ModelInfo]


class ErrorDetail(BaseModel):
    """Error detail in OpenAI format."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Error response in OpenAI format."""

    error: ErrorDetail


# =============================================================================
# Streaming Callback for Capturing Agent Events
# =============================================================================


class StreamingCallback(KawaiCallback):
    """Callback that captures agent events and puts them in a queue for streaming.

    This callback converts agent execution events into OpenAI-compatible streaming
    chunks that can be sent via Server-Sent Events.
    """

    def __init__(self, queue: asyncio.Queue, chunk_id: str, model: str) -> None:
        super().__init__()
        self._queue = queue
        self._chunk_id = chunk_id
        self._model = model
        self._created = int(time.time())

    def _create_chunk(
        self,
        content: str | None = None,
        role: str | None = None,
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ChatCompletionChunk:
        """Create a streaming chunk in OpenAI format."""
        delta = DeltaMessage(role=role, content=content, tool_calls=tool_calls)
        choice = ChatCompletionChunkChoice(
            index=0, delta=delta, finish_reason=finish_reason
        )
        return ChatCompletionChunk(
            id=self._chunk_id,
            created=self._created,
            model=self._model,
            choices=[choice],
        )

    def at_run_start(self, prompt: str, model: str):
        """Send initial chunk with role."""
        chunk = self._create_chunk(role="assistant", content="")
        self._queue.put_nowait(chunk)

    def at_reasoning(self, reasoning: str):
        """Stream reasoning as content."""
        chunk = self._create_chunk(content=f"\n\n**Reasoning:**\n{reasoning}\n")
        self._queue.put_nowait(chunk)

    def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]):
        """Stream tool call information."""
        tool_call_chunk = {
            "index": 0,
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_arguments),
            },
        }
        # Send tool call as content for visibility
        content = f"\n\n**Tool Call:** `{tool_name}`\n```json\n{json.dumps(tool_arguments, indent=2)}\n```\n"
        chunk = self._create_chunk(content=content)
        self._queue.put_nowait(chunk)

    def at_tool_result(self, tool_name: str, tool_result: str):
        """Stream tool result."""
        # Truncate very long results
        display_result = tool_result
        if len(display_result) > 2000:
            display_result = display_result[:2000] + "\n... [truncated]"

        content = f"\n\n**Tool Result ({tool_name}):**\n```\n{display_result}\n```\n"
        chunk = self._create_chunk(content=content)
        self._queue.put_nowait(chunk)

    def at_step_start(self, step_index: int):
        """Stream step separator."""
        chunk = self._create_chunk(content=f"\n\n---\n**Step {step_index + 1}**\n")
        self._queue.put_nowait(chunk)

    def at_planning_end(self, plan: str, updated_plan: bool):
        """Stream plan information."""
        title = "Updated Plan" if updated_plan else "Initial Plan"
        chunk = self._create_chunk(content=f"\n\n**{title}:**\n{plan}\n")
        self._queue.put_nowait(chunk)

    def at_warning(self, message: str):
        """Stream warning message."""
        chunk = self._create_chunk(content=f"\n\n⚠️ **Warning:** {message}\n")
        self._queue.put_nowait(chunk)

    def at_run_end(self, answer: str):
        """Send final answer and finish."""
        if answer:
            chunk = self._create_chunk(content=f"\n\n**Final Answer:**\n{answer}")
            self._queue.put_nowait(chunk)


# =============================================================================
# Conversation Manager for Stateful Sessions
# =============================================================================


class ConversationManager:
    """Manages stateful conversation sessions.

    Each session maintains its own agent instance with isolated memory,
    allowing multi-turn conversations with persistent context.
    """

    def __init__(self, agent_template: "KawaiReactAgent") -> None:
        """Initialize the conversation manager.

        Args:
            agent_template: Template agent used to create new conversation instances.
        """
        self._agent_template = agent_template
        self._conversations: dict[str, "KawaiReactAgent"] = {}

    def _create_agent_copy(self) -> "KawaiReactAgent":
        """Create a new agent instance with fresh memory."""
        from kawai.agents.react import KawaiReactAgent

        # Create new memory instance
        new_memory = BaseMemory()

        # Create new model with fresh memory
        new_model = OpenAIModel(
            model_id=self._agent_template.model.model_id,
            system_prompt=self._agent_template.model.system_prompt,
            base_url=self._agent_template.model.base_url,
            api_key_env_var=self._agent_template.model.api_key_env_var,
            max_tokens=self._agent_template.model.max_tokens,
            memory=new_memory,
        )

        # Create new agent with fresh model
        new_agent = KawaiReactAgent(
            model=new_model,
            tools=self._agent_template.tools.copy(),
            system_prompt=self._agent_template.system_prompt,
            instructions=self._agent_template.instructions,
            max_steps=self._agent_template.max_steps,
            planning_interval=self._agent_template.planning_interval,
            callbacks=[],  # Callbacks will be added per-request for streaming
        )

        return new_agent

    def create_conversation(self, session_id: str) -> "KawaiReactAgent":
        """Create a new conversation session.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            A new agent instance for this session.
        """
        agent = self._create_agent_copy()
        self._conversations[session_id] = agent
        return agent

    def get_conversation(self, session_id: str) -> "KawaiReactAgent | None":
        """Retrieve an existing conversation.

        Args:
            session_id: The session identifier to look up.

        Returns:
            The agent for this session, or None if not found.
        """
        return self._conversations.get(session_id)

    def get_or_create_conversation(self, session_id: str) -> "KawaiReactAgent":
        """Get existing conversation or create a new one.

        Args:
            session_id: The session identifier.

        Returns:
            The agent for this session.
        """
        if session_id not in self._conversations:
            return self.create_conversation(session_id)
        return self._conversations[session_id]

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: The session identifier to delete.

        Returns:
            True if the session was deleted, False if it didn't exist.
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False

    def list_conversations(self) -> list[str]:
        """List all active session IDs.

        Returns:
            List of active session identifiers.
        """
        return list(self._conversations.keys())


# =============================================================================
# Format Conversion Helpers
# =============================================================================


def extract_last_user_message(messages: list[ChatMessage]) -> str | None:
    """Extract the last user message from a list of messages.

    Args:
        messages: List of chat messages.

    Returns:
        The content of the last user message, or None if not found.
    """
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return None


def convert_to_openai_response(
    result: dict[str, Any], model: str, request_id: str | None = None
) -> ChatCompletionResponse:
    """Convert agent result to OpenAI ChatCompletion format.

    Args:
        result: The result dictionary from agent.run().
        model: The model identifier.
        request_id: Optional request ID for the response.

    Returns:
        ChatCompletionResponse in OpenAI format.
    """
    final_answer = result.get("final_answer")

    # Convert final answer to string
    if final_answer is None:
        content = "I was unable to complete the task within the allowed steps."
    elif isinstance(final_answer, str):
        content = final_answer
    else:
        content = json.dumps(final_answer)

    message = ChatCompletionMessage(role="assistant", content=content)

    choice = ChatCompletionChoice(
        index=0,
        message=message,
        finish_reason="stop" if result.get("completed") else "length",
    )

    response_id = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatCompletionResponse(
        id=response_id,
        model=model,
        choices=[choice],
        usage=UsageInfo(
            prompt_tokens=0,  # Not tracked currently
            completion_tokens=0,
            total_tokens=0,
        ),
    )


# =============================================================================
# FastAPI Application Factory
# =============================================================================


def create_app(agent: "KawaiReactAgent") -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        agent: The KawaiReactAgent instance to expose via the API.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Kawai Agent API",
        description="OpenAI-compatible API for KawaiReactAgent",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize conversation manager
    conversation_manager = ConversationManager(agent)

    # Store model info
    model_id = agent.model.model_id

    # -------------------------------------------------------------------------
    # Health Check Endpoint
    # -------------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "model": model_id}

    # -------------------------------------------------------------------------
    # Models Endpoint
    # -------------------------------------------------------------------------

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        """List available models."""
        models = [
            ModelInfo(id="kawai-agent", owned_by="kawai"),
            ModelInfo(id=model_id, owned_by="kawai"),
        ]
        return ModelListResponse(data=models)

    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str):
        """Get information about a specific model."""
        if model_name in ["kawai-agent", model_id]:
            return ModelInfo(id=model_name, owned_by="kawai")
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Model '{model_name}' not found",
                    type="invalid_request_error",
                    code="model_not_found",
                )
            ).model_dump(),
        )

    # -------------------------------------------------------------------------
    # Chat Completions Endpoint
    # -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest,
        x_session_id: str | None = Header(None, alias="X-Session-ID"),
    ):
        """Create a chat completion.

        Supports both streaming and non-streaming responses.
        Use X-Session-ID header for stateful conversations.
        """
        # Get or create session
        session_id = x_session_id or str(uuid.uuid4())
        session_agent = conversation_manager.get_or_create_conversation(session_id)

        # Extract the user's message
        user_message = extract_last_user_message(request.messages)
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message="No user message found in request",
                        type="invalid_request_error",
                        code="missing_user_message",
                    )
                ).model_dump(),
            )

        if request.stream:
            return await _handle_streaming_request(
                session_agent, user_message, model_id, session_id
            )
        else:
            return await _handle_non_streaming_request(
                session_agent, user_message, model_id, session_id
            )

    async def _handle_non_streaming_request(
        agent: "KawaiReactAgent",
        user_message: str,
        model: str,
        session_id: str,
    ) -> JSONResponse:
        """Handle a non-streaming chat completion request."""
        try:
            # Run agent synchronously (in thread pool for async compatibility)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: agent.run(user_message, force_provide_answer=True)
            )

            response = convert_to_openai_response(result, model)

            # Add session ID to response headers
            return JSONResponse(
                content=response.model_dump(),
                headers={"X-Session-ID": session_id},
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=str(e),
                        type="internal_error",
                        code="agent_error",
                    )
                ).model_dump(),
            )

    async def _handle_streaming_request(
        agent: "KawaiReactAgent",
        user_message: str,
        model: str,
        session_id: str,
    ) -> EventSourceResponse:
        """Handle a streaming chat completion request."""
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        async def event_generator():
            queue: asyncio.Queue = asyncio.Queue()
            streaming_callback = StreamingCallback(queue, chunk_id, model)

            # Add streaming callback to agent
            original_callbacks = agent.callbacks.copy()
            agent.callbacks = original_callbacks + [streaming_callback]

            # Run agent in background
            async def run_agent():
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: agent.run(user_message, force_provide_answer=True)
                    )
                except Exception as e:
                    # Send error as content
                    error_chunk = streaming_callback._create_chunk(
                        content=f"\n\n❌ **Error:** {str(e)}"
                    )
                    await queue.put(error_chunk)
                finally:
                    # Signal completion
                    await queue.put(None)
                    # Restore original callbacks
                    agent.callbacks = original_callbacks

            # Start agent task
            agent_task = asyncio.create_task(run_agent())

            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        # Send finish chunk
                        finish_chunk = streaming_callback._create_chunk(
                            finish_reason="stop"
                        )
                        yield {
                            "event": "message",
                            "data": finish_chunk.model_dump_json(),
                        }
                        yield {"event": "message", "data": "[DONE]"}
                        break

                    yield {"event": "message", "data": chunk.model_dump_json()}
            finally:
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass

        return EventSourceResponse(
            event_generator(),
            headers={"X-Session-ID": session_id},
        )

    # -------------------------------------------------------------------------
    # Conversation Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/v1/conversations")
    async def list_conversations():
        """List all active conversation sessions."""
        sessions = conversation_manager.list_conversations()
        return {"sessions": sessions, "count": len(sessions)}

    @app.delete("/v1/conversations/{session_id}")
    async def delete_conversation(session_id: str):
        """Delete a specific conversation session."""
        deleted = conversation_manager.delete_conversation(session_id)
        if deleted:
            return {"status": "deleted", "session_id": session_id}
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=f"Session '{session_id}' not found",
                    type="invalid_request_error",
                    code="session_not_found",
                )
            ).model_dump(),
        )

    # -------------------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions in OpenAI-compatible format."""
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(exc.detail),
                    type="api_error",
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=f"Internal server error: {str(exc)}",
                    type="internal_error",
                )
            ).model_dump(),
        )

    return app
