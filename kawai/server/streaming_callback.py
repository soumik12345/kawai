"""Streaming callback for Server-Sent Events (SSE) support."""

import asyncio
import json
from datetime import datetime
from typing import Any

from kawai.callback import KawaiCallback
from kawai.server.models import StreamEvent


class StreamingCallback(KawaiCallback):
    """Callback that captures agent events for SSE streaming.

    This callback captures all agent execution events and pushes them to an
    asyncio queue for real-time streaming via Server-Sent Events. The queue
    can be consumed by an async generator to yield SSE events.

    The callback is thread-safe and can be used when the agent runs in a
    background thread while the FastAPI server consumes events asynchronously.

    Attributes:
        _queue: Asyncio queue for buffering events.
        _loop: The event loop to use for queue operations.

    !!! example
        ```python
        import asyncio
        from kawai.server.streaming_callback import StreamingCallback

        async def stream_agent():
            callback = StreamingCallback()
            callback.set_event_loop(asyncio.get_event_loop())

            # Run agent in background thread
            # ...

            async for event in callback.stream_events():
                print(f"{event.type}: {event.data}")
        ```
    """

    _queue: asyncio.Queue | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _final_result: dict[str, Any] | None = None

    def __init__(self) -> None:
        super().__init__()
        self._queue = None
        self._loop = None
        self._final_result = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop and initialize the queue.

        Must be called before the agent starts running.

        Args:
            loop: The asyncio event loop to use for queue operations.
        """
        self._loop = loop
        self._queue = asyncio.Queue()

    def _put_event(self, event: StreamEvent) -> None:
        """Put an event on the queue from sync context.

        Uses call_soon_threadsafe to safely add events from the sync
        agent execution thread to the async queue.

        Args:
            event: The event to put on the queue.
        """
        if self._queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def _create_event(
        self, event_type: str, data: dict[str, Any] | None = None
    ) -> StreamEvent:
        """Create a StreamEvent with current timestamp.

        Args:
            event_type: The type of event.
            data: The event data.

        Returns:
            A StreamEvent instance.
        """
        return StreamEvent(
            type=event_type,
            data=data or {},
            timestamp=datetime.now().isoformat(),
        )

    def at_run_start(self, prompt: str, model: str) -> None:
        """Called when the agent run begins."""
        event = self._create_event(
            "run_start",
            {"prompt": prompt, "model": model},
        )
        self._put_event(event)

    def at_run_end(self, answer: Any) -> None:
        """Called when the agent run completes."""
        # Store the final result for access after streaming
        self._final_result = {"answer": answer}

        event = self._create_event(
            "run_end",
            {"answer": answer},
        )
        self._put_event(event)

        # Signal end of stream with a sentinel
        if self._queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def at_step_start(self, step_index: int) -> None:
        """Called at the beginning of each ReAct step."""
        event = self._create_event(
            "step_start",
            {"step_index": step_index},
        )
        self._put_event(event)

    def at_planning_end(self, plan: str, updated_plan: bool) -> None:
        """Called after the agent generates or updates a plan."""
        event = self._create_event(
            "planning",
            {"plan": plan, "updated_plan": updated_plan},
        )
        self._put_event(event)

    def at_reasoning(self, reasoning: str) -> None:
        """Called when the agent produces reasoning/thinking content."""
        event = self._create_event(
            "reasoning",
            {"reasoning": reasoning},
        )
        self._put_event(event)

    def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]) -> None:
        """Called when the agent is about to execute a tool."""
        event = self._create_event(
            "tool_call",
            {"tool_name": tool_name, "tool_arguments": tool_arguments},
        )
        self._put_event(event)

    def at_tool_result(self, tool_name: str, tool_result: str) -> None:
        """Called after a tool has been executed with its result."""
        # Try to parse result as JSON for structured data
        try:
            parsed_result = json.loads(tool_result)
        except (json.JSONDecodeError, TypeError):
            parsed_result = tool_result

        event = self._create_event(
            "tool_result",
            {"tool_name": tool_name, "tool_result": parsed_result},
        )
        self._put_event(event)

    def at_warning(self, message: str) -> None:
        """Called when the agent encounters a warning condition."""
        event = self._create_event(
            "warning",
            {"message": message},
        )
        self._put_event(event)

    async def stream_events(self):
        """Async generator that yields events from the queue.

        Yields events as they are produced by the agent. Stops when
        a None sentinel is received (indicating run_end).

        Yields:
            StreamEvent instances as they occur during agent execution.

        Raises:
            RuntimeError: If the event loop was not set.
        """
        if self._queue is None:
            raise RuntimeError(
                "Event loop not set. Call set_event_loop() before streaming."
            )

        while True:
            event = await self._queue.get()
            if event is None:
                # Sentinel received, end of stream
                break
            yield event

    def get_final_result(self) -> dict[str, Any] | None:
        """Get the final result after the agent run completes.

        Returns:
            Dictionary with 'answer' key, or None if run hasn't completed.
        """
        return self._final_result


def event_to_sse(event: StreamEvent) -> str:
    """Convert a StreamEvent to SSE format.

    Args:
        event: The StreamEvent to convert.

    Returns:
        JSON string (EventSourceResponse will add "data: " prefix automatically)
    """
    return event.model_dump_json()
