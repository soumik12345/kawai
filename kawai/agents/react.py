import json
from typing import Any

import weave
from pydantic import BaseModel, Field

from kawai.callback import KawaiCallback
from kawai.models.openai import OpenAIModel
from kawai.prompts import (
    INITIAL_PLAN_PROMPT,
    SYSTEM_PROMPT,
    UPDATE_PLAN_POST_MESSAGES,
    UPDATE_PLAN_PRE_MESSAGES,
)
from kawai.tools import FinalAnswerTool, KawaiTool


class KawaiReactAgent(BaseModel):
    """A [ReAct](https://arxiv.org/abs/2210.03629) agent that uses tool calling via OpenRouter API.

    This agent implements the ReAct paradigm where the agent iteratively:
    1. Reasons about what to do next
    2. Acts by calling a tool
    3. Observes the tool's output
    4. Repeats until the task is complete

    The agent maintains conversation history in OpenAI Chat Completions format and
    uses function calling to execute tools. All executions are automatically tracked
    by Weave for observability.

    Attributes:
        model (str): OpenRouter model identifier (e.g., "openai/gpt-4",
            "anthropic/claude-3-sonnet", "google/gemini-3-flash-preview").
        tools (list[KawaiTool]): List of tools available to the agent. FinalAnswerTool
            is automatically added if not present.
        system_prompt (str): System prompt that defines the agent's behavior. Defaults
            to `SYSTEM_PROMPT` which enforces strict ReAct pattern.
        instructions (str | None): Optional additional instructions appended to the
            system prompt. Use this to provide task-specific guidance without replacing
            the base ReAct behavior. Defaults to None.
        max_steps (int): Maximum number of ReAct steps before stopping. Defaults to 5.
        planning_interval (int | None): If set, agent generates/updates plans at this
            interval. None disables planning (default).
        tool_dict (dict[str, KawaiTool]): Internal mapping of tool names to tool objects.
            Automatically populated from tools list.
        callbacks (list[KawaiCallback]): List of callbacks for monitoring execution.
            Empty list by default.

    !!! example
        ```python
        import weave
        from kawai import KawaiReactAgent, WebSearchTool, KawaiLoggingCallback

        # Initialize Weave for tracking
        weave.init(project_name="my-project")

        # Create agent with web search capability
        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[WebSearchTool()],
            max_steps=10,
            planning_interval=3,  # Re-plan every 3 steps
            callbacks=[KawaiLoggingCallback()],
            instructions="Focus on finding recent sources from 2024."
        )

        # Run the agent
        result = agent.run("Who won the 2024 Nobel Prize in Physics?")
        print(result["final_answer"])
        ```

    Note:
        - Requires `OPENROUTER_API_KEY` environment variable
        - `FinalAnswerTool` is automatically added to complete tasks
        - Agent loops until `FinalAnswerTool` is called or max_steps reached
        - If max_steps is reached without completion, agent is forced to provide
          a final answer in one extra step
        - With planning enabled, agent creates and updates execution plans
    """

    model: OpenAIModel
    tools: list[KawaiTool]
    system_prompt: str = SYSTEM_PROMPT
    instructions: str | None = None
    max_steps: int = 5
    planning_interval: int | None = None
    tool_dict: dict[str, KawaiTool] = Field(default_factory=dict)
    callbacks: list[KawaiCallback] = []
    _compiled_system_prompt: str = ""

    def model_post_init(self, __context: Any) -> None:
        self.tool_dict = {tool.tool_name: tool for tool in self.tools}
        if "final_answer" not in self.tool_dict:
            final_answer_tool = FinalAnswerTool()
            self.tool_dict["final_answer"] = final_answer_tool
            self.tools.append(final_answer_tool)
        # Compile system prompt with instructions if provided
        self._compiled_system_prompt = self.system_prompt
        if self.instructions:
            self._compiled_system_prompt = (
                f"{self.system_prompt}\n\n{self.instructions}"
            )
        self.model.update_memory(content=self._compiled_system_prompt, role="system")

    def _get_tools_description(self) -> str:
        """Generate a description of available tools for planning prompts."""
        descriptions = []
        for tool in self.tools:
            if tool.tool_name != "final_answer":
                params = ", ".join(
                    [f"{p.param_name}: {p.tool_type}" for p in tool.parameters]
                )
                descriptions.append(f"- {tool.tool_name}({params}): {tool.description}")
        return "\n".join(descriptions)

    def _get_memory_summary(self, memory: list[dict[str, Any]]) -> str:
        """Generate a summary of the agent's memory for planning updates."""
        summary_parts = []
        for item in memory:
            if isinstance(item, dict):
                if item.get("role") == "user":
                    continue  # Skip the original task
                elif item.get("role") == "assistant":
                    content = item.get("content", "")
                    if content:
                        summary_parts.append(f"Assistant: {content}")
                    # Handle tool calls embedded in assistant message
                    tool_calls = item.get("tool_calls", [])
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        summary_parts.append(
                            f"Tool call: {func.get('name')}({func.get('arguments', '{}')})"
                        )
                elif item.get("role") == "tool":
                    output = item.get("content", "")
                    # Truncate long outputs
                    if len(output) > 500:
                        output = output[:500] + "..."
                    summary_parts.append(f"Tool output: {output}")
        return "\n".join(summary_parts)

    def _generate_initial_plan(self, task: str) -> tuple[str, list[dict[str, Any]]]:
        """Generate an initial plan for the task using the planning LLM.

        Creates a structured plan using the facts survey methodology:
        1. Facts given in the task
        2. Facts to look up
        3. Facts to derive
        4. Step-by-step high-level plan

        The plan is added to the conversation memory for the agent to follow.

        Args:
            task (str): The task description to plan for.
            memory (list[dict[str, Any]]): Current conversation memory.

        Returns:
            tuple[str, list[dict[str, Any]]]: A tuple containing:
                - The generated plan as a string
                - Updated memory with plan appended

        Note:
            Triggers the `at_planning_end` callback with `updated_plan=False`.
        """
        tools_description = self._get_tools_description()
        planning_prompt = INITIAL_PLAN_PROMPT.format(
            tools_description=tools_description, task=task
        )

        response = self.model.predict_from_messages(
            messages=[
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": planning_prompt},
            ]
        )
        plan = response.choices[0].message.content or ""

        for callback in self.callbacks:
            callback.at_planning_end(plan=plan, updated_plan=False)

        # Add the plan to the conversation for the agent to follow
        self.model.update_memory(
            content=f"I have created the following plan:\n\n{plan}", role="assistant"
        )
        self.model.update_memory(
            content="Now proceed and carry out this plan. Remember to provide your reasoning before each tool call.",
            role="user",
        )

        return plan

    def _generate_updated_plan(
        self, task: str, remaining_steps: int
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate an updated plan based on progress so far.

        Reviews execution history and updates the plan by:
        1. Analyzing what has been accomplished
        2. Updating the facts survey (given, learned, still to lookup, still to derive)
        3. Creating revised high-level steps for remaining work

        The updated plan is added to the conversation memory.

        Args:
            task (str): The original task description.
            memory (list[dict[str, Any]]): Current conversation memory including
                all previous steps and tool calls.
            remaining_steps (int): Number of steps remaining before max_steps limit.

        Returns:
            tuple[str, list[dict[str, Any]]]: A tuple containing:
                - The updated plan as a string
                - Updated memory with new plan appended

        Note:
            Triggers the `at_planning_end` callback with `updated_plan=True`.
        """
        tools_description = self._get_tools_description()
        memory_summary = self._get_memory_summary(self.model.memory.get_messages())

        pre_messages = UPDATE_PLAN_PRE_MESSAGES.format(task=task)
        post_messages = UPDATE_PLAN_POST_MESSAGES.format(
            remaining_steps=remaining_steps, tools_description=tools_description
        )

        planning_prompt = f"{pre_messages}\n\n{memory_summary}\n\n{post_messages}"
        response = self.model.predict_from_messages(
            messages=[
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": planning_prompt},
            ]
        )
        plan = response.choices[0].message.content or ""

        for callback in self.callbacks:
            callback.at_planning_end(plan=plan, updated_plan=True)

        # Add the plan to the conversation for the agent to follow
        self.model.update_memory(
            content=f"I have created the following plan:\n\n{plan}", role="assistant"
        )
        self.model.update_memory(
            content="Now proceed and carry out this plan. Remember to provide your reasoning before each tool call.",
            role="user",
        )

        return plan

    def _should_plan(self, step_idx: int) -> bool:
        """Determine if planning should occur at the current step."""
        if self.planning_interval is None:
            return False
        # Plan at step 0 (initial) and every planning_interval steps after
        if step_idx == 0:
            return True
        return step_idx % self.planning_interval == 0

    def execute_tool_from_response_call(
        self,
    ) -> tuple[bool, str | None]:
        """Execute one ReAct step: get LLM response, call tools, update memory.

        This method implements a single step of the ReAct loop:
        1. Calls the LLM with current memory and tool schemas
        2. Parses the LLM response for reasoning and tool calls
        3. Executes any tool calls and appends results to memory
        4. Handles errors and edge cases (no tool call, unknown tool, etc.)

        The method uses OpenAI Chat Completions format for all messages and
        properly links tool calls with their responses via tool_call_id.
        It operates on and modifies `self.model.memory` directly.

        Returns:
            tuple[bool, str | None]: A tuple containing:
                - Boolean indicating if final_answer was called (task complete)
                - The tool_call_id of the final_answer call, or None

        Note:
            - Operates on and modifies `self.model.memory` directly
            - Triggers callbacks: at_reasoning, at_tool_call, at_tool_result, at_warning
            - Tool errors are caught and returned as {"error": "..."} in memory
            - If model doesn't make a tool call, prompts it to continue
        """
        is_finished = False
        final_answer_call_id = None
        tools_schema = [tool.to_json_schema() for tool in self.tools]

        # Step 1: Get reasoning from the model (no tools available to force text response)
        reasoning_response = self.model.predict_from_memory(tools=None)
        reasoning_message = reasoning_response.choices[0].message

        if not reasoning_message.content:
            # Model didn't provide any reasoning - prompt it to think
            for callback in self.callbacks:
                callback.at_warning(
                    message="Model did not provide reasoning. Prompting to think first."
                )
            self.model.update_memory(
                content="You must first explain your reasoning in plain text. What are you trying to accomplish and what information do you need? Think step by step.",
                role="user",
            )
            return is_finished, final_answer_call_id

        # Log the reasoning
        for callback in self.callbacks:
            callback.at_reasoning(reasoning=reasoning_message.content)

        # Add reasoning to memory as assistant message
        self.model.update_memory(
            content=reasoning_message.content,
            role="assistant",
        )

        # Step 2: Now get the tool call with tools available
        # Add a prompt to make the tool call based on the reasoning
        self.model.update_memory(
            content="Now make a tool call based on your reasoning above.",
            role="user",
        )

        tool_response = self.model.predict_from_memory(tools=tools_schema)
        tool_message = tool_response.choices[0].message

        # Handle case where model still doesn't make a tool call
        if not tool_message.tool_calls:
            for callback in self.callbacks:
                callback.at_warning(
                    message="Model responded without making a tool call. Prompting to continue."
                )
            # Add the response to memory if it has content
            if tool_message.content:
                self.model.update_memory(
                    content=tool_message.content,
                    role="assistant",
                )
            # Remind the model to make a tool call
            self.model.update_memory(
                content="You must make a tool call. If you have enough information to answer, use the final_answer tool. Otherwise, use an appropriate tool to gather more information.",
                role="user",
            )
            return is_finished, final_answer_call_id

        # Add assistant message with tool_calls to memory
        self.model.update_memory(
            content=tool_message.content,
            role="assistant",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_message.tool_calls
            ],
        )

        # Process tool calls
        for tool_call in tool_message.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)

            # Log tool call arguments
            for callback in self.callbacks:
                callback.at_tool_call(
                    tool_name=tool_name, tool_arguments=tool_arguments
                )

            # Execute the tool
            tool_to_execute = self.tool_dict.get(tool_name)
            if not tool_to_execute:
                # Add tool response with role: "tool" (correct format)
                error_content = json.dumps({"error": f"Unknown tool: {tool_name}"})
                self.model.update_memory(
                    content=error_content, role="tool", tool_call_id=tool_call.id
                )
                # Log error result
                for callback in self.callbacks:
                    callback.at_tool_result(
                        tool_name=tool_name, tool_result=error_content
                    )
                continue

            try:
                tool_execution_result = tool_to_execute.forward(**tool_arguments)

                # Check if this is the final answer and track its call_id
                if tool_name == "final_answer":
                    is_finished = True
                    final_answer_call_id = tool_call.id

                # Add tool response with role: "tool" (correct format)
                output_content = (
                    json.dumps(tool_execution_result)
                    if not isinstance(tool_execution_result, str)
                    else tool_execution_result
                )
                self.model.update_memory(
                    content=output_content, role="tool", tool_call_id=tool_call.id
                )

                # Log tool result
                for callback in self.callbacks:
                    callback.at_tool_result(
                        tool_name=tool_name, tool_result=output_content
                    )
            except Exception as e:
                error_content = json.dumps({"error": str(e)})
                self.model.update_memory(
                    content=error_content, role="tool", tool_call_id=tool_call.id
                )
                # Log error result
                for callback in self.callbacks:
                    callback.at_tool_result(
                        tool_name=tool_name, tool_result=error_content
                    )

        return is_finished, final_answer_call_id

    def get_final_answer(self, step_index: int) -> str:
        """If we exhausted max_steps without calling final_answer, force one extra step"""
        for callback in self.callbacks:
            callback.at_warning(
                message=f"Reached max_steps ({self.max_steps}) without final answer. Forcing final_answer call in one extra step."
            )

        # Add a strong prompt to force the agent to provide a final answer
        self.model.update_memory(
            content=(
                "You have reached the maximum number of steps. You MUST now call the final_answer tool "
                "with your best answer based on all the information you have gathered so far. "
                "Provide a comprehensive summary of what you have learned and your conclusion."
            ),
            role="user",
        )

        # Execute one final step
        for callback in self.callbacks:
            callback.at_step_start(step_index=step_index + 1)

        is_finished, final_answer_call_id = self.execute_tool_from_response_call()

        # Extract the final answer if it was provided
        if is_finished and final_answer_call_id:
            for item in self.model.memory.get_messages():
                if (
                    isinstance(item, dict)
                    and item.get("role") == "tool"
                    and item.get("tool_call_id") == final_answer_call_id
                ):
                    output = item.get("content")
                    try:
                        final_answer = json.loads(output)
                    except (json.JSONDecodeError, TypeError):
                        final_answer = output
                    break
            step_index += 1  # Increment to reflect the extra step

        return final_answer

    @weave.op
    def run(self, prompt: str, force_provide_answer: bool = False) -> dict[str, Any]:
        """Execute the agent on a task and return the result.

        This is the main entry point for running the agent. It implements the full
        ReAct loop with optional planning:

        1. Initialize memory with system prompt and task
        2. (Optional) Generate initial plan if planning_interval is set
        3. Loop up to max_steps:
           - (Optional) Update plan at planning_interval
           - Get reasoning and tool call from LLM
           - Execute tool and observe result
           - If final_answer called, extract and return answer
        4. If max_steps reached without final_answer, force one extra step to get
           a final answer based on all information gathered
        5. Return comprehensive result dictionary

        The method is tracked by Weave as an operation for full observability.

        Args:
            prompt (str): The task description or question for the agent to solve.
            force_provide_answer (bool): Force to provide an answer even though `max_steps`
                have been exhausted.

        Returns:
            dict[str, Any]: A dictionary containing:
                - final_answer (Any): The final answer from FinalAnswerTool. If max_steps
                    is reached without completion, the agent is forced to provide a final
                    answer in one extra step
                - steps (int): Number of ReAct steps executed (may be max_steps + 1 if
                    forced final answer was triggered)
                - memory (list[dict[str, Any]]): Full conversation history in OpenAI
                    format, including all reasoning, tool calls, and results
                - completed (bool): Whether the task completed successfully (final_answer
                    was called, either naturally or forced)
                - plan (str | None): The final plan if planning was enabled, or None

        !!! example
            ```python
            agent = KawaiReactAgent(
                model="openai/gpt-4",
                tools=[WebSearchTool()],
                max_steps=5
            )

            result = agent.run("What is the population of Tokyo?")

            print(f"Answer: {result['final_answer']}")
            print(f"Took {result['steps']} steps")
            print(f"Completed: {result['completed']}")
            ```

        Note:
            - Triggers callbacks: at_run_start, at_step_start, at_run_end, at_warning
            - Also triggers planning and tool execution callbacks
            - If max_steps is exhausted, triggers at_warning before forcing final answer
            - Tracked by Weave - view traces at wandb.ai/weave
        """
        for callback in self.callbacks:
            callback.at_run_start(prompt=prompt, model=self.model.model_id)

        self.model.update_memory(content=prompt, role="user")
        final_answer = None
        is_finished = False
        step_index = -1
        current_plan: str | None = None

        for step_index in range(self.max_steps):
            for callback in self.callbacks:
                callback.at_step_start(step_index=step_index)

            # Check if we should generate/update the plan
            if self._should_plan(step_index):
                remaining_steps = self.max_steps - step_index
                current_plan = (
                    self._generate_initial_plan(task=prompt)
                    if step_index == 0
                    else self._generate_updated_plan(
                        task=prompt, remaining_steps=remaining_steps
                    )
                )

            is_finished, final_answer_call_id = self.execute_tool_from_response_call()

            # Call at_step_end with cumulative token usage
            token_usage = self.model.get_cumulative_token_usage()
            for callback in self.callbacks:
                callback.at_step_end(
                    step_index=step_index,
                    cumulative_input_tokens=token_usage["input_tokens"],
                    cumulative_output_tokens=token_usage["output_tokens"],
                    cumulative_total_tokens=token_usage["total_tokens"],
                )

            if is_finished and final_answer_call_id:
                # Find the specific tool response with the matching tool_call_id
                for item in self.model.memory.get_messages():
                    if (
                        isinstance(item, dict)
                        and item.get("role") == "tool"
                        and item.get("tool_call_id") == final_answer_call_id
                    ):
                        # FinalAnswerTool returns the answer directly (passthrough)
                        output = item.get("content")
                        try:
                            # Try to parse as JSON in case it was serialized
                            final_answer = json.loads(output)
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, use the raw value
                            final_answer = output
                        break
                break

        # If we exhausted max_steps without calling final_answer, force one extra step
        if not is_finished and force_provide_answer:
            final_answer = self.get_final_answer(step_index=step_index)

        token_usage = self.model.get_cumulative_token_usage()
        for callback in self.callbacks:
            callback.at_run_end(
                answer=final_answer,
                cumulative_input_tokens=token_usage["input_tokens"],
                cumulative_output_tokens=token_usage["output_tokens"],
                cumulative_total_tokens=token_usage["total_tokens"],
            )

        return {
            "final_answer": final_answer,
            "steps": step_index + 1,
            "memory": self.model.memory.get_messages(),
            "completed": is_finished,
            "plan": current_plan,
        }

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        session_timeout: int = 3600,
        enable_cors: bool = True,
        allowed_origins: list[str] | None = None,
        log_level: str = "info",
    ) -> None:
        """Start a FastAPI server to serve the agent via REST API.

        This method starts a web server that exposes the agent through HTTP endpoints,
        allowing you to query the agent from other applications or integrate it into
        existing infrastructure.

        Available endpoints:
            - `POST /chat`: Non-streaming chat endpoint. Send a message and get the final response.
            - `GET /stream`: Server-Sent Events streaming endpoint for real-time updates.
            - `GET /health`: Health check endpoint with server status and session count.
            - `GET /sessions/{session_id}`: Get information about a specific session.
            - `DELETE /sessions/{session_id}`: Delete a session.

        Sessions maintain conversation memory across requests, enabling multi-turn
        conversations with the agent.

        Args:
            host (str): Host address to bind the server to. Defaults to "0.0.0.0"
                (all interfaces).
            port (int): Port number to bind the server to. Defaults to 8000.
            session_timeout (int): Session expiration time in seconds. Sessions that
                haven't been accessed within this time are automatically cleaned up.
                Defaults to 3600 (1 hour).
            enable_cors (bool): Whether to enable CORS middleware for cross-origin
                requests from browsers. Defaults to True.
            allowed_origins (list[str] | None): List of allowed origins for CORS.
                If None and CORS is enabled, allows all origins ("*").
            log_level (str): Uvicorn log level. One of "critical", "error", "warning",
                "info", "debug", "trace". Defaults to "info".

        !!! example "Basic Usage"
            ```python
            import weave
            from kawai import KawaiReactAgent, WebSearchTool, OpenAIModel

            weave.init(project_name="kawai-server")

            model = OpenAIModel(
                model_id="google/gemini-3-flash-preview",
                base_url="https://openrouter.ai/api/v1",
                api_key_env_var="OPENROUTER_API_KEY",
            )

            agent = KawaiReactAgent(
                model=model,
                tools=[WebSearchTool()],
                max_steps=10,
            )

            # Start the server (blocking)
            agent.serve(port=8000)
            ```

        !!! example "Client Usage - Non-streaming"
            ```python
            import requests

            # Send a chat request
            response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "message": "What is the capital of France?",
                    "session_id": "user-123"  # Optional: maintains conversation
                }
            )

            result = response.json()
            print(result["answer"])  # The agent's response
            print(result["session_id"])  # Use this for follow-up messages
            ```

        !!! example "Client Usage - Streaming"
            ```python
            import requests
            import json

            # Connect to streaming endpoint
            response = requests.get(
                "http://localhost:8000/stream",
                params={
                    "message": "Search for latest AI news",
                    "session_id": "user-123"
                },
                stream=True
            )

            # Process events as they arrive
            for line in response.iter_lines():
                if line:
                    data = line.decode().removeprefix("data: ")
                    event = json.loads(data)
                    print(f"{event['type']}: {event['data']}")
            ```

        Note:
            - This method blocks until the server is shut down (Ctrl+C)
            - The server uses uvicorn as the ASGI server
            - Sessions are stored in memory and lost on server restart
            - For production use, consider using a reverse proxy like nginx
        """
        import uvicorn

        from kawai.server.app import create_app

        # Create the FastAPI app with this agent as the template
        app = create_app(
            agent=self,
            session_timeout=session_timeout,
            enable_cors=enable_cors,
            allowed_origins=allowed_origins,
        )

        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
        )
