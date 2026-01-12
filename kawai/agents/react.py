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
        memory_summary = self._get_memory_summary(self.model.memory)

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

    @weave.op
    def run(self, prompt: str) -> dict[str, Any]:
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
        4. Return comprehensive result dictionary

        The method is tracked by Weave as an operation for full observability.

        Args:
            prompt (str): The task description or question for the agent to solve.

        Returns:
            dict[str, Any]: A dictionary containing:
                - final_answer (Any): The final answer from FinalAnswerTool, or None
                    if max_steps reached without completion
                - steps (int): Number of ReAct steps executed
                - memory (list[dict[str, Any]]): Full conversation history in OpenAI
                    format, including all reasoning, tool calls, and results
                - completed (bool): Whether the task completed successfully (final_answer
                    was called)
                - plan (str | None): The final plan if planning was enabled, or None

        !!! example
            ```python
            agent = KawaiReactAgent(
                model="openai/gpt-4",
                tools=[WebSearchTool()],
                max_steps=5
            )

            result = agent.run("What is the population of Tokyo?")

            if result["completed"]:
                print(f"Answer: {result['final_answer']}")
                print(f"Took {result['steps']} steps")
            else:
                print("Task incomplete - reached max_steps")
            ```

        Note:
            - Triggers callbacks: at_run_start, at_step_start, at_run_end
            - Also triggers planning and tool execution callbacks
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

            if is_finished and final_answer_call_id:
                # Find the specific tool response with the matching tool_call_id
                for item in self.model.memory:
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

        for callback in self.callbacks:
            callback.at_run_end(answer=final_answer)

        return {
            "final_answer": final_answer,
            "steps": step_index + 1,
            "memory": self.model.memory,
            "completed": is_finished,
            "plan": current_plan,
        }
