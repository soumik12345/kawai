import json
import os
from typing import Any

import weave
from openai import OpenAI
from pydantic import BaseModel, Field

from kawai.callback import KawaiCallback
from kawai.prompts import (
    INITIAL_PLAN_PROMPT,
    SYSTEM_PROMPT,
    UPDATE_PLAN_POST_MESSAGES,
    UPDATE_PLAN_PRE_MESSAGES,
)
from kawai.tools import FinalAnswerTool, KawaiTool


class KawaiReactAgent(BaseModel):
    model: str
    tools: list[KawaiTool]
    system_prompt: str = SYSTEM_PROMPT
    max_steps: int = 5
    planning_interval: int | None = None
    tool_dict: dict[str, KawaiTool] = Field(default_factory=dict)
    callbacks: list[KawaiCallback] = []
    _client: OpenAI | None = None

    def model_post_init(self, __context: Any) -> None:
        self.tool_dict = {tool.tool_name: tool for tool in self.tools}
        if "final_answer" not in self.tool_dict:
            final_answer_tool = FinalAnswerTool()
            self.tool_dict["final_answer"] = final_answer_tool
            self.tools.append(final_answer_tool)
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

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

    def _generate_initial_plan(
        self, task: str, memory: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate an initial plan for the task."""
        tools_description = self._get_tools_description()
        planning_prompt = INITIAL_PLAN_PROMPT.format(
            tools_description=tools_description, task=task
        )

        planning_messages = [
            {"role": "system", "content": "You are a planning assistant."},
            {"role": "user", "content": planning_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=planning_messages,
        )

        plan = response.choices[0].message.content or ""

        for callback in self.callbacks:
            callback.at_planning_end(plan=plan, updated_plan=False)

        # Add the plan to the conversation for the agent to follow
        memory.append(
            {
                "role": "assistant",
                "content": f"I have created the following plan:\n\n{plan}",
            }
        )
        memory.append(
            {
                "role": "user",
                "content": "Now proceed and carry out this plan.",
            }
        )

        return plan, memory

    def _generate_updated_plan(
        self, task: str, memory: list[dict[str, Any]], remaining_steps: int
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate an updated plan based on progress so far."""
        tools_description = self._get_tools_description()
        memory_summary = self._get_memory_summary(memory)

        pre_messages = UPDATE_PLAN_PRE_MESSAGES.format(task=task)
        post_messages = UPDATE_PLAN_POST_MESSAGES.format(
            remaining_steps=remaining_steps, tools_description=tools_description
        )

        planning_prompt = f"{pre_messages}\n\n{memory_summary}\n\n{post_messages}"

        planning_messages = [
            {"role": "system", "content": "You are a planning assistant."},
            {"role": "user", "content": planning_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=planning_messages,
        )

        plan = response.choices[0].message.content or ""

        for callback in self.callbacks:
            callback.at_planning_end(plan=plan, updated_plan=True)

        # Add the plan to the conversation for the agent to follow
        # We use standard role-based messages that the API understands
        memory.append(
            {
                "role": "assistant",
                "content": f"I have updated my plan based on progress so far:\n\n{plan}",
            }
        )
        memory.append(
            {
                "role": "user",
                "content": "Now proceed and carry out this updated plan.",
            }
        )

        return plan, memory

    def _should_plan(self, step_idx: int) -> bool:
        """Determine if planning should occur at the current step."""
        if self.planning_interval is None:
            return False
        # Plan at step 0 (initial) and every planning_interval steps after
        if step_idx == 0:
            return True
        return step_idx % self.planning_interval == 0

    def execute_tool_from_response_call(
        self, memory: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        response = self._client.chat.completions.create(
            model=self.model,
            tools=[tool.to_json_schema() for tool in self.tools],
            messages=memory,
        )

        is_finished = False
        final_answer_call_id = None
        message = response.choices[0].message

        # Log reasoning if present (assistant's thinking before tool calls)
        if message.content:
            for callback in self.callbacks:
                callback.at_reasoning(reasoning=message.content)

        # Add assistant message to memory with tool_calls if present
        # This uses the correct OpenAI Chat Completions API format
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_message["content"] = message.content
        else:
            assistant_message["content"] = None

        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

        memory.append(assistant_message)

        # Process tool calls if present
        if message.tool_calls:
            for tool_call in message.tool_calls:
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
                    memory.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content,
                        }
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
                    memory.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_content,
                        }
                    )

                    # Log tool result
                    for callback in self.callbacks:
                        callback.at_tool_result(
                            tool_name=tool_name, tool_result=output_content
                        )
                except Exception as e:
                    error_content = json.dumps({"error": str(e)})
                    memory.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content,
                        }
                    )
                    # Log error result
                    for callback in self.callbacks:
                        callback.at_tool_result(
                            tool_name=tool_name, tool_result=error_content
                        )

        return memory, is_finished, final_answer_call_id

    @weave.op
    def run(self, prompt: str) -> dict[str, Any]:
        for callback in self.callbacks:
            callback.at_run_start(prompt=prompt, model=self.model)

        memory: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
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
                if step_index == 0:
                    # Generate initial plan
                    current_plan, memory = self._generate_initial_plan(prompt, memory)
                else:
                    # Update plan based on progress
                    current_plan, memory = self._generate_updated_plan(
                        prompt, memory, remaining_steps
                    )

            memory, is_finished, final_answer_call_id = (
                self.execute_tool_from_response_call(memory)
            )

            if is_finished and final_answer_call_id:
                # Find the specific tool response with the matching tool_call_id
                for item in memory:
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
            "memory": memory,
            "completed": is_finished,
            "plan": current_plan,
        }
