import json
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from kawai.tools.tool import FinalAnswerTool, KawaiTool


class ReactAgent(BaseModel):
    model: str
    tools: list[KawaiTool]
    system_prompt: str
    max_steps: int = 5
    tool_dict: dict[str, KawaiTool] = Field(default_factory=dict)
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

    def execute_tool_from_response_call(
        self, memory: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        response = self._client.responses.create(
            model=self.model,
            tools=[tool.to_json_schema() for tool in self.tools],
            input=memory,
        )

        is_finished = False
        final_answer_call_id = None

        # First, add all items from response.output to memory
        for item in response.output:
            if item.type == "function_call":
                # Add the function call itself to memory
                memory.append(
                    {
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                )
            elif item.type == "message":
                # Add message items to memory
                memory.append(
                    item.model_dump() if hasattr(item, "model_dump") else item
                )
            # Skip reasoning items - they should not be added back to input

        # Then, execute function calls and add their outputs
        for item in response.output:
            if item.type == "function_call":
                tool_to_execute = self.tool_dict.get(item.name)
                if not tool_to_execute:
                    memory.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(
                                {"error": f"Unknown tool: {item.name}"}
                            ),
                        }
                    )
                    continue
                try:
                    tool_execution_result = tool_to_execute.forward(
                        **json.loads(item.arguments)
                    )

                    # Check if this is the final answer and track its call_id
                    if item.name == "final_answer":
                        is_finished = True
                        final_answer_call_id = item.call_id

                    memory.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps(tool_execution_result)
                            if not isinstance(tool_execution_result, str)
                            else tool_execution_result,
                        }
                    )
                except Exception as e:
                    memory.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"error": str(e)}),
                        }
                    )
        return memory, is_finished, final_answer_call_id

    def run(self, prompt: str) -> dict[str, Any]:
        memory = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        final_answer = None
        is_finished = False
        step_idx = -1

        for step_idx in range(self.max_steps):
            memory, is_finished, final_answer_call_id = (
                self.execute_tool_from_response_call(memory)
            )

            if is_finished and final_answer_call_id:
                # Find the specific function_call_output with the matching call_id
                for item in memory:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "function_call_output"
                        and item.get("call_id") == final_answer_call_id
                    ):
                        try:
                            final_answer = json.loads(item["output"])
                        except (json.JSONDecodeError, KeyError):
                            final_answer = item.get("output")
                        break
                break

        return {
            "final_answer": final_answer,
            "steps": step_idx + 1,
            "memory": memory,
            "completed": is_finished,
        }
