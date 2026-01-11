from typing import Any, Literal

from pydantic import BaseModel


class KawaiToolParameter(BaseModel):
    param_name: str
    description: str
    tool_type: Literal["string", "number", "boolean", "object", "array"]
    required: bool = True


class KawaiTool(BaseModel):
    tool_name: str
    description: str
    parameters: list[KawaiToolParameter]

    def to_json_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.tool_name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter.param_name: {
                        "type": parameter.tool_type,
                        "description": parameter.description,
                    }
                    for parameter in self.parameters
                },
                "required": [
                    parameter.param_name
                    for parameter in self.parameters
                    if parameter.required
                ],
            },
        }

    def forward(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by the subclass")


class FinalAnswerTool(KawaiTool):
    tool_name: str = "final_answer"
    description: str = "Provides a final answer to the given problem."
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="answer",
            tool_type="string",
            description="The final answer to the problem",
        )
    ]

    def forward(self, **kwargs) -> dict[str, Any]:
        answer = kwargs.get("answer", "")
        return {"answer": answer}
