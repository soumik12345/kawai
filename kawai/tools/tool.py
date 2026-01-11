from typing import Any, Literal

import weave
from pydantic import BaseModel


class KawaiToolParameter(BaseModel):
    param_name: str
    description: str
    tool_type: Literal["string", "number", "boolean", "object", "array", "any"]
    required: bool = True
    nullable: bool = False


class KawaiTool(BaseModel):
    tool_name: str
    description: str
    parameters: list[KawaiToolParameter]

    def _get_parameter_schema(self, parameter: KawaiToolParameter) -> dict:
        """Generate JSON schema for a single parameter."""
        schema: dict[str, Any] = {"description": parameter.description}

        if parameter.tool_type == "any":
            # For "any" type, don't specify a type constraint
            # This allows any JSON value (string, number, object, array, boolean, null)
            pass
        elif parameter.nullable:
            schema["type"] = [parameter.tool_type, "null"]
        else:
            schema["type"] = parameter.tool_type

        return schema

    def to_json_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        parameter.param_name: self._get_parameter_schema(parameter)
                        for parameter in self.parameters
                    },
                    "required": [
                        parameter.param_name
                        for parameter in self.parameters
                        if parameter.required
                    ],
                    "additionalProperties": False,
                },
            },
        }

    @weave.op
    def forward(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("This method should be implemented by the subclass")
