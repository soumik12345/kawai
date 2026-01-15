from typing import Any, Literal

import weave
from pydantic import BaseModel


class KawaiToolParameter(BaseModel):
    """Defines a parameter for a KawaiTool.

    This class specifies the schema for a single parameter that a tool accepts,
    including its name, type, description, and validation constraints.

    Attributes:
        param_name (str): The name of the parameter as it will appear in tool calls.
        description (str): Human-readable description of what the parameter does.
        tool_type (Literal["string", "number", "boolean", "object", "array", "any"]): JSON schema
            type of the parameter. Supported types are:
                - "string": Text values
                - "number": Numeric values (int or float)
                - "boolean": True/False values
                - "object": JSON objects/dictionaries
                - "array": JSON arrays/lists
                - "any": Any JSON value (no type constraint)
        required (bool): Whether this parameter must be provided in tool calls. Defaults to True.
        nullable (bool): Whether the parameter can be null in addition to its declared type.
            Defaults to False.

    !!! example
        ```python
        param = KawaiToolParameter(
            param_name="query",
            description="The search query to execute",
            tool_type="string",
            required=True,
            nullable=False
        )
        ```
    """

    param_name: str
    description: str
    tool_type: Literal["string", "number", "boolean", "object", "array", "any"]
    required: bool = True
    nullable: bool = False


class KawaiTool(BaseModel):
    """Base class for all tools that can be used by KawaiReactAgent.

    Tools are functions that agents can call to interact with external systems,
    retrieve information, or perform actions. Each tool must define its name,
    description, parameters, and implement the `forward()` method.

    All tool executions are automatically tracked by Weave when the `forward()`
    method is decorated with `@weave.op`.

    Attributes:
        tool_name: Unique identifier for the tool. Used in function calling.
        description: Human-readable description of what the tool does. Shown to
            the agent to help it decide when to use this tool.
        parameters: List of KawaiToolParameter objects defining the tool's inputs.
        cacheable: Whether the tool's output can be cached. Defaults to True.
            Set to False for non-deterministic tools or tools with side effects.

    !!! example
        ```python
        class CalculatorTool(KawaiTool):
            tool_name: str = "calculator"
            description: str = "Performs basic arithmetic operations"
            parameters: list[KawaiToolParameter] = [
                KawaiToolParameter(
                    param_name="expression",
                    description="Mathematical expression to evaluate",
                    tool_type="string"
                )
            ]
            cacheable: bool = True

            @weave.op
            def forward(self, expression: str) -> dict[str, Any]:
                result = eval(expression)  # Note: unsafe, just for demo
                return {"result": result}
        ```

    Note:
        - Subclasses must implement the forward() method
        - The forward() method should be decorated with @weave.op for tracking
        - Return values should be JSON-serializable (dict, str, int, float, list, etc.)
    """

    tool_name: str
    description: str
    parameters: list[KawaiToolParameter]
    cacheable: bool = True

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
        """Convert the tool definition to OpenAI function calling JSON schema format.

        This method generates the schema that gets passed to the LLM so it knows
        how to call this tool. The schema follows the OpenAI function calling format.

        Returns:
            A dictionary containing the tool schema with structure:
            ```python
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "tool description",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...],
                        "additionalProperties": False
                    }
                }
            }
            ```

        Note:
            This is called automatically by the agent when registering tools.
            You typically don't need to call this method directly.
        """
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
    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the provided arguments.

        This is the main method that implements the tool's functionality. It must
        be overridden by subclasses to define what the tool actually does.

        Args:
            **kwargs: The tool parameters as specified in the parameters list.
                Parameter names and types must match the tool's parameter definitions.

        Returns:
            The tool's output, which should be JSON-serializable. Can be:
            - dict: For structured results
            - str: For simple text results
            - int/float: For numeric results
            - list: For multiple results
            - Any other JSON-serializable type

        Raises:
            NotImplementedError: If the subclass hasn't implemented this method.

        Note:
            - This method should be decorated with @weave.op in subclasses
            - Return values are automatically serialized to JSON by the agent
            - Errors raised here are caught and returned as {"error": "..."} to the agent
        """
        raise NotImplementedError("This method should be implemented by the subclass")
