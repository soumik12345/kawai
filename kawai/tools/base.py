from enum import Enum
from typing import Any

import weave
from pydantic import BaseModel


class ToolArgumentType(str, Enum):
    """Enumeration of supported tool argument types.

    This enum defines the valid data types that can be used for tool inputs and outputs.
    These types align with JSON schema types and are used for LLM tool calling.

    Attributes:
        STRING: String/text type
        BOOLEAN: Boolean type (true/false)
        INTEGER: Integer number type
        FLOAT: Floating-point number type
        BOOL: Alias for boolean type
        ARRAY: Array/list type
        OBJECT: Object/dictionary type
    """

    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    ARRAY = "array"
    OBJECT = "object"


class ToolArgument(BaseModel):
    """Pydantic model representing a tool argument specification.

    Attributes:
        type: The data type of the argument
        description: Human-readable description of what the argument represents
        nullable: Whether the argument can be None/null (default: False)
    """

    type: ToolArgumentType
    description: str
    nullable: bool = False


class KawaiTool(weave.Model):
    """Base class for creating tools that can be used with Kawai agents.

    KawaiTool provides a standardized interface for defining tools that agents can invoke.
    It supports automatic JSON schema generation for LLM tool calling and includes
    lifecycle management through initialization hooks.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of what the tool does
        inputs: Dictionary mapping parameter names to their specifications.
            Each specification should contain 'type', 'description', and optionally 'nullable'.
        output_type: The type of value returned by the tool
        output_schema: Optional detailed schema for complex output types
        _is_state_initialized: Internal flag tracking initialization state

    !!! example
        ```python
        class WeatherTool(KawaiTool):
            def __init__(self):
                super().__init__(
                    name="get_weather",
                    description="Get current weather for a location",
                    inputs={
                        "location": {
                            "type": "string",
                            "description": "City name or coordinates"
                        }
                    },
                    output_type=ToolArgumentType.STRING
                )

            def execute(self, location: str) -> str:
                return f"Weather in {location}: Sunny, 72°F"
        ```
    """

    name: str
    description: str
    inputs: dict[str, dict[str, Any]]
    output_type: ToolArgumentType
    output_schema: dict[str, Any] | None = None
    _is_state_initialized: bool = False

    def initialize(self):
        """Initialize the tool's state.

        This method is called automatically before the first execution of the tool.
        Override this method in subclasses to perform any setup operations such as
        loading models, establishing connections, or initializing resources.

        The base implementation simply marks the tool as initialized.
        """
        self._is_state_initialized = True

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool's main functionality.

        This is the core method that must be implemented by all tool subclasses.
        It contains the actual logic for what the tool does when invoked.

        Args:
            *args: Positional arguments passed to the tool
            **kwargs: Keyword arguments passed to the tool, typically matching
                the parameter names defined in the `inputs` attribute

        Returns:
            The result of the tool execution, with type matching `output_type`

        Raises:
            NotImplementedError: If not overridden in a subclass
        """
        raise NotImplementedError(
            "The `execute` method must be implemented by subclasses of `KawaiTool`."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the tool callable like a function.

        This method enables tools to be invoked directly as `tool(args)` rather than
        `tool.execute(args)`. It automatically handles initialization on first call.

        Args:
            *args: Positional arguments to pass to execute()
            **kwargs: Keyword arguments to pass to execute()

        Returns:
            The result from execute()
        """
        if not self._is_state_initialized:
            self.initialize()
        return self.execute(*args, **kwargs)

    def to_prompt(self) -> str:
        """Generate a human-readable text representation of the tool.

        Creates a formatted string describing the tool's name, description, inputs,
        and output type. This is useful for including tool descriptions in agent prompts.

        Returns:
            A formatted string describing the tool's interface

        !!! example
            ```
            get_weather: Get current weather for a location
                Takes inputs: {'location': {'type': 'string', 'description': 'City name'}}
                Returns an output of type: string
            ```
        """
        return (
            f"{self.name}: {self.description}\n"
            + "\tTakes inputs: "
            + str(self.inputs)
            + "\n\t"
            + f"Returns an output of type: {self.output_type}"
        )

    def to_json_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI/Anthropic tool calling JSON schema format.

        Generates a JSON schema representation of the tool that can be sent to LLMs
        for function/tool calling. The schema follows the OpenAI function calling format
        and is compatible with Anthropic's tool use API.

        Returns:
            A dictionary containing the tool's JSON schema with the following structure:
            - type: Always "function"
            - function: Object containing:
                - name: The tool's name
                - description: The tool's description
                - parameters: JSON schema for the tool's parameters including:
                    - type: Always "object"
                    - properties: Parameter specifications
                    - required: List of non-nullable parameter names

        !!! example
            ```python
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
            ```
        """
        properties = {}
        required = []

        for param_name, param_info in self.inputs.items():
            properties[param_name] = {
                "type": param_info["type"],
                "description": param_info["description"],
            }
            # Add nullable field if present
            if param_info.get("nullable", False):
                properties[param_name]["nullable"] = True
            else:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
