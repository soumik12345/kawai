import inspect
from typing import Callable, get_type_hints

from kawai.tools.base import KawaiTool, ToolArgumentType


def _python_type_to_tool_type(python_type: type) -> ToolArgumentType:
    """Convert Python type hints to ToolArgumentType."""
    # Handle string annotations
    if isinstance(python_type, str):
        python_type_lower = python_type.lower()
        if "str" in python_type_lower:
            return ToolArgumentType.STRING
        elif "int" in python_type_lower:
            return ToolArgumentType.INTEGER
        elif "float" in python_type_lower:
            return ToolArgumentType.FLOAT
        elif "bool" in python_type_lower:
            return ToolArgumentType.BOOLEAN
        elif "list" in python_type_lower or "array" in python_type_lower:
            return ToolArgumentType.ARRAY
        elif "dict" in python_type_lower:
            return ToolArgumentType.OBJECT
        return ToolArgumentType.STRING

    # Handle actual type objects
    if python_type == str:
        return ToolArgumentType.STRING
    elif python_type == int:
        return ToolArgumentType.INTEGER
    elif python_type == float:
        return ToolArgumentType.FLOAT
    elif python_type == bool:
        return ToolArgumentType.BOOLEAN
    elif hasattr(python_type, "__origin__"):
        # Handle generic types like List, Dict, etc.
        origin = getattr(python_type, "__origin__", None)
        if origin == list:
            return ToolArgumentType.ARRAY
        elif origin == dict:
            return ToolArgumentType.OBJECT

    # Default to string for unknown types
    return ToolArgumentType.STRING


def _extract_docstring_info(func: Callable) -> tuple[str, dict[str, str]]:
    """Extract description and parameter descriptions from docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return "", {}

    lines = docstring.split("\n")
    description_lines = []
    param_descriptions = {}
    in_args_section = False
    current_param = None
    current_param_desc = []

    for line in lines:
        stripped = line.strip()

        # Check if we're entering the Args/Arguments/Parameters section
        if stripped.lower() in ["args:", "arguments:", "parameters:", "params:"]:
            in_args_section = True
            continue

        # Check if we're leaving the args section (Returns, Raises, etc.)
        if in_args_section and stripped.lower().startswith(
            (
                "returns:",
                "return:",
                "raises:",
                "raise:",
                "yields:",
                "yield:",
                "examples:",
                "example:",
                "notes:",
                "note:",
            )
        ):
            in_args_section = False
            if current_param and current_param_desc:
                param_descriptions[current_param] = " ".join(current_param_desc).strip()
            continue

        if in_args_section:
            # Check if this is a parameter line (starts with parameter name)
            if stripped and not stripped.startswith(" ") and ":" in stripped:
                # Save previous parameter if exists
                if current_param and current_param_desc:
                    param_descriptions[current_param] = " ".join(
                        current_param_desc
                    ).strip()

                # Parse new parameter
                parts = stripped.split(":", 1)
                current_param = parts[0].strip()
                if len(parts) > 1:
                    current_param_desc = [parts[1].strip()]
                else:
                    current_param_desc = []
            elif current_param and stripped:
                # Continuation of parameter description
                current_param_desc.append(stripped)
        elif not in_args_section and stripped:
            # Part of main description
            description_lines.append(stripped)

    # Save last parameter if exists
    if current_param and current_param_desc:
        param_descriptions[current_param] = " ".join(current_param_desc).strip()

    description = " ".join(description_lines).strip()
    return description, param_descriptions


def kawaitool(func: Callable) -> KawaiTool:
    """
    Decorator to convert a Python function into a KawaiTool.

    This decorator extracts metadata from the function signature and docstring
    to create a KawaiTool instance that can be used with Kawai agents.

    Args:
        func: The function to convert into a tool.

    Returns:
        A KawaiTool instance wrapping the function.

    !!! example
        ```python
        @kawaitool
        def add_numbers(a: int, b: int) -> int:
            '''
            Adds two numbers together.

            Args:
                a: The first number
                b: The second number
            '''
            return a + b
        ```
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    # Extract docstring information
    description, param_descriptions = _extract_docstring_info(func)

    # If no description from docstring, use function name
    if not description:
        description = f"Tool: {func.__name__}"

    # Build inputs dictionary
    inputs = {}
    for param_name, param in sig.parameters.items():
        # Get type from type hints or annotation
        param_type = type_hints.get(param_name, param.annotation)
        if param_type == inspect.Parameter.empty:
            param_type = str  # Default to string

        # Convert to ToolArgumentType
        tool_type = _python_type_to_tool_type(param_type)

        # Get description from docstring
        param_desc = param_descriptions.get(param_name, f"Parameter {param_name}")

        # Check if parameter has a default value (making it nullable)
        nullable = param.default != inspect.Parameter.empty

        inputs[param_name] = {
            "type": tool_type.value,
            "description": param_desc,
            "nullable": nullable,
        }

    # Determine output type from return annotation
    return_type = type_hints.get("return", sig.return_annotation)
    if return_type == inspect.Signature.empty:
        output_type = ToolArgumentType.STRING
    else:
        output_type = _python_type_to_tool_type(return_type)

    # Create a KawaiTool subclass dynamically
    class DecoratedTool(KawaiTool):
        def __init__(self):
            super().__init__(
                name=func.__name__,
                description=description,
                inputs=inputs,
                output_type=output_type,
            )
            self._func = func

        def execute(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    # Create and return an instance
    tool_instance = DecoratedTool()

    # Preserve original function attributes
    tool_instance.__name__ = func.__name__
    tool_instance.__doc__ = func.__doc__

    return tool_instance
