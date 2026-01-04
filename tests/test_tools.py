"""
Parameterized unit tests for Kawai tools based on decorator examples.

This test suite validates the @kawaitool decorator functionality using
parameterized tests based on real-world examples from the examples directory.
"""

from typing import Any, Callable

import pytest

from kawai import kawaitool


# Test functions from examples
@kawaitool
def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers together.

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        The sum of a and b
    """
    return a + b


@kawaitool
def search_web(query: str, max_results: int = 10) -> str:
    """
    Performs a web search and returns results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (optional)

    Returns:
        A string containing the search results
    """
    # Simulated web search
    return f"Search results for '{query}' (showing {max_results} results)"


@kawaitool
def process_data(data: dict, format_output: bool = True) -> dict:
    """
    Processes input data and returns formatted results.

    Args:
        data: The input data dictionary to process
        format_output: Whether to format the output (optional)

    Returns:
        Processed data dictionary
    """
    processed = {k: str(v).upper() for k, v in data.items()}
    if format_output:
        processed["formatted"] = True
    return processed


class TestKawaiToolDecorator:
    """Test suite for @kawaitool decorator using parameterized tests."""

    @pytest.mark.parametrize(
        "tool_func,expected_name,expected_description,expected_output_type",
        [
            (
                add_numbers,
                "add_numbers",
                "Adds two numbers together.",
                "integer",
            ),
            (
                search_web,
                "search_web",
                "Performs a web search and returns results.",
                "string",
            ),
            (
                process_data,
                "process_data",
                "Processes input data and returns formatted results.",
                "object",
            ),
        ],
    )
    def test_tool_metadata(
        self,
        tool_func: Callable,
        expected_name: str,
        expected_description: str,
        expected_output_type: str,
    ):
        """Test that tools have correct metadata extracted from function signatures and docstrings."""
        assert tool_func.name == expected_name
        assert tool_func.description == expected_description
        assert tool_func.output_type.value == expected_output_type

    @pytest.mark.parametrize(
        "tool_func,expected_inputs",
        [
            (
                add_numbers,
                {
                    "a": {
                        "type": "integer",
                        "description": "The first number to add",
                        "nullable": False,
                    },
                    "b": {
                        "type": "integer",
                        "description": "The second number to add",
                        "nullable": False,
                    },
                },
            ),
            (
                search_web,
                {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                        "nullable": False,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (optional)",
                        "nullable": True,
                    },
                },
            ),
            (
                process_data,
                {
                    "data": {
                        "type": "object",
                        "description": "The input data dictionary to process",
                        "nullable": False,
                    },
                    "format_output": {
                        "type": "boolean",
                        "description": "Whether to format the output (optional)",
                        "nullable": True,
                    },
                },
            ),
        ],
    )
    def test_tool_inputs(
        self, tool_func: Callable, expected_inputs: dict[str, dict[str, Any]]
    ):
        """Test that tools have correct input specifications extracted from function signatures."""
        assert tool_func.inputs == expected_inputs

    @pytest.mark.parametrize(
        "tool_func,args,kwargs,expected_result",
        [
            (add_numbers, (5, 3), {}, 8),
            (add_numbers, (-10, 20), {}, 10),
            (
                search_web,
                ("Python tutorials",),
                {},
                "Search results for 'Python tutorials' (showing 10 results)",
            ),
            (
                search_web,
                ("AI news",),
                {"max_results": 5},
                "Search results for 'AI news' (showing 5 results)",
            ),
            (
                process_data,
                ({"name": "john", "city": "paris"},),
                {},
                {"name": "JOHN", "city": "PARIS", "formatted": True},
            ),
            (
                process_data,
                ({"name": "john", "city": "paris"},),
                {"format_output": False},
                {"name": "JOHN", "city": "PARIS"},
            ),
        ],
    )
    def test_tool_execution(
        self, tool_func: Callable, args: tuple, kwargs: dict, expected_result: Any
    ):
        """Test that tools execute correctly with various argument combinations."""
        result = tool_func(*args, **kwargs)
        assert result == expected_result

    @pytest.mark.parametrize(
        "tool_func,expected_schema_keys",
        [
            (add_numbers, {"type", "function"}),
            (search_web, {"type", "function"}),
            (process_data, {"type", "function"}),
        ],
    )
    def test_json_schema_structure(
        self, tool_func: Callable, expected_schema_keys: set[str]
    ):
        """Test that tools generate valid JSON schema structure."""
        schema = tool_func.to_json_schema()

        # Check top-level structure
        assert set(schema.keys()) == expected_schema_keys
        assert schema["type"] == "function"

        # Check function object structure
        function_obj = schema["function"]
        assert set(function_obj.keys()) == {"name", "description", "parameters"}

        # Check parameters structure
        parameters = function_obj["parameters"]
        assert parameters["type"] == "object"
        assert "properties" in parameters
        assert "required" in parameters

    @pytest.mark.parametrize(
        "tool_func,expected_required_params",
        [
            (add_numbers, ["a", "b"]),
            (search_web, ["query"]),
            (process_data, ["data"]),
        ],
    )
    def test_json_schema_required_params(
        self, tool_func: Callable, expected_required_params: list[str]
    ):
        """Test that JSON schema correctly identifies required vs optional parameters."""
        schema = tool_func.to_json_schema()
        required_params = schema["function"]["parameters"]["required"]
        assert required_params == expected_required_params

    @pytest.mark.parametrize(
        "tool_func,expected_prompt_elements",
        [
            (add_numbers, ("add_numbers", "Adds two numbers together", "integer")),
            (
                search_web,
                ("search_web", "Performs a web search and returns results", "string"),
            ),
            (
                process_data,
                (
                    "process_data",
                    "Processes input data and returns formatted results",
                    "object",
                ),
            ),
        ],
    )
    def test_to_prompt_format(
        self, tool_func: Callable, expected_prompt_elements: tuple[str, ...]
    ):
        """Test that to_prompt() generates properly formatted tool descriptions."""
        prompt = tool_func.to_prompt()

        # Check that all expected elements are present
        for element in expected_prompt_elements:
            assert element in prompt

        # Check prompt structure
        assert f"{tool_func.name}:" in prompt
        assert "Takes inputs:" in prompt
        assert "Returns an output of type:" in prompt

    @pytest.mark.parametrize(
        "tool_func",
        [add_numbers, search_web, process_data],
    )
    def test_initialization_state(self, tool_func: Callable):
        """Test that tools are properly initialized after creation."""
        # Tools should be initialized when created
        assert tool_func._is_state_initialized

        # Should be callable without additional initialization
        if tool_func == add_numbers:
            result = tool_func(1, 2)
            assert result == 3

    @pytest.mark.parametrize(
        "tool_func",
        [add_numbers, search_web, process_data],
    )
    def test_function_attributes_preserved(self, tool_func: Callable):
        """Test that original function attributes are preserved in the tool."""
        assert tool_func.__name__ == tool_func.name
        assert tool_func.__doc__ is not None
        assert tool_func.description in tool_func.__doc__
