import json
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax


class KawaiCallback(BaseModel):
    """Base class for implementing callbacks to monitor agent execution.

    Callbacks provide hooks into the agent's execution lifecycle, allowing you to
    observe, log, or react to various events during agent runs. All callback methods
    are optional - override only the ones you need.

    Common use cases:
    - Logging agent behavior to console or files
    - Collecting metrics and performance data
    - Integrating with monitoring systems
    - Custom visualization of agent reasoning
    - Debugging and development

    !!! example
        ```python
        class MyCustomCallback(KawaiCallback):
            def at_reasoning(self, reasoning: str):
                print(f"Agent is thinking: {reasoning}")

            def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]):
                print(f"Calling {tool_name} with args: {tool_arguments}")

        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[WebSearchTool()],
            callbacks=[MyCustomCallback()]
        )
        ```

    Note:
        - All methods have default no-op implementations
        - Callbacks are called synchronously during agent execution
        - Multiple callbacks can be attached to a single agent
        - Exceptions in callbacks will propagate and stop execution
    """

    def __init__(self) -> None:
        super().__init__()

    def at_run_start(self, prompt: str, model: str):
        """Called when the agent run begins.

        Args:
            prompt (str): The task or question given to the agent.
            model (str): The model identifier being used (e.g., "openai/gpt-4").
        """
        pass

    def at_run_end(self, answer: str):
        """Called when the agent run completes.

        Args:
            answer (str): The final answer from the agent, or None if max_steps
                was reached without completion.
        """
        pass

    def at_step_start(self, step_index: int):
        """Called at the beginning of each ReAct step.

        Args:
            step_index (int): The zero-based index of the current step.
        """
        pass

    def at_planning_end(self, plan: str, updated_plan: bool):
        """Called after the agent generates or updates a plan.

        Only triggered when planning_interval is set on the agent.

        Args:
            plan (str): The generated or updated plan text.
            updated_plan (bool): True if this is an updated plan, False if initial plan.
        """
        pass

    def at_reasoning(self, reasoning: str):
        """Called when the agent produces reasoning/thinking content.

        This is the agent's natural language explanation of what it's doing before
        it makes a tool call.

        Args:
            reasoning (str): The reasoning text produced by the agent.
        """
        pass

    def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]):
        """Called when the agent is about to execute a tool.

        Args:
            tool_name (str): The name of the tool being called.
            tool_arguments (dict[str, Any]): The arguments passed to the tool.
        """
        pass

    def at_tool_result(self, tool_name: str, tool_result: str):
        """Called after a tool has been executed with its result.

        Args:
            tool_name (str): The name of the tool that was executed.
            tool_result (str): The result from the tool, as a JSON string or plain string.
        """
        pass

    def at_warning(self, message: str):
        """Called when the agent encounters a warning condition.

        Common warnings include the model not making a tool call when expected.

        Args:
            message (str): Description of the warning condition.
        """
        pass


class KawaiLoggingCallback(KawaiCallback):
    """Rich console-based logging callback for visualizing agent execution.

    This callback provides formatted, colored console output for all agent events
    using the Rich library. It displays reasoning, tool calls, results, and warnings
    in visually distinct panels and styles.

    Output features:
    - Color-coded panels for different event types
    - Syntax highlighting for JSON tool arguments and results
    - Markdown rendering for reasoning and answers
    - Optional truncation of long outputs
    - Step separators with visual rules

    Attributes:
        truncate (bool): Whether to truncate long tool results. Defaults to False.
        max_result_length (int): Maximum length for tool results when truncate=True.
            Defaults to 1000 characters.

    !!! example
        ```python
        from kawai import KawaiReactAgent, WebSearchTool, KawaiLoggingCallback

        # Basic usage
        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[WebSearchTool()],
            callbacks=[KawaiLoggingCallback()]
        )

        # With truncation for long outputs
        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[WebSearchTool()],
            callbacks=[KawaiLoggingCallback(truncate=True, max_result_length=500)]
        )
        ```

    Visual Output:
        - Run start/end: Yellow bordered panels
        - Steps: Yellow horizontal rules
        - Reasoning: Cyan panels with markdown
        - Tool calls: Green panels with JSON syntax highlighting
        - Tool results: Magenta panels (JSON or markdown)
        - Warnings: Red panels

    Note:
        - Requires the `rich` library (included in dependencies)
        - Outputs to stdout using Rich Console
        - All content is formatted for terminal display
    """

    truncate: bool = False
    _console: Console | None = None
    max_result_length: int = 1000

    def __init__(self, truncate: bool = False, max_result_length: int = 1000) -> None:
        super().__init__()
        self._console = Console()
        self.truncate = truncate
        self.max_result_length = max_result_length

    def at_run_start(self, prompt: str, model: str):
        self._console.print(
            Panel(
                prompt,
                title="[bold yellow]New run[/bold yellow]",
                title_align="center",
                border_style="yellow",
                subtitle=f"Model - {model}",
                subtitle_align="right",
            )
        )

    def at_run_end(self, answer: str):
        self._console.print(
            Panel(
                Markdown(str(answer) if answer else "No answer provided"),
                title="[bold yellow]Final Answer[/bold yellow]",
                title_align="center",
                border_style="yellow",
            )
        )

    def at_step_start(self, step_index: int):
        self._console.print(
            Rule(title=f"[bold]Step {step_index}[/bold]", align="right", style="yellow")
        )

    def at_planning_end(self, plan: str, updated_plan: bool):
        title = "Initial Plan" if not updated_plan else "Updated Plan"
        self._console.print(
            Panel(
                Markdown(plan),
                title=f"[bold yellow]{title}[/bold yellow]",
                title_align="center",
                border_style="yellow",
            )
        )

    def at_reasoning(self, reasoning: str):
        self._console.print(
            Panel(
                Markdown(reasoning),
                title="[bold cyan]Reasoning[/bold cyan]",
                title_align="left",
                border_style="cyan",
            )
        )

    def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]):
        import json

        formatted_args = json.dumps(tool_arguments, indent=2)
        self._console.print(
            Panel(
                Syntax(formatted_args, "json", theme="monokai", word_wrap=True),
                title=f"[bold green]Tool Call: {tool_name}[/bold green]",
                title_align="left",
                border_style="green",
            )
        )

    def at_tool_result(self, tool_name: str, tool_result: str):
        # Try to parse as JSON to determine rendering format
        try:
            parsed_result = json.loads(tool_result)
            is_json = isinstance(parsed_result, (dict, list))
        except (json.JSONDecodeError, TypeError):
            is_json = False
            parsed_result = tool_result

        if is_json:
            display_result = json.dumps(parsed_result, indent=2)
            if self.truncate and len(display_result) > self.max_result_length:
                display_result = (
                    display_result[: self.max_result_length] + "\n... [truncated]"
                )
            content = Syntax(display_result, "json", theme="monokai", word_wrap=True)
        else:
            display_result = tool_result
            if self.truncate and len(display_result) > self.max_result_length:
                display_result = (
                    display_result[: self.max_result_length] + "\n... [truncated]"
                )
            content = Markdown(display_result)

        self._console.print(
            Panel(
                content,
                title=f"[bold magenta]Tool Result: {tool_name}[/bold magenta]",
                title_align="left",
                border_style="magenta",
            )
        )

    def at_warning(self, message: str):
        self._console.print(
            Panel(
                message,
                title="[bold red]Warning[/bold red]",
                title_align="left",
                border_style="red",
            )
        )
