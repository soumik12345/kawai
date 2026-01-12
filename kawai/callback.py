import json
from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax


class KawaiCallback(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def at_run_start(self, prompt: str, model: str):
        pass

    def at_run_end(self, answer: str):
        pass

    def at_step_start(self, step_index: int):
        pass

    def at_planning_end(self, plan: str, updated_plan: bool):
        pass

    def at_reasoning(self, reasoning: str):
        """Called when the agent produces reasoning/thinking content."""
        pass

    def at_tool_call(self, tool_name: str, tool_arguments: dict[str, Any]):
        """Called when the agent is about to execute a tool."""
        pass

    def at_tool_result(self, tool_name: str, tool_result: str):
        """Called after a tool has been executed with its result."""
        pass

    def at_warning(self, message: str):
        """Called when the agent encounters a warning condition."""
        pass


class KawaiLoggingCallback(KawaiCallback):
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
