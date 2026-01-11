from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule


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


class KawaiLoggingCallback(KawaiCallback):
    _console: Console | None = None

    def __init__(self) -> None:
        super().__init__()
        self._console = Console()

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
                Markdown(answer),
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
