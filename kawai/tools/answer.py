from typing import Any

import weave

from kawai.tools.tool import KawaiTool, KawaiToolParameter


class FinalAnswerTool(KawaiTool):
    """Tool for providing the final answer to complete a task.

    This is a special tool that MUST be called by the agent to complete any task.
    The agent will loop until either this tool is called or max_steps is reached.
    It's automatically added to every agent if not explicitly provided.

    The tool accepts any type of answer (string, number, dict, list, etc.) and
    returns it unchanged (passthrough behavior). The agent extracts this value
    as the final result.

    Attributes:
        tool_name: Always "final_answer"
        description: Brief description shown to the agent
        parameters: Single parameter "answer" of type "any"

    Example:
        ```python
        # Agent calls this when ready to complete task
        final_answer(answer="The population of Shanghai is 26 million")
        # Or with structured data
        final_answer(answer={"city": "Shanghai", "population": 26000000})
        ```

    Note:
        - This tool is required for task completion
        - The agent cannot complete a task without calling this
        - Automatically added to agent.tools if not present
    """

    tool_name: str = "final_answer"
    description: str = "Provides a final answer to the given problem."
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="answer",
            tool_type="any",
            description="The final answer to the problem",
        )
    ]

    @weave.op
    def forward(self, answer: Any) -> Any:
        """Return the answer unchanged.

        Args:
            answer: The final answer to the task. Can be any JSON-serializable type.

        Returns:
            The answer exactly as provided (passthrough).
        """
        return answer


class UserInputTool(KawaiTool):
    """Tool for prompting the user for interactive input during agent execution.

    This tool allows the agent to ask clarifying questions or gather additional
    information from the user via command-line input. Useful for interactive
    agents that need user decisions or information that can't be obtained
    through other means.

    Attributes:
        tool_name: Always "user_input"
        description: Brief description shown to the agent
        parameters: Single parameter "question" of type "string"

    Example:
        ```python
        # Agent usage in a task
        agent = KawaiReactAgent(
            model="openai/gpt-4",
            tools=[UserInputTool()],
            max_steps=10
        )

        # The agent might call this when it needs clarification:
        # user_input(question="What year should I filter the search results by?")
        # User then types their response at the prompt
        ```

    Note:
        - This tool blocks execution waiting for user input
        - Input is collected via the command line (stdin)
        - Returns the raw string entered by the user
    """

    tool_name: str = "user_input"
    description: str = "Asks for user's input on a specific question"
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="question",
            tool_type="string",
            description="The question to ask the user",
        )
    ]

    @weave.op
    def forward(self, question: str) -> str:
        """Prompt the user for input and return their response.

        Args:
            question: The question or prompt to display to the user.

        Returns:
            The user's input as a string.
        """
        user_input = input(f"{question} => Type your answer here:")
        return user_input
