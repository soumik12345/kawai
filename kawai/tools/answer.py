from typing import Any

from kawai.tools.tool import KawaiTool, KawaiToolParameter


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


class UserInputTool(KawaiTool):
    tool_name: str = "user_input"
    description: str = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="question",
            tool_type="string",
            description="The question to ask the user",
        )
    ]

    def forward(self, question):
        user_input = input(f"{question} => Type your answer here:")
        return user_input
