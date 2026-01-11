import rich
import weave
from dotenv import load_dotenv

from kawai import ReactAgent, KawaiTool, KawaiToolParameter

load_dotenv()
weave.init(project_name="kawai")


class HoroscopeTool(KawaiTool):
    tool_name: str = "get_horoscope"
    description: str = "Get today's horoscope for an astrological sign."
    parameters: list[KawaiToolParameter] = [
        KawaiToolParameter(
            param_name="sign",
            tool_type="string",
            description="An astrological sign like Taurus or Aquarius",
        )
    ]

    def forward(self, sign: str) -> dict[str, str]:
        return {"response": f"{sign}: Next Tuesday you will befriend a baby otter."}


agent = ReactAgent(
    model="openai/gpt-5",
    tools=[HoroscopeTool()],
    system_prompt="You are a helpful agent.",
)
result = weave.op(agent.run)(prompt="What is my horoscope? I am an Aquarius.")
rich.print("\n" + "=" * 50)
rich.print("RESULT:")
rich.print(result)
rich.print("=" * 50)
