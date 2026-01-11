import rich
import weave
from dotenv import load_dotenv

from kawai import KawaiReactAgent, WebSearchTool

load_dotenv()
weave.init(project_name="kawai")


agent = KawaiReactAgent(
    model="openai/gpt-5",
    tools=[WebSearchTool()],
)
result = agent.run(
    prompt="Who is the finance minister of West Bengal as of January 2026?"
)
rich.print("\n" + "=" * 50)
rich.print("RESULT:")
rich.print(result)
rich.print("=" * 50)
