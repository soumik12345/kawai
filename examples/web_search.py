import weave
from dotenv import load_dotenv

from kawai import KawaiLoggingCallback, KawaiReactAgent, WebSearchTool

load_dotenv()
weave.init(project_name="kawai")


agent = KawaiReactAgent(
    model="google/gemini-3-flash-preview",
    tools=[WebSearchTool()],
    planning_interval=3,
    instructions="Focus on finding sources from 2026.",
    callbacks=[KawaiLoggingCallback()],
)
result = agent.run(prompt="Who is the finance minister of West Bengal?")
