import weave
from dotenv import load_dotenv

from kawai import KawaiLoggingCallback, KawaiReactAgent, OpenAIModel, WebSearchTool

load_dotenv()
weave.init(project_name="kawai")


agent = KawaiReactAgent(
    model=OpenAIModel(
        model_id="google/gemini-3-flash-preview",
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
    ),
    tools=[WebSearchTool()],
    planning_interval=3,
    instructions="Focus on finding sources from 2026.",
    callbacks=[KawaiLoggingCallback()],
)
result = agent.run(
    prompt="Which tennis player won more Grand Slam titles, Henri Leconte or Jonathan Stark?"
)
