from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()

#  Ran this three times and using the OpenAI API logs (Traces), I can see the three responses with different output.

agent = Agent(name="Assistant", instructions="You are a helpful assistant that will always respond as if you were a pirate.")
result = Runner.run_sync(agent, "Write a short poem about Ben Banurji experimenting with the OpenAI API.")
print(result.final_output)