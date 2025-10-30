from agents import Agent, Runner
import asyncio
from dotenv import load_dotenv

load_dotenv()

## What I learnt: 
# 1. Agents can be specialized for different tasks (in this case language responses)
# 2. The Triage agent can route requests to the appropriate language-specific agent.
# 3. There are some issues with an agent this simple and without guardrails:
#    - When providing English text, the triage didn't pass to the English agent, rather responded itself.
#    - When speaking a non specified language (e.g. French), the English agent was used without any acknowledgement of the French input.. 

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Bonjour! Comment ça va?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())