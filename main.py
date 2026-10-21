from agent import TinyAgent

from llm import LLM
llm = LLM()
agent = TinyAgent(llm)
response = agent.run("Hi! How's life?")
print(response)
print(agent.trajectory.runs)
