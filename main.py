from agent import TinyAgent
from llm import LLM
from memory import Memory, SummarizationMemory

if False:
    llm = LLM()

    # Add memory to the Agent
    memory = Memory()
    agent_with_memory = TinyAgent(llm=llm, memory=memory)

    first_response = agent_with_memory.run("Hi! We are maarten and Jay, authors of 'An illustrated Guide to AI Agents'.")

    second_response = agent_with_memory.run("Hi! What are our names?")
    print(second_response)

    print(agent_with_memory.memory.get_messages())

llm = LLM()
memory = SummarizationMemory(llm)
agent = TinyAgent(llm=llm, memory=memory)

response = agent.run("Hi! my name is Sarah. Why are flamingos pink?")
print(response)
print(agent.memory.get_messages())
