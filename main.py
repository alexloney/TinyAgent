from agent import TinyAgent
from llm import LLM
from memory import Memory, SummarizationMemory
from embedding_model import EmbeddingModel

if False:
    llm = LLM()

    # Add memory to the Agent
    memory = Memory()
    agent_with_memory = TinyAgent(llm=llm, memory=memory)

    first_response = agent_with_memory.run("Hi! We are maarten and Jay, authors of 'An illustrated Guide to AI Agents'.")

    second_response = agent_with_memory.run("Hi! What are our names?")
    print(second_response)

    print(agent_with_memory.memory.get_messages())

if False:
    llm = LLM()
    memory = SummarizationMemory(llm)
    agent = TinyAgent(llm=llm, memory=memory)

    response = agent.run("Hi! my name is Sarah. Why are flamingos pink?")
    print(response)
    print(agent.memory.get_messages())

embedding_model = EmbeddingModel()

embedding_a = embedding_model.embed("I love flamingos.")
embedding_b = embedding_model.embed("Dolphins use echolocation.")
embedding_c = embedding_model.embed("Flamingos are pink birds.")

dot_ab = sum(x * y for x, y in zip(embedding_a, embedding_b))
norm_a = sum(x * x for x in embedding_a) ** 0.5
norm_b = sum(x * x for x in embedding_b) ** 0.5
similarity_ab = dot_ab / (norm_a * norm_b)

dot_ac = sum(x * y for x, y in zip(embedding_a, embedding_c))
norm_c = sum(x * x for x in embedding_c) ** 0.5
similarity_ac = dot_ac / (norm_a * norm_c)

print("Similarity between A and B:", similarity_ab)
print("Similarity between A and C:", similarity_ac)
