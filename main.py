from agent import TinyAgent
from llm import LLM
from memory import Memory, SummarizationMemory, RAGMemory
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

if False:
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

documents = [
    "Sarah works as a marine biologist studying coral reefs.",
    "Sarah lives in Lisbon, Portugal.",
    "Sarah's favorite hobby is rock climbing.",
    "Sarah favorite animal is flamingos.",
    "Sarah speaks fluent Spanish and Prtuguese.",
    "Ilse is a software engineer at a renewable energy startup.",
    "Ilse lives in Amsterdam, the Netherlands.",
    "Ilse plays the cello in a local string quartet.",
    "Ilse's favorite author is Brandon Sanderson.",
    "Ilse's favorite animal is dolphins."
]
embedding_model = EmbeddingModel()
memory = RAGMemory(documents=documents, embedding_model=embedding_model)

# Create the Agent and run a query
agent = TinyAgent(llm=LLM(), memory=memory)
response = agent.run("What is Sarah's favorite animal?")
print(response)
print(agent.memory.get_messages())
print("=" * 20)

response = agent.run("Where does Ilse live?")
print(response)
print(agent.memory.get_messages())
print("=" * 20)

response = agent.run("What is Sarah's favorite hobby?")
print(response)
print(agent.memory.get_messages())
print("=" * 20)

response = agent.run("What is Ilse's favorite animal?")
print(response)
print(agent.memory.get_messages())
print("=" * 20)
