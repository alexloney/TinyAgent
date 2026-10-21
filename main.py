from agent import TinyAgent

from llm import LLM
llm = LLM()
response = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
print(response)
