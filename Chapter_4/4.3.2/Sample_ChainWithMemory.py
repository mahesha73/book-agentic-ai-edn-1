# Sample_ChainWithMemory.py
# Section 4.3.2
# Page 103

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

output = chain.run({"input": "What is AI?"})
