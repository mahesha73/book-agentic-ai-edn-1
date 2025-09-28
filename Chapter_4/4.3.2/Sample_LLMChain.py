# Sample_LLMChain.py
# Section 4.3.2
# Page 102

from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=prompt)

output = chain.run(
    {
        "input": "Explain photosynthesis",
        "chat_history": "User asked about plant biology."
    }
)

print(output)
