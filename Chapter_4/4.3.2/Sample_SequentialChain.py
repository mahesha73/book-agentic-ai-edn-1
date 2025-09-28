# Sample_SequentialChain.py
# Section 4.3.2
# Page 102

from langchain.chains import SequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
seq_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["input"],
    output_variables=["final_output"]
    )

result = seq_chain.run({"input": "Explain photosynthesis"})
