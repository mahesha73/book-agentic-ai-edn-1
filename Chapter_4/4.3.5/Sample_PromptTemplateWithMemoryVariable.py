# Sample_PromptTemplateWithMemoryVariable.py
# Section 4.3.5
# Page 106

prompt = PromptTemplate(
    input_variables=["input", "history"],
    template="Conversation so far:\n{history}\nUser: {input}\nAssistant:"
)
