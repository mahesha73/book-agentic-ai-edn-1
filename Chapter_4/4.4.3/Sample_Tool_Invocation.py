# Sample_Tool_Invocation.py
# Section 4.4.3
# Page 111

from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[email_tool, search_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

response = agent.run("Please send an email to john@example.com about the meeting.")

print(response)
