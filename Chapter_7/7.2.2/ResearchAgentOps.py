# ResearchAgentOps.py
# Section 7.2.2
# Page 174

import agentops
from openai import OpenAI
import os

# 1. Initialize AgentOps with automatic instrumentation
agentops.init(os.environ.get("AGENTOPS_API_KEY"), tags=['research-agent-v1'])

# OpenAI client automatically instrumented
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Define agent using decorator
@agentops.agent(agent_id='research-summarizer-agent')
def research_agent(topic: str):
    """Agent that researches topics and provides summaries."""
    print(f"Agent starting research on: {topic}")

    # LLM call automatically captured
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": f"Please provide a concise summary of: {topic}"}
        ]
    )
    summary = response.choices[0].message.content
    print(f"Agent fi nished. Summary: {summary[:100]}...")

# 3. Record operation result
agentops.record(agentops.Event(result='Success', returns=summary))
if __name__ == "__main__":
    research_topic = "The impact of quantum computing on cryptography"
    research_agent(topic=research_topic)

# 4. End session
agentops.end_session(‘Success')
