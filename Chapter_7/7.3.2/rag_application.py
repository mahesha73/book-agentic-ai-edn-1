# rag_application.py
# Section 7.3.2
# Page 177

import os
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

# Wrap OpenAI client for automatic LLM tracing
client = wrap_openai(OpenAI())

@traceable(name="Retrieve-Documents")
def retrieve_documents(query: str) -> str:
    """Simulates retrieving relevant documents for a query."""
    print(f"Retrieving documents for: {query}")

    # In production, this would query a vector database
    return "Quantum computing leverages quantum-mechanical phenomena for computation."

@traceable(name="RAG-Chain")
def run_rag_chain(query: str):
    """Executes RAG process: retrieve documents then generate answer."""
    print("Starting RAG chain...")
    context = retrieve_documents(query)
    prompt = f"""Based on the following context, answer the user’s query.
        Context: {context}
        Query: {query}
        """

    # LLM call automatically traced
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    print(f"Generated Answer: {answer}")
    return answer

if __name__ == "__main__":
    user_query = "What is quantum computing?"
    run_rag_chain(user_query)
