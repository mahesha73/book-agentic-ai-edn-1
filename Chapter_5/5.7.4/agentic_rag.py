# agentic_rag.py
# Section 5.7.4
# Page 140

import os
from dotenv import load_dotenv
from llama_index import (
    SimpleDirectoryReader, # For reading documents from a directory
    GPTVectorStoreIndex,   # Vector-based index for semantic search
    LLMPredictor,          # Wrapper for LLM models
    ServiceContext         # Context bundling LLM predictor and other services
)
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM predictor with OpenAI GPT-4
llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        model_name="gpt-4",    # Using GPT-4 model
        temperature=0,         # Deterministic output
        openai_api_key=OPENAI_API_KEY
    )
)

# Create a service context with the LLM predictor
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Step 1: Data Ingestion - Load documents from the 'data/documents' directory
print("Loading documents...")
documents = SimpleDirectoryReader("data/documents").load_data()

# Step 2: Index Creation - Build a vector store index for semantic search
print("Building index...")
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# Save the index to disk for reuse
index.save_to_disk("index.json")
print("Index saved to disk.")

# Step 3 & 4: Query Handling and Response Generation
def answer_query(query: str) -> str:
    """
    Load the index from disk, query it with the user's question,
    and return the generated answer.
    """
    # Load the index
    index = GPTVectorStoreIndex.load_from_disk("index.json")
    # Query the index
    response = index.query(query)
    return response.response

# Main execution block
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    answer = answer_query(user_query)
    print("\nAnswer:\n", answer)
