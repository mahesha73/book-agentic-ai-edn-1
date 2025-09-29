# enhanced_agentic_rag.py
# Section 5.8.3
# Page 144

import os
from dotenv import load_dotenv
import cohere
from llama_index import (
    SimpleDirectoryReader,  # For reading documents from directory
    VectorStoreIndex,       # Vector-based index for semantic search
    LLMPredictor,           # Wrapper for LLM models
    ServiceContext,         # Context bundling services
    Document                # Document class for text processing
)
from llama_index.embeddings import CohereEmbedding
from llama_index.postprocessor import CohereRerank
from langchain.chat_models import ChatOpenAI
from typing import List

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client for additional operations
cohere_client = cohere.Client(COHERE_API_KEY)

class EnhancedAgenticRAG:
    """
    Enhanced Agentic RAG system using LlamaIndex
    with Cohere integration for improved embedding
    quality and retrieval precision.
    """

    def __init__(self):
        """Initialize the enhanced RAG system
           with Cohere integration."""
        # Initialize LLM predictor with OpenAI GPT-4
        self.llm_predictor = LLMPredictor(
            llm=ChatOpenAI(
                model_name="gpt-4",
                temperature=0,  # Deterministic responses
                openai_api_key=OPENAI_API_KEY
            )
        )

        # Initialize Cohere embedding model
        self.embed_model = CohereEmbedding(
            cohere_api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0",  # Latest Cohere
                                                embedding model
            input_type="search_document"      # Optimized for
                                                document search
        )

        # Create service context with enhanced components
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=self.llm_predictor,
            embed_model=self.embed_model
        )

        # Initialize Cohere reranker
        self.reranker = CohereRerank(
            api_key=COHERE_API_KEY,
            top_n=5,  # Return top 5 reranked results
            model="rerank-english-v2.0"  # Latest reranking model
        )

        self.index = None

    def load_and_index_documents(self, documents_path: str =
                                          "data/documents"):
        """
        Load documents from the specified path and
        create an enhanced index using Cohere embeddings.
        """
        print("Loading documents...")
        # Load documents using SimpleDirectoryReader
        documents = SimpleDirectoryReader(documents_path)
                    .load_data()

        print(f"Loaded {len(documents)} documents")
        print("Creating enhanced index with Cohere embeddings...")

        # Create vector store index with Cohere embeddings
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )

        # Save index for future use
        self.index.storage_context.persist(
              persist_dir="./enhanced_index")
        print("Enhanced index created and saved successfully!")

    def load_existing_index(self, index_path: str =
                            "./enhanced_index"):
        """Load an existing index from disk."""
        from llama_index import
                         load_index_from_storage, StorageContext

        try:
            # Load the storage context
            storage_context = StorageContext.from_defaults
                              (persist_dir=index_path)
            # Load the index
            self.index = load_index_from_storage(
                storage_context,
                service_context=self.service_context
            )
            print("Existing enhanced index loaded successfully!")
            return True
        except Exception as e:
            print(f"Could not load existing index: {e}")
            return False

    def query_with_reranking(self, query: str,
              similarity_top_k: int = 10) -> str:
        """
        Query the index with enhanced retrieval
        using Cohere reranking.

        Args:
            query: User's question
            similarity_top_k: Number of initial results
                              to retrieve before reranking

        Returns:
            Generated answer based on reranked,
            most relevant documents
        """
        if self.index is None:
            return "Index not loaded. Please load documents first."

        print(f"Processing query: {query}")

        # Create query engine with reranking
        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,  # Retrieve more
                                           candidates initially
            node_postprocessors=[self.reranker],  # Apply Cohere
                                                    reranking
            service_context=self.service_context
        )

        # Execute query with enhanced retrieval
        response = query_engine.query(query)

        return str(response)

    def get_retrieval_details(self, query: str,
                              similarity_top_k: int = 10):
        """
        Get detailed information about the retrieval
        process for analysis.
        """
        if self.index is None:
            return "Index not loaded. Please load documents first."

        # Create retriever for detailed analysis
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        )

        # Retrieve nodes before reranking
        nodes = retriever.retrieve(query)

        print(f"\nRetrieved {len(nodes)} documents before
                                             reranking:")
        for i, node in enumerate(nodes):
            print(f"{i+1}. Score: {node.score:.4f}")
            print(f"   Text preview: {node.text[:100]}...")
            print()

        return nodes

def main():
    """Main function demonstrating the enhanced
       agentic RAG system."""
    # Initialize the enhanced RAG system
    rag_system = EnhancedAgenticRAG()

    # Try to load existing index, otherwise create new one
    if not rag_system.load_existing_index():
        rag_system.load_and_index_documents()

    # Interactive query loop
    print("\n" + "="*50)
    print("Enhanced Agentic RAG System with Cohere Integration")
    print("="*50)
    print("Ask questions about your documents
          (type 'quit' to exit)")

    while True:
        user_query = input("\nEnter your question: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_query:
            continue

        # Get enhanced answer with reranking
        answer = rag_system.query_with_reranking(user_query)

        print(f"\nAnswer:\n{answer}")

        # Optional: Show retrieval details for analysis
        show_details = input("\nShow retrieval details? (y/n):
                             ").strip().lower()
        if show_details == 'y':
            rag_system.get_retrieval_details(user_query)

if __name__ == "__main__":
    main()
