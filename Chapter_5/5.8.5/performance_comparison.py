# performance_comparison.py
# Section 5.8.5
# Page 145

import time
from enhanced_agentic_rag import EnhancedAgenticRAG
from agentic_rag import answer_query  # From previous section

def compare_systems(test_queries: list):
    """
    Compare basic LlamaIndex implementation with Cohere-enhanced version.
    """
    print("Performance Comparison: Basic vs Enhanced RAG")
    print("=" * 60)

    # Initialize enhanced system
    enhanced_rag = EnhancedAgenticRAG()
    if not enhanced_rag.load_existing_index():
        enhanced_rag.load_and_index_documents()

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        # Basic LlamaIndex response
        start_time = time.time()
        basic_answer = answer_query(query)
        basic_time = time.time() - start_time

        # Enhanced response with Cohere
        start_time = time.time()
        enhanced_answer = enhanced_rag.query_with_reranking(query)
        enhanced_time = time.time() - start_time

        print(f"Basic Answer ({basic_time:.2f}s):")
        print(f"{basic_answer[:200]}...")
        print(f"\nEnhanced Answer ({enhanced_time:.2f}s):")
        print(f"{enhanced_answer[:200]}...")
        print("\n" + "="*60)

# Test queries for comparison
test_queries = [
    "What are the main benefits of using RAG systems?",
    "How does retrieval augmentation improve language models?",
    "What are the challenges in implementing agentic RAG?"
]

if __name__ == "__main__":
    compare_systems(test_queries)
