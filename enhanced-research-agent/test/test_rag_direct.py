"""
RAG Tools Direct Test Script

This script tests the RAG functionality directly without using the agent.
It creates a collection, indexes documents, and performs queries.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

# Import RAG tools
try:
    from src.tools.rag_indexer import get_file_content, chunk_text
    from src.tools.rag_google import GoogleEmbeddingRAGTool
    from src.tools.rag_openai import OpenAIEmbeddingRAGTool
    from src.tools.rag_base import BaseRAGTool
    
    rag_available = True
    print("RAG modules imported successfully.")
except ImportError as e:
    rag_available = False
    print(f"Error importing RAG modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

def test_google_rag():
    """Test Google Embedding RAG Tool"""
    print("\n=== Testing Google Embedding RAG Tool ===")
    
    # Initialize the RAG tool
    google_rag = GoogleEmbeddingRAGTool()
    
    # Test collection creation and indexing
    test_collection = "test_google_rag"
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    
    print(f"Indexing documents from {test_data_dir} into collection '{test_collection}'...")
    
    # Create and index the collection
    result = google_rag.index_collection(
        collection_name=test_collection,
        directory_path=test_data_dir,
        file_pattern="*.txt",
        chunk_size=500,
        chunk_overlap=50
    )
    
    print(f"Indexing result: {result}")
    
    # Test querying the collection
    queries = [
        "What are the different types of machine learning?",
        "Explain the concept of superposition in quantum computing.",
        "What are some applications of natural language processing?",
        "How do neural networks relate to machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = google_rag.query_collection(
            collection_name=test_collection,
            query=query,
            num_results=2
        )
        print(f"Results: {results[:500]}...")  # Print first 500 chars of results

def test_openai_rag():
    """Test OpenAI Embedding RAG Tool"""
    print("\n=== Testing OpenAI Embedding RAG Tool ===")
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Skipping OpenAI RAG test.")
        return
    
    # Initialize the RAG tool
    openai_rag = OpenAIEmbeddingRAGTool()
    
    # Test collection creation and indexing
    test_collection = "test_openai_rag"
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    
    print(f"Indexing documents from {test_data_dir} into collection '{test_collection}'...")
    
    # Create and index the collection
    result = openai_rag.index_collection(
        collection_name=test_collection,
        directory_path=test_data_dir,
        file_pattern="*.txt",
        chunk_size=500,
        chunk_overlap=50
    )
    
    print(f"Indexing result: {result}")
    
    # Test querying the collection
    queries = [
        "What are the different types of machine learning?",
        "Explain the concept of superposition in quantum computing.",
        "What are some applications of natural language processing?",
        "How do neural networks relate to machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = openai_rag.query_collection(
            collection_name=test_collection,
            query=query,
            num_results=2
        )
        print(f"Results: {results[:500]}...")  # Print first 500 chars of results

def main():
    """Main function to run tests"""
    print("=== RAG Direct Testing ===")
    
    # Test Google RAG
    try:
        test_google_rag()
    except Exception as e:
        print(f"Error during Google RAG test: {e}")
    
    # Test OpenAI RAG
    try:
        test_openai_rag()
    except Exception as e:
        print(f"Error during OpenAI RAG test: {e}")
    
    print("\n=== RAG Testing Complete ===")

if __name__ == "__main__":
    main()
