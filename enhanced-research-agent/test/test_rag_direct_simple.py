"""
Simplified RAG Direct Test Script

This script tests the RAG functionality directly without using the agent or indexer.
It manually adds documents to a collection and then queries them.
"""

import os
import sys
import glob
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

# Import RAG tools
try:
    from src.tools.rag_google import GoogleEmbeddingRAGTool
    from src.tools.rag_openai import OpenAIEmbeddingRAGTool
    
    rag_available = True
    print("RAG modules imported successfully.")
except ImportError as e:
    rag_available = False
    print(f"Error importing RAG modules: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

def test_google_rag_manual():
    """Test Google Embedding RAG Tool with manual document adding"""
    print("\n=== Testing Google Embedding RAG Tool (Manual) ===")
    
    # Initialize the RAG tool
    test_collection = "test_google_rag_manual"
    google_rag = GoogleEmbeddingRAGTool(collection_name=test_collection)
    
    # Test data directory
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    
    # Add documents manually
    doc_count = 0
    for file_path in glob.glob(os.path.join(test_data_dir, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create metadata
        filename = os.path.basename(file_path)
        metadata = {
            "filename": filename,
            "source": file_path,
            "type": "text"
        }
        
        # Add to collection
        doc_id = google_rag.add_document(
            text=content,
            metadata=metadata
        )
        
        print(f"Added document '{filename}' with ID: {doc_id}")
        doc_count += 1
    
    print(f"Added {doc_count} documents to collection '{test_collection}'")
    
    # Test querying the collection
    queries = [
        "What are the different types of machine learning?",
        "Explain the concept of superposition in quantum computing.",
        "What are some applications of natural language processing?",
        "How do neural networks relate to machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = google_rag.query_rag(
            query=query,
            limit=2,
            score_threshold=0.5  # Lower threshold for testing
        )
        print(f"Results:\n{results}")

def test_openai_rag_manual():
    """Test OpenAI Embedding RAG Tool with manual document adding"""
    print("\n=== Testing OpenAI Embedding RAG Tool (Manual) ===")
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Skipping OpenAI RAG test.")
        return
    
    # Initialize the RAG tool
    test_collection = "test_openai_rag_manual"
    openai_rag = OpenAIEmbeddingRAGTool(collection_name=test_collection)
    
    # Test data directory
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    
    # Add documents manually
    doc_count = 0
    for file_path in glob.glob(os.path.join(test_data_dir, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Create metadata
        filename = os.path.basename(file_path)
        metadata = {
            "filename": filename,
            "source": file_path,
            "type": "text"
        }
        
        # Add to collection
        doc_id = openai_rag.add_document(
            text=content,
            metadata=metadata
        )
        
        print(f"Added document '{filename}' with ID: {doc_id}")
        doc_count += 1
    
    print(f"Added {doc_count} documents to collection '{test_collection}'")
    
    # Test querying the collection
    queries = [
        "What are the different types of machine learning?",
        "Explain the concept of superposition in quantum computing.",
        "What are some applications of natural language processing?",
        "How do neural networks relate to machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = openai_rag.query_rag(
            query=query,
            limit=2,
            score_threshold=0.5  # Lower threshold for testing
        )
        print(f"Results:\n{results}")

def main():
    """Main function to run tests"""
    print("=== RAG Direct Testing (Simplified) ===")
    
    # Test Google RAG
    try:
        test_google_rag_manual()
    except Exception as e:
        print(f"Error during Google RAG test: {e}")
    
    # Test OpenAI RAG
    try:
        test_openai_rag_manual()
    except Exception as e:
        print(f"Error during OpenAI RAG test: {e}")
    
    print("\n=== RAG Testing Complete ===")

if __name__ == "__main__":
    main()
