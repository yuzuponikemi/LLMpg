"""
Test script for the RAG tools (Google Embeddings)

This script tests the RAG tools directly, bypassing the agent.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

from src.tools.rag_google import GoogleEmbeddingRAGTool

def main():
    print("=== Testing RAG Tools with Google Embeddings ===")
    
    # Collection name for the test
    collection_name = "test_google_rag_tool"
    
    # Initialize RAG tool
    try:
        rag_tool = GoogleEmbeddingRAGTool(collection_name=collection_name)
        print(f"Initialized GoogleEmbeddingRAGTool with collection: {collection_name}")
    except Exception as e:
        print(f"Error initializing RAG tool: {e}")
        return
    
    # Test data
    documents = [
        {
            "text": """Machine learning is a field of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. 
                    Types include supervised learning, unsupervised learning, and reinforcement learning.""",
            "metadata": {"source": "machine_learning_doc", "type": "text"}
        },
        {
            "text": """Quantum computing uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. 
                    Unlike classical computers that use bits, quantum computers use qubits.""",
            "metadata": {"source": "quantum_computing_doc", "type": "text"}
        },
        {
            "text": """Natural Language Processing (NLP) is a field that focuses on the interaction between computers and human language. 
                    Common NLP tasks include sentiment analysis, named entity recognition, and machine translation.""",
            "metadata": {"source": "nlp_doc", "type": "text"}
        }
    ]
    
    # Add documents
    print("\nAdding documents to the collection...")
    for i, doc in enumerate(documents):
        try:
            doc_id = rag_tool.add_document(
                text=doc["text"],
                metadata=doc["metadata"]
            )
            print(f"Added document {i+1} with ID: {doc_id}")
        except Exception as e:
            print(f"Error adding document {i+1}: {e}")
    
    # Test queries
    queries = [
        "What are the different types of machine learning?",
        "What is superposition in quantum computing?",
        "What are some NLP tasks?",
        "How do quantum computers differ from classical computers?"
    ]
    
    print("\nTesting queries...")
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: {query}")
        try:
            results = rag_tool.query_rag(
                query=query,
                limit=2,
                score_threshold=0.5
            )
            print(f"Results:\n{results}")
        except Exception as e:
            print(f"Error querying: {e}")
    
    print("\n=== RAG Tool Testing Complete ===")

if __name__ == "__main__":
    main()
