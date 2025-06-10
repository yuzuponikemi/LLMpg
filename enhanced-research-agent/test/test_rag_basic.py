"""
Basic RAG Test with Qdrant Client

This script tests the RAG functionality using the Qdrant client directly.
"""

import os
import sys
import glob
import uuid
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import google.generativeai as genai

# Initialize the Google Generative AI SDK
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Google API key not found. Please set GOOGLE_API_KEY in the .env file.")
    sys.exit(1)

def embed_text(text, model_name="models/embedding-001"):
    """Generate embeddings using Google's model"""
    embedding = genai.embed_content(
        model=model_name,
        content=text,
        task_type="RETRIEVAL_QUERY",
        title="",
    )
    return embedding["embedding"]

def main():
    print("=== Basic RAG Test with Qdrant Client ===")
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    print("Connected to Qdrant")
    
    # Collection name
    collection_name = "test_basic_rag"
    
    # Check if collection exists and recreate it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if collection_name in collection_names:
        print(f"Recreating collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
    
    # Create collection
    print(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # Google's embedding-001 model produces 768-dim vectors
            distance=Distance.COSINE,  # Use the enum directly
        )
    )
    
    # Add documents
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    doc_count = 0
    
    for file_path in glob.glob(os.path.join(test_data_dir, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate embedding
        embedding = embed_text(content)
        
        # Create metadata
        filename = os.path.basename(file_path)
        metadata = {
            "filename": filename,
            "source": file_path,
            "text": content,  # Include text in payload
            "type": "text"
        }        # Add point to collection
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate a UUID
                    vector=embedding,
                    payload=metadata
                )
            ]
        )
        
        print(f"Added document '{filename}' to collection")
        doc_count += 1
    
    print(f"Added {doc_count} documents to collection '{collection_name}'")
    
    # Test querying
    queries = [
        "What are the different types of machine learning?",
        "Explain the concept of superposition in quantum computing.",
        "What are some applications of natural language processing?",
        "How do neural networks relate to machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Generate query embedding
        query_embedding = embed_text(query)
        
        # Search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=2,
            with_payload=True,
            score_threshold=0.5
        )
        
        # Display results
        print(f"Found {len(search_results)} results:")
        for i, result in enumerate(search_results):
            filename = result.payload.get("filename", "Unknown")
            score = result.score
            text = result.payload.get("text", "No text available")
            
            print(f"Result {i+1} - {filename} (Score: {score:.4f}):")
            print(f"Excerpt: {text[:200]}...\n")

if __name__ == "__main__":
    main()
