"""
Example script demonstrating how to use the RAG capabilities
of the Enhanced Research Agent.

This script:
1. Creates a Qdrant collection
2. Indexes documents from a directory
3. Performs semantic searches using both Google and OpenAI embeddings
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to Python path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.tools.rag_google import GoogleEmbeddingRAGTool
from src.tools.rag_openai import OpenAIEmbeddingRAGTool
from src.tools.rag_indexer import index_directory

def main():
    """Main function demonstrating RAG capabilities"""
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not google_api_key:
        print("GOOGLE_API_KEY not found in environment variables.")
        return
    
    # Define Qdrant connection parameters
    qdrant_host = "localhost"
    qdrant_port = 6333
    
    print("=" * 60)
    print("RAG Capabilities Demo")
    print("=" * 60)
    
    # Test directory to index
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # 1. Index documents using Google embeddings
    print("\n1. Indexing documents with Google embeddings...")
    collection_name = "example_collection_google"
    
    # Create the collection and index the documents
    result = index_directory(
        directory_path=data_dir,
        collection_name=collection_name,
        embedding_type="google",
        file_pattern="*.json",  # Index only JSON files
        recursive=True,
        host=qdrant_host,
        port=qdrant_port
    )
    
    print(f"Indexed {result['files_indexed']} files with {result['chunks_indexed']} chunks")
    
    # 2. Create a RAG tool for querying
    print("\n2. Creating Google embedding RAG tool...")
    google_rag = GoogleEmbeddingRAGTool(
        collection_name=collection_name,
        host=qdrant_host,
        port=qdrant_port
    )
    
    # 3. Perform a sample query
    print("\n3. Performing a sample query with Google embeddings...")
    query = "sample data structure and format"
    results = google_rag.query_rag(query=query, limit=3)
    print(f"\nQuery: {query}")
    print(f"\nResults:\n{results}")
    
    # 4. Index with OpenAI embeddings if API key is available
    if openai_api_key:
        print("\n4. Indexing documents with OpenAI embeddings...")
        collection_name_openai = "example_collection_openai"
        
        # Create the collection and index the documents
        result = index_directory(
            directory_path=data_dir,
            collection_name=collection_name_openai,
            embedding_type="openai",
            file_pattern="*.json",  # Index only JSON files
            recursive=True,
            host=qdrant_host,
            port=qdrant_port
        )
        
        print(f"Indexed {result['files_indexed']} files with {result['chunks_indexed']} chunks")
        
        # 5. Create OpenAI RAG tool and query
        print("\n5. Querying with OpenAI embeddings...")
        openai_rag = OpenAIEmbeddingRAGTool(
            collection_name=collection_name_openai,
            host=qdrant_host,
            port=qdrant_port
        )
        
        results_openai = openai_rag.query_rag(query=query, limit=3)
        print(f"\nQuery: {query}")
        print(f"\nResults:\n{results_openai}")
    else:
        print("\nOpenAI API key not found. Skipping OpenAI embedding tests.")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
