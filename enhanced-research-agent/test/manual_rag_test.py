"""
Manual RAG Testing Guide

This script provides step-by-step instructions for manually testing
the RAG capabilities through the agent interface.

It will prepare test data and generate queries for you to try.
"""

import os
import sys
import glob
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

def setup_test_environment():
    """Set up the test environment and verify everything is ready"""
    print("Setting up RAG test environment...")
    
    # Check for test data directory
    test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
    test_files = glob.glob(os.path.join(test_data_dir, "*.txt"))
    
    if not test_files:
        print(f"ERROR: No test files found in {test_data_dir}")
        print("Please run test_rag_direct.py first to create the test files.")
        return None
    
    print(f"Found {len(test_files)} test files in {test_data_dir}:")
    for file in test_files:
        print(f"  - {os.path.basename(file)}")
    
    # Get absolute path for the test data directory
    abs_path = os.path.abspath(test_data_dir)
    
    return {
        "test_data_dir": abs_path,
        "test_files": test_files,
        "collection_name": "manual_rag_test"
    }

def print_manual_test_steps(test_env):
    """Print instructions for manual testing"""
    if not test_env:
        return
    
    print("\n" + "=" * 60)
    print("MANUAL RAG TESTING GUIDE")
    print("=" * 60)
    
    print("\nThis guide will help you manually test the RAG capabilities of the Enhanced Research Agent.")
    print("You will need to run the agent and enter the provided test queries.")
    
    print("\n" + "-" * 60)
    print("TEST 1: Indexing Test Documents")
    print("-" * 60)
    
    # Generate indexing query
    indexing_query = f"Please index the documents in the directory '{test_env['test_data_dir']}' into a collection called '{test_env['collection_name']}' using Google embeddings."
    
    print("1. Run the agent:")
    print("   python main.py")
    print("\n2. When the agent starts, enter this query:")
    print(f"   {indexing_query}")
    print("\n3. The agent should index the documents and confirm success.")
    
    print("\n" + "-" * 60)
    print("TEST 2: Querying Test Collection")
    print("-" * 60)
    
    # Generate test queries
    test_queries = [
        f"Using the '{test_env['collection_name']}' collection, what are the different types of machine learning?",
        f"Using the '{test_env['collection_name']}' collection, explain the concept of superposition in quantum computing.",
        f"Using the '{test_env['collection_name']}' collection, what are some applications of natural language processing?"
    ]
    
    print("After successful indexing, try these queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}:")
        print(f"   {query}")
    
    # Create a query.txt file for easy copying
    query_file = os.path.join(os.path.dirname(__file__), "..", "query.txt")
    with open(query_file, 'w', encoding='utf-8') as f:
        f.write("# RAG Test Queries\n\n")
        f.write("## Indexing Query:\n")
        f.write(indexing_query + "\n\n")
        f.write("## Test Queries:\n")
        for i, query in enumerate(test_queries, 1):
            f.write(f"{i}. {query}\n\n")
    
    print("\n" + "-" * 60)
    print("QUERY REFERENCE")
    print("-" * 60)
    print(f"All test queries have been saved to query.txt for easy copying.")
    
    print("\n" + "=" * 60)
    print("TESTING NOTES")
    print("=" * 60)
    print("- The agent should successfully index all test documents")
    print("- When querying, relevant information should be retrieved from the indexed documents")
    print("- The agent should combine the retrieved information with its own knowledge")
    print("- Check that the correct document is retrieved for each query")
    print("=" * 60)

def main():
    """Main function to set up and guide manual testing"""
    test_env = setup_test_environment()
    if test_env:
        print_manual_test_steps(test_env)

if __name__ == "__main__":
    main()
