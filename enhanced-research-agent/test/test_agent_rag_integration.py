"""
Agent RAG Integration Test

This script tests the RAG capabilities through direct interaction with the agent in eval mode.
It uses the agent's eval mode to send queries and capture the first response.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

# Import agent and necessary components
try:
    from src.logger.agent_logger import setup_logger
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    sys.exit(1)

# Setup logger for this module
logger = setup_logger(logger_name="agent_rag_test")

class AgentRAGTester:
    """Class to test RAG capabilities via the agent"""
    
    def __init__(self):
        """Initialize the tester"""
        print("Initializing RAG Tester...")
        
        # Define test data directory
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "rag_test_data")
        
        # Ensure test data exists
        if not os.path.exists(self.test_data_dir) or not os.listdir(self.test_data_dir):
            print(f"Error: Test data directory is empty or doesn't exist: {self.test_data_dir}")
            print("Please ensure you have the test data files before running this test.")
            sys.exit(1)
            
        print(f"Found test data in: {self.test_data_dir}")
        
        # Collection name for this test
        self.collection_name = "agent_rag_test"
        
        # Path to main.py
        self.main_script = os.path.join(os.path.dirname(__file__), "..", "main.py")
        
        # Directory for storing results
        self.results_dir = os.path.join(os.path.dirname(__file__), "test_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_eval_query(self, query, description=""):
        """
        Run a query in eval mode and return the response
        
        Args:
            query (str): The query to send to the agent
            description (str, optional): Description of the query for logging
        
        Returns:
            dict: The evaluation result containing the query and response
        """
        print(f"\n>>> {description}" if description else f"\n>>> Query: {query}")
        print("-" * 50)
        
        # Generate timestamp for the output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.results_dir, f"rag_eval_{timestamp}.json")
        
        # Construct the command to run the agent in eval mode
        command = [
            sys.executable,
            self.main_script,
            "--eval",
            "--query", query,
            "--output", output_file
        ]
        
        # Run the agent process
        try:
            print(f"Running agent in eval mode...")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running agent: {result.stderr}")
                return {"query": query, "response": f"Error: {result.stderr}", "success": False}
            
            # Read the results from the output file
            with open(output_file, 'r', encoding='utf-8') as f:
                eval_result = json.load(f)
            
            print(f"Response received and saved to: {output_file}")
            
            # Add success flag based on response length
            eval_result["success"] = len(eval_result.get("response", "")) > 100
            
            return eval_result
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {"query": query, "response": f"Error: {e}", "success": False}
    
    def run_indexing_test(self):
        """Test indexing documents with the agent"""
        print("\nTest 1: Indexing Documents")
        print("=" * 50)
        
        # Clean absolute path
        abs_path = os.path.abspath(self.test_data_dir)
        
        # Build the indexing query
        query = f"Please index the documents in the directory '{abs_path}' into a collection called '{self.collection_name}' using Google embeddings."
        
        result = self.run_eval_query(query, "Indexing test documents with Google embeddings")
        
        # Check if indexing was successful based on the response
        success = ("indexed successfully" in result.get("response", "").lower() or 
                  "created successfully" in result.get("response", "").lower())
        
        result["success"] = success
        return result
    
    def run_query_tests(self):
        """Test querying the indexed collection"""
        print("\nTest 2: Querying Indexed Collection")
        print("=" * 50)
        
        # Test queries
        queries = [
            {
                "query": f"Using the '{self.collection_name}' collection, what are the different types of machine learning?",
                "description": "Query about machine learning types"
            },
            {
                "query": f"Using the '{self.collection_name}' collection, explain the concept of superposition in quantum computing.",
                "description": "Query about quantum computing concepts" 
            },
            {
                "query": f"Using the '{self.collection_name}' collection, what are some applications of natural language processing?",
                "description": "Query about NLP applications"
            }
        ]
        
        results = []
        for test in queries:
            result = self.run_eval_query(test["query"], test["description"])
            results.append(result)
            time.sleep(1)  # Small delay between queries
            
        return results
    
    def run_all_tests(self):
        """Run all tests and return results"""
        print("\n" + "=" * 60)
        print("AGENT RAG INTEGRATION TEST (EVAL MODE)")
        print("=" * 60)
        
        results = {
            "indexing": None,
            "querying": None,
            "overall_success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test 1: Indexing
        results["indexing"] = self.run_indexing_test()
        
        # Only proceed with query tests if indexing was successful
        if results["indexing"]["success"]:
            # Test 2: Querying
            results["querying"] = self.run_query_tests()
            
            # Check overall success
            query_success_count = sum(1 for q in results["querying"] if q.get("success", False))
            results["overall_success"] = query_success_count >= 2  # At least 2 successful queries
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Indexing: {'✓ Success' if results['indexing']['success'] else '✗ Failed'}")
        
        if results["querying"]:
            print(f"Querying: {sum(1 for q in results['querying'] if q.get('success', False))}/{len(results['querying'])} successful")
        else:
            print("Querying: Not attempted (indexing failed)")
            
        print(f"Overall: {'✓ Success' if results['overall_success'] else '✗ Failed'}")
        print("=" * 60)
        
        return results

def main():
    """Main function to run the tests"""
    tester = AgentRAGTester()
    results = tester.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(tester.results_dir, f"agent_rag_eval_test_{timestamp}.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTest results saved to: {results_file}")
    
    # Return exit code based on overall success
    sys.exit(0 if results["overall_success"] else 1)

if __name__ == "__main__":
    main()
