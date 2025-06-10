"""
Test RAG capabilities using the agent's eval mode

This script tests the agent's RAG capabilities by:
1. First indexing the test documents into a collection
2. Then querying the collection using the agent in eval mode
3. Analyzing the responses to verify that the agent correctly retrieves and uses the information
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Add parent directory to Python path to import from enhanced-research-agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RAGEvalTester:
    """Class to test RAG capabilities using the agent's eval mode"""
    
    def __init__(self, collection_name="eval_rag_test"):
        """Initialize the tester with a collection name"""
        self.collection_name = collection_name
        
        # Path to the test data directory
        self.test_data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "rag_test_data"
        )
        
        # Path to the test results directory
        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_results"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Path to the main.py script
        self.main_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "main.py"
        )
        
        # Timestamp for the test run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_eval_query(self, query):
        """Run a query using the agent's eval mode"""
        output_file = os.path.join(
            self.results_dir,
            f"rag_eval_{self.timestamp}_{hash(query) % 10000}.json"
        )
        
        # Construct the command to run the agent in eval mode
        command = [
            sys.executable,
            self.main_script,
            "--eval",
            "--query", query,
            "--output", output_file
        ]
        
        # Run the command using os.system to capture output
        print(f"Running query: {query}")
        cmd_str = " ".join(f'"{x}"' if ' ' in x else x for x in command)
        exit_code = os.system(cmd_str)
        
        # Check if the command was successful
        if exit_code != 0:
            print(f"Error running eval query (exit code: {exit_code})")
            return {"query": query, "response": f"Error: Command failed with exit code {exit_code}"}
        
        # Read the results from the output file
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result
        except Exception as e:
            print(f"Error reading result file: {e}")
            return {"query": query, "response": f"Error: {str(e)}"}
    
    def index_test_data(self):
        """Index the test data into a collection"""
        # Clean absolute path
        abs_path = os.path.abspath(self.test_data_dir)
        
        # Create the query to index the test data
        query = f"Please index all documents in the directory '{abs_path}' into a collection called '{self.collection_name}' using Google embeddings."
        
        # Run the indexing query
        print("\n===== INDEXING TEST DATA =====")
        result = self.run_eval_query(query)
        
        # Check if indexing was successful
        success = any(phrase in result.get("response", "").lower() for phrase in 
                    ["indexed successfully", "created successfully", "collection created", "indexed documents", "indexing complete"])
        
        if not success:
            print("WARNING: Indexing may not have been successful. Check the response:")
            print(result.get("response", "No response"))
        else:
            print("Indexing appears to have been successful")
            
        return success
    
    def test_machine_learning_queries(self):
        """Test queries related to machine learning"""
        print("\n===== TESTING MACHINE LEARNING QUERIES =====")
        
        queries = [
            f"Using the '{self.collection_name}' collection, what are the different types of machine learning?",
            f"Using the '{self.collection_name}' collection, list at least 3 common machine learning algorithms."
        ]
        
        results = []
        for query in queries:
            result = self.run_eval_query(query)
            time.sleep(1)  # Small delay between queries
            results.append(result)
            
        return results
    
    def test_quantum_computing_queries(self):
        """Test queries related to quantum computing"""
        print("\n===== TESTING QUANTUM COMPUTING QUERIES =====")
        
        queries = [
            f"Using the '{self.collection_name}' collection, explain the concept of superposition in quantum computing.",
            f"Using the '{self.collection_name}' collection, what are the current challenges in quantum computing?"
        ]
        
        results = []
        for query in queries:
            result = self.run_eval_query(query)
            time.sleep(1)  # Small delay between queries
            results.append(result)
            
        return results
    
    def test_nlp_queries(self):
        """Test queries related to NLP"""
        print("\n===== TESTING NLP QUERIES =====")
        
        queries = [
            f"Using the '{self.collection_name}' collection, list at least 4 core NLP tasks.",
            f"Using the '{self.collection_name}' collection, what are the modern approaches to NLP?"
        ]
        
        results = []
        for query in queries:
            result = self.run_eval_query(query)
            time.sleep(1)  # Small delay between queries
            results.append(result)
            
        return results
    
    def evaluate_results(self, results, expected_content):
        """
        Evaluate the results against expected content
        
        Args:
            results: List of query results
            expected_content: List of strings that should appear in the responses
        
        Returns:
            dict: Evaluation results
        """
        evaluation = {
            "total_queries": len(results),
            "successful_queries": 0,
            "query_results": []
        }
        
        for i, result in enumerate(results):
            query = result.get("query", "Unknown query")
            response = result.get("response", "No response")
            
            # Check if the response contains any of the expected content
            found_content = [content for content in expected_content if content.lower() in response.lower()]
            success = len(found_content) > 0
            
            if success:
                evaluation["successful_queries"] += 1
                
            evaluation["query_results"].append({
                "query": query,
                "success": success,
                "found_content": found_content,
                "response_length": len(response)
            })
            
        evaluation["success_rate"] = evaluation["successful_queries"] / evaluation["total_queries"] if evaluation["total_queries"] > 0 else 0
        return evaluation
    
    def run_all_tests(self):
        """Run all RAG tests and evaluate the results"""
        # 1. First, index the test data
        indexing_success = self.index_test_data()
        
        # If indexing wasn't successful, still try the queries but warn about it
        if not indexing_success:
            print("WARNING: Indexing may not have been successful, but proceeding with queries")
        
        # Wait a moment after indexing
        time.sleep(2)
        
        # 2. Run all query tests
        ml_results = self.test_machine_learning_queries()
        qc_results = self.test_quantum_computing_queries()
        nlp_results = self.test_nlp_queries()
        
        # 3. Evaluate the results
        ml_evaluation = self.evaluate_results(ml_results, [
            "supervised learning", 
            "unsupervised learning", 
            "reinforcement learning",
            "linear regression",
            "logistic regression",
            "decision trees",
            "random forests",
            "support vector machines",
            "neural networks"
        ])
        
        qc_evaluation = self.evaluate_results(qc_results, [
            "superposition", 
            "entanglement",
            "qubits",
            "decoherence", 
            "error correction",
            "scalability"
        ])
        
        nlp_evaluation = self.evaluate_results(nlp_results, [
            "tokenization", 
            "part-of-speech tagging",
            "named entity recognition",
            "sentiment analysis",
            "machine translation",
            "question answering",
            "text summarization",
            "rule-based systems",
            "statistical methods",
            "deep learning",
            "transformer",
            "BERT",
            "GPT"
        ])
        
        # 4. Combine all results and evaluations
        all_results = ml_results + qc_results + nlp_results
        overall_evaluation = {
            "timestamp": self.timestamp,
            "collection_name": self.collection_name,
            "indexing_success": indexing_success,
            "machine_learning_evaluation": ml_evaluation,
            "quantum_computing_evaluation": qc_evaluation,
            "nlp_evaluation": nlp_evaluation,
            "overall_success_rate": (
                ml_evaluation["successful_queries"] + 
                qc_evaluation["successful_queries"] + 
                nlp_evaluation["successful_queries"]
            ) / len(all_results) if all_results else 0
        }
        
        # 5. Save the overall results
        results_file = os.path.join(self.results_dir, f"rag_eval_results_{self.timestamp}.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(overall_evaluation, f, indent=2)
            print(f"\nOverall evaluation results saved to: {results_file}")
        except Exception as e:
            print(f"Error saving overall evaluation results: {e}")
        
        # 6. Print summary
        print("\n===== TEST RESULTS SUMMARY =====")
        print(f"Total queries: {len(all_results)}")
        print(f"Machine Learning queries success rate: {ml_evaluation['success_rate']:.0%}")
        print(f"Quantum Computing queries success rate: {qc_evaluation['success_rate']:.0%}")
        print(f"NLP queries success rate: {nlp_evaluation['success_rate']:.0%}")
        print(f"Overall success rate: {overall_evaluation['overall_success_rate']:.0%}")
        
        return overall_evaluation

def main():
    """Parse command line arguments and run the tests"""
    parser = argparse.ArgumentParser(description="Test RAG capabilities using the agent's eval mode")
    parser.add_argument("--collection", default="eval_rag_test", help="Name for the test collection")
    args = parser.parse_args()
    
    # Run the tests
    tester = RAGEvalTester(collection_name=args.collection)
    tester.run_all_tests()

if __name__ == "__main__":
    main()
