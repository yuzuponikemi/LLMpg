"""
Test script for the agent's eval mode

This script demonstrates how to use the agent's eval mode for automated testing.
It sends a test prompt and captures the first response for evaluation.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

# Add parent directory to Python path to import from enhanced-research-agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_agent_eval(query, output_dir=None):
    """
    Run the agent in eval mode with the given query and save the results
    
    Args:
        query (str): The query to test
        output_dir (str, optional): Directory to save results, defaults to 'test_results'
        
    Returns:
        dict: The evaluation result containing the query and response
    """
    # Set default output directory if not provided
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamped filename for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"agent_eval_{timestamp}.json")
    
    # Construct the command to run the agent in eval mode
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    command = [
        sys.executable, 
        script_path, 
        "--eval",
        "--query", query,
        "--output", output_file
    ]
    
    # Run the agent process
    print(f"Running agent with query: {query}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running agent: {result.stderr}")
        return None
    
    # Read the results from the output file
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            eval_result = json.load(f)
        
        print(f"Results saved to: {output_file}")
        return eval_result
    except Exception as e:
        print(f"Error reading evaluation results: {e}")
        return None

def run_batch_eval(query_file, output_dir=None):
    """
    Run eval mode for multiple queries from a file
    
    Args:
        query_file (str): Path to a file containing one query per line
        output_dir (str, optional): Directory to save results
    
    Returns:
        list: List of evaluation results
    """
    # Read queries from the file
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading query file: {e}")
        return []
    
    # Run eval for each query
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\nRunning query {i}/{len(queries)}:")
        result = run_agent_eval(query, output_dir)
        if result:
            results.append(result)
    
    # Save combined results
    if results and output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = os.path.join(output_dir, f"batch_eval_{timestamp}.json")
        
        try:
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nCombined results saved to: {combined_file}")
        except Exception as e:
            print(f"Error saving combined results: {e}")
    
    return results

def main():
    """Parse command line arguments and run evaluation"""
    parser = argparse.ArgumentParser(description="Test the agent's eval mode")
    parser.add_argument("-q", "--query", help="Single query to test")
    parser.add_argument("-f", "--file", help="File containing multiple queries, one per line")
    parser.add_argument("-o", "--output-dir", help="Directory to save results")
    args = parser.parse_args()
    
    if not args.query and not args.file:
        parser.error("Either --query or --file must be provided")
    
    if args.file:
        print(f"Running batch evaluation from file: {args.file}")
        results = run_batch_eval(args.file, args.output_dir)
        print(f"\nProcessed {len(results)} queries successfully")
    else:
        print(f"Running single evaluation for query: {args.query}")
        result = run_agent_eval(args.query, args.output_dir)
        if result:
            print("\nEvaluation Result:")
            print(f"Query: {result['query']}")
            print(f"Response: {result['response']}")

if __name__ == "__main__":
    main()
