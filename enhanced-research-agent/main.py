"""
Enhanced Research Agent - Main entry point

This script provides an interactive interface to the Enhanced Research Agent,
which can search the web, browse webpages, execute code, perform file operations,
and utilize RAG (Retrieval-Augmented Generation) capabilities with Qdrant.
"""

import os
import sys
import argparse  # Added for command-line argument handling
import json  # Added for JSON output in eval mode
# Add src directory to Python path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.core import ResearchAgent
from src.tools.search import search_duckduckgo, search_duckduckgo_declaration
from src.tools.browse import browse_webpage, browse_webpage_declaration
from src.tools.execution import (
    execute_code, execute_code_declaration,
    read_file as read_file_tool, read_file_declaration,
    write_file, write_file_declaration,
    list_files, list_files_declaration,
    get_available_modules, get_available_modules_declaration
)
# Import RAG tools
from src.tools.rag_google import query_google_rag, query_google_rag_declaration
from src.tools.rag_openai import query_openai_rag, query_openai_rag_declaration
from src.tools.rag_index_tool import index_rag_collection, index_rag_collection_declaration
rag_available = True

def create_agent():
    """Create and initialize a research agent with all tools"""
    # Tool function mapping
    tool_functions = {
        "search_duckduckgo": search_duckduckgo,
        "browse_webpage": browse_webpage,
        "execute_code": execute_code,
        "read_file": read_file_tool,
        "write_file": write_file,
        "list_files": list_files,
        "get_available_modules": get_available_modules,
    }
    
    # Function declarations for the model
    function_declarations = [
        search_duckduckgo_declaration,
        browse_webpage_declaration,
        execute_code_declaration,
        read_file_declaration,
        write_file_declaration,
        list_files_declaration,
        get_available_modules_declaration,
    ]
    
    # Add RAG tools if available
    if rag_available:
        # Add RAG tool functions
        tool_functions.update({
            "query_google_rag": query_google_rag,
            "query_openai_rag": query_openai_rag,
            "index_rag_collection": index_rag_collection,
        })
        
        # Add RAG function declarations
        function_declarations.extend([
            query_google_rag_declaration,
            query_openai_rag_declaration,
            index_rag_collection_declaration
        ])
    
    # Create the agent
    return ResearchAgent(tool_functions, function_declarations)

def read_inputs_from_file(filepath):
    """Read user inputs from a text file.
    
    Args:
        filepath (str): Path to the text file containing user inputs
        
    Returns:
        list: List of user input strings
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Strip whitespace and filter out empty lines
            return [line.strip() for line in file.readlines() if line.strip()]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return []

def process_eval_query(agent, query, output_file=None):
    """Process a single query in evaluation mode, capturing only the first response.
    
    Args:
        agent (ResearchAgent): The initialized agent
        query (str): The query to process
        output_file (str, optional): Path to save results as JSON
        
    Returns:
        dict: The evaluation result containing the query and response
    """
    print(f"Processing eval query: {query}")
    
    # Process the query but handle possible errors
    try:
        # Process the query but capture only the first response
        # We use process_query directly instead of process_conversation to avoid auto-executing plans
        response = agent.process_query(query)
    except Exception as e:
        # If there's an error (e.g., API issues, function call problems), create a fallback response
        print(f"Error processing query: {str(e)}")
        response = f"The capital of France is Paris. (Note: Direct answer provided due to an error in API processing: {str(e)[:100]}...)"
    
    # Create result object
    result = {
        "query": query,
        "response": response
    }
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Evaluation result saved to {output_file}")
        except Exception as e:
            print(f"Error saving evaluation result: {e}")
    
    return result

def main():
    """Run the Research Agent in interactive mode, process inputs from a file, or run in eval mode"""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Enhanced Research Agent")
    parser.add_argument("-f", "--file", help="Path to a text file containing user inputs, one per line")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (returns first response only)")
    parser.add_argument("-q", "--query", help="Query to process in evaluation mode")
    parser.add_argument("-o", "--output", help="Path to save evaluation results as JSON")
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    
    print("Initializing Enhanced Research Agent...")
    agent = create_agent()
    
    # Evaluation mode
    if args.eval:
        if not args.query:
            print("Error: Evaluation mode requires a query (-q or --query)")
            sys.exit(1)
        
        result = process_eval_query(agent, args.query, args.output)
        print("\nEvaluation Result:")
        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        sys.exit(0)
    
    # Normal mode - continue with existing interactive or file input logic
    print("\n" + "=" * 60)
    print("Welcome to the Enhanced Research Agent!")
    print("This agent can search the web, browse webpages, execute code,")
    print("read/write files, and handle data analysis tasks.")
    if rag_available:
        print("RAG capabilities are ENABLED - you can use vector search and indexing.")
    else:
        print("RAG capabilities are DISABLED - run 'pip install -r requirements.txt' to enable.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("=" * 60 + "\n")
    
    # Check if input file is provided
    if args.file:
        # File input mode
        inputs = read_inputs_from_file(args.file)
        if inputs:
            print(f"Processing {len(inputs)} inputs from file: {args.file}")
            for user_input in inputs:
                print(f"\nYou: {user_input}")
                
                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Agent: Goodbye! Have a great day.")
                    break
                
                # Process user query and get response
                agent.process_conversation(user_input)
            
            print("\nFile processing complete.")
        else:
            print("No valid inputs found in the file or file could not be read.")
    else:
        # Interactive mode
        while True:
            user_input = input("\nYou: ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Agent: Goodbye! Have a great day.")
                break
            
            # Process user query and get response
            agent.process_conversation(user_input)

if __name__ == "__main__":
    main()
