"""
Enhanced Research Agent - Main entry point

This script provides an interactive interface to the Enhanced Research Agent,
which can search the web, browse webpages, execute code, and perform file operations.
"""

import os
import sys
import argparse  # Added for command-line argument handling
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

def create_agent():
    """Create and initialize a research agent with all tools"""    # Tool function mapping
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
        get_available_modules_declaration
    ]
    
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

def main():
    """Run the Research Agent in interactive mode or process inputs from a file"""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Enhanced Research Agent")
    parser.add_argument("-f", "--file", help="Path to a text file containing user inputs, one per line")
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
    
    print("Initializing Enhanced Research Agent...")
    agent = create_agent()
    
    print("\n" + "=" * 60)
    print("Welcome to the Enhanced Research Agent!")
    print("This agent can search the web, browse webpages, execute code,")
    print("read/write files, and handle data analysis tasks.")
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
