"""
Enhanced Research Agent - Main entry point

This script provides an interactive interface to the Enhanced Research Agent,
which can search the web, browse webpages, execute code, and perform file operations.
"""

import os
import sys
# Add src directory to Python path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.core import ResearchAgent
from src.tools.search import search_duckduckgo, search_duckduckgo_declaration
from src.tools.browse import browse_webpage, browse_webpage_declaration
from src.tools.execution import (
    execute_code, execute_code_declaration,
    read_file as read_file_tool, read_file_declaration,
    write_file, write_file_declaration,
    list_files, list_files_declaration
)

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
    }
    
    # Function declarations for the model
    function_declarations = [
        search_duckduckgo_declaration,
        browse_webpage_declaration,
        execute_code_declaration,
        read_file_declaration,
        write_file_declaration,
        list_files_declaration
    ]
    
    # Create the agent
    return ResearchAgent(tool_functions, function_declarations)

def main():
    """Run the Research Agent in interactive mode, taking user input"""
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
