"""
Test script to validate handling of multiple function calls in a single LLM response.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core import ResearchAgent
from src.tools.execution import get_tool_functions_and_declarations
from src.logger.agent_logger import setup_logger

# Setup logger
logger = setup_logger("test_multiple_function_calls")

def main():
    """Run the test to validate handling of multiple function calls."""
    logger.info("Starting multiple function calls test")
    
    # Load environment variables
    load_dotenv()
    
    # Get tool functions and declarations
    tool_functions, function_declarations = get_tool_functions_and_declarations()
    
    # Create agent
    agent = ResearchAgent(tool_functions, function_declarations)
    
    # Test query designed to trigger multiple function calls
    test_query = "Compare the population of Kyoto and Tokyo, and determine which city is larger."
    
    logger.info(f"Test query: '{test_query}'")
    logger.info("This query is designed to trigger multiple function calls (searching for both cities)")
    
    # Process the query
    start_time = time.time()
    response = agent.process_query(test_query)
    end_time = time.time()
    
    # Log the result
    logger.info(f"Query processed in {end_time - start_time:.2f} seconds")
    logger.info(f"Response length: {len(response)}")
    
    print("\n--- Test Result ---")
    print(response)
    print("\n--- End Result ---")
    
    logger.info("Multiple function calls test completed")

if __name__ == "__main__":
    main()
