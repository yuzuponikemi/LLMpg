"""
Test script for the Enhanced Research Agent
Runs a series of predetermined test cases to evaluate agent functionality
"""

import os
import sys
# Add the parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core import ResearchAgent
from src.tools.search import search_duckduckgo, search_duckduckgo_declaration
from src.tools.browse import browse_webpage, browse_webpage_declaration
from src.tools.execution import (
    execute_code, execute_code_declaration,
    read_file as read_file_tool, read_file_declaration,
    write_file, write_file_declaration,
    list_files, list_files_declaration
)

def create_test_agent():
    """Create and initialize a research agent for testing"""
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

def run_tests():
    """Run a series of tests on the Research Agent"""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Create sample data file if it doesn't exist
    if not os.path.exists("data/sample_data.csv"):
        # Create sample data file if it doesn't exist
        sample_data = """date,temperature,humidity,wind_speed,weather_condition
2025-04-20,22.5,65,12,Sunny
2025-04-21,18.7,72,15,Cloudy
2025-04-22,17.2,85,22,Rainy
2025-04-23,21.0,68,8,Partly Cloudy
2025-04-24,24.5,55,10,Sunny
2025-04-25,23.8,62,14,Sunny
2025-04-26,19.5,75,18,Cloudy
2025-04-27,16.8,88,25,Rainy
"""
        with open("data/sample_data.csv", "w") as f:
            f.write(sample_data)
    
    print("Creating research agent for testing...")
    agent = create_test_agent()
    
    print("\n===== TEST 1: SIMPLE CALCULATION =====")
    agent.process_conversation("Calculate the sum of numbers from 1 to 100")
    
    print("\n===== TEST 2: FILE OPERATIONS =====")
    agent.process_conversation("List the files in the data directory and tell me about any CSV files you find")
    
    print("\n===== TEST 3: DATA ANALYSIS =====")
    agent.process_conversation("Read the CSV file in data/sample_data.csv and calculate the average temperature")
    
    print("\n===== TEST 4: VISUALIZATION =====")
    agent.process_conversation("Create a visualization of the temperature and humidity data from sample_data.csv and save it as data/visualization.png")
    
    print("\n===== TEST 5: COMPLEX RESEARCH QUERY =====")
    agent.process_conversation("Explain how transformer models work in natural language processing")

if __name__ == "__main__":
    run_tests()
