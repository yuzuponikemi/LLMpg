"""
Consistency Test script for the Enhanced Research Agent
Runs a series of verifiable tests multiple times to evaluate consistency and reliability
"""

import os
import sys
import json
import re
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pathlib import Path
import argparse  # Added for command line argument parsing
import logging  # For detailed logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "consistency_test_debug.log"), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("consistency_test")

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

# Constants
NUM_REPETITIONS = 10  # Default number of times to run each test
TEST_RESULTS_DIR = "test_results"
SAMPLE_DATA_PATH = "data/sample_data.csv"
DATA_VERIFICATION_FILE = "data/verification_data.json"

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

def setup_test_environment():
    """Set up test environment with necessary files and directories"""
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    # Create sample data file if it doesn't exist
    if not os.path.exists(SAMPLE_DATA_PATH):
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
        with open(SAMPLE_DATA_PATH, "w") as f:
            f.write(sample_data)
    
    # Create verification data for test validation
    if not os.path.exists(DATA_VERIFICATION_FILE):
        verification_data = {
            "sum_1_to_100": 5050,
            "average_temperature": 20.5,  # (22.5+18.7+17.2+21.0+24.5+23.8+19.5+16.8)/8
            "csv_file_count": 1,
            "fibonacci_10th": 55,
            "prime_count_below_100": 25
        }
        with open(DATA_VERIFICATION_FILE, "w") as f:
            json.dump(verification_data, f, indent=2)

def extract_numeric_result(response):
    """Extract numeric result from agent's response text"""
    # Check if response is None or empty
    if response is None or not isinstance(response, str):
        logger.warning(f"Response is not valid: {type(response)}")
        return None
    
    logger.debug(f"Extracting numeric result from response: '{response}'")
    
    # First try to find a number following typical result indicators
    result_patterns = [
        r"result is (\d+\.?\d*)",
        r"equals (\d+\.?\d*)",
        r"= (\d+\.?\d*)",
        r"answer is (\d+\.?\d*)",
        r"is (\d+\.?\d*)",  # Added pattern for "is X"
        r"calculated (\d+\.?\d*)",
        r"sum.*?is (\d+\.?\d*)",  # Added pattern for "sum... is X"
        r"total.*?is (\d+\.?\d*)"  # Added pattern for "total... is X"
    ]
    for pattern in result_patterns:
        match = re.search(pattern, response.lower())
        if match:
            try:
                result = float(match.group(1))
                logger.info(f"Extracted number {result} using pattern '{pattern}'")
                return result
            except ValueError:
                logger.debug(f"Found match with pattern '{pattern}' but couldn't convert to float: {match.group(1)}")
                continue
    
    # As a fallback, look for numbers in the response
    numbers = re.findall(r"(\d+\.?\d*)", response)
    if numbers:
        logger.debug(f"Found raw numbers in response: {numbers}")
        # Try to get the most relevant number (usually the last one in the response)
        try:
            result = float(numbers[-1])
            logger.info(f"Extracted number {result} from all numbers (chose last one)")
            return result
        except ValueError:
            logger.debug(f"Couldn't convert last number to float: {numbers[-1]}")
            pass
    
    logger.warning("Failed to extract any numeric result from response")
    return None

def verify_result(response, expected, tolerance=0.01):
    """Verify if the response contains the expected result within tolerance"""
    logger.info(f"Verifying result. Expected: {expected} (type: {type(expected).__name__})")
    
    # Extract numeric result if we're expecting a number
    if isinstance(expected, (int, float)):
        extracted = extract_numeric_result(response)
        logger.info(f"Extracted value: {extracted} (type: {type(extracted).__name__ if extracted is not None else 'None'})")
        
        if extracted is not None:
            difference = abs(extracted - expected)
            within_tolerance = difference <= tolerance
            logger.info(f"Difference: {difference}, Tolerance: {tolerance}, Within tolerance: {within_tolerance}")
            return within_tolerance
        else:
            logger.warning("Could not extract a numeric value to compare with expected result")
            return False
    
    # For text results, check if the expected text is in the response
    elif isinstance(expected, str):
        is_match = expected.lower() in response.lower()
        logger.info(f"Text match result: {is_match}")
        return is_match
    
    # For lists/arrays
    elif isinstance(expected, list):
        # Check if all items in expected are in the response
        item_results = [(item, str(item).lower() in response.lower()) for item in expected]
        all_found = all(result for _, result in item_results)
        logger.info(f"List match results: {item_results}, All found: {all_found}")
        return all_found
    
    logger.warning(f"Unsupported expected result type: {type(expected).__name__}")
    return False

def run_test(agent, test_name, prompt, expected_result, tolerance=0.01, extract_func=None, num_repetitions=NUM_REPETITIONS):
    """Run a single test multiple times and collect results"""
    results = []
    start_time = time.time()
    
    logger.info(f"===== STARTING TEST: {test_name} =====")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Expected result: {expected_result} (type: {type(expected_result).__name__})")
    logger.info(f"Tolerance: {tolerance}")
    logger.info(f"Number of repetitions: {num_repetitions}")
    logger.info(f"Custom extractor: {'Yes' if extract_func else 'No'}")
    
    print(f"\n===== RUNNING TEST: {test_name} =====")
    for i in range(num_repetitions):
        logger.info(f"Starting repetition {i+1}/{num_repetitions}")
        print(f"  Repetition {i+1}/{num_repetitions}... ", end="", flush=True)
        
        # Run the test and capture the logs
        try:
            logger.debug(f"Sending prompt to agent: '{prompt}'")
            response = agent.process_conversation(prompt)
            logger.debug(f"Raw agent response: {type(response)} - '{response}'")
            
            # If response is None or empty, try to find the response in the logs
            if not response:
                logger.warning("Agent response is empty or None - attempting to extract from logs")
                # This is a fallback method - the agent might be logging its response instead of returning it
                log_paths = []
                for handler in logging.getLogger().handlers:
                    if hasattr(handler, 'baseFilename') and 'agent' in str(handler.baseFilename):
                        log_paths.append(handler.baseFilename)
                
                logger.debug(f"Found potential log paths: {log_paths}")
                
                # Also look in the known log locations
                agent_logs = [
                    os.path.join("logs", "agent.log"),
                    os.path.join("logs", "agent_default.log"),
                    os.path.join("examples", "logs", "agent.log"),
                    os.path.join("examples", "logs", "agent_default.log")
                ]
                
                # Try all potential log files
                all_logs = log_paths + agent_logs
                for log_path in all_logs:
                    if os.path.exists(log_path):
                        logger.debug(f"Searching in log file: {log_path}")
                        try:
                            with open(log_path, 'r') as f:
                                logs = f.read()
                                # Look for the response in the logs with different patterns
                                patterns = [
                                    r'AGENT RESPONSE: (.*?)(\n|$)',
                                    r'Agent: (.*?)(\n|$)',
                                    r'Agent response generated.*?:(.*?)(\n|$)'
                                ]
                                
                                for pattern in patterns:
                                    response_match = re.search(pattern, logs)
                                    if response_match:
                                        response = response_match.group(1).strip()
                                        logger.info(f"Extracted response from logs using pattern '{pattern}': '{response}'")
                                        break
                                
                                if response:  # If we found a response, stop searching
                                    break
                        except Exception as e:
                            logger.warning(f"Error reading log file {log_path}: {e}")
                            
        except Exception as e:
            error_msg = f"Error during test: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            response = f"ERROR: {str(e)}"
        
        # Make sure response is a string
        if response is None:
            logger.warning("Response is None, converting to empty string")
            response = ""
        elif not isinstance(response, str):
            logger.warning(f"Response is not a string: {type(response).__name__}, converting to string")
            response = str(response)
        
        logger.info(f"Final response text (first 100 chars): '{response[:100]}...' (length: {len(response)})")
            
        # Apply custom extractor if provided
        if extract_func is not None:
            logger.debug(f"Using custom extractor function: {extract_func.__name__}")
            extracted = extract_func(response)
        else:
            if isinstance(expected_result, (int, float)):
                logger.debug("Using default numeric extractor function")
                extracted = extract_numeric_result(response)
            else:
                logger.debug("No extraction needed for non-numeric expected result")
                extracted = None
        
        # Dump the complete response for debugging
        logger.debug(f"COMPLETE RESPONSE:\n{'-'*40}\n{response}\n{'-'*40}")
        
        # Verify the result
        success = verify_result(response, expected_result, tolerance)
        logger.info(f"Verification result: {'SUCCESS' if success else 'FAILURE'}")
        if not success:
            logger.warning(f"Test failed. Expected: {expected_result}, Extracted: {extracted}")
            # Try to show what might have gone wrong with the matching
            if isinstance(expected_result, (int, float)):
                numbers_in_response = re.findall(r"(\d+\.?\d*)", response)
                logger.debug(f"All numbers found in response: {numbers_in_response}")
                for pattern in [r"result is", r"answer is", r"sum.*is", r"is \d+"]:
                    matches = re.findall(f"{pattern}.*", response.lower())
                    if matches:
                        logger.debug(f"Context around potential answers ({pattern}): {matches}")
        else:
            logger.info(f"Test succeeded with extracted value: {extracted}")
        # Store the result
        results.append({
            "repetition": i+1,
            "success": success,
            "response": response,
            "extracted_value": extracted
        })
        
        print("✓" if success else "✗")
    
    elapsed_time = time.time() - start_time
    success_rate = (sum(1 for r in results if r["success"]) / num_repetitions) * 100
    
    # Summarize the test results
    summary = {
        "test_name": test_name,
        "prompt": prompt,
        "expected_result": expected_result,
        "success_rate": success_rate,
        "average_time": elapsed_time / num_repetitions,
        "total_time": elapsed_time,
        "results": results
    }
    
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Average Time: {elapsed_time / num_repetitions:.2f} seconds")
    print(f"  Total Time: {elapsed_time:.2f} seconds")
    
    return summary

def extract_file_count(response):
    """Extract the number of CSV files found in the response"""
    match = re.search(r"(\d+)\s+CSV file", response)
    if match:
        return int(match.group(1))
    return None

def get_available_tests():
    """Return a dictionary of available tests with their descriptions"""
    return {
        "simple_calculation": {
            "name": "SIMPLE_CALCULATION",
            "description": "Calculate the sum of numbers from 1 to 100",
            "prompt": "Calculate the sum of numbers from 1 to 100",
            "expected_key": "sum_1_to_100",
            "extract_func": None,
            "tolerance": 0.01
        },
        "file_operations": {
            "name": "FILE_OPERATIONS",
            "description": "List files in data directory and count CSV files",
            "prompt": "List the files in the data directory and tell me how many CSV files are there",
            "expected_key": "csv_file_count",
            "extract_func": extract_file_count,
            "tolerance": 0.01
        },
        "data_analysis": {
            "name": "DATA_ANALYSIS",
            "description": "Calculate average temperature from CSV data",
            "prompt": "Read the CSV file in data/sample_data.csv and calculate the average temperature",
            "expected_key": "average_temperature",
            "extract_func": None,
            "tolerance": 0.01
        },
        "fibonacci_calculation": {
            "name": "FIBONACCI_CALCULATION",
            "description": "Calculate the 10th Fibonacci number",
            "prompt": "Calculate the 10th Fibonacci number (starting with 1, 1 as the first two numbers)",
            "expected_key": "fibonacci_10th",
            "extract_func": None,
            "tolerance": 0.01
        },
        "prime_counting": {
            "name": "PRIME_COUNTING",
            "description": "Count prime numbers less than 100",
            "prompt": "Count the number of prime numbers less than 100",
            "expected_key": "prime_count_below_100",
            "extract_func": None,
            "tolerance": 0.01
        },
        "self_reflection": {
            "name": "SELF_REFLECTION",
            "description": "Find prime numbers with algorithm improvement through reflection",
            "prompt": "Find all prime numbers below 100, but please use an inefficient algorithm first then improve it based on your reflection",
            "expected_key": "prime_count_below_100",
            "extract_func": None,
            "tolerance": 0.01
        }
    }

def run_single_test(test_key, num_repetitions=NUM_REPETITIONS):
    """Run a single test by its key name"""
    # Setup
    setup_test_environment()
    
    # Load verification data
    with open(DATA_VERIFICATION_FILE, "r") as f:
        verification = json.load(f)
    
    print("Creating research agent for testing...")
    agent = create_test_agent()
    
    # Get available tests
    available_tests = get_available_tests()
    
    if test_key not in available_tests:
        print(f"Error: Test '{test_key}' not found.")
        print("Available tests:")
        for key, test in available_tests.items():
            print(f"  - {key}: {test['description']}")
        return None
    
    # Get the test configuration
    test_config = available_tests[test_key]
    
    # Run the single test
    result = run_test(
        agent=agent,
        test_name=test_config["name"],
        prompt=test_config["prompt"],
        expected_result=verification[test_config["expected_key"]],
        extract_func=test_config["extract_func"],
        tolerance=test_config["tolerance"],
        num_repetitions=num_repetitions
    )
    
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to JSON
    results_file = os.path.join(TEST_RESULTS_DIR, f"{test_key}_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)
    
    # Generate visualization for single test
    generate_single_test_visualization(result, test_key, timestamp)
    
    print(f"\nTest complete. Results saved to {results_file}")
    return result

def generate_single_test_visualization(result, test_key, timestamp):
    """Generate visualization for a single test result"""
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract repetition data
    repetitions = [r["repetition"] for r in result["results"]]
    success_values = [1 if r["success"] else 0 for r in result["results"]]
    
    # Bar chart for success rate by repetition
    bars = ax.bar(repetitions, success_values, color=['green' if s else 'red' for s in success_values])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Success (1=Pass, 0=Fail)')
    ax.set_xlabel('Repetition Number')
    ax.set_title(f'Test Results: {result["test_name"]} - Success Rate: {result["success_rate"]:.1f}%')
    ax.set_xticks(repetitions)
    
    # Add value labels on the bars
    for bar, success in zip(bars, success_values):
        label = "✓" if success else "✗"
        ax.text(bar.get_x() + bar.get_width()/2., 1.05,
                label, ha='center', va='bottom', fontsize=14)
    
    # Add info text
    info_text = f"Prompt: {result['prompt']}\n"
    info_text += f"Expected Result: {result['expected_result']}\n"
    info_text += f"Average Time: {result['average_time']:.2f}s\n"
    info_text += f"Total Time: {result['total_time']:.2f}s"
    
    plt.figtext(0.1, 0.01, info_text, wrap=True, horizontalalignment='left', fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save the figure
    output_file = os.path.join(TEST_RESULTS_DIR, f"{test_key}_results_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()

def run_all_consistency_tests(num_repetitions=NUM_REPETITIONS):
    """Run all consistency tests and generate report"""
    # Setup
    setup_test_environment()
    
    # Load verification data
    with open(DATA_VERIFICATION_FILE, "r") as f:
        verification = json.load(f)
    
    print("Creating research agent for testing...")
    agent = create_test_agent()
    
    # Initialize results list
    all_results = []
    
    # Get available tests
    available_tests = get_available_tests()
    
    # Run each test
    for test_key, test_config in available_tests.items():
        result = run_test(
            agent=agent,
            test_name=test_config["name"],
            prompt=test_config["prompt"],
            expected_result=verification[test_config["expected_key"]],
            extract_func=test_config["extract_func"],
            tolerance=test_config["tolerance"],
            num_repetitions=num_repetitions
        )
        all_results.append(result)
    
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results to JSON
    results_file = os.path.join(TEST_RESULTS_DIR, f"consistency_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualization
    generate_results_visualization(all_results, timestamp)
    
    print(f"\nConsistency test complete. Results saved to {results_file}")
    return all_results

def generate_results_visualization(results, timestamp):
    """Generate visualization of test results"""
    test_names = [r["test_name"] for r in results]
    success_rates = [r["success_rate"] for r in results]
    avg_times = [r["average_time"] for r in results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Bar chart for success rates
    bars = ax1.bar(test_names, success_rates, color='skyblue')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Consistency Test Success Rates')
    ax1.set_xticklabels(test_names, rotation=45, ha='right')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    # Bar chart for average times
    bars = ax2.bar(test_names, avg_times, color='lightgreen')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Average Response Time per Test')
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(TEST_RESULTS_DIR, f"consistency_results_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()

def list_available_tests():
    """List all available tests with their descriptions"""
    available_tests = get_available_tests()
    print("\nAvailable tests:")
    for key, test in available_tests.items():
        print(f"  - {key}: {test['description']}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run consistency tests for the Enhanced Research Agent")
    parser.add_argument(
        "--test", 
        type=str,
        help="Specific test to run. Use --list to see available tests."
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available tests"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=NUM_REPETITIONS,
        help=f"Number of times to repeat each test (default: {NUM_REPETITIONS})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TEST_RESULTS_DIR,
        help=f"Directory to save test results (default: {TEST_RESULTS_DIR})"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update constants based on arguments
    num_repetitions = args.repetitions
    TEST_RESULTS_DIR = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    # Handle command options
    if args.list:
        list_available_tests()
    elif args.test:
        # Run a specific test
        run_single_test(args.test, num_repetitions)
    else:
        # Run all tests
        run_all_consistency_tests(num_repetitions)
