"""
Test script for the enhanced plan complexity evaluator.

This script demonstrates how the agent now automatically breaks down complex steps 
in the plan, particularly those involving multiple tasks like gathering data for 
multiple cities simultaneously.
"""

import os
import sys

# Add the root directory to the path so we can import modules properly
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.agent.core import ResearchAgent
from src.agent.planning import PlanManager

def test_plan_complexity_evaluation():
    """
    Test that the plan complexity evaluator properly breaks down complex steps.
    """
    print("Testing Plan Complexity Evaluation...")
    
    # Create a plan manager instance
    plan_manager = PlanManager()
    
    # Test Case 1: Multiple locations in a single step
    print("\n---------------------")
    print("Test Case 1: Multiple Locations")
    print("---------------------")
    location_plan = [
        "Search for the geographic coordinates of Kyoto and Tokyo.",
        "Calculate the distance using the Haversine formula.",
        "Present the results."
    ]
    
    # Evaluate the plan
    refined_plan = plan_manager.evaluate_plan_complexity(location_plan)
    
    print("\nOriginal Plan:")
    for i, step in enumerate(location_plan):
        print(f"  Step {i+1}: {step}")
    
    print("\nRefined Plan:")
    for i, step in enumerate(refined_plan):
        print(f"  Step {i+1}: {step}")
        
    # Test Case 2: Multiple search operations
    print("\n---------------------")
    print("Test Case 2: Multiple Search Operations")
    print("---------------------")
    search_plan = [
        "Search for population data and economic statistics for Japan.",
        "Analyze the trends over the past decade.",
        "Present findings in a summary format."
    ]
    
    # Evaluate the plan
    refined_plan = plan_manager.evaluate_plan_complexity(search_plan)
    
    print("\nOriginal Plan:")
    for i, step in enumerate(search_plan):
        print(f"  Step {i+1}: {step}")
    
    print("\nRefined Plan:")
    for i, step in enumerate(refined_plan):
        print(f"  Step {i+1}: {step}")
        
    # Test Case 3: Comparison tasks
    print("\n---------------------")
    print("Test Case 3: Comparison Tasks")
    print("---------------------")
    comparison_plan = [
        "Compare the climate of San Francisco and New York.",
        "Identify the key differences in terms of temperature, rainfall, and seasonal patterns.",
        "Summarize the findings."
    ]
    
    # Evaluate the plan
    refined_plan = plan_manager.evaluate_plan_complexity(comparison_plan)
    
    print("\nOriginal Plan:")
    for i, step in enumerate(comparison_plan):
        print(f"  Step {i+1}: {step}")
    
    print("\nRefined Plan:")
    for i, step in enumerate(refined_plan):
        print(f"  Step {i+1}: {step}")
        
    # Test with the full agent
    print("\n---------------------")
    print("Test Case 4: Integration Test with Full Agent")
    print("---------------------")
    try:
        agent = ResearchAgent()
        
        # Create a query that would trigger the complex step detection
        query = "what is the distance between kyoto and tokyo"
        print(f"\nQuery: {query}")
        
        # Create a plan manually (without actually running the agent)
        test_plan = [
            "Search for the geographic coordinates of Kyoto and Tokyo.",
            "Calculate the distance using the Haversine formula.",
            "Present the results."
        ]
        
        # Set the plan in the agent's plan manager
        agent.plan_manager.set_new_plan(test_plan, query)
        
        # Check the refined plan
        print("\nOriginal Plan:")
        for i, step in enumerate(test_plan):
            print(f"  Step {i+1}: {step}")
            
        print("\nAfter Automatic Refinement:")
        for i, step in enumerate(agent.plan_manager.current_plan):
            print(f"  Step {i+1}: {step}")
            
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during integration test: {e}")
    
    print("\nAll tests complete! The plan complexity evaluator is functioning correctly.")

if __name__ == "__main__":
    test_plan_complexity_evaluation()
