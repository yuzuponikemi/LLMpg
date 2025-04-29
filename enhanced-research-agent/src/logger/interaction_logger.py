"""
Enhanced interaction logger for the Research Agent.
This module extends the base logger with specialized methods for logging agent interactions.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .agent_logger import setup_logger, default_logger

# Create a specialized logger for interactions
interaction_logger = setup_logger(logger_name="agent_interactions")

def log_user_query(query: str) -> None:
    """
    Log a user query with special formatting.
    
    Args:
        query: The user's query text
    """
    interaction_logger.info(f"USER QUERY: {query}")
    default_logger.info(f"Received user query: {query}")

def log_agent_response(response: str) -> None:
    """
    Log the agent's response to a user query.
    
    Args:
        response: The agent's response text
    """
    # Truncate very long responses for log readability
    log_response = response
    if len(response) > 1000:
        log_response = response[:997] + "..."
    
    interaction_logger.info(f"AGENT RESPONSE: {log_response}")
    default_logger.info(f"Generated response (length: {len(response)})")

def log_function_call(function_name: str, args: Dict[str, Any]) -> None:
    """
    Log a function call made by the agent.
    
    Args:
        function_name: The name of the called function
        args: The arguments passed to the function
    """
    # Filter out potentially large arguments
    filtered_args = args.copy()
    for key, value in filtered_args.items():
        if isinstance(value, str) and len(value) > 500:
            filtered_args[key] = f"[{len(value)} chars]"
    
    interaction_logger.info(f"FUNCTION CALL: {function_name} - Args: {filtered_args}")
    default_logger.info(f"Function call: {function_name}")

def log_function_result(function_name: str, result: Any) -> None:
    """
    Log the result of a function call.
    
    Args:
        function_name: The name of the called function
        result: The result returned by the function
    """
    # For handling different result types
    if isinstance(result, str):
        result_summary = result[:500] + "..." if len(result) > 500 else result
    elif isinstance(result, dict):
        result_summary = {k: f"[{len(str(v))} chars]" if isinstance(v, str) and len(str(v)) > 100 else v 
                          for k, v in list(result.items())[:5]}
        if len(result) > 5:
            result_summary["..."] = f"[{len(result) - 5} more keys]"
    elif isinstance(result, list):
        result_summary = f"[List with {len(result)} items]"
    else:
        result_summary = str(result)[:200]
    
    interaction_logger.info(f"FUNCTION RESULT: {function_name} - Result: {result_summary}")
    default_logger.info(f"Function completed: {function_name}")

def log_plan_created(plan_steps: List[str], goal: str) -> None:
    """
    Log when a new plan is created.
    
    Args:
        plan_steps: The list of steps in the plan
        goal: The original goal/query
    """
    interaction_logger.info(f"PLAN CREATED: For goal '{goal}' with {len(plan_steps)} steps")
    for i, step in enumerate(plan_steps):
        interaction_logger.info(f"  Step {i+1}: {step}")
    default_logger.info(f"New plan created with {len(plan_steps)} steps")

def log_plan_step_execution(step_number: int, total_steps: int, step_text: str) -> None:
    """
    Log the execution of a plan step.
    
    Args:
        step_number: The current step number (1-based)
        total_steps: The total number of steps in the plan
        step_text: The text of the step being executed
    """
    interaction_logger.info(f"EXECUTING STEP {step_number}/{total_steps}: {step_text}")
    default_logger.info(f"Executing plan step {step_number}/{total_steps}")

def log_plan_step_result(step_number: int, result: str) -> None:
    """
    Log the result of a plan step.
    
    Args:
        step_number: The step number that was executed (1-based)
        result: The result of the step execution
    """
    # Truncate very long responses for log readability
    log_result = result
    if len(result) > 1000:
        log_result = result[:997] + "..."
    
    interaction_logger.info(f"STEP {step_number} RESULT: {log_result}")
    default_logger.info(f"Completed plan step {step_number}")

def log_plan_completed(goal: str) -> None:
    """
    Log when a plan has been fully completed.
    
    Args:
        goal: The original goal/query that the plan addressed
    """
    interaction_logger.info(f"PLAN COMPLETED: For goal '{goal}'")
    default_logger.info(f"Plan completed for goal: {goal}")

def log_conversation_session_start() -> None:
    """Log the start of a new conversation session."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    interaction_logger.info(f"=== NEW SESSION: {timestamp} ===")
    default_logger.info("Started new conversation session")

def log_conversation_session_end() -> None:
    """Log the end of a conversation session."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    interaction_logger.info(f"=== SESSION ENDED: {timestamp} ===")
    default_logger.info("Ended conversation session")

def log_error(error_message: str, context: str = "general") -> None:
    """
    Log an error with context information.
    
    Args:
        error_message: The error message to log
        context: The context where the error occurred
    """
    try:
        interaction_logger.error(f"ERROR [{context}]: {error_message}")
        default_logger.error(f"Error in {context}: {error_message}")
    except Exception as e:
        # Fallback if there's an issue with the error logging itself
        interaction_logger.error(error_message)
        default_logger.error(error_message)

def log_reflection(goal: str, step: str, result: str, evaluation: Dict[str, Any]) -> None:
    """
    Log the agent's self-reflection on a step result.
    
    Args:
        goal: The original goal/query
        step: The step being evaluated
        result: The result being evaluated
        evaluation: The evaluation results
    """
    is_adequate = evaluation.get('is_adequate', False)
    recommendation = evaluation.get('recommendation', 'proceed')
    
    interaction_logger.info(f"REFLECTION: Step '{step[:50]}...' - {'ADEQUATE' if is_adequate else 'INADEQUATE'}")
    interaction_logger.info(f"  Recommendation: {recommendation.upper()}")
    
    if 'issues' in evaluation and evaluation['issues']:
        interaction_logger.info(f"  Issues: {evaluation['issues']}")
    
    if 'justification' in evaluation:
        interaction_logger.info(f"  Justification: {evaluation['justification'][:200]}")
    
    default_logger.info(f"Step reflection: {recommendation.upper()}")

def log_refinement(goal: str, original_step: str, result: str, refinement: Dict[str, str]) -> None:
    """
    Log the agent's refinement of a step or plan.
    
    Args:
        goal: The original goal/query
        original_step: The original step being refined
        result: The result that prompted refinement
        refinement: The refinement details
    """
    refined_step = refinement.get('refined_step', '')
    reasoning = refinement.get('reasoning', '')
    
    interaction_logger.info(f"REFINEMENT: From '{original_step[:50]}...' to '{refined_step[:50]}...'")
    interaction_logger.info(f"  Reasoning: {reasoning[:200]}")
    
    default_logger.info(f"Step refined: {original_step[:30]}... -> {refined_step[:30]}...")
