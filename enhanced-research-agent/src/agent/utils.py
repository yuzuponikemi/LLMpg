"""
Utility functions for the research agent
"""
import re
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="agent_utils")

def get_text_response(response):
    """Extracts the text response from the Gemini API result."""
    if response and response.candidates and response.candidates[0].content:
        # Combine text parts, properly handling function calls/responses
        text_parts = []
        function_call_count = 0
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text'):
                text_parts.append(part.text)
            elif hasattr(part, 'function_call') and part.function_call:
                function_call_count += 1
            # Skip parts with function_call attributes instead of trying to convert them
            # This prevents the "Could not convert `part.function_call` to text" warning
        
        result = "".join(text_parts)
        logger.info(f"Extracted text response (length: {len(result)}, function_calls: {function_call_count})")
        return result
    
    logger.warning("Could not extract text from response: no valid content found")
    return None

def get_function_call(response):
    """Checks if the response contains a function call and returns it."""
    if response and response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    logger.info(f"Function call found: {part.function_call.name}")
                    return part.function_call
    
    logger.debug("No function call found in response")
    return None

def get_function_calls(response):
    """
    Checks if the response contains multiple function calls and returns them as a list.
    Use this when the model might return multiple function calls in one response.
    
    Returns:
        list: A list of function call objects, or empty list if none found
    """
    function_calls = []
    
    if response and response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
    
    if function_calls:
        logger.info(f"Found {len(function_calls)} function calls: {', '.join(fc.name for fc in function_calls)}")
    else:
        logger.debug("No function calls found in response")
        
    return function_calls

def parse_plan_from_text(text):
    """
    Attempts to parse a numbered list representing a plan from text.
    Enhanced to detect various plan formats and indicators.
    """
    if not text: 
        return None
        
    # Look for plan indicators in the text
    plan_indicators = [
        r"here(?:'|')?s my plan",
        r"step(?:-| )by(?:-| )step plan",
        r"multi(?:-| )step plan",
        r"detailed plan",
        r"research plan",
        r"action plan",
        r"planned approach",
        r"I(?:'|')ll follow these steps",
        r"I will follow these steps",
        r"plan to ",
        r"let(?:'|')?s break this down"
    ]
    
    has_plan_indicator = any(re.search(pattern, text, re.IGNORECASE) for pattern in plan_indicators)
    
    # First try: Look for lines starting with optional whitespace, number, dot, space, then captures the rest
    steps = re.findall(r"^\s*\d+\.?\s+(.*?)(?:\n|$)", text, re.MULTILINE)
    
    # If regular numbered list not found, try finding steps with "Step X:" format
    if not steps:
        steps = re.findall(r"(?:^|\n)\s*(?:Step|STEP)\s+\d+:?\s+(.*?)(?:\n|$)", text, re.MULTILINE)
    
    # Also check for **Step X:** markdown format
    if not steps:
        steps = re.findall(r"\*\*\s*(?:Step|STEP)\s+\d+:?\s*\*\*\s*(.*?)(?:\n|$)", text, re.MULTILINE)
    
    if steps:
        # Basic cleanup
        steps = [step.strip() for step in steps if step.strip()]
        
        # If we have at least 2 steps, it's likely a plan
        if len(steps) >= 2:
            return steps
        # If we have exactly 1 step but there's a plan indicator, accept it as a plan
        elif len(steps) == 1 and has_plan_indicator:
            return steps
            
    # Check for Markdown or other formats with numbered lists
    if "1." in text and "2." in text:
        # Try extracting by finding lines with 1. 2. etc. without requiring them to be at line start
        steps = re.findall(r"(?:^|\n)\s*\d+\.?\s+(.*?)(?:\n|$)", text, re.MULTILINE)
        if steps and len(steps) >= 2:
            return [step.strip() for step in steps if step.strip()]
    
    # Check if the text contains confirmation of executing steps
    if re.search(r"executing step|starting with step|begin with|now I will|let's start|first step", text, re.IGNORECASE):
        # Try one more time with a more lenient pattern
        steps = re.findall(r"\d+\.?\s+(.*?)(?:\n|$)", text)
        if steps and len(steps) >= 1:
            return [step.strip() for step in steps if step.strip()]
            
    return None
