"""
Model manager for the enhanced research agent.
Handles model configuration, selection and generation with reflection capabilities.
"""

import google.generativeai as genai
from google.generativeai.types import Tool
from typing import Dict, Any, List, Optional, Union

from src.logger.agent_logger import setup_logger


# Setup logger for this module
logger = setup_logger(logger_name="model_manager")

class ModelManager:
    """
    Manages LLM model configurations and provides refined generation capabilities.
    Supports model switching and optional reflection-based refinement of responses.
    """
    
    # Model tier configurations
    MODEL_TIERS = {
        "light": {
            "model_name": "gemini-1.5-flash",
            "generation_config": {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 32,
                "max_output_tokens": 2048,
            }
        },
        "standard": {
            "model_name": "gemini-2.0-flash",
            "generation_config": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        },
        "advanced": {
            "model_name": "gemini-2.5-pro-preview-03-25",
            "generation_config": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        }
    }
    
    def __init__(self, reflection_evaluator=None, function_declarations=None):
        """
        Initialize the model manager.
        
        Args:
            reflection_evaluator: Optional ReflectionEvaluator instance for response refinement
            function_declarations: Optional list of function declarations for tool usage
        """
        self.reflection_evaluator = reflection_evaluator
        self.function_declarations = function_declarations
        self.chat_sessions = {}  # Store chat sessions for different models
        self.active_model_tier = "standard"  # Default to standard model
        
        # Initialize default model and chat
        self._initialize_model(self.active_model_tier)
        
    def _initialize_model(self, model_tier):
        """Initialize a model based on the specified tier"""
        if model_tier not in self.MODEL_TIERS:
            logger.warning(f"Model tier '{model_tier}' not found, defaulting to 'standard'")
            model_tier = "standard"
            
        config = self.MODEL_TIERS[model_tier]
        
        logger.info(f"Initializing {model_tier} model: {config['model_name']}")
        
        # Create the model with or without tools
        if self.function_declarations:
            model = genai.GenerativeModel(
                model_name=config["model_name"],
                generation_config=config["generation_config"],
                tools=[Tool(function_declarations=self.function_declarations)]
            )
        else:
            model = genai.GenerativeModel(
                model_name=config["model_name"],
                generation_config=config["generation_config"]
            )
            
        # Store the model and create chat session if not exists
        if model_tier not in self.chat_sessions:
            logger.info(f"Creating new chat session for {model_tier} model")
            self.chat_sessions[model_tier] = model.start_chat()
        
    def switch_model(self, model_tier, system_instruction=None):
        """
        Switch to a different model tier.
        
        Args:
            model_tier: The model tier to switch to ("light", "standard", or "advanced")
            system_instruction: Optional system instruction to initialize the chat
            
        Returns:
            bool: Whether the switch was successful
        """
        if model_tier not in self.MODEL_TIERS:
            logger.error(f"Invalid model tier: {model_tier}")
            return False
            
        logger.info(f"Switching from {self.active_model_tier} to {model_tier} model")
        
        # Initialize the model if it doesn't exist
        if model_tier not in self.chat_sessions:
            self._initialize_model(model_tier)
            
            # Initialize with system instruction if provided
            if system_instruction:
                logger.info("Initializing chat with system instruction")
                self.chat_sessions[model_tier].send_message(system_instruction)
                
        # Update active model tier
        self.active_model_tier = model_tier
        return True
    
    def send_message(self, message, model_tier=None, use_reflection=False,
                     reflection_context=None, max_retries=3, retry_delay=1):
        """
        Send a message to the LLM with optional reflection-based refinement.
        
        Args:
            message: The message to send
            model_tier: Optional model tier to use for this message
            use_reflection: Whether to use reflection for refinement
            reflection_context: Optional context for reflection (goal, step_description)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            
        Returns:
            The LLM response
        """
        # Import interaction logger here to avoid circular imports
        from src.logger.interaction_logger import log_llm_query, log_llm_response, log_multiple_function_calls_detected
        
        # Use specified model or active model
        tier = model_tier if model_tier else self.active_model_tier
        
        # Ensure the model is initialized
        if tier not in self.chat_sessions:
            self._initialize_model(tier)
        
        # Get the chat session for the selected model
        chat = self.chat_sessions[tier]
        
        # Send message with retry logic
        retry_count = 0
        last_exception = None
        
        while retry_count <= max_retries:
            try:
                logger.debug(f"Sending message to {tier} model (Attempt {retry_count + 1}/{max_retries + 1})")
                # Log the LLM query with model tier info
                log_llm_query(message, model=self.MODEL_TIERS[tier]["model_name"])
                response = chat.send_message(message)
                log_llm_response(response, model=self.MODEL_TIERS[tier]["model_name"])
                
                # Check for multiple function calls
                from src.agent.utils import get_function_calls
                function_calls = get_function_calls(response)
                if len(function_calls) > 1:
                    # Log the detection of multiple function calls
                    log_multiple_function_calls_detected(len(function_calls))
                    
                    # Create a modified system prompt to inform the model about one call at a time
                    if retry_count < max_retries:
                        retry_count += 1
                        retry_message = f"""
I noticed your response included {len(function_calls)} function calls in a single turn.
Due to system limitations, please make only one function call at a time.
For the task: {message}
First, execute just the first step/function, then I'll ask for the next steps afterward.
"""
                        logger.info(f"Requesting single function call (Attempt {retry_count}/{max_retries})")
                        message = retry_message
                        continue  # Retry with the updated message
                
                # If no reflection or evaluator not available, return the response directly
                if not use_reflection or not self.reflection_evaluator or not reflection_context:
                    return response
                      # Extract text from response for reflection
                try:
                    # First try to get text directly
                    if hasattr(response, 'text'):
                        response_text = response.text
                    # Next try to get text from candidates
                    elif hasattr(response, 'candidates') and response.candidates:
                        from src.agent.utils import get_text_response
                        response_text = get_text_response(response)
                    # Check for function calls
                    elif hasattr(response, 'candidates') and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'function_call'):
                                # If we have a function call, create a textual representation
                                func_name = part.function_call.name
                                response_text = f"[Function Call: {func_name}]"
                                break
                        else:
                            response_text = "No extractable text content"
                    else:
                        logger.warning("Could not extract text from response for reflection")
                        return response
                except Exception as text_error:
                    logger.warning(f"Error extracting text from response: {str(text_error)}")
                    response_text = "[Could not extract text from response]"
                    return response
                
                # Evaluate response quality with reflection
                goal = reflection_context.get("goal", "")
                step_description = reflection_context.get("step_description", "")
                
                evaluation = self.reflection_evaluator.evaluate_result(
                    goal=goal,
                    step_description=step_description,
                    result=response_text
                )
                
                logger.info(f"Response evaluation: {evaluation['recommendation']}")
                
                # If response is adequate, return it
                if evaluation["recommendation"] == "proceed":
                    logger.info("Reflection: Response evaluated as adequate")
                    return response
                
                # If response needs refinement, generate refined prompt and retry
                if retry_count < max_retries:
                    logger.info("Reflection: Refining response")
                    
                    refinement = self.reflection_evaluator.generate_step_refinement(
                        goal=goal,
                        step_description=step_description,
                        current_result=response_text,
                        issues=evaluation["issues"]
                    )
                    
                    # Create a refined prompt using the original message and refinement guidance
                    refined_prompt = f"""
Original request: {message}

The previous response had these issues: {evaluation['issues']}

Refinement guidance: {refinement['reasoning']}

Please provide an improved response addressing these issues.
"""
                    
                    # Increment retry count
                    retry_count += 1
                    
                    # Try with the refined prompt
                    logger.info(f"Sending refined prompt (Attempt {retry_count}/{max_retries})")
                    continue  # Will retry with the refined_prompt in next iteration
                
                # If max retries reached, return best response
                return response
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Error handling logic
                if "MALFORMED_FUNCTION_CALL" in str(e):
                    logger.warning(f"Detected MALFORMED_FUNCTION_CALL error. Retrying ({retry_count}/{max_retries})...")
                    if retry_count <= max_retries:
                        import time
                        time.sleep(retry_delay)
                        continue
                else:
                    # For other errors, log and retry as well
                    logger.warning(f"LLM API error: {str(e)}. Retrying ({retry_count}/{max_retries})...")
                    if retry_count <= max_retries:
                        import time
                        time.sleep(retry_delay)
                        continue
        
        # If we've exhausted all retries, log and re-raise the last exception
        logger.error(f"LLM API call failed after {max_retries} retries. Last error: {str(last_exception)}")
        raise last_exception