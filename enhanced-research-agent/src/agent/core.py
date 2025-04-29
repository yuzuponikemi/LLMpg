# filepath: c:\Users\yuzup\source\repos\llmdev\enhanced-research-agent\src\agent\core.py
"""
Core agent functionality for the research agent with enhanced logging
"""

import os
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import Tool

from src.agent.utils import get_text_response, get_function_call, parse_plan_from_text
from src.agent.planning import PlanManager
from src.agent.web import WebUtils
from src.agent.reflection import ReflectionEvaluator
from src.logger.agent_logger import setup_logger
from src.logger.interaction_logger import (
    log_user_query, log_agent_response, log_function_call, log_function_result,
    log_plan_created, log_plan_step_execution, log_plan_step_result, log_plan_completed,
    log_conversation_session_start, log_conversation_session_end, log_error,
    log_reflection, log_refinement
)

# Setup logger for this module
logger = setup_logger(logger_name="agent_core")

class ResearchAgent:
    """Core research agent with LLM interaction capabilities"""
    
# Replace the send_message method with our retry-enabled version
    def send_message_with_retry(self, message):
        """
        Send a message to the LLM with retry logic for handling API errors.
        
        Args:
            message: The message to send to the LLM
            
        Returns:
            The LLM response
            
        Raises:
            Exception: If all retries fail
        """
        retry_count = 0
        last_exception = None
        
        while retry_count <= self.max_retries:
            try:
                logger.debug(f"Sending message to LLM (Attempt {retry_count + 1}/{self.max_retries + 1})")
                response = self.chat.send_message(message)
                return response
                
            except Exception as e:
                last_exception = e
                retry_count += 1
                
                # Check if it's a MALFORMED_FUNCTION_CALL error
                if "MALFORMED_FUNCTION_CALL" in str(e):
                    logger.warning(f"Detected MALFORMED_FUNCTION_CALL error. Retrying ({retry_count}/{self.max_retries})...")
                    if retry_count <= self.max_retries:
                        import time
                        time.sleep(self.retry_delay)  # Small delay before retrying
                        continue
                else:
                    # For other errors, log and retry as well
                    logger.warning(f"LLM API error: {str(e)}. Retrying ({retry_count}/{self.max_retries})...")
                    if retry_count <= self.max_retries:
                        import time
                        time.sleep(self.retry_delay)  # Small delay before retrying
                        continue
        
        # If we've exhausted all retries, log and re-raise the last exception
        logger.error(f"LLM API call failed after {self.max_retries} retries. Last error: {str(last_exception)}")
        raise last_exception
    
    def __init__(self, tool_functions, function_declarations):
        """
        Initialize the research agent with the specified tools.
        
        Args:
            tool_functions: Dictionary mapping function names to their implementations
            function_declarations: List of function declarations for the LLM
        """
        logger.info("Initializing Research Agent...")
        
        # Store the tool functions
        self.tool_functions = tool_functions
        
        # Load API key
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        
        # Set default retry parameters
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Create the model with tools
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            },
            tools=[Tool(function_declarations=function_declarations)]
        )
        
        # Enhanced system instruction
        self.system_instruction = """You are a helpful research assistant specialized in scientific and coding topics. 
You can execute code, read files, write files, list directory contents, search the web, and browse webpages.

When asked to perform calculations, data analysis, or file operations, use the appropriate tools:
- Use 'execute_code' to run Python code for calculations or data analysis
- Use 'read_file' to access data files
- Use 'write_file' to save results
- Use 'list_files' to discover available files
- Use 'search_duckduckgo' for web searches
- Use 'browse_webpage' to browse specific URLs

For complex queries that involve multiple steps or require detailed research:
- Create a step-by-step plan as a numbered list
- Execute each step methodically
- Synthesize results into a comprehensive answer

Always provide clear explanations and analyze your results thoroughly."""
        
        # Initialize the chat session
        logger.info("Initializing chat session...")
        self.chat = self.model.start_chat()
        
        # Send the system instruction to initialize the chat
        self.chat.send_message(self.system_instruction)
        
        # Create a plan manager with memory capabilities
        self.plan_manager = PlanManager()
        
        # Create a web utilities object
        self.web_utils = WebUtils()
        
        # Create a reflection evaluator for self-reflection and refinement
        self.reflection_evaluator = ReflectionEvaluator()
        
        # Direct access to memory for agent-level operations
        self.memory = self.plan_manager.memory_manager
        
        # Track retry attempts for each step
        self.retry_counts = {}
        self.max_retries = 3
        
        logger.info("Agent initialized successfully!")
        # Start a new conversation session log
        log_conversation_session_start()
    
    def handle_function_call(self, response):
        """Process any function calls in the response"""
        if not response.candidates or not response.candidates[0].content:
            logger.debug("No valid candidates in response")
            return None
            
        candidate = response.candidates[0]
        
        # Check for function calls
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                args = dict(function_call.args)
                
                logger.info(f"Function call detected: {function_name}")
                log_function_call(function_name, args)
                
                # Call the function
                if function_name in self.tool_functions:
                    try:
                        function_to_call = self.tool_functions[function_name]
                        result = function_to_call(**args)
                        logger.info(f"Function executed: {function_name}")
                        log_function_result(function_name, result)
                        
                        # Auto URL extraction and browsing for search_duckduckgo
                        auto_browsed_content = None
                        if function_name == "search_duckduckgo" and self.plan_manager.original_goal:
                            # Extract URLs from search results
                            urls = self.web_utils.extract_urls_from_search_results(result)
                            logger.info(f"Extracted {len(urls)} URLs from search results")
                            
                            # Find the most relevant URL to auto-browse
                            for url_data in urls:
                                if self.web_utils.is_url_relevant(url_data, self.plan_manager.original_goal):
                                    logger.info(f"Auto-browsing relevant URL: {url_data['url']}")
                                    try:
                                        # Auto-browse the relevant URL
                                        auto_browsed_content = self.tool_functions["browse_webpage"](url=url_data['url'])
                                        logger.info(f"Successfully auto-browsed URL: {url_data['url']}")
                                        log_function_result("auto_browse_webpage", {"url": url_data['url']})
                                        break
                                    except Exception as browse_error:
                                        error_msg = str(browse_error)
                                        logger.error(f"Error auto-browsing URL: {error_msg}")
                                        log_error(error_msg, "auto_browsing")
                                        auto_browsed_content = None
                        
                        # Prepare the response - combine search result with auto-browsed content if available
                        response_data = result if isinstance(result, dict) else {"result": result}
                          # Add auto-browsed content to the same response if available
                        if auto_browsed_content:
                            logger.info("Adding auto-browsed content to function response")
                            if isinstance(response_data, dict):
                                response_data["auto_browsed_content"] = auto_browsed_content
                            else:
                                response_data = {
                                    "search_result": response_data,
                                    "auto_browsed_content": auto_browsed_content
                                }
                        
                        # Send combined function result back to model
                        # Format the response according to Gemini API expectations
                        # This ensures the function response parts match function call parts
                        try:
                            # Ensure the response data is properly sanitized for JSON
                            if isinstance(response_data, str):
                                # Handle potential unicode issues by replacing problematic characters
                                response_data = response_data.encode('utf-8', errors='replace').decode('utf-8')
                                # Convert string to a properly formatted response object
                                response_data = {"result": response_data}
                            elif isinstance(response_data, (int, float, bool)):
                                # Convert primitive types to a dictionary
                                response_data = {"result": response_data}
                            elif response_data is None:
                                # Handle None values
                                response_data = {"result": "Function executed successfully with no return value"}
                            
                            # Make sure complex objects are JSON serializable
                            try:
                                import json
                                # Test JSON serialization
                                json.dumps(response_data)
                            except (TypeError, OverflowError) as json_error:
                                logger.warning(f"Response data not JSON serializable: {str(json_error)}")
                                if isinstance(response_data, dict):
                                    # Try to sanitize the dictionary by converting problematic values to strings
                                    sanitized_data = {}
                                    for key, value in response_data.items():
                                        try:
                                            json.dumps({key: value})
                                            sanitized_data[key] = value
                                        except:
                                            sanitized_data[key] = str(value)
                                    response_data = sanitized_data
                                else:
                                    # Fallback for non-serializable objects
                                    response_data = {"result": str(response_data)}
                            
                            response_to_model = {
                                "function_response": {
                                    "name": function_name,
                                    "response": response_data
                                }
                            }
                            return self.send_message_with_retry(response_to_model)
                        except Exception as resp_error:
                            logger.error(f"Error formatting function response: {str(resp_error)}")
                            # Fallback to a simpler response if there are encoding issues
                            simplified_response = {
                                "function_response": {
                                    "name": function_name,
                                    "response": {"result": "Function executed successfully but result contains unsupported characters"}
                                }
                            }
                            return self.send_message_with_retry(simplified_response)
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Error executing function: {error_msg}")
                        log_error(error_msg, f"function_execution_{function_name}")
                        error_response = {
                            "function_response": {
                                "name": function_name,
                                "response": {"error": error_msg}
                            }
                        }
                        return self.send_message_with_retry(error_response)
                else:
                    logger.warning(f"Unknown function: {function_name}")
        
        return None
    def _execute_step(self, step_instruction):
        """Executes a single step of the plan with self-reflection and refinement"""
        logger.info(f"Executing step: {step_instruction}")
        
        # Initialize or get the retry count for this step
        step_key = f"{self.plan_manager.original_goal}_{step_instruction}"
        self.retry_counts[step_key] = self.retry_counts.get(step_key, 0)
        
        # Provide context about the original goal and previous steps
        context_prompt = f"Original goal: '{self.plan_manager.original_goal}'. "
        
        # Generate relevant context from memory
        memory_context = self.memory.generate_memory_context(step_instruction)
        if memory_context:
            context_prompt += f"\n\n{memory_context}\n\n"
        
        # Add previous step results for continuity
        if self.plan_manager.step_results:
            context_prompt += f"Results from previous steps are available in history. "
            
        # Add instruction for current step
        context_prompt += f"Now execute this step: {step_instruction}"

        response = self.send_message_with_retry(context_prompt)
        function_call = get_function_call(response)

        if function_call:
            step_result = self.handle_function_call(response)
            # Extract text from the response
            result_text = get_text_response(step_result) if step_result else "No result for this step."
        else:
            step_result = get_text_response(response)
            result_text = step_result if step_result else "No result for this step."
            
        # Self-reflection: evaluate the result quality
        evaluation = self.reflection_evaluator.evaluate_result(
            goal=self.plan_manager.original_goal,
            step_description=step_instruction,
            result=result_text
        )
        
        # Process the evaluation
        if evaluation["recommendation"] == "proceed":
            # Result is adequate, continue with the plan
            logger.info("Step result evaluated as adequate, proceeding with plan")
            return result_text
            
        elif evaluation["recommendation"] == "retry" and self.retry_counts[step_key] < self.max_retries:
            # Result is inadequate, retry the step with refinement
            logger.info(f"Step result inadequate, retrying (attempt {self.retry_counts[step_key] + 1}/{self.max_retries})")
            
            # Generate refinement for the step
            refinement = self.reflection_evaluator.generate_step_refinement(
                goal=self.plan_manager.original_goal,
                step_description=step_instruction,
                current_result=result_text,
                issues=evaluation["issues"]
            )
            
            # Increment retry count
            self.retry_counts[step_key] += 1
            
            # Execute the refined step
            refined_step = refinement["refined_step"]
            logger.info(f"Executing refined step: {refined_step}")
            
            # Recursive call with the refined step
            refined_result = self._execute_step(refined_step)
            
            # Return the refined result
            return refined_result
            
        elif evaluation["recommendation"] == "refine_plan":
            # The plan itself needs refinement
            logger.info("Evaluation suggests plan refinement is needed")
            
            # We'll return the current result but flag that the plan needs refinement
            # The plan refinement will be handled at the process_query level
            self.plan_manager.plan_needs_refinement = True
            self.plan_manager.refinement_issues = evaluation["issues"]
            
            return result_text + "\n\n[NOTE: This result indicates the plan may need refinement.]"
            
        else:
            # Max retries reached or default case
            if self.retry_counts[step_key] >= self.max_retries:
                logger.warning(f"Max retries ({self.max_retries}) reached for step, proceeding with best result")
                return result_text + "\n\n[NOTE: This result was obtained after multiple retry attempts.]"
            
            # Default fallback
            return result_text
    
    def process_query(self, query):
        """Process a user query with planning capabilities and self-reflection"""
        logger.info(f"Processing query: {query}")
        log_user_query(query)
        final_response = ""

        # Check if we are currently executing a plan
        if self.plan_manager.has_active_plan():
            # Check if the plan needs refinement based on previous step evaluation
            if self.plan_manager.plan_needs_refinement:
                logger.info("Plan needs refinement based on previous step evaluation")
                
                # Get executed steps with their results for context
                executed_steps = self.plan_manager.get_executed_steps_with_results()
                
                # Generate refined plan steps for remaining work
                refined_steps = self.reflection_evaluator.generate_plan_refinement(
                    goal=self.plan_manager.original_goal,
                    current_plan=self.plan_manager.current_plan,
                    executed_steps=executed_steps,
                    issues=self.plan_manager.refinement_issues
                )
                
                # Apply the refinement to the current plan
                self.plan_manager.refine_plan(refined_steps)
                
                # Inform user about the plan refinement
                plan_refinement_message = "I've refined the plan based on my evaluation of previous steps:\n"
                for i, step in enumerate(refined_steps, self.plan_manager.current_step_index + 1):
                    plan_refinement_message += f"{i}. {step}\n"
                logger.info(f"Plan refined with {len(refined_steps)} new steps")
                
                return plan_refinement_message
            
            # Execute the next step in the existing plan
            step_instruction = self.plan_manager.current_step()
            step_count = self.plan_manager.current_step_index + 1
            total_steps = len(self.plan_manager.current_plan)
            logger.info(f"Continuing Plan. Executing Step {step_count}/{total_steps}: {step_instruction}")
            log_plan_step_execution(step_count, total_steps, step_instruction)

            step_result = self._execute_step(step_instruction)
            # Store result with context for final summary
            self.plan_manager.store_step_result(step_instruction, step_result)
            log_plan_step_result(step_count, step_result)
            self.plan_manager.advance_to_next_step()

            # Check if plan is now complete
            if not self.plan_manager.has_active_plan():
                logger.info("Plan finished. Generating summary.")
                # Generate summary prompt and send to model
                summary_prompt = self.plan_manager.generate_plan_summary_prompt()
                response = self.chat.send_message(summary_prompt)
                final_response = get_text_response(response)
                # Reset plan state
                logger.info("Resetting plan state.")
                log_plan_completed(self.plan_manager.original_goal)
                self.plan_manager.reset_plan()
            else:
                # Plan not finished, inform user about the next step
                next_step_instruction = self.plan_manager.next_step()
                # Include result of the step just finished
                step_count = self.plan_manager.current_step_index
                final_response = f"Finished Step {step_count}. Result:\n{step_result}\n\n---> Now proceeding to Step {step_count + 1}: {next_step_instruction}"
        else:
            # No active plan, process the new user query
            logger.info(f"Processing new query: {query}")
            # Reset plan state just in case
            self.plan_manager.reset_plan()
            self.plan_manager.original_goal = query  # Store the goal

            # Check if query requires detailed info to enforce planning mode
            if self.plan_manager.requires_detailed_info(query):
                # Enforce the model to create a plan                planning_prompt = f"This query requires a methodical research approach: '{query}'\nYou MUST create a multi-step plan. Present your plan as a numbered list including specific steps for searching, data gathering, and analysis. Do not call functions directly."
                logger.info("Query appears to need detailed investigation. Enforcing planning mode.")
                response = self.send_message_with_retry(planning_prompt)
                # Check for a plan in the response
                text_response = get_text_response(response)
                logger.info(f"LLM response: {text_response}")
                # Use the enhanced plan extraction method from PlanManager
                parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)
                
                if not parsed_plan and not get_function_call(response):
                    # Try again with stronger enforcement
                    logger.info("No plan detected in first attempt. Trying again with stronger enforcement.")                    
                    planning_prompt = f"For this query: '{query}'\nYou MUST create a step-by-step plan as a numbered list. Do not answer directly. Do not call functions directly. First plan out the steps, then we will execute them one by one."
                    response = self.send_message_with_retry(planning_prompt)
                    text_response = get_text_response(response)
                    logger.info(f"LLM response: {text_response}")                    
                    parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)            
            else:                
                # Send the query as a message to the model
                response = self.send_message_with_retry(query)
                text_response = get_text_response(response)
                parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)
                
            function_call = get_function_call(response)

            if function_call and not self.plan_manager.requires_detailed_info(query):
                # Only allow direct function calls for simple queries
                logger.info("Simple query - LLM calling function directly (no plan).")
                function_response = self.handle_function_call(response)
                final_response = get_text_response(function_response)
                self.plan_manager.original_goal = None  # Clear goal as it was handled directly
            elif function_call:
                # For complex queries with function calls, convert to a plan
                logger.info("Complex query needs a plan but got function call. Creating synthetic plan.")
                function_name = function_call.name
                args = dict(function_call.args)
                # Create a synthetic plan with the function call as the first step and additional steps
                synthetic_plan = [f"Search for information about {args.get('query', query)}"]
                if function_name == "search_duckduckgo":
                    synthetic_plan.append(f"Browse relevant websites about {args.get('query', query)}")
                synthetic_plan.append(f"Synthesize findings into an explanation about {query}")
                
                # Store the plan
                self.plan_manager.set_new_plan(synthetic_plan, query)
                log_plan_created(synthetic_plan, query)
                
                logger.info("Created synthetic plan:")
                for i, step in enumerate(synthetic_plan):
                    logger.info(f"   {i+1}. {step}")
                
                # Execute first step using the function call
                logger.info(f"Executing synthetic plan Step 1 using function call {function_name}")
                log_plan_step_execution(1, len(synthetic_plan), synthetic_plan[0])
                function_response = self.handle_function_call(response)
                step_result = get_text_response(function_response)
                self.plan_manager.store_step_result(synthetic_plan[0], step_result)
                log_plan_step_result(1, step_result)
                self.plan_manager.advance_to_next_step()
                
                # Continue with the next step immediately (auto-execution of plan)
                return self.process_query("continue")  # Pass dummy input to trigger next step execution
            elif parsed_plan:
                logger.info("LLM generated a plan:")
                for i, step in enumerate(parsed_plan):
                    logger.info(f"   {i+1}. {step}")

                # Store the plan
                self.plan_manager.set_new_plan(parsed_plan, query)
                log_plan_created(parsed_plan, query)

                # Now, execute step 1 immediately
                first_step_instruction = self.plan_manager.current_step()
                logger.info(f"Plan detected. Executing Step 1: {first_step_instruction}")
                log_plan_step_execution(1, len(parsed_plan), first_step_instruction)

                # Store the plan text itself as the first part of the response
                plan_announcement = text_response + "\n\n"  # Add spacing

                # Execute step 1
                step_result = self._execute_step(first_step_instruction)
                self.plan_manager.store_step_result(first_step_instruction, step_result)
                log_plan_step_result(1, step_result)
                self.plan_manager.advance_to_next_step()

                # Check if plan is now complete (only 1 step)
                if not self.plan_manager.has_active_plan():
                    logger.info("One-step plan finished. Generating summary.")                    
                    summary_prompt = self.plan_manager.generate_one_step_summary_prompt()
                    response = self.send_message_with_retry(summary_prompt)
                    final_response = plan_announcement + get_text_response(response)
                    # Reset plan state
                    logger.info("Resetting plan state.")
                    log_plan_completed(self.plan_manager.original_goal)
                    self.plan_manager.reset_plan()
                else:
                    # Plan not finished, inform user about the next step
                    next_step_instruction = self.plan_manager.next_step()
                    final_response = f"{plan_announcement}Finished Step 1. Result:\n{step_result}\n\n---> Now proceeding to Step {self.plan_manager.current_step_index + 1}: {next_step_instruction}"
            else:
                # No plan, no function call, treat as single-step plan
                logger.info("No plan or function call detected. Treating as single-step plan.")
                single_step_plan = [f"{query}"]
                self.plan_manager.set_new_plan(single_step_plan, query)
                log_plan_created(single_step_plan, query)
                
                first_step_instruction = self.plan_manager.current_step()
                logger.info(f"Single-step plan detected. Executing: {first_step_instruction}")
                log_plan_step_execution(1, 1, first_step_instruction)
                
                step_result = self._execute_step(first_step_instruction)
                self.plan_manager.store_step_result(first_step_instruction, step_result)
                log_plan_step_result(1, step_result)
                self.plan_manager.advance_to_next_step()
                
                # Generate summary for single-step plan
                summary_prompt = self.plan_manager.generate_one_step_summary_prompt()
                response = self.chat.send_message(summary_prompt)
                final_response = get_text_response(response)
                logger.info("Resetting plan state.")
                log_plan_completed(self.plan_manager.original_goal)
                self.plan_manager.reset_plan()

        # Log the agent's response
        log_agent_response(final_response)
        return final_response if final_response else "Sorry, I couldn't generate a response."
    
    def process_conversation(self, query):
        """Process a query and automatically execute any multi-step plans"""
        logger.info("\nUser query: " + query)
        
        # Process initial user query
        agent_response = self.process_query(query)
        logger.info(f"Agent response generated (length: {len(agent_response)})")
        print(f"Agent: {agent_response}")
        
        # Automatically continue executing the plan until it's complete
        plan_in_progress = self.plan_manager.has_active_plan()
        
        while plan_in_progress:
            step_count = self.plan_manager.current_step_index + 1
            total_steps = len(self.plan_manager.current_plan)
            logger.info(f"\n[Auto-executing next step ({step_count}/{total_steps})]")
            
            # Execute the next step with a dummy "continue" command
            agent_response = self.process_query("continue")
            print(f"Agent: {agent_response}")
            
            # Check if we need to continue
            plan_in_progress = self.plan_manager.has_active_plan()
          # Ensure memories are saved after conversation ends
        if hasattr(self, 'memory'):
            logger.info("Saving important memories from this conversation")
            self.memory.persist_important_memories()
            self.memory.save_long_term_memory()
            
        return agent_response
            
    def recall_memory(self, query):
        """
        Allow users to explicitly query the agent's memory
        
        Args:
            query: The query to search for in memory
            
        Returns:
            A formatted string with relevant memories
        """
        if not hasattr(self, 'memory'):
            return "Memory system is not available."
            
        # Get relevant memories for this query
        memories = self.memory.get_relevant_memories(query, limit=10)
        
        if not memories:
            return f"I don't have any memories relevant to '{query}'."
            
        # Format memories for display
        response = f"Here's what I remember about '{query}':\n\n"
        
        for i, memory in enumerate(memories, 1):
            memory_type = memory["type"]
            content = memory["content"]
            source = memory["source"]
            relevance = memory["relevance"]
            
            response += f"{i}. "
            
            if memory_type == "facts":
                if isinstance(content, str):
                    response += f"Fact: {content}"
                else:
                    response += f"Fact: {str(content)[:200]}"
            elif memory_type == "conclusions":
                if isinstance(content, dict) and "key_conclusions" in content:
                    response += f"Conclusion: {content['key_conclusions'][0] if content['key_conclusions'] else str(content)[:200]}"
                else:
                    response += f"Conclusion: {str(content)[:200]}"
            elif memory_type == "web_content":
                if isinstance(content, dict):
                    response += f"From web: {content.get('step', '')} - {str(content.get('content', ''))[:150]}..."
                else:
                    response += f"From web: {str(content)[:150]}..."
            else:
                response += f"{memory_type.capitalize()}: {str(content)[:150]}..."
                
            response += f" (Source: {source})\n\n"
            
        return response
            
        return agent_response
