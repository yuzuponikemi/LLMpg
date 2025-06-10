# filepath: c:\Users\yuzup\source\repos\LLMpg\enhanced-research-agent\src\agent\core.py
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
from src.agent.model_manager import ModelManager
from src.logger.agent_logger import setup_logger
from src.logger.interaction_logger import (
    log_user_query, log_agent_response, log_function_call, log_function_result,
    log_plan_created, log_plan_step_execution, log_plan_step_result, log_plan_completed,
    log_conversation_session_start, log_conversation_session_end, log_error
)

# Setup logger for this module
logger = setup_logger(logger_name="agent_core")

class ResearchAgent:
    """Core research agent with LLM interaction capabilities"""
    
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
        
        # Set safety settings to allow function calls
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
        
        # Set default retry parameters
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Create reflection evaluator for self-reflection and refinement
        self.reflection_evaluator = ReflectionEvaluator()        # Enhanced system instruction
        self.system_instruction = """You are a helpful research assistant specialized in scientific and coding topics. 
You can execute code, read files, write files, list directory contents, search the web, browse webpages, and use RAG (Retrieval-Augmented Generation) for knowledge retrieval.

When asked to perform calculations, data analysis, or file operations, use the appropriate tools:
- Use 'execute_code' to run Python code for calculations or data analysis
- Use 'read_file' to access data files
- Use 'write_file' to save results
- Use 'list_files' to discover available files
- Use 'search_duckduckgo' for web searches
- Use 'browse_webpage' to browse specific URLs

For retrieval-augmented generation (RAG) tasks:
- Use 'query_google_rag' to retrieve relevant information from Qdrant collections using Google embeddings
- Use 'query_openai_rag' to retrieve relevant information from Qdrant collections using OpenAI embeddings
- Use 'index_rag_collection' to create new knowledge bases by indexing directories of files

IMPORTANT: You must make only ONE function call at a time. If a task requires multiple function calls, make them in sequence, waiting for each call to complete before making the next one.

IMPORTANT: When writing Python code for execution, you are limited to the following modules:
- Data processing: pandas, numpy, matplotlib.pyplot
- Standard library: math, random, statistics, datetime, calendar, collections
- More standard library: itertools, functools, re, csv, json, io, StringIO
- Utilities: base64, hashlib, time, uuid, urllib.parse, textwrap, string, copy
- Scientific (if available): scipy, sklearn modules, nltk, seaborn, plotly, statsmodels

For complex queries that involve multiple steps or require detailed research:
- Create a step-by-step plan as a numbered list
- Execute each step methodically
- Synthesize results into a comprehensive answer

Always provide clear explanations and analyze your results thoroughly."""
        
        # Create model manager for flexible model selection and reflection
        self.model_manager = ModelManager(
            reflection_evaluator=self.reflection_evaluator,
            function_declarations=function_declarations
        )
        
        # Send the system instruction to initialize the chat
        self.model_manager.send_message(self.system_instruction)
        
        # Create a plan manager with memory capabilities
        self.plan_manager = PlanManager()
        
        # Create a web utilities object
        self.web_utils = WebUtils()
        
        # Direct access to memory for agent-level operations
        self.memory = self.plan_manager.memory_manager
        
        # Track retry attempts for each step
        self.retry_counts = {}
        self.max_retries = 3
        
        logger.info("Agent initialized successfully!")
        # Start a new conversation session log
        log_conversation_session_start()

    def send_message_with_retry(self, message, use_reflection=False, reflection_context=None, model_tier=None):
        """
        Send a message to the LLM with reflection-based refinement and retry logic.
        
        Args:
            message: The message to send to the LLM
            use_reflection: Whether to use reflection to refine the response
            reflection_context: Context for reflection (goal, step_description)
            model_tier: Which model tier to use ("light", "standard", or "advanced")
            
        Returns:
            The LLM response
        """
        return self.model_manager.send_message(
            message=message,
            model_tier=model_tier,
            use_reflection=use_reflection,
            reflection_context=reflection_context,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
    
    def handle_function_call(self, response):
        """Process any function calls in the response"""
        if not response.candidates or not response.candidates[0].content:
            logger.debug("No valid candidates in response")
            return None
            
        candidate = response.candidates[0]
        
        # Get all function calls (first one only for backward compatibility)
        first_function_call = None
        
        # Check for function calls
        for part in candidate.content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                function_name = function_call.name
                args = dict(function_call.args)
                
                # Save the first one for processing
                if first_function_call is None:
                    first_function_call = function_call
                
                logger.info(f"Function call detected: {function_name}")
                log_function_call(function_name, args)
                
                # Only process the first function call to maintain compatibility
                # with the current architecture
                if function_call != first_function_call:
                    logger.info(f"Skipping additional function call: {function_name} (will process first one only)")
                    continue
                
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

        # Create reflection context for this step
        reflection_context = {
            "goal": self.plan_manager.original_goal,
            "step_description": step_instruction
        }
        
        # Use reflection by default for step execution
        response = self.send_message_with_retry(
            message=context_prompt,
            use_reflection=True,
            reflection_context=reflection_context,
            model_tier="standard"  # Use standard model for step execution
        )
        
        # Import the new multiple function call detector
        from src.agent.utils import get_function_call, get_function_calls
        
        # Check for multiple function calls
        function_calls = get_function_calls(response)
        
        if function_calls:
            if len(function_calls) > 1:
                logger.info(f"Multiple function calls detected ({len(function_calls)}) - processing first one only")
                
            # Process the first function call
            step_result = self.handle_function_call(response)
            # Extract text from the response
            result_text = get_text_response(step_result) if step_result else "No result for this step."
        else:
            step_result = get_text_response(response)
            result_text = step_result if step_result else "No result for this step."
            
        # Plan refinement assessment
        evaluation = self.reflection_evaluator.evaluate_result(
            goal=self.plan_manager.original_goal,
            step_description=step_instruction,
            result=result_text
        )
        
        # Process the evaluation for plan-level adjustments
        if evaluation["recommendation"] == "refine_plan":
            # The plan itself needs refinement
            logger.info("Evaluation suggests plan refinement is needed")
            
            # We'll return the current result but flag that the plan needs refinement
            # The plan refinement will be handled at the process_query level
            self.plan_manager.plan_needs_refinement = True
            self.plan_manager.refinement_issues = evaluation["issues"]
            
            return result_text + "\n\n[NOTE: This result indicates the plan may need refinement.]"
            
        else:
            # No plan adjustment needed, result is usable
            return result_text
        
    def process_query(self, query):
        """
        Process a user query with planning capabilities and self-reflection.
        
        This method handles:
        1. Continuing execution of an existing plan
        2. Plan refinement when needed
        3. Creating and executing new plans for complex queries
        4. Direct execution of simple queries
        
        Args:
            query: The user query string
            
        Returns:
            str: The response to provide to the user
        """
        logger.info(f"Processing query: {query}")
        log_user_query(query)
        final_response = ""

        # Check if we are currently executing a plan
        if self.plan_manager.has_active_plan():
            # Check if the plan needs refinement based on previous step evaluation
            if self.plan_manager.plan_needs_refinement:
                # Refine the plan using the new helper method
                return self._handle_plan_refinement()
            
            # Execute the next step in the existing plan
            step_result = self._handle_active_plan_execution()

            # Check if plan is now complete
            if not self.plan_manager.has_active_plan():
                # Handle plan completion using the new helper method
                return self._handle_plan_completion(query)
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
                # Enforce the model to create a plan
                parsed_plan, text_response, response = self._create_plan_for_complex_query(query)
                
                if not parsed_plan:
                    final_response = "I apologize, but I am unable to create a detailed plan for this query. Please try refining your question or providing more context."
                    logger.warning(final_response)
                    return final_response
                
                # Check for a plan in the response
                logger.info("Plan created successfully.")                # Execute first step immediately
                first_step_instruction = self.plan_manager.current_step()
                if first_step_instruction is None:
                    # Handle case where step content is None - use the first step from the parsed plan
                    if parsed_plan and len(parsed_plan) > 0:
                        first_step_instruction = parsed_plan[0]
                        logger.info(f"Retrieved first step from parsed_plan: {first_step_instruction}")
                    else:
                        first_step_instruction = "Execute the first step of the plan"
                        logger.warning(f"No step content available, using default: {first_step_instruction}")
                
                logger.info(f"Executing Step 1: {first_step_instruction}")
                log_plan_step_execution(1, len(parsed_plan), first_step_instruction)

                # Store the plan text itself as the first part of the response
                plan_announcement = text_response + "\n\n"  # Add spacing                # Execute step 1
                step_result = self._execute_step(first_step_instruction)
                self.plan_manager.store_step_result(first_step_instruction, step_result)
                log_plan_step_result(1, step_result)
                self.plan_manager.advance_to_next_step()

                # Always continue with the plan after step 1, never immediately generate a summary
                # Single-step plans will be handled by _handle_plan_completion when all steps finish
                next_step_instruction = self.plan_manager.next_step()
                
                # Check if we reached the end of the plan
                if next_step_instruction is None:
                    logger.info("Plan finished after first step. Generating summary.")
                    return self._handle_plan_completion(query)
                else:
                    # Plan not finished, inform user about the next step
                    final_response = f"{plan_announcement}Finished Step 1. Result:\n{step_result}\n\n---> Now proceeding to Step {self.plan_manager.current_step_index + 1}: {next_step_instruction}"
            else:                
                # Send the query as a message to the model
                parsed_plan, text_response, response = self._process_simple_query(query)

                function_call = get_function_call(response)

                if function_call and not self.plan_manager.requires_detailed_info(query):
                    # Only allow direct function calls for simple queries
                    logger.info("Simple query - LLM calling function directly (no plan).")
                    function_response = self.handle_function_call(response)
                    final_response = get_text_response(function_response)
                    self.plan_manager.original_goal = None  # Clear goal as it was handled directly
                elif function_call:
                    # For complex queries with function calls, convert to a plan
                    return self._create_and_execute_synthetic_plan(query, function_call, response)
                elif parsed_plan:
                    return self._execute_parsed_plan(parsed_plan, text_response, query)
                else:
                    # No plan, no function call, treat as single-step plan
                    return self._execute_single_step_plan(query)
        
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
    
    def set_model_tier(self, model_tier):
        """
        Switch to a different model tier
        
        Args:
            model_tier: One of "light", "standard", or "advanced"
            
        Returns:
            True if successful, False otherwise
        """
        return self.model_manager.switch_model(model_tier, self.system_instruction)
    
    def _handle_plan_refinement(self):
        """
        Refine the current plan based on previous step evaluation.
        
        Returns:
            str: A message describing the plan refinement
        """
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

    def _handle_active_plan_execution(self):
        """
        Execute the next step in an active plan.
        
        Returns:
            str: The response to provide to the user
        """
        step_instruction = self.plan_manager.current_step()
        step_count = self.plan_manager.current_step_index + 1
        total_steps = len(self.plan_manager.current_plan)
        
        logger.info(f"Executing Plan Step {step_count}/{total_steps}: {step_instruction}")
        log_plan_step_execution(step_count, total_steps, step_instruction)

        step_result = self._execute_step(step_instruction)
        self.plan_manager.store_step_result(step_instruction, step_result)
        log_plan_step_result(step_count, step_result)
        self.plan_manager.advance_to_next_step()

        return step_result
    
    def _handle_plan_completion(self, query=None):
        """
        Handle plan completion by generating a summary.
        
        Args:
            query: The original query if available
            
        Returns:
            str: The final summary response
        """
        logger.info("Plan finished. Generating summary.")
        
        # Generate summary prompt based on plan type
        if self.plan_manager.is_one_step_plan():
            summary_prompt = self.plan_manager.generate_one_step_summary_prompt()
            reflection_step_description = "Generate summary for simple query"
        else:
            summary_prompt = self.plan_manager.generate_plan_summary_prompt()
            reflection_step_description = "Generate final summary"
        
        # The goal is either the stored original goal or the current query
        reflection_goal = self.plan_manager.original_goal or query
        
        # Use reflection for final summary with advanced model
        response = self.send_message_with_retry(
            message=summary_prompt,
            use_reflection=True,
            reflection_context={
                "goal": reflection_goal, 
                "step_description": reflection_step_description
            },
            model_tier="advanced"  # Use advanced model for final summary
        )
        
        final_response = get_text_response(response)
        
        # Reset plan state
        logger.info("Resetting plan state.")
        log_plan_completed(self.plan_manager.original_goal)
        self.plan_manager.reset_plan()
        
        return final_response
    
    def _create_plan_for_complex_query(self, query):
        """
        Create a plan for a complex query that needs detailed investigation.
        
        Args:
            query: The user query
            
        Returns:
            tuple: (parsed_plan, text_response, model_response)
        """
        logger.info("Query appears to need detailed investigation. Enforcing planning mode.")
        
        # Enforce the model to create a plan
        planning_prompt = f"This query requires a methodical research approach: '{query}'\nYou MUST create a multi-step plan. Present your plan as a numbered list including specific steps for searching, data gathering, and analysis. Do not call functions directly."
        
        # Use standard model for planning with reflection
        response = self.send_message_with_retry(
            message=planning_prompt,
            use_reflection=True,
            reflection_context={"goal": query, "step_description": "Create a detailed research plan"},
            model_tier="standard"
        )
        
        # Check for a plan in the response
        text_response = get_text_response(response)
        logger.info(f"LLM planning response: {text_response}")
        # Extract plan from response
        parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)
        
        # If no plan was found and no function call, try again with stronger enforcement
        if not parsed_plan and not get_function_call(response):
            logger.info("No plan detected in first attempt. Trying again with stronger enforcement.")                    
            planning_prompt = f"For this query: '{query}'\nYou MUST create a step-by-step plan as a numbered list. Do not answer directly. Do not call functions directly. First plan out the steps, then we will execute them one by one."
            
            # Try with advanced model for better planning
            response = self.send_message_with_retry(
                message=planning_prompt,
                use_reflection=True,
                reflection_context={"goal": query, "step_description": "Create a detailed research plan"},
                model_tier="advanced"
            )
            
            text_response = get_text_response(response)
            logger.info(f"Second planning response: {text_response}")                    
            parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)
        
        return parsed_plan, text_response, response
    
    def _process_simple_query(self, query):
        """
        Process a simple query without enforcing planning.
        
        Args:
            query: The user query
            
        Returns:
            tuple: (parsed_plan, text_response, model_response)
        """
        logger.info("Processing as a simple query")
        
        # Use light model for simple queries
        response = self.send_message_with_retry(
            message=query,
            use_reflection=True,
            reflection_context={"goal": query, "step_description": query},
            model_tier="light"
        )
        
        text_response = get_text_response(response)
        parsed_plan = self.plan_manager.extract_plan_from_llm_response(text_response)
        
        return parsed_plan, text_response, response
        
    def _create_and_execute_synthetic_plan(self, query, function_call, response):
        """
        Create and begin executing a synthetic plan from a function call.
        
        Args:
            query: The user query
            function_call: The function call object
            response: The model response
            
        Returns:
            str: The response from the plan execution
        """
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
        
    def _execute_parsed_plan(self, parsed_plan, text_response, query):
        """
        Execute a plan that was parsed from the model response.
        
        Args:
            parsed_plan: The list of steps in the plan
            text_response: The raw text response from the model
            query: The user query
            
        Returns:
            str: The response after executing the first step
        """
        logger.info("LLM generated a plan:")
        for i, step in enumerate(parsed_plan):
            logger.info(f"   {i+1}. {step}")

        # Store the plan
        self.plan_manager.set_new_plan(parsed_plan, query)
        log_plan_created(parsed_plan, query)

        # Execute step 1 immediately
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
            return plan_announcement + self._handle_plan_completion(query)
        else:
            # Plan not finished, inform user about the next step
            next_step_instruction = self.plan_manager.next_step()
            return f"{plan_announcement}Finished Step 1. Result:\n{step_result}\n\n---> Now proceeding to Step {self.plan_manager.current_step_index + 1}: {next_step_instruction}"
    
    def _execute_single_step_plan(self, query):
        """
        Create and execute a single step plan for a simple query.
        
        Args:
            query: The user query
            
        Returns:
            str: The final response
        """
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
        
        return self._handle_plan_completion(query)
