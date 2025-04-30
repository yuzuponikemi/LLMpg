"""
Planning functionality for the research agent
"""

import time
import re
from typing import List, Optional, Dict, Any

from src.logger.agent_logger import setup_logger
from src.agent.memory import MemoryManager

# Setup logger for this module
logger = setup_logger(logger_name="plan_manager")

class PlanManager:
    """Manages the creation, execution, and tracking of plans"""
    
    def __init__(self, memory_manager=None):
        # Planning state variables
        self.current_plan = None  # List of steps to execute
        self.current_step_index = -1  # Current step being executed (-1 means no current step)
        self.original_goal = None  # Keep track of the original user query
        self.step_results = []  # Store results from each step of the plan
        
        # Plan refinement flags
        self.plan_needs_refinement = False
        self.refinement_issues = []
        
        # Initialize or use provided memory manager
        self.memory_manager = memory_manager if memory_manager else MemoryManager()    
    def evaluate_plan_complexity(self, plan_steps):
        """
        Uses LLM to analyze each step in the plan to identify and break down overly complex tasks.
        
        Args:
            plan_steps (list): The list of plan steps to evaluate
            
        Returns:
            list: A potentially modified plan with complex steps broken down
        """
        if not plan_steps:
            return plan_steps
        
        # Import here to avoid circular imports
        from src.agent.model_manager import ModelManager
        
        refined_plan = []
        model_manager = ModelManager()
        
        for step_idx, step in enumerate(plan_steps):
            logger.info(f"Evaluating complexity of step {step_idx+1}: {step}")
            
            # Prepare prompt for LLM to evaluate if the step is complex
            evaluation_prompt = self._create_step_evaluation_prompt(step)
            
            # Use lightweight model for step evaluation
            try:
                response = model_manager.send_message(
                    evaluation_prompt,
                    use_reflection=False,
                    model_tier="light"  # Use lightweight model for efficiency
                )
                
                # Extract text response
                from src.agent.utils import get_text_response
                eval_result = get_text_response(response)
                
                # Parse the evaluation result
                is_complex, reason = self._parse_step_evaluation(eval_result)
                
                if is_complex:
                    logger.info(f"LLM identified step {step_idx+1} as complex. Reason: {reason}")
                    
                    # Get breakdown from LLM
                    breakdown_prompt = self._create_step_breakdown_prompt(step, reason)
                    breakdown_response = model_manager.send_message(
                        breakdown_prompt,
                        use_reflection=False,
                        model_tier="light"
                    )
                    
                    breakdown_result = get_text_response(breakdown_response)
                    broken_steps = self._parse_step_breakdown(breakdown_result)
                    
                    if broken_steps and len(broken_steps) > 1:
                        logger.info(f"Breaking down step {step_idx+1} into {len(broken_steps)} subtasks")
                        refined_plan.extend(broken_steps)
                    else:
                        logger.warning(f"Failed to break down step {step_idx+1}, keeping original")
                        refined_plan.append(step)
                else:
                    logger.info(f"Step {step_idx+1} is not complex, keeping as is")
                    refined_plan.append(step)
                    
            except Exception as e:
                logger.error(f"Error during step complexity evaluation: {str(e)}")
                # On error, keep the original step
                refined_plan.append(step)
                
        return refined_plan
    
    def _break_down_complex_step(self, step):
        """
        Breaks down a complex step into multiple simpler steps.
        
        Args:
            step (str): The complex step to break down
            
        Returns:
            list: A list of simpler steps that accomplish the same task
        """
        # Pattern 1: Gathering data for multiple locations
        location_match = re.search(r'(?:coordinates|data|information|details) (?:for|about|of) (.*?) and (.*?)(?:\.|$)', step, re.IGNORECASE)
        if location_match:
            loc1, loc2 = location_match.groups()
            data_type = "information"
            # Try to extract what type of data we're looking for
            data_type_match = re.search(r'(coordinates|data|information|details)', step, re.IGNORECASE)
            if data_type_match:
                data_type = data_type_match.group(1)
            return [
                f"Search for {data_type} for {loc1}.",
                f"Search for {data_type} for {loc2}."
            ]
            
        # Pattern 2: Multiple search operations
        search_match = re.search(r'(?:search|find|gather|collect|look up) (.*?) and (.*?)(?:\.|$)', step, re.IGNORECASE)
        if search_match:
            item1, item2 = search_match.groups()
            return [
                f"Search for {item1}.",
                f"Search for {item2}."
            ]
            
        # Pattern 3: Compare or analyze multiple things
        compare_match = re.search(r'(?:compare|analyze|evaluate) (.*?) and (.*?)(?:\.|$)', step, re.IGNORECASE)
        if compare_match:
            item1, item2 = compare_match.groups()
            return [
                f"Gather information about {item1}.",
                f"Gather information about {item2}.",
                f"Compare the information for {item1} and {item2}."
            ]
            
        # Pattern 4: Between X and Y pattern
        between_match = re.search(r'between\s+(.*?)\s+and\s+(.*?)(?:\.|$)', step, re.IGNORECASE)
        if between_match:
            item1, item2 = between_match.groups()
            return [
                f"Gather information about {item1}.",
                f"Gather information about {item2}.",
                f"Calculate relationship between {item1} and {item2}."
            ]
            
        # Pattern 5: City names specifically
        cities_match = re.search(r'\b(New York|Los Angeles|Chicago|Tokyo|London|Paris|Berlin|Beijing|Moscow|Sydney|Toronto)\b.*?\band\b.*?\b(New York|Los Angeles|Chicago|Tokyo|London|Paris|Berlin|Beijing|Moscow|Sydney|Toronto)\b', step, re.IGNORECASE)
        if cities_match:
            city1, city2 = cities_match.groups()
            return [
                f"Research information about {city1}.",
                f"Research information about {city2}.",
                f"Compare the information about {city1} and {city2}."
            ]
            
        # Default: If we can't specifically break it down but it seems complex,
        # create a generic breakdown
        return [
            f"Evaluate how to break down this task: {step}",
            f"Execute the first part of the task based on the evaluation.",
            f"Execute any remaining parts of the task."
        ]
    
    def reset_plan(self):
        """Reset all plan-related state variables"""
        # Before clearing plan state, persist important information to long-term memory
        if hasattr(self, 'memory_manager'):
            self.memory_manager.persist_important_memories()
            self.memory_manager.clear_working_memory()
            
        self.current_plan = None
        self.current_step_index = -1
        self.original_goal = None
        self.step_results = []
        
        # Reset refinement flags
        self.plan_needs_refinement = False
        self.refinement_issues = []
        
    def has_active_plan(self):
        """Check if there's an active plan being executed"""
        return self.current_plan is not None and self.current_step_index < len(self.current_plan)
    
    def is_plan_complete(self):
        """Check if the current plan has been fully executed"""
        return self.current_plan is not None and self.current_step_index >= len(self.current_plan)
    
    def current_step(self):
        """Get the current step instruction"""
        if not self.has_active_plan():
            return None
        return self.current_plan[self.current_step_index]
    
    def next_step(self):
        """Get the next step instruction"""
        if not self.has_active_plan() or self.current_step_index + 1 >= len(self.current_plan):
            return None
        return self.current_plan[self.current_step_index + 1]
    
    def store_step_result(self, step_instruction, step_result):
        """Store the result of a step execution"""
        result_text = f"Result of Step {self.current_step_index + 1} ('{step_instruction}'):\n{step_result}\n"
        self.step_results.append(result_text)
        
        # Store in memory manager for better recall and persistence
        if hasattr(self, 'memory_manager'):
            self.memory_manager.store_step_result(
                step_instruction=step_instruction,
                step_result=step_result,
                goal=self.original_goal
            )
        
    def advance_to_next_step(self):
        """Move to the next step in the plan"""
        if self.has_active_plan():
            self.current_step_index += 1
            
    def set_new_plan(self, plan, goal):
        """
        Set a new plan and the associated goal.
        Automatically evaluates and refines the plan to break down complex steps.
        """
        # Evaluate and potentially refine the plan before setting it
        evaluated_plan = self.evaluate_plan_complexity(plan)
        
        # If plan was refined, log the changes
        if evaluated_plan != plan:
            logger.info(f"Plan was automatically refined: {len(plan)} steps -> {len(evaluated_plan)} steps")
            # Log detailed changes
            for i, (original, refined) in enumerate(zip(plan, evaluated_plan[:len(plan)])):
                if original != refined:
                    logger.info(f"Step {i+1} changed from '{original}' to '{refined}'")
            # Log any new steps added to the end
            if len(evaluated_plan) > len(plan):
                for i in range(len(plan), len(evaluated_plan)):
                    logger.info(f"New step {i+1} added: '{evaluated_plan[i]}'")
        
        # Set the refined plan
        self.current_plan = evaluated_plan
        self.current_step_index = 0
        self.original_goal = goal
        self.step_results = []
        
        # Initialize working memory for this plan
        if hasattr(self, 'memory_manager'):
            self.memory_manager.clear_working_memory()
            # Store the plan context in working memory
            self.memory_manager.working_memory["context"] = {
                "goal": goal,
                "plan_steps": evaluated_plan,
                "start_time": time.time()
            }
        
    def generate_plan_summary_prompt(self):
        """Generate a prompt for summarizing all steps in a completed plan"""
        if not self.step_results:
            return None
            
        summary_prompt = f"The original goal was: '{self.original_goal}'. The following steps were executed with their results:\n\n"
        summary_prompt += ''.join(self.step_results)
        
        # Add relevant memories from the memory manager if available
        if hasattr(self, 'memory_manager'):
            memory_context = self.memory_manager.generate_memory_context(self.original_goal)
            if memory_context:
                summary_prompt += f"\n\nAdditional context from memory:\n{memory_context}"
                
        summary_prompt += "\n\nPlease provide a final comprehensive summary based on these results."
        
        return summary_prompt
        
    def generate_one_step_summary_prompt(self):
        """Generate a prompt for summarizing a one-step plan"""
        if not self.step_results or len(self.step_results) != 1:
            return None
            
        summary_prompt = f"The original goal was: '{self.original_goal}'. The plan had one step with this result:\n\n"
        summary_prompt += self.step_results[0]
        
        # Add relevant memories from the memory manager if available
        if hasattr(self, 'memory_manager'):
            memory_context = self.memory_manager.generate_memory_context(self.original_goal)
            if memory_context:
                summary_prompt += f"\n\nAdditional context from memory:\n{memory_context}"
                
        summary_prompt += "\n\nPlease provide a final comprehensive summary based on these results."
        
        return summary_prompt
    
    def extract_plan_from_llm_response(self, text):
        """
        Enhanced function to extract a plan from LLM response text.
        This extends the basic parse_plan_from_text functionality with
        additional plan detection capabilities specific to the research agent.
        
        Args:
            text (str): The LLM response text
            
        Returns:
            list: A list of plan steps if a plan is detected, None otherwise
        """
        from src.agent.utils import parse_plan_from_text
        
        # First try the standard parser
        plan = parse_plan_from_text(text)
        if plan:
            logger.info(f"Standard plan parser detected {len(plan)} steps")
            return plan
        
        # More aggressive parsing if standard parser fails
        logger.info("Standard plan parsing failed, attempting enhanced parsing")
        
        # Look for any kind of numbered or bulleted list
        patterns = [
            r"^\s*\d+[\.\)]\s+(.*?)(?:\n|$)",  # Numbered lists (1. or 1) format)
            r"^\s*[\*\-\•]\s+(.*?)(?:\n|$)",    # Bullet points (*, -, •)
            r"^\s*(?:Step|STEP)\s+\d+:?\s+(.*?)(?:\n|$)",  # "Step X:" format
            r"(?:First|Second|Third|Fourth|Fifth|Lastly|Finally),\s+(.*?)(?:\n|$)"  # Natural language steps
        ]
        
        for pattern in patterns:
            steps = re.findall(pattern, text, re.MULTILINE)
            if steps and len(steps) >= 2:
                logger.info(f"Enhanced plan parser detected {len(steps)} steps using pattern: {pattern}")
                return [step.strip() for step in steps if step.strip()]
        
        # Last attempt: check for research-specific patterns
        research_patterns = [
            r"(?:search|research|analyze|gather|collect|compile|investigate)(?:ing)?\s+(?:for|about|on)\s+(.*?)(?:\n|$)",
            r"(?:examine|extract|identify|explore)\s+(.*?)(?:\n|$)",
            r"(?:summarize|synthesize|conclude|report)\s+(.*?)(?:\n|$)"
        ]
        
        research_steps = []
        for pattern in research_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            research_steps.extend(matches)
            
        if len(research_steps) >= 2:
            logger.info(f"Extracted {len(research_steps)} research-oriented steps")
            return [step.strip() for step in research_steps if step.strip()]
            
        logger.info("No plan detected after all parsing attempts")
        return None
        
    def requires_detailed_info(self, query):
        """
        Evaluates whether a query requires a structured research approach.
        Returns True for complex queries that would benefit from planning.
        Delegates to the advanced QueryEvaluator which uses LLM inference.
        """
        # Use the query evaluator implementation
        from src.agent.query_evaluator import QueryEvaluator
        
        # Create an evaluator instance and use it
        evaluator = QueryEvaluator()
        return evaluator.requires_detailed_info(query)
        
    def is_one_step_plan(self):
        """
        Check if the plan consists of only a single step.
        
        Returns:
            bool: True if the plan has exactly one step, False otherwise
        """
        return self.current_plan is not None and len(self.current_plan) == 1
        
    def refine_plan(self, refined_steps):
        """
        Update the current plan with a refined set of steps.
        Keeps the original goal but replaces remaining steps with refined ones.
        """
        if self.current_plan is None:
            logger.warning("Cannot refine plan: no active plan exists")
            return False
            
        # Replace remaining steps with refined steps
        self.current_plan = self.current_plan[:self.current_step_index] + refined_steps
        
        # Reset refinement flags
        self.plan_needs_refinement = False
        self.refinement_issues = []
        
        logger.info(f"Plan refined successfully with {len(refined_steps)} new steps")
        return True
        
    def get_executed_steps_with_results(self):
        """
        Get a list of all executed steps with their results
        
        Returns:
            list: A list of dictionaries with step instructions and results
        """
        executed_steps = []
        
        for i in range(min(self.current_step_index, len(self.step_results))):
            if i < len(self.current_plan) and i < len(self.step_results):
                executed_steps.append({
                    "step_number": i + 1,
                    "instruction": self.current_plan[i],
                    "result": self.step_results[i]
                })
                
        return executed_steps
    
    def _create_step_evaluation_prompt(self, step):
        """
        Creates a prompt for the LLM to evaluate if a step is complex and needs to be broken down.
        
        Args:
            step (str): The step to evaluate
            
        Returns:
            str: The prompt for the LLM
        """
        prompt = f"""Evaluate if the following research plan step is complex and should be broken down into simpler subtasks:

Step: "{step}"

A complex step typically:
1. Contains multiple distinct operations that could be done sequentially
2. Involves gathering or analyzing data about multiple distinct entities
3. Combines search/gathering tasks with analysis/calculation tasks
4. Requires multiple function calls to complete

Please respond in the following format:
- COMPLEX: [Yes/No]
- REASON: [Brief explanation of why the step is or is not complex]
"""
        return prompt
        
    def _parse_step_evaluation(self, evaluation_text):
        """
        Parses the LLM's evaluation of a step's complexity.
        
        Args:
            evaluation_text (str): The LLM's response to the evaluation prompt
            
        Returns:
            tuple: (is_complex, reason)
        """
        # Default values
        is_complex = False
        reason = "Could not determine complexity"
        
        # Check for "COMPLEX: Yes" pattern
        complex_match = re.search(r'COMPLEX:\s*(Yes|No)', evaluation_text, re.IGNORECASE)
        if complex_match and complex_match.group(1).lower() == 'yes':
            is_complex = True
        
        # Extract the reason
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()
            
        return is_complex, reason
        
    def _create_step_breakdown_prompt(self, step, reason):
        """
        Creates a prompt for the LLM to break down a complex step.
        
        Args:
            step (str): The complex step to break down
            reason (str): The reason why the step is complex
            
        Returns:
            str: The prompt for the LLM
        """
        prompt = f"""Break down the following complex research plan step into 2-5 simpler sequential subtasks that together accomplish the same goal:

Complex step: "{step}"

This step is complex because: {reason}

Guidelines for breaking it down:
- Each subtask should be a separate, distinct operation
- Include all necessary information in each subtask
- Make sure subtasks can be executed sequentially
- Each subtask should be focused on a single task/entity
- Don't use "step 1", "step 2" prefixes
- Make sure each subtask is clear, detailed, and can be executed without requiring further clarification

List your breakdown in this format:
SUBTASKS:
1. [First subtask]
2. [Second subtask]
...etc.
"""
        return prompt
        
    def _parse_step_breakdown(self, breakdown_text):
        """
        Parses the LLM's breakdown of a complex step.
        
        Args:
            breakdown_text (str): The LLM's response to the breakdown prompt
            
        Returns:
            list: The broken-down steps, or None if parsing failed
        """
        # Look for list items with numbers or bullet points
        patterns = [
            r'\d+\.\s*(.+?)(?=\d+\.|$)', # "1. task"  
            r'[-*•]\s*(.+?)(?=[-*•]|$)'  # "- task" or "* task" or "• task"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, breakdown_text, re.DOTALL)
            if matches:
                # Clean up the matches
                steps = [match.strip() for match in matches if match.strip()]
                if len(steps) > 1:  # Ensure we have at least 2 subtasks
                    return steps
        
        # If no structured list was found, try to find paragraph breaks
        paragraphs = [p.strip() for p in breakdown_text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
            
        # Look for "SUBTASKS:" followed by text
        subtasks_match = re.search(r'SUBTASKS:(.*?)(?:\n\n|$)', breakdown_text, re.IGNORECASE | re.DOTALL)
        if subtasks_match:
            subtask_text = subtasks_match.group(1).strip()
            return [line.strip() for line in subtask_text.split('\n') if line.strip()]
            
        # If all else fails, return None to indicate failure
        logger.warning("Failed to parse step breakdown response")
        return None
