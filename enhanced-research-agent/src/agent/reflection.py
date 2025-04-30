"""
Self-reflection and result evaluation module for the Enhanced Research Agent.
Enables the agent to evaluate its progress, check result quality, and refine its approach.
"""

import google.generativeai as genai
from src.logger.agent_logger import setup_logger
from src.logger.interaction_logger import log_reflection, log_refinement

# Setup logger for this module
logger = setup_logger(logger_name="agent_reflection")

class ReflectionEvaluator:
    """
    Provides self-reflection and evaluation capabilities to enable the agent to 
    assess the quality of results and refine its approach when needed.
    """
    
    def __init__(self):
        """Initialize the reflection evaluator with a Gemini model"""
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        logger.info("ReflectionEvaluator initialized")
        
    def evaluate_result(self, goal, step_description, result):
        """
        Evaluate if a step's result adequately addresses the requirement.
        
        Args:
            goal: The original goal or query
            step_description: Description of the current plan step
            result: The result produced by executing the step
            
        Returns:
            A dictionary containing evaluation results:
            - is_adequate: Whether the result adequately addresses the step (True/False)
            - issues: List of identified issues, if any
            - recommendation: Recommended action ("proceed", "retry", "refine_plan")
            - justification: Explanation of the evaluation
        """
        # Skip reflection for query evaluation or plan creation steps
        if "Create a detailed research plan" in step_description:
            logger.info(f"Skipping reflection for planning step: {step_description[:50]}...")
            return self._create_default_evaluation("proceed")
            
        logger.info(f"Evaluating result for step: {step_description[:50]}...")
        
        # Create the evaluation prompt
        evaluation_prompt = f"""
As an expert evaluator, assess whether the following result adequately addresses the requirement.

GOAL: {goal}

STEP: {step_description}

RESULT: {result}

Analyze whether the result:
1. Directly addresses the step requirement
2. Contains the necessary information
3. Is accurate and reliable
4. Is complete enough to proceed

Ignore any issues that are not relevant to meeting the step's objectives.

Provide your assessment in the following format:
- ADEQUATE: [Yes/No]
- ISSUES: [List any issues identified, or "None" if adequate]
- RECOMMENDATION: ["proceed" if adequate, "retry" if the step should be retried, or "refine_plan" if the plan needs adjustment]
- JUSTIFICATION: [Brief explanation of your evaluation]
"""

        try:
            # Get the model's evaluation
            response = self.model.generate_content(evaluation_prompt)
            if not response.text:
                logger.warning("Empty evaluation response")
                return self._create_default_evaluation("proceed")
                
            evaluation_text = response.text
            logger.debug(f"Raw evaluation: {evaluation_text}")
            
            # Parse the evaluation response
            evaluation = self._parse_evaluation_response(evaluation_text)
            
            # Log the reflection
            log_reflection(goal, step_description, result, evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error during result evaluation: {e}")
            # Return default evaluation as fallback
            return self._create_default_evaluation("proceed")
    
    def generate_step_refinement(self, goal, step_description, current_result, issues):
        """
        Generate a refined approach for a step when the current result is inadequate.
        
        Args:
            goal: The original goal or query
            step_description: Description of the current plan step
            current_result: The result that was deemed inadequate
            issues: Identified issues with the current result
            
        Returns:
            A dictionary containing the refinement plan:
            - refined_step: The refined step description
            - reasoning: Explanation of the refinement
        """
        logger.info(f"Generating refinement for step: {step_description[:50]}...")
        
        # Create the refinement prompt
        refinement_prompt = f"""
The following step execution did not produce adequate results for the given goal:

GOAL: {goal}

ORIGINAL STEP: {step_description}

CURRENT RESULT: {current_result}

IDENTIFIED ISSUES: {issues}

Please generate a refined approach for this step that addresses the identified issues.
Consider:
1. More specific search terms or parameters
2. Different tools or functions that might be more appropriate
3. Breaking the step into smaller sub-steps
4. Additional context or constraints that should be considered

Provide your refinement in the following format:
- REFINED STEP: [Clear instruction for the refined step]
- REASONING: [Brief explanation of why this refinement should work better]
"""

        try:
            # Get the model's refinement plan
            response = self.model.generate_content(refinement_prompt)
            if not response.text:
                logger.warning("Empty refinement response")
                return self._create_default_refinement(step_description)
                
            refinement_text = response.text
            logger.debug(f"Raw refinement: {refinement_text}")
            
            # Parse the refinement response
            refinement = self._parse_refinement_response(refinement_text)
            
            # Log the refinement
            log_refinement(step_description, refinement, issues)
            
            return refinement
            
        except Exception as e:
            logger.error(f"Error during step refinement: {e}")
            # Return default refinement as fallback
            return self._create_default_refinement(step_description)
    
    def generate_plan_refinement(self, goal, current_plan, executed_steps, issues):
        """
        Generate a refined overall plan when the current plan is inadequate.
        
        Args:
            goal: The original goal or query
            current_plan: The current plan steps
            executed_steps: Steps already executed with their results
            issues: Identified issues with the current plan
            
        Returns:
            A list of refined plan steps
        """
        logger.info("Generating refined plan...")
        
        # Format the executed steps and their results
        executed_steps_text = ""
        for i, (step, result) in enumerate(executed_steps, 1):
            executed_steps_text += f"Step {i}: {step}\nResult: {result}\n\n"
        
        # Format the remaining plan steps
        remaining_steps_text = ""
        if len(current_plan) > len(executed_steps):
            for i, step in enumerate(current_plan[len(executed_steps):], len(executed_steps) + 1):
                remaining_steps_text += f"Step {i}: {step}\n"
        else:
            remaining_steps_text = "No remaining steps."
        
        # Create the plan refinement prompt
        refinement_prompt = f"""
The current plan for addressing the goal needs refinement:

GOAL: {goal}

EXECUTED STEPS AND RESULTS:
{executed_steps_text}

REMAINING PLANNED STEPS:
{remaining_steps_text}

IDENTIFIED ISSUES: {issues}

Please generate a refined plan that addresses the identified issues.
You can:
1. Modify remaining steps
2. Add new steps
3. Remove unnecessary steps
4. Keep good steps from the original plan

The new plan should be presented as a numbered list, starting from step {len(executed_steps) + 1}.
"""

        try:
            # Get the model's refined plan
            response = self.model.generate_content(refinement_prompt)
            if not response.text:
                logger.warning("Empty plan refinement response")
                return current_plan[len(executed_steps):]  # Return remaining original steps as fallback
                
            refined_plan_text = response.text
            logger.debug(f"Raw plan refinement: {refined_plan_text}")
            
            # Parse the refined plan
            from src.agent.utils import parse_plan_from_text
            refined_steps = parse_plan_from_text(refined_plan_text)
            
            if not refined_steps or len(refined_steps) == 0:
                logger.warning("Failed to parse refined plan steps")
                return current_plan[len(executed_steps):]  # Return remaining original steps as fallback
            
            logger.info(f"Generated {len(refined_steps)} refined plan steps")
            return refined_steps
            
        except Exception as e:
            logger.error(f"Error during plan refinement: {e}")
            # Return remaining original steps as fallback
            return current_plan[len(executed_steps):]
    
    def _parse_evaluation_response(self, evaluation_text):
        """Parse the structured evaluation response from the LLM"""
        lines = evaluation_text.strip().split("\n")
        
        evaluation = {
            "is_adequate": False,
            "issues": [],
            "recommendation": "proceed",  # Default to proceed
            "justification": ""
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("- ADEQUATE:") or line.startswith("ADEQUATE:"):
                value = line.split(":", 1)[1].strip().lower()
                evaluation["is_adequate"] = "yes" in value
                
            elif line.startswith("- ISSUES:") or line.startswith("ISSUES:"):
                issues_text = line.split(":", 1)[1].strip()
                if "none" not in issues_text.lower():
                    # Split by commas or bullet points
                    if "," in issues_text:
                        evaluation["issues"] = [issue.strip() for issue in issues_text.split(",")]
                    elif "-" in issues_text:
                        evaluation["issues"] = [issue.strip() for issue in issues_text.split("-") if issue.strip()]
                    else:
                        evaluation["issues"] = [issues_text]
                        
            elif line.startswith("- RECOMMENDATION:") or line.startswith("RECOMMENDATION:"):
                rec = line.split(":", 1)[1].strip().lower()
                if "retry" in rec:
                    evaluation["recommendation"] = "retry"
                elif "refine_plan" in rec or "refine plan" in rec:
                    evaluation["recommendation"] = "refine_plan"
                else:
                    evaluation["recommendation"] = "proceed"
                    
            elif line.startswith("- JUSTIFICATION:") or line.startswith("JUSTIFICATION:"):
                evaluation["justification"] = line.split(":", 1)[1].strip()
        
        # Add safety checks
        if evaluation["recommendation"] != "proceed" and not evaluation["issues"]:
            evaluation["issues"] = ["Unspecified issue requiring action"]
            
        return evaluation
    
    def _parse_refinement_response(self, refinement_text):
        """Parse the structured refinement response from the LLM"""
        lines = refinement_text.strip().split("\n")
        
        refinement = {
            "refined_step": "",
            "reasoning": ""
        }
        
        refined_step_lines = []
        reasoning_lines = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("- REFINED STEP:") or line.startswith("REFINED STEP:"):
                current_section = "refined_step"
                refined_step = line.split(":", 1)[1].strip()
                if refined_step:  # Only add if there's content after the header
                    refined_step_lines.append(refined_step)
                    
            elif line.startswith("- REASONING:") or line.startswith("REASONING:"):
                current_section = "reasoning"
                reasoning = line.split(":", 1)[1].strip()
                if reasoning:  # Only add if there's content after the header
                    reasoning_lines.append(reasoning)
                    
            elif current_section == "refined_step":
                refined_step_lines.append(line)
                
            elif current_section == "reasoning":
                reasoning_lines.append(line)
        
        refinement["refined_step"] = " ".join(refined_step_lines)
        refinement["reasoning"] = " ".join(reasoning_lines)
        
        return refinement
    
    def _create_default_evaluation(self, recommendation="proceed"):
        """Create a default evaluation when parsing fails"""
        return {
            "is_adequate": recommendation == "proceed",
            "issues": [] if recommendation == "proceed" else ["Unable to determine specific issues"],
            "recommendation": recommendation,
            "justification": "Default evaluation due to processing limitation."
        }
    
    def _create_default_refinement(self, original_step):
        """Create a default refinement when parsing fails"""
        return {
            "refined_step": f"Try again with more specific parameters: {original_step}",
            "reasoning": "Default refinement due to processing limitation."
        }
