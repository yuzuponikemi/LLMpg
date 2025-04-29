"""
Advanced query complexity evaluator using LLM inference
"""
import google.generativeai as genai

# Import our agent logger
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="query_evaluator")

class QueryEvaluator:
    """
    Uses LLM to evaluate whether a query requires a structured research approach.
    """
    
    def __init__(self):
        """Initialize the query evaluator with a Gemini model"""
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
    def requires_detailed_info(self, query):
        """
        Use the LLM itself to evaluate whether a query requires a structured research approach.
        This implementation is designed for scientific and coding-oriented research.
        """
        # Craft a prompt that asks the model to evaluate the query with a focus on scientific/coding research
        evaluation_prompt = f"""
Evaluate this query: "{query}"

Assess whether this query requires a structured research approach with multiple steps.

Answer YES if the query exhibits these characteristics:
- Requires understanding complex systems, algorithms, or scientific concepts
- Involves technical implementation details or code architecture decisions
- Calls for comparing alternative approaches or methodologies
- Needs investigation of documentation, APIs, or scientific literature
- Would benefit from examples, code samples, or experimental data
- Involves debugging, troubleshooting, or understanding error conditions
- Requires synthesizing information from multiple sources or domains
- Involves data analysis, data processing, or visualization tasks
- Would need calculations or statistical analysis of numeric data

Answer NO if the query:
- Seeks basic factual information about a well-defined topic
- Requests a simple definition of a concept or term
- Can be completely answered with a single, straightforward response
- Does not require context from multiple sources
- Is a simple arithmetic calculation or conversion

Think like a scientist or software engineer - would you approach this question with a methodical research plan or answer it immediately?

Answer with ONLY 'YES' or 'NO'.
"""

        try:
            # Get the model's evaluation
            response = self.model.generate_content(evaluation_prompt)
            if response.text:
                evaluation = response.text.strip().upper()
                logger.info(f"Query evaluation result: {evaluation} for query: {query}")
                return evaluation == "YES"
            return False
        except Exception as e:
            logger.error(f"Error evaluating query complexity: {e}")
            # Fall back to simple keyword matching if the LLM evaluation fails
            query_lower = query.lower()
            # Keywords focused on scientific/coding research needs
            detail_indicators = [
                'how to implement', 'architecture', 'design pattern', 
                'optimize', 'debug', 'troubleshoot', 'compare', 
                'analyze', 'research', 'investigate', 'methodology',
                'algorithm', 'complexity', 'performance', 'alternative approach',
                'visualization', 'data analysis', 'csv', 'calculate', 'average'
            ]
            
            return any(indicator in query_lower for indicator in detail_indicators)
