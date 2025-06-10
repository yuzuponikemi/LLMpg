"""
RAG Integration Test with Main Agent

This script tests the RAG capabilities through the main agent,
demonstrating how RAG tools are used in the context of a complete agent session.
"""

import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

# Import agent and necessary components
from src.agent.core import ResearchAgent
from src.logger.agent_logger import setup_logger
from src.agent.memory_manager import MemoryManager
from src.agent.model_manager import ModelManager

# Set up logging
logger = setup_logger(logger_name="rag_agent_test")

def prepare_test_data():
    """Prepare test documents in a directory for RAG indexing"""
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(__file__), "rag_agent_test_data")
    os.makedirs(test_dir, exist_ok=True)
    
    # Test documents
    test_documents = [
        {
            "filename": "machine_learning_concepts.txt",
            "content": """Machine Learning Concepts

Machine learning is a field of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. 

Key machine learning paradigms include:
1. Supervised Learning: Training on labeled data to predict outcomes for unseen data.
2. Unsupervised Learning: Finding patterns and relationships in unlabeled data.
3. Reinforcement Learning: Learning through interaction with an environment to maximize rewards.

Common machine learning algorithms include:
- Linear Regression: For predicting continuous values
- Logistic Regression: For binary classification problems
- Decision Trees: Tree-like model of decisions
- Random Forests: Ensemble of decision trees
- Support Vector Machines: For classification and regression
- Neural Networks: Inspired by the human brain

Machine learning is applied in various domains including healthcare, finance, 
natural language processing, computer vision, and recommendation systems."""
        },
        {
            "filename": "ai_ethics.txt",
            "content": """AI Ethics and Responsible AI Development

AI ethics is the branch of ethics that focuses on the moral issues surrounding artificial intelligence systems.

Key ethical considerations in AI include:
1. Fairness and Bias: Ensuring AI systems don't discriminate against certain groups.
2. Transparency and Explainability: Making AI decision-making processes understandable.
3. Privacy: Protecting personal data used to train and operate AI systems.
4. Accountability: Determining responsibility when AI systems cause harm.
5. Security: Preventing AI systems from being manipulated or misused.

Responsible AI development practices include:
- Diverse and representative training data
- Regular auditing for bias and fairness issues
- Documentation of model limitations and intended uses
- Human oversight of AI systems
- Ongoing monitoring of deployed systems

Organizations like the IEEE, Partnership on AI, and various governmental bodies 
have proposed guidelines and frameworks for ethical AI development and deployment."""
        },
        {
            "filename": "data_science_workflow.txt",
            "content": """The Data Science Workflow

A typical data science workflow consists of several key stages that transform raw data into actionable insights.

The main stages of the data science workflow are:
1. Problem Definition: Clearly defining the business problem to be solved.
2. Data Collection: Gathering relevant data from various sources.
3. Data Cleaning: Handling missing values, outliers, and inconsistencies.
4. Exploratory Data Analysis (EDA): Understanding the data through visualization and statistical analysis.
5. Feature Engineering: Creating new features or transforming existing ones to improve model performance.
6. Model Selection: Choosing appropriate algorithms based on the problem type.
7. Model Training: Teaching the model to recognize patterns in the data.
8. Model Evaluation: Assessing model performance using appropriate metrics.
9. Model Deployment: Integrating the model into production systems.
10. Monitoring and Maintenance: Tracking model performance and updating as needed.

Tools commonly used throughout this workflow include:
- Programming languages: Python, R
- Data manipulation: Pandas, NumPy
- Visualization: Matplotlib, Seaborn, Plotly
- Machine learning: Scikit-learn, TensorFlow, PyTorch
- Big data processing: Spark, Hadoop
- Version control: Git
- Deployment: Docker, Kubernetes, MLflow"""
        }
    ]
    
    # Write test documents to files
    for doc in test_documents:
        file_path = os.path.join(test_dir, doc["filename"])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
    
    logger.info(f"Created {len(test_documents)} test documents in {test_dir}")
    return test_dir

def simulate_agent_conversation(agent, queries):
    """Simulate a conversation with the agent using a list of queries"""
    
    logger.info("Starting simulated conversation with agent")
    
    responses = []
    for i, query in enumerate(queries):
        logger.info(f"Query {i+1}: {query}")
        
        # Process the query
        agent_response = agent.process_query(query)
        
        # Record the response
        if isinstance(agent_response, tuple):
            # For newer agent versions that return (plan, text_response, response)
            text_response = agent_response[1]
        else:
            # For older agent versions that return just the text
            text_response = agent_response
            
        responses.append({
            "query": query,
            "response": text_response
        })
        
        logger.info(f"Agent responded to query {i+1}")
        
        # Add a short delay between queries to avoid rate limiting
        time.sleep(1)
    
    return responses

def main():
    """Main function to run the RAG agent integration test"""
    
    print("=== RAG Agent Integration Test ===")
    
    # Prepare test data
    print("Preparing test data...")
    test_data_dir = prepare_test_data()
    
    # Initialize the agent
    print("Initializing research agent...")
    agent = ResearchAgent()
    
    # RAG-specific queries to test
    rag_queries = [
        # First, index the test data
        f"Please index the documents in the directory '{test_data_dir}' into a collection called 'test_agent_rag' using Google embeddings.",
        
        # Query the indexed data
        "Using the 'test_agent_rag' collection, what are the different types of machine learning?",
        
        # More complex queries that combine RAG with other capabilities
        "Using the 'test_agent_rag' collection, what are the key ethical considerations in AI development? Then create a brief summary.",
        
        # Query that requires combining information from multiple documents
        "Using the 'test_agent_rag' collection, how might ethical considerations impact the data science workflow? Provide a thoughtful analysis."
    ]
    
    # Run the simulated conversation
    print("\nStarting simulated conversation...")
    responses = simulate_agent_conversation(agent, rag_queries)
    
    # Save the results
    results_dir = os.path.join(os.path.dirname(__file__), "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"rag_agent_test_{timestamp}.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2)
    
    print(f"\nTest completed. Results saved to: {results_file}")
    print("\n=== RAG Agent Integration Test Complete ===")

if __name__ == "__main__":
    main()
