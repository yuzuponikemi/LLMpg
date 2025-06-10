# RAG Testing Guide

This document explains the different test scripts available for testing the Retrieval-Augmented Generation (RAG) capabilities of the Enhanced Research Agent.

## Available Test Scripts

### 1. Basic RAG Tests
- `test_rag_basic.py`: Tests the basic functionality of Qdrant with Google embeddings directly, without using our custom RAG tools.
- `test_rag_direct_simple.py`: Tests our custom RAG tools (Google and OpenAI embeddings) directly, bypassing the agent.
- `test_rag_google_tool.py`: A focused test for the Google embedding RAG tool with in-memory test data.

### 2. Agent Integration Tests
- `test_agent_rag_integration.py`: Tests the RAG capabilities through direct interaction with the agent in an end-to-end scenario using the new eval mode.
- `test_agent_eval_mode.py`: Demonstrates how to use the agent's eval mode for automated testing.
- `test_rag_eval_mode.py`: Comprehensive RAG evaluation using the agent's eval mode to test indexing and querying capabilities.

## Test Data

The test scripts use test data located in the `test/rag_test_data` directory. These files contain sample text on various topics like machine learning, quantum computing, and natural language processing.

## Running the Tests

### Prerequisites
- Ensure you have all required dependencies installed: `pip install -r requirements.txt`
- Make sure your `.env` file contains the necessary API keys:
  - `GOOGLE_API_KEY` for Google embeddings
  - `OPENAI_API_KEY` for OpenAI embeddings (optional)
- Ensure Qdrant is running (typically on localhost:6333)

### Step 1: Basic Tests

First, run the basic tests to ensure the underlying RAG functionality works correctly:

```bash
# Test the basic Qdrant functionality with Google embeddings
python test/test_rag_basic.py

# Test our custom RAG tools directly
python test/test_rag_direct_simple.py

# Test the Google embedding RAG tool specifically
python test/test_rag_google_tool.py
```

### Step 2: Agent Integration Test

Once the basic tests pass, test the integration with the agent:

```bash
# Test RAG capabilities through the agent using eval mode
python test/test_agent_rag_integration.py

# Run comprehensive RAG evaluation tests using the agent's eval mode
python test/test_rag_eval_mode.py
```

You can also use the provided batch script for Windows:

```
run_rag_eval_tests.bat
```

### Step 3: Demo Example

To see a full demonstration of the RAG capabilities:

```bash
# Run the RAG demo
python examples/rag_demo.py
```

## Using the Eval Mode

The agent now includes an "eval mode" that allows external programs to send test prompts and capture the first response for evaluation. This is useful for automated testing and monitoring.

### Running the Agent in Eval Mode

```bash
# Run the agent in eval mode with a specific query
python main.py --eval --query "Your test query here" --output "results.json"
```

### Using the Test Agent Eval Mode Script

The `test_agent_eval_mode.py` script provides a convenient way to run multiple eval tests:

```bash
# Run a single eval test
python test/test_agent_eval_mode.py --query "Your test query here"

# Run multiple eval tests from a file
python test/test_agent_eval_mode.py --file "test_queries.txt"
```

### Using the RAG Eval Mode Script

The `test_rag_eval_mode.py` script provides a comprehensive test suite for RAG capabilities:

```bash
# Run the RAG eval mode tests with the default collection name
python test/test_rag_eval_mode.py

# Run the tests with a custom collection name
python test/test_rag_eval_mode.py --collection "my_test_collection"
```

This script:
1. Indexes the test documents from `test/rag_test_data` into a collection
2. Runs queries targeting each document type (machine learning, quantum computing, NLP)
3. Evaluates the responses to verify that the agent correctly retrieves and uses the information
4. Saves detailed results to the `test/test_results` directory

## Test Results

- Test results for the agent integration test are saved in the `test/test_results` directory.
- Each test run creates a timestamped JSON file with the test results.
- Eval mode results are saved as JSON files that include both the query and the agent's response.
- RAG eval mode tests generate a comprehensive evaluation report in JSON format.

## Troubleshooting

- If you encounter errors related to missing collections or empty results, ensure Qdrant is running.
- If embedding generation fails, check your API keys in the `.env` file.
- If the agent doesn't respond correctly, check the logs in the `logs` directory.
- If you see API errors with Unicode characters, these are usually harmless and handled by the eval mode's error recovery.

## Additional Resources

- See the `docs/rag_capabilities.md` file for more information on the RAG capabilities.
- Check the `examples/rag_demo.py` file for a full demonstration of the RAG capabilities.
