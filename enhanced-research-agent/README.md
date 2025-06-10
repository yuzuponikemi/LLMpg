# Enhanced Research Agent

A powerful AI research assistant capable of web searches, webpage browsing, code execution, file operations, and RAG (Retrieval-Augmented Generation) capabilities.

## Features

- **Web Search**: Find information from across the internet using DuckDuckGo
- **Web Browsing**: Extract content from specific web pages
- **Code Execution**: Run Python code safely in a controlled environment
- **File Operations**: Read and write files, list directory contents
- **Advanced Planning**: Break complex queries into step-by-step plans
- **Data Analysis**: Process, analyze, and visualize data from various sources
- **RAG Capabilities**: Retrieve information from vector databases using semantic search
  - Create searchable knowledge bases from documents and code
  - Query collections using Google or OpenAI embeddings
  - Custom embedding models for different document types
- **Evaluation Mode**: Run the agent with specific test prompts and capture the first response for automated testing

## Project Structure

```
enhanced-research-agent/
├── src/               # Source code
│   ├── agent/         # Core agent functionality
│   ├── tools/         # Tool implementations
│   └── logger/        # Logging functionality
├── data/              # Sample data files
├── logs/              # Log output files
├── examples/          # Example scripts
├── test/              # Test scripts and data
│   ├── rag_test_data/ # Test data for RAG tests
│   └── test_results/  # Test results
├── main.py            # Main entry point
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd enhanced-research-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Google API key: `GOOGLE_API_KEY=your-api-key`
   - Optionally add your OpenAI API key for OpenAI embeddings: `OPENAI_API_KEY=your-api-key`

5. Set up Qdrant (for RAG capabilities):
   - Using Docker:
     ```
     docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_data:/qdrant/storage \
       qdrant/qdrant
     ```
   - Or download from the [Qdrant website](https://qdrant.tech/documentation/quick-start/)

## Usage

### Interactive Mode

Run the agent in interactive mode:

```
python main.py
```

This starts a conversation where you can ask questions and the agent will respond.

### File Input Mode

Run the agent with inputs from a file:

```
python main.py -f inputs.txt
```

The file should contain one query per line.

### Evaluation Mode

Run the agent in evaluation mode to test with a specific prompt and capture only the first response:

```
python main.py --eval --query "Your test query here" --output "results.json"
```

This is useful for automated testing and benchmarking.

### Example Queries

The agent can handle a wide range of queries, including:

1. **Simple Information Requests**:
   ```
   What is the capital of France?
   ```

2. **Code Execution**:
   ```
   Calculate the sum of numbers from 1 to 100
   ```

3. **File Operations**:
   ```
   List the files in the data directory
   Read the sample_data.csv file
   ```

4. **Data Analysis**:
   ```
   Calculate the average temperature in sample_data.csv
   Create a visualization of the temperature and humidity data
   ```

5. **Complex Research**:
   ```
   Explain how transformer models work in natural language processing
   Compare different sorting algorithms and their time complexities
   ```

6. **RAG Queries**:
   ```
   Index the documents in the directory 'docs' into a collection called 'my_docs'
   Using the 'my_docs' collection, what are the key concepts of machine learning?
   ```

## Advanced Features

### Multi-step Planning

For complex queries, the agent automatically creates and executes multi-step plans:

1. The agent identifies when a query requires detailed research
2. It breaks down the task into logical steps
3. Each step is executed in sequence, with results from previous steps available for context
4. A final summary synthesizes information from all steps

### Tool Integration

The agent can use the following tools:

- `search_duckduckgo`: Web search using DuckDuckGo
- `browse_webpage`: Extract content from a specified URL
- `execute_code`: Run Python code in a sandboxed environment
- `read_file`: Access file contents
- `write_file`: Save data to a file
- `list_files`: Get directory listings
- `query_google_rag`: Retrieve information using Google embeddings
- `query_openai_rag`: Retrieve information using OpenAI embeddings
- `index_rag_collection`: Create knowledge bases from documents

### Automated Testing

The agent supports an evaluation mode that allows external programs to:
- Send test prompts
- Capture only the first response (without executing multi-step plans)
- Save results in JSON format for analysis

See the `test/test_agent_eval_mode.py` script for examples of how to use this feature.

## Examples

See the `examples/` directory for example scripts demonstrating various capabilities.

## Testing

See the `test/RAG_TESTING_GUIDE.md` for information on testing the RAG capabilities.

## Logging

Log files are stored in the `logs/` directory with timestamps in their filenames.

## License

[Specify your license here]
