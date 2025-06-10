# Using RAG Capabilities in the Enhanced Research Agent

This document explains how to use the Retrieval-Augmented Generation (RAG) capabilities of the Enhanced Research Agent. The agent can now create, index, and query vector databases using Qdrant, with support for multiple embedding models.

## Prerequisites

1. **Qdrant Server**: You need a running Qdrant server. The easiest way is to use Docker:

   ```bash
   docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   ```

2. **API Keys**: Depending on which embedding model you want to use, you'll need:
   - Google AI API key (required for Google embeddings)
   - OpenAI API key (optional, for OpenAI embeddings)

   Add these to your `.env` file:
   ```
   GOOGLE_API_KEY=your-google-api-key
   OPENAI_API_KEY=your-openai-api-key
   ```

## RAG Architecture

The RAG implementation consists of the following components:

1. **BaseRAGTool**: Abstract base class that handles connection to Qdrant and provides common functionality
2. **GoogleEmbeddingRAGTool**: Implementation using Google's text embeddings
3. **OpenAIEmbeddingRAGTool**: Implementation using OpenAI's text embeddings
4. **Indexing Utilities**: Functions to index files and directories

## Available Tools

The agent exposes three main tools for RAG functionality:

1. **index_rag_collection**: Index files from a directory into a Qdrant collection
2. **query_google_rag**: Query a collection using Google embeddings
3. **query_openai_rag**: Query a collection using OpenAI embeddings

## Example Usage

### Indexing Documents

You can create a knowledge base by indexing a directory of files:

```
# Index a directory of markdown files using Google embeddings
I need to index the documentation files in the 'docs' directory into a Qdrant collection called 'documentation' using Google embeddings.
```

The agent will use the `index_rag_collection` tool:

```python
index_rag_collection(
    directory_path="path/to/docs",
    collection_name="documentation",
    embedding_type="google",
    file_pattern="*.md",
    recursive=True
)
```

### Querying Collections

Once you have indexed documents, you can query them:

```
# Query the documentation collection for information about RAG
Find information about Retrieval-Augmented Generation in the 'documentation' collection.
```

The agent will use the `query_google_rag` or `query_openai_rag` tool:

```python
query_google_rag(
    collection_name="documentation",
    query="What is Retrieval-Augmented Generation?",
    limit=5,
    score_threshold=0.7
)
```

## Creating Custom Collections

You can create different collections for different types of documents:

1. **Documentation Collection**: For user guides, tutorials, etc.
2. **Code Collection**: For code snippets and examples
3. **Research Papers Collection**: For academic papers
4. **Project-Specific Collections**: For project-specific knowledge

Each collection can use a different embedding model optimized for its content type.

## Advanced Usage

### Custom Chunking

When indexing larger documents, they are split into chunks with some overlap. You can customize this:

```python
index_rag_collection(
    directory_path="path/to/large_docs",
    collection_name="large_documents",
    chunk_size=1500,  # Larger chunks
    overlap=300       # More overlap
)
```

### Filtering by Score

When retrieving documents, you can set a threshold to only return highly relevant results:

```python
query_google_rag(
    collection_name="documentation",
    query="What is RAG?",
    score_threshold=0.8  # Only return documents with similarity score ≥ 0.8
)
```

## Extending with New Embedding Models

You can create new RAG tool implementations by extending the `BaseRAGTool` class and implementing the `embed_text` and `embed_batch` methods. This allows you to use any embedding model of your choice.

Example for a hypothetical custom embedding model:

```python
class CustomEmbeddingRAGTool(BaseRAGTool):
    def __init__(self, collection_name, **kwargs):
        # Initialize your custom embedding model here
        self.model = load_custom_model()
        super().__init__(collection_name=collection_name, **kwargs)
    
    def embed_text(self, text):
        # Generate embeddings using your custom model
        return self.model.embed(text)
    
    def embed_batch(self, texts):
        # Generate batch embeddings
        return [self.model.embed(text) for text in texts]
```

## Troubleshooting

1. **Connection Issues**: Make sure Qdrant is running and accessible at the specified host and port
2. **API Key Errors**: Verify that your API keys are correctly set in the `.env` file
3. **Embedding Errors**: Check that you have the necessary permissions for the embedding models
4. **Empty Results**: Try lowering the `score_threshold` or using a more specific query
