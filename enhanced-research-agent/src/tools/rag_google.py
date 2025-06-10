"""
RAG tool implementation using Google's text embeddings
"""

import os
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration

from src.tools.rag_base import BaseRAGTool
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="google_rag_tool")

class GoogleEmbeddingRAGTool(BaseRAGTool):
    """
    RAG tool implementation using Google's text embeddings.
    
    This class extends BaseRAGTool and implements the embedding methods
    using Google's text embedding models.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "models/embedding-001",
        host: str = "localhost",
        port: int = 6333,
        **kwargs
    ):
        """
        Initialize the Google embedding-based RAG tool.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Google embedding model name
            host: Qdrant server hostname
            port: Qdrant HTTP port
            **kwargs: Additional arguments to pass to BaseRAGTool
        """
        # Google's text embedding model produces 768-dimensional vectors
        self.embedding_model = embedding_model
        
        # Determine vector size based on model
        if "embedding-001" in embedding_model:
            vector_size = 768
        else:
            # Default size, can be overridden in kwargs
            vector_size = kwargs.get("vector_size", 768)
        
        # Initialize the base class
        super().__init__(
            collection_name=collection_name,
            host=host,
            port=port,
            vector_size=vector_size,
            **kwargs
        )
        
        logger.info(f"Initialized Google embedding RAG tool with model: {embedding_model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for the provided text using Google's model.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_QUERY",
                title="",
            )
            
            # Extract the embedding values
            return embedding["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Google's model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Currently, we'll just process one at a time since the Google API 
            # doesn't have a batch embedding endpoint yet
            return [self.embed_text(text) for text in texts]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise

def query_google_rag(
    collection_name: str, 
    query: str, 
    limit: int = 5,
    score_threshold: float = 0.7,
    host: str = "localhost",
    port: int = 6333
) -> str:
    """
    Query a Qdrant collection using Google's text embeddings.
    
    This is the function that will be exposed to the agent.
    
    Args:
        collection_name: Name of the Qdrant collection to query
        query: The query text
        limit: Maximum number of documents to retrieve
        score_threshold: Minimum similarity score (0-1)
        host: Qdrant server hostname
        port: Qdrant HTTP port
        
    Returns:
        Formatted string with retrieved documents
    """
    logger.info(f"Calling Google RAG Tool with query: {query} on collection: {collection_name}")
    try:
        # Create the RAG tool for the specified collection
        rag_tool = GoogleEmbeddingRAGTool(
            collection_name=collection_name,
            host=host,
            port=port
        )
        
        # Perform the query
        results = rag_tool.query_rag(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )
        
        logger.info(f"Returning RAG results for collection: {collection_name}")
        return results
    except Exception as e:
        logger.error(f"Error during RAG query: {e}", exc_info=True)
        return f"An error occurred during the RAG query: {e}"

# --- Tool Schema Definition (for Gemini) ---
query_google_rag_declaration = FunctionDeclaration(
    name="query_google_rag",
    description="Retrieves relevant documents from a Qdrant vector database collection based on semantic similarity to the query. Use this for knowledge retrieval from specific document collections.",
    parameters={
        "type": "object",
        "properties": {
            "collection_name": {
                "type": "string",
                "description": "Name of the Qdrant collection to query (e.g., 'documentation', 'research_papers', 'code_snippets')"
            },
            "query": {
                "type": "string",
                "description": "The query text for which to find relevant documents"
            },            "limit": {
                "type": "integer",
                "description": "Maximum number of documents to retrieve (default: 5)"
            },
            "score_threshold": {
                "type": "number",
                "description": "Minimum similarity score (0-1) for retrieved documents (default: 0.7)"
            }
        },
        "required": ["collection_name", "query"]
    }
)
