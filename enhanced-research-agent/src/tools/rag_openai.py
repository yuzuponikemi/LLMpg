"""
RAG tool implementation using OpenAI's text embeddings
"""

import os
from typing import Dict, List, Optional, Any
import openai
from google.generativeai.types import FunctionDeclaration

from src.tools.rag_base import BaseRAGTool
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="openai_rag_tool")

class OpenAIEmbeddingRAGTool(BaseRAGTool):
    """
    RAG tool implementation using OpenAI's text embeddings.
    
    This class extends BaseRAGTool and implements the embedding methods
    using OpenAI's embedding models.
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        **kwargs
    ):
        """
        Initialize the OpenAI embedding-based RAG tool.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: OpenAI embedding model name
            api_key: OpenAI API key (if not set in environment)
            host: Qdrant server hostname
            port: Qdrant HTTP port
            **kwargs: Additional arguments to pass to BaseRAGTool
        """
        # Set up OpenAI API key
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            logger.error("OpenAI API key not provided and not found in environment")
            raise ValueError("OpenAI API key is required")
        
        self.embedding_model = embedding_model
        
        # Determine vector size based on model
        if embedding_model == "text-embedding-3-small":
            vector_size = 1536
        elif embedding_model == "text-embedding-3-large":
            vector_size = 3072
        elif embedding_model == "text-embedding-ada-002":
            vector_size = 1536
        else:
            # Default size, can be overridden in kwargs
            vector_size = kwargs.get("vector_size", 1536)
        
        # Initialize the base class
        super().__init__(
            collection_name=collection_name,
            host=host,
            port=port,
            vector_size=vector_size,
            **kwargs
        )
        
        logger.info(f"Initialized OpenAI embedding RAG tool with model: {embedding_model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for the provided text using OpenAI's model.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            # Truncate text if too long (OpenAI has token limits)
            if len(text) > 8000:
                logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
                text = text[:8000]
            
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float"
            )
            
            # Extract the embedding values
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using OpenAI's model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Truncate texts if too long
            truncated_texts = []
            for text in texts:
                if len(text) > 8000:
                    logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
                    truncated_texts.append(text[:8000])
                else:
                    truncated_texts.append(text)
            
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=truncated_texts,
                encoding_format="float"
            )
            
            # Extract the embedding values for each text
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise

def query_openai_rag(
    collection_name: str, 
    query: str, 
    limit: int = 5,
    score_threshold: float = 0.7,
    host: str = "localhost",
    port: int = 6333
) -> str:
    """
    Query a Qdrant collection using OpenAI's text embeddings.
    
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
    logger.info(f"Calling OpenAI RAG Tool with query: {query} on collection: {collection_name}")
    try:
        # Create the RAG tool for the specified collection
        rag_tool = OpenAIEmbeddingRAGTool(
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
query_openai_rag_declaration = FunctionDeclaration(
    name="query_openai_rag",
    description="Retrieves relevant documents from a Qdrant vector database collection using OpenAI embeddings based on semantic similarity to the query. Use this for knowledge retrieval from specific document collections.",
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
