"""
Base class and implementations for RAG (Retrieval-Augmented Generation) tools using Qdrant
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from google.generativeai.types import FunctionDeclaration

# Import the logger
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="rag_tools")

class BaseRAGTool(ABC):
    """
    Base class for RAG tools using Qdrant as vector database.
    
    This abstract class provides common functionality for all RAG tools,
    while specific embedding models and collection configurations are
    implemented in child classes.
    """
    
    def __init__(
        self, 
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        timeout: int = 10,
        vector_size: int = 1536,  # Default for most common embedding models
        distance: str = "Cosine"  # Changed from "COSINE" to "Cosine"
    ):
        """
        Initialize the base RAG tool.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            host: Qdrant server hostname
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
            prefer_grpc: Whether to prefer gRPC over HTTP
            api_key: API key for Qdrant Cloud (if applicable)
            timeout: Connection timeout in seconds
            vector_size: Size of vectors produced by the embedding model
            distance: Distance metric (COSINE, DOT, EUCLID)
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.api_key = api_key
        self.timeout = timeout
        self.vector_size = vector_size
        self.distance = distance
        
        # Connect to Qdrant
        self._connect()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        logger.info(f"Initialized RAG tool for collection: {collection_name}")
    
    def _connect(self) -> None:
        """Establish connection to Qdrant server"""
        try:
            # Handle both local and cloud deployments
            if self.host == "localhost" or self.host.startswith("http://"):
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    timeout=self.timeout
                )
            else:
                # Assume cloud deployment with API key
                self.client = QdrantClient(
                    url=self.host,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            
            logger.info(f"Successfully connected to Qdrant at {self.host}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")
    
    def _ensure_collection_exists(self) -> None:
        """Check if the collection exists and create it if it doesn't"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                  # Create the collection with appropriate settings
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}", exc_info=True)
            raise
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for the provided text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def add_document(
        self, 
        text: str, 
        metadata: Dict[str, Any],
        id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector database.
        
        Args:
            text: The document text
            metadata: Document metadata (e.g., source, title, etc.)
            id: Optional document ID (will be auto-generated if not provided)
            
        Returns:
            ID of the added document
        """
        try:
            # Generate embedding for the document
            embedding = self.embed_text(text)
              # Add document to the collection
            if id is None:
                # Generate a UUID for the document
                id = str(uuid.uuid4())
                
            # Add the document with the ID
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                ]
            )
            
            logger.info(f"Added document with ID {id} to collection {self.collection_name}")
            return id
        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            raise
    
    def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: The query text
            limit: Maximum number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of documents with metadata and similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embed_text(query)
            
            # Set up search filters
            search_params = {
                "hnsw_ef": 128,  # Higher values give better recall but slower search
                "exact": False    # False for approximate search, True for exact but slower
            }
            
            # Retrieve documents
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                search_params=search_params,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Extract payload and add score
                document = result.payload
                document["score"] = result.score
                formatted_results.append(document)
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query: {query}")
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            raise
    
    def delete_document(self, id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[id]
                )
            )
            logger.info(f"Deleted document with ID {id} from collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {e}", exc_info=True)
            return False
    
    def query_rag(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> str:
        """
        Perform a RAG query and return formatted results.
        
        This is the main method to be called from the agent.
        
        Args:
            query: The query text
            limit: Maximum number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            Formatted string with retrieved documents
        """
        try:
            # Retrieve relevant documents
            results = self.retrieve(query, limit, score_threshold)
            
            if not results:
                return "No relevant documents found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results):
                # Extract text and metadata
                text = result.get("text", "No text available")
                score = result.get("score", 0.0)
                
                # Extract other metadata (excluding text which was already used)
                metadata = {k: v for k, v in result.items() if k not in ["text", "score"]}
                
                # Format metadata as string
                metadata_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                
                # Add formatted result
                formatted_results.append(
                    f"Result {i+1} (Score: {score:.2f}):\n"
                    f"Metadata: {metadata_str}\n"
                    f"Content: {text}\n"
                )
            
            return "\n---\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error in RAG query: {e}", exc_info=True)
            return f"An error occurred during the RAG query: {e}"
