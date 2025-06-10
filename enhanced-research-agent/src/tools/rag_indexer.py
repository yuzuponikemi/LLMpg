"""
Utility functions for document indexing and management in Qdrant
"""

import os
import glob
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import pandas as pd

from src.tools.rag_base import BaseRAGTool
from src.tools.rag_google import GoogleEmbeddingRAGTool
from src.tools.rag_openai import OpenAIEmbeddingRAGTool
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="rag_indexer")

def get_file_content(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract content and metadata from a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (content, metadata)
    """
    file_path = Path(file_path)
    
    # Extract basic metadata
    metadata = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "extension": file_path.suffix.lower(),
        "size_bytes": file_path.stat().st_size,
        "last_modified": file_path.stat().st_mtime,
    }
    
    # Extract content based on file type
    try:
        extension = file_path.suffix.lower()
        
        # Text files
        if extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # JSON files - parse and convert to string
        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                content = json.dumps(json_data, indent=2)
                
                # Add more specific metadata
                if isinstance(json_data, dict):
                    metadata["json_keys"] = list(json_data.keys())
                metadata["json_size"] = len(json_data)
        
        # CSV files - parse with pandas and convert to string
        elif extension == '.csv':
            df = pd.read_csv(file_path)
            # Get a string representation with head and summary
            content = f"CSV Summary:\nShape: {df.shape}\n\nColumns: {', '.join(df.columns)}\n\nSample Data:\n{df.head().to_string()}"
            
            # Add more specific metadata
            metadata["rows"] = df.shape[0]
            metadata["columns"] = df.shape[1]
            metadata["column_names"] = list(df.columns)
        
        # Add more file types as needed
        
        else:
            # For unsupported file types
            content = f"Unsupported file type: {extension}"
            metadata["indexed"] = False
            logger.warning(f"Unsupported file type for indexing: {extension}")
        
        return content, metadata
    
    except Exception as e:
        logger.error(f"Error extracting content from {file_path}: {e}", exc_info=True)
        return f"Error extracting content: {e}", {"error": str(e), **metadata}

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of approximately equal size.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get the end position for this chunk
        end = start + chunk_size
        
        # If we're not at the end of the text, try to find a good breaking point
        if end < len(text):
            # Look for a newline character to break at
            newline_pos = text.rfind('\n', start, end)
            if newline_pos != -1 and newline_pos > start + chunk_size // 2:
                end = newline_pos + 1  # Include the newline
            else:
                # Otherwise try to break at a period, question mark, or exclamation point
                for char in ['. ', '? ', '! ']:
                    pos = text.rfind(char, start, end)
                    if pos != -1 and pos > start + chunk_size // 2:
                        end = pos + 2  # Include the punctuation and space
                        break
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move the start position, accounting for overlap
        start = end - overlap
        
        # Make sure we don't get stuck in a loop
        if start >= len(text) - overlap:
            break
    
    return chunks

def index_file(
    file_path: str,
    rag_tool: BaseRAGTool,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Index a file into a Qdrant collection.
    
    Args:
        file_path: Path to the file to index
        rag_tool: RAG tool instance to use for indexing
        chunk_size: Maximum size of each text chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document IDs created
    """
    try:
        # Extract content and metadata
        content, metadata = get_file_content(file_path)
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size, overlap)
        
        # Index each chunk
        doc_ids = []
        for i, chunk in enumerate(chunks):
            # Create chunk-specific metadata
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            }
            
            # Add the chunk to the vector database
            doc_id = rag_tool.add_document(
                text=chunk,
                metadata=chunk_metadata
            )
            doc_ids.append(doc_id)
        
        logger.info(f"Indexed file {file_path} into {len(chunks)} chunks")
        return doc_ids
    
    except Exception as e:
        logger.error(f"Error indexing file {file_path}: {e}", exc_info=True)
        raise

def index_directory(
    directory_path: str,
    collection_name: str,
    embedding_type: str = "google",
    file_pattern: str = "*.*",
    recursive: bool = True,
    chunk_size: int = 1000,
    overlap: int = 200,
    host: str = "localhost",
    port: int = 6333
) -> Dict[str, Any]:
    """
    Index all matching files in a directory into a Qdrant collection.
    
    Args:
        directory_path: Path to the directory to index
        collection_name: Name of the Qdrant collection to use
        embedding_type: Type of embedding to use ('google' or 'openai')
        file_pattern: Glob pattern for files to index
        recursive: Whether to search subdirectories recursively
        chunk_size: Maximum size of each text chunk in characters
        overlap: Number of characters to overlap between chunks
        host: Qdrant server hostname
        port: Qdrant HTTP port
        
    Returns:
        Dictionary with indexing statistics
    """
    try:
        # Create the RAG tool
        if embedding_type.lower() == "google":
            rag_tool = GoogleEmbeddingRAGTool(
                collection_name=collection_name,
                host=host,
                port=port
            )
        elif embedding_type.lower() == "openai":
            rag_tool = OpenAIEmbeddingRAGTool(
                collection_name=collection_name,
                host=host,
                port=port
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        
        # Find all matching files
        if recursive:
            search_pattern = os.path.join(directory_path, "**", file_pattern)
            files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(directory_path, file_pattern)
            files = glob.glob(search_pattern)
        
        if not files:
            logger.warning(f"No files found matching pattern: {search_pattern}")
            return {
                "success": False,
                "error": f"No files found matching pattern: {search_pattern}",
                "files_processed": 0,
                "files_indexed": 0,
                "chunks_indexed": 0
            }
        
        # Index each file
        stats = {
            "success": True,
            "files_processed": 0,
            "files_indexed": 0,
            "chunks_indexed": 0,
            "indexed_files": [],
            "failed_files": []
        }
        
        for file_path in files:
            try:
                stats["files_processed"] += 1
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Index the file
                doc_ids = index_file(
                    file_path=file_path,
                    rag_tool=rag_tool,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                # Update statistics
                stats["files_indexed"] += 1
                stats["chunks_indexed"] += len(doc_ids)
                stats["indexed_files"].append(file_path)
            
            except Exception as e:
                logger.error(f"Failed to index file {file_path}: {e}", exc_info=True)
                stats["failed_files"].append({
                    "file_path": file_path,
                    "error": str(e)
                })
        
        logger.info(f"Indexed {stats['files_indexed']} files with {stats['chunks_indexed']} chunks")
        return stats
    
    except Exception as e:
        logger.error(f"Error indexing directory {directory_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "files_processed": 0,
            "files_indexed": 0,
            "chunks_indexed": 0
        }
