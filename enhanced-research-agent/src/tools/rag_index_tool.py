"""
Tool interfaces for RAG indexing and management
"""

from google.generativeai.types import FunctionDeclaration
from src.tools.rag_indexer import index_directory
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="rag_index_tool")

def index_rag_collection(
    directory_path: str,
    collection_name: str,
    embedding_type: str = "google",
    file_pattern: str = "*.*",
    recursive: bool = True,
    chunk_size: int = 1000,
    overlap: int = 200,
    host: str = "localhost",
    port: int = 6333
) -> str:
    """
    Index all matching files in a directory into a Qdrant collection.
    
    This function is exposed to the agent as a tool.
    
    Args:
        directory_path: Path to the directory to index
        collection_name: Name of the Qdrant collection to use
        embedding_type: Type of embedding to use ('google' or 'openai')
        file_pattern: Glob pattern for files to index
        recursive: Whether to search subdirectories recursively
        chunk_size: Maximum size of each text chunk
        overlap: Number of characters to overlap between chunks
        host: Qdrant server hostname
        port: Qdrant HTTP port
        
    Returns:
        Formatted string with indexing results
    """
    logger.info(f"Indexing directory {directory_path} into collection {collection_name}")
    try:
        # Call the indexer function
        result = index_directory(
            directory_path=directory_path,
            collection_name=collection_name,
            embedding_type=embedding_type,
            file_pattern=file_pattern,
            recursive=recursive,
            chunk_size=chunk_size,
            overlap=overlap,
            host=host,
            port=port
        )
        
        # Format the result as a string
        if result["success"]:
            response = (
                f"Successfully indexed {result['files_indexed']} files into "
                f"{result['chunks_indexed']} chunks in collection '{collection_name}'.\n\n"
            )
            
            if result["files_indexed"] > 0:
                response += "Sample of indexed files:\n"
                for file_path in result["indexed_files"][:5]:  # Show first 5 files
                    response += f"- {file_path}\n"
                
                if len(result["indexed_files"]) > 5:
                    response += f"...and {len(result['indexed_files']) - 5} more files.\n"
            
            if result["failed_files"]:
                response += f"\nFailed to index {len(result['failed_files'])} files:\n"
                for failure in result["failed_files"][:5]:  # Show first 5 failures
                    response += f"- {failure['file_path']}: {failure['error']}\n"
                
                if len(result["failed_files"]) > 5:
                    response += f"...and {len(result['failed_files']) - 5} more failures.\n"
        else:
            response = f"Failed to index directory: {result.get('error', 'Unknown error')}"
        
        logger.info(f"Completed indexing with result: {response[:100]}...")
        return response
    
    except Exception as e:
        logger.error(f"Error in index_rag_collection: {e}", exc_info=True)
        return f"An error occurred during indexing: {e}"

# --- Tool Schema Definition (for Gemini) ---
index_rag_collection_declaration = FunctionDeclaration(
    name="index_rag_collection",
    description="Indexes files from a directory into a Qdrant vector database collection for retrieval-augmented generation (RAG). Use this to create searchable knowledge bases from local files.",
    parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Path to the directory containing files to index"
            },
            "collection_name": {
                "type": "string",
                "description": "Name of the Qdrant collection to store the indexed documents"
            },            "embedding_type": {
                "type": "string",
                "description": "Type of embedding model to use (google or openai)",
                "enum": ["google", "openai"]
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to index (e.g., '*.txt', '*.md', '*.py', default: '*.*')"
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search subdirectories recursively (default: true)"
            }
        },
        "required": ["directory_path", "collection_name"]
    }
)
