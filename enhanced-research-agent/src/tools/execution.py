"""Code Execution Tool for ResearchAgent

This module provides a safer environment for executing code requested by the LLM.
It uses a restricted execution environment with controlled imports and functions.
"""

import os
import io
import sys
import ast
import traceback
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import json
from src.logger.agent_logger import setup_logger

# Setup logger for this module
logger = setup_logger(logger_name="code_execution")

# Dictionary of allowed modules that can be imported in the sandbox
ALLOWED_MODULES = {
    'pandas': pd,
    'numpy': np,
    'matplotlib.pyplot': plt,
    'json': json,
    'io': io,
    'StringIO': StringIO,
    # Add other safe modules as needed
}

# Function declarations for Gemini API
from google.generativeai.types import FunctionDeclaration, Tool

execute_code_declaration = FunctionDeclaration(
    name="execute_code",
    description="Executes Python code in a restricted environment for data analysis, calculations, or visualizations. Use this tool when other tools cannot achieve the required computation.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Code should be well-formed and complete."
            },
            "description": {
                "type": "string", 
                "description": "A brief description of what the code is intended to do."
            }
        },
        "required": ["code", "description"]
    }
)

read_file_declaration = FunctionDeclaration(
    name="read_file",
    description="Reads the content of a file from the local filesystem. Use this for accessing data files.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path to the file to read."
            }
        },
        "required": ["file_path"]
    }
)

write_file_declaration = FunctionDeclaration(
    name="write_file",
    description="Writes content to a file in the local filesystem. Use this to save results or generated data.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The path where the file should be saved."
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file."
            }
        },
        "required": ["file_path", "content"]
    }
)

list_files_declaration = FunctionDeclaration(
    name="list_files",
    description="Lists files in a specified directory. Use this to discover available data files.",
    parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "The path to the directory to list files from."
            }
        },
        "required": ["directory_path"]
    }
)

class RestrictedNodeVisitor(ast.NodeVisitor):
    """AST Node visitor to check for potentially unsafe operations."""
    
    def __init__(self):
        self.errors = []
    
    def visit_Import(self, node):
        for name in node.names:
            if name.name not in ALLOWED_MODULES:
                self.errors.append(f"Import of module '{name.name}' is not allowed.")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module not in ALLOWED_MODULES:
            self.errors.append(f"Import from '{node.module}' is not allowed.")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Check for potentially dangerous built-in functions
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'compile', '__import__', 'open', 'globals', 'locals', 'getattr', 'setattr']:
                self.errors.append(f"Use of '{node.func.id}' function is not allowed.")
        elif isinstance(node.func, ast.Attribute):
            if hasattr(node.func, 'attr'):
                if node.func.attr in ['__globals__', '__code__', '__closure__', '__dict__']:
                    self.errors.append(f"Access to '{node.func.attr}' is not allowed.")
                # Disallow file operations on os module
                if hasattr(node.func, 'value') and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == 'os' and node.func.attr in ['system', 'popen', 'spawn', 'execl', 'execle', 'execlp', 'unlink', 'remove']:
                        self.errors.append(f"Use of 'os.{node.func.attr}' is not allowed.")
        self.generic_visit(node)


def is_code_safe(code_string):
    """Checks if the provided code string contains potentially unsafe operations."""
    try:
        parsed_ast = ast.parse(code_string)
        visitor = RestrictedNodeVisitor()
        visitor.visit(parsed_ast)
        
        if visitor.errors:
            return False, visitor.errors
        return True, []
    except SyntaxError as e:
        return False, [f"Syntax error in code: {str(e)}"]
    except Exception as e:
        return False, [f"Error analyzing code safety: {str(e)}"]


def execute_code(code, description=""):
    """
    Executes Python code in a restricted environment.
    
    Args:
        code (str): Python code string to execute
        description (str): Description of what the code does
        
    Returns:
        dict: Execution result with stdout, stderr, error messages, and any return value
    """
    logger.info(f"Executing code: {description}")
    debug_info = f"Code to execute:\n{code}"
    logger.debug(debug_info[:2000] + ("..." if len(debug_info) > 2000 else ""))  # Log truncated code for debugging
    
    result = {
        "stdout": "",
        "stderr": "",
        "error": None,
        "return_value": None,
        "has_plot": False,
        "plot_data": None,
    }
    
    # First check if the code is safe
    is_safe, safety_errors = is_code_safe(code)
    if not is_safe:
        logger.warning(f"Unsafe code detected: {safety_errors}")
        result["error"] = f"Code execution rejected due to safety concerns: {safety_errors}"
        return result
    
    # Create string buffers for stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
      # Create a restricted globals dictionary
    restricted_globals = {
        '__builtins__': {
            # Allow only specific builtins
            name: __builtins__[name] 
            for name in ['abs', 'all', 'any', 'bool', 'chr', 'dict', 'dir', 'divmod', 
                         'enumerate', 'filter', 'float', 'format', 'frozenset', 'hash', 
                         'hex', 'int', 'isinstance', 'issubclass', 'len', 'list', 'map',
                         'max', 'min', 'oct', 'ord', 'pow', 'print', 'range', 'repr',
                         'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum', 
                         'tuple', 'type', 'zip', '__import__']  # Added __import__ for import statements to work
        }
    }
    
    # Add allowed modules to globals
    restricted_globals.update(ALLOWED_MODULES)
    
    try:
        # Redirect stdout and stderr
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute the code
            exec_globals = {}
            exec(code, restricted_globals, exec_globals)
            
            # Check if a plot was created
            if plt.get_fignums():
                result["has_plot"] = True
                # Save the plot to a BytesIO object
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png')
                plt.close()
                img_data.seek(0)
                
                # Convert to base64 for easier transmission
                import base64
                result["plot_data"] = base64.b64encode(img_data.read()).decode()
                
        # Get stdout and stderr
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        
        # Look for variables that might be a result
        for var_name in ["result", "output", "data", "df", "answer"]:
            if var_name in exec_globals:
                result["return_value"] = str(exec_globals[var_name])
                # For DataFrames, display a proper representation
                if isinstance(exec_globals[var_name], pd.DataFrame):
                    result["return_value"] = exec_globals[var_name].to_string()
                break
                
    except Exception as e:
        # Catch any exceptions during execution
        result["error"] = f"Error during execution: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Code execution error: {str(e)}")
    
    return result


def read_file(file_path):
    """
    Reads the content of a file safely.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File content or error message
    """
    logger.info(f"Reading file: {file_path}")
    
    # Security: Basic path validation to avoid directory traversal
    normalized_path = os.path.normpath(file_path)
    if ".." in normalized_path:
        return {"error": "Path contains parent directory references (..), which is not allowed."}
    
    # Ensure the file exists
    if not os.path.exists(normalized_path):
        return {"error": f"File not found: {file_path}"}
    
    # Ensure it's a file, not a directory
    if not os.path.isfile(normalized_path):
        return {"error": f"Path is not a file: {file_path}"}
    
    try:
        # Read the file
        with open(normalized_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine the file type based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # For CSV files, automatically parse and return the first few rows for preview
        if ext == '.csv':
            try:
                df = pd.read_csv(normalized_path)
                return {
                    "content": content[:1000] + ("..." if len(content) > 1000 else ""),  # Truncated raw content
                    "preview": df.head().to_string(),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "file_type": "csv"
                }
            except Exception as e:
                # Fall back to raw content if parsing fails
                return {
                    "content": content[:2000] + ("..." if len(content) > 2000 else ""),
                    "error": f"Error parsing CSV: {str(e)}",
                    "file_type": "text"
                }
                
        # For JSON files, parse and return structure
        elif ext == '.json':
            try:
                data = json.loads(content)
                return {
                    "content": content[:1000] + ("..." if len(content) > 1000 else ""),
                    "structure": str(type(data)),
                    "sample": str(data)[:1000] + ("..." if len(str(data)) > 1000 else ""),
                    "file_type": "json"
                }
            except Exception as e:
                return {
                    "content": content[:2000] + ("..." if len(content) > 2000 else ""),
                    "error": f"Error parsing JSON: {str(e)}",
                    "file_type": "text"
                }
        
        # For text files, just return the content
        else:
            return {
                "content": content[:4000] + ("..." if len(content) > 4000 else ""),
                "file_type": "text"
            }
            
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return {"error": f"Error reading file: {str(e)}"}


def write_file(file_path, content):
    """
    Writes content to a file.
    
    Args:
        file_path (str): Path where to write the file
        content (str): Content to write
        
    Returns:
        dict: Success message or error
    """
    logger.info(f"Writing to file: {file_path}")
    
    # Security: Basic path validation
    normalized_path = os.path.normpath(file_path)
    if ".." in normalized_path:
        return {"error": "Path contains parent directory references (..), which is not allowed."}
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(normalized_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except Exception as e:
            return {"error": f"Could not create directory: {str(e)}"}
    
    try:
        # Write the file
        with open(normalized_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"File written successfully: {file_path}",
            "bytes": len(content)
        }
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {str(e)}")
        return {"error": f"Error writing file: {str(e)}"}


def list_files(directory_path):
    """
    Lists files in a directory.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        dict: List of files or error message
    """
    logger.info(f"Listing files in: {directory_path}")
    
    # Security: Basic path validation
    normalized_path = os.path.normpath(directory_path)
    if ".." in normalized_path:
        return {"error": "Path contains parent directory references (..), which is not allowed."}
    
    # Ensure the directory exists
    if not os.path.exists(normalized_path):
        return {"error": f"Directory not found: {directory_path}"}
    
    # Ensure it's a directory, not a file
    if not os.path.isdir(normalized_path):
        return {"error": f"Path is not a directory: {directory_path}"}
    
    try:
        # List the files
        file_list = os.listdir(normalized_path)
        
        # Get file details
        files = []
        for filename in file_list:
            file_path = os.path.join(normalized_path, filename)
            file_type = "directory" if os.path.isdir(file_path) else "file"
            file_size = os.path.getsize(file_path) if file_type == "file" else None
            files.append({
                "name": filename,
                "type": file_type,
                "size": file_size,
                "extension": os.path.splitext(filename)[1] if file_type == "file" else None
            })
        
        return {
            "directory": directory_path,
            "files": files
        }
    except Exception as e:
        logger.error(f"Error listing directory {directory_path}: {str(e)}")
        return {"error": f"Error listing directory: {str(e)}"}
