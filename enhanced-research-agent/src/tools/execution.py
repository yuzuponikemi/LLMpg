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
    # Data processing and analysis
    'pandas': pd,
    'numpy': np,
    'matplotlib.pyplot': plt,
    
    # Standard library modules
    'math': __import__('math'),
    'random': __import__('random'),
    'statistics': __import__('statistics'),
    'datetime': __import__('datetime'),
    'calendar': __import__('calendar'),
    'collections': __import__('collections'),
    'itertools': __import__('itertools'),
    'functools': __import__('functools'),
    're': __import__('re'),  # Regular expressions
    'csv': __import__('csv'),
    'json': json,
    'io': io,
    'StringIO': StringIO,
    'base64': __import__('base64'),
    'hashlib': __import__('hashlib'),
    'time': __import__('time'),
    'uuid': __import__('uuid'),
    'urllib.parse': __import__('urllib.parse'),
    'textwrap': __import__('textwrap'),
    'string': __import__('string'),
    'copy': __import__('copy'),
    'decimal': __import__('decimal'),
    'fractions': __import__('fractions'),
    'difflib': __import__('difflib'),  # For text comparison
    'heapq': __import__('heapq'),  # For heap queue algorithm
    
    # Science and numerical computing
    'scipy': __import__('scipy') if 'scipy' in sys.modules else None,
    'scipy.stats': __import__('scipy.stats', fromlist=['stats']) if 'scipy' in sys.modules else None,
    'scipy.optimize': __import__('scipy.optimize', fromlist=['optimize']) if 'scipy' in sys.modules else None,
    'scipy.spatial': __import__('scipy.spatial', fromlist=['spatial']) if 'scipy' in sys.modules else None,
    'scipy.signal': __import__('scipy.signal', fromlist=['signal']) if 'scipy' in sys.modules else None,
    'scipy.cluster': __import__('scipy.cluster', fromlist=['cluster']) if 'scipy' in sys.modules else None,
    
    # Machine Learning libraries
    'sklearn': __import__('sklearn') if 'sklearn' in sys.modules else None,
    'sklearn.preprocessing': __import__('sklearn.preprocessing', fromlist=['preprocessing']) if 'sklearn' in sys.modules else None,
    'sklearn.model_selection': __import__('sklearn.model_selection', fromlist=['model_selection']) if 'sklearn' in sys.modules else None,
    'sklearn.metrics': __import__('sklearn.metrics', fromlist=['metrics']) if 'sklearn' in sys.modules else None,
    'sklearn.cluster': __import__('sklearn.cluster', fromlist=['cluster']) if 'sklearn' in sys.modules else None,
    'sklearn.ensemble': __import__('sklearn.ensemble', fromlist=['ensemble']) if 'sklearn' in sys.modules else None,
    'sklearn.linear_model': __import__('sklearn.linear_model', fromlist=['linear_model']) if 'sklearn' in sys.modules else None,
    'sklearn.tree': __import__('sklearn.tree', fromlist=['tree']) if 'sklearn' in sys.modules else None,
    'sklearn.neighbors': __import__('sklearn.neighbors', fromlist=['neighbors']) if 'sklearn' in sys.modules else None,
    'sklearn.svm': __import__('sklearn.svm', fromlist=['svm']) if 'sklearn' in sys.modules else None,
    'sklearn.decomposition': __import__('sklearn.decomposition', fromlist=['decomposition']) if 'sklearn' in sys.modules else None,
    'sklearn.feature_extraction': __import__('sklearn.feature_extraction', fromlist=['feature_extraction']) if 'sklearn' in sys.modules else None,
    'sklearn.feature_extraction.text': __import__('sklearn.feature_extraction.text', fromlist=['text']) if 'sklearn' in sys.modules else None,
    'sklearn.pipeline': __import__('sklearn.pipeline', fromlist=['pipeline']) if 'sklearn' in sys.modules else None,
    
    # NLP libraries (if available)
    'nltk': __import__('nltk') if 'nltk' in sys.modules else None,
    'nltk.tokenize': __import__('nltk.tokenize', fromlist=['tokenize']) if 'nltk' in sys.modules else None,
    'nltk.stem': __import__('nltk.stem', fromlist=['stem']) if 'nltk' in sys.modules else None,
    'nltk.corpus': __import__('nltk.corpus', fromlist=['corpus']) if 'nltk' in sys.modules else None,
    
    # Data visualization enhancements
    'seaborn': __import__('seaborn') if 'seaborn' in sys.modules else None,
    'plotly': __import__('plotly') if 'plotly' in sys.modules else None,
    'plotly.express': __import__('plotly.express', fromlist=['express']) if 'plotly' in sys.modules else None,
    
    # Stats models (if available)
    'statsmodels': __import__('statsmodels') if 'statsmodels' in sys.modules else None,
    'statsmodels.api': __import__('statsmodels.api', fromlist=['api']) if 'statsmodels' in sys.modules else None,
    'statsmodels.formula.api': __import__('statsmodels.formula.api', fromlist=['api']) if 'statsmodels' in sys.modules else None,
}

def validate_imports(code_string):
    """
    Validate that the code only imports modules from the allowed list.
    
    Args:
        code_string: The Python code to analyze
        
    Returns:
        tuple: (is_valid, error_message) - is_valid is True if all imports are allowed,
               False otherwise with an error_message explaining the issue
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
        
        # Find all import statements
        for node in ast.walk(tree):
            # Check for direct imports (import x)
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name.split('.')[0]  # Get the base module name
                    if module_name not in ALLOWED_MODULES:
                        return False, f"Import of module '{module_name}' is not allowed. Only these modules are available: {', '.join(sorted(ALLOWED_MODULES.keys()))}"
            
            # Check for from imports (from x import y)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]  # Get the base module name
                    if module_name not in ALLOWED_MODULES:
                        return False, f"Import from module '{module_name}' is not allowed. Only these modules are available: {', '.join(sorted(ALLOWED_MODULES.keys()))}"
        
        # All imports are valid
        return True, ""
        
    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"
    except Exception as e:
        return False, f"Error validating imports: {str(e)}"

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

get_available_modules_declaration = FunctionDeclaration(
    name="get_available_modules",
    description="Returns information about all available modules that can be used in code execution. Use this before writing code to check which modules are available.",
    parameters={
        "type": "object",
        "properties": {},  # No parameters needed
        "required": []
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
    
    # Validate imports
    is_valid, import_error = validate_imports(code)
    if not is_valid:
        logger.warning(f"Invalid imports detected: {import_error}")
        result["error"] = f"Code execution rejected due to invalid imports: {import_error}"
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
    
    # Execute the code in the restricted environment
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Compile the code to detect syntax errors before execution
            compiled_code = compile(code, "<string>", "exec")
            
            # Execute the code with restricted globals and locals
            exec_locals = {}
            exec(compiled_code, restricted_globals, exec_locals)
            
        # Capture stdout and stderr
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        
        # Check for return value or variable definitions
        if "__return__" in exec_locals:
            result["return_value"] = exec_locals["__return__"]
        
        # Check if a matplotlib plot was created
        if "pyplot" in code or "plt" in code:
            if plt.get_fignums():
                result["has_plot"] = True
                # Save plot to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                import base64
                result["plot_data"] = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                plt.close('all')  # Close all plots
        
        logger.info("Code executed successfully")
        return result
    
    except ModuleNotFoundError as e:
        # Enhanced error message for module errors
        error_msg = f"Error: {str(e)}. Only these modules are available: {', '.join(sorted(ALLOWED_MODULES.keys()))}"
        logger.error(error_msg)
        result["error"] = error_msg
        result["stderr"] = stderr_buffer.getvalue() + f"\n{error_msg}"
        return result
    except ImportError as e:
        # Enhanced error message for import errors
        error_msg = f"Import Error: {str(e)}. Check that you're using the correct module name from the allowed modules list: {', '.join(sorted(ALLOWED_MODULES.keys()))}"
        logger.error(error_msg)
        result["error"] = error_msg
        result["stderr"] = stderr_buffer.getvalue() + f"\n{error_msg}"
        return result
    except Exception as e:
        # General error handling
        error_msg = f"{type(e).__name__}: {str(e)}"
        trace = traceback.format_exc()
        logger.error(f"Error executing code: {error_msg}\n{trace}")
        result["error"] = error_msg
        result["stderr"] = stderr_buffer.getvalue() + f"\n{trace}"
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


def get_available_modules():
    """
    Returns information about available modules that can be used in code execution.
    
    Returns:
        dict: Information about available modules organized by category
    """
    logger.info("Getting list of available modules")
    
    # Organize modules by category
    module_info = {
        "data_processing": [
            {"name": "pandas", "description": "Data analysis and manipulation library"},
            {"name": "numpy", "description": "Numerical computing with arrays and matrices"},
            {"name": "matplotlib.pyplot", "description": "Data visualization library"}
        ],
        "standard_library": [
            {"name": "math", "description": "Mathematical functions"},
            {"name": "random", "description": "Random number generation"},
            {"name": "statistics", "description": "Statistical functions"},
            {"name": "datetime", "description": "Date and time manipulation"},
            {"name": "calendar", "description": "Calendar-related functions"},
            {"name": "collections", "description": "Specialized container datatypes"},
            {"name": "itertools", "description": "Iterator functions for efficient looping"},
            {"name": "functools", "description": "Higher-order functions"},
            {"name": "re", "description": "Regular expressions"},
            {"name": "csv", "description": "CSV file reading and writing"},
            {"name": "json", "description": "JSON encoding and decoding"},
            {"name": "io", "description": "Core tools for working with streams"},
            {"name": "StringIO", "description": "In-memory text streams"},
        ],
        "utilities": [
            {"name": "base64", "description": "Base16, Base32, Base64, Base85 data encodings"},
            {"name": "hashlib", "description": "Secure hash and message digest algorithms"},
            {"name": "time", "description": "Time access and conversions"},
            {"name": "uuid", "description": "UUID objects according to RFC 4122"},
            {"name": "urllib.parse", "description": "Parse URLs into components"},
            {"name": "textwrap", "description": "Text wrapping and filling"},
            {"name": "string", "description": "Common string operations"},
            {"name": "copy", "description": "Shallow and deep copy operations"},
        ],
        "scientific": [
            {"name": "scipy", "description": "Scientific computing (if installed)"},
            {"name": "sklearn", "description": "Machine learning library (if installed)"},
            {"name": "nltk", "description": "Natural language toolkit (if installed)"},
            {"name": "seaborn", "description": "Statistical data visualization (if installed)"},
            {"name": "plotly", "description": "Interactive visualizations (if installed)"},
            {"name": "statsmodels", "description": "Statistical modeling (if installed)"}
        ]
    }
    
    # Check which optional scientific modules are actually available
    available_scientific = []
    for module in module_info["scientific"]:
        module_name = module["name"]
        if module_name in ALLOWED_MODULES and ALLOWED_MODULES[module_name] is not None:
            available_scientific.append(module)
    
    # Update with only available scientific modules
    module_info["scientific"] = available_scientific
    
    # Get a simple list of all available modules
    all_available_modules = sorted(ALLOWED_MODULES.keys())
    
    return {
        "categories": module_info,
        "all_modules": all_available_modules
    }
