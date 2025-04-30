"""
Logger module for the Gemini Research Agent.
This module provides a centralized logging configuration for the application.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Constants for log configuration
LOG_DIRECTORY = "logs"
LOG_FILE_FORMAT = "%Y-%m-%d_%H-%M-%S"
DEFAULT_LOG_LEVEL = logging.DEBUG
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

def setup_logger(logger_name="research_agent", log_level=DEFAULT_LOG_LEVEL, console=True, file_logging=True, log_file_name=None):
    """
    Configure and return a logger that can output to console and/or a log file.
    
    Args:
        logger_name: Name of the logger instance. Default is "research_agent".
        log_level: The level of logging (INFO, DEBUG, etc). Default is INFO.
        console: Whether to log to console. Default is True.
        file_logging: Whether to log to a file. Default is True.
        log_file_name: Custom log file name. If None, a timestamp-based name is used.
    
    Returns:
        A configured logger instance
    """
    # Create logs directory if it doesn't exist
    if file_logging and not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    
    # Setup logger
    newlogger = logging.getLogger(logger_name)
    newlogger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in newlogger.handlers[:]:
        newlogger.removeHandler(handler)
      # Setup formatters
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s'
    )
    
    # More detailed formatter for important interactions
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)-20s - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
      # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        newlogger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_logging:
        if not log_file_name:
            timestamp = datetime.now().strftime(LOG_FILE_FORMAT)
            log_file_name = f"agent.log"
        
        log_file_path = os.path.join(LOG_DIRECTORY, log_file_name)
        
        # Use RotatingFileHandler to limit log file size
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(detailed_formatter)
        newlogger.addHandler(file_handler)
    
    return newlogger

# Create a default/shared logger instance with a fixed file name
# This logger should be used only for general application logs that don't belong to a specific module
default_logger = setup_logger(logger_name="default_logger", log_file_name="agent_default.log")

# Convenience functions for different log levels
# IMPORTANT: These functions use the default_logger and should be used sparingly.
# For module-specific logging, create your own logger with a unique name:
#   module_logger = setup_logger(logger_name="module_name", log_file_name="module_name.log")
def debug(msg, *args, **kwargs):
    """Log a debug message using the default logger"""
    default_logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    """Log an info message using the default logger"""
    default_logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    """Log a warning message using the default logger"""
    default_logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    """Log an error message using the default logger"""
    default_logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    """Log a critical message using the default logger"""
    default_logger.critical(msg, *args, **kwargs)
