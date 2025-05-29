"""
Utility functions for the TEA module, including logging setup.
"""

import logging
import os
from pathlib import Path

def setup_logging(log_dir_path: Path, log_file_name_prefix: str) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        log_dir_path: Path object for the directory where log files will be stored.
        log_file_name_prefix: Prefix for the log file name (e.g., "tea_TARGET_ISO").

    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(log_dir_path, exist_ok=True)
    
    log_file_path = log_dir_path / f"{log_file_name_prefix}.log"

    # Create a logger
    logger = logging.getLogger(log_file_name_prefix) # Use prefix for logger name for clarity
    logger.setLevel(logging.DEBUG) # Set the base level for the logger

    # Prevent duplicate handlers if this function is called multiple times with the same logger name
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # Console output level
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG) # Log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging configured. Log file: {log_file_path}")
    
    return logger

# Example usage (optional, for testing the function directly)
if __name__ == '__main__':
    # This block will only execute if utils.py is run directly
    # It's useful for testing the setup_logging function in isolation
    current_script_dir = Path(__file__).resolve().parent
    default_log_dir = current_script_dir.parent / "logs_test" # Example test log directory
    
    # Test with a generic prefix
    test_logger_generic = setup_logging(default_log_dir, "test_log_generic")
    test_logger_generic.debug("This is a generic debug message.")
    test_logger_generic.info("This is a generic info message.")
    test_logger_generic.warning("This is a generic warning message.")

    # Test with a specific prefix
    test_logger_specific = setup_logging(default_log_dir, "test_log_specific_module")
    test_logger_specific.debug("This is a specific debug message for a module.")
    test_logger_specific.info("This is a specific info message for a module.")
    
    print(f"Test logs generated in: {default_log_dir}")
