"""
Utility functions for the TEA module, including logging setup.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(log_dir_path: Path, log_file_name_prefix: str, add_timestamp: bool = False) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        log_dir_path: Path object for the directory where log files will be stored.
        log_file_name_prefix: Prefix for the log file name (e.g., "tea_TARGET_ISO").
        add_timestamp: Whether to add timestamp to the log filename.

    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(log_dir_path, exist_ok=True)

    # Add timestamp to filename if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir_path / \
            f"{log_file_name_prefix}_{timestamp}.log"
    else:
        log_file_path = log_dir_path / f"{log_file_name_prefix}.log"

    # Create a logger
    # Use prefix for logger name for clarity
    logger = logging.getLogger(log_file_name_prefix)
    logger.setLevel(logging.DEBUG)  # Set the base level for the logger

    # Prevent duplicate handlers if this function is called multiple times with the same logger name
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler - reduced output level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)  # Only errors to console
    console_formatter = logging.Formatter(
        "TEA %(levelname)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging configured. Log file: {log_file_path}")

    return logger


def setup_tea_module_logger(target_iso: str,
                            reactor_name: str = None,
                            analysis_type: str = "tea",
                            add_timestamp: bool = False,
                            log_level: str = "INFO") -> logging.Logger:
    """
    Setup logger specifically for TEA module with proper categorization.

    Args:
        target_iso: Target ISO region
        reactor_name: Reactor name (preferred over ISO for identification)
        analysis_type: Type of analysis (tea, reactor_specific, etc.)
        add_timestamp: Whether to add timestamp to filename
        log_level: Logging level

    Returns:
        Configured logger instance
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    tea_log_dir = project_root / "output" / "logs" / "tea"
    
    # Create log filename prefix - prefer reactor name over ISO
    base_identifier = reactor_name if reactor_name else target_iso
    
    if analysis_type == "tea":
        prefix = f"tea_{base_identifier}"
    else:
        prefix = f"tea_{analysis_type}_{base_identifier}"
    
    return setup_logging(tea_log_dir, prefix, add_timestamp)


# Example usage (optional, for testing the function directly)
if __name__ == '__main__':
    # This block will only execute if utils.py is run directly
    # It's useful for testing the setup_logging function in isolation
    current_script_dir = Path(__file__).resolve().parent
    default_log_dir = current_script_dir.parent / \
        "logs_test"  # Example test log directory

    # Test with a generic prefix
    test_logger_generic = setup_logging(default_log_dir, "test_log_generic")
    test_logger_generic.debug("This is a generic debug message.")
    test_logger_generic.info("This is a generic info message.")
    test_logger_generic.warning("This is a generic warning message.")

    # Test with a specific prefix
    test_logger_specific = setup_logging(
        default_log_dir, "test_log_specific_module")
    test_logger_specific.debug(
        "This is a specific debug message for a module.")
    test_logger_specific.info("This is a specific info message for a module.")

    print(f"Test logs generated in: {default_log_dir}")
