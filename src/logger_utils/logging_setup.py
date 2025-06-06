"""
Centralized logging initialization used by all modules.
This module provides backward compatibility with the original logging_setup.py
"""

import logging
from pathlib import Path
from .basic_logging import setup_basic_logger

# Default configuration for backward compatibility
# Use absolute path to project root's output/logs directory
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "logs"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = "INFO"

# Global logger instance (for backward compatibility)
logger = None


def initialize_logger(
    name: str = "optimization",
    target_iso: str = None,
    log_dir: Path = None,
    log_level: str = None,
    log_format: str = None,
    filemode: str = "w"
):
    """
    Initialize the global logger instance.
    This function provides backward compatibility with the original logging setup.

    Args:
        name: Base name for the logger
        target_iso: Target ISO region (will be appended to name)
        log_dir: Directory for log files
        log_level: Logging level
        log_format: Log format string  
        filemode: File mode for log file
    """
    global logger

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL
    if log_format is None:
        log_format = DEFAULT_LOG_FORMAT

    # Create logger name
    logger_name = name
    if target_iso:
        logger_name = f"{name}_{target_iso}"

    # Setup logger using the basic logging module
    basic_logger = setup_basic_logger(
        name=logger_name,
        log_dir=log_dir,
        log_level=log_level,
        log_format=log_format,
        filemode=filemode
    )

    # Extract the underlying logging.Logger for backward compatibility
    logger = basic_logger.logger
    return logger


def get_logger():
    """Get the global logger instance."""
    global logger
    if logger is None:
        # Initialize with default settings if not already initialized
        initialize_logger()
    return logger


# Initialize default logger for backward compatibility
# This maintains compatibility with existing code that imports 'logger' directly
try:
    logger = initialize_logger()
except Exception:
    # Fallback to basic logging if initialization fails
    logging.basicConfig(
        level=logging.INFO,
        format=DEFAULT_LOG_FORMAT
    )
    logger = logging.getLogger("fallback_logger")
