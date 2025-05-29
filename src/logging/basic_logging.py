"""
Basic logging functionality for the optimization framework.
Provides simple, standardized logging setup for OPT module.
"""

import logging
import os
from pathlib import Path
from typing import Optional


class BasicLogger:
    """Basic logger wrapper with standardized formatting"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, msg: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, *args, **kwargs)


def setup_basic_logger(
    name: str,
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    filemode: str = "w"
) -> BasicLogger:
    """
    Setup a basic logger for optimization framework modules.

    Args:
        name: Logger name (usually module name)
        log_file: Complete path to log file (overrides log_dir)
        log_dir: Directory for log files (will create name.log)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        filemode: File mode for log file ('w' to overwrite, 'a' to append)

    Returns:
        BasicLogger instance
    """
    # Determine log file path
    if log_file is None:
        if log_dir is None:
            log_dir = Path("../logs")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(
        log_file, mode=filemode, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (only for INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"Basic logging initialized. Log file: {log_file}")

    return BasicLogger(logger)


def setup_opt_logger(target_iso: str, log_dir: Optional[Path] = None) -> BasicLogger:
    """
    Convenience function to setup logger for OPT module.

    Args:
        target_iso: Target ISO region (e.g., 'PJM', 'CAISO')
        log_dir: Optional log directory (defaults to ../logs)

    Returns:
        BasicLogger instance configured for optimization
    """
    if log_dir is None:
        log_dir = Path("../logs")

    return setup_basic_logger(
        name=f"opt_{target_iso}",
        log_dir=log_dir,
        log_level="INFO",
        filemode="w"  # Overwrite for each run
    )
