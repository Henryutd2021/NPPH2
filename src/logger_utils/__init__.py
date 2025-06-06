"""
Unified Logging System for Nuclear Power Plant Optimization Framework

This module provides logging functionality for both OPT and TEA modules:
- Basic logging setup for optimization processes
- Enhanced logging for TEA analysis with data tracking
- Progress indicators for long-running processes
"""

from .basic_logging import setup_basic_logger, BasicLogger
from .enhanced_logging import (
    EnhancedReactorLogger, ReactorLogSession, create_reactor_logger,
    DataIssueRecord, TEALoggerAdapter
)
from .progress_indicators import SolverProgressIndicator, TEAProgressIndicator

__all__ = [
    # Basic logging for OPT module
    'setup_basic_logger',
    'BasicLogger',

    # Enhanced logging for TEA module
    'EnhancedReactorLogger',
    'ReactorLogSession',
    'create_reactor_logger',
    'DataIssueRecord',
    'TEALoggerAdapter',

    # Progress indicators
    'SolverProgressIndicator',
    'TEAProgressIndicator'
]
