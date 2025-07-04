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
from .logging_setup import create_module_logger, initialize_logger, get_logger
from .unified_logging import (
    UnifiedLogger, ModuleType, get_unified_logger,
    create_opt_logger, create_tea_logger, create_lca_logger,
    close_all_loggers, list_active_loggers
)

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
    'TEAProgressIndicator',

    # Unified logging setup
    'create_module_logger',
    'initialize_logger',
    'get_logger',

    # Unified logging system
    'UnifiedLogger',
    'ModuleType',
    'get_unified_logger',
    'create_opt_logger',
    'create_tea_logger',
    'create_lca_logger',
    'close_all_loggers',
    'list_active_loggers'
]
