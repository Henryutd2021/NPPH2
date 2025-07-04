"""
Unified Logging System for Nuclear Power Plant Optimization Framework

This module provides a unified logging interface for OPT, TEA, and LCA modules
with proper categorization and timestamping.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from .basic_logging import setup_basic_logger


class ModuleType(Enum):
    """Supported module types"""
    OPT = "opt"
    TEA = "tea"
    LCA = "lca"


class UnifiedLogger:
    """Unified logger manager for all modules"""
    
    def __init__(self, base_log_dir: Optional[Path] = None):
        """
        Initialize unified logger
        
        Args:
            base_log_dir: Base directory for all logs (defaults to output/logs)
        """
        if base_log_dir is None:
            base_log_dir = Path(__file__).resolve().parent.parent.parent / "output" / "logs"
        
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize module directories
        for module in ModuleType:
            module_dir = self.base_log_dir / module.value
            module_dir.mkdir(parents=True, exist_ok=True)
        
        self._loggers: Dict[str, logging.Logger] = {}

    def create_logger(self,
                     module: ModuleType,
                     identifier: str,
                     log_level: str = "INFO",
                     add_timestamp: bool = False,
                     filemode: str = "w") -> logging.Logger:
        """
        Create a logger for a specific module and identifier
        
        Args:
            module: Module type (OPT, TEA, LCA)
            identifier: Identifier for the specific analysis (e.g., ISO region, plant name)
            log_level: Logging level
            add_timestamp: Whether to add timestamp to filename
            filemode: File mode for log file
            
        Returns:
            Configured logger instance
        """
        # Generate unique logger key
        logger_key = f"{module.value}_{identifier}"
        
        # Return existing logger if available
        if logger_key in self._loggers:
            return self._loggers[logger_key]
        
        # Create module log directory
        module_log_dir = self.base_log_dir / module.value
        
        # Generate filename
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{module.value}_{identifier}_{timestamp}.log"
        else:
            log_filename = f"{module.value}_{identifier}.log"
        
        log_file_path = module_log_dir / log_filename
        
        # Create logger using basic logging setup
        basic_logger = setup_basic_logger(
            name=logger_key,
            log_file=log_file_path,
            log_level=log_level,
            filemode=filemode
        )
        
        # Store logger
        self._loggers[logger_key] = basic_logger.logger
        
        return basic_logger.logger

    def get_opt_logger(self,
                      target_iso: str,
                      reactor_name: str = None,
                      add_timestamp: bool = False,
                      log_level: str = "INFO") -> logging.Logger:
        """
        Get logger for OPT module
        
        Args:
            target_iso: Target ISO region
            reactor_name: Reactor name (optional, will use ISO if not provided)
            add_timestamp: Whether to add timestamp
            log_level: Logging level
            
        Returns:
            OPT logger instance
        """
        # Use reactor name if provided, otherwise use ISO
        identifier = reactor_name if reactor_name else target_iso
        
        return self.create_logger(
            module=ModuleType.OPT,
            identifier=identifier,
            add_timestamp=add_timestamp,
            log_level=log_level,
            filemode="w"  # OPT typically overwrites
        )

    def get_tea_logger(self,
                      target_iso: str,
                      reactor_name: str = None,
                      analysis_type: str = "tea",
                      add_timestamp: bool = False,
                      log_level: str = "INFO") -> logging.Logger:
        """
        Get logger for TEA module
        
        Args:
            target_iso: Target ISO region
            reactor_name: Reactor name (optional, will use ISO if not provided)
            analysis_type: Type of TEA analysis
            add_timestamp: Whether to add timestamp
            log_level: Logging level
            
        Returns:
            TEA logger instance
        """
        # Build identifier based on reactor name or ISO
        base_identifier = reactor_name if reactor_name else target_iso
        
        if analysis_type == "tea":
            identifier = base_identifier
        else:
            identifier = f"{analysis_type}_{base_identifier}"
            
        return self.create_logger(
            module=ModuleType.TEA,
            identifier=identifier,
            add_timestamp=add_timestamp,
            log_level=log_level
        )

    def get_lca_logger(self,
                      plant_name: Optional[str] = None,
                      reactor_name: Optional[str] = None,
                      iso_region: Optional[str] = None,
                      analysis_type: str = "lca",
                      add_timestamp: bool = False,
                      log_level: str = "INFO") -> logging.Logger:
        """
        Get logger for LCA module
        
        Args:
            plant_name: Name of the plant
            reactor_name: Reactor name (preferred identifier)
            iso_region: ISO region
            analysis_type: Type of LCA analysis
            add_timestamp: Whether to add timestamp
            log_level: Logging level
            
        Returns:
            LCA logger instance
        """
        # Build identifier - prefer reactor_name, then plant_name, then iso_region
        identifier_parts = []
        if analysis_type != "lca":
            identifier_parts.append(analysis_type)
            
        if reactor_name:
            identifier_parts.append(reactor_name.replace(" ", "_"))
        elif plant_name:
            identifier_parts.append(plant_name.replace(" ", "_"))
        elif iso_region:
            identifier_parts.append(iso_region)
        else:
            identifier_parts.append("general")
            
        identifier = "_".join(identifier_parts)
        
        return self.create_logger(
            module=ModuleType.LCA,
            identifier=identifier,
            add_timestamp=add_timestamp,
            log_level=log_level
        )

    def list_loggers(self) -> Dict[str, str]:
        """
        List all active loggers
        
        Returns:
            Dictionary of logger names and their log files
        """
        logger_info = {}
        for logger_name, logger in self._loggers.items():
            # Get log file from file handler
            log_file = "unknown"
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file = handler.baseFilename
                    break
            logger_info[logger_name] = log_file
        return logger_info

    def close_all_loggers(self):
        """Close all loggers and their handlers"""
        for logger in self._loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        self._loggers.clear()


# Global unified logger instance
_unified_logger: Optional[UnifiedLogger] = None


def get_unified_logger() -> UnifiedLogger:
    """Get the global unified logger instance"""
    global _unified_logger
    if _unified_logger is None:
        _unified_logger = UnifiedLogger()
    return _unified_logger


def create_opt_logger(target_iso: str, reactor_name: str = None, **kwargs) -> logging.Logger:
    """Convenience function to create OPT logger"""
    return get_unified_logger().get_opt_logger(target_iso, reactor_name, **kwargs)


def create_tea_logger(target_iso: str, reactor_name: str = None, **kwargs) -> logging.Logger:
    """Convenience function to create TEA logger"""
    return get_unified_logger().get_tea_logger(target_iso, reactor_name, **kwargs)


def create_lca_logger(reactor_name: str = None, **kwargs) -> logging.Logger:
    """Convenience function to create LCA logger"""
    return get_unified_logger().get_lca_logger(reactor_name=reactor_name, **kwargs)


def close_all_loggers():
    """Close all active loggers"""
    global _unified_logger
    if _unified_logger:
        _unified_logger.close_all_loggers()
        _unified_logger = None


def list_active_loggers() -> Dict[str, str]:
    """List all active loggers and their log files"""
    return get_unified_logger().list_loggers() 