#!/usr/bin/env python3
"""
Unified path setup utility for executables
This module provides a consistent way to set up Python paths for importing src modules.
"""

import os
import sys
from pathlib import Path


def setup_src_paths():
    """
    Setup Python paths to enable importing from src directories.
    This function should be called at the beginning of executable scripts.
    
    Returns:
        str: The project root path
    """
    # Get the project root (parent of executables directory)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Add project root to sys.path (enables 'from src.xxx import' syntax)
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root_str


def get_project_root():
    """
    Get the project root path without modifying sys.path.
    
    Returns:
        str: The project root path
    """
    current_file = Path(__file__).resolve()
    return str(current_file.parent.parent) 