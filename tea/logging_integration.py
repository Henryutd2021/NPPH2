"""
Logging Integration Module for CS1 TEA Analysis
Provides adapters and decorators to integrate enhanced logging with existing TEA code
"""

import functools
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager
from .enhanced_logging import EnhancedReactorLogger


class TEALoggerAdapter:
    """Adapter to bridge enhanced logging with existing TEA functions"""
    
    def __init__(self, reactor_logger: EnhancedReactorLogger):
        self.logger = reactor_logger
        
    def log_tea_parameter(self, param_name: str, value: Any, unit: str = ""):
        """Log TEA parameter with standardized format"""
        self.logger.log_calculation_result(param_name, value, unit)
        
    def log_tea_phase(self, phase_name: str, description: str = ""):
        """Log TEA calculation phase"""
        self.logger.log_phase_start(f"TEA_{phase_name}", description)
        
    def complete_tea_phase(self, phase_name: str, duration: Optional[float] = None):
        """Complete TEA calculation phase"""
        self.logger.log_phase_complete(f"TEA_{phase_name}", duration)
        
    def log_data_validation_error(self, parameter: str, value: str, reason: str):
        """Log data validation errors"""
        self.logger.log_invalid_data(
            component="tea_validation",
            parameter=parameter,
            invalid_value=value,
            impact="medium"
        )
        
    def log_missing_input(self, parameter: str, fallback_value: str = None):
        """Log missing input parameters"""
        self.logger.log_missing_data(
            component="tea_input",
            parameter=parameter,
            fallback_value=fallback_value,
            impact="high"
        )


@contextmanager
def tea_calculation_phase(logger: EnhancedReactorLogger, phase_name: str, description: str = ""):
    """Context manager for TEA calculation phases"""
    import time
    start_time = time.time()
    logger.log_phase_start(f"TEA_{phase_name}", description)
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.log_phase_complete(f"TEA_{phase_name}", duration)


def log_tea_function(logger: EnhancedReactorLogger, phase_name: str = None):
    """Decorator to add logging to TEA functions"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_phase = phase_name or func.__name__
            
            with tea_calculation_phase(logger, function_phase, f"Executing {func.__name__}"):
                try:
                    result = func(*args, **kwargs)
                    logger.info(f"✅ {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"❌ {func.__name__} failed: {str(e)}")
                    raise
                    
        return wrapper
    return decorator


def monkey_patch_tea_functions(logger: EnhancedReactorLogger, tea_module):
    """Monkey patch TEA module functions to add logging"""
    
    # Store original functions
    original_functions = {}
    
    # List of functions to patch (add more as needed)
    functions_to_patch = [
        'calculate_lcoh',
        'calculate_npv',
        'calculate_capex',
        'calculate_opex',
        'load_system_parameters'
    ]
    
    for func_name in functions_to_patch:
        if hasattr(tea_module, func_name):
            original_functions[func_name] = getattr(tea_module, func_name)
            
            # Create logged version
            logged_func = log_tea_function(logger, func_name)(original_functions[func_name])
            
            # Replace the function
            setattr(tea_module, func_name, logged_func)
            
            logger.debug(f"Patched {func_name} with logging")
    
    return original_functions


def restore_tea_functions(tea_module, original_functions: Dict[str, Callable]):
    """Restore original TEA functions after analysis"""
    for func_name, original_func in original_functions.items():
        setattr(tea_module, func_name, original_func)


@contextmanager
def enhanced_tea_logging(reactor_logger: EnhancedReactorLogger, tea_module):
    """Context manager for enhanced TEA logging with automatic cleanup"""
    
    # Monkey patch functions
    original_functions = monkey_patch_tea_functions(reactor_logger, tea_module)
    
    try:
        # Provide adapter for easy access
        adapter = TEALoggerAdapter(reactor_logger)
        yield adapter
        
    finally:
        # Restore original functions
        restore_tea_functions(tea_module, original_functions)
        reactor_logger.info("🔄 TEA function logging cleanup completed") 