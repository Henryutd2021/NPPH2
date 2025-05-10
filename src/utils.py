"""Utility functions for model and result processing.

This module provides helper functions for parameter access, variable value retrieval,
and AS (Ancillary Services) related operations.
"""
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from logging_setup import logger


def get_param(model, param_name_base, time_index=None, default=0.0):
    """
    Safely gets a parameter value from the model, handling indexing,
    ISO specifics, and potential NaN values.

    Args:
        model: Pyomo model object
        param_name_base: Base name of the parameter
        time_index: Optional time index for indexed parameters
        default: Default value to return if parameter not found or invalid

    Returns:
        Parameter value or default
    """
    target_iso_local = getattr(model, 'TARGET_ISO', 'UNKNOWN')
    # Construct potential parameter names
    # e.g., p_RegU_ERCOT
    param_name_iso = f"{param_name_base}_{target_iso_local}"
    # e.g., ERCOT_p_RegU (Less common)
    param_name_iso_prefix = f"{target_iso_local}_{param_name_base}"

    param_to_get = None
    param_actual_name = "Not Found"

    # Prioritize ISO-specific parameter name (most common pattern)
    if hasattr(model, param_name_iso):
        param_to_get = getattr(model, param_name_iso)
        param_actual_name = param_name_iso
    # Fallback to base parameter name
    elif hasattr(model, param_name_base):
        param_to_get = getattr(model, param_name_base)
        param_actual_name = param_name_base
    # Fallback to ISO prefix (less common)
    elif hasattr(model, param_name_iso_prefix):
        param_to_get = getattr(model, param_name_iso_prefix)
        param_actual_name = param_name_iso_prefix
    else:
        return default

    try:
        val = None  # Initialize val
        if param_to_get is None:  # Should not happen if hasattr passed, but safety check
            val = None
        elif param_to_get.is_indexed():
            # Check index validity before accessing
            if time_index is not None and hasattr(param_to_get, 'index_set') and time_index in param_to_get.index_set():
                val = pyo.value(param_to_get[time_index], exception=False)
            else:
                val = None  # Index is invalid or None
        else:  # Scalar parameter
            val = pyo.value(param_to_get, exception=False)

        # Check for None or NaN
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            return default
        else:
            return val

    except Exception as e:
        logger.error(
            f"Unexpected error accessing parameter '{param_actual_name}' with index '{time_index}': {e}")
        return default


def get_var_value(model_component, time_index=None, default=0.0):
    """
    Safely gets a variable value from the model, handling indexing,
    None components, and potential solver non-population (returning default).

    Args:
        model_component: Pyomo variable component
        time_index: Optional time index for indexed variables
        default: Default value to return if variable not found or invalid

    Returns:
        Variable value or default
    """
    if model_component is None:
        return default
    try:
        val = None  # Initialize val
        if model_component.is_indexed():
            # Check if index set exists and index is valid
            if hasattr(model_component, 'index_set') and time_index is not None and time_index in model_component.index_set():
                # Allow returning None if variable not populated
                val = pyo.value(model_component[time_index], exception=False)
            else:
                val = None  # Index invalid or None
        else:  # Not indexed (e.g., pElectrolyzer_max)
            # Allow returning None
            val = pyo.value(model_component, exception=False)

        # Check for None or NaN
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val
    except Exception as e:
        comp_name = getattr(model_component, 'name', 'Unknown Component')
        logger.error(
            f"Unexpected error getting variable value for {comp_name}: {e}")
        return default


# --- AS Summation Helper Functions ---

def get_symbolic_as_bid_sum(m, t, service_list, component_suffix):
    """
    Returns symbolic sum expression for given AS BID services and component.
    Looks for variables named like: {service}_{component_suffix}

    Args:
        m: Pyomo model
        t: Time index
        service_list: List of service names
        component_suffix: Component name suffix (e.g., 'Turbine')

    Returns:
        Sum expression or 0.0
    """
    terms = []
    for service in service_list:
        var_name = f"{service}_{component_suffix}"  # This is for BID variables
        if hasattr(m, var_name):
            var_comp = getattr(m, var_name)
            if var_comp.is_indexed() and t in var_comp.index_set():
                terms.append(var_comp[t])
            elif not var_comp.is_indexed():
                logger.warning(
                    f"AS bid variable {var_name} is not indexed in get_symbolic_as_bid_sum.")
    return sum(terms) if terms else 0.0


def get_symbolic_as_deployed_sum(m, t, service_list, component_suffix):
    """
    Returns symbolic sum expression for given DEPLOYED services and component.
    Looks for variables named like: {service}_{component_suffix}_Deployed

    Args:
        m: Pyomo model
        t: Time index
        service_list: List of service names
        component_suffix: Component name suffix (e.g., 'Turbine')

    Returns:
        Sum expression or 0.0
    """
    terms = []
    for service in service_list:
        # Key change: added _Deployed
        var_name = f"{service}_{component_suffix}_Deployed"
        if hasattr(m, var_name):
            var_comp = getattr(m, var_name)
            if var_comp.is_indexed() and t in var_comp.index_set():
                terms.append(var_comp[t])
            elif not var_comp.is_indexed():
                logger.warning(
                    f"Deployed AS variable {var_name} is not indexed in get_symbolic_as_deployed_sum.")
    return sum(terms) if terms else 0.0


# --- Result Processing Helper ---
def get_total_deployed_as(m, t, service_name):
    """
    Helper to sum deployed AS from all relevant components for a given service
    AFTER optimization (using numerical values).

    Args:
        m: Pyomo model
        t: Time index
        service_name: Name of the ancillary service

    Returns:
        Total deployed AS from all components
    """
    total_deployed = 0.0
    # Access flags via model object 'm'
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)

    components = []
    if enable_electrolyzer:
        components.append('Electrolyzer')
    if enable_battery:
        components.append('Battery')
    # Consistent with AS logic: Turbine provides AS only if NPP and (Elec or Batt) enabled
    if enable_npp and (enable_electrolyzer or enable_battery):
        components.append('Turbine')

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            # Use get_var_value from this util module to get numerical value post-solve
            total_deployed += get_var_value(deployed_var, t, default=0.0)
    return total_deployed
