# src/utils.py

"""
Utility functions for model and result processing.

This module provides helper functions for parameter access, variable value retrieval,
and Ancillary Services (AS) related operations.
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from logging_setup import logger


def get_param(model, param_name_base, time_index=None, default=0.0):
    """
    Safely gets a parameter value from the model.

    Handles indexing, ISO specifics, and potential NaN values, returning
    a default if the parameter is not found or invalid.
    It checks for parameter names in the order:
    1. {param_name_base}_{TARGET_ISO} (e.g., p_RegU_ERCOT)
    2. {param_name_base} (e.g., p_RegU)
    3. {TARGET_ISO}_{param_name_base} (e.g., ERCOT_p_RegU - less common)
    """
    target_iso_local = getattr(model, "TARGET_ISO", "UNKNOWN")
    param_name_iso_suffix = f"{param_name_base}_{target_iso_local}"
    param_name_iso_prefix = f"{target_iso_local}_{param_name_base}"

    param_to_get = None
    param_actual_name = "Not Found"

    if hasattr(model, param_name_iso_suffix):
        param_to_get = getattr(model, param_name_iso_suffix)
        param_actual_name = param_name_iso_suffix
    elif hasattr(model, param_name_base):
        param_to_get = getattr(model, param_name_base)
        param_actual_name = param_name_base
    elif hasattr(model, param_name_iso_prefix):
        param_to_get = getattr(model, param_name_iso_prefix)
        param_actual_name = param_name_iso_prefix
    else:
        # Log if a parameter that might be expected is not found
        # logger.debug(f"Parameter '{param_name_base}' not found on model. Returning default: {default}")
        return default

    try:
        val = None
        if param_to_get is None:  # Should not happen if hasattr passed
            val = None
        elif param_to_get.is_indexed():
            if (
                time_index is not None
                and hasattr(param_to_get, "index_set")
                and time_index in param_to_get.index_set()
            ):
                val = pyo.value(param_to_get[time_index], exception=False)
            else:  # Invalid or None time_index for an indexed parameter
                # logger.debug(f"Invalid time_index for parameter. Returning default.")
                val = None
        else:  # Scalar parameter
            val = pyo.value(param_to_get, exception=False)

        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            # logger.debug(f"Parameter resolved to None/NaN. Returning default: {default}")
            return default
        else:
            return val
    except Exception as e:
        logger.error(
            f"Error accessing parameter '{param_actual_name}' with index '{time_index}': {e}"
        )
        return default


def get_var_value(model_component, time_index=None, default=0.0):
    """
    Safely gets a variable value from the model.

    Handles indexing, None components, and cases where variables might not be
    populated by the solver (returning default).
    """
    if model_component is None:
        return default
    try:
        val = None
        if model_component.is_indexed():
            if (
                hasattr(model_component, "index_set")
                and time_index is not None
                and time_index in model_component.index_set()
            ):
                val = pyo.value(
                    model_component[time_index], exception=False
                )  # Allow None if not populated
            else:  # Invalid or None time_index for an indexed variable
                val = None
        else:  # Scalar variable
            val = pyo.value(
                model_component, exception=False
            )  # Allow None if not populated

        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val
    except Exception as e:
        comp_name = getattr(model_component, "name", "Unknown Component")
        logger.error(f"Error getting variable value for {comp_name}: {e}")
        return default


def get_symbolic_as_bid_sum(m, t, service_list, component_suffix):
    """
    Returns symbolic sum expression for given AS BID services and component.
    Looks for variables named like: {service}_{component_suffix}
    """
    terms = []
    for service in service_list:
        var_name = f"{service}_{component_suffix}"
        if hasattr(m, var_name):
            var_comp = getattr(m, var_name)
            if var_comp.is_indexed() and t in var_comp.index_set():
                terms.append(var_comp[t])
            elif not var_comp.is_indexed():  # Should not happen for AS bids
                logger.warning(
                    f"AS bid variable {var_name} is not indexed in get_symbolic_as_bid_sum."
                )
    return sum(terms) if terms else 0.0


def get_symbolic_as_deployed_sum(m, t, service_list, component_suffix):
    """
    Returns symbolic sum expression for given DEPLOYED AS services and component.
    Looks for variables named like: {service}_{component_suffix}_Deployed
    """
    terms = []
    for service in service_list:
        var_name = f"{service}_{component_suffix}_Deployed"
        if hasattr(m, var_name):
            var_comp = getattr(m, var_name)
            if var_comp.is_indexed() and t in var_comp.index_set():
                terms.append(var_comp[t])
            elif not var_comp.is_indexed():  # Should not happen for deployed AS
                logger.warning(
                    f"Deployed AS variable {var_name} is not indexed in get_symbolic_as_deployed_sum."
                )
    return sum(terms) if terms else 0.0


def get_total_deployed_as(m, t, service_name):
    """
    Helper to sum deployed AS from all relevant components for a given service
    AFTER optimization (using numerical values).
    """
    total_deployed = 0.0
    enable_electrolyzer = getattr(m, "ENABLE_ELECTROLYZER", False)
    enable_battery = getattr(m, "ENABLE_BATTERY", False)
    enable_npp = getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)

    components = []
    if enable_electrolyzer:
        components.append("Electrolyzer")
    if enable_battery:
        components.append("Battery")
    if enable_npp and (enable_electrolyzer or enable_battery):  # Turbine AS condition
        components.append("Turbine")

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            total_deployed += get_var_value(deployed_var, t, default=0.0)
    return total_deployed
