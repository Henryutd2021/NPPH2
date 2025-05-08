# src/utils.py
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from logging_setup import logger # Assuming logger is accessible via direct import

def get_param(model, param_name_base, time_index=None, default=0.0):
    """
    Safely gets a parameter value from the model, handling indexing,
    ISO specifics, and potential NaN values.
    """
    target_iso_local = getattr(model, 'TARGET_ISO', 'UNKNOWN')
    param_name_iso = f"{param_name_base}_{target_iso_local}"
    param_to_get = None

    # Prioritize ISO-specific parameter name
    if hasattr(model, param_name_iso):
        param_to_get = getattr(model, param_name_iso)
    # Fallback to base parameter name
    elif hasattr(model, param_name_base):
        param_to_get = getattr(model, param_name_base)
    else:
        # Parameter not found with either name
        # logger.debug(f"Parameter '{param_name_base}' (or ISO variant) not found on model. Returning default.")
        return default

    try:
        if param_to_get.is_indexed():
            # Check index validity before accessing
            if time_index is not None and hasattr(param_to_get,'index_set') and time_index in param_to_get.index_set():
                val = pyo.value(param_to_get[time_index], exception=False)
            else:
                # logger.debug(f"Invalid time index '{time_index}' for indexed parameter '{param_name_base}'.")
                val = None # Index is invalid or None
        else: # Scalar parameter
            val = pyo.value(param_to_get, exception=False)

        # Check for None or NaN (common in pandas-loaded data)
        # Use both isnan and isna for broader compatibility
        if val is None or \
           (isinstance(val, float) and np.isnan(val)) or \
           pd.isna(val):
              # logger.debug(f"Parameter '{param_name_base}' value is None or NaN. Returning default.")
              return default
        else:
              return val

    except Exception as e:
        logger.error(f"Unexpected error accessing parameter '{param_name_base}' (attr name: {getattr(param_to_get, 'name', 'N/A')}) with index '{time_index}': {e}")
        return default

def get_var_value(model_component, time_index=None, default=0.0):
     """
     Safely gets a variable value from the model, handling indexing,
     None components, and potential solver non-population (returning default).
     """
     if model_component is None:
         # logger.debug("Attempted to get value from None model component.")
         return default
     try:
         val = None # Initialize val
         if model_component.is_indexed():
             # Check if index set exists and index is valid
             if hasattr(model_component,'index_set') and time_index is not None and time_index in model_component.index_set():
                 val = pyo.value(model_component[time_index], exception=False) # Allow returning None if variable not populated
             else:
                 # logger.debug(f"Invalid time index '{time_index}' for indexed variable '{getattr(model_component, 'name', 'N/A')}'.")
                 val = None # Index invalid or None
         else: # Not indexed
             val = pyo.value(model_component, exception=False) # Allow returning None

         # Check for None or NaN
         if val is None or (isinstance(val, float) and np.isnan(val)):
              # logger.debug(f"Variable '{getattr(model_component, 'name', 'N/A')}' value is None or NaN. Returning default.")
              return default
         return val
     except Exception as e:
         comp_name = getattr(model_component, 'name', 'Unknown Component')
         logger.error(f"Unexpected error getting variable value for {comp_name}: {e}")
         return default

# --- ADDED MISSING FUNCTION ---
def get_total_deployed_as(m, t, service_name):
    """Helper to sum deployed AS from all relevant components for a given service."""
    total_deployed = 0.0
    # Access flags via model object 'm'
    enable_electrolyzer = getattr(m,'ENABLE_ELECTROLYZER',False)
    enable_battery = getattr(m,'ENABLE_BATTERY',False)
    enable_npp = getattr(m,'ENABLE_NUCLEAR_GENERATOR',False)

    components = []
    if enable_electrolyzer: components.append('Electrolyzer')
    if enable_battery: components.append('Battery')
    # Consistent with AS logic: Turbine provides AS only if NPP and (Elec or Batt) enabled
    if enable_npp and (enable_electrolyzer or enable_battery):
        components.append('Turbine')

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            # Use get_var_value from this util module
            total_deployed += get_var_value(deployed_var, t, default=0.0)
    return total_deployed