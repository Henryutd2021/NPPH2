# src/result_processing.py
import pandas as pd
import pyomo.environ as pyo
import os
from logging_setup import logger
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Import necessary config flags to conditionally process results
# Assumes these flags are accessible, e.g., imported directly or via model object
from config import (
    TARGET_ISO, # Get target ISO from config
    ENABLE_NUCLEAR_GENERATOR, ENABLE_ELECTROLYZER, ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY,
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN, ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    CAN_PROVIDE_ANCILLARY_SERVICES # Use derived flag to check if AS processing is needed
    # SIMULATE_AS_DISPATCH_EXECUTION will be accessed via model object 'm'
)
# Assuming constraints.py is in the same directory or accessible via sys.path
try:
    from constraints import get_as_components as get_as_components_helper
except ImportError:
    logger.warning("Could not import get_as_components from constraints.py. Detailed AS component breakdown might be limited.")
    # Define a placeholder if needed
    def get_as_components_helper(m, t):
        if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
             return {'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0}
        up_bids = 0.0; down_bids = 0.0
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        down_services = ['RegDown', 'RampDown']
        for service in up_services:
            if hasattr(m, f'Total_{service}'): up_bids += get_var_value(getattr(m, f'Total_{service}', None), t)
        for service in down_services:
            if hasattr(m, f'Total_{service}'): down_bids += get_var_value(getattr(m, f'Total_{service}', None), t)
        return {'up_reserves_bid': up_bids, 'down_reserves_bid': down_bids}


# --- Helper Functions defined within result_processing.py scope ---
def get_pyomo_value(model_component, default=0.0):
    """Safely extract the value of a Pyomo component (Var or Param)."""
    try:
        val = pyo.value(model_component, exception=False)
        if val is None or (isinstance(val, float) and np.isnan(val)):
             return default
        return val
    except Exception as e:
        logger.debug(f"Unexpected error getting Pyomo value: {e}")
        return default

def get_param(model, param_name_base, time_index=None, default=0.0):
    """Safely gets a parameter value, handling indexing and ISO specifics."""
    # Ensure TARGET_ISO exists on model object
    target_iso_local = getattr(model, 'TARGET_ISO', 'UNKNOWN')
    param_name_iso = f"{param_name_base}_{target_iso_local}"
    param_to_get = None
    if hasattr(model, param_name_iso):
        param_to_get = getattr(model, param_name_iso)
    elif hasattr(model, param_name_base): # Fallback to base name
        param_to_get = getattr(model, param_name_base)
    else:
        # Parameter not found with either name
        return default

    try:
        if param_to_get.is_indexed():
            # Check index validity
            if time_index is not None and hasattr(param_to_get,'index_set') and time_index in param_to_get.index_set():
                val = pyo.value(param_to_get[time_index], exception=False)
            else: val = None
        else: val = pyo.value(param_to_get, exception=False)

        # Check for None or NaN (common in pandas-loaded data)
        try: is_invalid = val is None or pd.isna(val)
        except Exception: is_invalid = val is None

        return default if is_invalid else val

    except Exception as e:
        logger.error(f"Error accessing parameter '{param_name_base}' with index '{time_index}': {e}")
        return default

def get_var_value(model_component, time_index=None, default=0.0):
     """Safely gets a variable value, handling indexing."""
     if model_component is None: return default
     try:
         val = None # Initialize val
         if model_component.is_indexed():
             # Check if index set exists and index is valid
             if hasattr(model_component,'index_set') and time_index is not None and time_index in model_component.index_set():
                 val = pyo.value(model_component[time_index], exception=False)
             # else: val remains None
         else: # Not indexed
             val = pyo.value(model_component, exception=False)
         # Check for None or NaN
         if val is None or (isinstance(val, float) and np.isnan(val)):
              return default
         return val
     except Exception as e:
         # Use component name in log if available
         comp_name = getattr(model_component, 'name', 'Unknown Component')
         logger.debug(f"Error getting variable value for {comp_name}: {e}")
         return default

def get_total_deployed_as(m, t, service_name):
    """Helper to sum deployed AS from all relevant components for a given service."""
    total_deployed = 0.0
    components = []
    # Access flags via model object 'm'
    if getattr(m,'ENABLE_ELECTROLYZER',False): components.append('Electrolyzer')
    if getattr(m,'ENABLE_BATTERY',False): components.append('Battery')
    if getattr(m,'ENABLE_NUCLEAR_GENERATOR',False) and \
       (getattr(m,'ENABLE_ELECTROLYZER',False) or getattr(m,'ENABLE_BATTERY',False)):
        components.append('Turbine')

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            total_deployed += get_var_value(deployed_var, t, default=0.0)
    return total_deployed

def calculate_hourly_as_revenue(m: pyo.ConcreteModel, t: int) -> float:
    """
    Calculates hourly AS revenue rate ($/hr) using ISO-specific logic.
    Checks m.SIMULATE_AS_DISPATCH_EXECUTION to determine calculation method
    for energy/performance payments.
    """
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0

    iso_suffix = getattr(m, 'TARGET_ISO', 'UNKNOWN')
    total_hourly_as_revenue_rate = 0.0
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    try:
        lmp = get_param(m, 'energy_price', t, default=0.0)

        # --- ISO-Specific Logic with Conditional Energy/Performance Payment ---
        if iso_suffix == 'SPP':
            # Regulation Up
            service_iso = 'RegU'; internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0; perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Regulation Down
            service_iso = 'RegD'; internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0; perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Reserves
            reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                if bid < 1e-6: continue
                mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'CAISO':
            service_iso = 'RegU'; internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            service_iso = 'RegD'; internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
            for service_iso, internal_service in reserve_map.items():
                 bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                 if bid < 1e-6: continue
                 mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                 cap_payment = bid * win_rate * mcp
                 energy_payment = 0.0
                 if simulate_dispatch:
                     deployed_amount = get_total_deployed_as(m, t, internal_service)
                     energy_payment = deployed_amount * lmp
                 else:
                     deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                     energy_payment = bid * deploy_factor * lmp
                 total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'ERCOT':
            service_iso = 'RegU'; internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0; perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            service_iso = 'RegD'; internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
            mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0; perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
            for service_iso, internal_service in reserve_map.items():
                 bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                 if bid < 1e-6: continue
                 mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                 cap_payment = bid * win_rate * mcp
                 energy_payment = 0.0
                 if simulate_dispatch:
                     deployed_amount = get_total_deployed_as(m, t, internal_service)
                     energy_payment = deployed_amount * lmp
                 else:
                     deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                     energy_payment = bid * deploy_factor * lmp
                 total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'PJM':
            service = 'Reg'
            bid_up_var = getattr(m, 'Total_RegUp', None); bid_up = get_var_value(bid_up_var, t)
            bid_down_var = getattr(m, 'Total_RegDown', None); bid_down = get_var_value(bid_down_var, t)
            total_reg_bid = bid_up + bid_down
            mcp_cap = get_param(m, 'p_RegCap', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            cap_payment = total_reg_bid * win_rate * mcp_cap
            perf_payment = 0.0
            if simulate_dispatch:
                 deployed_up = get_total_deployed_as(m, t, 'RegUp')
                 deployed_down = get_total_deployed_as(m, t, 'RegDown')
                 perf_payment = (deployed_up - deployed_down) * lmp # Simplified
            else:
                 mileage = get_param(m, 'mileage_ratio', t, 1.0)
                 perf = get_param(m, 'performance_score', t, 1.0)
                 perf_payment = total_reg_bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + perf_payment + adder
            reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'}
            for service_iso, internal_service in reserve_map.items():
                 bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                 if bid < 1e-6: continue
                 mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                 cap_payment = bid * win_rate * mcp
                 energy_payment = 0.0
                 if simulate_dispatch:
                     deployed_amount = get_total_deployed_as(m, t, internal_service)
                     energy_payment = deployed_amount * lmp
                 else:
                     deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                     energy_payment = bid * deploy_factor * lmp
                 total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'NYISO':
             service = 'RegC'
             bid_up_var = getattr(m, 'Total_RegUp', None); bid_up = get_var_value(bid_up_var, t)
             bid_down_var = getattr(m, 'Total_RegDown', None); bid_down = get_var_value(bid_down_var, t)
             total_reg_bid = bid_up + bid_down
             mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
             cap_payment = total_reg_bid * win_rate * mcp_cap
             energy_perf_payment = 0.0
             if simulate_dispatch:
                 deployed_up = get_total_deployed_as(m, t, 'RegUp')
                 deployed_down = get_total_deployed_as(m, t, 'RegDown')
                 energy_perf_payment = (deployed_up - deployed_down) * lmp
             else:
                 mileage = 1.0; perf = 1.0
                 energy_perf_payment = total_reg_bid * mileage * perf * lmp
             total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
             reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': 'ThirtyMin'}
             for service_iso, internal_service in reserve_map.items():
                  bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                  if bid < 1e-6: continue
                  mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                  cap_payment = bid * win_rate * mcp
                  energy_payment = 0.0
                  if simulate_dispatch:
                      deployed_amount = get_total_deployed_as(m, t, internal_service)
                      energy_payment = deployed_amount * lmp
                  else:
                      deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                      energy_payment = bid * deploy_factor * lmp
                  total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'ISONE':
             reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': 'ThirtyMin'}
             for service_iso, internal_service in reserve_map.items():
                  bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                  if bid < 1e-6: continue
                  mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                  cap_payment = bid * win_rate * mcp
                  energy_payment = 0.0
                  if simulate_dispatch:
                      deployed_amount = get_total_deployed_as(m, t, internal_service)
                      energy_payment = deployed_amount * lmp
                  else:
                      deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                      energy_payment = bid * deploy_factor * lmp
                  total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'MISO':
             service = 'Reg'
             bid_up_var = getattr(m, 'Total_RegUp', None); bid_up = get_var_value(bid_up_var, t)
             bid_down_var = getattr(m, 'Total_RegDown', None); bid_down = get_var_value(bid_down_var, t)
             total_reg_bid = bid_up + bid_down
             mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
             cap_payment = total_reg_bid * win_rate * mcp_cap
             energy_perf_payment = 0.0
             if simulate_dispatch:
                 deployed_up = get_total_deployed_as(m, t, 'RegUp')
                 deployed_down = get_total_deployed_as(m, t, 'RegDown')
                 energy_perf_payment = (deployed_up - deployed_down) * lmp
             else:
                 mileage = 1.0; perf = 1.0
                 energy_perf_payment = total_reg_bid * mileage * perf * lmp
             total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
             reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'}
             for service_iso, internal_service in reserve_map.items():
                  bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
                  if bid < 1e-6: continue
                  mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                  cap_payment = bid * win_rate * mcp
                  energy_payment = 0.0
                  if simulate_dispatch:
                      deployed_amount = get_total_deployed_as(m, t, internal_service)
                      energy_payment = deployed_amount * lmp
                  else:
                      deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
                      energy_payment = bid * deploy_factor * lmp
                  total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        return total_hourly_as_revenue_rate

    except AttributeError as e:
        logger.error(f"Missing component during hourly AS revenue calc for t={t}, ISO={iso_suffix}: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}")
        return 0.0


# --- Main Results Extraction Function ---
def extract_results(model: pyo.ConcreteModel, target_iso: str, output_dir: str = '../output/Results_Standardized'):
    """
    Extracts comprehensive results from the solved Pyomo model, aligning with model.py,
    constraints.py, and revenue_cost.py logic, including conditional processing based on simulation mode.
    Uses dictionary-first approach to avoid DataFrame fragmentation.
    """
    logger.info(f"Extracting comprehensive results for {target_iso}...")
    # Ensure model object has TARGET_ISO and SIMULATE_AS_DISPATCH_EXECUTION flags
    if not hasattr(model, 'TARGET_ISO') or model.TARGET_ISO != target_iso:
         logger.warning(f"Model TARGET_ISO mismatches function arg ('{target_iso}'). Using model attribute if available.")
    model.TARGET_ISO = getattr(model, 'TARGET_ISO', target_iso) # Ensure it's set for get_param
    # Get simulation mode flag, default to False if not set on model
    simulate_dispatch_mode = getattr(model, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Get AS capability flag
    can_provide_as_local = getattr(model, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    logger.info(f"Results extraction mode: {'Dispatch Execution' if simulate_dispatch_mode else 'Bidding Strategy'}")

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hours = list(model.TimePeriods)
    if not hours:
         logger.error("Model has no TimePeriods defined. Cannot extract results.")
         return pd.DataFrame(), {} # Return empty structures

    # Initialize dictionary to hold data lists
    hourly_data = {}

    summary_results: Dict[str, Any] = {}
    time_factor = get_pyomo_value(model.delT_minutes, default=60.0) / 60.0 # Hours per time step

    # --- Extract Optimized System Sizes & Store in Summary ---
    summary_results['Target_ISO'] = target_iso
    summary_results['Simulation_Mode'] = 'Dispatch Execution' if simulate_dispatch_mode else 'Bidding Strategy'
    # Initialize values
    elec_capacity_val = 0.0
    batt_capacity_val = 0.0
    batt_power_val = 0.0

    if ENABLE_ELECTROLYZER:
        if hasattr(model, 'pElectrolyzer_max') and isinstance(model.pElectrolyzer_max, pyo.Var):
            elec_capacity_val = get_pyomo_value(model.pElectrolyzer_max, default=0.0)
            summary_results['Optimal_Electrolyzer_Capacity_MW'] = elec_capacity_val
        else:
            elec_capacity_val = get_param(model, 'pElectrolyzer_max_upper_bound', default=0.0)
            summary_results['Fixed_Electrolyzer_Capacity_MW'] = elec_capacity_val
    # Add constant column to hourly data
    hourly_data['Electrolyzer_Capacity_MW'] = [elec_capacity_val] * len(hours)


    if ENABLE_BATTERY:
        if hasattr(model, 'BatteryCapacity_MWh') and isinstance(model.BatteryCapacity_MWh, pyo.Var):
            batt_capacity_val = get_pyomo_value(model.BatteryCapacity_MWh, default=0.0)
            batt_power_val = get_pyomo_value(model.BatteryPower_MW, default=0.0)
            summary_results['Optimal_Battery_Capacity_MWh'] = batt_capacity_val
            summary_results['Optimal_Battery_Power_MW'] = batt_power_val
        else:
            batt_capacity_val = get_param(model, 'BatteryCapacity_max', default=0.0)
            power_ratio = get_param(model, 'BatteryPowerRatio', default=0.0)
            batt_power_val = batt_capacity_val * power_ratio
            summary_results['Fixed_Battery_Capacity_MWh'] = batt_capacity_val
            summary_results['Fixed_Battery_Power_MW'] = batt_power_val
    # Add constant columns to hourly data
    hourly_data['Battery_Capacity_MWh'] = [batt_capacity_val] * len(hours)
    hourly_data['Battery_Power_MW'] = [batt_power_val] * len(hours)


    # --- Extract Hourly Variables into Dictionary Lists ---
    logger.info("Extracting hourly variables...")

    hourly_data['pIES_MW'] = [get_var_value(getattr(model, 'pIES', None), t) for t in hours]

    if ENABLE_NUCLEAR_GENERATOR:
        hourly_data['pTurbine_MW'] = [get_var_value(getattr(model, 'pTurbine', None), t) for t in hours]
        hourly_data['qSteam_Turbine_MWth'] = [get_var_value(getattr(model, 'qSteam_Turbine', None), t) for t in hours]
    else:
        hourly_data['pTurbine_MW'] = [0.0] * len(hours)
        hourly_data['qSteam_Turbine_MWth'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER:
        hourly_data['pElectrolyzer_MW'] = [get_var_value(getattr(model, 'pElectrolyzer', None), t) for t in hours]
        hourly_data['pElectrolyzerSetpoint_MW'] = [get_var_value(getattr(model, 'pElectrolyzerSetpoint', None), t) for t in hours]
        hourly_data['mHydrogenProduced_kg_hr'] = [get_var_value(getattr(model, 'mHydrogenProduced', None), t) for t in hours]
        if not ENABLE_LOW_TEMP_ELECTROLYZER and hasattr(model, 'qSteam_Electrolyzer'):
             hourly_data['qSteam_Electrolyzer_MWth'] = [get_var_value(model.qSteam_Electrolyzer, t) for t in hours]
        else: hourly_data['qSteam_Electrolyzer_MWth'] = [0.0] * len(hours)
    else:
        hourly_data['pElectrolyzer_MW'] = [0.0] * len(hours)
        hourly_data['pElectrolyzerSetpoint_MW'] = [0.0] * len(hours)
        hourly_data['mHydrogenProduced_kg_hr'] = [0.0] * len(hours)
        hourly_data['qSteam_Electrolyzer_MWth'] = [0.0] * len(hours)

    if hasattr(model, 'pAuxiliary'):
         hourly_data['pAuxiliary_MW'] = [get_var_value(model.pAuxiliary, t) for t in hours]
    else:
         hourly_data['pAuxiliary_MW'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE:
        hourly_data['H2_Storage_Level_kg'] = [get_var_value(getattr(model, 'H2_storage_level', None), t) for t in hours]
        hourly_data['H2_to_Market_kg_hr'] = [get_var_value(getattr(model, 'H2_to_market', None), t) for t in hours]
        hourly_data['H2_from_Storage_kg_hr'] = [get_var_value(getattr(model, 'H2_from_storage', None), t) for t in hours]
        hourly_data['H2_to_Storage_Input_kg_hr'] = [get_var_value(getattr(model, 'H2_to_storage', None), t) for t in hours]
    else:
        hourly_data['H2_Storage_Level_kg'] = [0.0] * len(hours)
        hourly_data['H2_to_Market_kg_hr'] = [0.0] * len(hours)
        hourly_data['H2_from_Storage_kg_hr'] = [0.0] * len(hours)
        hourly_data['H2_to_Storage_Input_kg_hr'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER and ENABLE_STARTUP_SHUTDOWN:
        hourly_data['Electrolyzer_Status(0=off,1=on)'] = [get_var_value(getattr(model, 'uElectrolyzer', None), t, default=-1) for t in hours]
        hourly_data['Electrolyzer_Startup(0=no,1=yes)'] = [get_var_value(getattr(model, 'vElectrolyzerStartup', None), t, default=-1) for t in hours]
        hourly_data['Electrolyzer_Shutdown(0=no,1=yes)'] = [get_var_value(getattr(model, 'wElectrolyzerShutdown', None), t, default=-1) for t in hours]
        # Calculate summary stat later from the list/series
    else:
        hourly_data['Electrolyzer_Status(0=off,1=on)'] = [1.0 if ENABLE_ELECTROLYZER else 0.0] * len(hours)
        hourly_data['Electrolyzer_Startup(0=no,1=yes)'] = [0.0] * len(hours)
        hourly_data['Electrolyzer_Shutdown(0=no,1=yes)'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER and ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
        hourly_data['DegradationState_Units'] = [get_var_value(getattr(model, 'DegradationState', None), t) for t in hours]
        # Calculate summary stat later
    else: hourly_data['DegradationState_Units'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER and hasattr(model, 'pElectrolyzerRampPos'):
        hourly_data['pElectrolyzerRampPos_MW'] = [get_var_value(model.pElectrolyzerRampPos, t) for t in hours]
        hourly_data['pElectrolyzerRampNeg_MW'] = [get_var_value(model.pElectrolyzerRampNeg, t) for t in hours]
    else:
        hourly_data['pElectrolyzerRampPos_MW'] = [0.0] * len(hours)
        hourly_data['pElectrolyzerRampNeg_MW'] = [0.0] * len(hours)

    if ENABLE_ELECTROLYZER and not ENABLE_LOW_TEMP_ELECTROLYZER and hasattr(model, 'qSteamElectrolyzerRampPos'):
        hourly_data['qSteamElectrolyzerRampPos_MWth'] = [get_var_value(model.qSteamElectrolyzerRampPos, t) for t in hours]
        hourly_data['qSteamElectrolyzerRampNeg_MWth'] = [get_var_value(model.qSteamElectrolyzerRampNeg, t) for t in hours]
    else:
        hourly_data['qSteamElectrolyzerRampPos_MWth'] = [0.0] * len(hours)
        hourly_data['qSteamElectrolyzerRampNeg_MWth'] = [0.0] * len(hours)

    if ENABLE_BATTERY:
        hourly_data['Battery_SOC_MWh'] = [get_var_value(getattr(model, 'BatterySOC', None), t) for t in hours]
        hourly_data['Battery_Charge_MW'] = [get_var_value(getattr(model, 'BatteryCharge', None), t) for t in hours]
        hourly_data['Battery_Discharge_MW'] = [get_var_value(getattr(model, 'BatteryDischarge', None), t) for t in hours]
        hourly_data['Battery_Charge_Binary'] = [get_var_value(getattr(model, 'BatteryBinaryCharge', None), t) for t in hours]
        hourly_data['Battery_Discharge_Binary'] = [get_var_value(getattr(model, 'BatteryBinaryDischarge', None), t) for t in hours]
    else:
        hourly_data['Battery_SOC_MWh'] = [0.0] * len(hours)
        hourly_data['Battery_Charge_MW'] = [0.0] * len(hours)
        hourly_data['Battery_Discharge_MW'] = [0.0] * len(hours)
        hourly_data['Battery_Charge_Binary'] = [0.0] * len(hours)
        hourly_data['Battery_Discharge_Binary'] = [0.0] * len(hours)

    # --- Extract Ancillary Service Bids ---
    logger.info("Extracting AS bids...")
    as_service_list = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
    components_providing_as = []
    if ENABLE_ELECTROLYZER: components_providing_as.append('Electrolyzer')
    if ENABLE_BATTERY: components_providing_as.append('Battery')
    if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY): components_providing_as.append('Turbine')

    if can_provide_as_local:
        for comp in components_providing_as:
             for service in as_service_list:
                 var_name = f"{service}_{comp}"
                 col_name = f'{var_name}_Bid_MW'
                 hourly_data[col_name] = [get_var_value(getattr(model, var_name, None), t) for t in hours]

        for service in as_service_list:
            total_var_name = f"Total_{service}"
            col_name = f'{total_var_name}_Bid_MW'
            hourly_data[col_name] = [get_var_value(getattr(model, total_var_name, None), t) for t in hours]
    else:
        # Zero out bid columns if AS disabled
        all_components = ['Electrolyzer', 'Battery', 'Turbine', 'Total']
        for comp_label in all_components:
            for service in as_service_list:
                 base_name = f"{service}_{comp_label}" if comp_label != 'Total' else f"Total_{service}"
                 col_name = f"{base_name}_Bid_MW"
                 hourly_data[col_name] = [0.0] * len(hours)


    # --- Extract Deployed Ancillary Service Amounts (Conditionally) ---
    logger.info("Extracting Deployed AS amounts (if in Dispatch Simulation mode)...")
    if simulate_dispatch_mode and can_provide_as_local:
        logger.info("Dispatch Simulation Mode: Extracting *_Deployed variables.")
        for comp in components_providing_as:
            for service in as_service_list:
                deployed_var_name = f"{service}_{comp}_Deployed"
                col_name = f'{deployed_var_name}_MW'
                if hasattr(model, deployed_var_name):
                    hourly_data[col_name] = [get_var_value(getattr(model, deployed_var_name), t) for t in hours]
                # else: # Optionally add zero columns if var is missing
                    # hourly_data[col_name] = [0.0] * len(hours)
    # else: # No need for else, just won't add columns if not in this mode


    # --- Extract Input Prices/Factors ---
    logger.info("Extracting input prices and factors...")
    hourly_data['EnergyPrice_LMP_USDperMWh'] = [get_param(model, 'energy_price', t) for t in hours]
    if can_provide_as_local:
        iso_service_map = { # Define or import map
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
            'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
            'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'],
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
            'ISONE': ['Spin10', 'NSpin10', 'OR30'],
            'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
        }
        iso_suffix_local = model.TARGET_ISO # Use ISO from model
        if iso_suffix_local in iso_service_map:
            for service_iso in iso_service_map[iso_suffix_local]:
                param_base_name = service_iso
                # Price (p_*) - skip factors
                if not any(f in service_iso for f in ['factor', 'score', 'ratio']):
                     col_name = f'p_{service_iso}_{iso_suffix_local}'
                     hourly_data[col_name] = [get_param(model, f'p_{param_base_name}', t) for t in hours]
                # Deploy Factor
                col_name = f'deploy_factor_{service_iso}_{iso_suffix_local}'
                hourly_data[col_name] = [get_param(model, f'deploy_factor_{param_base_name}', t) for t in hours]
                # Adder
                col_name = f'loc_{service_iso}_{iso_suffix_local}'
                hourly_data[col_name] = [get_param(model, f'loc_{param_base_name}', t) for t in hours]
                # Winning Rate
                col_name = f'winning_rate_{service_iso}_{iso_suffix_local}'
                hourly_data[col_name] = [get_param(model, f'winning_rate_{param_base_name}', t, 1.0) for t in hours]
                # Mileage/Performance Factors
                if iso_suffix_local == 'CAISO' and service_iso in ['RegU', 'RegD']:
                     col_name = f'mileage_factor_{service_iso}_{iso_suffix_local}'
                     hourly_data[col_name] = [get_param(model, f'mileage_factor_{service_iso}', t, 1.0) for t in hours]
                if iso_suffix_local == 'PJM':
                     if service_iso == 'performance_score':
                          col_name = f'performance_score_{iso_suffix_local}'
                          hourly_data[col_name] = [get_param(model, 'performance_score', t, 1.0) for t in hours]
                     if service_iso == 'mileage_ratio':
                          col_name = f'mileage_ratio_{iso_suffix_local}'
                          hourly_data[col_name] = [get_param(model, 'mileage_ratio', t, 1.0) for t in hours]


    # --- *** Create DataFrame AFTER all data is collected *** ---
    logger.info("Creating final DataFrame from collected hourly data...")
    results_df = pd.DataFrame(hourly_data, index=pd.Index(hours, name='HourOfYear'))
    # --- End DataFrame Creation ---


    # --- Calculate Hourly Revenues/Costs/Profit (using the created results_df) ---
    logger.info("Calculating hourly revenues...")
    # Energy Revenue
    if 'pIES_MW' in results_df and 'EnergyPrice_LMP_USDperMWh' in results_df:
        results_df['Revenue_Energy_USD'] = results_df['pIES_MW'] * results_df['EnergyPrice_LMP_USDperMWh'] * time_factor
    else: results_df['Revenue_Energy_USD'] = 0.0

    # Hydrogen Revenue (Modified to potentially split components)
    h2_value_param = get_param(model, 'H2_value', default=0.0) if ENABLE_ELECTROLYZER else 0.0
    h2_subsidy_param = get_param(model, 'hydrogen_subsidy_per_kg', default=0.0) if ENABLE_ELECTROLYZER else 0.0

    results_df['Revenue_Hydrogen_Sales_USD'] = 0.0 # Initialize columns
    results_df['Revenue_Hydrogen_Subsidy_USD'] = 0.0

    if ENABLE_ELECTROLYZER and 'mHydrogenProduced_kg_hr' in results_df:
        # Calculate subsidy revenue based on total production
        results_df['Revenue_Hydrogen_Subsidy_USD'] = results_df['mHydrogenProduced_kg_hr'] * h2_subsidy_param * time_factor

        # Calculate sales revenue based on delivery method
        if ENABLE_H2_STORAGE and 'H2_to_Market_kg_hr' in results_df and 'H2_from_Storage_kg_hr' in results_df:
             results_df['Revenue_Hydrogen_Sales_USD'] = (results_df['H2_to_Market_kg_hr'] + results_df['H2_from_Storage_kg_hr']) * h2_value_param * time_factor
        elif not ENABLE_H2_STORAGE: # If no storage, sales revenue = production revenue (excluding subsidy)
             results_df['Revenue_Hydrogen_Sales_USD'] = results_df['mHydrogenProduced_kg_hr'] * h2_value_param * time_factor

    # Original Total Hydrogen Revenue calculation replaced by sum of components
    results_df['Revenue_Hydrogen_USD'] = results_df['Revenue_Hydrogen_Sales_USD'] + results_df['Revenue_Hydrogen_Subsidy_USD']

    # Ancillary Service Revenue (Calls internal helper that uses the model 'm')
    results_df['Revenue_Ancillary_USD'] = [calculate_hourly_as_revenue(model, t) * time_factor for t in hours]

    results_df['Revenue_Total_USD'] = results_df['Revenue_Energy_USD'] + results_df['Revenue_Hydrogen_USD'] + results_df['Revenue_Ancillary_USD']

    logger.info("Calculating hourly costs...")
    # Calculate individual cost columns (examples)
    if ENABLE_NUCLEAR_GENERATOR and hasattr(model, 'vom_turbine') and 'pTurbine_MW' in results_df:
        vom_turbine_param = get_param(model, 'vom_turbine', default=0.0)
        results_df['Cost_VOM_Turbine_USD'] = results_df['pTurbine_MW'] * vom_turbine_param * time_factor
    else: results_df['Cost_VOM_Turbine_USD'] = 0.0

    if ENABLE_ELECTROLYZER and hasattr(model, 'vom_electrolyzer') and 'pElectrolyzer_MW' in results_df:
        vom_electrolyzer_param = get_param(model, 'vom_electrolyzer', default=0.0)
        results_df['Cost_VOM_Electrolyzer_USD'] = results_df['pElectrolyzer_MW'] * vom_electrolyzer_param * time_factor
    else: results_df['Cost_VOM_Electrolyzer_USD'] = 0.0

    if ENABLE_BATTERY and hasattr(model, 'vom_battery_per_mwh_cycled') and 'Battery_Charge_MW' in results_df and 'Battery_Discharge_MW' in results_df:
        vom_battery_param = get_param(model, 'vom_battery_per_mwh_cycled', default=0.0)
        results_df['Cost_VOM_Battery_USD'] = (results_df['Battery_Charge_MW'] + results_df['Battery_Discharge_MW']) * vom_battery_param * time_factor
    else: results_df['Cost_VOM_Battery_USD'] = 0.0

    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_water_per_kg_h2') and 'mHydrogenProduced_kg_hr' in results_df:
        cost_water_param = get_param(model, 'cost_water_per_kg_h2', default=0.0)
        results_df['Cost_Water_USD'] = results_df['mHydrogenProduced_kg_hr'] * cost_water_param * time_factor
    else: results_df['Cost_Water_USD'] = 0.0

    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_electrolyzer_ramping') and 'pElectrolyzerRampPos_MW' in results_df and 'pElectrolyzerRampNeg_MW' in results_df:
        cost_ramp_param = get_param(model, 'cost_electrolyzer_ramping', default=0.0)
        results_df['Cost_Ramping_USD'] = (results_df['pElectrolyzerRampPos_MW'] + results_df['pElectrolyzerRampNeg_MW']) * cost_ramp_param
        # Ensure first hour ramp cost is zero if columns exist
        if not results_df.empty:
             results_df.loc[results_df.index.min(), 'Cost_Ramping_USD'] = 0.0
    else: results_df['Cost_Ramping_USD'] = 0.0

    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE and hasattr(model, 'vom_storage_cycle') and 'H2_to_Storage_Input_kg_hr' in results_df and 'H2_from_Storage_kg_hr' in results_df:
        cost_storage_param = get_param(model, 'vom_storage_cycle', default=0.0)
        results_df['Cost_Storage_Cycle_USD'] = (results_df['H2_to_Storage_Input_kg_hr'] + results_df['H2_from_Storage_kg_hr']) * cost_storage_param * time_factor
    else: results_df['Cost_Storage_Cycle_USD'] = 0.0

    if ENABLE_ELECTROLYZER and ENABLE_STARTUP_SHUTDOWN and hasattr(model, 'cost_startup_electrolyzer') and 'Electrolyzer_Startup(0=no,1=yes)' in results_df:
        cost_startup_param = get_param(model, 'cost_startup_electrolyzer', default=0.0)
        results_df['Cost_Startup_USD'] = results_df['Electrolyzer_Startup(0=no,1=yes)'] * cost_startup_param
    else: results_df['Cost_Startup_USD'] = 0.0

    # Calculate Total Hourly OPEX column
    cost_cols = ['Cost_VOM_Turbine_USD', 'Cost_VOM_Electrolyzer_USD', 'Cost_VOM_Battery_USD',
                 'Cost_Water_USD', 'Cost_Ramping_USD', 'Cost_Storage_Cycle_USD', 'Cost_Startup_USD']
    results_df['Cost_HourlyOpex_Total_USD'] = results_df[[col for col in cost_cols if col in results_df.columns]].sum(axis=1)

    logger.info("Calculating hourly profit...")
    results_df['Profit_Hourly_USD'] = results_df['Revenue_Total_USD'] - results_df['Cost_HourlyOpex_Total_USD']


    # --- Calculate Summary Statistics ---
    logger.info("Calculating summary statistics...")
    total_revenue = results_df['Revenue_Total_USD'].sum()
    total_hourly_opex = results_df['Cost_HourlyOpex_Total_USD'].sum()

    # Get totals for individual cost components (handling missing columns)
    cost_vom_turbine_total = results_df['Cost_VOM_Turbine_USD'].sum() if 'Cost_VOM_Turbine_USD' in results_df else 0.0
    cost_vom_electrolyzer_total = results_df['Cost_VOM_Electrolyzer_USD'].sum() if 'Cost_VOM_Electrolyzer_USD' in results_df else 0.0
    cost_vom_battery_total = results_df['Cost_VOM_Battery_USD'].sum() if 'Cost_VOM_Battery_USD' in results_df else 0.0
    cost_water_total = results_df['Cost_Water_USD'].sum() if 'Cost_Water_USD' in results_df else 0.0
    cost_ramping_total = results_df['Cost_Ramping_USD'].sum() if 'Cost_Ramping_USD' in results_df else 0.0
    cost_storage_cycle_total = results_df['Cost_Storage_Cycle_USD'].sum() if 'Cost_Storage_Cycle_USD' in results_df else 0.0
    cost_startup_total = results_df['Cost_Startup_USD'].sum() if 'Cost_Startup_USD' in results_df else 0.0


    # Calculate Annualized Capex (replicating logic from model.py)
    total_annualized_capex = 0.0
    electrolyzer_annual_capex = 0.0
    battery_annual_capex = 0.0
    total_hours_sim = len(hours) * time_factor
    scaling_factor = total_hours_sim / 8760.0 if total_hours_sim > 0 else 0.0

    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_electrolyzer_capacity'):
        cost_elec_cap_param = get_param(model, 'cost_electrolyzer_capacity', default=0.0)
        # Use elec_capacity_val calculated earlier
        electrolyzer_annual_capex = elec_capacity_val * cost_elec_cap_param * scaling_factor
        total_annualized_capex += electrolyzer_annual_capex

    if ENABLE_BATTERY:
        # Use batt_capacity_val and batt_power_val calculated earlier
        cost_batt_cap_mwh_yr = get_param(model, 'BatteryCapex_USD_per_MWh_year', default=0.0)
        cost_batt_pow_mw_yr = get_param(model, 'BatteryCapex_USD_per_MW_year', default=0.0)
        cost_batt_fom_mwh_yr = get_param(model, 'BatteryFixedOM_USD_per_MWh_year', default=0.0)
        battery_annual_capex = (batt_capacity_val * cost_batt_cap_mwh_yr +
                                batt_power_val * cost_batt_pow_mw_yr +
                                batt_capacity_val * cost_batt_fom_mwh_yr) * scaling_factor
        total_annualized_capex += battery_annual_capex

    # Populate summary results dictionary
    summary_results['Total_Revenue_USD'] = total_revenue
    summary_results['Total_Energy_Revenue_USD'] = results_df['Revenue_Energy_USD'].sum()
    summary_results['Total_Hydrogen_Revenue_USD'] = results_df['Revenue_Hydrogen_USD'].sum()
    if 'Revenue_Hydrogen_Sales_USD' in results_df:
        summary_results['Total_Hydrogen_Sales_Revenue_USD'] = results_df['Revenue_Hydrogen_Sales_USD'].sum()
    if 'Revenue_Hydrogen_Subsidy_USD' in results_df:
        summary_results['Total_Hydrogen_Subsidy_Revenue_USD'] = results_df['Revenue_Hydrogen_Subsidy_USD'].sum()
    summary_results['Total_Ancillary_Revenue_USD'] = results_df['Revenue_Ancillary_USD'].sum()

    summary_results['Total_Hourly_Opex_USD'] = total_hourly_opex
    summary_results['Total_VOM_Cost_USD'] = cost_vom_turbine_total + cost_vom_electrolyzer_total + cost_vom_battery_total
    summary_results['Total_Water_Cost_USD'] = cost_water_total
    summary_results['Total_Ramping_Cost_USD'] = cost_ramping_total
    summary_results['Total_Storage_Cycle_Cost_USD'] = cost_storage_cycle_total
    summary_results['Total_Startup_Cost_USD'] = cost_startup_total

    summary_results['Total_Annualized_Capex_USD'] = total_annualized_capex
    summary_results['Electrolyzer_Annualized_Capex_USD'] = electrolyzer_annual_capex
    summary_results['Battery_Annualized_Capex_USD'] = battery_annual_capex

    summary_results['Total_Profit_Calculated_USD'] = total_revenue - total_hourly_opex - total_annualized_capex
    summary_results['Objective_Value_USD'] = get_pyomo_value(model.TotalProfit_Objective, default=None)

    # Other Summary Metrics
    if 'mHydrogenProduced_kg_hr' in results_df:
        summary_results['Total_H2_Produced_kg'] = results_df['mHydrogenProduced_kg_hr'].sum() * time_factor
    else: summary_results['Total_H2_Produced_kg'] = 0.0

    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE:
        if 'H2_to_Market_kg_hr' in results_df:
             summary_results['Total_H2_to_Market_Direct_kg'] = results_df['H2_to_Market_kg_hr'].sum() * time_factor
        else: summary_results['Total_H2_to_Market_Direct_kg'] = 0.0
        if 'H2_from_Storage_kg_hr' in results_df:
             summary_results['Total_H2_from_Storage_kg'] = results_df['H2_from_Storage_kg_hr'].sum() * time_factor
        else: summary_results['Total_H2_from_Storage_kg'] = 0.0
        if 'H2_Storage_Level_kg' in results_df and not results_df.empty:
             summary_results['Final_H2_Storage_Level_kg'] = results_df['H2_Storage_Level_kg'].iloc[-1]
        else: summary_results['Final_H2_Storage_Level_kg'] = 0.0


    if elec_capacity_val > 1e-6 and 'pElectrolyzer_MW' in results_df:
        avg_elec_power_actual = results_df['pElectrolyzer_MW'].mean()
        capacity_factor_actual = avg_elec_power_actual / elec_capacity_val
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = capacity_factor_actual
    else: summary_results['Electrolyzer_Capacity_Factor_Actual'] = 0.0

    # Add startup summary from list if calculated earlier
    if 'Electrolyzer_Startup(0=no,1=yes)' in hourly_data:
        summary_results['Total_Electrolyzer_Startups'] = int(np.sum(hourly_data['Electrolyzer_Startup(0=no,1=yes)']))
    # Add degradation summary from list if calculated earlier
    if 'DegradationState_Units' in hourly_data and hourly_data['DegradationState_Units']:
         summary_results['Final_DegradationState_Units'] = hourly_data['DegradationState_Units'][-1]


    # --- Add Deployed AS Summary Stats (Conditionally) ---
    if simulate_dispatch_mode and can_provide_as_local:
        logger.info("Calculating total deployed AS amounts...")
        for service in as_service_list:
             total_deployed_mwh = 0.0
             # Sum total deployed column from results_df if it exists
             total_deployed_col_name = f"Total_{service}_Deployed_MW" # Assumes model.py creates this total column if needed
             if total_deployed_col_name in results_df.columns:
                   total_deployed_mwh = results_df[total_deployed_col_name].sum() * time_factor
             else:
                 # If total column doesn't exist, sum from component columns in results_df
                 temp_sum = 0.0
                 for comp in components_providing_as:
                      comp_col_name = f"{service}_{comp}_Deployed_MW"
                      if comp_col_name in results_df.columns:
                          temp_sum += results_df[comp_col_name].sum()
                 total_deployed_mwh = temp_sum * time_factor

             summary_results[f'Total_Deployed_{service}_MWh'] = total_deployed_mwh

    # --- Final Formatting and Saving ---
    if not summary_results:
        logger.warning("Summary results dictionary is empty. Skipping summary file generation.")
        if not results_df.empty:
             output_csv_path = results_dir / f'{target_iso}_Hourly_Results_Comprehensive.csv'
             try:
                results_df.round(4).to_csv(output_csv_path)
                logger.info(f"Comprehensive hourly results saved to {output_csv_path}")
                print(f"Comprehensive hourly results saved to {output_csv_path}")
             except Exception as e:
                logger.error(f"Failed to save hourly results CSV: {e}")
                print(f"Error: Failed to save hourly results CSV to {output_csv_path}")
        return results_df, summary_results # Return potentially empty dict

    # Round and save hourly results
    results_df = results_df.round(4)
    output_csv_path = results_dir / f'{target_iso}_Hourly_Results_Comprehensive.csv'
    try:
        results_df.to_csv(output_csv_path)
        logger.info(f"Comprehensive hourly results saved to {output_csv_path}")
        print(f"Comprehensive hourly results saved to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save hourly results CSV: {e}")
        print(f"Error: Failed to save hourly results CSV to {output_csv_path}")

    # Save summary results
    output_summary_path = results_dir / f'{target_iso}_Summary_Results.txt'
    try:
        logger.info(f"Attempting to write summary results to: {output_summary_path}")
        with open(output_summary_path, 'w') as f:
            f.write(f"--- Summary Results for {target_iso} ---\n")
            # Ensure Simulation_Mode is near the top
            f.write(f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}\n")
            f.write(f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}\n")
            # Write remaining items
            for key, value in summary_results.items():
                try:
                    if value is None: line = f"{key}: None\n"
                    elif isinstance(value, (float, np.floating)): line = f"{key}: {value:,.4f}\n"
                    elif isinstance(value, (int, np.integer)): line = f"{key}: {value:,}\n"
                    else: line = f"{key}: {value}\n"
                    f.write(line)
                except Exception as write_err:
                    logger.error(f"Error writing summary key '{key}' with value '{value}' (type: {type(value)}): {write_err}")
                    f.write(f"{key}: ERROR_WRITING_VALUE\n")

            # Compare Objective vs Calculated Profit
            obj_val = summary_results.get('Objective_Value_USD')
            calc_prof = summary_results.get('Total_Profit_Calculated_USD')
            if isinstance(obj_val, (int, float, np.number)) and isinstance(calc_prof, (int, float, np.number)):
                diff = calc_prof - obj_val
                f.write(f"\nObjective vs Calculated Profit Diff: {diff:,.4f}\n")
                if abs(diff) > 1.0: # Tolerance
                    f.write("WARNING: Significant difference between objective value and calculated profit!\n")
                    logger.warning("Significant difference between objective value and calculated profit detected.")
            else:
                 f.write("\nCould not compare Objective vs Calculated Profit (one or both invalid).\n")
        logger.info(f"Summary results successfully saved to {output_summary_path}")
        print(f"Summary results saved to {output_summary_path}")
    except Exception as e:
        logger.error(f"Failed to save summary results file: {e}", exc_info=True)
        print(f"Error: Failed to save summary results file to {output_summary_path}")

    # Print summary to console
    print("\n--- Summary ---")
    # Print mode first
    print(f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}")
    print(f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}")
    for key, value in summary_results.items():
         # Skip keys already printed
         if key in ['Simulation_Mode', 'Target_ISO']: continue
         try:
             if value is None: print(f"{key}: None")
             elif isinstance(value, (float, np.floating)): print(f"{key}: {value:,.4f}")
             elif isinstance(value, (int, np.integer)): print(f"{key}: {value:,}")
             else: print(f"{key}: {value}")
         except Exception as print_err:
              print(f"{key}: ERROR_PRINTING_VALUE ({print_err})")
    print("---------------")

    return results_df, summary_results

# --- End of extract_results function ---