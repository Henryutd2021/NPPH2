# src/result_processing.py
import pandas as pd
import pyomo.environ as pyo
import os
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Import necessary config flags to conditionally process results
from config import (
    TARGET_ISO, # Get target ISO from config
    ENABLE_NUCLEAR_GENERATOR, ENABLE_ELECTROLYZER, ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY,
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN, ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    CAN_PROVIDE_ANCILLARY_SERVICES # Use derived flag to check if AS processing is needed
)
# Assuming constraints.py is in the same directory or accessible via sys.path
# This helper is needed for AS details if required beyond bids (e.g., deployed amounts)
try:
    # get_as_components might be useful if calculating deployed reserves is needed later
    # For now, we mainly rely on bid variables and parameters from the model
    from constraints import get_as_components as get_as_components_helper
except ImportError:
    logging.warning("Could not import get_as_components from constraints.py. Detailed AS component breakdown might be limited.")
    # Define a placeholder if needed, though current implementation might not rely heavily on it
    def get_as_components_helper(m, t):
        # Placeholder returns zero/empty structures
        return {
            'up_reserves_bid_turbine': 0.0, 'down_reserves_bid_turbine': 0.0,
            'up_reserves_bid_h2': 0.0, 'down_reserves_bid_h2': 0.0,
            'up_reserves_bid_battery': 0.0, 'down_reserves_bid_battery': 0.0,
            'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0,
            # Add placeholders for deployed/iso if needed by future logic
            # 'up_deployed': 0.0, 'down_deployed': 0.0, 'iso_services': {} # Keep commented out
        }

# --- Helper Functions (get_pyomo_value, get_param) ---
def get_pyomo_value(model_component, default=0.0):
    """Safely extract the value of a Pyomo component (Var or Param)."""
    try:
        # Attempt to get the value, handling potential errors like component not existing or having no value
        val = pyo.value(model_component, exception=False)
        # Return the default if the value is None (component doesn't exist, has no value, or extraction failed)
        return default if val is None else val
    except Exception as e:
        # Log unexpected errors during value extraction
        logging.debug(f"Unexpected error getting Pyomo value: {e}")
        return default

def get_param(model, param_name_base, time_index=None, default=0.0):
    """Safely gets a parameter value, handling indexing and ISO specifics."""
    # Try ISO-specific name first
    param_name_iso = f"{param_name_base}_{model.TARGET_ISO}"
    param_to_get = None
    if hasattr(model, param_name_iso):
        param_to_get = getattr(model, param_name_iso)
    elif hasattr(model, param_name_base): # Fallback to base name
        param_to_get = getattr(model, param_name_base)
    else:
        # Parameter not found with either name
        return default

    # Now extract the value, handling indexed vs non-indexed
    try:
        if param_to_get.is_indexed():
            if time_index is not None and time_index in param_to_get:
                val = pyo.value(param_to_get[time_index], exception=False)
            else:
                val = None
        else: # Not indexed
            val = pyo.value(param_to_get, exception=False)

        # Check for None or NaN (common in pandas-loaded data)
        try:
            is_invalid = val is None or pd.isna(val)
        except Exception: # Handle cases where pd.isna might fail (e.g., non-numeric types)
            is_invalid = val is None

        return default if is_invalid else val

    except Exception as e:
        logger.error(f"Error accessing parameter '{param_name_base}' with index '{time_index}': {e}")
        return default

# Helper to safely get variable values
def get_var_value(model_component, time_index=None, default=0.0):
     """Safely gets a variable value, handling indexing."""
     if model_component is None:
         return default
     try:
         if model_component.is_indexed():
             if time_index is not None and time_index in model_component:
                 val = pyo.value(model_component[time_index], exception=False)
             else:
                 val = None # Invalid index access
         else: # Not indexed
             val = pyo.value(model_component, exception=False)

         # Return default if value is None (e.g., variable not solved, index invalid)
         return default if val is None else val
     except Exception as e:
         # Log unexpected errors during value extraction
         logger.debug(f"Error getting variable value: {e}")
         return default

# --- Replicated AS Revenue Calculation ---
# This function MUST exactly mirror the logic in revenue_cost.py for the target ISO
def calculate_hourly_as_revenue(m: pyo.ConcreteModel, t: int) -> float:
    """Calculates hourly AS revenue using ISO-specific logic based on revenue_cost.py."""
    # Return 0 immediately if the system cannot provide AS according to config
    if not CAN_PROVIDE_ANCILLARY_SERVICES:
        return 0.0

    iso_suffix = m.TARGET_ISO
    total_hourly_as_revenue_rate = 0.0 # Revenue rate for the hour ($/hr)
    try:
        # Get LMP for the current hour, needed for energy payment component of reserves
        lmp = get_param(m, 'energy_price', t, default=0.0)

        # Use the exact same logic structure as in revenue_cost.py
        if iso_suffix == 'SPP':
            # Regulation Up
            service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
            # Regulation Down
            service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
            # Reserves
            for service_iso in ['Spin', 'Sup', 'RamU', 'RamD', 'UncU']:
                internal_name_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None) # Get the variable object itself
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'CAISO':
            # Regulation Up (No LMP mileage payment in CAISO structure)
            service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            # Regulation Down
            service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            # Reserves
            for service_iso in ['Spin', 'NSpin', 'RMU', 'RMD']:
                internal_name_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'ERCOT':
            # Regulation Up
            service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
            # Regulation Down
            service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
            # Reserves
            for service_iso in ['Spin', 'NSpin', 'ECRS']:
                internal_name_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'PJM':
            # Regulation (Combined Up/Down Bid)
            service = 'Reg'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, 'p_RegCap', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = get_param(m, 'mileage_ratio', t, 1.0); perf = get_param(m, 'performance_score', t, 1.0)
            # Note: PJM has RegPerf price too. The user rule uses Mileage*Perf*LMP instead. Following user rule.
            cap_payment = bid * win_rate * mcp_cap
            perf_payment = mileage * perf * lmp # User rule for performance payment
            total_hourly_as_revenue_rate += cap_payment + perf_payment + adder
            # Reserves
            for service_iso in ['Syn', 'Rse', 'TMR']:
                internal_name_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': '30Min'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'NYISO':
            # Regulation Capacity
            service = 'RegC'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
            # Reserves
            for service_iso in ['Spin10', 'NSpin10', 'Res30']:
                internal_name_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': '30Min'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'ISONE':
            # Reserves (No Regulation in data?)
            for service_iso in ['Spin10', 'NSpin10', 'OR30']:
                internal_name_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': '30Min'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        elif iso_suffix == 'MISO':
            # Regulation
            service = 'Reg'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
            # Reserves
            for service_iso in ['Spin', 'Sup', 'STR', 'RamU', 'RamD']:
                internal_name_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': '30Min', 'RamU': 'RampUp', 'RamD': 'RampDown'}
                total_var_name = f"Total_{internal_name_map[service_iso]}"
                total_var = getattr(m, total_var_name, None)
                bid = get_var_value(total_var, t)
                mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                total_hourly_as_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder

        # Apply time factor (e.g., hours per timestep) to get total revenue for the period
        # NOTE: This time factor application is done in the main revenue rule that calls this hourly function
        # This function returns the HOURLY RATE ($/hr)
        return total_hourly_as_revenue_rate

    except AttributeError as e:
        logging.error(f"Missing parameter/variable during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}")
        return 0.0 # Return 0 if calculation fails for this hour due to missing data
    except Exception as e:
        logging.error(f"Error during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}")
        return 0.0 # Return 0 on other unexpected errors

# --- Main Results Extraction Function ---
def extract_results(model: pyo.ConcreteModel, target_iso: str, output_dir: str = '../Results_Standardized'):
    """
    Extracts comprehensive results from the solved Pyomo model, aligning with model.py,
    constraints.py, and revenue_cost.py logic.
    """
    logging.info(f"Extracting comprehensive results for {target_iso}...")
    # Ensure the model object has the TARGET_ISO attribute set, matching the input arg
    if not hasattr(model, 'TARGET_ISO') or model.TARGET_ISO != target_iso:
         logging.warning(f"Model TARGET_ISO ('{getattr(model, 'TARGET_ISO', 'Not Set')}') might not match function arg ('{target_iso}'). Using model attribute if available, else arg.")
         model.TARGET_ISO = getattr(model, 'TARGET_ISO', target_iso) # Ensure it's set for get_param

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hours = list(model.TimePeriods)
    results_df = pd.DataFrame(index=pd.Index(hours, name='HourOfYear'))
    summary_results: Dict[str, Any] = {}
    time_factor = get_pyomo_value(model.delT_minutes) / 60.0 # Hours per time step

    # --- Extract Optimized System Sizes ---
    summary_results['Target_ISO'] = target_iso
    if ENABLE_ELECTROLYZER and hasattr(model, 'pElectrolyzer_max') and isinstance(model.pElectrolyzer_max, pyo.Var):
        optimal_elec_capacity = get_pyomo_value(model.pElectrolyzer_max, default=None)
        summary_results['Optimal_Electrolyzer_Capacity_MW'] = optimal_elec_capacity
        results_df['Electrolyzer_Capacity_MW'] = optimal_elec_capacity if optimal_elec_capacity is not None else 0.0
    elif ENABLE_ELECTROLYZER: # If not a Var, assume it was fixed by bounds/params
        fixed_elec_capacity = get_param(model, 'pElectrolyzer_max_upper_bound', default=0.0)
        summary_results['Fixed_Electrolyzer_Capacity_MW'] = fixed_elec_capacity
        results_df['Electrolyzer_Capacity_MW'] = fixed_elec_capacity
    else:
        results_df['Electrolyzer_Capacity_MW'] = 0.0

    if ENABLE_BATTERY and hasattr(model, 'BatteryCapacity_MWh') and isinstance(model.BatteryCapacity_MWh, pyo.Var):
        optimal_batt_capacity = get_pyomo_value(model.BatteryCapacity_MWh, default=None)
        summary_results['Optimal_Battery_Capacity_MWh'] = optimal_batt_capacity
        results_df['Battery_Capacity_MWh'] = optimal_batt_capacity if optimal_batt_capacity is not None else 0.0
        optimal_batt_power = get_pyomo_value(model.BatteryPower_MW, default=None) # Get linked power
        summary_results['Optimal_Battery_Power_MW'] = optimal_batt_power
        results_df['Battery_Power_MW'] = optimal_batt_power if optimal_batt_power is not None else 0.0
    elif ENABLE_BATTERY: # If not a Var, assume fixed
        fixed_batt_capacity = get_param(model, 'BatteryCapacity_max', default=0.0) # Use max param as proxy?
        summary_results['Fixed_Battery_Capacity_MWh'] = fixed_batt_capacity
        results_df['Battery_Capacity_MWh'] = fixed_batt_capacity
        # Calculate power based on fixed capacity and ratio param
        power_ratio = get_param(model, 'BatteryPowerRatio', default=0.0)
        fixed_batt_power = fixed_batt_capacity * power_ratio
        summary_results['Fixed_Battery_Power_MW'] = fixed_batt_power
        results_df['Battery_Power_MW'] = fixed_batt_power
    else:
        results_df['Battery_Capacity_MWh'] = 0.0
        results_df['Battery_Power_MW'] = 0.0


    # --- Extract Hourly Variables ---
    logging.info("Extracting hourly variables...")
    results_df['pIES_MW'] = [get_var_value(model.pIES, t) for t in hours]

    if ENABLE_NUCLEAR_GENERATOR:
        results_df['pTurbine_MW'] = [get_var_value(model.pTurbine, t) for t in hours]
        results_df['qSteam_Turbine_MWth'] = [get_var_value(model.qSteam_Turbine, t) for t in hours]
    else:
        results_df['pTurbine_MW'] = 0.0
        results_df['qSteam_Turbine_MWth'] = 0.0

    if ENABLE_ELECTROLYZER:
        results_df['pElectrolyzer_MW'] = [get_var_value(model.pElectrolyzer, t) for t in hours] # Actual power
        results_df['pElectrolyzerSetpoint_MW'] = [get_var_value(model.pElectrolyzerSetpoint, t) for t in hours]
        results_df['mHydrogenProduced_kg_hr'] = [get_var_value(model.mHydrogenProduced, t) for t in hours] # Actual H2
        # Only extract steam if HTE mode is possible and electrolyzer enabled
        if not ENABLE_LOW_TEMP_ELECTROLYZER and hasattr(model, 'qSteam_Electrolyzer'):
             results_df['qSteam_Electrolyzer_MWth'] = [get_var_value(model.qSteam_Electrolyzer, t) for t in hours]
        else: results_df['qSteam_Electrolyzer_MWth'] = 0.0
    else:
        results_df['pElectrolyzer_MW'] = 0.0
        results_df['pElectrolyzerSetpoint_MW'] = 0.0
        results_df['mHydrogenProduced_kg_hr'] = 0.0
        results_df['qSteam_Electrolyzer_MWth'] = 0.0

    # Electrolyzer Optional Features
    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE:
        results_df['H2_Storage_Level_kg'] = [get_var_value(model.H2_storage_level, t) for t in hours]
        results_df['H2_to_Market_kg_hr'] = [get_var_value(model.H2_to_market, t) for t in hours]
        results_df['H2_from_Storage_kg_hr'] = [get_var_value(model.H2_from_storage, t) for t in hours]
        # Extract the input flow to storage
        results_df['H2_to_Storage_Input_kg_hr'] = [get_var_value(model.H2_to_storage, t) for t in hours]
    else:
        results_df['H2_Storage_Level_kg'] = 0.0
        results_df['H2_to_Market_kg_hr'] = 0.0
        results_df['H2_from_Storage_kg_hr'] = 0.0
        results_df['H2_to_Storage_Input_kg_hr'] = 0.0


    if ENABLE_ELECTROLYZER and ENABLE_STARTUP_SHUTDOWN:
        results_df['Electrolyzer_Status(0=off,1=on)'] = [get_var_value(model.uElectrolyzer, t, default=-1) for t in hours]
        results_df['Electrolyzer_Startup(0=no,1=yes)'] = [get_var_value(model.vElectrolyzerStartup, t, default=-1) for t in hours]
        results_df['Electrolyzer_Shutdown(0=no,1=yes)'] = [get_var_value(model.wElectrolyzerShutdown, t, default=-1) for t in hours]
        summary_results['Total_Electrolyzer_Startups'] = sum(results_df['Electrolyzer_Startup(0=no,1=yes)'].clip(0))
    else:
        results_df['Electrolyzer_Status(0=off,1=on)'] = 1.0 # Assume always on if SU/SD not modeled
        results_df['Electrolyzer_Startup(0=no,1=yes)'] = 0.0
        results_df['Electrolyzer_Shutdown(0=no,1=yes)'] = 0.0


    if ENABLE_ELECTROLYZER and ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
        results_df['DegradationState_Units'] = [get_var_value(model.DegradationState, t) for t in hours]
        summary_results['Final_DegradationState_Units'] = results_df['DegradationState_Units'].iloc[-1] if not results_df.empty else 0.0
    else:
         results_df['DegradationState_Units'] = 0.0

    # Ramping Variables (if costed/constrained)
    if ENABLE_ELECTROLYZER and hasattr(model, 'pElectrolyzerRampPos'):
        results_df['pElectrolyzerRampPos_MW'] = [get_var_value(model.pElectrolyzerRampPos, t) for t in hours]
        results_df['pElectrolyzerRampNeg_MW'] = [get_var_value(model.pElectrolyzerRampNeg, t) for t in hours]
    else:
        results_df['pElectrolyzerRampPos_MW'] = 0.0
        results_df['pElectrolyzerRampNeg_MW'] = 0.0
    if ENABLE_ELECTROLYZER and not ENABLE_LOW_TEMP_ELECTROLYZER and hasattr(model, 'qSteamElectrolyzerRampPos'):
        results_df['qSteamElectrolyzerRampPos_MWth'] = [get_var_value(model.qSteamElectrolyzerRampPos, t) for t in hours]
        results_df['qSteamElectrolyzerRampNeg_MWth'] = [get_var_value(model.qSteamElectrolyzerRampNeg, t) for t in hours]
    else:
        results_df['qSteamElectrolyzerRampPos_MWth'] = 0.0
        results_df['qSteamElectrolyzerRampNeg_MWth'] = 0.0

    # Battery Variables
    if ENABLE_BATTERY:
        results_df['Battery_SOC_MWh'] = [get_var_value(model.BatterySOC, t) for t in hours]
        results_df['Battery_Charge_MW'] = [get_var_value(model.BatteryCharge, t) for t in hours]
        results_df['Battery_Discharge_MW'] = [get_var_value(model.BatteryDischarge, t) for t in hours]
        results_df['Battery_Charge_Binary'] = [get_var_value(model.BatteryBinaryCharge, t) for t in hours]
        results_df['Battery_Discharge_Binary'] = [get_var_value(model.BatteryBinaryDischarge, t) for t in hours]
    else:
        results_df['Battery_SOC_MWh'] = 0.0
        results_df['Battery_Charge_MW'] = 0.0
        results_df['Battery_Discharge_MW'] = 0.0
        results_df['Battery_Charge_Binary'] = 0.0
        results_df['Battery_Discharge_Binary'] = 0.0

    # --- Extract Ancillary Service Bids ---
    logging.info("Extracting AS bids...")
    # Define the list of AS products used in the model
    as_service_list = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
    # Check if AS is possible at all
    if CAN_PROVIDE_ANCILLARY_SERVICES:
        # Extract component bids if components are enabled
        if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY):
            for service in as_service_list:
                var_name = f"{service}_Turbine"
                results_df[f'{var_name}_MW'] = [get_var_value(getattr(model, var_name, None), t) if hasattr(model, var_name) else 0.0 for t in hours]
        else: # Zero out turbine bids if turbine can't provide AS
             for service in as_service_list: results_df[f'{service}_Turbine_MW'] = 0.0

        if ENABLE_ELECTROLYZER:
            for service in as_service_list:
                 var_name = f"{service}_Electrolyzer"
                 results_df[f'{var_name}_MW'] = [get_var_value(getattr(model, var_name, None), t) if hasattr(model, var_name) else 0.0 for t in hours]
        else: # Zero out electrolyzer bids if disabled
             for service in as_service_list: results_df[f'{service}_Electrolyzer_MW'] = 0.0

        if ENABLE_BATTERY:
            for service in as_service_list:
                 var_name = f"{service}_Battery"
                 results_df[f'{var_name}_MW'] = [get_var_value(getattr(model, var_name, None), t) if hasattr(model, var_name) else 0.0 for t in hours]
        else: # Zero out battery bids if disabled
             for service in as_service_list: results_df[f'{service}_Battery_MW'] = 0.0

        # Extract total bids (These should exist as Vars if CAN_PROVIDE_AS is True)
        for service in as_service_list:
            total_var_name = f"Total_{service}"
            # Check if the total variable exists (it should if AS is enabled)
            if hasattr(model, total_var_name) and isinstance(getattr(model, total_var_name), pyo.Var):
                 results_df[f'{total_var_name}_MW'] = [get_var_value(getattr(model, total_var_name), t) for t in hours]
            else:
                 # If it's missing or not a Var (e.g., fixed Param), set column to 0
                 results_df[f'{total_var_name}_MW'] = 0.0
                 # Log a warning if it was expected to be a Var
                 if not (hasattr(model, total_var_name) and isinstance(getattr(model, total_var_name), pyo.Var)):
                      logging.debug(f"Total AS variable '{total_var_name}' not found or is not a Var, setting column to 0.")

    else: # If AS is impossible, zero out all AS bid columns
        logging.info("Ancillary services disabled by configuration. Setting all AS bid columns to 0.")
        for comp in ['Turbine', 'Electrolyzer', 'Battery', 'Total']:
             for service in as_service_list:
                 var_name = f"{service}_{comp}" if comp != 'Total' else f"Total_{service}"
                 results_df[f'{var_name}_MW'] = 0.0


    # --- Extract Input Prices/Factors ---
    logging.info("Extracting input prices and factors...")
    results_df['EnergyPrice_LMP_USDperMWh'] = [get_param(model, 'energy_price', t) for t in hours]

    # Extract AS parameters only if AS is possible
    if CAN_PROVIDE_ANCILLARY_SERVICES:
        # Use the same ISO mapping as model.py/revenue_cost.py to find relevant param names
        iso_service_map = { # Map ISO names to internal model names if needed (ensure consistency)
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
            'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
            'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'], # Include factors
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
            'ISONE': ['Spin10', 'NSpin10', 'OR30'],
            'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD'] # Added RamU/D based on model.py
        }
        iso_suffix = target_iso
        if iso_suffix in iso_service_map:
            for service_iso in iso_service_map[iso_suffix]:
                # Extract Price (skip for factors like performance_score)
                if not any(f in service_iso for f in ['factor', 'score', 'ratio']):
                     results_df[f'p_{service_iso}_{iso_suffix}'] = [get_param(model, f'p_{service_iso}', t) for t in hours]
                # Extract Deploy Factor
                results_df[f'deploy_factor_{service_iso}_{iso_suffix}'] = [get_param(model, f'deploy_factor_{service_iso}', t) for t in hours]
                # Extract Adder
                results_df[f'loc_{service_iso}_{iso_suffix}'] = [get_param(model, f'loc_{service_iso}', t) for t in hours]
                # Extract Winning Rate
                results_df[f'winning_rate_{service_iso}_{iso_suffix}'] = [get_param(model, f'winning_rate_{service_iso}', t, 1.0) for t in hours]
                # Extract specific mileage/performance factors
                if iso_suffix == 'CAISO' and service_iso in ['RegU', 'RegD']:
                     results_df[f'mileage_factor_{service_iso}_{iso_suffix}'] = [get_param(model, f'mileage_factor_{service_iso}', t, 1.0) for t in hours]
                if iso_suffix == 'PJM' and service_iso == 'performance_score':
                     results_df[f'performance_score_{iso_suffix}'] = [get_param(model, 'performance_score', t, 1.0) for t in hours]
                if iso_suffix == 'PJM' and service_iso == 'mileage_ratio':
                     results_df[f'mileage_ratio_{iso_suffix}'] = [get_param(model, 'mileage_ratio', t, 1.0) for t in hours]


    # --- Calculate Deployed AS Amounts (for verification/analysis) ---
    # --- *** FIX START: Comment out deployed calculation *** ---
    # logging.info("Calculating deployed AS amounts...")
    # deployed_up = []
    # deployed_down = []
    # for t in hours:
    #     # Call the helper from constraints.py
    #     # NOTE: This helper currently only calculates BIDS, not deployed amounts.
    #     #       Actual deployment calculation would need deployment signals/factors.
    #     #       We are commenting this out as the helper doesn't return these keys.
    #     as_info = get_as_components_helper(model, t)
    #     # deployed_up.append(as_info['up_deployed']) # This key doesn't exist in as_info
    #     # deployed_down.append(as_info['down_deployed']) # This key doesn't exist in as_info
    #     deployed_up.append(0.0) # Placeholder
    #     deployed_down.append(0.0) # Placeholder
    # results_df['Deployed_Up_Reserves_MW'] = deployed_up
    # results_df['Deployed_Down_Reserves_MW'] = deployed_down
    # --- *** FIX END *** ---


    # --- Calculate Hourly Revenues (Uses ISO-specific rules) ---
    logging.info("Calculating hourly revenues...")
    # Energy Revenue
    results_df['Revenue_Energy_USD'] = results_df['pIES_MW'] * results_df['EnergyPrice_LMP_USDperMWh'] * time_factor

    # Hydrogen Revenue
    h2_value_param = get_param(model, 'H2_value', default=0.0) if ENABLE_ELECTROLYZER else 0.0
    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE:
         # Revenue from H2 sold directly + H2 dispatched from storage
         results_df['Revenue_Hydrogen_USD'] = (results_df['H2_to_Market_kg_hr'] + results_df['H2_from_Storage_kg_hr']) * h2_value_param * time_factor
    elif ENABLE_ELECTROLYZER:
         # Revenue based on total production if no storage
         results_df['Revenue_Hydrogen_USD'] = results_df['mHydrogenProduced_kg_hr'] * h2_value_param * time_factor
    else:
        results_df['Revenue_Hydrogen_USD'] = 0.0

    # Ancillary Service Revenue (using the replicated calculation function)
    # This function calculates the total revenue for the hour based on bids and prices/factors
    results_df['Revenue_Ancillary_USD'] = [calculate_hourly_as_revenue(model, t) for t in hours]

    # Total Revenue
    results_df['Revenue_Total_USD'] = results_df['Revenue_Energy_USD'] + results_df['Revenue_Hydrogen_USD'] + results_df['Revenue_Ancillary_USD']


    # --- Calculate Hourly Costs (Replicating OpexCost_rule from revenue_cost.py) ---
    logging.info("Calculating hourly costs...")
    cost_vom_turbine = 0.0
    cost_vom_electrolyzer = 0.0
    cost_vom_battery = 0.0
    cost_water = 0.0
    cost_ramping = 0.0
    cost_storage_cycle = 0.0
    cost_startup = 0.0

    # VOM Costs
    if ENABLE_NUCLEAR_GENERATOR and hasattr(model, 'vom_turbine'):
        vom_turbine_param = get_param(model, 'vom_turbine', default=0.0)
        results_df['Cost_VOM_Turbine_USD'] = results_df['pTurbine_MW'] * vom_turbine_param * time_factor
        cost_vom_turbine = results_df['Cost_VOM_Turbine_USD'].sum()
    else: results_df['Cost_VOM_Turbine_USD'] = 0.0

    if ENABLE_ELECTROLYZER and hasattr(model, 'vom_electrolyzer'):
        vom_electrolyzer_param = get_param(model, 'vom_electrolyzer', default=0.0)
        results_df['Cost_VOM_Electrolyzer_USD'] = results_df['pElectrolyzer_MW'] * vom_electrolyzer_param * time_factor
        cost_vom_electrolyzer = results_df['Cost_VOM_Electrolyzer_USD'].sum()
    else: results_df['Cost_VOM_Electrolyzer_USD'] = 0.0

    # Battery VOM (Example: if cost is per MWh cycled)
    if ENABLE_BATTERY and hasattr(model, 'vom_battery_per_mwh_cycled'): # Check if this param exists
        vom_battery_param = get_param(model, 'vom_battery_per_mwh_cycled', default=0.0)
        results_df['Cost_VOM_Battery_USD'] = (results_df['Battery_Charge_MW'] + results_df['Battery_Discharge_MW']) * vom_battery_param * time_factor
        cost_vom_battery = results_df['Cost_VOM_Battery_USD'].sum()
    else: results_df['Cost_VOM_Battery_USD'] = 0.0

    # Water Cost
    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_water_per_kg_h2'):
        cost_water_param = get_param(model, 'cost_water_per_kg_h2', default=0.0)
        results_df['Cost_Water_USD'] = results_df['mHydrogenProduced_kg_hr'] * cost_water_param * time_factor
        cost_water = results_df['Cost_Water_USD'].sum()
    else: results_df['Cost_Water_USD'] = 0.0

    # Ramping Cost
    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_electrolyzer_ramping') and 'pElectrolyzerRampPos_MW' in results_df:
        cost_ramp_param = get_param(model, 'cost_electrolyzer_ramping', default=0.0)
        # Cost applies per MW ramped (sum of positive and negative ramps)
        results_df['Cost_Ramping_USD'] = (results_df['pElectrolyzerRampPos_MW'] + results_df['pElectrolyzerRampNeg_MW']) * cost_ramp_param
        results_df.loc[results_df.index.min(), 'Cost_Ramping_USD'] = 0.0 # No ramp cost at first hour
        cost_ramping = results_df['Cost_Ramping_USD'].sum()
    else: results_df['Cost_Ramping_USD'] = 0.0

    # H2 Storage Cycle Cost
    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE and hasattr(model, 'vom_storage_cycle'):
        cost_storage_param = get_param(model, 'vom_storage_cycle', default=0.0)
        # Cost applies per kg moved in OR out
        results_df['Cost_Storage_Cycle_USD'] = (results_df['H2_to_Storage_Input_kg_hr'] + results_df['H2_from_Storage_kg_hr']) * cost_storage_param * time_factor
        cost_storage_cycle = results_df['Cost_Storage_Cycle_USD'].sum()
    else: results_df['Cost_Storage_Cycle_USD'] = 0.0

    # Startup Cost
    if ENABLE_ELECTROLYZER and ENABLE_STARTUP_SHUTDOWN and hasattr(model, 'cost_startup_electrolyzer'):
        cost_startup_param = get_param(model, 'cost_startup_electrolyzer', default=0.0)
        results_df['Cost_Startup_USD'] = results_df['Electrolyzer_Startup(0=no,1=yes)'] * cost_startup_param
        cost_startup = results_df['Cost_Startup_USD'].sum()
    else: results_df['Cost_Startup_USD'] = 0.0

    # Total Hourly OPEX
    results_df['Cost_HourlyOpex_Total_USD'] = (results_df['Cost_VOM_Turbine_USD'] +
                                               results_df['Cost_VOM_Electrolyzer_USD'] +
                                               results_df['Cost_VOM_Battery_USD'] +
                                               results_df['Cost_Water_USD'] +
                                               results_df['Cost_Ramping_USD'] +
                                               results_df['Cost_Storage_Cycle_USD'] +
                                               results_df['Cost_Startup_USD'])


    # --- Calculate Hourly Profit ---
    logging.info("Calculating hourly profit...")
    results_df['Profit_Hourly_USD'] = results_df['Revenue_Total_USD'] - results_df['Cost_HourlyOpex_Total_USD']

    # --- Calculate Summary Statistics ---
    logging.info("Calculating summary statistics...")
    total_revenue = results_df['Revenue_Total_USD'].sum()
    total_hourly_opex = results_df['Cost_HourlyOpex_Total_USD'].sum()

    # Calculate Annualized Capex (replicating AnnualizedCapex_rule from model.py)
    total_annualized_capex = 0.0
    electrolyzer_annual_capex = 0.0
    battery_annual_capex = 0.0
    scaling_factor = len(hours) * time_factor / 8760.0 # Scale annual costs to simulation duration

    if ENABLE_ELECTROLYZER and hasattr(model, 'cost_electrolyzer_capacity'):
        cost_elec_cap_param = get_param(model, 'cost_electrolyzer_capacity', default=0.0) # Annual cost $/MW/year
        elec_capacity_mw = summary_results.get('Optimal_Electrolyzer_Capacity_MW') or summary_results.get('Fixed_Electrolyzer_Capacity_MW', 0.0)
        electrolyzer_annual_capex = elec_capacity_mw * cost_elec_cap_param * scaling_factor
        total_annualized_capex += electrolyzer_annual_capex

    if ENABLE_BATTERY:
        batt_cap_mwh = summary_results.get('Optimal_Battery_Capacity_MWh') or summary_results.get('Fixed_Battery_Capacity_MWh', 0.0)
        batt_pow_mw = summary_results.get('Optimal_Battery_Power_MW') or summary_results.get('Fixed_Battery_Power_MW', 0.0)
        cost_batt_cap_mwh_yr = get_param(model, 'BatteryCapex_USD_per_MWh_year', default=0.0)
        cost_batt_pow_mw_yr = get_param(model, 'BatteryCapex_USD_per_MW_year', default=0.0)
        cost_batt_fom_mwh_yr = get_param(model, 'BatteryFixedOM_USD_per_MWh_year', default=0.0)
        battery_annual_capex = (batt_cap_mwh * cost_batt_cap_mwh_yr +
                                batt_pow_mw * cost_batt_pow_mw_yr +
                                batt_cap_mwh * cost_batt_fom_mwh_yr) * scaling_factor
        total_annualized_capex += battery_annual_capex

    summary_results['Total_Revenue_USD'] = total_revenue
    summary_results['Total_Energy_Revenue_USD'] = results_df['Revenue_Energy_USD'].sum()
    summary_results['Total_Hydrogen_Revenue_USD'] = results_df['Revenue_Hydrogen_USD'].sum()
    summary_results['Total_Ancillary_Revenue_USD'] = results_df['Revenue_Ancillary_USD'].sum()

    summary_results['Total_Hourly_Opex_USD'] = total_hourly_opex
    summary_results['Total_VOM_Cost_USD'] = cost_vom_turbine + cost_vom_electrolyzer + cost_vom_battery
    summary_results['Total_Water_Cost_USD'] = cost_water
    summary_results['Total_Ramping_Cost_USD'] = cost_ramping
    summary_results['Total_Storage_Cycle_Cost_USD'] = cost_storage_cycle
    summary_results['Total_Startup_Cost_USD'] = cost_startup

    summary_results['Total_Annualized_Capex_USD'] = total_annualized_capex
    summary_results['Electrolyzer_Annualized_Capex_USD'] = electrolyzer_annual_capex
    summary_results['Battery_Annualized_Capex_USD'] = battery_annual_capex

    # Calculated Profit vs Objective Value
    summary_results['Total_Profit_Calculated_USD'] = total_revenue - total_hourly_opex - total_annualized_capex
    summary_results['Objective_Value_USD'] = get_pyomo_value(model.TotalProfit_Objective, default=None)

    # Other Summary Metrics
    summary_results['Total_H2_Produced_kg'] = results_df['mHydrogenProduced_kg_hr'].sum() * time_factor
    if ENABLE_ELECTROLYZER and ENABLE_H2_STORAGE:
        summary_results['Total_H2_to_Market_Direct_kg'] = results_df['H2_to_Market_kg_hr'].sum() * time_factor
        summary_results['Total_H2_from_Storage_kg'] = results_df['H2_from_Storage_kg_hr'].sum() * time_factor
        summary_results['Final_H2_Storage_Level_kg'] = results_df['H2_Storage_Level_kg'].iloc[-1] if not results_df.empty else 0.0

    # Capacity Factor based on actual power vs optimal/fixed capacity
    elec_capacity_val = summary_results.get('Optimal_Electrolyzer_Capacity_MW') or summary_results.get('Fixed_Electrolyzer_Capacity_MW', 0.0)
    if elec_capacity_val > 1e-6:
        avg_elec_power_actual = results_df['pElectrolyzer_MW'].mean()
        capacity_factor_actual = avg_elec_power_actual / elec_capacity_val
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = capacity_factor_actual
    else:
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = 0.0

    # --- Final Formatting and Saving ---
    # Add check for empty summary_results before writing
    if not summary_results:
        logging.warning("Summary results dictionary is empty. Skipping summary file generation.")
        print("Warning: Summary results dictionary is empty. Skipping summary file generation.")
        # Still save hourly results if they exist
        if not results_df.empty:
            output_csv_path = results_dir / f'{target_iso}_Hourly_Results_Comprehensive.csv'
            try:
                results_df.round(4).to_csv(output_csv_path)
                logging.info(f"Comprehensive hourly results saved to {output_csv_path}")
                print(f"Comprehensive hourly results saved to {output_csv_path}")
            except Exception as e:
                logging.error(f"Failed to save hourly results CSV: {e}")
                print(f"Error: Failed to save hourly results CSV to {output_csv_path}")
        return results_df, summary_results # Return potentially empty dict


    results_df = results_df.round(4) # Round numeric columns for cleaner output
    output_csv_path = results_dir / f'{target_iso}_Hourly_Results_Comprehensive.csv'
    try:
        results_df.to_csv(output_csv_path)
        logging.info(f"Comprehensive hourly results saved to {output_csv_path}")
        print(f"Comprehensive hourly results saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save hourly results CSV: {e}")
        print(f"Error: Failed to save hourly results CSV to {output_csv_path}")

    output_summary_path = results_dir / f'{target_iso}_Summary_Results.txt'
    try:
        logging.info(f"Attempting to write summary results to: {output_summary_path}")
        with open(output_summary_path, 'w') as f:
            f.write(f"--- Summary Results for {target_iso} ---\n")
            logging.debug(f"Summary dictionary contents: {summary_results}") # Log content before writing
            for key, value in summary_results.items():
                try:
                    # Handle None explicitly before formatting
                    if value is None:
                        line = f"{key}: None\n"
                    # --- *** FIX START: Remove np.float_ *** ---
                    # Format numbers nicely, check against float and np.float64
                    elif isinstance(value, (float, np.float64)):
                         line = f"{key}: {value:,.4f}\n"
                    # Check against int and np integer types
                    elif isinstance(value, (int, np.integer)): # np.integer covers np.int_, np.int64 etc.
                        line = f"{key}: {value:,}\n"
                    # --- *** FIX END *** ---
                    else: # Handle strings, bools, etc.
                        line = f"{key}: {value}\n"
                    f.write(line)
                except Exception as write_err:
                    # Log error during formatting/writing specific key-value pair
                    logging.error(f"Error writing summary key '{key}' with value '{value}' (type: {type(value)}): {write_err}")
                    f.write(f"{key}: ERROR_WRITING_VALUE\n") # Write placeholder on error

            # Compare Objective vs Calculated Profit
            obj_val = summary_results.get('Objective_Value_USD')
            calc_prof = summary_results.get('Total_Profit_Calculated_USD')
            # Check if both are numeric before comparing
            if isinstance(obj_val, (int, float, np.number)) and isinstance(calc_prof, (int, float, np.number)):
                diff = calc_prof - obj_val
                f.write(f"\nObjective vs Calculated Profit Diff: {diff:,.4f}\n")
                if abs(diff) > 1.0: # Use a tolerance (e.g., $1) for floating point issues
                    f.write("WARNING: Significant difference between objective value and calculated profit!\n")
                    logging.warning("Significant difference between objective value and calculated profit detected.")
            elif obj_val is None or calc_prof is None:
                 f.write("\nCould not compare Objective vs Calculated Profit (Objective or Calculated value is None).\n")
            else:
                 f.write("\nCould not compare Objective vs Calculated Profit (one or both non-numeric).\n")
        logging.info(f"Summary results successfully saved to {output_summary_path}")
        print(f"Summary results saved to {output_summary_path}")
    except Exception as e:
        logging.error(f"Failed to save summary results file: {e}", exc_info=True) # Log full traceback
        print(f"Error: Failed to save summary results file to {output_summary_path}")

    # Print summary to console as well
    print("\n--- Summary ---")
    for key, value in summary_results.items():
         try:
             if value is None: print(f"{key}: None")
             # --- *** FIX START: Remove np.float_ *** ---
             elif isinstance(value, (float, np.float64)): print(f"{key}: {value:,.4f}")
             elif isinstance(value, (int, np.integer)): print(f"{key}: {value:,}")
             # --- *** FIX END *** ---
             else: print(f"{key}: {value}")
         except Exception as print_err:
              print(f"{key}: ERROR_PRINTING_VALUE ({print_err})")
    print("---------------")

    return results_df, summary_results
