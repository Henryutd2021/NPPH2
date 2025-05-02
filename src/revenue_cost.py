# src/revenue_cost.py

"""Revenue and cost expression rules isolated here.

Refined Ancillary Service Revenue Logic (User Specified):
- AS Revenue is ZERO if only NPP is enabled.
- Reserves (Spin, Non-Spin, etc.): Revenue = Capacity Payment + Energy Payment + Adder
    - Capacity Payment = Bid * winning_rate[t] * MCP[t]
    - Energy Payment = Bid * deploy_factor[t] * LMP[t]
    - Adder = Locational Adder[t]
- Regulation: Revenue = Capacity Payment + Performance/Mileage Payment + Adder
    - Capacity Payment = Bid * winning_rate[t] * MCP_Capacity[t]
    - Performance/Mileage = Mileage[t] * Performance[t] * LMP[t]  # Using simplified LMP-based user rule
    - Adder = Locational Adder[t]

NOTE: Requires CAN_PROVIDE_ANCILLARY_SERVICES flag from config.py.
"""
import pyomo.environ as pyo
import pandas as pd
from logging_setup import logger
# Import necessary flags from config.py
from config import (
    ENABLE_H2_STORAGE, ENABLE_BATTERY, ENABLE_STARTUP_SHUTDOWN,
    ENABLE_ELECTROLYZER, ENABLE_NUCLEAR_GENERATOR,
    CAN_PROVIDE_ANCILLARY_SERVICES # Import derived flag
)

# ---------------------------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------------------------

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
        # Log a warning if it might be expected, or debug if optional
        # logger.debug(f"Parameter '{param_name_base}' (or ISO-specific) not found in model.")
        return default

    # Now extract the value, handling indexed vs non-indexed
    try:
        if param_to_get.is_indexed():
            if time_index is not None and time_index in param_to_get:
                val = pyo.value(param_to_get[time_index], exception=False)
            else:
                # Indexed param called without valid index, or index out of bounds
                # This might happen if trying to get a time-indexed param without passing t
                # logger.warning(f"Indexed parameter '{param_name_base}' accessed without valid index '{time_index}'.")
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

# ---------------------------------------------------------------------------
# REVENUE COMPONENTS
# ---------------------------------------------------------------------------

def EnergyRevenue_rule(m):
    """Calculate net energy market revenue for the entire period."""
    # This rule is always active, depends on pIES which reflects net grid exchange
    try:
        if not hasattr(m, 'pIES') or not hasattr(m, 'energy_price'):
             logger.error("Missing pIES or energy_price for EnergyRevenue_rule.")
             return 0.0
        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0 # Hours per time step
        total_revenue = 0
        for t in m.TimePeriods:
             # Revenue = Power Sold * Price * Duration
             # Power Sold is positive pIES, purchases are negative pIES
             total_revenue += get_var_value(m.pIES, t) * get_param(m, 'energy_price', t) * time_factor
        return total_revenue
    except Exception as e:
        logger.error(f"Error in EnergyRevenue_rule: {e}")
        return 0.0

def HydrogenRevenue_rule(m):
    """Calculate revenue from selling hydrogen for the entire period."""
    # Active only if electrolyzer is enabled
    if not ENABLE_ELECTROLYZER:
        return 0.0
    try:
        h2_value = get_param(m, 'H2_value', default=0.0)
        # Return 0 immediately if H2 has no value or negative value
        if h2_value <= 1e-6:
            return 0.0

        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
        total_revenue = 0

        if not ENABLE_H2_STORAGE:
            # Revenue based on production if no storage
            if not hasattr(m, 'mHydrogenProduced'):
                 logger.error("Missing mHydrogenProduced for HydrogenRevenue rule (no storage).")
                 return 0.0
            for t in m.TimePeriods:
                 total_revenue += h2_value * get_var_value(m.mHydrogenProduced, t) * time_factor
        else:
             # Revenue based on H2 leaving system boundary (direct market + from storage)
             if not hasattr(m, 'H2_to_market') or not hasattr(m, 'H2_from_storage'):
                 logger.error("Missing H2_to_market or H2_from_storage for HydrogenRevenue rule (with storage).")
                 return 0.0
             for t in m.TimePeriods:
                  h2_sold = get_var_value(m.H2_to_market, t) + get_var_value(m.H2_from_storage, t)
                  total_revenue += h2_value * h2_sold * time_factor
        return total_revenue
    except AttributeError as e:
        logger.error(f"Missing variable/parameter for HydrogenRevenue rule: {e}.")
        return 0.0
    except Exception as e:
        logger.error(f"Error in HydrogenRevenue rule: {e}")
        return 0.0

# --- ISO-Specific Ancillary Revenue Rules ---
# These calculate total revenue over the entire period

def _calculate_total_as_revenue(m, iso_rule_func):
    """Helper to sum hourly AS revenue over the period."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES:
        return 0.0 # No AS revenue if system cannot provide services

    total_as_revenue = 0
    time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
    try:
        for t in m.TimePeriods:
            # Calculate the hourly revenue *rate* using the specific ISO logic
            hourly_revenue_rate = iso_rule_func(m, t)
            total_as_revenue += hourly_revenue_rate * time_factor
        return total_as_revenue
    except Exception as e:
        # Log the error from the calling context (specific ISO rule)
        # Error is already logged within the hourly function if it fails there
        # This catches errors in the loop itself
        logger.error(f"Error summing hourly AS revenue: {e}")
        return 0.0

# Hourly calculation logic functions (called by the total calculators below)
def _calculate_hourly_spp_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation Up
    service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
    # Regulation Down
    service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
    # Reserves
    for service_iso in ['Spin', 'Sup', 'RamU', 'RamD', 'UncU']:
        internal_name_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None) # Get the variable object itself
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_caiso_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation Up (No LMP mileage payment in CAISO structure)
    service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + adder
    # Regulation Down
    service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + adder
    # Reserves
    for service_iso in ['Spin', 'NSpin', 'RMU', 'RMD']:
        internal_name_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_ercot_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation Up
    service = 'RegU'; bid = get_var_value(m.Total_RegUp, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
    # Regulation Down
    service = 'RegD'; bid = get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder
    # Reserves
    for service_iso in ['Spin', 'NSpin', 'ECRS']:
        internal_name_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_pjm_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation (Combined Up/Down Bid)
    service = 'Reg'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, 'p_RegCap', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = get_param(m, 'mileage_ratio', t, 1.0); perf = get_param(m, 'performance_score', t, 1.0)
    # Note: PJM has RegPerf price too. The user rule uses Mileage*Perf*LMP instead. Following user rule.
    cap_payment = bid * win_rate * mcp_cap
    perf_payment = mileage * perf * lmp # User rule for performance payment
    hourly_revenue_rate += cap_payment + perf_payment + adder
    # Reserves
    for service_iso in ['Syn', 'Rse', 'TMR']:
        internal_name_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': '30Min'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_nyiso_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation Capacity
    service = 'RegC'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
    # Reserves
    for service_iso in ['Spin10', 'NSpin10', 'Res30']:
        internal_name_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': '30Min'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_isone_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Reserves (No Regulation in data?)
    for service_iso in ['Spin10', 'NSpin10', 'OR30']:
        internal_name_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': '30Min'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate

def _calculate_hourly_miso_revenue(m, t):
    hourly_revenue_rate = 0
    lmp = get_param(m, 'energy_price', t)
    # Regulation
    service = 'Reg'; bid = get_var_value(m.Total_RegUp, t) + get_var_value(m.Total_RegDown, t); mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0); mileage = 1.0; perf = 1.0
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + (mileage * perf * lmp) + adder # User rule applied
    # Reserves
    for service_iso in ['Spin', 'Sup', 'STR', 'RamU', 'RamD']:
        internal_name_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': '30Min', 'RamU': 'RampUp', 'RamD': 'RampDown'}
        total_var_name = f"Total_{internal_name_map[service_iso]}"
        total_var = getattr(m, total_var_name, None)
        bid = get_var_value(total_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); deploy = get_param(m, f'deploy_factor_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        hourly_revenue_rate += (bid * win_rate * mcp) + (bid * deploy * lmp) + adder
    return hourly_revenue_rate


# --- Total AS Revenue Rules (for Objective Function) ---
# These call the helper to sum the results from the hourly calculators
def AncillaryRevenue_SPP_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_spp_revenue)
def AncillaryRevenue_CAISO_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_caiso_revenue)
def AncillaryRevenue_ERCOT_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_ercot_revenue)
def AncillaryRevenue_PJM_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_pjm_revenue)
def AncillaryRevenue_NYISO_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_nyiso_revenue)
def AncillaryRevenue_ISONE_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_isone_revenue)
def AncillaryRevenue_MISO_rule(m): return _calculate_total_as_revenue(m, _calculate_hourly_miso_revenue)


# ---------------------------------------------------------------------------
# OPERATIONAL COST COMPONENTS
# ---------------------------------------------------------------------------

def OpexCost_rule(m):
    """Calculate total hourly operational costs over the entire period."""
    # This rule calculates costs based on which components are enabled and operating
    total_opex = 0
    try:
        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0 # Hours per time step
        cost_vom_turbine = 0.0
        cost_vom_electrolyzer = 0.0
        cost_vom_battery = 0.0
        cost_water = 0.0
        cost_ramping = 0.0
        cost_storage_cycle = 0.0
        cost_startup = 0.0

        # Component VOM Costs
        if ENABLE_NUCLEAR_GENERATOR and hasattr(m, 'vom_turbine') and hasattr(m, 'pTurbine'):
            vom_rate = get_param(m, 'vom_turbine')
            for t in m.TimePeriods: cost_vom_turbine += vom_rate * get_var_value(m.pTurbine, t) * time_factor
        if ENABLE_ELECTROLYZER and hasattr(m, 'vom_electrolyzer') and hasattr(m, 'pElectrolyzer'):
            vom_rate = get_param(m, 'vom_electrolyzer')
            for t in m.TimePeriods: cost_vom_electrolyzer += vom_rate * get_var_value(m.pElectrolyzer, t) * time_factor
        # Add Battery VOM if defined (e.g., $/MWh cycled)
        if ENABLE_BATTERY and hasattr(m, 'vom_battery_per_mwh_cycled'): # Check if param exists in model
             vom_rate = get_param(m, 'vom_battery_per_mwh_cycled')
             for t in m.TimePeriods:
                  mwh_charged = get_var_value(m.BatteryCharge, t) * time_factor
                  mwh_discharged = get_var_value(m.BatteryDischarge, t) * time_factor
                  cost_vom_battery += vom_rate * (mwh_charged + mwh_discharged) # Cost applied per MWh in OR out

        # Water Cost
        if ENABLE_ELECTROLYZER and hasattr(m, 'cost_water_per_kg_h2') and hasattr(m, 'mHydrogenProduced'):
            cost_rate = get_param(m, 'cost_water_per_kg_h2')
            for t in m.TimePeriods: cost_water += cost_rate * get_var_value(m.mHydrogenProduced, t) * time_factor

        # Ramping Costs
        if ENABLE_ELECTROLYZER and hasattr(m, 'cost_electrolyzer_ramping') and hasattr(m, 'pElectrolyzerRampPos') and hasattr(m, 'pElectrolyzerRampNeg'):
            cost_rate = get_param(m, 'cost_electrolyzer_ramping')
            for t in m.TimePeriods:
                 if t > m.TimePeriods.first(): # Ramping occurs between periods
                      ramp_mw = get_var_value(m.pElectrolyzerRampPos, t) + get_var_value(m.pElectrolyzerRampNeg, t)
                      cost_ramping += cost_rate * ramp_mw
        # Add other ramping costs (steam?) if defined

        # H2 Storage Cycle Cost
        if ENABLE_H2_STORAGE and hasattr(m, 'vom_storage_cycle') and hasattr(m, 'H2_to_storage') and hasattr(m, 'H2_from_storage'):
             cost_rate = get_param(m, 'vom_storage_cycle')
             for t in m.TimePeriods:
                 # Cost applied per kg moved in OR out
                 kg_in = get_var_value(m.H2_to_storage, t) * time_factor
                 kg_out = get_var_value(m.H2_from_storage, t) * time_factor
                 cost_storage_cycle += cost_rate * (kg_in + kg_out)

        # Startup Costs
        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            cost_rate = get_param(m, 'cost_startup_electrolyzer')
            for t in m.TimePeriods: cost_startup += cost_rate * get_var_value(m.vElectrolyzerStartup, t)

        total_opex = (cost_vom_turbine +
                      cost_vom_electrolyzer +
                      cost_vom_battery + # Added battery VOM
                      cost_water +
                      cost_ramping +
                      cost_storage_cycle +
                      cost_startup)
        return total_opex

    except AttributeError as e:
        logger.error(f"Missing parameter/variable for OpexCost rule: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error in OpexCost rule: {e}")
        return 0.0