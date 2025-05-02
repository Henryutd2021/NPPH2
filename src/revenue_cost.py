# src/revenue_cost.py

"""Revenue and cost expression rules isolated here.

Handles two modes for Ancillary Service (AS) revenue calculation based on
the model's SIMULATE_AS_DISPATCH_EXECUTION flag:
1. Bidding Strategy Mode (Flag=False): Calculates energy/performance revenue
   based on bids and provided factors (deploy_factor, mileage, performance).
2. Dispatch Execution Mode (Flag=True): Calculates energy/performance revenue
   based on the optimized *Deployed* AS variables (e.g., RegUp_Electrolyzer_Deployed)
   and market prices (e.g., LMP).

Capacity revenue is always based on winning bids.
Requires CAN_PROVIDE_ANCILLARY_SERVICES flag from config.py (passed via model).
"""
import pyomo.environ as pyo
import pandas as pd
from logging_setup import logger
# Import necessary flags from config.py (retrieved via model object 'm')
# Assumes model object 'm' has attributes like:
# m.ENABLE_H2_STORAGE, m.ENABLE_BATTERY, m.ENABLE_STARTUP_SHUTDOWN,
# m.ENABLE_ELECTROLYZER, m.ENABLE_NUCLEAR_GENERATOR,
# m.CAN_PROVIDE_ANCILLARY_SERVICES, m.SIMULATE_AS_DISPATCH_EXECUTION

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (get_param, get_var_value)
# ---------------------------------------------------------------------------

def get_param(model, param_name_base, time_index=None, default=0.0):
    """Safely gets a parameter value, handling indexing and ISO specifics."""
    param_name_iso = f"{param_name_base}_{model.TARGET_ISO}"
    param_to_get = None
    if hasattr(model, param_name_iso):
        param_to_get = getattr(model, param_name_iso)
    elif hasattr(model, param_name_base):
        param_to_get = getattr(model, param_name_base)
    else:
        return default
    try:
        if param_to_get.is_indexed():
            if time_index is not None and time_index in param_to_get:
                val = pyo.value(param_to_get[time_index], exception=False)
            else: val = None
        else: val = pyo.value(param_to_get, exception=False)
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
         if model_component.is_indexed():
             # Ensure index is valid before accessing
             if time_index is not None and time_index in model_component.index_set():
                 val = pyo.value(model_component[time_index], exception=False)
             else: val = None
         else: val = pyo.value(model_component, exception=False)
         return default if val is None else val
     except Exception as e:
         logger.debug(f"Error getting variable value: {e}")
         return default

def get_total_deployed_as(m, t, service_name):
    """Helper to sum deployed AS from all relevant components for a given service."""
    total_deployed = 0.0
    # Check which components might provide this service and if their deployed vars exist
    components = []
    if m.ENABLE_ELECTROLYZER: components.append('Electrolyzer')
    if m.ENABLE_BATTERY: components.append('Battery')
    if m.ENABLE_NUCLEAR_GENERATOR and (m.ENABLE_ELECTROLYZER or m.ENABLE_BATTERY): components.append('Turbine')

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            total_deployed += get_var_value(deployed_var, t, default=0.0)
    return total_deployed


# ---------------------------------------------------------------------------
# REVENUE COMPONENTS
# ---------------------------------------------------------------------------

def EnergyRevenue_rule(m):
    """Calculate net energy market revenue for the entire period."""
    try:
        if not hasattr(m, 'pIES') or not hasattr(m, 'energy_price'):
             logger.error("Missing pIES or energy_price for EnergyRevenue_rule.")
             return 0.0
        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
        total_revenue = sum(get_var_value(m.pIES, t) * get_param(m, 'energy_price', t) * time_factor
                           for t in m.TimePeriods)
        return total_revenue
    except Exception as e:
        logger.error(f"Error in EnergyRevenue_rule: {e}")
        return 0.0

def HydrogenRevenue_rule(m):
    """Calculate revenue from selling hydrogen for the entire period."""
    if not m.ENABLE_ELECTROLYZER: return 0.0
    try:
        h2_value = get_param(m, 'H2_value', default=0.0)
        h2_subsidy = get_param(m, 'hydrogen_subsidy_per_kg', default=0.0) # Get subsidy value
        effective_h2_value = h2_value + h2_subsidy # Add subsidy to base value
        if effective_h2_value <= 1e-6: return 0.0 # Skip if total value is zero

        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
        total_revenue = 0

        if not m.ENABLE_H2_STORAGE:
            if not hasattr(m, 'mHydrogenProduced'):
                 logger.error("Missing mHydrogenProduced for HydrogenRevenue rule (no storage).")
                 return 0.0
            total_revenue = sum(effective_h2_value * get_var_value(m.mHydrogenProduced, t) * time_factor
                               for t in m.TimePeriods)
        else:
             if not hasattr(m, 'H2_to_market') or not hasattr(m, 'H2_from_storage') or not hasattr(m, 'mHydrogenProduced'): # Added check for mHydrogenProduced
                 logger.error("Missing H2_to_market, H2_from_storage, or mHydrogenProduced for HydrogenRevenue rule (with storage).")
                 return 0.0
             # --- Original Logic (Revenue only from sold/discharged H2) ---
             # total_revenue = sum(h2_value * (get_var_value(m.H2_to_market, t) + get_var_value(m.H2_from_storage, t)) * time_factor
             #                    for t in m.TimePeriods)

             # --- Modified Logic (Apply subsidy to ALL produced hydrogen, value to sold/discharged) ---
             revenue_from_sales = sum(h2_value * (get_var_value(m.H2_to_market, t) + get_var_value(m.H2_from_storage, t)) * time_factor
                                       for t in m.TimePeriods)
             revenue_from_subsidy = sum(h2_subsidy * get_var_value(m.mHydrogenProduced, t) * time_factor
                                       for t in m.TimePeriods) # Subsidy based on total production
             total_revenue = revenue_from_sales + revenue_from_subsidy
             
        return total_revenue
    except AttributeError as e:
        logger.error(f"Missing variable/parameter for HydrogenRevenue rule: {e}.")
        return 0.0
    except Exception as e:
        logger.error(f"Error in HydrogenRevenue rule: {e}")
        return 0.0

# --- ISO-Specific Ancillary Revenue Rules ---

def _calculate_total_as_revenue(m, iso_rule_func):
    """Helper to sum hourly AS revenue over the period."""
    # Use flag stored on model object
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
        return 0.0

    total_as_revenue = 0
    time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
    try:
        for t in m.TimePeriods:
            hourly_revenue_rate = iso_rule_func(m, t) # Calculate $/hr rate
            total_as_revenue += hourly_revenue_rate * time_factor # Multiply by duration
        return total_as_revenue
    except Exception as e:
        logger.error(f"Error summing hourly AS revenue: {e}")
        return 0.0

# --- Hourly Calculation Logic Functions (Conditional Logic Added) ---
# These now check m.SIMULATE_AS_DISPATCH_EXECUTION for energy/performance payments

def _calculate_hourly_spp_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
    mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
    cap_payment = bid * win_rate * mcp_cap
    energy_perf_payment = 0.0
    if simulate_dispatch:
        # Payment based on total deployed amount (summed across components)
        deployed_amount = get_total_deployed_as(m, t, internal_service)
        energy_perf_payment = deployed_amount * lmp # Simplified: Use LMP for performance value
    else:
        # Original logic: Based on bid & factors (using defaults mileage=1, perf=1)
        mileage = 1.0; perf = 1.0
        energy_perf_payment = bid * mileage * perf * lmp # Note: This uses total bid, not winning bid

    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
    mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
    cap_payment = bid * win_rate * mcp_cap
    energy_perf_payment = 0.0
    if simulate_dispatch:
        deployed_amount = get_total_deployed_as(m, t, internal_service)
        energy_perf_payment = deployed_amount * lmp # Simplified
    else:
        mileage = 1.0; perf = 1.0
        energy_perf_payment = bid * mileage * perf * lmp

    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp

        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_caiso_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up (No LMP mileage payment in CAISO structure by default)
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
    mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + adder
    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
    mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
    hourly_revenue_rate += (bid * win_rate * mcp_cap) + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp

        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_ercot_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up (Assuming similar structure to SPP based on user rule)
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
    mcp_cap = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
    cap_payment = bid * win_rate * mcp_cap
    energy_perf_payment = 0.0
    if simulate_dispatch:
        deployed_amount = get_total_deployed_as(m, t, internal_service)
        energy_perf_payment = deployed_amount * lmp
    else:
        mileage = 1.0; perf = 1.0 # Default factors
        energy_perf_payment = bid * mileage * perf * lmp
    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

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
        mileage = 1.0; perf = 1.0 # Default factors
        energy_perf_payment = bid * mileage * perf * lmp
    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp
        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_pjm_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation (Combined Up/Down Bid, has Cap and Perf MCPs)
    service = 'Reg' # Internal service prefix? Let's assume RegUp/RegDown track components
    bid_up = get_var_value(getattr(m, 'Total_RegUp', None), t)
    bid_down = get_var_value(getattr(m, 'Total_RegDown', None), t)
    total_reg_bid = bid_up + bid_down # Total capacity offered
    mcp_cap = get_param(m, 'p_RegCap', t) # Capacity price
    adder = get_param(m, f'loc_{service}', t)
    win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
    cap_payment = total_reg_bid * win_rate * mcp_cap
    perf_payment = 0.0
    if simulate_dispatch:
        # Performance payment depends on actual movement (deployment)
        # PJM Reg Market has complex performance scoring & mileage ratio.
        # Simplification: Use deployed RegUp/RegDown * LMP, similar to user rule for other ISOs
        deployed_up = get_total_deployed_as(m, t, 'RegUp')
        deployed_down = get_total_deployed_as(m, t, 'RegDown')
        # Net deployed energy equivalent * LMP (simplification)
        perf_payment = (deployed_up - deployed_down) * lmp # Highly simplified - real PJM is complex
    else:
        # Original user rule: Mileage*Perf*LMP based on total bid
        mileage = get_param(m, 'mileage_ratio', t, 1.0)
        perf = get_param(m, 'performance_score', t, 1.0)
        # PJM has p_RegPerf. User rule uses LMP. Sticking to user rule for consistency:
        perf_payment = total_reg_bid * mileage * perf * lmp
    hourly_revenue_rate += cap_payment + perf_payment + adder

    # Reserves
    reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'} # TMR maps to 30min internal name
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp
        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_nyiso_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Capacity (Similar to PJM Cap, user rule applied for performance)
    service = 'RegC' # Capacity price name
    bid_up = get_var_value(getattr(m, 'Total_RegUp', None), t)
    bid_down = get_var_value(getattr(m, 'Total_RegDown', None), t)
    total_reg_bid = bid_up + bid_down
    mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
    cap_payment = total_reg_bid * win_rate * mcp_cap
    energy_perf_payment = 0.0
    if simulate_dispatch:
        deployed_up = get_total_deployed_as(m, t, 'RegUp')
        deployed_down = get_total_deployed_as(m, t, 'RegDown')
        energy_perf_payment = (deployed_up - deployed_down) * lmp # Simplified LMP based payment
    else:
        # User rule applied (Mileage * Perf * LMP, with default factors)
        mileage = 1.0; perf = 1.0
        energy_perf_payment = total_reg_bid * mileage * perf * lmp
    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

    # Reserves
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': 'ThirtyMin'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp
        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_isone_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Reserves (No separate Regulation capacity price usually)
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': 'ThirtyMin'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp
        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

def _calculate_hourly_miso_revenue(m, t):
    hourly_revenue_rate = 0.0
    lmp = get_param(m, 'energy_price', t)
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation (Combined Up/Down Bid, single price 'p_Reg')
    service = 'Reg'
    bid_up = get_var_value(getattr(m, 'Total_RegUp', None), t)
    bid_down = get_var_value(getattr(m, 'Total_RegDown', None), t)
    total_reg_bid = bid_up + bid_down
    mcp_cap = get_param(m, f'p_{service}', t); adder = get_param(m, f'loc_{service}', t); win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
    cap_payment = total_reg_bid * win_rate * mcp_cap
    energy_perf_payment = 0.0
    if simulate_dispatch:
        deployed_up = get_total_deployed_as(m, t, 'RegUp')
        deployed_down = get_total_deployed_as(m, t, 'RegDown')
        energy_perf_payment = (deployed_up - deployed_down) * lmp # Simplified LMP based payment
    else:
        # User rule applied (Mileage * Perf * LMP, with default factors)
        mileage = 1.0; perf = 1.0
        energy_perf_payment = total_reg_bid * mileage * perf * lmp
    hourly_revenue_rate += cap_payment + energy_perf_payment + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None); bid = get_var_value(bid_var, t)
        mcp = get_param(m, f'p_{service_iso}', t); adder = get_param(m, f'loc_{service_iso}', t); win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
        cap_payment = bid * win_rate * mcp
        energy_payment = 0.0
        if simulate_dispatch:
            deployed_amount = get_total_deployed_as(m, t, internal_service)
            energy_payment = deployed_amount * lmp
        else:
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t)
            energy_payment = bid * deploy_factor * lmp
        hourly_revenue_rate += cap_payment + energy_payment + adder

    return hourly_revenue_rate

# --- Total AS Revenue Rules (Call the appropriate hourly calculator) ---
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
    # This rule calculates costs based on which components are enabled and operating.
    # It uses ACTUAL operation variables like pElectrolyzer, pTurbine, etc.
    # The values of these variables are determined differently based on the simulation mode,
    # but the cost calculation logic itself doesn't need to change based on the mode flag.
    total_opex = 0
    try:
        time_factor = get_param(m, 'delT_minutes', default=60.0) / 60.0
        cost_vom_turbine = 0.0
        cost_vom_electrolyzer = 0.0
        cost_vom_battery = 0.0
        cost_water = 0.0
        cost_ramping = 0.0
        cost_storage_cycle = 0.0
        cost_startup = 0.0

        # VOM Costs (based on ACTUAL power/operation)
        if m.ENABLE_NUCLEAR_GENERATOR and hasattr(m, 'vom_turbine') and hasattr(m, 'pTurbine'):
            vom_rate = get_param(m, 'vom_turbine')
            cost_vom_turbine = sum(vom_rate * get_var_value(m.pTurbine, t) * time_factor for t in m.TimePeriods)
        if m.ENABLE_ELECTROLYZER and hasattr(m, 'vom_electrolyzer') and hasattr(m, 'pElectrolyzer'):
            vom_rate = get_param(m, 'vom_electrolyzer')
            cost_vom_electrolyzer = sum(vom_rate * get_var_value(m.pElectrolyzer, t) * time_factor for t in m.TimePeriods)
        if m.ENABLE_BATTERY and hasattr(m, 'vom_battery_per_mwh_cycled') and hasattr(m, 'BatteryCharge') and hasattr(m, 'BatteryDischarge'):
             vom_rate = get_param(m, 'vom_battery_per_mwh_cycled')
             cost_vom_battery = sum(vom_rate * (get_var_value(m.BatteryCharge, t) + get_var_value(m.BatteryDischarge, t)) * time_factor for t in m.TimePeriods)

        # Water Cost (based on ACTUAL H2 production, derived from pElectrolyzer)
        if m.ENABLE_ELECTROLYZER and hasattr(m, 'cost_water_per_kg_h2') and hasattr(m, 'mHydrogenProduced'):
            cost_rate = get_param(m, 'cost_water_per_kg_h2')
            cost_water = sum(cost_rate * get_var_value(m.mHydrogenProduced, t) * time_factor for t in m.TimePeriods) # mHydrogenProduced depends on pElectrolyzer

        # Ramping Costs (based on ACTUAL power changes in pElectrolyzer)
        if m.ENABLE_ELECTROLYZER and hasattr(m, 'cost_electrolyzer_ramping') and hasattr(m, 'pElectrolyzerRampPos') and hasattr(m, 'pElectrolyzerRampNeg'):
            cost_rate = get_param(m, 'cost_electrolyzer_ramping')
            cost_ramping = sum(cost_rate * (get_var_value(m.pElectrolyzerRampPos, t) + get_var_value(m.pElectrolyzerRampNeg, t))
                              for t in m.TimePeriods if t > m.TimePeriods.first()) # Ramping occurs between periods

        # H2 Storage Cycle Cost
        if m.ENABLE_H2_STORAGE and hasattr(m, 'vom_storage_cycle') and hasattr(m, 'H2_to_storage') and hasattr(m, 'H2_from_storage'):
             cost_rate = get_param(m, 'vom_storage_cycle')
             cost_storage_cycle = sum(cost_rate * (get_var_value(m.H2_to_storage, t) + get_var_value(m.H2_from_storage, t)) * time_factor
                                     for t in m.TimePeriods)

        # Startup Costs
        if m.ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            cost_rate = get_param(m, 'cost_startup_electrolyzer')
            cost_startup = sum(cost_rate * get_var_value(m.vElectrolyzerStartup, t) for t in m.TimePeriods)

        total_opex = (cost_vom_turbine + cost_vom_electrolyzer + cost_vom_battery +
                      cost_water + cost_ramping + cost_storage_cycle + cost_startup)
        return total_opex

    except AttributeError as e:
        logger.error(f"Missing parameter/variable for OpexCost rule: {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Error in OpexCost rule: {e}")
        return 0.0