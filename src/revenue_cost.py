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
# MODIFICATION: Direct import from utils
from utils import get_param, get_var_value, get_total_deployed_as # get_param still useful for fixed values, get_var_value for post-processing (not rules)
# Import necessary flags from config.py (retrieved via model object 'm')
# Assumes model object 'm' has attributes like:
# m.ENABLE_H2_STORAGE, m.ENABLE_BATTERY, m.ENABLE_STARTUP_SHUTDOWN,
# m.ENABLE_ELECTROLYZER, m.ENABLE_NUCLEAR_GENERATOR,
# m.CAN_PROVIDE_ANCILLARY_SERVICES, m.SIMULATE_AS_DISPATCH_EXECUTION

# ---------------------------------------------------------------------------
# HELPER FUNCTION (Symbolic Sum for AS Deployed - Local to this module)
# ---------------------------------------------------------------------------
def _get_symbolic_deployed_sum(m, t, service_name):
    """Helper to create symbolic sum of deployed AS for a service."""
    total_deployed_expr = 0.0
    # Access flags via model object 'm'
    enable_electrolyzer = getattr(m,'ENABLE_ELECTROLYZER',False)
    enable_battery = getattr(m,'ENABLE_BATTERY',False)
    enable_npp = getattr(m,'ENABLE_NUCLEAR_GENERATOR',False)

    components = []
    if enable_electrolyzer: components.append('Electrolyzer')
    if enable_battery: components.append('Battery')
    if enable_npp and (enable_electrolyzer or enable_battery):
        components.append('Turbine')

    for comp_name in components:
        deployed_var_name = f"{service_name}_{comp_name}_Deployed"
        if hasattr(m, deployed_var_name):
            deployed_var = getattr(m, deployed_var_name)
            # Ensure var is indexed and index is valid
            if deployed_var.is_indexed() and t in deployed_var.index_set():
                total_deployed_expr += deployed_var[t]
            elif not deployed_var.is_indexed(): # Should not happen for time series var
                 logger.warning(f"Deployed AS variable {deployed_var_name} is not indexed. Skipping in sum.")
    return total_deployed_expr


# ---------------------------------------------------------------------------
# REVENUE COMPONENTS (Corrected for Symbolic Definition)
# ---------------------------------------------------------------------------

def EnergyRevenue_rule(m):
    """Calculate net energy market revenue expression for the objective function."""
    try:
        if not hasattr(m, 'pIES') or not hasattr(m, 'energy_price') or not hasattr(m, 'delT_minutes'):
             logger.error("Missing pIES, energy_price, or delT_minutes for EnergyRevenue_rule definition.")
             return 0.0 # Return 0 expression if components missing

        time_factor = pyo.value(m.delT_minutes) / 60.0 # Fixed time step duration in hours
        if time_factor <= 0:
             logger.critical("Invalid time_factor (<=0) in EnergyRevenue_rule definition.")
             raise ValueError("delT_minutes must result in a positive time_factor.")

        # Use symbolic components m.pIES[t] and m.energy_price[t]
        total_revenue_expr = sum(m.pIES[t] * m.energy_price[t] * time_factor
                                for t in m.TimePeriods)
        return total_revenue_expr

    except Exception as e:
        logger.critical(f"CRITICAL Error defining EnergyRevenue_rule expression: {e}", exc_info=True)
        raise # Re-raise exception

def HydrogenRevenue_rule(m):
    """Calculate revenue expression from selling hydrogen for the objective function."""
    # Use flags from model object
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_electrolyzer: return 0.0

    try:
        # Use Parameters directly (they are fixed during solve)
        h2_value = m.H2_value
        h2_subsidy = m.hydrogen_subsidy_per_kg

        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0:
             logger.critical("Invalid time_factor (<=0) in HydrogenRevenue_rule definition.")
             raise ValueError("delT_minutes must result in a positive time_factor.")

        total_revenue_expr = 0.0

        if not enable_h2_storage:
            if not hasattr(m, 'mHydrogenProduced'):
                 logger.error("Missing mHydrogenProduced for HydrogenRevenue rule definition (no storage).")
                 return 0.0
            # Apply value + subsidy to all produced hydrogen
            total_revenue_expr = sum((h2_value + h2_subsidy) * m.mHydrogenProduced[t] * time_factor
                                     for t in m.TimePeriods)
        else: # Storage enabled
             if not hasattr(m, 'H2_to_market') or not hasattr(m, 'H2_from_storage') or not hasattr(m, 'mHydrogenProduced'):
                 logger.error("Missing H2_to_market, H2_from_storage, or mHydrogenProduced for HydrogenRevenue rule definition (with storage).")
                 return 0.0
             # Apply subsidy to ALL produced hydrogen, value only to sold/discharged hydrogen
             revenue_from_sales_expr = sum(h2_value * (m.H2_to_market[t] + m.H2_from_storage[t]) * time_factor
                                            for t in m.TimePeriods)
             revenue_from_subsidy_expr = sum(h2_subsidy * m.mHydrogenProduced[t] * time_factor
                                             for t in m.TimePeriods)
             total_revenue_expr = revenue_from_sales_expr + revenue_from_subsidy_expr

        return total_revenue_expr
    except AttributeError as e:
        logger.critical(f"CRITICAL Missing variable/parameter for HydrogenRevenue rule definition: {e}.", exc_info=True)
        raise e # Re-raise exception
    except Exception as e:
        logger.critical(f"CRITICAL Error defining HydrogenRevenue rule expression: {e}", exc_info=True)
        raise # Re-raise exception


# --- ISO-Specific Ancillary Revenue Rules (Corrected for Symbolic Definition) ---

# --- Symbolic Hourly Calculation Logic Functions ---
# These helpers now return symbolic expressions for a single hour's revenue rate ($/hr)

def _symbolic_hourly_spp_revenue_expr(m, t):
    """Returns symbolic expression for SPP AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t] # Symbolic param
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set(): # Check var exists and indexed
        bid = bid_var[t] # Symbolic var
        # Access params symbolically
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service) # Use symbolic helper
            energy_perf_payment = deployed_amount_expr * lmp
        else:
            mileage = 1.0; perf = 1.0 # Assuming factors are 1 if not params
            energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Regulation Down (Similar symbolic implementation)
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
             deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
             energy_perf_payment = deployed_amount_expr * lmp
        else:
             mileage = 1.0; perf = 1.0
             energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Reserves (Similar symbolic implementation)
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            # Access params symbolically
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder

    return hourly_revenue_rate_expr

def _symbolic_hourly_caiso_revenue_expr(m, t):
    """Returns symbolic expression for CAISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Regulation Up
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        hourly_revenue_rate_expr += (bid * win_rate * mcp_cap) + adder
    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        hourly_revenue_rate_expr += (bid * win_rate * mcp_cap) + adder
    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
             bid = bid_var[t]
             mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
             adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
             win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
             deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
             cap_payment = bid * win_rate * mcp
             energy_payment = 0.0
             if simulate_dispatch:
                 deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                 energy_payment = deployed_amount_expr * lmp
             else:
                 energy_payment = bid * deploy_factor * lmp
             hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_ercot_revenue_expr(m, t):
    """Returns symbolic expression for ERCOT AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Regulation Up
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
            energy_perf_payment = deployed_amount_expr * lmp
        else:
            mileage = 1.0; perf = 1.0
            energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder
    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
             deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
             energy_perf_payment = deployed_amount_expr * lmp
        else:
             mileage = 1.0; perf = 1.0
             energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder
    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_pjm_revenue_expr(m, t):
    """Returns symbolic expression for PJM AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Regulation
    service = 'Reg'
    bid_up_var = getattr(m, 'Total_RegUp', None)
    bid_down_var = getattr(m, 'Total_RegDown', None)
    # Check if vars exist before using them
    if bid_up_var is not None and bid_up_var.is_indexed() and t in bid_up_var.index_set() and \
       bid_down_var is not None and bid_down_var.is_indexed() and t in bid_down_var.index_set():
        bid_up = bid_up_var[t]
        bid_down = bid_down_var[t]
        total_reg_bid = bid_up + bid_down
        mcp_cap = getattr(m, f'p_RegCap_{m.TARGET_ISO}')[t] # Specific PJM param
        adder = getattr(m, f'loc_{service}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service}_{m.TARGET_ISO}')[t]
        cap_payment = total_reg_bid * win_rate * mcp_cap
        perf_payment = 0.0
        if simulate_dispatch:
            deployed_up_expr = _get_symbolic_deployed_sum(m, t, 'RegUp')
            deployed_down_expr = _get_symbolic_deployed_sum(m, t, 'RegDown')
            perf_payment = (deployed_up_expr - deployed_down_expr) * lmp # Simplified
        else:
            mileage = getattr(m, f'mileage_ratio_{m.TARGET_ISO}')[t] # Specific PJM param
            perf = getattr(m, f'performance_score_{m.TARGET_ISO}')[t] # Specific PJM param
            perf_payment = total_reg_bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + perf_payment + adder
    # Reserves
    reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_nyiso_revenue_expr(m, t):
    """Returns symbolic expression for NYISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Regulation Capacity
    service = 'RegC'
    bid_up_var = getattr(m, 'Total_RegUp', None)
    bid_down_var = getattr(m, 'Total_RegDown', None)
    if bid_up_var is not None and bid_up_var.is_indexed() and t in bid_up_var.index_set() and \
       bid_down_var is not None and bid_down_var.is_indexed() and t in bid_down_var.index_set():
        bid_up = bid_up_var[t]
        bid_down = bid_down_var[t]
        total_reg_bid = bid_up + bid_down
        mcp_cap = getattr(m, f'p_{service}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service}_{m.TARGET_ISO}')[t]
        cap_payment = total_reg_bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_up_expr = _get_symbolic_deployed_sum(m, t, 'RegUp')
            deployed_down_expr = _get_symbolic_deployed_sum(m, t, 'RegDown')
            energy_perf_payment = (deployed_up_expr - deployed_down_expr) * lmp
        else:
            mileage = 1.0; perf = 1.0
            energy_perf_payment = total_reg_bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder
    # Reserves
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': 'ThirtyMin'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_isone_revenue_expr(m, t):
    """Returns symbolic expression for ISONE AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Reserves
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': 'ThirtyMin'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_miso_revenue_expr(m, t):
    """Returns symbolic expression for MISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Regulation
    service = 'Reg'
    bid_up_var = getattr(m, 'Total_RegUp', None)
    bid_down_var = getattr(m, 'Total_RegDown', None)
    if bid_up_var is not None and bid_up_var.is_indexed() and t in bid_up_var.index_set() and \
       bid_down_var is not None and bid_down_var.is_indexed() and t in bid_down_var.index_set():
        bid_up = bid_up_var[t]
        bid_down = bid_down_var[t]
        total_reg_bid = bid_up + bid_down
        mcp_cap = getattr(m, f'p_{service}_{m.TARGET_ISO}')[t]
        adder = getattr(m, f'loc_{service}_{m.TARGET_ISO}')[t]
        win_rate = getattr(m, f'winning_rate_{service}_{m.TARGET_ISO}')[t]
        cap_payment = total_reg_bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_up_expr = _get_symbolic_deployed_sum(m, t, 'RegUp')
            deployed_down_expr = _get_symbolic_deployed_sum(m, t, 'RegDown')
            energy_perf_payment = (deployed_up_expr - deployed_down_expr) * lmp # Simplified
        else:
            mileage = 1.0; perf = 1.0
            energy_perf_payment = total_reg_bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder
    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = getattr(m, f'p_{service_iso}_{m.TARGET_ISO}')[t]
            adder = getattr(m, f'loc_{service_iso}_{m.TARGET_ISO}')[t]
            win_rate = getattr(m, f'winning_rate_{service_iso}_{m.TARGET_ISO}')[t]
            deploy_factor = getattr(m, f'deploy_factor_{service_iso}_{m.TARGET_ISO}')[t]
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_symbolic_deployed_sum(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


# --- Total AS Revenue Rules (Assign correct hourly calculator) ---
def AncillaryRevenue_SPP_rule(m):
    """Returns symbolic expression for total SPP AS revenue."""
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_spp_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_SPP_rule: {e}", exc_info=True); raise

def AncillaryRevenue_CAISO_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_caiso_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_CAISO_rule: {e}", exc_info=True); raise

def AncillaryRevenue_ERCOT_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_ercot_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_ERCOT_rule: {e}", exc_info=True); raise

def AncillaryRevenue_PJM_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_pjm_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_PJM_rule: {e}", exc_info=True); raise

def AncillaryRevenue_NYISO_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_nyiso_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_NYISO_rule: {e}", exc_info=True); raise

def AncillaryRevenue_ISONE_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_isone_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_ISONE_rule: {e}", exc_info=True); raise

def AncillaryRevenue_MISO_rule(m):
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return sum(_symbolic_hourly_miso_revenue_expr(m, t) * time_factor for t in m.TimePeriods)
    except Exception as e: logger.critical(f"CRITICAL Error defining AncillaryRevenue_MISO_rule: {e}", exc_info=True); raise

# ---------------------------------------------------------------------------
# OPERATIONAL COST COMPONENTS (Corrected for Symbolic Definition)
# ---------------------------------------------------------------------------

def OpexCost_rule(m):
    """Calculate total hourly operational costs expression for the objective function."""
    total_opex_expr = 0.0
    # Use flags from model object
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)

    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0:
             logger.critical("Invalid time_factor (<=0) in OpexCost_rule definition.")
             raise ValueError("delT_minutes must result in a positive time_factor.")

        # VOM Costs (Use symbolic variables, Parameters accessed directly)
        cost_vom_turbine_expr = 0.0
        if enable_npp and hasattr(m, 'vom_turbine') and hasattr(m, 'pTurbine'):
            cost_vom_turbine_expr = sum(m.vom_turbine * m.pTurbine[t] * time_factor for t in m.TimePeriods)

        cost_vom_electrolyzer_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'vom_electrolyzer') and hasattr(m, 'pElectrolyzer'):
            cost_vom_electrolyzer_expr = sum(m.vom_electrolyzer * m.pElectrolyzer[t] * time_factor for t in m.TimePeriods)

        cost_vom_battery_expr = 0.0
        if enable_battery and hasattr(m, 'vom_battery_per_mwh_cycled') and hasattr(m, 'BatteryCharge') and hasattr(m, 'BatteryDischarge'):
             # Assuming rate is per MWh cycled (Charge OR Discharge equivalent)
             cost_vom_battery_expr = sum(m.vom_battery_per_mwh_cycled * (m.BatteryCharge[t] + m.BatteryDischarge[t]) / 2.0 * time_factor for t in m.TimePeriods)

        # Water Cost (Use symbolic mHydrogenProduced)
        cost_water_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'cost_water_per_kg_h2') and hasattr(m, 'mHydrogenProduced'):
            # Rate is $/kg, Var is kg/hr -> Total Cost = sum( rate * var * time_factor )
            cost_water_expr = sum(m.cost_water_per_kg_h2 * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)

        # Ramping Costs (Use symbolic ramp variables)
        cost_ramping_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'cost_electrolyzer_ramping') and hasattr(m, 'pElectrolyzerRampPos') and hasattr(m, 'pElectrolyzerRampNeg'):
            # Rate is $/MW ramped (per event)
            cost_ramping_expr = sum(m.cost_electrolyzer_ramping * (m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t])
                                   for t in m.TimePeriods if t > m.TimePeriods.first()) # Sum over t>1

        # H2 Storage Cycle Cost (Use symbolic storage variables)
        cost_storage_cycle_expr = 0.0
        if enable_h2_storage and hasattr(m, 'vom_storage_cycle') and hasattr(m, 'H2_to_storage') and hasattr(m, 'H2_from_storage'):
             # Rate is $/kg cycled, Vars are kg/hr -> Total Cost = sum( rate * (in + out) * time_factor )
             cost_storage_cycle_expr = sum(m.vom_storage_cycle * (m.H2_to_storage[t] + m.H2_from_storage[t]) * time_factor
                                          for t in m.TimePeriods)

        # Startup Costs (Use symbolic startup variable)
        cost_startup_expr = 0.0
        if enable_startup_shutdown and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            # Rate is $/startup event
            cost_startup_expr = sum(m.cost_startup_electrolyzer * m.vElectrolyzerStartup[t] for t in m.TimePeriods)

        # Sum all symbolic cost expressions
        total_opex_expr = (cost_vom_turbine_expr + cost_vom_electrolyzer_expr + cost_vom_battery_expr +
                           cost_water_expr + cost_ramping_expr + cost_storage_cycle_expr + cost_startup_expr)
        return total_opex_expr

    except AttributeError as e:
        logger.critical(f"CRITICAL Missing parameter/variable for OpexCost rule definition: {e}", exc_info=True)
        raise e # Re-raise exception
    except Exception as e:
        logger.critical(f"CRITICAL Error defining OpexCost rule expression: {e}", exc_info=True)
        raise # Re-raise exception

