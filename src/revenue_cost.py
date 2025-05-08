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
# --- IMPORT NEW HELPERS from utils.py ---
from utils import (
    get_param, # Used for fixed values like H2_value, costs, market params etc.
    get_symbolic_as_bid_sum,    # Used for summing component BID variables
    get_symbolic_as_deployed_sum # Used for summing component DEPLOYED variables
)
# Assumes model object 'm' has attributes like:
# m.ENABLE_H2_STORAGE, m.ENABLE_BATTERY, m.ENABLE_STARTUP_SHUTDOWN,
# m.ENABLE_ELECTROLYZER, m.ENABLE_NUCLEAR_GENERATOR,
# m.CAN_PROVIDE_ANCILLARY_SERVICES, m.SIMULATE_AS_DISPATCH_EXECUTION

# ---------------------------------------------------------------------------
# REVENUE COMPONENTS (Energy and Hydrogen - No changes needed)
# ---------------------------------------------------------------------------
def EnergyRevenue_rule(m):
    """Calculate net energy market revenue expression for the objective function."""
    try:
        if not hasattr(m, 'pIES') or not hasattr(m, 'energy_price') or not hasattr(m, 'delT_minutes'):
             logger.error("Missing pIES, energy_price, or delT_minutes for EnergyRevenue_rule definition.")
             return 0.0
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0: raise ValueError("delT_minutes must result in a positive time_factor.")
        total_revenue_expr = sum(m.pIES[t] * m.energy_price[t] * time_factor for t in m.TimePeriods)
        return total_revenue_expr
    except Exception as e: logger.critical(f"CRITICAL Error defining EnergyRevenue_rule expression: {e}", exc_info=True); raise

def HydrogenRevenue_rule(m):
    """Calculate revenue expression from selling hydrogen for the objective function."""
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_electrolyzer: return 0.0
    try:
        h2_value = m.H2_value # Param
        h2_subsidy = m.hydrogen_subsidy_per_kg # Param
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0: raise ValueError("delT_minutes must result in a positive time_factor.")
        total_revenue_expr = 0.0
        if not enable_h2_storage:
            if not hasattr(m, 'mHydrogenProduced'): logger.error("Missing mHydrogenProduced for H2 Revenue (no storage)."); return 0.0
            total_revenue_expr = sum((h2_value + h2_subsidy) * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)
        else:
             if not hasattr(m, 'H2_to_market') or not hasattr(m, 'H2_from_storage') or not hasattr(m, 'mHydrogenProduced'):
                 logger.error("Missing vars for H2 Revenue (with storage)."); return 0.0
             revenue_from_sales_expr = sum(h2_value * (m.H2_to_market[t] + m.H2_from_storage[t]) * time_factor for t in m.TimePeriods)
             revenue_from_subsidy_expr = sum(h2_subsidy * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)
             total_revenue_expr = revenue_from_sales_expr + revenue_from_subsidy_expr
        return total_revenue_expr
    except AttributeError as e: logger.critical(f"CRITICAL Missing var/param for HydrogenRevenue rule: {e}.", exc_info=True); raise e
    except Exception as e: logger.critical(f"CRITICAL Error defining HydrogenRevenue rule: {e}", exc_info=True); raise

# ---------------------------------------------------------------------------
# ANCILLARY SERVICE REVENUE (MODIFIED TO USE HELPERS FROM UTILS for ALL ISOs)
# ---------------------------------------------------------------------------

# --- Helper to get total deployed sum across all components ---
def _get_total_deployed_sum_for_service(m, t, internal_service):
    """Calculates the total symbolic deployed sum for a service across all eligible components."""
    total_deployed = 0.0
    enable_electrolyzer = getattr(m,'ENABLE_ELECTROLYZER',False)
    enable_battery = getattr(m,'ENABLE_BATTERY',False)
    enable_npp = getattr(m,'ENABLE_NUCLEAR_GENERATOR',False)

    if enable_electrolyzer:
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service], 'Electrolyzer')
    if enable_battery:
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service], 'Battery')
    if enable_npp and (enable_electrolyzer or enable_battery): # Turbine AS condition
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service], 'Turbine')
    return total_deployed


# --- Symbolic Hourly Calculation Logic Functions ---

def _symbolic_hourly_spp_revenue_expr(m, t):
    """Returns symbolic expression for SPP AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t] # Symbolic param
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t] # Symbolic Total Bid Var
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
            energy_perf_payment = deployed_amount_expr * lmp
        else: # Bidding mode
            mileage = get_param(m, f'mileage_factor_{service_iso}', t, default=1.0) # Example factor name
            perf = get_param(m, f'performance_factor_{service_iso}', t, default=1.0) # Example factor name
            energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
             deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
             energy_perf_payment = deployed_amount_expr * lmp
        else: # Bidding mode
             mileage = get_param(m, f'mileage_factor_{service_iso}', t, default=1.0)
             perf = get_param(m, f'performance_factor_{service_iso}', t, default=1.0)
             energy_perf_payment = bid * mileage * perf * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else: # Bidding mode
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
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = bid * win_rate * mcp_cap
        # CAISO Mileage payment (needs specific price param: p_RegMileage_CAISO ?)
        mileage_price = get_param(m, 'p_RegMileage', t, default=0.0) # Assumed name
        mileage_factor = get_param(m, f'mileage_factor_{service_iso}', t, default=1.0) # Factor specific to RegU
        mileage_payment = 0.0
        if simulate_dispatch:
            # Mileage payment usually based on awarded capacity * factor * price? Or deployed? Assuming awarded.
            mileage_payment = bid * win_rate * mileage_factor * mileage_price
        else: # Bidding mode
            mileage_payment = bid * mileage_factor * mileage_price # Estimate based on bid
        hourly_revenue_rate_expr += cap_payment + mileage_payment + adder

    # Regulation Down
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = bid * win_rate * mcp_cap
        mileage_price = get_param(m, 'p_RegMileage', t, default=0.0) # Assumed name
        mileage_factor = get_param(m, f'mileage_factor_{service_iso}', t, default=1.0) # Factor specific to RegD
        mileage_payment = 0.0
        if simulate_dispatch:
            mileage_payment = bid * win_rate * mileage_factor * mileage_price
        else: # Bidding mode
            mileage_payment = bid * mileage_factor * mileage_price
        hourly_revenue_rate_expr += cap_payment + mileage_payment + adder

    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'} # CAISO specific mapping
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
             bid = bid_var[t]
             mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
             adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
             win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
             deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
             cap_payment = bid * win_rate * mcp
             energy_payment = 0.0 # CAISO energy payment for reserves is complex, often settled in RT
             if simulate_dispatch:
                 deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                 energy_payment = deployed_amount_expr * lmp # Simplified, likely incorrect for CAISO DA
             else: # Bidding mode
                 energy_payment = bid * deploy_factor * lmp # Simplified estimate
             hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_ercot_revenue_expr(m, t):
    """Returns symbolic expression for ERCOT AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Up (RU)
    service_iso = 'RegU'; internal_service = 'RegUp'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
            energy_perf_payment = deployed_amount_expr * lmp
        else: # Bidding mode
            energy_perf_payment = bid * deploy_factor * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Regulation Down (RD)
    service_iso = 'RegD'; internal_service = 'RegDown'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
            energy_perf_payment = deployed_amount_expr * lmp
        else: # Bidding mode
            energy_perf_payment = bid * deploy_factor * lmp
        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Reserves (RRS - Spin, Non-Spin, ECRS)
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else: # Bidding mode
                energy_payment = bid * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_pjm_revenue_expr(m, t):
    """Returns symbolic expression for PJM AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation (Reg) - PJM uses combined Reg market with performance payments
    service_iso = 'Reg'; internal_service = 'Reg'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        reg_bid = bid_var[t]  # Single regulation bid variable

        # PJM specific parameter names (check model.py loading)
        mcp_cap = get_param(m, 'p_RegCap', t, default=0.0) # Capacity price
        mcp_perf = get_param(m, 'p_RegPerf', t, default=0.0) # Performance price
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0) # Locational adder? Unlikely for PJM Reg.
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        mileage_ratio = get_param(m, 'mileage_ratio', t, default=1.0) # PJM specific factor
        perf_score = get_param(m, 'performance_score', t, default=1.0) # PJM specific factor

        # Capacity Payment: Based on total offered capacity bid * win_rate * MCP_Cap
        cap_payment = reg_bid * win_rate * mcp_cap

        # Performance Payment: Complex in PJM. Simplified here.
        # Often: Cleared_MW * Performance_Score * Mileage_Ratio * MCP_Perf
        # Using reg_bid * win_rate as proxy for Cleared_MW
        perf_payment = 0.0
        if simulate_dispatch:
            # In dispatch mode, performance payment might be linked to actual deployed reg
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
            perf_payment = deployed_amount_expr * perf_score * mileage_ratio * mcp_perf
        else: # Bidding mode
            perf_payment = reg_bid * perf_score * mileage_ratio * mcp_perf # Estimate based on bid

        # Lost Opportunity Cost (LOC) - often part of PJM Reg payment, complex to model here. Assume adder covers it or ignore.

        hourly_revenue_rate_expr += cap_payment + perf_payment + adder

    # Reserves (Synchronized, Non-Synchronized, 30-Min)
    reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'} # PJM specific mapping
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0 # PJM reserves primarily capacity payment in DA
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp # Unlikely for DA revenue
            else: # Bidding mode
                energy_payment = bid * deploy_factor * lmp # Unlikely for DA revenue
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_nyiso_revenue_expr(m, t):
    """Returns symbolic expression for NYISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation Capacity (RegC) - NYISO has capacity and movement payments
    service_iso = 'RegC'; internal_service = 'Reg'
    bid_var = getattr(m, f'Total_{internal_service}', None)
    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        reg_bid = bid_var[t]  # Single regulation bid variable
        
        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0) # Capacity price (p_RegC_NYISO)
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0) # Locational adder?
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = reg_bid * win_rate * mcp_cap

        # Regulation Movement payment (complex, based on actual movement vs awarded)
        # Simplified: Assume some payment based on deployed amount if simulating, else ignore/placeholder
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
            # Movement payment might be proportional to (deployed_up - deployed_down) * movement_price ?
            # Using LMP as a very rough proxy for movement value/cost avoidance
            energy_perf_payment = deployed_amount_expr * lmp # Highly simplified
        # else: # Bidding mode - hard to estimate movement payment

        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Reserves (Spinning, 10-Min Non-Sync, 30-Min)
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': 'ThirtyMin'} # NYISO specific mapping
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0 # NYISO reserves primarily capacity payment in DA
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp # Unlikely for DA revenue
            else: # Bidding mode
                energy_payment = bid * deploy_factor * lmp # Unlikely for DA revenue
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_isone_revenue_expr(m, t):
    """Returns symbolic expression for ISONE AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # ISONE primarily has Reserves (TMSR, TMNSR, TMOR)
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': 'ThirtyMin'} # ISONE specific mapping (using internal names)
                                                                       # Check if param names match e.g. p_Spin10_ISONE
    for service_iso, internal_service in reserve_map.items(): # service_iso is key used for param lookup
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0) # ISONE has Reserve Constraint Penalty Factors (RCPFs) - complex
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0 # ISONE reserves primarily capacity payment in DA
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp # Unlikely for DA revenue
            else: # Bidding mode
                energy_payment = bid * deploy_factor * lmp # Unlikely for DA revenue
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr

def _symbolic_hourly_miso_revenue_expr(m, t):
    """Returns symbolic expression for MISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    # Regulation (Reg) - MISO has RegUp/RegDown distinct prices? Check params. Assume combined for now.
    service_iso = 'Reg'; internal_service_up = 'RegUp'; internal_service_down = 'RegDown'
    bid_up_var = getattr(m, f'Total_{internal_service_up}', None)
    bid_down_var = getattr(m, f'Total_{internal_service_down}', None)
    if (bid_up_var is not None and bid_up_var.is_indexed() and t in bid_up_var.index_set() and
        bid_down_var is not None and bid_down_var.is_indexed() and t in bid_down_var.index_set()):
        bid_up = bid_up_var[t]
        bid_down = bid_down_var[t]
        total_reg_bid = bid_up + bid_down

        mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0) # Assumes single p_Reg_MISO price
        adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
        cap_payment = total_reg_bid * win_rate * mcp_cap

        # MISO Reg mileage/performance payment? Simplified.
        energy_perf_payment = 0.0
        if simulate_dispatch:
            deployed_up_expr = _get_total_deployed_sum_for_service(m, t, internal_service_up)
            deployed_down_expr = _get_total_deployed_sum_for_service(m, t, internal_service_down)
            energy_perf_payment = (deployed_up_expr + deployed_down_expr) * lmp # Highly simplified proxy
        # else: # Bidding mode - hard to estimate

        hourly_revenue_rate_expr += cap_payment + energy_perf_payment + adder

    # Reserves (Spinning, Supplemental, Ramp Capability Up/Down, Short Term)
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'} # MISO specific mapping
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0 # MISO reserves primarily capacity payment in DA
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp # Unlikely for DA revenue
            else: # Bidding mode
                energy_payment = bid * deploy_factor * lmp # Unlikely for DA revenue
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


# --- Total AS Revenue Rules (Assign correct hourly calculator using factory) ---
def AncillaryRevenue_rule_factory(iso_hourly_revenue_func):
    """Factory to create the total AS revenue rule for an ISO."""
    def _rule(m):
        if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False): return 0.0
        try:
            time_factor = pyo.value(m.delT_minutes) / 60.0
            if time_factor <= 0: raise ValueError("delT_minutes must result in a positive time_factor.")
            return sum(iso_hourly_revenue_func(m, t) * time_factor for t in m.TimePeriods)
        except Exception as e:
            logger.critical(f"CRITICAL Error defining AncillaryRevenue rule using {iso_hourly_revenue_func.__name__}: {e}", exc_info=True)
            raise
    return _rule

AncillaryRevenue_SPP_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_spp_revenue_expr)
AncillaryRevenue_CAISO_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_caiso_revenue_expr)
AncillaryRevenue_ERCOT_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_ercot_revenue_expr)
AncillaryRevenue_PJM_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_pjm_revenue_expr)
AncillaryRevenue_NYISO_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_nyiso_revenue_expr)
AncillaryRevenue_ISONE_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_isone_revenue_expr)
AncillaryRevenue_MISO_rule = AncillaryRevenue_rule_factory(_symbolic_hourly_miso_revenue_expr)

# ---------------------------------------------------------------------------
# OPERATIONAL COST COMPONENTS (No changes needed here)
# ---------------------------------------------------------------------------
def OpexCost_rule(m):
    """Calculate total hourly operational costs expression for the objective function."""
    total_opex_expr = 0.0
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)

    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0: raise ValueError("delT_minutes must result in a positive time_factor.")

        cost_vom_turbine_expr = 0.0
        if enable_npp and hasattr(m, 'vom_turbine') and hasattr(m, 'pTurbine'):
            cost_vom_turbine_expr = sum(m.vom_turbine * m.pTurbine[t] * time_factor for t in m.TimePeriods)

        cost_vom_electrolyzer_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'vom_electrolyzer') and hasattr(m, 'pElectrolyzer'):
            cost_vom_electrolyzer_expr = sum(m.vom_electrolyzer * m.pElectrolyzer[t] * time_factor for t in m.TimePeriods) # Based on actual power

        cost_vom_battery_expr = 0.0
        if enable_battery and hasattr(m, 'vom_battery_per_mwh_cycled') and hasattr(m, 'BatteryCharge') and hasattr(m, 'BatteryDischarge'):
             cost_vom_battery_expr = sum(m.vom_battery_per_mwh_cycled * (m.BatteryCharge[t] + m.BatteryDischarge[t]) / 2.0 * time_factor for t in m.TimePeriods)

        cost_water_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'cost_water_per_kg_h2') and hasattr(m, 'mHydrogenProduced'):
            cost_water_expr = sum(m.cost_water_per_kg_h2 * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)

        cost_ramping_expr = 0.0
        # Ramping cost based on change in ACTUAL power (pElectrolyzer)
        if enable_electrolyzer and hasattr(m, 'cost_electrolyzer_ramping') and hasattr(m, 'pElectrolyzerRampPos') and hasattr(m, 'pElectrolyzerRampNeg'):
            cost_ramping_expr = sum(m.cost_electrolyzer_ramping * (m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t]) for t in m.TimePeriods if t > m.TimePeriods.first())

        cost_storage_cycle_expr = 0.0
        if enable_h2_storage and hasattr(m, 'vom_storage_cycle') and hasattr(m, 'H2_to_storage') and hasattr(m, 'H2_from_storage'):
             # Cost is per kg cycled, vars are kg/hr -> total cost = sum(rate * (in+out) * time_factor)
             cost_storage_cycle_expr = sum(m.vom_storage_cycle * (m.H2_to_storage[t] + m.H2_from_storage[t]) * time_factor for t in m.TimePeriods)

        cost_startup_expr = 0.0
        if enable_startup_shutdown and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            # Cost is per startup event
            cost_startup_expr = sum(m.cost_startup_electrolyzer * m.vElectrolyzerStartup[t] for t in m.TimePeriods)

        total_opex_expr = (cost_vom_turbine_expr + cost_vom_electrolyzer_expr + cost_vom_battery_expr +
                           cost_water_expr + cost_ramping_expr + cost_storage_cycle_expr + cost_startup_expr)
        return total_opex_expr

    except AttributeError as e: logger.critical(f"CRITICAL Missing parameter/variable for OpexCost rule: {e}", exc_info=True); raise e
    except Exception as e: logger.critical(f"CRITICAL Error defining OpexCost rule: {e}", exc_info=True); raise

