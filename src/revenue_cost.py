# src/revenue_cost.py

"""Revenue and cost expression rules isolated here so they can be unit‑tested
independently from the rest of the optimisation model.

Implements ISO-specific ancillary service revenue calculations based on:
- Reserves: (Bid * MCP) + (Bid * deploy_factor * LMP) + Adder
- Regulation: (Bid * MCP) + Performance/Mileage Payments + Adder
"""
import pyomo.environ as pyo
from logging_setup import logger
from config import ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN

# ---------------------------------------------------------------------------
# REVENUE COMPONENTS
# ---------------------------------------------------------------------------

# --- Energy and Hydrogen Revenue (Unchanged) ---
def EnergyRevenue_rule(m):
    try: return sum(m.energy_price[t] * m.pIES[t] for t in m.TimePeriods)
    except Exception as e: logger.error(f"Error in EnergyRevenue rule: {e}"); raise

def HydrogenRevenue_rule(m):
    if not ENABLE_H2_STORAGE:
        try: return sum(m.H2_value * m.mHydrogenProduced[t] for t in m.TimePeriods)
        except Exception as e: logger.error(f"Error in HydrogenRevenue (no storage) rule: {e}"); raise
    else:
        try:
            return sum(m.H2_value * (m.H2_to_market[t] + m.H2_from_storage[t]) for t in m.TimePeriods)
        except Exception as e: logger.error(f"Error in HydrogenRevenue (with storage) rule: {e}"); raise

# --- Helper Function for Safe Parameter Access ---
def get_param(model, param_name_base, time_index, default=0.0):
    """Safely gets a parameter value, returning default if it doesn't exist."""
    param_name = f"{param_name_base}_{model.TARGET_ISO}"
    if hasattr(model, param_name):
        param = getattr(model, param_name)
        if time_index in param:
             val = pyo.value(param[time_index], exception=False)
             return val if val is not None else default
        else:
             val = pyo.value(param, exception=False)
             return val if val is not None else default
    return default

# --- ISO-Specific Ancillary Revenue Rules ---
# (Keep all AncillaryRevenue_ISO_rule functions as they were in revenue_cost_iso_specific_v4)
def AncillaryRevenue_SPP_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_regu = pyo.value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            hourly_revenue += (bid_regu * mcp_regu) + adder_regu
            bid_regd = pyo.value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            hourly_revenue += (bid_regd * mcp_regd) + adder_regd
            bid_spin = pyo.value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            hourly_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_sup = pyo.value(m.Total_NSR[t]); mcp_sup = get_param(m, 'p_Sup', t); deploy_sup = get_param(m, 'deploy_factor_Sup', t, 0.0); adder_sup = get_param(m, 'loc_Sup', t, 0.0)
            hourly_revenue += (bid_sup * mcp_sup) + (bid_sup * deploy_sup * lmp) + adder_sup
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for SPP AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_SPP rule: {e}"); raise

def AncillaryRevenue_CAISO_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_regu = pyo.value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); mcp_rmu = get_param(m, 'p_RMU', t, 0.0); mileage_factor_regu = get_param(m, 'mileage_factor_RegU', t, 1.0); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            hourly_revenue += (bid_regu * mcp_regu) + (bid_regu * mileage_factor_regu * mcp_rmu) + adder_regu
            bid_regd = pyo.value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); mcp_rmd = get_param(m, 'p_RMD', t, 0.0); mileage_factor_regd = get_param(m, 'mileage_factor_RegD', t, 1.0); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            hourly_revenue += (bid_regd * mcp_regd) + (bid_regd * mileage_factor_regd * mcp_rmd) + adder_regd
            bid_spin = pyo.value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            hourly_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_nspin = pyo.value(m.Total_NSR[t]); mcp_nspin = get_param(m, 'p_NSpin', t); deploy_nspin = get_param(m, 'deploy_factor_NSpin', t, 0.0); adder_nspin = get_param(m, 'loc_NSpin', t, 0.0)
            hourly_revenue += (bid_nspin * mcp_nspin) + (bid_nspin * deploy_nspin * lmp) + adder_nspin
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for CAISO AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_CAISO rule: {e}"); raise

def AncillaryRevenue_ERCOT_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_regu = pyo.value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            hourly_revenue += (bid_regu * mcp_regu) + adder_regu
            bid_regd = pyo.value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            hourly_revenue += (bid_regd * mcp_regd) + adder_regd
            bid_spin = pyo.value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            hourly_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_nspin = pyo.value(m.Total_NSR[t]); mcp_nspin = get_param(m, 'p_NSpin', t); deploy_nspin = get_param(m, 'deploy_factor_NSpin', t, 0.0); adder_nspin = get_param(m, 'loc_NSpin', t, 0.0)
            hourly_revenue += (bid_nspin * mcp_nspin) + (bid_nspin * deploy_nspin * lmp) + adder_nspin
            bid_ecrs = pyo.value(m.Total_ECRS[t]) if hasattr(m, 'Total_ECRS') and isinstance(m.Total_ECRS, pyo.Var) else 0.0; mcp_ecrs = get_param(m, 'p_ECRS', t); deploy_ecrs = get_param(m, 'deploy_factor_ECRS', t, 0.0); adder_ecrs = get_param(m, 'loc_ECRS', t, 0.0)
            hourly_revenue += (bid_ecrs * mcp_ecrs) + (bid_ecrs * deploy_ecrs * lmp) + adder_ecrs
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for ERCOT AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_ERCOT rule: {e}"); raise

def AncillaryRevenue_PJM_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_reg = pyo.value(m.Total_RegUp[t]) + pyo.value(m.Total_RegDown[t]); mcp_reg_cap = get_param(m, 'p_RegCap', t); mcp_reg_perf = get_param(m, 'p_RegPerf', t); perf_score = get_param(m, 'performance_score', t, 1.0); mileage = get_param(m, 'mileage_ratio', t, 1.0); adder_reg = get_param(m, 'loc_Reg', t, 0.0)
            hourly_revenue += (bid_reg * mcp_reg_cap) + (bid_reg * mcp_reg_perf * perf_score * mileage) + adder_reg
            bid_syn = pyo.value(m.Total_SR[t]); mcp_syn = get_param(m, 'p_Syn', t); deploy_syn = get_param(m, 'deploy_factor_Syn', t, 0.0); adder_syn = get_param(m, 'loc_Syn', t, 0.0)
            hourly_revenue += (bid_syn * mcp_syn) + (bid_syn * deploy_syn * lmp) + adder_syn
            bid_rse = pyo.value(m.Total_NSR[t]); mcp_rse = get_param(m, 'p_Rse', t); deploy_rse = get_param(m, 'deploy_factor_Rse', t, 0.0); adder_rse = get_param(m, 'loc_Rse', t, 0.0)
            hourly_revenue += (bid_rse * mcp_rse) + (bid_rse * deploy_rse * lmp) + adder_rse
            bid_tmr = pyo.value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_tmr = get_param(m, 'p_TMR', t); deploy_tmr = get_param(m, 'deploy_factor_TMR', t, 0.0); adder_tmr = get_param(m, 'loc_TMR', t, 0.0)
            hourly_revenue += (bid_tmr * mcp_tmr) + (bid_tmr * deploy_tmr * lmp) + adder_tmr
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for PJM AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_PJM rule: {e}"); raise

def AncillaryRevenue_NYISO_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_regc = pyo.value(m.Total_RegUp[t]) + pyo.value(m.Total_RegDown[t]); mcp_regc = get_param(m, 'p_RegC', t); adder_regc = get_param(m, 'loc_RegC', t, 0.0)
            hourly_revenue += (bid_regc * mcp_regc) + adder_regc
            bid_spin10 = pyo.value(m.Total_SR[t]); mcp_spin10 = get_param(m, 'p_Spin10', t); deploy_spin10 = get_param(m, 'deploy_factor_Spin10', t, 0.0); adder_spin10 = get_param(m, 'loc_Spin10', t, 0.0)
            hourly_revenue += (bid_spin10 * mcp_spin10) + (bid_spin10 * deploy_spin10 * lmp) + adder_spin10
            bid_nspin10 = pyo.value(m.Total_NSR[t]); mcp_nspin10 = get_param(m, 'p_NSpin10', t); deploy_nspin10 = get_param(m, 'deploy_factor_NSpin10', t, 0.0); adder_nspin10 = get_param(m, 'loc_NSpin10', t, 0.0)
            hourly_revenue += (bid_nspin10 * mcp_nspin10) + (bid_nspin10 * deploy_nspin10 * lmp) + adder_nspin10
            bid_res30 = pyo.value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_res30 = get_param(m, 'p_Res30', t); deploy_res30 = get_param(m, 'deploy_factor_Res30', t, 0.0); adder_res30 = get_param(m, 'loc_Res30', t, 0.0)
            hourly_revenue += (bid_res30 * mcp_res30) + (bid_res30 * deploy_res30 * lmp) + adder_res30
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for NYISO AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_NYISO rule: {e}"); raise

def AncillaryRevenue_ISONE_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_spin10 = pyo.value(m.Total_SR[t]); mcp_spin10 = get_param(m, 'p_Spin10', t); deploy_spin10 = get_param(m, 'deploy_factor_Spin10', t, 0.0); adder_spin10 = get_param(m, 'loc_Spin10', t, 0.0)
            hourly_revenue += (bid_spin10 * mcp_spin10) + (bid_spin10 * deploy_spin10 * lmp) + adder_spin10
            bid_nspin10 = pyo.value(m.Total_NSR[t]); mcp_nspin10 = get_param(m, 'p_NSpin10', t); deploy_nspin10 = get_param(m, 'deploy_factor_NSpin10', t, 0.0); adder_nspin10 = get_param(m, 'loc_NSpin10', t, 0.0)
            hourly_revenue += (bid_nspin10 * mcp_nspin10) + (bid_nspin10 * deploy_nspin10 * lmp) + adder_nspin10
            bid_or30 = pyo.value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_or30 = get_param(m, 'p_OR30', t); deploy_or30 = get_param(m, 'deploy_factor_OR30', t, 0.0); adder_or30 = get_param(m, 'loc_OR30', t, 0.0)
            hourly_revenue += (bid_or30 * mcp_or30) + (bid_or30 * deploy_or30 * lmp) + adder_or30
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for ISONE AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_ISONE rule: {e}"); raise

def AncillaryRevenue_MISO_rule(m):
    total_as_revenue = 0
    try:
        for t in m.TimePeriods:
            hourly_revenue = 0; lmp = pyo.value(m.energy_price[t])
            bid_reg = pyo.value(m.Total_RegUp[t]) + pyo.value(m.Total_RegDown[t]); mcp_reg = get_param(m, 'p_Reg', t); adder_reg = get_param(m, 'loc_Reg', t, 0.0)
            hourly_revenue += (bid_reg * mcp_reg) + adder_reg
            bid_spin = pyo.value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            hourly_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_sup = pyo.value(m.Total_NSR[t]); mcp_sup = get_param(m, 'p_Sup', t); deploy_sup = get_param(m, 'deploy_factor_Sup', t, 0.0); adder_sup = get_param(m, 'loc_Sup', t, 0.0)
            hourly_revenue += (bid_sup * mcp_sup) + (bid_sup * deploy_sup * lmp) + adder_sup
            bid_str = pyo.value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_str = get_param(m, 'p_STR', t); deploy_str = get_param(m, 'deploy_factor_STR', t, 0.0); adder_str = get_param(m, 'loc_STR', t, 0.0)
            hourly_revenue += (bid_str * mcp_str) + (bid_str * deploy_str * lmp) + adder_str
            total_as_revenue += hourly_revenue
        return total_as_revenue
    except AttributeError as e: logger.error(f"Missing parameter/variable for MISO AS revenue: {e}"); raise
    except Exception as e: logger.error(f"Error in AncillaryRevenue_MISO rule: {e}"); raise


# --- Cost Components (OpexCost includes CAPEX now) ---
def OpexCost_rule(m):
    """Calculates operational costs, including startup and electrolyzer capacity cost."""
    try:
        # VOM Costs - Use symbolic variables directly
        turbine_vom_cost = sum(m.vom_turbine * m.pTurbine[t] for t in m.TimePeriods)
        electrolyzer_vom_cost = sum(m.vom_electrolyzer * m.pElectrolyzer[t] for t in m.TimePeriods) # Based on actual power
        water_cost = sum(m.cost_water_per_kg_h2 * m.mHydrogenProduced[t] for t in m.TimePeriods) # Based on actual H2

        # Ramping cost - Use symbolic helper variables
        ramping_cost = 0
        if hasattr(m, 'cost_electrolyzer_ramping') and m.cost_electrolyzer_ramping > 1e-9 and hasattr(m, 'pElectrolyzerRampPos'):
             ramping_cost = sum(m.cost_electrolyzer_ramping * (m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t])
                               for t in m.TimePeriods if t > m.TimePeriods.first())

        # Storage O&M cost - Use symbolic helper variables
        storage_cycle_cost = 0
        if ENABLE_H2_STORAGE and hasattr(m, 'vom_storage_cycle') and m.vom_storage_cycle > 1e-9 and hasattr(m, 'H2_net_to_storage'):
            storage_cycle_cost = sum(m.vom_storage_cycle * (m.H2_net_to_storage[t] + m.H2_from_storage[t])
                                      for t in m.TimePeriods)

        # Startup Cost - Use symbolic helper variable
        startup_cost = 0
        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            startup_cost = sum(m.cost_startup_electrolyzer * m.vElectrolyzerStartup[t] for t in m.TimePeriods)

        # Degradation Cost (Optional Penalty)
        degradation_penalty_cost = 0
        # ... (optional degradation penalty logic) ...

        # Electrolyzer Capital Cost (associated with the decision variable)
        electrolyzer_capex_cost = 0
        if hasattr(m, 'cost_electrolyzer_capacity') and hasattr(m, 'pElectrolyzer_max') and isinstance(m.pElectrolyzer_max, pyo.Var):
             # Keep symbolic multiplication
             electrolyzer_capex_cost = m.cost_electrolyzer_capacity * m.pElectrolyzer_max

        # Total Cost - Return symbolic expression
        total_hourly_cost = turbine_vom_cost + electrolyzer_vom_cost + water_cost + ramping_cost + storage_cycle_cost + startup_cost + degradation_penalty_cost
        total_cost = total_hourly_cost + electrolyzer_capex_cost
        return total_cost
    # Use more specific exception handling if possible
    except AttributeError as e: logger.error(f"AttributeError in OpexCost rule: {e}"); raise
    except TypeError as e: logger.error(f"TypeError in OpexCost rule: {e}"); raise
    except Exception as e: logger.error(f"Error in OpexCost rule: {e}"); raise

