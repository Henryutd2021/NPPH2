# src/result_processing.py
import pandas as pd
import pyomo.environ as pyo
import os
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from config import (
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_LOW_TEMP_ELECTROLYZER
)
# Assuming constraints.py is in the same directory or accessible via sys.path
try:
    from constraints import get_as_components
except ImportError:
    # Fallback or define a placeholder if constraints.py is not accessible
    # This is needed for calculating deployed reserves in the results
    logging.warning("Could not import get_as_components from constraints.py. Deployed AS calculation might be inaccurate.")
    def get_as_components(m, t): # Placeholder
        return {'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0, 'up_deployed': 0.0, 'down_deployed': 0.0, 'iso_services': {}}


# --- Helper Functions (get_pyomo_value, get_param) ---
def get_pyomo_value(model_component, default=0.0):
    try:
        if hasattr(model_component, 'extract_values'): val = pyo.value(model_component, exception=False)
        elif hasattr(model_component, 'value'): val = pyo.value(model_component, exception=False)
        else: val = pyo.value(model_component, exception=False)
        return default if val is None else val
    except Exception: return default

def get_param(model, param_name_base, time_index, default=0.0):
    param_name = f"{param_name_base}_{model.TARGET_ISO}"
    if hasattr(model, param_name):
        param = getattr(model, param_name)
        if time_index in param: return pyo.value(param[time_index], exception=False) or default
        else: return pyo.value(param, exception=False) or default
    return default

# --- Replicated AS Revenue Calculation ---
def calculate_hourly_as_revenue(m: pyo.ConcreteModel, t: int) -> float:
    """Calculates hourly AS revenue using ISO-specific logic."""
    # This function should exactly mirror the logic in revenue_cost.py
    iso_suffix = m.TARGET_ISO
    total_as_revenue = 0.0
    try:
        lmp = get_pyomo_value(m.energy_price[t])

        if iso_suffix == 'SPP':
            bid_regu = get_pyomo_value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            total_as_revenue += (bid_regu * mcp_regu) + adder_regu
            bid_regd = get_pyomo_value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            total_as_revenue += (bid_regd * mcp_regd) + adder_regd
            bid_spin = get_pyomo_value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            total_as_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_sup = get_pyomo_value(m.Total_NSR[t]); mcp_sup = get_param(m, 'p_Sup', t); deploy_sup = get_param(m, 'deploy_factor_Sup', t, 0.0); adder_sup = get_param(m, 'loc_Sup', t, 0.0)
            total_as_revenue += (bid_sup * mcp_sup) + (bid_sup * deploy_sup * lmp) + adder_sup

        elif iso_suffix == 'CAISO':
            bid_regu = get_pyomo_value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); mcp_rmu = get_param(m, 'p_RMU', t, 0.0); mileage_factor_regu = get_param(m, 'mileage_factor_RegU', t, 1.0); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            total_as_revenue += (bid_regu * mcp_regu) + (bid_regu * mileage_factor_regu * mcp_rmu) + adder_regu
            bid_regd = get_pyomo_value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); mcp_rmd = get_param(m, 'p_RMD', t, 0.0); mileage_factor_regd = get_param(m, 'mileage_factor_RegD', t, 1.0); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            total_as_revenue += (bid_regd * mcp_regd) + (bid_regd * mileage_factor_regd * mcp_rmd) + adder_regd
            bid_spin = get_pyomo_value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            total_as_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_nspin = get_pyomo_value(m.Total_NSR[t]); mcp_nspin = get_param(m, 'p_NSpin', t); deploy_nspin = get_param(m, 'deploy_factor_NSpin', t, 0.0); adder_nspin = get_param(m, 'loc_NSpin', t, 0.0)
            total_as_revenue += (bid_nspin * mcp_nspin) + (bid_nspin * deploy_nspin * lmp) + adder_nspin

        elif iso_suffix == 'ERCOT':
            bid_regu = get_pyomo_value(m.Total_RegUp[t]); mcp_regu = get_param(m, 'p_RegU', t); adder_regu = get_param(m, 'loc_RegU', t, 0.0)
            total_as_revenue += (bid_regu * mcp_regu) + adder_regu
            bid_regd = get_pyomo_value(m.Total_RegDown[t]); mcp_regd = get_param(m, 'p_RegD', t); adder_regd = get_param(m, 'loc_RegD', t, 0.0)
            total_as_revenue += (bid_regd * mcp_regd) + adder_regd
            bid_spin = get_pyomo_value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            total_as_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_nspin = get_pyomo_value(m.Total_NSR[t]); mcp_nspin = get_param(m, 'p_NSpin', t); deploy_nspin = get_param(m, 'deploy_factor_NSpin', t, 0.0); adder_nspin = get_param(m, 'loc_NSpin', t, 0.0)
            total_as_revenue += (bid_nspin * mcp_nspin) + (bid_nspin * deploy_nspin * lmp) + adder_nspin
            bid_ecrs = get_pyomo_value(m.Total_ECRS[t]) if hasattr(m, 'Total_ECRS') and isinstance(m.Total_ECRS, pyo.Var) else 0.0; mcp_ecrs = get_param(m, 'p_ECRS', t); deploy_ecrs = get_param(m, 'deploy_factor_ECRS', t, 0.0); adder_ecrs = get_param(m, 'loc_ECRS', t, 0.0)
            total_as_revenue += (bid_ecrs * mcp_ecrs) + (bid_ecrs * deploy_ecrs * lmp) + adder_ecrs

        elif iso_suffix == 'PJM':
             bid_reg = get_pyomo_value(m.Total_RegUp[t]) + get_pyomo_value(m.Total_RegDown[t]); mcp_reg_cap = get_param(m, 'p_RegCap', t); mcp_reg_perf = get_param(m, 'p_RegPerf', t); perf_score = get_param(m, 'performance_score', t, 1.0); mileage = get_param(m, 'mileage_ratio', t, 1.0); adder_reg = get_param(m, 'loc_Reg', t, 0.0)
             total_as_revenue += (bid_reg * mcp_reg_cap) + (bid_reg * mcp_reg_perf * perf_score * mileage) + adder_reg
             bid_syn = get_pyomo_value(m.Total_SR[t]); mcp_syn = get_param(m, 'p_Syn', t); deploy_syn = get_param(m, 'deploy_factor_Syn', t, 0.0); adder_syn = get_param(m, 'loc_Syn', t, 0.0)
             total_as_revenue += (bid_syn * mcp_syn) + (bid_syn * deploy_syn * lmp) + adder_syn
             bid_rse = get_pyomo_value(m.Total_NSR[t]); mcp_rse = get_param(m, 'p_Rse', t); deploy_rse = get_param(m, 'deploy_factor_Rse', t, 0.0); adder_rse = get_param(m, 'loc_Rse', t, 0.0)
             total_as_revenue += (bid_rse * mcp_rse) + (bid_rse * deploy_rse * lmp) + adder_rse
             bid_tmr = get_pyomo_value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_tmr = get_param(m, 'p_TMR', t); deploy_tmr = get_param(m, 'deploy_factor_TMR', t, 0.0); adder_tmr = get_param(m, 'loc_TMR', t, 0.0)
             total_as_revenue += (bid_tmr * mcp_tmr) + (bid_tmr * deploy_tmr * lmp) + adder_tmr

        elif iso_suffix == 'NYISO':
            bid_regc = get_pyomo_value(m.Total_RegUp[t]) + get_pyomo_value(m.Total_RegDown[t]); mcp_regc = get_param(m, 'p_RegC', t); adder_regc = get_param(m, 'loc_RegC', t, 0.0)
            total_as_revenue += (bid_regc * mcp_regc) + adder_regc
            bid_spin10 = get_pyomo_value(m.Total_SR[t]); mcp_spin10 = get_param(m, 'p_Spin10', t); deploy_spin10 = get_param(m, 'deploy_factor_Spin10', t, 0.0); adder_spin10 = get_param(m, 'loc_Spin10', t, 0.0)
            total_as_revenue += (bid_spin10 * mcp_spin10) + (bid_spin10 * deploy_spin10 * lmp) + adder_spin10
            bid_nspin10 = get_pyomo_value(m.Total_NSR[t]); mcp_nspin10 = get_param(m, 'p_NSpin10', t); deploy_nspin10 = get_param(m, 'deploy_factor_NSpin10', t, 0.0); adder_nspin10 = get_param(m, 'loc_NSpin10', t, 0.0)
            total_as_revenue += (bid_nspin10 * mcp_nspin10) + (bid_nspin10 * deploy_nspin10 * lmp) + adder_nspin10
            bid_res30 = get_pyomo_value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_res30 = get_param(m, 'p_Res30', t); deploy_res30 = get_param(m, 'deploy_factor_Res30', t, 0.0); adder_res30 = get_param(m, 'loc_Res30', t, 0.0)
            total_as_revenue += (bid_res30 * mcp_res30) + (bid_res30 * deploy_res30 * lmp) + adder_res30

        elif iso_suffix == 'ISONE':
            bid_spin10 = get_pyomo_value(m.Total_SR[t]); mcp_spin10 = get_param(m, 'p_Spin10', t); deploy_spin10 = get_param(m, 'deploy_factor_Spin10', t, 0.0); adder_spin10 = get_param(m, 'loc_Spin10', t, 0.0)
            total_as_revenue += (bid_spin10 * mcp_spin10) + (bid_spin10 * deploy_spin10 * lmp) + adder_spin10
            bid_nspin10 = get_pyomo_value(m.Total_NSR[t]); mcp_nspin10 = get_param(m, 'p_NSpin10', t); deploy_nspin10 = get_param(m, 'deploy_factor_NSpin10', t, 0.0); adder_nspin10 = get_param(m, 'loc_NSpin10', t, 0.0)
            total_as_revenue += (bid_nspin10 * mcp_nspin10) + (bid_nspin10 * deploy_nspin10 * lmp) + adder_nspin10
            bid_or30 = get_pyomo_value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_or30 = get_param(m, 'p_OR30', t); deploy_or30 = get_param(m, 'deploy_factor_OR30', t, 0.0); adder_or30 = get_param(m, 'loc_OR30', t, 0.0)
            total_as_revenue += (bid_or30 * mcp_or30) + (bid_or30 * deploy_or30 * lmp) + adder_or30

        # <<< Added MISO Logic >>>
        elif iso_suffix == 'MISO':
            bid_reg = get_pyomo_value(m.Total_RegUp[t]) + get_pyomo_value(m.Total_RegDown[t]); mcp_reg = get_param(m, 'p_Reg', t); adder_reg = get_param(m, 'loc_Reg', t, 0.0)
            total_as_revenue += (bid_reg * mcp_reg) + adder_reg
            bid_spin = get_pyomo_value(m.Total_SR[t]); mcp_spin = get_param(m, 'p_Spin', t); deploy_spin = get_param(m, 'deploy_factor_Spin', t, 0.0); adder_spin = get_param(m, 'loc_Spin', t, 0.0)
            total_as_revenue += (bid_spin * mcp_spin) + (bid_spin * deploy_spin * lmp) + adder_spin
            bid_sup = get_pyomo_value(m.Total_NSR[t]); mcp_sup = get_param(m, 'p_Sup', t); deploy_sup = get_param(m, 'deploy_factor_Sup', t, 0.0); adder_sup = get_param(m, 'loc_Sup', t, 0.0)
            total_as_revenue += (bid_sup * mcp_sup) + (bid_sup * deploy_sup * lmp) + adder_sup
            bid_str = get_pyomo_value(m.Total_30Min[t]) if hasattr(m, 'Total_30Min') and isinstance(m.Total_30Min, pyo.Var) else 0.0; mcp_str = get_param(m, 'p_STR', t); deploy_str = get_param(m, 'deploy_factor_STR', t, 0.0); adder_str = get_param(m, 'loc_STR', t, 0.0)
            total_as_revenue += (bid_str * mcp_str) + (bid_str * deploy_str * lmp) + adder_str

        return total_as_revenue
    except Exception as e:
        logging.error(f"Error during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}")
        return 0.0 # Return 0 if calculation fails for this hour


# --- Main Results Extraction Function ---
def extract_results(model: pyo.ConcreteModel, target_iso: str, output_dir: str = '../Results_Standardized'):
    """
    Extracts comprehensive results, reflecting actual electrolyzer power and H2 production
    based on known deployment factors.
    """
    logging.info(f"Extracting comprehensive results for {target_iso} (Actual Power Mode)...")
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    hours = list(model.TimePeriods)
    results_df = pd.DataFrame(index=pd.Index(hours, name='HourOfYear'))
    summary_results: Dict[str, Any] = {}

    # --- Extract Electrolyzer Capacity ---
    # (Keep existing logic)
    if hasattr(model, 'pElectrolyzer_max') and isinstance(model.pElectrolyzer_max, pyo.Var):
        optimal_capacity = get_pyomo_value(model.pElectrolyzer_max, default=None)
        summary_results['Optimal_Electrolyzer_Capacity_MW'] = optimal_capacity
        results_df['Electrolyzer_Capacity_MW'] = optimal_capacity if optimal_capacity is not None else 0.0
    else:
        optimal_capacity = get_pyomo_value(model.pElectrolyzer_max_upper_bound, default=None)
        summary_results['Fixed_Electrolyzer_Capacity_MW'] = optimal_capacity
        results_df['Electrolyzer_Capacity_MW'] = optimal_capacity if optimal_capacity is not None else 0.0


    # --- Extract Hourly Variables ---
    logging.info("Extracting hourly variables...")
    # (Keep existing extraction, using actual pElectrolyzer/mHydrogenProduced)
    results_df['pTurbine_MW'] = [get_pyomo_value(model.pTurbine[t]) for t in hours]
    results_df['qSteam_Turbine_MWth'] = [get_pyomo_value(model.qSteam_Turbine[t]) for t in hours]
    results_df['pElectrolyzer_MW'] = [get_pyomo_value(model.pElectrolyzer[t]) for t in hours] # Actual power
    results_df['pElectrolyzerSetpoint_MW'] = [get_pyomo_value(model.pElectrolyzerSetpoint[t]) for t in hours]
    results_df['mHydrogenProduced_kg_hr'] = [get_pyomo_value(model.mHydrogenProduced[t]) for t in hours] # Actual H2
    if not ENABLE_LOW_TEMP_ELECTROLYZER and hasattr(model, 'qSteam_Electrolyzer'):
        results_df['qSteam_Electrolyzer_MWth'] = [get_pyomo_value(model.qSteam_Electrolyzer[t]) for t in hours]
    else: results_df['qSteam_Electrolyzer_MWth'] = 0.0
    results_df['pIES_MW'] = [get_pyomo_value(model.pIES[t]) for t in hours]

    # Optional Features
    if ENABLE_H2_STORAGE:
        results_df['H2_Storage_Level_kg'] = [get_pyomo_value(model.H2_storage_level[t]) for t in hours]
        results_df['H2_to_Market_kg_hr'] = [get_pyomo_value(model.H2_to_market[t]) for t in hours]
        results_df['H2_from_Storage_kg_hr'] = [get_pyomo_value(model.H2_from_storage[t]) for t in hours]
        if hasattr(model, 'H2_net_to_storage'):
             results_df['H2_net_to_Storage_kg_hr'] = [get_pyomo_value(model.H2_net_to_storage[t]) for t in hours]
    if ENABLE_STARTUP_SHUTDOWN:
        results_df['Electrolyzer_Status (0=off, 1=on)'] = [get_pyomo_value(model.uElectrolyzer[t], default=-1) for t in hours]
        results_df['Electrolyzer_Startup (0=no, 1=yes)'] = [get_pyomo_value(model.vElectrolyzerStartup[t], default=-1) for t in hours]
        results_df['Electrolyzer_Shutdown (0=no, 1=yes)'] = [get_pyomo_value(model.wElectrolyzerShutdown[t], default=-1) for t in hours]
        summary_results['Total_Electrolyzer_Startups'] = sum(results_df['Electrolyzer_Startup (0=no, 1=yes)'].clip(0))
    if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
        results_df['DegradationState_Units'] = [get_pyomo_value(model.DegradationState[t]) for t in hours]
        summary_results['Final_DegradationState_Units'] = results_df['DegradationState_Units'].iloc[-1]
    if hasattr(model, 'pElectrolyzerRampPos'):
        results_df['pElectrolyzerRampPos_MW'] = [get_pyomo_value(model.pElectrolyzerRampPos[t]) for t in hours]
        results_df['pElectrolyzerRampNeg_MW'] = [get_pyomo_value(model.pElectrolyzerRampNeg[t]) for t in hours]
    if hasattr(model, 'qSteamElectrolyzerRampPos'):
        results_df['qSteamElectrolyzerRampPos_MWth'] = [get_pyomo_value(model.qSteamElectrolyzerRampPos[t]) for t in hours]
        results_df['qSteamElectrolyzerRampNeg_MWth'] = [get_pyomo_value(model.qSteamElectrolyzerRampNeg[t]) for t in hours]


    # Component AS Bids
    logging.info("Extracting component AS bids...")
    # (Keep existing extraction)
    results_df['RegUp_Turbine_MW'] = [get_pyomo_value(model.RegUp_Turbine[t]) for t in hours]
    results_df['RegDown_Turbine_MW'] = [get_pyomo_value(model.RegDown_Turbine[t]) for t in hours]
    results_df['SR_Turbine_MW'] = [get_pyomo_value(model.SR_Turbine[t]) for t in hours]
    results_df['NSR_Turbine_MW'] = [get_pyomo_value(model.NSR_Turbine[t]) for t in hours]
    results_df['RegUp_Electrolyzer_MW'] = [get_pyomo_value(model.RegUp_Electrolyzer[t]) for t in hours]
    results_df['RegDown_Electrolyzer_MW'] = [get_pyomo_value(model.RegDown_Electrolyzer[t]) for t in hours]
    results_df['SR_Electrolyzer_MW'] = [get_pyomo_value(model.SR_Electrolyzer[t]) for t in hours]
    results_df['NSR_Electrolyzer_MW'] = [get_pyomo_value(model.NSR_Electrolyzer[t]) for t in hours]
    if hasattr(model, 'ECRS_Turbine'):
        results_df['ECRS_Turbine_MW'] = [get_pyomo_value(model.ECRS_Turbine[t]) for t in hours]
        results_df['ECRS_Electrolyzer_MW'] = [get_pyomo_value(model.ECRS_Electrolyzer[t]) for t in hours]
    if hasattr(model, 'ThirtyMin_Turbine'):
        results_df['ThirtyMin_Turbine_MW'] = [get_pyomo_value(model.ThirtyMin_Turbine[t]) for t in hours]
        results_df['ThirtyMin_Electrolyzer_MW'] = [get_pyomo_value(model.ThirtyMin_Electrolyzer[t]) for t in hours]


    # Total AS Bids
    logging.info("Extracting total AS bids...")
    # (Keep existing extraction)
    results_df['Total_RegUp_MW'] = [get_pyomo_value(model.Total_RegUp[t]) for t in hours]
    results_df['Total_RegDown_MW'] = [get_pyomo_value(model.Total_RegDown[t]) for t in hours]
    results_df['Total_SR_MW'] = [get_pyomo_value(model.Total_SR[t]) for t in hours]
    results_df['Total_NSR_MW'] = [get_pyomo_value(model.Total_NSR[t]) for t in hours]
    if hasattr(model, 'Total_ECRS') and isinstance(model.Total_ECRS, pyo.Var):
        results_df['Total_ECRS_MW'] = [get_pyomo_value(model.Total_ECRS[t]) for t in hours]
    if hasattr(model, 'Total_30Min') and isinstance(model.Total_30Min, pyo.Var):
        results_df['Total_30Min_MW'] = [get_pyomo_value(model.Total_30Min[t]) for t in hours]


    # --- Extract Input Prices/Factors (Including all deploy factors) ---
    logging.info("Extracting input prices and factors...")
    results_df['EnergyPrice_LMP_USDperMWh'] = [get_pyomo_value(model.energy_price[t]) for t in hours]
    iso_suffix = target_iso
    # Define the services based on the ISO map used in generation/constraints
    iso_service_map = {
        'SPP': ['RegU', 'RegD', 'Spin', 'Sup'],
        'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
        'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
        'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'], # Add specific PJM factors
        'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
        'ISONE': ['Spin10', 'NSpin10', 'OR30'],
        'MISO': ['Reg', 'Spin', 'Sup', 'STR']
    }
    if iso_suffix in iso_service_map:
        for service in iso_service_map[iso_suffix]:
            # Extract Price, Deploy Factor, Adder for each service
            results_df[f'p_{service}_{iso_suffix}'] = [get_param(model, f'p_{service}', t) for t in hours]
            results_df[f'deploy_factor_{service}_{iso_suffix}'] = [get_param(model, f'deploy_factor_{service}', t) for t in hours]
            results_df[f'loc_{service}_{iso_suffix}'] = [get_param(model, f'loc_{service}', t) for t in hours]
            # Extract specific factors if they exist
            if iso_suffix == 'CAISO' and service in ['RegU', 'RegD']:
                 results_df[f'mileage_factor_{service}_{iso_suffix}'] = [get_param(model, f'mileage_factor_{service}', t, 1.0) for t in hours]
            # Note: PJM factors extracted via service='performance_score' etc. if named that way
            if iso_suffix == 'PJM' and service == 'performance_score':
                 results_df[f'performance_score_{iso_suffix}'] = [get_param(model, 'performance_score', t, 1.0) for t in hours]
            if iso_suffix == 'PJM' and service == 'mileage_ratio':
                 results_df[f'mileage_ratio_{iso_suffix}'] = [get_param(model, 'mileage_ratio', t, 1.0) for t in hours]


    # --- Calculate Deployed AS Amounts (for verification/analysis) ---
    logging.info("Calculating deployed AS amounts...")
    deployed_up = []
    deployed_down = []
    for t in hours:
        as_info = get_as_components(model, t) # Get deployed amounts based on bids and factors
        deployed_up.append(as_info['up_deployed'])
        deployed_down.append(as_info['down_deployed'])
    results_df['Deployed_Up_Reserves_MW'] = deployed_up
    results_df['Deployed_Down_Reserves_MW'] = deployed_down


    # --- Calculate Hourly Revenues (Uses ISO-specific rules) ---
    logging.info("Calculating hourly revenues...")
    results_df['Revenue_Energy_USD'] = results_df['pIES_MW'] * results_df['EnergyPrice_LMP_USDperMWh']
    h2_rev = []
    h2_value = get_pyomo_value(model.H2_value)
    if ENABLE_H2_STORAGE:
        for t_idx, t in enumerate(hours):
             # Revenue based on H2 leaving the system boundary (market + from_storage)
             h2_rev.append(h2_value * (results_df['H2_to_Market_kg_hr'].iloc[t_idx] + results_df['H2_from_Storage_kg_hr'].iloc[t_idx]))
    else:
        h2_rev = h2_value * results_df['mHydrogenProduced_kg_hr'] # Use actual H2 production if no storage
    results_df['Revenue_Hydrogen_USD'] = h2_rev
    results_df['Revenue_Ancillary_USD'] = [calculate_hourly_as_revenue(model, t) for t in hours] # Uses ISO-specific logic
    results_df['Revenue_Total_USD'] = results_df['Revenue_Energy_USD'] + results_df['Revenue_Hydrogen_USD'] + results_df['Revenue_Ancillary_USD']


    # --- Calculate Hourly Costs (Based on actual pElectrolyzer/mHydrogenProduced) ---
    logging.info("Calculating hourly costs...")
    results_df['Cost_VOM_Turbine_USD'] = get_pyomo_value(model.vom_turbine) * results_df['pTurbine_MW']
    results_df['Cost_VOM_Electrolyzer_USD'] = get_pyomo_value(model.vom_electrolyzer) * results_df['pElectrolyzer_MW'] # Use actual power
    results_df['Cost_Water_USD'] = get_pyomo_value(model.cost_water_per_kg_h2) * results_df['mHydrogenProduced_kg_hr'] # Use actual H2
    cost_ramp = 0.0
    if hasattr(model, 'cost_electrolyzer_ramping') and hasattr(model, 'pElectrolyzerRampPos'):
        cost_ramp_param = get_pyomo_value(model.cost_electrolyzer_ramping)
        results_df['Cost_Ramping_USD'] = cost_ramp_param * (results_df['pElectrolyzerRampPos_MW'] + results_df['pElectrolyzerRampNeg_MW'])
        results_df.loc[results_df.index.min(), 'Cost_Ramping_USD'] = 0.0
    else: results_df['Cost_Ramping_USD'] = 0.0
    cost_storage = 0.0
    if ENABLE_H2_STORAGE and hasattr(model, 'vom_storage_cycle') and hasattr(model, 'H2_net_to_Storage_kg_hr'):
        cost_storage_param = get_pyomo_value(model.vom_storage_cycle)
        results_df['Cost_Storage_Cycle_USD'] = cost_storage_param * (results_df['H2_net_to_Storage_kg_hr'] + results_df['H2_from_Storage_kg_hr'])
    else: results_df['Cost_Storage_Cycle_USD'] = 0.0
    cost_startup = 0.0
    if ENABLE_STARTUP_SHUTDOWN and hasattr(model, 'cost_startup_electrolyzer'):
        cost_startup_param = get_pyomo_value(model.cost_startup_electrolyzer)
        results_df['Cost_Startup_USD'] = cost_startup_param * results_df['Electrolyzer_Startup (0=no, 1=yes)']
    else: results_df['Cost_Startup_USD'] = 0.0

    results_df['Cost_HourlyOpex_Total_USD'] = (results_df['Cost_VOM_Turbine_USD'] +
                                               results_df['Cost_VOM_Electrolyzer_USD'] +
                                               results_df['Cost_Water_USD'] +
                                               results_df['Cost_Ramping_USD'] +
                                               results_df['Cost_Storage_Cycle_USD'] +
                                               results_df['Cost_Startup_USD'])


    # --- Calculate Hourly Profit ---
    logging.info("Calculating hourly profit...")
    results_df['Profit_Hourly_USD'] = results_df['Revenue_Total_USD'] - results_df['Cost_HourlyOpex_Total_USD']

    # --- Calculate Summary Statistics ---
    logging.info("Calculating summary statistics...")
    # (Calculations remain similar, but use actual H2 production)
    total_revenue = results_df['Revenue_Total_USD'].sum()
    total_hourly_opex = results_df['Cost_HourlyOpex_Total_USD'].sum()
    electrolyzer_capex = 0.0
    cost_elec_cap_param = get_pyomo_value(getattr(model, 'cost_electrolyzer_capacity', 0)) # Get param value safely
    if 'Optimal_Electrolyzer_Capacity_MW' in summary_results:
        electrolyzer_capex = cost_elec_cap_param * summary_results['Optimal_Electrolyzer_Capacity_MW']
    elif 'Fixed_Electrolyzer_Capacity_MW' in summary_results:
         electrolyzer_capex = cost_elec_cap_param * summary_results['Fixed_Electrolyzer_Capacity_MW']

    summary_results['Total_Revenue_USD'] = total_revenue
    summary_results['Total_Hourly_Opex_USD'] = total_hourly_opex
    summary_results['Electrolyzer_Capex_USD'] = electrolyzer_capex
    summary_results['Total_Profit_Calc_USD'] = total_revenue - total_hourly_opex - electrolyzer_capex
    summary_results['Objective_Value_USD'] = get_pyomo_value(model.TotalProfit_Objective, default=None)

    summary_results['Total_H2_Produced_kg'] = results_df['mHydrogenProduced_kg_hr'].sum() # Now reflects actual

    if ENABLE_H2_STORAGE:
        summary_results['Total_H2_to_Market_kg'] = results_df['H2_to_Market_kg_hr'].sum()
        summary_results['Total_H2_from_Storage_kg'] = results_df['H2_from_Storage_kg_hr'].sum()
        summary_results['Final_H2_Storage_Level_kg'] = results_df['H2_Storage_Level_kg'].iloc[-1]

    # Capacity Factor based on actual power
    elec_capacity_val = summary_results.get('Optimal_Electrolyzer_Capacity_MW') or summary_results.get('Fixed_Electrolyzer_Capacity_MW', 0)
    if elec_capacity_val > 1e-6:
        avg_elec_power_actual = results_df['pElectrolyzer_MW'].mean()
        capacity_factor_actual = avg_elec_power_actual / elec_capacity_val
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = capacity_factor_actual


    # --- Final Formatting and Saving ---
    # (Keep existing saving logic)
    results_df = results_df.round(4)
    output_csv_path = results_dir / f'{target_iso}_Hourly_Results_Comprehensive.csv'
    results_df.to_csv(output_csv_path)
    logging.info(f"Comprehensive hourly results saved to {output_csv_path}")
    print(f"Comprehensive hourly results saved to {output_csv_path}")
    output_summary_path = results_dir / f'{target_iso}_Summary_Results.txt'
    with open(output_summary_path, 'w') as f:
        f.write(f"--- Summary Results for {target_iso} ---\n")
        for key, value in summary_results.items():
            if isinstance(value, (float, np.float64)): f.write(f"{key}: {value:,.2f}\n")
            else: f.write(f"{key}: {value}\n")
        if summary_results.get('Objective_Value_USD') is not None:
             obj_val = summary_results['Objective_Value_USD']
             calc_prof = summary_results['Total_Profit_Calc_USD']
             # Check if values are numeric before subtracting
             if isinstance(obj_val, (int, float, np.number)) and isinstance(calc_prof, (int, float, np.number)):
                 diff = calc_prof - obj_val
                 f.write(f"\nObjective vs Calculated Profit Diff: {diff:,.2f}\n")
                 if abs(diff) > 1e-2: f.write("WARNING: Significant difference between objective value and calculated profit!\n")
             else:
                 f.write("\nCould not compare Objective vs Calculated Profit (non-numeric values).\n")

    logging.info(f"Summary results saved to {output_summary_path}")
    print(f"Summary results saved to {output_summary_path}")
    print("--- Summary ---")
    for key, value in summary_results.items():
         if isinstance(value, (float, np.float64)): print(f"{key}: {value:,.2f}")
         else: print(f"{key}: {value}")
    print("---------------")

    return results_df, summary_results
