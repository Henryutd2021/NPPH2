# src/result_processing.py
import pandas as pd
import pyomo.environ as pyo
import os
from logging_setup import logger
from pathlib import Path
from typing import Dict, Any
import numpy as np

# MODIFICATION: Changed relative import to direct import
from utils import get_param, get_var_value, get_total_deployed_as

# Import necessary config flags to conditionally process results
from config import (
    TARGET_ISO,
    ENABLE_NUCLEAR_GENERATOR, ENABLE_ELECTROLYZER, ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY,
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN, ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    CAN_PROVIDE_ANCILLARY_SERVICES
)
# Import constraint helper if available, define placeholder otherwise
try:
    from constraints import get_as_components as get_as_components_helper
except ImportError:
    logger.warning(
        "Could not import get_as_components from constraints.py. Detailed AS component breakdown might be limited.")

    def get_as_components_helper(m, t):  # Placeholder using imported utils
        if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
            return {'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0}
        up_bids = 0.0
        down_bids = 0.0
        up_services = ['RegUp', 'SR', 'NSR',
                       'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        down_services = ['RegDown', 'RampDown']
        for service in up_services:
            total_var = getattr(m, f'Total_{service}', None)
            if total_var is not None:
                up_bids += get_var_value(total_var, t, default=0.0)  # Use util
        for service in down_services:
            total_var = getattr(m, f'Total_{service}', None)
            if total_var is not None:
                down_bids += get_var_value(total_var,
                                           t, default=0.0)  # Use util
        # Return detailed breakdown if needed by downstream code, otherwise simple up/down is fine for placeholder
        as_info = {'up_reserves_bid': up_bids, 'down_reserves_bid': down_bids}
        # Add component specific keys with 0 as this placeholder doesn't calculate them
        for comp in ['turbine', 'h2', 'battery']:
            as_info[f'up_reserves_bid_{comp}'] = 0.0
            as_info[f'down_reserves_bid_{comp}'] = 0.0
        return as_info


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS -- REMOVED (Now imported from utils)
# ---------------------------------------------------------------------------


# --- Hourly AS Revenue Calculation (Uses imported utils) ---
def calculate_hourly_as_revenue(m: pyo.ConcreteModel, t: int) -> float:
    """
    Calculates hourly AS revenue rate ($/hr) using ISO-specific logic.
    Checks m.SIMULATE_AS_DISPATCH_EXECUTION to determine calculation method
    for energy/performance payments. Uses imported utils.
    """
    if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
        return 0.0

    iso_suffix = getattr(m, 'TARGET_ISO', 'UNKNOWN')
    total_hourly_as_revenue_rate = 0.0
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)

    try:
        lmp = get_param(m, 'energy_price', t, default=0.0)  # Use util

        # --- ISO-Specific Logic ---
        # (This logic remains the same, but relies on the imported get_param, get_var_value, get_total_deployed_as)
        # Example for SPP (ensure all ISOs use imported helpers)
        if iso_suffix == 'SPP':
            # Regulation Up
            service_iso = 'RegU'
            internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)  # Use util
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)  # Use util
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(
                    m, t, internal_service)  # Use util
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Regulation Down
            service_iso = 'RegD'
            internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Reserves
            reserve_map = {'Spin': 'SR', 'Sup': 'NSR',
                           'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t, default=0.0)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
                adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
                win_rate = get_param(
                    m, f'winning_rate_{service_iso}', t, default=1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t, default=0.0)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'CAISO':
            # Regulation Up
            service_iso = 'RegU'
            internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            # Regulation Down
            service_iso = 'RegD'
            internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)
            total_hourly_as_revenue_rate += (bid * win_rate * mcp_cap) + adder
            # Reserves
            reserve_map = {'Spin': 'SR', 'NSpin': 'NSR',
                           'RMU': 'RampUp', 'RMD': 'RampDown'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t, default=0.0)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
                adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
                win_rate = get_param(
                    m, f'winning_rate_{service_iso}', t, default=1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t, default=0.0)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'ERCOT':
            # Regulation Up
            service_iso = 'RegU'
            internal_service = 'RegUp'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Regulation Down
            service_iso = 'RegD'
            internal_service = 'RegDown'
            bid_var = getattr(m, f'Total_{internal_service}', None)
            bid = get_var_value(bid_var, t, default=0.0)
            mcp_cap = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(
                m, f'winning_rate_{service_iso}', t, default=1.0)
            cap_payment = bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_amount = get_total_deployed_as(m, t, internal_service)
                energy_perf_payment = deployed_amount * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            # Reserves
            reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t, default=0.0)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
                adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
                win_rate = get_param(
                    m, f'winning_rate_{service_iso}', t, default=1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t, default=0.0)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'PJM':
            service = 'Reg'
            bid_up_var = getattr(m, 'Total_RegUp', None)
            bid_up = get_var_value(bid_up_var, t)
            bid_down_var = getattr(m, 'Total_RegDown', None)
            bid_down = get_var_value(bid_down_var, t)
            total_reg_bid = bid_up + bid_down
            mcp_cap = get_param(m, 'p_RegCap', t)
            adder = get_param(m, f'loc_{service}', t)
            win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            cap_payment = total_reg_bid * win_rate * mcp_cap
            perf_payment = 0.0
            if simulate_dispatch:
                deployed_up = get_total_deployed_as(m, t, 'RegUp')
                deployed_down = get_total_deployed_as(m, t, 'RegDown')
                perf_payment = (deployed_up - deployed_down) * \
                    lmp  # Simplified
            else:
                mileage = get_param(m, 'mileage_ratio', t, 1.0)
                perf = get_param(m, 'performance_score', t, 1.0)
                perf_payment = total_reg_bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + perf_payment + adder
            reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t)
                adder = get_param(m, f'loc_{service_iso}', t)
                win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'NYISO':
            service = 'RegC'
            bid_up_var = getattr(m, 'Total_RegUp', None)
            bid_up = get_var_value(bid_up_var, t)
            bid_down_var = getattr(m, 'Total_RegDown', None)
            bid_down = get_var_value(bid_down_var, t)
            total_reg_bid = bid_up + bid_down
            mcp_cap = get_param(m, f'p_{service}', t)
            adder = get_param(m, f'loc_{service}', t)
            win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            cap_payment = total_reg_bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_up = get_total_deployed_as(m, t, 'RegUp')
                deployed_down = get_total_deployed_as(m, t, 'RegDown')
                energy_perf_payment = (deployed_up - deployed_down) * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = total_reg_bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            reserve_map = {'Spin10': 'SR',
                           'NSpin10': 'NSR', 'Res30': 'ThirtyMin'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t)
                adder = get_param(m, f'loc_{service_iso}', t)
                win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'ISONE':
            reserve_map = {'Spin10': 'SR',
                           'NSpin10': 'NSR', 'OR30': 'ThirtyMin'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t)
                adder = get_param(m, f'loc_{service_iso}', t)
                win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        elif iso_suffix == 'MISO':
            service = 'Reg'
            bid_up_var = getattr(m, 'Total_RegUp', None)
            bid_up = get_var_value(bid_up_var, t)
            bid_down_var = getattr(m, 'Total_RegDown', None)
            bid_down = get_var_value(bid_down_var, t)
            total_reg_bid = bid_up + bid_down
            mcp_cap = get_param(m, f'p_{service}', t)
            adder = get_param(m, f'loc_{service}', t)
            win_rate = get_param(m, f'winning_rate_{service}', t, 1.0)
            cap_payment = total_reg_bid * win_rate * mcp_cap
            energy_perf_payment = 0.0
            if simulate_dispatch:
                deployed_up = get_total_deployed_as(m, t, 'RegUp')
                deployed_down = get_total_deployed_as(m, t, 'RegDown')
                energy_perf_payment = (deployed_up - deployed_down) * lmp
            else:
                mileage = 1.0
                perf = 1.0
                energy_perf_payment = total_reg_bid * mileage * perf * lmp
            total_hourly_as_revenue_rate += cap_payment + energy_perf_payment + adder
            reserve_map = {'Spin': 'SR', 'Sup': 'NSR',
                           'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'}
            for service_iso, internal_service in reserve_map.items():
                bid_var = getattr(m, f'Total_{internal_service}', None)
                bid = get_var_value(bid_var, t)
                if bid < 1e-6:
                    continue
                mcp = get_param(m, f'p_{service_iso}', t)
                adder = get_param(m, f'loc_{service_iso}', t)
                win_rate = get_param(m, f'winning_rate_{service_iso}', t, 1.0)
                cap_payment = bid * win_rate * mcp
                energy_payment = 0.0
                if simulate_dispatch:
                    deployed_amount = get_total_deployed_as(
                        m, t, internal_service)
                    energy_payment = deployed_amount * lmp
                else:
                    deploy_factor = get_param(
                        m, f'deploy_factor_{service_iso}', t)
                    energy_payment = bid * deploy_factor * lmp
                total_hourly_as_revenue_rate += cap_payment + energy_payment + adder

        return total_hourly_as_revenue_rate

    except AttributeError as e:
        logger.error(
            f"Missing component during hourly AS revenue calc for t={t}, ISO={iso_suffix}: {e}")
        return 0.0
    except Exception as e:
        logger.error(
            f"Error during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}", exc_info=True)
        return 0.0


# --- Main Results Extraction Function ---
def extract_results(model: pyo.ConcreteModel, target_iso: str, output_dir: str = '../output/Results_Standardized'):
    """
    Extracts comprehensive results from the solved Pyomo model, aligning with model.py,
    constraints.py, and revenue_cost.py logic, including conditional processing based on simulation mode.
    Uses dictionary-first approach to avoid DataFrame fragmentation. Uses imported utils.
    """
    logger.info(f"Extracting comprehensive results for {target_iso}...")
    if not hasattr(model, 'TARGET_ISO') or model.TARGET_ISO != target_iso:
        logger.warning(
            f"Model TARGET_ISO mismatches function arg ('{target_iso}'). Using model attribute if available, else function arg.")
    # Ensure model object has TARGET_ISO attribute, prefer model's if exists, else use arg
    model.TARGET_ISO = getattr(model, 'TARGET_ISO', target_iso)
    target_iso_local = model.TARGET_ISO  # Use consistent ISO value

    simulate_dispatch_mode = getattr(
        model, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    # Use flag from model if set
    can_provide_as_local = getattr(
        model, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    logger.info(
        f"Results extraction mode: {'Dispatch Execution' if simulate_dispatch_mode else 'Bidding Strategy'}")

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, 'TimePeriods') or not list(model.TimePeriods):
        logger.error(
            "Model has no TimePeriods defined. Cannot extract results.")
        return pd.DataFrame(), {}
    hours = list(model.TimePeriods)

    hourly_data = {}
    summary_results: Dict[str, Any] = {}
    time_factor = get_param(model, 'delT_minutes',
                            default=60.0) / 60.0  # Use util
    if time_factor <= 0:
        logger.error(
            "Invalid time_factor (<=0) obtained. Cannot proceed with result extraction.")
        return pd.DataFrame(), {}

    # --- Extract Optimized System Sizes & Store in Summary ---
    summary_results['Target_ISO'] = target_iso_local
    summary_results['Simulation_Mode'] = 'Dispatch Execution' if simulate_dispatch_mode else 'Bidding Strategy'

    elec_capacity_val = 0.0
    batt_capacity_val = 0.0
    batt_power_val = 0.0

    # Get config flags from the model object if they exist, otherwise use imported config
    enable_electrolyzer = getattr(
        model, 'ENABLE_ELECTROLYZER', ENABLE_ELECTROLYZER)
    enable_battery = getattr(model, 'ENABLE_BATTERY', ENABLE_BATTERY)
    enable_npp = getattr(model, 'ENABLE_NUCLEAR_GENERATOR',
                         ENABLE_NUCLEAR_GENERATOR)
    # Use LTE_MODE if available
    enable_lte = getattr(model, 'LTE_MODE', ENABLE_LOW_TEMP_ELECTROLYZER)
    enable_h2_storage = getattr(model, 'ENABLE_H2_STORAGE', ENABLE_H2_STORAGE)
    enable_startup_shutdown = getattr(
        model, 'ENABLE_STARTUP_SHUTDOWN', ENABLE_STARTUP_SHUTDOWN)
    enable_degradation = getattr(
        model, 'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING', ENABLE_ELECTROLYZER_DEGRADATION_TRACKING)

    if enable_electrolyzer:
        elec_cap_component = getattr(model, 'pElectrolyzer_max', None)
        if isinstance(elec_cap_component, pyo.Param):
            elec_capacity_val = pyo.value(
                elec_cap_component)  # Already a float
            summary_results['Fixed_Electrolyzer_Capacity_MW'] = elec_capacity_val
        elif isinstance(elec_cap_component, pyo.Var):
            elec_capacity_val = get_var_value(elec_cap_component, default=0.0)
            summary_results['Optimal_Electrolyzer_Capacity_MW'] = elec_capacity_val
        else:  # Not found or other type
            # Fallback if var was expected but not found
            elec_capacity_val = get_param(
                model, 'pElectrolyzer_max_upper_bound', default=0.0)
            # Or handle as error
            summary_results['Assumed_Electrolyzer_Capacity_MW'] = elec_capacity_val
    hourly_data['Electrolyzer_Capacity_MW'] = [elec_capacity_val] * len(hours)

    if enable_battery:
        batt_cap_component = getattr(model, 'BatteryCapacity_MWh', None)
        batt_pow_component = getattr(model, 'BatteryPower_MW', None)

        if isinstance(batt_cap_component, pyo.Param):
            batt_capacity_val = pyo.value(batt_cap_component)
            # Should also be Param if capacity is Param
            batt_power_val = pyo.value(batt_pow_component)
            summary_results['Fixed_Battery_Capacity_MWh'] = batt_capacity_val
            summary_results['Fixed_Battery_Power_MW'] = batt_power_val
        elif isinstance(batt_cap_component, pyo.Var):
            batt_capacity_val = get_var_value(batt_cap_component, default=0.0)
            # Will be optimized or linked
            batt_power_val = get_var_value(batt_pow_component, default=0.0)
            summary_results['Optimal_Battery_Capacity_MWh'] = batt_capacity_val
            summary_results['Optimal_Battery_Power_MW'] = batt_power_val
        else:  # Not found or other type
            batt_capacity_val = get_param(
                model, 'BatteryCapacity_max', default=0.0)
            power_ratio = get_param(model, 'BatteryPowerRatio', default=0.0)
            batt_power_val = batt_capacity_val * power_ratio
            summary_results['Assumed_Battery_Capacity_MWh'] = batt_capacity_val
            summary_results['Assumed_Battery_Power_MW'] = batt_power_val
    hourly_data['Battery_Capacity_MWh'] = [batt_capacity_val] * len(hours)
    hourly_data['Battery_Power_MW'] = [batt_power_val] * len(hours)

    # --- Extract Hourly Variables into Dictionary Lists (Using get_var_value) ---
    logger.info("Extracting hourly variables...")
    var_extract_list = [
        # (Internal Var Name, Output Col Name, Always Extract?, Default if Disabled)
        ('pIES', 'pIES_MW', True, 0.0),
        ('pTurbine', 'pTurbine_MW', enable_npp, 0.0),
        ('qSteam_Turbine', 'qSteam_Turbine_MWth', enable_npp, 0.0),
        ('pElectrolyzer', 'pElectrolyzer_MW', enable_electrolyzer, 0.0),
        ('pElectrolyzerSetpoint', 'pElectrolyzerSetpoint_MW', enable_electrolyzer, 0.0),
        ('mHydrogenProduced', 'mHydrogenProduced_kg_hr', enable_electrolyzer, 0.0),
        ('qSteam_Electrolyzer', 'qSteam_Electrolyzer_MWth',
         enable_electrolyzer and not enable_lte, 0.0),
        ('pAuxiliary', 'pAuxiliary_MW', True, 0.0),  # Check hasattr later
        ('H2_storage_level', 'H2_Storage_Level_kg', enable_h2_storage, 0.0),
        ('H2_to_market', 'H2_to_Market_kg_hr', enable_h2_storage, 0.0),
        ('H2_from_storage', 'H2_from_Storage_kg_hr', enable_h2_storage, 0.0),
        ('H2_to_storage', 'H2_to_Storage_Input_kg_hr', enable_h2_storage, 0.0),
        ('uElectrolyzer', 'Electrolyzer_Status(0=off,1=on)', enable_startup_shutdown,
         (1.0 if enable_electrolyzer else 0.0)),  # Default depends if elec enabled
        ('vElectrolyzerStartup', 'Electrolyzer_Startup(0=no,1=yes)',
         enable_startup_shutdown, 0.0),
        ('wElectrolyzerShutdown', 'Electrolyzer_Shutdown(0=no,1=yes)',
         enable_startup_shutdown, 0.0),
        ('DegradationState', 'DegradationState_Units', enable_degradation, 0.0),
        ('pElectrolyzerRampPos', 'pElectrolyzerRampPos_MW',
         enable_electrolyzer, 0.0),  # Check hasattr later
        ('pElectrolyzerRampNeg', 'pElectrolyzerRampNeg_MW',
         enable_electrolyzer, 0.0),  # Check hasattr later
        ('qSteamElectrolyzerRampPos', 'qSteamElectrolyzerRampPos_MWth',
         enable_electrolyzer and not enable_lte, 0.0),  # Check hasattr later
        ('qSteamElectrolyzerRampNeg', 'qSteamElectrolyzerRampNeg_MWth',
         enable_electrolyzer and not enable_lte, 0.0),  # Check hasattr later
        ('BatterySOC', 'Battery_SOC_MWh', enable_battery, 0.0),
        ('BatteryCharge', 'Battery_Charge_MW', enable_battery, 0.0),
        ('BatteryDischarge', 'Battery_Discharge_MW', enable_battery, 0.0),
        ('BatteryBinaryCharge', 'Battery_Charge_Binary', enable_battery, 0.0),
        ('BatteryBinaryDischarge', 'Battery_Discharge_Binary', enable_battery, 0.0),
    ]

    for var_name, col_name, is_enabled, default_val in var_extract_list:
        if is_enabled:
            # Use getattr to handle missing optional vars
            var_component = getattr(model, var_name, None)
            if var_component is not None:
                hourly_data[col_name] = [get_var_value(
                    var_component, t, default=default_val) for t in hours]  # Use util
            # elif not is_enabled: # This condition is redundant with the outer 'if is_enabled'
                # hourly_data[col_name] = [default_val] * len(hours)
            else:  # Var expected based on flags but not found
                logger.warning(
                    f"Variable '{var_name}' expected but not found on model. Filling '{col_name}' with default {default_val}.")
                hourly_data[col_name] = [default_val] * len(hours)
        else:  # Feature disabled
            hourly_data[col_name] = [default_val] * len(hours)

    # --- Extract Ancillary Service Bids ---
    logger.info("Extracting AS bids...")
    as_service_list = ['RegUp', 'RegDown', 'SR', 'NSR',
                       'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
    components_providing_as = []
    if enable_electrolyzer:
        components_providing_as.append('Electrolyzer')
    if enable_battery:
        components_providing_as.append('Battery')
    if enable_npp and (enable_electrolyzer or enable_battery):
        components_providing_as.append('Turbine')

    all_as_components_labels = ['Electrolyzer',
                                'Battery', 'Turbine', 'Total']  # Include Total

    for comp_label in all_as_components_labels:
        for service in as_service_list:
            is_total = (comp_label == 'Total')
            base_name = f"Total_{service}" if is_total else f"{service}_{comp_label}"
            col_name = f"{base_name}_Bid_MW"

            if can_provide_as_local and hasattr(model, base_name):
                var_comp = getattr(model, base_name)
                if isinstance(var_comp, pyo.Var):
                    hourly_data[col_name] = [get_var_value(
                        var_comp, t, default=0.0) for t in hours]  # Use util
                else:  # It's a Param (likely fixed to 0)
                    hourly_data[col_name] = [get_param(
                        model, base_name, t, default=0.0) for t in hours]  # Use get_param for Param
            else:  # AS disabled or this specific var doesn't exist
                hourly_data[col_name] = [0.0] * len(hours)

    # --- Extract Deployed Ancillary Service Amounts (Conditionally) ---
    logger.info(
        "Extracting Deployed AS amounts (if in Dispatch Simulation mode)...")
    if simulate_dispatch_mode and can_provide_as_local:
        logger.info(
            "Dispatch Simulation Mode: Extracting *_Deployed variables.")
        for comp in components_providing_as:  # Use list of components actually providing AS
            for service in as_service_list:
                deployed_var_name = f"{service}_{comp}_Deployed"
                col_name = f'{deployed_var_name}_MW'
                if hasattr(model, deployed_var_name):
                    hourly_data[col_name] = [get_var_value(
                        getattr(model, deployed_var_name), t, default=0.0) for t in hours]  # Use util
                # else: # Optionally add zero columns if var missing for this component/service combo
                    # hourly_data[col_name] = [0.0] * len(hours)
    # else: (No need for else, columns are just not added if not in this mode)

    # --- Extract Input Prices/Factors ---
    logger.info("Extracting input prices and factors...")
    hourly_data['EnergyPrice_LMP_USDperMWh'] = [
        get_param(model, 'energy_price', t, default=0.0) for t in hours]  # Use util
    if can_provide_as_local:
        iso_service_map = {  # Define or import map - Must match model.py
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
            'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
            'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'],
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
            'ISONE': ['Spin10', 'NSpin10', 'OR30'],
            'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
        }

        if target_iso_local in iso_service_map:
            # Use the key from the map (e.g., 'RegU', 'Syn')
            for service_iso_key in iso_service_map[target_iso_local]:

                # Define expected param types based on the key
                param_types_to_extract = []
                is_factor_type_param = any(f in service_iso_key.lower() for f in [
                                           'factor', 'score', 'ratio'])

                if not is_factor_type_param:  # Add standard price/adder params if not a factor key
                    param_types_to_extract.extend(
                        [('p', 0.0), ('loc', 0.0), ('winning_rate', 1.0), ('deploy_factor', 0.0)])

                # Add specific factors based on ISO and key
                if target_iso_local == 'CAISO' and service_iso_key in ['RegU', 'RegD']:
                    param_types_to_extract.append(('mileage_factor', 1.0))
                if target_iso_local == 'PJM':
                    # Note: PJM factors are directly named, not prefixed with service_iso_key
                    if service_iso_key == 'performance_score':
                        param_types_to_extract.append(
                            ('performance_score', 1.0))
                    if service_iso_key == 'mileage_ratio':
                        param_types_to_extract.append(('mileage_ratio', 1.0))

                # Extract each relevant parameter type for this service_iso_key
                for param_prefix, default_val in param_types_to_extract:
                    # Construct the base name used for the param on the model
                    # This needs careful mapping based on how model.py adds params
                    param_base_name_on_model = ""
                    output_col_name = ""
                    if param_prefix in ['performance_score', 'mileage_ratio'] and target_iso_local == 'PJM':
                        # These are special cases in PJM map, param name doesn't include service_iso_key
                        # e.g., 'performance_score'
                        param_base_name_on_model = f"{param_prefix}"
                        # e.g., performance_score_PJM
                        output_col_name = f"{param_prefix}_{target_iso_local}"
                    else:
                        # Standard case: param name includes service_iso_key
                        # e.g., p_RegU, loc_Spin
                        param_base_name_on_model = f"{param_prefix}_{service_iso_key}"
                        # e.g., p_RegU_SPP
                        output_col_name = f"{param_prefix}_{service_iso_key}_{target_iso_local}"

                    # Extract using get_param util - it handles adding _TARGET_ISO automatically
                    hourly_data[output_col_name] = [get_param(
                        model, param_base_name_on_model, t, default=default_val) for t in hours]

    # --- Create DataFrame AFTER all data is collected ---
    logger.info("Creating final DataFrame from collected hourly data...")
    try:
        results_df = pd.DataFrame(
            hourly_data, index=pd.Index(hours, name='HourOfYear'))
    except ValueError as ve:
        logger.error(
            f"Error creating DataFrame, likely due to inconsistent array lengths: {ve}")
        # Log lengths for debugging
        for k, v in hourly_data.items():
            logger.debug(
                f"Length of '{k}': {len(v) if isinstance(v, list) else 'N/A'}")
        return pd.DataFrame(), {}  # Return empty structures

    # --- Calculate Hourly Revenues/Costs/Profit (using the created results_df) ---
    logger.info("Calculating hourly revenues...")
    if 'pIES_MW' in results_df.columns and 'EnergyPrice_LMP_USDperMWh' in results_df.columns:
        results_df['Revenue_Energy_USD'] = results_df['pIES_MW'] * \
            results_df['EnergyPrice_LMP_USDperMWh'] * time_factor
    else:
        results_df['Revenue_Energy_USD'] = 0.0

    h2_value_param = get_param(
        model, 'H2_value', default=0.0) if enable_electrolyzer else 0.0
    h2_subsidy_param = get_param(
        model, 'hydrogen_subsidy_per_kg', default=0.0) if enable_electrolyzer else 0.0
    results_df['Revenue_Hydrogen_Sales_USD'] = 0.0
    results_df['Revenue_Hydrogen_Subsidy_USD'] = 0.0
    if enable_electrolyzer and 'mHydrogenProduced_kg_hr' in results_df.columns:
        results_df['Revenue_Hydrogen_Subsidy_USD'] = results_df['mHydrogenProduced_kg_hr'] * \
            h2_subsidy_param * time_factor
        if enable_h2_storage and 'H2_to_Market_kg_hr' in results_df.columns and 'H2_from_Storage_kg_hr' in results_df.columns:
            results_df['Revenue_Hydrogen_Sales_USD'] = (
                results_df['H2_to_Market_kg_hr'] + results_df['H2_from_Storage_kg_hr']) * h2_value_param * time_factor
        elif not enable_h2_storage:
            results_df['Revenue_Hydrogen_Sales_USD'] = results_df['mHydrogenProduced_kg_hr'] * \
                h2_value_param * time_factor
    results_df['Revenue_Hydrogen_USD'] = results_df['Revenue_Hydrogen_Sales_USD'] + \
        results_df['Revenue_Hydrogen_Subsidy_USD']

    results_df['Revenue_Ancillary_USD'] = [
        calculate_hourly_as_revenue(model, t) * time_factor for t in hours]
    results_df['Revenue_Total_USD'] = results_df[['Revenue_Energy_USD',
                                                  'Revenue_Hydrogen_USD', 'Revenue_Ancillary_USD']].sum(axis=1)

    logger.info("Calculating hourly costs...")
    # Define cost components based on features and check column existence
    cost_calc_list = [
        ('Cost_VOM_Turbine_USD', enable_npp, 'vom_turbine', 'pTurbine_MW'),
        ('Cost_VOM_Electrolyzer_USD', enable_electrolyzer,
         'vom_electrolyzer', 'pElectrolyzer_MW'),
        ('Cost_VOM_Battery_USD', enable_battery, 'vom_battery_per_mwh_cycled', [
         'Battery_Charge_MW', 'Battery_Discharge_MW']),  # Special case: needs sum + division?
        ('Cost_Water_USD', enable_electrolyzer, 'cost_water_per_kg_h2',
         'mHydrogenProduced_kg_hr'),  # Needs time factor? No, rate is per kg
        ('Cost_Ramping_USD', enable_electrolyzer, 'cost_electrolyzer_ramping', [
         'pElectrolyzerRampPos_MW', 'pElectrolyzerRampNeg_MW']),  # Special case: no time factor
        ('Cost_Storage_Cycle_USD', enable_h2_storage, 'vom_storage_cycle', [
         'H2_to_Storage_Input_kg_hr', 'H2_from_Storage_kg_hr']),  # Special case: needs time factor? No, rate is per kg
        ('Cost_Startup_USD', enable_startup_shutdown, 'cost_startup_electrolyzer',
         'Electrolyzer_Startup(0=no,1=yes)')  # Special case: no time factor
    ]

    cost_cols_for_total = []
    for cost_col, is_enabled, param_name, source_col_or_list in cost_calc_list:
        if is_enabled:
            cost_rate_param = get_param(
                model, param_name, default=0.0)  # Use util
            if cost_rate_param > 1e-9:  # Only calculate if cost rate is non-zero
                if isinstance(source_col_or_list, list):  # Multiple source columns
                    if all(col in results_df.columns for col in source_col_or_list):
                        source_sum = results_df[source_col_or_list].sum(axis=1)
                        # Rate is per MWh cycled, sum(charge+discharge)/2 ? Assume rate is per MWh thruput (charge OR discharge)
                        if cost_col == 'Cost_VOM_Battery_USD':
                            # Divide by 2 if rate is per round trip MWh
                            results_df[cost_col] = source_sum * \
                                cost_rate_param * time_factor / 2.0
                        # Costs applied per event/ramp, not per hour duration
                        elif cost_col == 'Cost_Ramping_USD' or cost_col == 'Cost_Startup_USD':
                            results_df[cost_col] = source_sum * cost_rate_param
                        elif cost_col == 'Cost_Storage_Cycle_USD':  # Rate is $/kg cycled
                            # Sum is kg/hr, rate is $/kg -> $/hr. Multiply by time_factor
                            results_df[cost_col] = source_sum * cost_rate_param
                            results_df[cost_col] = results_df[cost_col] * \
                                time_factor
                        else:  # VOMs (already handled), should not happen
                            results_df[cost_col] = 0.0
                        cost_cols_for_total.append(cost_col)
                    else:
                        results_df[cost_col] = 0.0
                else:  # Single source column
                    source_col = source_col_or_list
                    if source_col in results_df.columns:
                        if cost_col == 'Cost_Water_USD':  # Rate is $/kg, source is kg/hr
                            results_df[cost_col] = results_df[source_col] * \
                                cost_rate_param * time_factor  # Total kg * $/kg
                        else:  # VOMs: MW * $/MWh * hr -> $
                            results_df[cost_col] = results_df[source_col] * \
                                cost_rate_param * time_factor
                        cost_cols_for_total.append(cost_col)
                    else:
                        results_df[cost_col] = 0.0
            else:
                results_df[cost_col] = 0.0  # Cost rate is zero
        else:
            results_df[cost_col] = 0.0  # Feature disabled

    # Correct Ramping and Startup cost for first hour if applicable
    if 'Cost_Ramping_USD' in results_df.columns and not results_df.empty:
        results_df.loc[results_df.index.min(), 'Cost_Ramping_USD'] = 0.0

    results_df['Cost_HourlyOpex_Total_USD'] = results_df[cost_cols_for_total].sum(
        axis=1)

    logger.info("Calculating hourly profit...")
    results_df['Profit_Hourly_USD'] = results_df['Revenue_Total_USD'] - \
        results_df['Cost_HourlyOpex_Total_USD']

    # --- Calculate Summary Statistics ---
    logger.info("Calculating summary statistics...")
    summary_results['Total_Revenue_USD'] = results_df['Revenue_Total_USD'].sum()
    summary_results['Total_Energy_Revenue_USD'] = results_df['Revenue_Energy_USD'].sum()
    summary_results['Total_Hydrogen_Revenue_USD'] = results_df['Revenue_Hydrogen_USD'].sum()
    if 'Revenue_Hydrogen_Sales_USD' in results_df.columns:
        summary_results['Total_Hydrogen_Sales_Revenue_USD'] = results_df['Revenue_Hydrogen_Sales_USD'].sum()
    if 'Revenue_Hydrogen_Subsidy_USD' in results_df.columns:
        summary_results['Total_Hydrogen_Subsidy_Revenue_USD'] = results_df['Revenue_Hydrogen_Subsidy_USD'].sum()
    summary_results['Total_Ancillary_Revenue_USD'] = results_df['Revenue_Ancillary_USD'].sum()

    summary_results['Total_Hourly_Opex_USD'] = results_df['Cost_HourlyOpex_Total_USD'].sum()
    # Sum individual cost components from results_df, checking existence
    summary_results['Total_VOM_Cost_USD'] = sum(results_df[col].sum() for col in [
                                                'Cost_VOM_Turbine_USD', 'Cost_VOM_Electrolyzer_USD', 'Cost_VOM_Battery_USD'] if col in results_df.columns)
    summary_results['Total_Water_Cost_USD'] = results_df['Cost_Water_USD'].sum(
    ) if 'Cost_Water_USD' in results_df.columns else 0.0
    summary_results['Total_Ramping_Cost_USD'] = results_df['Cost_Ramping_USD'].sum(
    ) if 'Cost_Ramping_USD' in results_df.columns else 0.0
    summary_results['Total_Storage_Cycle_Cost_USD'] = results_df['Cost_Storage_Cycle_USD'].sum(
    ) if 'Cost_Storage_Cycle_USD' in results_df.columns else 0.0
    summary_results['Total_Startup_Cost_USD'] = results_df['Cost_Startup_USD'].sum(
    ) if 'Cost_Startup_USD' in results_df.columns else 0.0

    # Calculate Annualized Capex (replicating logic from model.py)
    total_annualized_capex = 0.0
    electrolyzer_annual_capex = 0.0
    battery_annual_capex = 0.0
    total_hours_sim = len(hours) * time_factor
    # Use 8760 hours for annualization regardless of simulation length
    scaling_factor = total_hours_sim / 8760.0 if total_hours_sim > 0 else 0.0

    if enable_electrolyzer:
        cost_elec_cap_param = get_param(
            model, 'cost_electrolyzer_capacity', default=0.0)  # Use util
        electrolyzer_annual_capex = elec_capacity_val * \
            cost_elec_cap_param * scaling_factor
        total_annualized_capex += electrolyzer_annual_capex

    if enable_battery:
        cost_batt_cap_mwh_yr = get_param(
            model, 'BatteryCapex_USD_per_MWh_year', default=0.0)  # Use util
        cost_batt_pow_mw_yr = get_param(
            model, 'BatteryCapex_USD_per_MW_year', default=0.0)  # Use util
        cost_batt_fom_mwh_yr = get_param(
            model, 'BatteryFixedOM_USD_per_MWh_year', default=0.0)  # Use util
        battery_annual_capex = (batt_capacity_val * cost_batt_cap_mwh_yr +
                                batt_power_val * cost_batt_pow_mw_yr +
                                batt_capacity_val * cost_batt_fom_mwh_yr) * scaling_factor
        total_annualized_capex += battery_annual_capex

    summary_results['Total_Annualized_Capex_USD'] = total_annualized_capex
    summary_results['Electrolyzer_Annualized_Capex_USD'] = electrolyzer_annual_capex
    summary_results['Battery_Annualized_Capex_USD'] = battery_annual_capex

    summary_results['Total_Profit_Calculated_USD'] = summary_results['Total_Revenue_USD'] - \
        summary_results['Total_Hourly_Opex_USD'] - \
        summary_results['Total_Annualized_Capex_USD']
    # Extract objective value using get_var_value for safety, although objectives usually aren't None if solved
    summary_results['Objective_Value_USD'] = get_var_value(
        getattr(model, 'TotalProfit_Objective', None), default=None)  # Use util

    # Other Summary Metrics
    if 'mHydrogenProduced_kg_hr' in results_df.columns:
        summary_results['Total_H2_Produced_kg'] = results_df['mHydrogenProduced_kg_hr'].sum(
        ) * time_factor
    else:
        summary_results['Total_H2_Produced_kg'] = 0.0

    if enable_h2_storage:
        summary_results['Total_H2_to_Market_Direct_kg'] = results_df['H2_to_Market_kg_hr'].sum(
        ) * time_factor if 'H2_to_Market_kg_hr' in results_df.columns else 0.0
        summary_results['Total_H2_from_Storage_kg'] = results_df['H2_from_Storage_kg_hr'].sum(
        ) * time_factor if 'H2_from_Storage_kg_hr' in results_df.columns else 0.0
        if 'H2_Storage_Level_kg' in results_df.columns and not results_df.empty:
            summary_results['Final_H2_Storage_Level_kg'] = results_df['H2_Storage_Level_kg'].iloc[-1]
        else:
            summary_results['Final_H2_Storage_Level_kg'] = 0.0

    if elec_capacity_val > 1e-6 and 'pElectrolyzer_MW' in results_df.columns:
        avg_elec_power_actual = results_df['pElectrolyzer_MW'].mean()
        capacity_factor_actual = avg_elec_power_actual / \
            elec_capacity_val if elec_capacity_val > 0 else 0.0
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = capacity_factor_actual
    else:
        summary_results['Electrolyzer_Capacity_Factor_Actual'] = 0.0

    # Check original extracted list
    if 'Electrolyzer_Startup(0=no,1=yes)' in hourly_data:
        summary_results['Total_Electrolyzer_Startups'] = int(
            np.sum(hourly_data['Electrolyzer_Startup(0=no,1=yes)']))
    if 'DegradationState_Units' in hourly_data and hourly_data['DegradationState_Units']:
        # Get last valid state
        last_state = pd.Series(hourly_data['DegradationState_Units']).iloc[-1]
        summary_results['Final_DegradationState_Units'] = last_state if pd.notna(
            last_state) else 0.0

    # Add Deployed AS Summary Stats
    if simulate_dispatch_mode and can_provide_as_local:
        logger.info("Calculating total deployed AS amounts...")
        for service in as_service_list:
            total_deployed_mwh = 0.0
            temp_sum_mw = 0.0
            for comp in components_providing_as:
                comp_col_name = f"{service}_{comp}_Deployed_MW"
                if comp_col_name in results_df.columns:
                    temp_sum_mw += results_df[comp_col_name].sum()
            total_deployed_mwh = temp_sum_mw * time_factor
            summary_results[f'Total_Deployed_{service}_MWh'] = total_deployed_mwh

    # --- Final Formatting and Saving ---
    if not summary_results:
        logger.warning(
            "Summary results dictionary is empty. Skipping summary file generation.")
    else:
        # Save summary results
        output_summary_path = results_dir / \
            f'{target_iso_local}_Summary_Results.txt'
        try:
            logger.info(
                f"Attempting to write summary results to: {output_summary_path}")
            with open(output_summary_path, 'w') as f:
                f.write(f"--- Summary Results for {target_iso_local} ---\n")
                f.write(
                    f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}\n")
                f.write(
                    f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}\n")
                # Write remaining items, ensuring specific order or grouping if desired
                key_order = [k for k in summary_results.keys() if k not in [
                    'Simulation_Mode', 'Target_ISO']]
                # Separate Objective and Calculated Profit for clarity
                obj_val_key = 'Objective_Value_USD'
                calc_prof_key = 'Total_Profit_Calculated_USD'
                if obj_val_key in key_order:
                    key_order.remove(obj_val_key)
                if calc_prof_key in key_order:
                    key_order.remove(calc_prof_key)

                for key in key_order:
                    value = summary_results[key]
                    try:
                        if value is None:
                            line = f"{key}: None\n"
                        elif isinstance(value, (float, np.floating)):
                            line = f"{key}: {value:,.4f}\n"
                        elif isinstance(value, (int, np.integer)):
                            line = f"{key}: {value:,}\n"
                        else:
                            line = f"{key}: {value}\n"
                        f.write(line)
                    except Exception as write_err:
                        logger.error(
                            f"Error writing summary key '{key}' with value '{value}' (type: {type(value)}): {write_err}")
                        f.write(f"{key}: ERROR_WRITING_VALUE\n")

                # Write Objective and Calculated Profit at the end with comparison
                f.write("\n--- Profitability ---\n")
                obj_val = summary_results.get(obj_val_key)
                calc_prof = summary_results.get(calc_prof_key)
                if obj_val is not None:
                    f.write(f"{obj_val_key}: {obj_val:,.4f}\n")
                else:
                    f.write(f"{obj_val_key}: N/A (Solver Failed?)\n")
                if calc_prof is not None:
                    f.write(f"{calc_prof_key}: {calc_prof:,.4f}\n")
                else:
                    f.write(f"{calc_prof_key}: N/A (Calculation Error?)\n")

                if isinstance(obj_val, (int, float, np.number)) and isinstance(calc_prof, (int, float, np.number)):
                    diff = calc_prof - obj_val
                    f.write(
                        f"Objective vs Calculated Profit Diff: {diff:,.4f}\n")
                    if abs(diff) > 1.0:  # Tolerance for float differences
                        f.write(
                            "WARNING: Significant difference between objective value and calculated profit!\n")
                        logger.warning(
                            "Significant difference between objective value and calculated profit detected.")
                else:
                    f.write(
                        "\nCould not compare Objective vs Calculated Profit (one or both invalid).\n")

            logger.info(
                f"Summary results successfully saved to {output_summary_path}")
            print(f"Summary results saved to {output_summary_path}")
        except Exception as e:
            logger.error(
                f"Failed to save summary results file: {e}", exc_info=True)
            print(
                f"Error: Failed to save summary results file to {output_summary_path}")

    # Save hourly results dataframe regardless of summary success
    if not results_df.empty:
        results_df = results_df.round(4)  # Round before saving
        output_csv_path = results_dir / \
            f'{target_iso_local}_Hourly_Results_Comprehensive.csv'
        try:
            results_df.to_csv(output_csv_path)
            logger.info(
                f"Comprehensive hourly results saved to {output_csv_path}")
            print(f"Comprehensive hourly results saved to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save hourly results CSV: {e}")
            print(
                f"Error: Failed to save hourly results CSV to {output_csv_path}")
    else:
        logger.warning("Hourly results DataFrame is empty. Skipping CSV save.")

    # Print summary to console (optional, as it's saved to file)
    if summary_results:
        print("\n--- Summary (also saved to file) ---")
        print(
            f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}")
        print(f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}")
        obj_val_console = summary_results.get('Objective_Value_USD', 'N/A')
        calc_prof_console = summary_results.get(
            'Total_Profit_Calculated_USD', 'N/A')
        if isinstance(obj_val_console, (int, float, np.number)):
            print(f"Objective_Value_USD: {obj_val_console:,.4f}")
        else:
            print(f"Objective_Value_USD: {obj_val_console}")
        if isinstance(calc_prof_console, (int, float, np.number)):
            print(f"Total_Profit_Calculated_USD: {calc_prof_console:,.4f}")
        else:
            print(f"Total_Profit_Calculated_USD: {calc_prof_console}")
        # Print a few other key metrics maybe
        print(
            f"Total_H2_Produced_kg: {summary_results.get('Total_H2_Produced_kg', 0.0):,.2f}")
        print("---------------------------------")

    return results_df, summary_results

# --- End of extract_results function ---
