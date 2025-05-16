# src/revenue_cost.py

"""
Revenue and cost expression rules.

Handles two modes for Ancillary Service (AS) revenue calculation based on
the model's SIMULATE_AS_DISPATCH_EXECUTION flag:
1. Bidding Strategy Mode (Flag=False): Calculates energy/performance revenue
   based on bids and provided factors (deploy_factor, mileage, performance).
2. Dispatch Execution Mode (Flag=True): Calculates energy/performance revenue
   based on the optimized *Deployed* AS variables and market prices (LMP).

Capacity revenue is always based on winning bids.
Requires CAN_PROVIDE_ANCILLARY_SERVICES flag from config.py (passed via model).
"""
import pyomo.environ as pyo
from logging_setup import logger
from utils import (
    get_param,
    get_symbolic_as_deployed_sum
)


def energy_revenue_rule(m):
    """Calculate net energy market revenue expression."""
    try:
        if not hasattr(m, 'pIES') or not hasattr(m, 'energy_price') or not hasattr(m, 'delT_minutes'):
            logger.error("Missing pIES, energy_price, or delT_minutes for EnergyRevenue_rule.")
            return 0.0
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0:
            raise ValueError("delT_minutes must result in a positive time_factor.")
        total_revenue_expr = sum(
            m.pIES[t] * m.energy_price[t] * time_factor for t in m.TimePeriods)
        return total_revenue_expr
    except Exception as e:
        logger.critical(f"CRITICAL Error defining EnergyRevenue_rule expression: {e}", exc_info=True)
        raise


def hydrogen_revenue_rule(m):
    """Calculate revenue expression from selling hydrogen."""
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_electrolyzer:
        return 0.0
    try:
        h2_value = m.H2_value
        h2_subsidy = m.hydrogen_subsidy_per_kg
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0:
            raise ValueError("delT_minutes must result in a positive time_factor.")
        total_revenue_expr = 0.0
        if not enable_h2_storage:
            if not hasattr(m, 'mHydrogenProduced'):
                logger.error("Missing mHydrogenProduced for H2 Revenue (no storage).")
                return 0.0
            total_revenue_expr = sum(
                (h2_value + h2_subsidy) * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)
        else:
            if not hasattr(m, 'H2_to_market') or not hasattr(m, 'H2_from_storage') or not hasattr(m, 'mHydrogenProduced'):
                logger.error("Missing vars for H2 Revenue (with storage).")
                return 0.0
            revenue_from_sales_expr = sum(
                h2_value * (m.H2_to_market[t] + m.H2_from_storage[t]) * time_factor for t in m.TimePeriods)
            revenue_from_subsidy_expr = sum(
                h2_subsidy * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)
            total_revenue_expr = revenue_from_sales_expr + revenue_from_subsidy_expr
        return total_revenue_expr
    except AttributeError as e:
        logger.critical(f"CRITICAL Missing var/param for HydrogenRevenue rule: {e}.", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"CRITICAL Error defining HydrogenRevenue rule: {e}", exc_info=True)
        raise


def _get_total_deployed_sum_for_service(m, t, internal_service_name):
    """Calculates the total symbolic deployed sum for a service across all eligible components."""
    total_deployed = 0.0
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)

    if enable_electrolyzer:
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service_name], 'Electrolyzer')
    if enable_battery:
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service_name], 'Battery')
    if enable_npp and (enable_electrolyzer or enable_battery):
        total_deployed += get_symbolic_as_deployed_sum(m, t, [internal_service_name], 'Turbine')
    return total_deployed


def _calculate_standard_regulation_revenue(m, t, iso_service_name, internal_service_name, lmp):
    """
    Standard calculation logic for a single regulation service (Up or Down).
    iso_service_name: The name used for parameter lookup (e.g., 'RegU', 'RegD', 'RegUp', 'RegDown').
    internal_service_name: The name used for bid variables (e.g., 'RegUp', 'RegDown').
    """
    revenue_expr = 0.0
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    bid_var = getattr(m, f'Total_{internal_service_name}', None)

    if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
        bid = bid_var[t]
        mcp_cap = get_param(m, f'p_{iso_service_name}', t, default=0.0)
        adder = get_param(m, f'loc_{iso_service_name}', t, default=0.0)
        win_rate = get_param(m, f'winning_rate_{iso_service_name}', t, default=1.0)
        
        cap_payment = bid * win_rate * mcp_cap
        energy_perf_payment = 0.0

        if simulate_dispatch:
            deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service_name)
            energy_perf_payment = deployed_amount_expr * lmp
        else:
            # Standardized calculation for all ISOs.
            # Assumes model.py loads parameters like 'mileage_factor_RegUp_PJM', 'performance_factor_RegUp_PJM' etc.
            # The get_param function will try to fetch f'{param_base_name}_{TARGET_ISO}' first.
            mileage = get_param(m, f'mileage_factor_{iso_service_name}', t, default=1.0)
            perf = get_param(m, f'performance_factor_{iso_service_name}', t, default=1.0)
            energy_perf_payment = bid * win_rate * mileage * perf * lmp
        
        revenue_expr = cap_payment + energy_perf_payment + adder
    return revenue_expr


def _symbolic_hourly_spp_revenue_expr(m, t):
    """Returns symbolic expression for SPP AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # Regulation Up (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegU', 'RegUp', lmp)
    # Regulation Down (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegD', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'RamU': 'RampUp', 'RamD': 'RampDown', 'UncU': 'UncU'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_caiso_revenue_expr(m, t):
    """Returns symbolic expression for CAISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # Regulation Up (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegU', 'RegUp', lmp)
    # Regulation Down (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegD', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'RMU': 'RampUp', 'RMD': 'RampDown'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_ercot_revenue_expr(m, t):
    """Returns symbolic expression for ERCOT AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # Regulation Up (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegU', 'RegUp', lmp)
    # Regulation Down (using standard calculation)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegD', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Spin': 'SR', 'NSpin': 'NSR', 'ECRS': 'ECRS'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_pjm_revenue_expr(m, t):
    """Returns symbolic expression for PJM AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # PJM Regulation is now split into RegUp and RegDown, following standard calculation.
    # iso_service_name for parameter lookup will be 'RegUp' and 'RegDown'.
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegUp', 'RegUp', lmp)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegDown', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Syn': 'SR', 'Rse': 'NSR', 'TMR': 'ThirtyMin'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0 
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_nyiso_revenue_expr(m, t):
    """Returns symbolic expression for NYISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # NYISO Regulation is now split into RegUp and RegDown.
    # iso_service_name for parameter lookup will be 'RegUp' and 'RegDown'.
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegUp', 'RegUp', lmp)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegDown', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'Res30': 'ThirtyMin'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_isone_revenue_expr(m, t):
    """Returns symbolic expression for ISONE AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # Reserves
    reserve_map = {'Spin10': 'SR', 'NSpin10': 'NSR', 'OR30': 'ThirtyMin'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def _symbolic_hourly_miso_revenue_expr(m, t):
    """Returns symbolic expression for MISO AS revenue rate for hour t."""
    hourly_revenue_rate_expr = 0.0
    lmp = m.energy_price[t]

    # MISO Regulation is now split into RegUp and RegDown.
    # iso_service_name for parameter lookup will be 'RegUp' and 'RegDown'.
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegUp', 'RegUp', lmp)
    hourly_revenue_rate_expr += _calculate_standard_regulation_revenue(m, t, 'RegDown', 'RegDown', lmp)

    # Reserves
    reserve_map = {'Spin': 'SR', 'Sup': 'NSR', 'STR': 'ThirtyMin', 'RamU': 'RampUp', 'RamD': 'RampDown'}
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    for service_iso, internal_service in reserve_map.items():
        bid_var = getattr(m, f'Total_{internal_service}', None)
        if bid_var is not None and bid_var.is_indexed() and t in bid_var.index_set():
            bid = bid_var[t]
            mcp = get_param(m, f'p_{service_iso}', t, default=0.0)
            adder = get_param(m, f'loc_{service_iso}', t, default=0.0)
            win_rate = get_param(m, f'winning_rate_{service_iso}', t, default=1.0)
            
            cap_payment = bid * win_rate * mcp
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_expr = _get_total_deployed_sum_for_service(m, t, internal_service)
                energy_payment = deployed_amount_expr * lmp
            else:
                deploy_factor = get_param(m, f'deploy_factor_{service_iso}', t, default=0.0)
                energy_payment = bid * win_rate * deploy_factor * lmp
            hourly_revenue_rate_expr += cap_payment + energy_payment + adder
    return hourly_revenue_rate_expr


def ancillary_revenue_rule_factory(iso_hourly_revenue_func):
    """Factory to create the total AS revenue rule for an ISO."""
    def _rule(m):
        if not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
            return 0.0
        try:
            time_factor = pyo.value(m.delT_minutes) / 60.0
            if time_factor <= 0:
                raise ValueError("delT_minutes must result in a positive time_factor.")
            return sum(iso_hourly_revenue_func(m, t) * time_factor for t in m.TimePeriods)
        except Exception as e:
            logger.critical(
                f"CRITICAL Error defining AncillaryRevenue rule using {iso_hourly_revenue_func.__name__}: {e}", exc_info=True)
            raise
    return _rule


AncillaryRevenue_SPP_rule = ancillary_revenue_rule_factory(_symbolic_hourly_spp_revenue_expr)
AncillaryRevenue_CAISO_rule = ancillary_revenue_rule_factory(_symbolic_hourly_caiso_revenue_expr)
AncillaryRevenue_ERCOT_rule = ancillary_revenue_rule_factory(_symbolic_hourly_ercot_revenue_expr)
AncillaryRevenue_PJM_rule = ancillary_revenue_rule_factory(_symbolic_hourly_pjm_revenue_expr)
AncillaryRevenue_NYISO_rule = ancillary_revenue_rule_factory(_symbolic_hourly_nyiso_revenue_expr)
AncillaryRevenue_ISONE_rule = ancillary_revenue_rule_factory(_symbolic_hourly_isone_revenue_expr)
AncillaryRevenue_MISO_rule = ancillary_revenue_rule_factory(_symbolic_hourly_miso_revenue_expr)


def opex_cost_rule(m):
    """Calculate total hourly operational costs expression."""
    total_opex_expr = 0.0
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)

    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 0:
            raise ValueError("delT_minutes must result in a positive time_factor.")

        cost_vom_turbine_expr = 0.0
        if enable_npp and hasattr(m, 'vom_turbine') and hasattr(m, 'pTurbine'):
            cost_vom_turbine_expr = sum(
                m.vom_turbine * m.pTurbine[t] * time_factor for t in m.TimePeriods)

        cost_vom_electrolyzer_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'vom_electrolyzer') and hasattr(m, 'pElectrolyzer'):
            cost_vom_electrolyzer_expr = sum(
                m.vom_electrolyzer * m.pElectrolyzer[t] * time_factor for t in m.TimePeriods)

        cost_vom_battery_expr = 0.0
        if enable_battery and hasattr(m, 'vom_battery_per_mwh_cycled') and hasattr(m, 'BatteryCharge') and hasattr(m, 'BatteryDischarge'):
            cost_vom_battery_expr = sum(m.vom_battery_per_mwh_cycled * (
                m.BatteryCharge[t] + m.BatteryDischarge[t]) / 2.0 * time_factor for t in m.TimePeriods)

        cost_water_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'cost_water_per_kg_h2') and hasattr(m, 'mHydrogenProduced'):
            cost_water_expr = sum(
                m.cost_water_per_kg_h2 * m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)

        cost_ramping_expr = 0.0
        if enable_electrolyzer and hasattr(m, 'cost_electrolyzer_ramping') and hasattr(m, 'pElectrolyzerRampPos') and hasattr(m, 'pElectrolyzerRampNeg'):
            cost_ramping_expr = sum(m.cost_electrolyzer_ramping * (
                m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t]) for t in m.TimePeriods if t > m.TimePeriods.first())

        cost_storage_cycle_expr = 0.0
        if enable_h2_storage and hasattr(m, 'vom_storage_cycle') and hasattr(m, 'H2_to_storage') and hasattr(m, 'H2_from_storage'):
            cost_storage_cycle_expr = sum(m.vom_storage_cycle * (
                m.H2_to_storage[t] + m.H2_from_storage[t]) * time_factor for t in m.TimePeriods)

        cost_startup_expr = 0.0
        if enable_startup_shutdown and hasattr(m, 'cost_startup_electrolyzer') and hasattr(m, 'vElectrolyzerStartup'):
            cost_startup_expr = sum(
                m.cost_startup_electrolyzer * m.vElectrolyzerStartup[t] for t in m.TimePeriods)

        total_opex_expr = (cost_vom_turbine_expr + cost_vom_electrolyzer_expr + cost_vom_battery_expr +
                           cost_water_expr + cost_ramping_expr + cost_storage_cycle_expr + cost_startup_expr)
        return total_opex_expr

    except AttributeError as e:
        logger.critical(f"CRITICAL Missing parameter/variable for OpexCost rule: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"CRITICAL Error defining OpexCost rule: {e}", exc_info=True)
        raise

# Rename rules to be consistent (optional, but good practice)
EnergyRevenue_rule = energy_revenue_rule
HydrogenRevenue_rule = hydrogen_revenue_rule
OpexCost_rule = opex_cost_rule
