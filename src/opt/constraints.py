"""
Constraints module for the optimization model.

This module contains all constraint rules used by the optimization model.
These functions are imported by model.py to create the constraints.
"""

import pyomo.environ as pyo

# Config flags are used by Pyomo rules, so direct import is fine here.
from config import (
    CAN_PROVIDE_ANCILLARY_SERVICES,
    ENABLE_BATTERY,
    ENABLE_ELECTROLYZER,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_H2_STORAGE,
    ENABLE_LOW_TEMP_ELECTROLYZER,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_STARTUP_SHUTDOWN,
)
from src.logger_utils.logging_setup import logger
from utils import get_symbolic_as_bid_sum, get_symbolic_as_deployed_sum

# This map defines which services are considered "actually available" or "expected" for each ISO.
# It will be used by link_deployed_to_bid_rule to determine if a warning for missing parameters is appropriate.
ACTUAL_ISO_SERVICES_PROVIDED = {
    "SPP": ["RegU", "RegD", "Spin", "Sup", "RamU", "RamD", "UncU"],
    "CAISO": ["RegU", "RegD", "Spin", "NSpin", "RMU", "RMD"],
    "ERCOT": ["RegU", "RegD", "Spin", "NSpin", "ECRS"],
    "PJM": ["RegUp", "RegDown", "Syn", "Rse", "TMR"],
    "NYISO": ["RegUp", "RegDown", "Spin10", "NSpin10", "Res30"],
    "ISONE": ["Spin10", "NSpin10", "OR30"],
    "MISO": ["RegUp", "RegDown", "Spin", "Sup", "STR", "RamU", "RamD"],
}


def steam_balance_rule(m, t):
    """Links total steam production to turbine and HTE electrolyzer use."""
    if not getattr(m, "ENABLE_NUCLEAR_GENERATOR", False):
        return pyo.Constraint.Skip
    try:
        turbine_steam = m.qSteam_Turbine[t] if hasattr(
            m, "qSteam_Turbine") else 0
        hte_steam = 0
        if (
            getattr(m, "ENABLE_ELECTROLYZER", False)
            and not getattr(m, "LTE_MODE", False)
            and hasattr(m, "qSteam_Electrolyzer")
        ):
            hte_steam = m.qSteam_Electrolyzer[t]
        return turbine_steam + hte_steam == m.qSteam_Total
    except Exception as e:
        logger.error(f"Error in steam_balance rule @t={t}: {e}", exc_info=True)
        raise


def power_balance_rule(m, t):
    """Ensures power generation equals consumption + net grid interaction."""
    try:
        turbine_power = (
            m.pTurbine[t]
            if getattr(m, "ENABLE_NUCLEAR_GENERATOR", False) and hasattr(m, "pTurbine")
            else 0
        )
        battery_discharge = (
            m.BatteryDischarge[t]
            if getattr(m, "ENABLE_BATTERY", False) and hasattr(m, "BatteryDischarge")
            else 0
        )

        electrolyzer_power = 0
        if getattr(m, "ENABLE_ELECTROLYZER", False) and hasattr(m, "pElectrolyzer"):
            electrolyzer_power = m.pElectrolyzer[t]  # Actual power consumed

        battery_charge = (
            m.BatteryCharge[t]
            if getattr(m, "ENABLE_BATTERY", False) and hasattr(m, "BatteryCharge")
            else 0
        )
        auxiliary_power = (
            m.pAuxiliary[t] if hasattr(m, "pAuxiliary") else 0
        )  # Exists if aux_power_consumption_per_kg_h2 > 0

        return (
            turbine_power
            + battery_discharge
            - electrolyzer_power
            - battery_charge
            - auxiliary_power
            == m.pIES[t]
        )
    except Exception as e:
        logger.error(f"Error in power_balance rule @t={t}: {e}", exc_info=True)
        raise


def constant_turbine_power_rule(m, t):
    """Fixes turbine power if LTE mode is active."""
    if not (
        getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
        and getattr(m, "ENABLE_ELECTROLYZER", False)
        and getattr(m, "LTE_MODE", False)
    ):
        return pyo.Constraint.Skip
    try:
        if hasattr(m, "pTurbine") and hasattr(m, "pTurbine_LTE_setpoint"):
            return m.pTurbine[t] == m.pTurbine_LTE_setpoint
        else:  # Should not happen if flags are set correctly
            logger.warning(
                f"Skipping constant_turbine_power rule @t={t}: Missing pTurbine or pTurbine_LTE_setpoint."
            )
            return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in constant_turbine_power rule @t={t}: {e}", exc_info=True)
        raise


def link_auxiliary_power_rule(m, t):
    """Links auxiliary power consumption to hydrogen production rate."""
    if not hasattr(m, "pAuxiliary") or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):  # pAuxiliary only exists if param > 0
        return pyo.Constraint.Skip
    try:
        if hasattr(m, "mHydrogenProduced") and hasattr(
            m, "aux_power_consumption_per_kg_h2"
        ):
            aux_rate = pyo.value(
                m.aux_power_consumption_per_kg_h2
            )  # MW_aux per kg/hr H2
            return m.pAuxiliary[t] == m.mHydrogenProduced[t] * aux_rate
        else:
            logger.warning(
                f"Skipping link_auxiliary_power rule @t={t}: Missing components."
            )
            return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in link_auxiliary_power rule @t={t}: {e}", exc_info=True)
        raise


def link_setpoint_to_actual_power_if_not_simulating_dispatch_rule(m, t):
    """If not simulating dispatch, actual electrolyzer power equals setpoint."""
    if not getattr(m, "ENABLE_ELECTROLYZER", False) or getattr(
        m, "SIMULATE_AS_DISPATCH_EXECUTION", False
    ):
        return pyo.Constraint.Skip
    try:
        if hasattr(m, "pElectrolyzer") and hasattr(m, "pElectrolyzerSetpoint"):
            return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t]
        return pyo.Constraint.Skip  # Should not happen if electrolyzer enabled
    except Exception as e:
        logger.error(
            f"Error in link_setpoint_to_actual_power rule @t={t}: {e}",
            exc_info=True,
        )
        raise


def electrolyzer_setpoint_min_power_rule(m, t):
    """Ensures the electrolyzer setpoint respects minimum turn-down if operational."""
    if not getattr(m, "ENABLE_ELECTROLYZER", False) or not hasattr(
        m, "pElectrolyzerSetpoint"
    ):
        return pyo.Constraint.Skip
    try:
        min_power_param = getattr(m, "pElectrolyzer_min", None)
        if min_power_param is None:
            logger.warning(
                f"Skipping setpoint_min_power @t={t}: pElectrolyzer_min not found."
            )
            return pyo.Constraint.Skip

        if getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) and hasattr(m, "uElectrolyzer"):
            return m.pElectrolyzerSetpoint[t] >= min_power_param * m.uElectrolyzer[t]
        else:  # If not using on/off logic, assume setpoint must always be above min if min_power > 0
            return (
                m.pElectrolyzerSetpoint[t] >= min_power_param
                if pyo.value(min_power_param) > 1e-6
                else pyo.Constraint.Skip
            )
    except Exception as e:
        logger.error(
            f"Error in electrolyzer_setpoint_min_power rule @t={t}: {e}",
            exc_info=True,
        )
        raise


# --- H2 STORAGE RULES ---
def h2_storage_balance_adj_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        discharge_eff = pyo.value(m.storage_discharge_eff)
        charge_eff = pyo.value(m.storage_charge_eff)
        discharge_term = (
            m.H2_from_storage[t] / discharge_eff if discharge_eff > 1e-9 else 0
        )
        charge_term = m.H2_to_storage[t] * charge_eff
        if t == m.TimePeriods.first():
            return (
                m.H2_storage_level[t]
                == m.H2_storage_level_initial + charge_term - discharge_term
            )
        else:
            return (
                m.H2_storage_level[t]
                == m.H2_storage_level[t - 1] + charge_term - discharge_term
            )
    except Exception as e:
        logger.error(f"H2 Storage Balance Error @t={t}: {e}", exc_info=True)
        raise


def h2_prod_dispatch_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        return m.mHydrogenProduced[t] == m.H2_to_market[t] + m.H2_to_storage[t]
    except Exception as e:
        logger.error(f"H2 Prod Dispatch Error @t={t}: {e}", exc_info=True)
        raise


def h2_storage_charge_limit_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        return m.H2_to_storage[t] <= m.H2_storage_charge_rate_max
    except Exception as e:
        logger.error(f"H2 Charge Limit Error @t={t}: {e}", exc_info=True)
        raise


def h2_storage_discharge_limit_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max
    except Exception as e:
        logger.error(f"H2 Discharge Limit Error @t={t}: {e}", exc_info=True)
        raise


def h2_storage_level_max_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        return m.H2_storage_level[t] <= m.H2_storage_capacity_max
    except Exception as e:
        logger.error(f"H2 Level Max Error @t={t}: {e}", exc_info=True)
        raise


def h2_storage_level_min_rule(m, t):
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        return m.H2_storage_level[t] >= m.H2_storage_capacity_min
    except Exception as e:
        logger.error(f"H2 Level Min Error @t={t}: {e}", exc_info=True)
        raise


# --- RAMP RATE RULES ---
def Electrolyzer_RampUp_rule(m, t):
    if not getattr(m, "ENABLE_ELECTROLYZER", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return (
            m.pElectrolyzer[t] - m.pElectrolyzer[t - 1]
            <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        )
    except Exception as e:
        logger.error(f"Elec RampUp Error @t={t}: {e}", exc_info=True)
        raise


def Electrolyzer_RampDown_rule(m, t):
    if not getattr(m, "ENABLE_ELECTROLYZER", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return (
            m.pElectrolyzer[t - 1] - m.pElectrolyzer[t]
            <= m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        )
    except Exception as e:
        logger.error(f"Elec RampDown Error @t={t}: {e}", exc_info=True)
        raise


def Turbine_RampUp_rule(m, t):
    if (
        not getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
        or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False))
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return m.pTurbine[t] - m.pTurbine[t - 1] <= m.RU_Turbine_hourly * time_factor
    except Exception as e:
        logger.error(f"Turbine RampUp Error @t={t}: {e}", exc_info=True)
        raise


def Turbine_RampDown_rule(m, t):
    if (
        not getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
        or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False))
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return m.pTurbine[t - 1] - m.pTurbine[t] <= m.RD_Turbine_hourly * time_factor
    except Exception as e:
        logger.error(f"Turbine RampDown Error @t={t}: {e}", exc_info=True)
        raise


def Steam_Electrolyzer_Ramp_rule(m, t):
    if (
        not (
            getattr(m, "ENABLE_ELECTROLYZER", False)
            and not getattr(m, "LTE_MODE", False)
        )
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip  # HTE only
    try:
        if hasattr(m, "qSteamElectrolyzerRampPos") and hasattr(
            m, "qSteamElectrolyzerRampNeg"
        ):
            time_factor = pyo.value(m.delT_minutes) / 60.0
            return (
                m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t]
                <= m.Ramp_qSteam_Electrolyzer_limit * time_factor
            )
        return pyo.Constraint.Skip
    except Exception as e:
        logger.error(f"Steam Elec Ramp Error @t={t}: {e}", exc_info=True)
        raise


# --- H2 CAPACITY FACTOR RULE ---
def h2_CapacityFactor_rule(m):
    if not getattr(m, "ENABLE_H2_CAP_FACTOR", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        total_hours_sim = len(m.TimePeriods) * \
            (pyo.value(m.delT_minutes) / 60.0)
        if (
            not hasattr(m, "pElectrolyzer_max")
            or pyo.value(m.pElectrolyzer_max) <= 1e-6
        ):
            if (
                isinstance(m.pElectrolyzer_max, pyo.Param)
                and pyo.value(m.pElectrolyzer_max) <= 1e-6
            ):
                return pyo.Constraint.Skip
            if (
                hasattr(m, "pElectrolyzer_max_upper_bound")
                and pyo.value(m.pElectrolyzer_max_upper_bound) <= 1e-6
            ):
                return pyo.Constraint.Skip
            max_elec_power_limit = (
                pyo.value(m.pElectrolyzer_max_upper_bound)
                if hasattr(m, "pElectrolyzer_max_upper_bound")
                else pyo.value(m.pElectrolyzer_max)
            )
        else:
            max_elec_power_limit = pyo.value(m.pElectrolyzer_max)

        if max_elec_power_limit <= 1e-6:
            return pyo.Constraint.Skip
        if not hasattr(m, "pElectrolyzer_efficiency_breakpoints") or not list(
            m.pElectrolyzer_efficiency_breakpoints
        ):
            return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last()
        if not hasattr(m, "ke_H2_inv_values") or max_power_bp not in m.ke_H2_inv_values:
            return pyo.Constraint.Skip
        max_h2_rate_kg_per_mwh = pyo.value(m.ke_H2_inv_values[max_power_bp])
        if max_h2_rate_kg_per_mwh < 1e-9:
            return pyo.Constraint.Skip

        max_h2_rate_kg_per_hr_est = max_elec_power_limit * max_h2_rate_kg_per_mwh
        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours_sim
        if max_potential_h2_kg_total_est <= 1e-6:
            return pyo.Constraint.Skip

        total_actual_production_kg = sum(
            m.mHydrogenProduced[t] * (pyo.value(m.delT_minutes) / 60.0)
            for t in m.TimePeriods
        )
        target_production_kg = (
            m.h2_target_capacity_factor * max_potential_h2_kg_total_est
        )
        return total_actual_production_kg >= target_production_kg
    except Exception as e:
        logger.error(f"H2 Cap Factor Error: {e}", exc_info=True)
        raise


# --- STARTUP/SHUTDOWN RULES ---
def electrolyzer_on_off_logic_rule(m, t):
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        if t == m.TimePeriods.first():
            return (
                m.uElectrolyzer[t] - m.uElectrolyzer_initial
                == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
            )
        else:
            return (
                m.uElectrolyzer[t] - m.uElectrolyzer[t - 1]
                == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
            )
    except Exception as e:
        logger.error(f"SU/SD Logic Error @t={t}: {e}", exc_info=True)
        raise


# Applies to actual physical power
def electrolyzer_min_power_when_on_rule(m, t):
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t] >= m.pElectrolyzer_min * m.uElectrolyzer[t]
    except Exception as e:
        logger.error(f"SU/SD Min Power Error @t={t}: {e}", exc_info=True)
        raise


def electrolyzer_max_power_rule(m, t):  # Applies to actual physical power
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t] <= m.pElectrolyzer_max * m.uElectrolyzer[t]
    except Exception as e:
        logger.error(f"SU/SD Max Power Error @t={t}: {e}", exc_info=True)
        raise


# Applies to actual physical power
def electrolyzer_min_power_sds_disabled_rule(m, t):
    if getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        return (
            m.pElectrolyzer[t] >= m.pElectrolyzer_min
            if pyo.value(m.pElectrolyzer_min) > 1e-6
            else pyo.Constraint.Skip
        )
    except Exception as e:
        logger.error(
            f"Min Power (SDS Disabled) Error @t={t}: {e}", exc_info=True)
        raise


def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1
    except Exception as e:
        logger.error(f"SU/SD Exclusivity Error @t={t}: {e}", exc_info=True)
        raise


def electrolyzer_min_uptime_rule(m, t):
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        min_uptime = pyo.value(m.MinUpTimeElectrolyzer)
        if t < min_uptime:
            return pyo.Constraint.Skip
        start_idx = max(m.TimePeriods.first(), t - min_uptime + 1)
        if not all(i in m.TimePeriods for i in range(start_idx, t + 1)):
            return pyo.Constraint.Skip
        return (
            sum(m.uElectrolyzer[i] for i in range(start_idx, t + 1))
            >= min_uptime * m.vElectrolyzerStartup[t]
        )
    except Exception as e:
        logger.error(f"Min Uptime Error @t={t}: {e}", exc_info=True)
        raise


def electrolyzer_min_downtime_rule(m, t):
    if not getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        min_downtime = pyo.value(m.MinDownTimeElectrolyzer)
        if t < min_downtime:
            return pyo.Constraint.Skip
        start_idx = max(m.TimePeriods.first(), t - min_downtime + 1)
        if not all(i in m.TimePeriods for i in range(start_idx, t + 1)):
            return pyo.Constraint.Skip
        return (
            sum((1 - m.uElectrolyzer[i]) for i in range(start_idx, t + 1))
            >= min_downtime * m.wElectrolyzerShutdown[t]
        )
    except Exception as e:
        logger.error(f"Min Downtime Error @t={t}: {e}", exc_info=True)
        raise


# --- ELECTROLYZER DEGRADATION RULE ---
# Based on actual power m.pElectrolyzer
def electrolyzer_degradation_rule(m, t):
    if not getattr(m, "ENABLE_ELECTROLYZER_DEGRADATION_TRACKING", False) or not getattr(
        m, "ENABLE_ELECTROLYZER", False
    ):
        return pyo.Constraint.Skip
    try:
        relative_load_expr = 0
        max_cap_var = m.pElectrolyzer_max
        epsilon = 1e-6
        # Ensure pElectrolyzer_max is treated as a value if it's a Param, or use the variable directly
        current_max_cap = (
            pyo.value(max_cap_var)
            if isinstance(max_cap_var, pyo.Param)
            else max_cap_var
        )

        relative_load_expr = m.pElectrolyzer[t] / (current_max_cap + epsilon)
        time_factor = pyo.value(m.delT_minutes) / 60.0
        op_factor = pyo.value(m.DegradationFactorOperation)
        startup_factor = pyo.value(m.DegradationFactorStartup)
        degradation_increase_op = op_factor * relative_load_expr * time_factor
        degradation_increase_su = 0.0
        if getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) and hasattr(
            m, "vElectrolyzerStartup"
        ):
            degradation_increase_su = m.vElectrolyzerStartup[t] * \
                startup_factor
        total_degradation_increase = degradation_increase_op + degradation_increase_su
        if t == m.TimePeriods.first():
            return (
                m.DegradationState[t]
                == m.DegradationStateInitial + total_degradation_increase
            )
        else:
            if (t - 1) not in m.TimePeriods or not hasattr(m, "DegradationState"):
                return pyo.Constraint.Skip
            return (
                m.DegradationState[t]
                == m.DegradationState[t - 1] + total_degradation_increase
            )
    except Exception as e:
        logger.error(f"Elec Degradation Error @t={t}: {e}", exc_info=True)
        raise


# --- BATTERY STORAGE RULES ---
def battery_soc_balance_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        initial_soc = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
        time_factor = pyo.value(m.delT_minutes) / 60.0
        charge_eff = pyo.value(m.BatteryChargeEff)
        discharge_eff = pyo.value(m.BatteryDischargeEff)
        discharge_term = (
            m.BatteryDischarge[t] /
            discharge_eff if discharge_eff > 1e-9 else 0
        )
        charge_term = m.BatteryCharge[t] * charge_eff
        if t == m.TimePeriods.first():
            return (
                m.BatterySOC[t]
                == initial_soc + (charge_term - discharge_term) * time_factor
            )
        else:
            return (
                m.BatterySOC[t]
                == m.BatterySOC[t - 1] + (charge_term - discharge_term) * time_factor
            )
    except Exception as e:
        logger.error(f"Battery SOC Balance Error @t={t}: {e}", exc_info=True)
        raise


def battery_charge_limit_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        return m.BatteryCharge[t] <= m.BatteryPower_MW * m.BatteryBinaryCharge[t]
    except Exception as e:
        logger.error(f"Battery Charge Limit Error @t={t}: {e}", exc_info=True)
        raise


def battery_discharge_limit_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        return m.BatteryDischarge[t] <= m.BatteryPower_MW * m.BatteryBinaryDischarge[t]
    except Exception as e:
        logger.error(
            f"Battery Discharge Limit Error @t={t}: {e}", exc_info=True)
        raise


def battery_binary_exclusivity_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        return m.BatteryBinaryCharge[t] + m.BatteryBinaryDischarge[t] <= 1
    except Exception as e:
        logger.error(
            f"Battery Binary Exclusivity Error @t={t}: {e}", exc_info=True)
        raise


def battery_soc_max_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        return m.BatterySOC[t] <= m.BatteryCapacity_MWh
    except Exception as e:
        logger.error(f"Battery SOC Max Error @t={t}: {e}", exc_info=True)
        raise


def battery_soc_min_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        return m.BatterySOC[t] >= m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
    except Exception as e:
        logger.error(f"Battery SOC Min Error @t={t}: {e}", exc_info=True)
        raise


def battery_ramp_up_rule(m, t):  # Charge ramp
    if not getattr(m, "ENABLE_BATTERY", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return m.BatteryCharge[t] - m.BatteryCharge[t - 1] <= ramp_limit_mw
    except Exception as e:
        logger.error(f"Battery RampUp Error @t={t}: {e}", exc_info=True)
        raise


def battery_ramp_down_rule(m, t):  # Charge ramp
    if not getattr(m, "ENABLE_BATTERY", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return m.BatteryCharge[t - 1] - m.BatteryCharge[t] <= ramp_limit_mw
    except Exception as e:
        logger.error(f"Battery RampDown Error @t={t}: {e}", exc_info=True)
        raise


def battery_discharge_ramp_up_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return m.BatteryDischarge[t] - m.BatteryDischarge[t - 1] <= ramp_limit_mw
    except Exception as e:
        logger.error(
            f"Battery Discharge RampUp Error @t={t}: {e}", exc_info=True)
        raise


def battery_discharge_ramp_down_rule(m, t):
    if not getattr(m, "ENABLE_BATTERY", False) or t == m.TimePeriods.first():
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return m.BatteryDischarge[t - 1] - m.BatteryDischarge[t] <= ramp_limit_mw
    except Exception as e:
        logger.error(
            f"Battery Discharge RampDown Error @t={t}: {e}", exc_info=True)
        raise


def battery_cyclic_soc_lower_rule(m):
    if not getattr(m, "ENABLE_BATTERY", False) or not pyo.value(
        m.BatteryRequireCyclicSOC
    ):
        return pyo.Constraint.Skip
    try:
        last_t = m.TimePeriods.last()
        initial_soc_expr = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
        tolerance = 0.01  # MWh
        return m.BatterySOC[last_t] >= initial_soc_expr - tolerance
    except Exception as e:
        logger.error(f"Battery Cyclic SOC Lower Error: {e}", exc_info=True)
        raise


def battery_cyclic_soc_upper_rule(m):
    if not getattr(m, "ENABLE_BATTERY", False) or not pyo.value(
        m.BatteryRequireCyclicSOC
    ):
        return pyo.Constraint.Skip
    try:
        last_t = m.TimePeriods.last()
        initial_soc_expr = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
        tolerance = 0.01  # MWh
        return m.BatterySOC[last_t] <= initial_soc_expr + tolerance
    except Exception as e:
        logger.error(f"Battery Cyclic SOC Upper Error: {e}", exc_info=True)
        raise


def battery_power_capacity_link_rule(m):
    if not getattr(m, "ENABLE_BATTERY", False) or isinstance(
        getattr(m, "BatteryCapacity_MWh", None), pyo.Param
    ):
        return pyo.Constraint.Skip  # Only if optimizing capacity
    try:
        return m.BatteryPower_MW == m.BatteryCapacity_MWh * m.BatteryPowerRatio
    except Exception as e:
        logger.error(f"Battery Power-Capacity Link Error: {e}", exc_info=True)
        raise


def battery_min_cap_rule(m):
    if not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip
    try:
        # In model.py, BatteryCapacity_min_param is used for the bound when optimizing
        if (
            hasattr(m, "BatteryCapacity_min_param")
            and pyo.value(m.BatteryCapacity_min_param) > 1e-6
        ):
            return m.BatteryCapacity_MWh >= m.BatteryCapacity_min_param
        return pyo.Constraint.Skip
    except Exception as e:
        logger.error(f"Battery Min Cap Error: {e}", exc_info=True)
        raise


# --- ANCILLARY SERVICE CAPABILITY RULES ---
def Turbine_AS_Zero_rule(m, t):
    turbine_as_disabled = not (
        getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        and getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
    ) or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False))
    if turbine_as_disabled:
        all_services = [
            "RegUp",
            "RegDown",
            "SR",
            "NSR",
            "ECRS",
            "ThirtyMin",
            "RampUp",
            "RampDown",
            "UncU",
        ]
        return get_symbolic_as_bid_sum(m, t, all_services, "Turbine") == 0.0
    return pyo.Constraint.Skip


def Turbine_AS_Pmax_rule(m, t):  # Upward capability
    if not (
        getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        and getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
    ) or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False)):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Turbine")
        return m.pTurbine[t] + up_bids <= m.pTurbine_max
    except Exception as e:
        logger.error(f"Turbine AS Pmax Error @t={t}: {e}", exc_info=True)
        raise


def Turbine_AS_Pmin_rule(m, t):  # Downward capability
    if not (
        getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        and getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
    ) or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False)):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(m, t, down_services, "Turbine")
        return m.pTurbine[t] - down_bids >= m.pTurbine_min
    except Exception as e:
        logger.error(f"Turbine AS Pmin Error @t={t}: {e}", exc_info=True)
        raise


def Turbine_AS_RU_rule(m, t):
    if (
        not (
            getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
            and getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
        )
        or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False))
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Turbine")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return (m.pTurbine[t] + up_bids) - m.pTurbine[
            t - 1
        ] <= m.RU_Turbine_hourly * time_factor
    except Exception as e:
        logger.error(f"Turbine AS RU Error @t={t}: {e}", exc_info=True)
        raise


def Turbine_AS_RD_rule(m, t):
    if (
        not (
            getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
            and getattr(m, "ENABLE_NUCLEAR_GENERATOR", False)
        )
        or (getattr(m, "LTE_MODE", False) and getattr(m, "ENABLE_ELECTROLYZER", False))
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(m, t, down_services, "Turbine")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return (
            m.pTurbine[t - 1] - (m.pTurbine[t] - down_bids)
            <= m.RD_Turbine_hourly * time_factor
        )
    except Exception as e:
        logger.error(f"Turbine AS RD Error @t={t}: {e}", exc_info=True)
        raise


def Electrolyzer_AS_Pmax_rule(
    m, t
):  # Capability to increase load (Down-reserve from setpoint)
    if not (
        getattr(m, "ENABLE_ELECTROLYZER", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(
            m, t, down_services, "Electrolyzer")
        max_power_limit = m.pElectrolyzer_max
        if getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) and hasattr(m, "uElectrolyzer"):
            max_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_max
        return m.pElectrolyzerSetpoint[t] + down_bids <= max_power_limit
    except Exception as e:
        logger.error(f"Elec AS Pmax Error @t={t}: {e}", exc_info=True)
        raise


def Electrolyzer_AS_Pmin_rule(
    m, t
):  # Capability to decrease load (Up-reserve from setpoint)
    if not (
        getattr(m, "ENABLE_ELECTROLYZER", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Electrolyzer")
        min_power_limit = m.pElectrolyzer_min
        if getattr(m, "ENABLE_STARTUP_SHUTDOWN", False) and hasattr(m, "uElectrolyzer"):
            min_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_min
        return m.pElectrolyzerSetpoint[t] - up_bids >= min_power_limit
    except Exception as e:
        logger.error(f"Elec AS Pmin Error @t={t}: {e}", exc_info=True)
        raise


def Electrolyzer_AS_RU_rule(
    m, t
):  # Ramp for Down-reserve (Increasing Load from actual previous power)
    if (
        not (
            getattr(m, "ENABLE_ELECTROLYZER", False)
            and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        )
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(
            m, t, down_services, "Electrolyzer")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        ramp_limit = (
            m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        )
        # Compare (Setpoint + Down Bids) with previous actual power m.pElectrolyzer[t-1]
        return (m.pElectrolyzerSetpoint[t] + down_bids) - m.pElectrolyzer[
            t - 1
        ] <= ramp_limit
    except Exception as e:
        logger.error(f"Elec AS RU Error @t={t}: {e}", exc_info=True)
        raise


def Electrolyzer_AS_RD_rule(
    m, t
):  # Ramp for Up-reserve (Decreasing Load from actual previous power)
    if (
        not (
            getattr(m, "ENABLE_ELECTROLYZER", False)
            and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        )
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Electrolyzer")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        ramp_limit = (
            m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        )
        # Compare (Setpoint - Up Bids) with previous actual power m.pElectrolyzer[t-1]
        return (
            m.pElectrolyzer[t - 1] - (m.pElectrolyzerSetpoint[t] - up_bids)
            <= ramp_limit
        )
    except Exception as e:
        logger.error(f"Elec AS RD Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_Pmax_rule(m, t):  # Down-reserve (charge capability)
    if not (
        getattr(m, "ENABLE_BATTERY", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(m, t, down_services, "Battery")
        # Available headroom for charging based on current charge rate and binary status
        return (
            down_bids
            <= m.BatteryPower_MW * m.BatteryBinaryCharge[t] - m.BatteryCharge[t]
        )
    except Exception as e:
        logger.error(f"Battery AS Pmax Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_Pmin_rule(m, t):  # Up-reserve (discharge capability)
    if not (
        getattr(m, "ENABLE_BATTERY", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Battery")
        # Available headroom for discharging based on current discharge rate and binary status
        return (
            up_bids
            <= m.BatteryPower_MW * m.BatteryBinaryDischarge[t] - m.BatteryDischarge[t]
        )
    except Exception as e:
        logger.error(f"Battery AS Pmin Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_SOC_Up_rule(m, t):  # Energy for Up-reserve
    if not (
        getattr(m, "ENABLE_BATTERY", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Battery")
        discharge_eff = pyo.value(m.BatteryDischargeEff)
        as_duration = pyo.value(m.AS_Duration)
        energy_needed = up_bids * (
            as_duration /
            discharge_eff if discharge_eff > 1e-9 else float("inf")
        )
        min_soc = m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
        return m.BatterySOC[t] - energy_needed >= min_soc
    except Exception as e:
        logger.error(f"Battery AS SOC Up Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_SOC_Down_rule(m, t):  # Energy for Down-reserve
    if not (
        getattr(m, "ENABLE_BATTERY", False)
        and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(m, t, down_services, "Battery")
        charge_eff = pyo.value(m.BatteryChargeEff)
        as_duration = pyo.value(m.AS_Duration)
        energy_absorbed = down_bids * as_duration * charge_eff
        return m.BatterySOC[t] + energy_absorbed <= m.BatteryCapacity_MWh
    except Exception as e:
        logger.error(f"Battery AS SOC Down Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_RU_rule(m, t):  # Ramp for Down-reg (Increasing Charge)
    if (
        not (
            getattr(m, "ENABLE_BATTERY", False)
            and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        )
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        down_services = ["RegDown", "RampDown"]
        down_bids = get_symbolic_as_bid_sum(m, t, down_services, "Battery")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return (m.BatteryCharge[t] + down_bids) - m.BatteryCharge[t - 1] <= ramp_limit
    except Exception as e:
        logger.error(f"Battery AS RU Error @t={t}: {e}", exc_info=True)
        raise


def Battery_AS_RD_rule(m, t):  # Ramp for Up-reg (Increasing Discharge)
    if (
        not (
            getattr(m, "ENABLE_BATTERY", False)
            and getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        )
        or t == m.TimePeriods.first()
    ):
        return pyo.Constraint.Skip
    try:
        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        up_bids = get_symbolic_as_bid_sum(m, t, up_services, "Battery")
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6:
            return pyo.Constraint.Skip
        ramp_limit = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return (m.BatteryDischarge[t] + up_bids) - m.BatteryDischarge[
            t - 1
        ] <= ramp_limit
    except Exception as e:
        logger.error(f"Battery AS RD Error @t={t}: {e}", exc_info=True)
        raise


# --- ANCILLARY SERVICE LINKING RULES (BIDS) ---
def link_total_as_rule(m, t, service_name):
    """Generic rule to link component AS BIDS to the total bid for a service."""
    if not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False):
        return pyo.Constraint.Skip
    try:
        total_var = getattr(m, f"Total_{service_name}", None)
        if (
            total_var is None
            or not isinstance(total_var, pyo.Var)
            or not (total_var.is_indexed() and t in total_var.index_set())
        ):
            return pyo.Constraint.Skip

        turbine_bid = 0.0
        if getattr(m, "ENABLE_NUCLEAR_GENERATOR", False) and (
            getattr(m, "ENABLE_ELECTROLYZER", False)
            or getattr(m, "ENABLE_BATTERY", False)
        ):
            turbine_bid = get_symbolic_as_bid_sum(
                m, t, [service_name], "Turbine")

        electro_bid = 0.0
        if getattr(m, "ENABLE_ELECTROLYZER", False):
            electro_bid = get_symbolic_as_bid_sum(
                m, t, [service_name], "Electrolyzer")

        battery_bid = 0.0
        if getattr(m, "ENABLE_BATTERY", False):
            battery_bid = get_symbolic_as_bid_sum(
                m, t, [service_name], "Battery")

        return total_var[t] == turbine_bid + electro_bid + battery_bid
    except AttributeError as e:
        logger.debug(
            f"Attribute error linking total for service {service_name} at time {t}: {e}."
        )
        return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in link_total_as_rule for {service_name} @t={t}: {e}",
            exc_info=True,
        )
        raise


def battery_regulation_balance_rule(m, t):
    """Ensures RegUp and RegDown bids are equal for Battery at each time period."""
    if not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False) or not getattr(m, "ENABLE_BATTERY", False):
        return pyo.Constraint.Skip

    try:
        if (hasattr(m, "RegUp_Battery") and hasattr(m, "RegDown_Battery") and
            isinstance(getattr(m, "RegUp_Battery"), pyo.Var) and
            isinstance(getattr(m, "RegDown_Battery"), pyo.Var) and
                t in m.RegUp_Battery.index_set() and t in m.RegDown_Battery.index_set()):
            return m.RegUp_Battery[t] == m.RegDown_Battery[t]
        else:
            return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in battery_regulation_balance_rule @t={t}: {e}", exc_info=True)
        raise


def electrolyzer_regulation_balance_rule(m, t):
    """Ensures RegUp and RegDown bids are equal for Electrolyzer at each time period."""
    if not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False) or not getattr(m, "ENABLE_ELECTROLYZER", False):
        return pyo.Constraint.Skip

    try:
        if (hasattr(m, "RegUp_Electrolyzer") and hasattr(m, "RegDown_Electrolyzer") and
            isinstance(getattr(m, "RegUp_Electrolyzer"), pyo.Var) and
            isinstance(getattr(m, "RegDown_Electrolyzer"), pyo.Var) and
                t in m.RegUp_Electrolyzer.index_set() and t in m.RegDown_Electrolyzer.index_set()):
            return m.RegUp_Electrolyzer[t] == m.RegDown_Electrolyzer[t]
        else:
            return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in electrolyzer_regulation_balance_rule @t={t}: {e}", exc_info=True)
        raise


def turbine_regulation_balance_rule(m, t):
    """Ensures RegUp and RegDown bids are equal for Turbine at each time period."""
    if not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False) or not getattr(m, "ENABLE_NUCLEAR_GENERATOR", False):
        return pyo.Constraint.Skip

    # Only apply if turbine can provide AS (when other components are also enabled)
    if not (getattr(m, "ENABLE_ELECTROLYZER", False) or getattr(m, "ENABLE_BATTERY", False)):
        return pyo.Constraint.Skip

    try:
        if (hasattr(m, "RegUp_Turbine") and hasattr(m, "RegDown_Turbine") and
            isinstance(getattr(m, "RegUp_Turbine"), pyo.Var) and
            isinstance(getattr(m, "RegDown_Turbine"), pyo.Var) and
                t in m.RegUp_Turbine.index_set() and t in m.RegDown_Turbine.index_set()):
            return m.RegUp_Turbine[t] == m.RegDown_Turbine[t]
        else:
            return pyo.Constraint.Skip
    except Exception as e:
        logger.error(
            f"Error in turbine_regulation_balance_rule @t={t}: {e}", exc_info=True)
        raise


def link_Total_RegUp_rule(m, t):
    return link_total_as_rule(m, t, "RegUp")


def link_Total_RegDown_rule(m, t):
    return link_total_as_rule(m, t, "RegDown")


def link_Total_SR_rule(m, t):
    return link_total_as_rule(m, t, "SR")


def link_Total_NSR_rule(m, t):
    return link_total_as_rule(m, t, "NSR")


def link_Total_ECRS_rule(m, t):
    return link_total_as_rule(m, t, "ECRS")


def link_Total_30Min_rule(m, t):
    return link_total_as_rule(m, t, "ThirtyMin")


def link_Total_RampUp_rule(m, t):
    return link_total_as_rule(m, t, "RampUp")


def link_Total_RampDown_rule(m, t):
    return link_total_as_rule(m, t, "RampDown")


def link_Total_UncU_rule(m, t):
    return link_total_as_rule(m, t, "UncU")


# --- CONDITIONAL RULES for DISPATCH EXECUTION MODE ---
def link_deployed_to_bid_rule(m, t, internal_service_name, component_name):
    """
    Links DEPLOYED AS amount to the component's winning bid.
    If necessary parameters (winning_rate, deploy_factor) are missing FOR A DEFINED SERVICE,
    a warning is logged and the deployed amount for that service at that time is constrained to 0.
    If the service is not defined for the ISO (per ACTUAL_ISO_SERVICES_PROVIDED),
    missing parameters are expected, and deployed amount is constrained to 0 without a warning.
    """
    if not getattr(m, "SIMULATE_AS_DISPATCH_EXECUTION", False) or not getattr(
        m, "CAN_PROVIDE_ANCILLARY_SERVICES", False
    ):
        return pyo.Constraint.Skip
    try:
        deployed_var_name = f"{internal_service_name}_{component_name}_Deployed"
        bid_var_name = f"{internal_service_name}_{component_name}"

        if not (hasattr(m, deployed_var_name) and hasattr(m, bid_var_name)):
            logger.debug(
                f"Skipping link_deployed for {deployed_var_name} or {bid_var_name}: base variable(s) not found on model for {component_name} at t={t}."
            )
            if hasattr(m, deployed_var_name):
                deployed_var_obj = getattr(m, deployed_var_name)
                if deployed_var_obj.is_indexed() and t in deployed_var_obj.index_set():
                    return deployed_var_obj[t] == 0.0
            return pyo.Constraint.Skip

        deployed_var = getattr(m, deployed_var_name)[t]
        bid_var = getattr(m, bid_var_name)[t]
        target_iso = getattr(m, "TARGET_ISO", "UNKNOWN")

        # Map internal_service_name to the ISO-specific key used for parameter loading in model.py
        iso_specific_service_key = internal_service_name
        if target_iso == "SPP":
            if internal_service_name == "RegUp":
                iso_specific_service_key = "RegU"
            elif internal_service_name == "RegDown":
                iso_specific_service_key = "RegD"
            elif internal_service_name == "SR":
                iso_specific_service_key = "Spin"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "Sup"
        elif target_iso == "CAISO":
            if internal_service_name == "RegUp":
                iso_specific_service_key = "RegU"
            elif internal_service_name == "RegDown":
                iso_specific_service_key = "RegD"
            elif internal_service_name == "SR":
                iso_specific_service_key = "Spin"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "NSpin"
            elif internal_service_name == "RampUp":
                iso_specific_service_key = "RMU"
            elif internal_service_name == "RampDown":
                iso_specific_service_key = "RMD"
        elif target_iso == "ERCOT":
            if internal_service_name == "RegUp":
                iso_specific_service_key = "RegU"
            elif internal_service_name == "RegDown":
                iso_specific_service_key = "RegD"
            elif internal_service_name == "SR":
                iso_specific_service_key = "Spin"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "NSpin"
            # UncU and ECRS use their internal names as keys in ACTUAL_ISO_SERVICES_PROVIDED for ERCOT
        elif target_iso == "PJM":
            if internal_service_name == "SR":
                iso_specific_service_key = "Syn"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "Rse"
            elif internal_service_name == "ThirtyMin":
                iso_specific_service_key = "TMR"
        elif target_iso == "NYISO":
            if internal_service_name == "RegUp":
                iso_specific_service_key = "RegU"
            elif internal_service_name == "RegDown":
                iso_specific_service_key = "RegD"
            if internal_service_name == "SR":
                iso_specific_service_key = "Spin10"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "NSpin10"
            elif internal_service_name == "ThirtyMin":
                iso_specific_service_key = "Res30"
        elif target_iso == "ISONE":
            if internal_service_name == "SR":
                iso_specific_service_key = "Spin10"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "NSpin10"
            elif internal_service_name == "ThirtyMin":
                iso_specific_service_key = "OR30"
        elif target_iso == "MISO":
            if internal_service_name == "SR":
                iso_specific_service_key = "Spin"
            elif internal_service_name == "NSR":
                iso_specific_service_key = "Sup"
            elif internal_service_name == "ThirtyMin":
                iso_specific_service_key = "STR"

        win_rate_param_name = f"winning_rate_{iso_specific_service_key}_{target_iso}"
        deploy_factor_param_name = (
            f"deploy_factor_{iso_specific_service_key}_{target_iso}"
        )

        is_service_actually_provided = False
        if (
            target_iso in ACTUAL_ISO_SERVICES_PROVIDED
            and iso_specific_service_key in ACTUAL_ISO_SERVICES_PROVIDED[target_iso]
        ):
            is_service_actually_provided = True

        # Determine if the service is regulation or reserve for parameter checking
        is_regulation_service_flag = (
            "RegU" in internal_service_name
            or "RegD" in internal_service_name
            or "RegUp" in internal_service_name
            or "RegDown" in internal_service_name
        )

        params_on_model_sufficient = False
        required_params_missing_log_msg = ""

        if is_regulation_service_flag:
            # Regulation services primarily need winning_rate for this rule.
            # Deploy_factor is effectively 1.0 for them in dispatch mode.
            if hasattr(m, win_rate_param_name):
                params_on_model_sufficient = True
            else:
                required_params_missing_log_msg = f"'{win_rate_param_name}'"
        else:  # Reserve services
            if hasattr(m, win_rate_param_name) and hasattr(m, deploy_factor_param_name):
                params_on_model_sufficient = True
            else:
                missing = []
                if not hasattr(m, win_rate_param_name):
                    missing.append(f"'{win_rate_param_name}'")
                if not hasattr(m, deploy_factor_param_name):
                    missing.append(f"'{deploy_factor_param_name}'")
                required_params_missing_log_msg = " or ".join(missing)

        if not params_on_model_sufficient:
            if is_service_actually_provided:
                logger.warning(
                    f"Required parameter(s) {required_params_missing_log_msg} not found on model for "
                    f"defined service '{iso_specific_service_key}' (internal: '{internal_service_name}') of {component_name} "
                    f"for {target_iso} at t={t}. Constraining {deployed_var_name} to 0.0. "
                    f"This may indicate an issue with parameter loading in model.py or data generation."
                )
            else:
                logger.debug(
                    f"Service '{iso_specific_service_key}' (internal: '{internal_service_name}') is not listed in "
                    f"ACTUAL_ISO_SERVICES_PROVIDED for {target_iso}, or its essential parameters "
                    f"({required_params_missing_log_msg if required_params_missing_log_msg else 'winning_rate/deploy_factor'}) "
                    f"are missing from the model. Constraining {deployed_var_name} to 0.0."
                )
            return deployed_var == 0.0

        # If parameters exist on the model, proceed with the linking
        win_rate_param_val = getattr(m, win_rate_param_name)[t]

        effective_deploy_factor = 1.0  # Default for regulation
        if not is_regulation_service_flag:  # For reserves, get the deploy_factor
            effective_deploy_factor = getattr(m, deploy_factor_param_name)[t]

        return deployed_var == bid_var * win_rate_param_val * effective_deploy_factor
    except Exception as e:
        logger.error(
            f"Error in link_deployed_to_bid_rule for {internal_service_name} of {component_name} @t={t}: {e}",
            exc_info=True,
        )
        raise


def define_actual_electrolyzer_power_rule(m, t):
    """Defines actual pElectrolyzer based on Setpoint and Deployed AS."""
    if (
        not getattr(m, "SIMULATE_AS_DISPATCH_EXECUTION", False)
        or not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
        or not getattr(m, "ENABLE_ELECTROLYZER", False)
    ):
        return pyo.Constraint.Skip
    try:
        if not hasattr(m, "pElectrolyzerSetpoint") or not hasattr(m, "pElectrolyzer"):
            logger.error(
                "Missing pElectrolyzerSetpoint or pElectrolyzer for dispatch definition."
            )
            return pyo.Constraint.Skip  # Should not happen

        up_services = [
            "RegUp",
            "SR",
            "NSR",
            "RampUp",
            "UncU",
            "ThirtyMin",
            "ECRS",
        ]
        down_services = ["RegDown", "RampDown"]

        total_up_deployed_expr = get_symbolic_as_deployed_sum(
            m, t, up_services, "Electrolyzer"
        )
        total_down_deployed_expr = get_symbolic_as_deployed_sum(
            m, t, down_services, "Electrolyzer"
        )

        return (
            m.pElectrolyzer[t]
            == m.pElectrolyzerSetpoint[t]
            - total_up_deployed_expr
            + total_down_deployed_expr
        )
    except Exception as e:
        logger.error(
            f"Error in define_actual_electrolyzer_power_rule @t={t}: {e}",
            exc_info=True,
        )
        raise


def restrict_grid_purchase_rule(m, t):
    """
    Restricts grid power purchase to only when down-regulation services are deployed.
    - Grid purchase is limited to the amount of deployed down-regulation services
    - Uses pGridPurchase variable which represents the actual amount of power purchased from the grid
    """
    if not getattr(m, "SIMULATE_AS_DISPATCH_EXECUTION", False) or not getattr(
        m, "CAN_PROVIDE_ANCILLARY_SERVICES", False
    ):
        # No grid purchase allowed if not in dispatch mode
        return m.pGridPurchase[t] == 0

    # Down services that can be deployed
    down_services = ["RampDown"]  # "RegDown", "ECRS"

    # Calculate total down services deployed across all components
    total_down_deployed = 0.0
    components_with_deployed_vars = []

    if getattr(m, "ENABLE_ELECTROLYZER", False):
        components_with_deployed_vars.append("Electrolyzer")
    if getattr(m, "ENABLE_BATTERY", False):
        components_with_deployed_vars.append("Battery")

    # Sum all deployed down services
    for comp_name in components_with_deployed_vars:
        for service in down_services:
            deployed_var_name = f"{service}_{comp_name}_Deployed"
            if hasattr(m, deployed_var_name) and t in getattr(m, deployed_var_name):
                total_down_deployed += getattr(m, deployed_var_name)[t]

    # Grid purchase is directly limited to the amount of down services deployed
    return m.pGridPurchase[t] <= total_down_deployed


def h2_constant_sales_rate_rule(m, t):
    """Enforce constant hydrogen sales rate throughout optimization period."""
    if not getattr(m, "ENABLE_H2_STORAGE", False) or not getattr(m, "ENABLE_ELECTROLYZER", False):
        return pyo.Constraint.Skip
    if not hasattr(m, "H2_constant_sales_rate"):
        return pyo.Constraint.Skip

    # Skip the first time period to allow storage buildup
    if t == m.TimePeriods.first():
        return pyo.Constraint.Skip

    try:
        # Total hydrogen sold to market must equal constant rate (after first period)
        return m.H2_to_market[t] + m.H2_from_storage[t] == m.H2_constant_sales_rate
    except Exception as e:
        logger.error(
            f"H2 Constant Sales Rate Error @t={t}: {e}", exc_info=True)
        raise


def h2_storage_balance_constraint_rule(m, t):
    """Ensure hydrogen storage balance with constant sales rate constraint."""
    if not getattr(m, "ENABLE_H2_STORAGE", False):
        return pyo.Constraint.Skip
    try:
        discharge_eff = pyo.value(m.storage_discharge_eff)
        charge_eff = pyo.value(m.storage_charge_eff)

        # Account for efficiency losses
        discharge_term = (
            m.H2_from_storage[t] / discharge_eff if discharge_eff > 1e-9 else 0
        )
        charge_term = m.H2_to_storage[t] * charge_eff

        if t == m.TimePeriods.first():
            return (
                m.H2_storage_level[t]
                == m.H2_storage_level_initial + charge_term - discharge_term
            )
        else:
            return (
                m.H2_storage_level[t]
                == m.H2_storage_level[t - 1] + charge_term - discharge_term
            )
    except Exception as e:
        logger.error(
            f"H2 Storage Balance Constraint Error @t={t}: {e}", exc_info=True)
        raise


def h2_total_production_balance_rule(m):
    """Ensure total hydrogen production is sufficient to meet total sales over optimization period."""
    if not getattr(m, "ENABLE_H2_STORAGE", False) or not getattr(m, "ENABLE_ELECTROLYZER", False):
        return pyo.Constraint.Skip
    if not hasattr(m, "H2_constant_sales_rate"):
        return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0

        # Total production over the period
        total_production = sum(
            m.mHydrogenProduced[t] * time_factor for t in m.TimePeriods)

        # Total sales: variable sales in first period + constant rate for remaining periods
        first_period = m.TimePeriods.first()
        remaining_periods = len(m.TimePeriods) - 1

        first_period_sales = (
            m.H2_to_market[first_period] + m.H2_from_storage[first_period]) * time_factor
        constant_period_sales = m.H2_constant_sales_rate * \
            (remaining_periods * time_factor)
        total_sales = first_period_sales + constant_period_sales

        # Production must be sufficient to meet sales (allowing for excess storage)
        initial_storage = pyo.value(m.H2_storage_level_initial)
        final_storage = m.H2_storage_level[m.TimePeriods.last()]

        return total_production + initial_storage >= total_sales + final_storage
    except Exception as e:
        logger.error(f"H2 Total Production Balance Error: {e}", exc_info=True)
        raise


def h2_no_direct_sales_rule(m, t):
    """Ensure all hydrogen production goes through storage before sales when storage is enabled."""
    if not getattr(m, "ENABLE_H2_STORAGE", False) or not getattr(m, "ENABLE_ELECTROLYZER", False):
        return pyo.Constraint.Skip
    try:
        # When storage is enabled, no direct sales allowed - all H2 must go through storage
        return m.H2_to_market[t] == 0
    except Exception as e:
        logger.error(f"H2 No Direct Sales Error @t={t}: {e}", exc_info=True)
        raise
