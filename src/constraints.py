# src/constraints.py
import pyomo.environ as pyo
from logging_setup import logger
from config import (ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN,
                    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_H2_CAP_FACTOR,
                    ENABLE_LOW_TEMP_ELECTROLYZER)

# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def build_piecewise_constraints(model: pyo.ConcreteModel, *, component_prefix: str,
                                input_var_name: str, output_var_name: str,
                                breakpoint_set_name: str, value_param_name: str, n_segments=None) -> None:
    """Attach SOS2 piece‑wise linear constraints *in‑place* to `model`."""
    # ... (Keep the restored build_piecewise_constraints function) ...
    logger.info("Building piece‑wise constraints for %s using SOS2", component_prefix)
    input_var = getattr(model, input_var_name)
    output_var = getattr(model, output_var_name)
    breakpoint_set = getattr(model, breakpoint_set_name)
    value_param = getattr(model, value_param_name)
    if not breakpoint_set.ordered:
         logger.warning(f"Breakpoint set {breakpoint_set_name} for {component_prefix} is not ordered.")
         try:
             sorted_breakpoints = sorted(list(breakpoint_set))
             breakpoint_set = pyo.Set(initialize=sorted_breakpoints, ordered=True)
             setattr(model, breakpoint_set_name, breakpoint_set)
             logger.info(f"Replaced {breakpoint_set_name} with an ordered version.")
         except TypeError:
              logger.error(f"Cannot sort breakpoint set {breakpoint_set_name}.")
              raise ValueError(f"Breakpoint set {breakpoint_set_name} must be ordered.")
    lam_var_name = f"lambda_{component_prefix}"
    lam = pyo.Var(model.TimePeriods, breakpoint_set, bounds=(0, 1), within=pyo.NonNegativeReals)
    setattr(model, lam_var_name, lam)
    def _sum_rule(m, t): return sum(lam[t, bp] for bp in breakpoint_set) == 1
    model.add_component(f"{component_prefix}_sum_lambda", pyo.Constraint(model.TimePeriods, rule=_sum_rule))
    def _input_link(m, t): return input_var[t] == sum(lam[t, bp] * bp for bp in breakpoint_set)
    model.add_component(f"{component_prefix}_input_link", pyo.Constraint(model.TimePeriods, rule=_input_link))
    def _output_link(m, t): return output_var[t] == sum(lam[t, bp] * value_param[bp] for bp in breakpoint_set)
    model.add_component(f"{component_prefix}_output_link", pyo.Constraint(model.TimePeriods, rule=_output_link))
    def _sos2_rule(m, t): return [lam[t, bp] for bp in breakpoint_set]
    model.add_component(f"SOS2_{component_prefix}", pyo.SOSConstraint(model.TimePeriods, rule=_sos2_rule, sos=2))


# ---------------------------------------------------------------------------
# PHYSICAL BALANCE RULES
# ---------------------------------------------------------------------------
# (steam_balance_rule, power_balance_rule, constant_turbine_power_rule - unchanged)
def steam_balance_rule(m, t):
    try:
        if m.LTE_MODE: return m.qSteam_Turbine[t] == m.qSteam_Total
        else: return m.qSteam_Turbine[t] + m.qSteam_Electrolyzer[t] == m.qSteam_Total
    except Exception as e: logger.error(f"Error in steam_balance rule @t={t}: {e}"); raise

def power_balance_rule(m, t):
    try: return m.pIES[t] + m.pElectrolyzer[t] == m.pTurbine[t]
    except Exception as e: logger.error(f"Error in power_balance rule @t={t}: {e}"); raise

def constant_turbine_power_rule(m,t):
    if not m.LTE_MODE: return pyo.Constraint.Skip
    try: return m.pTurbine[t] == m.pTurbine_LTE_setpoint
    except Exception as e: logger.error(f"Error in constant_turbine_power rule @t={t}: {e}"); raise

# --- H2 Storage Rules (Unchanged, use actual mHydrogenProduced) ---
# ... (h2_storage_balance_rule, etc.) ...
def h2_storage_balance_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        charge_term = (m.mHydrogenProduced[t] - m.H2_to_market[t]) * m.storage_charge_eff
        discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
        return m.H2_storage_level[t] == m.H2_storage_level_initial + charge_term - discharge_term
    else:
        charge_term = (m.mHydrogenProduced[t] - m.H2_to_market[t]) * m.storage_charge_eff
        discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
        return m.H2_storage_level[t] == m.H2_storage_level[t-1] + charge_term - discharge_term

def h2_storage_charge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return (m.mHydrogenProduced[t] - m.H2_to_market[t]) <= m.H2_storage_charge_rate_max

def h2_storage_discharge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max

def h2_storage_level_max_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] <= m.H2_storage_capacity_max

def h2_storage_level_min_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] >= m.H2_storage_capacity_min

def h2_direct_market_link_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_to_market[t] <= m.mHydrogenProduced[t]


# --- Ramp Rate Rules (Apply to ACTUAL pElectrolyzer) ---
# (Electrolyzer_RampUp/Down_rule, Turbine_RampUp/Down_rule, Steam_Electrolyzer_Ramp_rule - unchanged)
def Electrolyzer_RampUp_rule(m, t):
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t):
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t-1] - m.pElectrolyzer[t] <= m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampDown rule @t={t}: {e}"); raise

def Turbine_RampUp_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t] - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampUp rule @t={t}: {e}"); raise

def Turbine_RampDown_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t-1] - m.pTurbine[t] <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampDown rule @t={t}: {e}"); raise

def Steam_Electrolyzer_Ramp_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        if hasattr(m, 'qSteamElectrolyzerRampPos') and hasattr(m, 'qSteamElectrolyzerRampNeg'):
             return m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t] <= m.Ramp_qSteam_Electrolyzer_limit
        else: return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in Steam_Electrolyzer_Ramp rule @t={t}: {e}"); raise


# --- Production Requirement Rule (Uses actual mHydrogenProduced) ---
def h2_CapacityFactor_rule(m):
    # (Keep existing logic)
    if not ENABLE_H2_CAP_FACTOR: return pyo.Constraint.Skip
    try:
        total_hours = len(m.TimePeriods)
        max_elec_power_ub = pyo.value(m.pElectrolyzer_max.ub)
        if max_elec_power_ub <= 1e-6: return pyo.Constraint.Skip
        if not hasattr(m, 'pElectrolyzer_efficiency_breakpoints') or not m.pElectrolyzer_efficiency_breakpoints: return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last()
        eff_at_max_bp = pyo.value(m.ke_H2_values.get(max_power_bp, None))
        if eff_at_max_bp is None or eff_at_max_bp <= 1e-9: return pyo.Constraint.Skip
        max_h2_rate_kg_per_hr_est = max_elec_power_ub / eff_at_max_bp
        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours
        if max_potential_h2_kg_total_est <= 1e-6: return pyo.Constraint.Skip
        return sum(m.mHydrogenProduced[t] for t in m.TimePeriods) >= m.h2_target_capacity_factor * max_potential_h2_kg_total_est
    except Exception as e: logger.error(f"Error in h2_CapacityFactor rule: {e}"); raise


# --- Startup/Shutdown Constraints (Apply to ACTUAL pElectrolyzer) ---
# (Keep existing logic for on_off, min/max power, exclusivity, min uptime/downtime)
def electrolyzer_on_off_logic_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        return m.uElectrolyzer[t] - m.uElectrolyzer_initial == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
    else:
        return m.uElectrolyzer[t] - m.uElectrolyzer[t-1] == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]

def electrolyzer_min_power_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.pElectrolyzer[t] >= m.uElectrolyzer[t] * m.pElectrolyzer_min

def electrolyzer_max_power_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.pElectrolyzer[t] <= m.uElectrolyzer[t] * m.pElectrolyzer_max

def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1

def electrolyzer_min_uptime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    min_uptime = m.MinUpTimeElectrolyzer
    if t < min_uptime: return pyo.Constraint.Skip
    expr = sum(m.uElectrolyzer[i] for i in range(t - min_uptime + 1, t + 1)) >= min_uptime * m.vElectrolyzerStartup[t - min_uptime + 1]
    return expr

def electrolyzer_min_downtime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    min_downtime = m.MinDownTimeElectrolyzer
    if t < min_downtime: return pyo.Constraint.Skip
    expr = sum(m.uElectrolyzer[i] for i in range(t - min_downtime + 1, t + 1)) <= min_downtime * (1 - m.wElectrolyzerShutdown[t - min_downtime + 1])
    return expr


# --- Electrolyzer Degradation Tracking Rule (Uses ACTUAL pElectrolyzer) ---
def electrolyzer_degradation_rule(m, t):
    if not ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: return pyo.Constraint.Skip
    try:
        # Linear Approximation using upper bound of capacity variable
        relative_load_expr = 0 # Initialize as 0 or an expression evaluating to 0
        # Use pyo.value() only on parameters/fixed variables during construction
        max_cap_ub = pyo.value(m.pElectrolyzer_max.ub)

        if max_cap_ub > 1e-6:
             # Keep relative_load as a symbolic expression
            relative_load_expr = m.pElectrolyzer[t] / max_cap_ub
        # else: relative_load_expr remains 0 (or pyo.Expression(expr=0))

        # Calculate degradation increase symbolically
        # Note: m.uElectrolyzer[t] is a binary variable, multiplication is okay
        degradation_increase = m.uElectrolyzer[t] * m.DegradationFactorOperation * relative_load_expr

        if ENABLE_STARTUP_SHUTDOWN:
             # m.vElectrolyzerStartup[t] is also binary
             degradation_increase += m.vElectrolyzerStartup[t] * m.DegradationFactorStartup

        # Return symbolic constraint expression
        if t == m.TimePeriods.first():
            # m.DegradationStateInitial is a Param, okay to use directly
            return m.DegradationState[t] == m.DegradationStateInitial + degradation_increase
        else:
            # m.DegradationState[t-1] is a variable, use symbolically
            return m.DegradationState[t] == m.DegradationState[t-1] + degradation_increase
    except Exception as e:
        logger.error(f"Error defining electrolyzer_degradation rule @t={t}: {e}")
        # Return a trivial constraint or skip if definition fails
        return pyo.Constraint.Skip


# --- Ancillary Service Definitions Helper ---
# (Keep existing get_as_components helper function)
def get_as_components(m, t):
    bids = {'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0, 'up_deployed': 0.0, 'down_deployed': 0.0, 'iso_services': {}}
    iso = m.TARGET_ISO
    service_details = {
        'SPP': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'Sup': 'E'},
        'CAISO': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'NSpin': 'E', 'RMU': 'R', 'RMD': 'R'},
        'ERCOT': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'NSpin': 'E', 'ECRS': 'E'},
        'PJM': {'Reg': 'R', 'Syn': 'E', 'Rse': 'E', 'TMR': 'E'},
        'NYISO': {'RegC': 'R', 'Spin10': 'E', 'NSpin10': 'E', 'Res30': 'E'},
        'ISONE': {'Spin10': 'E', 'NSpin10': 'E', 'OR30': 'E'},
        'MISO': {'Reg': 'R', 'Spin': 'E', 'Sup': 'E', 'STR': 'E'}
    }
    if iso not in service_details: return bids

    for service, type in service_details[iso].items():
        up_bid_e = 0.0; down_bid_e = 0.0
        if service in ['RegU', 'Reg', 'RegC'] and hasattr(m, 'RegUp_Electrolyzer'): up_bid_e = m.RegUp_Electrolyzer[t]
        elif service in ['RegD'] and hasattr(m, 'RegDown_Electrolyzer'): down_bid_e = m.RegDown_Electrolyzer[t]
        elif service in ['Spin', 'Syn', 'Spin10'] and hasattr(m, 'SR_Electrolyzer'): up_bid_e = m.SR_Electrolyzer[t]
        elif service in ['NSpin', 'Sup', 'Rse', 'NSpin10'] and hasattr(m, 'NSR_Electrolyzer'): up_bid_e = m.NSR_Electrolyzer[t]
        elif service == 'ECRS' and hasattr(m, 'ECRS_Electrolyzer'): up_bid_e = m.ECRS_Electrolyzer[t]
        elif service in ['TMR', 'Res30', 'OR30', 'STR'] and hasattr(m, 'ThirtyMin_Electrolyzer'): up_bid_e = m.ThirtyMin_Electrolyzer[t]
        if iso == 'PJM' and service == 'Reg':
             if hasattr(m, 'RegUp_Electrolyzer'): up_bid_e = m.RegUp_Electrolyzer[t]
             if hasattr(m, 'RegDown_Electrolyzer'): down_bid_e = m.RegDown_Electrolyzer[t]

        electrolyzer_bid = up_bid_e + down_bid_e
        bids['iso_services'][service] = electrolyzer_bid
        deploy_factor_param = f'deploy_factor_{service}_{iso}'; deploy_factor = 0.0
        if hasattr(m, deploy_factor_param):
            param_obj = getattr(m, deploy_factor_param)
            if t in param_obj: deploy_factor = pyo.value(param_obj[t], exception=False) or 0.0
            else: deploy_factor = pyo.value(param_obj, exception=False) or 0.0

        up_bid_e_val = pyo.value(up_bid_e); down_bid_e_val = pyo.value(down_bid_e)
        if service in ['RegU', 'Spin', 'NSpin', 'ECRS', 'TMR', 'Syn', 'Rse', 'Spin10', 'NSpin10', 'Res30', 'OR30', 'STR'] or (iso == 'PJM' and service == 'Reg'):
            bids['up_reserves_bid'] += up_bid_e_val
            if deploy_factor > 1e-6: bids['up_deployed'] += up_bid_e_val * deploy_factor
        if service in ['RegD'] or (iso == 'PJM' and service == 'Reg'):
            bids['down_reserves_bid'] += down_bid_e_val
            if deploy_factor > 1e-6: bids['down_deployed'] += down_bid_e_val * deploy_factor
    return bids


# --- Electrolyzer Setpoint Linking Rule ---
def Electrolyzer_Setpoint_Link_rule(m, t):
    # (Keep existing logic)
    try:
        as_info = get_as_components(m, t)
        return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t] - as_info['up_deployed'] + as_info['down_deployed']
    except Exception as e: logger.error(f"Error in Electrolyzer_Setpoint_Link rule @t={t}: {e}"); raise


# --- Ancillary Service Provision Capability Rules (Based on Setpoint) ---
# (Turbine_AS_Zero_rule, Turbine_AS_Pmax/min/RU/RD_rule - unchanged)
def Turbine_AS_Zero_rule(m, t):
    if not m.LTE_MODE: return pyo.Constraint.Skip
    try:
        zero_as = 0
        if hasattr(m, 'RegUp_Turbine'): zero_as += m.RegUp_Turbine[t]
        if hasattr(m, 'RegDown_Turbine'): zero_as += m.RegDown_Turbine[t]
        if hasattr(m, 'SR_Turbine'): zero_as += m.SR_Turbine[t]
        if hasattr(m, 'NSR_Turbine'): zero_as += m.NSR_Turbine[t]
        if hasattr(m, 'ECRS_Turbine'): zero_as += m.ECRS_Turbine[t]
        if hasattr(m, 'ThirtyMin_Turbine'): zero_as += m.ThirtyMin_Turbine[t]
        return zero_as == 0
    except Exception as e: logger.error(f"Error in Turbine_AS_Zero rule @t={t}: {e}"); raise

def Turbine_AS_Pmax_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Turbine'): up_reserve += m.RegUp_Turbine[t]
        if hasattr(m, 'SR_Turbine'): up_reserve += m.SR_Turbine[t]
        if hasattr(m, 'NSR_Turbine'): up_reserve += m.NSR_Turbine[t]
        if hasattr(m, 'ECRS_Turbine'): up_reserve += m.ECRS_Turbine[t]
        if hasattr(m, 'ThirtyMin_Turbine'): up_reserve += m.ThirtyMin_Turbine[t]
        return m.pTurbine[t] + up_reserve <= m.pTurbine_max
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Turbine'): down_reserve += m.RegDown_Turbine[t]
        return m.pTurbine[t] - down_reserve >= m.pTurbine_min
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmin rule @t={t}: {e}"); raise

def Turbine_AS_RU_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Turbine'): up_reserve += m.RegUp_Turbine[t]
        if hasattr(m, 'SR_Turbine'): up_reserve += m.SR_Turbine[t]
        if hasattr(m, 'NSR_Turbine'): up_reserve += m.NSR_Turbine[t]
        if hasattr(m, 'ECRS_Turbine'): up_reserve += m.ECRS_Turbine[t]
        if hasattr(m, 'ThirtyMin_Turbine'): up_reserve += m.ThirtyMin_Turbine[t]
        return (m.pTurbine[t] + up_reserve) - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RU rule @t={t}: {e}"); raise

def Turbine_AS_RD_rule(m, t):
    if m.LTE_MODE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Turbine'): down_reserve += m.RegDown_Turbine[t]
        return m.pTurbine[t-1] - (m.pTurbine[t] - down_reserve) <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise


# (Electrolyzer_AS_Pmax/Pmin_rule - unchanged, based on Setpoint and Bid Capability)
def Electrolyzer_AS_Pmax_rule(m, t): # Capability to provide full DOWN reserve bid
    try:
        as_info = get_as_components(m, t)
        total_down_bid_capability = as_info['down_reserves_bid']
        if not ENABLE_STARTUP_SHUTDOWN:
            return m.pElectrolyzerSetpoint[t] + total_down_bid_capability <= m.pElectrolyzer_max
        else:
            return m.pElectrolyzerSetpoint[t] + total_down_bid_capability <= m.uElectrolyzer[t] * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Capability to provide full UP reserve bid
    try:
        as_info = get_as_components(m, t)
        total_up_bid_capability = as_info['up_reserves_bid']
        if not ENABLE_STARTUP_SHUTDOWN:
             return m.pElectrolyzerSetpoint[t] - total_up_bid_capability >= m.pElectrolyzer_min
        else:
            # Ensure setpoint itself is feasible when on
            setpoint_feasibility = m.pElectrolyzerSetpoint[t] >= m.uElectrolyzer[t] * m.pElectrolyzer_min
            reserve_feasibility = m.pElectrolyzerSetpoint[t] - total_up_bid_capability >= m.uElectrolyzer[t] * m.pElectrolyzer_min
            # Combine or return the stricter one (reserve_feasibility implies setpoint_feasibility if total_up_bid >=0)
            return reserve_feasibility
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise


# (Electrolyzer_AS_RU/RD_rule - unchanged, check ramp capability relative to setpoint extremes)
def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for DOWN-reg
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_capability = as_info['down_reserves_bid']
        ramp_limit_mw = m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        return (m.pElectrolyzerSetpoint[t] + total_down_bid_capability) - m.pElectrolyzer[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for UP-reg
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_capability = as_info['up_reserves_bid']
        ramp_limit_mw = m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        return m.pElectrolyzer[t-1] - (m.pElectrolyzerSetpoint[t] - total_up_bid_capability) <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise


# --- Link Component AS to Total System AS Rules ---
# (Modified for LTE mode)
def link_Total_RegUp_rule(m, t):
    try:
        regup_h2 = m.RegUp_Electrolyzer[t] if hasattr(m, 'RegUp_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_RegUp[t] == regup_h2
        else:
            regup_turbine = m.RegUp_Turbine[t] if hasattr(m, 'RegUp_Turbine') else 0
            return m.Total_RegUp[t] == regup_turbine + regup_h2
    except Exception as e: logger.error(f"Error in link_Total_RegUp rule @t={t}: {e}"); raise

def link_Total_RegDown_rule(m, t):
    try:
        regdown_h2 = m.RegDown_Electrolyzer[t] if hasattr(m, 'RegDown_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_RegDown[t] == regdown_h2
        else:
            regdown_turbine = m.RegDown_Turbine[t] if hasattr(m, 'RegDown_Turbine') else 0
            return m.Total_RegDown[t] == regdown_turbine + regdown_h2
    except Exception as e: logger.error(f"Error in link_Total_RegDown rule @t={t}: {e}"); raise

def link_Total_SR_rule(m, t):
    try:
        sr_h2 = m.SR_Electrolyzer[t] if hasattr(m, 'SR_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_SR[t] == sr_h2
        else:
            sr_turbine = m.SR_Turbine[t] if hasattr(m, 'SR_Turbine') else 0
            return m.Total_SR[t] == sr_turbine + sr_h2
    except Exception as e: logger.error(f"Error in link_Total_SR rule @t={t}: {e}"); raise

def link_Total_NSR_rule(m, t):
    try:
        nsr_h2 = m.NSR_Electrolyzer[t] if hasattr(m, 'NSR_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_NSR[t] == nsr_h2
        else:
            nsr_turbine = m.NSR_Turbine[t] if hasattr(m, 'NSR_Turbine') else 0
            return m.Total_NSR[t] == nsr_turbine + nsr_h2
    except Exception as e: logger.error(f"Error in link_Total_NSR rule @t={t}: {e}"); raise

def link_Total_ECRS_rule(m, t):
    try:
        if not isinstance(m.Total_ECRS, pyo.Var): return pyo.Constraint.Skip
        ecrs_h2 = m.ECRS_Electrolyzer[t] if hasattr(m, 'ECRS_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_ECRS[t] == ecrs_h2
        else:
            ecrs_turbine = m.ECRS_Turbine[t] if hasattr(m, 'ECRS_Turbine') else 0
            return m.Total_ECRS[t] == ecrs_turbine + ecrs_h2
    except Exception as e: logger.error(f"Error in link_Total_ECRS rule @t={t}: {e}"); raise

def link_Total_30Min_rule(m, t):
    try:
        if not isinstance(m.Total_30Min, pyo.Var): return pyo.Constraint.Skip
        thirtymin_h2 = m.ThirtyMin_Electrolyzer[t] if hasattr(m, 'ThirtyMin_Electrolyzer') else 0
        if m.LTE_MODE: return m.Total_30Min[t] == thirtymin_h2
        else:
            thirtymin_turbine = m.ThirtyMin_Turbine[t] if hasattr(m, 'ThirtyMin_Turbine') else 0
            return m.Total_30Min[t] == thirtymin_turbine + thirtymin_h2
    except Exception as e: logger.error(f"Error in link_Total_30Min rule @t={t}: {e}"); raise

