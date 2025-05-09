# src/constraints.py
import pyomo.environ as pyo
from logging_setup import logger
from config import ( # Import flags needed for conditional constraints
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_H2_CAP_FACTOR,
    ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY, ENABLE_ELECTROLYZER,
    ENABLE_NUCLEAR_GENERATOR, CAN_PROVIDE_ANCILLARY_SERVICES
)
# --- IMPORT NEW HELPERS from utils.py ---
from utils import (
    get_symbolic_as_bid_sum,
    get_symbolic_as_deployed_sum
)

# ---------------------------------------------------------------------------
# GENERIC HELPERS (Piecewise - kept local as it's complex and specific to constraint setup)
# ---------------------------------------------------------------------------
def build_piecewise_constraints(model: pyo.ConcreteModel, *, component_prefix: str,
                                input_var_name: str, output_var_name: str,
                                breakpoint_set_name: str, value_param_name: str) -> None:
    """Attach SOS2 piece‑wise linear constraints *in‑place* to `model`."""
    # (Keep the implementation of build_piecewise_constraints as provided before)
    logger.info("Building piece‑wise constraints for %s using SOS2 (constraints.py local helper)", component_prefix)
    # ... (rest of build_piecewise_constraints implementation remains the same) ...
    if not hasattr(model, input_var_name): logger.error(f"Input variable '{input_var_name}' not found for PWL {component_prefix}."); return
    if not hasattr(model, output_var_name): logger.error(f"Output variable '{output_var_name}' not found for PWL {component_prefix}."); return
    if not hasattr(model, breakpoint_set_name): logger.error(f"Breakpoint set '{breakpoint_set_name}' not found for PWL {component_prefix}."); return

    input_var = getattr(model, input_var_name)
    output_var = getattr(model, output_var_name)
    breakpoint_set_orig = getattr(model, breakpoint_set_name)

    value_data_source = None
    if isinstance(value_param_name, str):
        if hasattr(model, value_param_name): value_data_source = getattr(model, value_param_name)
        else: logger.error(f"Value parameter/dict named '{value_param_name}' not found for PWL {component_prefix}."); return
    elif isinstance(value_param_name, (dict, pyo.Param)): # Allow passing dict or Param directly
        value_data_source = value_param_name
    else:
        logger.error(f"Value source '{value_param_name}' for PWL {component_prefix} is not a string name, dict, or Pyomo Param."); return


    if not (isinstance(value_data_source, pyo.Param) or isinstance(value_data_source, dict)):
        logger.error(f"Value source for '{component_prefix}' (from '{value_param_name}') is not a Pyomo Param or dict."); return

    breakpoint_set_to_use = breakpoint_set_orig
    if not breakpoint_set_orig.isordered():
         logger.warning(f"Breakpoint set {breakpoint_set_name} for {component_prefix} is not ordered. Attempting to sort and replace.")
         try:
             sorted_breakpoints_values = sorted(list(pyo.value(bp) for bp in breakpoint_set_orig))
             ordered_set_attr_name = f"_ordered_{breakpoint_set_name}_{component_prefix}"
             if hasattr(model, ordered_set_attr_name):
                 breakpoint_set_to_use = getattr(model, ordered_set_attr_name)
                 logger.info(f"Using existing dynamically created ordered breakpoint set: {ordered_set_attr_name}")
             else:
                 new_ordered_set = pyo.Set(initialize=sorted_breakpoints_values, ordered=True, name=ordered_set_attr_name)
                 setattr(model, ordered_set_attr_name, new_ordered_set)
                 breakpoint_set_to_use = new_ordered_set
                 logger.info(f"Created and using new ordered breakpoint set: {ordered_set_attr_name} for {component_prefix}.")
         except Exception as e:
              logger.error(f"Cannot sort or replace breakpoint set {breakpoint_set_name} for {component_prefix}: {e}")
              raise ValueError(f"Breakpoint set {breakpoint_set_name} must be ordered for SOS2. Automatic sorting failed.")

    lam_var_name = f"lambda_{component_prefix}"
    if not hasattr(model, lam_var_name):
        lam = pyo.Var(model.TimePeriods, breakpoint_set_to_use, bounds=(0, 1), within=pyo.NonNegativeReals)
        setattr(model, lam_var_name, lam)
    else: lam = getattr(model, lam_var_name)

    sum_lambda_constr_name = f"{component_prefix}_sum_lambda"
    input_link_constr_name = f"{component_prefix}_input_link"
    output_link_constr_name = f"{component_prefix}_output_link"
    sos2_constr_name = f"SOS2_{component_prefix}"

    if not hasattr(model, sum_lambda_constr_name):
        def _sum_rule(m, t): return sum(lam[t, bp] for bp in breakpoint_set_to_use) == 1
        model.add_component(sum_lambda_constr_name, pyo.Constraint(model.TimePeriods, rule=_sum_rule))

    if not hasattr(model, input_link_constr_name):
        def _input_link(m, t): return input_var[t] == sum(lam[t, bp] * bp for bp in breakpoint_set_to_use)
        model.add_component(input_link_constr_name, pyo.Constraint(model.TimePeriods, rule=_input_link))

    if not hasattr(model, output_link_constr_name):
        def _output_link(m, t):
            if isinstance(value_data_source, pyo.Param):
                return output_var[t] == sum(lam[t, bp] * value_data_source[bp] for bp in breakpoint_set_to_use)
            elif isinstance(value_data_source, dict): # Check if it's a dict (e.g., ke_H2_inv_values)
                return output_var[t] == sum(lam[t, bp] * value_data_source.get(bp, 0.0) for bp in breakpoint_set_to_use)
            logger.critical(f"Value source for {component_prefix} output link is invalid type in _output_link. This is unexpected.")
            return pyo.Constraint.Skip
        model.add_component(output_link_constr_name, pyo.Constraint(model.TimePeriods, rule=_output_link))

    if not hasattr(model, sos2_constr_name):
        def _sos2_rule(m, t): return [lam[t, bp] for bp in breakpoint_set_to_use]
        model.add_component(sos2_constr_name, pyo.SOSConstraint(model.TimePeriods, rule=_sos2_rule, sos=2))


# ---------------------------------------------------------------------------
# PHYSICAL BALANCE RULES (No changes needed here)
# ---------------------------------------------------------------------------
def steam_balance_rule(m, t):
    """Links total steam production to turbine and HTE electrolyzer use."""
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not enable_npp: return pyo.Constraint.Skip
    try:
        turbine_steam = m.qSteam_Turbine[t] if hasattr(m, 'qSteam_Turbine') else 0
        hte_steam = m.qSteam_Electrolyzer[t] if enable_electrolyzer and not lte_mode and hasattr(m, 'qSteam_Electrolyzer') else 0
        total_steam_available = m.qSteam_Total
        return turbine_steam + hte_steam == total_steam_available
    except Exception as e: logger.error(f"Error in steam_balance rule @t={t}: {e}"); raise

def power_balance_rule(m, t):
    """Ensures power generation equals consumption + net grid interaction."""
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    try:
        turbine_power = m.pTurbine[t] if enable_npp and hasattr(m, 'pTurbine') else 0
        battery_discharge = m.BatteryDischarge[t] if enable_battery and hasattr(m, 'BatteryDischarge') else 0
        electrolyzer_power = m.pElectrolyzer[t] if enable_electrolyzer and hasattr(m, 'pElectrolyzer') else 0 # Actual power
        battery_charge = m.BatteryCharge[t] if enable_battery and hasattr(m, 'BatteryCharge') else 0
        auxiliary_power = m.pAuxiliary[t] if hasattr(m, 'pAuxiliary') else 0
        return turbine_power + battery_discharge - electrolyzer_power - battery_charge - auxiliary_power == m.pIES[t]
    except Exception as e: logger.error(f"Error in power_balance rule @t={t}: {e}"); raise

def constant_turbine_power_rule(m,t):
    """Fixes turbine power if LTE mode is active."""
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (enable_npp and enable_electrolyzer and lte_mode): return pyo.Constraint.Skip
    try:
        if hasattr(m, 'pTurbine') and hasattr(m, 'pTurbine_LTE_setpoint'):
            return m.pTurbine[t] == m.pTurbine_LTE_setpoint
        else: logger.warning(f"Skipping constant_turbine_power rule @t={t}: Missing components."); return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in constant_turbine_power rule @t={t}: {e}"); raise

def link_auxiliary_power_rule(m, t):
    """Links auxiliary power consumption to hydrogen production rate."""
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not hasattr(m, 'pAuxiliary') or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        if hasattr(m, 'mHydrogenProduced') and hasattr(m, 'aux_power_consumption_per_kg_h2'):
            aux_rate = pyo.value(m.aux_power_consumption_per_kg_h2) # MW_aux per kg/hr H2
            # mHydrogenProduced is kg/hr, aux_rate is MW/(kg/hr) -> result is MW
            return m.pAuxiliary[t] == m.mHydrogenProduced[t] * aux_rate
        else: logger.warning(f"Skipping link_auxiliary_power rule @t={t}: Missing components."); return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in link_auxiliary_power rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# NEW RULE: Link Setpoint to Actual Power if NOT Simulating Dispatch
# ---------------------------------------------------------------------------
def link_setpoint_to_actual_power_if_not_simulating_dispatch_rule(m, t):
    """If not simulating dispatch, actual power equals setpoint."""
    simulate_dispatch = getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer or simulate_dispatch: return pyo.Constraint.Skip
    try:
        if hasattr(m, 'pElectrolyzer') and hasattr(m, 'pElectrolyzerSetpoint'):
            return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t]
        return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in link_setpoint_to_actual_power rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# NEW RULE: Electrolyzer Setpoint Minimum Power
# ---------------------------------------------------------------------------
def electrolyzer_setpoint_min_power_rule(m, t):
    """Ensures the electrolyzer setpoint respects minimum turn-down if operational."""
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer or not hasattr(m, 'pElectrolyzerSetpoint'): return pyo.Constraint.Skip
    try:
        min_power_param = getattr(m, 'pElectrolyzer_min', None)
        if min_power_param is None: logger.warning(f"Skipping setpoint_min_power @t={t}: pElectrolyzer_min not found."); return pyo.Constraint.Skip

        enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
        if enable_startup_shutdown and hasattr(m, 'uElectrolyzer'):
            return m.pElectrolyzerSetpoint[t] >= min_power_param * m.uElectrolyzer[t]
        else: # If not using on/off logic, assume setpoint must always be above min
            return m.pElectrolyzerSetpoint[t] >= min_power_param
    except Exception as e: logger.error(f"Error in electrolyzer_setpoint_min_power rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# H2 STORAGE RULES (No changes needed here)
# ---------------------------------------------------------------------------
# ... (h2_storage_balance_adj_rule, h2_prod_dispatch_rule, etc. remain the same) ...
def h2_storage_balance_adj_rule(m, t):
     enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
     if not enable_h2_storage: return pyo.Constraint.Skip
     try:
         discharge_eff = pyo.value(m.storage_discharge_eff) # Efficiency is param
         charge_eff = pyo.value(m.storage_charge_eff)     # Efficiency is param
         discharge_term = (m.H2_from_storage[t] / discharge_eff if discharge_eff > 1e-9 else 0)
         charge_term = m.H2_to_storage[t] * charge_eff

         if t == m.TimePeriods.first():
             return m.H2_storage_level[t] == m.H2_storage_level_initial + charge_term - discharge_term
         else:
             return m.H2_storage_level[t] == m.H2_storage_level[t-1] + charge_term - discharge_term
     except Exception as e: logger.error(f"Error in h2_storage_balance rule @t={t}: {e}"); raise

def h2_prod_dispatch_rule(m, t):
     enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
     if not enable_h2_storage: return pyo.Constraint.Skip
     try:
         return m.mHydrogenProduced[t] == m.H2_to_market[t] + m.H2_to_storage[t]
     except Exception as e: logger.error(f"Error in h2_prod_dispatch rule @t={t}: {e}"); raise

def h2_storage_charge_limit_rule(m, t):
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_h2_storage: return pyo.Constraint.Skip
    try: return m.H2_to_storage[t] <= m.H2_storage_charge_rate_max
    except Exception as e: logger.error(f"Error in h2_charge_limit rule @t={t}: {e}"); raise

def h2_storage_discharge_limit_rule(m, t):
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_h2_storage: return pyo.Constraint.Skip
    try: return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max
    except Exception as e: logger.error(f"Error in h2_discharge_limit rule @t={t}: {e}"); raise

def h2_storage_level_max_rule(m, t):
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_h2_storage: return pyo.Constraint.Skip
    try: return m.H2_storage_level[t] <= m.H2_storage_capacity_max
    except Exception as e: logger.error(f"Error in h2_level_max rule @t={t}: {e}"); raise

def h2_storage_level_min_rule(m, t):
    enable_h2_storage = getattr(m, 'ENABLE_H2_STORAGE', False)
    if not enable_h2_storage: return pyo.Constraint.Skip
    try: return m.H2_storage_level[t] >= m.H2_storage_capacity_min
    except Exception as e: logger.error(f"Error in h2_level_min rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# RAMP RATE RULES (No changes needed here, they apply to actual power m.pElectrolyzer)
# ---------------------------------------------------------------------------
# ... (Electrolyzer_RampUp_rule, Electrolyzer_RampDown_rule, etc. remain the same) ...
def Electrolyzer_RampUp_rule(m, t):
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Applies to actual power m.pElectrolyzer
        return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
    except Exception as e: logger.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t):
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Applies to actual power m.pElectrolyzer
        return m.pElectrolyzer[t-1] - m.pElectrolyzer[t] <= m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
    except Exception as e: logger.error(f"Error in Electrolyzer_RampDown rule @t={t}: {e}"); raise

def Turbine_RampUp_rule(m, t):
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not enable_npp: return pyo.Constraint.Skip
    if lte_mode and enable_electrolyzer: return pyo.Constraint.Skip # Skip if turbine fixed
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return m.pTurbine[t] - m.pTurbine[t-1] <= m.RU_Turbine_hourly * time_factor
    except Exception as e: logger.error(f"Error in Turbine_RampUp rule @t={t}: {e}"); raise

def Turbine_RampDown_rule(m, t):
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not enable_npp: return pyo.Constraint.Skip
    if lte_mode and enable_electrolyzer: return pyo.Constraint.Skip # Skip if turbine fixed
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return m.pTurbine[t-1] - m.pTurbine[t] <= m.RD_Turbine_hourly * time_factor
    except Exception as e: logger.error(f"Error in Turbine_RampDown rule @t={t}: {e}"); raise

def Steam_Electrolyzer_Ramp_rule(m, t):
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (enable_electrolyzer and not lte_mode): return pyo.Constraint.Skip # HTE only
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        # Check if ramp variables exist (implies limit is finite and costed/constrained)
        if hasattr(m, 'qSteamElectrolyzerRampPos') and hasattr(m, 'qSteamElectrolyzerRampNeg'):
             time_factor = pyo.value(m.delT_minutes) / 60.0
             return m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t] <= m.Ramp_qSteam_Electrolyzer_limit * time_factor
        else:
             return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in Steam_Electrolyzer_Ramp rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# PRODUCTION REQUIREMENT RULE (h2_CapacityFactor_rule) (No changes needed here)
# ---------------------------------------------------------------------------
# ... (h2_CapacityFactor_rule remains the same) ...
def h2_CapacityFactor_rule(m):
    enable_h2_cap_factor = getattr(m, 'ENABLE_H2_CAP_FACTOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_h2_cap_factor or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        total_hours_sim = len(m.TimePeriods) * (pyo.value(m.delT_minutes) / 60.0)

        p_elec_max_comp = getattr(m, 'pElectrolyzer_max', None)
        if p_elec_max_comp is None: return pyo.Constraint.Skip

        max_elec_power_limit = pyo.value(m.pElectrolyzer_max_upper_bound)
        if max_elec_power_limit <= 1e-6: return pyo.Constraint.Skip

        if not hasattr(m, 'pElectrolyzer_efficiency_breakpoints') or not list(m.pElectrolyzer_efficiency_breakpoints):
            logger.warning("H2 Cap Factor: Missing or empty electrolyzer efficiency breakpoints.")
            return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last()

        if not hasattr(m, 'ke_H2_inv_values') or max_power_bp not in m.ke_H2_inv_values:
             logger.warning("H2 Cap Factor: Missing inverse efficiency values (ke_H2_inv_values).")
             return pyo.Constraint.Skip
        max_h2_rate_kg_per_mwh = pyo.value(m.ke_H2_inv_values[max_power_bp])
        if max_h2_rate_kg_per_mwh < 1e-9:
             logger.warning("H2 Cap Factor: Near-zero efficiency at max breakpoint.")
             return pyo.Constraint.Skip

        max_h2_rate_kg_per_hr_est = max_elec_power_limit * max_h2_rate_kg_per_mwh
        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours_sim
        if max_potential_h2_kg_total_est <= 1e-6: return pyo.Constraint.Skip

        total_actual_production_kg = sum(m.mHydrogenProduced[t] * (pyo.value(m.delT_minutes) / 60.0) for t in m.TimePeriods)
        target_production_kg = m.h2_target_capacity_factor * max_potential_h2_kg_total_est

        return total_actual_production_kg >= target_production_kg
    except Exception as e: logger.error(f"Error in h2_CapacityFactor rule: {e}"); raise

# ---------------------------------------------------------------------------
# STARTUP/SHUTDOWN CONSTRAINTS (No changes needed here)
# ---------------------------------------------------------------------------
# ... (electrolyzer_on_off_logic_rule, etc. remain the same, including electrolyzer_min_power_sds_disabled_rule adjustment) ...
def electrolyzer_on_off_logic_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        if t == m.TimePeriods.first():
            return m.uElectrolyzer[t] - m.uElectrolyzer_initial == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
        else:
            return m.uElectrolyzer[t] - m.uElectrolyzer[t-1] == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
    except Exception as e: logger.error(f"Error in SU/SD logic rule @t={t}: {e}"); raise

def electrolyzer_min_power_when_on_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    # This rule applies to actual physical power m.pElectrolyzer
    try: return m.pElectrolyzer[t] >= m.pElectrolyzer_min * m.uElectrolyzer[t]
    except Exception as e: logger.error(f"Error in SU/SD min power rule @t={t}: {e}"); raise

def electrolyzer_max_power_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    # This rule applies to actual physical power m.pElectrolyzer
    try: return m.pElectrolyzer[t] <= m.pElectrolyzer_max * m.uElectrolyzer[t]
    except Exception as e: logger.error(f"Error in SU/SD max power rule @t={t}: {e}"); raise

def electrolyzer_min_power_sds_disabled_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if enable_startup_shutdown or not enable_electrolyzer: # Skip if SDS is enabled or no electrolyzer
        return pyo.Constraint.Skip
    # If SDS is disabled, electrolyzer should still respect its min power if > 0
    # This applies to actual physical power m.pElectrolyzer
    try:
        if pyo.value(m.pElectrolyzer_min) > 1e-6 : # Only apply if min power is meaningfully positive
             return m.pElectrolyzer[t] >= m.pElectrolyzer_min
        return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in min_power_sds_disabled rule @t={t}: {e}"); raise


def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    try: return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1
    except Exception as e: logger.error(f"Error in SU/SD exclusivity rule @t={t}: {e}"); raise

def electrolyzer_min_uptime_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        min_uptime = pyo.value(m.MinUpTimeElectrolyzer)
        if t < min_uptime: return pyo.Constraint.Skip
        start_idx = max(m.TimePeriods.first(), t - min_uptime + 1)
        if not all(i in m.TimePeriods for i in range(start_idx, t + 1)):
             logger.warning(f"Min uptime constraint skipped at t={t} due to index range issues relative to TimePeriods.")
             return pyo.Constraint.Skip
        return sum(m.uElectrolyzer[i] for i in range(start_idx, t + 1)) >= min_uptime * m.vElectrolyzerStartup[t]
    except Exception as e: logger.error(f"Error in min uptime rule @t={t}: {e}"); raise

def electrolyzer_min_downtime_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        min_downtime = pyo.value(m.MinDownTimeElectrolyzer)
        if t < min_downtime: return pyo.Constraint.Skip
        start_idx = max(m.TimePeriods.first(), t - min_downtime + 1)
        if not all(i in m.TimePeriods for i in range(start_idx, t + 1)):
             logger.warning(f"Min downtime constraint skipped at t={t} due to index range issues relative to TimePeriods.")
             return pyo.Constraint.Skip
        return sum((1 - m.uElectrolyzer[i]) for i in range(start_idx, t + 1)) >= min_downtime * m.wElectrolyzerShutdown[t]
    except Exception as e: logger.error(f"Error in min downtime rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# ELECTROLYZER DEGRADATION RULE (No changes needed here)
# ---------------------------------------------------------------------------
# ... (electrolyzer_degradation_rule remains the same) ...
def electrolyzer_degradation_rule(m, t):
    enable_degradation = getattr(m, 'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_degradation or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        relative_load_expr = 0
        max_cap_var = m.pElectrolyzer_max
        epsilon = 1e-6
        # Degradation should be based on actual power m.pElectrolyzer
        relative_load_expr = m.pElectrolyzer[t] / (max_cap_var + epsilon)

        time_factor = pyo.value(m.delT_minutes) / 60.0
        op_factor = pyo.value(m.DegradationFactorOperation)
        startup_factor = pyo.value(m.DegradationFactorStartup)

        degradation_increase_op = op_factor * relative_load_expr * time_factor
        degradation_increase_su = 0.0

        enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
        if enable_startup_shutdown and hasattr(m, 'vElectrolyzerStartup'):
            degradation_increase_su = m.vElectrolyzerStartup[t] * startup_factor

        total_degradation_increase = degradation_increase_op + degradation_increase_su

        if t == m.TimePeriods.first():
            return m.DegradationState[t] == m.DegradationStateInitial + total_degradation_increase
        else:
            if (t-1) not in m.TimePeriods:
                logger.warning(f"Skipping degradation constraint at t={t} because t-1 is not in TimePeriods.")
                return pyo.Constraint.Skip
            if not hasattr(m, 'DegradationState'):
                 logger.error(f"DegradationState variable missing at t={t}.")
                 return pyo.Constraint.Skip
            return m.DegradationState[t] == m.DegradationState[t-1] + total_degradation_increase
    except Exception as e:
        logger.error(f"Error defining electrolyzer_degradation rule @t={t}: {e}", exc_info=True)
        raise

# ---------------------------------------------------------------------------
# BATTERY STORAGE RULES (No changes needed here)
# ---------------------------------------------------------------------------
# ... (battery rules remain the same) ...
def battery_soc_balance_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try:
        initial_soc = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Symbolic expression using Var
        time_factor = pyo.value(m.delT_minutes) / 60.0
        charge_eff = pyo.value(m.BatteryChargeEff) # Param value
        discharge_eff = pyo.value(m.BatteryDischargeEff) # Param value
        discharge_term = (m.BatteryDischarge[t] / discharge_eff if discharge_eff > 1e-9 else 0)
        charge_term = m.BatteryCharge[t] * charge_eff

        if t == m.TimePeriods.first():
            return m.BatterySOC[t] == initial_soc + (charge_term - discharge_term) * time_factor
        else:
            return m.BatterySOC[t] == m.BatterySOC[t-1] + (charge_term - discharge_term) * time_factor
    except Exception as e: logger.error(f"Error in battery_soc_balance rule @t={t}: {e}"); raise

def battery_charge_limit_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try: return m.BatteryCharge[t] <= m.BatteryPower_MW * m.BatteryBinaryCharge[t] # Links to Var BatteryPower_MW
    except Exception as e: logger.error(f"Error in battery_charge_limit rule @t={t}: {e}"); raise

def battery_discharge_limit_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try: return m.BatteryDischarge[t] <= m.BatteryPower_MW * m.BatteryBinaryDischarge[t] # Links to Var BatteryPower_MW
    except Exception as e: logger.error(f"Error in battery_discharge_limit rule @t={t}: {e}"); raise

def battery_binary_exclusivity_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try: return m.BatteryBinaryCharge[t] + m.BatteryBinaryDischarge[t] <= 1
    except Exception as e: logger.error(f"Error in battery_binary_exclusivity rule @t={t}: {e}"); raise

def battery_soc_max_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try: return m.BatterySOC[t] <= m.BatteryCapacity_MWh # Compares with Var BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in battery_soc_max rule @t={t}: {e}"); raise

def battery_soc_min_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    try: return m.BatterySOC[t] >= m.BatterySOC_min_fraction * m.BatteryCapacity_MWh # Compares with expression using Var BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in battery_soc_min rule @t={t}: {e}"); raise

# Corrected Battery Ramp Rules
def battery_ramp_up_rule(m, t): # Charge ramp
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor # Corrected: uses symbolic Var BatteryCapacity_MWh * time_factor
        return m.BatteryCharge[t] - m.BatteryCharge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in battery_ramp_up rule @t={t}: {e}"); raise

def battery_ramp_down_rule(m, t): # Charge ramp
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor # Corrected
        return m.BatteryCharge[t-1] - m.BatteryCharge[t] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in battery_ramp_down rule @t={t}: {e}"); raise

def battery_discharge_ramp_up_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor # Corrected
        return m.BatteryDischarge[t] - m.BatteryDischarge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in battery_discharge_ramp_up rule @t={t}: {e}"); raise

def battery_discharge_ramp_down_rule(m, t):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor # Corrected
        return m.BatteryDischarge[t-1] - m.BatteryDischarge[t] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in battery_discharge_ramp_down rule @t={t}: {e}"); raise

def battery_cyclic_soc_lower_rule(m):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    try:
        last_t = m.TimePeriods.last()
        initial_soc_expr = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Symbolic
        tolerance = 0.01 # MWh tolerance - fixed parameter
        return m.BatterySOC[last_t] >= initial_soc_expr - tolerance
    except Exception as e: logger.error(f"Error in battery_cyclic_soc_lower rule: {e}"); raise

def battery_cyclic_soc_upper_rule(m):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    try:
        last_t = m.TimePeriods.last()
        initial_soc_expr = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Symbolic
        tolerance = 0.01 # MWh tolerance
        return m.BatterySOC[last_t] <= initial_soc_expr + tolerance
    except Exception as e: logger.error(f"Error in battery_cyclic_soc_upper rule: {e}"); raise

def battery_power_capacity_link_rule(m):
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not enable_battery: return pyo.Constraint.Skip
    # If BatteryCapacity_MWh is a Param, it means power is also a Param (due to new logic in model.py)
    # So, this linking constraint is not needed / would be an equality of two params.
    if isinstance(getattr(m, 'BatteryCapacity_MWh', None), pyo.Param):
        return pyo.Constraint.Skip
    try: 
        # This rule applies only when BatteryPower_MW and BatteryCapacity_MWh are Vars
        return m.BatteryPower_MW == m.BatteryCapacity_MWh * m.BatteryPowerRatio 
    except Exception as e: 
        logger.error(f"Error in battery_power_capacity_link rule: {e}")
        raise

def battery_min_cap_rule(m):
     enable_battery = getattr(m, 'ENABLE_BATTERY', False)
     if not enable_battery: return pyo.Constraint.Skip
     try:
         if hasattr(m, 'BatteryCapacity_min') and pyo.value(m.BatteryCapacity_min) > 1e-6:
              return m.BatteryCapacity_MWh >= m.BatteryCapacity_min # Compares Var to Param
         else:
              return pyo.Constraint.Skip
     except Exception as e: logger.error(f"Error in battery_min_cap rule: {e}"); raise

# ---------------------------------------------------------------------------
# ANCILLARY SERVICE CAPABILITY RULES (Using get_symbolic_as_bid_sum from utils)
# ---------------------------------------------------------------------------
def Turbine_AS_Zero_rule(m, t):
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    turbine_as_disabled = not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer)
    if turbine_as_disabled:
        all_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
        zero_sum_expr = get_symbolic_as_bid_sum(m, t, all_services, 'Turbine') # Use imported helper
        return zero_sum_expr == 0.0
    return pyo.Constraint.Skip

def Turbine_AS_Pmax_rule(m, t): # Upward capability
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_turbine = get_symbolic_as_bid_sum(m, t, up_services, 'Turbine') # Use imported helper
        return m.pTurbine[t] + up_reserve_bids_turbine <= m.pTurbine_max
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t): # Downward capability
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_turbine = get_symbolic_as_bid_sum(m, t, down_services, 'Turbine') # Use imported helper
        return m.pTurbine[t] - down_reserve_bids_turbine >= m.pTurbine_min
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmin rule @t={t}: {e}"); raise

def Turbine_AS_RU_rule(m, t): # Ramp-Up capability check
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_turbine = get_symbolic_as_bid_sum(m, t, up_services, 'Turbine') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return (m.pTurbine[t] + up_reserve_bids_turbine) - m.pTurbine[t-1] <= m.RU_Turbine_hourly * time_factor
    except Exception as e: logger.error(f"Error in Turbine_AS_RU rule @t={t}: {e}"); raise

def Turbine_AS_RD_rule(m, t): # Ramp-Down capability check
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_turbine = get_symbolic_as_bid_sum(m, t, down_services, 'Turbine') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        return m.pTurbine[t-1] - (m.pTurbine[t] - down_reserve_bids_turbine) <= m.RD_Turbine_hourly * time_factor
    except Exception as e: logger.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmax_rule(m, t): # Capability to increase load (Down-reserve)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_h2 = get_symbolic_as_bid_sum(m, t, down_services, 'Electrolyzer') # Use imported helper
        max_power_limit_expr = m.pElectrolyzer_max
        if enable_startup_shutdown and hasattr(m, 'uElectrolyzer'):
            max_power_limit_expr = m.uElectrolyzer[t] * m.pElectrolyzer_max
        return m.pElectrolyzerSetpoint[t] + down_reserve_bids_h2 <= max_power_limit_expr
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Capability to decrease load (Up-reserve)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_h2 = get_symbolic_as_bid_sum(m, t, up_services, 'Electrolyzer') # Use imported helper
        min_power_limit_expr = m.pElectrolyzer_min
        if enable_startup_shutdown and hasattr(m, 'uElectrolyzer'):
             min_power_limit_expr = m.uElectrolyzer[t] * m.pElectrolyzer_min
        return m.pElectrolyzerSetpoint[t] - up_reserve_bids_h2 >= min_power_limit_expr
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise

def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for Down-reserve (Increasing Load based on Setpoint)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_h2 = get_symbolic_as_bid_sum(m, t, down_services, 'Electrolyzer') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        ramp_limit_mw = m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        return (m.pElectrolyzerSetpoint[t] + down_reserve_bids_h2) - m.pElectrolyzer[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for Up-reserve (Decreasing Load based on Setpoint)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_h2 = get_symbolic_as_bid_sum(m, t, up_services, 'Electrolyzer') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        ramp_limit_mw = m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
        return m.pElectrolyzer[t-1] - (m.pElectrolyzerSetpoint[t] - up_reserve_bids_h2) <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise

def Battery_AS_Pmax_rule(m, t):
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, down_services, 'Battery') # Use imported helper
        available_charge_headroom = m.BatteryPower_MW - m.BatteryCharge[t]
        return down_reserve_bids_battery <= available_charge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmax rule @t={t}: {e}"); raise

def Battery_AS_Pmin_rule(m, t):
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, up_services, 'Battery') # Use imported helper
        available_discharge_headroom = m.BatteryPower_MW - m.BatteryDischarge[t]
        return up_reserve_bids_battery <= available_discharge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmin rule @t={t}: {e}"); raise

def Battery_AS_SOC_Up_rule(m, t):
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, up_services, 'Battery') # Use imported helper
        discharge_eff = pyo.value(m.BatteryDischargeEff)
        as_duration = pyo.value(m.AS_Duration)
        energy_needed_expr = up_reserve_bids_battery * (as_duration / discharge_eff if discharge_eff > 1e-9 else float('inf'))
        min_soc_level_expr = m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
        return m.BatterySOC[t] - energy_needed_expr >= min_soc_level_expr
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Up rule @t={t}: {e}"); raise

def Battery_AS_SOC_Down_rule(m, t):
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, down_services, 'Battery') # Use imported helper
        charge_eff = pyo.value(m.BatteryChargeEff)
        as_duration = pyo.value(m.AS_Duration)
        energy_absorbed_expr = down_reserve_bids_battery * as_duration * charge_eff
        return m.BatterySOC[t] + energy_absorbed_expr <= m.BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Down rule @t={t}: {e}"); raise

def Battery_AS_RU_rule(m, t): # Ramp capability for Down-reg (Increasing Charge)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, down_services, 'Battery') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw_expr = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return (m.BatteryCharge[t] + down_reserve_bids_battery) - m.BatteryCharge[t-1] <= ramp_limit_mw_expr
    except Exception as e: logger.error(f"Error in Battery_AS_RU rule @t={t}: {e}"); raise

def Battery_AS_RD_rule(m, t): # Ramp capability for Up-reg (Increasing Discharge)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = get_symbolic_as_bid_sum(m, t, up_services, 'Battery') # Use imported helper
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw_expr = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        return (m.BatteryDischarge[t] + up_reserve_bids_battery) - m.BatteryDischarge[t-1] <= ramp_limit_mw_expr
    except Exception as e: logger.error(f"Error in Battery_AS_RD rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# ANCILLARY SERVICE LINKING RULES (BIDS) (Using get_symbolic_as_bid_sum from utils)
# ---------------------------------------------------------------------------
def link_total_as_rule(m, t, service_name):
    """Generic rule to link component AS BIDS to the total bid for a service."""
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    if not can_provide_as: return pyo.Constraint.Skip
    try:
        total_var = getattr(m, f"Total_{service_name}", None)
        if total_var is None or not isinstance(total_var, pyo.Var) or not (total_var.is_indexed() and t in total_var.index_set()):
             return pyo.Constraint.Skip

        enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
        enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
        enable_battery = getattr(m, 'ENABLE_BATTERY', False)

        turbine_bid_expr = 0.0
        if enable_npp and (enable_electrolyzer or enable_battery):
            turbine_bid_expr = get_symbolic_as_bid_sum(m, t, [service_name], 'Turbine') # Use imported helper

        electro_bid_expr = 0.0
        if enable_electrolyzer:
            electro_bid_expr = get_symbolic_as_bid_sum(m, t, [service_name], 'Electrolyzer') # Use imported helper

        battery_bid_expr = 0.0
        if enable_battery:
             battery_bid_expr = get_symbolic_as_bid_sum(m, t, [service_name], 'Battery') # Use imported helper

        return total_var[t] == turbine_bid_expr + electro_bid_expr + battery_bid_expr
    except AttributeError as e: logger.debug(f"Attribute error linking service {service_name} at time {t}: {e}"); return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in link_total_as_rule for {service_name} @t={t}: {e}"); raise

# Explicit rules calling the generic linker for BIDS
def link_Total_RegUp_rule(m, t): return link_total_as_rule(m, t, 'RegUp')
def link_Total_RegDown_rule(m, t): return link_total_as_rule(m, t, 'RegDown')
def link_Total_SR_rule(m, t): return link_total_as_rule(m, t, 'SR')
def link_Total_NSR_rule(m, t): return link_total_as_rule(m, t, 'NSR')
def link_Total_ECRS_rule(m, t): return link_total_as_rule(m, t, 'ECRS')
def link_Total_30Min_rule(m, t): return link_total_as_rule(m, t, 'ThirtyMin')
def link_Total_RampUp_rule(m, t): return link_total_as_rule(m, t, 'RampUp')
def link_Total_RampDown_rule(m, t): return link_total_as_rule(m, t, 'RampDown')
def link_Total_UncU_rule(m, t): return link_total_as_rule(m, t, 'UncU')


# ---------------------------------------------------------------------------
# CONDITIONAL RULES for DISPATCH EXECUTION MODE
# ---------------------------------------------------------------------------
def link_deployed_to_bid_rule(m, t, service_name, component_name):
    """Links DEPLOYED AS amount to the component's winning bid."""
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
        return pyo.Constraint.Skip
    try:
        deployed_var_name = f"{service_name}_{component_name}_Deployed"
        bid_var_name = f"{service_name}_{component_name}"
        target_iso = getattr(m, 'TARGET_ISO', 'UNKNOWN')
        iso_service_key_for_param = service_name

        if not (hasattr(m, deployed_var_name) and hasattr(m, bid_var_name)): return pyo.Constraint.Skip

        win_rate_param_name_on_model = f"winning_rate_{iso_service_key_for_param}_{target_iso}"
        deploy_factor_param_name_on_model = f"deploy_factor_{iso_service_key_for_param}_{target_iso}"

        if not (hasattr(m, win_rate_param_name_on_model) and hasattr(m, deploy_factor_param_name_on_model)): return pyo.Constraint.Skip

        deployed_var = getattr(m, deployed_var_name)[t]
        bid_var = getattr(m, bid_var_name)[t]
        win_rate_param = getattr(m, win_rate_param_name_on_model)[t]
        deploy_factor_param = getattr(m, deploy_factor_param_name_on_model)[t]

        return deployed_var == bid_var * win_rate_param * deploy_factor_param
    except Exception as e: logger.error(f"Error in link_deployed_to_bid_rule for {service_name} {component_name} @t={t}: {e}"); raise

def define_actual_electrolyzer_power_rule(m, t):
    """Defines actual pElectrolyzer based on Setpoint and Deployed AS."""
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False) or \
       not getattr(m, 'ENABLE_ELECTROLYZER', False):
        return pyo.Constraint.Skip
    try:
        if not hasattr(m, 'pElectrolyzerSetpoint') or not hasattr(m, 'pElectrolyzer'):
             logger.error("Missing pElectrolyzerSetpoint or pElectrolyzer for dispatch definition."); return pyo.Constraint.Skip

        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        down_services = ['RegDown', 'RampDown']

        # Use the correct helper from utils to sum DEPLOYED quantities
        total_up_deployed_expr = get_symbolic_as_deployed_sum(m, t, up_services, 'Electrolyzer')
        total_down_deployed_expr = get_symbolic_as_deployed_sum(m, t, down_services, 'Electrolyzer')

        return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t] - total_up_deployed_expr + total_down_deployed_expr
    except Exception as e: logger.error(f"Error in define_actual_electrolyzer_power_rule @t={t}: {e}"); raise
