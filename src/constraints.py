# src/constraints.py
import pyomo.environ as pyo
from logging_setup import logger
from config import ( # Assuming SIMULATE_AS_DISPATCH_EXECUTION is imported or passed via model
    ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_H2_CAP_FACTOR,
    ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY, ENABLE_ELECTROLYZER,
    ENABLE_NUCLEAR_GENERATOR, CAN_PROVIDE_ANCILLARY_SERVICES
    # NOTE: You might need to import SIMULATE_AS_DISPATCH_EXECUTION here if not passed via model object 'm'
)

# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------
# Using the improved PWL helper from model.py locally or ensure it's imported
# For completeness, including the improved version here:
def build_piecewise_constraints(model: pyo.ConcreteModel, *, component_prefix: str,
                                input_var_name: str, output_var_name: str,
                                breakpoint_set_name: str, value_param_name: str) -> None: # Removed n_segments
    """Attach SOS2 piece‑wise linear constraints *in‑place* to `model`."""
    logger.info("Building piece‑wise constraints for %s using SOS2", component_prefix)

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
    else: value_data_source = value_param_name

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

    # Ensure constraints are not added multiple times if helper is somehow called repeatedly for same component
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
            elif isinstance(value_data_source, dict):
                return output_var[t] == sum(lam[t, bp] * value_data_source.get(bp, 0.0) for bp in breakpoint_set_to_use)
            logger.critical(f"Value source for {component_prefix} output link is invalid type in _output_link. This is unexpected.")
            return pyo.Constraint.Skip
        model.add_component(output_link_constr_name, pyo.Constraint(model.TimePeriods, rule=_output_link))

    if not hasattr(model, sos2_constr_name):
        def _sos2_rule(m, t): return [lam[t, bp] for bp in breakpoint_set_to_use]
        model.add_component(sos2_constr_name, pyo.SOSConstraint(model.TimePeriods, rule=_sos2_rule, sos=2))


# ---------------------------------------------------------------------------
# PHYSICAL BALANCE RULES
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
        # Ensure qSteam_Total is accessed as a parameter value if fixed, or symbolically if variable
        total_steam_available = m.qSteam_Total # Assuming qSteam_Total is Param
        return turbine_steam + hte_steam == total_steam_available
    except Exception as e:
        logger.error(f"Error in steam_balance rule @t={t}: {e}")
        raise

def power_balance_rule(m, t):
    """Ensures power generation equals consumption + net grid interaction."""
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    try:
        turbine_power = m.pTurbine[t] if enable_npp and hasattr(m, 'pTurbine') else 0
        battery_discharge = m.BatteryDischarge[t] if enable_battery and hasattr(m, 'BatteryDischarge') else 0
        electrolyzer_power = m.pElectrolyzer[t] if enable_electrolyzer and hasattr(m, 'pElectrolyzer') else 0
        battery_charge = m.BatteryCharge[t] if enable_battery and hasattr(m, 'BatteryCharge') else 0
        auxiliary_power = m.pAuxiliary[t] if hasattr(m, 'pAuxiliary') else 0

        return turbine_power + battery_discharge - electrolyzer_power - battery_charge - auxiliary_power == m.pIES[t]
    except Exception as e:
        logger.error(f"Error in power_balance rule @t={t}: {e}")
        raise

def constant_turbine_power_rule(m,t):
    """Fixes turbine power if LTE mode is active."""
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (enable_npp and enable_electrolyzer and lte_mode): return pyo.Constraint.Skip
    try:
        # Check if both components exist before constraining
        if hasattr(m, 'pTurbine') and hasattr(m, 'pTurbine_LTE_setpoint'):
            return m.pTurbine[t] == m.pTurbine_LTE_setpoint
        else:
            logger.warning(f"Skipping constant_turbine_power rule @t={t}: Missing pTurbine or pTurbine_LTE_setpoint.")
            return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in constant_turbine_power rule @t={t}: {e}"); raise

def link_auxiliary_power_rule(m, t):
    """Links auxiliary power consumption to hydrogen production rate."""
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not hasattr(m, 'pAuxiliary') or not enable_electrolyzer:
        return pyo.Constraint.Skip
    try:
        if hasattr(m, 'mHydrogenProduced') and hasattr(m, 'aux_power_consumption_per_kg_h2'):
            # Use parameter value directly, as it's fixed
            aux_rate = pyo.value(m.aux_power_consumption_per_kg_h2)
            return m.pAuxiliary[t] == m.mHydrogenProduced[t] * aux_rate / 1000.0
        else:
             logger.warning(f"Skipping link_auxiliary_power rule @t={t}: Missing components.")
             return pyo.Constraint.Skip
    except Exception as e:
        logger.error(f"Error in link_auxiliary_power rule @t={t}: {e}")
        raise

# ---------------------------------------------------------------------------
# H2 STORAGE RULES (Assumed Symbolically Correct)
# ---------------------------------------------------------------------------
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
# RAMP RATE RULES (Corrected)
# ---------------------------------------------------------------------------
def Electrolyzer_RampUp_rule(m, t):
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Ensure pElectrolyzer_max is accessed symbolically if it's a Var
        # If pElectrolyzer_max is Var: needs careful handling (linearization or robust bounds)
        # Assuming pElectrolyzer_max is treated as fixed for the ramp calculation (based on its variable value)
        # A truly robust formulation might use Big-M or alternative methods if pElecMax is variable.
        # For now, using it symbolically as per the correction.
        return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor
    except Exception as e: logger.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t):
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_electrolyzer: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        time_factor = pyo.value(m.delT_minutes) / 60.0
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
# PRODUCTION REQUIREMENT RULE (h2_CapacityFactor_rule)
# This rule calculates potential max production based on pElectrolyzer_max.
# If pElectrolyzer_max is a Var, using pyo.value() inside is problematic for standard constraints.
# This might need reformulation if pElectrolyzer_max is variable, e.g., using linearization
# or defining it relative to the upper bound parameter. Assuming pElectrolyzer_max behaves like a fixed parameter within this rule's context for now.
# ---------------------------------------------------------------------------
def h2_CapacityFactor_rule(m):
    enable_h2_cap_factor = getattr(m, 'ENABLE_H2_CAP_FACTOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_h2_cap_factor or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        total_hours_sim = len(m.TimePeriods) * (pyo.value(m.delT_minutes) / 60.0)

        # *** Check if pElectrolyzer_max is Var or Param ***
        p_elec_max_comp = getattr(m, 'pElectrolyzer_max', None)
        if p_elec_max_comp is None: return pyo.Constraint.Skip # Cannot calculate if missing

        # If pElectrolyzer_max is a variable, using its value directly in a standard constraint is complex.
        # A common approach is to define potential production based on the *upper bound parameter*.
        max_elec_power_limit = pyo.value(m.pElectrolyzer_max_upper_bound)
        # Or, if pElectrolyzer_max is *fixed* to a value (even if technically a Var), its value could be used.
        # Using the upper bound parameter seems safer for defining potential.
        if max_elec_power_limit <= 1e-6: return pyo.Constraint.Skip

        if not hasattr(m, 'pElectrolyzer_efficiency_breakpoints') or not list(m.pElectrolyzer_efficiency_breakpoints):
            logger.warning("H2 Cap Factor: Missing or empty electrolyzer efficiency breakpoints.")
            return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last() # Assumes sorted

        if not hasattr(m, 'ke_H2_inv_values') or max_power_bp not in m.ke_H2_inv_values:
             logger.warning("H2 Cap Factor: Missing inverse efficiency values (ke_H2_inv_values).")
             return pyo.Constraint.Skip
        max_h2_rate_kg_per_mwh = pyo.value(m.ke_H2_inv_values[max_power_bp])
        if max_h2_rate_kg_per_mwh < 1e-9:
             logger.warning("H2 Cap Factor: Near-zero efficiency at max breakpoint.")
             return pyo.Constraint.Skip

        # Calculate potential based on the max capacity *parameter*
        max_h2_rate_kg_per_hr_est = max_elec_power_limit * max_h2_rate_kg_per_mwh
        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours_sim
        if max_potential_h2_kg_total_est <= 1e-6: return pyo.Constraint.Skip

        # Actual production uses symbolic sum
        total_actual_production_kg = sum(m.mHydrogenProduced[t] * (pyo.value(m.delT_minutes) / 60.0) for t in m.TimePeriods)
        target_production_kg = m.h2_target_capacity_factor * max_potential_h2_kg_total_est

        return total_actual_production_kg >= target_production_kg
    except Exception as e: logger.error(f"Error in h2_CapacityFactor rule: {e}"); raise


# ---------------------------------------------------------------------------
# STARTUP/SHUTDOWN CONSTRAINTS (Assumed Symbolically Correct)
# ---------------------------------------------------------------------------
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
    try: return m.pElectrolyzer[t] >= m.pElectrolyzer_min * m.uElectrolyzer[t] # Min power is Param
    except Exception as e: logger.error(f"Error in SU/SD min power rule @t={t}: {e}"); raise

def electrolyzer_max_power_rule(m, t):
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_startup_shutdown or not enable_electrolyzer: return pyo.Constraint.Skip
    try: return m.pElectrolyzer[t] <= m.pElectrolyzer_max * m.uElectrolyzer[t] # Max power is Var
    except Exception as e: logger.error(f"Error in SU/SD max power rule @t={t}: {e}"); raise

def electrolyzer_min_power_sds_disabled_rule(m, t):
    # Kept as Skip, assuming PWL curve handles min load if SDS is off.
    return pyo.Constraint.Skip

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
# ELECTROLYZER DEGRADATION RULE (Corrected)
# ---------------------------------------------------------------------------
def electrolyzer_degradation_rule(m, t):
    enable_degradation = getattr(m, 'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not enable_degradation or not enable_electrolyzer: return pyo.Constraint.Skip
    try:
        relative_load_expr = 0
        # Use symbolic pElectrolyzer_max (Var)
        max_cap_var = m.pElectrolyzer_max
        # Need to handle division by variable: use a small epsilon or reformulation if max_cap_var can be 0
        # Assuming bounds enforce max_cap_var > 0 if electrolyzer is built. Add epsilon for safety.
        epsilon = 1e-6
        relative_load_expr = m.pElectrolyzer[t] / (max_cap_var + epsilon)

        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Degradation factors are Params, access their values
        op_factor = pyo.value(m.DegradationFactorOperation)
        startup_factor = pyo.value(m.DegradationFactorStartup)

        degradation_increase_op = op_factor * relative_load_expr * time_factor
        degradation_increase_su = 0.0

        enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
        if enable_startup_shutdown and hasattr(m, 'vElectrolyzerStartup'):
            # Use symbolic startup variable
            degradation_increase_su = m.vElectrolyzerStartup[t] * startup_factor

        total_degradation_increase = degradation_increase_op + degradation_increase_su

        if t == m.TimePeriods.first():
            # Initial state is Param
            return m.DegradationState[t] == m.DegradationStateInitial + total_degradation_increase
        else:
            if (t-1) not in m.TimePeriods:
                logger.warning(f"Skipping degradation constraint at t={t} because t-1 is not in TimePeriods.")
                return pyo.Constraint.Skip
            if not hasattr(m, 'DegradationState'):
                 logger.error(f"DegradationState variable missing at t={t}.")
                 return pyo.Constraint.Skip
            # Link to previous state symbolically
            return m.DegradationState[t] == m.DegradationState[t-1] + total_degradation_increase
    except Exception as e:
        logger.error(f"Error defining electrolyzer_degradation rule @t={t}: {e}", exc_info=True)
        raise # Re-raise errors during definition


# ---------------------------------------------------------------------------
# BATTERY STORAGE RULES (Corrected Ramp Limits)
# ---------------------------------------------------------------------------
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
    try: return m.BatteryPower_MW == m.BatteryCapacity_MWh * m.BatteryPowerRatio # Links two Vars via a Param
    except Exception as e: logger.error(f"Error in battery_power_capacity_link rule: {e}"); raise

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
# ANCILLARY SERVICE HELPER (Removed - was causing issues in constraint defs)
# ---------------------------------------------------------------------------
# def get_as_components(m, t): # REMOVED

# ---------------------------------------------------------------------------
# ANCILLARY SERVICE CAPABILITY RULES (Corrected - Use Symbolic Vars)
# ---------------------------------------------------------------------------

# --- Helper to get symbolic sum of AS bids ---
def _get_symbolic_as_sum(m, t, service_list, component_suffix):
    """ Returns symbolic sum expression for given services and component """
    terms = []
    for service in service_list:
        var_name = f"{service}_{component_suffix}"
        if hasattr(m, var_name):
            var_comp = getattr(m, var_name)
            if var_comp.is_indexed() and t in var_comp.index_set():
                terms.append(var_comp[t])
            elif not var_comp.is_indexed(): # Should not happen for bids by time
                 logger.warning(f"AS bid variable {var_name} is not indexed.")
    return sum(terms) if terms else 0.0 # Return 0.0 if list is empty

# --- Turbine AS Capability ---
def Turbine_AS_Zero_rule(m, t):
    # This rule forces bids to zero if turbine cannot provide AS
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    
    # Determine if turbine AS is disabled specifically
    turbine_as_disabled = not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer)
    
    if turbine_as_disabled:
        all_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
        zero_sum_expr = _get_symbolic_as_sum(m, t, all_services, 'Turbine')
        return zero_sum_expr == 0.0
    return pyo.Constraint.Skip # Otherwise, capability is checked below

def Turbine_AS_Pmax_rule(m, t): # Upward capability
    # Check basic flags
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    
    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS'] # List all upward services
        up_reserve_bids_turbine = _get_symbolic_as_sum(m, t, up_services, 'Turbine')
        return m.pTurbine[t] + up_reserve_bids_turbine <= m.pTurbine_max
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t): # Downward capability
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    lte_mode = getattr(m, 'LTE_MODE', False)
    if not (can_provide_as and enable_npp) or (lte_mode and enable_electrolyzer): return pyo.Constraint.Skip
    
    try:
        down_services = ['RegDown', 'RampDown'] # List all downward services
        down_reserve_bids_turbine = _get_symbolic_as_sum(m, t, down_services, 'Turbine')
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
        up_reserve_bids_turbine = _get_symbolic_as_sum(m, t, up_services, 'Turbine')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Check if turbine can ramp from previous actual power to current power + upward reserve obligation
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
        down_reserve_bids_turbine = _get_symbolic_as_sum(m, t, down_services, 'Turbine')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Check if turbine can ramp down from previous actual power to current power - downward reserve obligation
        return m.pTurbine[t-1] - (m.pTurbine[t] - down_reserve_bids_turbine) <= m.RD_Turbine_hourly * time_factor
    except Exception as e: logger.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise

# --- Electrolyzer AS Capability ---
def Electrolyzer_AS_Pmax_rule(m, t): # Capability to increase load (Down-reserve)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip

    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_h2 = _get_symbolic_as_sum(m, t, down_services, 'Electrolyzer')

        max_power_limit_expr = m.pElectrolyzer_max # Max capacity is Var
        if enable_startup_shutdown and hasattr(m, 'uElectrolyzer'):
            max_power_limit_expr = m.uElectrolyzer[t] * m.pElectrolyzer_max # Scale by on/off status (symbolic)

        # Capability check: Setpoint + DownBid <= Max Limit
        return m.pElectrolyzerSetpoint[t] + down_reserve_bids_h2 <= max_power_limit_expr
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Capability to decrease load (Up-reserve)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    enable_startup_shutdown = getattr(m, 'ENABLE_STARTUP_SHUTDOWN', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip

    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_h2 = _get_symbolic_as_sum(m, t, up_services, 'Electrolyzer')

        min_power_limit_expr = m.pElectrolyzer_min # Min power is Param
        if enable_startup_shutdown and hasattr(m, 'uElectrolyzer'):
             min_power_limit_expr = m.uElectrolyzer[t] * m.pElectrolyzer_min # Scale by on/off status (symbolic)

        # Capability check: Setpoint - UpBid >= Min Limit
        return m.pElectrolyzerSetpoint[t] - up_reserve_bids_h2 >= min_power_limit_expr
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise

def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for Down-reserve (Increasing Load)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip

    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_h2 = _get_symbolic_as_sum(m, t, down_services, 'Electrolyzer')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        # Use symbolic Var pElectrolyzer_max for ramp limit calculation
        ramp_limit_mw = m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor

        # Capability check: Setpoint+DownBid (potential future state) minus previous ACTUAL state <= Ramp Limit
        return (m.pElectrolyzerSetpoint[t] + down_reserve_bids_h2) - m.pElectrolyzer[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for Up-reserve (Decreasing Load)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
    if not (enable_electrolyzer and can_provide_as): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip

    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_h2 = _get_symbolic_as_sum(m, t, up_services, 'Electrolyzer')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        ramp_limit_mw = m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max * time_factor # Use symbolic Var pElectrolyzer_max

        # Capability check: Previous ACTUAL state minus Setpoint-UpBid (potential future state) <= Ramp Limit
        return m.pElectrolyzer[t-1] - (m.pElectrolyzerSetpoint[t] - up_reserve_bids_h2) <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise

# --- Battery AS Capability ---
def Battery_AS_Pmax_rule(m, t): # Down-regulation capability (Charging headroom)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip

    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = _get_symbolic_as_sum(m, t, down_services, 'Battery')
        # BatteryPower_MW is Var, linked to capacity
        available_charge_headroom = m.BatteryPower_MW - m.BatteryCharge[t]
        return down_reserve_bids_battery <= available_charge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmax rule @t={t}: {e}"); raise

def Battery_AS_Pmin_rule(m, t): # Up-regulation capability (Discharging headroom)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip

    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = _get_symbolic_as_sum(m, t, up_services, 'Battery')
        available_discharge_headroom = m.BatteryPower_MW - m.BatteryDischarge[t]
        return up_reserve_bids_battery <= available_discharge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmin rule @t={t}: {e}"); raise

def Battery_AS_SOC_Up_rule(m, t): # Energy constraint for Up-reg (Discharging)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip

    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = _get_symbolic_as_sum(m, t, up_services, 'Battery')
        discharge_eff = pyo.value(m.BatteryDischargeEff) # Param value
        as_duration = pyo.value(m.AS_Duration) # Param value
        # Calculate energy needed symbolically relative to bids
        energy_needed_expr = up_reserve_bids_battery * (as_duration / discharge_eff if discharge_eff > 1e-9 else float('inf'))
        # Min SOC level is symbolic expression based on Var BatteryCapacity_MWh
        min_soc_level_expr = m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
        return m.BatterySOC[t] - energy_needed_expr >= min_soc_level_expr
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Up rule @t={t}: {e}"); raise

def Battery_AS_SOC_Down_rule(m, t): # Energy constraint for Down-reg (Charging)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as): return pyo.Constraint.Skip

    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = _get_symbolic_as_sum(m, t, down_services, 'Battery')
        charge_eff = pyo.value(m.BatteryChargeEff) # Param value
        as_duration = pyo.value(m.AS_Duration) # Param value
        energy_absorbed_expr = down_reserve_bids_battery * as_duration * charge_eff
        # Max SOC level uses symbolic Var BatteryCapacity_MWh
        return m.BatterySOC[t] + energy_absorbed_expr <= m.BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Down rule @t={t}: {e}"); raise

def Battery_AS_RU_rule(m, t): # Ramp capability for Down-reg (Increasing Charge)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as) or t == m.TimePeriods.first(): return pyo.Constraint.Skip

    try:
        down_services = ['RegDown', 'RampDown']
        down_reserve_bids_battery = _get_symbolic_as_sum(m, t, down_services, 'Battery')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        # Use corrected symbolic ramp limit
        ramp_limit_mw_expr = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        # Capability check: (Charge[t] + DownBid) - Charge[t-1] <= Ramp Limit
        return (m.BatteryCharge[t] + down_reserve_bids_battery) - m.BatteryCharge[t-1] <= ramp_limit_mw_expr
    except Exception as e: logger.error(f"Error in Battery_AS_RU rule @t={t}: {e}"); raise

def Battery_AS_RD_rule(m, t): # Ramp capability for Up-reg (Increasing Discharge)
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    enable_battery = getattr(m, 'ENABLE_BATTERY', False)
    if not (enable_battery and can_provide_as) or t == m.TimePeriods.first(): return pyo.Constraint.Skip

    try:
        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        up_reserve_bids_battery = _get_symbolic_as_sum(m, t, up_services, 'Battery')
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        # Use corrected symbolic ramp limit
        ramp_limit_mw_expr = m.BatteryRampRate * m.BatteryCapacity_MWh * time_factor
        # Capability check: (Discharge[t] + UpBid) - Discharge[t-1] <= Ramp Limit
        return (m.BatteryDischarge[t] + up_reserve_bids_battery) - m.BatteryDischarge[t-1] <= ramp_limit_mw_expr
    except Exception as e: logger.error(f"Error in Battery_AS_RD rule @t={t}: {e}"); raise


# ---------------------------------------------------------------------------
# ANCILLARY SERVICE LINKING RULES (BIDS) (Assumed Symbolically Correct)
# ---------------------------------------------------------------------------
def link_total_as_rule(m, t, service_name):
    """Generic rule to link component AS BIDS to the total bid for a service."""
    can_provide_as = getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False)
    if not can_provide_as: return pyo.Constraint.Skip
    try:
        total_var = getattr(m, f"Total_{service_name}", None)
        if total_var is None or not isinstance(total_var, pyo.Var) or not (total_var.is_indexed() and t in total_var.index_set()):
             return pyo.Constraint.Skip # Skip if Total isn't a Var for this service/time

        # Use helper to get symbolic sum for components
        enable_npp = getattr(m, 'ENABLE_NUCLEAR_GENERATOR', False)
        enable_electrolyzer = getattr(m, 'ENABLE_ELECTROLYZER', False)
        enable_battery = getattr(m, 'ENABLE_BATTERY', False)

        turbine_bid_expr = 0.0
        if enable_npp and (enable_electrolyzer or enable_battery):
            turbine_bid_expr = _get_symbolic_as_sum(m, t, [service_name], 'Turbine') # Sum for just this service

        electro_bid_expr = 0.0
        if enable_electrolyzer:
            electro_bid_expr = _get_symbolic_as_sum(m, t, [service_name], 'Electrolyzer')

        battery_bid_expr = 0.0
        if enable_battery:
             battery_bid_expr = _get_symbolic_as_sum(m, t, [service_name], 'Battery')

        return total_var[t] == turbine_bid_expr + electro_bid_expr + battery_bid_expr
    except AttributeError as e:
        logger.debug(f"Attribute error linking service {service_name} at time {t}: {e}")
        return pyo.Constraint.Skip
    except Exception as e:
        logger.error(f"Error in link_total_as_rule for {service_name} @t={t}: {e}")
        raise

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
# CONDITIONAL RULES for DISPATCH EXECUTION MODE (Assumed Symbolically Correct)
# ---------------------------------------------------------------------------
def link_deployed_to_bid_rule(m, t, service_name, component_name):
    """
    Generic rule to link DEPLOYED AS amount to the component's winning bid.
    Active only when SIMULATE_AS_DISPATCH_EXECUTION is True.
    """
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
        return pyo.Constraint.Skip

    try:
        deployed_var_name = f"{service_name}_{component_name}_Deployed"
        bid_var_name = f"{service_name}_{component_name}"
        target_iso = getattr(m, 'TARGET_ISO', 'UNKNOWN')
        # Use service_name directly assuming param names align, adjust if mapping needed
        iso_service_key_for_param = service_name
        win_rate_param_name = f"winning_rate_{iso_service_key_for_param}_{target_iso}"
        deploy_factor_param_name = f"deploy_factor_{iso_service_key_for_param}_{target_iso}"

        # Check existence of all components symbolically
        if not (hasattr(m, deployed_var_name) and hasattr(m, bid_var_name) and \
                hasattr(m, win_rate_param_name) and hasattr(m, deploy_factor_param_name)):
            return pyo.Constraint.Skip # Skip silently if any part is missing

        deployed_var = getattr(m, deployed_var_name)[t]
        bid_var = getattr(m, bid_var_name)[t]
        win_rate_param = getattr(m, win_rate_param_name)[t] # Parameters accessed symbolically
        deploy_factor_param = getattr(m, deploy_factor_param_name)[t] # Parameters accessed symbolically

        # Constraint: Deployed = Bid * WinRate * DeployFactor (Symbolic expression)
        return deployed_var == bid_var * win_rate_param * deploy_factor_param
    except Exception as e:
        logger.error(f"Error in link_deployed_to_bid_rule for {service_name} {component_name} @t={t}: {e}")
        raise # Re-raise error

# --- Rule to define actual electrolyzer power based on setpoint and deployment ---
def define_actual_electrolyzer_power_rule(m, t):
    """
    Explicitly defines pElectrolyzer based on Setpoint and Deployed AS.
    Active only when SIMULATE_AS_DISPATCH_EXECUTION is True.
    """
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False) or \
       not getattr(m, 'ENABLE_ELECTROLYZER', False):
        return pyo.Constraint.Skip

    try:
        if not hasattr(m, 'pElectrolyzerSetpoint') or not hasattr(m, 'pElectrolyzer'):
             logger.error("Missing pElectrolyzerSetpoint or pElectrolyzer for dispatch definition.")
             return pyo.Constraint.Skip

        up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS']
        down_services = ['RegDown', 'RampDown']

        total_up_deployed_expr = _get_symbolic_as_sum(m, t, up_services, 'Electrolyzer')
        total_down_deployed_expr = _get_symbolic_as_sum(m, t, down_services, 'Electrolyzer')

        # Actual Power = Setpoint - Deployed Up-Reserves + Deployed Down-Reserves (Symbolic)
        return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t] - total_up_deployed_expr + total_down_deployed_expr
    except Exception as e:
        logger.error(f"Error in define_actual_electrolyzer_power_rule @t={t}: {e}")
        raise