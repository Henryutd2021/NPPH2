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
# build_piecewise_constraints remains unchanged...
def build_piecewise_constraints(model: pyo.ConcreteModel, *, component_prefix: str,
                                input_var_name: str, output_var_name: str,
                                breakpoint_set_name: str, value_param_name: str, n_segments=None) -> None:
    """Attach SOS2 piece‑wise linear constraints *in‑place* to `model`."""
    logger.info("Building piece‑wise constraints for %s using SOS2", component_prefix)
    input_var = getattr(model, input_var_name)
    output_var = getattr(model, output_var_name)
    breakpoint_set = getattr(model, breakpoint_set_name)
    value_param = getattr(model, value_param_name)
    # --- MODIFICATION START ---
    if not breakpoint_set.isordered(): # NEW Line
    # --- MODIFICATION END ---
         logger.warning(f"Breakpoint set {breakpoint_set_name} for {component_prefix} is not ordered.")
         try:
             sorted_breakpoints = sorted(list(breakpoint_set))
             # Re-create the set as ordered if possible (might not be necessary if only checking)
             # breakpoint_set = pyo.Set(initialize=sorted_breakpoints, ordered=True)
             # setattr(model, breakpoint_set_name, breakpoint_set)
             logger.info(f"Proceeding with potentially unsorted breakpoint set {breakpoint_set_name}.")
         except TypeError:
              logger.error(f"Cannot sort breakpoint set {breakpoint_set_name}.")
              raise ValueError(f"Breakpoint set {breakpoint_set_name} must be compatible with SOS2 constraints (implicitly ordered).")


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
def steam_balance_rule(m, t):
    """Links total steam production to turbine and HTE electrolyzer use."""
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    try:
        turbine_steam = m.qSteam_Turbine[t] if hasattr(m, 'qSteam_Turbine') else 0
        hte_steam = m.qSteam_Electrolyzer[t] if ENABLE_ELECTROLYZER and not m.LTE_MODE and hasattr(m, 'qSteam_Electrolyzer') else 0
        total_steam_available = m.qSteam_Total
        return turbine_steam + hte_steam == total_steam_available
    except Exception as e:
        logger.error(f"Error in steam_balance rule @t={t}: {e}")
        raise

def power_balance_rule(m, t):
    """Ensures power generation equals consumption + net grid interaction."""
    try:
        turbine_power = m.pTurbine[t] if ENABLE_NUCLEAR_GENERATOR and hasattr(m, 'pTurbine') else 0
        battery_discharge = m.BatteryDischarge[t] if ENABLE_BATTERY and hasattr(m, 'BatteryDischarge') else 0
        # pElectrolyzer's value depends on the simulation mode via define_actual_electrolyzer_power_rule when AS dispatch is simulated
        electrolyzer_power = m.pElectrolyzer[t] if ENABLE_ELECTROLYZER and hasattr(m, 'pElectrolyzer') else 0
        battery_charge = m.BatteryCharge[t] if ENABLE_BATTERY and hasattr(m, 'BatteryCharge') else 0
        # Include auxiliary power if the variable exists (meaning consumption rate > 0)
        auxiliary_power = m.pAuxiliary[t] if hasattr(m, 'pAuxiliary') else 0

        return turbine_power + battery_discharge - electrolyzer_power - battery_charge - auxiliary_power == m.pIES[t] # ADDED auxiliary_power
    except Exception as e:
        logger.error(f"Error in power_balance rule @t={t}: {e}")
        raise

def constant_turbine_power_rule(m,t):
    """Fixes turbine power if LTE mode is active."""
    if not (ENABLE_NUCLEAR_GENERATOR and ENABLE_ELECTROLYZER and m.LTE_MODE): return pyo.Constraint.Skip
    try:
        return m.pTurbine[t] == m.pTurbine_LTE_setpoint
    except Exception as e: logger.error(f"Error in constant_turbine_power rule @t={t}: {e}"); raise

def link_auxiliary_power_rule(m, t):
    """Links auxiliary power consumption to hydrogen production rate."""
    # Apply this constraint only if the auxiliary power variable exists
    if not hasattr(m, 'pAuxiliary') or not ENABLE_ELECTROLYZER:
        return pyo.Constraint.Skip
    try:
        # pAuxiliary (MW) = mHydrogenProduced (kg/hr) * aux_power_consumption (kWh/kg) / 1000 (kW/MW)
        # Note: mHydrogenProduced is an hourly rate variable (kg/hr)
        # aux_power_consumption_per_kg_h2 is in kWh/kg
        # Resulting pAuxiliary should be in MW
        # Conversion: (kg/hr) * (kWh/kg) -> kW. Divide by 1000 for MW.
        return m.pAuxiliary[t] == m.mHydrogenProduced[t] * m.aux_power_consumption_per_kg_h2 / 1000.0
    except Exception as e:
        logger.error(f"Error in link_auxiliary_power rule @t={t}: {e}")
        raise

# ---------------------------------------------------------------------------
# H2 STORAGE RULES
# ---------------------------------------------------------------------------
def h2_storage_balance_adj_rule(m, t): # Renamed from h2_storage_balance_rule
     """Adjusted storage balance using H2_to_storage variable."""
     if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
     # Ensure efficiency params are non-zero before division
     discharge_eff = pyo.value(m.storage_discharge_eff)
     discharge_term = (m.H2_from_storage[t] / discharge_eff if discharge_eff > 1e-9 else 0)
     charge_term = m.H2_to_storage[t] * m.storage_charge_eff

     if t == m.TimePeriods.first():
         return m.H2_storage_level[t] == m.H2_storage_level_initial + charge_term - discharge_term
     else:
         return m.H2_storage_level[t] == m.H2_storage_level[t-1] + charge_term - discharge_term

def h2_prod_dispatch_rule(m, t):
     """Links hydrogen production to market sales and storage input flow."""
     if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
     # H2 produced = H2 sold directly + H2 flow designated for storage (before eff loss)
     return m.mHydrogenProduced[t] == m.H2_to_market[t] + m.H2_to_storage[t]

def h2_storage_charge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_to_storage[t] <= m.H2_storage_charge_rate_max

def h2_storage_discharge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max

def h2_storage_level_max_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] <= m.H2_storage_capacity_max

def h2_storage_level_min_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] >= m.H2_storage_capacity_min

# ---------------------------------------------------------------------------
# RAMP RATE RULES
# ---------------------------------------------------------------------------
def Electrolyzer_RampUp_rule(m, t):
    # Limits the change in ACTUAL power pElectrolyzer
    if not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t):
    # Limits the change in ACTUAL power pElectrolyzer
    if not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        return m.pElectrolyzer[t-1] - m.pElectrolyzer[t] <= m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampDown rule @t={t}: {e}"); raise

def Turbine_RampUp_rule(m, t):
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    if m.LTE_MODE and ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t] - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampUp rule @t={t}: {e}"); raise

def Turbine_RampDown_rule(m, t):
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    if m.LTE_MODE and ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t-1] - m.pTurbine[t] <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampDown rule @t={t}: {e}"); raise

def Steam_Electrolyzer_Ramp_rule(m, t):
    # Only relevant for HTE mode and if ramp is constrained/costed
    if not (ENABLE_ELECTROLYZER and not m.LTE_MODE): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        if hasattr(m, 'qSteamElectrolyzerRampPos') and hasattr(m, 'qSteamElectrolyzerRampNeg'):
             return m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t] <= m.Ramp_qSteam_Electrolyzer_limit
        else:
             return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in Steam_Electrolyzer_Ramp rule @t={t}: {e}"); raise

# ---------------------------------------------------------------------------
# PRODUCTION REQUIREMENT RULE
# ---------------------------------------------------------------------------
def h2_CapacityFactor_rule(m):
    if not ENABLE_H2_CAP_FACTOR or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    try:
        total_hours_sim = len(m.TimePeriods) * (pyo.value(m.delT_minutes) / 60.0)
        max_elec_power_ub = pyo.value(m.pElectrolyzer_max)
        if max_elec_power_ub <= 1e-6: return pyo.Constraint.Skip

        if not hasattr(m, 'pElectrolyzer_efficiency_breakpoints') or not m.pElectrolyzer_efficiency_breakpoints:
            logger.warning("Cannot calculate H2 Capacity Factor: Missing electrolyzer efficiency breakpoints.")
            return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last()

        if not hasattr(m, 'ke_H2_inv_values') or max_power_bp not in m.ke_H2_inv_values:
             logger.warning("Cannot calculate H2 Capacity Factor: Missing inverse efficiency values (ke_H2_inv_values).")
             return pyo.Constraint.Skip
        max_h2_rate_kg_per_mwh = pyo.value(m.ke_H2_inv_values[max_power_bp])
        if max_h2_rate_kg_per_mwh < 1e-9:
             logger.warning("Cannot calculate H2 Capacity Factor: Near-zero efficiency at max breakpoint.")
             return pyo.Constraint.Skip

        max_h2_rate_kg_per_hr_est = max_elec_power_ub * max_h2_rate_kg_per_mwh
        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours_sim
        if max_potential_h2_kg_total_est <= 1e-6: return pyo.Constraint.Skip

        total_actual_production_kg = sum(m.mHydrogenProduced[t] * (m.delT_minutes / 60.0) for t in m.TimePeriods)
        return total_actual_production_kg >= m.h2_target_capacity_factor * max_potential_h2_kg_total_est
    except Exception as e: logger.error(f"Error in h2_CapacityFactor rule: {e}"); raise

# ---------------------------------------------------------------------------
# STARTUP/SHUTDOWN CONSTRAINTS
# ---------------------------------------------------------------------------
def electrolyzer_on_off_logic_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        return m.uElectrolyzer[t] - m.uElectrolyzer_initial == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
    else:
        return m.uElectrolyzer[t] - m.uElectrolyzer[t-1] == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]

def electrolyzer_min_power_when_on_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # Constrains ACTUAL power pElectrolyzer
    return m.pElectrolyzer[t] >= m.pElectrolyzer_min * m.uElectrolyzer[t]

def electrolyzer_max_power_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # Constrains ACTUAL power pElectrolyzer
    return m.pElectrolyzer[t] <= m.pElectrolyzer_max * m.uElectrolyzer[t]

def electrolyzer_min_power_sds_disabled_rule(m, t):
    if ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # Constrains ACTUAL power pElectrolyzer
    # Ensure pElectrolyzer is >= min value if it's non-zero (implicit ON)
    # Need to handle the case where pElectrolyzer can be zero
    # This might be better handled by the piecewise constraint domain if pElectrolyzer=0 results in mHydrogenProduced=0
    # Or, if non-zero operation is strictly required when enabled:
    # return m.pElectrolyzer[t] >= m.pElectrolyzer_min
    # Let's assume the optimizer can choose zero power if needed, so skip strict enforcement here.
    # The main purpose is fulfilled by electrolyzer_capacity_limit_rule (pElec <= pMax).
    # Min power is implicitly handled by efficiency curve starting point if using PWL approx.
    return pyo.Constraint.Skip # Revisit if strict minimum needed when SU/SD off

def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1

def electrolyzer_min_uptime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    min_uptime = pyo.value(m.MinUpTimeElectrolyzer) # Get integer value
    if t < min_uptime: return pyo.Constraint.Skip
    start_idx = max(m.TimePeriods.first(), t - min_uptime + 1)
    return sum(m.uElectrolyzer[i] for i in range(start_idx, t + 1)) >= min_uptime * m.vElectrolyzerStartup[t] # Corrected: should be v[t] triggering the lookback

def electrolyzer_min_downtime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    min_downtime = pyo.value(m.MinDownTimeElectrolyzer) # Get integer value
    if t < min_downtime: return pyo.Constraint.Skip
    start_idx = max(m.TimePeriods.first(), t - min_downtime + 1)
    return sum((1 - m.uElectrolyzer[i]) for i in range(start_idx, t + 1)) >= min_downtime * m.wElectrolyzerShutdown[t] # Corrected: should be w[t] triggering the lookback

# ---------------------------------------------------------------------------
# ELECTROLYZER DEGRADATION RULE
# ---------------------------------------------------------------------------
def electrolyzer_degradation_rule(m, t):
    if not ENABLE_ELECTROLYZER_DEGRADATION_TRACKING or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    try:
        relative_load_expr = 0
        max_cap = pyo.value(m.pElectrolyzer_max)
        if max_cap > 1e-6:
            relative_load_expr = m.pElectrolyzer[t] / max_cap
        else:
             relative_load_expr = 0

        time_factor = pyo.value(m.delT_minutes) / 60.0
        degradation_increase = m.DegradationFactorOperation * relative_load_expr * time_factor

        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'vElectrolyzerStartup'):
            # Ensure startup var is indexed correctly
            if m.vElectrolyzerStartup.is_indexed() and t in m.vElectrolyzerStartup.index_set():
                degradation_increase += m.vElectrolyzerStartup[t] * m.DegradationFactorStartup
            elif not m.vElectrolyzerStartup.is_indexed(): # Should not happen for time series var
                 logger.warning("vElectrolyzerStartup is not indexed in degradation rule.")

        if t == m.TimePeriods.first():
            return m.DegradationState[t] == m.DegradationStateInitial + degradation_increase
        else:
            # --- MODIFICATION START ---
            # Check if previous time step exists in the TimePeriods set
            # if not hasattr(m.DegradationState, t-1): return pyo.Constraint.Skip # OLD Line - Incorrect usage
            if (t-1) not in m.TimePeriods: # NEW Line - Correct check
                logger.warning(f"Skipping degradation constraint at t={t} because t-1 is not in TimePeriods.")
                return pyo.Constraint.Skip
            # --- MODIFICATION END ---
            # Ensure previous state variable exists (defensive check)
            if not hasattr(m, 'DegradationState') or not m.DegradationState.is_indexed():
                 logger.error(f"DegradationState variable issue at t={t}.")
                 return pyo.Constraint.Skip
            # Access previous state using index
            return m.DegradationState[t] == m.DegradationState[t-1] + degradation_increase
    except Exception as e:
        # Log the specific error and time step
        logger.error(f"Error defining electrolyzer_degradation rule @t={t}: {e}", exc_info=True) # Add traceback
        return pyo.Constraint.Skip # Fallback to skip constraint on error


# ---------------------------------------------------------------------------
# BATTERY STORAGE RULES
# ---------------------------------------------------------------------------
def battery_soc_balance_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
    time_factor = pyo.value(m.delT_minutes) / 60.0
    charge_eff = pyo.value(m.BatteryChargeEff)
    discharge_eff = pyo.value(m.BatteryDischargeEff)
    discharge_term = (m.BatteryDischarge[t] / discharge_eff if discharge_eff > 1e-9 else 0)
    charge_term = m.BatteryCharge[t] * charge_eff

    if t == m.TimePeriods.first():
        return m.BatterySOC[t] == initial_soc_mwh + (charge_term - discharge_term) * time_factor
    else:
        return m.BatterySOC[t] == m.BatterySOC[t-1] + (charge_term - discharge_term) * time_factor

def battery_charge_limit_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryCharge[t] <= m.BatteryPower_MW * m.BatteryBinaryCharge[t]

def battery_discharge_limit_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryDischarge[t] <= m.BatteryPower_MW * m.BatteryBinaryDischarge[t]

def battery_binary_exclusivity_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryBinaryCharge[t] + m.BatteryBinaryDischarge[t] <= 1

def battery_soc_max_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatterySOC[t] <= m.BatteryCapacity_MWh

def battery_soc_min_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatterySOC[t] >= m.BatterySOC_min_fraction * m.BatteryCapacity_MWh

def battery_ramp_up_rule(m, t): # Charge ramp
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = pyo.value(m.delT_minutes) / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryCharge[t] - m.BatteryCharge[t-1] <= ramp_limit_mw

def battery_ramp_down_rule(m, t): # Charge ramp
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = pyo.value(m.delT_minutes) / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryCharge[t-1] - m.BatteryCharge[t] <= ramp_limit_mw

def battery_discharge_ramp_up_rule(m, t):
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = pyo.value(m.delT_minutes) / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryDischarge[t] - m.BatteryDischarge[t-1] <= ramp_limit_mw

def battery_discharge_ramp_down_rule(m, t):
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = pyo.value(m.delT_minutes) / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryDischarge[t-1] - m.BatteryDischarge[t] <= ramp_limit_mw

def battery_cyclic_soc_lower_rule(m):
    if not ENABLE_BATTERY or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    last_t = m.TimePeriods.last()
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
    tolerance = 0.01 # MWh tolerance
    return m.BatterySOC[last_t] >= initial_soc_mwh - tolerance

def battery_cyclic_soc_upper_rule(m):
    if not ENABLE_BATTERY or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    last_t = m.TimePeriods.last()
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh
    tolerance = 0.01 # MWh tolerance
    return m.BatterySOC[last_t] <= initial_soc_mwh + tolerance

def battery_power_capacity_link_rule(m):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryPower_MW == m.BatteryCapacity_MWh * m.BatteryPowerRatio

def battery_min_cap_rule(m):
     if not ENABLE_BATTERY: return pyo.Constraint.Skip
     if hasattr(m, 'BatteryCapacity_min') and pyo.value(m.BatteryCapacity_min) > 1e-6:
          return m.BatteryCapacity_MWh >= m.BatteryCapacity_min
     else:
          return pyo.Constraint.Skip

# ---------------------------------------------------------------------------
# ANCILLARY SERVICE DEFINITIONS HELPER
# ---------------------------------------------------------------------------
def get_as_components(m, t):
    """Helper to organize AS BIDS for each component."""
    as_components = {
        'up_reserves_bid_turbine': 0.0, 'down_reserves_bid_turbine': 0.0,
        'up_reserves_bid_h2': 0.0, 'down_reserves_bid_h2': 0.0,
        'up_reserves_bid_battery': 0.0, 'down_reserves_bid_battery': 0.0,
        'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0
    }
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES: return as_components

    try:
        up_reserves_turbine, down_reserves_turbine = [], []
        up_reserves_h2, down_reserves_h2 = [], []
        up_reserves_battery, down_reserves_battery = [], []
        internal_as_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']

        for service in internal_as_services:
            is_down_reserve = 'Down' in service or service == 'RegD'
            # Turbine Bids
            if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY):
                var_name = f"{service}_Turbine"
                if hasattr(m, var_name) and isinstance(getattr(m, var_name), pyo.Var):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_turbine.append(var)
                    else: up_reserves_turbine.append(var)
            # Electrolyzer Bids
            if ENABLE_ELECTROLYZER:
                var_name = f"{service}_Electrolyzer"
                if hasattr(m, var_name) and isinstance(getattr(m, var_name), pyo.Var):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_h2.append(var)
                    else: up_reserves_h2.append(var)
            # Battery Bids
            if ENABLE_BATTERY:
                var_name = f"{service}_Battery"
                if hasattr(m, var_name) and isinstance(getattr(m, var_name), pyo.Var):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_battery.append(var)
                    else: up_reserves_battery.append(var)

        as_components['up_reserves_bid_turbine'] = sum(up_reserves_turbine) if up_reserves_turbine else 0.0
        as_components['down_reserves_bid_turbine'] = sum(down_reserves_turbine) if down_reserves_turbine else 0.0
        as_components['up_reserves_bid_h2'] = sum(up_reserves_h2) if up_reserves_h2 else 0.0
        as_components['down_reserves_bid_h2'] = sum(down_reserves_h2) if down_reserves_h2 else 0.0
        as_components['up_reserves_bid_battery'] = sum(up_reserves_battery) if up_reserves_battery else 0.0
        as_components['down_reserves_bid_battery'] = sum(down_reserves_battery) if down_reserves_battery else 0.0
        as_components['up_reserves_bid'] = (as_components['up_reserves_bid_turbine'] +
                                            as_components['up_reserves_bid_h2'] +
                                            as_components['up_reserves_bid_battery'])
        as_components['down_reserves_bid'] = (as_components['down_reserves_bid_turbine'] +
                                              as_components['down_reserves_bid_h2'] +
                                              as_components['down_reserves_bid_battery'])
        return as_components
    except Exception as e:
        logger.error(f"Error in get_as_components helper @t={t}: {e}")
        # Return zeros on error
        return {k: 0.0 for k in as_components}


# ---------------------------------------------------------------------------
# ANCILLARY SERVICE CAPABILITY RULES (Based on Bids and Setpoints)
# These ensure the system *can* provide the bid amount if called upon.
# ---------------------------------------------------------------------------

# --- Turbine AS Capability ---
def Turbine_AS_Zero_rule(m, t):
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER):
        zero_as = 0
        internal_as_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
        for service in internal_as_services:
            var_name = f"{service}_Turbine"
            if hasattr(m, var_name) and isinstance(getattr(m, var_name), pyo.Var):
                 zero_as += getattr(m, var_name)[t]
        return zero_as == 0
    return pyo.Constraint.Skip

def Turbine_AS_Pmax_rule(m, t):
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        return m.pTurbine[t] + as_info['up_reserves_bid_turbine'] <= m.pTurbine_max
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t):
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        return m.pTurbine[t] - as_info['down_reserves_bid_turbine'] >= m.pTurbine_min
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmin rule @t={t}: {e}"); raise

def Turbine_AS_RU_rule(m, t):
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        # Turbine must be able to ramp up from previous period's actual power
        # to the current period's actual power PLUS the upward bid requirement
        return (m.pTurbine[t] + as_info['up_reserves_bid_turbine']) - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RU rule @t={t}: {e}"); raise

def Turbine_AS_RD_rule(m, t):
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        # Turbine must be able to ramp down from previous period's actual power
        # to the current period's actual power MINUS the downward bid requirement
        return m.pTurbine[t-1] - (m.pTurbine[t] - as_info['down_reserves_bid_turbine']) <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise

# --- Electrolyzer AS Capability ---
def Electrolyzer_AS_Pmax_rule(m, t): # Capability to increase load (Down-reserve)
    if not (ENABLE_ELECTROLYZER and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_h2 = as_info['down_reserves_bid_h2']
        max_power_limit = m.pElectrolyzer_max
        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'uElectrolyzer'):
            max_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_max
        # Capability check: Setpoint + DownBid <= Max Limit
        return m.pElectrolyzerSetpoint[t] + total_down_bid_h2 <= max_power_limit
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Capability to decrease load (Up-reserve)
    if not (ENABLE_ELECTROLYZER and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_h2 = as_info['up_reserves_bid_h2']
        min_power_limit = m.pElectrolyzer_min
        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'uElectrolyzer'):
             min_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_min
        # Capability check: Setpoint - UpBid >= Min Limit
        return m.pElectrolyzerSetpoint[t] - total_up_bid_h2 >= min_power_limit
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise

def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for Down-reserve (Increasing Load)
    if not (ENABLE_ELECTROLYZER and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_h2 = as_info['down_reserves_bid_h2']
        ramp_limit_mw = m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        # Capability check: Setpoint+DownBid (potential future state) minus previous ACTUAL state <= Ramp Limit
        return (m.pElectrolyzerSetpoint[t] + total_down_bid_h2) - m.pElectrolyzer[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for Up-reserve (Decreasing Load)
    if not (ENABLE_ELECTROLYZER and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_h2 = as_info['up_reserves_bid_h2']
        ramp_limit_mw = m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        # Capability check: Previous ACTUAL state minus Setpoint-UpBid (potential future state) <= Ramp Limit
        return m.pElectrolyzer[t-1] - (m.pElectrolyzerSetpoint[t] - total_up_bid_h2) <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise

# --- Battery AS Capability ---
def Battery_AS_Pmax_rule(m, t): # Down-regulation capability (Charging headroom)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        available_charge_headroom = m.BatteryPower_MW - m.BatteryCharge[t]
        return as_info['down_reserves_bid_battery'] <= available_charge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmax rule @t={t}: {e}"); raise

def Battery_AS_Pmin_rule(m, t): # Up-regulation capability (Discharging headroom)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        available_discharge_headroom = m.BatteryPower_MW - m.BatteryDischarge[t]
        return as_info['up_reserves_bid_battery'] <= available_discharge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmin rule @t={t}: {e}"); raise

def Battery_AS_SOC_Up_rule(m, t): # Energy constraint for Up-reg (Discharging)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        discharge_eff = pyo.value(m.BatteryDischargeEff)
        energy_needed = as_info['up_reserves_bid_battery'] * (pyo.value(m.AS_Duration) / discharge_eff if discharge_eff > 1e-9 else 0)
        min_soc_level = m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
        return m.BatterySOC[t] - energy_needed >= min_soc_level
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Up rule @t={t}: {e}"); raise

def Battery_AS_SOC_Down_rule(m, t): # Energy constraint for Down-reg (Charging)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        energy_absorbed = as_info['down_reserves_bid_battery'] * pyo.value(m.AS_Duration) * m.BatteryChargeEff
        return m.BatterySOC[t] + energy_absorbed <= m.BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Down rule @t={t}: {e}"); raise

def Battery_AS_RU_rule(m, t): # Ramp capability for Down-reg (Increasing Charge)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
        # Capability check: (Charge[t] + DownBid) - Charge[t-1] <= Ramp Limit
        return (m.BatteryCharge[t] + as_info['down_reserves_bid_battery']) - m.BatteryCharge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Battery_AS_RU rule @t={t}: {e}"); raise

def Battery_AS_RD_rule(m, t): # Ramp capability for Up-reg (Increasing Discharge)
    if not (ENABLE_BATTERY and m.CAN_PROVIDE_ANCILLARY_SERVICES) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        time_factor = pyo.value(m.delT_minutes) / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
        # Capability check: (Discharge[t] + UpBid) - Discharge[t-1] <= Ramp Limit
        return (m.BatteryDischarge[t] + as_info['up_reserves_bid_battery']) - m.BatteryDischarge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Battery_AS_RD rule @t={t}: {e}"); raise


# ---------------------------------------------------------------------------
# ANCILLARY SERVICE LINKING RULES (BIDS)
# ---------------------------------------------------------------------------
def link_total_as_rule(m, t, service_name):
    """Generic rule to link component AS BIDS to the total bid for a service."""
    if not m.CAN_PROVIDE_ANCILLARY_SERVICES: return pyo.Constraint.Skip
    try:
        total_var = getattr(m, f"Total_{service_name}", None)
        # Only link if the Total_* variable is actually a Var (not a Param fixed to 0)
        if total_var is None or not isinstance(total_var, pyo.Var):
             return pyo.Constraint.Skip

        turbine_bid = 0
        if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY) and hasattr(m, f"{service_name}_Turbine"):
            # Check if component bid var is defined and is a Var
            comp_var = getattr(m, f"{service_name}_Turbine")
            if isinstance(comp_var, pyo.Var): turbine_bid = comp_var[t]

        electro_bid = 0
        if ENABLE_ELECTROLYZER and hasattr(m, f"{service_name}_Electrolyzer"):
            comp_var = getattr(m, f"{service_name}_Electrolyzer")
            if isinstance(comp_var, pyo.Var): electro_bid = comp_var[t]

        battery_bid = 0
        if ENABLE_BATTERY and hasattr(m, f"{service_name}_Battery"):
             comp_var = getattr(m, f"{service_name}_Battery")
             if isinstance(comp_var, pyo.Var): battery_bid = comp_var[t]

        return total_var[t] == turbine_bid + electro_bid + battery_bid
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
def link_Total_30Min_rule(m, t): return link_total_as_rule(m, t, 'ThirtyMin') # Use consistent internal name
def link_Total_RampUp_rule(m, t): return link_total_as_rule(m, t, 'RampUp')
def link_Total_RampDown_rule(m, t): return link_total_as_rule(m, t, 'RampDown')
def link_Total_UncU_rule(m, t): return link_total_as_rule(m, t, 'UncU')


# ---------------------------------------------------------------------------
# CONDITIONAL RULES for DISPATCH EXECUTION MODE
# ---------------------------------------------------------------------------

def link_deployed_to_bid_rule(m, t, service_name, component_name):
    """
    Generic rule to link DEPLOYED AS amount to the component's winning bid.
    Active only when SIMULATE_AS_DISPATCH_EXECUTION is True.
    Assumes deploy_factor represents the deployed fraction of the winning bid.
    """
    # --- Conditional Activation ---
    # Check flags using m.<FLAG_NAME> notation (assuming flags are set on model object)
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False):
        return pyo.Constraint.Skip

    deployed_var_name = f"{service_name}_{component_name}_Deployed"
    bid_var_name = f"{service_name}_{component_name}"
    # Construct param names dynamically based on convention used in model.py
    # Ensure these param names match exactly how they are defined in model.py
    base_param_name = service_name # Adjust if ISO specific names used e.g. 'Syn' for PJM SR
    # Find the ISO-specific name if needed (logic might be complex depending on model.py structure)
    iso_service_name = service_name # Placeholder - needs actual mapping if used
    win_rate_param_name = f"winning_rate_{iso_service_name}_{m.TARGET_ISO}"
    deploy_factor_param_name = f"deploy_factor_{iso_service_name}_{m.TARGET_ISO}"

    # Check if all necessary model components (Vars and Params) exist for this t
    if not hasattr(m, deployed_var_name) or not hasattr(m, bid_var_name) or \
       not hasattr(m, win_rate_param_name) or not hasattr(m, deploy_factor_param_name):
        # Silently skip if components are missing (e.g., service not defined for this component/ISO)
        return pyo.Constraint.Skip

    # Check if Params are indexed by TimePeriods
    win_rate_param = getattr(m, win_rate_param_name)
    deploy_factor_param = getattr(m, deploy_factor_param_name)
    if not (win_rate_param.is_indexed() and deploy_factor_param.is_indexed() and \
            t in win_rate_param and t in deploy_factor_param):
         logger.warning(f"Skipping deployment link constraint for {service_name} {component_name} at t={t} due to missing index in parameter.")
         return pyo.Constraint.Skip

    try:
        deployed_var = getattr(m, deployed_var_name)[t]
        bid_var = getattr(m, bid_var_name)[t]
        win_rate = win_rate_param[t]
        deploy_factor = deploy_factor_param[t]

        # Constraint: Deployed = Bid * WinRate * DeployFactor
        # This assumes deploy_factor is the fraction deployed.
        # If deploy_factor is only for revenue, you might need Deployed <= Bid * WinRate
        # and another mechanism (e.g., objective) to determine deployed amount.
        # Using equality based on previous discussion:
        return deployed_var == bid_var * win_rate * deploy_factor
    except Exception as e:
        logger.error(f"Error in link_deployed_to_bid_rule for {service_name} {component_name} @t={t}: {e}")
        raise # Re-raise error


# --- Rule to define actual electrolyzer power based on setpoint and deployment ---
def define_actual_electrolyzer_power_rule(m, t):
    """
    Explicitly defines pElectrolyzer based on Setpoint and Deployed AS.
    Active only when SIMULATE_AS_DISPATCH_EXECUTION is True.
    """
    # --- Conditional Activation ---
    if not getattr(m, 'SIMULATE_AS_DISPATCH_EXECUTION', False) or \
       not getattr(m, 'CAN_PROVIDE_ANCILLARY_SERVICES', False) or \
       not ENABLE_ELECTROLYZER: # Also requires electrolyzer to be enabled
        return pyo.Constraint.Skip

    # Check if necessary variables exist
    if not hasattr(m, 'pElectrolyzerSetpoint') or not hasattr(m, 'pElectrolyzer'):
         logger.error("Missing pElectrolyzerSetpoint or pElectrolyzer for dispatch definition.")
         return pyo.Constraint.Skip # Or raise error

    # --- Safely sum deployed amounts ---
    total_up_deployed = 0
    total_down_deployed = 0
    # List relevant AS types for summing deployed amounts
    up_services = ['RegUp', 'SR', 'NSR', 'RampUp', 'UncU', 'ThirtyMin', 'ECRS'] # Include all potential up reserves
    down_services = ['RegDown', 'RampDown'] # Include all potential down reserves

    # Sum UP reserves deployed by electrolyzer
    for service in up_services:
         var_name = f"{service}_Electrolyzer_Deployed"
         if hasattr(m, var_name):
              # Check if indexed and index exists
              deployed_var = getattr(m, var_name)
              if deployed_var.is_indexed() and t in deployed_var:
                   total_up_deployed += deployed_var[t]
              elif not deployed_var.is_indexed(): # Should not happen for time series var
                   logger.warning(f"Deployed variable {var_name} is not indexed.")

    # Sum DOWN reserves deployed by electrolyzer
    for service in down_services:
         var_name = f"{service}_Electrolyzer_Deployed"
         if hasattr(m, var_name):
              deployed_var = getattr(m, var_name)
              if deployed_var.is_indexed() and t in deployed_var:
                  total_down_deployed += deployed_var[t]
              elif not deployed_var.is_indexed():
                   logger.warning(f"Deployed variable {var_name} is not indexed.")
    # --- End summing ---

    try:
        # The core relationship: Actual Power = Setpoint - Deployed Up + Deployed Down
        return m.pElectrolyzer[t] == m.pElectrolyzerSetpoint[t] - total_up_deployed + total_down_deployed
    except Exception as e:
        logger.error(f"Error in define_actual_electrolyzer_power_rule @t={t}: {e}")
        raise

