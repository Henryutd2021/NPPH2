# src/constraints.py
import pyomo.environ as pyo
from logging_setup import logger
from config import (ENABLE_H2_STORAGE, ENABLE_STARTUP_SHUTDOWN,
                    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_H2_CAP_FACTOR,
                    ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY, ENABLE_ELECTROLYZER,
                    ENABLE_NUCLEAR_GENERATOR, CAN_PROVIDE_ANCILLARY_SERVICES) # Import new derived flag

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
def steam_balance_rule(m, t):
    """Links total steam production to turbine and HTE electrolyzer use."""
    # This rule inherently requires the generator to be enabled to provide steam
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    try:
        turbine_steam = m.qSteam_Turbine[t] if hasattr(m, 'qSteam_Turbine') else 0
        # HTE steam: Only if HTE is enabled (implies Electrolyzer=True and LTE=False)
        hte_steam = m.qSteam_Electrolyzer[t] if ENABLE_ELECTROLYZER and not m.LTE_MODE and hasattr(m, 'qSteam_Electrolyzer') else 0
        total_steam_available = m.qSteam_Total # Already checked ENABLE_NUCLEAR_GENERATOR

        return turbine_steam + hte_steam == total_steam_available
    except Exception as e:
        logger.error(f"Error in steam_balance rule @t={t}: {e}")
        raise

def power_balance_rule(m, t):
    """Ensures power generation equals consumption + net grid interaction."""
    # This rule must always hold, adapting based on enabled components
    try:
        # --- Power Generation Sources ---
        turbine_power = m.pTurbine[t] if ENABLE_NUCLEAR_GENERATOR and hasattr(m, 'pTurbine') else 0
        battery_discharge = m.BatteryDischarge[t] if ENABLE_BATTERY and hasattr(m, 'BatteryDischarge') else 0

        # --- Power Consumption Sinks ---
        electrolyzer_power = m.pElectrolyzer[t] if ENABLE_ELECTROLYZER and hasattr(m, 'pElectrolyzer') else 0
        battery_charge = m.BatteryCharge[t] if ENABLE_BATTERY and hasattr(m, 'BatteryCharge') else 0

        # Total Generation = Total Consumption + Net Grid Sales (pIES)
        # Rearranged: Generation - Consumption = pIES (Net Grid Exchange)
        return turbine_power + battery_discharge - electrolyzer_power - battery_charge == m.pIES[t]

    except Exception as e:
        logger.error(f"Error in power_balance rule @t={t}: {e}")
        raise

# constant_turbine_power_rule: Logic is correct, applies only when Gen+LTE enabled
def constant_turbine_power_rule(m,t):
    """Fixes turbine power if LTE mode is active."""
    if not (ENABLE_NUCLEAR_GENERATOR and ENABLE_ELECTROLYZER and m.LTE_MODE): return pyo.Constraint.Skip # Only applies if Gen + LTE enabled
    try:
        return m.pTurbine[t] == m.pTurbine_LTE_setpoint
    except Exception as e: logger.error(f"Error in constant_turbine_power rule @t={t}: {e}"); raise

# --- H2 Storage Rules (Unchanged logic, but only active if ENABLE_H2_STORAGE) ---
def h2_storage_balance_adj_rule(m, t): # Renamed from h2_storage_balance_rule in model.py call
     """Adjusted storage balance using H2_to_storage variable."""
     if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
     if t == m.TimePeriods.first():
         charge_term = m.H2_to_storage[t] * m.storage_charge_eff
         # Discharge amount is H2_from_storage, efficiency loss means more is taken from tank
         discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
         return m.H2_storage_level[t] == m.H2_storage_level_initial + charge_term - discharge_term
     else:
         charge_term = m.H2_to_storage[t] * m.storage_charge_eff
         discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
         return m.H2_storage_level[t] == m.H2_storage_level[t-1] + charge_term - discharge_term

def h2_prod_dispatch_rule(m, t):
     """Links hydrogen production to market sales and storage input flow."""
     if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
     # H2 produced = H2 sold directly + H2 flow designated for storage (before eff loss)
     return m.mHydrogenProduced[t] == m.H2_to_market[t] + m.H2_to_storage[t]

def h2_storage_charge_limit_rule(m, t):
    """Limits the flow rate *into* storage (before eff loss)."""
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_to_storage[t] <= m.H2_storage_charge_rate_max

def h2_storage_discharge_limit_rule(m, t):
    """Limits the flow rate *out* of storage."""
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max

def h2_storage_level_max_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] <= m.H2_storage_capacity_max

def h2_storage_level_min_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] >= m.H2_storage_capacity_min

# --- Ramp Rate Rules (Apply only if component enabled) ---
def Electrolyzer_RampUp_rule(m, t):
    if not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        # Ramp limit based on % of optimized max capacity
        return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t):
    if not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        # Ramp limit based on % of optimized max capacity
        return m.pElectrolyzer[t-1] - m.pElectrolyzer[t] <= m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
    except Exception as e: logger.error(f"Error in Electrolyzer_RampDown rule @t={t}: {e}"); raise

def Turbine_RampUp_rule(m, t):
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    if m.LTE_MODE and ENABLE_ELECTROLYZER: return pyo.Constraint.Skip # No ramp if LTE mode active
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t] - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampUp rule @t={t}: {e}"); raise

def Turbine_RampDown_rule(m, t):
    if not ENABLE_NUCLEAR_GENERATOR: return pyo.Constraint.Skip
    if m.LTE_MODE and ENABLE_ELECTROLYZER: return pyo.Constraint.Skip # No ramp if LTE mode active
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t-1] - m.pTurbine[t] <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_RampDown rule @t={t}: {e}"); raise

def Steam_Electrolyzer_Ramp_rule(m, t):
    # Only relevant for HTE mode and if ramp is constrained/costed
    if not (ENABLE_ELECTROLYZER and not m.LTE_MODE): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        # Check if the ramp variables exist (they are created conditionally in model.py)
        if hasattr(m, 'qSteamElectrolyzerRampPos') and hasattr(m, 'qSteamElectrolyzerRampNeg'):
             # Limit the sum of positive and negative ramps (total change)
             return m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t] <= m.Ramp_qSteam_Electrolyzer_limit
        else:
             # If ramp variables don't exist, skip the constraint
             return pyo.Constraint.Skip
    except Exception as e: logger.error(f"Error in Steam_Electrolyzer_Ramp rule @t={t}: {e}"); raise

# --- Production Requirement Rule (Only if ENABLE_H2_CAP_FACTOR) ---
def h2_CapacityFactor_rule(m):
    if not ENABLE_H2_CAP_FACTOR or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    try:
        total_hours = len(m.TimePeriods)
        # Use the optimized electrolyzer capacity value
        max_elec_power_ub = pyo.value(m.pElectrolyzer_max)
        if max_elec_power_ub <= 1e-6: return pyo.Constraint.Skip # Avoid division by zero if capacity is zero

        # Estimate max H2 production rate using efficiency at max capacity breakpoint
        if not hasattr(m, 'pElectrolyzer_efficiency_breakpoints') or not m.pElectrolyzer_efficiency_breakpoints:
            logger.warning("Cannot calculate H2 Capacity Factor: Missing electrolyzer efficiency breakpoints.")
            return pyo.Constraint.Skip
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last() # Assumes last breakpoint corresponds to max capacity

        # Find the efficiency (ke) corresponding to the max power breakpoint
        # Use the inverse efficiency (kg/MWh) precomputed in model.py
        if not hasattr(m, 'ke_H2_inv_values') or max_power_bp not in m.ke_H2_inv_values:
             logger.warning("Cannot calculate H2 Capacity Factor: Missing inverse efficiency values (ke_H2_inv_values).")
             return pyo.Constraint.Skip

        max_h2_rate_kg_per_mwh = pyo.value(m.ke_H2_inv_values[max_power_bp])
        if max_h2_rate_kg_per_mwh < 1e-9: # Check if efficiency is near zero
             logger.warning("Cannot calculate H2 Capacity Factor: Near-zero efficiency at max breakpoint.")
             return pyo.Constraint.Skip

        # Estimate max production rate at the optimized capacity
        # This assumes efficiency at max_elec_power_ub is same as at max_power_bp, which might not be exact if PWL
        max_h2_rate_kg_per_hr_est = max_elec_power_ub * max_h2_rate_kg_per_mwh

        max_potential_h2_kg_total_est = max_h2_rate_kg_per_hr_est * total_hours * (m.delT_minutes / 60.0) # Account for time step duration
        if max_potential_h2_kg_total_est <= 1e-6: return pyo.Constraint.Skip

        # Total actual production must meet the target fraction of potential production
        return sum(m.mHydrogenProduced[t] * (m.delT_minutes / 60.0) for t in m.TimePeriods) >= m.h2_target_capacity_factor * max_potential_h2_kg_total_est
    except Exception as e: logger.error(f"Error in h2_CapacityFactor rule: {e}"); raise

# --- Startup/Shutdown Constraints (Only if ENABLE_STARTUP_SHUTDOWN) ---
def electrolyzer_on_off_logic_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        return m.uElectrolyzer[t] - m.uElectrolyzer_initial == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
    else:
        return m.uElectrolyzer[t] - m.uElectrolyzer[t-1] == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]

# Rule for minimum power when ON (only if SU/SD enabled)
def electrolyzer_min_power_when_on_rule(m, t):
    """Ensures electrolyzer power is >= min power * ON_status."""
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # pElectrolyzer >= pMin * uElectrolyzer
    return m.pElectrolyzer[t] >= m.pElectrolyzer_min * m.uElectrolyzer[t]

# Rule for maximum power when ON (only if SU/SD enabled)
def electrolyzer_max_power_rule(m, t):
    """Ensures electrolyzer power is <= max capacity * ON_status."""
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # pElectrolyzer <= pMax(Var) * uElectrolyzer
    return m.pElectrolyzer[t] <= m.pElectrolyzer_max * m.uElectrolyzer[t]

# Rule for minimum power when SU/SD is DISABLED
def electrolyzer_min_power_sds_disabled_rule(m, t):
    """Ensures electrolyzer power is >= min power (when SU/SD is off)."""
    if ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    # If SU/SD is off, electrolyzer is implicitly always ON if operating
    # Enforce the minimum operating level directly
    return m.pElectrolyzer[t] >= m.pElectrolyzer_min

def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1

def electrolyzer_min_uptime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    min_uptime = m.MinUpTimeElectrolyzer
    if t < min_uptime: return pyo.Constraint.Skip
    start_idx = max(m.TimePeriods.first(), t - min_uptime + 1)
    if start_idx > t : return pyo.Constraint.Skip
    return sum(m.uElectrolyzer[i] for i in range(start_idx, t + 1)) >= min_uptime * m.vElectrolyzerStartup[start_idx]

def electrolyzer_min_downtime_rule(m, t):
    if not ENABLE_STARTUP_SHUTDOWN or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    min_downtime = m.MinDownTimeElectrolyzer
    if t < min_downtime: return pyo.Constraint.Skip
    start_idx = max(m.TimePeriods.first(), t - min_downtime + 1)
    if start_idx > t : return pyo.Constraint.Skip
    return sum((1 - m.uElectrolyzer[i]) for i in range(start_idx, t + 1)) >= min_downtime * m.wElectrolyzerShutdown[start_idx]


# --- Electrolyzer Degradation Tracking Rule (Only if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING) ---
def electrolyzer_degradation_rule(m, t):
    if not ENABLE_ELECTROLYZER_DEGRADATION_TRACKING or not ENABLE_ELECTROLYZER: return pyo.Constraint.Skip
    try:
        relative_load_expr = 0
        max_cap = pyo.value(m.pElectrolyzer_max) # Use optimized value

        if max_cap > 1e-6:
            # Degradation scales with actual load relative to max capacity
            # Use pElectrolyzer (actual power) for degradation calculation
            relative_load_expr = m.pElectrolyzer[t] / max_cap
        else:
             relative_load_expr = 0 # No degradation if capacity is zero

        # Degradation from operation (proportional to load * duration)
        time_factor = m.delT_minutes / 60.0
        degradation_increase = m.DegradationFactorOperation * relative_load_expr * time_factor

        if ENABLE_STARTUP_SHUTDOWN:
            # Add degradation from startups (occurs at the start of hour t if vElectrolyzerStartup[t]=1)
            degradation_increase += m.vElectrolyzerStartup[t] * m.DegradationFactorStartup

        # Return symbolic constraint expression
        if t == m.TimePeriods.first():
            return m.DegradationState[t] == m.DegradationStateInitial + degradation_increase
        else:
            return m.DegradationState[t] == m.DegradationState[t-1] + degradation_increase
    except Exception as e:
        logger.error(f"Error defining electrolyzer_degradation rule @t={t}: {e}")
        return pyo.Constraint.Skip


# --- Ancillary Service Definitions Helper ---
def get_as_components(m, t):
    """Helper to organize all ancillary service components in one place, considering enabled technologies."""
    # Initialize dictionary with zero values
    as_components = {
        'up_reserves_bid_turbine': 0.0, 'down_reserves_bid_turbine': 0.0,
        'up_reserves_bid_h2': 0.0, 'down_reserves_bid_h2': 0.0,
        'up_reserves_bid_battery': 0.0, 'down_reserves_bid_battery': 0.0,
        'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0
    }

    # If system CANNOT provide AS, return the zero dictionary immediately
    if not CAN_PROVIDE_ANCILLARY_SERVICES:
        return as_components

    try:
        up_reserves_turbine, down_reserves_turbine = [], []
        up_reserves_h2, down_reserves_h2 = [], []
        up_reserves_battery, down_reserves_battery = [], []

        # Define the list of internal AS service names used for variables
        internal_as_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']

        for service in internal_as_services:
            # Determine direction (Up or Down reserve)
            is_down_reserve = 'Down' in service or service == 'RegD' # Simple check

            # Turbine Contribution (Only if Gen enabled AND (Elec OR Batt enabled))
            if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY):
                var_name = f"{service}_Turbine"
                if hasattr(m, var_name):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_turbine.append(var)
                    else: up_reserves_turbine.append(var)

            # Electrolyzer Contribution
            if ENABLE_ELECTROLYZER:
                var_name = f"{service}_Electrolyzer"
                if hasattr(m, var_name):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_h2.append(var)
                    else: up_reserves_h2.append(var)

            # Battery Contribution
            if ENABLE_BATTERY:
                var_name = f"{service}_Battery"
                if hasattr(m, var_name):
                    var = getattr(m, var_name)[t]
                    if is_down_reserve: down_reserves_battery.append(var)
                    else: up_reserves_battery.append(var)

        # Calculate sums for each category
        as_components['up_reserves_bid_turbine'] = sum(up_reserves_turbine)
        as_components['down_reserves_bid_turbine'] = sum(down_reserves_turbine)
        as_components['up_reserves_bid_h2'] = sum(up_reserves_h2)
        as_components['down_reserves_bid_h2'] = sum(down_reserves_h2)
        as_components['up_reserves_bid_battery'] = sum(up_reserves_battery)
        as_components['down_reserves_bid_battery'] = sum(down_reserves_battery)

        # Total up/down reserves totals across ALL enabled components
        as_components['up_reserves_bid'] = (as_components['up_reserves_bid_turbine'] +
                                            as_components['up_reserves_bid_h2'] +
                                            as_components['up_reserves_bid_battery'])
        as_components['down_reserves_bid'] = (as_components['down_reserves_bid_turbine'] +
                                              as_components['down_reserves_bid_h2'] +
                                              as_components['down_reserves_bid_battery'])

        return as_components
    except Exception as e:
        logger.error(f"Error in get_as_components helper @t={t}: {e}")
        # Return zeros on error to avoid downstream issues
        # Re-initialize to ensure all keys exist with zero values
        return {
            'up_reserves_bid_turbine': 0.0, 'down_reserves_bid_turbine': 0.0,
            'up_reserves_bid_h2': 0.0, 'down_reserves_bid_h2': 0.0,
            'up_reserves_bid_battery': 0.0, 'down_reserves_bid_battery': 0.0,
            'up_reserves_bid': 0.0, 'down_reserves_bid': 0.0
        }


# ---------------------------------------------------------------------------
# BATTERY STORAGE RULES (Only active if ENABLE_BATTERY)
# ---------------------------------------------------------------------------
def battery_soc_balance_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    # Use the optimized capacity value (Var)
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Calculate initial MWh based on Var
    time_factor = m.delT_minutes / 60.0
    if t == m.TimePeriods.first():
        return m.BatterySOC[t] == initial_soc_mwh + \
               (m.BatteryCharge[t] * m.BatteryChargeEff - m.BatteryDischarge[t] / m.BatteryDischargeEff) * time_factor
    else:
        return m.BatterySOC[t] == m.BatterySOC[t-1] + \
               (m.BatteryCharge[t] * m.BatteryChargeEff - m.BatteryDischarge[t] / m.BatteryDischargeEff) * time_factor

def battery_charge_limit_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    # Limit charge by optimized power rating (Var)
    return m.BatteryCharge[t] <= m.BatteryPower_MW * m.BatteryBinaryCharge[t]

def battery_discharge_limit_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    # Limit discharge by optimized power rating (Var)
    return m.BatteryDischarge[t] <= m.BatteryPower_MW * m.BatteryBinaryDischarge[t]

def battery_binary_exclusivity_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryBinaryCharge[t] + m.BatteryBinaryDischarge[t] <= 1

def battery_soc_max_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    # Limit SOC by optimized capacity (Var)
    return m.BatterySOC[t] <= m.BatteryCapacity_MWh

def battery_soc_min_rule(m, t):
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    # Ensure min SOC is respected (fraction * optimized capacity Var)
    return m.BatterySOC[t] >= m.BatterySOC_min_fraction * m.BatteryCapacity_MWh

def battery_ramp_up_rule(m, t): # Charge ramp
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    # Ramp rate relative to optimized MWh capacity (Var), converted to MW over interval
    time_factor = m.delT_minutes / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip # Avoid division by zero
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryCharge[t] - m.BatteryCharge[t-1] <= ramp_limit_mw

def battery_ramp_down_rule(m, t): # Charge ramp
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = m.delT_minutes / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryCharge[t-1] - m.BatteryCharge[t] <= ramp_limit_mw

def battery_discharge_ramp_up_rule(m, t):
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = m.delT_minutes / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryDischarge[t] - m.BatteryDischarge[t-1] <= ramp_limit_mw

def battery_discharge_ramp_down_rule(m, t):
    if not ENABLE_BATTERY or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    time_factor = m.delT_minutes / 60.0
    if time_factor <= 1e-6: return pyo.Constraint.Skip
    ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
    return m.BatteryDischarge[t-1] - m.BatteryDischarge[t] <= ramp_limit_mw

# --- *** FIX START: Split Cyclic SOC into two constraints *** ---
def battery_cyclic_soc_lower_rule(m):
    """Ensures final SOC is >= initial SOC (within tolerance)."""
    if not ENABLE_BATTERY or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    last_t = m.TimePeriods.last()
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Based on optimized capacity (Var)
    tolerance = 0.01 # MWh tolerance
    return m.BatterySOC[last_t] >= initial_soc_mwh - tolerance

def battery_cyclic_soc_upper_rule(m):
    """Ensures final SOC is <= initial SOC (within tolerance)."""
    if not ENABLE_BATTERY or not pyo.value(m.BatteryRequireCyclicSOC): return pyo.Constraint.Skip
    last_t = m.TimePeriods.last()
    initial_soc_mwh = m.BatterySOC_initial_fraction * m.BatteryCapacity_MWh # Based on optimized capacity (Var)
    tolerance = 0.01 # MWh tolerance
    return m.BatterySOC[last_t] <= initial_soc_mwh + tolerance
# --- *** FIX END *** ---

def battery_power_capacity_link_rule(m):
    """Links battery power rating (Var) to its energy capacity (Var) via P/E ratio (Param)."""
    if not ENABLE_BATTERY: return pyo.Constraint.Skip
    return m.BatteryPower_MW == m.BatteryCapacity_MWh * m.BatteryPowerRatio

def battery_min_cap_rule(m):
     """Enforces minimum battery capacity directly if min_cap > 0."""
     if not ENABLE_BATTERY: return pyo.Constraint.Skip
     # If a minimum is defined (>0), enforce it. Otherwise, capacity >= 0 is handled by NonNegativeReals bounds.
     if hasattr(m, 'BatteryCapacity_min') and pyo.value(m.BatteryCapacity_min) > 1e-6:
          return m.BatteryCapacity_MWh >= m.BatteryCapacity_min
     else:
          return pyo.Constraint.Skip # Default non-negativity applies


# --- Battery Ancillary Service Rules (Only if ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES) ---
def Battery_AS_Pmax_rule(m, t): # Down-regulation capability (Charging headroom)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_battery = as_info['down_reserves_bid_battery']
        # Available charge headroom: Optimized Power (Var) - Current Charge (Var)
        available_charge_headroom = m.BatteryPower_MW - m.BatteryCharge[t]
        return total_down_bid_battery <= available_charge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmax rule @t={t}: {e}"); raise

def Battery_AS_Pmin_rule(m, t): # Up-regulation capability (Discharging headroom)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_battery = as_info['up_reserves_bid_battery']
        # Available discharge headroom: Optimized Power (Var) - Current Discharge (Var)
        available_discharge_headroom = m.BatteryPower_MW - m.BatteryDischarge[t]
        return total_up_bid_battery <= available_discharge_headroom
    except Exception as e: logger.error(f"Error in Battery_AS_Pmin rule @t={t}: {e}"); raise

def Battery_AS_SOC_Up_rule(m, t): # Energy constraint for Up-reg (Discharging)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_battery = as_info['up_reserves_bid_battery']
        # Energy needed = Power * Duration / DischargeEfficiency
        energy_needed = total_up_bid_battery * (m.AS_Duration / m.BatteryDischargeEff if pyo.value(m.BatteryDischargeEff) > 1e-6 else 0)
        # Check against min SOC fraction * optimized capacity (Var)
        min_soc_level = m.BatterySOC_min_fraction * m.BatteryCapacity_MWh
        return m.BatterySOC[t] - energy_needed >= min_soc_level
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Up rule @t={t}: {e}"); raise

def Battery_AS_SOC_Down_rule(m, t): # Energy constraint for Down-reg (Charging)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_battery = as_info['down_reserves_bid_battery']
        # Energy absorbed = Power * Duration * ChargeEfficiency
        energy_absorbed = total_down_bid_battery * m.AS_Duration * m.BatteryChargeEff
        # Check against optimized capacity (Var)
        return m.BatterySOC[t] + energy_absorbed <= m.BatteryCapacity_MWh
    except Exception as e: logger.error(f"Error in Battery_AS_SOC_Down rule @t={t}: {e}"); raise

def Battery_AS_RU_rule(m, t): # Ramp capability for Down-reg (Increasing Charge)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_battery = as_info['down_reserves_bid_battery']
        # Ramp limit based on optimized capacity (Var)
        time_factor = m.delT_minutes / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
        # (Charge[t] + Down_Bid) - Charge[t-1] <= RampLimit
        return (m.BatteryCharge[t] + total_down_bid_battery) - m.BatteryCharge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Battery_AS_RU rule @t={t}: {e}"); raise

def Battery_AS_RD_rule(m, t): # Ramp capability for Up-reg (Increasing Discharge)
    if not (ENABLE_BATTERY and CAN_PROVIDE_ANCILLARY_SERVICES) or t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_battery = as_info['up_reserves_bid_battery']
        # Ramp limit based on optimized capacity (Var)
        time_factor = m.delT_minutes / 60.0
        if time_factor <= 1e-6: return pyo.Constraint.Skip
        ramp_limit_mw = m.BatteryRampRate * m.BatteryCapacity_MWh / time_factor
        # (Discharge[t] + Up_Bid) - Discharge[t-1] <= RampLimit
        return (m.BatteryDischarge[t] + total_up_bid_battery) - m.BatteryDischarge[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Battery_AS_RD rule @t={t}: {e}"); raise


# ---------------------------------------------------------------------------
# ANCILLARY SERVICE PROVISION & LINKING RULES
# ---------------------------------------------------------------------------
def link_total_as_rule(m, t, service_name):
    """Generic rule to link component AS bids to the total bid for a service."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES: return pyo.Constraint.Skip
    try:
        total_var = getattr(m, f"Total_{service_name}", None)
        if total_var is None or not isinstance(total_var, pyo.Var):
             return pyo.Constraint.Skip
        turbine_bid = 0
        if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY) and hasattr(m, f"{service_name}_Turbine"):
            turbine_bid = getattr(m, f"{service_name}_Turbine")[t]
        electro_bid = 0
        if ENABLE_ELECTROLYZER and hasattr(m, f"{service_name}_Electrolyzer"):
            electro_bid = getattr(m, f"{service_name}_Electrolyzer")[t]
        battery_bid = 0
        if ENABLE_BATTERY and hasattr(m, f"{service_name}_Battery"):
             battery_bid = getattr(m, f"{service_name}_Battery")[t]
        return total_var[t] == turbine_bid + electro_bid + battery_bid
    except AttributeError as e:
        logger.debug(f"Attribute error linking service {service_name} at time {t}: {e}")
        return pyo.Constraint.Skip
    except Exception as e:
        logger.error(f"Error in link_total_as_rule for {service_name} @t={t}: {e}")
        raise

# Explicit rules calling the generic linker
def link_Total_RegUp_rule(m, t): return link_total_as_rule(m, t, 'RegUp')
def link_Total_RegDown_rule(m, t): return link_total_as_rule(m, t, 'RegDown')
def link_Total_SR_rule(m, t): return link_total_as_rule(m, t, 'SR')
def link_Total_NSR_rule(m, t): return link_total_as_rule(m, t, 'NSR')
def link_Total_ECRS_rule(m, t): return link_total_as_rule(m, t, 'ECRS')
def link_Total_30Min_rule(m, t): return link_total_as_rule(m, t, 'ThirtyMin')
def link_Total_RampUp_rule(m, t): return link_total_as_rule(m, t, 'RampUp')
def link_Total_RampDown_rule(m, t): return link_total_as_rule(m, t, 'RampDown')
def link_Total_UncU_rule(m, t): return link_total_as_rule(m, t, 'UncU')


# --- Ancillary Service Provision Capability Rules ---
def Turbine_AS_Zero_rule(m, t):
    """Ensures turbine provides no AS if AS capability is disabled OR generator disabled OR in LTE mode."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER):
        zero_as = 0
        internal_as_services = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU']
        for service in internal_as_services:
            var_name = f"{service}_Turbine"
            if hasattr(m, var_name):
                attr = getattr(m, var_name)
                if isinstance(attr, pyo.Var) and attr.is_indexed():
                     zero_as += attr[t]
        return zero_as == 0
    return pyo.Constraint.Skip

def Turbine_AS_Pmax_rule(m, t):
    """Turbine headroom constraint for providing upward reserves."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        up_reserve_turbine = as_info['up_reserves_bid_turbine']
        return m.pTurbine[t] + up_reserve_turbine <= m.pTurbine_max
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t):
    """Turbine footroom constraint for providing downward reserves."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        down_reserve_turbine = as_info['down_reserves_bid_turbine']
        return m.pTurbine[t] - down_reserve_turbine >= m.pTurbine_min
    except Exception as e: logger.error(f"Error in Turbine_AS_Pmin rule @t={t}: {e}"); raise

def Turbine_AS_RU_rule(m, t):
    """Turbine ramp-up capability constraint for upward reserves."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        up_reserve_turbine = as_info['up_reserves_bid_turbine']
        return (m.pTurbine[t] + up_reserve_turbine) - m.pTurbine[t-1] <= m.RU_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RU rule @t={t}: {e}"); raise

def Turbine_AS_RD_rule(m, t):
    """Turbine ramp-down capability constraint for downward reserves."""
    if not CAN_PROVIDE_ANCILLARY_SERVICES or not ENABLE_NUCLEAR_GENERATOR or (m.LTE_MODE and ENABLE_ELECTROLYZER): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        down_reserve_turbine = as_info['down_reserves_bid_turbine']
        return m.pTurbine[t-1] - (m.pTurbine[t] - down_reserve_turbine) <= m.RD_Turbine_hourly
    except Exception as e: logger.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise

# Electrolyzer AS Capability
def Electrolyzer_AS_Pmax_rule(m, t): # Capability to increase load (Down-reserve)
    if not (ENABLE_ELECTROLYZER and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_h2 = as_info['down_reserves_bid_h2']
        max_power_limit = m.pElectrolyzer_max
        if ENABLE_STARTUP_SHUTDOWN:
            max_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_max
        return m.pElectrolyzerSetpoint[t] + total_down_bid_h2 <= max_power_limit
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Capability to decrease load (Up-reserve)
    if not (ENABLE_ELECTROLYZER and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_h2 = as_info['up_reserves_bid_h2']
        min_power_limit = m.pElectrolyzer_min
        if ENABLE_STARTUP_SHUTDOWN:
             min_power_limit = m.uElectrolyzer[t] * m.pElectrolyzer_min
        return m.pElectrolyzerSetpoint[t] - total_up_bid_h2 >= min_power_limit
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise

def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for Down-reserve (Increasing Load)
    if not (ENABLE_ELECTROLYZER and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_down_bid_h2 = as_info['down_reserves_bid_h2']
        ramp_limit_mw = m.RU_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        return (m.pElectrolyzerSetpoint[t] + total_down_bid_h2) - m.pElectrolyzer[t-1] <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for Up-reserve (Decreasing Load)
    if not (ENABLE_ELECTROLYZER and CAN_PROVIDE_ANCILLARY_SERVICES): return pyo.Constraint.Skip
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        as_info = get_as_components(m, t)
        total_up_bid_h2 = as_info['up_reserves_bid_h2']
        ramp_limit_mw = m.RD_Electrolyzer_percent_hourly * m.pElectrolyzer_max
        return m.pElectrolyzer[t-1] - (m.pElectrolyzerSetpoint[t] - total_up_bid_h2) <= ramp_limit_mw
    except Exception as e: logger.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise
