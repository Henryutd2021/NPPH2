import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.common.errors import ApplicationError
from pyomo.util.infeasible import log_infeasible_constraints
import timeit
import logging
import os
import sys # Import sys for exiting

# =============================================================================
# Configuration
# =============================================================================
# SELECT THE TARGET ISO HERE:
TARGET_ISO = 'PJM'  # Options: 'CAISO', 'ERCOT', 'ISONE', 'MISO', 'NYISO', 'PJM', 'SPP'

HOURS_IN_YEAR = 8760
# HOURS_IN_YEAR = 24 * 7 # For testing with a smaller dataset (e.g., 1 week)

# Configure logging
log_filename = f'{TARGET_ISO}_optimization_standardized.log' # New log file name
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w') # Overwrite log file each run

# --- Feature Flags (Global Scope) ---
ENABLE_H2_STORAGE = True
ENABLE_H2_CAP_FACTOR = False # Usually disable if storage is enabled
ENABLE_NONLINEAR_TURBINE_EFF = True # Defined Globally
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING = True
ENABLE_STARTUP_SHUTDOWN = True # Enable Mixed-Integer Programming features

# =============================================================================
# Helper Functions for Piecewise Linearization (Updated for Standard Names)
# =============================================================================
def build_piecewise_constraints(model, component_prefix, input_var_name, output_var_name,
                                breakpoint_set_name, value_param_name, n_segments=3):
    """
    Generic helper to build piecewise linear constraints using SOS2.
    Args:
        model: The Pyomo model.
        component_prefix: String prefix for new vars/constraints (e.g., 'ElectrolyzerPower', 'TurbinePower').
        input_var_name: Name of the input variable (e.g., 'pElectrolyzer', 'qSteam_Turbine').
        output_var_name: Name of the output variable (e.g., 'mHydrogenProduced', 'pTurbine').
        breakpoint_set_name: Name of the Set containing breakpoint values for the input variable.
        value_param_name: Name of the Param (indexed by breakpoints) holding output values at breakpoints.
        n_segments: Number of linear segments.
    """
    logging.info(f"Building piecewise constraints for {output_var_name} vs {input_var_name}...")

    input_var = getattr(model, input_var_name)
    breakpoint_set = getattr(model, breakpoint_set_name)
    value_param = getattr(model, value_param_name)

    # Define lambda variables
    lambda_var_name = f'lambda_{component_prefix}'
    # lambda_var is indexed by model.TimePeriods and breakpoint_set
    lambda_var = pyo.Var(model.TimePeriods, breakpoint_set, within=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    setattr(model, lambda_var_name, lambda_var)

    # Sum constraint
    sos2_sum_rule_name = f'sos2_sum_rule_{component_prefix}'
    def sos2_sum_rule(m, t):
        return sum(lambda_var[t, bp] for bp in breakpoint_set) == 1.0
    setattr(model, sos2_sum_rule_name, pyo.Constraint(model.TimePeriods, rule=sos2_sum_rule))

    # Input variable constraint
    input_link_rule_name = f'input_link_rule_{component_prefix}'
    def input_link_rule(m, t):
        return input_var[t] == sum(lambda_var[t, bp] * bp for bp in breakpoint_set)
    setattr(model, input_link_rule_name, pyo.Constraint(model.TimePeriods, rule=input_link_rule))

    # Output variable constraint
    output_link_rule_name = f'output_link_rule_{component_prefix}'
    output_var = getattr(model, output_var_name) # Get the output variable object
    def output_link_rule(m, t):
        # Ensure the value_param actually contains the output values corresponding to breakpoints
        return output_var[t] == sum(lambda_var[t, bp] * value_param[bp] for bp in breakpoint_set)
    setattr(model, output_link_rule_name, pyo.Constraint(model.TimePeriods, rule=output_link_rule))

    # --- SOS2 constraint (Defined using a rule) ---
    sos2_constr_name = f'sos2_{component_prefix}'
    # Define the rule that generates the list of variables for the SOSConstraint for each time period
    def _sos2_rule(m, t):
        # For a given time 't', the variables in the SOS set are lambda_var[t, bp]
        # where bp iterates over the breakpoint_set
        # The rule itself just returns the list of variables for that index 't'
        return [lambda_var[t, bp] for bp in breakpoint_set]

    # Create the indexed constraint using the rule AND specify sos=2
    setattr(model, sos2_constr_name, pyo.SOSConstraint(model.TimePeriods,
                                                       rule=_sos2_rule,
                                                       sos=2)) # Specify SOS type here

    logging.info(f"Finished piecewise constraints for {output_var_name}.")


# =============================================================================
# Constraint Rule Definitions (Updated & Renamed)
# =============================================================================

# --- Physical System Balance Rules ---
def steam_balance_rule(m, t):
    # Steam balance: Input steam = Turbine steam + Electrolyzer steam
    try: return m.qSteam_Turbine[t] + m.qSteam_Electrolyzer[t] == m.qSteam_Total # qCS -> qSteam_Total
    except Exception as e: logging.error(f"Error in steam_balance rule @t={t}: {e}"); raise

def power_balance_rule(m, t):
    # Power balance: Turbine output = Grid sales/exchange + Electrolyzer consumption
    # pIES: Power to Integrated Energy System (Grid). Positive = Export, Negative = Import (if allowed by bounds)
    try: return m.pIES[t] + m.pElectrolyzer[t] == m.pTurbine[t]
    except Exception as e: logging.error(f"Error in power_balance rule @t={t}: {e}"); raise

# --- Hydrogen Storage Rules (Logic Unchanged, names updated) ---
def h2_storage_balance_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        # Initial balance
        charge_term = (m.mHydrogenProduced[t] - m.H2_to_market[t]) * m.storage_charge_eff
        discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
        return m.H2_storage_level[t] == m.H2_storage_level_initial + charge_term - discharge_term
    else:
        # Subsequent balance
        charge_term = (m.mHydrogenProduced[t] - m.H2_to_market[t]) * m.storage_charge_eff
        discharge_term = (m.H2_from_storage[t] / m.storage_discharge_eff if pyo.value(m.storage_discharge_eff) > 1e-6 else 0)
        return m.H2_storage_level[t] == m.H2_storage_level[t-1] + charge_term - discharge_term

def h2_storage_charge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    # Limit amount charged per hour based on net production going to storage
    return (m.mHydrogenProduced[t] - m.H2_to_market[t]) <= m.H2_storage_charge_rate_max

def h2_storage_discharge_limit_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    # Limit amount discharged per hour
    return m.H2_from_storage[t] <= m.H2_storage_discharge_rate_max

def h2_storage_level_max_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] <= m.H2_storage_capacity_max

def h2_storage_level_min_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    return m.H2_storage_level[t] >= m.H2_storage_capacity_min

def h2_direct_market_link_rule(m, t):
    if not ENABLE_H2_STORAGE: return pyo.Constraint.Skip
    # Cannot send more H2 directly to market than produced
    return m.H2_to_market[t] <= m.mHydrogenProduced[t]


# --- Ramp Rate Rules (Hourly) ---
def Electrolyzer_RampUp_rule(m, t): # Ramp limit on ELECTROLYZER POWER INPUT
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_hourly * m.pElectrolyzer_max # Ramp based on % of Max Power
    except Exception as e: logging.error(f"Error in Electrolyzer_RampUp rule @t={t}: {e}"); raise

def Electrolyzer_RampDown_rule(m, t): # Ramp limit on ELECTROLYZER POWER INPUT
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pElectrolyzer[t-1] - m.pElectrolyzer[t] <= m.RD_Electrolyzer_hourly * m.pElectrolyzer_max # Ramp based on % of Max Power
    except Exception as e: logging.error(f"Error in Electrolyzer_RampDown rule @t={t}: {e}"); raise

def Turbine_RampUp_rule(m, t): # Ramp limit on TURBINE ELECTRICAL OUTPUT
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t] - m.pTurbine[t-1] <= m.RU_Turbine_hourly * m.pTurbine_max # Ramp based on % of Max Power
    except Exception as e: logging.error(f"Error in Turbine_RampUp rule @t={t}: {e}"); raise

def Turbine_RampDown_rule(m, t): # Ramp limit on TURBINE ELECTRICAL OUTPUT
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try: return m.pTurbine[t-1] - m.pTurbine[t] <= m.RD_Turbine_hourly * m.pTurbine_max # Ramp based on % of Max Power
    except Exception as e: logging.error(f"Error in Turbine_RampDown rule @t={t}: {e}"); raise

# Optional: Ramp limit on steam extraction for electrolyzer (LINEARIZED)
def Steam_Electrolyzer_Ramp_rule(m, t):
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    # This rule now uses the helper variables defined in the linearization constraint
    try:
        # The absolute value of the ramp is ramp_pos + ramp_neg
        return m.qSteamElectrolyzerRampPos[t] + m.qSteamElectrolyzerRampNeg[t] <= m.Ramp_qSteam_Electrolyzer_limit
    except Exception as e: logging.error(f"Error in Steam_Electrolyzer_Ramp rule @t={t}: {e}"); raise

# --- Production Requirement Rule (Optional) ---
def h2_CapacityFactor_rule(m):
    if not ENABLE_H2_CAP_FACTOR: return pyo.Constraint.Skip
    try:
        total_hours = len(m.TimePeriods)
        # Requires estimate of H2 production at max power using efficiency parameters
        # CAUTION: Ensure ke_H2_values correctly represents H2 production rate per MW input,
        # or adjust the calculation based on its actual physical meaning.
        # Assuming ke_H2_values[max_bp] is specific energy cons. (MWh/kg), max prod = pMax / ke
        max_power_bp = m.pElectrolyzer_efficiency_breakpoints.last()
        max_elec_power = pyo.value(m.pElectrolyzer_max)
        # This assumes ke_H2_values holds MWh/kg. If it holds kg/MWh, invert the division.
        eff_at_max = pyo.value(m.ke_H2_values.get(max_power_bp, None)) # Get efficiency at max power breakpoint

        if max_elec_power <= 1e-6 or eff_at_max is None or eff_at_max <= 1e-9:
            logging.warning(f"Cannot calculate max potential H2 production (Power={max_elec_power}, Eff={eff_at_max}). Skipping h2_CapacityFactor_rule.")
            return pyo.Constraint.Skip

        # Calculate max H2 production rate (kg/hr) at max power
        max_h2_rate_kg_per_hr = max_elec_power / eff_at_max # kg/hr = MW / (MWh/kg)
        max_potential_h2_kg_total = max_h2_rate_kg_per_hr * total_hours

        if max_potential_h2_kg_total <= 1e-6:
             logging.warning(f"Max potential H2 production is near zero ({max_potential_h2_kg_total} kg). Skipping h2_CapacityFactor_rule.")
             return pyo.Constraint.Skip

        # Ensure total produced hydrogen meets the capacity factor requirement
        return sum(m.mHydrogenProduced[t] for t in m.TimePeriods) >= m.h2_target_capacity_factor * max_potential_h2_kg_total
    except Exception as e: logging.error(f"Error in h2_CapacityFactor rule: {e}"); raise


# --- Startup/Shutdown Constraints (MIP) ---
def electrolyzer_on_off_logic_rule(m, t):
    # Link binary status variable uElectrolyzer to startup/shutdown events
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        # Assume initial state is defined (e.g., m.uElectrolyzer_initial = 1 if starting on)
        return m.uElectrolyzer[t] - m.uElectrolyzer_initial == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]
    else:
        return m.uElectrolyzer[t] - m.uElectrolyzer[t-1] == m.vElectrolyzerStartup[t] - m.wElectrolyzerShutdown[t]

def electrolyzer_min_power_rule(m, t):
    # If on (uElectrolyzer=1), power must be >= min power
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.pElectrolyzer[t] >= m.uElectrolyzer[t] * m.pElectrolyzer_min

def electrolyzer_max_power_rule(m, t):
    # If on (uElectrolyzer=1), power must be <= max power
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.pElectrolyzer[t] <= m.uElectrolyzer[t] * m.pElectrolyzer_max

def electrolyzer_startup_shutdown_exclusivity_rule(m, t):
    # Cannot startup and shutdown in the same period
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    return m.vElectrolyzerStartup[t] + m.wElectrolyzerShutdown[t] <= 1

def electrolyzer_min_uptime_rule(m, t):
    # If it just started up (vElectrolyzerStartup=1), it must stay on for MinUpTime hours
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    min_uptime = m.MinUpTimeElectrolyzer # Parameter: minimum hours to stay on
    if t < min_uptime: return pyo.Constraint.Skip # Not enough history
    # Sum of 'on' status must be >= min_uptime if a startup occurred at the beginning of the window
    expr = sum(m.uElectrolyzer[i] for i in range(t - min_uptime + 1, t + 1)) >= min_uptime * m.vElectrolyzerStartup[t - min_uptime + 1]
    return expr

def electrolyzer_min_downtime_rule(m, t):
    # If it just shut down (wElectrolyzerShutdown=1), it must stay off for MinDownTime hours
    if not ENABLE_STARTUP_SHUTDOWN: return pyo.Constraint.Skip
    min_downtime = m.MinDownTimeElectrolyzer # Parameter: minimum hours to stay off
    if t < min_downtime: return pyo.Constraint.Skip # Not enough history
    # Sum of 'on' status uElectrolyzer[i] must be 0 for i from t-min_downtime+1 to t IF wElectrolyzerShutdown[t-min_downtime+1] == 1
    # Equivalent formulation: Sum of status 'on' <= min_downtime * (1 - shutdown_trigger)
    expr = sum(m.uElectrolyzer[i] for i in range(t - min_downtime + 1, t + 1)) <= min_downtime * (1 - m.wElectrolyzerShutdown[t - min_downtime + 1])
    return expr

# --- Electrolyzer Degradation Tracking Rule ---
def electrolyzer_degradation_rule(m, t):
    if not ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: return pyo.Constraint.Skip
    if t == m.TimePeriods.first():
        # Initial degradation calculation
        degradation_increase = m.uElectrolyzer[t] * m.DegradationFactorOperation * (m.pElectrolyzer[t] / m.pElectrolyzer_max if m.pElectrolyzer_max > 1e-6 else 0) \
                             + m.vElectrolyzerStartup[t] * m.DegradationFactorStartup
        return m.DegradationState[t] == m.DegradationStateInitial + degradation_increase
    else:
        # Subsequent degradation calculation
        degradation_increase = m.uElectrolyzer[t] * m.DegradationFactorOperation * (m.pElectrolyzer[t] / m.pElectrolyzer_max if m.pElectrolyzer_max > 1e-6 else 0) \
                             + m.vElectrolyzerStartup[t] * m.DegradationFactorStartup
        # Optional: Add ramp degradation term using linearized ramp variables
        # if hasattr(m, 'DegradationFactorRamp') and t > m.TimePeriods.first():
        #      degradation_increase += m.DegradationFactorRamp * (m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t])
        return m.DegradationState[t] == m.DegradationState[t-1] + degradation_increase


# --- Ancillary Service Provision Capability Rules (Updated for MIP & Standard Names) ---
def Turbine_AS_Pmax_rule(m, t): # Turbine providing Up-Regulation / Reserves
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Turbine'): up_reserve += m.RegUp_Turbine[t]
        if hasattr(m, 'SR_Turbine'): up_reserve += m.SR_Turbine[t]            # Spin/Sync/RRS/TMSR
        if hasattr(m, 'NSR_Turbine'): up_reserve += m.NSR_Turbine[t]           # NonSpin/Supp/NSR/TMNSR
        if hasattr(m, 'ECRS_Turbine'): up_reserve += m.ECRS_Turbine[t]         # ERCOT ECRS
        if hasattr(m, 'ThirtyMin_Turbine'): up_reserve += m.ThirtyMin_Turbine[t] # PJM SecR / NYISO 30Min / ISONE TMOR / MISO STR
        return m.pTurbine[t] + up_reserve <= m.pTurbine_max
    except Exception as e: logging.error(f"Error in Turbine_AS_Pmax rule @t={t}: {e}"); raise

def Turbine_AS_Pmin_rule(m, t): # Turbine providing Down-Regulation
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Turbine'): down_reserve += m.RegDown_Turbine[t]
        return m.pTurbine[t] - down_reserve >= m.pTurbine_min
    except Exception as e: logging.error(f"Error in Turbine_AS_Pmin rule @t={t}: {e}"); raise

def Turbine_AS_RU_rule(m, t): # Turbine ramp capability for Up-Regulation / Reserves
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Turbine'): up_reserve += m.RegUp_Turbine[t]
        if hasattr(m, 'SR_Turbine'): up_reserve += m.SR_Turbine[t]
        if hasattr(m, 'NSR_Turbine'): up_reserve += m.NSR_Turbine[t]
        if hasattr(m, 'ECRS_Turbine'): up_reserve += m.ECRS_Turbine[t]
        if hasattr(m, 'ThirtyMin_Turbine'): up_reserve += m.ThirtyMin_Turbine[t]
        # Max power output achievable considering reserves must be within hourly ramp-up limit
        return (m.pTurbine[t] + up_reserve) - m.pTurbine[t-1] <= m.RU_Turbine_hourly * m.pTurbine_max
    except Exception as e: logging.error(f"Error in Turbine_AS_RU rule @t={t}: {e}"); raise

def Turbine_AS_RD_rule(m, t): # Turbine ramp capability for Down-Regulation
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Turbine'): down_reserve += m.RegDown_Turbine[t]
        # Min power output achievable considering reserves must be within hourly ramp-down limit
        return m.pTurbine[t-1] - (m.pTurbine[t] - down_reserve) <= m.RD_Turbine_hourly * m.pTurbine_max
    except Exception as e: logging.error(f"Error in Turbine_AS_RD rule @t={t}: {e}"); raise

# --- Electrolyzer AS rules now depend on uElectrolyzer ---
def Electrolyzer_AS_Pmax_rule(m, t): # Electrolyzer providing DOWN-reg (INCREASING load)
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Electrolyzer'): down_reserve += m.RegDown_Electrolyzer[t]

        if not ENABLE_STARTUP_SHUTDOWN: # Original rule if not MIP
            return m.pElectrolyzer[t] + down_reserve <= m.pElectrolyzer_max
        else: # If MIP: Can only provide down-reg if ON. Max power is pElectrolyzer_max.
            # Max power achievable = pElectrolyzer_max. Also, reserve is only possible if uElectrolyzer=1.
            return m.pElectrolyzer[t] + down_reserve <= m.uElectrolyzer[t] * m.pElectrolyzer_max
    except Exception as e: logging.error(f"Error in Electrolyzer_AS_Pmax rule @t={t}: {e}"); raise

def Electrolyzer_AS_Pmin_rule(m, t): # Electrolyzer providing UP-reg (DECREASING load)
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Electrolyzer'): up_reserve += m.RegUp_Electrolyzer[t]
        if hasattr(m, 'SR_Electrolyzer'): up_reserve += m.SR_Electrolyzer[t]
        if hasattr(m, 'NSR_Electrolyzer'): up_reserve += m.NSR_Electrolyzer[t]
        if hasattr(m, 'ECRS_Electrolyzer'): up_reserve += m.ECRS_Electrolyzer[t]
        if hasattr(m, 'ThirtyMin_Electrolyzer'): up_reserve += m.ThirtyMin_Electrolyzer[t]

        if not ENABLE_STARTUP_SHUTDOWN: # Original rule if not MIP
             return m.pElectrolyzer[t] - up_reserve >= m.pElectrolyzer_min
        else: # If MIP: Can only provide up-reg if ON. Min power is pElectrolyzer_min.
            # Min power achievable = pElectrolyzer_min. Reserve only possible if uElectrolyzer=1.
            return m.pElectrolyzer[t] - up_reserve >= m.uElectrolyzer[t] * m.pElectrolyzer_min
    except Exception as e: logging.error(f"Error in Electrolyzer_AS_Pmin rule @t={t}: {e}"); raise

# Ramp rules for AS consider on/off status implicitly via pElectrolyzer bounds/values
def Electrolyzer_AS_RU_rule(m, t): # Ramp capability for DOWN-reg (INCREASING load)
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        down_reserve = 0
        if hasattr(m, 'RegDown_Electrolyzer'): down_reserve += m.RegDown_Electrolyzer[t]
        # Check if the potential load increase is within ramp-up capability
        # This implicitly handles uElectrolyzer=0 case as pElectrolyzer[t] and down_reserve[t] would be 0 (or constrained to 0).
        return (m.pElectrolyzer[t] + down_reserve) - m.pElectrolyzer[t-1] <= m.RU_Electrolyzer_hourly * m.pElectrolyzer_max
    except Exception as e: logging.error(f"Error in Electrolyzer_AS_RU rule @t={t}: {e}"); raise

def Electrolyzer_AS_RD_rule(m, t): # Ramp capability for UP-reg (DECREASING load)
    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
    try:
        up_reserve = 0
        if hasattr(m, 'RegUp_Electrolyzer'): up_reserve += m.RegUp_Electrolyzer[t]
        if hasattr(m, 'SR_Electrolyzer'): up_reserve += m.SR_Electrolyzer[t]
        if hasattr(m, 'NSR_Electrolyzer'): up_reserve += m.NSR_Electrolyzer[t]
        if hasattr(m, 'ECRS_Electrolyzer'): up_reserve += m.ECRS_Electrolyzer[t]
        if hasattr(m, 'ThirtyMin_Electrolyzer'): up_reserve += m.ThirtyMin_Electrolyzer[t]
        # Check if the potential load decrease is within ramp-down capability
        return m.pElectrolyzer[t-1] - (m.pElectrolyzer[t] - up_reserve) <= m.RD_Electrolyzer_hourly * m.pElectrolyzer_max
    except Exception as e: logging.error(f"Error in Electrolyzer_AS_RD rule @t={t}: {e}"); raise

# --- Link Component AS to Total System AS Rules (Logic Unchanged, names updated) ---
def link_Total_RegUp_rule(m, t):
    try:
        regup_turbine = m.RegUp_Turbine[t] if hasattr(m, 'RegUp_Turbine') else 0
        regup_h2 = m.RegUp_Electrolyzer[t] if hasattr(m, 'RegUp_Electrolyzer') else 0
        return m.Total_RegUp[t] == regup_turbine + regup_h2
    except Exception as e: logging.error(f"Error in link_Total_RegUp rule @t={t}: {e}"); raise

def link_Total_RegDown_rule(m, t):
    try:
        regdown_turbine = m.RegDown_Turbine[t] if hasattr(m, 'RegDown_Turbine') else 0
        regdown_h2 = m.RegDown_Electrolyzer[t] if hasattr(m, 'RegDown_Electrolyzer') else 0
        return m.Total_RegDown[t] == regdown_turbine + regdown_h2
    except Exception as e: logging.error(f"Error in link_Total_RegDown rule @t={t}: {e}"); raise

def link_Total_SR_rule(m, t): # Sync/Spin/RRS/TMSR
    try:
        sr_turbine = m.SR_Turbine[t] if hasattr(m, 'SR_Turbine') else 0
        sr_h2 = m.SR_Electrolyzer[t] if hasattr(m, 'SR_Electrolyzer') else 0
        return m.Total_SR[t] == sr_turbine + sr_h2
    except Exception as e: logging.error(f"Error in link_Total_SR rule @t={t}: {e}"); raise

def link_Total_NSR_rule(m, t): # NonSync/Supp/NSR/TMNSR/PrimaryRes
    try:
        nsr_turbine = m.NSR_Turbine[t] if hasattr(m, 'NSR_Turbine') else 0
        nsr_h2 = m.NSR_Electrolyzer[t] if hasattr(m, 'NSR_Electrolyzer') else 0
        return m.Total_NSR[t] == nsr_turbine + nsr_h2
    except Exception as e: logging.error(f"Error in link_Total_NSR rule @t={t}: {e}"); raise

def link_Total_ECRS_rule(m, t): # ERCOT ECRS
    try:
        # Check if Total_ECRS is a Variable (only for ERCOT)
        if isinstance(m.Total_ECRS, pyo.Var) and hasattr(m, 'ECRS_Turbine') and hasattr(m, 'ECRS_Electrolyzer'):
             return m.Total_ECRS[t] == m.ECRS_Turbine[t] + m.ECRS_Electrolyzer[t]
        elif isinstance(m.Total_ECRS, pyo.Var):
             # If it's a Var but components don't exist (shouldn't happen with proper setup), set to 0
             return m.Total_ECRS[t] == 0
        else:
             # If Total_ECRS is a Param (for non-ERCOT), skip the constraint
             return pyo.Constraint.Skip
    except Exception as e: logging.error(f"Error in link_Total_ECRS rule @t={t}: {e}"); raise

def link_Total_30Min_rule(m, t): # PJM SecR / NYISO 30Min / ISONE TMOR / MISO STR
    try:
        # Check if Total_30Min is a Variable (only for relevant ISOs)
        if isinstance(m.Total_30Min, pyo.Var) and hasattr(m, 'ThirtyMin_Turbine') and hasattr(m, 'ThirtyMin_Electrolyzer'):
             return m.Total_30Min[t] == m.ThirtyMin_Turbine[t] + m.ThirtyMin_Electrolyzer[t]
        elif isinstance(m.Total_30Min, pyo.Var):
             return m.Total_30Min[t] == 0 # Should not happen if set up correctly
        else:
             return pyo.Constraint.Skip # Skip if Param
    except Exception as e: logging.error(f"Error in link_Total_30Min rule @t={t}: {e}"); raise

# =============================================================================
# Revenue and Cost Calculation Rules (Updated for Startup Cost & Standard Names)
# =============================================================================

# --- Revenue Components ---
def EnergyRevenue_rule(m):
    # Revenue from selling energy to the grid (pIES > 0)
    # Cost from buying energy from the grid (pIES < 0) - handled if price is LMP
    try: return sum(m.energy_price[t] * m.pIES[t] for t in m.TimePeriods)
    except Exception as e: logging.error(f"Error in EnergyRevenue rule: {e}"); raise

def HydrogenRevenue_rule(m):
    # Revenue from selling hydrogen
    if not ENABLE_H2_STORAGE:
        try: return sum(m.H2_value * m.mHydrogenProduced[t] for t in m.TimePeriods)
        except Exception as e: logging.error(f"Error in HydrogenRevenue (no storage) rule: {e}"); raise
    else:
        # Revenue from H2 sent directly to market + H2 dispatched from storage
        try:
            return sum(m.H2_value * (m.H2_to_market[t] + m.H2_from_storage[t]) for t in m.TimePeriods)
        except Exception as e: logging.error(f"Error in HydrogenRevenue (with storage) rule: {e}"); raise


# --- Ancillary Revenue Functions (Using Standardized Total AS Vars) ---
# These functions define the revenue based on the TOTAL ancillary service provided by the combined system.
# The mapping from ISO service names (e.g., RRS, TMSR) to the model variables (Total_SR)
# happens implicitly through the variable definitions and linking constraints.

def AncillaryRevenue_CAISO_rule(m):
    # CAISO Services: RegUp, RegDown, Spin, NonSpin
    try:
        return sum(((m.p_RegUp_CAISO[t] * m.Total_RegUp[t]) +
                    (m.p_RegDown_CAISO[t] * m.Total_RegDown[t]) +
                    (m.p_Spin_CAISO[t] * m.Total_SR[t]) +       # Spin -> Total_SR
                    (m.p_NonSpin_CAISO[t] * m.Total_NSR[t])     # NonSpin -> Total_NSR
                   ) for t in m.TimePeriods)
    except Exception as e: logging.error(f"Error in AncillaryRevenue_CAISO rule: {e}"); raise

def AncillaryRevenue_ERCOT_rule(m):
    # ERCOT Services: RegUp, RegDown, RRS (->Spin), ECRS, NonSpin
    try:
        revenue = sum(((m.p_RegUp_ERCOT[t] * m.Total_RegUp[t]) +
                       (m.p_RegDown_ERCOT[t] * m.Total_RegDown[t]) +
                       (m.p_RRS_ERCOT[t] * m.Total_SR[t]) +          # RRS -> Total_SR
                       (m.p_ECRS_ERCOT[t] * m.Total_ECRS[t]) +       # ECRS -> Total_ECRS
                       (m.p_NonSpin_ERCOT[t] * m.Total_NSR[t])       # NonSpin -> Total_NSR
                      ) for t in m.TimePeriods)
        return revenue
    except Exception as e: logging.error(f"Error in AncillaryRevenue_ERCOT rule: {e}"); raise

def AncillaryRevenue_ISONE_rule(m):
    # ISONE Services: Reg (Combined), TMSR (->Spin), TMNSR (->NonSpin), TMOR (->30Min)
    try:
        return sum(((m.p_Reg_ISONE[t] * (m.Total_RegUp[t] + m.Total_RegDown[t])) + # Reg -> Total_RegUp/Down
                    (m.p_TMSR_ISONE[t] * m.Total_SR[t]) +       # TMSR -> Total_SR
                    (m.p_TMNSR_ISONE[t] * m.Total_NSR[t]) +     # TMNSR -> Total_NSR
                    (m.p_TMOR_ISONE[t] * m.Total_30Min[t])      # TMOR -> Total_30Min
                   ) for t in m.TimePeriods)
    except Exception as e: logging.error(f"Error in AncillaryRevenue_ISONE_RT rule: {e}"); raise

def AncillaryRevenue_MISO_rule(m):
    # MISO Services: Reg (Combined), Spin, Supp (->NonSpin), STR (->30Min)
    # Includes locational price adders
    try:
        return sum(((m.p_Reg_MISO[t] * (m.Total_RegUp[t] + m.Total_RegDown[t])) + m.loc_Reg_MISO[t] + # Reg -> Total_RegUp/Down
                    (m.p_Spin_MISO[t] * m.Total_SR[t]) + m.loc_Spin_MISO[t] +        # Spin -> Total_SR
                    (m.p_Supp_MISO[t] * m.Total_NSR[t]) + m.loc_Supp_MISO[t] +       # Supp -> Total_NSR
                    (m.p_STR_MISO[t] * m.Total_30Min[t]) + m.loc_STR_MISO[t]         # STR -> Total_30Min
                   ) for t in m.TimePeriods)
    except Exception as e: logging.error(f"Error in AncillaryRevenue_MISO rule: {e}"); raise

def AncillaryRevenue_NYISO_rule(m):
    # NYISO Services: Reg Capacity, Spin10 (->Spin), 10MinNonSync (->NonSpin), 30MinRes (->30Min)
    # Includes performance factor and locational adders
    try:
        return sum(((m.p_Reg_NYISO[t] * (m.Total_RegUp[t] + m.Total_RegDown[t])) * m.reg_performance_factor_NYISO[t] + m.loc_Reg_NYISO[t] + # RegC -> Total_RegUp/Down
                    (m.p_Spin_NYISO[t] * m.Total_SR[t]) + m.loc_Spin_NYISO[t] +                             # Spin10 -> Total_SR
                    (m.p_10MinNonSync_NYISO[t] * m.Total_NSR[t]) + m.loc_10MinNonSync_NYISO[t] +           # NSpin10 -> Total_NSR
                    (m.p_30Min_NYISO[t] * m.Total_30Min[t]) + m.loc_30Min_NYISO[t]                          # Res30 -> Total_30Min
                   ) for t in m.TimePeriods)
    except Exception as e: logging.error(f"Error in AncillaryRevenue_NYISO rule: {e}"); raise

def AncillaryRevenue_PJM_rule(m):
    # PJM Services: Reg (Capacity + Performance), Syn (->Spin), PrimaryRes (->NonSpin), 30MinRes (->SecR -> 30Min)
    # Includes mileage, performance score, and locational adders
    try:
        reg_revenue = sum(((m.p_RegCap_PJM[t] * (m.Total_RegUp[t] + m.Total_RegDown[t])) + # Reg Capacity
                           (m.p_RegPerf_PJM[t] * (m.Total_RegUp[t] + m.Total_RegDown[t]) * m.performance_score_PJM[t] * m.mileage_ratio_PJM[t]) + # Reg Performance
                           m.loc_Reg_PJM[t] # Locational Adder for Reg
                          ) for t in m.TimePeriods)
        sr_revenue = sum(((m.p_SR_PJM[t] * m.Total_SR[t]) + m.loc_SR_PJM[t]) for t in m.TimePeriods) # Syn -> Total_SR
        nsr_revenue = sum(((m.p_NSR_PJM[t] * m.Total_NSR[t]) + m.loc_NSR_PJM[t]) for t in m.TimePeriods) # PrimaryRes -> Total_NSR
        secr_revenue = sum(((m.p_SecR_PJM[t] * m.Total_30Min[t]) + m.loc_SecR_PJM[t]) for t in m.TimePeriods) # 30MinRes (SecR) -> Total_30Min
        return reg_revenue + sr_revenue + nsr_revenue + secr_revenue
    except Exception as e: logging.error(f"Error in AncillaryRevenue_PJM rule: {e}"); raise

def AncillaryRevenue_SPP_rule(m):
    # SPP Services: RegUp, RegDown, Spin, Supp (->NonSpin)
    # Includes Mileage/Deployment Factors conceptually
    try:
        # Calculate hourly revenue components applying factors
        total_revenue = 0
        for t in m.TimePeriods:
            # Regulation revenue with potential mileage factor
            reg_up_rev = (m.p_RegUp_SPP[t] * m.Total_RegUp[t]) * getattr(m, 'RT_Mileage_AS_Reg_SPP', 1.0)
            reg_down_rev = (m.p_RegDown_SPP[t] * m.Total_RegDown[t]) * getattr(m, 'RT_Mileage_AS_Reg_SPP', 1.0)
            # Spinning reserve revenue with potential deployment factor
            spin_rev = (m.p_Spin_SPP[t] * m.Total_SR[t]) * getattr(m, 'RT_DeployFactor_SR', 1.0) # Spin -> Total_SR
            # Supplemental reserve revenue with potential deployment factor
            supp_rev = (m.p_Supp_SPP[t] * m.Total_NSR[t]) * getattr(m, 'RT_DeployFactor_NSR', 1.0) # Supp -> Total_NSR

            total_revenue += reg_up_rev + reg_down_rev + spin_rev + supp_rev

        return total_revenue
    except Exception as e: logging.error(f"Error in AncillaryRevenue_SPP rule: {e}"); raise

# --- Cost Components (Updated) ---
def OpexCost_rule(m):
    """Calculates operational costs, including startup."""
    try:
        # VOM Costs
        turbine_vom_cost = sum(m.vom_turbine * m.pTurbine[t] for t in m.TimePeriods)
        electrolyzer_vom_cost = sum(m.vom_electrolyzer * m.pElectrolyzer[t] for t in m.TimePeriods)
        water_cost = sum(m.cost_water_per_kg_h2 * m.mHydrogenProduced[t] for t in m.TimePeriods)

        # Ramping cost (proxy for degradation/stress) - LINEARIZED
        ramping_cost = 0
        if hasattr(m, 'cost_electrolyzer_ramping') and m.cost_electrolyzer_ramping > 1e-9:
             # Use linearized helper variables
             ramping_cost = sum(m.cost_electrolyzer_ramping * (m.pElectrolyzerRampPos[t] + m.pElectrolyzerRampNeg[t])
                               for t in m.TimePeriods if t > m.TimePeriods.first()) # Skip t=1

        # Storage O&M cost (per kg cycled) - LINEARIZED
        storage_cycle_cost = 0
        if ENABLE_H2_STORAGE and hasattr(m, 'vom_storage_cycle') and m.vom_storage_cycle > 1e-9:
            # *** Use linearized helper variable H2_net_to_storage ***
            storage_cycle_cost = sum(m.vom_storage_cycle * (m.H2_net_to_storage[t] + m.H2_from_storage[t])
                                      for t in m.TimePeriods)

        # Startup Cost
        startup_cost = 0
        if ENABLE_STARTUP_SHUTDOWN and hasattr(m, 'cost_startup_electrolyzer'):
            startup_cost = sum(m.cost_startup_electrolyzer * m.vElectrolyzerStartup[t] for t in m.TimePeriods)

        # Degradation Cost (Optional)
        degradation_penalty_cost = 0
        # if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING and hasattr(m, 'cost_degradation_penalty_per_unit'):
        #     degradation_penalty_cost = m.cost_degradation_penalty_per_unit * m.DegradationState[m.TimePeriods.last()]

        total_cost = turbine_vom_cost + electrolyzer_vom_cost + water_cost + ramping_cost + storage_cycle_cost + startup_cost + degradation_penalty_cost
        return total_cost
    except Exception as e: logging.error(f"Error in OpexCost rule: {e}"); raise


# =============================================================================
# Model Creation Function (Updated for Advanced Features & Standard Names)
# =============================================================================

# Placeholder for the electrolyzer-specific piecewise function if needed
# def build_piecewise_efficiency_constraints(model):
#    """ Placeholder: Implement specific piecewise logic for electrolyzer
#        using standardized names (pElectrolyzer, mHydrogenProduced, qSteam_Electrolyzer)
#        and linking them via ke_H2_values and kt_H2_values.
#        This might involve multiple piecewise relationships.
#    """
#    logging.warning("build_piecewise_efficiency_constraints function is not defined. Electrolyzer efficiency is not constrained.")
#    # Example (Conceptual - needs correct implementation based on physics)
#    # build_piecewise_constraints(model, component_prefix='ElectrolyzerProd', ...) # Link pElec to mH2
#    # build_piecewise_constraints(model, component_prefix='ElectrolyzerSteam', ...) # Link pElec to qSteam
#    pass


# *** MODIFIED Function Signature ***
def create_model(data_inputs, target_iso, use_nonlinear_turbine_eff_setting):
    """Creates the Pyomo ConcreteModel based on input data and target ISO."""
    model = pyo.ConcreteModel(f"Optimize_Profit_Standardized_{target_iso}")
    model.TARGET_ISO = target_iso # Store for conditional logic

    logging.info(f"Creating STANDARDIZED model for {target_iso}...")
    # =========================================================================
    # SETS & PARAMETERS
    # =========================================================================
    logging.info("Loading parameters...")

    try: # Wrap parameter loading in try-except
        # Time Periods
        nT = HOURS_IN_YEAR
        model.TimePeriods = pyo.Set(initialize=pyo.RangeSet(1, nT), ordered=True)

        # --- System Parameters (Common & New) ---
        df_system = data_inputs['df_system'] # Assumes a DataFrame loaded from sys_data_advanced.csv
        # Helper function to get values safely
        def get_sys_param(param_name, default=None):
            try:
                val = df_system.loc[param_name, 'Value']
                # Basic type check/conversion if needed
                if pd.isna(val) and default is not None: return default
                # Add specific type conversions if necessary, e.g., int for counts
                if 'MinUpTime' in param_name or 'MinDownTime' in param_name or 'initial_status' in param_name:
                    return int(val)
                return float(val) # Default to float
            except KeyError:
                if default is not None:
                    logging.warning(f"System parameter '{param_name}' not found in input file. Using default value: {default}")
                    return default
                else:
                    logging.error(f"Essential system parameter '{param_name}' not found in input file!")
                    raise ValueError(f"Missing essential parameter: {param_name}")


        model.delT_minutes = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('delT_minutes', 60.0)) # Time step duration
        model.qSteam_Total = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Total_MWth')) # Was qCS

        # Turbine Parameters
        model.convertTtE_const = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4)) # Constant efficiency fallback
        qSteam_Turbine_min_mwth = get_sys_param('qSteam_Turbine_min_MWth')
        qSteam_Turbine_max_mwth = get_sys_param('qSteam_Turbine_max_MWth')
        model.qSteam_Turbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=qSteam_Turbine_min_mwth)
        model.qSteam_Turbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=qSteam_Turbine_max_mwth)
        # Electrical limits derived from thermal if needed, or specified directly
        model.pTurbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_min_MW', qSteam_Turbine_min_mwth * model.convertTtE_const))
        model.pTurbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_max_MW', qSteam_Turbine_max_mwth * model.convertTtE_const))
        model.RU_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_RampUp_Rate_Percent_per_Min', 1.0) * 60 / 100) # Convert %/min to fraction/hr
        model.RD_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_RampDown_Rate_Percent_per_Min', 1.0) * 60 / 100) # Convert %/min to fraction/hr

        # Real-Time Deployment/Performance Factors (Add defaults if not required for all ISOs)
        model.RT_DeployFactor_SR = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('RT_DeployFactor_SR', 1.0)) # Example for Spinning Reserve
        model.RT_DeployFactor_NSR = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('RT_DeployFactor_NSR', 1.0)) # Example for Non-Spinning Reserve
        # Add similar parameters for other reserve types (ECRS, ThirtyMin, etc.) as needed
        if target_iso == 'SPP':
            model.RT_Mileage_AS_Reg_SPP = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('RT_Mileage_AS_Reg_SPP', 1.0)) # Default to 1 if missing

        # Steam to Electrolyzer Ramp Limit
        model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Electrolyzer_Ramp_Limit_MWth_per_Hour', float('inf')))

        # --- Local flag to track if non-linear turbine setup is successful ---
        nonlinear_turbine_enabled_in_model = False # Start assuming it's not enabled

        # *** Use passed argument here ***
        if use_nonlinear_turbine_eff_setting:
            # Parameters for Turbine Piecewise Efficiency
            # ***** YOU NEED TO PROVIDE THESE BREAKPOINT VALUES in sys_data_advanced.csv *****
            try:
                q_bps_str = df_system.loc['qSteam_Turbine_Breakpoints_MWth', 'Value']
                p_vals_str = df_system.loc['pTurbine_Outputs_at_Breakpoints_MW', 'Value']
                q_breakpoints = sorted([float(x.strip()) for x in q_bps_str.split(',')])
                p_values = [float(x.strip()) for x in p_vals_str.split(',')]
                if len(q_breakpoints) != len(p_values):
                    raise ValueError("Turbine breakpoint and output value list lengths differ.")
                # *** Ensure breakpoint set is correctly defined ***
                model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=q_breakpoints, ordered=True)
                pTurbine_vals_at_qTurbine_bp = dict(zip(q_breakpoints, p_values))
                model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize=pTurbine_vals_at_qTurbine_bp)
                # Update min/max power based on piecewise definition if more accurate
                model.pTurbine_min = pyo.Param(mutable=True, initialize=min(p_values))
                model.pTurbine_max = pyo.Param(mutable=True, initialize=max(p_values))
                # *** Set local flag to True only if setup succeeds ***
                nonlinear_turbine_enabled_in_model = True
                logging.info("Successfully loaded and enabled non-linear turbine efficiency.")
            except Exception as e:
                logging.error(f"Error loading/parsing turbine piecewise data: {e}. Check sys_data_advanced.csv. Falling back to constant efficiency.")
                # Keep nonlinear_turbine_enabled_in_model = False (already default)

        # Electrolyzer Parameters
        model.pElectrolyzer_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pElectrolyzer_min_MW'))
        model.pElectrolyzer_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pElectrolyzer_max_MW'))
        model.RU_Electrolyzer_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Electrolyzer_RampUp_Rate_Percent_per_Min', 10.0) * 60 / 100) # %/min to frac/hr
        model.RD_Electrolyzer_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Electrolyzer_RampDown_Rate_Percent_per_Min', 10.0) * 60 / 100) # %/min to frac/hr

        # Electrolyzer Efficiency (Piecewise Parameters)
        # ***** ENSURE THESE ARE LOADED/DEFINED in sys_data_advanced.csv *****
        try:
            p_elec_bps_str = df_system.loc['pElectrolyzer_Breakpoints_MW', 'Value']
            ke_vals_str = df_system.loc['ke_H2_Values_MWh_per_kg', 'Value']
            kt_vals_str = df_system.loc['kt_H2_Values_MWth_per_kg', 'Value']
            p_elec_breakpoints = sorted([float(x.strip()) for x in p_elec_bps_str.split(',')])
            ke_values = [float(x.strip()) for x in ke_vals_str.split(',')]
            kt_values = [float(x.strip()) for x in kt_vals_str.split(',')]
            if not (len(p_elec_breakpoints) == len(ke_values) == len(kt_values)):
                 raise ValueError("Electrolyzer breakpoint and value list lengths differ.")
            model.pElectrolyzer_efficiency_breakpoints = pyo.Set(initialize=p_elec_breakpoints, ordered=True)
            ke_vals_dict = dict(zip(p_elec_breakpoints, ke_values))
            kt_vals_dict = dict(zip(p_elec_breakpoints, kt_values))
            model.ke_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=ke_vals_dict, within=pyo.NonNegativeReals)
            model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_vals_dict, within=pyo.NonNegativeReals)
            logging.info("Loaded electrolyzer piecewise parameters (ke, kt). Ensure build_piecewise_efficiency_constraints uses them correctly.")
        except Exception as e:
            logging.error(f"Error loading/parsing electrolyzer piecewise data: {e}. Cannot run piecewise electrolyzer model.")
            # Consider adding fallback to constant efficiency if essential
            raise ValueError("Failed to load electrolyzer efficiency data.")


        # Grid Export/Import Limits (pIES: Power to Integrated Energy System)
        model.pIES_min = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_min_MW', -model.pTurbine_max)) # Allow import up to turbine max? Default: no import
        model.pIES_max = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_max_MW', model.pTurbine_max)) # Allow export up to turbine max

        # Hydrogen Value & Costs (Updated for Startup)
        model.H2_value = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_value_USD_per_kg'))
        model.vom_turbine = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_turbine_USD_per_MWh', 0))
        model.vom_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_electrolyzer_USD_per_MWh', 0))
        model.cost_water_per_kg_h2 = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_water_USD_per_kg_h2', 0))
        model.cost_electrolyzer_ramping = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_electrolyzer_ramping_USD_per_MW_ramp', 0))

        if ENABLE_STARTUP_SHUTDOWN:
            model.cost_startup_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_startup_electrolyzer_USD_per_startup', 0))
            model.MinUpTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinUpTimeElectrolyzer_hours', 1))
            model.MinDownTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinDownTimeElectrolyzer_hours', 1))
            model.uElectrolyzer_initial = pyo.Param(within=pyo.Binary, initialize=get_sys_param('uElectrolyzer_initial_status_0_or_1', 0)) # Initial on/off state

        # Degradation Parameters
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
            model.DegradationStateInitial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationStateInitial_Units', 0.0)) # Units depend on definition
            model.DegradationFactorOperation = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorOperation_Units_per_Hour_at_MaxLoad', 0.0))
            model.DegradationFactorStartup = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorStartup_Units_per_Startup', 0.0))
            # model.cost_degradation_penalty_per_unit = pyo.Param(...) # Optional penalty cost

        # Hydrogen Storage Parameters
        if ENABLE_H2_STORAGE:
            model.H2_storage_capacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_capacity_max_kg'))
            model.H2_storage_capacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_capacity_min_kg', 0))
            model.H2_storage_level_initial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_level_initial_kg', 0))
            model.H2_storage_charge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_charge_rate_max_kg_per_hr'))
            model.H2_storage_discharge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_discharge_rate_max_kg_per_hr'))
            model.storage_charge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_charge_eff_fraction', 1.0))
            model.storage_discharge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_discharge_eff_fraction', 1.0))
            model.vom_storage_cycle = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_storage_cycle_USD_per_kg_cycled', 0))

        # Hydrogen Capacity Factor Requirement
        if ENABLE_H2_CAP_FACTOR:
            model.h2_target_capacity_factor = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('h2_target_capacity_factor_fraction', 0.0))

        # --- Energy Price ---
        df_price = data_inputs['df_price_hourly']
        if len(df_price) < nT: raise ValueError(f"Energy price data has only {len(df_price)} rows, expected {nT}.")
        # Ensure the column name matches your CSV file
        energy_price_col = 'Price ($/MWh)' # Adjust if your column name is different
        if energy_price_col not in df_price.columns: raise ValueError(f"Energy price column '{energy_price_col}' not found in price file.")
        def energy_price_init(m, t): return df_price[energy_price_col].iloc[t-1]
        model.energy_price = pyo.Param(model.TimePeriods, initialize=energy_price_init)

        # --- Ancillary Service Prices & ISO-Specific Parameters ---
        df_ANSprice = data_inputs['df_ANSprice_hourly']
        if len(df_ANSprice) < nT: raise ValueError(f"ANS price data has only {len(df_ANSprice)} rows, expected {nT}.")
        df_ANSmile = data_inputs.get('df_ANSmile_hourly', None) # Optional mileage/performance data

        # Helper to get price, checking column existence
        def get_param_val(t, col_name, default=0):
             if col_name in df_ANSprice.columns:
                 # Use .iloc[t-1] for 1-based indexing of TimePeriods
                 return df_ANSprice[col_name].iloc[t-1] if t-1 < len(df_ANSprice) else default
             else:
                 #logging.warning(f"ANS Price column '{col_name}' not found for ISO {target_iso}. Using default {default}.")
                 return default

        # Helper to get mileage/performance, checking column existence
        def get_smile_val(t, col_name, default=1.0):
            if df_ANSmile is not None and col_name in df_ANSmile.columns:
                 return df_ANSmile[col_name].iloc[t-1] if t-1 < len(df_ANSmile) else default
            else:
                # Only warn if the file was expected (e.g., for PJM)
                #if target_iso == 'PJM' and df_ANSmile is None:
                #    logging.warning(f"PJM Mileage file not loaded. Using default {default} for '{col_name}'.")
                #elif df_ANSmile is not None and col_name not in df_ANSmile.columns:
                #    logging.warning(f"Mileage column '{col_name}' not found. Using default {default}.")
                return default

        # --- Load AS Prices based on TARGET_ISO ---
        # Ensure column names match exactly what's in your Price_ANS_hourly.csv for each ISO
        if target_iso == 'CAISO':
            model.p_RegUp_CAISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegUp_ASMP', 0))
            model.p_RegDown_CAISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegDown_ASMP', 0))
            model.p_Spin_CAISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_ASMP', 0))
            model.p_NonSpin_CAISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'NonSpin_ASMP', 0))
            # Note: CAISO mileage price is often separate or included in capacity; check data source. Not included here based on original code.
        elif target_iso == 'ERCOT':
            model.p_RegUp_ERCOT = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegUp_MCP', 0))
            model.p_RegDown_ERCOT = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegDown_MCP', 0))
            model.p_RRS_ERCOT = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RRS_MCP', 0)) # RRS -> SR
            model.p_ECRS_ERCOT = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'ECRS_MCP', 0))
            model.p_NonSpin_ERCOT = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'NonSpin_MCP', 0))
        elif target_iso == 'ISONE':
            model.p_Reg_ISONE = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_RMCP', 0)) # Combined Reg price
            model.p_TMSR_ISONE = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'TMSR_SRMCP', 0)) # TMSR -> SR
            model.p_TMNSR_ISONE = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'TMNSR_NSRMCP', 0)) # TMNSR -> NSR
            model.p_TMOR_ISONE = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'TMOR_MCP', 0)) # TMOR -> 30Min
        elif target_iso == 'MISO':
            model.p_Reg_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_MCP', 0)) # Combined Reg price
            model.loc_Reg_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_LOC_Adder', 0.0)) # Example name for adder
            model.p_Spin_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_MCP', 0))
            model.loc_Spin_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_LOC_Adder', 0.0)) # Example name
            model.p_Supp_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Supp_MCP', 0)) # Supp -> NSR
            model.loc_Supp_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Supp_LOC_Adder', 0.0)) # Example name
            model.p_STR_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'STR_MCP', 0)) # STR -> 30Min (verify mapping if needed)
            model.loc_STR_MISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'STR_LOC_Adder', 0.0)) # Example name
        elif target_iso == 'NYISO':
            model.p_Reg_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_MCP', 0)) # Reg Capacity price
            model.reg_performance_factor_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_smile_val(t,'Reg_Perf_Factor', 1.0)) # Example name
            model.loc_Reg_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_LOC_Adder', 0.0)) # Example name
            model.p_Spin_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_MCP', 0)) # Spin10 -> SR
            model.loc_Spin_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_LOC_Adder', 0.0)) # Example name
            model.p_10MinNonSync_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, '10MinNonSync_MCP', 0)) # NSpin10 -> NSR
            model.loc_10MinNonSync_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, '10MinNonSync_LOC_Adder', 0.0)) # Example name
            model.p_30Min_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, '30Min_MCP', 0)) # Res30 -> 30Min
            model.loc_30Min_NYISO = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, '30Min_LOC_Adder', 0.0)) # Example name
        elif target_iso == 'PJM':
            # Assumes PJM has separate Capacity and Performance prices for Reg
            model.p_RegCap_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Regulation Capacity Up ($/MWh)', 0)) # Check exact column name
            model.p_RegPerf_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Regulation Up Mileage ($/MWh)', 0)) # Check exact column name
            model.mileage_ratio_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_smile_val(t, 'MileageUp', 1.0)) # From MileageMultiplier file
            model.performance_score_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_smile_val(t, 'PerformanceScore', 1.0)) # Placeholder if not in data
            model.loc_Reg_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Reg_LOC_Adder', 0.0)) # Example name
            model.p_SR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Synchronized Reserve ($/MWh)', 0)) # Syn -> SR
            model.loc_SR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'SR_LOC_Adder', 0.0)) # Example name
            model.p_NSR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Primary Reserve ($/MWh)', 0)) # Primary Reserve -> NSR
            model.loc_NSR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'NSR_LOC_Adder', 0.0)) # Example name
            model.p_SecR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Thirty Minutes Reserve ($/MWh)', 0)) # 30Min Reserve (SecR) -> 30Min
            model.loc_SecR_PJM = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'SecR_LOC_Adder', 0.0)) # Example name
        elif target_iso == 'SPP':
            model.p_RegUp_SPP = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegUp_MCP', 0))
            model.p_RegDown_SPP = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'RegDown_MCP', 0))
            model.p_Spin_SPP = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Spin_MCP', 0)) # Spin -> SR
            model.p_Supp_SPP = pyo.Param(model.TimePeriods, initialize=lambda m, t: get_param_val(t, 'Supp_MCP', 0)) # Supp -> NSR
        else:
            logging.warning(f"No specific AS parameters loaded for TARGET_ISO='{target_iso}'. Ancillary revenue will be zero.")


    except FileNotFoundError as e:
        logging.error(f"Essential data file not found: {e}. Cannot create model.", exc_info=True)
        raise # Re-raise exception to stop model creation
    except KeyError as e:
        logging.error(f"Missing expected column or parameter index in input data: {e}. Cannot create model.", exc_info=True)
        raise
    except ValueError as e:
         logging.error(f"Data validation error: {e}. Cannot create model.", exc_info=True)
         raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during parameter loading: {e}", exc_info=True)
        raise

    # =========================================================================
    # VARIABLES
    # =========================================================================
    logging.info("Defining variables...")
    try: # Wrap variable definitions
        # --- Physical System Variables ---
        # Steam flow to turbine (MWth)
        model.qSteam_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.qSteam_Turbine_min, model.qSteam_Turbine_max))
        # Turbine Power Output (MW) - Bounds updated if piecewise enabled
        model.pTurbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.pTurbine_min, model.pTurbine_max))

        # Electrolyzer power consumption (MW) - Bounds set by MIP constraints if enabled
        model.pElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds set later if MIP
        # Hydrogen production rate (kg/hr)
        model.mHydrogenProduced = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        # Electrolyzer thermal power consumption (MWth)
        model.qSteam_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

        # --- Grid Interaction Variable ---
        # Power to/from Integrated Energy System (Grid): Positive=Export, Negative=Import
        model.pIES = pyo.Var(model.TimePeriods, within=pyo.Reals, bounds=(model.pIES_min, model.pIES_max))

        # --- Hydrogen Storage Variables (if enabled) ---
        if ENABLE_H2_STORAGE:
            model.H2_storage_level = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.H2_storage_capacity_min, model.H2_storage_capacity_max))
            model.H2_from_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, model.H2_storage_discharge_rate_max)) # H2 dispatched FROM storage
            model.H2_to_market = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # H2 produced and sent DIRECTLY to market/use
            # *** ADDED Helper variable for linearized storage cost ***
            if hasattr(model, 'vom_storage_cycle') and model.vom_storage_cycle > 1e-9:
                 model.H2_net_to_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)


        # --- Startup/Shutdown Variables (MIP) ---
        if ENABLE_STARTUP_SHUTDOWN:
            model.uElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.Binary) # Electrolyzer On/Off status
            model.vElectrolyzerStartup = pyo.Var(model.TimePeriods, within=pyo.Binary) # Startup event
            model.wElectrolyzerShutdown = pyo.Var(model.TimePeriods, within=pyo.Binary) # Shutdown event
        else: # If MIP not enabled, assume electrolyzer is always available within its operational range
            model.pElectrolyzer.setlb(model.pElectrolyzer_min)
            model.pElectrolyzer.setub(model.pElectrolyzer_max)


        # --- Degradation State Variable ---
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
            model.DegradationState = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Tracks accumulated degradation

        # --- Ancillary Service Variables (Component Level) ---
        # Define AS vars for both components, even if one component might not provide a specific service (optimization will set it to 0)
        model.RegUp_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegUp_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegDown_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegDown_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

        # Spinning/Synchronized/Responsive Reserves (SR)
        model.SR_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.SR_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

        # Non-Spinning/Supplemental/Primary Reserves (NSR)
        model.NSR_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.NSR_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

        # ISO-Specific Reserves
        if target_iso == 'ERCOT':
            model.ECRS_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.ECRS_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        # Services like PJM SecR, NYISO 30Min, ISONE TMOR, MISO STR map to 'ThirtyMin'
        if target_iso in ['PJM', 'NYISO', 'ISONE', 'MISO']:
            model.ThirtyMin_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.ThirtyMin_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

        # --- Ancillary Service Variables (Total System Provision) ---
        model.Total_RegUp = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_RegDown = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_SR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)    # Covers Spin/Syn/RRS/TMSR
        model.Total_NSR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)   # Covers NonSpin/Supp/NSR/TMNSR/PrimaryRes

        # Define Total ECRS and 30Min as Vars only for relevant ISOs, otherwise use Params initialized to 0
        if target_iso == 'ERCOT':
            model.Total_ECRS = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        else:
            model.Total_ECRS = pyo.Param(model.TimePeriods, initialize=0.0, within=pyo.NonNegativeReals) # Use Param for non-ERCOT

        if target_iso in ['PJM', 'NYISO', 'ISONE', 'MISO']:
            model.Total_30Min = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0) # Covers SecR/30Min/TMOR/STR
        else:
            model.Total_30Min = pyo.Param(model.TimePeriods, initialize=0.0, within=pyo.NonNegativeReals) # Use Param otherwise

        # *** ADDED Helper Variables for Linearizing Ramping Cost ***
        if hasattr(model, 'cost_electrolyzer_ramping') and model.cost_electrolyzer_ramping > 1e-9:
            model.pElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            model.pElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

        # *** ADDED Helper Variables for Linearizing Steam Ramp Limit ***
        if model.Ramp_qSteam_Electrolyzer_limit < float('inf'):
            model.qSteamElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            model.qSteamElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)


    except Exception as e:
        logging.error(f"Error during variable definition: {e}", exc_info=True)
        raise

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    logging.info("Defining constraints...")
    try: # Wrap constraints
        # --- Physical System Constraints ---
        model.steam_balance_constr = pyo.Constraint(model.TimePeriods, rule=steam_balance_rule)
        model.power_balance_constr = pyo.Constraint(model.TimePeriods, rule=power_balance_rule)
        model.Turbine_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampUp_rule)
        model.Turbine_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampDown_rule)
        model.Electrolyzer_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampUp_rule)
        model.Electrolyzer_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampDown_rule)

        # Optional steam ramp constraint (Linearized)
        if model.Ramp_qSteam_Electrolyzer_limit < float('inf'):
             # Add constraint linking actual ramp to helper variables
             def qSteam_ramp_linearization_rule(m, t):
                 if t == m.TimePeriods.first():
                     return pyo.Constraint.Skip
                 else:
                     return m.qSteam_Electrolyzer[t] - m.qSteam_Electrolyzer[t-1] == m.qSteamElectrolyzerRampPos[t] - m.qSteamElectrolyzerRampNeg[t]
             model.qSteam_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=qSteam_ramp_linearization_rule)

             # Add the actual ramp limit constraint using helper variables
             model.Steam_Electrolyzer_Ramp_constr = pyo.Constraint(model.TimePeriods, rule=Steam_Electrolyzer_Ramp_rule)
             logging.info("Added Linearized qSteam_Electrolyzer Ramp Constraint.")

        # --- Component Efficiency & Production Constraints ---
        # Electrolyzer (Requires custom piecewise function implementation)
        # build_piecewise_efficiency_constraints(model) # CALL YOUR SPECIFIC FUNCTION HERE
        logging.warning("Electrolyzer efficiency relationship (pElectrolyzer, mHydrogenProduced, qSteam_Electrolyzer) is NOT YET CONSTRAINED. Implement 'build_piecewise_efficiency_constraints'. Using simple fallback.")
        # --- Fallback/Placeholder: Assume fixed efficiency if piecewise function not ready ---
        # This is likely incorrect but prevents the model from being completely undefined
        def simple_h2_prod_rule(m, t):
             # Estimate H2 production based on average efficiency (replace with better logic)
             avg_ke = sum(m.ke_H2_values[bp] for bp in m.pElectrolyzer_efficiency_breakpoints) / len(m.pElectrolyzer_efficiency_breakpoints) if len(m.pElectrolyzer_efficiency_breakpoints)>0 else 50
             if avg_ke < 1e-6: avg_ke = 50 # Avoid division by zero
             return m.mHydrogenProduced[t] == m.pElectrolyzer[t] / avg_ke
        def simple_steam_cons_rule(m, t):
             avg_kt = sum(m.kt_H2_values[bp] for bp in m.pElectrolyzer_efficiency_breakpoints) / len(m.pElectrolyzer_efficiency_breakpoints) if len(m.pElectrolyzer_efficiency_breakpoints)>0 else 5
             return m.qSteam_Electrolyzer[t] == m.mHydrogenProduced[t] * avg_kt
        model.simple_h2_prod_constr = pyo.Constraint(model.TimePeriods, rule=simple_h2_prod_rule)
        model.simple_steam_cons_constr = pyo.Constraint(model.TimePeriods, rule=simple_steam_cons_rule)
        # --- End Placeholder ---


        # Turbine (Piecewise or Linear)
        # *** Use the local flag determined during parameter loading ***
        if nonlinear_turbine_enabled_in_model:
            build_piecewise_constraints(model, component_prefix='TurbinePower',
                                        input_var_name='qSteam_Turbine', output_var_name='pTurbine',
                                        breakpoint_set_name='qTurbine_efficiency_breakpoints',
                                        value_param_name='pTurbine_values_at_qTurbine_bp',
                                        n_segments=len(model.qTurbine_efficiency_breakpoints)-1)
            logging.info("Applying non-linear turbine efficiency constraints.")
        else:
            # If linear, define pTurbine using the constant efficiency
            def linear_pTurbine_rule(m,t):
                return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const
            model.linear_pTurbine_constr = pyo.Constraint(model.TimePeriods, rule=linear_pTurbine_rule)
            logging.info("Applying linear turbine efficiency constraint.")


        # --- Hydrogen Production/Storage Constraints ---
        if ENABLE_H2_STORAGE:
            model.h2_storage_balance_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_balance_rule)
            model.h2_storage_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_charge_limit_rule)
            model.h2_storage_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_discharge_limit_rule)
            model.h2_storage_level_max_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_max_rule)
            model.h2_storage_level_min_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_min_rule)
            model.h2_direct_market_link_constr = pyo.Constraint(model.TimePeriods, rule=h2_direct_market_link_rule)
            # *** ADDED Constraint for Linearized Storage Cost ***
            if hasattr(model, 'vom_storage_cycle') and model.vom_storage_cycle > 1e-9:
                def h2_net_to_storage_rule(m, t):
                    # H2_net_to_storage must be at least the net flow (prod - direct_market)
                    # Objective will drive it down to 0 if (prod - direct_market) is negative
                    return m.H2_net_to_storage[t] >= m.mHydrogenProduced[t] - m.H2_to_market[t]
                model.h2_net_to_storage_constr = pyo.Constraint(model.TimePeriods, rule=h2_net_to_storage_rule)

        # Optional Capacity Factor Constraint
        if ENABLE_H2_CAP_FACTOR and not ENABLE_H2_STORAGE: # Usually exclusive with storage
             model.h2_prod_req_constr = pyo.Constraint(rule=h2_CapacityFactor_rule)


        # --- Startup/Shutdown Constraints (MIP) ---
        if ENABLE_STARTUP_SHUTDOWN:
            model.electrolyzer_on_off_logic_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_on_off_logic_rule)
            model.electrolyzer_min_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_rule)
            model.electrolyzer_max_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_max_power_rule)
            model.electrolyzer_startup_shutdown_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_startup_shutdown_exclusivity_rule)
            model.electrolyzer_min_uptime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_uptime_rule)
            model.electrolyzer_min_downtime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_downtime_rule)

        # --- Degradation Tracking Constraint ---
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
            model.electrolyzer_degradation_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_degradation_rule)

        # --- Ancillary Service Capability Constraints (Updated for MIP) ---
        model.Turbine_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmax_rule)
        model.Turbine_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmin_rule)
        model.Turbine_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RU_rule)
        model.Turbine_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RD_rule)
        model.Electrolyzer_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmax_rule) # Uses MIP logic if enabled
        model.Electrolyzer_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmin_rule) # Uses MIP logic if enabled
        model.Electrolyzer_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RU_rule) # Uses MIP logic if enabled
        model.Electrolyzer_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RD_rule) # Uses MIP logic if enabled

        # --- Link Component AS to Total System AS Constraints ---
        model.link_Total_RegUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegUp_rule)
        model.link_Total_RegDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegDown_rule)
        model.link_Total_SR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_SR_rule)
        model.link_Total_NSR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_NSR_rule)
        # Link ISO-specific totals only if they are Variables (i.e., for the relevant ISO)
        if isinstance(model.Total_ECRS, pyo.Var):
             model.link_Total_ECRS_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_ECRS_rule)
        if isinstance(model.Total_30Min, pyo.Var):
             model.link_Total_30Min_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_30Min_rule)

        # *** ADDED Constraint to Linearize Absolute Value for Ramping Cost ***
        if hasattr(model, 'cost_electrolyzer_ramping') and model.cost_electrolyzer_ramping > 1e-9:
            def electrolyzer_ramp_linearization_rule(m, t):
                if t == m.TimePeriods.first():
                    # No ramp cost for the first period, can skip or set helpers to 0
                     return pyo.Constraint.Skip # Cost rule already skips t=1
                else:
                    # Defines the ramp difference using the positive and negative helper variables
                    return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] == m.pElectrolyzerRampPos[t] - m.pElectrolyzerRampNeg[t]
            model.electrolyzer_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_ramp_linearization_rule)


    except Exception as e:
        logging.error(f"Error during constraint definition: {e}", exc_info=True)
        raise

    # =========================================================================
    # OBJECTIVE FUNCTION (Maximize Profit)
    # =========================================================================
    logging.info("Defining objective function (Maximize Profit)...")
    try: # Wrap objective definition
        # --- Define Revenue Components using Expressions ---
        model.EnergyRevenue = pyo.Expression(rule=EnergyRevenue_rule)
        model.HydrogenRevenue = pyo.Expression(rule=HydrogenRevenue_rule)

        # --- Define Ancillary Revenue Expression (Selects rule based on ISO) ---
        if target_iso == 'CAISO': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_CAISO_rule)
        elif target_iso == 'ERCOT': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_ERCOT_rule)
        elif target_iso == 'ISONE': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_ISONE_rule)
        elif target_iso == 'MISO': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_MISO_rule)
        elif target_iso == 'NYISO': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_NYISO_rule)
        elif target_iso == 'PJM': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_PJM_rule)
        elif target_iso == 'SPP': model.AncillaryRevenue = pyo.Expression(rule=AncillaryRevenue_SPP_rule)
        else:
            logging.warning(f"No AncillaryRevenue rule defined for TARGET_ISO='{target_iso}'. Setting to 0.")
            model.AncillaryRevenue = pyo.Expression(initialize=0.0)

        # --- Define Cost Components using Expressions ---
        model.OpexCost = pyo.Expression(rule=OpexCost_rule) # Includes VOM, water, ramping, storage cycle, startup

        # --- Define the Main Objective (Profit = Revenue - Cost) ---
        def TotalProfit_Objective_rule(m):
            return m.EnergyRevenue + m.AncillaryRevenue + m.HydrogenRevenue - m.OpexCost
        model.TotalProfit_Objective = pyo.Objective(rule=TotalProfit_Objective_rule, sense=pyo.maximize)

    except Exception as e:
        logging.error(f"Error during objective definition: {e}", exc_info=True)
        raise

    logging.info("Standardized model created successfully.")
    return model

# =============================================================================
# Data Loading Function (Needs update for new parameter names in CSV)
# =============================================================================

def load_data(target_iso, base_input_dir='./input/hourly_data'):
    """Loads hourly input data from specified files for the target ISO."""
    logging.info(f"Loading hourly data for STANDARDIZED model: {target_iso} from {base_input_dir}")
    iso_path = os.path.join(base_input_dir, target_iso)
    common_path = base_input_dir # Assuming system data is common

    # Define required files - ensure sys_data_advanced.csv contains ALL necessary parameters
    # with names matching those used in get_sys_param() calls within create_model()
    required_files = {
        'df_price_hourly': os.path.join(iso_path, 'Price_hourly.csv'),
        'df_ANSprice_hourly': os.path.join(iso_path, 'Price_ANS_hourly.csv'),
        # --- Ensure this CSV has rows with index matching parameter names used in create_model ---
        # --- e.g., 'qSteam_Total_MWth', 'pTurbine_min_MW', 'H2_value_USD_per_kg', etc. ---
        'df_system': os.path.join(common_path, 'sys_data_advanced.csv'),
    }
    # Optional files (e.g., mileage for PJM)
    if target_iso == 'PJM':
         # Ensure this file exists and has expected columns if needed for PJM revenue
         mileage_file = os.path.join(iso_path, 'MileageMultiplier_hourly.csv')
         if os.path.exists(mileage_file):
             required_files['df_ANSmile_hourly'] = mileage_file
         else:
             logging.warning(f"Optional PJM mileage file not found: {mileage_file}")

    data = {}
    all_files_found = True
    try:
        # --- Load Files ---
        for key, file_path in required_files.items():
            if not os.path.exists(file_path):
                 # Handle missing essential files
                if key in ['df_price_hourly', 'df_ANSprice_hourly', 'df_system']:
                    logging.error(f"Essential data file not found: {file_path}")
                    all_files_found = False
                # Optional files might just issue a warning (handled above for mileage)
                continue # Skip loading if file not found

            logging.info(f"Loading {key} from {file_path}")
            if 'df_system' in key: # System parameters file requires index_col=0
                 data[key] = pd.read_csv(file_path, index_col=0)
                 # Validate required columns/index in system file
                 if 'Value' not in data[key].columns:
                     raise ValueError(f"System data file {file_path} missing 'Value' column.")
            else: # Hourly data files
                 data[key] = pd.read_csv(file_path)

            # Basic validation
            if data[key] is None: raise ValueError(f"DataFrame '{key}' failed to load from {file_path}.")
            if data[key].empty: raise ValueError(f"DataFrame '{key}' loaded from {file_path} is empty.")
            if '_hourly' in key and len(data[key]) < HOURS_IN_YEAR:
                logging.warning(f"DataFrame '{key}' has {len(data[key])} rows, less than expected {HOURS_IN_YEAR}.")

        if not all_files_found:
            logging.error("One or more essential data files were not found.")
            return None # Indicate failure

        # --- Parameter Mapping ---
        # The mapping now happens *inside* the create_model function using get_sys_param()
        # and direct loading of price dataframes. No explicit mapping needed here anymore.
        logging.info("Data files loaded successfully. Parameter extraction occurs during model creation.")

    except FileNotFoundError as e:
        logging.error(f"Error loading data file: {e}.", exc_info=True)
        return None
    except ValueError as e:
         logging.error(f"Data validation error: {e}", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return None

    return data # Return the dictionary of loaded dataframes


# =============================================================================
# Main Execution Block (Updated for Standardized Model & MIP)
# =============================================================================

if __name__ == '__main__':
    start_time = timeit.default_timer()
    print(f"--- Starting Standardized Optimization for {TARGET_ISO} ---")
    logging.info(f"--- Starting Standardized Optimization for {TARGET_ISO} ---")
    logging.info(f"Feature Flags: Storage={ENABLE_H2_STORAGE}, CapFactor={ENABLE_H2_CAP_FACTOR}, NLTurbine={ENABLE_NONLINEAR_TURBINE_EFF}, Degradation={ENABLE_ELECTROLYZER_DEGRADATION_TRACKING}, StartStop={ENABLE_STARTUP_SHUTDOWN}")

    # --- 1. Load Data ---
    input_data = load_data(target_iso=TARGET_ISO) # Loads dataframes

    if input_data is None:
        print(f"Exiting due to data loading errors. Check log file '{log_filename}'.")
        logging.critical("Data loading failed. Exiting.")
        sys.exit(1)

    # --- 2. Create Model ---
    model_instance = None
    try:
        # *** MODIFIED Call: Pass the global flag ***
        model_instance = create_model(input_data, TARGET_ISO, ENABLE_NONLINEAR_TURBINE_EFF)
    except (ValueError, KeyError, AttributeError) as e: # Catch errors during model creation (e.g., missing params)
        print(f"Model creation failed: {e}. Check log file '{log_filename}'.")
        logging.error(f"Model creation failed: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e: # Catch unexpected errors
         print(f"An unexpected error occurred during model creation: {e}. Check log file '{log_filename}'.")
         logging.error(f"Unexpected error during model creation: {e}", exc_info=True)
         sys.exit(1)


    # --- 3. Solve Model ---
    if model_instance:
        logging.info("Solving the STANDARDIZED model ({} hours)...".format(HOURS_IN_YEAR))
        print("Solving the STANDARDIZED model ({} hours)...".format(HOURS_IN_YEAR))
        if ENABLE_STARTUP_SHUTDOWN or ENABLE_NONLINEAR_TURBINE_EFF: # Check global flag here for message
             print("Solver may take longer due to MIP/MINLP features.")

        # Select a solver capable of handling the problem type:
        # - MILP: If only ENABLE_STARTUP_SHUTDOWN is True (and efficiencies are linear)
        # - MIQCP/MINLP: If both MIP and non-linear features are enabled.
        # Choose a solver available in your environment (e.g., cplex, gurobi,cbc, scip, baron, couenne)
        solver_name = 'gurobi' # Gurobi is generally good for MILP/MIQCP
        # solver_name = 'cbc' # CBC is a good open-source option for MILP
        try:
            solver = SolverFactory(solver_name)
            logging.info(f"Using solver: {solver_name}")
        except ApplicationError:
             logging.error(f"Solver '{solver_name}' not found or path incorrect. Please install or check configuration.")
             print(f"Solver '{solver_name}' not found. Please install it or choose a different solver.")
             sys.exit(1)


        # Add solver options if needed (examples)
        # solver.options['timelimit'] = 3600 # e.g., 1 hour time limit
        # solver.options['mipgap'] = 0.01 # e.g., 1% optimality gap for MIP

        try:
            results = solver.solve(model_instance, tee=True) # tee=True shows solver output in console

            # --- 4. Process Results ---
            logging.info("Processing results...")
            print("Processing results...")
            termination_condition = results.solver.termination_condition
            solver_status = results.solver.status
            logging.info(f"Solver Status: {solver_status}, Termination Condition: {termination_condition}")


            # Check for optimal or feasible solution
            solution_found = False
            if solver_status == SolverStatus.ok:
                if termination_condition == TerminationCondition.optimal:
                    logging.info("Solver found an optimal solution.")
                    print("\nSolver found an optimal solution.")
                    solution_found = True
                elif termination_condition == TerminationCondition.feasible:
                     logging.warning("Solver found a feasible solution (may not be optimal due to gap/time limits).")
                     print(f"\nSolver found a feasible solution (may not be optimal). Termination Condition: {termination_condition}")
                     solution_found = True
                # Add other acceptable conditions if applicable (e.g., locallyOptimal for MINLP)

            if solution_found:
                 # Extract Profit and components
                total_profit = pyo.value(model_instance.TotalProfit_Objective)
                energy_revenue = pyo.value(model_instance.EnergyRevenue)
                ancillary_revenue = pyo.value(model_instance.AncillaryRevenue)
                hydrogen_revenue = pyo.value(model_instance.HydrogenRevenue)
                opex_cost = pyo.value(model_instance.OpexCost)

                # Extract startup costs if enabled
                startup_costs_total = 0
                if ENABLE_STARTUP_SHUTDOWN and hasattr(model_instance, 'cost_startup_electrolyzer'):
                    startup_costs_total = sum(pyo.value(model_instance.cost_startup_electrolyzer * model_instance.vElectrolyzerStartup[t])
                                              for t in model_instance.TimePeriods)

                print(f"\n------ Annual Results ({TARGET_ISO} - Standardized) ------")
                print(f"Total Profit: ${total_profit:,.2f}")
                print(f"  Total Revenue: ${energy_revenue + ancillary_revenue + hydrogen_revenue:,.2f}")
                print(f"    Energy Revenue: ${energy_revenue:,.2f}")
                print(f"    Ancillary Revenue: ${ancillary_revenue:,.2f}")
                print(f"    Hydrogen Revenue: ${hydrogen_revenue:,.2f}")
                print(f"  Total Opex Cost: ${opex_cost:,.2f}")
                if ENABLE_STARTUP_SHUTDOWN:
                    print(f"    Startup Costs (Electrolyzer): ${startup_costs_total:,.2f}")
                # Optionally break down Opex further based on OpexCost_rule components

                logging.info(f"Total Profit: {total_profit:.2f}, EnergyRev: {energy_revenue:.2f}, AncillaryRev: {ancillary_revenue:.2f}, HydrogenRev: {hydrogen_revenue:.2f}, OpexCost: {opex_cost:.2f}, StartupCost: {startup_costs_total:.2f}")

                # --- 5. Extract Hourly Data and Save ---
                logging.info("Extracting hourly results...")
                print("Extracting and saving hourly results...")
                result_dir='./Results_Standardized' # Save to a new directory
                if not os.path.exists(result_dir): os.makedirs(result_dir)

                hours = list(model_instance.TimePeriods)
                results_df = pd.DataFrame(index=hours)
                results_df.index.name = 'HourOfYear'

                # Extract key variables
                results_df['pIES_MW'] = [pyo.value(model_instance.pIES[t]) for t in hours]
                results_df['pTurbine_MW'] = [pyo.value(model_instance.pTurbine[t]) for t in hours]
                results_df['qSteam_Turbine_MWth'] = [pyo.value(model_instance.qSteam_Turbine[t]) for t in hours]
                results_df['pElectrolyzer_MW'] = [pyo.value(model_instance.pElectrolyzer[t]) for t in hours]
                results_df['qSteam_Electrolyzer_MWth'] = [pyo.value(model_instance.qSteam_Electrolyzer[t]) for t in hours]
                results_df['mHydrogenProduced_kg_hr'] = [pyo.value(model_instance.mHydrogenProduced[t]) for t in hours]

                if ENABLE_H2_STORAGE:
                     results_df['H2_Storage_Level_kg'] = [pyo.value(model_instance.H2_storage_level[t]) for t in hours]
                     results_df['H2_From_Storage_kg_hr'] = [pyo.value(model_instance.H2_from_storage[t]) for t in hours]
                     results_df['H2_To_Market_Direct_kg_hr'] = [pyo.value(model_instance.H2_to_market[t]) for t in hours]
                     # Calculate net flow to storage for clarity
                     results_df['H2_To_Storage_kg_hr'] = results_df['mHydrogenProduced_kg_hr'] - results_df['H2_To_Market_Direct_kg_hr']
                     # *** ADDED extraction for linearized storage cost helper ***
                     if hasattr(model_instance, 'H2_net_to_storage'):
                         results_df['H2_Net_To_Storage_kg_hr'] = [pyo.value(model_instance.H2_net_to_storage[t]) for t in hours]


                # Add MIP variables if enabled
                if ENABLE_STARTUP_SHUTDOWN:
                     results_df['uElectrolyzer_Status'] = [pyo.value(model_instance.uElectrolyzer[t]) for t in hours]
                     results_df['vElectrolyzer_Startup'] = [pyo.value(model_instance.vElectrolyzerStartup[t]) for t in hours]
                     results_df['wElectrolyzer_Shutdown'] = [pyo.value(model_instance.wElectrolyzerShutdown[t]) for t in hours]

                # Add Degradation State if enabled
                if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                     results_df['DegradationState_Units'] = [pyo.value(model_instance.DegradationState[t]) for t in hours]

                # Add Prices and Hourly Revenue/Cost Components
                results_df['EnergyPrice_USDperMWh'] = [pyo.value(model_instance.energy_price[t]) for t in hours]
                results_df['HourlyEnergyRevenue_USD'] = results_df['EnergyPrice_USDperMWh'] * results_df['pIES_MW']

                if ENABLE_H2_STORAGE:
                    results_df['HourlyHydrogenRevenue_USD'] = pyo.value(model_instance.H2_value) * (results_df['H2_To_Market_Direct_kg_hr'] + results_df['H2_From_Storage_kg_hr'])
                else:
                    results_df['HourlyHydrogenRevenue_USD'] = pyo.value(model_instance.H2_value) * results_df['mHydrogenProduced_kg_hr']

                # Add hourly AS revenue breakdown (requires extracting total AS vars and prices)
                # This provides a more granular view than just averaging the total
                hourly_as_revenue = []
                for t in hours:
                    as_rev_t = 0
                    try: # Calculate based on the specific ISO's rule logic
                        if target_iso == 'CAISO':
                             as_rev_t = (pyo.value(model_instance.p_RegUp_CAISO[t]) * pyo.value(model_instance.Total_RegUp[t]) +
                                         pyo.value(model_instance.p_RegDown_CAISO[t]) * pyo.value(model_instance.Total_RegDown[t]) +
                                         pyo.value(model_instance.p_Spin_CAISO[t]) * pyo.value(model_instance.Total_SR[t]) +
                                         pyo.value(model_instance.p_NonSpin_CAISO[t]) * pyo.value(model_instance.Total_NSR[t]))
                        elif target_iso == 'ERCOT':
                             as_rev_t = (pyo.value(model_instance.p_RegUp_ERCOT[t]) * pyo.value(model_instance.Total_RegUp[t]) +
                                         pyo.value(model_instance.p_RegDown_ERCOT[t]) * pyo.value(model_instance.Total_RegDown[t]) +
                                         pyo.value(model_instance.p_RRS_ERCOT[t]) * pyo.value(model_instance.Total_SR[t]) +
                                         pyo.value(model_instance.p_ECRS_ERCOT[t]) * pyo.value(model_instance.Total_ECRS[t]) + # Total_ECRS is Var only for ERCOT
                                         pyo.value(model_instance.p_NonSpin_ERCOT[t]) * pyo.value(model_instance.Total_NSR[t]))
                        # Add elif blocks for other ISOs similarly...
                        elif target_iso == 'SPP':
                              reg_up_rev = (pyo.value(model_instance.p_RegUp_SPP[t]) * pyo.value(model_instance.Total_RegUp[t])) * pyo.value(getattr(model_instance, 'RT_Mileage_AS_Reg_SPP', 1.0))
                              reg_down_rev = (pyo.value(model_instance.p_RegDown_SPP[t]) * pyo.value(model_instance.Total_RegDown[t])) * pyo.value(getattr(model_instance, 'RT_Mileage_AS_Reg_SPP', 1.0))
                              spin_rev = (pyo.value(model_instance.p_Spin_SPP[t]) * pyo.value(model_instance.Total_SR[t])) * pyo.value(getattr(model_instance, 'RT_DeployFactor_SR', 1.0))
                              supp_rev = (pyo.value(model_instance.p_Supp_SPP[t]) * pyo.value(model_instance.Total_NSR[t])) * pyo.value(getattr(model_instance, 'RT_DeployFactor_NSR', 1.0))
                              as_rev_t = reg_up_rev + reg_down_rev + spin_rev + supp_rev
                        # ... other ISOs
                        elif target_iso == 'PJM': # Example for PJM hourly revenue
                            reg_rev_t = (pyo.value(model_instance.p_RegCap_PJM[t]) * (pyo.value(model_instance.Total_RegUp[t]) + pyo.value(model_instance.Total_RegDown[t]))) + \
                                        (pyo.value(model_instance.p_RegPerf_PJM[t]) * (pyo.value(model_instance.Total_RegUp[t]) + pyo.value(model_instance.Total_RegDown[t])) * \
                                         pyo.value(model_instance.performance_score_PJM[t]) * pyo.value(model_instance.mileage_ratio_PJM[t])) + \
                                         pyo.value(model_instance.loc_Reg_PJM[t])
                            sr_rev_t = (pyo.value(model_instance.p_SR_PJM[t]) * pyo.value(model_instance.Total_SR[t])) + pyo.value(model_instance.loc_SR_PJM[t])
                            nsr_rev_t = (pyo.value(model_instance.p_NSR_PJM[t]) * pyo.value(model_instance.Total_NSR[t])) + pyo.value(model_instance.loc_NSR_PJM[t])
                            secr_rev_t = (pyo.value(model_instance.p_SecR_PJM[t]) * pyo.value(model_instance.Total_30Min[t])) + pyo.value(model_instance.loc_SecR_PJM[t])
                            as_rev_t = reg_rev_t + sr_rev_t + nsr_rev_t + secr_rev_t
                        # Add other ISOs here...

                    except Exception as e:
                        logging.warning(f"Could not calculate hourly AS revenue for t={t}: {e}")
                    hourly_as_revenue.append(as_rev_t)
                results_df['HourlyAncillaryRevenue_USD'] = hourly_as_revenue

                 # Calculate hourly OPEX (approximated - true costs like startup occur discretely)
                hourly_opex = []
                for t in hours:
                     opex_t = (pyo.value(model_instance.vom_turbine * model_instance.pTurbine[t]) +
                               pyo.value(model_instance.vom_electrolyzer * model_instance.pElectrolyzer[t]) +
                               pyo.value(model_instance.cost_water_per_kg_h2 * model_instance.mHydrogenProduced[t]))
                     # Ramping Cost (Linearized)
                     if hasattr(model_instance, 'cost_electrolyzer_ramping') and t > model_instance.TimePeriods.first():
                          opex_t += pyo.value(model_instance.cost_electrolyzer_ramping * (model_instance.pElectrolyzerRampPos[t] + model_instance.pElectrolyzerRampNeg[t]))
                     # Storage Cycle Cost (Linearized)
                     if ENABLE_H2_STORAGE and hasattr(model_instance, 'vom_storage_cycle') and model_instance.vom_storage_cycle > 1e-9:
                          opex_t += pyo.value(model_instance.vom_storage_cycle * (model_instance.H2_net_to_storage[t] + model_instance.H2_from_storage[t]))
                     # Startup Cost
                     if ENABLE_STARTUP_SHUTDOWN and hasattr(model_instance, 'cost_startup_electrolyzer'):
                          opex_t += pyo.value(model_instance.cost_startup_electrolyzer * model_instance.vElectrolyzerStartup[t])
                     hourly_opex.append(opex_t)
                results_df['HourlyOpexCost_USD'] = hourly_opex

                results_df['HourlyProfit_USD'] = results_df['HourlyEnergyRevenue_USD'] + results_df['HourlyAncillaryRevenue_USD'] + results_df['HourlyHydrogenRevenue_USD'] - results_df['HourlyOpexCost_USD']

                results_df = results_df.round(4)

                # Save detailed results
                output_filename = f'{TARGET_ISO}_Hourly_Results_Standardized.csv'
                results_path = os.path.join(result_dir, output_filename)
                try:
                    results_df.to_csv(results_path)
                    logging.info(f"Saved detailed hourly results to {results_path}")
                    print(f"\nSaved detailed hourly results to {results_path}")
                except Exception as e:
                     logging.error(f"Failed to save results to CSV: {e}", exc_info=True)
                     print(f"Error: Failed to save results to {results_path}")


                # Print Summary Stats
                print(f"\n------ Hourly Summary Stats ({TARGET_ISO} - Standardized) ------")
                print(f"Avg pIES (Grid Exchange): {results_df['pIES_MW'].mean():.2f} MW")
                print(f"Avg pTurbine: {results_df['pTurbine_MW'].mean():.2f} MW")
                print(f"Avg pElectrolyzer: {results_df['pElectrolyzer_MW'].mean():.2f} MW")
                print(f"Avg mHydrogenProduced: {results_df['mHydrogenProduced_kg_hr'].mean():.2f} kg/hr")
                if ENABLE_H2_STORAGE: print(f"Avg H2 Storage Level: {results_df['H2_Storage_Level_kg'].mean():,.1f} kg")
                if ENABLE_STARTUP_SHUTDOWN: print(f"Total Electrolyzer Startups: {results_df['vElectrolyzer_Startup'].sum():.0f}")
                if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: print(f"Final Degradation State: {results_df['DegradationState_Units'].iloc[-1]:.2f} Units")
                print(f"Avg Hourly Profit: ${results_df['HourlyProfit_USD'].mean():,.2f}")


            # Handle non-optimal terminations
            elif termination_condition == pyo.TerminationCondition.infeasible:
                 logging.error("Solver determined the problem is infeasible.")
                 print("\nSolver determined the problem is infeasible. Check constraints, parameters, and log file.")
                 # Log infeasible constraints (can be verbose)
                 print("Logging infeasible constraints to log file...")
                 logging.info("Attempting to log infeasible constraints...")
                 log_infeasible_constraints(model_instance, log_expression=True, log_variables=True)
                 logging.info("Finished logging infeasible constraints.")
            elif termination_condition == pyo.TerminationCondition.maxTimeLimit:
                 logging.warning("Solver reached the time limit.")
                 print("\nSolver reached the time limit. Solution may be suboptimal or infeasible.")
                 # Optionally try to process suboptimal solution if available (check solver status/results object)
            elif termination_condition == pyo.TerminationCondition.error or solver_status == SolverStatus.error:
                 logging.error("Solver encountered an error.")
                 print("\nSolver encountered an error. Check solver logs and model formulation.")
            elif termination_condition == pyo.TerminationCondition.unbounded:
                 logging.error("Solver determined the problem is unbounded.")
                 print("\nSolver determined the problem is unbounded. Check objective function and constraints.")
            else: # Other non-optimal/error conditions
                 logging.error(f"Solver finished with non-optimal status: {solver_status} and condition: {termination_condition}")
                 print(f"\nSolver finished with non-optimal status: {solver_status} / condition: {termination_condition}. Check log file.")

        except Exception as e:
            logging.error(f"An error occurred during optimization or results processing: {e}", exc_info=True)
            print(f"An error occurred during optimization/results processing: {e}. Check log file '{log_filename}'.")

    # --- End ---
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    logging.info(f'Total Execution Time: {total_time:.2f} seconds')
    print(f'\nTotal Execution Time: {total_time:.2f} seconds')
