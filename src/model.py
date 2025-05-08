# src/model.py
import pyomo.environ as pyo
from logging_setup import logger
import pandas as pd
import numpy as np # Import numpy for isnan check if needed
from config import (
    TARGET_ISO, HOURS_IN_YEAR,
    ENABLE_NUCLEAR_GENERATOR, ENABLE_ELECTROLYZER, ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY,
    ENABLE_H2_STORAGE, ENABLE_H2_CAP_FACTOR, ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_STARTUP_SHUTDOWN,
    CAN_PROVIDE_ANCILLARY_SERVICES,
    SIMULATE_AS_DISPATCH_EXECUTION
)
# Import constraint rules selectively based on enabled features
from constraints import (
    # build_piecewise_constraints is defined locally below
    steam_balance_rule, power_balance_rule,
    constant_turbine_power_rule,
    link_Total_RegUp_rule, link_Total_RegDown_rule, link_Total_SR_rule, link_Total_NSR_rule,
    link_Total_ECRS_rule, link_Total_30Min_rule, link_Total_RampUp_rule, link_Total_RampDown_rule, # Corrected: link_Total_30Min_rule
    link_Total_UncU_rule,link_auxiliary_power_rule,
    Turbine_AS_Zero_rule, Turbine_AS_Pmax_rule, Turbine_AS_Pmin_rule, Turbine_AS_RU_rule, Turbine_AS_RD_rule,
    Electrolyzer_AS_Pmax_rule, Electrolyzer_AS_Pmin_rule, Electrolyzer_AS_RU_rule, Electrolyzer_AS_RD_rule,
    Battery_AS_Pmax_rule, Battery_AS_Pmin_rule, Battery_AS_SOC_Up_rule, Battery_AS_SOC_Down_rule, Battery_AS_RU_rule, Battery_AS_RD_rule,
    Turbine_RampUp_rule, Turbine_RampDown_rule,
    Electrolyzer_RampUp_rule, Electrolyzer_RampDown_rule,
    Steam_Electrolyzer_Ramp_rule,
    electrolyzer_on_off_logic_rule, electrolyzer_min_power_when_on_rule, electrolyzer_max_power_rule,
    electrolyzer_startup_shutdown_exclusivity_rule, electrolyzer_min_uptime_rule, electrolyzer_min_downtime_rule,
    electrolyzer_min_power_sds_disabled_rule, # This rule was adjusted in constraints.py
    electrolyzer_degradation_rule, h2_CapacityFactor_rule,
    h2_prod_dispatch_rule, h2_storage_charge_limit_rule, h2_storage_discharge_limit_rule,
    h2_storage_level_max_rule, h2_storage_level_min_rule, h2_storage_balance_adj_rule,
    battery_soc_balance_rule, battery_charge_limit_rule, battery_discharge_limit_rule,
    battery_binary_exclusivity_rule, battery_soc_max_rule, battery_soc_min_rule,
    battery_ramp_up_rule, battery_ramp_down_rule, battery_discharge_ramp_up_rule,
    battery_discharge_ramp_down_rule,
    battery_cyclic_soc_lower_rule, battery_cyclic_soc_upper_rule,
    battery_power_capacity_link_rule, battery_min_cap_rule,
    link_deployed_to_bid_rule,
    define_actual_electrolyzer_power_rule, # This rule was modified in constraints.py to use _get_symbolic_as_deployed_sum
    # --- NEWLY ADDED IMPORTS from constraints.py ---
    link_setpoint_to_actual_power_if_not_simulating_dispatch_rule,
    electrolyzer_setpoint_min_power_rule
)
# Import revenue rules
from revenue_cost import (
    EnergyRevenue_rule, HydrogenRevenue_rule, AncillaryRevenue_CAISO_rule,
    AncillaryRevenue_ERCOT_rule, AncillaryRevenue_ISONE_rule, AncillaryRevenue_MISO_rule,
    AncillaryRevenue_NYISO_rule, AncillaryRevenue_PJM_rule, AncillaryRevenue_SPP_rule,
    OpexCost_rule,
)

# --- HELPER FUNCTION (get_sys_param) ---
df_system = None # Global to be accessible by get_sys_param

def get_sys_param(param_name, default=None, required=False):
    """Safely gets a parameter value from the system DataFrame, handling type conversions."""
    global df_system
    if df_system is None:
        if required: raise ValueError("df_system not loaded in get_sys_param.")
        return default
    try:
        if param_name not in df_system.index:
            if required: raise ValueError(f"Missing essential system parameter: {param_name}")
            return default if default is not None else None
        val = df_system.loc[param_name, 'Value']
        if pd.isna(val):
            if required: raise ValueError(f"Missing essential system parameter (NaN value): {param_name}")
            return default if default is not None else None
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower == 'true': return True
            elif val_lower == 'false': return False
            elif any(indicator in param_name.lower() for indicator in ['enable', 'require', 'use', 'is_']):
                 logger.warning(f"Parameter '{param_name}' looks boolean but value is '{val}'. Interpreting as False unless explicitly 'true'.")
                 return False
        if any(indicator in param_name for indicator in ['MinUpTime', 'MinDownTime', 'initial_status', 'Lifetime_years', 'plant_lifetime_years', 'hours']):
            try: return int(float(val))
            except (ValueError, TypeError) as e:
                logger.error(f"Parameter '{param_name}' expected int, got '{val}'. Error: {e}")
                if required: raise
                return default
        if any(indicator in param_name for indicator in ['Breakpoints', 'Values', 'Outputs']):
             return str(val).strip()
        try: return float(val)
        except (ValueError, TypeError):
            logger.debug(f"Parameter '{param_name}' value '{val}' not converted to numeric/bool. Returning as string or default.")
            return val if default is None else default
    except KeyError:
        if required: raise ValueError(f"Missing essential system parameter (KeyError): {param_name}")
        return default if default is not None else None
    except Exception as e:
        logger.error(f"Unexpected error retrieving system parameter '{param_name}': {e}")
        if required: raise
        return default

# --- Piecewise Helper (Local to model.py) ---
# This helper is used directly within model.py during model construction.
# constraints.py also has a copy for its internal use if needed, but this one is for model.py.
def build_piecewise_constraints(model: pyo.ConcreteModel, *, component_prefix: str,
                                input_var_name: str, output_var_name: str,
                                breakpoint_set_name: str, value_param_name: str) -> None:
    """Attach SOS2 piece‑wise linear constraints *in‑place* to `model`."""
    logger.info("Building piece‑wise constraints for %s using SOS2 (model.py local helper)", component_prefix)

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


# --- Main Model Creation Function ---
def create_model(data_inputs, target_iso: str, simulate_dispatch: bool) -> pyo.ConcreteModel:
    """Creates the Pyomo ConcreteModel based on data and configuration flags."""
    model = pyo.ConcreteModel(f"Optimize_Profit_Standardized_{target_iso}")
    model.TARGET_ISO = target_iso

    # Store config flags on model for easier access in rules
    model.ENABLE_NUCLEAR_GENERATOR = ENABLE_NUCLEAR_GENERATOR
    model.ENABLE_ELECTROLYZER = ENABLE_ELECTROLYZER
    model.ENABLE_BATTERY = ENABLE_BATTERY
    model.ENABLE_H2_STORAGE = ENABLE_H2_STORAGE
    model.LTE_MODE = ENABLE_LOW_TEMP_ELECTROLYZER if ENABLE_ELECTROLYZER else False
    model.CAN_PROVIDE_ANCILLARY_SERVICES = CAN_PROVIDE_ANCILLARY_SERVICES
    model.SIMULATE_AS_DISPATCH_EXECUTION = simulate_dispatch # Passed from main.py

    logger.info(f"Creating STANDARDIZED model for {target_iso} with features:")
    logger.info(f"  Nuclear Generator: {model.ENABLE_NUCLEAR_GENERATOR}")
    logger.info(f"  Electrolyzer: {model.ENABLE_ELECTROLYZER} (LTE Mode: {model.LTE_MODE})")
    logger.info(f"  Battery: {model.ENABLE_BATTERY}")
    logger.info(f"  Ancillary Service Capability: {model.CAN_PROVIDE_ANCILLARY_SERVICES}")
    logger.info(f"  >> AS Simulation Mode: {'Dispatch Execution' if model.SIMULATE_AS_DISPATCH_EXECUTION else 'Bidding Strategy'}")
    logger.info(f"  H2 Storage: {model.ENABLE_H2_STORAGE}")
    logger.info(f"  Nonlinear Turbine: {ENABLE_NONLINEAR_TURBINE_EFF and model.ENABLE_NUCLEAR_GENERATOR}") # Use model.attr
    logger.info(f"  Degradation: {ENABLE_ELECTROLYZER_DEGRADATION_TRACKING and model.ENABLE_ELECTROLYZER}")
    logger.info(f"  Startup/Shutdown: {ENABLE_STARTUP_SHUTDOWN and model.ENABLE_ELECTROLYZER}")
    logger.info(f"  H2 Cap Factor: {ENABLE_H2_CAP_FACTOR and model.ENABLE_ELECTROLYZER}")

    # =========================================================================
    # SETS & PARAMETERS
    # =========================================================================
    logger.info("Loading parameters...")
    try:
        if 'df_price_hourly' not in data_inputs or data_inputs['df_price_hourly'] is None:
             raise ValueError("Essential data 'df_price_hourly' not found in data_inputs.")
        nT = len(data_inputs['df_price_hourly'])
        if nT == 0: raise ValueError("Price data is empty.")
        model.TimePeriods = pyo.RangeSet(1, nT)

        if 'df_system' not in data_inputs or data_inputs['df_system'] is None:
             raise ValueError("Essential data 'df_system' not found in data_inputs.")
        global df_system
        df_system = data_inputs['df_system']

        model.delT_minutes = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('delT_minutes', 60.0, required=True))
        model.AS_Duration = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('AS_Duration', 0.25)) # hours
        model.plant_lifetime_years = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('plant_lifetime_years', 30))

        if model.ENABLE_NUCLEAR_GENERATOR: # Use model.attr for consistency
            model.qSteam_Total = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Total_MWth', required=True))
            model.qSteam_Turbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_min_MWth', required=True))
            model.qSteam_Turbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_max_MWth', required=True))
            model.pTurbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_min_MW', required=True))
            p_turb_max_val = get_sys_param('pTurbine_max_MW', required=True)
            model.pTurbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=p_turb_max_val)
            ramp_up_pct_min = get_sys_param('Turbine_RampUp_Rate_Percent_per_Min', 1.0)
            ramp_down_pct_min = get_sys_param('Turbine_RampDown_Rate_Percent_per_Min', 1.0)
            model.RU_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_up_pct_min * 60 / 100 * p_turb_max_val)
            model.RD_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_down_pct_min * 60 / 100 * p_turb_max_val)
            model.vom_turbine = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_turbine_USD_per_MWh', 0))
            model.convertTtE_const = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4))
            model.nonlinear_turbine_enabled_in_model = False # Flag to track if PWL is successfully set up
            if ENABLE_NONLINEAR_TURBINE_EFF: # Check global config flag for intent
                 try:
                    q_bps_str = get_sys_param('qSteam_Turbine_Breakpoints_MWth', required=True)
                    p_vals_str = get_sys_param('pTurbine_Outputs_at_Breakpoints_MW', required=True)
                    if not isinstance(q_bps_str, str) or not isinstance(p_vals_str, str): raise TypeError("Turbine breakpoint/output data not string.")
                    q_breakpoints = sorted([float(x.strip()) for x in q_bps_str.split(',') if x.strip()])
                    p_values = [float(x.strip()) for x in p_vals_str.split(',') if x.strip()]
                    if not q_breakpoints: raise ValueError("Turbine steam breakpoints list is empty.")
                    if len(q_breakpoints) != len(p_values): raise ValueError("Turbine steam breakpoints and power output lists have different lengths.")
                    model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=q_breakpoints, ordered=True)
                    pTurbine_vals_at_qTurbine_bp_dict = dict(zip(q_breakpoints, p_values))
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize=pTurbine_vals_at_qTurbine_bp_dict)
                    model.nonlinear_turbine_enabled_in_model = True
                    logger.info("Enabled non-linear turbine efficiency with provided breakpoints.")
                 except Exception as e:
                    logger.error(f"Error loading non-linear turbine piecewise data: {e}. Falling back to constant efficiency representation if this was intended.")
                    # Fallback to a simple 2-point linear representation based on min/max and const_eff
                    q_turb_min_val_fb = get_sys_param('qSteam_Turbine_min_MWth', required=True) # pyo.value(model.qSteam_Turbine_min)
                    q_turb_max_val_fb = get_sys_param('qSteam_Turbine_max_MWth', required=True) # pyo.value(model.qSteam_Turbine_max)
                    conv_const_fb = get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4) # pyo.value(model.convertTtE_const)
                    model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=[q_turb_min_val_fb, q_turb_max_val_fb], ordered=True)
                    min_p_fallback = q_turb_min_val_fb * conv_const_fb
                    max_p_fallback = q_turb_max_val_fb * conv_const_fb
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize={q_turb_min_val_fb: min_p_fallback, q_turb_max_val_fb: max_p_fallback})
                    model.nonlinear_turbine_enabled_in_model = True # Still uses PWL, just a linear one
                    logger.warning("Fell back to a linear piecewise representation for turbine efficiency using min/max points.")

            if model.ENABLE_ELECTROLYZER and not model.LTE_MODE: # HTE needs steam ramp limit
                model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour', float('inf')))
            if model.ENABLE_ELECTROLYZER and model.LTE_MODE: # LTE has fixed turbine setpoint
                model.pTurbine_LTE_setpoint = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_LTE_setpoint_MW', p_turb_max_val, required=False))

        if model.ENABLE_ELECTROLYZER:
            elec_type_suffix = "LTE" if model.LTE_MODE else "HTE"
            logger.info(f"Loading parameters for {elec_type_suffix} electrolyzer...")
            model.hydrogen_subsidy_per_kg = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('hydrogen_subsidy_per_kg', 0.0))
            model.aux_power_consumption_per_kg_h2 = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('aux_power_consumption_per_kg_h2', 0.0)) # MW_aux per kg/hr H2
            p_elec_max_ub = get_sys_param('pElectrolyzer_max_upper_bound_MW', required=True); p_elec_max_lb = get_sys_param('pElectrolyzer_max_lower_bound_MW', 0.0)
            model.pElectrolyzer_max_upper_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_max_ub); model.pElectrolyzer_max_lower_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_max_lb)
            p_elec_min_val = get_sys_param(f'pElectrolyzer_min_MW_{elec_type_suffix}', default=get_sys_param('pElectrolyzer_min_MW', required=True))
            model.pElectrolyzer_min = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_min_val)
            ramp_up_elec_pct_min = get_sys_param(f'Electrolyzer_RampUp_Rate_Percent_per_Min_{elec_type_suffix}', default=get_sys_param('Electrolyzer_RampUp_Rate_Percent_per_Min', 10.0))
            ramp_down_elec_pct_min = get_sys_param(f'Electrolyzer_RampDown_Rate_Percent_per_Min_{elec_type_suffix}', default=get_sys_param('Electrolyzer_RampDown_Rate_Percent_per_Min', 10.0))
            model.RU_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_up_elec_pct_min * 60 / 100) # Fraction of Pmax per hour
            model.RD_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_down_elec_pct_min * 60 / 100) # Fraction of Pmax per hour
            model.vom_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'vom_electrolyzer_USD_per_MWh_{elec_type_suffix}', default=get_sys_param('vom_electrolyzer_USD_per_MWh', 0)))
            model.cost_water_per_kg_h2 = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_water_USD_per_kg_h2', 0))
            model.cost_electrolyzer_ramping = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'cost_electrolyzer_ramping_USD_per_MW_ramp_{elec_type_suffix}', default=get_sys_param('cost_electrolyzer_ramping_USD_per_MW_ramp', 0)))
            model.cost_electrolyzer_capacity = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'cost_electrolyzer_capacity_USD_per_MW_year_{elec_type_suffix}', default=get_sys_param('cost_electrolyzer_capacity_USD_per_MW_year', 0)))
            try:
                p_elec_bps_str = get_sys_param(f'pElectrolyzer_Breakpoints_MW_{elec_type_suffix}', default=get_sys_param('pElectrolyzer_Breakpoints_MW', required=True))
                ke_vals_str = get_sys_param(f'ke_H2_Values_MWh_per_kg_{elec_type_suffix}', default=get_sys_param('ke_H2_Values_MWh_per_kg', required=True)) # MWh_elec / kg_H2
                if not isinstance(p_elec_bps_str, str) or not isinstance(ke_vals_str, str): raise TypeError(f"Elec breakpoint/ke data not string for {elec_type_suffix}.")
                p_elec_breakpoints = sorted([float(x.strip()) for x in p_elec_bps_str.split(',') if x.strip()])
                ke_values = [float(x.strip()) for x in ke_vals_str.split(',') if x.strip()]
                if not p_elec_breakpoints: raise ValueError(f"Elec power breakpoints list empty for {elec_type_suffix}.")
                if len(p_elec_breakpoints) != len(ke_values): raise ValueError(f"Elec breakpoints/ke lengths differ for {elec_type_suffix}.")
                model.pElectrolyzer_efficiency_breakpoints = pyo.Set(initialize=p_elec_breakpoints, ordered=True)
                ke_vals_dict = dict(zip(p_elec_breakpoints, ke_values)); model.ke_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=ke_vals_dict, within=pyo.NonNegativeReals)
                if not model.LTE_MODE: # HTE specific
                    kt_vals_str = get_sys_param(f'kt_H2_Values_MWth_per_kg_{elec_type_suffix}', default=get_sys_param('kt_H2_Values_MWth_per_kg', required=True)) # MWth_steam / kg_H2
                    if not isinstance(kt_vals_str, str): raise TypeError(f"HTE kt_H2 data not string for {elec_type_suffix}.")
                    kt_values = [float(x.strip()) for x in kt_vals_str.split(',') if x.strip()]
                    if len(p_elec_breakpoints) != len(kt_values): raise ValueError(f"HTE breakpoints/kt lengths differ for {elec_type_suffix}.")
                    kt_vals_dict = dict(zip(p_elec_breakpoints, kt_values)); model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_vals_dict, within=pyo.NonNegativeReals)
                else: # LTE: kt is effectively zero
                    kt_zero_dict = {bp: 0.0 for bp in p_elec_breakpoints}; model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_zero_dict) # Ensure it exists for HTE logic
                logger.info(f"Loaded {elec_type_suffix} electrolyzer piecewise parameters (ke, kt).")
            except Exception as e: logger.error(f"Error loading {elec_type_suffix} electrolyzer piecewise data: {e}."); raise ValueError(f"Failed to load essential {elec_type_suffix} electrolyzer efficiency data.")

            if ENABLE_STARTUP_SHUTDOWN: # Check global config flag
                 model.cost_startup_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'cost_startup_electrolyzer_USD_per_startup_{elec_type_suffix}', default=get_sys_param('cost_startup_electrolyzer_USD_per_startup', 0)))
                 model.MinUpTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param(f'MinUpTimeElectrolyzer_hours_{elec_type_suffix}', default=get_sys_param('MinUpTimeElectrolyzer_hours', 1)))
                 model.MinDownTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param(f'MinDownTimeElectrolyzer_hours_{elec_type_suffix}', default=get_sys_param('MinDownTimeElectrolyzer_hours', 1)))
                 init_status_raw = get_sys_param('uElectrolyzer_initial_status_0_or_1', 0); init_status = 1 if int(float(init_status_raw)) == 1 else 0
                 model.uElectrolyzer_initial = pyo.Param(within=pyo.Binary, initialize=init_status)

            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: # Check global config flag
                 model.DegradationStateInitial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationStateInitial_Units', 0.0))
                 model.DegradationFactorOperation = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_{elec_type_suffix}', default=get_sys_param('DegradationFactorOperation_Units_per_Hour_at_MaxLoad', 0.0)))
                 model.DegradationFactorStartup = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param(f'DegradationFactorStartup_Units_per_Startup_{elec_type_suffix}', default=get_sys_param('DegradationFactorStartup_Units_per_Startup', 0.0)))

            if ENABLE_H2_CAP_FACTOR: # Check global config flag
                model.h2_target_capacity_factor = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('h2_target_capacity_factor_fraction', 0.0)) # e.g., 0.9 for 90%

            model.H2_value = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_value_USD_per_kg', required=True))

            if model.ENABLE_H2_STORAGE: # Use model.attr
                 h2_storage_max = get_sys_param('H2_storage_capacity_max_kg', required=True); h2_storage_min = get_sys_param('H2_storage_capacity_min_kg', 0)
                 model.H2_storage_capacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=h2_storage_max); model.H2_storage_capacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=h2_storage_min)
                 initial_level_raw = get_sys_param('H2_storage_level_initial_kg', h2_storage_min); initial_level = max(h2_storage_min, min(h2_storage_max, float(initial_level_raw)))
                 model.H2_storage_level_initial = pyo.Param(within=pyo.NonNegativeReals, initialize=initial_level)
                 model.H2_storage_charge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_charge_rate_max_kg_per_hr', required=True))
                 model.H2_storage_discharge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_discharge_rate_max_kg_per_hr', required=True))
                 model.storage_charge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_charge_eff_fraction', 1.0))
                 model.storage_discharge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_discharge_eff_fraction', 1.0))
                 model.vom_storage_cycle = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_storage_cycle_USD_per_kg_cycled', 0))

        if model.ENABLE_BATTERY: # Use model.attr
            logger.info("Configuring battery storage parameters...")
            batt_cap_max = get_sys_param('BatteryCapacity_max_MWh', required=True); batt_cap_min = get_sys_param('BatteryCapacity_min_MWh', 0.0)
            model.BatteryCapacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=batt_cap_max); model.BatteryCapacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=batt_cap_min)
            model.BatteryPowerRatio = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryPowerRatio_MW_per_MWh', 0.25, required=True)) # MW/MWh
            model.BatteryChargeEff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryChargeEff', 0.95)); model.BatteryDischargeEff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryDischargeEff', 0.95))
            model.BatterySOC_min_fraction = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatterySOC_min_fraction', 0.10)); model.BatterySOC_initial_fraction = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatterySOC_initial_fraction', 0.50))
            batt_cyclic_val_str = get_sys_param('BatteryRequireCyclicSOC', "True")
            batt_cyclic_val = True if batt_cyclic_val_str else False
            model.BatteryRequireCyclicSOC = pyo.Param(within=pyo.Boolean, initialize=batt_cyclic_val)
            model.BatteryRampRate = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryRampRate_fraction_per_hour', 1.0)) # Fraction of Capacity_MWh per hour
            model.BatteryCapex_USD_per_MWh_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryCapex_USD_per_MWh_year', 0.0))
            model.BatteryCapex_USD_per_MW_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryCapex_USD_per_MW_year', 0.0))
            model.BatteryFixedOM_USD_per_MWh_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryFixedOM_USD_per_MWh_year', 0.0))
            vom_batt_val = get_sys_param('vom_battery_per_mwh_cycled', None) # $/MWh cycled (charge+discharge)/2
            if vom_batt_val is not None: model.vom_battery_per_mwh_cycled = pyo.Param(within=pyo.NonNegativeReals, initialize=float(vom_batt_val))

        # Grid Interaction Params
        p_ies_min_default_val = -(p_turb_max_val) if model.ENABLE_NUCLEAR_GENERATOR and 'p_turb_max_val' in locals() else -1000.0
        p_ies_max_default_val = p_turb_max_val if model.ENABLE_NUCLEAR_GENERATOR and 'p_turb_max_val' in locals() else 1000.0
        model.pIES_min = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_min_MW', p_ies_min_default_val))
        model.pIES_max = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_max_MW', p_ies_max_default_val))

        # Hourly Data Params - Energy Price
        df_price = data_inputs['df_price_hourly']
        if len(df_price) < nT: raise ValueError(f"Energy price data length mismatch ({len(df_price)} vs {nT}).")
        energy_price_col = 'Price ($/MWh)'; # Standardized column name
        if energy_price_col not in df_price.columns: raise ValueError(f"'{energy_price_col}' not found in price data.")
        energy_price_dict = {t_idx: df_price[energy_price_col].iloc[t_idx-1] for t_idx in model.TimePeriods}
        model.energy_price = pyo.Param(model.TimePeriods, initialize=energy_price_dict, within=pyo.Reals)

        # Hourly Data Params - AS Prices/Factors
        df_ANSprice = data_inputs.get('df_ANSprice_hourly', None)
        df_ANSmile = data_inputs.get('df_ANSmile_hourly', None) # Mileage, performance score
        df_ANSdeploy = data_inputs.get('df_ANSdeploy_hourly', None) # Deployment factor
        df_ANSwinrate = data_inputs.get('df_ANSwinrate_hourly', None) # Winning rate

        def get_hourly_param_from_df_model(t_idx, df, col_name, default=0.0, required_param=False):
             """Helper to safely get data from optional hourly dataframes for model params."""
             if df is None:
                 if required_param: raise ValueError(f"Required data file for {col_name} not loaded.")
                 return default
             filename = getattr(df, 'attrs', {}).get('filename', 'DataFrame')
             if col_name in df.columns:
                 try:
                     if not (0 <= t_idx - 1 < len(df)): raise IndexError("Index out of bounds")
                     val = df[col_name].iloc[t_idx-1]
                     return default if pd.isna(val) else val
                 except IndexError:
                      logger.warning(f"Index {t_idx-1} out of bounds for '{col_name}' in {filename} (len {len(df)}). Defaulting.")
                      if required_param: raise ValueError(f"Index error accessing required parameter '{col_name}'.")
                      return default
                 except Exception as e_param:
                      logger.error(f"Error reading '{col_name}' @ index {t_idx-1} from {filename}: {e_param}")
                      if required_param: raise
                      return default
             else:
                 if required_param: raise ValueError(f"Required column '{col_name}' not in {filename}.")
                 return default

        iso_service_map = {
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'], 'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'], 'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'],
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'], 'ISONE': ['Spin10', 'NSpin10', 'OR30'], 'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
        }
        if target_iso not in iso_service_map: raise ValueError(f"AS definitions missing for TARGET_ISO: {target_iso}")
        logger.info(f"Loading AS parameters for {target_iso}...")
        if df_ANSprice is not None: df_ANSprice.attrs = {'filename': 'Price_ANS_hourly.csv'}
        if df_ANSmile is not None: df_ANSmile.attrs = {'filename': 'MileageMultiplier_hourly.csv'}
        if df_ANSdeploy is not None: df_ANSdeploy.attrs = {'filename': 'DeploymentFactor_hourly.csv'}
        if df_ANSwinrate is not None: df_ANSwinrate.attrs = {'filename': 'WinningRate_hourly.csv'}

        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            for service_key_in_map in iso_service_map[target_iso]:
                csv_col_base = f"{service_key_in_map}_{target_iso}" # e.g., RegU_ERCOT
                param_name_base_on_model = f"{service_key_in_map}" # e.g., RegU (used for param prefix)

                is_factor_type_param = any(f in service_key_in_map.lower() for f in ['factor', 'score', 'ratio'])

                # Price (p_*) - Skip for factor-like keys that are not prices themselves (e.g. RegCap, RegPerf are prices)
                if not is_factor_type_param or service_key_in_map in ['RegCap', 'RegPerf']:
                     price_col_name_in_csv = f'p_{csv_col_base}' # e.g., p_RegU_ERCOT
                     param_dict = {t: get_hourly_param_from_df_model(t, df_ANSprice, price_col_name_in_csv, default=0.0) for t in model.TimePeriods}
                     model.add_component(f'p_{param_name_base_on_model}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.Reals))

                # Deploy Factor (deploy_factor_*) - default 0.0 if not applicable/found
                deploy_col_name_in_csv = f'deploy_factor_{csv_col_base}'
                param_dict = {t: get_hourly_param_from_df_model(t, df_ANSdeploy, deploy_col_name_in_csv, default=0.0) for t in model.TimePeriods}
                model.add_component(f'deploy_factor_{param_name_base_on_model}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.PercentFraction))

                # Adder (loc_*) - default 0.0
                loc_col_name_in_csv = f'loc_{csv_col_base}'
                param_dict = {t: get_hourly_param_from_df_model(t, df_ANSprice, loc_col_name_in_csv, default=0.0) for t in model.TimePeriods}
                model.add_component(f'loc_{param_name_base_on_model}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.Reals))

                # Winning Rate (winning_rate_*) - default 1.0 (assume 100% win if not specified)
                win_col_name_in_csv = f'winning_rate_{csv_col_base}'
                param_dict = {t: get_hourly_param_from_df_model(t, df_ANSwinrate, win_col_name_in_csv, default=1.0) for t in model.TimePeriods}
                model.add_component(f'winning_rate_{param_name_base_on_model}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.PercentFraction))

                # Specific factors (mileage, performance)
                if target_iso == 'CAISO' and service_key_in_map in ['RegU', 'RegD']: # CAISO has mileage_factor_RegU_CAISO etc.
                     mileage_col_name_in_csv = f'mileage_factor_{service_key_in_map}_{target_iso}'
                     param_dict = {t: get_hourly_param_from_df_model(t, df_ANSmile, mileage_col_name_in_csv, default=1.0) for t in model.TimePeriods}
                     model.add_component(f'mileage_factor_{service_key_in_map}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
                if target_iso == 'PJM': # PJM has specific named factors
                     if service_key_in_map == 'performance_score':
                          perf_col_name_in_csv = f'performance_score_{target_iso}' # e.g. performance_score_PJM
                          param_dict = {t: get_hourly_param_from_df_model(t, df_ANSmile, perf_col_name_in_csv, default=1.0) for t in model.TimePeriods}
                          model.add_component(f'performance_score_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
                     if service_key_in_map == 'mileage_ratio':
                          mileage_col_name_in_csv = f'mileage_ratio_{target_iso}' # e.g. mileage_ratio_PJM
                          param_dict = {t: get_hourly_param_from_df_model(t, df_ANSmile, mileage_col_name_in_csv, default=1.0) for t in model.TimePeriods}
                          model.add_component(f'mileage_ratio_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
        else: logger.info("Ancillary services disabled by configuration. Skipping AS parameter loading.")

    except Exception as e: logger.error(f"Error during parameter loading: {e}", exc_info=True); raise

    # =========================================================================
    # VARIABLES
    # =========================================================================
    logger.info("Defining variables...")
    try:
        p_ies_min_val = pyo.value(model.pIES_min); p_ies_max_val = pyo.value(model.pIES_max)
        model.pIES = pyo.Var(model.TimePeriods, within=pyo.Reals, bounds=(p_ies_min_val, p_ies_max_val))

        if model.ENABLE_NUCLEAR_GENERATOR:
            q_turb_min_val = pyo.value(model.qSteam_Turbine_min); q_turb_max_val = pyo.value(model.qSteam_Turbine_max)
            p_turb_min_val_loc = pyo.value(model.pTurbine_min); p_turb_max_val_loc = pyo.value(model.pTurbine_max)
            model.qSteam_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(q_turb_min_val, q_turb_max_val))
            model.pTurbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(p_turb_min_val_loc, p_turb_max_val_loc))

        if model.ENABLE_ELECTROLYZER:
            p_elec_max_lb_val = pyo.value(model.pElectrolyzer_max_lower_bound); p_elec_max_ub_val = pyo.value(model.pElectrolyzer_max_upper_bound)
            model.pElectrolyzer_max = pyo.Var(within=pyo.NonNegativeReals, bounds=(p_elec_max_lb_val, p_elec_max_ub_val), initialize=p_elec_max_ub_val) # Optimized capacity
            # Actual operating power of electrolyzer
            model.pElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds will be set by constraints like pElec <= pElecMax * uElec
            # Setpoint for electrolyzer, used for AS calculations and can differ from actual pElectrolyzer in dispatch mode
            model.pElectrolyzerSetpoint = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds also by constraints
            model.mHydrogenProduced = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            if pyo.value(model.aux_power_consumption_per_kg_h2) > 1e-6 : model.pAuxiliary = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            if not model.LTE_MODE: model.qSteam_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

            if ENABLE_STARTUP_SHUTDOWN: # Use global config flag
                model.uElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.Binary)
                model.vElectrolyzerStartup = pyo.Var(model.TimePeriods, within=pyo.Binary)
                model.wElectrolyzerShutdown = pyo.Var(model.TimePeriods, within=pyo.Binary)
            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: # Use global config flag
                model.DegradationState = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            # Ramping cost variables (MW ramped, not power itself)
            if pyo.value(model.cost_electrolyzer_ramping) > 1e-9: # Only if ramping has a cost
                model.pElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
                model.pElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            if not model.LTE_MODE and pyo.value(model.Ramp_qSteam_Electrolyzer_limit) < float('inf'): # HTE steam ramping vars
                model.qSteamElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
                model.qSteamElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

            if model.ENABLE_H2_STORAGE:
                 h2_storage_min_val_loc = pyo.value(model.H2_storage_capacity_min); h2_storage_max_val_loc = pyo.value(model.H2_storage_capacity_max)
                 h2_charge_max_val_loc = pyo.value(model.H2_storage_charge_rate_max); h2_discharge_max_val_loc = pyo.value(model.H2_storage_discharge_rate_max)
                 model.H2_storage_level = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(h2_storage_min_val_loc, h2_storage_max_val_loc))
                 model.H2_to_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, h2_charge_max_val_loc)) # kg/hr
                 model.H2_from_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, h2_discharge_max_val_loc)) # kg/hr
                 model.H2_to_market = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # kg/hr

        if model.ENABLE_BATTERY:
            batt_cap_lb_val_loc = pyo.value(model.BatteryCapacity_min); batt_cap_ub_val_loc = pyo.value(model.BatteryCapacity_max)
            model.BatteryCapacity_MWh = pyo.Var(within=pyo.NonNegativeReals, bounds=(batt_cap_lb_val_loc, batt_cap_ub_val_loc), initialize=(batt_cap_lb_val_loc + batt_cap_ub_val_loc) / 2)
            model.BatteryPower_MW = pyo.Var(within=pyo.NonNegativeReals) # Bounds linked to Capacity by constraint
            model.BatterySOC = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds linked to Capacity by constraint
            model.BatteryCharge = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds linked to Power_MW by constraint
            model.BatteryDischarge = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Bounds linked to Power_MW by constraint
            model.BatteryBinaryCharge = pyo.Var(model.TimePeriods, within=pyo.Binary)
            model.BatteryBinaryDischarge = pyo.Var(model.TimePeriods, within=pyo.Binary)

        # Ancillary Service Bid Variables
        as_service_list_internal = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU'] # Internal names
        model.component_as_bid_vars = {} # To store lists of var names per component
        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            logger.info("Defining Ancillary Service Bid Variables...")
            components_for_as_bids = []
            if model.ENABLE_NUCLEAR_GENERATOR and (model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY): components_for_as_bids.append('Turbine')
            if model.ENABLE_ELECTROLYZER: components_for_as_bids.append('Electrolyzer')
            if model.ENABLE_BATTERY: components_for_as_bids.append('Battery')

            for comp_name in components_for_as_bids:
                model.component_as_bid_vars[comp_name] = []
                for s_internal in as_service_list_internal:
                    var_name = f"{s_internal}_{comp_name}"
                    model.add_component(var_name, pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0))
                    model.component_as_bid_vars[comp_name].append(var_name)
            # Total AS Bids (sum of component bids)
            for s_internal in as_service_list_internal:
                model.add_component(f"Total_{s_internal}", pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0))
        else: # AS disabled, fix Total bids to 0 using Parameters
            logger.info("Ancillary Services disabled by configuration. Fixing AS Total bids to 0.")
            for s_internal in as_service_list_internal:
                model.add_component(f"Total_{s_internal}", pyo.Param(model.TimePeriods, initialize=0.0, within=pyo.NonNegativeReals))

        # Ancillary Service Deployed Variables (only if simulating dispatch)
        model.component_as_deployed_vars = {}
        if model.SIMULATE_AS_DISPATCH_EXECUTION and model.CAN_PROVIDE_ANCILLARY_SERVICES:
            logger.info("Defining Ancillary Service Deployed Variables...")
            # Determine components that can have deployed AS variables
            components_providing_as_deployment = []
            if model.ENABLE_ELECTROLYZER: components_providing_as_deployment.append('Electrolyzer')
            if model.ENABLE_BATTERY: components_providing_as_deployment.append('Battery')
            if model.ENABLE_NUCLEAR_GENERATOR and (model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY):
                 components_providing_as_deployment.append('Turbine')

            for comp_name in components_providing_as_deployment:
                 model.component_as_deployed_vars[comp_name] = []
                 for s_internal in as_service_list_internal:
                    bid_var_name = f"{s_internal}_{comp_name}"
                   
                    iso = model.TARGET_ISO
                    win_param   = f"winning_rate_{s_internal}_{iso}"
                    deploy_param= f"deploy_factor_{s_internal}_{iso}"
                    if (hasattr(model, bid_var_name)
                        and hasattr(model, win_param)
                        and hasattr(model, deploy_param)
                        and isinstance(getattr(model, bid_var_name), pyo.Var)):
                        deployed_var_name = f"{s_internal}_{comp_name}_Deployed"
                        model.add_component(
                            deployed_var_name,
                            pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
                        )
                        model.component_as_deployed_vars[comp_name].append(deployed_var_name)
                    else:
                        logger.warning(
                            f"Skip Deployed var for {s_internal}_{comp_name}: "
                            f"missing {win_param} or {deploy_param}"
                        )
    except Exception as e: logger.error(f"Error during variable definition: {e}", exc_info=True); raise

    # =========================================================================
    # PRECOMPUTE / UPDATE PARAMS BASED ON VARIABLES (e.g. inverse efficiencies)
    # =========================================================================
    try:
        if model.ENABLE_ELECTROLYZER:
            if hasattr(model, 'ke_H2_values') and hasattr(model, 'pElectrolyzer_efficiency_breakpoints'):
                 if not list(model.pElectrolyzer_efficiency_breakpoints): raise ValueError("Cannot precompute ke_H2_inv_values: pElectrolyzer_efficiency_breakpoints is empty.")
                 # ke_H2_values is MWh_elec / kg_H2. Inverse is kg_H2 / MWh_elec
                 model.ke_H2_inv_values = { bp: (1.0 / model.ke_H2_values[bp] if abs(pyo.value(model.ke_H2_values[bp])) > 1e-9 else 1e9) for bp in model.pElectrolyzer_efficiency_breakpoints }
            else: raise ValueError("Missing ke_H2_values or pElectrolyzer_efficiency_breakpoints for precomputation of inverse efficiency.")

            if not model.LTE_MODE: # HTE specific precomputation for steam consumption PWL
                 if hasattr(model, 'kt_H2_values') and hasattr(model, 'ke_H2_inv_values') and hasattr(model, 'pElectrolyzer_efficiency_breakpoints'):
                      if not list(model.pElectrolyzer_efficiency_breakpoints): raise ValueError("Cannot precompute qSteam_values_at_pElec_bp: pElectrolyzer_efficiency_breakpoints is empty.")
                      # q_steam = p_elec * (kt_H2 / ke_H2) = p_elec * kt_H2 * ke_H2_inv
                      # So, at breakpoint p_bp, the q_steam output is p_bp * kt_H2[p_bp] * ke_H2_inv[p_bp]
                      q_steam_at_pElec_bp_dict = {
                          p_bp: (p_bp * pyo.value(model.kt_H2_values[p_bp]) * pyo.value(model.ke_H2_inv_values[p_bp]))
                          for p_bp in model.pElectrolyzer_efficiency_breakpoints
                      }
                      model.qSteam_values_at_pElec_bp = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=q_steam_at_pElec_bp_dict)
                      logger.info("Calculated qSteam values at pElectrolyzer breakpoints for HTE.")
                 else:
                      logger.warning("kt_H2_values, ke_H2_inv_values, or pElectrolyzer_efficiency_breakpoints missing. Cannot calculate qSteam_values_at_pElec_bp for HTE.")
                      if hasattr(model, 'pElectrolyzer_efficiency_breakpoints') and list(model.pElectrolyzer_efficiency_breakpoints):
                          model.qSteam_values_at_pElec_bp = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=0.0, default=0.0) # Fallback
    except Exception as e: logger.error(f"Error during precomputation: {e}", exc_info=True); raise

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    logger.info("Defining constraints...")
    try:
        # --- Physical System Constraints ---
        model.power_balance_constr = pyo.Constraint(model.TimePeriods, rule=power_balance_rule)
        if hasattr(model, 'pAuxiliary'): model.link_auxiliary_power_constr = pyo.Constraint(model.TimePeriods, rule=link_auxiliary_power_rule)

        if model.ENABLE_NUCLEAR_GENERATOR:
            model.steam_balance_constr = pyo.Constraint(model.TimePeriods, rule=steam_balance_rule)
            if model.nonlinear_turbine_enabled_in_model and hasattr(model, 'pTurbine_values_at_qTurbine_bp'):
                 build_piecewise_constraints(model, component_prefix='TurbinePower',
                                             input_var_name='qSteam_Turbine', output_var_name='pTurbine',
                                             breakpoint_set_name='qTurbine_efficiency_breakpoints',
                                             value_param_name='pTurbine_values_at_qTurbine_bp') # Pass the Param object
            elif ENABLE_NONLINEAR_TURBINE_EFF: # Intent was there, but params failed
                 logger.warning("Nonlinear turbine enabled by config but PWL params missing/invalid. Using linear fallback constraint for turbine.")
                 def linear_pTurbine_rule(m,t): return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const
                 model.linear_pTurbine_constr = pyo.Constraint(model.TimePeriods, rule=linear_pTurbine_rule)
            else: # Linear efficiency by explicit choice (ENABLE_NONLINEAR_TURBINE_EFF is False)
                def linear_pTurbine_rule(m,t): return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const
                model.linear_pTurbine_constr = pyo.Constraint(model.TimePeriods, rule=linear_pTurbine_rule)

            model.Turbine_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampUp_rule)
            model.Turbine_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampDown_rule)
            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.turbine_as_zero_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Zero_rule)
                model.Turbine_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmax_rule)
                model.Turbine_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmin_rule)
                model.Turbine_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RU_rule)
                model.Turbine_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RD_rule)
            if model.ENABLE_ELECTROLYZER and model.LTE_MODE:
                model.const_turbine_power_constr = pyo.Constraint(model.TimePeriods, rule=constant_turbine_power_rule)

        # --- Electrolyzer Constraints ---
        if model.ENABLE_ELECTROLYZER:
            # Max capacity limit for actual power (pElectrolyzer) is handled by electrolyzer_max_power_rule if SDS enabled,
            # or needs to be explicit if SDS is not enabled.
            # Max capacity limit for setpoint (pElectrolyzerSetpoint)
            def electrolyzer_setpoint_capacity_limit_rule(m,t): return m.pElectrolyzerSetpoint[t] <= m.pElectrolyzer_max
            model.electrolyzer_setpoint_capacity_limit_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_setpoint_capacity_limit_rule)

            # --- ADDING NEW ELECTROLYZER SETPOINT MIN POWER CONSTRAINT ---
            model.electrolyzer_setpoint_min_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_setpoint_min_power_rule)


            if hasattr(model, 'ke_H2_inv_values'): # This is a dict precomputed on the model
                 build_piecewise_constraints(model, component_prefix='HydrogenProduction',
                                             input_var_name='pElectrolyzer', # Actual power produces H2
                                             output_var_name='mHydrogenProduced',
                                             breakpoint_set_name='pElectrolyzer_efficiency_breakpoints',
                                             value_param_name=model.ke_H2_inv_values) # Pass the dict
            else: logger.error("Cannot build HydrogenProduction piecewise: ke_H2_inv_values dict missing.")

            if not model.LTE_MODE and hasattr(model, 'qSteam_values_at_pElec_bp'): # HTE steam consumption
                 build_piecewise_constraints(model, component_prefix='SteamConsumption',
                                             input_var_name='pElectrolyzer', # Actual power consumes steam
                                             output_var_name='qSteam_Electrolyzer',
                                             breakpoint_set_name='pElectrolyzer_efficiency_breakpoints',
                                             value_param_name='qSteam_values_at_pElec_bp') # Pass the Param
            elif not model.LTE_MODE: logger.warning("Cannot build SteamConsumption piecewise for HTE: qSteam_values_at_pElec_bp Param missing.")

            model.Electrolyzer_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampUp_rule)
            model.Electrolyzer_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampDown_rule)

            if not model.LTE_MODE and hasattr(model, 'qSteamElectrolyzerRampPos'): # HTE Steam Ramping
                 def qSteam_ramp_linearization_rule(m, t):
                     if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                     if not hasattr(m, 'qSteam_Electrolyzer'): return pyo.Constraint.Skip
                     return m.qSteam_Electrolyzer[t] - m.qSteam_Electrolyzer[t-1] == m.qSteamElectrolyzerRampPos[t] - m.qSteamElectrolyzerRampNeg[t]
                 model.qSteam_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=qSteam_ramp_linearization_rule)
                 model.Steam_Electrolyzer_Ramp_constr = pyo.Constraint(model.TimePeriods, rule=Steam_Electrolyzer_Ramp_rule)

            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.Electrolyzer_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmax_rule)
                model.Electrolyzer_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmin_rule)
                model.Electrolyzer_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RU_rule)
                model.Electrolyzer_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RD_rule)

            if ENABLE_STARTUP_SHUTDOWN: # Use global config flag
                model.electrolyzer_on_off_logic_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_on_off_logic_rule)
                model.electrolyzer_min_power_when_on_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_when_on_rule)
                model.electrolyzer_max_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_max_power_rule) # This links pElec to pElecMax * uElec
                model.electrolyzer_startup_shutdown_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_startup_shutdown_exclusivity_rule)
                model.electrolyzer_min_uptime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_uptime_rule)
                model.electrolyzer_min_downtime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_downtime_rule)
            else: # SDS Disabled
                # If SDS is disabled, pElectrolyzer must be >= pElectrolyzer_min (if min > 0)
                # and pElectrolyzer <= pElectrolyzer_max (the optimized capacity)
                model.electrolyzer_min_power_sds_disabled_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_sds_disabled_rule)
                def electrolyzer_max_power_sds_disabled_rule(m,t): return m.pElectrolyzer[t] <= m.pElectrolyzer_max
                model.electrolyzer_max_power_sds_disabled_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_max_power_sds_disabled_rule)


            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: # Use global config flag
                model.electrolyzer_degradation_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_degradation_rule)
            if ENABLE_H2_CAP_FACTOR: # Use global config flag
                model.h2_prod_req_constr = pyo.Constraint(rule=h2_CapacityFactor_rule)

            # Linearization for electrolyzer power ramping cost (if cost_electrolyzer_ramping > 0)
            if hasattr(model, 'pElectrolyzerRampPos'): # This var only exists if cost > 0
                def electrolyzer_ramp_linearization_rule(m, t):
                    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                    # This defines RampPos/Neg based on change in ACTUAL power m.pElectrolyzer
                    return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] == m.pElectrolyzerRampPos[t] - m.pElectrolyzerRampNeg[t]
                model.electrolyzer_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_ramp_linearization_rule)

            if model.ENABLE_H2_STORAGE:
                model.h2_storage_balance_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_balance_adj_rule)
                model.h2_prod_dispatch_constr = pyo.Constraint(model.TimePeriods, rule=h2_prod_dispatch_rule)
                model.h2_storage_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_charge_limit_rule)
                model.h2_storage_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_discharge_limit_rule)
                model.h2_storage_level_max_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_max_rule)
                model.h2_storage_level_min_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_min_rule)

            # Conditional linking of pElectrolyzer, pElectrolyzerSetpoint, and Deployed AS
            if model.SIMULATE_AS_DISPATCH_EXECUTION:
                # This rule defines pElectrolyzer based on Setpoint and Deployed AS
                model.define_actual_electrolyzer_power_constr = pyo.Constraint(model.TimePeriods, rule=define_actual_electrolyzer_power_rule)
            else:
                # --- ADDING NEW CONSTRAINT for non-dispatch simulation mode ---
                model.link_setpoint_to_actual_power_constr = pyo.Constraint(model.TimePeriods, rule=link_setpoint_to_actual_power_if_not_simulating_dispatch_rule)


        # --- Battery Constraints ---
        if model.ENABLE_BATTERY:
            model.battery_soc_balance_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_balance_rule)
            model.battery_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=battery_charge_limit_rule)
            model.battery_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_limit_rule)
            model.battery_binary_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=battery_binary_exclusivity_rule)
            model.battery_soc_max_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_max_rule)
            model.battery_soc_min_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_min_rule)
            model.battery_charge_ramp_up_constr = pyo.Constraint(model.TimePeriods, rule=battery_ramp_up_rule) # Charge ramp up
            model.battery_charge_ramp_down_constr = pyo.Constraint(model.TimePeriods, rule=battery_ramp_down_rule) # Charge ramp down
            model.battery_discharge_ramp_up_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_ramp_up_rule)
            model.battery_discharge_ramp_down_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_ramp_down_rule)
            if pyo.value(model.BatteryRequireCyclicSOC):
                model.battery_cyclic_soc_lower_constr = pyo.Constraint(rule=battery_cyclic_soc_lower_rule)
                model.battery_cyclic_soc_upper_constr = pyo.Constraint(rule=battery_cyclic_soc_upper_rule)
            model.battery_power_capacity_link_constr = pyo.Constraint(rule=battery_power_capacity_link_rule)
            model.battery_min_cap_constr = pyo.Constraint(rule=battery_min_cap_rule)
            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.Battery_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_Pmax_rule)
                model.Battery_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_Pmin_rule)
                model.Battery_AS_SOC_Up_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_SOC_Up_rule)
                model.Battery_AS_SOC_Down_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_SOC_Down_rule)
                model.Battery_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_RU_rule)
                model.Battery_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_RD_rule)

        # --- Ancillary Service Linking Constraints (BIDS) ---
        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            # Check if Total_Service is a Var (meaning AS bids are being optimized)
            if hasattr(model, 'Total_RegUp') and isinstance(model.Total_RegUp, pyo.Var): model.link_Total_RegUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegUp_rule)
            if hasattr(model, 'Total_RegDown') and isinstance(model.Total_RegDown, pyo.Var): model.link_Total_RegDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegDown_rule)
            if hasattr(model, 'Total_SR') and isinstance(model.Total_SR, pyo.Var): model.link_Total_SR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_SR_rule)
            if hasattr(model, 'Total_NSR') and isinstance(model.Total_NSR, pyo.Var): model.link_Total_NSR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_NSR_rule)
            if hasattr(model, 'Total_ECRS') and isinstance(model.Total_ECRS, pyo.Var): model.link_Total_ECRS_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_ECRS_rule)
            if hasattr(model, 'Total_ThirtyMin') and isinstance(model.Total_ThirtyMin, pyo.Var): model.link_Total_30Min_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_30Min_rule) # Corrected rule name
            if hasattr(model, 'Total_RampUp') and isinstance(model.Total_RampUp, pyo.Var): model.link_Total_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RampUp_rule)
            if hasattr(model, 'Total_RampDown') and isinstance(model.Total_RampDown, pyo.Var): model.link_Total_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RampDown_rule)
            if hasattr(model, 'Total_UncU') and isinstance(model.Total_UncU, pyo.Var): model.link_Total_UncU_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_UncU_rule)

        # --- Link Deployed AS to Bids (Dynamically) - If in Dispatch Simulation Mode ---
        if model.SIMULATE_AS_DISPATCH_EXECUTION and model.CAN_PROVIDE_ANCILLARY_SERVICES:
            logger.info("Adding generic link_deployed_to_bid_rule constraints...")
            components_with_deployed_vars = []
            if model.ENABLE_ELECTROLYZER and model.component_as_deployed_vars.get('Electrolyzer'): components_with_deployed_vars.append('Electrolyzer')
            if model.ENABLE_BATTERY and model.component_as_deployed_vars.get('Battery'): components_with_deployed_vars.append('Battery')
            if model.ENABLE_NUCLEAR_GENERATOR and (model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY) and model.component_as_deployed_vars.get('Turbine'):
                 components_with_deployed_vars.append('Turbine')

            for comp_name_iter in components_with_deployed_vars:
                for service_internal_iter in as_service_list_internal: # Use internal service names
                    deployed_var_name_check = f"{service_internal_iter}_{comp_name_iter}_Deployed"
                    if hasattr(model, deployed_var_name_check): # Check if this specific deployed var exists
                        # Create a unique rule for each service and component combination
                        def _rule_factory_deployed_link(s_name, c_name): # s_name is internal service name
                            def _actual_rule(m_inner, t_inner):
                                return link_deployed_to_bid_rule(m_inner, t_inner, s_name, c_name)
                            return _actual_rule
                        constr_name_deployed = f"link_deployed_{service_internal_iter}_{comp_name_iter}_constr"
                        model.add_component(constr_name_deployed,
                                            pyo.Constraint(model.TimePeriods,
                                                           rule=_rule_factory_deployed_link(service_internal_iter, comp_name_iter)))
                    # else:
                        # logger.debug(f"Deployed var {deployed_var_name_check} not found for component {comp_name_iter}, service {service_internal_iter}. Skipping link_deployed_to_bid constraint.")


    except Exception as e: logger.error(f"Error during constraint definition: {e}", exc_info=True); raise

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================
    logger.info("Defining objective function (Maximize Profit)...")
    try:
        model.EnergyRevenueExpr = pyo.Expression(rule=EnergyRevenue_rule)
        model.HydrogenRevenueExpr = pyo.Expression(rule=HydrogenRevenue_rule)

        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            iso_revenue_rule_map = {
                'CAISO': AncillaryRevenue_CAISO_rule, 'ERCOT': AncillaryRevenue_ERCOT_rule,
                'ISONE': AncillaryRevenue_ISONE_rule, 'MISO': AncillaryRevenue_MISO_rule,
                'NYISO': AncillaryRevenue_NYISO_rule, 'PJM': AncillaryRevenue_PJM_rule,
                'SPP': AncillaryRevenue_SPP_rule,
            }
            if target_iso in iso_revenue_rule_map:
                model.AncillaryRevenueExpr = pyo.Expression(rule=iso_revenue_rule_map[target_iso])
            else:
                logger.warning(f"No AS Revenue rule defined for TARGET_ISO='{target_iso}'. Setting AS Revenue to 0.")
                model.AncillaryRevenueExpr = pyo.Expression(initialize=0.0)
        else:
            model.AncillaryRevenueExpr = pyo.Expression(initialize=0.0)

        model.OpexCostExpr = pyo.Expression(rule=OpexCost_rule)

        def AnnualizedCapex_rule(m):
             total_annual_capex_expr = 0.0
             try:
                 delT_min_val = pyo.value(m.delT_minutes)
                 total_hours_sim = len(m.TimePeriods) * (delT_min_val / 60.0)
                 # Annualization should be based on a full year (8760 hours)
                 # The scaling factor adjusts the total capex (which is for the plant lifetime)
                 # to an equivalent annual cost for the simulation period.
                 # However, the cost parameters (e.g., cost_electrolyzer_capacity_USD_per_MW_year) are already per year.
                 # So, we need to scale the *total* capex (if calculated for lifetime) or use annual costs directly.
                 # The current parameters like 'cost_electrolyzer_capacity_USD_per_MW_year' are already annualized.
                 # The scaling factor is to make the objective value reflect the profit over the *simulated period*.
                 scaling_factor_for_period = total_hours_sim / HOURS_IN_YEAR # Sim period as fraction of year

             except Exception as e:
                  logger.error(f"Error getting fixed values in AnnualizedCapex_rule: {e}")
                  return 0.0

             if m.ENABLE_ELECTROLYZER and hasattr(m, 'cost_electrolyzer_capacity') and hasattr(m, 'pElectrolyzer_max'):
                  cost_elec_cap_param_per_year = m.cost_electrolyzer_capacity # This is $/MW-year
                  # Cost for the simulated period = AnnualCost * (SimHours / AnnualHours)
                  total_annual_capex_expr += m.pElectrolyzer_max * cost_elec_cap_param_per_year * scaling_factor_for_period

             if m.ENABLE_BATTERY and hasattr(m, 'BatteryCapacity_MWh') and hasattr(m, 'BatteryPower_MW'):
                 batt_cap_cost_param_per_year = m.BatteryCapex_USD_per_MWh_year
                 batt_pow_cost_param_per_year = m.BatteryCapex_USD_per_MW_year
                 batt_fom_cost_param_per_year = m.BatteryFixedOM_USD_per_MWh_year # Fixed OM is also like an annualized capex here

                 battery_annual_cost = (m.BatteryCapacity_MWh * batt_cap_cost_param_per_year +
                                        m.BatteryPower_MW * batt_pow_cost_param_per_year +
                                        m.BatteryCapacity_MWh * batt_fom_cost_param_per_year)
                 total_annual_capex_expr += battery_annual_cost * scaling_factor_for_period
             return total_annual_capex_expr
        model.AnnualizedCapexExpr = pyo.Expression(rule=AnnualizedCapex_rule)

        def TotalProfit_Objective_rule(m):
            try:
                energy_rev = getattr(m, 'EnergyRevenueExpr', 0.0)
                as_rev = getattr(m, 'AncillaryRevenueExpr', 0.0)
                h2_rev = getattr(m, 'HydrogenRevenueExpr', 0.0)
                opex_cost = getattr(m, 'OpexCostExpr', 0.0)
                capex_cost_for_period = getattr(m, 'AnnualizedCapexExpr', 0.0) # This is already scaled for the period

                total_revenue = energy_rev + as_rev + h2_rev
                total_cost = opex_cost + capex_cost_for_period
                return total_revenue - total_cost
            except Exception as e:
                 logger.critical(f"Error defining TotalProfit_Objective_rule expression: {e}", exc_info=True)
                 raise
        model.TotalProfit_Objective = pyo.Objective(rule=TotalProfit_Objective_rule, sense=pyo.maximize)

    except Exception as e: logger.error(f"Error during objective definition: {e}", exc_info=True); raise

    logger.info("Standardized model created successfully.")
    return model

