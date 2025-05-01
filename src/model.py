# src/model.py
# ... (other imports) ...
import pyomo.environ as pyo
from logging_setup import logger
from config import HOURS_IN_YEAR
import pandas as pd

from config import (ENABLE_NONLINEAR_TURBINE_EFF, ENABLE_H2_STORAGE,
                    ENABLE_H2_CAP_FACTOR, ENABLE_STARTUP_SHUTDOWN,
                    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
                    ENABLE_LOW_TEMP_ELECTROLYZER
                    )
# Import constraint rules, including the new one
from constraints import (
    build_piecewise_constraints, steam_balance_rule, power_balance_rule,
    constant_turbine_power_rule, h2_storage_balance_rule, h2_storage_charge_limit_rule,
    h2_storage_discharge_limit_rule, h2_storage_level_max_rule, h2_storage_level_min_rule,
    h2_direct_market_link_rule, Electrolyzer_RampUp_rule, Electrolyzer_RampDown_rule,
    Turbine_RampUp_rule, Turbine_RampDown_rule, Steam_Electrolyzer_Ramp_rule,
    h2_CapacityFactor_rule, electrolyzer_on_off_logic_rule, electrolyzer_min_power_rule,
    electrolyzer_max_power_rule, electrolyzer_startup_shutdown_exclusivity_rule,
    electrolyzer_min_uptime_rule, electrolyzer_min_downtime_rule, electrolyzer_degradation_rule,
    Turbine_AS_Pmax_rule, Turbine_AS_Pmin_rule, Turbine_AS_RU_rule, Turbine_AS_RD_rule,
    Turbine_AS_Zero_rule, Electrolyzer_AS_Pmax_rule, Electrolyzer_AS_Pmin_rule,
    Electrolyzer_AS_RU_rule, Electrolyzer_AS_RD_rule, # Ensure these are defined
    Electrolyzer_Setpoint_Link_rule, # Added new constraint rule
    link_Total_RegUp_rule, link_Total_RegDown_rule, link_Total_SR_rule, link_Total_NSR_rule,
    link_Total_ECRS_rule, link_Total_30Min_rule,
)
# Import revenue rules
from revenue_cost import (
    EnergyRevenue_rule, HydrogenRevenue_rule, AncillaryRevenue_CAISO_rule,
    AncillaryRevenue_ERCOT_rule, AncillaryRevenue_ISONE_rule, AncillaryRevenue_MISO_rule,
    AncillaryRevenue_NYISO_rule, AncillaryRevenue_PJM_rule, AncillaryRevenue_SPP_rule,
    OpexCost_rule,
)

def create_model(data_inputs, target_iso: str, use_nonlinear_turbine_eff_setting: bool) -> pyo.ConcreteModel:
    # ... (initial model setup) ...
    model = pyo.ConcreteModel(f"Optimize_Profit_Standardized_{target_iso}")
    model.TARGET_ISO = target_iso
    model.LTE_MODE = ENABLE_LOW_TEMP_ELECTROLYZER

    logger.info(f"Creating STANDARDIZED model for {target_iso}...")
    # =========================================================================
    # SETS & PARAMETERS
    # =========================================================================
    logger.info("Loading parameters...")
    try:
        # ... (TimePeriods, df_system loading, get_sys_param helper) ...
        nT = HOURS_IN_YEAR
        model.TimePeriods = pyo.Set(initialize=pyo.RangeSet(1, nT), ordered=True)
        df_system = data_inputs['df_system']
        def get_sys_param(param_name, default=None):
           # (Keep existing helper function)
            try:
                val = df_system.loc[param_name, 'Value']
                if pd.isna(val) and default is not None: return default
                if 'MinUpTime' in param_name or 'MinDownTime' in param_name or 'initial_status' in param_name:
                    return int(val)
                if isinstance(val, str) and ',' in val and ('Breakpoints' in param_name or 'Values' in param_name):
                     return val
                return float(val)
            except KeyError:
                if default is not None:
                    logger.warning(f"System parameter '{param_name}' not found. Using default: {default}")
                    return default
                else:
                    logger.error(f"Essential system parameter '{param_name}' not found!")
                    raise ValueError(f"Missing essential parameter: {param_name}")

        # ... (Load other system parameters: delT, qSteam_Total, Turbine, Electrolyzer bounds/cost, etc.) ...
        # General
        model.delT_minutes = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('delT_minutes', 60.0))
        model.qSteam_Total = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Total_MWth'))

        # Turbine
        model.convertTtE_const = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4))
        model.qSteam_Turbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_min_MWth'))
        model.qSteam_Turbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_max_MWth'))
        model.pTurbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_min_MW'))
        model.pTurbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_max_MW'))
        model.RU_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_RampUp_Rate_Percent_per_Min', 1.0) * 60 / 100 * model.pTurbine_max)
        model.RD_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_RampDown_Rate_Percent_per_Min', 1.0) * 60 / 100 * model.pTurbine_max)
        if model.LTE_MODE:
             model.pTurbine_LTE_setpoint = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_LTE_setpoint_MW', model.pTurbine_max))
        if not model.LTE_MODE:
            model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Electrolyzer_Ramp_Limit_MWth_per_Hour', float('inf')))
        else:
            model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(initialize=0.0)

        # --- Turbine Piecewise ---
        nonlinear_turbine_enabled_in_model = False
        if use_nonlinear_turbine_eff_setting:
             try:
                q_bps_str = get_sys_param('qSteam_Turbine_Breakpoints_MWth')
                p_vals_str = get_sys_param('pTurbine_Outputs_at_Breakpoints_MW')
                q_breakpoints = sorted([float(x.strip()) for x in q_bps_str.split(',')])
                p_values = [float(x.strip()) for x in p_vals_str.split(',')]
                if len(q_breakpoints) != len(p_values): raise ValueError("Turbine breakpoint lengths differ.")
                model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=q_breakpoints, ordered=True)
                pTurbine_vals_at_qTurbine_bp = dict(zip(q_breakpoints, p_values))
                model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize=pTurbine_vals_at_qTurbine_bp)
                model.pTurbine_min = pyo.Param(mutable=True, initialize=min(p_values))
                model.pTurbine_max = pyo.Param(mutable=True, initialize=max(p_values))
                nonlinear_turbine_enabled_in_model = True
                logger.info("Enabled non-linear turbine efficiency.")
             except Exception as e:
                logger.error(f"Error loading turbine piecewise data: {e}. Falling back to constant.")

        # Electrolyzer
        model.pElectrolyzer_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pElectrolyzer_min_MW'))
        model.pElectrolyzer_max_upper_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pElectrolyzer_max_upper_bound_MW'))
        model.pElectrolyzer_max_lower_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pElectrolyzer_max_lower_bound_MW', 0))
        model.RU_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Electrolyzer_RampUp_Rate_Percent_per_Min', 10.0) * 60 / 100)
        model.RD_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Electrolyzer_RampDown_Rate_Percent_per_Min', 10.0) * 60 / 100)

        # --- Electrolyzer Piecewise ---
        try:
            p_elec_bps_str = get_sys_param('pElectrolyzer_Breakpoints_MW')
            ke_vals_str = get_sys_param('ke_H2_Values_MWh_per_kg')
            p_elec_breakpoints = sorted([float(x.strip()) for x in p_elec_bps_str.split(',')])
            ke_values = [float(x.strip()) for x in ke_vals_str.split(',')]
            model.pElectrolyzer_efficiency_breakpoints = pyo.Set(initialize=p_elec_breakpoints, ordered=True)
            ke_vals_dict = dict(zip(p_elec_breakpoints, ke_values))
            model.ke_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=ke_vals_dict, within=pyo.NonNegativeReals)
            if not model.LTE_MODE: # Only load kt if HTE
                kt_vals_str = get_sys_param('kt_H2_Values_MWth_per_kg')
                kt_values = [float(x.strip()) for x in kt_vals_str.split(',')]
                if not (len(p_elec_breakpoints) == len(ke_values) == len(kt_values)):
                     raise ValueError("HTE Electrolyzer breakpoint lengths differ.")
                kt_vals_dict = dict(zip(p_elec_breakpoints, kt_values))
                model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_vals_dict, within=pyo.NonNegativeReals)
            else:
                 model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=0.0, within=pyo.NonNegativeReals)
            if len(p_elec_breakpoints) != len(ke_values):
                 raise ValueError("Electrolyzer ke breakpoint lengths differ.")
            logger.info("Loaded electrolyzer piecewise parameters (ke, kt).")
        except Exception as e:
            logger.error(f"Error loading electrolyzer piecewise data: {e}.")
            raise ValueError("Failed to load electrolyzer efficiency data.")

        # Grid
        model.pIES_min = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_min_MW', -model.pTurbine_max))
        model.pIES_max = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_max_MW', model.pTurbine_max))

        # Costs & Value
        model.H2_value = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_value_USD_per_kg'))
        model.vom_turbine = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_turbine_USD_per_MWh', 0))
        model.vom_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_electrolyzer_USD_per_MWh', 0))
        model.cost_water_per_kg_h2 = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_water_USD_per_kg_h2', 0))
        model.cost_electrolyzer_ramping = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_electrolyzer_ramping_USD_per_MW_ramp', 0))
        model.cost_electrolyzer_capacity = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_electrolyzer_capacity_USD_per_MW', 0))

        # Startup/Shutdown
        if ENABLE_STARTUP_SHUTDOWN:
            model.cost_startup_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_startup_electrolyzer_USD_per_startup', 0))
            model.MinUpTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinUpTimeElectrolyzer_hours', 1))
            model.MinDownTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinDownTimeElectrolyzer_hours', 1))
            model.uElectrolyzer_initial = pyo.Param(within=pyo.Binary, initialize=get_sys_param('uElectrolyzer_initial_status_0_or_1', 0))

        # Degradation
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
            model.DegradationStateInitial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationStateInitial_Units', 0.0))
            model.DegradationFactorOperation = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorOperation_Units_per_Hour_at_MaxLoad', 0.0))
            model.DegradationFactorStartup = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorStartup_Units_per_Startup', 0.0))

        # H2 Storage
        if ENABLE_H2_STORAGE:
            model.H2_storage_capacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_capacity_max_kg'))
            model.H2_storage_capacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_capacity_min_kg', 0))
            model.H2_storage_level_initial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_level_initial_kg', 0))
            model.H2_storage_charge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_charge_rate_max_kg_per_hr'))
            model.H2_storage_discharge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_discharge_rate_max_kg_per_hr'))
            model.storage_charge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_charge_eff_fraction', 1.0))
            model.storage_discharge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_discharge_eff_fraction', 1.0))
            model.vom_storage_cycle = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_storage_cycle_USD_per_kg_cycled', 0))

        # H2 Capacity Factor
        if ENABLE_H2_CAP_FACTOR:
             model.h2_target_capacity_factor = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('h2_target_capacity_factor_fraction', 0.0))

        # --- Energy Price ---
        df_price = data_inputs['df_price_hourly']
        if len(df_price) < nT: raise ValueError(f"Energy price data has {len(df_price)} rows, expected {nT}.")
        energy_price_col = 'Price ($/MWh)'
        if energy_price_col not in df_price.columns: raise ValueError(f"Energy price column '{energy_price_col}' not found.")
        model.energy_price = pyo.Param(model.TimePeriods, initialize=lambda m, t: df_price[energy_price_col].iloc[t-1])

        # --- Ancillary Service Prices & Factors ---
        df_ANSprice = data_inputs['df_ANSprice_hourly']
        if len(df_ANSprice) < nT: raise ValueError(f"ANS price data has {len(df_ANSprice)} rows, expected {nT}.")
        df_ANSmile = data_inputs.get('df_ANSmile_hourly', None)
        df_ANSdeploy = data_inputs.get('df_ANSdeploy_hourly', None)

        # Helper to get parameter values safely
        def get_hourly_param(t, df, col_name, default=0.0, required=False):
             if df is None:
                 if required: raise ValueError(f"Required data file (e.g., for {col_name}) not loaded.")
                 return default
             # Store filename for better error messages
             filename = getattr(df, 'attrs', {}).get('filename', 'DataFrame')
             if col_name in df.columns:
                 # Use iloc for positional access (t-1 because model index is 1-based)
                 return df[col_name].iloc[t-1] if t-1 < len(df) else default
             else:
                 if required: raise ValueError(f"Required column '{col_name}' not found in {filename}.")
                 # Optional: Log warning if optional column not found
                 # logger.warning(f"Column '{col_name}' not found in {filename}. Using default {default}.")
                 return default

        # --- Define ISO Service Map (used for loading correct parameters) ---
        # <<< Added MISO entry here >>>
        iso_service_map = {
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup'],
            'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
            'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'], # Include specific PJM factors/prices
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
            'ISONE': ['Spin10', 'NSpin10', 'OR30'],
            'MISO': ['Reg', 'Spin', 'Sup', 'STR'] # Correct MISO services
        }
        # <<< Check added here >>>
        if target_iso not in iso_service_map:
            logger.error(f"Ancillary service definitions not found for ISO: {target_iso} in model.py's iso_service_map.")
            raise ValueError(f"Ancillary service definitions not found for ISO: {target_iso}")

        # --- Load ISO-Specific AS Parameters (Prices, Adders, Factors) ---
        logger.info(f"Loading AS parameters for {target_iso}...")
        # Add filename attribute for better error messages
        if df_ANSprice is not None: df_ANSprice.attrs['filename'] = 'Price_ANS_hourly.csv'
        if df_ANSmile is not None: df_ANSmile.attrs['filename'] = 'MileageMultiplier_hourly.csv'
        if df_ANSdeploy is not None: df_ANSdeploy.attrs['filename'] = 'DeploymentFactor_hourly.csv'

        # Load parameters for the target ISO's services
        for service in iso_service_map[target_iso]:
            # Price (Required, except for PJM Reg which uses Cap/Perf)
            is_pjm_reg_component = (target_iso == 'PJM' and service in ['RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'])
            if not is_pjm_reg_component: # Load standard price if not a PJM Reg component
                price_col = f'p_{service}_{target_iso}'
                model.add_component(price_col, pyo.Param(model.TimePeriods, initialize=lambda m, t, sc=service: get_hourly_param(t, df_ANSprice, f'p_{sc}_{target_iso}', required=True), within=pyo.Reals))

            # Deployment Factor (Required for ALL services now for physical model)
            deploy_col = f'deploy_factor_{service}_{target_iso}'
            # Skip deploy factor for PJM factors/split prices
            if not is_pjm_reg_component:
                model.add_component(deploy_col, pyo.Param(model.TimePeriods, initialize=lambda m, t, sc=service: get_hourly_param(t, df_ANSdeploy, f'deploy_factor_{sc}_{target_iso}', default=0.0, required=True), within=pyo.PercentFraction))

            # Locational Adder (Optional)
            loc_col = f'loc_{service}_{target_iso}'
            if not is_pjm_reg_component: # PJM Reg adder loaded separately if needed
                 model.add_component(loc_col, pyo.Param(model.TimePeriods, initialize=lambda m, t, sc=service: get_hourly_param(t, df_ANSprice, f'loc_{sc}_{target_iso}', default=0.0), within=pyo.Reals))

            # Mileage/Performance Factors/Prices (ISO/Service Specific)
            if target_iso == 'CAISO' and service in ['RegU', 'RegD']:
                 mf_col = f'mileage_factor_{service}_{target_iso}'
                 model.add_component(mf_col, pyo.Param(model.TimePeriods, initialize=lambda m, t, sc=service: get_hourly_param(t, df_ANSmile, f'mileage_factor_{sc}_{target_iso}', default=1.0), within=pyo.NonNegativeReals))
            # Load specific PJM factors/prices (handled by including them in the iso_service_map loop)
            if target_iso == 'PJM' and service == 'RegCap':
                 model.add_component(f'p_RegCap_{target_iso}', pyo.Param(model.TimePeriods, initialize=lambda m, t: get_hourly_param(t, df_ANSprice, f'p_RegCap_{target_iso}', 0.0, required=True), within=pyo.Reals))
            if target_iso == 'PJM' and service == 'RegPerf':
                  model.add_component(f'p_RegPerf_{target_iso}', pyo.Param(model.TimePeriods, initialize=lambda m, t: get_hourly_param(t, df_ANSprice, f'p_RegPerf_{target_iso}', 0.0, required=True), within=pyo.Reals))
            if target_iso == 'PJM' and service == 'performance_score':
                 model.add_component(f'performance_score_{target_iso}', pyo.Param(model.TimePeriods, initialize=lambda m, t: get_hourly_param(t, df_ANSmile, f'performance_score_{target_iso}', 1.0, required=True), within=pyo.NonNegativeReals))
            if target_iso == 'PJM' and service == 'mileage_ratio':
                 model.add_component(f'mileage_ratio_{target_iso}', pyo.Param(model.TimePeriods, initialize=lambda m, t: get_hourly_param(t, df_ANSmile, f'mileage_ratio_{target_iso}', 1.0, required=True), within=pyo.NonNegativeReals))
            # Add locational adder for combined PJM Reg if needed
            if target_iso == 'PJM' and service == 'Reg':
                 model.add_component(f'loc_Reg_{target_iso}', pyo.Param(model.TimePeriods, initialize=lambda m, t: get_hourly_param(t, df_ANSprice, f'loc_Reg_{target_iso}', default=0.0), within=pyo.Reals))


    except Exception as e:
        logger.error(f"An unexpected error occurred during parameter loading: {e}", exc_info=True)
        raise

    # =========================================================================
    # VARIABLES
    # =========================================================================
    # ... (Variable definitions remain the same as in model_deploy_factor_v2) ...
    logger.info("Defining variables...")
    try:
        model.qSteam_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.qSteam_Turbine_min, model.qSteam_Turbine_max))
        model.pTurbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.pTurbine_min, model.pTurbine_max))
        model.pElectrolyzer_max = pyo.Var(within=pyo.NonNegativeReals, bounds=(model.pElectrolyzer_max_lower_bound, model.pElectrolyzer_max_upper_bound), initialize=model.pElectrolyzer_max_upper_bound)
        model.pElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        model.pElectrolyzerSetpoint = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        model.mHydrogenProduced = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        if not model.LTE_MODE: model.qSteam_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        else: model.qSteam_Electrolyzer = pyo.Param(model.TimePeriods, initialize=0.0)
        model.pIES = pyo.Var(model.TimePeriods, within=pyo.Reals, bounds=(model.pIES_min, model.pIES_max))
        if ENABLE_H2_STORAGE:
            model.H2_storage_level = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(model.H2_storage_capacity_min, model.H2_storage_capacity_max))
            model.H2_from_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, model.H2_storage_discharge_rate_max))
            model.H2_to_market = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            if hasattr(model, 'vom_storage_cycle') and model.vom_storage_cycle > 1e-9: model.H2_net_to_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        if ENABLE_STARTUP_SHUTDOWN:
            model.uElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.Binary)
            model.vElectrolyzerStartup = pyo.Var(model.TimePeriods, within=pyo.Binary)
            model.wElectrolyzerShutdown = pyo.Var(model.TimePeriods, within=pyo.Binary)
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: model.DegradationState = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        model.RegUp_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegUp_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegDown_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.RegDown_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.SR_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.SR_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.NSR_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.NSR_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        if target_iso == 'ERCOT':
            model.ECRS_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.ECRS_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        if target_iso in ['PJM', 'NYISO', 'ISONE', 'MISO']:
            model.ThirtyMin_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.ThirtyMin_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_RegUp = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_RegDown = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_SR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        model.Total_NSR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        if target_iso == 'ERCOT': model.Total_ECRS = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        else: model.Total_ECRS = pyo.Param(model.TimePeriods, initialize=0.0, within=pyo.NonNegativeReals)
        if target_iso in ['PJM', 'NYISO', 'ISONE', 'MISO']: model.Total_30Min = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
        else: model.Total_30Min = pyo.Param(model.TimePeriods, initialize=0.0, within=pyo.NonNegativeReals)
        if hasattr(model, 'cost_electrolyzer_ramping') and model.cost_electrolyzer_ramping > 1e-9:
            model.pElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            model.pElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
        if not model.LTE_MODE and model.Ramp_qSteam_Electrolyzer_limit < float('inf'):
             model.qSteamElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
             model.qSteamElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
    except Exception as e: logger.error(f"Error during variable definition: {e}", exc_info=True); raise


    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    # ... (Constraints remain the same as in constraints_deploy_factor_v5) ...
    logger.info("Defining constraints...")
    try:
        # Physical System
        model.steam_balance_constr = pyo.Constraint(model.TimePeriods, rule=steam_balance_rule)
        model.power_balance_constr = pyo.Constraint(model.TimePeriods, rule=power_balance_rule)
        model.Turbine_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampUp_rule)
        model.Turbine_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampDown_rule)
        model.Electrolyzer_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampUp_rule)
        model.Electrolyzer_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampDown_rule)
        if model.LTE_MODE:
            model.constant_turbine_power_constr = pyo.Constraint(model.TimePeriods, rule=constant_turbine_power_rule)
            model.turbine_as_zero_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Zero_rule)
        if not model.LTE_MODE and model.Ramp_qSteam_Electrolyzer_limit < float('inf'):
             def qSteam_ramp_linearization_rule(m, t):
                 if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                 return m.qSteam_Electrolyzer[t] - m.qSteam_Electrolyzer[t-1] == m.qSteamElectrolyzerRampPos[t] - m.qSteamElectrolyzerRampNeg[t]
             model.qSteam_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=qSteam_ramp_linearization_rule)
             model.Steam_Electrolyzer_Ramp_constr = pyo.Constraint(model.TimePeriods, rule=Steam_Electrolyzer_Ramp_rule)

        # Electrolyzer Power/Setpoint Link
        model.Electrolyzer_Setpoint_Link_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_Setpoint_Link_rule)
        def electrolyzer_capacity_limit_rule(m, t): return m.pElectrolyzer[t] <= m.pElectrolyzer_max
        model.electrolyzer_capacity_limit_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_capacity_limit_rule)
        # Add constraint for setpoint as well? pSetpoint <= pMax? Usually handled by AS Pmax/Pmin rules.
        def electrolyzer_setpoint_capacity_limit_rule(m,t): return m.pElectrolyzerSetpoint[t] <= m.pElectrolyzer_max
        model.electrolyzer_setpoint_capacity_limit_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_setpoint_capacity_limit_rule)


        # Component Efficiency & Production
        # <<< Placeholder - Replace with actual piecewise logic >>>
        def h2_production_from_actual_power_rule(m, t):
            avg_ke = 50.0; return m.mHydrogenProduced[t] * avg_ke == m.pElectrolyzer[t] if avg_ke > 1e-6 else m.mHydrogenProduced[t] == 0
        model.h2_production_constr = pyo.Constraint(model.TimePeriods, rule=h2_production_from_actual_power_rule)
        logger.warning("Using PLACEHOLDER linear efficiency for H2 production constraint.")
        def steam_consumption_rule(m, t):
             if m.LTE_MODE: return pyo.Constraint.Skip
             avg_kt = 8.0; return m.qSteam_Electrolyzer[t] == m.mHydrogenProduced[t] * avg_kt
        model.steam_consumption_constr = pyo.Constraint(model.TimePeriods, rule=steam_consumption_rule)
        logger.warning("Using PLACEHOLDER linear efficiency for steam consumption constraint.")
        # Turbine Efficiency
        if nonlinear_turbine_enabled_in_model:
            build_piecewise_constraints(model, component_prefix='TurbinePower', input_var_name='qSteam_Turbine', output_var_name='pTurbine', breakpoint_set_name='qTurbine_efficiency_breakpoints', value_param_name='pTurbine_values_at_qTurbine_bp')
        else:
            def linear_pTurbine_rule(m,t): return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const
            model.linear_pTurbine_constr = pyo.Constraint(model.TimePeriods, rule=linear_pTurbine_rule)

        # H2 Storage
        if ENABLE_H2_STORAGE:
            model.h2_storage_balance_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_balance_rule)
            model.h2_storage_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_charge_limit_rule)
            model.h2_storage_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_discharge_limit_rule)
            model.h2_storage_level_max_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_max_rule)
            model.h2_storage_level_min_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_min_rule)
            model.h2_direct_market_link_constr = pyo.Constraint(model.TimePeriods, rule=h2_direct_market_link_rule)
            if hasattr(model, 'vom_storage_cycle') and model.vom_storage_cycle > 1e-9:
                def h2_net_to_storage_rule(m, t): return m.H2_net_to_storage[t] >= m.mHydrogenProduced[t] - m.H2_to_market[t]
                model.h2_net_to_storage_constr = pyo.Constraint(model.TimePeriods, rule=h2_net_to_storage_rule)

        # Optional Capacity Factor
        if ENABLE_H2_CAP_FACTOR: model.h2_prod_req_constr = pyo.Constraint(rule=h2_CapacityFactor_rule)

        # Startup/Shutdown
        if ENABLE_STARTUP_SHUTDOWN:
            model.electrolyzer_on_off_logic_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_on_off_logic_rule)
            model.electrolyzer_min_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_rule)
            model.electrolyzer_max_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_max_power_rule)
            model.electrolyzer_startup_shutdown_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_startup_shutdown_exclusivity_rule)
            model.electrolyzer_min_uptime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_uptime_rule)
            model.electrolyzer_min_downtime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_downtime_rule)

        # Degradation
        if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: model.electrolyzer_degradation_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_degradation_rule)

        # Ancillary Service Capability
        model.Turbine_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmax_rule)
        model.Turbine_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmin_rule)
        model.Turbine_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RU_rule)
        model.Turbine_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RD_rule)
        model.Electrolyzer_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmax_rule)
        model.Electrolyzer_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmin_rule)
        model.Electrolyzer_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RU_rule) # Check ramp capability
        model.Electrolyzer_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RD_rule) # Check ramp capability

        # Link Component AS to Total
        model.link_Total_RegUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegUp_rule)
        model.link_Total_RegDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegDown_rule)
        model.link_Total_SR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_SR_rule)
        model.link_Total_NSR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_NSR_rule)
        if isinstance(model.Total_ECRS, pyo.Var): model.link_Total_ECRS_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_ECRS_rule)
        if isinstance(model.Total_30Min, pyo.Var): model.link_Total_30Min_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_30Min_rule)

        # Ramping Cost Linearization
        if hasattr(model, 'cost_electrolyzer_ramping') and model.cost_electrolyzer_ramping > 1e-9:
            def electrolyzer_ramp_linearization_rule(m, t):
                if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] == m.pElectrolyzerRampPos[t] - m.pElectrolyzerRampNeg[t]
            model.electrolyzer_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_ramp_linearization_rule)

    except Exception as e: logger.error(f"Error during constraint definition: {e}", exc_info=True); raise

    # =========================================================================
    # OBJECTIVE FUNCTION (Maximize Profit)
    # =========================================================================
    logger.info("Defining objective function (Maximize Profit)...")
    try:
        model.EnergyRevenue = pyo.Expression(rule=EnergyRevenue_rule)
        model.HydrogenRevenue = pyo.Expression(rule=HydrogenRevenue_rule)
        iso_revenue_rule_map = {
            'CAISO': AncillaryRevenue_CAISO_rule, 'ERCOT': AncillaryRevenue_ERCOT_rule,
            'ISONE': AncillaryRevenue_ISONE_rule, 'MISO': AncillaryRevenue_MISO_rule,
            'NYISO': AncillaryRevenue_NYISO_rule, 'PJM': AncillaryRevenue_PJM_rule,
            'SPP': AncillaryRevenue_SPP_rule,
        }
        if target_iso in iso_revenue_rule_map:
            model.AncillaryRevenue = pyo.Expression(rule=iso_revenue_rule_map[target_iso])
        else:
            logger.warning(f"No AncillaryRevenue rule for TARGET_ISO='{target_iso}'. Setting to 0.")
            model.AncillaryRevenue = pyo.Expression(initialize=0.0)
        model.OpexCost = pyo.Expression(rule=OpexCost_rule)
        def TotalProfit_Objective_rule(m): return m.EnergyRevenue + m.AncillaryRevenue + m.HydrogenRevenue - m.OpexCost
        model.TotalProfit_Objective = pyo.Objective(rule=TotalProfit_Objective_rule, sense=pyo.maximize)
    except Exception as e: logger.error(f"Error during objective definition: {e}", exc_info=True); raise

    logger.info("Standardized model created successfully.")
    return model
