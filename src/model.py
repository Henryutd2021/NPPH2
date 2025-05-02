# src/model.py
import pyomo.environ as pyo
from logging_setup import logger
import pandas as pd
from config import (
    TARGET_ISO, HOURS_IN_YEAR,
    ENABLE_NUCLEAR_GENERATOR, ENABLE_ELECTROLYZER, ENABLE_LOW_TEMP_ELECTROLYZER, ENABLE_BATTERY,
    ENABLE_H2_STORAGE, ENABLE_H2_CAP_FACTOR, ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING, ENABLE_STARTUP_SHUTDOWN,
    CAN_PROVIDE_ANCILLARY_SERVICES # Import derived flag
)
# Import constraint rules selectively based on enabled features
from constraints import (
    build_piecewise_constraints, steam_balance_rule, power_balance_rule,
    constant_turbine_power_rule,
    link_Total_RegUp_rule, link_Total_RegDown_rule, link_Total_SR_rule, link_Total_NSR_rule,
    link_Total_ECRS_rule, link_Total_30Min_rule, link_Total_RampUp_rule, link_Total_RampDown_rule,
    link_Total_UncU_rule,
    # Import rules conditionally later in the function
)
# Import revenue rules
from revenue_cost import (
    EnergyRevenue_rule, HydrogenRevenue_rule, AncillaryRevenue_CAISO_rule,
    AncillaryRevenue_ERCOT_rule, AncillaryRevenue_ISONE_rule, AncillaryRevenue_MISO_rule,
    AncillaryRevenue_NYISO_rule, AncillaryRevenue_PJM_rule, AncillaryRevenue_SPP_rule,
    OpexCost_rule,
)

# --- HELPER FUNCTION (MODIFIED FOR BOOLEAN HANDLING) ---
def get_sys_param(param_name, default=None, required=False):
    """Safely gets a parameter value from the system DataFrame, handling type conversions."""
    # This function is called within create_model, assumes df_system exists in data_inputs
    global df_system # Access the df_system loaded within create_model

    try:
        val = df_system.loc[param_name, 'Value']

        # Handle missing values
        if pd.isna(val):
            if required:
                raise ValueError(f"Missing essential system parameter: {param_name}")
            if default is not None:
                # logger.warning(f"System param '{param_name}' is NaN. Using default: {default}")
                return default
            else:
                # logger.debug(f"Optional system param '{param_name}' is NaN and no default provided.")
                return None # Return None if no default and not required

        # --- Type Conversion Logic ---
        # 1. Boolean Check (Case-insensitive)
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower == 'true':
                return True
            elif val_lower == 'false':
                return False
            # Add specific checks if parameter name implies boolean
            elif 'Require' in param_name or 'Enable' in param_name or 'Use' in param_name:
                 logger.warning(f"Parameter '{param_name}' looks boolean but value is '{val}'. Attempting interpretation (defaulting to False if unclear).")
                 # Could add more sophisticated checks or default to False/error
                 return False # Defaulting to False if unsure

        # 2. Integer Check (for specific parameters)
        if 'MinUpTime' in param_name or 'MinDownTime' in param_name or \
           'initial_status' in param_name or 'Lifetime_years' in param_name or \
           'plant_lifetime_years' in param_name: # Added plant_lifetime
            try:
                return int(float(val)) # Use float first to handle "1.0" etc.
            except (ValueError, TypeError) as e:
                logger.error(f"Parameter '{param_name}' expected int, got '{val}'. Error: {e}")
                if required: raise
                return default # Return default if conversion fails and not required

        # 3. Comma-separated String Check (for breakpoints/values)
        # Keep this check relatively early if numbers might be misinterpreted as bools/ints
        if isinstance(val, str) and ',' in val and \
           ('Breakpoints' in param_name or 'Values' in param_name or 'Outputs' in param_name):
             # Return the raw string for parsing later in the model setup
             return val.strip()

        # 4. Float Check (default for numeric parameters)
        try:
            return float(val)
        except (ValueError, TypeError):
            # If it's not float, bool, int, or comma-list, return as string (or default)
            logger.debug(f"Parameter '{param_name}' value '{val}' not converted to numeric/bool. Returning as string or default.")
            # Only return string if no default is provided, otherwise return default
            return val if default is None else default

    except KeyError:
        # Parameter name not found in the DataFrame index
        if required:
            raise ValueError(f"Missing essential system parameter: {param_name}")
        if default is not None:
            # logger.warning(f"System param '{param_name}' not found. Using default: {default}")
            return default
        else:
            # logger.debug(f"Optional system param '{param_name}' not found and no default provided.")
            return None # Return None if optional and not found
    except Exception as e:
        logger.error(f"Unexpected error retrieving system parameter '{param_name}': {e}")
        if required: raise
        return default # Fallback to default on unexpected errors


def create_model(data_inputs, target_iso: str) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel(f"Optimize_Profit_Standardized_{target_iso}")
    model.TARGET_ISO = target_iso
    # Store config flags on model for easier access in constraint rules
    model.ENABLE_NUCLEAR_GENERATOR = ENABLE_NUCLEAR_GENERATOR
    model.ENABLE_ELECTROLYZER = ENABLE_ELECTROLYZER
    model.ENABLE_BATTERY = ENABLE_BATTERY
    model.ENABLE_H2_STORAGE = ENABLE_H2_STORAGE
    model.LTE_MODE = ENABLE_LOW_TEMP_ELECTROLYZER if ENABLE_ELECTROLYZER else False # LTE only relevant if electrolyzer enabled
    # Store derived AS capability flag
    model.CAN_PROVIDE_ANCILLARY_SERVICES = CAN_PROVIDE_ANCILLARY_SERVICES

    logger.info(f"Creating STANDARDIZED model for {target_iso} with features:")
    logger.info(f"  Nuclear Generator: {ENABLE_NUCLEAR_GENERATOR}")
    logger.info(f"  Electrolyzer: {ENABLE_ELECTROLYZER} (LTE Mode: {model.LTE_MODE})")
    logger.info(f"  Battery: {ENABLE_BATTERY}")
    logger.info(f"  Ancillary Service Capability: {CAN_PROVIDE_ANCILLARY_SERVICES}") # Log AS capability
    logger.info(f"  H2 Storage: {ENABLE_H2_STORAGE}")
    logger.info(f"  Nonlinear Turbine: {ENABLE_NONLINEAR_TURBINE_EFF and ENABLE_NUCLEAR_GENERATOR}")
    logger.info(f"  Degradation: {ENABLE_ELECTROLYZER_DEGRADATION_TRACKING and ENABLE_ELECTROLYZER}")
    logger.info(f"  Startup/Shutdown: {ENABLE_STARTUP_SHUTDOWN and ENABLE_ELECTROLYZER}")
    logger.info(f"  H2 Cap Factor: {ENABLE_H2_CAP_FACTOR and ENABLE_ELECTROLYZER}")

    # =========================================================================
    # SETS & PARAMETERS
    # =========================================================================
    logger.info("Loading parameters...")
    try:
        # Time periods
        if 'df_price_hourly' not in data_inputs or data_inputs['df_price_hourly'] is None:
             raise ValueError("Essential data 'df_price_hourly' not found in data_inputs.")
        nT = len(data_inputs['df_price_hourly'])
        if nT == 0: raise ValueError("Price data is empty.")
        logger.info(f"Model time horizon set to {nT} periods.")
        model.TimePeriods = pyo.Set(initialize=pyo.RangeSet(1, nT), ordered=True)

        # System parameters file
        if 'df_system' not in data_inputs or data_inputs['df_system'] is None:
             raise ValueError("Essential data 'df_system' not found in data_inputs.")
        global df_system # Make df_system accessible to the helper function
        df_system = data_inputs['df_system']

        # --- Load Parameters based on Enabled Features ---
        # Use the modified get_sys_param helper which now handles types
        model.delT_minutes = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('delT_minutes', 60.0, required=True))
        model.AS_Duration = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('AS_Duration', 0.25)) # Expects float
        model.plant_lifetime_years = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('plant_lifetime_years', 30)) # Expects int

        # Nuclear Generator Parameters
        if ENABLE_NUCLEAR_GENERATOR:
            model.qSteam_Total = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Total_MWth', required=True))
            model.qSteam_Turbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_min_MWth', required=True))
            model.qSteam_Turbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('qSteam_Turbine_max_MWth', required=True))
            model.pTurbine_min = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_min_MW', required=True))
            model.pTurbine_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_max_MW', required=True))
            ramp_up_pct_min = get_sys_param('Turbine_RampUp_Rate_Percent_per_Min', 1.0)
            ramp_down_pct_min = get_sys_param('Turbine_RampDown_Rate_Percent_per_Min', 1.0)
            # Calculate ramp rates based on pTurbine_max (ensure it's loaded first)
            p_turb_max_val = get_sys_param('pTurbine_max_MW', required=True) # Get value again for calculation
            model.RU_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_up_pct_min * 60 / 100 * p_turb_max_val)
            model.RD_Turbine_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_down_pct_min * 60 / 100 * p_turb_max_val)
            model.vom_turbine = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_turbine_USD_per_MWh', 0))
            model.convertTtE_const = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4))

            model.nonlinear_turbine_enabled_in_model = False
            if ENABLE_NONLINEAR_TURBINE_EFF:
                 try:
                    q_bps_str = get_sys_param('qSteam_Turbine_Breakpoints_MWth', required=True)
                    p_vals_str = get_sys_param('pTurbine_Outputs_at_Breakpoints_MW', required=True)
                    # Ensure they are strings before splitting
                    if not isinstance(q_bps_str, str) or not isinstance(p_vals_str, str):
                         raise TypeError("Turbine breakpoint/value data is not a comma-separated string.")
                    q_breakpoints = sorted([float(x.strip()) for x in q_bps_str.split(',')])
                    p_values = [float(x.strip()) for x in p_vals_str.split(',')]
                    if len(q_breakpoints) != len(p_values): raise ValueError("Turbine breakpoint lengths differ.")
                    if not q_breakpoints: raise ValueError("Turbine breakpoints cannot be empty.")
                    model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=q_breakpoints, ordered=True)
                    pTurbine_vals_at_qTurbine_bp = dict(zip(q_breakpoints, p_values))
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize=pTurbine_vals_at_qTurbine_bp)
                    model.nonlinear_turbine_enabled_in_model = True
                    logger.info("Enabled non-linear turbine efficiency.")
                 except Exception as e:
                    logger.error(f"Error loading turbine piecewise data: {e}. Falling back to constant efficiency.")
                    # Fallback logic needs parameters loaded first
                    q_turb_min_val = get_sys_param('qSteam_Turbine_min_MWth', required=True)
                    q_turb_max_val = get_sys_param('qSteam_Turbine_max_MWth', required=True)
                    conv_const = get_sys_param('Turbine_Thermal_Elec_Efficiency_Const', 0.4)
                    model.qTurbine_efficiency_breakpoints = pyo.Set(initialize=[q_turb_min_val, q_turb_max_val], ordered=True)
                    min_p_fallback = q_turb_min_val * conv_const
                    max_p_fallback = q_turb_max_val * conv_const
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(model.qTurbine_efficiency_breakpoints, initialize={q_turb_min_val: min_p_fallback, q_turb_max_val: max_p_fallback})

            # HTE Steam Ramp Limit (Only relevant if HTE enabled)
            if ENABLE_ELECTROLYZER and not model.LTE_MODE:
                model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour', float('inf')))
            else:
                model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(initialize=float('inf')) # Set default if not HTE

            # LTE Turbine Setpoint (Only relevant if LTE enabled)
            if ENABLE_ELECTROLYZER and model.LTE_MODE:
                 model.pTurbine_LTE_setpoint = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('pTurbine_LTE_setpoint_MW', p_turb_max_val, required=False)) # Default to max power if not specified

        # Electrolyzer Parameters
        if ENABLE_ELECTROLYZER:
            # Get max capacity bounds first for calculations
            p_elec_max_ub = get_sys_param('pElectrolyzer_max_upper_bound_MW', required=True)
            p_elec_max_lb = get_sys_param('pElectrolyzer_max_lower_bound_MW', 0.0)
            p_elec_min_val = get_sys_param('pElectrolyzer_min_MW', required=True) # Min operating power

            model.pElectrolyzer_min = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_min_val)
            model.pElectrolyzer_max_upper_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_max_ub)
            model.pElectrolyzer_max_lower_bound = pyo.Param(within=pyo.NonNegativeReals, initialize=p_elec_max_lb)

            ramp_up_elec_pct_min = get_sys_param('Electrolyzer_RampUp_Rate_Percent_per_Min', 10.0)
            ramp_down_elec_pct_min = get_sys_param('Electrolyzer_RampDown_Rate_Percent_per_Min', 10.0)
            # Ramp rates are % of MAX capacity (upper bound)
            model.RU_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_up_elec_pct_min * 60 / 100)
            model.RD_Electrolyzer_percent_hourly = pyo.Param(within=pyo.NonNegativeReals, initialize=ramp_down_elec_pct_min * 60 / 100)

            model.vom_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_electrolyzer_USD_per_MWh', 0))
            model.cost_water_per_kg_h2 = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_water_USD_per_kg_h2', 0))
            model.cost_electrolyzer_ramping = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_electrolyzer_ramping_USD_per_MW_ramp', 0))
            model.cost_electrolyzer_capacity = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_electrolyzer_capacity_USD_per_MW_year', 0)) # Annual cost

            try:
                p_elec_bps_str = get_sys_param('pElectrolyzer_Breakpoints_MW', required=True)
                ke_vals_str = get_sys_param('ke_H2_Values_MWh_per_kg', required=True)
                if not isinstance(p_elec_bps_str, str) or not isinstance(ke_vals_str, str):
                     raise TypeError("Electrolyzer breakpoint/ke data is not a comma-separated string.")
                p_elec_breakpoints = sorted([float(x.strip()) for x in p_elec_bps_str.split(',')])
                ke_values = [float(x.strip()) for x in ke_vals_str.split(',')]
                if not p_elec_breakpoints: raise ValueError("Electrolyzer power breakpoints empty.")
                if len(p_elec_breakpoints) != len(ke_values): raise ValueError("Electrolyzer breakpoints/ke lengths differ.")
                model.pElectrolyzer_efficiency_breakpoints = pyo.Set(initialize=p_elec_breakpoints, ordered=True)
                ke_vals_dict = dict(zip(p_elec_breakpoints, ke_values))
                if any(v <= 1e-9 for v in ke_values): logger.warning("Found zero/near-zero ke_H2 values.")
                model.ke_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=ke_vals_dict, within=pyo.NonNegativeReals)

                if not model.LTE_MODE: # Only load kt if HTE mode
                    kt_vals_str = get_sys_param('kt_H2_Values_MWth_per_kg', required=True)
                    if not isinstance(kt_vals_str, str):
                         raise TypeError("Electrolyzer kt data is not a comma-separated string.")
                    kt_values = [float(x.strip()) for x in kt_vals_str.split(',')]
                    if len(p_elec_breakpoints) != len(kt_values): raise ValueError("HTE breakpoints/kt lengths differ.")
                    kt_vals_dict = dict(zip(p_elec_breakpoints, kt_values))
                    model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_vals_dict, within=pyo.NonNegativeReals)
                else: # If LTE, create kt param filled with zeros
                    kt_zero_dict = {bp: 0.0 for bp in p_elec_breakpoints}
                    model.kt_H2_values = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=kt_zero_dict)
                logger.info("Loaded electrolyzer piecewise parameters (ke, kt).")
            except Exception as e:
                logger.error(f"Error loading electrolyzer piecewise data: {e}.")
                raise ValueError("Failed to load essential electrolyzer efficiency data.")

            if ENABLE_STARTUP_SHUTDOWN:
                model.cost_startup_electrolyzer = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('cost_startup_electrolyzer_USD_per_startup', 0))
                model.MinUpTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinUpTimeElectrolyzer_hours', 1))
                model.MinDownTimeElectrolyzer = pyo.Param(within=pyo.PositiveIntegers, initialize=get_sys_param('MinDownTimeElectrolyzer_hours', 1))
                # Initial status should be 0 or 1, convert from potential float/string
                init_status_raw = get_sys_param('uElectrolyzer_initial_status_0_or_1', 0)
                init_status = 1 if int(float(init_status_raw)) == 1 else 0
                model.uElectrolyzer_initial = pyo.Param(within=pyo.Binary, initialize=init_status)

            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                model.DegradationStateInitial = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationStateInitial_Units', 0.0))
                model.DegradationFactorOperation = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorOperation_Units_per_Hour_at_MaxLoad', 0.0))
                model.DegradationFactorStartup = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('DegradationFactorStartup_Units_per_Startup', 0.0))

            if ENABLE_H2_CAP_FACTOR:
                 model.h2_target_capacity_factor = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('h2_target_capacity_factor_fraction', 0.0))

            model.H2_value = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_value_USD_per_kg', required=True))

            if ENABLE_H2_STORAGE:
                h2_storage_max = get_sys_param('H2_storage_capacity_max_kg', required=True)
                h2_storage_min = get_sys_param('H2_storage_capacity_min_kg', 0)
                model.H2_storage_capacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=h2_storage_max)
                model.H2_storage_capacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=h2_storage_min)
                initial_level_raw = get_sys_param('H2_storage_level_initial_kg', h2_storage_min)
                # Ensure initial level is within bounds
                initial_level = max(h2_storage_min, min(h2_storage_max, initial_level_raw))
                model.H2_storage_level_initial = pyo.Param(within=pyo.NonNegativeReals, initialize=initial_level)
                model.H2_storage_charge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_charge_rate_max_kg_per_hr', required=True))
                model.H2_storage_discharge_rate_max = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('H2_storage_discharge_rate_max_kg_per_hr', required=True))
                model.storage_charge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_charge_eff_fraction', 1.0))
                model.storage_discharge_eff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('storage_discharge_eff_fraction', 1.0))
                model.vom_storage_cycle = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_storage_cycle_USD_per_kg_cycled', 0))

        # Battery Parameters
        if ENABLE_BATTERY:
            logger.info("Configuring battery storage parameters...")
            batt_cap_max = get_sys_param('BatteryCapacity_max_MWh', required=True)
            batt_cap_min = get_sys_param('BatteryCapacity_min_MWh', 0.0)
            model.BatteryCapacity_max = pyo.Param(within=pyo.NonNegativeReals, initialize=batt_cap_max)
            model.BatteryCapacity_min = pyo.Param(within=pyo.NonNegativeReals, initialize=batt_cap_min)
            model.BatteryPowerRatio = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryPowerRatio_MW_per_MWh', 0.25, required=True))
            model.BatteryChargeEff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryChargeEff', 0.95))
            model.BatteryDischargeEff = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryDischargeEff', 0.95))
            model.BatterySOC_min_fraction = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatterySOC_min_fraction', 0.10))
            model.BatterySOC_initial_fraction = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatterySOC_initial_fraction', 0.50))
            # Ensure the value from get_sys_param is a Python boolean before initializing
            batt_cyclic_val = get_sys_param('BatteryRequireCyclicSOC', True) # get_sys_param now handles conversion
            model.BatteryRequireCyclicSOC = pyo.Param(within=pyo.Boolean, initialize=batt_cyclic_val)
            model.BatteryRampRate = pyo.Param(within=pyo.PercentFraction, initialize=get_sys_param('BatteryRampRate_fraction_per_hour', 1.0))
            model.BatteryCapex_USD_per_MWh_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryCapex_USD_per_MWh_year', 0.0)) # Annual cost
            model.BatteryCapex_USD_per_MW_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryCapex_USD_per_MW_year', 0.0)) # Annual cost
            model.BatteryFixedOM_USD_per_MWh_year = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('BatteryFixedOM_USD_per_MWh_year', 0.0)) # Annual cost
            # Optional VOM per MWh cycled
            if get_sys_param('vom_battery_per_mwh_cycled', None) is not None:
                 model.vom_battery_per_mwh_cycled = pyo.Param(within=pyo.NonNegativeReals, initialize=get_sys_param('vom_battery_per_mwh_cycled'))


        # Grid Interaction Parameters
        default_grid_max = p_turb_max_val if ENABLE_NUCLEAR_GENERATOR else 1000.0
        default_grid_min = -default_grid_max # Symmetric default
        model.pIES_min = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_min_MW', default_grid_min))
        model.pIES_max = pyo.Param(within=pyo.Reals, initialize=get_sys_param('pIES_max_MW', default_grid_max))

        # --- Hourly Data Parameters ---
        df_price = data_inputs['df_price_hourly']
        if len(df_price) < nT: raise ValueError(f"Energy price data missing rows ({len(df_price)} vs {nT}).")
        energy_price_col = 'Price ($/MWh)' # Adjust if column name differs
        if energy_price_col not in df_price.columns: raise ValueError(f"'{energy_price_col}' not found in price data.")
        # Ensure index matches RangeSet(1, nT)
        energy_price_dict = {t: df_price[energy_price_col].iloc[t-1] for t in model.TimePeriods}
        model.energy_price = pyo.Param(model.TimePeriods, initialize=energy_price_dict, within=pyo.Reals)

        # Load Optional DataFrames if present
        df_ANSprice = data_inputs.get('df_ANSprice_hourly', None)
        df_ANSmile = data_inputs.get('df_ANSmile_hourly', None)
        df_ANSdeploy = data_inputs.get('df_ANSdeploy_hourly', None)
        df_ANSwinrate = data_inputs.get('df_ANSwinrate_hourly', None)

        # Helper to load hourly data safely
        def get_hourly_param_from_df(t, df, col_name, default=0.0, required=False):
             if df is None:
                 if required: raise ValueError(f"Required data file for {col_name} not loaded.")
                 return default
             filename = getattr(df, 'attrs', {}).get('filename', 'DataFrame') # Get filename if stored
             if col_name in df.columns:
                 try:
                     # Use t-1 for 0-based iloc access from 1-based model.TimePeriods index t
                     val = df[col_name].iloc[t-1]
                     return val if pd.notna(val) else default
                 except IndexError:
                      logger.warning(f"Index {t-1} out of bounds for '{col_name}' in {filename} (length {len(df)}). Using default {default}.")
                      return default
                 except Exception as e:
                      logger.error(f"Error reading '{col_name}' at index {t-1} from {filename}: {e}")
                      if required: raise
                      return default
             else:
                 # Column not found
                 if required: raise ValueError(f"Required column '{col_name}' not in {filename}.")
                 # logger.debug(f"Optional column '{col_name}' not found in {filename}. Using default {default}.")
                 return default

        # Define ISO service map locally or import if needed elsewhere
        iso_service_map = {
            'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
            'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
            'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
            'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'],
            'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
            'ISONE': ['Spin10', 'NSpin10', 'OR30'],
            'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
        }
        if target_iso not in iso_service_map:
            raise ValueError(f"AS definitions missing for ISO: {target_iso}")

        logger.info(f"Loading AS parameters for {target_iso}...")
        if df_ANSprice is not None: df_ANSprice.attrs['filename'] = 'Price_ANS_hourly.csv'
        if df_ANSmile is not None: df_ANSmile.attrs['filename'] = 'MileageMultiplier_hourly.csv'
        if df_ANSdeploy is not None: df_ANSdeploy.attrs['filename'] = 'DeploymentFactor_hourly.csv'
        if df_ANSwinrate is not None: df_ANSwinrate.attrs['filename'] = 'WinningRate_hourly.csv'

        # Load ISO-Specific AS Parameters only if AS is possible
        if CAN_PROVIDE_ANCILLARY_SERVICES:
            for service in iso_service_map[target_iso]:
                is_factor = any(f in service for f in ['factor', 'score', 'ratio'])
                param_col_pattern = f"{service}_{target_iso}"

                # Price (p_*) - Load from df_ANSprice, skip if it's a factor param
                if not is_factor:
                     price_col_name = f'p_{param_col_pattern}'
                     # Only require price if the corresponding df exists
                     req_price = df_ANSprice is not None
                     param_dict = {t: get_hourly_param_from_df(t, df_ANSprice, price_col_name, default=0.0, required=False) for t in model.TimePeriods}
                     model.add_component(f'p_{service}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.Reals))

                # Deploy Factor (deploy_factor_*) - Load from df_ANSdeploy (optional)
                deploy_col_name = f'deploy_factor_{param_col_pattern}'
                param_dict = {t: get_hourly_param_from_df(t, df_ANSdeploy, deploy_col_name, default=0.0) for t in model.TimePeriods}
                model.add_component(f'deploy_factor_{service}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.PercentFraction))

                # Adder (loc_*) - Load from df_ANSprice (optional)
                loc_col_name = f'loc_{param_col_pattern}'
                param_dict = {t: get_hourly_param_from_df(t, df_ANSprice, loc_col_name, default=0.0) for t in model.TimePeriods}
                model.add_component(f'loc_{service}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.Reals))

                # Winning Rate (winning_rate_*) - Load from df_ANSwinrate (optional, default 1.0)
                win_col_name = f'winning_rate_{param_col_pattern}'
                param_dict = {t: get_hourly_param_from_df(t, df_ANSwinrate, win_col_name, default=1.0) for t in model.TimePeriods}
                model.add_component(f'winning_rate_{service}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.PercentFraction))

                # Mileage/Performance Factors - Load from df_ANSmile (optional)
                if target_iso == 'CAISO' and service in ['RegU', 'RegD']:
                     mileage_col_name = f'mileage_factor_{service}_{target_iso}'
                     param_dict = {t: get_hourly_param_from_df(t, df_ANSmile, mileage_col_name, default=1.0) for t in model.TimePeriods}
                     model.add_component(f'mileage_factor_{service}_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
                if target_iso == 'PJM':
                     if service == 'performance_score':
                          perf_col_name = f'performance_score_{target_iso}'
                          param_dict = {t: get_hourly_param_from_df(t, df_ANSmile, perf_col_name, default=1.0) for t in model.TimePeriods}
                          model.add_component(f'performance_score_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
                     if service == 'mileage_ratio':
                          mileage_col_name = f'mileage_ratio_{target_iso}'
                          param_dict = {t: get_hourly_param_from_df(t, df_ANSmile, mileage_col_name, default=1.0) for t in model.TimePeriods}
                          model.add_component(f'mileage_ratio_{target_iso}', pyo.Param(model.TimePeriods, initialize=param_dict, within=pyo.NonNegativeReals))
        else:
             logger.info("Ancillary services disabled by configuration. Skipping AS parameter loading.")


    except Exception as e:
        logger.error(f"Error during parameter loading: {e}", exc_info=True)
        raise

    # =========================================================================
    # VARIABLES
    # =========================================================================
    logger.info("Defining variables...")
    try:
        # Grid Interaction (always present)
        p_ies_min_val = pyo.value(model.pIES_min)
        p_ies_max_val = pyo.value(model.pIES_max)
        model.pIES = pyo.Var(model.TimePeriods, within=pyo.Reals, bounds=(p_ies_min_val, p_ies_max_val))

        # Nuclear Generator Variables
        if ENABLE_NUCLEAR_GENERATOR:
            q_turb_min_val = pyo.value(model.qSteam_Turbine_min)
            q_turb_max_val = pyo.value(model.qSteam_Turbine_max)
            p_turb_min_val = pyo.value(model.pTurbine_min)
            p_turb_max_val = pyo.value(model.pTurbine_max)
            model.qSteam_Turbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(q_turb_min_val, q_turb_max_val))
            model.pTurbine = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(p_turb_min_val, p_turb_max_val))

        # Electrolyzer Variables
        if ENABLE_ELECTROLYZER:
            p_elec_max_lb_val = pyo.value(model.pElectrolyzer_max_lower_bound)
            p_elec_max_ub_val = pyo.value(model.pElectrolyzer_max_upper_bound)
            p_elec_min_op_val = pyo.value(model.pElectrolyzer_min) # Min operating level when ON
            # Optimized capacity variable
            model.pElectrolyzer_max = pyo.Var(within=pyo.NonNegativeReals, bounds=(p_elec_max_lb_val, p_elec_max_ub_val), initialize=p_elec_max_ub_val)
            # Hourly operation variables (bounds depend on optimized capacity and min operating level)
            # Bounds will be set dynamically by constraints using pElectrolyzer_max and pElectrolyzer_min
            model.pElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Actual power consumption
            model.pElectrolyzerSetpoint = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals) # Target before AS
            model.mHydrogenProduced = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            if not model.LTE_MODE:
                 model.qSteam_Electrolyzer = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            # else: qSteam_Electrolyzer is effectively 0, no variable needed

            if ENABLE_STARTUP_SHUTDOWN:
                model.uElectrolyzer = pyo.Var(model.TimePeriods, within=pyo.Binary)
                model.vElectrolyzerStartup = pyo.Var(model.TimePeriods, within=pyo.Binary)
                model.wElectrolyzerShutdown = pyo.Var(model.TimePeriods, within=pyo.Binary)
            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                model.DegradationState = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            # Ramping variables only needed if costed/constrained significantly
            if pyo.value(model.cost_electrolyzer_ramping) > 1e-9:
                model.pElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
                model.pElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            if not model.LTE_MODE and pyo.value(model.Ramp_qSteam_Electrolyzer_limit) < float('inf'):
                 model.qSteamElectrolyzerRampPos = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
                 model.qSteamElectrolyzerRampNeg = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

            if ENABLE_H2_STORAGE:
                h2_storage_min_val = pyo.value(model.H2_storage_capacity_min)
                h2_storage_max_val = pyo.value(model.H2_storage_capacity_max)
                h2_charge_max_val = pyo.value(model.H2_storage_charge_rate_max)
                h2_discharge_max_val = pyo.value(model.H2_storage_discharge_rate_max)
                model.H2_storage_level = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(h2_storage_min_val, h2_storage_max_val))
                # Flow rate INTO storage tank (before eff loss)
                model.H2_to_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, h2_charge_max_val))
                # Flow rate OUT of storage tank (before eff loss applied in balance)
                model.H2_from_storage = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, bounds=(0, h2_discharge_max_val))
                # Flow rate directly to market/offtake
                model.H2_to_market = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)

        # Battery Variables
        if ENABLE_BATTERY:
            batt_cap_lb_val = pyo.value(model.BatteryCapacity_min)
            batt_cap_ub_val = pyo.value(model.BatteryCapacity_max)
            # Optimized capacity variable
            model.BatteryCapacity_MWh = pyo.Var(within=pyo.NonNegativeReals, bounds=(batt_cap_lb_val, batt_cap_ub_val), initialize=(batt_cap_lb_val + batt_cap_ub_val) / 2)
            # Power linked to capacity - bounds set by constraint
            model.BatteryPower_MW = pyo.Var(within=pyo.NonNegativeReals)
            # SOC bounds depend on optimized capacity - set by constraint
            model.BatterySOC = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            # Charge/Discharge bounds depend on optimized power - set by constraint
            model.BatteryCharge = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            model.BatteryDischarge = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals)
            # Binary variables for charge/discharge state
            model.BatteryBinaryCharge = pyo.Var(model.TimePeriods, within=pyo.Binary)
            model.BatteryBinaryDischarge = pyo.Var(model.TimePeriods, within=pyo.Binary)

        # --- Ancillary Service Variables ---
        # Define component and total AS vars ONLY IF the system CAN provide AS
        as_service_list = ['RegUp', 'RegDown', 'SR', 'NSR', 'ECRS', 'ThirtyMin', 'RampUp', 'RampDown', 'UncU'] # Internal names
        if CAN_PROVIDE_ANCILLARY_SERVICES:
            logger.info("Defining Ancillary Service Variables...")
            # Component Bids (only for enabled components)
            if ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY):
                for service in as_service_list: model.add_component(f"{service}_Turbine", pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0))
            if ENABLE_ELECTROLYZER:
                for service in as_service_list: model.add_component(f"{service}_Electrolyzer", pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0))
            if ENABLE_BATTERY:
                for service in as_service_list: model.add_component(f"{service}_Battery", pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0))

            # Total AS Bids (Define as Vars)
            # Always define core AS Vars if CAN_PROVIDE_ANCILLARY_SERVICES is True
            model.Total_RegUp = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.Total_RegDown = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.Total_SR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            model.Total_NSR = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

            # Conditionally define optional AS Vars based on whether their price parameter exists
            if hasattr(model, f'p_ECRS_{target_iso}'):
                model.Total_ECRS = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['TMR', 'Res30', 'OR30', 'STR']):
                model.Total_30Min = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['RMU', 'RamU']):
                model.Total_RampUp = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['RMD', 'RamD']):
                model.Total_RampDown = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)
            if hasattr(model, f'p_UncU_{target_iso}'):
                model.Total_UncU = pyo.Var(model.TimePeriods, within=pyo.NonNegativeReals, initialize=0.0)

        else: # CAN_PROVIDE_ANCILLARY_SERVICES is False
            logger.info("Ancillary Services disabled by configuration. Fixing AS bids to 0.")
            # Fix Total AS bids to 0 using Parameters
            model.Total_RegUp = pyo.Param(model.TimePeriods, initialize=0.0)
            model.Total_RegDown = pyo.Param(model.TimePeriods, initialize=0.0)
            model.Total_SR = pyo.Param(model.TimePeriods, initialize=0.0)
            model.Total_NSR = pyo.Param(model.TimePeriods, initialize=0.0)
            # Define others as Params=0 only if they *might* have been Vars otherwise (check for price param)
            if hasattr(model, f'p_ECRS_{target_iso}'): model.Total_ECRS = pyo.Param(model.TimePeriods, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['TMR', 'Res30', 'OR30', 'STR']): model.Total_30Min = pyo.Param(model.TimePeriods, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['RMU', 'RamU']): model.Total_RampUp = pyo.Param(model.TimePeriods, initialize=0.0)
            if any(hasattr(model, f'p_{s}_{target_iso}') for s in ['RMD', 'RamD']): model.Total_RampDown = pyo.Param(model.TimePeriods, initialize=0.0)
            if hasattr(model, f'p_UncU_{target_iso}'): model.Total_UncU = pyo.Param(model.TimePeriods, initialize=0.0)


    except Exception as e: logger.error(f"Error during variable definition: {e}", exc_info=True); raise

    # =========================================================================
    # PRECOMPUTE / UPDATE PARAMS BASED ON VARIABLES
    # =========================================================================
    try:
        if ENABLE_ELECTROLYZER:
            # Calculate inverse efficiency 1 / (MWh/kg) = kg/MWh
            model.ke_H2_inv_values = {
                 bp: 1.0 / model.ke_H2_values[bp] if abs(pyo.value(model.ke_H2_values[bp])) > 1e-9 else 1e9 # Avoid division by zero
                 for bp in model.pElectrolyzer_efficiency_breakpoints
            }
            # Calculate steam consumption at breakpoints if HTE
            if not model.LTE_MODE:
                 # Check if kt_H2_values exists before trying to use it
                 if hasattr(model, 'kt_H2_values'):
                      q_steam_at_pElec_bp = {
                          p_bp: (pyo.value(model.kt_H2_values[p_bp]) * pyo.value(model.ke_H2_inv_values[p_bp]) * p_bp
                                 if abs(pyo.value(model.ke_H2_values[p_bp])) > 1e-9 else 0)
                          for p_bp in model.pElectrolyzer_efficiency_breakpoints
                      }
                      model.qSteam_values_at_pElec_bp = pyo.Param(model.pElectrolyzer_efficiency_breakpoints, initialize=q_steam_at_pElec_bp)
                      logger.info("Calculated qSteam values at pElectrolyzer breakpoints for HTE.")
                 else:
                      logger.warning("kt_H2_values param missing, cannot calculate qSteam_values_at_pElec_bp for HTE.")


    except Exception as e: logger.error(f"Error during precomputation: {e}", exc_info=True); raise

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    logger.info("Defining constraints...")
    try:
        # --- Physical System Constraints ---
        model.power_balance_constr = pyo.Constraint(model.TimePeriods, rule=power_balance_rule)

        if ENABLE_NUCLEAR_GENERATOR:
            model.steam_balance_constr = pyo.Constraint(model.TimePeriods, rule=steam_balance_rule)
            if model.nonlinear_turbine_enabled_in_model:
                # Ensure the value param exists before building
                if hasattr(model, 'pTurbine_values_at_qTurbine_bp'):
                     build_piecewise_constraints(model, component_prefix='TurbinePower', input_var_name='qSteam_Turbine', output_var_name='pTurbine', breakpoint_set_name='qTurbine_efficiency_breakpoints', value_param_name='pTurbine_values_at_qTurbine_bp')
                else:
                     logger.error("Cannot build TurbinePower piecewise: pTurbine_values_at_qTurbine_bp missing.")
                     # Fallback to linear might be needed here if essential
            else: # Linear efficiency
                def linear_pTurbine_rule(m,t): return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const
                model.linear_pTurbine_constr = pyo.Constraint(model.TimePeriods, rule=linear_pTurbine_rule)

            # Ramp constraints only apply if not LTE mode
            from constraints import Turbine_RampUp_rule, Turbine_RampDown_rule
            model.Turbine_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampUp_rule)
            model.Turbine_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_RampDown_rule)

            if CAN_PROVIDE_ANCILLARY_SERVICES: # Turbine AS constraints only relevant if AS possible
                from constraints import Turbine_AS_Zero_rule, Turbine_AS_Pmax_rule, Turbine_AS_Pmin_rule, Turbine_AS_RU_rule, Turbine_AS_RD_rule
                model.turbine_as_zero_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Zero_rule)
                model.Turbine_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmax_rule)
                model.Turbine_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_Pmin_rule)
                model.Turbine_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RU_rule)
                model.Turbine_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Turbine_AS_RD_rule)
            if ENABLE_ELECTROLYZER and model.LTE_MODE:
                 model.const_turbine_power_constr = pyo.Constraint(model.TimePeriods, rule=constant_turbine_power_rule)

        if ENABLE_ELECTROLYZER:
            # Link actual power to optimized capacity
            def electrolyzer_capacity_limit_rule(m, t): return m.pElectrolyzer[t] <= m.pElectrolyzer_max
            model.electrolyzer_capacity_limit_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_capacity_limit_rule)
            # Link setpoint to optimized capacity
            def electrolyzer_setpoint_capacity_limit_rule(m,t): return m.pElectrolyzerSetpoint[t] <= m.pElectrolyzer_max
            model.electrolyzer_setpoint_capacity_limit_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_setpoint_capacity_limit_rule)

            # Piecewise constraints for H2 production and Steam consumption
            if hasattr(model, 'ke_H2_inv_values'):
                 build_piecewise_constraints(model, component_prefix='HydrogenProduction', input_var_name='pElectrolyzer', output_var_name='mHydrogenProduced', breakpoint_set_name='pElectrolyzer_efficiency_breakpoints', value_param_name='ke_H2_inv_values')
            else: logger.error("Cannot build HydrogenProduction piecewise: ke_H2_inv_values missing.")

            if not model.LTE_MODE and hasattr(model, 'qSteam_values_at_pElec_bp'):
                 build_piecewise_constraints(model, component_prefix='SteamConsumption', input_var_name='pElectrolyzer', output_var_name='qSteam_Electrolyzer', breakpoint_set_name='pElectrolyzer_efficiency_breakpoints', value_param_name='qSteam_values_at_pElec_bp')
            elif not model.LTE_MODE:
                 logger.error("Cannot build SteamConsumption piecewise for HTE: qSteam_values_at_pElec_bp missing.")

            # Ramp constraints
            from constraints import Electrolyzer_RampUp_rule, Electrolyzer_RampDown_rule
            model.Electrolyzer_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampUp_rule)
            model.Electrolyzer_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_RampDown_rule)
            # Steam ramp constraints (only if HTE and constrained)
            if not model.LTE_MODE and hasattr(model, 'qSteamElectrolyzerRampPos'):
                 from constraints import Steam_Electrolyzer_Ramp_rule
                 # Linearization link: dQ = Qramp_pos - Qramp_neg
                 def qSteam_ramp_linearization_rule(m, t):
                     if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                     # Need to handle potential missing qSteam_Electrolyzer if LTE? No, this block only runs if not LTE.
                     return m.qSteam_Electrolyzer[t] - m.qSteam_Electrolyzer[t-1] == m.qSteamElectrolyzerRampPos[t] - m.qSteamElectrolyzerRampNeg[t]
                 model.qSteam_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=qSteam_ramp_linearization_rule)
                 # Ramp limit constraint
                 model.Steam_Electrolyzer_Ramp_constr = pyo.Constraint(model.TimePeriods, rule=Steam_Electrolyzer_Ramp_rule)

            if CAN_PROVIDE_ANCILLARY_SERVICES: # Electrolyzer AS constraints only relevant if AS possible
                from constraints import Electrolyzer_AS_Pmax_rule, Electrolyzer_AS_Pmin_rule, Electrolyzer_AS_RU_rule, Electrolyzer_AS_RD_rule
                model.Electrolyzer_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmax_rule)
                model.Electrolyzer_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_Pmin_rule)
                model.Electrolyzer_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RU_rule)
                model.Electrolyzer_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Electrolyzer_AS_RD_rule)

            # Apply appropriate min/max power rules based on SU/SD flag
            if ENABLE_STARTUP_SHUTDOWN:
                # Import SU/SD specific rules
                from constraints import (electrolyzer_on_off_logic_rule,
                                         electrolyzer_min_power_when_on_rule, # Use new rule
                                         electrolyzer_max_power_rule, # Use existing rule for max power when ON
                                         electrolyzer_startup_shutdown_exclusivity_rule,
                                         electrolyzer_min_uptime_rule,
                                         electrolyzer_min_downtime_rule)
                model.electrolyzer_on_off_logic_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_on_off_logic_rule)
                # Add the min/max power constraints linked to ON/OFF status
                model.electrolyzer_min_power_when_on_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_when_on_rule)
                model.electrolyzer_max_power_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_max_power_rule)
                # Add other SU/SD constraints
                model.electrolyzer_startup_shutdown_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_startup_shutdown_exclusivity_rule)
                model.electrolyzer_min_uptime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_uptime_rule)
                model.electrolyzer_min_downtime_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_downtime_rule)
            else:
                # If SU/SD is disabled, only enforce the constant minimum operating level
                from constraints import electrolyzer_min_power_sds_disabled_rule
                model.electrolyzer_min_power_sds_disabled_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_min_power_sds_disabled_rule)
                # The upper bound (pElec <= pElec_max) is already handled by electrolyzer_capacity_limit_rule

            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                from constraints import electrolyzer_degradation_rule
                model.electrolyzer_degradation_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_degradation_rule)
            if ENABLE_H2_CAP_FACTOR:
                from constraints import h2_CapacityFactor_rule
                model.h2_prod_req_constr = pyo.Constraint(rule=h2_CapacityFactor_rule)

            # Ramping cost linearization (if cost > 0)
            if hasattr(model, 'pElectrolyzerRampPos'):
                def electrolyzer_ramp_linearization_rule(m, t):
                    if t == m.TimePeriods.first(): return pyo.Constraint.Skip
                    # Link change in actual power to ramp variables
                    return m.pElectrolyzer[t] - m.pElectrolyzer[t-1] == m.pElectrolyzerRampPos[t] - m.pElectrolyzerRampNeg[t]
                model.electrolyzer_ramp_linearization_constr = pyo.Constraint(model.TimePeriods, rule=electrolyzer_ramp_linearization_rule)

            if ENABLE_H2_STORAGE:
                # Import H2 storage constraint rules
                from constraints import (h2_prod_dispatch_rule, h2_storage_charge_limit_rule,
                                         h2_storage_discharge_limit_rule, h2_storage_level_max_rule,
                                         h2_storage_level_min_rule, h2_storage_balance_adj_rule) # Import the adjusted rule

                model.h2_storage_balance_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_balance_adj_rule)
                # Link production to market sales and input to storage
                model.h2_prod_dispatch_constr = pyo.Constraint(model.TimePeriods, rule=h2_prod_dispatch_rule)
                # Limit charge/discharge flow rates
                model.h2_storage_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_charge_limit_rule)
                model.h2_storage_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_discharge_limit_rule)
                # Storage level bounds
                model.h2_storage_level_max_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_max_rule)
                model.h2_storage_level_min_constr = pyo.Constraint(model.TimePeriods, rule=h2_storage_level_min_rule)

        if ENABLE_BATTERY:
            from constraints import (
                battery_soc_balance_rule, battery_charge_limit_rule, battery_discharge_limit_rule,
                battery_binary_exclusivity_rule, battery_soc_max_rule, battery_soc_min_rule,
                battery_ramp_up_rule, battery_ramp_down_rule, battery_discharge_ramp_up_rule,
                battery_discharge_ramp_down_rule,
                battery_cyclic_soc_lower_rule, battery_cyclic_soc_upper_rule, # Import new rules
                battery_power_capacity_link_rule,
                battery_min_cap_rule # Use the simpler min cap rule
            )
            model.battery_soc_balance_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_balance_rule)
            model.battery_charge_limit_constr = pyo.Constraint(model.TimePeriods, rule=battery_charge_limit_rule)
            model.battery_discharge_limit_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_limit_rule)
            model.battery_binary_exclusivity_constr = pyo.Constraint(model.TimePeriods, rule=battery_binary_exclusivity_rule)
            model.battery_soc_max_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_max_rule)
            model.battery_soc_min_constr = pyo.Constraint(model.TimePeriods, rule=battery_soc_min_rule)
            # Charge Ramping
            model.battery_charge_ramp_up_constr = pyo.Constraint(model.TimePeriods, rule=battery_ramp_up_rule)
            model.battery_charge_ramp_down_constr = pyo.Constraint(model.TimePeriods, rule=battery_ramp_down_rule)
            # Discharge Ramping
            model.battery_discharge_ramp_up_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_ramp_up_rule)
            model.battery_discharge_ramp_down_constr = pyo.Constraint(model.TimePeriods, rule=battery_discharge_ramp_down_rule)
            # --- *** FIX START: Add split cyclic SOC constraints *** ---
            if pyo.value(model.BatteryRequireCyclicSOC):
                model.battery_cyclic_soc_lower_constr = pyo.Constraint(rule=battery_cyclic_soc_lower_rule)
                model.battery_cyclic_soc_upper_constr = pyo.Constraint(rule=battery_cyclic_soc_upper_rule)
            # --- *** FIX END *** ---
            # Link Power and Capacity
            model.battery_power_capacity_link_constr = pyo.Constraint(rule=battery_power_capacity_link_rule)
            # Enforce Minimum Capacity (if applicable)
            model.battery_min_cap_constr = pyo.Constraint(rule=battery_min_cap_rule)

            if CAN_PROVIDE_ANCILLARY_SERVICES: # Battery AS constraints only relevant if AS possible
                from constraints import (Battery_AS_Pmax_rule, Battery_AS_Pmin_rule, Battery_AS_SOC_Up_rule,
                                         Battery_AS_SOC_Down_rule, Battery_AS_RU_rule, Battery_AS_RD_rule)
                model.Battery_AS_Pmax_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_Pmax_rule)
                model.Battery_AS_Pmin_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_Pmin_rule)
                model.Battery_AS_SOC_Up_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_SOC_Up_rule)
                model.Battery_AS_SOC_Down_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_SOC_Down_rule)
                model.Battery_AS_RU_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_RU_rule)
                model.Battery_AS_RD_constr = pyo.Constraint(model.TimePeriods, rule=Battery_AS_RD_rule)

        # --- Ancillary Service Linking Constraints ---
        # Only add linking constraints if AS is possible (Total_* are Vars)
        if CAN_PROVIDE_ANCILLARY_SERVICES:
            model.link_Total_RegUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegUp_rule)
            model.link_Total_RegDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RegDown_rule)
            model.link_Total_SR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_SR_rule)
            model.link_Total_NSR_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_NSR_rule)
            # Conditionally add links based on whether Total_* is a Var
            if isinstance(getattr(model, 'Total_ECRS', None), pyo.Var): model.link_Total_ECRS_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_ECRS_rule)
            if isinstance(getattr(model, 'Total_30Min', None), pyo.Var): model.link_Total_30Min_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_30Min_rule)
            if isinstance(getattr(model, 'Total_RampUp', None), pyo.Var): model.link_Total_RampUp_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RampUp_rule)
            if isinstance(getattr(model, 'Total_RampDown', None), pyo.Var): model.link_Total_RampDown_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_RampDown_rule)
            if isinstance(getattr(model, 'Total_UncU', None), pyo.Var): model.link_Total_UncU_constr = pyo.Constraint(model.TimePeriods, rule=link_Total_UncU_rule)

    except Exception as e: logger.error(f"Error during constraint definition: {e}", exc_info=True); raise

    # =========================================================================
    # OBJECTIVE FUNCTION (Maximize Profit)
    # =========================================================================
    logger.info("Defining objective function (Maximize Profit)...")
    try:
        # Energy Revenue (always calculated)
        model.EnergyRevenueExpr = pyo.Expression(rule=EnergyRevenue_rule)
        # Hydrogen Revenue (calculated if electrolyzer enabled)
        model.HydrogenRevenueExpr = pyo.Expression(rule=HydrogenRevenue_rule) # Rule handles ENABLE_ELECTROLYZER internally

        # Ancillary Service Revenue (calculated ONLY if CAN_PROVIDE_ANCILLARY_SERVICES is True)
        if CAN_PROVIDE_ANCILLARY_SERVICES:
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
            # If AS not possible, set expression to 0
            model.AncillaryRevenueExpr = pyo.Expression(initialize=0.0)

        # Operational Costs (calculated based on enabled components)
        model.OpexCostExpr = pyo.Expression(rule=OpexCost_rule) # Rule handles enabled components internally

        # Capital Costs (Annualized)
        def AnnualizedCapex_rule(m):
            total_annual_capex = 0.0
            # Simulation duration relative to a year
            scaling_factor = len(m.TimePeriods) * (pyo.value(m.delT_minutes) / 60.0) / 8760.0

            # Electrolyzer CAPEX (Annualized cost based on optimized capacity)
            if ENABLE_ELECTROLYZER and hasattr(m, 'cost_electrolyzer_capacity') and pyo.value(m.cost_electrolyzer_capacity) > 1e-9:
                 total_annual_capex += m.pElectrolyzer_max * m.cost_electrolyzer_capacity * scaling_factor

            # Battery CAPEX & Fixed O&M (Annualized costs based on optimized capacity/power)
            if ENABLE_BATTERY:
                 # Ensure params exist before using them
                 batt_cap_cost = pyo.value(getattr(m, 'BatteryCapex_USD_per_MWh_year', 0.0))
                 batt_pow_cost = pyo.value(getattr(m, 'BatteryCapex_USD_per_MW_year', 0.0))
                 batt_fom_cost = pyo.value(getattr(m, 'BatteryFixedOM_USD_per_MWh_year', 0.0))
                 total_annual_capex += (m.BatteryCapacity_MWh * batt_cap_cost +
                                        m.BatteryPower_MW * batt_pow_cost +
                                        m.BatteryCapacity_MWh * batt_fom_cost) * scaling_factor
            return total_annual_capex
        model.AnnualizedCapexExpr = pyo.Expression(rule=AnnualizedCapex_rule)

        # Total Profit Calculation
        def TotalProfit_Objective_rule(m):
            total_revenue = m.EnergyRevenueExpr + m.AncillaryRevenueExpr + m.HydrogenRevenueExpr
            total_opex = m.OpexCostExpr
            total_capex = m.AnnualizedCapexExpr
            return total_revenue - total_opex - total_capex
        model.TotalProfit_Objective = pyo.Objective(rule=TotalProfit_Objective_rule, sense=pyo.maximize)

    except Exception as e: logger.error(f"Error during objective definition: {e}", exc_info=True); raise

    logger.info("Standardized model created successfully.")
    return model

# --- Piecewise Helper (Keep as is) ---
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
