# src/model.py
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from config import (
    CAN_PROVIDE_ANCILLARY_SERVICES,
    ENABLE_BATTERY,
    ENABLE_ELECTROLYZER,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_H2_STORAGE,
    ENABLE_LOW_TEMP_ELECTROLYZER,
    ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_STARTUP_SHUTDOWN,
    HOURS_IN_YEAR,
    SIMULATE_AS_DISPATCH_EXECUTION,
    TARGET_ISO,
)
from constraints import (
    Battery_AS_Pmax_rule,
    Battery_AS_Pmin_rule,
    Battery_AS_RD_rule,
    Battery_AS_RU_rule,
    Battery_AS_SOC_Down_rule,
    Battery_AS_SOC_Up_rule,
    Electrolyzer_AS_Pmax_rule,
    Electrolyzer_AS_Pmin_rule,
    Electrolyzer_AS_RD_rule,
    Electrolyzer_AS_RU_rule,
    Electrolyzer_RampDown_rule,
    Electrolyzer_RampUp_rule,
    Steam_Electrolyzer_Ramp_rule,
    Turbine_AS_Pmax_rule,
    Turbine_AS_Pmin_rule,
    Turbine_AS_RD_rule,
    Turbine_AS_RU_rule,
    Turbine_AS_Zero_rule,
    Turbine_RampDown_rule,
    Turbine_RampUp_rule,
    battery_binary_exclusivity_rule,
    battery_charge_limit_rule,
    battery_cyclic_soc_lower_rule,
    battery_cyclic_soc_upper_rule,
    battery_discharge_limit_rule,
    battery_discharge_ramp_down_rule,
    battery_discharge_ramp_up_rule,
    battery_min_cap_rule,
    battery_power_capacity_link_rule,
    battery_ramp_down_rule,
    battery_ramp_up_rule,
    battery_regulation_balance_rule,
    battery_soc_balance_rule,
    battery_soc_max_rule,
    battery_soc_min_rule,
    constant_turbine_power_rule,
    define_actual_electrolyzer_power_rule,
    electrolyzer_degradation_rule,
    electrolyzer_max_power_rule,
    electrolyzer_min_downtime_rule,
    electrolyzer_min_power_sds_disabled_rule,
    electrolyzer_min_power_when_on_rule,
    electrolyzer_min_uptime_rule,
    electrolyzer_on_off_logic_rule,
    electrolyzer_regulation_balance_rule,
    electrolyzer_setpoint_min_power_rule,
    electrolyzer_startup_shutdown_exclusivity_rule,
    h2_CapacityFactor_rule,
    h2_prod_dispatch_rule,
    h2_storage_balance_adj_rule,
    h2_storage_charge_limit_rule,
    h2_storage_discharge_limit_rule,
    h2_storage_level_max_rule,
    h2_storage_level_min_rule,
    link_auxiliary_power_rule,
    link_deployed_to_bid_rule,
    link_setpoint_to_actual_power_if_not_simulating_dispatch_rule,
    link_Total_30Min_rule,
    link_Total_ECRS_rule,
    link_Total_NSR_rule,
    link_Total_RampDown_rule,
    link_Total_RampUp_rule,
    link_Total_RegDown_rule,
    link_Total_RegUp_rule,
    link_Total_SR_rule,
    link_Total_UncU_rule,
    power_balance_rule,
    restrict_grid_purchase_rule,
    steam_balance_rule,
    turbine_regulation_balance_rule,
)
from logging_setup import logger
from revenue_cost import (
    AncillaryRevenue_CAISO_rule,
    AncillaryRevenue_ERCOT_rule,
    AncillaryRevenue_ISONE_rule,
    AncillaryRevenue_MISO_rule,
    AncillaryRevenue_NYISO_rule,
    AncillaryRevenue_PJM_rule,
    AncillaryRevenue_SPP_rule,
    EnergyRevenue_rule,
    HydrogenRevenue_rule,
    OpexCost_rule,
)

df_system = None


def get_sys_param(param_name, default=None, required=False):
    """Safely gets a parameter value from the system DataFrame."""
    global df_system
    if df_system is None:
        if required:
            raise ValueError("df_system not loaded in get_sys_param.")
        return default
    try:
        if param_name not in df_system.index:
            if required:
                raise ValueError(
                    f"Missing essential system parameter: {param_name}")
            return default
        val = df_system.loc[param_name, "Value"]
        if pd.isna(val):
            if required:
                raise ValueError(
                    f"Missing essential system parameter (NaN value): {param_name}"
                )
            return default
        if isinstance(val, str):
            val_lower = val.strip().lower()
            if val_lower == "true":
                return True
            elif val_lower == "false":
                return False
            elif any(
                indicator in param_name.lower()
                for indicator in ["enable", "require", "use", "is_"]
            ):
                logger.warning(
                    f"Parameter '{param_name}' looks boolean but value is '{val}'. Interpreting as False unless explicitly 'true'."
                )
                return False
        if any(
            indicator in param_name
            for indicator in [
                "MinUpTime",
                "MinDownTime",
                "initial_status",
                "Lifetime_years",
                "plant_lifetime_years",
                "hours",
            ]
        ):
            try:
                return int(float(val))
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Parameter '{param_name}' expected int, got '{val}'. Error: {e}"
                )
                if required:
                    raise
                return default
        if any(
            indicator in param_name
            for indicator in ["Breakpoints", "Values", "Outputs"]
        ):
            return str(val).strip()
        try:
            return float(val)
        except (ValueError, TypeError):
            logger.debug(
                f"Parameter '{param_name}' value '{val}' not converted. Returning as is or default."
            )
            return val if default is None and not required else default
    except KeyError:
        if required:
            raise ValueError(
                f"Missing essential system parameter (KeyError): {param_name}"
            )
        return default
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving system parameter '{param_name}': {e}"
        )
        if required:
            raise
        return default


def build_piecewise_constraints(
    model: pyo.ConcreteModel,
    *,
    component_prefix: str,
    input_var_name: str,
    output_var_name: str,
    breakpoint_set_name: str,
    value_param_name: str,
) -> None:
    """Attach SOS2 piece-wise linear constraints in-place to `model`."""
    logger.info("Building piece-wise constraints for %s using SOS2",
                component_prefix)

    if not hasattr(model, input_var_name):
        logger.error(
            f"Input variable '{input_var_name}' not found for PWL {component_prefix}."
        )
        return
    if not hasattr(model, output_var_name):
        logger.error(
            f"Output variable '{output_var_name}' not found for PWL {component_prefix}."
        )
        return
    if not hasattr(model, breakpoint_set_name):
        logger.error(
            f"Breakpoint set '{breakpoint_set_name}' not found for PWL {component_prefix}."
        )
        return

    input_var = getattr(model, input_var_name)
    output_var = getattr(model, output_var_name)
    breakpoint_set_orig = getattr(model, breakpoint_set_name)

    value_data_source = None
    if isinstance(value_param_name, str):
        if hasattr(model, value_param_name):
            value_data_source = getattr(model, value_param_name)
        else:
            logger.error(
                f"Value parameter/dict named '{value_param_name}' not found for PWL {component_prefix}."
            )
            return
    elif isinstance(value_param_name, (dict, pyo.Param)):
        value_data_source = value_param_name
    else:
        logger.error(
            f"Value source '{value_param_name}' for PWL {component_prefix} is invalid type."
        )
        return

    if not (
        isinstance(value_data_source, pyo.Param) or isinstance(
            value_data_source, dict)
    ):
        logger.error(
            f"Value source for '{component_prefix}' is not a Pyomo Param or dict."
        )
        return

    breakpoint_set_to_use = breakpoint_set_orig
    if not breakpoint_set_orig.isordered():
        logger.warning(
            f"Breakpoint set {breakpoint_set_name} for {component_prefix} is not ordered. Sorting."
        )
        try:
            sorted_breakpoints_values = sorted(
                list(pyo.value(bp) for bp in breakpoint_set_orig)
            )
            ordered_set_attr_name = f"_ordered_{breakpoint_set_name}_{component_prefix}"
            if hasattr(model, ordered_set_attr_name):
                breakpoint_set_to_use = getattr(model, ordered_set_attr_name)
            else:
                new_ordered_set = pyo.Set(
                    initialize=sorted_breakpoints_values,
                    ordered=True,
                    name=ordered_set_attr_name,
                )
                setattr(model, ordered_set_attr_name, new_ordered_set)
                breakpoint_set_to_use = new_ordered_set
        except Exception as e:
            logger.error(
                f"Cannot sort breakpoint set {breakpoint_set_name} for {component_prefix}: {e}"
            )
            raise ValueError(
                f"Breakpoint set {breakpoint_set_name} must be ordered for SOS2."
            )

    lam_var_name = f"lambda_{component_prefix}"
    if not hasattr(model, lam_var_name):
        lam = pyo.Var(
            model.TimePeriods,
            breakpoint_set_to_use,
            bounds=(0, 1),
            within=pyo.NonNegativeReals,
        )
        setattr(model, lam_var_name, lam)
    else:
        lam = getattr(model, lam_var_name)

    sum_lambda_constr_name = f"{component_prefix}_sum_lambda"
    input_link_constr_name = f"{component_prefix}_input_link"
    output_link_constr_name = f"{component_prefix}_output_link"
    sos2_constr_name = f"SOS2_{component_prefix}"

    if not hasattr(model, sum_lambda_constr_name):

        def _sum_rule(m, t):
            return sum(lam[t, bp] for bp in breakpoint_set_to_use) == 1

        model.add_component(
            sum_lambda_constr_name,
            pyo.Constraint(model.TimePeriods, rule=_sum_rule),
        )

    if not hasattr(model, input_link_constr_name):

        def _input_link(m, t):
            return input_var[t] == sum(lam[t, bp] * bp for bp in breakpoint_set_to_use)

        model.add_component(
            input_link_constr_name,
            pyo.Constraint(model.TimePeriods, rule=_input_link),
        )

    if not hasattr(model, output_link_constr_name):

        def _output_link(m, t):
            if isinstance(value_data_source, pyo.Param):
                return output_var[t] == sum(
                    lam[t, bp] * value_data_source[bp] for bp in breakpoint_set_to_use
                )
            elif isinstance(value_data_source, dict):
                return output_var[t] == sum(
                    lam[t, bp] * value_data_source.get(bp, 0.0)
                    for bp in breakpoint_set_to_use
                )
            logger.critical(
                f"Value source for {component_prefix} output link is invalid type."
            )
            return pyo.Constraint.Skip

        model.add_component(
            output_link_constr_name,
            pyo.Constraint(model.TimePeriods, rule=_output_link),
        )

    if not hasattr(model, sos2_constr_name):

        def _sos2_rule(m, t):
            return [lam[t, bp] for bp in breakpoint_set_to_use]

        model.add_component(
            sos2_constr_name,
            pyo.SOSConstraint(model.TimePeriods, rule=_sos2_rule, sos=2),
        )


def create_model(
    data_inputs, target_iso: str, simulate_dispatch: bool
) -> pyo.ConcreteModel:
    """Creates the Pyomo ConcreteModel based on data and configuration flags."""
    model = pyo.ConcreteModel(f"Optimize_Profit_Standardized_{target_iso}")
    model.TARGET_ISO = target_iso

    model.ENABLE_NUCLEAR_GENERATOR = ENABLE_NUCLEAR_GENERATOR
    model.ENABLE_ELECTROLYZER = ENABLE_ELECTROLYZER
    model.ENABLE_BATTERY = ENABLE_BATTERY
    model.ENABLE_H2_STORAGE = ENABLE_H2_STORAGE
    model.LTE_MODE = ENABLE_LOW_TEMP_ELECTROLYZER if ENABLE_ELECTROLYZER else False
    model.CAN_PROVIDE_ANCILLARY_SERVICES = CAN_PROVIDE_ANCILLARY_SERVICES
    model.SIMULATE_AS_DISPATCH_EXECUTION = simulate_dispatch

    logger.info(f"Creating STANDARDIZED model for {target_iso} with features:")
    logger.info(f"  Nuclear Generator: {model.ENABLE_NUCLEAR_GENERATOR}")
    logger.info(
        f"  Electrolyzer: {model.ENABLE_ELECTROLYZER} (LTE Mode: {model.LTE_MODE})"
    )
    logger.info(f"  Battery: {model.ENABLE_BATTERY}")
    logger.info(
        f"  Ancillary Service Capability: {model.CAN_PROVIDE_ANCILLARY_SERVICES}"
    )
    logger.info(
        f"  >> AS Simulation Mode: {'Dispatch Execution' if model.SIMULATE_AS_DISPATCH_EXECUTION else 'Bidding Strategy'}"
    )
    logger.info(f"  H2 Storage: {model.ENABLE_H2_STORAGE}")
    logger.info(
        f"  Nonlinear Turbine: {ENABLE_NONLINEAR_TURBINE_EFF and model.ENABLE_NUCLEAR_GENERATOR}"
    )
    logger.info(
        f"  Degradation: {ENABLE_ELECTROLYZER_DEGRADATION_TRACKING and model.ENABLE_ELECTROLYZER}"
    )
    logger.info(
        f"  Startup/Shutdown: {ENABLE_STARTUP_SHUTDOWN and model.ENABLE_ELECTROLYZER}"
    )
    logger.info(
        f"  H2 Cap Factor: {ENABLE_H2_CAP_FACTOR and model.ENABLE_ELECTROLYZER}"
    )

    logger.info("Loading parameters...")
    try:
        if (
            "df_price_hourly" not in data_inputs
            or data_inputs["df_price_hourly"] is None
        ):
            raise ValueError("Essential data 'df_price_hourly' not found.")
        nT = len(data_inputs["df_price_hourly"])
        if nT == 0:
            raise ValueError("Price data is empty.")
        model.TimePeriods = pyo.RangeSet(1, nT)

        if "df_system" not in data_inputs or data_inputs["df_system"] is None:
            raise ValueError("Essential data 'df_system' not found.")
        global df_system
        df_system = data_inputs["df_system"]

        user_elec_cap_mw_str = get_sys_param(
            "user_specified_electrolyzer_capacity_MW", default=None
        )
        user_batt_power_mw_str = get_sys_param(
            "user_specified_battery_power_MW", default=None
        )
        user_batt_energy_mwh_str = get_sys_param(
            "user_specified_battery_energy_MWh", default=None
        )

        user_elec_cap_mw = None
        if user_elec_cap_mw_str is not None:
            try:
                user_elec_cap_mw = float(user_elec_cap_mw_str)
                if user_elec_cap_mw < 0:
                    logger.warning(
                        "User specified electrolyzer capacity negative. Optimizing."
                    )
                    user_elec_cap_mw = None
            except ValueError:
                logger.warning(
                    f"Invalid user_specified_electrolyzer_capacity_MW. Optimizing."
                )

        user_batt_power_mw = None
        if user_batt_power_mw_str is not None:
            try:
                user_batt_power_mw = float(user_batt_power_mw_str)
                if user_batt_power_mw < 0:
                    logger.warning(
                        "User specified battery power negative. Optimizing.")
                    user_batt_power_mw = None
            except ValueError:
                logger.warning(
                    f"Invalid user_specified_battery_power_MW. Optimizing.")

        user_batt_energy_mwh = None
        if user_batt_energy_mwh_str is not None:
            try:
                user_batt_energy_mwh = float(user_batt_energy_mwh_str)
                if user_batt_energy_mwh < 0:
                    logger.warning(
                        "User specified battery energy negative. Optimizing."
                    )
                    user_batt_energy_mwh = None
            except ValueError:
                logger.warning(
                    f"Invalid user_specified_battery_energy_MWh. Optimizing."
                )

        battery_capacity_fixed = False
        if user_batt_power_mw is not None and user_batt_energy_mwh is not None:
            battery_capacity_fixed = True
        elif user_batt_power_mw is not None or user_batt_energy_mwh is not None:
            logger.warning(
                "Battery power and energy must BOTH be specified to fix. Optimizing."
            )
            user_batt_power_mw = None
            user_batt_energy_mwh = None

        model.delT_minutes = pyo.Param(
            within=pyo.NonNegativeReals,
            initialize=get_sys_param("delT_minutes", 60.0, required=True),
        )
        model.AS_Duration = pyo.Param(
            within=pyo.NonNegativeReals,
            initialize=get_sys_param("AS_Duration", 0.25),
        )
        model.plant_lifetime_years = pyo.Param(
            within=pyo.PositiveIntegers,
            initialize=get_sys_param("plant_lifetime_years", 30),
        )
        model.HOURS_IN_YEAR = pyo.Param(
            within=pyo.PositiveReals, initialize=HOURS_IN_YEAR
        )

        if model.ENABLE_NUCLEAR_GENERATOR:
            model.qSteam_Total = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("qSteam_Total_MWth", required=True),
            )
            model.qSteam_Turbine_min = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "qSteam_Turbine_min_MWth", required=True),
            )
            model.qSteam_Turbine_max = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "qSteam_Turbine_max_MWth", required=True),
            )
            model.pTurbine_min = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("pTurbine_min_MW", required=True),
            )
            p_turb_max_val = get_sys_param("pTurbine_max_MW", required=True)
            model.pTurbine_max = pyo.Param(
                within=pyo.NonNegativeReals, initialize=p_turb_max_val
            )
            ramp_up_pct_min = get_sys_param(
                "Turbine_RampUp_Rate_Percent_per_Min", 1.0)
            ramp_down_pct_min = get_sys_param(
                "Turbine_RampDown_Rate_Percent_per_Min", 1.0
            )
            model.RU_Turbine_hourly = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=ramp_up_pct_min * 60 / 100 * p_turb_max_val,
            )
            model.RD_Turbine_hourly = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=ramp_down_pct_min * 60 / 100 * p_turb_max_val,
            )
            model.vom_turbine = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("vom_turbine_USD_per_MWh", 0),
            )
            model.convertTtE_const = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "Turbine_Thermal_Elec_Efficiency_Const", 0.4),
            )
            model.nonlinear_turbine_enabled_in_model = False
            if ENABLE_NONLINEAR_TURBINE_EFF:
                try:
                    q_bps_str = get_sys_param(
                        "qSteam_Turbine_Breakpoints_MWth", required=True
                    )
                    p_vals_str = get_sys_param(
                        "pTurbine_Outputs_at_Breakpoints_MW", required=True
                    )
                    if not isinstance(q_bps_str, str) or not isinstance(
                        p_vals_str, str
                    ):
                        raise TypeError(
                            "Turbine breakpoint/output data not string.")
                    q_breakpoints = sorted(
                        [float(x.strip())
                         for x in q_bps_str.split(",") if x.strip()]
                    )
                    p_values = [
                        float(x.strip()) for x in p_vals_str.split(",") if x.strip()
                    ]
                    if not q_breakpoints:
                        raise ValueError(
                            "Turbine steam breakpoints list is empty.")
                    if len(q_breakpoints) != len(p_values):
                        raise ValueError(
                            "Turbine steam breakpoints and power output lists have different lengths."
                        )
                    model.qTurbine_efficiency_breakpoints = pyo.Set(
                        initialize=q_breakpoints, ordered=True
                    )
                    pTurbine_vals_at_qTurbine_bp_dict = dict(
                        zip(q_breakpoints, p_values)
                    )
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(
                        model.qTurbine_efficiency_breakpoints,
                        initialize=pTurbine_vals_at_qTurbine_bp_dict,
                    )
                    model.nonlinear_turbine_enabled_in_model = True
                    logger.info(
                        "Enabled non-linear turbine efficiency with provided breakpoints."
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading non-linear turbine piecewise data: {e}. Falling back to linear."
                    )
                    q_turb_min_val_fb = get_sys_param(
                        "qSteam_Turbine_min_MWth", required=True
                    )
                    q_turb_max_val_fb = get_sys_param(
                        "qSteam_Turbine_max_MWth", required=True
                    )
                    conv_const_fb = get_sys_param(
                        "Turbine_Thermal_Elec_Efficiency_Const", 0.4
                    )
                    model.qTurbine_efficiency_breakpoints = pyo.Set(
                        initialize=[q_turb_min_val_fb, q_turb_max_val_fb],
                        ordered=True,
                    )
                    min_p_fallback = q_turb_min_val_fb * conv_const_fb
                    max_p_fallback = q_turb_max_val_fb * conv_const_fb
                    model.pTurbine_values_at_qTurbine_bp = pyo.Param(
                        model.qTurbine_efficiency_breakpoints,
                        initialize={
                            q_turb_min_val_fb: min_p_fallback,
                            q_turb_max_val_fb: max_p_fallback,
                        },
                    )
                    model.nonlinear_turbine_enabled_in_model = (
                        True  # Still uses PWL, just linear
                    )
                    logger.warning(
                        "Fell back to a linear piecewise representation for turbine efficiency."
                    )

            if model.ENABLE_ELECTROLYZER and not model.LTE_MODE:
                model.Ramp_qSteam_Electrolyzer_limit = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour",
                        float("inf"),
                    ),
                )
            if model.ENABLE_ELECTROLYZER and model.LTE_MODE:
                model.pTurbine_LTE_setpoint = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "pTurbine_LTE_setpoint_MW",
                        p_turb_max_val,
                        required=False,
                    ),
                )

        if model.ENABLE_ELECTROLYZER:
            elec_type_suffix = "LTE" if model.LTE_MODE else "HTE"
            logger.info(
                f"Loading parameters for {elec_type_suffix} electrolyzer...")
            model.hydrogen_subsidy_per_kg = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "hydrogen_subsidy_value_usd_per_kg", 0.0),
            )
            model.aux_power_consumption_per_kg_h2 = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "aux_power_consumption_per_kg_h2", 0.0),
            )

            if user_elec_cap_mw is None:  # Optimizing capacity
                model.pElectrolyzer_max_upper_bound = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "pElectrolyzer_max_upper_bound_MW", required=True
                    ),
                )
                model.pElectrolyzer_max_lower_bound = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "pElectrolyzer_max_lower_bound_MW", 0.0),
                )

            p_elec_min_val = get_sys_param(
                f"pElectrolyzer_min_MW_{elec_type_suffix}",
                default=get_sys_param("pElectrolyzer_min_MW", required=True),
            )
            model.pElectrolyzer_min = pyo.Param(
                within=pyo.NonNegativeReals, initialize=p_elec_min_val
            )  # This is the crucial pElectrolyzer_min param

            ramp_up_elec_pct_min = get_sys_param(
                f"Electrolyzer_RampUp_Rate_Percent_per_Min_{elec_type_suffix}",
                default=get_sys_param(
                    "Electrolyzer_RampUp_Rate_Percent_per_Min", 10.0),
            )
            ramp_down_elec_pct_min = get_sys_param(
                f"Electrolyzer_RampDown_Rate_Percent_per_Min_{elec_type_suffix}",
                default=get_sys_param(
                    "Electrolyzer_RampDown_Rate_Percent_per_Min", 10.0
                ),
            )
            model.RU_Electrolyzer_percent_hourly = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=ramp_up_elec_pct_min * 60 / 100,
            )
            model.RD_Electrolyzer_percent_hourly = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=ramp_down_elec_pct_min * 60 / 100,
            )
            model.vom_electrolyzer = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    f"vom_electrolyzer_USD_per_MWh_{elec_type_suffix}",
                    default=get_sys_param("vom_electrolyzer_USD_per_MWh", 0),
                ),
            )
            model.cost_water_per_kg_h2 = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("cost_water_USD_per_kg_h2", 0),
            )
            model.cost_electrolyzer_ramping = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    f"cost_electrolyzer_ramping_USD_per_MW_ramp_{elec_type_suffix}",
                    default=get_sys_param(
                        "cost_electrolyzer_ramping_USD_per_MW_ramp", 0
                    ),
                ),
            )
            model.cost_electrolyzer_capacity = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    f"cost_electrolyzer_capacity_USD_per_MW_year_{elec_type_suffix}",
                    default=get_sys_param(
                        "cost_electrolyzer_capacity_USD_per_MW_year", 0
                    ),
                ),
            )
            try:
                p_elec_bps_str = get_sys_param(
                    f"pElectrolyzer_Breakpoints_MW_{elec_type_suffix}",
                    default=get_sys_param(
                        "pElectrolyzer_Breakpoints_MW", required=True
                    ),
                )
                ke_vals_str = get_sys_param(
                    f"ke_H2_Values_MWh_per_kg_{elec_type_suffix}",
                    default=get_sys_param(
                        "ke_H2_Values_MWh_per_kg", required=True),
                )
                if not isinstance(p_elec_bps_str, str) or not isinstance(
                    ke_vals_str, str
                ):
                    raise TypeError(
                        f"Elec breakpoint/ke data not string for {elec_type_suffix}."
                    )
                p_elec_breakpoints = sorted(
                    [float(x.strip())
                     for x in p_elec_bps_str.split(",") if x.strip()]
                )
                ke_values = [
                    float(x.strip()) for x in ke_vals_str.split(",") if x.strip()
                ]
                if not p_elec_breakpoints:
                    raise ValueError(
                        f"Elec power breakpoints list empty for {elec_type_suffix}."
                    )
                if len(p_elec_breakpoints) != len(ke_values):
                    raise ValueError(
                        f"Elec breakpoints/ke lengths differ for {elec_type_suffix}."
                    )
                model.pElectrolyzer_efficiency_breakpoints = pyo.Set(
                    initialize=p_elec_breakpoints, ordered=True
                )
                ke_vals_dict = dict(zip(p_elec_breakpoints, ke_values))
                model.ke_H2_values = pyo.Param(
                    model.pElectrolyzer_efficiency_breakpoints,
                    initialize=ke_vals_dict,
                    within=pyo.NonNegativeReals,
                )
                if not model.LTE_MODE:
                    kt_vals_str = get_sys_param(
                        f"kt_H2_Values_MWth_per_kg_{elec_type_suffix}",
                        default=get_sys_param(
                            "kt_H2_Values_MWth_per_kg", required=True
                        ),
                    )
                    if not isinstance(kt_vals_str, str):
                        raise TypeError(
                            f"HTE kt_H2 data not string for {elec_type_suffix}."
                        )
                    kt_values = [
                        float(x.strip()) for x in kt_vals_str.split(",") if x.strip()
                    ]
                    if len(p_elec_breakpoints) != len(kt_values):
                        raise ValueError(
                            f"HTE breakpoints/kt lengths differ for {elec_type_suffix}."
                        )
                    kt_vals_dict = dict(zip(p_elec_breakpoints, kt_values))
                    model.kt_H2_values = pyo.Param(
                        model.pElectrolyzer_efficiency_breakpoints,
                        initialize=kt_vals_dict,
                        within=pyo.NonNegativeReals,
                    )
                else:
                    kt_zero_dict = {bp: 0.0 for bp in p_elec_breakpoints}
                    model.kt_H2_values = pyo.Param(
                        model.pElectrolyzer_efficiency_breakpoints,
                        initialize=kt_zero_dict,
                    )
                logger.info(
                    f"Loaded {elec_type_suffix} electrolyzer piecewise parameters (ke, kt)."
                )
            except Exception as e:
                logger.error(
                    f"Error loading {elec_type_suffix} electrolyzer piecewise data: {e}."
                )
                raise ValueError(
                    f"Failed to load essential {elec_type_suffix} electrolyzer efficiency data."
                )

            if ENABLE_STARTUP_SHUTDOWN:
                model.cost_startup_electrolyzer = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        f"cost_startup_electrolyzer_USD_per_startup_{elec_type_suffix}",
                        default=get_sys_param(
                            "cost_startup_electrolyzer_USD_per_startup", 0
                        ),
                    ),
                )
                model.MinUpTimeElectrolyzer = pyo.Param(
                    within=pyo.PositiveIntegers,
                    initialize=get_sys_param(
                        f"MinUpTimeElectrolyzer_hours_{elec_type_suffix}",
                        default=get_sys_param(
                            "MinUpTimeElectrolyzer_hours", 1),
                    ),
                )
                model.MinDownTimeElectrolyzer = pyo.Param(
                    within=pyo.PositiveIntegers,
                    initialize=get_sys_param(
                        f"MinDownTimeElectrolyzer_hours_{elec_type_suffix}",
                        default=get_sys_param(
                            "MinDownTimeElectrolyzer_hours", 1),
                    ),
                )
                init_status_raw = get_sys_param(
                    "uElectrolyzer_initial_status_0_or_1", 0
                )
                init_status = 1 if int(float(init_status_raw)) == 1 else 0
                model.uElectrolyzer_initial = pyo.Param(
                    within=pyo.Binary, initialize=init_status
                )

            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                model.DegradationStateInitial = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "DegradationStateInitial_Units", 0.0),
                )
                model.DegradationFactorOperation = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        f"DegradationFactorOperation_Units_per_Hour_at_MaxLoad_{elec_type_suffix}",
                        default=get_sys_param(
                            "DegradationFactorOperation_Units_per_Hour_at_MaxLoad",
                            0.0,
                        ),
                    ),
                )
                model.DegradationFactorStartup = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        f"DegradationFactorStartup_Units_per_Startup_{elec_type_suffix}",
                        default=get_sys_param(
                            "DegradationFactorStartup_Units_per_Startup", 0.0
                        ),
                    ),
                )

            if ENABLE_H2_CAP_FACTOR:
                model.h2_target_capacity_factor = pyo.Param(
                    within=pyo.PercentFraction,
                    initialize=get_sys_param(
                        "h2_target_capacity_factor_fraction", 0.0),
                )

            model.H2_value = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("H2_value_USD_per_kg", required=True),
            )

            if model.ENABLE_H2_STORAGE:
                h2_storage_max = get_sys_param(
                    "H2_storage_capacity_max_kg", required=True
                )
                h2_storage_min = get_sys_param("H2_storage_capacity_min_kg", 0)
                model.H2_storage_capacity_max = pyo.Param(
                    within=pyo.NonNegativeReals, initialize=h2_storage_max
                )
                model.H2_storage_capacity_min = pyo.Param(
                    within=pyo.NonNegativeReals, initialize=h2_storage_min
                )
                initial_level_raw = get_sys_param(
                    "H2_storage_level_initial_kg", h2_storage_min
                )
                initial_level = max(
                    h2_storage_min,
                    min(h2_storage_max, float(initial_level_raw)),
                )
                model.H2_storage_level_initial = pyo.Param(
                    within=pyo.NonNegativeReals, initialize=initial_level
                )
                model.H2_storage_charge_rate_max = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "H2_storage_charge_rate_max_kg_per_hr", required=True
                    ),
                )
                model.H2_storage_discharge_rate_max = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "H2_storage_discharge_rate_max_kg_per_hr",
                        required=True,
                    ),
                )
                model.storage_charge_eff = pyo.Param(
                    within=pyo.PercentFraction,
                    initialize=get_sys_param(
                        "storage_charge_eff_fraction", 1.0),
                )
                model.storage_discharge_eff = pyo.Param(
                    within=pyo.PercentFraction,
                    initialize=get_sys_param(
                        "storage_discharge_eff_fraction", 1.0),
                )
                model.vom_storage_cycle = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "vom_storage_cycle_USD_per_kg_cycled", 0),
                )

        if model.ENABLE_BATTERY:
            logger.info("Configuring battery storage parameters...")
            if not battery_capacity_fixed:  # Optimizing capacity
                model.BatteryCapacity_max_param = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param(
                        "BatteryCapacity_max_MWh", required=True),
                )
                model.BatteryCapacity_min_param = pyo.Param(
                    within=pyo.NonNegativeReals,
                    initialize=get_sys_param("BatteryCapacity_min_MWh", 0.0),
                )

            model.BatteryPowerRatio = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "BatteryPowerRatio_MW_per_MWh", 0.25, required=True
                ),
            )
            model.BatteryChargeEff = pyo.Param(
                within=pyo.PercentFraction,
                initialize=get_sys_param("BatteryChargeEff", 0.95),
            )
            model.BatteryDischargeEff = pyo.Param(
                within=pyo.PercentFraction,
                initialize=get_sys_param("BatteryDischargeEff", 0.95),
            )
            model.BatterySOC_min_fraction = pyo.Param(
                within=pyo.PercentFraction,
                initialize=get_sys_param("BatterySOC_min_fraction", 0.10),
            )
            model.BatterySOC_initial_fraction = pyo.Param(
                within=pyo.PercentFraction,
                initialize=get_sys_param("BatterySOC_initial_fraction", 0.50),
            )
            batt_cyclic_val_str = get_sys_param(
                "BatteryRequireCyclicSOC", "True")
            batt_cyclic_val = (
                True if str(batt_cyclic_val_str).strip(
                ).lower() == "true" else False
            )
            model.BatteryRequireCyclicSOC = pyo.Param(
                within=pyo.Boolean, initialize=batt_cyclic_val
            )
            model.BatteryRampRate = pyo.Param(
                within=pyo.PercentFraction,
                initialize=get_sys_param(
                    "BatteryRampRate_fraction_per_hour", 1.0),
            )
            model.BatteryCapex_USD_per_MWh_year = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("BatteryCapex_USD_per_MWh_year", 0.0),
            )
            model.BatteryCapex_USD_per_MW_year = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param("BatteryCapex_USD_per_MW_year", 0.0),
            )
            model.BatteryFixedOM_USD_per_MWh_year = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=get_sys_param(
                    "BatteryFixedOM_USD_per_MWh_year", 0.0),
            )
            vom_batt_val = get_sys_param("vom_battery_per_mwh_cycled", None)
            model.vom_battery_per_mwh_cycled = pyo.Param(
                within=pyo.NonNegativeReals,
                initialize=(float(vom_batt_val)
                            if vom_batt_val is not None else 0.0),
            )

        p_ies_min_default_val = (
            -(p_turb_max_val)
            if model.ENABLE_NUCLEAR_GENERATOR and "p_turb_max_val" in locals()
            else -1000.0
        )
        p_ies_max_default_val = (
            p_turb_max_val
            if model.ENABLE_NUCLEAR_GENERATOR and "p_turb_max_val" in locals()
            else 1000.0
        )
        model.pIES_min = pyo.Param(
            within=pyo.Reals,
            initialize=get_sys_param("pIES_min_MW", p_ies_min_default_val),
        )
        model.pIES_max = pyo.Param(
            within=pyo.Reals,
            initialize=get_sys_param("pIES_max_MW", p_ies_max_default_val),
        )

        df_price = data_inputs["df_price_hourly"]
        if len(df_price) < nT:
            raise ValueError(
                f"Energy price data length mismatch ({len(df_price)} vs {nT})."
            )
        energy_price_col = "Price ($/MWh)"
        if energy_price_col not in df_price.columns:
            raise ValueError(f"'{energy_price_col}' not found in price data.")
        energy_price_dict = {
            t_idx: df_price[energy_price_col].iloc[t_idx - 1]
            for t_idx in model.TimePeriods
        }
        model.energy_price = pyo.Param(
            model.TimePeriods, initialize=energy_price_dict, within=pyo.Reals
        )

        df_ANSprice = data_inputs.get("df_ANSprice_hourly", None)
        df_ANSmile = data_inputs.get(
            "df_ANSmile_hourly", None
        )  # For mileage_factor and performance_factor
        df_ANSdeploy = data_inputs.get("df_ANSdeploy_hourly", None)
        df_ANSwinrate = data_inputs.get("df_ANSwinrate_hourly", None)

        def get_hourly_param_from_df_model(
            t_idx, df, col_name, default=0.0, required_param=False
        ):
            """Helper to safely get data from optional hourly dataframes for model params."""
            if df is None:
                if required_param:
                    raise ValueError(
                        f"Required data file for {col_name} not loaded.")
                return default
            filename = getattr(df, "attrs", {}).get("filename", "DataFrame")
            if col_name in df.columns:
                try:
                    if not (0 <= t_idx - 1 < len(df)):
                        raise IndexError(f"Index {t_idx - 1} out of bounds")
                    val = df[col_name].iloc[t_idx - 1]
                    return default if pd.isna(val) else float(val)
                except IndexError:
                    logger.warning(
                        f"Index {t_idx - 1} out of bounds for '{col_name}' in {filename}. Defaulting."
                    )
                    if required_param:
                        raise ValueError(
                            f"Index error accessing required parameter '{col_name}'."
                        )
                    return default
                except Exception as e_param:
                    logger.error(
                        f"Error reading '{col_name}' @ index {t_idx - 1} from {filename}: {e_param}"
                    )
                    if required_param:
                        raise
                    return default
            else:  # Column not found
                # For optional factors, this is acceptable, they will use default.
                # For required prices (p_*), this would be an issue if df_ANSprice was mandatory and column missing.
                # logger.debug(f"Column '{col_name}' not in {filename}. Returning default {default}.")
                if required_param:  # Should not happen if file structure is correct
                    raise ValueError(
                        f"Required column '{col_name}' not in {filename}.")
                return default

        iso_service_map = {  # Standardized map
            "SPP": ["RegU", "RegD", "Spin", "Sup", "RamU", "RamD", "UncU"],
            "CAISO": ["RegU", "RegD", "Spin", "NSpin", "RMU", "RMD"],
            "ERCOT": ["RegU", "RegD", "Spin", "NSpin", "ECRS"],
            "PJM": ["RegUp", "RegDown", "Syn", "Rse", "TMR"],
            "NYISO": ["RegUp", "RegDown", "Spin10", "NSpin10", "Res30"],
            "ISONE": ["Spin10", "NSpin10", "OR30"],
            "MISO": ["RegUp", "RegDown", "Spin", "Sup", "STR", "RamU", "RamD"],
        }

        if target_iso not in iso_service_map:
            raise ValueError(
                f"AS definitions missing for TARGET_ISO: {target_iso}")
        logger.info(
            f"Loading AS parameters for {target_iso} using standardized naming..."
        )
        if df_ANSprice is not None:
            df_ANSprice.attrs = {"filename": "Price_ANS_hourly.csv"}
        if df_ANSmile is not None:
            df_ANSmile.attrs = {"filename": "MileageMultiplier_hourly.csv"}
        if df_ANSdeploy is not None:
            df_ANSdeploy.attrs = {"filename": "DeploymentFactor_hourly.csv"}
        if df_ANSwinrate is not None:
            df_ANSwinrate.attrs = {"filename": "WinningRate_hourly.csv"}

        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            for service_key_in_map in iso_service_map[target_iso]:
                # param_name_base_on_model is the key used by revenue_cost.py (e.g. 'RegUp', 'Spin')
                param_name_base_on_model = service_key_in_map
                # csv_col_base is used to construct column names in CSV files (e.g., 'RegUp_PJM', 'Spin_SPP')
                csv_col_base = f"{service_key_in_map}_{target_iso}"

                # Price (p_*)
                price_col_name_in_csv = f"p_{csv_col_base}"
                param_dict = {
                    t: get_hourly_param_from_df_model(
                        t, df_ANSprice, price_col_name_in_csv, default=0.0
                    )
                    for t in model.TimePeriods
                }
                model.add_component(
                    f"p_{param_name_base_on_model}_{target_iso}",
                    pyo.Param(
                        model.TimePeriods,
                        initialize=param_dict,
                        within=pyo.Reals,
                    ),
                )

                # Adder (loc_*)
                loc_col_name_in_csv = f"loc_{csv_col_base}"
                param_dict = {
                    t: get_hourly_param_from_df_model(
                        t, df_ANSprice, loc_col_name_in_csv, default=0.0
                    )
                    for t in model.TimePeriods
                }
                model.add_component(
                    f"loc_{param_name_base_on_model}_{target_iso}",
                    pyo.Param(
                        model.TimePeriods,
                        initialize=param_dict,
                        within=pyo.Reals,
                    ),
                )

                # Winning Rate (winning_rate_*)
                win_col_name_in_csv = f"winning_rate_{csv_col_base}"
                param_dict = {
                    t: get_hourly_param_from_df_model(
                        t, df_ANSwinrate, win_col_name_in_csv, default=1.0
                    )
                    for t in model.TimePeriods
                }
                model.add_component(
                    f"winning_rate_{param_name_base_on_model}_{target_iso}",
                    pyo.Param(
                        model.TimePeriods,
                        initialize=param_dict,
                        within=pyo.PercentFraction,
                    ),
                )

                is_regulation_service = (
                    "RegU" in service_key_in_map
                    or "RegD" in service_key_in_map
                    or "RegUp" in service_key_in_map
                    or "RegDown" in service_key_in_map
                )
                if is_regulation_service:
                    # Mileage Factor (mileage_factor_*)
                    mileage_col_name_in_csv = f"mileage_factor_{csv_col_base}"
                    param_dict_mileage = {
                        t: get_hourly_param_from_df_model(
                            t, df_ANSmile, mileage_col_name_in_csv, default=1.0
                        )
                        for t in model.TimePeriods
                    }
                    model.add_component(
                        f"mileage_factor_{param_name_base_on_model}_{target_iso}",
                        pyo.Param(
                            model.TimePeriods,
                            initialize=param_dict_mileage,
                            within=pyo.NonNegativeReals,
                        ),
                    )

                    # Performance Factor (performance_factor_*)
                    perf_col_name_in_csv = f"performance_factor_{csv_col_base}"
                    param_dict_perf = {
                        t: get_hourly_param_from_df_model(
                            t, df_ANSmile, perf_col_name_in_csv, default=1.0
                        )
                        for t in model.TimePeriods
                    }  # Assuming perf factors are also in df_ANSmile
                    model.add_component(
                        f"performance_factor_{param_name_base_on_model}_{target_iso}",
                        pyo.Param(
                            model.TimePeriods,
                            initialize=param_dict_perf,
                            within=pyo.NonNegativeReals,
                        ),
                    )
                else:  # Reserve service
                    # Deploy Factor (deploy_factor_*)
                    deploy_col_name_in_csv = f"deploy_factor_{csv_col_base}"
                    param_dict_deploy = {
                        t: get_hourly_param_from_df_model(
                            t,
                            df_ANSdeploy,
                            deploy_col_name_in_csv,
                            default=0.0,
                        )
                        for t in model.TimePeriods
                    }
                    model.add_component(
                        f"deploy_factor_{param_name_base_on_model}_{target_iso}",
                        pyo.Param(
                            model.TimePeriods,
                            initialize=param_dict_deploy,
                            within=pyo.PercentFraction,
                        ),
                    )
        else:
            logger.info(
                "Ancillary services disabled. Skipping AS parameter loading.")

    except Exception as e:
        logger.error(f"Error during parameter loading: {e}", exc_info=True)
        raise

    logger.info("Defining variables...")
    try:
        p_ies_min_val = pyo.value(model.pIES_min)
        p_ies_max_val = pyo.value(model.pIES_max)
        model.pIES = pyo.Var(
            model.TimePeriods,
            within=pyo.Reals,
            bounds=(p_ies_min_val, p_ies_max_val),
        )

        model.pGridPurchase = pyo.Var(
            model.TimePeriods,
            within=pyo.NonNegativeReals,
            initialize=0.0
        )

        model.pGridSale = pyo.Var(
            model.TimePeriods,
            within=pyo.NonNegativeReals,
            initialize=0.0
        )

        def grid_interaction_decomposition_rule(m, t):
            return m.pIES[t] == m.pGridSale[t] - m.pGridPurchase[t]

        model.grid_interaction_decomposition_constr = pyo.Constraint(
            model.TimePeriods, rule=grid_interaction_decomposition_rule
        )

        if model.ENABLE_NUCLEAR_GENERATOR:
            q_turb_min_val = pyo.value(model.qSteam_Turbine_min)
            q_turb_max_val = pyo.value(model.qSteam_Turbine_max)
            p_turb_min_val_loc = pyo.value(model.pTurbine_min)
            p_turb_max_val_loc = pyo.value(model.pTurbine_max)
            model.qSteam_Turbine = pyo.Var(
                model.TimePeriods,
                within=pyo.NonNegativeReals,
                bounds=(q_turb_min_val, q_turb_max_val),
            )
            model.pTurbine = pyo.Var(
                model.TimePeriods,
                within=pyo.NonNegativeReals,
                bounds=(p_turb_min_val_loc, p_turb_max_val_loc),
            )

        if model.ENABLE_ELECTROLYZER:
            # Define pElectrolyzer_max (Var or Param)
            if user_elec_cap_mw is not None:  # Fixed capacity
                if not hasattr(model, "pElectrolyzer_max"):
                    model.pElectrolyzer_max = pyo.Param(
                        within=pyo.NonNegativeReals,
                        initialize=user_elec_cap_mw,
                    )
                    logger.info(
                        f"Using fixed electrolyzer capacity: {user_elec_cap_mw} MW"
                    )
            else:  # Optimizing capacity
                if not hasattr(model, "pElectrolyzer_max_lower_bound") or not hasattr(
                    model, "pElectrolyzer_max_upper_bound"
                ):
                    raise ValueError(
                        "Electrolyzer capacity bounds parameters missing for optimization."
                    )
                if not hasattr(model, "pElectrolyzer_max"):
                    model.pElectrolyzer_max = pyo.Var(
                        within=pyo.NonNegativeReals,
                        bounds=(
                            pyo.value(model.pElectrolyzer_max_lower_bound),
                            pyo.value(model.pElectrolyzer_max_upper_bound),
                        ),
                        initialize=pyo.value(
                            model.pElectrolyzer_max_upper_bound),
                    )  # Initialize to upper bound
                    logger.info(
                        f"Optimizing electrolyzer capacity between {pyo.value(model.pElectrolyzer_max_lower_bound)} and {pyo.value(model.pElectrolyzer_max_upper_bound)} MW"
                    )

            if not hasattr(model, "pElectrolyzer_max"):  # Should be defined by now
                raise AttributeError(
                    "Failed to define model.pElectrolyzer_max as Var or Param."
                )

            # Add constraint: pElectrolyzer_max >= pElectrolyzer_min if pElectrolyzer_max is a variable
            if user_elec_cap_mw is None:  # pElectrolyzer_max is a Var
                if hasattr(model, "pElectrolyzer_max") and hasattr(
                    model, "pElectrolyzer_min"
                ):

                    def enforce_min_capacity_rule(m):
                        return m.pElectrolyzer_max >= m.pElectrolyzer_min

                    model.enforce_min_electrolyzer_capacity_constr = pyo.Constraint(
                        rule=enforce_min_capacity_rule
                    )
                    logger.info(
                        "Added constraint: pElectrolyzer_max >= pElectrolyzer_min for optimized capacity."
                    )
                else:  # Should not happen if logic is correct
                    logger.error(
                        "Could not add pElectrolyzer_max >= pElectrolyzer_min constraint: params/vars missing."
                    )

            model.pElectrolyzer = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals
            )
            model.pElectrolyzerSetpoint = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals
            )
            model.mHydrogenProduced = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals
            )
            if (
                hasattr(model, "aux_power_consumption_per_kg_h2")
                and pyo.value(model.aux_power_consumption_per_kg_h2) > 1e-6
            ):
                model.pAuxiliary = pyo.Var(
                    model.TimePeriods,
                    within=pyo.NonNegativeReals,
                    initialize=0.0,
                )
            if not model.LTE_MODE:  # HTE specific
                model.qSteam_Electrolyzer = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )

            if ENABLE_STARTUP_SHUTDOWN:
                model.uElectrolyzer = pyo.Var(
                    model.TimePeriods, within=pyo.Binary)
                model.vElectrolyzerStartup = pyo.Var(
                    model.TimePeriods, within=pyo.Binary
                )
                model.wElectrolyzerShutdown = pyo.Var(
                    model.TimePeriods, within=pyo.Binary
                )
            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                model.DegradationState = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )
            if (
                hasattr(model, "cost_electrolyzer_ramping")
                and pyo.value(model.cost_electrolyzer_ramping) > 1e-9
            ):
                model.pElectrolyzerRampPos = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )
                model.pElectrolyzerRampNeg = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )
            if (
                not model.LTE_MODE
                and hasattr(model, "Ramp_qSteam_Electrolyzer_limit")
                and pyo.value(model.Ramp_qSteam_Electrolyzer_limit) < float("inf")
            ):
                model.qSteamElectrolyzerRampPos = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )
                model.qSteamElectrolyzerRampNeg = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )

            if model.ENABLE_H2_STORAGE:
                h2_storage_min_val_loc = pyo.value(
                    model.H2_storage_capacity_min)
                h2_storage_max_val_loc = pyo.value(
                    model.H2_storage_capacity_max)
                h2_charge_max_val_loc = pyo.value(
                    model.H2_storage_charge_rate_max)
                h2_discharge_max_val_loc = pyo.value(
                    model.H2_storage_discharge_rate_max
                )
                model.H2_storage_level = pyo.Var(
                    model.TimePeriods,
                    within=pyo.NonNegativeReals,
                    bounds=(h2_storage_min_val_loc, h2_storage_max_val_loc),
                )
                model.H2_to_storage = pyo.Var(
                    model.TimePeriods,
                    within=pyo.NonNegativeReals,
                    bounds=(0, h2_charge_max_val_loc),
                )
                model.H2_from_storage = pyo.Var(
                    model.TimePeriods,
                    within=pyo.NonNegativeReals,
                    bounds=(0, h2_discharge_max_val_loc),
                )
                model.H2_to_market = pyo.Var(
                    model.TimePeriods, within=pyo.NonNegativeReals
                )

        if model.ENABLE_BATTERY:
            if battery_capacity_fixed:
                if not hasattr(model, "BatteryPower_MW"):
                    model.BatteryPower_MW = pyo.Param(
                        within=pyo.NonNegativeReals,
                        initialize=user_batt_power_mw,
                    )
                if not hasattr(model, "BatteryCapacity_MWh"):
                    model.BatteryCapacity_MWh = pyo.Param(
                        within=pyo.NonNegativeReals,
                        initialize=user_batt_energy_mwh,
                    )
                logger.info(
                    f"Using fixed battery power: {user_batt_power_mw} MW and energy: {user_batt_energy_mwh} MWh"
                )
            else:  # Optimizing
                if not hasattr(model, "BatteryCapacity_min_param") or not hasattr(
                    model, "BatteryCapacity_max_param"
                ):
                    raise ValueError(
                        "Battery capacity bounds parameters missing for optimization."
                    )
                if not hasattr(model, "BatteryCapacity_MWh"):
                    model.BatteryCapacity_MWh = pyo.Var(
                        within=pyo.NonNegativeReals,
                        bounds=(
                            pyo.value(model.BatteryCapacity_min_param),
                            pyo.value(model.BatteryCapacity_max_param),
                        ),
                        initialize=(
                            pyo.value(model.BatteryCapacity_min_param)
                            + pyo.value(model.BatteryCapacity_max_param)
                        )
                        / 2,
                    )
                if not hasattr(model, "BatteryPower_MW"):
                    model.BatteryPower_MW = pyo.Var(
                        within=pyo.NonNegativeReals
                    )  # Linked to MWh by ratio constraint
                logger.info(
                    f"Optimizing battery energy capacity (between {pyo.value(model.BatteryCapacity_min_param)}-{pyo.value(model.BatteryCapacity_max_param)} MWh) and power."
                )

            if not hasattr(model, "BatteryPower_MW") or not hasattr(
                model, "BatteryCapacity_MWh"
            ):
                raise AttributeError(
                    "Failed to define battery power/capacity components."
                )

            model.BatterySOC = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals)
            model.BatteryCharge = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals
            )
            model.BatteryDischarge = pyo.Var(
                model.TimePeriods, within=pyo.NonNegativeReals
            )
            model.BatteryBinaryCharge = pyo.Var(
                model.TimePeriods, within=pyo.Binary)
            model.BatteryBinaryDischarge = pyo.Var(
                model.TimePeriods, within=pyo.Binary)

        as_service_list_internal = [
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
        model.component_as_bid_vars = {}
        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            logger.info("Defining Ancillary Service Bid Variables...")
            components_for_as_bids = []
            if model.ENABLE_NUCLEAR_GENERATOR and (
                model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY
            ):
                components_for_as_bids.append("Turbine")
            if model.ENABLE_ELECTROLYZER:
                components_for_as_bids.append("Electrolyzer")
            if model.ENABLE_BATTERY:
                components_for_as_bids.append("Battery")

            for comp_name in components_for_as_bids:
                model.component_as_bid_vars[comp_name] = []
                for s_internal in as_service_list_internal:
                    var_name = f"{s_internal}_{comp_name}"
                    model.add_component(
                        var_name,
                        pyo.Var(
                            model.TimePeriods,
                            within=pyo.NonNegativeReals,
                            initialize=0.0,
                        ),
                    )
                    model.component_as_bid_vars[comp_name].append(var_name)
            for s_internal in as_service_list_internal:  # Total bids
                model.add_component(
                    f"Total_{s_internal}",
                    pyo.Var(
                        model.TimePeriods,
                        within=pyo.NonNegativeReals,
                        initialize=0.0,
                    ),
                )
        else:  # AS disabled, fix Total bids to 0 using Parameters
            logger.info(
                "Ancillary Services disabled by configuration. Fixing AS Total bids to 0."
            )
            for s_internal in as_service_list_internal:
                model.add_component(
                    f"Total_{s_internal}",
                    pyo.Param(
                        model.TimePeriods,
                        initialize=0.0,
                        within=pyo.NonNegativeReals,
                    ),
                )

        model.component_as_deployed_vars = {}
        if (
            model.SIMULATE_AS_DISPATCH_EXECUTION
            and model.CAN_PROVIDE_ANCILLARY_SERVICES
        ):
            logger.info("Defining Ancillary Service Deployed Variables...")
            components_providing_as_deployment = (
                []
            )  # Redefine for clarity in this scope
            if model.ENABLE_ELECTROLYZER:
                components_providing_as_deployment.append("Electrolyzer")
            if model.ENABLE_BATTERY:
                components_providing_as_deployment.append("Battery")
            if model.ENABLE_NUCLEAR_GENERATOR and (
                model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY
            ):
                components_providing_as_deployment.append("Turbine")

            for comp_name in components_providing_as_deployment:
                model.component_as_deployed_vars[comp_name] = []
                for s_internal in as_service_list_internal:
                    # Check if corresponding bid var exists
                    bid_var_name = f"{s_internal}_{comp_name}"
                    if hasattr(model, bid_var_name) and isinstance(
                        getattr(model, bid_var_name), pyo.Var
                    ):
                        deployed_var_name = f"{s_internal}_{comp_name}_Deployed"
                        if not hasattr(model, deployed_var_name):  # Add if not exists
                            model.add_component(
                                deployed_var_name,
                                pyo.Var(
                                    model.TimePeriods,
                                    within=pyo.NonNegativeReals,
                                    initialize=0.0,
                                ),
                            )
                        model.component_as_deployed_vars[comp_name].append(
                            deployed_var_name
                        )
    except Exception as e:
        logger.error(f"Error during variable definition: {e}", exc_info=True)
        raise

    # Precompute inverse efficiencies etc.
    try:
        if model.ENABLE_ELECTROLYZER:
            if hasattr(model, "ke_H2_values") and hasattr(
                model, "pElectrolyzer_efficiency_breakpoints"
            ):
                if not list(model.pElectrolyzer_efficiency_breakpoints):
                    raise ValueError(
                        "pElectrolyzer_efficiency_breakpoints is empty for ke_H2_inv precomputation."
                    )
                model.ke_H2_inv_values = {
                    bp: (
                        1.0 / model.ke_H2_values[bp]
                        if abs(pyo.value(model.ke_H2_values[bp])) > 1e-9
                        else 1e9
                    )
                    for bp in model.pElectrolyzer_efficiency_breakpoints
                }
            else:
                raise ValueError(
                    "Missing ke_H2_values or pElectrolyzer_efficiency_breakpoints for precomputation."
                )

            if (
                not model.LTE_MODE
            ):  # HTE specific precomputation for steam consumption PWL
                if (
                    hasattr(model, "kt_H2_values")
                    and hasattr(model, "ke_H2_inv_values")
                    and hasattr(model, "pElectrolyzer_efficiency_breakpoints")
                ):
                    if not list(model.pElectrolyzer_efficiency_breakpoints):
                        raise ValueError(
                            "pElectrolyzer_efficiency_breakpoints is empty for qSteam_values precomputation."
                        )
                    q_steam_at_pElec_bp_dict = {
                        p_bp: (
                            p_bp
                            * pyo.value(model.kt_H2_values[p_bp])
                            * pyo.value(model.ke_H2_inv_values[p_bp])
                        )
                        for p_bp in model.pElectrolyzer_efficiency_breakpoints
                    }
                    model.qSteam_values_at_pElec_bp = pyo.Param(
                        model.pElectrolyzer_efficiency_breakpoints,
                        initialize=q_steam_at_pElec_bp_dict,
                    )
                    logger.info(
                        "Calculated qSteam values at pElectrolyzer breakpoints for HTE."
                    )
                else:  # Should not happen if HTE is correctly configured
                    logger.warning(
                        "kt_H2_values, ke_H2_inv_values, or breakpoints missing. Cannot calculate qSteam_values_at_pElec_bp for HTE."
                    )
                    if hasattr(model, "pElectrolyzer_efficiency_breakpoints") and list(
                        model.pElectrolyzer_efficiency_breakpoints
                    ):
                        model.qSteam_values_at_pElec_bp = pyo.Param(
                            model.pElectrolyzer_efficiency_breakpoints,
                            initialize=0.0,
                            default=0.0,
                        )  # Fallback
    except Exception as e:
        logger.error(f"Error during precomputation: {e}", exc_info=True)
        raise

    logger.info("Defining constraints...")
    try:
        # Ensure essential components exist before adding constraints that use them
        if model.ENABLE_ELECTROLYZER and not hasattr(model, "pElectrolyzer_max"):
            raise AttributeError(
                "model.pElectrolyzer_max was not defined before constraints section."
            )
        if model.ENABLE_BATTERY and (
            not hasattr(model, "BatteryPower_MW")
            or not hasattr(model, "BatteryCapacity_MWh")
        ):
            raise AttributeError(
                "Battery power/capacity components were not defined before constraints section."
            )

        # Physical System Constraints
        model.power_balance_constr = pyo.Constraint(
            model.TimePeriods, rule=power_balance_rule
        )

        model.restrict_grid_purchase_constr = pyo.Constraint(
            model.TimePeriods, rule=restrict_grid_purchase_rule
        )

        if hasattr(model, "pAuxiliary"):  # Only add if pAuxiliary var exists
            model.link_auxiliary_power_constr = pyo.Constraint(
                model.TimePeriods, rule=link_auxiliary_power_rule
            )

        if model.ENABLE_NUCLEAR_GENERATOR:
            model.steam_balance_constr = pyo.Constraint(
                model.TimePeriods, rule=steam_balance_rule
            )
            if model.nonlinear_turbine_enabled_in_model and hasattr(
                model, "pTurbine_values_at_qTurbine_bp"
            ):
                build_piecewise_constraints(
                    model,
                    component_prefix="TurbinePower",
                    input_var_name="qSteam_Turbine",
                    output_var_name="pTurbine",
                    breakpoint_set_name="qTurbine_efficiency_breakpoints",
                    value_param_name="pTurbine_values_at_qTurbine_bp",
                )
            elif ENABLE_NONLINEAR_TURBINE_EFF:  # Intent was there, but params failed
                logger.warning(
                    "Nonlinear turbine enabled by config but PWL params missing/invalid. Using linear fallback constraint for turbine."
                )

                def linear_pTurbine_rule(m, t):
                    return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const

                model.linear_pTurbine_constr = pyo.Constraint(
                    model.TimePeriods, rule=linear_pTurbine_rule
                )
            else:  # Linear efficiency by explicit choice

                def linear_pTurbine_rule(m, t):
                    return m.pTurbine[t] == m.qSteam_Turbine[t] * m.convertTtE_const

                model.linear_pTurbine_constr = pyo.Constraint(
                    model.TimePeriods, rule=linear_pTurbine_rule
                )

            model.Turbine_RampUp_constr = pyo.Constraint(
                model.TimePeriods, rule=Turbine_RampUp_rule
            )
            model.Turbine_RampDown_constr = pyo.Constraint(
                model.TimePeriods, rule=Turbine_RampDown_rule
            )
            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.turbine_as_zero_constr = pyo.Constraint(
                    model.TimePeriods, rule=Turbine_AS_Zero_rule
                )
                model.Turbine_AS_Pmax_constr = pyo.Constraint(
                    model.TimePeriods, rule=Turbine_AS_Pmax_rule
                )
                model.Turbine_AS_Pmin_constr = pyo.Constraint(
                    model.TimePeriods, rule=Turbine_AS_Pmin_rule
                )
                model.Turbine_AS_RU_constr = pyo.Constraint(
                    model.TimePeriods, rule=Turbine_AS_RU_rule
                )
                model.Turbine_AS_RD_constr = pyo.Constraint(
                    model.TimePeriods, rule=Turbine_AS_RD_rule
                )
            if (
                model.ENABLE_ELECTROLYZER and model.LTE_MODE
            ):  # LTE mode fixes turbine output
                model.const_turbine_power_constr = pyo.Constraint(
                    model.TimePeriods, rule=constant_turbine_power_rule
                )

        # Electrolyzer Constraints
        if model.ENABLE_ELECTROLYZER:

            def electrolyzer_setpoint_capacity_limit_rule(m, t):
                return m.pElectrolyzerSetpoint[t] <= m.pElectrolyzer_max

            model.electrolyzer_setpoint_capacity_limit_constr = pyo.Constraint(
                model.TimePeriods,
                rule=electrolyzer_setpoint_capacity_limit_rule,
            )
            model.electrolyzer_setpoint_min_power_constr = pyo.Constraint(
                model.TimePeriods, rule=electrolyzer_setpoint_min_power_rule
            )
            model.mH2_rate_at_pElec_bp = {
                bp: bp * value for bp, value in model.ke_H2_inv_values.items()
            }

            if hasattr(
                model, "ke_H2_inv_values"
            ):  # This is a dict precomputed on the model
                build_piecewise_constraints(
                    model,
                    component_prefix="HydrogenProduction",
                    input_var_name="pElectrolyzer",  # Actual power produces H2
                    output_var_name="mHydrogenProduced",
                    breakpoint_set_name="pElectrolyzer_efficiency_breakpoints",
                    value_param_name="mH2_rate_at_pElec_bp",
                )  # Pass the dict
            else:
                logger.error(
                    "Cannot build HydrogenProduction piecewise: ke_H2_inv_values dict missing."
                )

            if not model.LTE_MODE and hasattr(
                model, "qSteam_values_at_pElec_bp"
            ):  # HTE steam consumption
                build_piecewise_constraints(
                    model,
                    component_prefix="SteamConsumption",
                    input_var_name="pElectrolyzer",  # Actual power consumes steam
                    output_var_name="qSteam_Electrolyzer",
                    breakpoint_set_name="pElectrolyzer_efficiency_breakpoints",
                    value_param_name="qSteam_values_at_pElec_bp",
                )  # Pass the Param
            elif not model.LTE_MODE:  # HTE but params missing
                logger.warning(
                    "Cannot build SteamConsumption piecewise for HTE: qSteam_values_at_pElec_bp Param missing."
                )

            model.Electrolyzer_RampUp_constr = pyo.Constraint(
                model.TimePeriods, rule=Electrolyzer_RampUp_rule
            )
            model.Electrolyzer_RampDown_constr = pyo.Constraint(
                model.TimePeriods, rule=Electrolyzer_RampDown_rule
            )

            if not model.LTE_MODE and hasattr(
                model, "qSteamElectrolyzerRampPos"
            ):  # HTE Steam Ramping

                def qSteam_ramp_linearization_rule(m, t):
                    if t == m.TimePeriods.first():
                        return pyo.Constraint.Skip
                    if not hasattr(m, "qSteam_Electrolyzer"):
                        return pyo.Constraint.Skip  # Should exist for HTE
                    return (
                        m.qSteam_Electrolyzer[t] - m.qSteam_Electrolyzer[t - 1]
                        == m.qSteamElectrolyzerRampPos[t]
                        - m.qSteamElectrolyzerRampNeg[t]
                    )

                model.qSteam_ramp_linearization_constr = pyo.Constraint(
                    model.TimePeriods, rule=qSteam_ramp_linearization_rule
                )
                model.Steam_Electrolyzer_Ramp_constr = pyo.Constraint(
                    model.TimePeriods, rule=Steam_Electrolyzer_Ramp_rule
                )

            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.Electrolyzer_AS_Pmax_constr = pyo.Constraint(
                    model.TimePeriods, rule=Electrolyzer_AS_Pmax_rule
                )
                model.Electrolyzer_AS_Pmin_constr = pyo.Constraint(
                    model.TimePeriods, rule=Electrolyzer_AS_Pmin_rule
                )
                model.Electrolyzer_AS_RU_constr = pyo.Constraint(
                    model.TimePeriods, rule=Electrolyzer_AS_RU_rule
                )
                model.Electrolyzer_AS_RD_constr = pyo.Constraint(
                    model.TimePeriods, rule=Electrolyzer_AS_RD_rule
                )

            if ENABLE_STARTUP_SHUTDOWN:
                model.electrolyzer_on_off_logic_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_on_off_logic_rule
                )
                model.electrolyzer_min_power_when_on_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_min_power_when_on_rule
                )
                model.electrolyzer_max_power_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_max_power_rule
                )
                model.electrolyzer_startup_shutdown_exclusivity_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=electrolyzer_startup_shutdown_exclusivity_rule,
                )
                model.electrolyzer_min_uptime_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_min_uptime_rule
                )
                model.electrolyzer_min_downtime_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_min_downtime_rule
                )
            else:  # SDS Disabled
                model.electrolyzer_min_power_sds_disabled_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=electrolyzer_min_power_sds_disabled_rule,
                )

                def electrolyzer_max_power_sds_disabled_rule(m, t):
                    return m.pElectrolyzer[t] <= m.pElectrolyzer_max

                model.electrolyzer_max_power_sds_disabled_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=electrolyzer_max_power_sds_disabled_rule,
                )

            if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING:
                model.electrolyzer_degradation_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_degradation_rule
                )
            if ENABLE_H2_CAP_FACTOR:
                model.h2_prod_req_constr = pyo.Constraint(
                    rule=h2_CapacityFactor_rule)

            if hasattr(
                model, "pElectrolyzerRampPos"
            ):  # This var only exists if cost > 0

                def electrolyzer_ramp_linearization_rule(m, t):
                    if t == m.TimePeriods.first():
                        return pyo.Constraint.Skip
                    return (
                        m.pElectrolyzer[t] - m.pElectrolyzer[t - 1]
                        == m.pElectrolyzerRampPos[t] - m.pElectrolyzerRampNeg[t]
                    )

                model.electrolyzer_ramp_linearization_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=electrolyzer_ramp_linearization_rule,
                )

            if model.ENABLE_H2_STORAGE:
                model.h2_storage_balance_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_storage_balance_adj_rule
                )
                model.h2_prod_dispatch_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_prod_dispatch_rule
                )
                model.h2_storage_charge_limit_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_storage_charge_limit_rule
                )
                model.h2_storage_discharge_limit_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_storage_discharge_limit_rule
                )
                model.h2_storage_level_max_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_storage_level_max_rule
                )
                model.h2_storage_level_min_constr = pyo.Constraint(
                    model.TimePeriods, rule=h2_storage_level_min_rule
                )

            # Conditional linking of pElectrolyzer, pElectrolyzerSetpoint, and Deployed AS
            if model.SIMULATE_AS_DISPATCH_EXECUTION:
                model.define_actual_electrolyzer_power_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=define_actual_electrolyzer_power_rule,
                )
            else:  # Bidding strategy mode
                model.link_setpoint_to_actual_power_constr = pyo.Constraint(
                    model.TimePeriods,
                    rule=link_setpoint_to_actual_power_if_not_simulating_dispatch_rule,
                )

        # Battery Constraints
        if model.ENABLE_BATTERY:
            model.battery_soc_balance_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_soc_balance_rule
            )
            model.battery_charge_limit_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_charge_limit_rule
            )
            model.battery_discharge_limit_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_discharge_limit_rule
            )
            model.battery_binary_exclusivity_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_binary_exclusivity_rule
            )
            model.battery_soc_max_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_soc_max_rule
            )
            model.battery_soc_min_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_soc_min_rule
            )
            model.battery_charge_ramp_up_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_ramp_up_rule
            )
            model.battery_charge_ramp_down_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_ramp_down_rule
            )
            model.battery_discharge_ramp_up_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_discharge_ramp_up_rule
            )
            model.battery_discharge_ramp_down_constr = pyo.Constraint(
                model.TimePeriods, rule=battery_discharge_ramp_down_rule
            )
            if hasattr(model, "BatteryRequireCyclicSOC") and pyo.value(
                model.BatteryRequireCyclicSOC
            ):
                model.battery_cyclic_soc_lower_constr = pyo.Constraint(
                    rule=battery_cyclic_soc_lower_rule
                )
                model.battery_cyclic_soc_upper_constr = pyo.Constraint(
                    rule=battery_cyclic_soc_upper_rule
                )

            if not battery_capacity_fixed:  # Only add if optimizing capacity
                model.battery_power_capacity_link_constr = pyo.Constraint(
                    rule=battery_power_capacity_link_rule
                )
                model.battery_min_cap_constr = pyo.Constraint(
                    rule=battery_min_cap_rule)

            if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                model.Battery_AS_Pmax_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_Pmax_rule
                )
                model.Battery_AS_Pmin_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_Pmin_rule
                )
                model.Battery_AS_SOC_Up_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_SOC_Up_rule
                )
                model.Battery_AS_SOC_Down_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_SOC_Down_rule
                )
                model.Battery_AS_RU_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_RU_rule
                )
                model.Battery_AS_RD_constr = pyo.Constraint(
                    model.TimePeriods, rule=Battery_AS_RD_rule
                )

        # Ancillary Service Linking Constraints (BIDS)
        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            # Check if Total_Service is a Var (meaning AS bids are being optimized)
            if hasattr(model, "Total_RegUp") and isinstance(model.Total_RegUp, pyo.Var):
                model.link_Total_RegUp_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_RegUp_rule
                )
            if hasattr(model, "Total_RegDown") and isinstance(
                model.Total_RegDown, pyo.Var
            ):
                model.link_Total_RegDown_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_RegDown_rule
                )
            if hasattr(model, "Total_SR") and isinstance(model.Total_SR, pyo.Var):
                model.link_Total_SR_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_SR_rule
                )
            if hasattr(model, "Total_NSR") and isinstance(model.Total_NSR, pyo.Var):
                model.link_Total_NSR_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_NSR_rule
                )
            if hasattr(model, "Total_ECRS") and isinstance(model.Total_ECRS, pyo.Var):
                model.link_Total_ECRS_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_ECRS_rule
                )
            if hasattr(model, "Total_ThirtyMin") and isinstance(
                model.Total_ThirtyMin, pyo.Var
            ):
                model.link_Total_30Min_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_30Min_rule
                )
            if hasattr(model, "Total_RampUp") and isinstance(
                model.Total_RampUp, pyo.Var
            ):
                model.link_Total_RampUp_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_RampUp_rule
                )
            if hasattr(model, "Total_RampDown") and isinstance(
                model.Total_RampDown, pyo.Var
            ):
                model.link_Total_RampDown_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_RampDown_rule
                )
            if hasattr(model, "Total_UncU") and isinstance(model.Total_UncU, pyo.Var):
                model.link_Total_UncU_constr = pyo.Constraint(
                    model.TimePeriods, rule=link_Total_UncU_rule
                )

            # Regulation Balance Constraints (RegUp == RegDown for each component)
            if model.ENABLE_BATTERY:
                model.battery_regulation_balance_constr = pyo.Constraint(
                    model.TimePeriods, rule=battery_regulation_balance_rule
                )
            if model.ENABLE_ELECTROLYZER:
                model.electrolyzer_regulation_balance_constr = pyo.Constraint(
                    model.TimePeriods, rule=electrolyzer_regulation_balance_rule
                )
            if model.ENABLE_NUCLEAR_GENERATOR and (model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY):
                model.turbine_regulation_balance_constr = pyo.Constraint(
                    model.TimePeriods, rule=turbine_regulation_balance_rule
                )

        # Link Deployed AS to Bids (Dynamically) - If in Dispatch Simulation Mode
        if (
            model.SIMULATE_AS_DISPATCH_EXECUTION
            and model.CAN_PROVIDE_ANCILLARY_SERVICES
        ):
            logger.info(
                "Adding generic link_deployed_to_bid_rule constraints...")
            components_with_deployed_vars = (
                []
            )  # Determine which components have deployed vars
            if model.ENABLE_ELECTROLYZER and model.component_as_deployed_vars.get(
                "Electrolyzer"
            ):
                components_with_deployed_vars.append("Electrolyzer")
            if model.ENABLE_BATTERY and model.component_as_deployed_vars.get("Battery"):
                components_with_deployed_vars.append("Battery")
            if (
                model.ENABLE_NUCLEAR_GENERATOR
                and (model.ENABLE_ELECTROLYZER or model.ENABLE_BATTERY)
                and model.component_as_deployed_vars.get("Turbine")
            ):
                components_with_deployed_vars.append("Turbine")

            for comp_name_iter in components_with_deployed_vars:
                for service_internal_iter in as_service_list_internal:
                    deployed_var_name_check = (
                        f"{service_internal_iter}_{comp_name_iter}_Deployed"
                    )
                    if hasattr(
                        model, deployed_var_name_check
                    ):  # Check if this specific deployed var exists

                        def _rule_factory_deployed_link(
                            s_name, c_name
                        ):  # Closure to capture s_name, c_name
                            def _actual_rule(m_inner, t_inner):
                                return link_deployed_to_bid_rule(
                                    m_inner, t_inner, s_name, c_name
                                )

                            return _actual_rule

                        constr_name_deployed = f"link_deployed_{service_internal_iter}_{comp_name_iter}_constr"
                        if not hasattr(
                            model, constr_name_deployed
                        ):  # Add if not exists
                            model.add_component(
                                constr_name_deployed,
                                pyo.Constraint(
                                    model.TimePeriods,
                                    rule=_rule_factory_deployed_link(
                                        service_internal_iter, comp_name_iter
                                    ),
                                ),
                            )
    except Exception as e:
        logger.error(f"Error during constraint definition: {e}", exc_info=True)
        raise

    logger.info("Defining objective function (Maximize Profit)...")
    try:
        model.EnergyRevenueExpr = pyo.Expression(rule=EnergyRevenue_rule)
        model.HydrogenRevenueExpr = pyo.Expression(rule=HydrogenRevenue_rule)

        if model.CAN_PROVIDE_ANCILLARY_SERVICES:
            iso_revenue_rule_map = {
                "CAISO": AncillaryRevenue_CAISO_rule,
                "ERCOT": AncillaryRevenue_ERCOT_rule,
                "ISONE": AncillaryRevenue_ISONE_rule,
                "MISO": AncillaryRevenue_MISO_rule,
                "NYISO": AncillaryRevenue_NYISO_rule,
                "PJM": AncillaryRevenue_PJM_rule,
                "SPP": AncillaryRevenue_SPP_rule,
            }
            if target_iso in iso_revenue_rule_map:
                model.AncillaryRevenueExpr = pyo.Expression(
                    rule=iso_revenue_rule_map[target_iso]
                )
            else:
                logger.warning(
                    f"No AS Revenue rule defined for TARGET_ISO='{target_iso}'. Setting AS Revenue to 0."
                )
                model.AncillaryRevenueExpr = pyo.Expression(initialize=0.0)
        else:
            model.AncillaryRevenueExpr = pyo.Expression(initialize=0.0)

        model.OpexCostExpr = pyo.Expression(rule=OpexCost_rule)

        def AnnualizedCapex_rule(m):
            total_annual_capex_expr = 0.0
            try:
                delT_min_val = pyo.value(m.delT_minutes)
                total_hours_sim = len(m.TimePeriods) * (delT_min_val / 60.0)
                hours_in_year_val = pyo.value(
                    m.HOURS_IN_YEAR
                )  # Use value from model param
                scaling_factor_for_period = (
                    total_hours_sim / hours_in_year_val if hours_in_year_val > 0 else 0
                )
            except Exception as e:
                logger.error(
                    f"Error getting fixed values in AnnualizedCapex_rule: {e}")
                return 0.0  # Should not happen if params are loaded

            if (
                m.ENABLE_ELECTROLYZER
                and hasattr(m, "cost_electrolyzer_capacity")
                and hasattr(m, "pElectrolyzer_max")
            ):
                cost_elec_cap_param_per_year = m.cost_electrolyzer_capacity
                total_annual_capex_expr += (
                    m.pElectrolyzer_max
                    * cost_elec_cap_param_per_year
                    * scaling_factor_for_period
                )

            if (
                m.ENABLE_BATTERY
                and hasattr(m, "BatteryCapacity_MWh")
                and hasattr(m, "BatteryPower_MW")
            ):
                batt_cap_cost_param_per_year = m.BatteryCapex_USD_per_MWh_year
                batt_pow_cost_param_per_year = m.BatteryCapex_USD_per_MW_year
                batt_fom_cost_param_per_year = m.BatteryFixedOM_USD_per_MWh_year
                battery_annual_cost = (
                    m.BatteryCapacity_MWh * batt_cap_cost_param_per_year
                    + m.BatteryPower_MW * batt_pow_cost_param_per_year
                    + m.BatteryCapacity_MWh * batt_fom_cost_param_per_year
                )
                total_annual_capex_expr += (
                    battery_annual_cost * scaling_factor_for_period
                )
            return total_annual_capex_expr

        model.AnnualizedCapexExpr = pyo.Expression(rule=AnnualizedCapex_rule)

        def TotalProfit_Objective_rule(m):
            try:
                energy_rev = getattr(m, "EnergyRevenueExpr", 0.0)
                as_rev = getattr(m, "AncillaryRevenueExpr", 0.0)
                h2_rev = getattr(m, "HydrogenRevenueExpr", 0.0)
                opex_cost = getattr(m, "OpexCostExpr", 0.0)
                capex_cost_for_period = getattr(
                    m, "AnnualizedCapexExpr", 0.0
                )  # Cost for the simulation period
                total_revenue = energy_rev + as_rev + h2_rev
                total_cost = opex_cost + capex_cost_for_period
                return total_revenue - total_cost
            except Exception as e:
                logger.critical(
                    f"Error defining TotalProfit_Objective_rule expression: {e}",
                    exc_info=True,
                )
                raise

        model.TotalProfit_Objective = pyo.Objective(
            rule=TotalProfit_Objective_rule, sense=pyo.maximize
        )

    except Exception as e:
        logger.error(f"Error during objective definition: {e}", exc_info=True)
        raise

    logger.info("Standardized model created successfully.")
    return model
