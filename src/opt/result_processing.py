# src/result_processing.py
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from config import (
    ENABLE_BATTERY,
    ENABLE_ELECTROLYZER,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_H2_STORAGE,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_STARTUP_SHUTDOWN,
)
from src.logger_utils.logging_setup import logger
from utils import get_param, get_total_deployed_as, get_var_value


def _get_regulation_revenue_component_results(
    m: pyo.ConcreteModel,
    t: int,
    lmp: float,
    iso_service_name: str,
    internal_service_name: str,
    simulate_dispatch: bool,
) -> float:
    """
    Calculates revenue for a single regulation service component (Up or Down) post-
        solve.
    Uses numerical values from the solved model.

    UPDATED: Both regulation and reserve services now use actual deploy_factor values
    for deployed amount calculations in dispatch simulation mode.
    """
    revenue = 0.0
    bid_var_component = getattr(m, f"Total_{internal_service_name}", None)
    bid_value = get_var_value(bid_var_component, t, default=0.0)

    # In bidding strategy mode, if bid is zero, revenue from this component is zero.
    # In dispatch simulation, even if bid was zero (e.g. fixed), deployed could be non-zero
    # if forced by other constraints, but typically awarded capacity (bid*win_rate) would be zero,
    # leading to zero capacity payment.
    # Energy payment in dispatch mode is based on actual deployment.
    if bid_value < 1e-6 and not simulate_dispatch:
        return 0.0

    mcp_cap_val = get_param(m, f"p_{iso_service_name}", t, default=0.0)
    adder_val = get_param(m, f"loc_{iso_service_name}", t, default=0.0)
    win_rate_val = get_param(
        m, f"winning_rate_{iso_service_name}", t, default=1.0)

    cap_payment = bid_value * win_rate_val * mcp_cap_val
    energy_perf_payment = 0.0

    if simulate_dispatch:
        # UPDATED: In dispatch mode, use actual deployed amounts
        # These now include deploy_factor effects for both regulation and reserve services
        deployed_amount_val = get_total_deployed_as(
            m, t, internal_service_name)
        energy_perf_payment = deployed_amount_val * lmp
    else:
        # UPDATED: In bidding strategy mode, regulation services now also use deploy_factor
        # Previously regulation services used deploy_factor = 1.0 implicitly
        mileage_val = get_param(
            m, f"mileage_factor_{iso_service_name}", t, default=1.0)
        perf_val = get_param(
            m, f"performance_factor_{iso_service_name}", t, default=1.0
        )
        # Note: In bidding mode, deploy_factor is used for estimating energy payments
        deploy_factor_val = get_param(
            m, f"deploy_factor_{iso_service_name}", t, default=1.0)
        energy_perf_payment = bid_value * win_rate_val * \
            mileage_val * perf_val * deploy_factor_val * lmp

    revenue = cap_payment + energy_perf_payment + adder_val
    return revenue


def calculate_hourly_as_revenue(m: pyo.ConcreteModel, t: int) -> float:
    """
    Calculates hourly AS revenue rate ($/hr) using ISO-specific logic
        and numerical results.
    Checks m.SIMULATE_AS_DISPATCH_EXECUTION to determine calculation method
    for energy/performance payments.
    """
    if not getattr(m, "CAN_PROVIDE_ANCILLARY_SERVICES", False):
        return 0.0

    iso_suffix = getattr(m, "TARGET_ISO", "UNKNOWN")
    total_hourly_as_revenue_rate = 0.0
    simulate_dispatch = getattr(m, "SIMULATE_AS_DISPATCH_EXECUTION", False)

    try:
        lmp_val = get_param(m, "energy_price", t, default=0.0)

        # Standardized Regulation Revenue Calculation for all ISOs
        # PJM, NYISO, MISO now use 'RegUp'/'RegDown' as iso_service_name for params
        if iso_suffix in ["PJM", "NYISO", "MISO"]:
            total_hourly_as_revenue_rate += _get_regulation_revenue_component_results(
                m, t, lmp_val, "RegUp", "RegUp", simulate_dispatch
            )
            total_hourly_as_revenue_rate += _get_regulation_revenue_component_results(
                m, t, lmp_val, "RegDown", "RegDown", simulate_dispatch
            )
        # SPP, CAISO, ERCOT use 'RegU'/'RegD' as iso_service_name for params
        elif iso_suffix in ["SPP", "CAISO", "ERCOT"]:
            total_hourly_as_revenue_rate += _get_regulation_revenue_component_results(
                m,
                t,
                lmp_val,
                "RegU",
                "RegUp",
                simulate_dispatch,  # internal_service_name is always RegUp/RegDown
            )
            total_hourly_as_revenue_rate += _get_regulation_revenue_component_results(
                m, t, lmp_val, "RegD", "RegDown", simulate_dispatch
            )

        # Reserve Revenue Calculation (uses deploy_factor or deployed amount)
        reserve_map = {}
        if iso_suffix == "SPP":
            reserve_map = {
                "Spin": "SR",
                "Sup": "NSR",
                "RamU": "RampUp",
                "RamD": "RampDown",
                "UncU": "UncU",
            }
        elif iso_suffix == "CAISO":
            reserve_map = {
                "Spin": "SR",
                "NSpin": "NSR",
                "RMU": "RampUp",
                "RMD": "RampDown",
            }
        elif iso_suffix == "ERCOT":
            reserve_map = {"Spin": "SR", "NSpin": "NSR", "ECRS": "ECRS"}
        elif iso_suffix == "PJM":
            reserve_map = {"Syn": "SR", "Rse": "NSR", "TMR": "ThirtyMin"}
        elif iso_suffix == "NYISO":
            reserve_map = {
                "Spin10": "SR",
                "NSpin10": "NSR",
                "Res30": "ThirtyMin",
            }
        elif iso_suffix == "ISONE":  # ISONE only has reserves in this map
            reserve_map = {
                "Spin10": "SR",
                "NSpin10": "NSR",
                "OR30": "ThirtyMin",
            }
        elif iso_suffix == "MISO":
            reserve_map = {
                "Spin": "SR",
                "Sup": "NSR",
                "STR": "ThirtyMin",
                "RamU": "RampUp",
                "RamD": "RampDown",
            }

        for service_iso, internal_service in reserve_map.items():
            bid_var_comp = getattr(m, f"Total_{internal_service}", None)
            bid_val = get_var_value(bid_var_comp, t, default=0.0)
            if bid_val < 1e-6 and not simulate_dispatch:
                continue

            mcp_val = get_param(m, f"p_{service_iso}", t, default=0.0)
            adder_val = get_param(m, f"loc_{service_iso}", t, default=0.0)
            win_rate_val = get_param(
                m, f"winning_rate_{service_iso}", t, default=1.0)

            cap_payment = bid_val * win_rate_val * mcp_val
            energy_payment = 0.0
            if simulate_dispatch:
                deployed_amount_val = get_total_deployed_as(
                    m, t, internal_service)
                energy_payment = deployed_amount_val * lmp_val
            else:
                deploy_factor_val = get_param(
                    m, f"deploy_factor_{service_iso}", t, default=0.0
                )
                energy_payment = bid_val * win_rate_val * deploy_factor_val * lmp_val

            total_hourly_as_revenue_rate += cap_payment + energy_payment + adder_val

        return total_hourly_as_revenue_rate

    except AttributeError as e:
        logger.error(
            f"Missing component during hourly AS revenue calc for t={t}, ISO={iso_suffix}: {e}"
        )
        return 0.0
    except Exception as e:
        logger.error(
            f"Error during hourly AS revenue calculation for t={t}, ISO={iso_suffix}: {e}",
            exc_info=True,
        )
        return 0.0


def extract_results(
    model: pyo.ConcreteModel,
    target_iso: str,
    output_dir: str = "../output/opt/Results_Standardized",
):
    """
    Extracts comprehensive results from the solved Pyomo model.
    """
    logger.info(f"Extracting comprehensive results for {target_iso}...")
    model.TARGET_ISO = getattr(
        model, "TARGET_ISO", target_iso
    )  # Ensure model object has it
    target_iso_local = model.TARGET_ISO

    simulate_dispatch_mode = getattr(
        model, "SIMULATE_AS_DISPATCH_EXECUTION", False)
    can_provide_as_local = getattr(
        model, "CAN_PROVIDE_ANCILLARY_SERVICES", False)
    logger.info(
        f"Results extraction mode: {'Dispatch Execution' if simulate_dispatch_mode else 'Bidding Strategy'}"
    )

    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "TimePeriods") or not list(model.TimePeriods):
        logger.error(
            "Model has no TimePeriods defined. Cannot extract results.")
        return pd.DataFrame(), {}
    hours = list(model.TimePeriods)

    hourly_data = {}
    summary_results: Dict[str, Any] = {}
    time_factor = get_param(model, "delT_minutes", default=60.0) / 60.0
    if time_factor <= 0:
        logger.error("Invalid time_factor (<=0). Cannot extract results.")
        return pd.DataFrame(), {}

    summary_results["Target_ISO"] = target_iso_local
    summary_results["Simulation_Mode"] = (
        "Dispatch Execution" if simulate_dispatch_mode else "Bidding Strategy"
    )

    elec_capacity_val = 0.0
    batt_capacity_val = 0.0
    batt_power_val = 0.0

    enable_electrolyzer = getattr(
        model, "ENABLE_ELECTROLYZER", ENABLE_ELECTROLYZER)
    enable_battery = getattr(model, "ENABLE_BATTERY", ENABLE_BATTERY)
    enable_npp = getattr(model, "ENABLE_NUCLEAR_GENERATOR",
                         ENABLE_NUCLEAR_GENERATOR)
    enable_lte = getattr(
        model, "LTE_MODE", False
    )  # LTE_MODE is set on model during creation
    enable_h2_storage = getattr(model, "ENABLE_H2_STORAGE", ENABLE_H2_STORAGE)
    enable_startup_shutdown = getattr(
        model, "ENABLE_STARTUP_SHUTDOWN", ENABLE_STARTUP_SHUTDOWN
    )
    enable_degradation = getattr(
        model,
        "ENABLE_ELECTROLYZER_DEGRADATION_TRACKING",
        ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    )

    if enable_electrolyzer:
        elec_cap_component = getattr(model, "pElectrolyzer_max", None)
        if isinstance(elec_cap_component, pyo.Param):
            elec_capacity_val = pyo.value(elec_cap_component)
            summary_results["Fixed_Electrolyzer_Capacity_MW"] = elec_capacity_val
        elif isinstance(elec_cap_component, pyo.Var):
            elec_capacity_val = get_var_value(elec_cap_component, default=0.0)
            summary_results["Optimal_Electrolyzer_Capacity_MW"] = elec_capacity_val
        else:
            elec_capacity_val = get_param(
                model, "pElectrolyzer_max_upper_bound", default=0.0
            )
            summary_results["Assumed_Electrolyzer_Capacity_MW"] = elec_capacity_val
    hourly_data["Electrolyzer_Capacity_MW"] = [elec_capacity_val] * len(hours)

    if enable_battery:
        batt_cap_component = getattr(model, "BatteryCapacity_MWh", None)
        batt_pow_component = getattr(model, "BatteryPower_MW", None)
        if isinstance(batt_cap_component, pyo.Param):  # Fixed capacity
            batt_capacity_val = pyo.value(batt_cap_component)
            batt_power_val = (
                pyo.value(
                    batt_pow_component) if batt_pow_component is not None else 0.0
            )
            summary_results["Fixed_Battery_Capacity_MWh"] = batt_capacity_val
            summary_results["Fixed_Battery_Power_MW"] = batt_power_val
        elif isinstance(batt_cap_component, pyo.Var):  # Optimized capacity
            batt_capacity_val = get_var_value(batt_cap_component, default=0.0)
            batt_power_val = get_var_value(batt_pow_component, default=0.0)
            summary_results["Optimal_Battery_Capacity_MWh"] = batt_capacity_val
            summary_results["Optimal_Battery_Power_MW"] = batt_power_val
        else:  # Fallback or not defined as expected
            batt_capacity_val = get_param(
                model, "BatteryCapacity_max_param", default=0.0
            )  # Using _param to indicate it's the bound
            power_ratio = get_param(model, "BatteryPowerRatio", default=0.0)
            batt_power_val = batt_capacity_val * power_ratio
            summary_results["Assumed_Battery_Capacity_MWh"] = batt_capacity_val
            summary_results["Assumed_Battery_Power_MW"] = batt_power_val
    hourly_data["Battery_Capacity_MWh"] = [batt_capacity_val] * len(hours)
    hourly_data["Battery_Power_MW"] = [batt_power_val] * len(hours)

    # Extract H2 storage capacity and constant sales rate for optimal sizing mode
    h2_storage_capacity_val = 0.0
    h2_constant_sales_rate_val = 0.0

    if enable_h2_storage:
        # Check if optimal storage sizing is enabled
        enable_optimal_h2_sizing = getattr(
            model, "ENABLE_OPTIMAL_H2_STORAGE_SIZING", False)

        if enable_optimal_h2_sizing:
            # Extract optimal storage capacity
            h2_storage_cap_component = getattr(
                model, "H2_storage_capacity_optimal", None)
            if h2_storage_cap_component is not None:
                h2_storage_capacity_val = get_var_value(
                    h2_storage_cap_component, default=0.0)
                summary_results["Optimal_H2_Storage_Capacity_kg"] = h2_storage_capacity_val
            else:
                # Fallback to fixed capacity
                h2_storage_cap_component = getattr(
                    model, "H2_storage_capacity_max", None)
                if h2_storage_cap_component is not None:
                    h2_storage_capacity_val = pyo.value(h2_storage_cap_component) if isinstance(
                        h2_storage_cap_component, pyo.Param) else get_var_value(h2_storage_cap_component, default=0.0)
                    summary_results["Fixed_H2_Storage_Capacity_kg"] = h2_storage_capacity_val

            # Extract constant sales rate
            h2_sales_rate_component = getattr(
                model, "H2_constant_sales_rate", None)
            if h2_sales_rate_component is not None:
                h2_constant_sales_rate_val = get_var_value(
                    h2_sales_rate_component, default=0.0)
                summary_results["Optimal_H2_Constant_Sales_Rate_kg_hr"] = h2_constant_sales_rate_val
        else:
            # Fixed capacity mode
            h2_storage_cap_component = getattr(
                model, "H2_storage_capacity_max", None)
            if h2_storage_cap_component is not None:
                h2_storage_capacity_val = pyo.value(h2_storage_cap_component) if isinstance(
                    h2_storage_cap_component, pyo.Param) else get_var_value(h2_storage_cap_component, default=0.0)
                summary_results["Fixed_H2_Storage_Capacity_kg"] = h2_storage_capacity_val

    hourly_data["H2_Storage_Capacity_kg"] = [
        h2_storage_capacity_val] * len(hours)
    hourly_data["H2_Constant_Sales_Rate_kg_hr"] = [
        h2_constant_sales_rate_val] * len(hours)

    logger.info("Extracting hourly variables...")
    var_extract_list = [
        ("pIES", "pIES_MW", True, 0.0),
        ("pTurbine", "pTurbine_MW", enable_npp, 0.0),
        ("qSteam_Turbine", "qSteam_Turbine_MWth", enable_npp, 0.0),
        ("pElectrolyzer", "pElectrolyzer_MW", enable_electrolyzer, 0.0),
        (
            "pElectrolyzerSetpoint",
            "pElectrolyzerSetpoint_MW",
            enable_electrolyzer,
            0.0,
        ),
        (
            "mHydrogenProduced",
            "mHydrogenProduced_kg_hr",
            enable_electrolyzer,
            0.0,
        ),
        (
            "qSteam_Electrolyzer",
            "qSteam_Electrolyzer_MWth",
            enable_electrolyzer and not enable_lte,
            0.0,
        ),
        ("pAuxiliary", "pAuxiliary_MW", True, 0.0),
        ("H2_storage_level", "H2_Storage_Level_kg", enable_h2_storage, 0.0),
        # ("H2_to_market", "H2_to_Market_kg_hr", enable_h2_storage, 0.0),  # Removed: All H2 now goes through storage
        ("H2_from_storage", "H2_from_Storage_kg_hr", enable_h2_storage, 0.0),
        ("H2_to_storage", "H2_to_Storage_Input_kg_hr", enable_h2_storage, 0.0),
        (
            "uElectrolyzer",
            "Electrolyzer_Status(0=off,1=on)",
            enable_startup_shutdown,
            (1.0 if enable_electrolyzer else 0.0),
        ),
        (
            "vElectrolyzerStartup",
            "Electrolyzer_Startup(0=no,1=yes)",
            enable_startup_shutdown,
            0.0,
        ),
        (
            "wElectrolyzerShutdown",
            "Electrolyzer_Shutdown(0=no,1=yes)",
            enable_startup_shutdown,
            0.0,
        ),
        (
            "DegradationState",
            "DegradationState_Units",
            enable_degradation,
            0.0,
        ),
        (
            "pElectrolyzerRampPos",
            "pElectrolyzerRampPos_MW",
            enable_electrolyzer,
            0.0,
        ),
        (
            "pElectrolyzerRampNeg",
            "pElectrolyzerRampNeg_MW",
            enable_electrolyzer,
            0.0,
        ),
        (
            "qSteamElectrolyzerRampPos",
            "qSteamElectrolyzerRampPos_MWth",
            enable_electrolyzer and not enable_lte,
            0.0,
        ),
        (
            "qSteamElectrolyzerRampNeg",
            "qSteamElectrolyzerRampNeg_MWth",
            enable_electrolyzer and not enable_lte,
            0.0,
        ),
        ("BatterySOC", "Battery_SOC_MWh", enable_battery, 0.0),
        ("BatteryCharge", "Battery_Charge_MW", enable_battery, 0.0),
        ("BatteryDischarge", "Battery_Discharge_MW", enable_battery, 0.0),
        ("BatteryBinaryCharge", "Battery_Charge_Binary", enable_battery, 0.0),
        (
            "BatteryBinaryDischarge",
            "Battery_Discharge_Binary",
            enable_battery,
            0.0,
        ),
        # **NEW: Extract grid purchase/sale data for battery source attribution**
        ("pGridPurchase", "pGridPurchase_MW", True, 0.0),
        ("pGridSale", "pGridSale_MW", True, 0.0),
    ]
    for var_name, col_name, is_enabled_flag, default_val in var_extract_list:
        if is_enabled_flag:
            var_component = getattr(model, var_name, None)
            if var_component is not None:
                hourly_data[col_name] = [
                    get_var_value(var_component, t, default=default_val) for t in hours
                ]
            else:
                logger.warning(
                    f"Variable '{var_name}' expected but not found. Filling '{col_name}' with {default_val}."
                )
                hourly_data[col_name] = [default_val] * len(hours)
        else:
            hourly_data[col_name] = [default_val] * len(hours)

    logger.info("Extracting AS bids...")
    # Define a map to get the internal service names for the target ISO
    # This ensures we only create columns for relevant ancillary services.
    reserve_map_by_iso = {
        "SPP": {
            "Spin": "SR",
            "Sup": "NSR",
            "RamU": "RampUp",
            "RamD": "RampDown",
            "UncU": "UncU",
        },
        "CAISO": {
            "Spin": "SR",
            "NSpin": "NSR",
            "RMU": "RampUp",
            "RMD": "RampDown",
        },
        "ERCOT": {"Spin": "SR", "NSpin": "NSR", "ECRS": "ECRS"},
        "PJM": {"Syn": "SR", "Rse": "NSR", "TMR": "ThirtyMin"},
        "NYISO": {
            "Spin10": "SR",
            "NSpin10": "NSR",
            "Res30": "ThirtyMin",
        },
        "ISONE": {
            "Spin10": "SR",
            "NSpin10": "NSR",
            "OR30": "ThirtyMin",
        },
        "MISO": {
            "Spin": "SR",
            "Sup": "NSR",
            "STR": "ThirtyMin",
            "RamU": "RampUp",
            "RamD": "RampDown",
        },
    }
    # ISOs with standard Regulation Up/Down services
    isos_with_reg = ["PJM", "NYISO", "MISO", "SPP", "CAISO", "ERCOT"]

    as_service_list_internal = []
    if target_iso_local in reserve_map_by_iso:
        as_service_list_internal.extend(
            reserve_map_by_iso[target_iso_local].values())
    if target_iso_local in isos_with_reg:
        as_service_list_internal.extend(["RegUp", "RegDown"])

    # Remove duplicates and sort for consistent column order
    as_service_list_internal = sorted(list(set(as_service_list_internal)))

    if as_service_list_internal:
        logger.info(
            f"Extracting AS bids for {target_iso_local} services: {as_service_list_internal}"
        )
    else:
        logger.info(
            f"No ancillary services defined for {target_iso_local} in this script. Skipping AS bid extraction."
        )

    components_providing_as = []
    if enable_electrolyzer:
        components_providing_as.append("Electrolyzer")
    if enable_battery:
        components_providing_as.append("Battery")
    if enable_npp and (enable_electrolyzer or enable_battery):
        components_providing_as.append("Turbine")

    all_as_components_labels = ["Electrolyzer", "Battery", "Turbine", "Total"]
    for comp_label in all_as_components_labels:
        for service_internal in as_service_list_internal:
            is_total = comp_label == "Total"
            base_name = (
                f"Total_{service_internal}"
                if is_total
                else f"{service_internal}_{comp_label}"
            )
            col_name = f"{base_name}_Bid_MW"
            if can_provide_as_local and hasattr(model, base_name):
                var_comp = getattr(model, base_name)
                hourly_data[col_name] = [
                    get_var_value(var_comp, t, default=0.0) for t in hours
                ]
            else:
                hourly_data[col_name] = [0.0] * len(hours)

    if simulate_dispatch_mode and can_provide_as_local:
        logger.info(
            "Dispatch Simulation Mode: Extracting *_Deployed variables.")
        for comp in components_providing_as:
            for service_internal in as_service_list_internal:
                deployed_var_name = f"{service_internal}_{comp}_Deployed"
                col_name = f"{deployed_var_name}_MW"
                if hasattr(model, deployed_var_name):
                    hourly_data[col_name] = [
                        get_var_value(
                            getattr(model, deployed_var_name), t, default=0.0)
                        for t in hours
                    ]
                else:  # Only add zero columns if var missing but component could provide it
                    if comp in components_providing_as:  # Defensive check
                        hourly_data[col_name] = [0.0] * len(hours)

    logger.info("Extracting input prices and factors...")
    hourly_data["EnergyPrice_LMP_USDperMWh"] = [
        get_param(model, "energy_price", t, default=0.0) for t in hours
    ]
    if can_provide_as_local:
        # This map must align with model.py's iso_service_map for parameter loading
        iso_service_map_for_inputs = {
            "SPP": ["RegU", "RegD", "Spin", "Sup", "RamU", "RamD", "UncU"],
            "CAISO": ["RegU", "RegD", "Spin", "NSpin", "RMU", "RMD"],
            "ERCOT": ["RegU", "RegD", "Spin", "NSpin", "ECRS"],
            "PJM": ["RegUp", "RegDown", "Syn", "Rse", "TMR"],
            "NYISO": ["RegUp", "RegDown", "Spin10", "NSpin10", "Res30"],
            "ISONE": ["Spin10", "NSpin10", "OR30"],
            "MISO": ["RegUp", "RegDown", "Spin", "Sup", "STR", "RamU", "RamD"],
        }
        if target_iso_local in iso_service_map_for_inputs:
            for service_key in iso_service_map_for_inputs[target_iso_local]:
                # Parameter types to extract for each service_key
                param_configs = [
                    ("p", 0.0),
                    ("loc", 0.0),
                    ("winning_rate", 1.0),
                ]
                # Add factors based on service type (regulation vs reserve)
                is_regulation_service = (
                    "RegU" in service_key
                    or "RegD" in service_key
                    or "RegUp" in service_key
                    or "RegDown" in service_key
                )
                if is_regulation_service:
                    param_configs.extend(
                        [
                            ("mileage_factor", 1.0),
                            ("performance_factor", 1.0),
                            ("deploy_factor", 0.0),
                        ]  # Also extract deploy_factor for regulation services
                    )
                else:  # Reserve service
                    param_configs.append(("deploy_factor", 0.0))

                for param_prefix, default_val in param_configs:
                    # Construct param name as it exists on the model (e.g., p_RegUp_PJM)
                    # The get_param in utils handles TARGET_ISO suffixing.
                    # Here, param_base_name_on_model is 'p_RegUp', 'mileage_factor_RegDown', etc.
                    param_base_name_on_model = f"{param_prefix}_{service_key}"
                    # For CSV column
                    output_col_name = f"{param_prefix}_{service_key}_{target_iso_local}"

                    hourly_data[output_col_name] = [
                        get_param(
                            model,
                            param_base_name_on_model,
                            t,
                            default=default_val,
                        )
                        for t in hours
                    ]

    logger.info("Creating final DataFrame from collected hourly data...")
    try:
        results_df = pd.DataFrame(
            hourly_data, index=pd.Index(hours, name="HourOfYear"))
    except ValueError as ve:
        logger.error(
            f"Error creating DataFrame, likely due to inconsistent array lengths: {ve}"
        )
        for k, v_list in hourly_data.items():
            logger.debug(
                f"Length of '{k}': {len(v_list) if isinstance(v_list, list) else 'N/A'}"
            )
        return pd.DataFrame(), {}

    logger.info("Calculating hourly revenues...")
    if (
        "pIES_MW" in results_df.columns
        and "EnergyPrice_LMP_USDperMWh" in results_df.columns
    ):
        results_df["Revenue_Energy_USD"] = (
            results_df["pIES_MW"]
            * results_df["EnergyPrice_LMP_USDperMWh"]
            * time_factor
        )
    else:
        results_df["Revenue_Energy_USD"] = 0.0

    h2_value_param = (
        get_param(model, "H2_value",
                  default=0.0) if enable_electrolyzer else 0.0
    )
    h2_subsidy_param = (
        get_param(model, "hydrogen_subsidy_per_kg", default=0.0)
        if enable_electrolyzer
        else 0.0
    )
    results_df["Revenue_Hydrogen_Sales_USD"] = 0.0
    results_df["Revenue_Hydrogen_Subsidy_USD"] = 0.0
    if enable_electrolyzer and "mHydrogenProduced_kg_hr" in results_df.columns:
        results_df["Revenue_Hydrogen_Subsidy_USD"] = (
            results_df["mHydrogenProduced_kg_hr"] *
            h2_subsidy_param * time_factor
        )
        if (
            enable_h2_storage
            and "H2_from_Storage_kg_hr" in results_df.columns
        ):
            # Revenue should be based on actual sales quantity (H2_from_storage)
            # since all hydrogen must go through storage (h2_no_direct_sales_rule enforces H2_to_market = 0)
            results_df["Revenue_Hydrogen_Sales_USD"] = (
                results_df["H2_from_Storage_kg_hr"]
                * h2_value_param
                * time_factor
            )
        elif not enable_h2_storage:
            results_df["Revenue_Hydrogen_Sales_USD"] = (
                results_df["mHydrogenProduced_kg_hr"] *
                h2_value_param * time_factor
            )
    results_df["Revenue_Hydrogen_USD"] = (
        results_df["Revenue_Hydrogen_Sales_USD"]
        + results_df["Revenue_Hydrogen_Subsidy_USD"]
    )

    results_df["Revenue_Ancillary_USD"] = [
        calculate_hourly_as_revenue(model, t) * time_factor for t in hours
    ]
    results_df["Revenue_Total_USD"] = results_df[
        ["Revenue_Energy_USD", "Revenue_Hydrogen_USD", "Revenue_Ancillary_USD"]
    ].sum(axis=1)

    logger.info("Calculating hourly costs...")
    cost_calc_list = [
        ("Cost_VOM_Turbine_USD", enable_npp, "vom_turbine", "pTurbine_MW"),
        (
            "Cost_VOM_Electrolyzer_USD",
            enable_electrolyzer,
            "vom_electrolyzer",
            "pElectrolyzer_MW",
        ),
        (
            "Cost_VOM_Battery_USD",
            enable_battery,
            "vom_battery_per_mwh_cycled",
            ["Battery_Charge_MW", "Battery_Discharge_MW"],
        ),
        (
            "Cost_Water_USD",
            enable_electrolyzer,
            "cost_water_per_kg_h2",
            "mHydrogenProduced_kg_hr",
        ),
        (
            "Cost_Ramping_USD",
            enable_electrolyzer,
            "cost_electrolyzer_ramping",
            ["pElectrolyzerRampPos_MW", "pElectrolyzerRampNeg_MW"],
        ),
        (
            "Cost_Storage_Cycle_USD",
            enable_h2_storage,
            "vom_storage_cycle",
            ["H2_to_Storage_Input_kg_hr", "H2_from_Storage_kg_hr"],
        ),
        (
            "Cost_Startup_USD",
            enable_startup_shutdown,
            "cost_startup_electrolyzer",
            "Electrolyzer_Startup(0=no,1=yes)",
        ),
    ]
    cost_cols_for_total = []
    for cost_col, is_enabled, param_name, source_col_or_list in cost_calc_list:
        if is_enabled:
            cost_rate_param = get_param(model, param_name, default=0.0)
            if cost_rate_param > 1e-9:
                if isinstance(source_col_or_list, list):
                    if all(col in results_df.columns for col in source_col_or_list):
                        source_sum = results_df[source_col_or_list].sum(axis=1)
                        if cost_col == "Cost_VOM_Battery_USD":
                            results_df[cost_col] = (
                                source_sum * cost_rate_param * time_factor / 2.0
                            )
                        elif cost_col in [
                            "Cost_Ramping_USD",
                            "Cost_Startup_USD",
                        ]:
                            results_df[cost_col] = source_sum * cost_rate_param
                        elif cost_col == "Cost_Storage_Cycle_USD":
                            results_df[cost_col] = (
                                source_sum * cost_rate_param * time_factor
                            )
                        else:
                            results_df[cost_col] = 0.0
                        cost_cols_for_total.append(cost_col)
                    else:
                        results_df[cost_col] = 0.0
                else:  # Single source column
                    source_col = source_col_or_list
                    if source_col in results_df.columns:
                        if cost_col == "Cost_Water_USD":
                            results_df[cost_col] = (
                                results_df[source_col] *
                                cost_rate_param * time_factor
                            )
                        else:
                            results_df[cost_col] = (
                                results_df[source_col] *
                                cost_rate_param * time_factor
                            )
                        cost_cols_for_total.append(cost_col)
                    else:
                        results_df[cost_col] = 0.0
            else:
                results_df[cost_col] = 0.0
        else:
            results_df[cost_col] = 0.0
    if "Cost_Ramping_USD" in results_df.columns and not results_df.empty:
        results_df.loc[results_df.index.min(), "Cost_Ramping_USD"] = 0.0

    # Add calculation for NPP Fuel Cost and Fixed O&M Cost if NPP is enabled
    if enable_npp:
        # NPP Fuel Cost - Use total thermal energy output for accurate fuel cost calculation
        npp_fuel_cost_param = get_param(model, "npp_fuel_cost", default=0.0)
        if npp_fuel_cost_param > 0:
            # Check if thermal energy data is available for more accurate calculation
            if "qSteam_Turbine_MWth" in results_df.columns:
                # Calculate fuel cost based on total thermal energy output
                total_thermal_energy_mwth = results_df["qSteam_Turbine_MWth"].copy(
                )

                # Add thermal energy for electrolyzer (HTE mode only)
                if ("qSteam_Electrolyzer_MWth" in results_df.columns and
                        not enable_lte):
                    total_thermal_energy_mwth += results_df["qSteam_Electrolyzer_MWth"]

                # Convert electrical fuel cost ($/MWh_elec) to thermal fuel cost ($/MWh_th)
                thermal_efficiency = get_param(
                    model, "convertTtE_const", default=0.4)
                if thermal_efficiency > 0:
                    fuel_cost_per_mwh_thermal = npp_fuel_cost_param * thermal_efficiency
                    results_df["Cost_Fuel_NPP_USD"] = (
                        total_thermal_energy_mwth * fuel_cost_per_mwh_thermal * time_factor
                    )
                else:
                    # Fallback to original calculation if thermal efficiency not available
                    if "pTurbine_MW" in results_df.columns:
                        results_df["Cost_Fuel_NPP_USD"] = (
                            results_df["pTurbine_MW"] *
                            npp_fuel_cost_param * time_factor
                        )
                    else:
                        results_df["Cost_Fuel_NPP_USD"] = 0.0
            elif "pTurbine_MW" in results_df.columns:
                # Fallback to original calculation if thermal data not available
                results_df["Cost_Fuel_NPP_USD"] = (
                    results_df["pTurbine_MW"] *
                    npp_fuel_cost_param * time_factor
                )
            else:
                results_df["Cost_Fuel_NPP_USD"] = 0.0

            if results_df["Cost_Fuel_NPP_USD"].sum() > 0:
                cost_cols_for_total.append("Cost_Fuel_NPP_USD")
        else:
            results_df["Cost_Fuel_NPP_USD"] = 0.0

        # NPP Fixed O&M Cost
        npp_fixed_om_cost_param = get_param(
            model, "npp_fixed_om_cost", default=0.0)

        # Use nameplate capacity if available, otherwise fallback to pTurbine_max
        npp_nameplate_capacity = get_param(
            model, "npp_nameplate_capacity_mw", default=0.0)
        if npp_nameplate_capacity <= 0:
            # Fallback to pTurbine_max if nameplate capacity not available
            npp_nameplate_capacity = get_param(
                model, "pTurbine_max", default=0.0)

        if npp_fixed_om_cost_param > 0 and npp_nameplate_capacity > 0:
            # This is an annual cost, so we need to scale it to the simulation period
            # The cost is per hour of the simulation
            # Use actual nameplate capacity for fixed O&M calculation
            annual_total_fom = npp_fixed_om_cost_param * npp_nameplate_capacity
            hourly_fom = annual_total_fom / \
                get_param(model, "HOURS_IN_YEAR", default=8760.0)
            results_df["Cost_Fixed_OM_NPP_USD"] = hourly_fom * time_factor
            cost_cols_for_total.append("Cost_Fixed_OM_NPP_USD")
        else:
            results_df["Cost_Fixed_OM_NPP_USD"] = 0.0
    else:
        results_df["Cost_Fuel_NPP_USD"] = 0.0
        results_df["Cost_Fixed_OM_NPP_USD"] = 0.0

    # Define NPP OPEX components
    npp_opex_components = [
        "Cost_VOM_Turbine_USD",
        "Cost_Fuel_NPP_USD",
        "Cost_Fixed_OM_NPP_USD"
    ]

    # Define H2 and Battery OPEX components
    h2_battery_opex_components = [
        "Cost_VOM_Electrolyzer_USD",
        "Cost_VOM_Battery_USD",
        "Cost_Water_USD",
        "Cost_Ramping_USD",
        "Cost_Storage_Cycle_USD",
        "Cost_Startup_USD"
    ]

    # Calculate NPP OPEX total
    npp_opex_cols = [
        col for col in npp_opex_components if col in results_df.columns]
    results_df["Cost_HourlyOpex_NPP_USD"] = (
        results_df[npp_opex_cols].sum(axis=1) if npp_opex_cols else 0.0
    )

    # Calculate H2 and Battery OPEX total
    h2_battery_opex_cols = [
        col for col in h2_battery_opex_components if col in results_df.columns]
    results_df["Cost_HourlyOpex_H2_Battery_USD"] = (
        results_df[h2_battery_opex_cols].sum(
            axis=1) if h2_battery_opex_cols else 0.0
    )

    # Calculate total OPEX (sum of both categories)
    results_df["Cost_HourlyOpex_Total_USD"] = (
        results_df["Cost_HourlyOpex_NPP_USD"] +
        results_df["Cost_HourlyOpex_H2_Battery_USD"]
    )

    logger.info("Calculating hourly profit...")
    results_df["Profit_Hourly_USD"] = (
        results_df["Revenue_Total_USD"] -
        results_df["Cost_HourlyOpex_Total_USD"]
    )

    logger.info("Calculating summary statistics...")
    summary_results["Total_Revenue_USD"] = results_df["Revenue_Total_USD"].sum()
    summary_results["Total_Energy_Revenue_USD"] = results_df["Revenue_Energy_USD"].sum()
    summary_results["Total_Hydrogen_Revenue_USD"] = results_df[
        "Revenue_Hydrogen_USD"
    ].sum()
    if "Revenue_Hydrogen_Sales_USD" in results_df.columns:
        summary_results["Total_Hydrogen_Sales_Revenue_USD"] = results_df[
            "Revenue_Hydrogen_Sales_USD"
        ].sum()
    if "Revenue_Hydrogen_Subsidy_USD" in results_df.columns:
        summary_results["Total_Hydrogen_Subsidy_Revenue_USD"] = results_df[
            "Revenue_Hydrogen_Subsidy_USD"
        ].sum()
    summary_results["Total_Ancillary_Revenue_USD"] = results_df[
        "Revenue_Ancillary_USD"
    ].sum()
    summary_results["Total_Hourly_Opex_USD"] = results_df[
        "Cost_HourlyOpex_Total_USD"
    ].sum()
    # Add NPP OPEX and H2/Battery OPEX breakdown
    summary_results["Total_NPP_Opex_USD"] = results_df[
        "Cost_HourlyOpex_NPP_USD"
    ].sum()
    summary_results["Total_H2_Battery_Opex_USD"] = results_df[
        "Cost_HourlyOpex_H2_Battery_USD"
    ].sum()

    # Verify OPEX breakdown consistency
    breakdown_total = summary_results["Total_NPP_Opex_USD"] + \
        summary_results["Total_H2_Battery_Opex_USD"]
    original_total = summary_results["Total_Hourly_Opex_USD"]
    if abs(breakdown_total - original_total) > 0.01:
        logger.warning(
            f"OPEX breakdown inconsistency: NPP+H2/Battery={breakdown_total:.2f} vs Total={original_total:.2f}"
        )

    summary_results["Total_VOM_Cost_USD"] = sum(
        results_df[col].sum()
        for col in [
            "Cost_VOM_Turbine_USD",
            "Cost_VOM_Electrolyzer_USD",
            "Cost_VOM_Battery_USD",
        ]
        if col in results_df.columns
    )
    summary_results["Total_Water_Cost_USD"] = (
        results_df["Cost_Water_USD"].sum()
        if "Cost_Water_USD" in results_df.columns
        else 0.0
    )
    summary_results["Total_Ramping_Cost_USD"] = (
        results_df["Cost_Ramping_USD"].sum()
        if "Cost_Ramping_USD" in results_df.columns
        else 0.0
    )
    summary_results["Total_Storage_Cycle_Cost_USD"] = (
        results_df["Cost_Storage_Cycle_USD"].sum()
        if "Cost_Storage_Cycle_USD" in results_df.columns
        else 0.0
    )
    summary_results["Total_Startup_Cost_USD"] = (
        results_df["Cost_Startup_USD"].sum()
        if "Cost_Startup_USD" in results_df.columns
        else 0.0
    )
    # Add NPP costs to summary
    summary_results["Total_Fuel_NPP_Cost_USD"] = (
        results_df["Cost_Fuel_NPP_USD"].sum()
        if "Cost_Fuel_NPP_USD" in results_df.columns
        else 0.0
    )
    summary_results["Total_Fixed_OM_NPP_Cost_USD"] = (
        results_df["Cost_Fixed_OM_NPP_USD"].sum()
        if "Cost_Fixed_OM_NPP_USD" in results_df.columns
        else 0.0
    )

    total_annualized_capex = 0.0
    electrolyzer_annual_capex = 0.0
    battery_annual_capex = 0.0
    h2_storage_annual_capex = 0.0  # Add H2 storage CAPEX tracking
    total_hours_sim = len(hours) * time_factor
    scaling_factor = (
        total_hours_sim / get_param(model, "HOURS_IN_YEAR", default=8760.0)
        if total_hours_sim > 0 and get_param(model, "HOURS_IN_YEAR", default=8760.0) > 0
        else 0.0
    )
    if enable_electrolyzer:
        cost_elec_cap_param = get_param(
            model, "cost_electrolyzer_capacity", default=0.0
        )
        electrolyzer_annual_capex = (
            elec_capacity_val * cost_elec_cap_param * scaling_factor
        )
        total_annualized_capex += electrolyzer_annual_capex
    if enable_battery:
        cost_batt_cap_mwh_yr = get_param(
            model, "BatteryCapex_USD_per_MWh_year", default=0.0
        )
        cost_batt_pow_mw_yr = get_param(
            model, "BatteryCapex_USD_per_MW_year", default=0.0
        )
        cost_batt_fom_mwh_yr = get_param(
            model, "BatteryFixedOM_USD_per_MWh_year", default=0.0
        )
        battery_annual_capex = (
            batt_capacity_val * cost_batt_cap_mwh_yr
            + batt_power_val * cost_batt_pow_mw_yr
            + batt_capacity_val * cost_batt_fom_mwh_yr
        ) * scaling_factor
        total_annualized_capex += battery_annual_capex

    # Add H2 Storage CAPEX calculation (matching the model's AnnualizedCapex_rule logic)
    if enable_h2_storage:
        cost_h2_storage_capex_param = get_param(
            model, "cost_h2_storage_capex", default=0.0
        )
        h2_storage_annual_capex = (
            h2_storage_capacity_val * cost_h2_storage_capex_param * scaling_factor
        )
        total_annualized_capex += h2_storage_annual_capex

    summary_results["Total_Annualized_Capex_USD"] = total_annualized_capex
    summary_results["Electrolyzer_Annualized_Capex_USD"] = electrolyzer_annual_capex
    summary_results["Battery_Annualized_Capex_USD"] = battery_annual_capex
    # Add H2 storage CAPEX to summary
    summary_results["H2_Storage_Annualized_Capex_USD"] = h2_storage_annual_capex
    summary_results["Total_Profit_Calculated_USD"] = (
        summary_results["Total_Revenue_USD"]
        - summary_results["Total_Hourly_Opex_USD"]
        - summary_results["Total_Annualized_Capex_USD"]
    )
    summary_results["Objective_Value_USD"] = get_var_value(
        getattr(model, "TotalProfit_Objective", None), default=None
    )

    if "mHydrogenProduced_kg_hr" in results_df.columns:
        summary_results["Total_H2_Produced_kg"] = (
            results_df["mHydrogenProduced_kg_hr"].sum() * time_factor
        )
    else:
        summary_results["Total_H2_Produced_kg"] = 0.0
    if enable_h2_storage:
        # All hydrogen now goes through storage, so H2_to_Market_Direct_kg is always 0
        summary_results["Total_H2_to_Market_Direct_kg"] = 0.0
        summary_results["Total_H2_from_Storage_kg"] = (
            results_df["H2_from_Storage_kg_hr"].sum() * time_factor
            if "H2_from_Storage_kg_hr" in results_df.columns
            else 0.0
        )
        if "H2_Storage_Level_kg" in results_df.columns and not results_df.empty:
            summary_results["Final_H2_Storage_Level_kg"] = results_df[
                "H2_Storage_Level_kg"
            ].iloc[-1]
        else:
            summary_results["Final_H2_Storage_Level_kg"] = 0.0
    if elec_capacity_val > 1e-6 and "pElectrolyzer_MW" in results_df.columns:
        avg_elec_power_actual = results_df["pElectrolyzer_MW"].mean()
        summary_results["Electrolyzer_Capacity_Factor_Actual"] = (
            avg_elec_power_actual / elec_capacity_val if elec_capacity_val > 0 else 0.0
        )
    else:
        summary_results["Electrolyzer_Capacity_Factor_Actual"] = 0.0
    if "Electrolyzer_Startup(0=no,1=yes)" in hourly_data:
        summary_results["Total_Electrolyzer_Startups"] = int(
            np.sum(hourly_data["Electrolyzer_Startup(0=no,1=yes)"])
        )
    if (
        "DegradationState_Units" in hourly_data
        and hourly_data["DegradationState_Units"]
    ):
        last_state = pd.Series(hourly_data["DegradationState_Units"]).iloc[-1]
        summary_results["Final_DegradationState_Units"] = (
            last_state if pd.notna(last_state) else 0.0
        )

    if simulate_dispatch_mode and can_provide_as_local:
        logger.info("Calculating total deployed AS amounts...")
        for (
            service_internal
        ) in as_service_list_internal:  # Use internal names for deployed vars
            total_deployed_mwh = 0.0
            temp_sum_mw = 0.0
            for comp in components_providing_as:
                comp_col_name = f"{service_internal}_{comp}_Deployed_MW"
                if comp_col_name in results_df.columns:
                    temp_sum_mw += results_df[comp_col_name].sum()
            total_deployed_mwh = temp_sum_mw * time_factor
            summary_results[f"Total_Deployed_{service_internal}_MWh"] = (
                total_deployed_mwh
            )

    if not summary_results:
        logger.warning("Summary results dictionary is empty.")
    else:
        output_summary_path = results_dir / \
            f"{target_iso_local}_Summary_Results.txt"
        try:
            logger.info(
                f"Attempting to write summary results to: {output_summary_path}"
            )
            with open(output_summary_path, "w") as f:
                f.write(f"--- Summary Results for {target_iso_local} ---\n")
                f.write(
                    f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}\n"
                )
                f.write(
                    f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}\n")
                key_order = [
                    k
                    for k in summary_results.keys()
                    if k
                    not in [
                        "Simulation_Mode",
                        "Target_ISO",
                        "Objective_Value_USD",
                        "Total_Profit_Calculated_USD",
                    ]
                ]
                for key in key_order:
                    value = summary_results[key]
                    try:
                        if value is None:
                            line = f"{key}: None\n"
                        elif isinstance(value, (float, np.floating)):
                            line = f"{key}: {value:,.4f}\n"
                        elif isinstance(value, (int, np.integer)):
                            line = f"{key}: {value:,}\n"
                        else:
                            line = f"{key}: {value}\n"
                        f.write(line)
                    except Exception as write_err:
                        logger.error(
                            f"Error writing summary key '{key}' value '{value}': {write_err}"
                        )
                        f.write(f"{key}: ERROR_WRITING_VALUE\n")
                f.write("\n--- Profitability ---\n")
                obj_val = summary_results.get("Objective_Value_USD")
                calc_prof = summary_results.get("Total_Profit_Calculated_USD")
                f.write(
                    f"Objective_Value_USD: {obj_val:,.4f}\n"
                    if obj_val is not None
                    else "Objective_Value_USD: N/A (Solver Failed?)\n"
                )
                f.write(
                    f"Total_Profit_Calculated_USD: {calc_prof:,.4f}\n"
                    if calc_prof is not None
                    else "Total_Profit_Calculated_USD: N/A (Calculation Error?)\n"
                )
                if isinstance(obj_val, (int, float, np.number)) and isinstance(
                    calc_prof, (int, float, np.number)
                ):
                    diff = calc_prof - obj_val
                    f.write(
                        f"Objective vs Calculated Profit Diff: {diff:,.4f}\n")
                    if abs(diff) > 1.0:
                        f.write(
                            "WARNING: Significant difference between objective and calculated profit!\n"
                        )
                        logger.warning(
                            "Significant difference: objective vs calculated profit."
                        )
                else:
                    f.write("\nCould not compare Objective vs Calculated Profit.\n")
            logger.info(f"Summary results saved to {output_summary_path}")
            print(f"Summary results saved to {output_summary_path}")
        except Exception as e:
            logger.error(
                f"Failed to save summary results file: {e}", exc_info=True)
            print(
                f"Error: Failed to save summary results file to {output_summary_path}"
            )

    if not results_df.empty:
        results_df = results_df.round(4)
        output_csv_path = (
            results_dir /
            f"{target_iso_local}_Hourly_Results_Comprehensive.csv"
        )
        try:
            results_df.to_csv(output_csv_path)
            logger.info(
                f"Comprehensive hourly results saved to {output_csv_path}")
            print(f"Comprehensive hourly results saved to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save hourly results CSV: {e}")
            print(
                f"Error: Failed to save hourly results CSV to {output_csv_path}")
    else:
        logger.warning("Hourly results DataFrame is empty. Skipping CSV save.")

    if summary_results:
        print("\n--- Summary (also saved to file) ---")
        print(
            f"Simulation_Mode: {summary_results.get('Simulation_Mode', 'N/A')}")
        print(f"Target_ISO: {summary_results.get('Target_ISO', 'N/A')}")
        obj_val_console = summary_results.get("Objective_Value_USD", "N/A")
        calc_prof_console = summary_results.get(
            "Total_Profit_Calculated_USD", "N/A")
        print(
            f"Objective_Value_USD: {obj_val_console:,.4f}"
            if isinstance(obj_val_console, (int, float, np.number))
            else f"Objective_Value_USD: {obj_val_console}"
        )
        print(
            f"Total_Profit_Calculated_USD: {calc_prof_console:,.4f}"
            if isinstance(calc_prof_console, (int, float, np.number))
            else f"Total_Profit_Calculated_USD: {calc_prof_console}"
        )
        print(
            f"Total_H2_Produced_kg: {summary_results.get('Total_H2_Produced_kg', 0.0):,.2f}"
        )
        print("---------------------------------")

    return results_df, summary_results
