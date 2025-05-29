"""
Data loading functions for the TEA module.
"""

import logging
import pandas as pd
from pathlib import Path
# import numpy as np # Not directly used

# Attempt to import from tea.config, handling potential circular imports or module not found
try:
    from .config import (  # Changed to relative import
        HOURS_IN_YEAR,
        CAPEX_COMPONENTS,
        # Use alias to avoid direct modification
        PROJECT_LIFETIME_YEARS as DEFAULT_PROJECT_LIFETIME_YEARS,
        DISCOUNT_RATE as DEFAULT_DISCOUNT_RATE,
        CONSTRUCTION_YEARS as DEFAULT_CONSTRUCTION_YEARS,
        TAX_RATE as DEFAULT_TAX_RATE,
        OM_COMPONENTS as DEFAULT_OM_COMPONENTS,
        NUCLEAR_INTEGRATED_CONFIG as DEFAULT_NUCLEAR_INTEGRATED_CONFIG
    )
except ImportError:
    # Fallback values if tea.config is not available during initial setup or specific contexts
    # This is a safeguard, ideally tea.config should always be resolvable
    HOURS_IN_YEAR = 8760
    CAPEX_COMPONENTS = {}  # Define a default empty dict
    DEFAULT_PROJECT_LIFETIME_YEARS = 30
    DEFAULT_DISCOUNT_RATE = 0.08
    DEFAULT_CONSTRUCTION_YEARS = 2
    DEFAULT_TAX_RATE = 0.21
    DEFAULT_OM_COMPONENTS = {}
    DEFAULT_NUCLEAR_INTEGRATED_CONFIG = {}


logger = logging.getLogger(__name__)

# Helper function (can be module-level or stay nested if preferred)


def _get_param_value(params_dict, key, default_val, type_converter, param_logger):
    val = params_dict.get(key)
    if val is None or pd.isna(val):
        param_logger.info(
            f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_val}"
        )
        return default_val
    try:
        return type_converter(val)
    except (ValueError, TypeError):
        param_logger.warning(
            f"Invalid value '{val}' for '{key}' in sys_data. Using default: {default_val}"
        )
        return default_val


def load_tea_sys_params(iso_target: str, input_base_dir: Path) -> tuple[dict, int, float, int, float, dict, dict]:
    """
    Loads TEA-relevant system parameters.

    Returns a tuple containing:
    - params (dict): The raw parameters loaded from CSV.
    - project_lifetime (int): Project lifetime in years.
    - discount_rate (float): Discount rate as a fraction.
    - construction_years (int): Construction period in years.
    - tax_rate (float): Corporate tax rate as a fraction.
    - om_components (dict): O&M components dictionary, potentially updated.
    - nuclear_config (dict): Nuclear integrated config dictionary, potentially updated.
    """
    logger.debug(f"load_tea_sys_params called for ISO: {iso_target}")
    params = {}

    # Use aliased default values from config
    project_lifetime_years = DEFAULT_PROJECT_LIFETIME_YEARS
    discount_rate = DEFAULT_DISCOUNT_RATE
    construction_years = DEFAULT_CONSTRUCTION_YEARS
    tax_rate = DEFAULT_TAX_RATE
    # Deep copy mutable defaults to avoid modifying them globally if they are updated
    om_components = {k: v.copy() if isinstance(
        v, dict) else v for k, v in DEFAULT_OM_COMPONENTS.items()}
    nuclear_integrated_config = {k: v.copy() if isinstance(
        v, dict) else v for k, v in DEFAULT_NUCLEAR_INTEGRATED_CONFIG.items()}

    try:
        sys_data_file_path = input_base_dir / "hourly_data" / "sys_data_advanced.csv"
        if not sys_data_file_path.exists():
            sys_data_file_path = input_base_dir / "sys_data_advanced.csv"
        logger.debug(f"Attempting to load sys_data from: {sys_data_file_path}")

        if sys_data_file_path.exists():
            df_system = pd.read_csv(sys_data_file_path, index_col=0)
            param_keys = [
                "hydrogen_subsidy_value_usd_per_kg",
                "hydrogen_subsidy_duration_years",
                "user_specified_electrolyzer_capacity_MW",
                "user_specified_h2_storage_capacity_kg",
                "user_specified_battery_capacity_MWh",
                "user_specified_battery_power_MW",
                "plant_lifetime_years",
                "baseline_nuclear_annual_revenue_USD",
                "enable_incremental_analysis",
                "discount_rate_fraction",
                "project_construction_years",
                "corporate_tax_rate_fraction",
                "BatteryFixedOM_USD_per_MW_year",
                "BatteryFixedOM_USD_per_MWh_year",
                "enable_nuclear_integrated_analysis",
                "nuclear_plant_capacity_MW",
                "nuclear_project_lifetime_years",
                "nuclear_construction_years",
                "enable_nuclear_capex_costs",
                "enable_nuclear_opex_costs",
            ]
            for key in param_keys:
                if key in df_system.index:
                    value_series = df_system.loc[key, "Value"]
                    params[key] = (
                        value_series.iloc[0]
                        if isinstance(value_series, pd.Series)
                        else value_series
                    )
                else:
                    params[key] = None
            logger.info(
                f"Successfully loaded TEA relevant params from {sys_data_file_path}"
            )
        else:
            logger.warning(
                f"sys_data_advanced.csv not found at {sys_data_file_path}. TEA will use defaults for some parameters."
            )

    except Exception as e:
        logger.error(
            f"Error loading TEA system data from {sys_data_file_path}: {e}")
        logger.debug(f"Error in load_tea_sys_params: {e}", exc_info=True)

    project_lifetime_years = _get_param_value(
        params,
        "plant_lifetime_years",
        project_lifetime_years,  # Default from config
        lambda x: int(float(x)),
        logger,
    )
    discount_rate = _get_param_value(
        params, "discount_rate_fraction", discount_rate, float, logger  # Default from config
    )
    construction_years = _get_param_value(
        params,
        "project_construction_years",
        construction_years,  # Default from config
        lambda x: int(float(x)),
        logger,
    )
    tax_rate = _get_param_value(
        params, "corporate_tax_rate_fraction", tax_rate, float, logger  # Default from config
    )

    # Update Battery O&M from loaded params if OM_COMPONENTS is structured as expected
    if "Fixed_OM_Battery" in om_components:
        om_components["Fixed_OM_Battery"]["base_cost_per_mw_year"] = _get_param_value(
            params, "BatteryFixedOM_USD_per_MW_year",
            om_components["Fixed_OM_Battery"].get("base_cost_per_mw_year", 0),
            float, logger
        )
        om_components["Fixed_OM_Battery"]["base_cost_per_mwh_year"] = _get_param_value(
            params, "BatteryFixedOM_USD_per_MWh_year",
            om_components["Fixed_OM_Battery"].get("base_cost_per_mwh_year", 0),
            float, logger
        )
    else:
        logger.warning(
            "OM_COMPONENTS structure missing 'Fixed_OM_Battery'. Cannot update from params.")

    # Update Nuclear Integrated System Configuration from loaded params
    nuclear_integrated_config["enabled"] = _get_param_value(
        params, "enable_nuclear_integrated_analysis",
        nuclear_integrated_config.get("enabled", False), bool, logger
    )
    nuclear_integrated_config["nuclear_plant_capacity_mw"] = _get_param_value(
        params, "nuclear_plant_capacity_MW",
        nuclear_integrated_config.get(
            "nuclear_plant_capacity_mw", 1000), float, logger
    )
    nuclear_integrated_config["project_lifetime_years"] = _get_param_value(
        params, "nuclear_project_lifetime_years",
        nuclear_integrated_config.get(
            "project_lifetime_years", 60), lambda x: int(float(x)), logger
    )
    nuclear_integrated_config["construction_years"] = _get_param_value(
        params, "nuclear_construction_years",
        nuclear_integrated_config.get(
            "construction_years", 8), lambda x: int(float(x)), logger
    )
    nuclear_integrated_config["enable_nuclear_capex"] = _get_param_value(
        params, "enable_nuclear_capex_costs",
        nuclear_integrated_config.get(
            "enable_nuclear_capex", False), bool, logger
    )
    nuclear_integrated_config["enable_nuclear_opex"] = _get_param_value(
        params, "enable_nuclear_opex_costs",
        nuclear_integrated_config.get(
            "enable_nuclear_opex", False), bool, logger
    )

    logger.debug(
        f"load_tea_sys_params finished. Project Lifetime: {project_lifetime_years}, Discount Rate: {discount_rate}"
    )
    logger.debug(f"Nuclear Integrated Config: {nuclear_integrated_config}")

    # Return both the raw params and the potentially updated config dictionaries/values
    return params, project_lifetime_years, discount_rate, construction_years, tax_rate, om_components, nuclear_integrated_config


def load_hourly_results(filepath: Path) -> pd.DataFrame | None:
    """Loads and validates hourly results from the optimization run."""
    logger.info(f"Loading hourly results from: {filepath}")
    if not filepath.exists():
        logger.error(f"Results file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)

        base_required_cols = [
            "Profit_Hourly_USD",
            "Revenue_Total_USD",
            "Cost_HourlyOpex_Total_USD",
            "mHydrogenProduced_kg_hr",
            "pElectrolyzer_MW",
            "pTurbine_MW",
            "EnergyPrice_LMP_USDperMWh",
        ]

        degradation_cols = [
            col for col in df.columns if "degradation" in col.lower()]
        if degradation_cols:
            logger.debug(f"Found degradation columns: {degradation_cols}")
        else:
            logger.debug("No degradation columns found in results file.")

        missing_base_cols = [
            col for col in base_required_cols if col not in df.columns]
        if missing_base_cols:
            logger.error(
                f"Missing essential base columns in results file: {missing_base_cols}"
            )
            return None

        capacity_cols_needed_for_capex = set()
        if CAPEX_COMPONENTS:  # Check if CAPEX_COMPONENTS is not empty
            for comp_details in CAPEX_COMPONENTS.values():
                cap_key = comp_details.get("applies_to_component_capacity_key")
                if cap_key:
                    capacity_cols_needed_for_capex.add(cap_key)
        else:
            logger.warning(
                "CAPEX_COMPONENTS is empty. Cannot determine capacity columns needed for CAPEX.")

        logger.debug(
            f"Capacity columns needed for CAPEX: {capacity_cols_needed_for_capex}"
        )

        capacity_related_cols = [
            col
            for col in df.columns
            if any(term in col.lower() for term in ["capacity", "mw", "mwh", "kg"])
        ]
        logger.debug(
            f"Available capacity-related columns in results file: {capacity_related_cols}"
        )

        if "Electrolyzer_Capacity_MW" in df.columns:
            unique_vals = df["Electrolyzer_Capacity_MW"].unique()
            logger.debug(
                f"Electrolyzer_Capacity_MW unique values: {unique_vals}")
        else:
            logger.warning(
                "Electrolyzer_Capacity_MW not found in results file!")
            potential_electrolyzer_cols = [
                col
                for col in df.columns
                if "electrolyzer" in col.lower() and "capacity" in col.lower()
            ]
            if potential_electrolyzer_cols:
                logger.debug(
                    f"Found potential electrolyzer capacity columns: {potential_electrolyzer_cols}"
                )
                df["Electrolyzer_Capacity_MW"] = df[potential_electrolyzer_cols[0]]
                logger.info(
                    f"Using {potential_electrolyzer_cols[0]} as fallback for Electrolyzer_Capacity_MW"
                )
            else:
                logger.warning(
                    "No alternative electrolyzer capacity columns found!")

        for cap_col_key in capacity_cols_needed_for_capex:
            if cap_col_key not in df.columns:
                logger.warning(
                    f"Capacity column '{cap_col_key}' (needed for CAPEX learning rate/scaling) "
                    f"is missing from results file '{filepath}'. "
                    f"Assuming 0 capacity for this component in this run."
                )
                df[cap_col_key] = 0.0

        all_required_cols = base_required_cols + \
            list(capacity_cols_needed_for_capex)
        all_required_cols = sorted(list(set(all_required_cols)))

        missing_final_cols = [
            col for col in all_required_cols if col not in df.columns]
        if missing_final_cols:
            logger.error(
                f"Still missing columns after attempting to add defaults: {missing_final_cols}"
            )
            return None

        return df
    except Exception as e:
        logger.error(
            f"Error loading or processing results file {filepath}: {e}",
            exc_info=True,
        )
        return None
