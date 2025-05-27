# runs/tea.py
"""
Technical Economic Analysis (TEA) script for the nuclear-hydrogen optimization framework.
This script performs comprehensive lifecycle analysis including:
- Capital and operational costs (with learning rate adjustments for CAPEX)
- Revenue streams from multiple sources
- Financial metrics (NPV, IRR, LCOH, etc.)
- Sensitivity analysis
- Visualization of results
"""

import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
import seaborn as sns

# Configure logging, redirect logs to file
SCRIPT_DIR_PATH = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR_PATH.parent / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Add src directory to Python path
SRC_PATH = SCRIPT_DIR_PATH.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# Framework imports
logger = None
TARGET_ISO = "DEFAULT_ISO_FALLBACK"
HOURS_IN_YEAR = 8760

try:
    from logging_setup import logger

    logger.info("Using framework logger configuration")

    # Add file handler, keep existing console handler
    log_file_path = LOG_DIR / f"tea_{TARGET_ISO}.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    from config import ENABLE_BATTERY, HOURS_IN_YEAR, TARGET_ISO
    from data_io import load_hourly_data
    from utils import get_param

except ImportError as e_import:
    if logger is None:
        # Set up log file path using default TARGET_ISO
        log_file_path = LOG_DIR / f"tea_{TARGET_ISO}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - TEA_FALLBACK - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file_path),
            ],
        )
        logger = logging.getLogger(__name__)
    logger.error(
        f"Failed to import from optimization framework (ImportError): {e_import}. TEA script might not function correctly."
    )
    TARGET_ISO = "DEFAULT_ISO_IMPORT_ERROR"
    HOURS_IN_YEAR = 8760
    ENABLE_BATTERY = False

except Exception as e_other:
    if logger is None:
        # Set up log file path using default TARGET_ISO
        log_file_path = LOG_DIR / f"tea_{TARGET_ISO}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - TEA_FALLBACK - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file_path),
            ],
        )
        logger = logging.getLogger(__name__)
    logger.error(
        f"A non-ImportError occurred during framework imports: {e_other}. TEA script might not function correctly.",
        exc_info=True,
    )
    TARGET_ISO = "DEFAULT_ISO_OTHER_ERROR"
    HOURS_IN_YEAR = 8760
    ENABLE_BATTERY = False

# Log configuration information to log file
logger.debug(
    f"Framework imports section finished. TARGET_ISO set to: {TARGET_ISO}, ENABLE_BATTERY: {ENABLE_BATTERY}"
)

# TEA Configuration
BASE_OUTPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent / "TEA_results"
BASE_INPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent / "input"

# TEA Parameters
PROJECT_LIFETIME_YEARS = 30
DISCOUNT_RATE = 0.08
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21

# CAPEX Components (with learning rate structure)
CAPEX_COMPONENTS = {
    "Electrolyzer_System": {
        "total_base_cost_for_ref_size": 100_000_000,  # 50MW * 1000 * $2000
        "reference_total_capacity_mw": 50,
        "applies_to_component_capacity_key": "Electrolyzer_Capacity_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "H2_Storage_System": {
        "total_base_cost_for_ref_size": 10_000_000,  # 10,000kg * $1000
        "reference_total_capacity_mw": 10000,  # Assuming kg
        "applies_to_component_capacity_key": "H2_Storage_Capacity_kg",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "Battery_System_Energy": {  # Cost component for MWh capacity
        "total_base_cost_for_ref_size": 23_600_000,  # 100MWh * 1000 * $236
        "reference_total_capacity_mw": 100,
        "applies_to_component_capacity_key": "Battery_Capacity_MWh",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-1: 1.0},
    },
    "Battery_System_Power": {  # Cost component for MW power
        "total_base_cost_for_ref_size": 5_000_000,
        "reference_total_capacity_mw": 25,  # Here unit is MW
        "applies_to_component_capacity_key": "Battery_Power_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-1: 1.0},
    },
    "Grid_Integration": {
        "total_base_cost_for_ref_size": 5_000_000,
        "reference_total_capacity_mw": 0,
        "applies_to_component_capacity_key": None,
        "learning_rate_decimal": 0,
        "payment_schedule_years": {-1: 1.0},
    },
    "NPP_Modifications": {
        "total_base_cost_for_ref_size": 2_000_000,
        "reference_total_capacity_mw": 0,
        "applies_to_component_capacity_key": None,
        "learning_rate_decimal": 0,
        "payment_schedule_years": {-2: 1.0},
    },
}

# O&M Components
OM_COMPONENTS = {
    "Fixed_OM_General": {
        "base_cost_percent_of_capex": 0.02,
        "size_dependent": True,
        "inflation_rate": 0.02,
    },
    "Fixed_OM_Battery": {
        "base_cost_per_mw_year": 25_000,
        "base_cost_per_mwh_year": 0,
        "inflation_rate": 0.02,
    },
}

# Replacement Schedule
REPLACEMENT_SCHEDULE = {
    "Electrolyzer_Stack": {
        "cost_percent_initial_capex": 0.30,
        "years": [10, 20],
        "size_dependent": True,
    },
    "H2_Storage_Components": {
        "cost": 5_000_000,
        "years": [15],
        "size_dependent": True,
    },
    "Battery_Augmentation_Replacement": {
        "cost_percent_initial_capex": 0.60,
        "years": [10],
        "size_dependent": True,
    },
}

# Nuclear Power Plant Integrated System Configuration
# For comprehensive nuclear + hydrogen analysis (60-year lifecycle)
NUCLEAR_INTEGRATED_CONFIG = {
    "enabled": True,  # Default enabled for greenfield nuclear-hydrogen analysis
    "project_lifetime_years": 60,  # Nuclear plant lifetime
    "construction_years": 8,  # Nuclear construction period
    # Default nuclear plant capacity (will be overridden by actual reactor size)
    "nuclear_plant_capacity_mw": 1000,
    # Include nuclear construction costs in greenfield analysis
    "enable_nuclear_capex": True,
    "enable_nuclear_opex": True,  # Include nuclear O&M costs in greenfield analysis
}

# Nuclear Plant CAPEX Components (separate from existing components)
NUCLEAR_CAPEX_COMPONENTS = {
    "Nuclear_Power_Plant": {
        # $10B for 1000MW plant (more realistic)
        "total_base_cost_for_ref_size": 10_000_000_000,
        "reference_total_capacity_mw": 1000,  # Reference capacity in MW
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.05,  # 5% learning rate for nuclear construction
        # 8-year construction (year 0-7)
        "payment_schedule_years": {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.05},
    },
    "Nuclear_Site_Preparation": {
        "total_base_cost_for_ref_size": 300_000_000,  # Reduced site preparation costs
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.02,
        "payment_schedule_years": {0: 0.8, 1: 0.2},  # Front-loaded
    },
    "Nuclear_Safety_Systems": {
        "total_base_cost_for_ref_size": 1_500_000_000,  # Reduced safety systems cost
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.03,
        "payment_schedule_years": {3: 0.3, 4: 0.4, 5: 0.3},
    },
    "Nuclear_Grid_Connection": {
        "total_base_cost_for_ref_size": 200_000_000,  # Reduced grid connection cost
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {6: 0.6, 7: 0.4},
    },
}

# Nuclear Plant O&M Components (separate from existing components)
NUCLEAR_OM_COMPONENTS = {
    "Nuclear_Fixed_OM": {
        "base_cost_per_mw_year": 120_000,  # $120k/MW/year for nuclear O&M
        "inflation_rate": 0.025,  # Slightly higher inflation for nuclear O&M
    },
    "Nuclear_Fuel_Cost": {
        "base_cost_per_mwh": 8.0,  # $8/MWh fuel cost (from PDF data)
        "inflation_rate": 0.02,
    },
    "Nuclear_Security": {
        "base_cost_per_mw_year": 15_000,  # $15k/MW/year for security
        "inflation_rate": 0.03,
    },
    "Nuclear_Regulatory": {
        "base_cost_per_mw_year": 8_000,  # $8k/MW/year for regulatory compliance
        "inflation_rate": 0.025,
    },
    "Nuclear_Waste_Management": {
        "base_cost_per_mwh": 1.0,  # $1/MWh for waste management
        "inflation_rate": 0.02,
    },
}

# Nuclear Plant Major Refurbishments/Replacements
NUCLEAR_REPLACEMENT_SCHEDULE = {
    "Steam_Generator_Replacement": {
        "cost_percent_initial_capex": 0.08,  # 8% of initial nuclear CAPEX
        "years": [25],  # Mid-life steam generator replacement
        "size_dependent": True,
    },
    "Reactor_Pressure_Vessel_Head": {
        "cost_percent_initial_capex": 0.03,  # 3% of initial nuclear CAPEX
        "years": [20, 40],  # Periodic replacements
        "size_dependent": True,
    },
    "Major_Maintenance_Outage": {
        "cost_percent_initial_capex": 0.02,  # 2% of initial nuclear CAPEX
        "years": [15, 30, 45],  # Major outages every 15 years
        "size_dependent": True,
    },
    "Control_Rod_Drive_Mechanisms": {
        "cost_percent_initial_capex": 0.015,  # 1.5% of initial nuclear CAPEX
        "years": [18, 36, 54],  # Replacement every 18 years
        "size_dependent": True,
    },
}

logger.debug("Global configurations and constants defined.")


def load_tea_sys_params(iso_target: str, input_base_dir: Path) -> dict:
    """Loads TEA-relevant system parameters."""
    logger.debug(f"load_tea_sys_params called for ISO: {iso_target}")
    params = {}
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
                "user_specified_battery_capacity_MWh",  # Added for battery
                "user_specified_battery_power_MW",  # Added for battery
                "plant_lifetime_years",
                "baseline_nuclear_annual_revenue_USD",
                "enable_incremental_analysis",
                "discount_rate_fraction",
                "project_construction_years",
                "corporate_tax_rate_fraction",
                "BatteryFixedOM_USD_per_MW_year",  # Added for battery O&M
                "BatteryFixedOM_USD_per_MWh_year",  # Added for battery O&M
                # Nuclear integrated system parameters
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

    global PROJECT_LIFETIME_YEARS, DISCOUNT_RATE, CONSTRUCTION_YEARS, TAX_RATE, OM_COMPONENTS

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

    PROJECT_LIFETIME_YEARS = _get_param_value(
        params,
        "plant_lifetime_years",
        PROJECT_LIFETIME_YEARS,
        lambda x: int(float(x)),
        logger,
    )
    DISCOUNT_RATE = _get_param_value(
        params, "discount_rate_fraction", DISCOUNT_RATE, float, logger
    )
    CONSTRUCTION_YEARS = _get_param_value(
        params,
        "project_construction_years",
        CONSTRUCTION_YEARS,
        lambda x: int(float(x)),
        logger,
    )
    TAX_RATE = _get_param_value(
        params, "corporate_tax_rate_fraction", TAX_RATE, float, logger
    )

    # Update Battery O&M from loaded params
    OM_COMPONENTS["Fixed_OM_Battery"]["base_cost_per_mw_year"] = _get_param_value(
        params, "BatteryFixedOM_USD_per_MW_year", 0, float, logger
    )
    OM_COMPONENTS["Fixed_OM_Battery"]["base_cost_per_mwh_year"] = _get_param_value(
        params, "BatteryFixedOM_USD_per_MWh_year", 0, float, logger
    )

    # Update Nuclear Integrated System Configuration from loaded params
    global NUCLEAR_INTEGRATED_CONFIG
    NUCLEAR_INTEGRATED_CONFIG["enabled"] = _get_param_value(
        params, "enable_nuclear_integrated_analysis", False, bool, logger
    )
    NUCLEAR_INTEGRATED_CONFIG["nuclear_plant_capacity_mw"] = _get_param_value(
        params, "nuclear_plant_capacity_MW", 1000, float, logger
    )
    NUCLEAR_INTEGRATED_CONFIG["project_lifetime_years"] = _get_param_value(
        params, "nuclear_project_lifetime_years", 60, lambda x: int(
            float(x)), logger
    )
    NUCLEAR_INTEGRATED_CONFIG["construction_years"] = _get_param_value(
        params, "nuclear_construction_years", 8, lambda x: int(
            float(x)), logger
    )
    NUCLEAR_INTEGRATED_CONFIG["enable_nuclear_capex"] = _get_param_value(
        params, "enable_nuclear_capex_costs", False, bool, logger
    )
    NUCLEAR_INTEGRATED_CONFIG["enable_nuclear_opex"] = _get_param_value(
        params, "enable_nuclear_opex_costs", False, bool, logger
    )

    logger.debug(
        f"load_tea_sys_params finished. Project Lifetime: {PROJECT_LIFETIME_YEARS}, Discount Rate: {DISCOUNT_RATE}"
    )
    logger.debug(f"Nuclear Integrated Config: {NUCLEAR_INTEGRATED_CONFIG}")
    return params


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

        # Check if there is electrolyzer degradation data
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
        for comp_details in CAPEX_COMPONENTS.values():
            cap_key = comp_details.get("applies_to_component_capacity_key")
            if cap_key:
                capacity_cols_needed_for_capex.add(cap_key)

        # Record capacity-related column information to log file
        logger.debug(
            f"Capacity columns needed for CAPEX: {capacity_cols_needed_for_capex}"
        )

        # Record available capacity-related columns to log file
        capacity_related_cols = [
            col
            for col in df.columns
            if any(term in col.lower() for term in ["capacity", "mw", "mwh", "kg"])
        ]
        logger.debug(
            f"Available capacity-related columns in results file: {capacity_related_cols}"
        )

        # Check for electrolyzer capacity specifically
        if "Electrolyzer_Capacity_MW" in df.columns:
            unique_vals = df["Electrolyzer_Capacity_MW"].unique()
            logger.debug(
                f"Electrolyzer_Capacity_MW unique values: {unique_vals}")
        else:
            logger.warning(
                "Electrolyzer_Capacity_MW not found in results file!")

            # Try to find alternative column names that might contain electrolyzer capacity
            potential_electrolyzer_cols = [
                col
                for col in df.columns
                if "electrolyzer" in col.lower() and "capacity" in col.lower()
            ]
            if potential_electrolyzer_cols:
                logger.debug(
                    f"Found potential electrolyzer capacity columns: {potential_electrolyzer_cols}"
                )

                # Use the first potential column as fallback
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


def calculate_annual_metrics(df: pd.DataFrame, tea_sys_params: dict) -> dict | None:
    """Calculates comprehensive annual metrics from hourly results."""
    if df is None:
        return None
    metrics = {}
    try:
        num_hours = len(df)
        if num_hours == 0:
            logger.error("Hourly results DataFrame is empty.")
            return None
        annualization_factor = (
            HOURS_IN_YEAR / num_hours if num_hours > 0 and HOURS_IN_YEAR > 0 else 1.0
        )

        metrics["Annual_Profit"] = df["Profit_Hourly_USD"].sum()
        metrics["Annual_Revenue"] = df["Revenue_Total_USD"].sum()
        metrics["Annual_Opex_Cost_from_Opt"] = df["Cost_HourlyOpex_Total_USD"].sum()
        metrics["Energy_Revenue"] = df.get(
            "Revenue_Energy_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["AS_Revenue"] = df.get(
            "Revenue_Ancillary_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Sales_Revenue"] = df.get(
            "Revenue_Hydrogen_Sales_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Subsidy_Revenue"] = df.get(
            "Revenue_Hydrogen_Subsidy_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Total_Revenue"] = (
            metrics["H2_Sales_Revenue"] + metrics["H2_Subsidy_Revenue"]
        )
        metrics["VOM_Turbine_Cost"] = df.get(
            "Cost_VOM_Turbine_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["VOM_Electrolyzer_Cost"] = df.get(
            "Cost_VOM_Electrolyzer_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["VOM_Battery_Cost"] = df.get(
            "Cost_VOM_Battery_USD", pd.Series(0.0, dtype="float64")
        ).sum()  # From optimization results
        metrics["Startup_Cost"] = df.get(
            "Cost_Startup_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["Water_Cost"] = df.get(
            "Cost_Water_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["Ramping_Cost"] = df.get(
            "Cost_Ramping_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Storage_Cycle_Cost"] = df.get(
            "Cost_Storage_Cycle_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Production_kg_annual"] = (
            df["mHydrogenProduced_kg_hr"].sum() * annualization_factor
        )

        # Extract degradation information
        degradation_cols = [
            col for col in df.columns if "degradation" in col.lower()]
        if degradation_cols:
            for col in degradation_cols:
                # Add degradation information to metrics
                metrics[f"{col}_avg"] = df[col].mean()
                metrics[f"{col}_end"] = df[col].iloc[-1] if not df[col].empty else 0
                logger.debug(
                    f"Added degradation metric - {col}_avg: {metrics[f'{col}_avg']}"
                )
                logger.debug(
                    f"Added degradation metric - {col}_end: {metrics[f'{col}_end']}"
                )

        # Check for user-specified capacity values in tea_sys_params first
        user_spec_electrolyzer_cap = None
        if (
            "user_specified_electrolyzer_capacity_MW" in tea_sys_params
            and tea_sys_params["user_specified_electrolyzer_capacity_MW"] is not None
        ):
            try:
                user_spec_electrolyzer_cap = float(
                    tea_sys_params["user_specified_electrolyzer_capacity_MW"]
                )
                if user_spec_electrolyzer_cap > 0:
                    logger.debug(
                        f"Found user-specified electrolyzer capacity: {user_spec_electrolyzer_cap} MW"
                    )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid user-specified electrolyzer capacity value: {tea_sys_params['user_specified_electrolyzer_capacity_MW']}"
                )

        # Try to get electrolyzer capacity from results file first
        if (
            "Electrolyzer_Capacity_MW" in df.columns
            and not df["Electrolyzer_Capacity_MW"].empty
        ):
            opt_electrolyzer_cap = df["Electrolyzer_Capacity_MW"].iloc[0]
            logger.debug(
                f"Found electrolyzer capacity in results file: {opt_electrolyzer_cap} MW"
            )

            # Compare with user-specified value if available
            if (
                user_spec_electrolyzer_cap is not None
                and user_spec_electrolyzer_cap > 0
            ):
                # More than 1 MW difference
                if abs(opt_electrolyzer_cap - user_spec_electrolyzer_cap) > 1:
                    logger.warning(
                        f"Warning - electrolyzer capacity from results ({opt_electrolyzer_cap} MW) differs significantly from user-specified value ({user_spec_electrolyzer_cap} MW)"
                    )
                    # Use the user-specified value if it's different from optimization result
                    metrics["Electrolyzer_Capacity_MW"] = user_spec_electrolyzer_cap
                    logger.debug(
                        f"Using user-specified electrolyzer capacity: {user_spec_electrolyzer_cap} MW"
                    )
                else:
                    metrics["Electrolyzer_Capacity_MW"] = opt_electrolyzer_cap
            else:
                metrics["Electrolyzer_Capacity_MW"] = opt_electrolyzer_cap
        elif user_spec_electrolyzer_cap is not None and user_spec_electrolyzer_cap > 0:
            # Use user-specified value as fallback if available
            metrics["Electrolyzer_Capacity_MW"] = user_spec_electrolyzer_cap
            logger.debug(
                f"Using user-specified electrolyzer capacity as fallback: {user_spec_electrolyzer_cap} MW"
            )
        else:
            # Last resort: use default value
            default_cap = 0
            metrics["Electrolyzer_Capacity_MW"] = default_cap
            logger.debug(
                f"No electrolyzer capacity found in results or user parameters. Using default: {default_cap} MW"
            )

        # Log final capacity value used
        logger.debug(
            f"Final electrolyzer capacity used for calculations: {metrics['Electrolyzer_Capacity_MW']} MW"
        )

        # Similar logic for H2 storage capacity
        if (
            "H2_Storage_Capacity_kg" in df.columns
        ):  # This column is added by load_hourly_results if needed
            metrics["H2_Storage_Capacity_kg"] = (
                df["H2_Storage_Capacity_kg"].iloc[0]
                if not df["H2_Storage_Capacity_kg"].empty
                else 0
            )
            logger.debug(
                f"H2 storage capacity from results: {metrics['H2_Storage_Capacity_kg']} kg"
            )
        else:  # Should not happen if load_hourly_results works as intended
            user_spec_h2_storage = tea_sys_params.get(
                "user_specified_h2_storage_capacity_kg"
            )
            if user_spec_h2_storage is not None and not pd.isna(user_spec_h2_storage):
                try:
                    metrics["H2_Storage_Capacity_kg"] = float(
                        user_spec_h2_storage)
                    logger.debug(
                        f"Using user-specified H2 storage capacity: {metrics['H2_Storage_Capacity_kg']} kg"
                    )
                except (ValueError, TypeError):
                    metrics["H2_Storage_Capacity_kg"] = 0
                    logger.debug(
                        f"Invalid user-specified H2 storage value: {user_spec_h2_storage}. Using 0 kg"
                    )
            else:
                metrics["H2_Storage_Capacity_kg"] = 0
                logger.debug(
                    "H2_Storage_Capacity_kg column unexpectedly missing in calculate_annual_metrics and no user value available."
                )

        # H2 Constant Sales Rate (for optimal storage sizing mode)
        if "H2_Constant_Sales_Rate_kg_hr" in df.columns:
            metrics["H2_Constant_Sales_Rate_kg_hr"] = (
                df["H2_Constant_Sales_Rate_kg_hr"].iloc[0]
                if not df["H2_Constant_Sales_Rate_kg_hr"].empty
                else 0
            )
            logger.debug(
                f"H2 constant sales rate from results: {metrics['H2_Constant_Sales_Rate_kg_hr']} kg/hr"
            )
        else:
            metrics["H2_Constant_Sales_Rate_kg_hr"] = 0
            logger.debug("H2 constant sales rate not found in results")

        # If we have summary results with optimal values, prefer those
        for summary_col in df.columns:
            if "Optimal_H2_Constant_Sales_Rate_kg_hr" in summary_col:
                optimal_rate = df[summary_col].iloc[0] if not df[summary_col].empty else 0
                if optimal_rate > 0:
                    metrics["Optimal_H2_Constant_Sales_Rate_kg_hr"] = optimal_rate
                    logger.debug(
                        f"Found optimal H2 constant sales rate: {optimal_rate} kg/hr")

        # Battery Capacity and Power (from results DataFrame, ensured by load_hourly_results)
        metrics["Battery_Capacity_MWh"] = (
            df["Battery_Capacity_MWh"].iloc[0]
            if "Battery_Capacity_MWh" in df and not df["Battery_Capacity_MWh"].empty
            else 0
        )
        metrics["Battery_Power_MW"] = (
            df["Battery_Power_MW"].iloc[0]
            if "Battery_Power_MW" in df and not df["Battery_Power_MW"].empty
            else 0
        )
        logger.debug(
            f"Battery capacity: {metrics['Battery_Capacity_MWh']} MWh, power: {metrics['Battery_Power_MW']} MW"
        )

        # Try to find Battery SOC column by checking various possible names
        battery_soc_col = None
        possible_battery_soc_cols = [
            "BatterySOC_MWh",
            "Battery_SOC_MWh",
            "BatterySOC",
            "Battery_SOC",
            "SOC_Battery_MWh",
        ]

        # Debug available columns
        battery_cols = [
            col
            for col in df.columns
            if "battery" in col.lower() or "soc" in col.lower()
        ]
        logger.debug(f"Found battery-related columns: {battery_cols}")

        for col_name in possible_battery_soc_cols:
            if col_name in df.columns:
                battery_soc_col = col_name
                logger.debug(f"Found battery SOC column: {col_name}")
                break

        if battery_soc_col is not None and metrics["Battery_Capacity_MWh"] > 1e-6:
            # Calculate average Battery SOC as percentage of total capacity
            metrics["Battery_SOC_percent"] = (
                df[battery_soc_col].mean() / metrics["Battery_Capacity_MWh"]
            ) * 100
            logger.debug(
                f"Battery average SOC calculated: {metrics['Battery_SOC_percent']}% from column {battery_soc_col}"
            )
        else:
            metrics["Battery_SOC_percent"] = 0
            logger.debug(
                "Battery SOC set to 0 (capacity is zero or SOC data missing)")

        # Calculate H2 Storage SOC (similar approach as battery)
        h2_storage_soc_col = None
        possible_h2_soc_cols = [
            "H2_Storage_Level_kg",  # This is the actual column name in results
            "H2_Storage_SOC_kg",
            "H2StorageSOC_kg",
            "H2StorageSOC",
            "H2_Storage_SOC",
            "mH2Storage_kg",
        ]

        # Debug available H2 storage columns
        h2_storage_cols = [
            col
            for col in df.columns
            if "h2" in col.lower()
            and ("storage" in col.lower() or "inventory" in col.lower())
        ]
        logger.debug(f"Found H2 storage-related columns: {h2_storage_cols}")

        for col_name in possible_h2_soc_cols:
            if col_name in df.columns:
                h2_storage_soc_col = col_name
                logger.debug(f"Found H2 storage SOC column: {col_name}")
                break

        if h2_storage_soc_col is not None and metrics["H2_Storage_Capacity_kg"] > 1e-6:
            # Calculate average H2 Storage SOC as percentage of total capacity
            metrics["H2_Storage_SOC_percent"] = (
                df[h2_storage_soc_col].mean() / metrics["H2_Storage_Capacity_kg"]
            ) * 100
            logger.debug(
                f"H2 Storage average SOC calculated: {metrics['H2_Storage_SOC_percent']}% from column {h2_storage_soc_col}"
            )
        else:
            metrics["H2_Storage_SOC_percent"] = 0
            logger.debug(
                "H2 Storage SOC set to 0 (capacity is zero or SOC data missing)"
            )

        # Keep old Battery CF calculation for backward compatibility
        if (
            metrics["Battery_Power_MW"] > 1e-6
            and "BatteryCharge_MW" in df
            and "BatteryDischarge_MW" in df
        ):
            avg_batt_usage = (
                df["BatteryCharge_MW"].mean() + df["BatteryDischarge_MW"].mean()
            ) / 2
            metrics["Battery_CF_percent"] = (
                avg_batt_usage / metrics["Battery_Power_MW"]
            ) * 100
        else:
            metrics["Battery_CF_percent"] = 0

        # **NEW: Calculate annual battery charging electricity consumption**
        # This is critical for TEA opportunity cost calculations
        # Handle different possible column names for battery charging data
        battery_charge_col = None
        possible_battery_charge_cols = [
            "BatteryCharge_MW", "Battery_Charge_MW", "BatteryCharge"]
        for col_name in possible_battery_charge_cols:
            if col_name in df.columns:
                battery_charge_col = col_name
                break

        if battery_charge_col is not None and len(df) > 0:
            # Annual battery charging electricity (MWh)
            battery_charge_mwh_annual = df[battery_charge_col].sum(
            ) * annualization_factor
            metrics["Annual_Battery_Charge_MWh"] = battery_charge_mwh_annual

            # Distinguish charging source based on grid interaction
            # When pGridPurchase > 0, battery is likely charging from grid
            # When pGridPurchase == 0 and battery is charging, it's from NPP excess
            grid_purchase_col = None
            possible_grid_purchase_cols = [
                "pGridPurchase_MW", "pGridPurchase", "GridPurchase_MW"]
            for col_name in possible_grid_purchase_cols:
                if col_name in df.columns:
                    grid_purchase_col = col_name
                    break

            if grid_purchase_col is not None:
                # Battery charging from grid (when grid purchase > 0)
                battery_grid_charge_mask = (df[battery_charge_col] > 0) & (
                    df[grid_purchase_col] > 0)
                battery_charge_from_grid_mwh = df.loc[battery_grid_charge_mask, battery_charge_col].sum(
                ) * annualization_factor

                # Battery charging from NPP (when no grid purchase and battery charging)
                battery_npp_charge_mask = (df[battery_charge_col] > 0) & (
                    df[grid_purchase_col] <= 1e-6)
                battery_charge_from_npp_mwh = df.loc[battery_npp_charge_mask, battery_charge_col].sum(
                ) * annualization_factor

                metrics["Annual_Battery_Charge_From_Grid_MWh"] = battery_charge_from_grid_mwh
                metrics["Annual_Battery_Charge_From_NPP_MWh"] = battery_charge_from_npp_mwh

                logger.debug(f"Battery charging breakdown:")
                logger.debug(
                    f"  Total: {battery_charge_mwh_annual:.2f} MWh/year")
                logger.debug(
                    f"  From Grid: {battery_charge_from_grid_mwh:.2f} MWh/year")
                logger.debug(
                    f"  From NPP: {battery_charge_from_npp_mwh:.2f} MWh/year")
            else:
                # Fallback: assume all charging is from NPP if no grid purchase data
                metrics["Annual_Battery_Charge_From_Grid_MWh"] = 0.0
                metrics["Annual_Battery_Charge_From_NPP_MWh"] = battery_charge_mwh_annual
                logger.warning(
                    "No pGridPurchase_MW data found. Assuming all battery charging is from NPP.")
        else:
            # No battery charging data
            metrics["Annual_Battery_Charge_MWh"] = 0.0
            metrics["Annual_Battery_Charge_From_Grid_MWh"] = 0.0
            metrics["Annual_Battery_Charge_From_NPP_MWh"] = 0.0

        # Try to get Turbine_Capacity_MW from results file first
        metrics["Turbine_Capacity_MW"] = (
            df.get("Turbine_Capacity_MW", pd.Series(
                0.0, dtype="float64")).iloc[0]
            if "Turbine_Capacity_MW" in df and not df["Turbine_Capacity_MW"].empty
            else 0
        )

        # If Turbine_Capacity_MW is 0 or very small, try to get it from tea_sys_params (pTurbine_max_MW)
        if metrics["Turbine_Capacity_MW"] <= 1e-6:
            if (
                "pTurbine_max_MW" in tea_sys_params
                and tea_sys_params["pTurbine_max_MW"] is not None
            ):
                try:
                    user_spec_turbine_cap = float(
                        tea_sys_params["pTurbine_max_MW"])
                    if user_spec_turbine_cap > 0:
                        metrics["Turbine_Capacity_MW"] = user_spec_turbine_cap
                        logger.debug(
                            f"Using pTurbine_max_MW from sys_params as Turbine capacity: {user_spec_turbine_cap} MW"
                        )
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid pTurbine_max_MW value: {tea_sys_params['pTurbine_max_MW']}"
                    )

        # Calculate Turbine CF - improved to ensure we use data correctly
        if "pTurbine_MW" in df.columns:
            # Debug turbine power data
            logger.debug(
                f"pTurbine_MW data available - Min: {df['pTurbine_MW'].min()}, Max: {df['pTurbine_MW'].max()}, Mean: {df['pTurbine_MW'].mean()}"
            )

            if metrics["Turbine_Capacity_MW"] <= 1e-6:
                # If we still don't have a capacity value, try to use the max observed value from the data
                if df["pTurbine_MW"].max() > 0:
                    metrics["Turbine_Capacity_MW"] = df["pTurbine_MW"].max()
                    logger.debug(
                        f"Using maximum observed pTurbine_MW as capacity: {metrics['Turbine_Capacity_MW']} MW"
                    )
                # If max is still zero, try getting pTurbine_max_MW from tea_sys_params directly
                elif (
                    "pTurbine_max_MW" in tea_sys_params
                    and tea_sys_params["pTurbine_max_MW"] is not None
                ):
                    try:
                        turbine_max = float(tea_sys_params["pTurbine_max_MW"])
                        if turbine_max > 0:
                            metrics["Turbine_Capacity_MW"] = turbine_max
                            logger.debug(
                                f"Forced setting of Turbine capacity to pTurbine_max_MW: {turbine_max} MW"
                            )
                    except (ValueError, TypeError):
                        pass

            # Now calculate the CF with the best capacity value we have
            if metrics["Turbine_Capacity_MW"] > 1e-6:
                metrics["Turbine_CF_percent"] = (
                    df["pTurbine_MW"].mean() / metrics["Turbine_Capacity_MW"]
                ) * 100
                logger.debug(
                    f"Turbine CF calculated: {metrics['Turbine_CF_percent']}% (Capacity: {metrics['Turbine_Capacity_MW']} MW)")
            else:
                metrics["Turbine_CF_percent"] = 0
                logger.debug(
                    "Turbine CF set to 0 (valid capacity value couldn't be determined)"
                )
        else:
            metrics["Turbine_CF_percent"] = 0
            logger.debug(
                "Turbine CF set to 0 (pTurbine_MW column not found in data)")
        metrics["Annual_Electrolyzer_MWh"] = (
            df["pElectrolyzer_MW"].sum() * annualization_factor
            if "pElectrolyzer_MW" in df
            else 0
        )

        # Calculate Electrolyzer Capacity Factor
        if (
            "pElectrolyzer_MW" in df.columns
            and metrics["Electrolyzer_Capacity_MW"] > 1e-6
        ):
            metrics["Electrolyzer_CF_percent"] = (
                df["pElectrolyzer_MW"].mean() / metrics["Electrolyzer_Capacity_MW"]
            ) * 100
            logger.debug(
                f"Electrolyzer CF calculated: {metrics['Electrolyzer_CF_percent']}%"
            )
        else:
            metrics["Electrolyzer_CF_percent"] = 0
            logger.debug(
                "Electrolyzer CF set to 0 (capacity or power data missing)")

        if "EnergyPrice_LMP_USDperMWh" in df.columns:
            metrics["Avg_Electricity_Price_USD_per_MWh"] = df[
                "EnergyPrice_LMP_USDperMWh"
            ].mean()
            if "pElectrolyzer_MW" in df.columns and df["pElectrolyzer_MW"].sum() > 0:
                weighted_price = (
                    df["EnergyPrice_LMP_USDperMWh"] * df["pElectrolyzer_MW"]
                ).sum() / df["pElectrolyzer_MW"].sum()
                metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = weighted_price
            else:
                metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = metrics[
                    "Avg_Electricity_Price_USD_per_MWh"
                ]
        else:
            metrics["Avg_Electricity_Price_USD_per_MWh"] = 40.0
            metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = 40.0

        # **ENHANCEMENT: Extract plant-specific thermal parameters**
        # Try to get thermal capacity from results dataframe first (added by cs1_tea.py)
        if "Thermal_Capacity_MWt" in df.columns and not df["Thermal_Capacity_MWt"].empty:
            metrics["thermal_capacity_mwt"] = df["Thermal_Capacity_MWt"].iloc[0]
            logger.debug(
                f"Thermal capacity from results: {metrics['thermal_capacity_mwt']} MWt")
        elif "thermal_capacity_mwt" in tea_sys_params and tea_sys_params["thermal_capacity_mwt"] is not None:
            try:
                metrics["thermal_capacity_mwt"] = float(
                    tea_sys_params["thermal_capacity_mwt"])
                logger.debug(
                    f"Thermal capacity from sys_params: {metrics['thermal_capacity_mwt']} MWt")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid thermal capacity value in sys_params: {tea_sys_params['thermal_capacity_mwt']}")
                metrics["thermal_capacity_mwt"] = 0
        else:
            metrics["thermal_capacity_mwt"] = 0

        # Try to get thermal efficiency from results dataframe first (added by cs1_tea.py)
        if "Thermal_Efficiency" in df.columns and not df["Thermal_Efficiency"].empty:
            metrics["thermal_efficiency"] = df["Thermal_Efficiency"].iloc[0]
            logger.debug(
                f"Thermal efficiency from results: {metrics['thermal_efficiency']:.4f}")
        elif "thermal_efficiency" in tea_sys_params and tea_sys_params["thermal_efficiency"] is not None:
            try:
                metrics["thermal_efficiency"] = float(
                    tea_sys_params["thermal_efficiency"])
                logger.debug(
                    f"Thermal efficiency from sys_params: {metrics['thermal_efficiency']:.4f}")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid thermal efficiency value in sys_params: {tea_sys_params['thermal_efficiency']}")
                metrics["thermal_efficiency"] = 0
        else:
            metrics["thermal_efficiency"] = 0

        # **NEW: Add Ancillary Services related statistics**
        # Calculate AS revenue statistics
        if "Revenue_Ancillary_USD" in df.columns:
            as_revenue_hourly = df["Revenue_Ancillary_USD"]
            metrics["AS_Revenue_Total"] = as_revenue_hourly.sum()
            metrics["AS_Revenue_Average_Hourly"] = as_revenue_hourly.mean()
            metrics["AS_Revenue_Maximum_Hourly"] = as_revenue_hourly.max()
            metrics["AS_Revenue_Hours_Positive"] = (
                as_revenue_hourly > 0).sum()
            metrics["AS_Revenue_Utilization_Rate"] = (
                metrics["AS_Revenue_Hours_Positive"] /
                len(as_revenue_hourly) * 100
                if len(as_revenue_hourly) > 0 else 0
            )

            # Calculate AS revenue per MW of capacity
            if metrics["Electrolyzer_Capacity_MW"] > 0:
                metrics["AS_Revenue_per_MW_Electrolyzer"] = (
                    metrics["AS_Revenue_Total"] /
                    metrics["Electrolyzer_Capacity_MW"]
                )
            if metrics["Battery_Power_MW"] > 0:
                metrics["AS_Revenue_per_MW_Battery"] = (
                    metrics["AS_Revenue_Total"] / metrics["Battery_Power_MW"]
                )

            logger.debug(f"AS Revenue statistics calculated:")
            logger.debug(
                f"  Total AS Revenue: ${metrics['AS_Revenue_Total']:,.2f}")
            logger.debug(
                f"  AS Utilization Rate: {metrics['AS_Revenue_Utilization_Rate']:.1f}%")

        # Look for bid data columns and calculate AS bid statistics
        as_bid_columns = [
            col for col in df.columns if "_Bid_MW" in col and "Total_" in col]
        if as_bid_columns:
            metrics["AS_Total_Bid_Services"] = len(as_bid_columns)

            # Calculate total bid capacity across all services
            total_bid_capacity = 0
            for bid_col in as_bid_columns:
                service_max_bid = df[bid_col].max()
                total_bid_capacity += service_max_bid

                # Individual service statistics
                service_name = bid_col.replace(
                    "Total_", "").replace("_Bid_MW", "")
                metrics[f"AS_Max_Bid_{service_name}_MW"] = service_max_bid
                metrics[f"AS_Avg_Bid_{service_name}_MW"] = df[bid_col].mean()

            metrics["AS_Total_Max_Bid_Capacity_MW"] = total_bid_capacity

            # AS capacity utilization relative to system capacity
            if metrics["Electrolyzer_Capacity_MW"] > 0:
                metrics["AS_Bid_Utilization_vs_Electrolyzer"] = (
                    total_bid_capacity /
                    metrics["Electrolyzer_Capacity_MW"] * 100
                )

        # Look for deployed data columns (if in dispatch simulation mode)
        as_deployed_columns = [
            col for col in df.columns if "_Deployed_MW" in col]
        if as_deployed_columns:
            total_deployed_energy = 0
            for deployed_col in as_deployed_columns:
                service_deployed_total = df[deployed_col].sum(
                ) * annualization_factor
                total_deployed_energy += service_deployed_total

                # Individual service deployment statistics
                service_name = deployed_col.replace("_Deployed_MW", "")
                metrics[f"AS_Total_Deployed_{service_name}_MWh"] = service_deployed_total
                metrics[f"AS_Avg_Deployed_{service_name}_MW"] = df[deployed_col].mean(
                )

            metrics["AS_Total_Deployed_Energy_MWh"] = total_deployed_energy

            # Deployment efficiency (deployed vs bid)
            for deployed_col in as_deployed_columns:
                service_base = deployed_col.replace("_Deployed_MW", "")
                corresponding_bid_col = f"Total_{service_base}_Bid_MW"
                if corresponding_bid_col in df.columns:
                    avg_deployed = df[deployed_col].mean()
                    avg_bid = df[corresponding_bid_col].mean()
                    if avg_bid > 0:
                        deployment_efficiency = (avg_deployed / avg_bid) * 100
                        metrics[f"AS_Deployment_Efficiency_{service_base}_percent"] = deployment_efficiency

        # **NEW: Calculate High Temperature Electrolyzer (HTE) heat opportunity cost**
        # For HTE systems, calculate the opportunity cost of steam consumption that could have been used for electricity generation
        metrics["HTE_Heat_Opportunity_Cost_Annual_USD"] = 0.0
        metrics["HTE_Steam_Consumption_Annual_MWth"] = 0.0
        metrics["HTE_Mode_Detected"] = False

        if "qSteam_Electrolyzer_MWth" in df.columns:
            # Check if this is an HTE system by looking for non-zero steam consumption
            steam_consumption_hourly = df["qSteam_Electrolyzer_MWth"]
            total_steam_consumption_annual = steam_consumption_hourly.sum() * \
                annualization_factor

            # Threshold to detect HTE mode (>1 MWth annually)
            if total_steam_consumption_annual > 1.0:
                metrics["HTE_Mode_Detected"] = True
                metrics["HTE_Steam_Consumption_Annual_MWth"] = total_steam_consumption_annual

                # Calculate opportunity cost using thermal efficiency and electricity price
                thermal_efficiency = metrics.get("thermal_efficiency", 0.0)
                avg_electricity_price = metrics.get(
                    "Avg_Electricity_Price_USD_per_MWh", 40.0)

                if thermal_efficiency > 0:
                    # Convert steam thermal energy to equivalent electrical energy using thermal efficiency
                    # Opportunity cost = steam_thermal_MWth * thermal_efficiency * electricity_price
                    lost_electricity_generation_mwh = total_steam_consumption_annual * thermal_efficiency
                    heat_opportunity_cost_annual = lost_electricity_generation_mwh * avg_electricity_price

                    metrics["HTE_Heat_Opportunity_Cost_Annual_USD"] = heat_opportunity_cost_annual
                    metrics["HTE_Lost_Electricity_Generation_Annual_MWh"] = lost_electricity_generation_mwh

                    logger.info(f"HTE Heat Opportunity Cost Analysis:")
                    logger.info(
                        f"  Annual steam consumption: {total_steam_consumption_annual:,.1f} MWth")
                    logger.info(
                        f"  Thermal efficiency: {thermal_efficiency:.4f}")
                    logger.info(
                        f"  Average electricity price: ${avg_electricity_price:.2f}/MWh")
                    logger.info(
                        f"  Lost electricity generation: {lost_electricity_generation_mwh:,.1f} MWh")
                    logger.info(
                        f"  Heat opportunity cost: ${heat_opportunity_cost_annual:,.2f}/year")

                    # Add per-kg hydrogen cost component
                    h2_production_annual = metrics.get(
                        "H2_Production_kg_annual", 0)
                    if h2_production_annual > 0:
                        heat_opportunity_cost_per_kg_h2 = heat_opportunity_cost_annual / h2_production_annual
                        metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = heat_opportunity_cost_per_kg_h2
                        logger.info(
                            f"  Heat opportunity cost per kg H2: ${heat_opportunity_cost_per_kg_h2:.3f}/kg")
                    else:
                        metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = 0.0
                else:
                    logger.warning(
                        "HTE system detected but thermal efficiency is 0 or missing. Cannot calculate heat opportunity cost.")
                    metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = 0.0
                    metrics["HTE_Lost_Electricity_Generation_Annual_MWh"] = 0.0
            else:
                logger.debug(
                    "LTE mode detected (no significant steam consumption by electrolyzer)")
        else:
            logger.debug(
                "No steam consumption data found - likely LTE mode or steam consumption not tracked")

    except KeyError as e:
        logger.error(
            f"Missing column in hourly results for annual metrics calculation: {e}"
        )
        return None
    except Exception as e:
        logger.error(f"Error calculating annual metrics: {e}", exc_info=True)
        return None
    return metrics


def calculate_lcoh_breakdown(
    annual_metrics: dict,
    capex_breakdown: dict,
    project_lifetime: int,
    construction_period: int,
    discount_rate: float,
    annual_h2_production_kg: float,
) -> dict:
    """
    Calculate detailed LCOH breakdown by cost factors and their percentages.

    Args:
        annual_metrics: Dictionary containing annual operational metrics
        capex_breakdown: Dictionary containing CAPEX breakdown by component
        project_lifetime: Project lifetime in years
        construction_period: Construction period in years
        discount_rate: Discount rate for present value calculations
        annual_h2_production_kg: Annual hydrogen production in kg

    Returns:
        Dictionary containing LCOH breakdown by cost factors and percentages
    """
    logger.info("Calculating detailed LCOH breakdown by cost factors")

    if annual_h2_production_kg <= 0:
        logger.warning(
            "Annual H2 production is zero or negative. Cannot calculate LCOH breakdown.")
        return {}

    # Calculate present value of total H2 production over project lifetime
    pv_total_h2_production = 0
    for op_idx in range(project_lifetime - construction_period):
        year_idx = op_idx + construction_period
        pv_factor = (1 + discount_rate) ** year_idx
        pv_total_h2_production += annual_h2_production_kg / pv_factor

    if pv_total_h2_production <= 0:
        logger.warning(
            "Present value of H2 production is zero or negative. Cannot calculate LCOH breakdown.")
        return {}

    lcoh_breakdown = {}

    # 1. CAPEX Components (annualized and converted to $/kg H2)
    logger.debug("Calculating CAPEX contribution to LCOH:")

    # Calculate Capital Recovery Factor (CRF)
    if discount_rate > 0:
        crf = (discount_rate * (1 + discount_rate) ** project_lifetime) / \
              ((1 + discount_rate) ** project_lifetime - 1)
    else:
        crf = 1 / project_lifetime

    total_capex_lcoh_contribution = 0
    capex_lcoh_components = {}

    for component, capex_cost in capex_breakdown.items():
        if capex_cost > 0:
            # Annualize CAPEX using CRF
            annualized_capex = capex_cost * crf
            # Convert to $/kg H2
            capex_lcoh_per_kg = annualized_capex / annual_h2_production_kg
            capex_lcoh_components[f"CAPEX_{component}"] = capex_lcoh_per_kg
            total_capex_lcoh_contribution += capex_lcoh_per_kg

            logger.debug(
                f"   {component}: ${capex_cost:,.0f} -> ${capex_lcoh_per_kg:.3f}/kg H2")

    # 2. Fixed O&M Costs
    logger.debug("Calculating Fixed O&M contribution to LCOH:")

    # Get annual fixed O&M costs from stored data
    annual_fixed_om_costs = annual_metrics.get("annual_fixed_om_costs", [])
    if annual_fixed_om_costs:
        # Calculate present value of all fixed O&M costs
        pv_fixed_om_total = 0
        for year_idx, annual_cost in enumerate(annual_fixed_om_costs):
            pv_factor = (1 + discount_rate) ** (year_idx + construction_period)
            pv_fixed_om_total += annual_cost / pv_factor

        fixed_om_lcoh_per_kg = pv_fixed_om_total / pv_total_h2_production
        lcoh_breakdown["Fixed_OM"] = fixed_om_lcoh_per_kg
        logger.debug(
            f"   Fixed O&M: PV ${pv_fixed_om_total:,.0f} -> ${fixed_om_lcoh_per_kg:.3f}/kg H2")
    else:
        # Fallback: use current year fixed O&M and assume constant
        total_capex = annual_metrics.get("total_capex", 0)
        fixed_om_rate = 0.02  # 2% of CAPEX as default
        annual_fixed_om = total_capex * fixed_om_rate
        fixed_om_lcoh_per_kg = annual_fixed_om / annual_h2_production_kg
        lcoh_breakdown["Fixed_OM"] = fixed_om_lcoh_per_kg
        logger.debug(
            f"   Fixed O&M (estimated): ${annual_fixed_om:,.0f}/year -> ${fixed_om_lcoh_per_kg:.3f}/kg H2")

    # 3. Variable Operating Costs (H2 production related only)
    logger.debug("Calculating Variable OPEX contribution to LCOH:")

    variable_opex_components = {
        "VOM_Electrolyzer": annual_metrics.get("VOM_Electrolyzer_Cost", 0),
        # VOM_Turbine excluded - this is NPP baseline cost, not H2 production cost
        "VOM_Battery": annual_metrics.get("VOM_Battery_Cost", 0),
        "Water_Cost": annual_metrics.get("Water_Cost", 0),
        "Startup_Cost": annual_metrics.get("Startup_Cost", 0),
        "Ramping_Cost": annual_metrics.get("Ramping_Cost", 0),
        "H2_Storage_Cycle_Cost": annual_metrics.get("H2_Storage_Cycle_Cost", 0),
    }

    total_variable_opex_lcoh = 0
    for component, annual_cost in variable_opex_components.items():
        if annual_cost > 0:
            # Convert annual cost to $/kg H2
            cost_per_kg = annual_cost / annual_h2_production_kg
            lcoh_breakdown[component] = cost_per_kg
            total_variable_opex_lcoh += cost_per_kg
            logger.debug(
                f"   {component}: ${annual_cost:,.0f}/year -> ${cost_per_kg:.3f}/kg H2")

    # 4. Electricity Opportunity Cost (for electrolyzer and battery charging from NPP)
    logger.debug(
        "Calculating Electricity Opportunity Cost contribution to LCOH:")

    # Electrolyzer electricity opportunity cost
    annual_electrolyzer_mwh = annual_metrics.get("Annual_Electrolyzer_MWh", 0)
    avg_electricity_price = annual_metrics.get(
        "Avg_Electricity_Price_USD_per_MWh", 40.0)
    electrolyzer_opportunity_cost = annual_electrolyzer_mwh * avg_electricity_price

    if electrolyzer_opportunity_cost > 0:
        electrolyzer_opp_cost_per_kg = electrolyzer_opportunity_cost / annual_h2_production_kg
        lcoh_breakdown["Electricity_Opportunity_Cost_Electrolyzer"] = electrolyzer_opp_cost_per_kg
        logger.debug(
            f"   Electrolyzer Electricity Opportunity Cost: ${electrolyzer_opportunity_cost:,.0f}/year -> ${electrolyzer_opp_cost_per_kg:.3f}/kg H2")

    # Battery charging opportunity cost (NPP charging)
    battery_npp_charge_mwh = annual_metrics.get(
        "Annual_Battery_Charge_From_NPP_MWh", 0)
    battery_opportunity_cost = battery_npp_charge_mwh * avg_electricity_price

    if battery_opportunity_cost > 0:
        battery_opp_cost_per_kg = battery_opportunity_cost / annual_h2_production_kg
        lcoh_breakdown["Electricity_Opportunity_Cost_Battery"] = battery_opp_cost_per_kg
        logger.debug(
            f"   Battery Electricity Opportunity Cost: ${battery_opportunity_cost:,.0f}/year -> ${battery_opp_cost_per_kg:.3f}/kg H2")

    # Battery direct charging cost (grid charging)
    battery_grid_charge_mwh = annual_metrics.get(
        "Annual_Battery_Charge_From_Grid_MWh", 0)
    battery_direct_cost = battery_grid_charge_mwh * avg_electricity_price

    if battery_direct_cost > 0:
        battery_direct_cost_per_kg = battery_direct_cost / annual_h2_production_kg
        lcoh_breakdown["Electricity_Direct_Cost_Battery"] = battery_direct_cost_per_kg
        logger.debug(
            f"   Battery Direct Electricity Cost: ${battery_direct_cost:,.0f}/year -> ${battery_direct_cost_per_kg:.3f}/kg H2")

    # 5. HTE Heat Opportunity Cost (if applicable)
    hte_heat_opportunity_cost = annual_metrics.get(
        "HTE_Heat_Opportunity_Cost_Annual_USD", 0)
    if hte_heat_opportunity_cost > 0:
        hte_cost_per_kg = hte_heat_opportunity_cost / annual_h2_production_kg
        lcoh_breakdown["HTE_Heat_Opportunity_Cost"] = hte_cost_per_kg
        logger.debug(
            f"   HTE Heat Opportunity Cost: ${hte_heat_opportunity_cost:,.0f}/year -> ${hte_cost_per_kg:.3f}/kg H2")

    # 6. Replacement Costs
    logger.debug("Calculating Replacement Costs contribution to LCOH:")

    # Stack replacement costs
    annual_stack_replacement_costs = annual_metrics.get(
        "annual_stack_replacement_costs", [])
    if annual_stack_replacement_costs:
        pv_stack_replacement_total = 0
        for year_idx, annual_cost in enumerate(annual_stack_replacement_costs):
            if annual_cost > 0:
                pv_factor = (1 + discount_rate) ** (year_idx +
                                                    construction_period)
                pv_stack_replacement_total += annual_cost / pv_factor

        if pv_stack_replacement_total > 0:
            stack_replacement_lcoh_per_kg = pv_stack_replacement_total / pv_total_h2_production
            lcoh_breakdown["Stack_Replacement"] = stack_replacement_lcoh_per_kg
            logger.debug(
                f"   Stack Replacement: PV ${pv_stack_replacement_total:,.0f} -> ${stack_replacement_lcoh_per_kg:.3f}/kg H2")

    # Other replacement costs
    annual_other_replacement_costs = annual_metrics.get(
        "annual_other_replacement_costs", [])
    if annual_other_replacement_costs:
        pv_other_replacement_total = 0
        for year_idx, annual_cost in enumerate(annual_other_replacement_costs):
            if annual_cost > 0:
                pv_factor = (1 + discount_rate) ** (year_idx +
                                                    construction_period)
                pv_other_replacement_total += annual_cost / pv_factor

        if pv_other_replacement_total > 0:
            other_replacement_lcoh_per_kg = pv_other_replacement_total / pv_total_h2_production
            lcoh_breakdown["Other_Replacements"] = other_replacement_lcoh_per_kg
            logger.debug(
                f"   Other Replacements: PV ${pv_other_replacement_total:,.0f} -> ${other_replacement_lcoh_per_kg:.3f}/kg H2")

    # Add CAPEX components to main breakdown
    lcoh_breakdown.update(capex_lcoh_components)

    # Calculate total LCOH and percentages
    total_lcoh = sum(lcoh_breakdown.values())
    lcoh_percentages = {}

    if total_lcoh > 0:
        for component, cost_per_kg in lcoh_breakdown.items():
            percentage = (cost_per_kg / total_lcoh) * 100
            lcoh_percentages[component] = percentage

    # Prepare final results
    lcoh_analysis = {
        "total_lcoh_usd_per_kg": total_lcoh,
        "lcoh_breakdown_usd_per_kg": lcoh_breakdown,
        "lcoh_percentages": lcoh_percentages,
        "pv_total_h2_production_kg": pv_total_h2_production,
    }

    # Log summary
    logger.info(f"LCOH Breakdown Analysis Summary:")
    logger.info(f"   Total LCOH: ${total_lcoh:.3f}/kg H2")
    logger.info(f"   Number of cost components: {len(lcoh_breakdown)}")

    # Log top 5 cost contributors
    sorted_components = sorted(
        lcoh_breakdown.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"   Top 5 cost contributors:")
    for i, (component, cost) in enumerate(sorted_components[:5]):
        percentage = lcoh_percentages.get(component, 0)
        logger.info(
            f"     {i+1}. {component}: ${cost:.3f}/kg ({percentage:.1f}%)")

    # **NEW: Add sensitivity analysis for key cost drivers**
    logger.debug("Calculating LCOH sensitivity analysis...")

    # Define sensitivity ranges (±20% for component costs)
    sensitivity_range = 0.20
    sensitivity_analysis = {}

    # Analyze sensitivity for top 5 cost contributors
    for component, base_cost in sorted_components[:5]:
        component_sensitivity = {}

        # Calculate LCOH impact for ±20% change in this component
        for change_pct in [-sensitivity_range, sensitivity_range]:
            adjusted_cost = base_cost * (1 + change_pct)
            cost_difference = adjusted_cost - base_cost
            new_total_lcoh = total_lcoh + cost_difference
            lcoh_change = new_total_lcoh - total_lcoh

            component_sensitivity[f"{change_pct*100:+.0f}%"] = {
                "lcoh_change": lcoh_change,
                "new_total_lcoh": new_total_lcoh,
                "impact_percentage": (lcoh_change / total_lcoh) * 100
            }

        sensitivity_analysis[component] = component_sensitivity

    # Add sensitivity analysis to results
    lcoh_analysis["sensitivity_analysis"] = sensitivity_analysis

    logger.debug(
        f"Sensitivity analysis completed for {len(sensitivity_analysis)} components")

    return lcoh_analysis


def calculate_cash_flows(
    annual_metrics: dict,
    project_lifetime: int,
    construction_period: int,
    h2_subsidy_value: float,
    h2_subsidy_duration: int,
    capex_details: dict,
    om_details: dict,
    replacement_details: dict,
    optimized_capacities: dict,
) -> np.ndarray:
    logger.info(
        f"Calculating cash flows for {project_lifetime} years. Construction period: {construction_period} years."
    )

    # Log the optimized capacities received for debugging
    logger.debug(f"Optimized capacities received by calculate_cash_flows:")
    for cap_key, cap_val in optimized_capacities.items():
        logger.debug(f"   {cap_key} = {cap_val}")

    cash_flows_array = np.zeros(project_lifetime + construction_period)
    total_capex_sum_after_learning = 0

    # Store initial CAPEX for battery for replacement calculation
    initial_battery_capex_energy = 0
    initial_battery_capex_power = 0
    # Store initial electrolyzer CAPEX for stack replacement
    initial_electrolyzer_capex = 0

    # Add CAPEX breakdown for reporting and visualization
    capex_breakdown = {}

    for component_name, comp_data in capex_details.items():
        base_cost_for_ref_size = comp_data.get(
            "total_base_cost_for_ref_size", 0)
        ref_capacity = comp_data.get("reference_total_capacity_mw", 0)
        lr_decimal = comp_data.get("learning_rate_decimal", 0)
        capacity_key = comp_data.get("applies_to_component_capacity_key")
        payment_schedule = comp_data.get("payment_schedule_years", {})

        # Get optimized capacity for this component (or default if not available)
        actual_optimized_capacity = optimized_capacities.get(
            capacity_key, ref_capacity if capacity_key else 0
        )

        # Detailed logging of capacity and cost calculation
        logger.debug(f"Processing CAPEX for '{component_name}':")
        logger.debug(f"   Linked to capacity key: {capacity_key}")
        logger.debug(f"   Reference capacity: {ref_capacity}")
        logger.debug(
            f"   Actual optimized capacity: {actual_optimized_capacity}")
        logger.debug(
            f"   Base cost for reference size: ${base_cost_for_ref_size:,.2f}")
        logger.debug(f"   Learning rate: {lr_decimal*100:.1f}%")

        adjusted_total_component_cost = 0.0
        if capacity_key and actual_optimized_capacity == 0 and ref_capacity > 0:
            logger.info(
                f"Component '{component_name}' was sized to 0 (e.g., MW or kg). Its CAPEX will be 0."
            )
            logger.debug(f"   Component sized to 0. CAPEX will be 0.")
            adjusted_total_component_cost = 0.0
        elif (
            lr_decimal > 0
            and ref_capacity > 0
            and actual_optimized_capacity > 0
            and capacity_key
        ):
            progress_ratio = 1 - lr_decimal
            b = math.log(progress_ratio) / \
                math.log(2) if 0 < progress_ratio < 1 else 0
            scale_factor = actual_optimized_capacity / ref_capacity
            adjusted_total_component_cost = base_cost_for_ref_size * \
                (scale_factor**b)
            logger.debug(
                f"   Applying learning rate. Scale factor: {scale_factor:.3f}, LR exponent b: {b:.4f}"
            )
            logger.debug(
                f"   Formula: {base_cost_for_ref_size:,.2f} * ({scale_factor:.3f} ^ {b:.4f}) = ${adjusted_total_component_cost:,.2f}"
            )
            logger.info(
                f"Component '{component_name}': Ref Cost=${base_cost_for_ref_size:,.0f} (Ref Cap:{ref_capacity}), Optimized Cap:{actual_optimized_capacity}, LR:{lr_decimal*100}%, Adjusted Total Cost=${adjusted_total_component_cost:,.0f}"
            )
        elif actual_optimized_capacity > 0 and ref_capacity > 0 and capacity_key:
            scale_factor = actual_optimized_capacity / ref_capacity
            adjusted_total_component_cost = base_cost_for_ref_size * scale_factor
            logger.debug(
                f"   Linear scaling without learning rate. Scale factor: {scale_factor:.3f}"
            )
            logger.debug(
                f"   Formula: {base_cost_for_ref_size:,.2f} * {scale_factor:.3f} = ${adjusted_total_component_cost:,.2f}"
            )
            logger.info(
                f"Component '{component_name}': Ref Cost=${base_cost_for_ref_size:,.0f} (Ref Cap:{ref_capacity}), Optimized Cap:{actual_optimized_capacity}, No LR, Linearly Scaled Total Cost=${adjusted_total_component_cost:,.0f}"
            )
        elif not capacity_key:
            adjusted_total_component_cost = base_cost_for_ref_size
            logger.debug(
                f"   Fixed component, no capacity scaling. Cost: ${adjusted_total_component_cost:,.2f}"
            )
            logger.info(
                f"Component '{component_name}': Fixed Cost=${adjusted_total_component_cost:,.0f} (does not scale)."
            )
        else:
            adjusted_total_component_cost = 0.0
            if base_cost_for_ref_size > 0 and capacity_key:
                logger.info(
                    f"Component '{component_name}' has 0 optimized capacity. Its CAPEX is 0."
                )
                logger.debug(
                    f"   Component has 0 optimized capacity. CAPEX is 0.")

        # Store CAPEX component cost in breakdown dictionary
        friendly_component_name = component_name.replace("_", " ")
        capex_breakdown[friendly_component_name] = adjusted_total_component_cost

        # Save CAPEX for each component for subsequent calculations
        if component_name == "Battery_System_Energy":
            initial_battery_capex_energy = adjusted_total_component_cost
        if component_name == "Battery_System_Power":
            initial_battery_capex_power = adjusted_total_component_cost
        if component_name == "Electrolyzer_System":
            initial_electrolyzer_capex = adjusted_total_component_cost

        total_capex_sum_after_learning += adjusted_total_component_cost
        for constr_year_offset, share in payment_schedule.items():
            project_year_index = construction_period + constr_year_offset
            if 0 <= project_year_index < construction_period:
                cash_flows_array[project_year_index] -= (
                    adjusted_total_component_cost * share
                )
            else:
                logger.warning(
                    f"Payment schedule year {constr_year_offset} for component '{component_name}' is outside construction period."
                )

    # Store CAPEX breakdown and total in annual_metrics for reporting and visualization
    annual_metrics["capex_breakdown"] = capex_breakdown
    annual_metrics["total_capex"] = total_capex_sum_after_learning
    # Save electrolyzer CAPEX for LCOH calculation
    annual_metrics["electrolyzer_capex"] = initial_electrolyzer_capex

    logger.debug(f"Final CAPEX breakdown:")
    for comp, cost in sorted(capex_breakdown.items(), key=lambda x: x[1], reverse=True):
        logger.debug(f"   {comp}: ${cost:,.2f}")
    logger.debug(
        f"Total CAPEX after learning rate/scaling adjustments: ${total_capex_sum_after_learning:,.2f}"
    )

    logger.info(
        f"Total CAPEX after learning rate/scaling adjustments: ${total_capex_sum_after_learning:,.2f}"
    )
    initial_total_battery_capex = (
        initial_battery_capex_energy + initial_battery_capex_power
    )

    base_annual_profit_from_opt = annual_metrics.get(
        "Annual_Revenue", 0
    ) - annual_metrics.get("Annual_Opex_Cost_from_Opt", 0)

    # New variables to store annual O&M, stack replacement, and other costs (for LCOH calculation)
    annual_fixed_om_costs = []
    annual_stack_replacement_costs = []
    annual_other_replacement_costs = []

    for op_year_idx in range(project_lifetime - construction_period):
        current_project_year_idx = op_year_idx + construction_period
        operational_year_num = op_year_idx + 1
        current_year_profit_before_fixed_om_repl_tax = base_annual_profit_from_opt
        if operational_year_num > h2_subsidy_duration:
            current_year_profit_before_fixed_om_repl_tax -= annual_metrics.get(
                "H2_Subsidy_Revenue", 0
            )

        # **NEW: HTE Heat Opportunity Cost - subtract from profit (add to costs)**
        # For high temperature electrolyzers, include the opportunity cost of steam that could have been used for electricity generation
        hte_heat_opportunity_cost_annual = annual_metrics.get(
            "HTE_Heat_Opportunity_Cost_Annual_USD", 0.0)
        if hte_heat_opportunity_cost_annual > 0:
            current_year_profit_before_fixed_om_repl_tax -= hte_heat_opportunity_cost_annual
            logger.debug(
                f"Year {operational_year_num} HTE Heat Opportunity Cost: ${hte_heat_opportunity_cost_annual:,.2f}")

        # Fixed O&M - based on 2% of total CAPEX (instead of a fixed amount)
        if (
            om_details.get("Fixed_OM_General", {}).get(
                "base_cost_percent_of_capex", 0)
            > 0
        ):
            fixed_om_percent = om_details.get("Fixed_OM_General", {}).get(
                "base_cost_percent_of_capex", 0.02
            )
            fixed_om_general_cost = (
                total_capex_sum_after_learning
                * fixed_om_percent
                * (
                    (
                        1
                        + om_details.get("Fixed_OM_General", {}).get(
                            "inflation_rate", 0
                        )
                    )
                    ** op_year_idx
                )
            )
            logger.debug(
                f"Year {operational_year_num} Fixed O&M: ${fixed_om_general_cost:,.2f} ({fixed_om_percent*100}% of CAPEX)")
        else:
            # Traditional method, fixed amount (deprecated, but kept for compatibility)
            fixed_om_general_cost = om_details.get("Fixed_OM_General", {}).get(
                "base_cost", 0
            ) * (
                (1 + om_details.get("Fixed_OM_General", {}).get("inflation_rate", 0))
                ** op_year_idx
            )

        annual_fixed_om_costs.append(fixed_om_general_cost)
        current_year_profit_before_fixed_om_repl_tax -= fixed_om_general_cost

        # Battery Fixed O&M (if battery is enabled and capacity > 0)
        if ENABLE_BATTERY and optimized_capacities.get("Battery_Capacity_MWh", 0) > 0:
            batt_fixed_om_per_mw = om_details.get("Fixed_OM_Battery", {}).get(
                "base_cost_per_mw_year", 0
            )
            batt_fixed_om_per_mwh = om_details.get("Fixed_OM_Battery", {}).get(
                "base_cost_per_mwh_year", 0
            )
            batt_inflation = om_details.get("Fixed_OM_Battery", {}).get(
                "inflation_rate", 0
            )

            batt_power_mw = optimized_capacities.get("Battery_Power_MW", 0)
            batt_capacity_mwh = optimized_capacities.get(
                "Battery_Capacity_MWh", 0)

            battery_fixed_om_cost_this_year = (
                batt_power_mw * batt_fixed_om_per_mw
                + batt_capacity_mwh * batt_fixed_om_per_mwh
            ) * ((1 + batt_inflation) ** op_year_idx)
            annual_fixed_om_costs[-1] += battery_fixed_om_cost_this_year
            current_year_profit_before_fixed_om_repl_tax -= (
                battery_fixed_om_cost_this_year
            )

        replacement_cost_this_year = 0
        stack_replacement_cost_this_year = 0
        other_replacement_cost_this_year = 0

        for rep_comp_name, comp_data in replacement_details.items():
            if operational_year_num in comp_data.get("years", []):
                # Stack replacement based on a percentage of electrolyzer CAPEX
                if (
                    rep_comp_name == "Electrolyzer_Stack"
                    and "cost_percent_initial_capex" in comp_data
                ):
                    # Electrolyzer stack replacement based on electrolyzer CAPEX percentage
                    percent = comp_data.get("cost_percent_initial_capex", 0.30)
                    cost_val = initial_electrolyzer_capex * percent
                    logger.debug(
                        f"Year {operational_year_num} Stack Replacement: ${cost_val:,.2f} ({percent*100}% of Electrolyzer CAPEX)"
                    )
                    stack_replacement_cost_this_year += cost_val
                elif (
                    rep_comp_name == "Battery_Augmentation_Replacement"
                    and comp_data.get("cost_percent_initial_capex", 0) > 0
                ):
                    # Battery replacement based on battery CAPEX percentage
                    cost_val = (
                        initial_total_battery_capex
                        * comp_data["cost_percent_initial_capex"]
                    )
                    other_replacement_cost_this_year += cost_val
                else:
                    # Other component replacements, using traditional fixed cost
                    cost_val = comp_data.get("cost", 0)
                    other_replacement_cost_this_year += cost_val

                replacement_cost_this_year += cost_val

        annual_stack_replacement_costs.append(stack_replacement_cost_this_year)
        annual_other_replacement_costs.append(other_replacement_cost_this_year)
        current_year_profit_before_fixed_om_repl_tax -= replacement_cost_this_year

        # Calculate taxable income
        taxable_income = current_year_profit_before_fixed_om_repl_tax
        tax_amount = taxable_income * TAX_RATE if taxable_income > 0 else 0
        cash_flows_array[current_project_year_idx] = taxable_income - tax_amount

    # Save annual cost data to metrics for LCOH calculation
    annual_metrics["annual_fixed_om_costs"] = annual_fixed_om_costs
    annual_metrics["annual_stack_replacement_costs"] = annual_stack_replacement_costs
    annual_metrics["annual_other_replacement_costs"] = annual_other_replacement_costs

    return cash_flows_array


def calculate_financial_metrics(
    cash_flows_input: np.ndarray,
    discount_rt: float,
    annual_h2_prod_kg: float,
    project_lt: int,
    construction_p: int,
) -> dict:
    metrics_results = {}
    cf_array = np.array(cash_flows_input, dtype=float)
    try:
        metrics_results["NPV_USD"] = npf.npv(discount_rt, cf_array)
    except Exception:
        metrics_results["NPV_USD"] = np.nan
    try:
        if any(cf > 0 for cf in cf_array) and any(cf < 0 for cf in cf_array):
            metrics_results["IRR_percent"] = npf.irr(cf_array) * 100
        else:
            metrics_results["IRR_percent"] = np.nan
    except Exception:
        metrics_results["IRR_percent"] = np.nan
    cumulative_cash_flow = np.cumsum(cf_array)
    positive_indices = np.where(cumulative_cash_flow >= 0)[0]
    if positive_indices.size > 0:
        first_positive_idx = positive_indices[0]
        if first_positive_idx == 0 and cf_array[0] >= 0:
            metrics_results["Payback_Period_Years"] = 0
        elif (
            first_positive_idx > 0 and cumulative_cash_flow[first_positive_idx - 1] < 0
        ):
            metrics_results["Payback_Period_Years"] = (
                (first_positive_idx - 1)
                + abs(cumulative_cash_flow[first_positive_idx - 1])
                / (
                    cumulative_cash_flow[first_positive_idx]
                    - cumulative_cash_flow[first_positive_idx - 1]
                )
                - construction_p
                + 1
            )
        else:
            metrics_results["Payback_Period_Years"] = (
                first_positive_idx - construction_p + 1
            )
    else:
        metrics_results["Payback_Period_Years"] = np.nan

    # Note: LCOH calculation is now handled by the detailed breakdown method in calculate_lcoh_breakdown()
    # This provides more accurate cost attribution and component-level analysis

    # Calculate ROI (Return on Investment) = NPV / Total CAPEX
    # This will be populated later when CAPEX data is available
    metrics_results["ROI"] = np.nan

    return metrics_results


def calculate_incremental_metrics(
    optimized_cash_flows: np.ndarray,
    baseline_annual_revenue: float,
    project_lifetime: int,
    construction_period: int,
    discount_rt: float,
    tax_rt: float,
    annual_metrics_optimized: dict,
    capex_components_incremental: dict,
    om_components_incremental: dict,
    replacement_schedule_incremental: dict,
    h2_subsidy_val: float,
    h2_subsidy_yrs: int,
    optimized_capacities_inc: dict,
) -> dict:
    logger.info("Calculating incremental financial metrics.")
    inc_metrics = {}

    # **FIXED: Ensure baseline_cash_flows has the same length as optimized_cash_flows**
    # optimized_cash_flows might include construction period, so match its length
    total_project_years = len(optimized_cash_flows)
    baseline_cash_flows = np.zeros(total_project_years)

    # **FIXED: Calculate actual baseline costs instead of using 30% assumption**
    # Calculate baseline operating costs based on actual turbine VOM costs
    baseline_annual_opex = annual_metrics_optimized.get("VOM_Turbine_Cost", 0)
    baseline_annual_profit_before_tax = baseline_annual_revenue - baseline_annual_opex

    logger.info(f"Baseline revenue calculation:")
    logger.info(f"  Annual baseline revenue: ${baseline_annual_revenue:,.2f}")
    logger.info(
        f"  Annual baseline OPEX (VOM Turbine): ${baseline_annual_opex:,.2f}")
    logger.info(
        f"  Annual baseline profit before tax: ${baseline_annual_profit_before_tax:,.2f}")

    for i in range(construction_period, total_project_years):
        baseline_cash_flows[i] = baseline_annual_profit_before_tax * (
            1 - tax_rt if baseline_annual_profit_before_tax > 0 else 1
        )

    pure_incremental_cf = np.zeros(total_project_years)
    total_incremental_capex_sum_after_learning = 0
    initial_inc_battery_capex_energy = 0  # For incremental battery replacement
    initial_inc_battery_capex_power = 0  # For incremental battery replacement

    for comp_name, comp_data in capex_components_incremental.items():
        base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
        ref_cap = comp_data.get("reference_total_capacity_mw", 0)
        lr = comp_data.get("learning_rate_decimal", 0)
        cap_key = comp_data.get("applies_to_component_capacity_key")
        pay_sched = comp_data.get("payment_schedule_years", {})

        actual_opt_cap_inc = optimized_capacities_inc.get(
            cap_key, ref_cap if cap_key else 0
        )
        adj_cost_inc = 0.0
        if cap_key and actual_opt_cap_inc == 0 and ref_cap > 0:
            adj_cost_inc = 0.0
        elif lr > 0 and ref_cap > 0 and actual_opt_cap_inc > 0 and cap_key:
            pr = 1 - lr
            b = math.log(pr) / math.log(2) if 0 < pr < 1 else 0
            adj_cost_inc = base_cost * ((actual_opt_cap_inc / ref_cap) ** b)
        elif actual_opt_cap_inc > 0 and ref_cap > 0 and cap_key:
            adj_cost_inc = base_cost * (actual_opt_cap_inc / ref_cap)
        elif not cap_key:
            adj_cost_inc = base_cost

        if comp_name == "Battery_System_Energy":
            initial_inc_battery_capex_energy = adj_cost_inc
        if comp_name == "Battery_System_Power":
            initial_inc_battery_capex_power = adj_cost_inc

        total_incremental_capex_sum_after_learning += adj_cost_inc
        for constr_yr_offset, share in pay_sched.items():
            if 0 <= construction_period + constr_yr_offset < construction_period:
                pure_incremental_cf[construction_period + constr_yr_offset] -= (
                    adj_cost_inc * share
                )
    inc_metrics["Total_Incremental_CAPEX_Learned_USD"] = (
        total_incremental_capex_sum_after_learning
    )
    initial_total_inc_battery_capex = (
        initial_inc_battery_capex_energy + initial_inc_battery_capex_power
    )

    # **FIXED: Incremental revenue calculation - only include new revenue sources**
    h2_rev_annual = annual_metrics_optimized.get("H2_Total_Revenue", 0)
    as_rev_annual = annual_metrics_optimized.get("AS_Revenue", 0)

    # Get electricity price for opportunity cost calculations
    avg_elec_price = annual_metrics_optimized.get(
        "Avg_Electricity_Price_USD_per_MWh", 40.0)

    # **IMPROVED: Distinguish battery charging costs based on electricity source**
    opp_cost_battery_annual = 0.0
    direct_cost_battery_annual = 0.0

    if ENABLE_BATTERY and optimized_capacities_inc.get("Battery_Capacity_MWh", 0) > 0:
        # Battery charging from NPP is an opportunity cost (lost revenue from not selling to grid)
        battery_charge_from_npp_mwh = annual_metrics_optimized.get(
            "Annual_Battery_Charge_From_NPP_MWh", 0)
        opp_cost_battery_annual = battery_charge_from_npp_mwh * avg_elec_price

        # Battery charging from grid is a direct operating cost (electricity purchase)
        battery_charge_from_grid_mwh = annual_metrics_optimized.get(
            "Annual_Battery_Charge_From_Grid_MWh", 0)
        direct_cost_battery_annual = battery_charge_from_grid_mwh * avg_elec_price

        logger.info(f"Battery charging cost breakdown:")
        logger.info(
            f"  Opportunity cost (NPP charging): ${opp_cost_battery_annual:,.2f}/year")
        logger.info(
            f"  Direct cost (grid charging): ${direct_cost_battery_annual:,.2f}/year")

    # **NEW: Calculate AS opportunity cost - electricity not sold to grid due to AS deployment**
    # This includes both regulation and reserve services that reduce electricity sales
    as_opportunity_cost_annual = 0.0

    # Get total deployed AS energy across all services and components
    total_as_deployed_mwh = 0.0
    as_deployment_keys = [
        "AS_Total_Deployed_ECRS_Battery_MWh", "AS_Total_Deployed_ECRS_Electrolyzer_MWh", "AS_Total_Deployed_ECRS_Turbine_MWh",
        "AS_Total_Deployed_RegDown_Battery_MWh", "AS_Total_Deployed_RegDown_Electrolyzer_MWh", "AS_Total_Deployed_RegDown_Turbine_MWh",
        "AS_Total_Deployed_RegUp_Battery_MWh", "AS_Total_Deployed_RegUp_Electrolyzer_MWh", "AS_Total_Deployed_RegUp_Turbine_MWh",
        "AS_Total_Deployed_NSR_Battery_MWh", "AS_Total_Deployed_NSR_Electrolyzer_MWh", "AS_Total_Deployed_NSR_Turbine_MWh",
        "AS_Total_Deployed_SR_Battery_MWh", "AS_Total_Deployed_SR_Electrolyzer_MWh", "AS_Total_Deployed_SR_Turbine_MWh",
    ]

    for key in as_deployment_keys:
        if "Turbine" in key:  # Only count turbine AS deployment as lost electricity sales
            total_as_deployed_mwh += annual_metrics_optimized.get(key, 0)

    as_opportunity_cost_annual = total_as_deployed_mwh * avg_elec_price

    logger.info(f"AS opportunity cost calculation:")
    logger.info(
        f"  Total AS deployed by turbine: {total_as_deployed_mwh:,.2f} MWh/year")
    logger.info(
        f"  AS opportunity cost: ${as_opportunity_cost_annual:,.2f}/year")

    vom_annual_inc = sum(
        annual_metrics_optimized.get(k, 0)
        for k in [
            "VOM_Electrolyzer_Cost",
            "VOM_Battery_Cost",
            "Water_Cost",
            "Startup_Cost",
            "Ramping_Cost",
            "H2_Storage_Cycle_Cost",
        ]
    )
    # Add direct cost of battery charging from grid to VOM costs
    vom_annual_inc += direct_cost_battery_annual

    # **FIXED: Electrolyzer opportunity cost calculation**
    opp_cost_elec_annual = annual_metrics_optimized.get(
        "Annual_Electrolyzer_MWh", 0
    ) * avg_elec_price
    # Add NPP charging to opportunity cost
    opp_cost_elec_annual += opp_cost_battery_annual

    # **NEW: Total opportunity cost includes AS opportunity cost**
    total_opportunity_cost_annual = opp_cost_elec_annual + as_opportunity_cost_annual

    for op_idx in range(total_project_years - construction_period):
        proj_yr_idx = op_idx + construction_period
        op_yr_num = op_idx + 1
        cur_h2_rev = h2_rev_annual - (
            annual_metrics_optimized.get("H2_Subsidy_Revenue", 0)
            if op_yr_num > h2_subsidy_yrs
            else 0
        )

        # **FIXED: Incremental revenue = H2 revenue + AS revenue - AS opportunity cost**
        rev_inc = cur_h2_rev + as_rev_annual - as_opportunity_cost_annual
        costs_inc = vom_annual_inc + opp_cost_elec_annual

        # Incremental Fixed O&M (General + Battery specific)
        fixed_om_inc_general_base = om_components_incremental.get(
            "Fixed_OM_General", {}
        ).get(
            "base_cost", 0
        )  # If there's a general incremental fixed OM
        fixed_om_inc_general_inflation = om_components_incremental.get(
            "Fixed_OM_General", {}
        ).get("inflation_rate", 0)
        costs_inc += fixed_om_inc_general_base * (
            (1 + fixed_om_inc_general_inflation) ** op_idx
        )

        if (
            ENABLE_BATTERY
            and optimized_capacities_inc.get("Battery_Capacity_MWh", 0) > 0
        ):
            batt_fixed_om_per_mw_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}
            ).get("base_cost_per_mw_year", 0)
            batt_fixed_om_per_mwh_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}
            ).get("base_cost_per_mwh_year", 0)
            batt_inflation_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}
            ).get("inflation_rate", 0)
            batt_power_inc = optimized_capacities_inc.get(
                "Battery_Power_MW", 0)
            batt_capacity_inc = optimized_capacities_inc.get(
                "Battery_Capacity_MWh", 0)
            costs_inc += (
                batt_power_inc * batt_fixed_om_per_mw_inc
                + batt_capacity_inc * batt_fixed_om_per_mwh_inc
            ) * ((1 + batt_inflation_inc) ** op_idx)

        # Incremental Replacements
        for (
            rep_comp_name_inc,
            rep_data_inc,
        ) in replacement_schedule_incremental.items():
            if op_yr_num in rep_data_inc.get("years", []):
                cost_val_inc = rep_data_inc.get("cost", 0)
                if (
                    rep_comp_name_inc == "Battery_Augmentation_Replacement"
                    and rep_data_inc.get("cost_percent_initial_capex", 0) > 0
                ):
                    cost_val_inc = (
                        initial_total_inc_battery_capex
                        * rep_data_inc["cost_percent_initial_capex"]
                    )
                costs_inc += cost_val_inc

        profit_inc_pre_tax = rev_inc - costs_inc
        tax_inc = profit_inc_pre_tax * tax_rt if profit_inc_pre_tax > 0 else 0
        pure_incremental_cf[proj_yr_idx] += profit_inc_pre_tax - tax_inc

    inc_metrics["NPV_USD"] = npf.npv(discount_rt, pure_incremental_cf)
    try:
        inc_metrics["IRR_percent"] = (
            npf.irr(pure_incremental_cf) * 100
            if any(cf != 0 for cf in pure_incremental_cf)
            else np.nan
        )
    except:
        inc_metrics["IRR_percent"] = np.nan

    cum_pure_inc_cf = np.cumsum(pure_incremental_cf)
    pos_idx_pure = np.where(cum_pure_inc_cf >= 0)[0]
    if pos_idx_pure.size > 0:
        first_pos = pos_idx_pure[0]
        if first_pos == 0 and pure_incremental_cf[0] >= 0:
            inc_metrics["Payback_Period_Years"] = 0
        elif first_pos > 0 and cum_pure_inc_cf[first_pos - 1] < 0:
            inc_metrics["Payback_Period_Years"] = (
                (first_pos - 1)
                + abs(cum_pure_inc_cf[first_pos - 1])
                / (cum_pure_inc_cf[first_pos] - cum_pure_inc_cf[first_pos - 1])
                - construction_period
                + 1
            )
        else:
            inc_metrics["Payback_Period_Years"] = first_pos - \
                construction_period + 1
    else:
        inc_metrics["Payback_Period_Years"] = np.nan

    # Note: Incremental LCOH calculation removed - use detailed breakdown method instead
    # The detailed LCOH breakdown provides more accurate cost attribution

    inc_metrics["pure_incremental_cash_flows"] = pure_incremental_cf
    inc_metrics["traditional_incremental_cash_flows"] = (
        optimized_cash_flows - baseline_cash_flows
    )
    inc_metrics["Annual_Electricity_Opportunity_Cost_USD"] = total_opportunity_cost_annual
    inc_metrics["Annual_AS_Opportunity_Cost_USD"] = as_opportunity_cost_annual
    inc_metrics["Annual_Baseline_OPEX_USD"] = baseline_annual_opex
    return inc_metrics


def plot_results(
    annual_metrics_data: dict,
    financial_metrics_data: dict,
    cash_flows_data: np.ndarray,
    plot_dir: Path,
    construction_p: int,
    incremental_metrics_data: dict | None = None,
):
    os.makedirs(plot_dir, exist_ok=True)

    try:
        plt.style.use("seaborn")
    except:
        try:
            plt.style.use("ggplot")
        except:
            plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.3
        }
    )
    years_axis = np.arange(1, len(cash_flows_data) + 1)
    cumulative_cf_plot = np.cumsum(cash_flows_data)
    fig, ax1 = plt.subplots()
    bars = ax1.bar(
        years_axis,
        cash_flows_data,
        color="cornflowerblue",
        alpha=0.7,
        label="Annual Cash Flow",
    )
    for i, val in enumerate(cash_flows_data):
        if val < 0:
            bars[i].set_color("salmon")
    ax2 = ax1.twinx()
    ax2.plot(
        years_axis,
        cumulative_cf_plot,
        "forestgreen",
        marker="o",
        markersize=4,
        label="Cumulative Cash Flow",
    )
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_xlabel("Project Year")
    ax1.set_ylabel("Annual Cash Flow (USD)")
    ax2.set_ylabel("Cumulative Cash Flow (USD)")
    if construction_p > 0:
        ax1.axvline(
            construction_p + 0.5,
            color="black",
            linestyle="--",
            lw=1,
            label="Operations Start",
        )
    ax1.set_title("Project Cash Flow Profile", fontweight="bold")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "cash_flow_profile.png", dpi=300)
    plt.close(fig)

    if (
        incremental_metrics_data
        and "pure_incremental_cash_flows" in incremental_metrics_data
    ):
        inc_cf_data = incremental_metrics_data["pure_incremental_cash_flows"]
        fig_inc, ax1_inc = plt.subplots()
        inc_cumulative_cf_plot = np.cumsum(inc_cf_data)
        inc_bars = ax1_inc.bar(
            years_axis,
            inc_cf_data,
            color="mediumpurple",
            alpha=0.7,
            label="Incremental Annual CF",
        )
        for i, val in enumerate(inc_cf_data):
            if val < 0:
                inc_bars[i].set_color("lightcoral")
        ax2_inc = ax1_inc.twinx()
        ax2_inc.plot(
            years_axis,
            inc_cumulative_cf_plot,
            "darkorange",
            marker="s",
            markersize=4,
            label="Cumulative Incremental CF",
        )
        ax1_inc.axhline(0, color="grey", lw=0.8)
        ax1_inc.set_xlabel("Project Year")
        ax1_inc.set_ylabel("Incremental Annual CF (USD)")
        ax2_inc.set_ylabel("Cumulative Incremental CF (USD)")
        if construction_p > 0:
            ax1_inc.axvline(
                construction_p + 0.5,
                color="black",
                linestyle="--",
                lw=1,
                label="Operations Start",
            )
        ax1_inc.set_title(
            "Pure Incremental Cash Flow Profile (H2/Battery System)",
            fontweight="bold",
        )
        inc_handles1, inc_labels1 = ax1_inc.get_legend_handles_labels()
        inc_handles2, inc_labels2 = ax2_inc.get_legend_handles_labels()
        ax1_inc.legend(
            inc_handles1 + inc_handles2, inc_labels1 + inc_labels2, loc="best"
        )
        if "Annual_Electricity_Opportunity_Cost_USD" in incremental_metrics_data:
            ax1_inc.text(
                0.02,
                0.02,
                f"Annual Electricity Opportunity Cost: ${incremental_metrics_data['Annual_Electricity_Opportunity_Cost_USD']:,.0f}",
                transform=ax1_inc.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
            )
        plt.tight_layout()
        plt.savefig(plot_dir / "incremental_cash_flow_profile.png", dpi=300)
        plt.close(fig_inc)

    # Add CAPEX breakdown visualization
    if (
        hasattr(annual_metrics_data, "capex_breakdown")
        and annual_metrics_data["capex_breakdown"]
    ):
        capex_data = annual_metrics_data["capex_breakdown"]
        # Filter out zero values
        capex_filtered = {k: v for k, v in capex_data.items() if v > 1e-3}

        if capex_filtered:
            # CAPEX Pie Chart
            fig_capex_pie, ax_capex_pie = plt.subplots()
            ax_capex_pie.pie(
                capex_filtered.values(),
                labels=[f"{k}\n(${v:,.0f})" for k,
                        v in capex_filtered.items()],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("crest", len(capex_filtered)),
            )
            ax_capex_pie.set_title(
                "CAPEX Breakdown by Component", fontweight="bold")
            ax_capex_pie.axis("equal")
            plt.tight_layout()
            plt.savefig(plot_dir / "capex_breakdown_pie.png", dpi=300)
            plt.close(fig_capex_pie)

            # CAPEX Bar Chart
            fig_capex_bar, ax_capex_bar = plt.subplots()
            capex_items = list(capex_filtered.items())
            # Sort by value descending
            capex_items.sort(key=lambda x: x[1], reverse=True)

            bar_labels = [k for k, v in capex_items]
            bar_values = [v for k, v in capex_items]

            bars = ax_capex_bar.bar(
                bar_labels,
                bar_values,
                color=sns.color_palette("crest", len(capex_items)),
            )
            ax_capex_bar.set_ylabel("Cost (USD)")
            ax_capex_bar.set_title("CAPEX by Component", fontweight="bold")

            # Add value labels on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax_capex_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(bar_values),
                    f"${height:,.0f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / "capex_breakdown_bar.png", dpi=300)
            plt.close(fig_capex_bar)

            # CAPEX and total cost as stacked bar
            if "total_capex" in annual_metrics_data:
                total_capex = annual_metrics_data["total_capex"]
                fig_total, ax_total = plt.subplots()

                # Calculate other costs (total project cost minus CAPEX)
                total_cost = abs(sum(cf for cf in cash_flows_data if cf < 0))
                opex_replacements = (
                    total_cost - total_capex if total_cost > total_capex else 0
                )

                # Create stacked bar chart
                categories = ["Total Project Cost"]
                capex_bar = ax_total.bar(
                    categories, [total_capex], label="CAPEX", color="steelblue"
                )
                opex_bar = ax_total.bar(
                    categories,
                    [opex_replacements],
                    bottom=[total_capex],
                    label="OPEX & Replacements",
                    color="lightcoral",
                )

                ax_total.set_ylabel("Cost (USD)")
                ax_total.set_title("Project Cost Structure", fontweight="bold")
                ax_total.legend()

                # Add value labels
                for bar in [capex_bar, opex_bar]:
                    for rect in bar:
                        height = rect.get_height()
                        if height > 0:
                            ax_total.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                rect.get_y() + height / 2.0,
                                f"${height:,.0f}\n({height/total_cost*100:.1f}%)",
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                            )

                plt.tight_layout()
                plt.savefig(plot_dir / "total_cost_structure.png", dpi=300)
                plt.close(fig_total)

    rev_sources = {
        k: annual_metrics_data.get(k, 0)
        for k in [
            "Energy_Revenue",
            "AS_Revenue",
            "H2_Sales_Revenue",
            "H2_Subsidy_Revenue",
        ]
    }
    rev_plot = {
        k.replace("_Revenue", ""): v for k, v in rev_sources.items() if v > 1e-3
    }
    if rev_plot:
        fig_rev, ax_rev = plt.subplots()
        ax_rev.pie(
            rev_plot.values(),
            labels=[f"{k}\n(${v:,.0f})" for k, v in rev_plot.items()],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("viridis", len(rev_plot)),
        )
        ax_rev.set_title("Annual Revenue Breakdown", fontweight="bold")
        ax_rev.axis("equal")
        plt.tight_layout()
        plt.savefig(plot_dir / "revenue_breakdown.png", dpi=300)
        plt.close(fig_rev)

    opex_sources = {
        k: annual_metrics_data.get(k, 0)
        for k in [
            "VOM_Turbine_Cost",
            "VOM_Electrolyzer_Cost",
            "VOM_Battery_Cost",
            "Startup_Cost",
            "Water_Cost",
            "Ramping_Cost",
            "H2_Storage_Cycle_Cost",
        ]
    }
    opex_sources["Fixed OM (General)"] = OM_COMPONENTS.get("Fixed_OM_General", {}).get(
        "base_cost", 0
    )  # Updated key
    # Add battery fixed OM if applicable
    if ENABLE_BATTERY and annual_metrics_data.get("Battery_Capacity_MWh", 0) > 0:
        batt_om_mw_cost = OM_COMPONENTS.get("Fixed_OM_Battery", {}).get(
            "base_cost_per_mw_year", 0
        ) * annual_metrics_data.get("Battery_Power_MW", 0)
        batt_om_mwh_cost = OM_COMPONENTS.get("Fixed_OM_Battery", {}).get(
            "base_cost_per_mwh_year", 0
        ) * annual_metrics_data.get("Battery_Capacity_MWh", 0)
        opex_sources["Fixed OM (Battery)"] = batt_om_mw_cost + batt_om_mwh_cost

    opex_plot = {k.replace("_Cost", ""): v for k,
                 v in opex_sources.items() if v > 1e-3}
    if opex_plot:
        fig_opex, ax_opex = plt.subplots()
        ax_opex.pie(
            opex_plot.values(),
            labels=[f"{k}\n(${v:,.0f})" for k, v in opex_plot.items()],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("rocket", len(opex_plot)),
        )
        ax_opex.set_title(
            "Annual Operational Cost Breakdown (Base Year)", fontweight="bold"
        )
        ax_opex.axis("equal")
        plt.tight_layout()
        plt.savefig(plot_dir / "opex_cost_breakdown.png", dpi=300)
        plt.close(fig_opex)

    fin_metrics = {
        k: financial_metrics_data.get(k, np.nan)
        for k in [
            "NPV_USD",
            "IRR_percent",
            "Payback_Period_Years",
            "LCOH_USD_per_kg",
        ]
    }
    fin_valid = {
        k.replace("_USD", " (USD)")
        .replace("_percent", " (%)")
        .replace("_Years", " (Years)")
        .replace("_per_kg", " (USD/kg)"): v
        for k, v in fin_metrics.items()
        if not pd.isna(v)
    }
    if fin_valid:
        npv_key = "NPV (USD)" if "NPV (USD)" in fin_valid else None
        npv_value = fin_valid.get("NPV (USD)", None)

        other_metrics = {k: v for k,
                         v in fin_valid.items() if k != "NPV (USD)"}

        fig_fin = plt.figure(figsize=(12, 8))

        if npv_key and npv_value is not None:
            ax_npv = plt.subplot(2, 1, 1)
            npv_bar = ax_npv.barh(
                [npv_key],
                [npv_value],
                color=sns.color_palette("mako", 1)[0]
            )
            ax_npv.set_xlabel("Value (USD)")
            ax_npv.set_title("Net Present Value (NPV)", fontweight="bold")

            ax_npv.text(
                npv_value + 0.01 * abs(npv_value) if npv_value != 0 else 0.01,
                0,
                f"${npv_value:,.2f}",
                va="center",
                ha="left" if npv_value >= 0 else "right",
            )

        if other_metrics:
            ax_other = plt.subplot(2, 1, 2)
            other_bars = ax_other.barh(
                list(other_metrics.keys()),
                list(other_metrics.values()),
                color=sns.color_palette("mako", len(other_metrics))
            )
            ax_other.set_xlabel("Value")
            ax_other.set_title("Other Financial Metrics", fontweight="bold")

            for i, (k, v) in enumerate(other_metrics.items()):
                if "IRR" in k:
                    label_text = f"{v:.2f}%"
                elif "LCOH" in k:
                    label_text = f"${v:.2f}/kg"
                elif "Payback" in k:
                    label_text = f"{v:.2f} years"
                else:
                    label_text = f"{v:.2f}"

                ax_other.text(
                    v + 0.01 * abs(v) if v != 0 else 0.01,
                    i,
                    label_text,
                    va="center",
                    ha="left" if v >= 0 else "right",
                )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(plot_dir / "financial_metrics_summary.png", dpi=300)
        plt.close(fig_fin)

        # Updated to include Electrolyzer CF, Turbine CF, Battery SOC, and H2 Storage SOC
    cf_data = {
        "Electrolyzer_CF_percent": annual_metrics_data.get(
            "Electrolyzer_CF_percent", np.nan
        ),
        "Turbine_CF_percent": annual_metrics_data.get("Turbine_CF_percent", np.nan),
        "Battery_SOC_percent": annual_metrics_data.get("Battery_SOC_percent", np.nan),
        "H2_Storage_SOC_percent": annual_metrics_data.get(
            "H2_Storage_SOC_percent", np.nan
        ),
    }

    # Create a dictionary with friendly names for plotting
    plot_labels = {
        "Electrolyzer_CF_percent": "Electrolyzer CF (%)",
        "Turbine_CF_percent": "Turbine CF (%)",
        "Battery_SOC_percent": "Battery Avg SOC (%)",
        "H2_Storage_SOC_percent": "H2 Storage Avg SOC (%)",
    }

    cf_valid = {
        plot_labels[k]: v
        for k, v in cf_data.items()
        if not pd.isna(v) and v is not None
    }
    if cf_valid:
        fig_cf, ax_cf = plt.subplots(figsize=(10, 6))
        # Use more colors to accommodate possible 4 metrics
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ]
        bars = ax_cf.bar(
            range(len(cf_valid)),
            list(cf_valid.values()),
            color=colors[: len(cf_valid)],
        )
        ax_cf.set_ylabel("Percentage (%)")
        ax_cf.set_title("System Performance Metrics", fontweight="bold")

        # Add value labels and ensure enough space to display
        for i, (k, v) in enumerate(cf_valid.items()):
            # If value is small, label above bar; if value is large, label inside bar
            if v < 10:
                ax_cf.text(i, v + 2, f"{v:.1f}%", ha="center")
            else:
                ax_cf.text(
                    i,
                    v - 5 if v > 15 else v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    color="white" if v > 15 else "black",
                    fontweight="bold",
                )

        # Adjust x-axis labels
        ax_cf.set_xticks(range(len(cf_valid)))
        ax_cf.set_xticklabels(list(cf_valid.keys()), rotation=15, ha="center")

        # Adjust y-axis range to ensure all labels are visible
        ax_cf.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(plot_dir / "capacity_factors.png", dpi=300)
        plt.close(fig_cf)

    # **NEW: Comprehensive LCOH Analysis Dashboard**
    if "lcoh_breakdown_analysis" in annual_metrics_data:
        lcoh_analysis = annual_metrics_data["lcoh_breakdown_analysis"]
        lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
        lcoh_percentages = lcoh_analysis.get("lcoh_percentages", {})
        total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

        if lcoh_breakdown and total_lcoh > 0:
            logger.info("Creating comprehensive LCOH analysis dashboard...")

            # Create a comprehensive 2x3 subplot figure
            fig_lcoh_dashboard = plt.figure(figsize=(24, 16))

            # Filter out very small components (< 1% of total)
            significant_components = {
                k: v for k, v in lcoh_breakdown.items()
                if lcoh_percentages.get(k, 0) >= 1.0
            }

            # Group small components together
            small_components_total = sum(
                v for k, v in lcoh_breakdown.items()
                if lcoh_percentages.get(k, 0) < 1.0
            )

            if small_components_total > 0:
                significant_components["Other (< 1%)"] = small_components_total

                # Sort by value for better visualization
                sorted_components = sorted(
                    significant_components.items(), key=lambda x: x[1], reverse=True)

            # Define color scheme for different cost categories
            def get_component_color(component_name):
                if component_name.startswith("CAPEX_"):
                    if "Electrolyzer" in component_name:
                        return '#1f77b4'  # Blue for Electrolyzer CAPEX
                    elif "H2 Storage" in component_name or "H2_Storage" in component_name:
                        return '#2ca02c'  # Green for H2 Storage CAPEX
                    elif "Battery" in component_name:
                        return '#ff7f0e'  # Orange for Battery CAPEX
                    elif "NPP" in component_name or "Npp" in component_name:
                        return '#d62728'  # Red for NPP Modifications
                    else:
                        return '#9467bd'  # Purple for other CAPEX
                elif "Opportunity_Cost" in component_name or "Direct_Cost" in component_name:
                    return '#ff9999'  # Light red for electricity costs
                elif "Fixed_OM" in component_name:
                    return '#90EE90'  # Light green for O&M
                elif "Replacement" in component_name:
                    return '#FFB6C1'  # Light pink for replacements
                else:
                    return '#DDA0DD'  # Plum for other OPEX

            # Subplot 1: High-Level Breakdown (CAPEX vs OPEX)
            ax1 = plt.subplot(2, 3, 1)
            total_capex_lcoh = sum(
                v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
            total_opex_lcoh = total_lcoh - total_capex_lcoh

            categories = ['CAPEX\n(Capital Recovery)',
                          'OPEX\n(Operating Costs)']
            values = [total_capex_lcoh, total_opex_lcoh]
            colors = ['#1f77b4', '#ff7f0e']

            bars1 = ax1.bar(categories, values, color=colors, alpha=0.8)
            for bar, value in zip(bars1, values):
                percentage = (value / total_lcoh) * 100
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() / 2.0,
                    f"${value:.3f}/kg\n({percentage:.1f}%)",
                    ha="center", va="center", fontweight="bold", color="white", fontsize=11
                )

            ax1.set_ylabel("Cost (USD/kg H2)", fontsize=12)
            ax1.set_title("LCOH High-Level Breakdown",
                          fontweight="bold", fontsize=14)
            ax1.set_ylim(0, max(values) * 1.1)

            # Subplot 2: Detailed Component Breakdown (Pie Chart)
            ax2 = plt.subplot(2, 3, 2)

            # Prepare data for pie chart
            pie_labels = []
            pie_values = []
            pie_colors = []

            for component, value in sorted_components:
                percentage = (value / total_lcoh) * 100
                # Clean up component names for display
                display_name = component.replace(
                    "CAPEX_", "").replace("_", " ").title()
                if "Electricity Opportunity Cost" in display_name:
                    display_name = display_name.replace(
                        "Electricity Opportunity Cost", "Elec. Opp. Cost")
                elif "Npp Modifications" in display_name:
                    display_name = "NPP Modifications"

                pie_labels.append(
                    f"{display_name}\n${value:.3f}/kg\n({percentage:.1f}%)")
                pie_values.append(value)
                pie_colors.append(get_component_color(component))

            wedges, texts, autotexts = ax2.pie(
                pie_values,
                labels=pie_labels,
                colors=pie_colors,
                autopct='',  # We include percentage in labels
                startangle=90,
                textprops={'fontsize': 9}
            )

            ax2.set_title("LCOH Component Breakdown",
                          fontweight="bold", fontsize=14)

            # Subplot 3: CAPEX Components Detail
            ax3 = plt.subplot(2, 3, 4)

            # Filter CAPEX components
            capex_components = {
                k.replace("CAPEX_", ""): v for k, v in sorted_components
                if k.startswith("CAPEX_")
            }

            if capex_components:
                capex_names = list(capex_components.keys())
                capex_values = list(capex_components.values())
                capex_colors = [get_component_color(
                    f"CAPEX_{name}") for name in capex_names]

                bars3 = ax3.bar(range(len(capex_names)),
                                capex_values, color=capex_colors, alpha=0.8)

                # Add value labels
                for i, (bar, value) in enumerate(zip(bars3, capex_values)):
                    percentage = (value / total_lcoh) * 100
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(capex_values) * 0.01,
                        f"${value:.3f}\n({percentage:.1f}%)",
                        ha="center", va="bottom", fontsize=9
                    )

                ax3.set_ylabel("Cost (USD/kg H2)", fontsize=12)
                ax3.set_title("CAPEX Components",
                              fontweight="bold", fontsize=14)
                ax3.set_xticks(range(len(capex_names)))
                ax3.set_xticklabels([name.replace("_", " ").title() for name in capex_names],
                                    rotation=45, ha="right", fontsize=10)

            # Subplot 4: OPEX Components Detail
            ax4 = plt.subplot(2, 3, 5)

            # Filter OPEX components
            opex_components = {
                k: v for k, v in sorted_components
                if not k.startswith("CAPEX_")
            }

            if opex_components:
                opex_names = list(opex_components.keys())
                opex_values = list(opex_components.values())
                opex_colors = [get_component_color(
                    name) for name in opex_names]

                bars4 = ax4.bar(range(len(opex_names)),
                                opex_values, color=opex_colors, alpha=0.8)

                # Add value labels
                for i, (bar, value) in enumerate(zip(bars4, opex_values)):
                    percentage = (value / total_lcoh) * 100
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(opex_values) * 0.01,
                        f"${value:.3f}\n({percentage:.1f}%)",
                        ha="center", va="bottom", fontsize=9
                    )

                ax4.set_ylabel("Cost (USD/kg H2)", fontsize=12)
                ax4.set_title("OPEX Components",
                              fontweight="bold", fontsize=14)
                ax4.set_xticks(range(len(opex_names)))
                ax4.set_xticklabels([name.replace("_", " ").title() for name in opex_names],
                                    rotation=45, ha="right", fontsize=10)

            # Subplot 5: Cost Efficiency Analysis
            ax5 = plt.subplot(2, 3, 3)

            # Calculate cost efficiency metrics
            annual_h2_production = annual_metrics_data.get(
                "Annual_H2_Production_kg", 1)
            electrolyzer_capacity = annual_metrics_data.get(
                "Electrolyzer_Capacity_MW", 1)
            capacity_factor = annual_metrics_data.get(
                "Electrolyzer_Capacity_Factor", 0.5)

            # Calculate key efficiency metrics
            efficiency_metrics = {
                'H2 Production\nRate (kg/day)': annual_h2_production / 365,
                'Capacity\nFactor (%)': capacity_factor * 100,
                'LCOH per\nCapacity Factor': total_lcoh / (capacity_factor * 100) if capacity_factor > 0 else 0,
                'Cost per\nMW Capacity': total_lcoh * annual_h2_production / electrolyzer_capacity if electrolyzer_capacity > 0 else 0,
                'Energy\nEfficiency (%)': annual_metrics_data.get("Electrolyzer_Efficiency", 0.7) * 100
            }

            # Create efficiency metrics bar chart
            metrics_names = list(efficiency_metrics.keys())
            metrics_values = list(efficiency_metrics.values())

            # Normalize values for better visualization (scale to 0-100)
            max_val = max(metrics_values) if metrics_values else 1
            normalized_values = [v / max_val * 100 for v in metrics_values]

            colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB']
            bars5 = ax5.bar(range(len(metrics_names)), normalized_values,
                            color=colors[:len(metrics_names)], alpha=0.8)

            # Add value labels with original values
            for i, (bar, orig_val) in enumerate(zip(bars5, metrics_values)):
                if 'kg/day' in metrics_names[i]:
                    label = f'{orig_val:.0f}'
                elif '%' in metrics_names[i]:
                    label = f'{orig_val:.1f}%'
                elif 'LCOH per' in metrics_names[i]:
                    label = f'${orig_val:.4f}'
                elif 'Cost per' in metrics_names[i]:
                    label = f'${orig_val:.0f}'
                else:
                    label = f'{orig_val:.2f}'

                ax5.text(bar.get_x() + bar.get_width() / 2.0,
                         bar.get_height() + 2,
                         label, ha="center", va="bottom", fontsize=9)

            ax5.set_ylabel("Normalized Score (0-100)", fontsize=10)
            ax5.set_title("Production Efficiency Metrics",
                          fontweight="bold", fontsize=12)
            ax5.set_xticks(range(len(metrics_names)))
            ax5.set_xticklabels(metrics_names, rotation=45,
                                ha="right", fontsize=9)
            ax5.set_ylim(0, 110)
            ax5.grid(True, alpha=0.3)

            # Subplot 6: Cost Structure Comparison
            ax6 = plt.subplot(2, 3, 6)

            # Create a stacked bar showing cost structure
            cost_categories = {
                'Capital\nRecovery': sum(v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_")),
                'Electricity\nCosts': sum(v for k, v in lcoh_breakdown.items()
                                          if "Opportunity_Cost" in k or "Direct_Cost" in k),
                'Fixed\nO&M': sum(v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k),
                'Variable\nOPEX': sum(v for k, v in lcoh_breakdown.items()
                                      if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost",
                                               "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"]),
                'Replacements': sum(v for k, v in lcoh_breakdown.items() if "Replacement" in k),
                'Other': sum(v for k, v in lcoh_breakdown.items()
                             if not any(term in k for term in ["CAPEX_", "Opportunity_Cost", "Direct_Cost",
                                                               "Fixed_OM", "VOM_", "Water_Cost", "Startup_Cost",
                                                               "Ramping_Cost", "H2_Storage_Cycle_Cost", "Replacement"]))
            }

            # Filter out zero values
            cost_categories = {k: v for k,
                               v in cost_categories.items() if v > 1e-6}

            if cost_categories:
                categories = list(cost_categories.keys())
                values = list(cost_categories.values())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                          '#9467bd', '#8c564b'][:len(categories)]

                bars6 = ax6.bar(categories, values, color=colors, alpha=0.8)

                # Add value labels and percentages
                for bar, value in zip(bars6, values):
                    percentage = (value / total_lcoh) * 100
                    ax6.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(values) * 0.01,
                        f"${value:.3f}\n({percentage:.1f}%)",
                        ha="center", va="bottom", fontsize=9
                    )

                ax6.set_ylabel("Cost (USD/kg H2)", fontsize=10)
                ax6.set_title("Cost Structure by Category",
                              fontweight="bold", fontsize=12)
                ax6.set_xticklabels(categories, rotation=45,
                                    ha="right", fontsize=9)

            # Add overall title and summary information
            fig_lcoh_dashboard.suptitle(
                f"Levelized Cost of Hydrogen (LCOH) Analysis\nTotal LCOH: ${total_lcoh:.3f}/kg H2",
                fontsize=18, fontweight="bold", y=0.98
            )

            # Add summary text box
            summary_text = f"""
Key Insights:
• Total LCOH: ${total_lcoh:.3f}/kg H2
• CAPEX Contribution: ${total_capex_lcoh:.3f}/kg ({total_capex_lcoh/total_lcoh*100:.1f}%)
• OPEX Contribution: ${total_opex_lcoh:.3f}/kg ({total_opex_lcoh/total_lcoh*100:.1f}%)
• Number of Cost Components: {len(lcoh_breakdown)}
            """

            fig_lcoh_dashboard.text(0.02, 0.02, summary_text.strip(),
                                    fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                                                           facecolor="lightgray", alpha=0.8),
                                    verticalalignment='bottom')

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3, wspace=0.3)
            plt.savefig(plot_dir / "lcoh_comprehensive_analysis.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig_lcoh_dashboard)

            # **NEW: Create LCOH benchmarking and trends analysis with sensitivity analysis**
            logger.info("Creating LCOH benchmarking analysis...")

            # Create a larger figure with 2x3 subplots to include sensitivity analysis
            fig_benchmark, ((ax_bench, ax_waterfall, ax_tornado), (ax_trends, ax_breakdown_pie, ax_heatmap)) = plt.subplots(
                2, 3, figsize=(24, 12))

            # Subplot 1: LCOH Benchmarking against industry standards
            benchmark_data = {
                'Current LCOH': total_lcoh,
                'DOE 2030 Target': 2.0,  # DOE hydrogen target
                'Steam Methane\nReforming': 1.5,  # Typical SMR cost
                'Grid Electrolysis': 5.0,  # Typical grid electrolysis
                'Renewable\nElectrolysis': 3.5   # Renewable electrolysis
            }

            bench_names = list(benchmark_data.keys())
            bench_values = list(benchmark_data.values())
            bench_colors = ['#FF6B6B', '#4ECDC4',
                            '#45B7D1', '#96CEB4', '#FFEAA7']

            bars_bench = ax_bench.bar(
                bench_names, bench_values, color=bench_colors, alpha=0.8)

            # Add value labels
            for bar, value in zip(bars_bench, bench_values):
                ax_bench.text(bar.get_x() + bar.get_width() / 2.0,
                              bar.get_height() + 0.05,
                              f'${value:.2f}', ha="center", va="bottom", fontsize=10, fontweight='bold')

            ax_bench.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
            ax_bench.set_title("LCOH Benchmarking Analysis",
                               fontweight="bold", fontsize=14)
            ax_bench.set_xticklabels(
                bench_names, rotation=45, ha="right", fontsize=10)
            ax_bench.grid(True, alpha=0.3, axis='y')
            ax_bench.axhline(y=2.0, color='red', linestyle='--',
                             alpha=0.7, label='DOE 2030 Target')
            ax_bench.legend()

            # Subplot 2: LCOH Waterfall Chart
            # Show how each major component contributes to total LCOH
            waterfall_components = [
                ('Base', 0),
                ('CAPEX', sum(v for k, v in lcoh_breakdown.items()
                 if k.startswith("CAPEX_"))),
                ('Electricity', sum(v for k, v in lcoh_breakdown.items()
                 if "Opportunity_Cost" in k or "Direct_Cost" in k)),
                ('Fixed O&M', sum(v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k)),
                ('Variable OPEX', sum(v for k, v in lcoh_breakdown.items()
                                      if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost", "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"])),
                ('Replacements', sum(
                    v for k, v in lcoh_breakdown.items() if "Replacement" in k)),
                ('Total', total_lcoh)
            ]

            # Calculate cumulative values for waterfall
            cumulative = 0
            x_pos = range(len(waterfall_components))
            heights = []
            bottoms = []
            colors_waterfall = []

            for i, (name, value) in enumerate(waterfall_components):
                if name == 'Base':
                    heights.append(0)
                    bottoms.append(0)
                    colors_waterfall.append('lightgray')
                elif name == 'Total':
                    heights.append(total_lcoh)
                    bottoms.append(0)
                    colors_waterfall.append('darkgreen')
                else:
                    heights.append(value)
                    bottoms.append(cumulative)
                    colors_waterfall.append('steelblue')
                    cumulative += value

            bars_waterfall = ax_waterfall.bar(x_pos, heights, bottom=bottoms,
                                              color=colors_waterfall, alpha=0.8)

            # Add connecting lines
            for i in range(1, len(waterfall_components)-1):
                if waterfall_components[i][1] > 0:
                    ax_waterfall.plot([i-0.4, i+0.4], [bottoms[i], bottoms[i]],
                                      'k--', alpha=0.5, linewidth=1)

            # Add value labels
            for i, (bar, (name, value)) in enumerate(zip(bars_waterfall, waterfall_components)):
                if name not in ['Base', 'Total'] and value > 0:
                    ax_waterfall.text(bar.get_x() + bar.get_width() / 2.0,
                                      bar.get_height() + bottoms[i] + 0.02,
                                      f'${value:.3f}', ha="center", va="bottom", fontsize=9)

            ax_waterfall.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
            ax_waterfall.set_title(
                "LCOH Waterfall Analysis", fontweight="bold", fontsize=14)
            ax_waterfall.set_xticks(x_pos)
            ax_waterfall.set_xticklabels([comp[0] for comp in waterfall_components],
                                         rotation=45, ha="right", fontsize=10)
            ax_waterfall.grid(True, alpha=0.3, axis='y')

            # Subplot 3: Cost trends and projections
            years = list(range(2024, 2035))
            # Simulate cost reduction trends based on learning curves
            capex_reduction = [1.0 * (0.95 ** (year - 2024))
                               for year in years]  # 5% annual reduction
            electricity_cost_trend = [
                1.0 * (0.97 ** (year - 2024)) for year in years]  # 3% annual reduction

            current_capex = sum(
                v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
            current_elec = sum(v for k, v in lcoh_breakdown.items(
            ) if "Opportunity_Cost" in k or "Direct_Cost" in k)
            current_other = total_lcoh - current_capex - current_elec

            projected_lcoh = [current_capex * capex_red + current_elec * elec_red + current_other
                              for capex_red, elec_red in zip(capex_reduction, electricity_cost_trend)]

            ax_trends.plot(years, projected_lcoh, 'b-', linewidth=3,
                           marker='o', markersize=6, label='Projected LCOH')
            ax_trends.axhline(y=2.0, color='red', linestyle='--',
                              alpha=0.7, label='DOE 2030 Target')
            ax_trends.axhline(y=total_lcoh, color='green',
                              linestyle='-', alpha=0.7, label='Current LCOH')

            ax_trends.set_xlabel("Year", fontsize=12)
            ax_trends.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
            ax_trends.set_title("LCOH Projection (2024-2034)",
                                fontweight="bold", fontsize=14)
            ax_trends.grid(True, alpha=0.3)
            ax_trends.legend()
            ax_trends.set_ylim(0, max(projected_lcoh) * 1.1)

            # Subplot 4: Enhanced pie chart with cost efficiency indicators
            # Filter significant components for pie chart
            significant_components_pie = {
                k: v for k, v in lcoh_breakdown.items() if v/total_lcoh >= 0.05}
            other_components_sum = sum(
                v for k, v in lcoh_breakdown.items() if v/total_lcoh < 0.05)

            if other_components_sum > 0:
                significant_components_pie['Other Components'] = other_components_sum

            pie_labels_enhanced = []
            pie_values_enhanced = list(significant_components_pie.values())
            pie_colors_enhanced = []

            for component in significant_components_pie.keys():
                clean_name = component.replace(
                    "CAPEX_", "").replace("_", " ").title()
                if "Electricity Opportunity Cost" in clean_name:
                    clean_name = "Electricity Cost"
                elif "Npp Modifications" in clean_name:
                    clean_name = "NPP Modifications"

                percentage = (
                    significant_components_pie[component] / total_lcoh) * 100
                pie_labels_enhanced.append(
                    f"{clean_name}\n${significant_components_pie[component]:.3f}\n({percentage:.1f}%)")
                pie_colors_enhanced.append(get_component_color(component))

            wedges_enhanced, texts_enhanced, autotexts_enhanced = ax_breakdown_pie.pie(
                pie_values_enhanced,
                labels=pie_labels_enhanced,
                colors=pie_colors_enhanced,
                autopct='',
                startangle=90,
                textprops={'fontsize': 9}
            )

            ax_breakdown_pie.set_title(
                "Major LCOH Components\n(>5% of total)", fontweight="bold", fontsize=12)

            # Subplot 5: Tornado Chart for LCOH Sensitivity (top right)
            sensitivity_data = lcoh_analysis.get("sensitivity_analysis", {})
            if sensitivity_data:
                # Prepare data for tornado chart
                tornado_data = []
                for component, sensitivity in sensitivity_data.items():
                    neg_impact = abs(sensitivity["-20%"]["lcoh_change"])
                    pos_impact = sensitivity["+20%"]["lcoh_change"]
                    total_impact = neg_impact + pos_impact

                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Electricity Opportunity Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"

                    tornado_data.append({
                        'component': clean_name,
                        'neg_impact': -neg_impact,
                        'pos_impact': pos_impact,
                        'total_impact': total_impact
                    })

                # Sort by total impact and take top 5 for space
                tornado_data.sort(
                    key=lambda x: x['total_impact'], reverse=True)
                # Limit to top 5 for better visibility
                tornado_data = tornado_data[:5]

                # Create tornado chart
                y_pos = range(len(tornado_data))
                components = [item['component'] for item in tornado_data]
                neg_impacts = [item['neg_impact'] for item in tornado_data]
                pos_impacts = [item['pos_impact'] for item in tornado_data]

                # Use consistent colors with benchmarking theme
                bars_neg = ax_tornado.barh(
                    y_pos, neg_impacts, color='#FF6B6B', alpha=0.8, label='-20% Cost')
                bars_pos = ax_tornado.barh(
                    y_pos, pos_impacts, color='#4ECDC4', alpha=0.8, label='+20% Cost')

                # Add value labels
                for i, (neg, pos) in enumerate(zip(neg_impacts, pos_impacts)):
                    if neg < -0.01:
                        ax_tornado.text(neg - 0.005, i, f'${abs(neg):.3f}',
                                        ha='right', va='center', fontsize=8, fontweight='bold')
                    if pos > 0.01:
                        ax_tornado.text(pos + 0.005, i, f'${pos:.3f}',
                                        ha='left', va='center', fontsize=8, fontweight='bold')

                ax_tornado.set_yticks(y_pos)
                ax_tornado.set_yticklabels(components, fontsize=9)
                ax_tornado.set_xlabel('LCOH Change (USD/kg H2)', fontsize=10)
                ax_tornado.set_title('LCOH Sensitivity Analysis\n(Top 5 Cost Drivers)',
                                     fontweight='bold', fontsize=12)
                ax_tornado.axvline(x=0, color='black',
                                   linestyle='-', alpha=0.5)
                ax_tornado.legend(loc='lower right', fontsize=8)
                ax_tornado.grid(True, alpha=0.3, axis='x')

                # Add current LCOH reference
                ax_tornado.text(0.02, 0.98, f'Current: ${total_lcoh:.3f}/kg',
                                transform=ax_tornado.transAxes, fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="yellow", alpha=0.7),
                                verticalalignment='top')

            # Subplot 6: Multi-Parameter Sensitivity Heatmap (bottom right)
            if sensitivity_data:
                # Create sensitivity matrix for heatmap
                parameters = list(sensitivity_data.keys())[
                    :5]  # Limit to top 5 parameters
                change_levels = ['-20%', '0%', '+20%']  # Simplified for space

                # Calculate LCOH values for different change levels
                sensitivity_matrix = []
                parameter_labels = []

                for param in parameters:
                    clean_name = param.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Elec. Opp. Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"
                    elif len(clean_name) > 15:  # Truncate long names
                        clean_name = clean_name[:12] + "..."
                    parameter_labels.append(clean_name)

                    param_sensitivity = sensitivity_data[param]

                    # Calculate values for change levels
                    row_values = []
                    for change in change_levels:
                        if change == '0%':
                            row_values.append(total_lcoh)
                        elif change in param_sensitivity:
                            row_values.append(
                                param_sensitivity[change]['new_total_lcoh'])
                        else:
                            row_values.append(total_lcoh)

                    sensitivity_matrix.append(row_values)

                # Create heatmap with consistent color scheme
                sensitivity_array = np.array(sensitivity_matrix)

                # Use colors consistent with benchmarking theme
                vmin = sensitivity_array.min()
                vmax = sensitivity_array.max()
                vcenter = total_lcoh

                # Use a color scheme that matches the overall theme
                from matplotlib.colors import TwoSlopeNorm
                norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                im = ax_heatmap.imshow(
                    sensitivity_array, cmap='RdYlBu_r', norm=norm, aspect='auto')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
                cbar.set_label('LCOH (USD/kg)', fontsize=9)
                cbar.ax.tick_params(labelsize=8)

                # Set ticks and labels
                ax_heatmap.set_xticks(range(len(change_levels)))
                ax_heatmap.set_xticklabels(change_levels, fontsize=9)
                ax_heatmap.set_yticks(range(len(parameter_labels)))
                ax_heatmap.set_yticklabels(parameter_labels, fontsize=9)

                # Add text annotations
                for i in range(len(parameter_labels)):
                    for j in range(len(change_levels)):
                        value = sensitivity_array[i, j]
                        # Choose text color based on background
                        text_color = 'white' if abs(
                            value - vcenter) > (vmax - vmin) * 0.3 else 'black'
                        ax_heatmap.text(j, i, f'${value:.3f}', ha='center', va='center',
                                        color=text_color, fontsize=8, fontweight='bold')

                ax_heatmap.set_xlabel('Parameter Change', fontsize=10)
                ax_heatmap.set_ylabel('Cost Parameters', fontsize=10)
                ax_heatmap.set_title('LCOH Sensitivity Heatmap\n(Parameter Impact)',
                                     fontweight='bold', fontsize=12)

                # Add reference
                ax_heatmap.text(0.02, 0.98, f'Base: ${total_lcoh:.3f}/kg',
                                transform=ax_heatmap.transAxes, fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor="white", alpha=0.8),
                                verticalalignment='top')
            else:
                # If no sensitivity data, show a message
                ax_tornado.text(0.5, 0.5, 'No Sensitivity\nData Available',
                                transform=ax_tornado.transAxes, ha='center', va='center',
                                fontsize=12, fontweight='bold')
                ax_tornado.set_title(
                    'LCOH Sensitivity Analysis', fontweight='bold', fontsize=12)

                ax_heatmap.text(0.5, 0.5, 'No Sensitivity\nData Available',
                                transform=ax_heatmap.transAxes, ha='center', va='center',
                                fontsize=12, fontweight='bold')
                ax_heatmap.set_title(
                    'LCOH Sensitivity Heatmap', fontweight='bold', fontsize=12)

            plt.tight_layout()
            plt.savefig(plot_dir / "lcoh_benchmarking_analysis.png",
                        dpi=300, bbox_inches='tight')
            plt.close(fig_benchmark)

            logger.info("LCOH benchmarking analysis created successfully.")

            logger.info(
                "Comprehensive LCOH analysis dashboard created successfully.")

    logger.info(f"Plots saved to {plot_dir}")


def generate_report(
    annual_metrics_rpt: dict,
    financial_metrics_rpt: dict,
    output_file_path: Path,
    target_iso_rpt: str,
    capex_data: dict,
    om_data: dict,
    replacement_data: dict,
    project_lt_rpt: int,
    construction_p_rpt: int,
    discount_rt_rpt: float,
    tax_rt_rpt: float,
    incremental_metrics_rpt: dict | None = None,
):
    logger.info(f"Generating TEA report: {output_file_path}")

    # **ENHANCEMENT: Use plant-specific title if available**
    current_module = sys.modules[__name__]
    if hasattr(current_module, 'PLANT_REPORT_TITLE'):
        report_title = getattr(current_module, 'PLANT_REPORT_TITLE')
        subtitle_info = f"ISO Region: {target_iso_rpt}"
        logger.info(f"Using plant-specific report title: {report_title}")
    else:
        report_title = target_iso_rpt
        subtitle_info = f"Target ISO: {target_iso_rpt}"
        logger.info(f"Using default ISO report title: {report_title}")

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(
            f"Technical Economic Analysis Report - {report_title}\n"
            + "=" * (30 + len(report_title))
            + "\n\n"
        )
        f.write("1. Project Configuration\n" + "-" * 25 + "\n")
        f.write(f"  ISO Region                      : {target_iso_rpt}\n")
        f.write(
            f"  Project Lifetime                : {project_lt_rpt} years\n")
        f.write(
            f"  Construction Period             : {construction_p_rpt} years\n")
        f.write(
            f"  Discount Rate                   : {discount_rt_rpt*100:.2f}%\n")
        f.write(f"  Corporate Tax Rate              : {tax_rt_rpt*100:.1f}%\n")

        # **ENHANCEMENT: Add plant-specific technical parameters**
        if annual_metrics_rpt:
            # Add Turbine Capacity (moved from System Capacities section)
            turbine_capacity = annual_metrics_rpt.get("Turbine_Capacity_MW", 0)
            if turbine_capacity > 0:
                f.write(
                    f"  Turbine Capacity                : {turbine_capacity:,.2f} MW\n")

            # Add Thermal Capacity if available
            thermal_capacity = annual_metrics_rpt.get(
                "thermal_capacity_mwt", 0)
            if thermal_capacity == 0:
                thermal_capacity = annual_metrics_rpt.get(
                    "Thermal_Capacity_MWt", 0)
            if thermal_capacity > 0:
                f.write(
                    f"  Thermal Capacity                : {thermal_capacity:,.2f} MWt\n")

            # Add Thermal Efficiency if available
            thermal_efficiency = annual_metrics_rpt.get(
                "thermal_efficiency", 0)
            if thermal_efficiency == 0:
                thermal_efficiency = annual_metrics_rpt.get(
                    "Thermal_Efficiency", 0)
            if thermal_efficiency > 0:
                f.write(
                    f"  Thermal Efficiency              : {thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)\n")

        f.write("\n")

        # Add CAPEX breakdown section
        f.write("2. Capital Expenditure (CAPEX) Breakdown\n" + "-" * 42 + "\n")
        if annual_metrics_rpt and "capex_breakdown" in annual_metrics_rpt:
            capex_breakdown = annual_metrics_rpt["capex_breakdown"]
            total_capex = annual_metrics_rpt.get(
                "total_capex", sum(capex_breakdown.values())
            )

            # Sort by values in descending order
            for component, cost in sorted(
                capex_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                if cost > 0:
                    percentage = (cost / total_capex *
                                  100) if total_capex > 0 else 0
                    f.write(
                        f"  {component:<30}             : ${cost:,.0f} ({percentage:.1f}%)\n")

            f.write(
                f"  \n  Total CAPEX                     : ${total_capex:,.0f}\n\n")
        else:
            f.write("  No CAPEX breakdown data available.\n\n")

        # Add a new section for actual capacity values used in calculations
        f.write("3. Optimization Results - System Capacities\n" + "-" * 45 + "\n")
        if annual_metrics_rpt:
            # Show the actual capacity values that were used for calculations
            capacity_metrics = {
                "Electrolyzer Capacity": annual_metrics_rpt.get(
                    "Electrolyzer_Capacity_MW", 0
                ),
                "Hydrogen Storage Capacity": annual_metrics_rpt.get(
                    "H2_Storage_Capacity_kg", 0
                ),
                "Battery Energy Capacity": annual_metrics_rpt.get(
                    "Battery_Capacity_MWh", 0
                ),
                "Battery Power Capacity": annual_metrics_rpt.get("Battery_Power_MW", 0),
            }
            capacity_units = {
                "Electrolyzer Capacity": "MW",
                "Hydrogen Storage Capacity": "kg",
                "Battery Energy Capacity": "MWh",
                "Battery Power Capacity": "MW",
            }

            for name, value in capacity_metrics.items():
                unit = capacity_units.get(name, "")
                f.write(f"  {name:<30}          : {value:,.2f} {unit}\n")

            # Add hydrogen constant sales rate if available (for optimal storage sizing mode)
            h2_constant_sales_rate = annual_metrics_rpt.get(
                "Optimal_H2_Constant_Sales_Rate_kg_hr", 0)
            if h2_constant_sales_rate == 0:
                h2_constant_sales_rate = annual_metrics_rpt.get(
                    "H2_Constant_Sales_Rate_kg_hr", 0)

            if h2_constant_sales_rate > 0:
                f.write(
                    f"  Optimal H2 Constant Sales Rate  : {h2_constant_sales_rate:,.2f} kg/hr\n")

                # Calculate and show daily/annual production rates
                daily_sales = h2_constant_sales_rate * 24
                annual_sales = daily_sales * 365
                f.write(
                    f"  Optimal H2 Daily Sales Rate     : {daily_sales:,.2f} kg/day\n")
                f.write(
                    f"  Optimal H2 Annual Sales Rate    : {annual_sales:,.0f} kg/year\n")

            # **REMOVED: Reference capacities section as requested**
            f.write("\n")

        # 4. Representative Annual Performance (from Optimization)
        f.write(
            "4. Representative Annual Performance (from Optimization)\n"
            + "-" * 58
            + "\n"
        )
        if annual_metrics_rpt:
            # Only show AS results for services provided by this ISO
            iso_service_map = {
                "SPP": ["RegU", "RegD", "Spin", "Sup", "RamU", "RamD", "UncU"],
                "CAISO": ["RegU", "RegD", "Spin", "NSpin", "RMU", "RMD"],
                "ERCOT": ["RegU", "RegD", "Spin", "NSpin", "ECRS"],
                "PJM": ["RegUp", "RegDown", "Syn", "Rse", "TMR"],
                "NYISO": ["RegUp", "RegDown", "Spin10", "NSpin10", "Res30"],
                "ISONE": ["Spin10", "NSpin10", "OR30"],
                "MISO": ["RegUp", "RegDown", "Spin", "Sup", "STR", "RamU", "RamD"],
            }
            iso = target_iso_rpt
            as_services = iso_service_map.get(iso, [])
            metrics_to_skip = [
                "capex_breakdown",
                "total_capex",
                "Electrolyzer_Capacity_MW",
                "H2_Storage_Capacity_kg",
                "Battery_Capacity_MWh",
                "Battery_Power_MW",
                "Turbine_Capacity_MW",
                # Skip AS revenue here as it will be shown in the AS section
                "AS_Revenue",
                "Revenue_Ancillary_USD",
                # Skip battery charging metrics here as they will be shown in battery section
                "Annual_Battery_Charge_MWh",
                "Annual_Battery_Charge_From_Grid_MWh",
                "Annual_Battery_Charge_From_NPP_MWh",
            ]
            for k, v in sorted(annual_metrics_rpt.items()):
                # Only show AS-related metrics for this ISO's services
                if any(
                    k.startswith(f"AS_Max_Bid_{s}") or k.startswith(f"AS_Avg_Bid_{s}") or
                    k.startswith(f"AS_Total_Deployed_{s}") or k.startswith(f"AS_Avg_Deployed_{s}") or
                    k.startswith(f"AS_Deployment_Efficiency_{s}")
                    for s in as_services
                ):
                    # Format AS metric names properly
                    display_name = k.replace('_', ' ').replace(
                        'As ', 'AS ').replace('Mw', 'MW').replace('Mwh', 'MWh')
                    if isinstance(v, (int, float)) and not pd.isna(v):
                        if 'Revenue' in k or 'Cost' in k:
                            formatted_value = f"${v:,.2f}"
                        elif k.endswith('_percent') or 'Utilization' in k:
                            formatted_value = f"{v:.2f}%" if v < 10 else f"{v:.0f}%"
                        elif 'Hours' in k:
                            formatted_value = f"{v:,.0f}"
                        else:
                            formatted_value = f"{v:,.2f}"
                    else:
                        formatted_value = str(v)
                    f.write(f"  {display_name:<45}: {formatted_value}\n")
                # Show non-AS metrics as usual
                elif not any(x in k for x in ["AS_Max_Bid_", "AS_Avg_Bid_", "AS_Total_Deployed_", "AS_Avg_Deployed_", "AS_Deployment_Efficiency_"]) and k not in metrics_to_skip:
                    # Format non-AS metric names properly
                    display_name = k.replace('_', ' ').replace('Cf ', 'Capacity Factor ').replace('Soc ', 'SOC ').replace('Vom ', 'VOM ').replace(
                        'Opex ', 'OPEX ').replace('Capex', 'CAPEX').replace('Mw', 'MW').replace('Mwh', 'MWh').replace('Usd', 'USD')
                    if isinstance(v, (int, float)) and not pd.isna(v):
                        if 'Revenue' in k or 'Cost' in k or 'USD' in k:
                            formatted_value = f"${v:,.2f}"
                        elif k.endswith('_percent') or 'Percent' in display_name:
                            formatted_value = f"{v:.2f}%"
                        elif 'Hours' in k:
                            formatted_value = f"{v:,.0f}"
                        elif 'Price' in k and 'USD' in k:
                            formatted_value = f"${v:.2f}"
                        else:
                            formatted_value = f"{v:,.2f}"
                    else:
                        formatted_value = str(v)
                    f.write(f"  {display_name:<45}: {formatted_value}\n")
        else:
            f.write("  No annual metrics data available.\n")

        # **NEW: Battery Performance and Charging Analysis Section**
        battery_capacity = annual_metrics_rpt.get(
            "Battery_Capacity_MWh", 0) if annual_metrics_rpt else 0
        if battery_capacity > 0:
            f.write(
                "\n4.1. Battery Performance and Charging Analysis\n" + "-" * 47 + "\n")

            # Battery charging breakdown
            total_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_MWh", 0)
            grid_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_Grid_MWh", 0)
            npp_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_NPP_MWh", 0)

            f.write(f"  Battery Charging Electricity Consumption:\n")
            f.write(
                f"    Total Annual Charging: {total_charge:,.2f} MWh/year\n")

            if total_charge > 0:
                grid_pct = (grid_charge / total_charge) * 100
                npp_pct = (npp_charge / total_charge) * 100
                f.write(
                    f"    From Grid Purchase: {grid_charge:,.2f} MWh/year ({grid_pct:.1f}%)\n")
                f.write(
                    f"    From NPP (Opportunity Cost): {npp_charge:,.2f} MWh/year ({npp_pct:.1f}%)\n")

            # Battery utilization metrics
            battery_cf = annual_metrics_rpt.get("Battery_CF_percent", 0)
            battery_soc = annual_metrics_rpt.get("Battery_SOC_percent", 0)

            f.write(f"\n  Battery Utilization:\n")
            f.write(f"    Capacity Factor: {battery_cf:.2f}%\n")
            f.write(f"    Average State of Charge: {battery_soc:.2f}%\n")

            # Economic implications
            avg_price = annual_metrics_rpt.get(
                "Avg_Electricity_Price_USD_per_MWh", 0)
            if avg_price > 0 and total_charge > 0:
                total_charge_cost = total_charge * avg_price
                grid_charge_cost = grid_charge * avg_price
                npp_opportunity_cost = npp_charge * avg_price

                f.write(
                    f"\n  Economic Impact (at avg price ${avg_price:.2f}/MWh):\n")
                f.write(
                    f"    Total Charging Cost: ${total_charge_cost:,.2f}/year\n")
                f.write(
                    f"    Direct Cost (Grid): ${grid_charge_cost:,.2f}/year\n")
                f.write(
                    f"    Opportunity Cost (NPP): ${npp_opportunity_cost:,.2f}/year\n")

        # 5. Ancillary Services Performance
        f.write("\n5. Ancillary Services Performance\n" + "-" * 35 + "\n")
        if annual_metrics_rpt:
            # Total AS revenue - use the new calculated metrics
            as_revenue = annual_metrics_rpt.get("AS_Revenue_Total", 0)
            if as_revenue == 0:
                as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)
            if as_revenue == 0:
                as_revenue = annual_metrics_rpt.get("Revenue_Ancillary_USD", 0)

            f.write(
                f"  Total Ancillary Services Revenue            : ${as_revenue:,.2f}\n")

            # AS revenue statistics
            as_avg_hourly = annual_metrics_rpt.get(
                "AS_Revenue_Average_Hourly", 0)
            as_max_hourly = annual_metrics_rpt.get(
                "AS_Revenue_Maximum_Hourly", 0)
            as_utilization = annual_metrics_rpt.get(
                "AS_Revenue_Utilization_Rate", 0)

            if as_avg_hourly > 0 or as_max_hourly > 0:
                f.write(
                    f"  Average Hourly AS Revenue                   : ${as_avg_hourly:,.2f}\n")
                f.write(
                    f"  Maximum Hourly AS Revenue                   : ${as_max_hourly:,.2f}\n")
                f.write(
                    f"  AS Revenue Utilization Rate                 : {as_utilization:.1f}%\n")

            # Revenue per MW statistics
            as_rev_per_mw_elec = annual_metrics_rpt.get(
                "AS_Revenue_per_MW_Electrolyzer", 0)
            as_rev_per_mw_batt = annual_metrics_rpt.get(
                "AS_Revenue_per_MW_Battery", 0)
            if as_rev_per_mw_elec > 0:
                f.write(
                    f"  AS Revenue per MW Electrolyzer              : ${as_rev_per_mw_elec:,.2f}/MW\n")
            if as_rev_per_mw_batt > 0:
                f.write(
                    f"  AS Revenue per MW Battery Power             : ${as_rev_per_mw_batt:,.2f}/MW\n")

            # Revenue breakdown by AS type if available
            as_revenue_types = {
                "Energy Revenue": annual_metrics_rpt.get("Energy_Revenue", 0),
                "H2 Sales Revenue": annual_metrics_rpt.get("H2_Sales_Revenue", 0),
                "H2 Subsidy Revenue": annual_metrics_rpt.get("H2_Subsidy_Revenue", 0),
                "Ancillary Services Revenue": as_revenue,
            }

            total_revenue = annual_metrics_rpt.get("Annual_Revenue", 0)
            if total_revenue > 0:
                f.write(f"\n  Revenue Mix:\n")
                for rev_type, rev_amount in as_revenue_types.items():
                    if rev_amount > 0:
                        percentage = (rev_amount / total_revenue) * 100
                        f.write(
                            f"    {rev_type:<30}            : ${rev_amount:,.2f} ({percentage:.1f}%)\n")

            # AS revenue as percentage of total revenue
            if total_revenue > 0 and as_revenue > 0:
                as_percentage = (as_revenue / total_revenue) * 100
                f.write(
                    f"\n  AS Revenue as % of Total Revenue            : {as_percentage:.2f}%\n")

            # AS Bidding Statistics
            as_total_services = annual_metrics_rpt.get(
                "AS_Total_Bid_Services", 0)
            as_total_bid_capacity = annual_metrics_rpt.get(
                "AS_Total_Max_Bid_Capacity_MW", 0)
            as_bid_utilization = annual_metrics_rpt.get(
                "AS_Bid_Utilization_vs_Electrolyzer", 0)

            if as_total_services > 0 or as_total_bid_capacity > 0:
                f.write(f"\n  AS Bidding Performance:\n")
                if as_total_services > 0:
                    f.write(
                        f"    Number of AS Services Bid                 : {as_total_services}\n")
                if as_total_bid_capacity > 0:
                    f.write(
                        f"    Total Maximum Bid Capacity                : {as_total_bid_capacity:,.2f} MW\n")
                if as_bid_utilization > 0:
                    f.write(
                        f"    Bid Capacity vs Electrolyzer             : {as_bid_utilization:.1f}%\n")

            # Individual AS service breakdown
            as_service_metrics = {}
            for key, value in annual_metrics_rpt.items():
                if key.startswith("AS_Max_Bid_") and key.endswith("_MW"):
                    service_name = key.replace(
                        "AS_Max_Bid_", "").replace("_MW", "")
                    avg_key = f"AS_Avg_Bid_{service_name}_MW"
                    avg_value = annual_metrics_rpt.get(avg_key, 0)
                    if value > 0 or avg_value > 0:
                        as_service_metrics[service_name] = {
                            "max_bid": value,
                            "avg_bid": avg_value
                        }

            if as_service_metrics:
                f.write(f"\n  AS Service Breakdown:\n")
                for service, data in sorted(as_service_metrics.items()):
                    f.write(
                        f"    {service:<15}: Max {data['max_bid']:>6.1f} MW, Avg {data['avg_bid']:>6.1f} MW\n")

            # Deployment statistics (if available)
            as_total_deployed = annual_metrics_rpt.get(
                "AS_Total_Deployed_Energy_MWh", 0)
            if as_total_deployed > 0:
                f.write(f"\n  AS Deployment Performance:\n")
                f.write(
                    f"    Total Deployed Energy                     : {as_total_deployed:,.2f} MWh\n")

                # Individual deployment metrics
                deployment_metrics = {}
                for key, value in annual_metrics_rpt.items():
                    if key.startswith("AS_Total_Deployed_") and key.endswith("_MWh"):
                        service_name = key.replace(
                            "AS_Total_Deployed_", "").replace("_MWh", "")
                        avg_key = f"AS_Avg_Deployed_{service_name}_MW"
                        eff_key = f"AS_Deployment_Efficiency_{service_name}_percent"
                        avg_value = annual_metrics_rpt.get(avg_key, 0)
                        eff_value = annual_metrics_rpt.get(eff_key, 0)
                        if value > 0:
                            deployment_metrics[service_name] = {
                                "total_mwh": value,
                                "avg_mw": avg_value,
                                "efficiency": eff_value
                            }

                if deployment_metrics:
                    f.write(f"    \n    Service-Specific Deployment:\n")
                    for service, data in sorted(deployment_metrics.items()):
                        eff_str = f", Efficiency {data['efficiency']:.1f}%" if data['efficiency'] > 0 else ""
                        f.write(
                            f"      {service:<30}                      : {data['total_mwh']:>10,.1f} MWh    (Avg {data['avg_mw']:>6.1f} MW{eff_str})\n")

            # System utilization metrics
            capacity_factors = {
                "Electrolyzer Capacity Factor": annual_metrics_rpt.get("Electrolyzer_CF_percent", 0),
                "Turbine Capacity Factor": annual_metrics_rpt.get("Turbine_CF_percent", 0),
                "Battery SOC": annual_metrics_rpt.get("Battery_SOC_percent", 0),
            }

            non_zero_cfs = {k: v for k, v in capacity_factors.items() if v > 0}
            if non_zero_cfs:
                f.write(f"\n  System Utilization (affects AS capability):\n")
                for cf_name, cf_value in non_zero_cfs.items():
                    f.write(
                        f"    {cf_name:<30}              : {cf_value:.2f}%\n")
        else:
            f.write("  No ancillary services data available.\n")

        # 6. Lifecycle Financial Metrics (Total System)
        f.write("\n6. Lifecycle Financial Metrics (Total System)\n" + "-" * 46 + "\n")
        if financial_metrics_rpt:
            # Use pre-calculated ROI if available, otherwise calculate it
            roi = financial_metrics_rpt.get("ROI")
            if roi is None or pd.isna(roi):
                npv = financial_metrics_rpt.get("NPV_USD")
                total_capex = annual_metrics_rpt.get(
                    "total_capex") if annual_metrics_rpt else None
                if npv is not None and total_capex and total_capex > 0:
                    roi = npv / total_capex
                else:
                    roi = None

            for k, v in sorted(financial_metrics_rpt.items()):
                if k == "IRR_percent":
                    lbl = "IRR (%)"
                elif k == "LCOH_USD_per_kg":
                    lbl = "LCOH (USD/kg)"
                elif k == "NPV_USD":
                    lbl = "NPV (USD)"
                elif k == "Payback_Period_Years":
                    lbl = "Payback Period (Years)"
                else:
                    lbl = (
                        k.replace("_USD", " (USD)")
                        .replace("_percent", " (%)")
                        .replace("_Years", " (Years)")
                        .replace("_per_kg", " (USD/kg)")
                        .replace("_", " ")
                        .title()
                    )

                if isinstance(v, (int, float)) and not pd.isna(v):
                    if k == "IRR_percent":
                        formatted_value = f"{v:.2f}%"
                    elif k == "LCOH_USD_per_kg":
                        formatted_value = f"${v:.3f}"
                    elif "USD" in k:
                        formatted_value = f"${v:,.2f}"
                    else:
                        formatted_value = f"{v:,.2f}"
                else:
                    formatted_value = str(v)

                f.write(f"  {lbl:<45}: {formatted_value}\n")

            if roi is not None:
                f.write(
                    f"  Return on Investment (ROI)                  : {roi:.4f}\n")
        else:
            f.write("  No financial metrics data available.\n")

        # **NEW: Detailed LCOH Analysis Section**
        if annual_metrics_rpt and "lcoh_breakdown_analysis" in annual_metrics_rpt:
            f.write(
                "\n6.1. Detailed Levelized Cost of Hydrogen (LCOH) Analysis\n" + "-" * 58 + "\n")

            lcoh_analysis = annual_metrics_rpt["lcoh_breakdown_analysis"]
            lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
            lcoh_percentages = lcoh_analysis.get("lcoh_percentages", {})
            total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

            f.write(f"  Total LCOH: ${total_lcoh:.3f}/kg H2\n\n")

            # LCOH Component Breakdown
            f.write("  LCOH Component Breakdown:\n")
            sorted_components = sorted(
                lcoh_breakdown.items(), key=lambda x: x[1], reverse=True)

            for component, cost in sorted_components:
                if cost > 0.001:  # Only show significant components
                    percentage = lcoh_percentages.get(component, 0)
                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Electricity Opportunity Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"

                    f.write(
                        f"    {clean_name:<35}: ${cost:>8.3f}/kg ({percentage:>5.1f}%)\n")

            # Cost Category Analysis
            f.write("\n  Cost Category Analysis:\n")

            # Calculate category totals
            capex_total = sum(
                v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
            electricity_total = sum(v for k, v in lcoh_breakdown.items()
                                    if "Opportunity_Cost" in k or "Direct_Cost" in k)
            fixed_om_total = sum(
                v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k)
            variable_opex_total = sum(v for k, v in lcoh_breakdown.items()
                                      if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost",
                                               "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"])
            replacement_total = sum(
                v for k, v in lcoh_breakdown.items() if "Replacement" in k)

            categories = [
                ("Capital Recovery (CAPEX)", capex_total),
                ("Electricity Costs", electricity_total),
                ("Fixed O&M", fixed_om_total),
                ("Variable OPEX", variable_opex_total),
                ("Equipment Replacements", replacement_total)
            ]

            for cat_name, cat_cost in categories:
                if cat_cost > 0.001:
                    cat_percentage = (cat_cost / total_lcoh) * \
                        100 if total_lcoh > 0 else 0
                    f.write(
                        f"    {cat_name:<35}: ${cat_cost:>8.3f}/kg ({cat_percentage:>5.1f}%)\n")

            # LCOH Benchmarking
            f.write("\n  LCOH Benchmarking:\n")
            benchmarks = [
                ("DOE 2030 Target", 2.0),
                ("Steam Methane Reforming (typical)", 1.5),
                ("Grid Electrolysis (typical)", 5.0),
                ("Renewable Electrolysis (typical)", 3.5)
            ]

            for bench_name, bench_value in benchmarks:
                comparison = "✓ Below" if total_lcoh < bench_value else "✗ Above"
                difference = abs(total_lcoh - bench_value)
                f.write(
                    f"    vs {bench_name:<30}: {comparison} by ${difference:.3f}/kg\n")

            # Cost Efficiency Metrics
            if annual_metrics_rpt:
                f.write("\n  Cost Efficiency Metrics:\n")

                annual_h2_production = annual_metrics_rpt.get(
                    "Annual_H2_Production_kg", 0)
                electrolyzer_capacity = annual_metrics_rpt.get(
                    "Electrolyzer_Capacity_MW", 0)
                capacity_factor = annual_metrics_rpt.get(
                    "Electrolyzer_Capacity_Factor", 0)

                if annual_h2_production > 0:
                    daily_production = annual_h2_production / 365
                    f.write(
                        f"    Daily H2 Production Rate            : {daily_production:,.0f} kg/day\n")

                if capacity_factor > 0:
                    f.write(
                        f"    Electrolyzer Capacity Factor        : {capacity_factor*100:.1f}%\n")
                    lcoh_per_cf = total_lcoh / \
                        (capacity_factor * 100) if capacity_factor > 0 else 0
                    f.write(
                        f"    LCOH per Capacity Factor Point      : ${lcoh_per_cf:.4f}/kg per 1%\n")

                if electrolyzer_capacity > 0 and annual_h2_production > 0:
                    specific_production = annual_h2_production / electrolyzer_capacity
                    f.write(
                        f"    Specific H2 Production              : {specific_production:,.0f} kg/MW/year\n")

                    cost_per_mw_capacity = total_lcoh * annual_h2_production / electrolyzer_capacity
                    f.write(
                        f"    Cost per MW Electrolyzer Capacity   : ${cost_per_mw_capacity:,.0f}/MW\n")

                # Energy efficiency
                electrolyzer_efficiency = annual_metrics_rpt.get(
                    "Electrolyzer_Efficiency", 0)
                if electrolyzer_efficiency > 0:
                    f.write(
                        f"    Electrolyzer Energy Efficiency      : {electrolyzer_efficiency*100:.1f}%\n")

            # Sensitivity insights (if available)
            sensitivity_data = lcoh_analysis.get("sensitivity_analysis", {})
            if sensitivity_data:
                f.write("\n  Key Cost Drivers (Sensitivity Analysis):\n")

                # Find top 3 most sensitive components
                sensitivity_ranking = []
                for component, sensitivity in sensitivity_data.items():
                    impact = abs(
                        sensitivity["-20%"]["lcoh_change"]) + sensitivity["+20%"]["lcoh_change"]
                    sensitivity_ranking.append((component, impact))

                sensitivity_ranking.sort(key=lambda x: x[1], reverse=True)

                for i, (component, impact) in enumerate(sensitivity_ranking[:3]):
                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Electricity Opportunity Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"

                    neg_impact = abs(
                        sensitivity_data[component]["-20%"]["lcoh_change"])
                    pos_impact = sensitivity_data[component]["+20%"]["lcoh_change"]

                    f.write(
                        f"    {i+1}. {clean_name:<30}: ±20% cost → -${neg_impact:.3f}/+${pos_impact:.3f}/kg\n")

            # Cost reduction opportunities
            f.write("\n  Potential Cost Reduction Opportunities:\n")

            # Identify largest cost components for reduction focus
            top_components = sorted_components[:3]
            for i, (component, cost) in enumerate(top_components):
                if cost > 0.1:  # Only suggest for significant components
                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        suggestion = "Optimize electrolyzer operation during low-price periods"
                    elif "Electrolyzer" in clean_name:
                        suggestion = "Technology learning curve, economies of scale"
                    elif "Battery" in clean_name:
                        suggestion = "Battery cost reduction trends, optimal sizing"
                    elif "Npp Modifications" in clean_name:
                        suggestion = "Standardized modification packages, shared costs"
                    else:
                        suggestion = "Process optimization, bulk purchasing"

                    potential_reduction = cost * 0.2  # Assume 20% reduction potential
                    new_lcoh = total_lcoh - potential_reduction

                    f.write(
                        f"    {i+1}. {clean_name} (-20%): ${new_lcoh:.3f}/kg (${potential_reduction:.3f} reduction)\n")
                    f.write(f"       Strategy: {suggestion}\n")

        if incremental_metrics_rpt:
            f.write(
                "\n7. Incremental Financial Metrics (H2/Battery System vs. Baseline)\n"
                + "-" * 68
                + "\n"
            )

            # **NEW: Baseline analysis section**
            baseline_opex = incremental_metrics_rpt.get(
                "Annual_Baseline_OPEX_USD", 0)
            if baseline_opex > 0:
                f.write("  Baseline Nuclear Plant Analysis:\n")
                # Calculate baseline revenue from total system analysis
                total_energy_rev = annual_metrics_rpt.get("Energy_Revenue", 0)
                f.write(
                    f"    Annual Baseline Revenue (Electricity Sales): ${total_energy_rev:,.2f}\n")
                f.write(
                    f"    Annual Baseline OPEX (Turbine VOM): ${baseline_opex:,.2f}\n")
                baseline_profit = total_energy_rev - baseline_opex
                f.write(
                    f"    Annual Baseline Profit: ${baseline_profit:,.2f}\n")
                if total_energy_rev > 0:
                    margin = (baseline_profit / total_energy_rev) * 100
                    f.write(f"    Baseline Profit Margin: {margin:.1f}%\n")
                f.write("\n")

            # **NEW: Battery charging cost breakdown in incremental analysis**
            if annual_metrics_rpt and annual_metrics_rpt.get("Battery_Capacity_MWh", 0) > 0:
                f.write("  Battery Charging Cost Analysis:\n")

                # Get battery charging data
                total_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_MWh", 0)
                grid_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_Grid_MWh", 0)
                npp_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_NPP_MWh", 0)
                avg_price = annual_metrics_rpt.get(
                    "Avg_Electricity_Price_USD_per_MWh", 0)

                if total_charge > 0 and avg_price > 0:
                    grid_cost = grid_charge * avg_price
                    npp_opportunity_cost = npp_charge * avg_price

                    f.write(
                        f"    Direct Operating Cost (Grid Charging): ${grid_cost:,.2f}/year\n")
                    f.write(
                        f"    Opportunity Cost (NPP Charging): ${npp_opportunity_cost:,.2f}/year\n")
                    f.write(
                        f"    Total Battery Charging Cost: ${grid_cost + npp_opportunity_cost:,.2f}/year\n")

                    if total_charge > 0:
                        f.write(
                            f"    Cost Breakdown: {grid_charge/total_charge*100:.1f}% Direct, {npp_charge/total_charge*100:.1f}% Opportunity\n")
                f.write("\n")

            # **NEW: AS opportunity cost analysis**
            as_opp_cost = incremental_metrics_rpt.get(
                "Annual_AS_Opportunity_Cost_USD", 0)
            if as_opp_cost > 0:
                f.write("  Ancillary Services Opportunity Cost Analysis:\n")
                as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)
                net_as_benefit = as_revenue - as_opp_cost
                f.write(f"    AS Revenue: ${as_revenue:,.2f}/year\n")
                f.write(
                    f"    AS Opportunity Cost (Lost Electricity Sales): ${as_opp_cost:,.2f}/year\n")
                f.write(f"    Net AS Benefit: ${net_as_benefit:,.2f}/year\n")
                if as_revenue > 0:
                    net_margin = (net_as_benefit / as_revenue) * 100
                    f.write(f"    Net AS Margin: {net_margin:.1f}%\n")
                f.write("\n")

            for k_inc in [
                "Annual_Electricity_Opportunity_Cost_USD",
                "Total_Incremental_CAPEX_Learned_USD",
            ]:
                if k_inc in incremental_metrics_rpt:
                    label = k_inc.replace(
                        '_', ' ').title().replace('(USD)', '(USD)')
                    f.write(
                        f"  {label:<35}  : ${incremental_metrics_rpt[k_inc]:,.2f}\n"
                    )

            # Add ROI calculation for incremental
            inc_npv = incremental_metrics_rpt.get("NPV_USD")
            inc_total_capex = incremental_metrics_rpt.get(
                "Total_Incremental_CAPEX_Learned_USD")
            inc_roi = None
            if inc_npv is not None and inc_total_capex and inc_total_capex > 0:
                inc_roi = inc_npv / inc_total_capex

            for k, v in sorted(incremental_metrics_rpt.items()):
                if k in [
                    "pure_incremental_cash_flows",
                    "traditional_incremental_cash_flows",
                    "Annual_Electricity_Opportunity_Cost_USD",
                    "Total_Incremental_CAPEX_Learned_USD",
                    "LCOH_USD_per_kg",  # Skip LCOH - use detailed breakdown analysis instead
                ]:
                    continue
                if k == "IRR_percent":
                    lbl = "IRR (%)"
                elif k == "NPV_USD":
                    lbl = "NPV (USD)"
                elif k == "Payback_Period_Years":
                    lbl = "Payback Period (Years)"
                else:
                    lbl = (
                        k.replace("_USD", " (USD)")
                        .replace("_percent", " (%)")
                        .replace("_Years", " (Years)")
                        .replace("_per_kg", " (USD/kg)")
                        .replace("_", " ")
                        .title()
                    )

                if isinstance(v, (int, float)) and not pd.isna(v):
                    if k == "IRR_percent":
                        formatted_value = f"{v:.2f}%"
                    elif "USD" in k:
                        formatted_value = f"${v:,.2f}"
                    else:
                        formatted_value = f"{v:,.2f}"
                else:
                    formatted_value = str(v)

                f.write(
                    f"  Incremental {lbl:<25}         : {formatted_value}\n")

            if inc_roi is not None:
                f.write(
                    f"  Incremental Return on Investment (ROI)      : {inc_roi:.4f}\n")

        # 移除LCOH Analysis & Comparison部分 (原第8部分)

        section_num = 7 if not incremental_metrics_rpt else 8
        f.write(
            f"\n{section_num}. Cost Assumptions (Base Year)\n"
            + "-" * 32
            + "\n  CAPEX Components:\n"
        )
        for comp, det in sorted(capex_data.items()):
            f.write(
                f"    {comp:<30}                     : ${det.get('total_base_cost_for_ref_size',0):,.0f} (Ref Cap: {det.get('reference_total_capacity_mw',0)}, LR: {det.get('learning_rate_decimal',0)*100}%, Pay Sched: {det.get('payment_schedule_years',{})})\n"
            )
        f.write("    \n  O&M Components (Annual Base):\n")
        for comp, det in sorted(om_data.items()):
            if comp == "Fixed_OM_Battery":
                f.write(
                    f"    {comp:<30}                          : ${det.get('base_cost_per_mw_year',0):,.2f}/MW/yr + ${det.get('base_cost_per_mwh_year',0):,.2f}/MWh/yr (Inflation: {det.get('inflation_rate',0)*100:.1f}%)\n"
                )
            else:
                f.write(
                    f"    {comp:<30}                          : ${det.get('base_cost',0):,.0f} (Inflation: {det.get('inflation_rate',0)*100:.1f}%)\n"
                )
        f.write("    \n  Major Replacements:\n")
        for comp, det in sorted(replacement_data.items()):
            f.write(
                f"    {comp:<30}                        : Cost: {'{:.2f}% of Initial CAPEX'.format(det.get('cost_percent_initial_capex',0)*100) if 'cost_percent_initial_capex' in det else '${:,.0f}'.format(det.get('cost',0))} (Years: {det.get('years',[])})\n"
            )

        # **GREENFIELD NUCLEAR-HYDROGEN SYSTEM ANALYSIS SECTION**
        # Add supplementary analysis results if available
        if annual_metrics_rpt and "greenfield_nuclear_analysis" in annual_metrics_rpt:
            greenfield_results = annual_metrics_rpt["greenfield_nuclear_analysis"]
            f.write(
                f"\n{section_num + 1}. Greenfield Nuclear-Hydrogen System\n" + "-" * 75 + "\n")
            f.write(
                "This analysis calculates the economics of building both nuclear plant\n")
            f.write(
                "and hydrogen production system from scratch (greenfield development).\n\n")

            f.write("System Configuration:\n")
            f.write(
                f"  Analysis Type                   : {greenfield_results.get('analysis_type', 'N/A')}\n")
            f.write(
                f"  Nuclear Capacity                : {greenfield_results.get('nuclear_capacity_mw', 0):,.0f} MW\n")
            f.write(
                f"  Project Lifetime                : {greenfield_results.get('project_lifetime_years', 0)} years\n")
            f.write(
                f"  Construction Period             : {greenfield_results.get('construction_period_years', 0)} years\n")
            f.write(
                f"  Discount Rate                   : {greenfield_results.get('discount_rate', 0)*100:.1f}%\n")

            f.write("\nCapital Investment Breakdown:\n")
            nuclear_capex = greenfield_results.get('nuclear_capex_usd', 0)
            h2_capex = greenfield_results.get('hydrogen_system_capex_usd', 0)
            total_capex = greenfield_results.get('total_system_capex_usd', 0)

            if total_capex > 0:
                nuclear_pct = nuclear_capex / total_capex * 100
                h2_pct = h2_capex / total_capex * 100
            else:
                nuclear_pct = h2_pct = 0

            f.write(
                f"  Nuclear Plant CAPEX             : ${nuclear_capex:,.0f} ({nuclear_pct:.1f}%)\n")
            f.write(
                f"  Hydrogen System CAPEX           : ${h2_capex:,.0f} ({h2_pct:.1f}%)\n")
            f.write(
                f"  Total System CAPEX              : ${total_capex:,.0f}\n")
            f.write(
                f"  CAPEX per MW Nuclear            : ${greenfield_results.get('capex_per_mw_nuclear', 0):,.0f}/MW\n")
            f.write(
                f"  CAPEX per kg H2/year            : ${greenfield_results.get('capex_per_kg_h2_annual', 0):,.0f}/kg\n")

            f.write("\nProduction Metrics:\n")
            f.write(
                f"  Annual H2 Production            : {greenfield_results.get('annual_h2_production_kg', 0):,.0f} kg/year\n")
            f.write(
                f"  Annual Nuclear Generation       : {greenfield_results.get('annual_nuclear_generation_mwh', 0):,.0f} MWh/year\n")
            f.write(
                f"  H2 Production per MW Nuclear    : {greenfield_results.get('h2_production_per_mw_nuclear', 0):,.0f} kg/MW/year\n")
            f.write(
                f"  Nuclear Capacity Factor         : {greenfield_results.get('nuclear_capacity_factor', 0)*100:.1f}%\n")
            f.write(
                f"  Electricity to H2 Efficiency    : {greenfield_results.get('electricity_to_h2_efficiency', 0)*100:.1f}%\n")

            # Investment breakdown with replacements
            f.write("\nInvestment Breakdown (60-year lifecycle):\n")
            h2_initial = greenfield_results.get('h2_initial_capex_usd', 0)
            h2_replacement = greenfield_results.get(
                'h2_replacement_capex_usd', 0)
            f.write(
                f"  H2 System Initial CAPEX         : ${h2_initial:,.0f}\n")
            f.write(
                f"  H2 System Replacement CAPEX     : ${h2_replacement:,.0f}\n")
            f.write(
                f"    Electrolyzer Replacements     : {greenfield_results.get('electrolyzer_replacements_count', 0)} times\n")
            f.write(
                f"    H2 Storage Replacements       : {greenfield_results.get('h2_storage_replacements_count', 0)} times\n")
            f.write(
                f"    Battery Replacements          : {greenfield_results.get('battery_replacements_count', 0)} times\n")
            f.write(
                f"  Enhanced Maintenance Factor     : {greenfield_results.get('enhanced_maintenance_factor', 1.0):.1f}x\n")

            f.write("\nAnnual Performance:\n")
            f.write(
                f"  Total Annual Revenue            : ${greenfield_results.get('annual_total_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    H2 Revenue                    : ${greenfield_results.get('annual_h2_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Electricity Revenue           : ${greenfield_results.get('annual_electricity_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Ancillary Services Revenue    : ${greenfield_results.get('annual_as_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    H2 Subsidy Revenue            : ${greenfield_results.get('annual_h2_subsidy_revenue_usd', 0):,.0f}\n")
            f.write(
                f"  Total Annual OPEX               : ${greenfield_results.get('annual_total_opex_usd', 0):,.0f}\n")
            f.write(
                f"  Net Annual Revenue              : ${greenfield_results.get('annual_net_revenue_usd', 0):,.0f}\n")

            f.write("\nFinancial Results (60-year lifecycle):\n")
            f.write(
                f"  Net Present Value (NPV)         : ${greenfield_results.get('npv_usd', 0):,.0f}\n")
            f.write(
                f"  Internal Rate of Return (IRR)   : {greenfield_results.get('irr_percent', 0):.2f}%\n")
            f.write(
                f"  Return on Investment (ROI)      : {greenfield_results.get('roi_percent', 0):.2f}%\n")
            f.write(
                f"  Payback Period                  : {greenfield_results.get('payback_period_years', 0):.1f} years\n")

            f.write("\nLevelized Costs:\n")
            f.write(
                f"  LCOH (Integrated System)        : ${greenfield_results.get('lcoh_integrated_usd_per_kg', 0):.3f}/kg\n")
            f.write(
                f"  Nuclear LCOE                    : ${greenfield_results.get('nuclear_lcoe_usd_per_mwh', 0):.2f}/MWh\n")

            f.write("\nCash Flow Summary (Present Value):\n")
            f.write(
                f"  Total Revenue (PV)              : ${greenfield_results.get('total_revenue_pv_usd', 0):,.0f}\n")
            f.write(
                f"  Total Costs (PV)                : ${greenfield_results.get('total_costs_pv_usd', 0):,.0f}\n")
            f.write(
                f"  Net Cash Flow (PV)              : ${greenfield_results.get('net_cash_flow_pv_usd', 0):,.0f}\n")

            f.write("\nKey Insights:\n")
            if greenfield_results.get('npv_usd', 0) > 0:
                f.write(
                    "  • The greenfield nuclear-hydrogen system shows positive NPV\n")
            else:
                f.write(
                    "  • The greenfield nuclear-hydrogen system shows negative NPV\n")

            if greenfield_results.get('irr_percent', 0) > greenfield_results.get('discount_rate', 0.08) * 100:
                f.write(
                    "  • IRR exceeds the discount rate, indicating attractive returns\n")
            else:
                f.write(
                    "  • IRR is below the discount rate, indicating marginal returns\n")

            payback = greenfield_results.get('payback_period_years', 999)
            if payback < 15:
                f.write(
                    "  • Relatively short payback period suggests good investment recovery\n")
            elif payback < 25:
                f.write(
                    "  • Moderate payback period typical for large infrastructure projects\n")
            else:
                f.write(
                    "  • Long payback period indicates high capital requirements\n")

            f.write(
                "\nNote: This greenfield analysis assumes building both nuclear plant and\n")
            f.write(
                "hydrogen system from zero, with both systems designed for 60-year operation.\n")
            f.write(
                "The analysis includes periodic replacement of H2 system components:\n")
            f.write(
                "• Electrolyzers replaced every 20 years (2 replacements)\n")
            f.write(
                "• H2 storage systems replaced every 30 years (1 replacement)\n")
            f.write(
                "• Batteries replaced every 15 years (3 replacements)\n")
            f.write(
                "• Enhanced maintenance costs (+20%) for extended lifecycle operation\n")
            f.write(
                "This provides a comprehensive view of long-term integrated system economics.\n\n")

        # Add lifecycle comparison analysis if available
        if annual_metrics_rpt and "lifecycle_comparison_analysis" in annual_metrics_rpt:
            comparison_results = annual_metrics_rpt["lifecycle_comparison_analysis"]
            f.write(
                f"\n{section_num + 2}. Lifecycle Comparison Analysis: 60-Year vs 80-Year\n" + "-" * 75 + "\n")
            f.write(
                "This section compares the financial performance of 60-year vs 80-year project lifecycles\n")
            f.write(
                "to evaluate the impact of extending project duration on investment returns.\n\n")

            # Extract results for both scenarios
            lifecycle_60 = comparison_results.get("lifecycle_60", {})
            lifecycle_80 = comparison_results.get("lifecycle_80", {})
            comparison = comparison_results.get("comparison", {})

            f.write("Financial Performance Comparison:\n")
            f.write(
                f"{'Metric':<35} {'60-Year':<20} {'80-Year':<20} {'Difference':<15}\n")
            f.write("-" * 90 + "\n")

            # Investment comparison
            investment_60 = lifecycle_60.get('total_investment_usd', 0)
            investment_80 = lifecycle_80.get('total_investment_usd', 0)
            f.write(f"{'Total Investment':<35} ${investment_60/1e9:.2f}B{'':<11} ${investment_80/1e9:.2f}B{'':<11} ${comparison.get('additional_investment_usd', 0)/1e9:+.2f}B\n")

            # NPV comparison
            npv_60 = lifecycle_60.get('npv_usd', 0)
            npv_80 = lifecycle_80.get('npv_usd', 0)
            f.write(f"{'Net Present Value (NPV)':<35} ${npv_60/1e9:.2f}B{'':<11} ${npv_80/1e9:.2f}B{'':<11} ${comparison.get('npv_improvement_usd', 0)/1e9:+.2f}B\n")

            # ROI comparison
            roi_60 = lifecycle_60.get('roi_percent', 0)
            roi_80 = lifecycle_80.get('roi_percent', 0)
            f.write(f"{'Return on Investment (ROI)':<35} {roi_60:.2f}%{'':<14} {roi_80:.2f}%{'':<14} {comparison.get('roi_improvement_percent', 0):+.2f}%\n")

            # LCOH comparison
            lcoh_60 = lifecycle_60.get('lcoh_usd_per_kg', 0)
            lcoh_80 = lifecycle_80.get('lcoh_usd_per_kg', 0)
            f.write(f"{'LCOH (USD/kg)':<35} ${lcoh_60:.3f}/kg{'':<11} ${lcoh_80:.3f}/kg{'':<11} ${comparison.get('lcoh_improvement_usd_per_kg', 0):+.3f}/kg\n")

            # Payback comparison
            payback_60 = lifecycle_60.get('payback_years', 0)
            payback_80 = lifecycle_80.get('payback_years', 0)
            f.write(f"{'Payback Period':<35} {payback_60:.1f} years{'':<10} {payback_80:.1f} years{'':<10} {comparison.get('payback_improvement_years', 0):+.1f} years\n")

            f.write("\nReplacement Schedule Comparison:\n")
            h2_60 = lifecycle_60.get('h2_replacements', {})
            h2_80 = lifecycle_80.get('h2_replacements', {})
            f.write(
                f"{'Component':<20} {'60-Year Count':<15} {'80-Year Count':<15} {'Additional':<12}\n")
            f.write("-" * 62 + "\n")
            f.write(f"{'Electrolyzer':<20} {h2_60.get('electrolyzer', 0):<15} {h2_80.get('electrolyzer', 0):<15} {h2_80.get('electrolyzer', 0) - h2_60.get('electrolyzer', 0):<12}\n")
            f.write(f"{'H2 Storage':<20} {h2_60.get('storage', 0):<15} {h2_80.get('storage', 0):<15} {h2_80.get('storage', 0) - h2_60.get('storage', 0):<12}\n")
            f.write(f"{'Battery':<20} {h2_60.get('battery', 0):<15} {h2_80.get('battery', 0):<15} {h2_80.get('battery', 0) - h2_60.get('battery', 0):<12}\n")

            f.write("\nKey Findings:\n")
            additional_investment = comparison.get(
                'additional_investment_usd', 0)
            npv_improvement = comparison.get('npv_improvement_usd', 0)
            recommended_lifecycle = comparison.get(
                'recommended_lifecycle', '60-year lifecycle')

            f.write(
                f"  • Extending to 80-year lifecycle requires additional ${additional_investment/1e9:.2f}B investment\n")
            if npv_improvement > 0:
                f.write(
                    f"  • NPV improves by ${npv_improvement/1e9:.2f}B with 80-year lifecycle\n")
                f.write(
                    f"  • Additional 20 years of operation provide sufficient returns\n")
            else:
                f.write(
                    f"  • NPV worsens by ${abs(npv_improvement)/1e9:.2f}B with 80-year lifecycle\n")
                f.write(
                    f"  • Additional investment does not generate sufficient returns\n")

            lcoh_improvement = comparison.get('lcoh_improvement_usd_per_kg', 0)
            if lcoh_improvement > 0:
                f.write(
                    f"  • LCOH improves by ${lcoh_improvement:.3f}/kg with extended lifecycle\n")
            else:
                f.write(
                    f"  • LCOH increases by ${abs(lcoh_improvement):.3f}/kg with extended lifecycle\n")

            f.write(
                f"\nRecommendation: {recommended_lifecycle} provides better financial performance\n")

            f.write(
                "\nNote: This comparison uses identical annual performance metrics and cost assumptions\n")
            f.write(
                "for both scenarios, varying only the project duration and associated replacement\n")
            f.write(
                "schedules. The analysis considers the time value of money through NPV calculations\n")
            f.write(
                "and evaluates whether additional investment for extended operation is justified.\n\n")

        f.write(
            "\nReport generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        )
    logger.info(f"TEA report saved to {output_file_path}")


# Nuclear Power Plant Integrated Analysis Functions

def calculate_nuclear_capex_breakdown(nuclear_capacity_mw: float) -> dict:
    """
    Calculate nuclear power plant CAPEX breakdown with learning curve adjustments.

    Args:
        nuclear_capacity_mw: Nuclear plant capacity in MW

    Returns:
        Dictionary containing nuclear CAPEX breakdown
    """
    logger.debug(
        f"Calculating nuclear CAPEX for {nuclear_capacity_mw} MW capacity")

    nuclear_capex_breakdown = {}
    total_nuclear_capex = 0

    for component_name, component_details in NUCLEAR_CAPEX_COMPONENTS.items():
        ref_capacity = component_details["reference_total_capacity_mw"]
        ref_cost = component_details["total_base_cost_for_ref_size"]
        learning_rate = component_details["learning_rate_decimal"]

        # Apply scaling and learning curve
        if ref_capacity > 0:
            capacity_ratio = nuclear_capacity_mw / ref_capacity
            # Learning curve based on cumulative capacity deployment
            learning_factor = (1 - learning_rate)  # Simplified learning curve
            scaled_cost = ref_cost * capacity_ratio * learning_factor
        else:
            scaled_cost = ref_cost

        nuclear_capex_breakdown[component_name] = scaled_cost
        total_nuclear_capex += scaled_cost

        logger.debug(f"Nuclear {component_name}: ${scaled_cost:,.0f}")

    nuclear_capex_breakdown["Total_Nuclear_CAPEX"] = total_nuclear_capex
    logger.info(f"Total Nuclear CAPEX: ${total_nuclear_capex:,.0f}")

    return nuclear_capex_breakdown


def calculate_nuclear_annual_opex(nuclear_capacity_mw: float, annual_generation_mwh: float, year: int = 1) -> dict:
    """
    Calculate nuclear power plant annual OPEX.

    Args:
        nuclear_capacity_mw: Nuclear plant capacity in MW
        annual_generation_mwh: Annual electricity generation in MWh
        year: Project year for inflation adjustment

    Returns:
        Dictionary containing nuclear annual OPEX breakdown
    """
    logger.debug(f"Calculating nuclear annual OPEX for year {year}")

    nuclear_opex_breakdown = {}
    total_nuclear_opex = 0

    for component_name, component_details in NUCLEAR_OM_COMPONENTS.items():
        inflation_rate = component_details["inflation_rate"]
        inflation_factor = (1 + inflation_rate) ** (year - 1)

        if "base_cost_per_mw_year" in component_details:
            # Fixed O&M cost per MW
            base_cost = component_details["base_cost_per_mw_year"]
            annual_cost = base_cost * nuclear_capacity_mw * inflation_factor
        elif "base_cost_per_mwh" in component_details:
            # Variable O&M cost per MWh
            base_cost = component_details["base_cost_per_mwh"]
            annual_cost = base_cost * annual_generation_mwh * inflation_factor
        else:
            annual_cost = 0

        nuclear_opex_breakdown[component_name] = annual_cost
        total_nuclear_opex += annual_cost

        logger.debug(
            f"Nuclear {component_name} (Year {year}): ${annual_cost:,.0f}")

    nuclear_opex_breakdown["Total_Nuclear_OPEX"] = total_nuclear_opex
    logger.debug(
        f"Total Nuclear OPEX (Year {year}): ${total_nuclear_opex:,.0f}")

    return nuclear_opex_breakdown


def calculate_greenfield_nuclear_hydrogen_system(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict
) -> dict:
    """
    Calculate financial metrics for a greenfield nuclear-hydrogen integrated system.

    This analysis assumes building both nuclear plant and hydrogen system from zero,
    with both systems designed for 60-year operation through periodic replacement 
    and enhanced maintenance.
    """
    logger.info("=" * 80)
    logger.info("GREENFIELD NUCLEAR-HYDROGEN INTEGRATED SYSTEM ANALYSIS")
    logger.info(
        "60-Year Lifecycle Analysis with Periodic Replacement & Maintenance")
    logger.info("=" * 80)

    # Project parameters - aligned 60-year lifecycle for both systems
    project_lifetime = 60  # Years
    construction_period = 8  # Years for nuclear construction
    discount_rate = DISCOUNT_RATE

    # Get subsidy parameters with proper type handling
    h2_subsidy_val = float(tea_sys_params.get(
        "hydrogen_subsidy_value_usd_per_kg", 0))
    h2_subsidy_duration_raw = tea_sys_params.get(
        "hydrogen_subsidy_duration_years", 10)
    try:
        h2_subsidy_yrs = int(float(str(h2_subsidy_duration_raw))
                             ) if h2_subsidy_duration_raw else 10
    except (ValueError, TypeError):
        h2_subsidy_yrs = 10

    logger.info(f"Greenfield System Configuration:")
    logger.info(f"  Nuclear capacity: {nuclear_capacity_mw:.1f} MW")
    logger.info(
        f"  Project lifetime: {project_lifetime} years (both nuclear and H2 systems)")
    logger.info(f"  Construction period: {construction_period} years")
    logger.info(f"  Discount rate: {discount_rate:.1%}")
    logger.info(
        f"  H2 subsidy: ${h2_subsidy_val:.2f}/kg for {h2_subsidy_yrs} years")

    # === 1. NUCLEAR SYSTEM COSTS ===
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw)
    nuclear_total_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]

    logger.info(f"\nNuclear Plant Investment:")
    logger.info(f"  Initial CAPEX: ${nuclear_total_capex:,.0f}")
    logger.info(
        f"  Cost per MW: ${nuclear_total_capex/nuclear_capacity_mw:,.0f}/MW")

    # === 2. HYDROGEN SYSTEM COSTS (Enhanced for 80-year lifecycle) ===
    # Original hydrogen system CAPEX from optimization
    h2_electrolyzer_capex = annual_metrics.get("total_capex", 0)
    electrolyzer_capacity_mw = annual_metrics.get(
        "Electrolyzer_Capacity_MW", 0)
    h2_storage_capacity_kg = annual_metrics.get("H2_Storage_Capacity_kg", 0)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 0)

    # Enhanced H2 system design for 60-year operation
    # Calculate replacement costs over 60-year lifecycle

    # Electrolyzer replacements (every 20 years: years 20, 40)
    electrolyzer_unit_cost = 1200  # $/kW (updated 2024 cost target)
    electrolyzer_replacement_cost = electrolyzer_capacity_mw * \
        1000 * electrolyzer_unit_cost
    total_electrolyzer_replacements = electrolyzer_replacement_cost * \
        2  # 2 replacements (20, 40)

    # H2 Storage system replacements (every 30 years: year 30)
    # $/kg capacity (updated underground storage cost)
    h2_storage_unit_cost = 400
    h2_storage_replacement_cost = h2_storage_capacity_kg * h2_storage_unit_cost
    total_h2_storage_replacements = h2_storage_replacement_cost * \
        1  # 1 replacement (30)

    # Battery replacements (every 15 years: years 15, 30, 45)
    battery_unit_cost = 150_000  # $/MWh (updated 2024 grid storage cost)
    battery_replacement_cost = battery_capacity_mwh * battery_unit_cost
    total_battery_replacements = battery_replacement_cost * \
        3  # 3 replacements (15, 30, 45)

    # Enhanced maintenance for 60-year operation (20% higher than standard)
    enhanced_maintenance_factor = 1.2

    # Total H2 system investment over 60 years
    total_h2_capex = h2_electrolyzer_capex + total_electrolyzer_replacements + \
        total_h2_storage_replacements + total_battery_replacements

    logger.info(f"\nHydrogen System Investment (60-year lifecycle):")
    logger.info(f"  Initial H2 System CAPEX: ${h2_electrolyzer_capex:,.0f}")
    logger.info(
        f"  Electrolyzer Replacements (2x): ${total_electrolyzer_replacements:,.0f}")
    logger.info(
        f"  H2 Storage Replacements (1x): ${total_h2_storage_replacements:,.0f}")
    logger.info(
        f"  Battery Replacements (3x): ${total_battery_replacements:,.0f}")
    logger.info(f"  Total H2 System CAPEX: ${total_h2_capex:,.0f}")

    # === 3. TOTAL SYSTEM INVESTMENT ===
    total_system_capex = nuclear_total_capex + total_h2_capex

    logger.info(f"\nTotal System Investment:")
    logger.info(
        f"  Nuclear Plant: ${nuclear_total_capex:,.0f} ({nuclear_total_capex/total_system_capex*100:.1f}%)")
    logger.info(
        f"  H2 System: ${total_h2_capex:,.0f} ({total_h2_capex/total_system_capex*100:.1f}%)")
    logger.info(f"  Total: ${total_system_capex:,.0f}")

    # === 4. PRODUCTION METRICS ===
    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)
    annual_nuclear_generation = nuclear_capacity_mw * 8760 * 0.9  # 90% capacity factor
    nuclear_capacity_factor = 0.9

    # Efficiency metrics
    electricity_to_h2_efficiency = (annual_h2_production * 33.3) / \
        annual_nuclear_generation if annual_nuclear_generation > 0 else 0
    h2_production_per_mw = annual_h2_production / \
        nuclear_capacity_mw if nuclear_capacity_mw > 0 else 0

    logger.info(f"\nProduction Metrics:")
    logger.info(f"  Annual H2 Production: {annual_h2_production:,.0f} kg/year")
    logger.info(
        f"  Annual Nuclear Generation: {annual_nuclear_generation:,.0f} MWh/year")
    logger.info(f"  Nuclear Capacity Factor: {nuclear_capacity_factor:.1%}")
    logger.info(
        f"  H2 Production per MW Nuclear: {h2_production_per_mw:,.0f} kg/MW/year")
    logger.info(
        f"  Electricity-to-H2 Efficiency: {electricity_to_h2_efficiency:.1%}")

    # === 5. FINANCIAL ANALYSIS ===

    # Annual revenues - use actual prices from optimization results where available
    h2_price = 5.0  # $/kg (target price for clean hydrogen)
    # Use actual electricity price from optimization results if available
    electricity_price = annual_metrics.get(
        "Avg_Electricity_Price_USD_per_MWh", 60.0)

    annual_h2_revenue = annual_h2_production * h2_price
    annual_electricity_revenue = annual_nuclear_generation * electricity_price

    # **ADDED: Include Ancillary Services revenue from optimization results**
    annual_as_revenue = annual_metrics.get("AS_Revenue_Total", 0)
    if annual_as_revenue == 0:
        annual_as_revenue = annual_metrics.get("AS_Revenue", 0)

    total_annual_revenue = annual_h2_revenue + \
        annual_electricity_revenue + annual_as_revenue

    # Apply hydrogen subsidy during subsidy period
    h2_subsidy_revenue = annual_h2_production * \
        h2_subsidy_val if h2_subsidy_yrs > 0 else 0

    # Annual operating costs (enhanced for 60-year operation)
    nuclear_annual_opex = nuclear_capacity_mw * 120_000  # $120k/MW/year
    h2_annual_opex = total_h2_capex * 0.025 * \
        enhanced_maintenance_factor  # 2.5% of CAPEX, enhanced
    total_annual_opex = nuclear_annual_opex + h2_annual_opex

    # Net annual cash flow (during operations)
    annual_net_revenue = total_annual_revenue - total_annual_opex
    annual_net_revenue_with_subsidy = annual_net_revenue + h2_subsidy_revenue

    logger.info(f"\nAnnual Financial Performance:")
    logger.info(
        f"  H2 Revenue: ${annual_h2_revenue:,.0f} (${h2_price:.2f}/kg)")
    logger.info(
        f"  Electricity Revenue: ${annual_electricity_revenue:,.0f} (${electricity_price:.2f}/MWh)")
    logger.info(
        f"  Ancillary Services Revenue: ${annual_as_revenue:,.0f}")
    logger.info(
        f"  H2 Subsidy Revenue: ${h2_subsidy_revenue:,.0f} (first {h2_subsidy_yrs} years)")
    logger.info(f"  Total Annual Revenue: ${total_annual_revenue:,.0f}")
    logger.info(f"  Total Annual OPEX: ${total_annual_opex:,.0f}")
    logger.info(f"  Net Annual Revenue: ${annual_net_revenue:,.0f}")

    # === 6. NPV CALCULATION ===

    # Present value of revenues and costs
    total_revenue_pv = 0
    total_costs_pv = total_system_capex  # Initial investment

    for year in range(construction_period + 1, construction_period + project_lifetime + 1):
        operating_year = year - construction_period

        # Calculate annual revenue with subsidy tapering
        if operating_year <= h2_subsidy_yrs:
            year_revenue = annual_net_revenue_with_subsidy
        else:
            year_revenue = annual_net_revenue

        # Apply discount factor
        discount_factor = (1 + discount_rate) ** (year - 1)
        total_revenue_pv += year_revenue / discount_factor

        # Add replacement costs in specific years (60-year lifecycle)
        replacement_cost = 0
        if operating_year == 15:
            replacement_cost += battery_replacement_cost  # First battery replacement
        if operating_year == 20:
            replacement_cost += electrolyzer_replacement_cost  # First electrolyzer replacement
        if operating_year == 30:
            replacement_cost += h2_storage_replacement_cost + \
                battery_replacement_cost  # Storage + 2nd battery
        if operating_year == 40:
            # Second electrolyzer replacement
            replacement_cost += electrolyzer_replacement_cost
        if operating_year == 45:
            replacement_cost += battery_replacement_cost  # Third battery replacement

        if replacement_cost > 0:
            total_costs_pv += replacement_cost / discount_factor

    # Financial metrics
    npv = total_revenue_pv - total_costs_pv

    # IRR calculation (simplified)
    if npv > 0:
        irr_estimate = (total_revenue_pv /
                        total_costs_pv) ** (1/project_lifetime) - 1
        irr_percent = irr_estimate * 100
    else:
        irr_percent = 0

    # Payback period
    cumulative_cash_flow = -total_system_capex
    payback_years = project_lifetime  # Default to full lifetime if no payback

    for year in range(1, project_lifetime + 1):
        if year <= h2_subsidy_yrs:
            year_cash_flow = annual_net_revenue_with_subsidy
        else:
            year_cash_flow = annual_net_revenue
        cumulative_cash_flow += year_cash_flow
        if cumulative_cash_flow > 0 and payback_years == project_lifetime:
            payback_years = year
            break

    # Levelized costs
    total_production_pv = 0
    for year in range(1, project_lifetime + 1):
        discount_factor = (1 + discount_rate) ** year
        total_production_pv += annual_h2_production / discount_factor

    lcoh_integrated = total_costs_pv / \
        total_production_pv if total_production_pv > 0 else 0

    # Nuclear LCOE
    total_generation_pv = 0
    for year in range(1, project_lifetime + 1):
        discount_factor = (1 + discount_rate) ** year
        total_generation_pv += annual_nuclear_generation / discount_factor

    nuclear_lcoe = nuclear_total_capex / \
        total_generation_pv if total_generation_pv > 0 else 0

    logger.info(f"\nFinancial Results (60-year lifecycle):")
    logger.info(f"  Net Present Value (NPV): ${npv:,.0f}")
    logger.info(f"  Internal Rate of Return (IRR): {irr_percent:.2f}%")
    logger.info(
        f"  Return on Investment (ROI): {(npv/total_system_capex*100):.2f}%")
    logger.info(f"  Payback Period: {payback_years:.1f} years")
    logger.info(f"  LCOH (Integrated System): ${lcoh_integrated:.3f}/kg")
    logger.info(f"  Nuclear LCOE: ${nuclear_lcoe:.2f}/MWh")

    # Investment efficiency metrics
    capex_per_mw_nuclear = total_system_capex / \
        nuclear_capacity_mw if nuclear_capacity_mw > 0 else 0
    capex_per_kg_h2_annual = total_system_capex / \
        annual_h2_production if annual_h2_production > 0 else 0

    logger.info(f"\nInvestment Efficiency:")
    logger.info(
        f"  Total Investment per MW Nuclear: ${capex_per_mw_nuclear:,.0f}/MW")
    logger.info(
        f"  Total Investment per kg H2/year: ${capex_per_kg_h2_annual:,.0f}/kg")

    logger.info("=" * 80)

    # === 7. COMPILE RESULTS ===
    greenfield_results = {
        "analysis_type": "greenfield_nuclear_hydrogen_system",
        "description": "Complete 60-year integrated system (nuclear + hydrogen) built from zero",

        # System configuration
        "nuclear_capacity_mw": nuclear_capacity_mw,
        "project_lifetime_years": project_lifetime,
        "construction_period_years": construction_period,
        "discount_rate": discount_rate,

        # Investment breakdown
        "nuclear_capex_usd": nuclear_total_capex,
        "hydrogen_system_capex_usd": total_h2_capex,
        "total_system_capex_usd": total_system_capex,
        "h2_initial_capex_usd": h2_electrolyzer_capex,
        "h2_replacement_capex_usd": total_h2_capex - h2_electrolyzer_capex,

        # Production metrics
        "annual_h2_production_kg": annual_h2_production,
        "annual_nuclear_generation_mwh": annual_nuclear_generation,
        "nuclear_capacity_factor": nuclear_capacity_factor,
        "electricity_to_h2_efficiency": electricity_to_h2_efficiency,
        "h2_production_per_mw_nuclear": h2_production_per_mw,

        # Financial metrics
        "npv_usd": npv,
        "irr_percent": irr_percent,
        "payback_period_years": payback_years,
        "roi_percent": (npv / total_system_capex * 100) if total_system_capex > 0 else 0,

        # Unit costs
        "lcoh_integrated_usd_per_kg": lcoh_integrated,
        "nuclear_lcoe_usd_per_mwh": nuclear_lcoe,

        # Cash flow summary
        "total_revenue_pv_usd": total_revenue_pv,
        "total_costs_pv_usd": total_costs_pv,
        "net_cash_flow_pv_usd": npv,

        # Investment efficiency
        "capex_per_mw_nuclear": capex_per_mw_nuclear,
        "capex_per_kg_h2_annual": capex_per_kg_h2_annual,

        # Annual performance
        "annual_total_revenue_usd": total_annual_revenue,
        "annual_h2_revenue_usd": annual_h2_revenue,
        "annual_electricity_revenue_usd": annual_electricity_revenue,
        "annual_as_revenue_usd": annual_as_revenue,
        "annual_h2_subsidy_revenue_usd": h2_subsidy_revenue,
        "annual_total_opex_usd": total_annual_opex,
        "annual_net_revenue_usd": annual_net_revenue,

        # Replacement schedule summary
        "electrolyzer_replacements_count": 2,
        "h2_storage_replacements_count": 1,
        "battery_replacements_count": 3,
        "enhanced_maintenance_factor": enhanced_maintenance_factor,
    }

    return greenfield_results


def calculate_lifecycle_comparison_analysis(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict
) -> dict:
    """
    Calculate comparison between 60-year and 80-year project lifecycles
    for nuclear-hydrogen integrated systems.
    """
    logger.info("\n" + "=" * 80)
    logger.info("LIFECYCLE COMPARISON ANALYSIS: 60-YEAR vs 80-YEAR")
    logger.info("=" * 80)

    # Base parameters (same for both scenarios)
    construction_period = 8
    discount_rate = 0.08

    # Investment costs - use calculated nuclear CAPEX instead of hardcoded value
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw)
    nuclear_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]
    h2_initial_capex = annual_metrics.get("total_capex", 509_860_000)

    # Annual performance metrics - use actual optimization results
    annual_h2_production = annual_metrics.get(
        "H2_Production_kg_annual", 42_620_782)
    annual_nuclear_generation = nuclear_capacity_mw * 8760 * 0.9

    # Use actual revenue calculations from optimization results
    h2_price = 5.0  # $/kg (target price for clean hydrogen)
    # Use actual electricity price from optimization results if available
    electricity_price = annual_metrics.get(
        "Avg_Electricity_Price_USD_per_MWh", 60.0)
    annual_h2_revenue = annual_h2_production * h2_price
    annual_electricity_revenue = annual_nuclear_generation * electricity_price
    annual_as_revenue = annual_metrics.get("AS_Revenue_Total", 0)
    if annual_as_revenue == 0:
        annual_as_revenue = annual_metrics.get("AS_Revenue", 0)

    # Get subsidy parameters from tea_sys_params
    h2_subsidy_val = float(tea_sys_params.get(
        "hydrogen_subsidy_value_usd_per_kg", 0))
    h2_subsidy_duration_raw = tea_sys_params.get(
        "hydrogen_subsidy_duration_years", 10)
    try:
        h2_subsidy_years = int(
            float(str(h2_subsidy_duration_raw))) if h2_subsidy_duration_raw else 10
    except (ValueError, TypeError):
        h2_subsidy_years = 10
    annual_h2_subsidy = annual_h2_production * h2_subsidy_val

    # Component costs for replacements
    electrolyzer_capacity_mw = annual_metrics.get(
        "Electrolyzer_Capacity_MW", 50)
    h2_storage_capacity_kg = annual_metrics.get(
        "H2_Storage_Capacity_kg", 10_000_000)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 100)

    electrolyzer_unit_cost = 1200  # $/kW (updated cost)
    h2_storage_unit_cost = 400  # $/kg (updated cost)
    battery_unit_cost = 150_000  # $/MWh (updated cost)

    electrolyzer_replacement_cost = electrolyzer_capacity_mw * \
        1000 * electrolyzer_unit_cost
    h2_storage_replacement_cost = h2_storage_capacity_kg * h2_storage_unit_cost
    battery_replacement_cost = battery_capacity_mwh * battery_unit_cost

    # Operating costs
    nuclear_annual_opex = nuclear_capacity_mw * 120_000
    enhanced_maintenance_factor = 1.2

    # Total annual revenue
    total_annual_revenue = annual_h2_revenue + \
        annual_electricity_revenue + annual_as_revenue

    # === 60-Year Analysis ===
    project_lifetime_60 = 60
    total_h2_replacement_60 = (electrolyzer_replacement_cost * 2 +
                               h2_storage_replacement_cost * 1 +
                               battery_replacement_cost * 3)
    total_system_capex_60 = nuclear_capex + \
        h2_initial_capex + total_h2_replacement_60

    h2_annual_opex_60 = (h2_initial_capex + total_h2_replacement_60) * \
        0.025 * enhanced_maintenance_factor
    total_annual_opex_60 = nuclear_annual_opex + h2_annual_opex_60
    annual_net_revenue_60 = total_annual_revenue - total_annual_opex_60
    annual_net_revenue_with_subsidy_60 = annual_net_revenue_60 + annual_h2_subsidy

    # NPV calculation for 60 years
    total_revenue_pv_60 = 0
    total_costs_pv_60 = total_system_capex_60

    replacements_60 = {15: battery_replacement_cost, 20: electrolyzer_replacement_cost,
                       30: h2_storage_replacement_cost + battery_replacement_cost,
                       40: electrolyzer_replacement_cost, 45: battery_replacement_cost}

    for year in range(construction_period + 1, construction_period + project_lifetime_60 + 1):
        operating_year = year - construction_period
        year_revenue = annual_net_revenue_with_subsidy_60 if operating_year <= h2_subsidy_years else annual_net_revenue_60
        discount_factor = (1 + discount_rate) ** (year - 1)
        total_revenue_pv_60 += year_revenue / discount_factor
        if operating_year in replacements_60:
            total_costs_pv_60 += replacements_60[operating_year] / \
                discount_factor

    npv_60 = total_revenue_pv_60 - total_costs_pv_60

    # === 80-Year Analysis ===
    project_lifetime_80 = 80
    total_h2_replacement_80 = (electrolyzer_replacement_cost * 3 +
                               h2_storage_replacement_cost * 2 +
                               battery_replacement_cost * 5)
    total_system_capex_80 = nuclear_capex + \
        h2_initial_capex + total_h2_replacement_80

    h2_annual_opex_80 = (h2_initial_capex + total_h2_replacement_80) * \
        0.025 * enhanced_maintenance_factor
    total_annual_opex_80 = nuclear_annual_opex + h2_annual_opex_80
    annual_net_revenue_80 = total_annual_revenue - total_annual_opex_80
    annual_net_revenue_with_subsidy_80 = annual_net_revenue_80 + annual_h2_subsidy

    # NPV calculation for 80 years
    total_revenue_pv_80 = 0
    total_costs_pv_80 = total_system_capex_80

    replacements_80 = {15: battery_replacement_cost, 20: electrolyzer_replacement_cost,
                       30: h2_storage_replacement_cost + battery_replacement_cost,
                       40: electrolyzer_replacement_cost, 45: battery_replacement_cost,
                       60: electrolyzer_replacement_cost + h2_storage_replacement_cost + battery_replacement_cost,
                       75: battery_replacement_cost}

    for year in range(construction_period + 1, construction_period + project_lifetime_80 + 1):
        operating_year = year - construction_period
        year_revenue = annual_net_revenue_with_subsidy_80 if operating_year <= h2_subsidy_years else annual_net_revenue_80
        discount_factor = (1 + discount_rate) ** (year - 1)
        total_revenue_pv_80 += year_revenue / discount_factor
        if operating_year in replacements_80:
            total_costs_pv_80 += replacements_80[operating_year] / \
                discount_factor

    npv_80 = total_revenue_pv_80 - total_costs_pv_80

    # Calculate LCOH for both scenarios
    total_production_pv_60 = sum(annual_h2_production / (1 + discount_rate)
                                 ** year for year in range(1, project_lifetime_60 + 1))
    total_production_pv_80 = sum(annual_h2_production / (1 + discount_rate)
                                 ** year for year in range(1, project_lifetime_80 + 1))

    lcoh_60 = total_costs_pv_60 / \
        total_production_pv_60 if total_production_pv_60 > 0 else 0
    lcoh_80 = total_costs_pv_80 / \
        total_production_pv_80 if total_production_pv_80 > 0 else 0

    # Payback period calculation
    def calculate_payback(capex, net_revenue_with_sub, net_revenue, subsidy_yrs, lifetime):
        cumulative_cash_flow = -capex
        for year in range(1, lifetime + 1):
            year_cash_flow = net_revenue_with_sub if year <= subsidy_yrs else net_revenue
            cumulative_cash_flow += year_cash_flow
            if cumulative_cash_flow > 0:
                return year
        return lifetime

    payback_60 = calculate_payback(total_system_capex_60, annual_net_revenue_with_subsidy_60,
                                   annual_net_revenue_60, h2_subsidy_years, project_lifetime_60)
    payback_80 = calculate_payback(total_system_capex_80, annual_net_revenue_with_subsidy_80,
                                   annual_net_revenue_80, h2_subsidy_years, project_lifetime_80)

    # Calculate improvements
    additional_investment = total_system_capex_80 - total_system_capex_60
    npv_improvement = npv_80 - npv_60
    roi_60 = (npv_60 / total_system_capex_60) * \
        100 if total_system_capex_60 > 0 else 0
    roi_80 = (npv_80 / total_system_capex_80) * \
        100 if total_system_capex_80 > 0 else 0
    roi_improvement = roi_80 - roi_60
    lcoh_improvement = lcoh_60 - lcoh_80
    payback_improvement = payback_60 - payback_80

    logger.info(f"60-Year Lifecycle Results:")
    logger.info(f"  Total Investment: ${total_system_capex_60/1e9:.2f}B")
    logger.info(f"  NPV: ${npv_60/1e9:.2f}B")
    logger.info(f"  ROI: {roi_60:.2f}%")
    logger.info(f"  LCOH: ${lcoh_60:.3f}/kg")
    logger.info(f"  Payback Period: {payback_60:.1f} years")

    logger.info(f"\n80-Year Lifecycle Results:")
    logger.info(f"  Total Investment: ${total_system_capex_80/1e9:.2f}B")
    logger.info(f"  NPV: ${npv_80/1e9:.2f}B")
    logger.info(f"  ROI: {roi_80:.2f}%")
    logger.info(f"  LCOH: ${lcoh_80:.3f}/kg")
    logger.info(f"  Payback Period: {payback_80:.1f} years")

    logger.info(f"\nComparison Summary:")
    logger.info(
        f"  Additional Investment (80-year): ${additional_investment/1e9:+.2f}B")
    logger.info(f"  NPV Improvement: ${npv_improvement/1e9:+.2f}B")
    logger.info(f"  ROI Improvement: {roi_improvement:+.2f}%")
    logger.info(f"  LCOH Improvement: ${lcoh_improvement:+.3f}/kg")
    logger.info(f"  Payback Improvement: {payback_improvement:+.1f} years")

    recommendation = "80-year lifecycle" if npv_improvement > 0 else "60-year lifecycle"
    logger.info(
        f"\nRecommendation: {recommendation} shows better financial performance")

    # Return comparison results
    comparison_results = {
        "lifecycle_60": {
            "project_lifetime": project_lifetime_60,
            "total_investment_usd": total_system_capex_60,
            "npv_usd": npv_60,
            "roi_percent": roi_60,
            "lcoh_usd_per_kg": lcoh_60,
            "payback_years": payback_60,
            "h2_replacements": {"electrolyzer": 2, "storage": 1, "battery": 3}
        },
        "lifecycle_80": {
            "project_lifetime": project_lifetime_80,
            "total_investment_usd": total_system_capex_80,
            "npv_usd": npv_80,
            "roi_percent": roi_80,
            "lcoh_usd_per_kg": lcoh_80,
            "payback_years": payback_80,
            "h2_replacements": {"electrolyzer": 3, "storage": 2, "battery": 5}
        },
        "comparison": {
            "additional_investment_usd": additional_investment,
            "npv_improvement_usd": npv_improvement,
            "roi_improvement_percent": roi_improvement,
            "lcoh_improvement_usd_per_kg": lcoh_improvement,
            "payback_improvement_years": payback_improvement,
            "recommended_lifecycle": recommendation
        }
    }

    return comparison_results


def main():
    """Main execution function for TEA analysis."""
    global log_file_path

    logger.debug("main() function started in tea.py.")
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - TEA_MAIN_FALLBACK - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file_path),
            ],
        )
        globals()["logger"] = logging.getLogger(__name__)

    logger.info("--- Starting Technical Economic Analysis ---")
    current_target_iso = TARGET_ISO
    logger.info(f"Using Target ISO: {current_target_iso}")

    # Update log file path with correct ISO
    new_log_file_path = LOG_DIR / f"tea_{current_target_iso}.log"
    if log_file_path != new_log_file_path:
        # Remove old handlers
        if logger and logger.handlers:
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)

        # Rename the log file if it exists
        if log_file_path.exists():
            try:
                log_file_path.rename(new_log_file_path)
                logger.info(f"Log file renamed to: {new_log_file_path}")
            except Exception as e:
                logger.warning(f"Could not rename log file: {e}")

        # Update log_file_path and add new handler
        log_file_path = new_log_file_path
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    tea_base_output_dir = BASE_OUTPUT_DIR_DEFAULT
    os.makedirs(tea_base_output_dir, exist_ok=True)
    tea_output_file = (
        tea_base_output_dir / f"{current_target_iso}_TEA_Summary_Report.txt"
    )
    plot_output_dir = tea_base_output_dir / f"Plots_{current_target_iso}"
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.debug(
        f"Output paths configured. Report: {tea_output_file}, Plots: {plot_output_dir}"
    )

    tea_sys_params = load_tea_sys_params(
        current_target_iso, BASE_INPUT_DIR_DEFAULT)

    def get_float_param(params_dict, key, default_value, logger_instance):
        val = params_dict.get(key)
        if val is None or pd.isna(val):
            logger_instance.info(
                f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_value}"
            )
            return float(default_value)
        try:
            return float(val)
        except (ValueError, TypeError):
            logger_instance.warning(
                f"Invalid value for parameter '{key}': '{val}'. Using default: {default_value}"
            )
            return float(default_value)

    def get_int_param(params_dict, key, default_value, logger_instance):
        val = params_dict.get(key)
        if val is None or pd.isna(val):
            logger_instance.info(
                f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_value}"
            )
            return int(default_value)
        try:
            return int(float(val))
        except (ValueError, TypeError):
            logger_instance.warning(
                f"Invalid value for parameter '{key}': '{val}'. Using default: {default_value}"
            )
            return int(default_value)

    h2_subsidy_val = get_float_param(
        tea_sys_params, "hydrogen_subsidy_value_usd_per_kg", 0.0, logger
    )
    # Fixed: ensure proper type handling for hydrogen subsidy duration
    h2_subsidy_duration_raw = tea_sys_params.get(
        "hydrogen_subsidy_duration_years", 10)
    try:
        h2_subsidy_duration_int = int(
            float(str(h2_subsidy_duration_raw))) if h2_subsidy_duration_raw else 10
    except (ValueError, TypeError):
        h2_subsidy_duration_int = 10

    h2_subsidy_yrs = min(h2_subsidy_duration_int, PROJECT_LIFETIME_YEARS)
    baseline_revenue_val = get_float_param(
        tea_sys_params, "baseline_nuclear_annual_revenue_USD", 0.0, logger
    )

    run_incremental_raw = tea_sys_params.get("enable_incremental_analysis")
    if run_incremental_raw is None or pd.isna(run_incremental_raw):
        run_incremental = True
        logger.info(
            "'enable_incremental_analysis' not found or NA in sys_params. Defaulting to True."
        )
    else:
        try:
            if isinstance(run_incremental_raw, str):
                run_incremental = run_incremental_raw.lower() in [
                    "true",
                    "1",
                    "yes",
                ]
            else:
                run_incremental = bool(int(float(run_incremental_raw)))
        except (ValueError, TypeError):
            run_incremental = True
            logger.warning(
                f"Invalid value for 'enable_incremental_analysis': {run_incremental_raw}. Defaulting to True."
            )

    logger.debug("TEA system parameters loaded and processed in main.")

    opt_results_dir = SCRIPT_DIR_PATH.parent / "output" / "Results_Standardized"
    results_file_path = (
        opt_results_dir /
        f"{current_target_iso}_Hourly_Results_Comprehensive.csv"
    )
    logger.info(f"Loading results from: {results_file_path}")
    if not results_file_path.exists():
        logger.error(
            f"Optimization results file not found: {results_file_path}. Exiting TEA."
        )
        print(
            f"Error: Optimization results file not found at {results_file_path}")
        return False

    hourly_res_df = load_hourly_results(results_file_path)
    if hourly_res_df is None:
        logger.error("Failed to load optimization results. Exiting TEA.")
        return False
    logger.info("Hourly results loaded successfully.")

    annual_metrics_results = calculate_annual_metrics(
        hourly_res_df, tea_sys_params)
    if annual_metrics_results is None:
        logger.error("Failed to calculate annual metrics. Exiting TEA.")
        return False
    logger.info("Annual metrics calculated.")

    optimized_caps = {
        "Electrolyzer_Capacity_MW": annual_metrics_results.get(
            "Electrolyzer_Capacity_MW", 0
        ),
        "H2_Storage_Capacity_kg": annual_metrics_results.get(
            "H2_Storage_Capacity_kg", 0
        ),
        # Added for battery
        "Battery_Capacity_MWh": annual_metrics_results.get("Battery_Capacity_MWh", 0),
        # Added for battery
        "Battery_Power_MW": annual_metrics_results.get("Battery_Power_MW", 0),
    }
    logger.debug(f"Optimized capacities for cash flow: {optimized_caps}")

    cash_flows_results = calculate_cash_flows(
        annual_metrics=annual_metrics_results,
        project_lifetime=PROJECT_LIFETIME_YEARS,
        construction_period=CONSTRUCTION_YEARS,
        h2_subsidy_value=h2_subsidy_val,
        h2_subsidy_duration=h2_subsidy_yrs,
        capex_details=CAPEX_COMPONENTS,
        om_details=OM_COMPONENTS,
        replacement_details=REPLACEMENT_SCHEDULE,
        optimized_capacities=optimized_caps,
    )
    logger.info("Cash flows calculated.")

    financial_metrics_results = calculate_financial_metrics(
        cash_flows_input=cash_flows_results,
        discount_rt=DISCOUNT_RATE,
        annual_h2_prod_kg=annual_metrics_results.get(
            "H2_Production_kg_annual", 0),
        project_lt=PROJECT_LIFETIME_YEARS,
        construction_p=CONSTRUCTION_YEARS,
    )

    # Calculate ROI (Return on Investment) = NPV / Total CAPEX
    total_capex = annual_metrics_results.get("total_capex", 0)
    npv = financial_metrics_results.get("NPV_USD", 0)
    if total_capex > 0 and npv is not None and not pd.isna(npv):
        financial_metrics_results["ROI"] = npv / total_capex
        logger.info(f"ROI calculated: {financial_metrics_results['ROI']:.4f}")
    else:
        financial_metrics_results["ROI"] = np.nan
        logger.warning("ROI calculation failed - missing CAPEX or NPV data")

    logger.info("Financial metrics calculated.")

    # **Greenfield Nuclear-Hydrogen Integrated Analysis (now enabled by default)**
    # This is the main analysis module for building nuclear plant + hydrogen system from scratch
    greenfield_nuclear_metrics = None
    logger.info("Greenfield nuclear-hydrogen integrated analysis is enabled")
    logger.info(
        "This analysis calculates building nuclear plant + hydrogen system from zero")

    # Get actual reactor capacity from tea_sys_params (typically from plant-specific data)
    actual_nuclear_capacity = get_float_param(
        tea_sys_params,
        "nameplate_capacity_mw",  # From plant-specific parameters
        get_float_param(tea_sys_params, "pTurbine_max_MW",
                        NUCLEAR_INTEGRATED_CONFIG.get("nuclear_plant_capacity_mw", 1000), logger),
        logger
    )

    logger.info(
        f"Using actual nuclear plant capacity: {actual_nuclear_capacity:.1f} MW")

    greenfield_nuclear_metrics = calculate_greenfield_nuclear_hydrogen_system(
        annual_metrics=annual_metrics_results,
        nuclear_capacity_mw=actual_nuclear_capacity,
        tea_sys_params=tea_sys_params
    )

    if greenfield_nuclear_metrics:
        logger.info(
            "Greenfield nuclear-hydrogen integrated analysis completed successfully")
        # Store greenfield results separately (don't overwrite existing results)
        annual_metrics_results["greenfield_nuclear_analysis"] = greenfield_nuclear_metrics

        # Perform lifecycle comparison analysis (60-year vs 80-year)
        lifecycle_comparison_metrics = calculate_lifecycle_comparison_analysis(
            annual_metrics=annual_metrics_results,
            nuclear_capacity_mw=actual_nuclear_capacity,
            tea_sys_params=tea_sys_params
        )

        if lifecycle_comparison_metrics:
            logger.info("Lifecycle comparison analysis completed successfully")
            annual_metrics_results["lifecycle_comparison_analysis"] = lifecycle_comparison_metrics
        else:
            logger.warning("Lifecycle comparison analysis failed")
    else:
        logger.warning(
            "Greenfield nuclear-hydrogen integrated analysis failed")

    # **Calculate detailed LCOH breakdown by cost factors**
    lcoh_breakdown_results = None
    h2_production_annual = annual_metrics_results.get(
        "H2_Production_kg_annual", 0)
    if h2_production_annual > 0 and "capex_breakdown" in annual_metrics_results:
        lcoh_breakdown_results = calculate_lcoh_breakdown(
            annual_metrics=annual_metrics_results,
            capex_breakdown=annual_metrics_results["capex_breakdown"],
            project_lifetime=PROJECT_LIFETIME_YEARS,
            construction_period=CONSTRUCTION_YEARS,
            discount_rate=DISCOUNT_RATE,
            annual_h2_production_kg=h2_production_annual,
        )
        if lcoh_breakdown_results:
            # Store LCOH breakdown in annual_metrics for reporting and visualization
            annual_metrics_results["lcoh_breakdown_analysis"] = lcoh_breakdown_results

            # Add detailed LCOH to financial metrics (replacing simplified method)
            total_lcoh = lcoh_breakdown_results.get("total_lcoh_usd_per_kg", 0)
            financial_metrics_results["LCOH_USD_per_kg"] = total_lcoh

            logger.info(
                f"LCOH breakdown analysis completed. Total LCOH: ${total_lcoh:.3f}/kg")
        else:
            logger.warning("LCOH breakdown analysis failed.")
    else:
        logger.warning("Skipping LCOH breakdown analysis - insufficient data.")

    incremental_fin_metrics = None
    if run_incremental:
        logger.info("Starting incremental analysis.")
        # Incremental components now explicitly include Battery if ENABLE_BATTERY is true
        # Added NPP for nuclear plant modifications
        incremental_capex_keys = ["Electrolyzer", "H2_Storage", "NPP"]
        if ENABLE_BATTERY:
            incremental_capex_keys.append("Battery")
        incremental_capex = {
            k: v
            for k, v in CAPEX_COMPONENTS.items()
            if any(sub in k for sub in incremental_capex_keys)
        }

        # Incremental O&M should also consider battery fixed O&M
        incremental_om = {
            "Fixed_OM_General": OM_COMPONENTS.get("Fixed_OM_General", {})
        }  # General incremental fixed OM
        if ENABLE_BATTERY:
            incremental_om["Fixed_OM_Battery"] = OM_COMPONENTS.get(
                "Fixed_OM_Battery", {}
            )  # Battery specific fixed OM

        incremental_replacements_keys = ["Electrolyzer", "H2_Storage"]
        if ENABLE_BATTERY:
            incremental_replacements_keys.append("Battery")
        incremental_replacements = {
            k: v
            for k, v in REPLACEMENT_SCHEDULE.items()
            if any(sub in k for sub in incremental_replacements_keys)
        }

        if baseline_revenue_val <= 0 and "Energy_Revenue" in annual_metrics_results:
            turbine_max_cap_param = tea_sys_params.get("pTurbine_max_MW")
            turbine_max_cap = get_float_param(
                tea_sys_params,
                "pTurbine_max_MW",
                annual_metrics_results.get("Turbine_Capacity_MW", 300),
                logger,
            )
            avg_lmp_val = annual_metrics_results.get(
                "Avg_Electricity_Price_USD_per_MWh", 40
            )
            baseline_revenue_val = turbine_max_cap * HOURS_IN_YEAR * avg_lmp_val
            logger.info(
                f"Estimated baseline nuclear revenue: ${baseline_revenue_val:,.2f}"
            )

        incremental_fin_metrics = calculate_incremental_metrics(
            optimized_cash_flows=cash_flows_results,
            baseline_annual_revenue=baseline_revenue_val,
            project_lifetime=PROJECT_LIFETIME_YEARS,
            construction_period=CONSTRUCTION_YEARS,
            discount_rt=DISCOUNT_RATE,
            tax_rt=TAX_RATE,
            annual_metrics_optimized=annual_metrics_results,
            capex_components_incremental=incremental_capex,
            om_components_incremental=incremental_om,
            replacement_schedule_incremental=incremental_replacements,
            h2_subsidy_val=h2_subsidy_val,
            h2_subsidy_yrs=h2_subsidy_yrs,
            optimized_capacities_inc=optimized_caps,
        )
        logger.info("Incremental metrics calculated.")

    logger.info("Generating plots...")
    plot_results(
        annual_metrics_data=annual_metrics_results,
        financial_metrics_data=financial_metrics_results,
        cash_flows_data=cash_flows_results,
        plot_dir=plot_output_dir,
        construction_p=CONSTRUCTION_YEARS,
        incremental_metrics_data=incremental_fin_metrics,
    )
    logger.info("Plots generated successfully.")

    logger.info("Generating final report...")
    generate_report(
        annual_metrics_rpt=annual_metrics_results,
        financial_metrics_rpt=financial_metrics_results,
        output_file_path=tea_output_file,
        target_iso_rpt=current_target_iso,
        capex_data=CAPEX_COMPONENTS,
        om_data=OM_COMPONENTS,
        replacement_data=REPLACEMENT_SCHEDULE,
        project_lt_rpt=PROJECT_LIFETIME_YEARS,
        construction_p_rpt=CONSTRUCTION_YEARS,
        discount_rt_rpt=DISCOUNT_RATE,
        tax_rt_rpt=TAX_RATE,
        incremental_metrics_rpt=incremental_fin_metrics,
    )
    logger.info("Report generation finished.")

    logger.info("--- Technical Economic Analysis completed successfully ---")
    print(f"\nTEA Analysis completed for {current_target_iso}.")
    print(f"  Summary Report: {tea_output_file}")
    print(f"  Plots: {plot_output_dir}")
    print(f"  Log file: {log_file_path}")
    return True


if __name__ == "__main__":
    try:
        main_success = main()
        if not main_success:
            print("TEA analysis failed. Check the log file for details.")
            sys.exit(1)
        sys.exit(0)
    except Exception as e_main:
        print(f"An unhandled error occurred in TEA main: {e_main}")
        if logger:
            logger.critical(
                f"An unhandled error occurred in TEA main: {e_main}",
                exc_info=True,
            )
        else:
            print("CRITICAL: Failed to initialize logger.")
            traceback.print_exc()
        sys.exit(2)


def calculate_nuclear_replacement_costs(nuclear_capex_breakdown: dict, year: int) -> dict:
    """
    Calculate nuclear power plant replacement costs for a specific year.

    Args:
        nuclear_capex_breakdown: Nuclear CAPEX breakdown dictionary
        year: Project year

    Returns:
        Dictionary containing replacement costs for the year
    """
    nuclear_replacement_costs = {}
    total_replacement_cost = 0

    total_nuclear_capex = nuclear_capex_breakdown.get("Total_Nuclear_CAPEX", 0)

    for component_name, component_details in NUCLEAR_REPLACEMENT_SCHEDULE.items():
        replacement_years = component_details["years"]

        if year in replacement_years:
            if "cost_percent_initial_capex" in component_details:
                cost_percent = component_details["cost_percent_initial_capex"]
                replacement_cost = total_nuclear_capex * cost_percent
            else:
                replacement_cost = component_details.get("cost", 0)

            nuclear_replacement_costs[component_name] = replacement_cost
            total_replacement_cost += replacement_cost

            logger.debug(
                f"Nuclear {component_name} replacement in year {year}: ${replacement_cost:,.0f}")

    if total_replacement_cost > 0:
        nuclear_replacement_costs["Total_Nuclear_Replacement"] = total_replacement_cost
        logger.info(
            f"Total Nuclear replacement costs (Year {year}): ${total_replacement_cost:,.0f}")

    return nuclear_replacement_costs


def calculate_nuclear_integrated_cash_flows(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    project_lifetime: int,
    construction_period: int,
    h2_subsidy_value: float,
    h2_subsidy_duration: int,
    capex_details: dict,
    om_details: dict,
    replacement_details: dict,
    optimized_capacities: dict,
) -> np.ndarray:
    """
    Calculate cash flows for nuclear integrated system (nuclear + hydrogen production).

    This extends the existing cash flow calculation to include nuclear plant costs.
    """
    logger.info("Calculating nuclear integrated cash flows")

    # Calculate hydrogen system cash flows using nuclear project timeline
    h2_cash_flows = calculate_cash_flows(
        annual_metrics=annual_metrics,
        project_lifetime=project_lifetime,  # Use nuclear timeline (60 years)
        # Use nuclear construction period (8 years)
        construction_period=construction_period,
        h2_subsidy_value=h2_subsidy_value,
        h2_subsidy_duration=h2_subsidy_duration,
        capex_details=capex_details,
        om_details=om_details,
        replacement_details=replacement_details,
        optimized_capacities=optimized_capacities,
    )

    # Calculate nuclear CAPEX breakdown
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw)

    # Estimate annual nuclear generation (assuming 90% capacity factor)
    capacity_factor = 0.90
    annual_generation_mwh = nuclear_capacity_mw * HOURS_IN_YEAR * capacity_factor

    # Initialize nuclear cash flows array
    total_years = construction_period + project_lifetime
    nuclear_cash_flows = np.zeros(total_years)

    # Add nuclear CAPEX during construction period
    for component_name, component_details in NUCLEAR_CAPEX_COMPONENTS.items():
        payment_schedule = component_details["payment_schedule_years"]
        component_capex = nuclear_capex_breakdown[component_name]

        for year_offset, payment_fraction in payment_schedule.items():
            # Year offset from start of project (0-based)
            year_index = year_offset
            if 0 <= year_index < total_years:
                capex_payment = component_capex * payment_fraction
                # Negative for costs
                nuclear_cash_flows[year_index] -= capex_payment
                logger.debug(
                    f"Nuclear {component_name} CAPEX payment in year {year_index}: ${capex_payment:,.0f}")

    # Add nuclear OPEX during operational years
    for year in range(construction_period, total_years):
        operational_year = year - construction_period + 1
        nuclear_opex = calculate_nuclear_annual_opex(
            nuclear_capacity_mw, annual_generation_mwh, operational_year
        )
        # Negative for costs
        nuclear_cash_flows[year] -= nuclear_opex["Total_Nuclear_OPEX"]

        # Add nuclear replacement costs
        nuclear_replacements = calculate_nuclear_replacement_costs(
            nuclear_capex_breakdown, operational_year
        )
        if nuclear_replacements:
            total_replacement = nuclear_replacements.get(
                "Total_Nuclear_Replacement", 0)
            nuclear_cash_flows[year] -= total_replacement  # Negative for costs

    # Combine hydrogen and nuclear cash flows
    integrated_cash_flows = h2_cash_flows + nuclear_cash_flows

    logger.info("Nuclear integrated cash flows calculated successfully")
    logger.debug(
        f"Total nuclear CAPEX impact: ${np.sum(nuclear_cash_flows[nuclear_cash_flows < 0]):,.0f}")

    return integrated_cash_flows


def calculate_nuclear_integrated_financial_metrics(
    annual_metrics: dict,
    nuclear_capacity_mw: float = None,
    enable_nuclear_analysis: bool = False
) -> dict:
    """
    Calculate financial metrics for nuclear integrated system.

    This function either runs standard analysis or nuclear integrated analysis
    based on configuration.
    """
    if not enable_nuclear_analysis or not NUCLEAR_INTEGRATED_CONFIG.get("enabled", False):
        logger.info(
            "Nuclear integrated analysis disabled - running standard analysis")
        return None

    # Use nuclear configuration parameters
    nuclear_lifetime = NUCLEAR_INTEGRATED_CONFIG["project_lifetime_years"]
    nuclear_construction = NUCLEAR_INTEGRATED_CONFIG["construction_years"]

    if nuclear_capacity_mw is None:
        nuclear_capacity_mw = NUCLEAR_INTEGRATED_CONFIG["nuclear_plant_capacity_mw"]

    logger.info(
        f"Running nuclear integrated analysis for {nuclear_capacity_mw} MW nuclear capacity")

    # Get optimized capacities (these should include Nuclear_Plant_Capacity_MW)
    optimized_caps = {
        "Electrolyzer_Capacity_MW": annual_metrics.get("Electrolyzer_Capacity_MW", 0),
        "H2_Storage_Capacity_kg": annual_metrics.get("H2_Storage_Capacity_kg", 0),
        "Battery_Capacity_MWh": annual_metrics.get("Battery_Capacity_MWh", 0),
        "Battery_Power_MW": annual_metrics.get("Battery_Power_MW", 0),
        "Nuclear_Plant_Capacity_MW": nuclear_capacity_mw,
    }

    # Get subsidy parameters (these should be available from tea_sys_params)
    h2_subsidy_val = annual_metrics.get("h2_subsidy_value_usd_per_kg", 0)
    h2_subsidy_yrs = annual_metrics.get("h2_subsidy_duration_years", 0)

    # Calculate nuclear integrated cash flows
    integrated_cash_flows = calculate_nuclear_integrated_cash_flows(
        annual_metrics=annual_metrics,
        nuclear_capacity_mw=nuclear_capacity_mw,
        project_lifetime=nuclear_lifetime,
        construction_period=nuclear_construction,
        h2_subsidy_value=h2_subsidy_val,
        h2_subsidy_duration=h2_subsidy_yrs,
        capex_details=CAPEX_COMPONENTS,
        om_details=OM_COMPONENTS,
        replacement_details=REPLACEMENT_SCHEDULE,
        optimized_capacities=optimized_caps,
    )

    # Calculate financial metrics for nuclear integrated system
    nuclear_financial_metrics = calculate_financial_metrics(
        cash_flows_input=integrated_cash_flows,
        discount_rt=DISCOUNT_RATE,
        annual_h2_prod_kg=annual_metrics.get("H2_Production_kg_annual", 0),
        project_lt=nuclear_lifetime,
        construction_p=nuclear_construction,
    )

    # Add nuclear-specific metrics
    nuclear_capex = calculate_nuclear_capex_breakdown(nuclear_capacity_mw)
    nuclear_financial_metrics["Nuclear_Total_CAPEX_USD"] = nuclear_capex["Total_Nuclear_CAPEX"]
    nuclear_financial_metrics["Nuclear_Capacity_MW"] = nuclear_capacity_mw
    nuclear_financial_metrics["Nuclear_Project_Lifetime_Years"] = nuclear_lifetime
    nuclear_financial_metrics["Nuclear_Construction_Period_Years"] = nuclear_construction

    # Calculate nuclear-specific LCOE (Levelized Cost of Electricity)
    annual_generation_mwh = nuclear_capacity_mw * \
        HOURS_IN_YEAR * 0.90  # 90% capacity factor
    if annual_generation_mwh > 0:
        # Calculate LCOE as the electricity cost that makes NPV = 0
        total_costs_pv = -np.sum([cf / (1 + DISCOUNT_RATE) ** i for i,
                                 cf in enumerate(integrated_cash_flows) if cf < 0])
        total_generation_pv = np.sum([annual_generation_mwh / (1 + DISCOUNT_RATE) ** i for i in range(
            nuclear_construction, nuclear_construction + nuclear_lifetime)])
        nuclear_financial_metrics["Nuclear_LCOE_USD_per_MWh"] = total_costs_pv / \
            total_generation_pv if total_generation_pv > 0 else 0

    logger.info("Nuclear integrated financial metrics calculated successfully")
    return nuclear_financial_metrics


def calculate_greenfield_integrated_cash_flows(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    project_lifetime: int,
    construction_period: int,
    h2_subsidy_value: float,
    h2_subsidy_duration: int,
    capex_details: dict,
    om_details: dict,
    replacement_details: dict,
    optimized_capacities: dict,
) -> np.ndarray:
    """
    Calculate cash flows for greenfield nuclear-hydrogen integrated system.
    This is similar to nuclear_integrated_cash_flows but designed for the greenfield analysis.
    """
    logger.info("Calculating greenfield nuclear-hydrogen integrated cash flows")

    # Use the existing nuclear integrated cash flow calculation
    # This already combines nuclear and hydrogen system costs
    integrated_cash_flows = calculate_nuclear_integrated_cash_flows(
        annual_metrics=annual_metrics,
        nuclear_capacity_mw=nuclear_capacity_mw,
        project_lifetime=project_lifetime,
        construction_period=construction_period,
        h2_subsidy_value=h2_subsidy_value,
        h2_subsidy_duration=h2_subsidy_duration,
        capex_details=capex_details,
        om_details=om_details,
        replacement_details=replacement_details,
        optimized_capacities=optimized_capacities,
    )

    logger.info("Greenfield integrated cash flows calculated successfully")
    return integrated_cash_flows
