"""
Sensitivity analysis script for the nuclear-hydrogen optimization framework.
This script performs comprehensive sensitivity analysis on various parameters
and generates detailed reports and visualizations.
"""

import copy
import json
import logging
import os
import shutil

# Import optimization framework from ../src
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import (
    ENABLE_BATTERY,
    ENABLE_ELECTROLYZER,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_H2_STORAGE,
    ENABLE_LOW_TEMP_ELECTROLYZER,
    ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_STARTUP_SHUTDOWN,
    SIMULATE_AS_DISPATCH_EXECUTION,
    TARGET_ISO,
)
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results

# --- Configuration ---
BASE_INPUT_DIR = "../input/hourly_data"
BASE_SYS_DATA_FILE = os.path.join(BASE_INPUT_DIR, "sys_data_advanced.csv")
SENSITIVITY_OUTPUT_DIR = "../sensitivity_analysis_results"
SENSITIVITY_OUTPUT_CSV = os.path.join(
    SENSITIVITY_OUTPUT_DIR, "sensitivity_analysis_results.csv"
)
TEMP_RUN_DIR_BASE = "../temp_sensitivity_runs"

# Parameters to vary: {parameter_name_in_csv: [min_change_%, max_change_%, num_steps]}
PARAMETERS_TO_VARY = {
    # Economic Parameters
    "H2_value_USD_per_kg": [-30, 30, 7],  # Hydrogen market value
    "vom_electrolyzer_USD_per_MWh": [-40, 40, 9],  # O&M cost
    "vom_electrolyzer_USD_per_MWh_LTE": [-40, 40, 9],  # LTE-specific O&M
    "vom_electrolyzer_USD_per_MWh_HTE": [-40, 40, 9],  # HTE-specific O&M
    "vom_turbine_USD_per_MWh": [-30, 30, 7],  # Turbine O&M
    "cost_startup_electrolyzer_USD_per_startup_LTE": [
        -50,
        50,
        7,
    ],  # LTE startup cost
    "cost_startup_electrolyzer_USD_per_startup_HTE": [
        -50,
        50,
        7,
    ],  # HTE startup cost
    "cost_water_USD_per_kg_h2": [-30, 30, 7],  # Water cost
    "cost_h2_storage_USD_per_kg": [-40, 40, 9],  # H2 storage cost
    "cost_electrolyzer_capacity_USD_per_MW_year_LTE": [
        -20,
        20,
        5,
    ],  # LTE capex
    "cost_electrolyzer_capacity_USD_per_MW_year_HTE": [
        -20,
        20,
        5,
    ],  # HTE capex
    "cost_electrolyzer_ramping_USD_per_MW_ramp": [-30, 30, 7],  # Ramping cost
    "hydrogen_subsidy_value_usd_per_kg": [
        0,
        5,
        6,
    ],  # Absolute values, not percentage change
    "hydrogen_subsidy_duration_years": [5, 20, 4],  # New: subsidy duration
    "BatteryCapex_USD_per_MWh_year": [-25, 25, 6],  # Battery energy capex
    "BatteryCapex_USD_per_MW_year": [-25, 25, 6],  # Battery power capex
    "BatteryFixedOM_USD_per_MWh_year": [-30, 30, 7],  # Battery fixed O&M
    # Technical Parameters
    "pElectrolyzer_max_upper_bound_MW": [-30, 30, 7],  # Electrolyzer capacity
    "pTurbine_max_MW": [-20, 20, 5],  # Turbine capacity
    "H2_storage_capacity_max_kg": [-40, 40, 9],  # H2 storage size
    "Turbine_RampUp_Rate_Percent_per_Min": [-50, 100, 7],  # Turbine ramp rate
    "Turbine_RampDown_Rate_Percent_per_Min": [
        -50,
        100,
        7,
    ],  # Turbine ramp down rate
    "Electrolyzer_RampUp_Rate_Percent_per_Min_LTE": [
        -50,
        100,
        7,
    ],  # LTE ramp rate
    "Electrolyzer_RampDown_Rate_Percent_per_Min_LTE": [
        -50,
        100,
        7,
    ],  # LTE ramp down rate
    "Electrolyzer_RampUp_Rate_Percent_per_Min_HTE": [
        -50,
        100,
        7,
    ],  # HTE ramp rate
    "Electrolyzer_RampDown_Rate_Percent_per_Min_HTE": [
        -50,
        100,
        7,
    ],  # HTE ramp down rate
    "ke_H2_Values_MWh_per_kg": [-15, 15, 7],  # Electrolyzer efficiency
    "ke_H2_Values_MWh_per_kg_LTE": [-15, 15, 7],  # LTE efficiency
    "ke_H2_Values_MWh_per_kg_HTE": [-15, 15, 7],  # HTE efficiency
    "Turbine_Thermal_Elec_Efficiency_Const": [
        -15,
        15,
        7,
    ],  # Turbine efficiency
    "DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE": [
        0,
        300,
        7,
    ],  # LTE degradation
    "DegradationFactorOperation_Units_per_Hour_at_MaxLoad_HTE": [
        0,
        300,
        7,
    ],  # HTE degradation
    "DegradationFactorStartup_Units_per_Startup_LTE": [
        0,
        300,
        7,
    ],  # LTE startup degradation
    "DegradationFactorStartup_Units_per_Startup_HTE": [
        0,
        300,
        7,
    ],  # HTE startup degradation
    "BatteryChargeEff": [-15, 15, 7],  # Battery charge efficiency
    "BatteryDischargeEff": [-15, 15, 7],  # Battery discharge efficiency
    "storage_charge_eff_fraction": [
        -15,
        15,
        7,
    ],  # H2 storage charge efficiency
    "storage_discharge_eff_fraction": [
        -15,
        15,
        7,
    ],  # H2 storage discharge efficiency
    "H2_storage_charge_rate_max_kg_per_hr": [
        -30,
        30,
        7,
    ],  # H2 storage charge rate
    "H2_storage_discharge_rate_max_kg_per_hr": [
        -30,
        30,
        7,
    ],  # H2 storage discharge rate
    # Market Parameters
    "energy_price_multiplier": [-40, 40, 9],  # Overall energy price scaling
    "ancillary_service_price_multiplier": [
        -40,
        40,
        9,
    ],  # Overall AS price scaling
}

# Additional scenario analysis parameters
SCENARIO_ANALYSES = {
    "technology_combinations": [
        {
            "name": "Nuclear + LTE",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": True,
                "ENABLE_BATTERY": False,
                "ENABLE_H2_STORAGE": False,
            },
        },
        {
            "name": "Nuclear + HTE",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": False,
                "ENABLE_BATTERY": False,
                "ENABLE_H2_STORAGE": False,
            },
        },
        {
            "name": "Nuclear + LTE + Storage",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": True,
                "ENABLE_BATTERY": False,
                "ENABLE_H2_STORAGE": True,
            },
        },
        {
            "name": "Nuclear + HTE + Storage",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": False,
                "ENABLE_BATTERY": False,
                "ENABLE_H2_STORAGE": True,
            },
        },
        {
            "name": "Nuclear + LTE + Battery",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": True,
                "ENABLE_BATTERY": True,
                "ENABLE_H2_STORAGE": False,
            },
        },
        {
            "name": "Nuclear + HTE + Battery",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": False,
                "ENABLE_BATTERY": True,
                "ENABLE_H2_STORAGE": False,
            },
        },
        {
            "name": "Full System (LTE)",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": True,
                "ENABLE_BATTERY": True,
                "ENABLE_H2_STORAGE": True,
            },
        },
        {
            "name": "Full System (HTE)",
            "config": {
                "ENABLE_NUCLEAR_GENERATOR": True,
                "ENABLE_ELECTROLYZER": True,
                "ENABLE_LOW_TEMP_ELECTROLYZER": False,
                "ENABLE_BATTERY": True,
                "ENABLE_H2_STORAGE": True,
            },
        },
    ],
    "market_scenarios": [
        {
            "name": "Base Case",
            "energy_price_multiplier": 1.0,
            "ancillary_service_price_multiplier": 1.0,
        },
        {
            "name": "High Energy Prices",
            "energy_price_multiplier": 1.5,
            "ancillary_service_price_multiplier": 1.0,
        },
        {
            "name": "High AS Prices",
            "energy_price_multiplier": 1.0,
            "ancillary_service_price_multiplier": 1.5,
        },
        {
            "name": "Low Energy Prices",
            "energy_price_multiplier": 0.7,
            "ancillary_service_price_multiplier": 1.0,
        },
        {
            "name": "Low AS Prices",
            "energy_price_multiplier": 1.0,
            "ancillary_service_price_multiplier": 0.7,
        },
        {
            "name": "High Overall Prices",
            "energy_price_multiplier": 1.3,
            "ancillary_service_price_multiplier": 1.3,
        },
        {
            "name": "Low Overall Prices",
            "energy_price_multiplier": 0.7,
            "ancillary_service_price_multiplier": 0.7,
        },
    ],
    "h2_value_scenarios": [
        {
            "name": "Current H2 Value",
            "H2_value_USD_per_kg": 4.0,
            "hydrogen_subsidy_value_usd_per_kg": 0.0,
            "hydrogen_subsidy_duration_years": 10,
        },
        {
            "name": "Medium H2 Value",
            "H2_value_USD_per_kg": 6.0,
            "hydrogen_subsidy_value_usd_per_kg": 0.0,
            "hydrogen_subsidy_duration_years": 10,
        },
        {
            "name": "High H2 Value",
            "H2_value_USD_per_kg": 10.0,
            "hydrogen_subsidy_value_usd_per_kg": 0.0,
            "hydrogen_subsidy_duration_years": 10,
        },
        {
            "name": "Current H2 + Short Subsidy",
            "H2_value_USD_per_kg": 4.0,
            "hydrogen_subsidy_value_usd_per_kg": 3.0,
            "hydrogen_subsidy_duration_years": 5,
        },
        {
            "name": "Current H2 + Medium Subsidy",
            "H2_value_USD_per_kg": 4.0,
            "hydrogen_subsidy_value_usd_per_kg": 3.0,
            "hydrogen_subsidy_duration_years": 10,
        },
        {
            "name": "Current H2 + Long Subsidy",
            "H2_value_USD_per_kg": 4.0,
            "hydrogen_subsidy_value_usd_per_kg": 3.0,
            "hydrogen_subsidy_duration_years": 20,
        },
    ],
    "fixed_size_scenarios": [
        {
            "name": "Small Electrolyzer (100MW)",
            "user_specified_electrolyzer_capacity_MW": 100,
        },
        {
            "name": "Medium Electrolyzer (250MW)",
            "user_specified_electrolyzer_capacity_MW": 250,
        },
        {
            "name": "Large Electrolyzer (500MW)",
            "user_specified_electrolyzer_capacity_MW": 500,
        },
        {
            "name": "Small Battery (50MW/200MWh)",
            "user_specified_battery_power_MW": 50,
            "user_specified_battery_energy_MWh": 200,
        },
        {
            "name": "Medium Battery (100MW/400MWh)",
            "user_specified_battery_power_MW": 100,
            "user_specified_battery_energy_MWh": 400,
        },
        {
            "name": "Large Battery (200MW/800MWh)",
            "user_specified_battery_power_MW": 200,
            "user_specified_battery_energy_MWh": 800,
        },
        {
            "name": "Integrated System (250MW Elec, 100MW/400MWh Batt)",
            "user_specified_electrolyzer_capacity_MW": 250,
            "user_specified_battery_power_MW": 100,
            "user_specified_battery_energy_MWh": 400,
        },
    ],
}

# Result metrics to track
RESULT_METRICS = [
    "Total Profit",
    "Energy Revenue",
    "Hydrogen Revenue",
    "Ancillary Service Revenue",
    "Total Cost",
    "H2 Production (kg)",
    "Energy Generation (MWh)",
    "AS Capacity (MW)",
    "Electrolyzer Capacity Factor",
    "Turbine Capacity Factor",
    "H2 Storage Utilization",
    "Battery Cycles",
]


# --- Logging Setup ---
def setup_logging():
    """Setup logging configuration with timestamp-based log file."""
    os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sensitivity_log_file = os.path.join(
        SENSITIVITY_OUTPUT_DIR, f"sensitivity_analysis_{timestamp}.log"
    )
    logging.basicConfig(
        filename=sensitivity_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    # Add console handler for better visibility
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)
    return sensitivity_log_file


def prepare_output_dirs():
    """Prepare all output directories needed for analysis."""
    dirs_to_create = [
        SENSITIVITY_OUTPUT_DIR,
        os.path.join(SENSITIVITY_OUTPUT_DIR, "parameter_analysis"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "scenario_analysis"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "technology_comparison"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "market_scenarios"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "h2_value_scenarios"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "correlation_analysis"),
        os.path.join(SENSITIVITY_OUTPUT_DIR, "raw_data"),
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    return dirs_to_create


def generate_run_config():
    """Generate a configuration tracking file for this sensitivity run."""
    config_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "base_input_dir": BASE_INPUT_DIR,
        "parameters_analyzed": list(PARAMETERS_TO_VARY.keys()),
        "scenario_analyses": {k: len(v) for k, v in SCENARIO_ANALYSES.items()},
        "result_metrics": RESULT_METRICS,
        "base_config": {
            "TARGET_ISO": TARGET_ISO,
            "ENABLE_NUCLEAR_GENERATOR": ENABLE_NUCLEAR_GENERATOR,
            "ENABLE_ELECTROLYZER": ENABLE_ELECTROLYZER,
            "ENABLE_LOW_TEMP_ELECTROLYZER": ENABLE_LOW_TEMP_ELECTROLYZER,
            "ENABLE_BATTERY": ENABLE_BATTERY,
            "ENABLE_H2_STORAGE": ENABLE_H2_STORAGE,
            "ENABLE_H2_CAP_FACTOR": ENABLE_H2_CAP_FACTOR,
            "ENABLE_NONLINEAR_TURBINE_EFF": ENABLE_NONLINEAR_TURBINE_EFF,
            "ENABLE_ELECTROLYZER_DEGRADATION_TRACKING": ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
            "ENABLE_STARTUP_SHUTDOWN": ENABLE_STARTUP_SHUTDOWN,
            "SIMULATE_AS_DISPATCH_EXECUTION": SIMULATE_AS_DISPATCH_EXECUTION,
        },
    }
    with open(
        os.path.join(
            SENSITIVITY_OUTPUT_DIR,
            f"run_config_{config_data['timestamp']}.json",
        ),
        "w",
    ) as f:
        json.dump(config_data, f, indent=4)
    return config_data


def modify_sys_data(original_file, temp_file, param_name, new_value):
    """Reads the sys_data CSV, modifies a parameter, saves to temp file."""
    try:
        df = pd.read_csv(original_file, index_col=0)
        if param_name not in df.index:
            logging.warning(
                f"Parameter '{param_name}' not found in {original_file}. Skipping modification."
            )
            shutil.copyfile(original_file, temp_file)
            return False
        logging.info(
            f"Modifying '{param_name}' from {df.loc[param_name, 'Value']} to {new_value}"
        )
        df.loc[param_name, "Value"] = new_value
        df.to_csv(temp_file)
        return True
    except Exception as e:
        logging.error(
            f"Error modifying parameter '{param_name}' in {original_file}: {e}",
            exc_info=True,
        )
        if os.path.exists(original_file) and not os.path.exists(temp_file):
            shutil.copyfile(original_file, temp_file)
        return False


def modify_hourly_data(
    input_dir, energy_price_multiplier=None, as_price_multiplier=None
):
    """
    Modifies hourly price data according to multipliers.

    Args:
        input_dir: Directory containing hourly data files
        energy_price_multiplier: Multiplier for energy prices (None = no change)
        as_price_multiplier: Multiplier for ancillary service prices (None = no change)
    """
    iso_dirs = [
        d
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
        and d in ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]
    ]

    try:
        # Apply energy price multiplier if provided
        if energy_price_multiplier is not None and energy_price_multiplier != 1.0:
            for iso in iso_dirs:
                price_file = os.path.join(input_dir, iso, "Price_hourly.csv")
                if os.path.exists(price_file):
                    df = pd.read_csv(price_file)
                    price_col = "Price ($/MWh)"
                    if price_col in df.columns:
                        df[price_col] = df[price_col] * energy_price_multiplier
                        df.to_csv(price_file, index=False)
                        logging.info(
                            f"Applied energy price multiplier {energy_price_multiplier} to {price_file}"
                        )

        # Apply AS price multiplier if provided
        if as_price_multiplier is not None and as_price_multiplier != 1.0:
            for iso in iso_dirs:
                ans_price_file = os.path.join(input_dir, iso, "Price_ANS_hourly.csv")
                if os.path.exists(ans_price_file):
                    df = pd.read_csv(ans_price_file)
                    # Apply multiplier to all price columns (exclude timestamps, etc.)
                    for col in df.columns:
                        # Skip non-price columns like timestamps
                        if col.lower() not in [
                            "hour",
                            "timestamp",
                            "date",
                            "time",
                            "datetime",
                        ]:
                            df[col] = df[col] * as_price_multiplier
                    df.to_csv(ans_price_file, index=False)
                    logging.info(
                        f"Applied AS price multiplier {as_price_multiplier} to {ans_price_file}"
                    )

        return True
    except Exception as e:
        logging.error(f"Error modifying hourly data: {e}", exc_info=True)
        return False


def set_config_flags(temp_dir, config_dict):
    """
    Creates a temporary config.py file with updated flags.

    Args:
        temp_dir: Directory where to create the config.py file
        config_dict: Dictionary of configuration flags to set

    Returns:
        Path to the created config file
    """
    try:
        temp_config_file = os.path.join(temp_dir, "config.py")

        # Create a basic config template with default values
        config_template = """# Temporary config file generated for sensitivity analysis
# {timestamp}

# ISO setting
TARGET_ISO = "{target_iso}"

# Hours in year
HOURS_IN_YEAR = 8760

# Feature flags
ENABLE_NUCLEAR_GENERATOR = {enable_nuclear}
ENABLE_ELECTROLYZER = {enable_electrolyzer}
ENABLE_LOW_TEMP_ELECTROLYZER = {enable_lte}
ENABLE_BATTERY = {enable_battery}
ENABLE_H2_STORAGE = {enable_h2_storage}
ENABLE_H2_CAP_FACTOR = {enable_h2_cap}
ENABLE_NONLINEAR_TURBINE_EFF = {enable_nonlinear}
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING = {enable_degradation}
ENABLE_STARTUP_SHUTDOWN = {enable_startup}
SIMULATE_AS_DISPATCH_EXECUTION = {simulate_dispatch}

# Derived setting (calculated)
CAN_PROVIDE_ANCILLARY_SERVICES = ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY)
"""
        # Set default values that will be overridden by config_dict
        default_flags = {
            "target_iso": TARGET_ISO,
            "enable_nuclear": ENABLE_NUCLEAR_GENERATOR,
            "enable_electrolyzer": ENABLE_ELECTROLYZER,
            "enable_lte": ENABLE_LOW_TEMP_ELECTROLYZER,
            "enable_battery": ENABLE_BATTERY,
            "enable_h2_storage": ENABLE_H2_STORAGE,
            "enable_h2_cap": ENABLE_H2_CAP_FACTOR,
            "enable_nonlinear": ENABLE_NONLINEAR_TURBINE_EFF,
            "enable_degradation": ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
            "enable_startup": ENABLE_STARTUP_SHUTDOWN,
            "simulate_dispatch": SIMULATE_AS_DISPATCH_EXECUTION,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Override with provided values
        for k, v in config_dict.items():
            if k == "TARGET_ISO":
                default_flags["target_iso"] = v
            elif k == "ENABLE_NUCLEAR_GENERATOR":
                default_flags["enable_nuclear"] = v
            elif k == "ENABLE_ELECTROLYZER":
                default_flags["enable_electrolyzer"] = v
            elif k == "ENABLE_LOW_TEMP_ELECTROLYZER":
                default_flags["enable_lte"] = v
            elif k == "ENABLE_BATTERY":
                default_flags["enable_battery"] = v
            elif k == "ENABLE_H2_STORAGE":
                default_flags["enable_h2_storage"] = v
            elif k == "ENABLE_H2_CAP_FACTOR":
                default_flags["enable_h2_cap"] = v
            elif k == "ENABLE_NONLINEAR_TURBINE_EFF":
                default_flags["enable_nonlinear"] = v
            elif k == "ENABLE_ELECTROLYZER_DEGRADATION_TRACKING":
                default_flags["enable_degradation"] = v
            elif k == "ENABLE_STARTUP_SHUTDOWN":
                default_flags["enable_startup"] = v
            elif k == "SIMULATE_AS_DISPATCH_EXECUTION":
                default_flags["simulate_dispatch"] = v

        # Format the template with our values
        config_content = config_template.format(**default_flags)

        # Write to file
        with open(temp_config_file, "w") as f:
            f.write(config_content)

        return temp_config_file
    except Exception as e:
        logging.error(f"Error creating config file: {e}", exc_info=True)
        return None


def run_single_optimization(
    input_dir, iso=None, solver_name="gurobi", config_override=None
):
    """
    Runs the optimization for a given input directory. Returns a dictionary of results.

    Args:
        input_dir: Directory containing input data
        iso: ISO to run for (default: None, which uses TARGET_ISO from config)
        solver_name: Solver to use (default: gurobi)
        config_override: Dictionary with config flags to override (default: None)

    Returns:
        Dictionary of results with metrics from RESULT_METRICS
    """
    try:
        # Create custom config file if config_override is provided
        custom_config_file = None
        if config_override:
            custom_config_file = set_config_flags(input_dir, config_override)
            if custom_config_file:
                logging.info(f"Created custom config at {custom_config_file}")
                # Make sure Python can import from this directory
                sys.path.insert(0, input_dir)
            else:
                logging.warning("Failed to create custom config, using default")

        # Determine which ISO to use
        target_iso = iso if iso else TARGET_ISO

        # Load data
        data = load_hourly_data(target_iso, base_dir=input_dir)
        if data is None:
            logging.error(f"Data loading failed for {input_dir}")
            return {metric: np.nan for metric in RESULT_METRICS}

        # Create model
        model = create_model(
            data, target_iso, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION
        )

        # Solve
        solver = SolverFactory(solver_name)

        # Check solver availability
        if not solver.available():
            logging.warning(
                f"Solver {solver_name} not available, trying alternative solvers..."
            )
            for alt_solver in ["cbc", "glpk", "ipopt"]:
                solver = SolverFactory(alt_solver)
                if solver.available():
                    logging.info(f"Using alternative solver: {alt_solver}")
                    break
            else:
                logging.error("No available solvers found.")
                return {metric: np.nan for metric in RESULT_METRICS}

        # Configure solver options based on solver type
        if solver.name == "gurobi":
            solver.options["TimeLimit"] = 3600  # 1 hour time limit
            solver.options["MIPGap"] = 0.01  # 1% MIP gap
        elif solver.name == "cbc":
            solver.options["sec"] = 3600  # 1 hour time limit
            solver.options["ratio"] = 0.01  # 1% MIP gap
        elif solver.name == "glpk":
            solver.options["tmlim"] = 3600  # 1 hour time limit

        logging.info(f"Using solver: {solver.name}")
        results = solver.solve(model, tee=False)

        if results.solver.status != SolverStatus.ok:
            logging.error(f"Solver failed: {results.solver.status}")
            return {metric: np.nan for metric in RESULT_METRICS}

        if results.solver.termination_condition not in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            logging.warning(
                f"Solver termination: {results.solver.termination_condition}"
            )

        # Extract results
        try:
            result_dict = {}

            # Map metrics from RESULT_METRICS to model attributes
            metric_mapping = {
                "Total Profit": "TotalProfit_Objective",
                "Energy Revenue": "EnergyRevenueExpr",
                "Hydrogen Revenue": "HydrogenRevenueExpr",
                "Ancillary Service Revenue": "AncillaryRevenueExpr",
                "Total Cost": "TotalCostExpr",  # OpexCostExpr + AnnualizedCapexExpr
                "H2 Production (kg)": "Total_H2_produced",
                "Energy Generation (MWh)": "Total_Energy_Generation",
                "AS Capacity (MW)": "Total_AS_Capacity",
                "Electrolyzer Capacity Factor": "Electrolyzer_Capacity_Factor",
                "Turbine Capacity Factor": "Turbine_Capacity_Factor",
                "H2 Storage Utilization": "H2_Storage_Utilization",
                "Battery Cycles": "Battery_Cycles",
            }

            # Get metrics from model
            for metric, model_attr in metric_mapping.items():
                attr_name = model_attr.replace(" ", "_")
                if hasattr(model, attr_name):
                    result_dict[metric] = pyo.value(getattr(model, attr_name))
                else:
                    result_dict[metric] = np.nan

            # Calculate derived metrics if not directly available from model
            if "Total Cost" not in result_dict or np.isnan(result_dict["Total Cost"]):
                opex = 0.0
                capex = 0.0
                if hasattr(model, "OpexCostExpr"):
                    opex = pyo.value(model.OpexCostExpr)
                if hasattr(model, "AnnualizedCapexExpr"):
                    capex = pyo.value(model.AnnualizedCapexExpr)
                result_dict["Total Cost"] = opex + capex

            # Extract summary results for additional metrics
            try:
                # Extract full results to get summary
                results_df, summary_results = extract_results(
                    model,
                    target_iso,
                    output_dir=os.path.join(input_dir, "results"),
                )

                # Map summary results to our metrics if needed
                if "H2 Production (kg)" not in result_dict or np.isnan(
                    result_dict["H2 Production (kg)"]
                ):
                    result_dict["H2 Production (kg)"] = summary_results.get(
                        "Total_H2_Produced_kg", 0.0
                    )

                if "Electrolyzer Capacity Factor" not in result_dict or np.isnan(
                    result_dict["Electrolyzer Capacity Factor"]
                ):
                    result_dict["Electrolyzer Capacity Factor"] = summary_results.get(
                        "Electrolyzer_Capacity_Factor_Actual", 0.0
                    )

                # Add any other metrics we care about
                for key, value in summary_results.items():
                    if (
                        key not in result_dict
                        and "Total_" in key
                        and isinstance(value, (int, float, np.number))
                    ):
                        result_dict[key] = value

            except Exception as e:
                logging.warning(f"Error extracting detailed results: {e}")
                # Continue with what we have so far

            # Clean up path if we added it
            if config_override and input_dir in sys.path:
                sys.path.remove(input_dir)

            logging.info(f"Run completed with results: {result_dict}")
            return result_dict

        except Exception as e:
            logging.error(f"Error extracting results: {e}")

            # Clean up path if we added it
            if config_override and input_dir in sys.path:
                sys.path.remove(input_dir)

            return {metric: np.nan for metric in RESULT_METRICS}

    except Exception as e:
        logging.error(f"Exception during optimization: {e}", exc_info=True)

        # Clean up path if we added it
        if config_override and input_dir in sys.path:
            sys.path.remove(input_dir)

        return {metric: np.nan for metric in RESULT_METRICS}


def plot_sensitivity_results(results_df, output_dir):
    """
    Generate enhanced plots for sensitivity analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a summary plot across all parameters
    plt.figure(figsize=(12, 8))

    # Group by parameter, get max percentage change for each metric
    summary_data = []
    for param in results_df["Parameter"].unique():
        if param == "Base Case":
            continue

        param_data = results_df[results_df["Parameter"] == param]
        param_summary = {"Parameter": param}

        for metric in RESULT_METRICS:
            if f"{metric}_Change_Pct" in param_data.columns:
                change_values = param_data[f"{metric}_Change_Pct"].abs()
                if not change_values.empty and not all(np.isnan(change_values)):
                    param_summary[f"{metric}_Max_Change"] = change_values.max()

        summary_data.append(param_summary)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Plot the top N most sensitive parameters for each metric
        for metric in RESULT_METRICS:
            change_col = f"{metric}_Max_Change"
            if change_col in summary_df.columns:
                # Get top 10 parameters with highest impact
                top_params = summary_df.sort_values(
                    by=change_col, ascending=False
                ).head(10)

                if not top_params.empty:
                    plt.figure(figsize=(10, 6))
                    ax = sns.barplot(data=top_params, x="Parameter", y=change_col)
                    ax.set_title(f"Top 10 Parameters Affecting {metric}")
                    ax.set_xlabel("Parameter")
                    ax.set_ylabel("Maximum Absolute % Change")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            output_dir,
                            f'top_parameters_{metric.replace(" ", "_")}.png',
                        )
                    )
                    plt.close()

    # Plot for each parameter and metric
    for param in results_df["Parameter"].unique():
        if param == "Base Case":
            continue

        param_data = results_df[results_df["Parameter"] == param]

        # Create subplots for each metric
        n_metrics = len(RESULT_METRICS)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics))
        fig.suptitle(f"Sensitivity Analysis: {param}")

        for idx, metric in enumerate(RESULT_METRICS):
            ax = axes[idx] if n_metrics > 1 else axes

            # Plot absolute values
            ax.plot(
                param_data["Variation (%)"],
                param_data[metric],
                "b-",
                label="Absolute Value",
            )

            # Plot percentage change if applicable
            if f"{metric}_Change_Pct" in param_data.columns:
                ax2 = ax.twinx()
                ax2.plot(
                    param_data["Variation (%)"],
                    param_data[f"{metric}_Change_Pct"],
                    "r--",
                    label="% Change",
                )
                ax2.set_ylabel("Percentage Change (%)")

            ax.set_xlabel("Parameter Variation (%)")
            ax.set_ylabel(metric)
            ax.grid(True)

            if idx == 0:
                ax.legend(loc="upper left")
                if f"{metric}_Change_Pct" in param_data.columns:
                    ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'sensitivity_{param.replace(" ", "_")}.png')
        )
        plt.close()

        # Create correlation matrix for this parameter
        corr_data = param_data.filter(
            regex="^(?!.*_Change_Pct).*$"
        )  # Exclude % change columns
        corr_data = corr_data.select_dtypes(
            include=[np.number]
        )  # Keep only numeric columns

        if len(corr_data.columns) > 2:  # Need at least parameter value + 2 metrics
            plt.figure(figsize=(10, 8))
            corr_matrix = corr_data.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                fmt=".2f",
            )
            plt.title(f"Correlation Matrix for {param}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'correlation_{param.replace(" ", "_")}.png')
            )
            plt.close()


def create_tornado_plots(results_df, output_dir):
    """
    Create tornado plots to show relative impact of parameters on key metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # For each metric, create a tornado plot
    for metric in RESULT_METRICS:
        change_col = f"{metric}_Change_Pct"

        # Skip if this metric doesn't have % change data
        if change_col not in results_df.columns:
            continue

        # Get all parameters and their max absolute % changes
        impact_data = []

        for param in results_df["Parameter"].unique():
            if param == "Base Case":
                continue

            param_data = results_df[results_df["Parameter"] == param]

            if change_col in param_data.columns:
                max_increase = param_data[change_col].max()
                max_decrease = param_data[change_col].min()

                # Only include if there's actual variation
                if not (np.isnan(max_increase) and np.isnan(max_decrease)):
                    impact_data.append(
                        {
                            "Parameter": param,
                            "Max Increase (%)": (
                                max_increase if not np.isnan(max_increase) else 0
                            ),
                            "Max Decrease (%)": (
                                max_decrease if not np.isnan(max_decrease) else 0
                            ),
                        }
                    )

        if impact_data:
            impact_df = pd.DataFrame(impact_data)

            # Sort by total impact (abs sum of increase and decrease)
            impact_df["Total Impact"] = abs(impact_df["Max Increase (%)"]) + abs(
                impact_df["Max Decrease (%)"]
            )
            impact_df = impact_df.sort_values("Total Impact", ascending=False).head(
                15
            )  # Top 15

            # Create tornado plot
            plt.figure(figsize=(10, 8))

            # Get parameters in order of impact
            params = impact_df["Parameter"].tolist()

            # Plot horizontal bars
            y_pos = range(len(params))
            plt.barh(
                y_pos,
                impact_df["Max Increase (%)"],
                height=0.8,
                color="green",
                alpha=0.6,
                label="Max Increase",
            )
            plt.barh(
                y_pos,
                impact_df["Max Decrease (%)"],
                height=0.8,
                color="red",
                alpha=0.6,
                label="Max Decrease",
            )

            # Set y-tick labels to parameter names
            plt.yticks(y_pos, params)

            # Add grid, title, labels
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.title(f"Impact of Parameters on {metric}")
            plt.xlabel("Percentage Change (%)")
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f'tornado_{metric.replace(" ", "_")}.png')
            )
            plt.close()


def analyze_scenario_results(scenario_results, scenario_type, output_dir):
    """
    Analyze results from scenario analysis

    Args:
        scenario_results: List of dictionaries with scenario results
        scenario_type: Type of scenario analysis (technology, market, h2_value)
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    if not scenario_results:
        logging.warning(f"No results for {scenario_type} scenario analysis")
        return

    # Convert results to DataFrame
    results_df = pd.DataFrame(scenario_results)

    # Save raw results
    csv_file = os.path.join(output_dir, f"{scenario_type}_scenarios.csv")
    results_df.to_csv(csv_file, index=False)

    # Get metrics columns (exclude scenario name/description columns)
    metric_cols = [col for col in results_df.columns if col in RESULT_METRICS]

    if not metric_cols:
        logging.warning(f"No metrics found in {scenario_type} results")
        return

    # Plot comparisons
    for metric in metric_cols:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=results_df, x="Scenario", y=metric)
        ax.set_title(f'{metric} by {scenario_type.replace("_", " ").title()} Scenario')
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{scenario_type}_{metric.replace(" ", "_")}.png')
        )
        plt.close()

    # Create a heatmap for all metrics
    plt.figure(figsize=(12, 8))

    # Normalize each metric column for better visualization
    normalized_df = results_df.copy()
    for col in metric_cols:
        if normalized_df[col].std() > 0:  # Avoid division by zero
            normalized_df[col] = (
                normalized_df[col] - normalized_df[col].mean()
            ) / normalized_df[col].std()
        else:
            normalized_df[col] = 0  # If no variation, set to 0

    # Create pivot table for heatmap
    pivot_data = normalized_df.set_index("Scenario")[metric_cols]

    # Plot heatmap
    sns.heatmap(pivot_data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(
        f'Normalized Metrics by {scenario_type.replace("_", " ").title()} Scenario'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{scenario_type}_heatmap.png"))
    plt.close()


def analyze_size_combinations(output_dir=None):
    """
    A comprehensive sensitivity analysis of electrolyzer and cell size combinations is performed.
    This function tests the system performance for different combinations of electrolyzer capacity
    and cell power/energy capacity.

    Args:
        output_dir: The output directory, default to a dedicated directory in SENSITIVITY_OUTPUT_DIR

    Returns:
        DataFrame containing the results
    """
    if output_dir is None:
        output_dir = os.path.join(SENSITIVITY_OUTPUT_DIR, "size_combination_analysis")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_RUN_DIR_BASE, exist_ok=True)

    # Set the range of electrolyzer capacity and cell size
    electrolyzer_sizes = [100, 200, 300, 400, 500]  # MW
    battery_power_sizes = [50, 100, 150, 200]  # MW
    battery_energy_durations = [2, 4, 6, 8]  # hours

    # Calculate total number of runs
    total_combinations = (
        len(electrolyzer_sizes)
        * len(battery_power_sizes)
        * len(battery_energy_durations)
    )
    logging.info(
        f"Starting size combination analysis with {total_combinations} combinations"
    )

    # Create a list to store results
    all_results = []

    # Track current run
    current_run = 0

    # Run optimization for each size combination
    for elec_size in electrolyzer_sizes:
        for batt_power in battery_power_sizes:
            for batt_duration in battery_energy_durations:
                # Calculate battery energy capacity
                batt_energy = batt_power * batt_duration

                # Create run directory
                run_name = f"E{elec_size}_BP{batt_power}_BE{batt_energy}"
                run_dir = os.path.join(TEMP_RUN_DIR_BASE, f"size_combo_{run_name}")

                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir)
                shutil.copytree(BASE_INPUT_DIR, run_dir)

                # Update progress
                current_run += 1
                print(
                    f"Running size combination {current_run}/{total_combinations}: "
                    f"Electrolyzer={elec_size}MW, Battery={batt_power}MW/{batt_energy}MWh"
                )

                # Modify system data file
                sys_data_file = os.path.join(run_dir, "sys_data_advanced.csv")
                temp_file_1 = os.path.join(run_dir, "temp_sys_data_1.csv")
                temp_file_2 = os.path.join(run_dir, "temp_sys_data_2.csv")

                # Apply changes
                modify_sys_data(
                    BASE_SYS_DATA_FILE,
                    temp_file_1,
                    "user_specified_electrolyzer_capacity_MW",
                    elec_size,
                )
                modify_sys_data(
                    temp_file_1,
                    temp_file_2,
                    "user_specified_battery_power_MW",
                    batt_power,
                )
                modify_sys_data(
                    temp_file_2,
                    sys_data_file,
                    "user_specified_battery_energy_MWh",
                    batt_energy,
                )

                # Run optimization
                results = run_single_optimization(run_dir, iso=TARGET_ISO)

                # Record results
                result_dict = {
                    "Electrolyzer Size (MW)": elec_size,
                    "Battery Power (MW)": batt_power,
                    "Battery Duration (h)": batt_duration,
                    "Battery Energy (MWh)": batt_energy,
                    "E/B Power Ratio": (
                        elec_size / batt_power if batt_power > 0 else float("inf")
                    ),
                }
                result_dict.update(results)
                all_results.append(result_dict)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(output_dir, "size_combination_results.csv")
    results_df.to_csv(results_csv, index=False)

    # Create heatmap visualizing key metrics
    # Create a heatmap for each key metric
    key_metrics = [
        "Total Profit",
        "Energy Revenue",
        "Hydrogen Revenue",
        "Ancillary Service Revenue",
        "Total Cost",
        "H2 Production (kg)",
    ]

    for metric in key_metrics:
        if metric in results_df.columns:
            for duration in results_df["Battery Duration (h)"].unique():
                # Filter data for specific battery duration
                df_duration = results_df[results_df["Battery Duration (h)"] == duration]

                if len(df_duration) > 0:
                    # Create pivot table
                    pivot_data = df_duration.pivot(
                        index="Electrolyzer Size (MW)",
                        columns="Battery Power (MW)",
                        values=metric,
                    )

                    # Plot heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_data, annot=True, cmap="viridis", fmt=".2f")
                    plt.title(f"{metric} - Battery Duration: {duration}h")
                    plt.tight_layout()

                    # Save chart
                    metric_name = (
                        metric.replace(" ", "_").replace("(", "").replace(")", "")
                    )
                    plt.savefig(
                        os.path.join(
                            output_dir,
                            f"heatmap_{metric_name}_duration_{duration}h.png",
                        )
                    )
                    plt.close()

    # Create best size combination analysis
    try:
        # Sort by Total Profit to find the optimal combinations
        top_combinations = results_df.sort_values(
            by="Total Profit", ascending=False
        ).head(10)

        # Plot bar chart comparing top combinations
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_combinations)), top_combinations["Total Profit"])
        plt.xticks(
            range(len(top_combinations)),
            [
                f"E{row['Electrolyzer Size (MW)']}|B{row['Battery Power (MW)']}|{row['Battery Duration (h)']}h"
                for _, row in top_combinations.iterrows()
            ],
            rotation=45,
        )
        plt.ylabel("Total Profit ($)")
        plt.title("Top 10 Size Combinations by Profit")

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_combinations_by_profit.png"))
        plt.close()

        # Save top combinations
        top_combinations.to_csv(
            os.path.join(output_dir, "top_size_combinations.csv"), index=False
        )

    except Exception as e:
        logging.error(f"Error creating top combinations analysis: {e}")

    # Clean up temporary files
    if os.path.exists(TEMP_RUN_DIR_BASE):
        shutil.rmtree(TEMP_RUN_DIR_BASE)

    return results_df


def sensitivity_analysis():
    """
    Perform comprehensive sensitivity analysis including parameter variations,
    technology configurations, market scenarios, and H2 value scenarios.
    """
    # Setup logging and output directories
    log_file = setup_logging()
    output_dirs = prepare_output_dirs()
    config_data = generate_run_config()

    os.makedirs(TEMP_RUN_DIR_BASE, exist_ok=True)

    # Calculate total number of runs
    total_runs = 1  # Base case
    for _, (_, _, steps) in PARAMETERS_TO_VARY.items():
        total_runs += steps

    # Add scenario analyses
    total_runs += len(SCENARIO_ANALYSES["technology_combinations"])
    total_runs += len(SCENARIO_ANALYSES["market_scenarios"])
    total_runs += len(SCENARIO_ANALYSES["h2_value_scenarios"])
    total_runs += len(SCENARIO_ANALYSES["fixed_size_scenarios"])

    current_run = 0
    all_results = []

    logging.info(f"Starting sensitivity analysis with {total_runs} total runs...")
    print(f"Starting sensitivity analysis with {total_runs} total runs...")

    # Get base case result
    base_temp_dir = os.path.join(TEMP_RUN_DIR_BASE, "base_case")
    if os.path.exists(base_temp_dir):
        shutil.rmtree(base_temp_dir)
    shutil.copytree(BASE_INPUT_DIR, base_temp_dir)

    print(f"Running base case ({current_run+1}/{total_runs})...")
    base_results = run_single_optimization(base_temp_dir, iso=TARGET_ISO)
    current_run += 1

    base_result_dict = {
        "Parameter": "Base Case",
        "Variation (%)": 0,
        "Parameter Value": "Base",
    }
    base_result_dict.update(base_results)
    all_results.append(base_result_dict)

    # =========================================================================
    # Part 1: Parameter Sensitivity Analysis
    # =========================================================================
    logging.info("Starting parameter sensitivity analysis...")

    # Sensitivity for each parameter
    for param_name, (min_pct, max_pct, steps) in PARAMETERS_TO_VARY.items():
        logging.info(
            f"Running sensitivity for {param_name} from {min_pct}% to {max_pct}%..."
        )
        print(f"Running sensitivity for {param_name} from {min_pct}% to {max_pct}%...")

        # Read base value
        base_df = pd.read_csv(BASE_SYS_DATA_FILE, index_col=0)
        if param_name not in base_df.index:
            logging.warning(
                f"Parameter '{param_name}' not found in sys_data_advanced.csv, skipping."
            )
            continue

        # --- Check parameter value type ---
        base_value_raw = base_df.loc[param_name, "Value"]

        try:
            base_value = float(base_value_raw)
            # If successful, proceed with standard percentage or absolute variation

            # Check if this parameter uses absolute values instead of percentages
            use_absolute = False
            if (
                min_pct * max_pct >= 0 and min_pct >= 0
            ):  # If both min & max are >= 0, likely absolute
                if param_name in [
                    "hydrogen_subsidy_value_usd_per_kg",
                    "hydrogen_subsidy_duration_years",
                ]:
                    use_absolute = True
                    pct_range = np.linspace(
                        min_pct, max_pct, steps
                    )  # Here pct_range is actually absolute values
                    logging.info(
                        f"Using absolute values for {param_name} from {min_pct} to {max_pct}"
                    )
                else:
                    pct_range = np.linspace(min_pct, max_pct, steps)
            else:
                pct_range = np.linspace(min_pct, max_pct, steps)

            for pct in pct_range:
                # Calculate new value
                if use_absolute:
                    current_value = pct  # Use absolute value
                    # Calculate display percentage relative to base for reporting
                    display_pct = f"{pct}"
                    if base_value != 0:
                        display_pct += (
                            f" ({(pct-base_value)/abs(base_value)*100:.1f}%) "
                        )
                    else:
                        display_pct += " (N/A %)"
                else:
                    current_value = base_value * (1 + pct / 100)
                    display_pct = f"{pct:.1f}%"

                run_dir = os.path.join(
                    TEMP_RUN_DIR_BASE,
                    f"{param_name.replace(' ', '_')}_{pct:.2f}",
                )
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir)
                shutil.copytree(BASE_INPUT_DIR, run_dir)

                # Modify sys_data_advanced.csv
                sys_data_file = os.path.join(run_dir, "sys_data_advanced.csv")
                modify_sys_data(
                    BASE_SYS_DATA_FILE,
                    sys_data_file,
                    param_name,
                    current_value,
                )

                # Run optimization
                print(
                    f"Running {param_name} at {display_pct} variation ({current_run+1}/{total_runs})..."
                )
                results = run_single_optimization(run_dir, iso=TARGET_ISO)
                current_run += 1

                # Calculate changes from base case
                result_dict = {
                    "Parameter": param_name,
                    "Variation (%)": pct,  # Store the original 'pct' value from the range
                    "Parameter Value": current_value,
                }

                # Add results and calculate percentage changes
                for metric in RESULT_METRICS:
                    result_dict[metric] = results[metric]
                    if (
                        not np.isnan(results[metric])
                        and not np.isnan(base_results[metric])
                        and base_results[metric] != 0
                    ):
                        result_dict[f"{metric}_Change_Pct"] = (
                            (results[metric] - base_results[metric])
                            / abs(base_results[metric])
                            * 100
                        )
                    else:
                        result_dict[f"{metric}_Change_Pct"] = np.nan

                all_results.append(result_dict)

        except ValueError:
            # If conversion to float fails, it's likely a string representing a list
            logging.warning(
                f"Parameter '{param_name}' has non-numeric value '{base_value_raw}'. Skipping standard parameter sensitivity for this item."
            )
            # Optionally, add a placeholder or log that this parameter is skipped
            # You might want to handle these specifically in scenario analysis if needed.
            continue  # Skip to the next parameter in the loop

    # =========================================================================
    # Part 2: Technology Configuration Analysis
    # =========================================================================
    logging.info("Starting technology configuration analysis...")
    tech_scenario_results = []

    for scenario in SCENARIO_ANALYSES["technology_combinations"]:
        scenario_name = scenario["name"]
        config = scenario["config"]

        logging.info(f"Running technology scenario: {scenario_name}")
        print(
            f"Running technology scenario: {scenario_name} ({current_run+1}/{total_runs})..."
        )

        run_dir = os.path.join(
            TEMP_RUN_DIR_BASE, f"tech_{scenario_name.replace(' ', '_')}"
        )
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(BASE_INPUT_DIR, run_dir)

        # Run optimization with config override
        results = run_single_optimization(
            run_dir, iso=TARGET_ISO, config_override=config
        )
        current_run += 1

        # Record results
        result_dict = {"Scenario": scenario_name, "Config": str(config)}
        result_dict.update(results)
        tech_scenario_results.append(result_dict)

    # =========================================================================
    # Part 3: Market Scenario Analysis
    # =========================================================================
    logging.info("Starting market scenario analysis...")
    market_scenario_results = []

    for scenario in SCENARIO_ANALYSES["market_scenarios"]:
        scenario_name = scenario["name"]
        energy_multiplier = scenario["energy_price_multiplier"]
        as_multiplier = scenario["ancillary_service_price_multiplier"]

        logging.info(f"Running market scenario: {scenario_name}")
        print(
            f"Running market scenario: {scenario_name} ({current_run+1}/{total_runs})..."
        )

        run_dir = os.path.join(
            TEMP_RUN_DIR_BASE, f"market_{scenario_name.replace(' ', '_')}"
        )
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(BASE_INPUT_DIR, run_dir)

        # Modify hourly price data
        modify_hourly_data(run_dir, energy_multiplier, as_multiplier)

        # Run optimization
        results = run_single_optimization(run_dir, iso=TARGET_ISO)
        current_run += 1

        # Record results
        result_dict = {
            "Scenario": scenario_name,
            "Energy Price Multiplier": energy_multiplier,
            "AS Price Multiplier": as_multiplier,
        }
        result_dict.update(results)
        market_scenario_results.append(result_dict)

    # =========================================================================
    # Part 4: H2 Value Scenario Analysis
    # =========================================================================
    logging.info("Starting H2 value scenario analysis...")
    h2_scenario_results = []

    for scenario in SCENARIO_ANALYSES["h2_value_scenarios"]:
        scenario_name = scenario["name"]
        h2_value = scenario["H2_value_USD_per_kg"]
        h2_subsidy = scenario["hydrogen_subsidy_value_usd_per_kg"]
        h2_duration = scenario["hydrogen_subsidy_duration_years"]

        logging.info(f"Running H2 value scenario: {scenario_name}")
        print(
            f"Running H2 value scenario: {scenario_name} ({current_run+1}/{total_runs})..."
        )

        run_dir = os.path.join(
            TEMP_RUN_DIR_BASE, f"h2value_{scenario_name.replace(' ', '_')}"
        )
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(BASE_INPUT_DIR, run_dir)

        # Modify sys_data_advanced.csv for H2 value and subsidy
        sys_data_file = os.path.join(run_dir, "sys_data_advanced.csv")
        temp_sys_data_1 = os.path.join(run_dir, "temp_sys_data_1.csv")
        temp_sys_data_2 = os.path.join(run_dir, "temp_sys_data_2.csv")

        # Apply changes in sequence
        modify_sys_data(
            BASE_SYS_DATA_FILE,
            temp_sys_data_1,
            "H2_value_USD_per_kg",
            h2_value,
        )
        modify_sys_data(
            temp_sys_data_1,
            temp_sys_data_2,
            "hydrogen_subsidy_value_usd_per_kg",
            h2_subsidy,
        )
        modify_sys_data(
            temp_sys_data_2,
            sys_data_file,
            "hydrogen_subsidy_duration_years",
            h2_duration,
        )

        # Run optimization
        results = run_single_optimization(run_dir, iso=TARGET_ISO)
        current_run += 1

        # Record results
        result_dict = {
            "Scenario": scenario_name,
            "H2 Value ($/kg)": h2_value,
            "H2 Subsidy ($/kg)": h2_subsidy,
            "H2 Subsidy Duration (years)": h2_duration,
        }
        result_dict.update(results)
        h2_scenario_results.append(result_dict)

    # =========================================================================
    # Part 5: Fixed Size Analysis - Equipment Size Sensitivity
    # =========================================================================
    logging.info("Starting fixed size scenario analysis...")
    fixed_size_results = []

    for scenario in SCENARIO_ANALYSES["fixed_size_scenarios"]:
        scenario_name = scenario["name"]

        logging.info(f"Running fixed size scenario: {scenario_name}")
        print(
            f"Running fixed size scenario: {scenario_name} ({current_run+1}/{total_runs})..."
        )

        run_dir = os.path.join(
            TEMP_RUN_DIR_BASE, f"fixed_size_{scenario_name.replace(' ', '_')}"
        )
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        shutil.copytree(BASE_INPUT_DIR, run_dir)

        # Modify sys_data_advanced.csv with the specified equipment sizes
        sys_data_file = os.path.join(run_dir, "sys_data_advanced.csv")
        temp_file = os.path.join(run_dir, "temp_sys_data.csv")

        # Copy base file to temp file first
        shutil.copyfile(BASE_SYS_DATA_FILE, temp_file)
        current_file = temp_file

        # Apply each specification in sequence
        for param, value in scenario.items():
            if param != "name" and value is not None:  # Skip 'name' and None values
                new_temp = os.path.join(run_dir, f"temp_sys_data_{param}.csv")
                modify_sys_data(current_file, new_temp, param, value)
                current_file = new_temp

        # Final move to sys_data_file
        shutil.move(current_file, sys_data_file)

        # Run optimization
        results = run_single_optimization(run_dir, iso=TARGET_ISO)
        current_run += 1

        # Record results
        result_dict = {"Scenario": scenario_name}

        # Add each parameter as a column
        for param, value in scenario.items():
            if param != "name":
                param_display = (
                    param.replace("user_specified_", "").replace("_", " ").title()
                )
                result_dict[param_display] = value

        result_dict.update(results)
        fixed_size_results.append(result_dict)

    # =========================================================================
    # Process and save all results
    # =========================================================================

    # Save parameter sensitivity results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SENSITIVITY_OUTPUT_CSV, index=False)

    # Save scenario analysis results
    pd.DataFrame(tech_scenario_results).to_csv(
        os.path.join(SENSITIVITY_OUTPUT_DIR, "technology_scenario_results.csv"),
        index=False,
    )
    pd.DataFrame(market_scenario_results).to_csv(
        os.path.join(SENSITIVITY_OUTPUT_DIR, "market_scenario_results.csv"),
        index=False,
    )
    pd.DataFrame(h2_scenario_results).to_csv(
        os.path.join(SENSITIVITY_OUTPUT_DIR, "h2_value_scenario_results.csv"),
        index=False,
    )
    pd.DataFrame(fixed_size_results).to_csv(
        os.path.join(SENSITIVITY_OUTPUT_DIR, "fixed_size_scenario_results.csv"),
        index=False,
    )

    # Generate plots
    logging.info("Generating parameter sensitivity plots...")
    plot_sensitivity_results(
        results_df, os.path.join(SENSITIVITY_OUTPUT_DIR, "parameter_analysis")
    )

    logging.info("Generating tornado plots...")
    create_tornado_plots(
        results_df, os.path.join(SENSITIVITY_OUTPUT_DIR, "parameter_analysis")
    )

    logging.info("Analyzing technology scenarios...")
    analyze_scenario_results(
        tech_scenario_results,
        "technology",
        os.path.join(SENSITIVITY_OUTPUT_DIR, "technology_comparison"),
    )

    logging.info("Analyzing market scenarios...")
    analyze_scenario_results(
        market_scenario_results,
        "market",
        os.path.join(SENSITIVITY_OUTPUT_DIR, "market_scenarios"),
    )

    logging.info("Analyzing H2 value scenarios...")
    analyze_scenario_results(
        h2_scenario_results,
        "h2_value",
        os.path.join(SENSITIVITY_OUTPUT_DIR, "h2_value_scenarios"),
    )

    logging.info("Analyzing fixed size scenarios...")
    analyze_scenario_results(
        fixed_size_results,
        "fixed_size",
        os.path.join(SENSITIVITY_OUTPUT_DIR, "fixed_size_scenarios"),
    )

    # Generate correlation analysis across all numeric results
    logging.info("Generating correlation analysis...")
    try:
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = results_df[numeric_cols].corr()

        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
        )
        plt.title("Correlation Matrix of Parameters and Results")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                SENSITIVITY_OUTPUT_DIR,
                "correlation_analysis",
                "full_correlation_matrix.png",
            )
        )
        plt.close()

        # Save correlation matrix to CSV
        correlation_matrix.to_csv(
            os.path.join(
                SENSITIVITY_OUTPUT_DIR,
                "correlation_analysis",
                "correlation_matrix.csv",
            )
        )
    except Exception as e:
        logging.error(f"Error generating correlation analysis: {e}", exc_info=True)

    # Clean up temporary files
    if os.path.exists(TEMP_RUN_DIR_BASE):
        shutil.rmtree(TEMP_RUN_DIR_BASE)

    logging.info("Sensitivity analysis completed successfully")
    print(
        f"\nSensitivity analysis completed. Results saved to {SENSITIVITY_OUTPUT_DIR}"
    )
    print(f"Log file: {log_file}")

    # Add size combination analysis after the original analysis (best to run optionally to avoid too long)
    run_size_combination = False  # Set to True to enable combination analysis

    if run_size_combination:
        logging.info("Starting electrolyzer & battery size combination analysis...")
        print("\nRunning electrolyzer & battery size combination analysis...")
        analyze_size_combinations()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for nuclear-hydrogen optimization model"
    )

    # Add command line arguments
    parser.add_argument(
        "--all", action="store_true", help="Run all sensitivity analyses"
    )
    parser.add_argument(
        "--params",
        action="store_true",
        help="Run only parameter sensitivity analysis",
    )
    parser.add_argument(
        "--tech",
        action="store_true",
        help="Run only technology configuration analysis",
    )
    parser.add_argument(
        "--market",
        action="store_true",
        help="Run only market scenario analysis",
    )
    parser.add_argument(
        "--h2value",
        action="store_true",
        help="Run only H2 value scenario analysis",
    )
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="Run only fixed size scenario analysis",
    )
    parser.add_argument(
        "--combinations",
        action="store_true",
        help="Run only size combination analysis (time-consuming)",
    )
    parser.add_argument(
        "--iso",
        type=str,
        default=TARGET_ISO,
        help=f"Specify ISO region (default: {TARGET_ISO})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=SENSITIVITY_OUTPUT_DIR,
        help=f"Specify output directory (default: {SENSITIVITY_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Set ISO and output directory
    target_iso = args.iso  # Use local variable instead of modifying global variable
    output_dir = args.output  # Use local variable instead of modifying global variable

    # Output parameter settings information
    if args.iso != TARGET_ISO:
        print(f"Using ISO region: {target_iso}")

    if args.output != SENSITIVITY_OUTPUT_DIR:
        print(f"Output directory set to: {output_dir}")

    # If no options are specified, run complete sensitivity analysis by default
    if not any(
        [
            args.all,
            args.params,
            args.tech,
            args.market,
            args.h2value,
            args.sizes,
            args.combinations,
        ]
    ):
        print(
            "No analysis type specified, running complete sensitivity analysis. Use --help to view options."
        )
        sensitivity_analysis()  # Use default global variables
    else:
        # Run all analyses
        if args.all:
            # Create a temporary environment, modify configuration variables, run analysis, and restore
            temp_target_iso = TARGET_ISO
            temp_output_dir = SENSITIVITY_OUTPUT_DIR

            try:
                # Temporarily redirect output directory and ISO settings, rather than directly modifying global variables
                BASE_INPUT_DIR_local = os.path.join(
                    output_dir, os.path.basename(BASE_INPUT_DIR)
                )
                SENSITIVITY_OUTPUT_CSV_local = os.path.join(
                    output_dir, os.path.basename(SENSITIVITY_OUTPUT_CSV)
                )
                TEMP_RUN_DIR_BASE_local = os.path.join(
                    output_dir, os.path.basename(TEMP_RUN_DIR_BASE)
                )

                # Run specified analysis
                sensitivity_analysis()
                if args.combinations:
                    analyze_size_combinations(
                        output_dir=os.path.join(output_dir, "size_combination_analysis")
                    )
            finally:
                # Restore global settings (actually not modified, but good practice)
                pass
        else:
            # If specific analysis type is specified, run only the specified analysis
            if args.params:
                print("Running parameter sensitivity analysis...")
                # Use parameters instead of global variables

            if args.tech:
                print("Running technology configuration analysis...")
                # Use parameters instead of global variables

            if args.market:
                print("Running market scenario analysis...")
                # Use parameters instead of global variables

            if args.h2value:
                print("Running H2 value scenario analysis...")
                # Use parameters instead of global variables

            if args.sizes:
                print("Running fixed size scenario analysis...")
                # Use parameters instead of global variables

            if args.combinations:
                print("Running equipment size combination analysis...")
                analyze_size_combinations(
                    output_dir=os.path.join(output_dir, "size_combination_analysis")
                )
