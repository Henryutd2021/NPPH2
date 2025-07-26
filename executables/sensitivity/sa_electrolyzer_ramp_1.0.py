#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Electrolyzer Ramp Rate = 1.0
This script tests the optimization framework with Electrolyzer HTE ramp rate set to 1.0 (both up and down)
"""


import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import os
import sys
import shutil
import pandas as pd
import logging
import timeit
from pathlib import Path
from datetime import datetime

# Setup Python paths for importing src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from path_setup import setup_src_paths
setup_src_paths()
from src.opt.result_processing import extract_results
from src.opt.model import create_model
from src.opt.data_io import load_hourly_data
from src.opt.config import TARGET_ISO, SIMULATE_AS_DISPATCH_EXECUTION


# Configuration
BASE_INPUT_DIR = "../input/hourly_data"
BASE_SYS_DATA_FILE = os.path.join(BASE_INPUT_DIR, "sys_data_advanced.csv")
OUTPUT_DIR = "../../output/sensitivity_analysis/electrolyzer_ramp_1.0"
TEMP_DIR = "../../temp_sensitivity_runs/electrolyzer_ramp_1.0"

# Parameters to modify
ELECTROLYZER_RAMP_UP_RATE = 1.0
ELECTROLYZER_RAMP_DOWN_RATE = 1.0


def setup_logging():
    """Setup logging configuration"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configure logging
    log_filename = os.path.join(
        OUTPUT_DIR, f"electrolyzer_ramp_1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(
        "=== Electrolyzer Ramp Rate 1.0 Sensitivity Analysis Started ===")


def prepare_directories():
    """Create necessary directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Copy input data to temp directory
    if os.path.exists(BASE_INPUT_DIR):
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        shutil.copytree(BASE_INPUT_DIR, TEMP_DIR)
        logging.info(f"Copied input data from {BASE_INPUT_DIR} to {TEMP_DIR}")
    else:
        logging.error(f"Base input directory not found: {BASE_INPUT_DIR}")
        raise FileNotFoundError(
            f"Base input directory not found: {BASE_INPUT_DIR}")


def modify_sys_data(original_file, temp_file, param_name, new_value):
    """Modify system data parameters"""
    try:
        df = pd.read_csv(original_file, index_col=0)
        if param_name not in df.index:
            logging.warning(
                f"Parameter '{param_name}' not found in {original_file}")
            return False

        old_value = df.loc[param_name, 'Value']
        logging.info(
            f"Modifying '{param_name}' from {old_value} to {new_value}")
        df.loc[param_name, 'Value'] = new_value
        df.to_csv(temp_file)
        return True
    except Exception as e:
        logging.error(f"Error modifying parameter '{param_name}': {e}")
        return False


def run_optimization():
    """Run the optimization with modified parameters"""
    try:
        # Modify system data file
        temp_sys_file = os.path.join(TEMP_DIR, "sys_data_advanced.csv")

        # Modify electrolyzer ramp rates (HTE)
        success1 = modify_sys_data(BASE_SYS_DATA_FILE, temp_sys_file,
                                   "Electrolyzer_RampUp_Rate_Percent_per_Min_HTE", ELECTROLYZER_RAMP_UP_RATE)
        success2 = modify_sys_data(temp_sys_file, temp_sys_file,
                                   "Electrolyzer_RampDown_Rate_Percent_per_Min_HTE", ELECTROLYZER_RAMP_DOWN_RATE)

        if not (success1 and success2):
            logging.error("Failed to modify system parameters")
            return None

        # Load data
        logging.info("Loading hourly data...")
        data = load_hourly_data(TARGET_ISO, base_dir=TEMP_DIR)
        if data is None:
            logging.error("Failed to load hourly data")
            return None

        # Create model
        logging.info("Creating optimization model...")
        model = create_model(
            data, TARGET_ISO, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION)
        if model is None:
            logging.error("Failed to create optimization model")
            return None

        # Solve model
        logging.info("Solving optimization model...")
        solver = SolverFactory("gurobi")

        # Check solver availability
        if not solver.available():
            logging.warning(
                "Gurobi not available, trying alternative solvers...")
            for alt_solver in ["cbc", "glpk"]:
                solver = SolverFactory(alt_solver)
                if solver.available():
                    logging.info(f"Using alternative solver: {alt_solver}")
                    break
            else:
                logging.error("No available solvers found")
                return None

        # Configure solver options
        if solver.name == "gurobi":
            solver.options["TimeLimit"] = 3600  # 1 hour time limit
            solver.options["MIPGap"] = 0.0005  # 1% MIP gap
        elif solver.name == "cbc":
            solver.options["sec"] = 3600  # 1 hour time limit
            solver.options["ratio"] = 0.01  # 1% MIP gap

        # Solve
        results = solver.solve(model, tee=True)

        # Check solve status
        if results.solver.status != SolverStatus.ok:
            logging.error(f"Solver failed: {results.solver.status}")
            return None

        if results.solver.termination_condition not in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            logging.warning(
                f"Solver termination: {results.solver.termination_condition}")

        # Extract results
        logging.info("Extracting results...")
        results_df, summary_results = extract_results(
            model, TARGET_ISO, output_dir=OUTPUT_DIR)

        # Save results
        results_filename = os.path.join(
            OUTPUT_DIR, "electrolyzer_ramp_1.0_results.csv")
        summary_filename = os.path.join(
            OUTPUT_DIR, "electrolyzer_ramp_1.0_summary.json")

        if results_df is not None:
            results_df.to_csv(results_filename, index=False)
            logging.info(f"Results saved to {results_filename}")

        if summary_results is not None:
            import json
            with open(summary_filename, 'w') as f:
                json.dump(summary_results, f, indent=2)
            logging.info(f"Summary results saved to {summary_filename}")

        return {
            'results_df': results_df,
            'summary_results': summary_results,
            'model': model,
            'solver_results': results
        }

    except Exception as e:
        logging.error(f"Error during optimization: {e}", exc_info=True)
        return None


def cleanup():
    """Clean up temporary files"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logging.info(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logging.warning(f"Error cleaning up temporary directory: {e}")


def main():
    """Main execution function"""
    start_time = timeit.default_timer()

    try:
        # Setup
        setup_logging()
        prepare_directories()

        # Run optimization
        logging.info(
            f"Starting optimization with Electrolyzer RampUp Rate = {ELECTROLYZER_RAMP_UP_RATE}, RampDown Rate = {ELECTROLYZER_RAMP_DOWN_RATE}")
        results = run_optimization()

        if results is None:
            logging.error("Optimization failed")
            return 1

        # Log summary
        elapsed_time = timeit.default_timer() - start_time
        logging.info(
            f"Optimization completed successfully in {elapsed_time:.2f} seconds")

        if results['summary_results']:
            logging.info("Key Results:")
            for key, value in results['summary_results'].items():
                if isinstance(value, (int, float)):
                    logging.info(f"  {key}: {value:.2f}")
                else:
                    logging.info(f"  {key}: {value}")

        logging.info(
            "=== Electrolyzer Ramp Rate 1.0 Sensitivity Analysis Completed ===")
        return 0

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
