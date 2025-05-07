"""
Sensitivity analysis script for the nuclear-hydrogen optimization framework.
This script performs comprehensive sensitivity analysis on various parameters
and generates detailed reports and visualizations.
"""

import os
import shutil
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import copy
from datetime import datetime

# Import optimization framework from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from config import (
    TARGET_ISO,
    ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_ELECTROLYZER,
    ENABLE_BATTERY,
    ENABLE_H2_STORAGE,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_STARTUP_SHUTDOWN,
    SIMULATE_AS_DISPATCH_EXECUTION
)

# --- Configuration ---
BASE_INPUT_DIR = "../input/hourly_data"
BASE_SYS_DATA_FILE = os.path.join(BASE_INPUT_DIR, "sys_data_advanced.csv")
SENSITIVITY_OUTPUT_DIR = "../sensitivity_analysis_results"
SENSITIVITY_OUTPUT_CSV = os.path.join(SENSITIVITY_OUTPUT_DIR, "sensitivity_analysis_results.csv")
TEMP_RUN_DIR_BASE = "../temp_sensitivity_runs"

# Parameters to vary: {parameter_name_in_csv: [min_change_%, max_change_%, num_steps]}
PARAMETERS_TO_VARY = {
    # Economic Parameters
    'H2_value_USD_per_kg': [-20, 20, 5],
    'vom_electrolyzer_USD_per_MWh': [-30, 30, 7],
    'vom_turbine_USD_per_MWh': [-30, 30, 5],
    'cost_startup_electrolyzer_USD_per_startup': [-50, 50, 5],
    'cost_water_USD_per_kg_h2': [-20, 20, 5],
    'cost_h2_storage_USD_per_kg': [-30, 30, 5],
    
    # Technical Parameters
    'pElectrolyzer_max_MW': [-20, 20, 5],
    'pTurbine_max_MW': [-15, 15, 5],
    'H2_storage_capacity_max_kg': [-30, 30, 5],
    'Turbine_RampUp_Rate_Percent_per_Min': [-50, 100, 5],
    'Electrolyzer_RampUp_Rate_Percent_per_Min': [-50, 100, 5],
    'Electrolyzer_Efficiency': [-10, 10, 5],
    'Turbine_Efficiency': [-10, 10, 5],
    
    # Market Parameters
    'energy_price_multiplier': [-20, 20, 5],
    'ancillary_service_price_multiplier': [-30, 30, 5],
}

# Result metrics to track
RESULT_METRICS = [
    'Total Profit',
    'Energy Revenue',
    'Hydrogen Revenue',
    'Ancillary Service Revenue',
    'Total Cost',
    'H2 Production (kg)',
    'Energy Generation (MWh)',
    'AS Capacity (MW)'
]

# --- Logging Setup ---
def setup_logging():
    """Setup logging configuration"""
    os.makedirs(SENSITIVITY_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sensitivity_log_file = os.path.join(SENSITIVITY_OUTPUT_DIR, f'sensitivity_analysis_{timestamp}.log')
    logging.basicConfig(
        filename=sensitivity_log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    return sensitivity_log_file

def modify_sys_data(original_file, temp_file, param_name, new_value):
    """Reads the sys_data CSV, modifies a parameter, saves to temp file."""
    try:
        df = pd.read_csv(original_file, index_col=0)
        if param_name not in df.index:
            logging.warning(f"Parameter '{param_name}' not found in {original_file}. Skipping modification.")
            shutil.copyfile(original_file, temp_file)
            return False
        logging.info(f"Modifying '{param_name}' from {df.loc[param_name, 'Value']} to {new_value}")
        df.loc[param_name, 'Value'] = new_value
        df.to_csv(temp_file)
        return True
    except Exception as e:
        logging.error(f"Error modifying parameter '{param_name}' in {original_file}: {e}", exc_info=True)
        if os.path.exists(original_file) and not os.path.exists(temp_file):
            shutil.copyfile(original_file, temp_file)
        return False

def run_single_optimization(input_dir, iso=None, solver_name="gurobi"):
    """
    Runs the optimization for a given input directory. Returns a dictionary of results.
    """
    try:
        # Load data
        data = load_hourly_data(iso, base_dir=input_dir)
        if data is None:
            logging.error(f"Data loading failed for {input_dir}")
            return {metric: np.nan for metric in RESULT_METRICS}

        # Create model
        model = create_model(data, iso, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION)
        
        # Solve
        solver = SolverFactory(solver_name)
        
        # Check solver availability
        if not solver.available():
            logging.warning(f"Solver {solver_name} not available, trying alternative solvers...")
            for alt_solver in ["cbc", "glpk"]:
                solver = SolverFactory(alt_solver)
                if solver.available():
                    logging.info(f"Using alternative solver: {alt_solver}")
                    break
            else:
                logging.error("No available solvers found.")
                return {metric: np.nan for metric in RESULT_METRICS}
        
        logging.info(f"Using solver: {solver.name}")
        results = solver.solve(model, tee=False)
        
        if results.solver.status != SolverStatus.ok:
            logging.error(f"Solver failed: {results.solver.status}")
            return {metric: np.nan for metric in RESULT_METRICS}
            
        if results.solver.termination_condition not in {TerminationCondition.optimal, TerminationCondition.feasible}:
            logging.warning(f"Solver termination: {results.solver.termination_condition}")
        
        # Extract results
        try:
            result_dict = {}
            for metric in RESULT_METRICS:
                if hasattr(model, metric.replace(' ', '_')):
                    result_dict[metric] = pyo.value(getattr(model, metric.replace(' ', '_')))
                else:
                    result_dict[metric] = np.nan
            
            logging.info(f"Run completed with results: {result_dict}")
            return result_dict
            
        except Exception as e:
            logging.error(f"Error extracting results: {e}")
            return {metric: np.nan for metric in RESULT_METRICS}
            
    except Exception as e:
        logging.error(f"Exception during optimization: {e}", exc_info=True)
        return {metric: np.nan for metric in RESULT_METRICS}

def plot_sensitivity_results(results_df, output_dir):
    """
    Generate plots for sensitivity analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot for each parameter and metric
    for param in results_df['Parameter'].unique():
        if param == 'Base Case':
            continue
            
        param_data = results_df[results_df['Parameter'] == param]
        
        # Create subplots for each metric
        n_metrics = len(RESULT_METRICS)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5*n_metrics))
        fig.suptitle(f'Sensitivity Analysis: {param}')
        
        for idx, metric in enumerate(RESULT_METRICS):
            ax = axes[idx] if n_metrics > 1 else axes
            
            # Plot absolute values
            ax.plot(param_data['Variation (%)'], param_data[metric], 'b-', label='Absolute Value')
            
            # Plot percentage change if applicable
            if f'{metric}_Change_Pct' in param_data.columns:
                ax2 = ax.twinx()
                ax2.plot(param_data['Variation (%)'], param_data[f'{metric}_Change_Pct'], 'r--', label='% Change')
                ax2.set_ylabel('Percentage Change (%)')
            
            ax.set_xlabel('Parameter Variation (%)')
            ax.set_ylabel(metric)
            ax.grid(True)
            
            if idx == 0:
                ax.legend(loc='upper left')
                if f'{metric}_Change_Pct' in param_data.columns:
                    ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sensitivity_{param.replace(" ", "_")}.png'))
        plt.close()

def sensitivity_analysis():
    """
    Perform comprehensive sensitivity analysis
    """
    # Setup logging
    log_file = setup_logging()
    os.makedirs(TEMP_RUN_DIR_BASE, exist_ok=True)
    
    # Calculate total number of runs
    total_runs = 1  # Base case
    for _, (_, _, steps) in PARAMETERS_TO_VARY.items():
        total_runs += steps
    
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
        'Parameter': 'Base Case',
        'Variation (%)': 0,
        'Parameter Value': 'Base'
    }
    base_result_dict.update(base_results)
    all_results.append(base_result_dict)

    # Sensitivity for each parameter
    for param_name, (min_pct, max_pct, steps) in PARAMETERS_TO_VARY.items():
        logging.info(f"Running sensitivity for {param_name} from {min_pct}% to {max_pct}%...")
        print(f"Running sensitivity for {param_name} from {min_pct}% to {max_pct}%...")
        
        # Read base value
        base_df = pd.read_csv(BASE_SYS_DATA_FILE, index_col=0)
        if param_name not in base_df.index:
            logging.warning(f"Parameter '{param_name}' not found in sys_data_advanced.csv, skipping.")
            continue
            
        base_value = float(base_df.loc[param_name, 'Value'])
        pct_range = np.linspace(min_pct, max_pct, steps)
        
        for pct in pct_range:
            current_value = base_value * (1 + pct / 100)
            run_dir = os.path.join(TEMP_RUN_DIR_BASE, f"{param_name.replace(' ', '_')}_{pct:.2f}")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            shutil.copytree(BASE_INPUT_DIR, run_dir)
            
            # Modify sys_data_advanced.csv
            sys_data_file = os.path.join(run_dir, "sys_data_advanced.csv")
            modify_sys_data(BASE_SYS_DATA_FILE, sys_data_file, param_name, current_value)
            
            # Run optimization
            print(f"Running {param_name} at {pct}% variation ({current_run+1}/{total_runs})...")
            results = run_single_optimization(run_dir, iso=TARGET_ISO)
            current_run += 1
            
            # Calculate changes from base case
            result_dict = {
                'Parameter': param_name,
                'Variation (%)': pct,
                'Parameter Value': current_value
            }
            
            # Add results and calculate percentage changes
            for metric in RESULT_METRICS:
                result_dict[metric] = results[metric]
                if not np.isnan(results[metric]) and not np.isnan(base_results[metric]) and base_results[metric] != 0:
                    result_dict[f'{metric}_Change_Pct'] = (results[metric] - base_results[metric]) / abs(base_results[metric]) * 100
                else:
                    result_dict[f'{metric}_Change_Pct'] = np.nan
            
            all_results.append(result_dict)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SENSITIVITY_OUTPUT_CSV, index=False)
    
    # Generate plots
    plot_sensitivity_results(results_df, SENSITIVITY_OUTPUT_DIR)
    
    # Clean up temporary files
    if os.path.exists(TEMP_RUN_DIR_BASE):
        shutil.rmtree(TEMP_RUN_DIR_BASE)
    
    logging.info("Sensitivity analysis completed successfully")
    print(f"\nSensitivity analysis completed. Results saved to {SENSITIVITY_OUTPUT_DIR}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    sensitivity_analysis()