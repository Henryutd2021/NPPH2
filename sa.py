import os
import shutil
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

# Import optimization framework from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# --- Configuration ---
BASE_INPUT_DIR = "./input/hourly_data"
BASE_SYS_DATA_FILE = os.path.join(BASE_INPUT_DIR, "sys_data_advanced.csv")
SENSITIVITY_OUTPUT_CSV = "sensitivity_analysis_results.csv"
TEMP_RUN_DIR_BASE = "./temp_sensitivity_runs"

# Parameters to vary: {parameter_name_in_csv: [min_change_%, max_change_%, num_steps]}
PARAMETERS_TO_VARY = {
    'H2_value_USD_per_kg': [-20, 20, 5],
    'vom_electrolyzer_USD_per_MWh': [-30, 30, 7],
    'cost_startup_electrolyzer_USD_per_startup': [-50, 50, 5],
    'pElectrolyzer_max_MW': [-20, 20, 5],
    'pTurbine_max_MW': [-15, 15, 5],
    'H2_storage_capacity_max_kg': [-30, 30, 5],
    'Turbine_RampUp_Rate_Percent_per_Min': [-50, 100, 5],
    'vom_turbine_USD_per_MWh': [-30, 30, 5],
    'cost_water_USD_per_kg_h2': [-20, 20, 5],
    # Add more parameters as needed, names must match the first column of sys_data_advanced.csv
}

RESULT_METRIC_TARGET = 'Total Profit'

# --- Logging Setup ---
sensitivity_log_file = 'sensitivity_analysis.log'
logging.basicConfig(filename=sensitivity_log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')


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
    Runs the optimization for a given input directory. Returns total profit (objective value).
    """
    try:
        # Load data
        data = load_hourly_data(iso, base_dir=input_dir)
        if data is None:
            logging.error(f"Data loading failed for {input_dir}")
            return np.nan
        # Create model
        model = create_model(data, iso, use_nonlinear_turbine_eff_setting=True)
        # Solve
        solver = SolverFactory(solver_name)
        
        # Check if gurobi is available, otherwise try other solvers
        if solver_name == "gurobi" and not SolverFactory(solver_name).available():
            logging.warning(f"Solver {solver_name} not available, trying cbc...")
            solver = SolverFactory("cbc")
            if not solver.available():
                logging.warning("CBC not available, trying glpk...")
                solver = SolverFactory("glpk")
                if not solver.available():
                    logging.error("No available solvers found.")
                    return np.nan
        
        logging.info(f"Using solver: {solver.name}")
        results = solver.solve(model, tee=False)
        if results.solver.status != SolverStatus.ok:
            logging.error(f"Solver failed: {results.solver.status}")
            return np.nan
        if results.solver.termination_condition not in {TerminationCondition.optimal, TerminationCondition.feasible}:
            logging.warning(f"Solver termination: {results.solver.termination_condition}")
            
        # Extract results
        try:
            profit = pyo.value(model.TotalProfit_Objective)
            
            # Additional results
            energy_revenue = pyo.value(model.EnergyRevenue)
            hydrogen_revenue = pyo.value(model.HydrogenRevenue)
            
            logging.info(f"Run completed with profit: {profit}, Energy Revenue: {energy_revenue}, H2 Revenue: {hydrogen_revenue}")
            
            # Optional: Save detailed results for this run if needed
            # extract_results(model, iso, output_dir=f"{TEMP_RUN_DIR_BASE}/results_{os.path.basename(input_dir)}")
            
            return profit
        except Exception as e:
            logging.error(f"Error extracting results: {e}")
            return np.nan
            
    except Exception as e:
        logging.error(f"Exception during optimization: {e}", exc_info=True)
        return np.nan


def sensitivity_analysis():
    """
    Perform sensitivity analysis for each parameter in PARAMETERS_TO_VARY.
    For each parameter, vary it over the specified range, run optimization, and record the result.
    """
    os.makedirs(TEMP_RUN_DIR_BASE, exist_ok=True)
    all_results = []
    
    # Calculate total number of runs for progress tracking
    total_runs = 1  # Base case
    for _, (_, _, steps) in PARAMETERS_TO_VARY.items():
        total_runs += steps
    
    current_run = 0
    
    logging.info(f"Starting sensitivity analysis with {total_runs} total runs...")
    print(f"Starting sensitivity analysis with {total_runs} total runs...")

    # Get base case result
    base_temp_dir = os.path.join(TEMP_RUN_DIR_BASE, "base_case")
    if os.path.exists(base_temp_dir):
        shutil.rmtree(base_temp_dir)
    shutil.copytree(BASE_INPUT_DIR, base_temp_dir)
    
    print(f"Running base case ({current_run+1}/{total_runs})...")
    base_profit = run_single_optimization(base_temp_dir, iso=None)
    current_run += 1
    
    all_results.append({
        'Parameter': 'Base Case',
        'Variation (%)': 0,
        'Parameter Value': 'Base',
        RESULT_METRIC_TARGET: base_profit
    })

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
            profit = run_single_optimization(run_dir, iso=None)
            current_run += 1
            
            # Calculate percent change from base case
            if base_profit and not np.isnan(base_profit) and base_profit != 0 and not np.isnan(profit):
                pct_change = (profit - base_profit) / abs(base_profit) * 100
            else:
                pct_change = np.nan
                
            all_results.append({
                'Parameter': param_name,
                'Variation (%)': pct,
                'Parameter Value': current_value,
                RESULT_METRIC_TARGET: profit,
                'Profit_Change_Pct': pct_change
            })
            
            # Save results after each run so we don't lose everything if there's an error
            temp_results_df = pd.DataFrame(all_results)
            temp_results_df.to_csv(SENSITIVITY_OUTPUT_CSV, index=False)

    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SENSITIVITY_OUTPUT_CSV, index=False)
    logging.info(f"Sensitivity analysis complete. Results saved to {SENSITIVITY_OUTPUT_CSV}")
    print(f"Sensitivity analysis complete. Results saved to {SENSITIVITY_OUTPUT_CSV}")

    # Create additional detailed results CSV with elasticity values
    try:
        detailed_results = []
        for param_name in results_df['Parameter'].unique():
            if param_name == 'Base Case':
                continue
                
            param_data = results_df[results_df['Parameter'] == param_name]
            base_param_value = base_df.loc[param_name, 'Value']
            
            for _, row in param_data.iterrows():
                param_value = row['Parameter Value']
                profit = row[RESULT_METRIC_TARGET]
                param_pct_change = (param_value - base_param_value) / base_param_value * 100
                
                if not np.isnan(profit) and not np.isnan(base_profit) and base_profit != 0 and param_pct_change != 0:
                    # Calculate elasticity: % change in profit / % change in parameter
                    profit_pct_change = (profit - base_profit) / base_profit * 100
                    elasticity = profit_pct_change / param_pct_change
                else:
                    elasticity = np.nan
                    
                detailed_results.append({
                    'Parameter': param_name,
                    'Base_Value': base_param_value,
                    'New_Value': param_value,
                    'Parameter_Pct_Change': param_pct_change,
                    'Base_Profit': base_profit,
                    'New_Profit': profit,
                    'Profit_Pct_Change': (profit - base_profit) / base_profit * 100 if not np.isnan(profit) and not np.isnan(base_profit) and base_profit != 0 else np.nan,
                    'Elasticity': elasticity
                })
                
        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            detailed_filename = "sensitivity_detailed_results.csv"
            detailed_df.to_csv(detailed_filename, index=False)
            print(f"Detailed elasticity results saved to {detailed_filename}")
    except Exception as e:
        logging.error(f"Error generating detailed results: {e}", exc_info=True)
    
    # Plotting (Tornado Plot)
    try:
        base_profit = results_df[results_df['Parameter'] == 'Base Case'][RESULT_METRIC_TARGET].iloc[0]
        sensitivity_data = []
        varied_params = results_df['Parameter'].unique()
        varied_params = [p for p in varied_params if p != 'Base Case']
        
        for param in varied_params:
            param_results = results_df[results_df['Parameter'] == param]
            if param_results[RESULT_METRIC_TARGET].isnull().all():
                logging.warning(f"Skipping {param} in plot due to all NaN results.")
                continue
                
            # Filter out NaN values before calculating min/max
            valid_results = param_results[~param_results[RESULT_METRIC_TARGET].isnull()]
            if len(valid_results) == 0:
                continue
                
            min_profit = valid_results[RESULT_METRIC_TARGET].min()
            max_profit = valid_results[RESULT_METRIC_TARGET].max()
            
            # Only include if there's a significant difference
            if abs(max_profit - min_profit) > 1e-6:
                # Get param values that produced min/max
                min_param_value = valid_results.loc[valid_results[RESULT_METRIC_TARGET].idxmin(), 'Parameter Value']
                max_param_value = valid_results.loc[valid_results[RESULT_METRIC_TARGET].idxmax(), 'Parameter Value']
                
                sensitivity_data.append({
                    'Parameter': param,
                    'Min Result': min_profit,
                    'Max Result': max_profit,
                    'Min Param Value': min_param_value,
                    'Max Param Value': max_param_value
                })
                
        if sensitivity_data:
            sens_df = pd.DataFrame(sensitivity_data)
            sens_df['Range'] = sens_df['Max Result'] - sens_df['Min Result']
            sens_df = sens_df.sort_values(by='Range', ascending=True)
            
            # Create tornado plot
            fig, ax = plt.subplots(figsize=(12, len(sens_df) * 0.7))
            y_pos = np.arange(len(sens_df))
            
            # Plot bars
            ax.barh(y_pos, sens_df['Max Result'] - base_profit, left=base_profit, color='green', alpha=0.7, 
                    label='Positive Impact')
            ax.barh(y_pos, sens_df['Min Result'] - base_profit, left=base_profit, color='red', alpha=0.7,
                    label='Negative Impact')
            
            # Add parameter values at min/max
            for i, (_, row) in enumerate(sens_df.iterrows()):
                # Format values for display
                min_val = f"{row['Min Param Value']:.2f}" if isinstance(row['Min Param Value'], (int, float)) else row['Min Param Value']
                max_val = f"{row['Max Param Value']:.2f}" if isinstance(row['Max Param Value'], (int, float)) else row['Max Param Value']
                
                # Add text for min value (left side)
                if row['Min Result'] < base_profit:
                    ax.text(row['Min Result'], i, f" {min_val}", va='center', ha='left', fontsize=8)
                
                # Add text for max value (right side)
                if row['Max Result'] > base_profit:
                    ax.text(row['Max Result'], i, f" {max_val}", va='center', ha='left', fontsize=8)
            
            # Set labels and title
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sens_df['Parameter'])
            ax.set_xlabel(f'{RESULT_METRIC_TARGET} ($)')
            ax.set_title('Sensitivity Analysis Tornado Plot')
            ax.axvline(base_profit, color='black', linestyle='--', label=f'Base Case: ${base_profit:,.2f}')
            ax.legend()
            
            plt.tight_layout()
            plot_filename = 'sensitivity_tornado_plot.png'
            plt.savefig(plot_filename)
            print(f"Saved tornado plot to {plot_filename}")
            
            # Additional plot: Parameter elasticity chart
            try:
                # Create elasticity dataframe
                elasticity_data = []
                for param in sens_df['Parameter']:
                    param_data = detailed_df[detailed_df['Parameter'] == param]
                    # Use average elasticity for each parameter
                    avg_elasticity = param_data['Elasticity'].mean()
                    if not np.isnan(avg_elasticity):
                        elasticity_data.append({
                            'Parameter': param,
                            'Elasticity': avg_elasticity
                        })
                
                if elasticity_data:
                    el_df = pd.DataFrame(elasticity_data)
                    el_df = el_df.sort_values(by='Elasticity', ascending=False)
                    
                    # Plot elasticity
                    fig, ax = plt.subplots(figsize=(10, len(el_df) * 0.5))
                    colors = ['green' if x > 0 else 'red' for x in el_df['Elasticity']]
                    ax.barh(el_df['Parameter'], el_df['Elasticity'], color=colors)
                    ax.set_xlabel('Elasticity (% change in profit / % change in parameter)')
                    ax.set_title('Parameter Elasticity Chart')
                    ax.axvline(0, color='black', linestyle='--')
                    
                    plt.tight_layout()
                    elasticity_plot = 'parameter_elasticity_chart.png'
                    plt.savefig(elasticity_plot)
                    print(f"Saved elasticity chart to {elasticity_plot}")
            except Exception as e:
                logging.error(f"Failed to generate elasticity chart: {e}", exc_info=True)
            
        else:
            print("No valid sensitivity data to plot.")
    except Exception as e:
        logging.error(f"Failed to generate tornado plot: {e}", exc_info=True)
        print("Failed to generate tornado plot.")
    
    print("--- Sensitivity Analysis Finished ---")


if __name__ == "__main__":
    sensitivity_analysis()