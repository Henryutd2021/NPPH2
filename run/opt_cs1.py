#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuclear Power Plant Optimization Controller

This script works as a controller that:
1. Reads nuclear power plant data from CSV files
2. Prepares input data for the optimization framework
3. Calls the framework's optimization functions
4. Saves hourly optimization results to CSV file
"""

import os
import sys
import pandas as pd
import warnings
import threading
import time
from pathlib import Path
warnings.filterwarnings("ignore")

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add src directory to Python path for importing optimization framework modules
# Get the absolute path to the workspace root (parent of opt directory)
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_opt_path = os.path.join(workspace_root, 'src', 'opt')
src_logger_utils_path = os.path.join(workspace_root, 'src', 'logger_utils')
src_root_path = os.path.join(workspace_root, 'src')

# Add all necessary paths to sys.path, including workspace root for absolute 'src' imports
for path in [workspace_root, src_opt_path, src_logger_utils_path, src_root_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Debug: Print the paths being added
print(f"Added paths to sys.path:")
print(f"  - {workspace_root} (for 'src' absolute imports)")
print(f"  - {src_opt_path}")
print(f"  - {src_logger_utils_path}")
print(f"  - {src_root_path}")

# Try to import the optimization framework modules
try:
    from config import HOURS_IN_YEAR, TARGET_ISO
    from data_io import load_hourly_data
    from model import create_model
    from result_processing import extract_results
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
    import pyomo.environ as pyo
    # Import the framework's optimization module
    from optimization_utils import run_plant_optimization
    # Import unified progress indicator
    from progress_indicators import SolverProgressIndicator
    optimization_framework_available = True
    print("Successfully imported optimization framework modules")
except ImportError as e:
    print(f"Error: Optimization framework modules not available: {e}")
    optimization_framework_available = False
    sys.exit(1)  # Exit if framework is not available


def calculate_crf(discount_rate, lifetime_years):
    """Calculates the Capital Recovery Factor."""
    if lifetime_years <= 0:
        return 0
    if discount_rate == 0:
        return 1.0 / lifetime_years if lifetime_years > 0 else 0
    try:
        factor = (1 + discount_rate)**lifetime_years
        if abs(factor - 1.0) < 1e-9:  # Avoid division by zero if factor is very close to 1
            return 0 if lifetime_years <= 0 else (1.0 / lifetime_years if discount_rate == 0 else 0)
        return (discount_rate * factor) / (factor - 1)
    except (OverflowError, ValueError):
        print(
            f"Warning: CRF calculation failed for rate={discount_rate}, life={lifetime_years}. Returning rate.")
        return discount_rate


def adjust_system_params_for_remaining_life(system_params_df, remaining_years, thermal_efficiency=None):
    """
    Adjusts the system parameters based on the remaining life of the nuclear power plant.

    Args:
        system_params_df: DataFrame containing the system parameters
        remaining_years: Remaining operational years of the nuclear power plant
        thermal_efficiency: (Optional) Actual thermal efficiency to use for breakpoints

    Returns:
        DataFrame: Modified system parameters with updated CRF-based values
    """
    # Create a copy of the system parameters dataframe
    df = system_params_df.copy()

    # Get discount rate from the dataframe
    discount_rate = float(df.loc['discount_rate', 'Value'])

    # Update plant lifetime years to the remaining years
    df.loc['plant_lifetime_years', 'Value'] = remaining_years

    # Define the equipment lifetimes and cost parameters
    equipment_params = {
        'LTE': {  # Low Temperature Electrolyzer (PEM)
            'default_lifetime': 20,
            'capex_usd_per_kw': 2000.0,
            'params_to_update': {
                'cost_electrolyzer_capacity_USD_per_MW_year_LTE': lambda crf: 2000.0 * 1000.0 * crf
            }
        },
        'HTE': {  # High Temperature Electrolyzer (SOEC)
            'default_lifetime': 20,
            'capex_usd_per_kw': 2500.0,
            'params_to_update': {
                'cost_electrolyzer_capacity_USD_per_MW_year_HTE': lambda crf: 2500.0 * 1000.0 * crf
            }
        },
        'Battery': {
            'default_lifetime': 15,
            'capex_mwh_total_usd_per_kwh': 236.0,
            'duration_hours': 4.0,
            'params_to_update': {
                'BatteryCapex_USD_per_MWh_year': lambda crf: 236.0 * 1000.0 * crf,
                'BatteryCapex_USD_per_MW_year': lambda crf: 236.0 * 1000.0 * 4.0 * crf,
                # 1% of capex
                'BatteryFixedOM_USD_per_MWh_year': lambda crf: (236.0 * 0.01) * 1000.0
            }
        }
    }

    # Calculate effective lifetimes - min of equipment life and plant remaining life
    for equip_type, equip_data in equipment_params.items():
        effective_lifetime = min(
            equip_data['default_lifetime'], remaining_years)
        equip_crf = calculate_crf(discount_rate, effective_lifetime)

        # Update the parameters in the dataframe
        for param_name, calc_func in equip_data['params_to_update'].items():
            new_value = calc_func(equip_crf)
            if param_name in df.index:
                df.loc[param_name, 'Value'] = new_value

    # Update hydrogen subsidy duration to min of 10 years or remaining life
    hydrogen_subsidy_duration = min(10, remaining_years)
    df.loc['hydrogen_subsidy_duration_years',
           'Value'] = hydrogen_subsidy_duration

    if 'qSteam_Turbine_Breakpoints_MWth' in df.index:
        q_breakpoints_str = str(
            df.loc['qSteam_Turbine_Breakpoints_MWth', 'Value'])
        try:
            q_breakpoints = [float(x.strip())
                             for x in q_breakpoints_str.split(',') if x.strip()]
            if thermal_efficiency is not None:
                eff = float(thermal_efficiency)
            elif 'Thermal_Efficiency' in df.index:
                eff = float(df.loc['Thermal_Efficiency', 'Value'])
            elif 'Turbine_Thermal_Elec_Efficiency_Const' in df.index:
                eff = float(
                    df.loc['Turbine_Thermal_Elec_Efficiency_Const', 'Value'])
            else:
                raise ValueError('No available efficiency for breakpoints')
            p_outputs = [q * eff for q in q_breakpoints]
            p_outputs_str = ', '.join(f'{p:.4f}' for p in p_outputs)
            df.loc['pTurbine_Outputs_at_Breakpoints_MW', 'Value'] = p_outputs_str
        except Exception as e:
            print(
                f"Warning: Failed to update pTurbine_Outputs_at_Breakpoints_MW: {e}")

    return df


def create_output_directory():
    """Create the output directory for storing results."""
    output_dir = "../output/opt/cs1"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_npp_data(file_path):
    """Load nuclear power plant data from CSV file."""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Clean data - remove rows with ISO value "None"
        df = df[df["ISO"].notna() & (df["ISO"] != "None")]

        # Convert numeric columns
        numeric_columns = [
            "Licensed Power (MWt)", "Nameplate Capacity (MW)",
            "Summer Capacity (MW)", "Winter Capacity (MW)",
            "Minimum Load (MW)", "remaining"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(
                    ",", "").astype(float)

        # Calculate thermal efficiency
        df["Thermal_Efficiency"] = df["Nameplate Capacity (MW)"] / \
            df["Licensed Power (MWt)"]

        return df

    except Exception as e:
        print(f"Error loading NPP data: {str(e)}")
        sys.exit(1)


def run_plant_optimization_with_dynamic_params(plant_params, output_dir, verbose=True):
    """
    Run optimization for a nuclear power plant with dynamically adjusted system parameters
    based on the plant's remaining operational life.

    This function wraps the original run_plant_optimization function with custom
    parameter adjustment.
    """
    plant_id = plant_params["plant_id"]
    iso_region = plant_params["iso_region"]
    remaining_years = plant_params["remaining_years"]

    print(
        f"Running optimization for {plant_params['plant_name']} Unit {plant_params['generator_id']} ({iso_region}, {remaining_years} years)")

    # Load hourly data for this ISO
    hourly_data = load_hourly_data(iso_region, base_dir="../input/hourly_data")
    if hourly_data is None:
        print(f"Error: Failed to load hourly data for ISO: {iso_region}")
        return None

    # Get the original system parameters
    df_system = hourly_data.get('df_system')
    if df_system is None:
        print("Error: System parameters not found in hourly data")
        return None

    # Adjust system parameters based on remaining life
    adjusted_df_system = adjust_system_params_for_remaining_life(
        df_system, remaining_years, thermal_efficiency=plant_params.get('thermal_efficiency'))

    # **MODIFICATION: Set plant-specific parameters in the system DataFrame**
    # This ensures the model uses plant-specific values from the beginning

    # Set thermal capacity (total steam available)
    adjusted_df_system.loc['qSteam_Total_MWth',
                           'Value'] = plant_params["thermal_capacity_mwt"]

    # Set maximum turbine power capacity
    adjusted_df_system.loc['pTurbine_max_MW',
                           'Value'] = plant_params["nameplate_capacity_mw"]

    # Set thermal efficiency
    adjusted_df_system.loc['Turbine_Thermal_Elec_Efficiency_Const',
                           'Value'] = plant_params["thermal_efficiency"]

    # Calculate and set turbine steam limits based on thermal capacity and efficiency
    thermal_efficiency = plant_params["thermal_efficiency"]
    thermal_capacity_mwt = plant_params["thermal_capacity_mwt"]
    nameplate_capacity_mw = plant_params["nameplate_capacity_mw"]

    # Set maximum steam to turbine equal to total thermal capacity
    # This allows turbine to use all available steam if needed
    adjusted_df_system.loc['qSteam_Turbine_max_MWth',
                           'Value'] = thermal_capacity_mwt

    # Set minimum turbine power based on 100 MWt steam consumption
    # This provides a reasonable minimum load while maintaining flexibility
    min_steam_100mwt = 100.0  # MWt
    min_power_from_100mwt = min_steam_100mwt * thermal_efficiency
    adjusted_df_system.loc['pTurbine_min_MW', 'Value'] = min_power_from_100mwt

    # Set minimum steam to turbine
    adjusted_df_system.loc['qSteam_Turbine_min_MWth',
                           'Value'] = min_steam_100mwt

    # Update pTurbine_Outputs_at_Breakpoints_MW to cover the full range
    # Create breakpoints that span from minimum to maximum operation
    if 'qSteam_Turbine_Breakpoints_MWth' in adjusted_df_system.index:
        # Get existing breakpoints and ensure they cover the full range
        q_breakpoints_str = str(
            adjusted_df_system.loc['qSteam_Turbine_Breakpoints_MWth', 'Value'])
        try:
            q_breakpoints = [float(x.strip())
                             for x in q_breakpoints_str.split(',') if x.strip()]

            # Ensure we have breakpoints that cover the full operational range
            # Add minimum and maximum if not already present
            min_steam = min_steam_100mwt
            max_steam = thermal_capacity_mwt

            # Create a comprehensive set of breakpoints
            if len(q_breakpoints) < 3:  # If not enough breakpoints, create new ones
                q_breakpoints = [min_steam,
                                 thermal_capacity_mwt * 0.5, max_steam]
            else:
                # Ensure existing breakpoints cover the range
                q_breakpoints[0] = min(q_breakpoints[0], min_steam)
                q_breakpoints[-1] = max(q_breakpoints[-1], max_steam)

            # Sort breakpoints
            q_breakpoints = sorted(set(q_breakpoints))

            # Calculate corresponding power outputs
            p_outputs = [q * thermal_efficiency for q in q_breakpoints]

            # Update the parameters
            q_breakpoints_str = ', '.join(f'{q:.2f}' for q in q_breakpoints)
            p_outputs_str = ', '.join(f'{p:.4f}' for p in p_outputs)

            adjusted_df_system.loc['qSteam_Turbine_Breakpoints_MWth',
                                   'Value'] = q_breakpoints_str
            adjusted_df_system.loc['pTurbine_Outputs_at_Breakpoints_MW',
                                   'Value'] = p_outputs_str

        except Exception as e:
            print(f"Warning: Could not update turbine breakpoints: {e}")

    # Update the hourly data with the adjusted system parameters
    hourly_data['df_system'] = adjusted_df_system

    # Now call the original run_plant_optimization function
    print("Creating optimization model...")

    # Create the model
    model = create_model(
        hourly_data,
        iso_region,
        simulate_dispatch=True
    )

    if model is None:
        print(
            f"Error: Failed to create optimization model for plant: {plant_id}")
        return None

    # **REMOVED: Old parameter override code that didn't work correctly**
    # The model now uses the correct plant-specific parameters from the system DataFrame

    # Create solver
    solver_name = None
    for solver_name_candidate in ["gurobi", "cplex", "glpk", "cbc"]:
        try:
            solver = SolverFactory(solver_name_candidate)
            solver_name = solver_name_candidate
            print(f"Using {solver_name} solver")
            break
        except:
            pass

    if solver_name is None:
        print("Error: No suitable solver found. Exiting optimization process.")
        return None

    # Solver options
    solver_options = {}
    if solver_name == "gurobi":
        solver_options = {"MIPGap": 0.0005}  # "TimeLimit": 600,
        target_gap = 0.0005
    elif solver_name == "cplex":
        solver_options = {"timelimit": 600, "mipgap": 0.01}
        target_gap = 0.01
    else:
        target_gap = 0.01

    # Get plant name and generator ID for file naming
    plant_name = plant_params["plant_name"]
    generator_id = int(plant_params["generator_id"])
    file_prefix = f"{plant_name}_{generator_id}_{iso_region}_{int(remaining_years)}"

    # Create log file for solver output monitoring
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    solver_log_file = os.path.join(log_dir, f"{file_prefix}_solver_output.log")

    # Clear/create a fresh log file for this optimization
    if os.path.exists(solver_log_file):
        os.remove(solver_log_file)

    # Add logging to solver options
    if solver_name == "gurobi":
        solver_options["LogFile"] = solver_log_file
    elif solver_name == "cplex":
        solver_options["logfile"] = solver_log_file

    # Solve model
    print("Solving optimization model...")

    # Create and start progress indicator with gap monitoring
    progress = SolverProgressIndicator(
        f"Optimizing {plant_params['plant_name']} Unit {plant_params['generator_id']}",
        target_gap=target_gap)
    progress.start(solver_name=solver_name, log_file=solver_log_file)

    try:
        results = solver.solve(model, tee=False, options=solver_options)
    finally:
        # Always stop the progress indicator, even if solve fails
        progress.stop()

    # Check solver status
    solver_status = results.solver.status
    termination_condition = results.solver.termination_condition

    if solver_status == SolverStatus.ok and (
        termination_condition == TerminationCondition.optimal or
        termination_condition == TerminationCondition.feasible
    ):
        print("Optimization completed successfully")

        # Extract results
        results_df, summary_results = extract_results(
            model, target_iso=iso_region)

        # Save hourly results to the output directory
        results_file = os.path.join(
            output_dir, f"{file_prefix}_hourly_results.csv")
        results_df.to_csv(results_file, index=False)

        print(f"Results saved to: {results_file}")

        return True
    else:
        print(
            f"Optimization failed - Status: {results.solver.status}, Condition: {results.solver.termination_condition}")
        return None


def main():
    """Main function to control the execution workflow."""
    print("Starting Nuclear Power Plant Optimization")

    # Create output directory
    output_dir = create_output_directory()

    # Load NPP data
    npp_data_file = "../input/hourly_data/NPPs info.csv"

    print(f"Loading NPP data from: {npp_data_file}")
    npp_df = load_npp_data(npp_data_file)

    # Process each nuclear power plant
    processed_count = 0
    total_plants = len(npp_df)

    for idx, row in npp_df.iterrows():
        plant_data = row.to_dict()

        # Skip reactors with remaining operational life of less than 10 years
        if plant_data.get('remaining', 0) < 10:
            continue

        # Get Plant Name and Generator ID for file naming
        plant_name = plant_data['Plant Name']
        generator_id = plant_data['Generator ID']
        plant_id = f"{plant_data['Plant Code']}_{plant_data['Generator ID']}"
        iso_region = plant_data["ISO"]

        processed_count += 1
        print(
            f"\n[{processed_count}/{total_plants}] Processing: {plant_name} Unit {generator_id} ({iso_region})")

        # Prepare plant parameters for optimization
        plant_params = {
            "plant_id": plant_id,
            "plant_name": plant_name,
            "generator_id": generator_id,
            "iso_region": iso_region,
            "thermal_capacity_mwt": plant_data["Licensed Power (MWt)"],
            "nameplate_capacity_mw": plant_data["Nameplate Capacity (MW)"],
            "min_load_mw": plant_data["Minimum Load (MW)"],
            "thermal_efficiency": plant_data["Thermal_Efficiency"],
            "remaining_years": plant_data["remaining"]
        }

        # Run optimization using the framework
        try:
            # Call the modified optimization function with dynamic parameter adjustment
            success = run_plant_optimization_with_dynamic_params(
                plant_params=plant_params,
                output_dir=output_dir,
                verbose=False
            )

            # Check if optimization was successful
            if not success:
                print(
                    f"Optimization failed for {plant_name} Unit {generator_id}")
                continue

        except Exception as e:
            print(
                f"Error processing {plant_name} Unit {generator_id}: {str(e)}")
            # Continue with the next plant
            continue

    print(f"\nOptimization completed. Results saved in: {output_dir}")
    print("Nuclear Power Plant Optimization Completed")


if __name__ == "__main__":
    main()
