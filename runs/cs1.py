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
from pathlib import Path
warnings.filterwarnings("ignore")

# Add src directory to Python path for importing optimization framework modules
src_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

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
    print(
        f"Adjusting system parameters for remaining life: {remaining_years} years")

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
        print(
            f"{equip_type} effective lifetime: {effective_lifetime} years, CRF: {equip_crf:.5f}")

        # Update the parameters in the dataframe
        for param_name, calc_func in equip_data['params_to_update'].items():
            new_value = calc_func(equip_crf)
            if param_name in df.index:
                df.loc[param_name, 'Value'] = new_value
                print(f"Updated {param_name} to {new_value}")
            else:
                print(
                    f"Warning: Parameter {param_name} not found in system parameters")

    # Update hydrogen subsidy duration to min of 10 years or remaining life
    hydrogen_subsidy_duration = min(10, remaining_years)
    df.loc['hydrogen_subsidy_duration_years',
           'Value'] = hydrogen_subsidy_duration
    print(
        f"Updated hydrogen subsidy duration to {hydrogen_subsidy_duration} years")

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
            print(
                f"Dynamically updated pTurbine_Outputs_at_Breakpoints_MW to: {p_outputs_str}")
        except Exception as e:
            print(
                f"Warning: Failed to update pTurbine_Outputs_at_Breakpoints_MW: {e}")
    else:
        print("Warning: qSteam_Turbine_Breakpoints_MWth not found in system parameters")

    return df


def create_output_directory():
    """Create the output directory for storing results."""
    output_dir = "../output/cs1"
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

    if verbose:
        print(f"\n{'='*50}")
        print(
            f"Preparing optimization for {plant_params['plant_name']} - {plant_params['generator_id']}")
        print(
            f"ISO Region: {iso_region}, Remaining Life: {remaining_years} years")
        print(f"{'='*50}")

    # Load hourly data for this ISO
    hourly_data = load_hourly_data(iso_region)
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

    # Update the hourly data with the adjusted system parameters
    hourly_data['df_system'] = adjusted_df_system

    # Now call the original run_plant_optimization function
    if verbose:
        print("Running optimization with adjusted system parameters")

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

    # Override model parameters with plant-specific data
    if hasattr(model, 'pTurbine_max_MW'):
        model.pTurbine_max_MW = plant_params["nameplate_capacity_mw"]
    elif hasattr(model, 'pNuclear_max_MW'):
        model.pNuclear_max_MW = plant_params["nameplate_capacity_mw"]

    if hasattr(model, 'Turbine_Thermal_Elec_Efficiency_Const'):
        model.Turbine_Thermal_Elec_Efficiency_Const = plant_params["thermal_efficiency"]
    elif hasattr(model, 'Nuclear_Thermal_Elec_Efficiency_Const'):
        model.Nuclear_Thermal_Elec_Efficiency_Const = plant_params["thermal_efficiency"]

    min_load = plant_params["min_load_mw"]
    if hasattr(model, 'pTurbine_min_MW') and min_load > 0:
        model.pTurbine_min_MW = min_load
    elif hasattr(model, 'pNuclear_min_MW') and min_load > 0:
        model.pNuclear_min_MW = min_load

    # Create solver
    solver_name = None
    for solver_name_candidate in ["gurobi", "cplex", "glpk", "cbc"]:
        try:
            solver = SolverFactory(solver_name_candidate)
            solver_name = solver_name_candidate
            if verbose:
                print(f"Using {solver_name} solver")
            break
        except:
            if verbose:
                print(
                    f"Solver {solver_name_candidate} not available, trying another...")

    if solver_name is None:
        print("Error: No suitable solver found. Exiting optimization process.")
        return None

    # Solver options
    solver_options = {}
    if solver_name == "gurobi":
        solver_options = {"MIPGap": 0.0005}  # "TimeLimit": 600,
    elif solver_name == "cplex":
        solver_options = {"timelimit": 600, "mipgap": 0.01}

    # Get plant name and generator ID for file naming
    plant_name = plant_params["plant_name"]
    generator_id = int(plant_params["generator_id"])
    file_prefix = f"{plant_name}_{generator_id}_{iso_region}_{int(remaining_years)}"

    # Solve model
    if verbose:
        print(f"Solving optimization model for plant: {plant_id}")

    results = solver.solve(model, tee=verbose, options=solver_options)

    # Check solver status
    solver_status = results.solver.status
    termination_condition = results.solver.termination_condition

    if solver_status == SolverStatus.ok and (
        termination_condition == TerminationCondition.optimal or
        termination_condition == TerminationCondition.feasible
    ):
        if verbose:
            print(f"Optimization completed successfully for plant: {plant_id}")

        # Extract results
        results_df, summary_results = extract_results(
            model, target_iso=iso_region)

        # Save hourly results to the output directory
        results_file = os.path.join(
            output_dir, f"{file_prefix}_hourly_results.csv")
        results_df.to_csv(results_file, index=False)

        if verbose:
            print(f"Hourly results saved to: {results_file}")

        return True
    else:
        if verbose:
            print(f"Optimization failed for plant: {plant_id}")
            print(f"Solver status: {results.solver.status}")
            print(
                f"Termination condition: {results.solver.termination_condition}")

        print("Error: Optimization failed. Exiting optimization process.")
        return None


def main():
    """Main function to control the execution workflow."""
    print("Starting Nuclear Power Plant Optimization")

    # Create output directory
    output_dir = create_output_directory()

    # Load NPP data
    npp_data_file = "../input/hourly_data/NPPs info.csv"

    print(f"Using NPP data file: {npp_data_file}")
    npp_df = load_npp_data(npp_data_file)

    # Process each nuclear power plant
    for idx, row in npp_df.iterrows():
        plant_data = row.to_dict()

        # Skip reactors with remaining operational life of less than 10 years
        if plant_data.get('remaining', 0) < 10:
            print(f"Skipping plant {plant_data.get('Plant Name', 'Unknown Plant')} - Unit {plant_data.get('Generator ID', 'N/A')} due to remaining operational life < 10 years ({plant_data.get('remaining', 0)} years).")
            continue

        # Get Plant Name and Generator ID for file naming
        plant_name = plant_data['Plant Name']
        generator_id = plant_data['Generator ID']
        plant_id = f"{plant_data['Plant Code']}_{plant_data['Generator ID']}"
        iso_region = plant_data["ISO"]

        print(
            f"\nProcessing plant {idx+1}/{len(npp_df)}: {plant_name} - Unit {generator_id}")
        print(f"ISO Region: {iso_region}")

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
                verbose=True
            )

            # Check if optimization was successful
            if not success:
                print(
                    f"Optimization failed for {plant_name} - Unit {generator_id}. Skipping to next plant.")
                continue

        except Exception as e:
            print(
                f"Error processing plant {plant_name} - Unit {generator_id}: {str(e)}")
            # Continue with the next plant
            continue

    print(f"All hourly results saved in {output_dir}")
    print("Nuclear Power Plant Optimization Completed")


if __name__ == "__main__":
    main()
