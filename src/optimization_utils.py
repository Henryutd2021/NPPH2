"""
Optimization utilities for Nuclear Power Plants

This module contains functions for running optimization on nuclear power plants
using the existing optimization framework.
"""

import os
import json
import sys
import pandas as pd
from pathlib import Path

# Import framework modules
from config import HOURS_IN_YEAR, TARGET_ISO
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pyomo.environ as pyo


def run_plant_optimization(plant_params, output_paths, verbose=True):
    """
    Run optimization for a nuclear power plant using the existing optimization framework.

    Args:
        plant_params: Dictionary containing plant parameters
            Required keys:
                - plant_id: Plant identifier
                - plant_name: Plant name
                - generator_id: Generator ID
                - iso_region: ISO region code
                - thermal_capacity_mwt: Thermal capacity in MWt
                - nameplate_capacity_mw: Nameplate capacity in MW
                - min_load_mw: Minimum load in MW
                - thermal_efficiency: Thermal efficiency
                - remaining_years: Remaining operational years

        output_paths: Dictionary containing output paths
            Required keys:
                - plant_output_dir: Directory for plant-specific outputs
                - optimization_results_dir: Directory for optimization results

        verbose: Whether to print detailed logs

    Returns:
        dict: Optimization results
    """
    # Extract parameters
    plant_id = plant_params["plant_id"]
    iso_region = plant_params["iso_region"]
    thermal_capacity = plant_params["thermal_capacity_mwt"]
    nameplate_capacity = plant_params["nameplate_capacity_mw"]
    min_load = plant_params["min_load_mw"]
    thermal_efficiency = plant_params["thermal_efficiency"]
    remaining_years = plant_params["remaining_years"]

    # Extract names for file naming
    plant_name = plant_params.get("plant_name", "Plant")
    generator_id = plant_params.get("generator_id", "Unit")
    file_prefix = f"{plant_name}_{generator_id}"

    # Extract output paths
    plant_output_dir = output_paths["plant_output_dir"]
    opt_dir = output_paths["optimization_results_dir"]

    if verbose:
        print(f"\n{'='*50}")
        print(f"Running optimization for {plant_name} - {generator_id}")
        print(f"ISO Region: {iso_region}")
        print(f"{'='*50}")

    # Load hourly data for this plant's ISO
    if verbose:
        print(f"Loading hourly data for ISO: {iso_region}")

    hourly_data = load_hourly_data(iso_region)

    if hourly_data is None:
        print(f"Error: Failed to load hourly data for ISO: {iso_region}")
        print("Exiting optimization process.")
        return None

    # Create the model
    if verbose:
        print(f"Creating optimization model for plant: {plant_id}")

    model = create_model(
        hourly_data,
        iso_region,
        simulate_dispatch=True
    )

    if model is None:
        print(
            f"Error: Failed to create optimization model for plant: {plant_id}")
        print("Exiting optimization process.")
        return None

    # Override model parameters with plant-specific data
    if hasattr(model, 'pTurbine_max_MW'):
        model.pTurbine_max_MW = nameplate_capacity
    elif hasattr(model, 'pNuclear_max_MW'):
        model.pNuclear_max_MW = nameplate_capacity

    if hasattr(model, 'Turbine_Thermal_Elec_Efficiency_Const'):
        model.Turbine_Thermal_Elec_Efficiency_Const = thermal_efficiency
    elif hasattr(model, 'Nuclear_Thermal_Elec_Efficiency_Const'):
        model.Nuclear_Thermal_Elec_Efficiency_Const = thermal_efficiency

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
        solver_options = {"TimeLimit": 600, "MIPGap": 0.01}
    elif solver_name == "cplex":
        solver_options = {"timelimit": 600, "mipgap": 0.01}

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

        # Save detailed results
        results_file = os.path.join(
            plant_output_dir, f"{file_prefix}_hourly_results.csv")
        results_df.to_csv(results_file, index=False)

        # Save summary results
        summary_file = os.path.join(
            plant_output_dir, f"{file_prefix}_summary_results.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_results, f, indent=2)

        # Extract total profit
        total_profit = 0
        if hasattr(model, 'TotalProfit_Objective'):
            try:
                total_profit = pyo.value(model.TotalProfit_Objective)
            except:
                total_profit = summary_results.get("TotalProfit", 0)
        else:
            total_profit = summary_results.get("TotalProfit", 0)

        # Create plant result
        plant_result = {
            "plant_id": plant_id,
            "plant_name": plant_name,
            "generator_id": generator_id,
            "iso_region": iso_region,
            "optimal_capacity_factor": summary_results.get("CapacityFactor", 0),
            "optimal_revenue": summary_results.get("TotalRevenue", 0),
            "optimal_cost": summary_results.get("TotalCost", 0),
            "optimal_profit": total_profit,
            "optimization_status": "success",
            "thermal_efficiency": thermal_efficiency,
            "thermal_capacity_mwt": thermal_capacity,
            "nameplate_capacity_mw": nameplate_capacity,
            "min_load_mw": min_load,
            "remaining_years": remaining_years
        }

        # Save plant result
        opt_result_file = os.path.join(
            opt_dir, f"{file_prefix}_optimization_results.json")
        with open(opt_result_file, 'w') as f:
            json.dump(plant_result, f, indent=4)

        return plant_result
    else:
        if verbose:
            print(f"Optimization failed for plant: {plant_id}")
            print(f"Solver status: {results.solver.status}")
            print(
                f"Termination condition: {results.solver.termination_condition}")

        print("Error: Optimization failed. Exiting optimization process.")
        return None
