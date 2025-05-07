"""
Main entry point for the nuclear-hydrogen optimization framework.
This script integrates the core optimization functionality with enhanced features.

Run:
    python main.py --iso PJM --solver gurobi --hours 8760
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path
from argparse import ArgumentParser

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from config import (
    TARGET_ISO,
    ENABLE_NONLINEAR_TURBINE_EFF,
    LOG_FILE,
    SIMULATE_AS_DISPATCH_EXECUTION,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_ELECTROLYZER,
    ENABLE_BATTERY,
    ENABLE_H2_STORAGE,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_STARTUP_SHUTDOWN
)
from logging_setup import logger
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> ArgumentParser:
    """
    Parse command line arguments
    """
    p = ArgumentParser(description="Nuclear-Hydrogen Flexibility Optimization Model")
    p.add_argument("--iso", default=TARGET_ISO, 
                  help="Target ISO (default: %(default)s)")
    p.add_argument("--solver", default="gurobi", 
                  help="Solver name registered with Pyomo")
    p.add_argument("--hours", type=int, default=8760,
                  help="Number of hours to simulate (default: 8760)")
    return p

def print_configuration():
    """
    Print current configuration settings
    """
    logger.info("Current Configuration:")
    logger.info("----------------------")
    logger.info(f"Target ISO: {TARGET_ISO}")
    logger.info(f"Nuclear Generator: {ENABLE_NUCLEAR_GENERATOR}")
    logger.info(f"Electrolyzer: {ENABLE_ELECTROLYZER}")
    logger.info(f"Battery Storage: {ENABLE_BATTERY}")
    logger.info(f"H2 Storage: {ENABLE_H2_STORAGE}")
    logger.info(f"H2 Cap Factor: {ENABLE_H2_CAP_FACTOR}")
    logger.info(f"Electrolyzer Degradation: {ENABLE_ELECTROLYZER_DEGRADATION_TRACKING}")
    logger.info(f"Startup/Shutdown: {ENABLE_STARTUP_SHUTDOWN}")
    logger.info(f"Nonlinear Turbine: {ENABLE_NONLINEAR_TURBINE_EFF}")
    logger.info(f"AS Dispatch Simulation: {SIMULATE_AS_DISPATCH_EXECUTION}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    """
    Main execution function
    """
    args = parse_cli().parse_args(argv)
    
    # Start timing
    t0 = timeit.default_timer()
    logger.info("=== Starting optimization run for %s ===", args.iso)
    
    # Print configuration
    print_configuration()
    
    # 1. Load data
    logger.info("Loading hourly data...")
    data = load_hourly_data(args.iso)
    if data is None:
        logger.critical("Data loading failed – terminating.")
        sys.exit(1)
    
    # 2. Create model
    logger.info("Creating optimization model...")
    model = create_model(data, args.iso, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION)
    
    # 3. Solve model
    logger.info("Solving optimization model...")
    solver = SolverFactory(args.solver)
    results = solver.solve(model, tee=True)
    
    # 4. Process results
    logger.info("Processing results...")
    results_df = extract_results(model, target_iso=args.iso)
    
    # Check solver status
    if results.solver.status != SolverStatus.ok:
        logger.error("Solver failed: %s", results.solver.status)
        sys.exit(2)
    
    if results.solver.termination_condition not in {TerminationCondition.optimal, TerminationCondition.feasible}:
        logger.warning("Solver termination: %s", results.solver.termination_condition)
    
    # Calculate and display results
    profit = pyo.value(model.TotalProfit_Objective)
    runtime = timeit.default_timer() - t0
    
    logger.info("Optimization completed successfully")
    logger.info("Total profit = $%.2f", profit)
    logger.info("Runtime = %.2f seconds", runtime)
    logger.info("Log saved to %s", LOG_FILE)
    
    print("\nResults Summary:")
    print("---------------")
    print(f"Total profit = ${profit:,.2f}")
    print(f"Runtime = {runtime:.2f} seconds")
    print(f"Detailed logs: {LOG_FILE}")

if __name__ == "__main__":
    main()
