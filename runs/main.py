# runs/main.py
"""
Main entry point for the nuclear-hydrogen optimization framework.
This script integrates the core optimization functionality with enhanced features
and includes debugging logic for infeasible/unbounded models.

Run:
    python main.py --iso PJM --solver gurobi --hours 8760
"""

from __future__ import annotations

import sys
import timeit
import traceback  # Import traceback for detailed error logging
from argparse import ArgumentParser
from pathlib import Path

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from config import HOURS_IN_YEAR  # <-- Added this import
from config import (
    ENABLE_BATTERY,
    ENABLE_ELECTROLYZER,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_H2_STORAGE,
    ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_STARTUP_SHUTDOWN,
    LOG_FILE,
    SIMULATE_AS_DISPATCH_EXECUTION,
    TARGET_ISO,
)
from data_io import load_hourly_data
from logging_setup import logger  # Make sure logger is initialized early
from model import create_model
from result_processing import extract_results

# --- MODIFICATION: Added HOURS_IN_YEAR to the import ---

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cli() -> ArgumentParser:
    """
    Parse command line arguments
    """
    p = ArgumentParser(description="Nuclear-Hydrogen Flexibility Optimization Model")
    p.add_argument(
        "--iso", default=TARGET_ISO, help="Target ISO (default: %(default)s)"
    )
    p.add_argument(
        "--solver", default="gurobi", help="Solver name registered with Pyomo"
    )
    # --- MODIFICATION: Set default hours from config ---
    p.add_argument(
        "--hours",
        type=int,
        default=HOURS_IN_YEAR,  # Use imported HOURS_IN_YEAR as default
        help="Number of hours to simulate (default: from config)",
    )
    # --- Add option to enable IIS/Debugging ---
    p.add_argument(
        "--debug-infeasibility",
        action="store_true",
        help="Enable IIS computation if model is infeasible (requires compatible solver like Gurobi/CPLEX)",
    )
    return p


def print_configuration(args):
    """
    Print current configuration settings
    """
    logger.info("Current Configuration:")
    logger.info("----------------------")
    logger.info(f"Target ISO: {args.iso}")  # Use args.iso
    logger.info(f"Solver: {args.solver}")
    logger.info(f"Simulation Hours: {args.hours}")  # Use args.hours
    logger.info(f"Debug Infeasibility: {args.debug_infeasibility}")
    logger.info(f"Nuclear Generator: {ENABLE_NUCLEAR_GENERATOR}")
    logger.info(f"Electrolyzer: {ENABLE_ELECTROLYZER}")
    logger.info(f"Battery Storage: {ENABLE_BATTERY}")
    logger.info(f"H2 Storage: {ENABLE_H2_STORAGE}")
    logger.info(f"H2 Cap Factor: {ENABLE_H2_CAP_FACTOR}")
    logger.info(f"Electrolyzer Degradation: {ENABLE_ELECTROLYZER_DEGRADATION_TRACKING}")
    logger.info(f"Startup/Shutdown: {ENABLE_STARTUP_SHUTDOWN}")
    logger.info(f"Nonlinear Turbine: {ENABLE_NONLINEAR_TURBINE_EFF}")
    logger.info(f"AS Dispatch Simulation: {SIMULATE_AS_DISPATCH_EXECUTION}")
    logger.info("----------------------")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None):
    """
    Main execution function
    """
    # --- ADDED: Basic print to confirm script start ---
    print("--- main.py script started ---")
    logger.info("--- main.py script started ---")

    args = parse_cli().parse_args(argv)

    # Start timing
    t0 = timeit.default_timer()
    logger.info("=== Starting optimization run for %s ===", args.iso)
    # Also print to console
    print(f"=== Starting optimization run for {args.iso} ===")

    # Print configuration using parsed args
    print_configuration(args)  # This logs the config

    model = None  # Initialize model to None
    data = None  # Initialize data to None
    solver = None  # Initialize solver to None
    results = None  # Initialize results to None

    try:
        # 1. Load data
        logger.info("Step 1: Loading hourly data...")
        print("Step 1: Loading hourly data...")
        # --- MODIFICATION: Use imported HOURS_IN_YEAR for comparison ---
        if args.hours != HOURS_IN_YEAR:
            logger.warning(
                f"Simulating for {args.hours} hours, but config HOURS_IN_YEAR is {HOURS_IN_YEAR}. Ensure data loading and model use the correct number of hours."
            )
            # NOTE: data_io.py and potentially model.py might need adjustment
            # if args.hours should override config HOURS_IN_YEAR for data slicing / TimePeriods definition.
            # For now, we just warn. The model will likely use HOURS_IN_YEAR from config.

        data = load_hourly_data(args.iso)  # Pass base_dir if needed
        if data is None:
            logger.critical("Data loading failed – terminating.")
            print("ERROR: Data loading failed.")
            sys.exit(1)
        logger.info("Step 1: Data loading complete.")
        print("Step 1: Data loading complete.")

        # 2. Create model
        logger.info("Step 2: Creating optimization model...")
        print("Step 2: Creating optimization model...")
        # Pass simulate_dispatch flag correctly
        model = create_model(
            data, args.iso, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION
        )
        # Check if model creation returned None (could indicate internal error)
        if model is None:
            logger.critical("Model creation failed (returned None) - terminating.")
            print("ERROR: Model creation failed.")
            sys.exit(2)
        logger.info("Step 2: Model creation complete.")
        print("Step 2: Model creation complete.")

        # 3. Solve model
        logger.info(f"Step 3: Setting up solver: {args.solver}...")
        print(f"Step 3: Setting up solver: {args.solver}...")
        solver = SolverFactory(args.solver)
        logger.info("Step 3: Solver factory setup complete.")
        print("Step 3: Solver factory setup complete.")

        # --- Add Solver Options for Debugging (Example for Gurobi) ---
        solver_options = {}  # Initialize as empty dict
        if args.solver == "gurobi":
            solver_options["MIPGap"] = 0.0005  # 0.05% gap
            # solver_options["Heuristics"] = 0.8
            # solver_options["Threads"] = 64
            # solver_options["Cuts"] = 2
            if args.debug_infeasibility:
                logger.info(
                    "Gurobi IIS computation is implicitly enabled on infeasibility."
                )
                # solver_options['ResultFile'] = f"{args.iso}_infeasible_model.ilp" # Optional: Write IIS directly if solver supports
                # solver_options['InfUnbdInfo'] = 1 # Optional: Get more info for infeasible/unbounded

        elif args.solver == "cplex":
            if args.debug_infeasibility:
                solver_options["iisfind"] = 1
                logger.info("CPLEX IIS computation enabled.")

        # --- Solve ---
        logger.info("Step 3b: Calling solver.solve()...")
        print("Step 3b: Calling solver.solve()...")
        # --- MODIFICATION: Always pass the solver_options dictionary ---
        results = solver.solve(model, tee=True, options=solver_options)
        logger.info("Step 3b: solver.solve() call finished.")
        print("Step 3b: solver.solve() call finished.")

    # --- MODIFICATION: Catch exceptions during setup/solve ---
    except Exception as e:
        logger.critical(
            f"An exception occurred during model setup or solver call: {e}",
            exc_info=True,
        )
        print(f"ERROR during model setup or solver call: {e}")
        # Print traceback for more details
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
        sys.exit(3)  # Exit if setup/solve crashes

    # --- Check if results object is valid ---
    if results is None:
        logger.critical("Solver did not return a results object. Terminating.")
        print("ERROR: Solver did not return a results object.")
        sys.exit(3)

    # 4. Process results and Check Solver Status
    logger.info("Step 4: Processing results and checking solver status...")
    print("Step 4: Processing results and checking solver status...")
    solver_status = results.solver.status
    term_cond = results.solver.termination_condition
    logger.info(f"Solver Status: {solver_status}")
    logger.info(f"Termination Condition: {term_cond}")
    print(f"Solver Status: {solver_status}")
    print(f"Termination Condition: {term_cond}")

    results_df = None
    summary_results = {}
    profit = None

    # --- MODIFICATION: Wrap result extraction in try...except ---
    try:
        if solver_status == SolverStatus.ok and term_cond in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            logger.info("Solver found an optimal or feasible solution.")
            print("Solver found an optimal or feasible solution.")
            # Extract results only if a feasible/optimal solution was found
            results_df, summary_results = extract_results(model, target_iso=args.iso)
            profit = pyo.value(model.TotalProfit_Objective)
            logger.info("Total profit = $%.2f", profit)

        elif term_cond == TerminationCondition.infeasible:
            logger.error("Model determined to be INFEASIBLE.")
            print("ERROR: Model determined to be INFEASIBLE.")
            if args.debug_infeasibility:
                logger.info(
                    "Attempting to identify Irreducible Inconsistent Subsystem (IIS)..."
                )
                print("Attempting to write infeasible model file for IIS analysis...")
                try:
                    log_dir = Path("../logs")
                    log_dir.mkdir(parents=True, exist_ok=True)
                    infeasible_lp_file = log_dir / f"{args.iso}_infeasible_model.lp"
                    # --- MODIFICATION: Add try-except around model.write ---
                    try:
                        model.write(
                            str(infeasible_lp_file),
                            io_options={"symbolic_solver_labels": True},
                        )
                        logger.info(f"Wrote infeasible model to {infeasible_lp_file}")
                        print(f"Wrote infeasible model to {infeasible_lp_file}")
                        logger.info(
                            f"Try running IIS analysis directly with your solver:"
                        )
                        print(f"Try running IIS analysis directly with your solver:")
                        if args.solver == "gurobi":
                            logger.info(
                                f"  gurobi_cl ResultFile={args.iso}_iis.ilp {infeasible_lp_file}"
                            )
                            print(
                                f"  gurobi_cl ResultFile={args.iso}_iis.ilp {infeasible_lp_file}"
                            )
                        elif args.solver == "cplex":
                            logger.info(
                                f"  Run CPLEX interactive, read {infeasible_lp_file}, then use 'tools iis'"
                            )
                            print(
                                f"  Run CPLEX interactive, read {infeasible_lp_file}, then use 'tools iis'"
                            )
                    except Exception as write_e:
                        logger.error(
                            f"Could not write infeasible model file: {write_e}",
                            exc_info=True,
                        )
                        print(
                            f"ERROR: Could not write infeasible model file: {write_e}"
                        )

                except Exception as path_e:  # Catch potential Path issues
                    logger.error(
                        f"Error setting up path for infeasible model file: {path_e}",
                        exc_info=True,
                    )
                    print(
                        f"ERROR: Error setting up path for infeasible model file: {path_e}"
                    )
            else:
                logger.warning(
                    "Run with --debug-infeasibility to enable IIS computation attempt (requires compatible solver)."
                )
                print(
                    "INFO: Run with --debug-infeasibility to enable IIS computation attempt."
                )
            # --- MODIFICATION: Ensure exit even if file write fails ---
            sys.exit(4)  # Exit code for infeasible

        elif term_cond == TerminationCondition.unbounded:
            logger.error("Model determined to be UNBOUNDED.")
            print("ERROR: Model determined to be UNBOUNDED.")
            logger.warning(
                "Diagnosing unbounded models often requires checking objective function terms and variable bounds."
            )
            try:
                log_dir = Path("../logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                unbounded_lp_file = log_dir / f"{args.iso}_unbounded_model.lp"
                try:
                    model.write(
                        str(unbounded_lp_file),
                        io_options={"symbolic_solver_labels": True},
                    )
                    logger.info(
                        f"Wrote potentially unbounded model to {unbounded_lp_file}"
                    )
                    print(f"Wrote potentially unbounded model to {unbounded_lp_file}")
                except Exception as write_e:
                    logger.error(
                        f"Could not write unbounded model file: {write_e}",
                        exc_info=True,
                    )
                    print(f"ERROR: Could not write unbounded model file: {write_e}")
            except Exception as path_e:
                logger.error(
                    f"Error setting up path for unbounded model file: {path_e}",
                    exc_info=True,
                )
                print(
                    f"ERROR: Error setting up path for unbounded model file: {path_e}"
                )
            sys.exit(5)  # Exit code for unbounded

        elif (
            solver_status == SolverStatus.warning
            and term_cond == TerminationCondition.maxTimeLimit
        ):
            logger.warning("Solver reached maximum time limit.")
            print("WARNING: Solver reached maximum time limit.")
            # Try to extract results if available, but they might be suboptimal
            if (
                hasattr(model, "TotalProfit_Objective")
                and model.TotalProfit_Objective.value is not None
            ):
                profit = pyo.value(model.TotalProfit_Objective)
                logger.warning(f"Suboptimal profit found = $%.2f", profit)
                results_df, summary_results = extract_results(
                    model, target_iso=args.iso
                )  # Attempt extraction
            else:
                logger.warning(
                    "No objective value available after reaching time limit."
                )

        elif solver_status == SolverStatus.error:
            logger.error(
                f"Solver reported an ERROR. Termination condition: {term_cond}"
            )
            print(
                f"ERROR: Solver reported an ERROR. Termination condition: {term_cond}"
            )
            if hasattr(results, "problem") and hasattr(results.problem, "message"):
                logger.error(f"Problem message: {results.problem.message}")
                print(f"Problem message: {results.problem.message}")
            sys.exit(6)  # Exit code for solver error

        else:  # Other statuses (aborted, unknown, etc.)
            logger.error(
                f"Solver finished with unexpected status: {solver_status}, termination condition: {term_cond}"
            )
            print(
                f"ERROR: Solver finished with unexpected status: {solver_status}, termination condition: {term_cond}"
            )
            sys.exit(7)  # Exit code for other issues

    except Exception as e:
        logger.error(
            f"An exception occurred during result processing or status checking: {e}",
            exc_info=True,
        )
        print(f"ERROR during result processing or status checking: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
        sys.exit(8)  # Different exit code for result processing errors

    # Calculate runtime and log final messages only if exited normally or with suboptimal solution
    runtime = timeit.default_timer() - t0
    logger.info("Optimization process finished.")
    if profit is not None:
        logger.info("Final profit = $%.2f", profit)
    else:
        logger.info(
            "No final profit value available (model might have been infeasible, unbounded, or errored)."
        )
    logger.info("Runtime = %.2f seconds", runtime)
    logger.info("Log saved to %s", LOG_FILE)

    print("\nRun Summary:")
    print("---------------")
    print(f"Solver Status: {solver_status}")
    print(f"Termination Condition: {term_cond}")
    if profit is not None:
        print(f"Total profit = ${profit:,.2f}")
    else:
        print("Total profit: N/A")
    print(f"Runtime = {runtime:.2f} seconds")
    print(f"Detailed logs: {LOG_FILE}")
    if term_cond == TerminationCondition.infeasible and args.debug_infeasibility:
        infeasible_lp_file = Path(f"../logs/{args.iso}_infeasible_model.lp")
        print(f"Infeasible model saved to {infeasible_lp_file}")
        print(f"Try running IIS analysis using your solver's command line tools.")
    elif term_cond == TerminationCondition.unbounded:
        unbounded_lp_file = Path(f"../logs/{args.iso}_unbounded_model.lp")
        print(f"Potentially unbounded model saved to {unbounded_lp_file}")


if __name__ == "__main__":
    try:
        main()
        print("--- main.py script finished normally ---")
        logger.info("--- main.py script finished normally ---")
    except SystemExit as e:
        print(f"--- main.py script exited with code {e.code} ---")
        logger.warning(f"--- main.py script exited with code {e.code} ---")
    except Exception as e:
        print(
            f"--- main.py script encountered an unhandled top-level exception: {e} ---"
        )
        logger.critical(
            f"--- main.py script encountered an unhandled top-level exception: {e} ---",
            exc_info=True,
        )
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
