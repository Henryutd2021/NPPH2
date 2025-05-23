# runs/main.py
"""
Main entry point for the nuclear-hydrogen optimization framework.
This script integrates the core optimization functionality with enhanced features
and includes debugging logic for infeasible/unbounded models.

Run:
    python main.py --iso PJM --solver gurobi --hours 8760
"""

from __future__ import annotations
from result_processing import extract_results
from model import create_model
from logging_setup import logger  # Make sure logger is initialized early
from data_io import load_hourly_data
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
from config import HOURS_IN_YEAR  # <-- Added this import

import sys
import timeit
import traceback  # Import traceback for detailed error logging
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
import os
import math

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))


# --- MODIFICATION: Added HOURS_IN_YEAR to the import ---

# ---------------------------------------------------------------------------
# Progress Indicator
# ---------------------------------------------------------------------------

class SolverProgressIndicator:
    """
    A progress indicator for optimization solving process that shows real progress
    based on MIP gap information from the solver output.
    """

    def __init__(self, description="Solving optimization model", target_gap=0.0005):
        self.description = description
        self.target_gap = target_gap  # Target MIP gap (e.g., 0.0005 = 0.05%)
        self.running = False
        self.thread = None
        self.start_time = None
        self.current_gap = None
        self.best_bound = None
        self.best_solution = None
        self.log_file = None
        self.iterations = 0

    def _parse_gurobi_line(self, line):
        """Parse Gurobi output line to extract gap information."""
        try:
            line = line.strip()
            if not line:
                return False

            # Pattern 1: Heuristic solution lines: "H  150     0                    2.234056e+08 2.234056e+08  0.00%     0s"
            if line.startswith('H') and '%' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if '%' in part:
                        gap_str = part.replace('%', '')
                        try:
                            self.current_gap = float(gap_str) / 100.0
                            return True
                        except ValueError:
                            pass

            # Pattern 2: Regular iteration lines with gap: "   150     0 2.234056e+08 2.234056e+08  0.00%     0s"
            elif line[0].isdigit() and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part and part != '%':
                        gap_str = part.replace('%', '')
                        try:
                            self.current_gap = float(gap_str) / 100.0
                            return True
                        except ValueError:
                            pass

            # Pattern 3: Final gap information: "Best objective 1.234567e+08, best bound 1.234567e+08, gap 0.00%"
            elif 'gap' in line.lower() and '%' in line:
                import re
                # Look for "gap X.XX%" pattern
                match = re.search(r'gap\s+(\d+\.?\d*)%', line, re.IGNORECASE)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True

            # Pattern 4: Presolve or other status lines that might contain gap
            elif 'gap:' in line.lower() and '%' in line:
                import re
                match = re.search(r'gap:\s*(\d+\.?\d*)%', line, re.IGNORECASE)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True

        except (ValueError, IndexError, AttributeError):
            pass
        return False

    def _parse_cplex_line(self, line):
        """Parse CPLEX output line to extract gap information."""
        try:
            # CPLEX format varies, but often contains "gap" keyword
            if 'gap' in line.lower() and '%' in line:
                # Extract percentage value
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True
        except (ValueError, IndexError):
            pass
        return False

    def _monitor_solver_output(self, solver_name):
        """Monitor solver output file for gap information."""
        # Wait for log file to be created (up to 30 seconds)
        wait_time = 0
        max_wait = 30
        while not os.path.exists(self.log_file) and wait_time < max_wait and self.running:
            time.sleep(0.5)
            wait_time += 0.5

        if not os.path.exists(self.log_file):
            print(
                f"Warning: Log file {self.log_file} not created after {max_wait}s")
            return

        print(f"Gap monitoring started for {self.log_file}")

        try:
            # Start reading from the end of the file to avoid old content
            last_position = 0
            # Get initial file size to start reading from current end
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    f.seek(0, 2)  # Go to end
                    last_position = f.tell()
                    print(f"Starting to monitor from position {last_position}")

            gap_found_count = 0
            while self.running:
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        if new_content:
                            # Process each line in the new content
                            lines = new_content.split('\n')
                            # Exclude last potentially incomplete line
                            for line in lines[:-1]:
                                if line.strip():
                                    if solver_name.lower() == 'gurobi':
                                        if self._parse_gurobi_line(line):
                                            gap_found_count += 1
                                            if gap_found_count <= 5:  # Show more for debugging
                                                print(
                                                    f"Gap found: {self.current_gap*100:.3f}% from NEW line: {line[:80]}...")
                                    elif solver_name.lower() == 'cplex':
                                        if self._parse_cplex_line(line):
                                            gap_found_count += 1
                                            if gap_found_count <= 5:
                                                print(
                                                    f"Gap found: {self.current_gap*100:.3f}% from NEW line: {line[:80]}...")
                            last_position = f.tell()
                        else:
                            time.sleep(0.2)  # Wait for more content
                except IOError:
                    # File might be locked by solver, wait and retry
                    time.sleep(0.5)
        except Exception as e:
            print(f"Error in gap monitoring: {e}")
        finally:
            if gap_found_count > 0:
                print(
                    f"Gap monitoring finished. Total NEW gaps found: {gap_found_count}")
            else:
                print(
                    f"Gap monitoring finished. No NEW gaps found in {self.log_file}")

    def _calculate_progress(self):
        """Calculate progress based on current gap and target gap."""
        if self.current_gap is None:
            return 0.0

        if self.current_gap <= self.target_gap:
            return 100.0

        # Assume initial gap is 100% and calculate progress
        # Progress = (initial_gap - current_gap) / (initial_gap - target_gap)
        # For simplicity, we'll use logarithmic scaling since gap decreases exponentially
        initial_gap = 1.0  # Assume 100% initial gap

        if self.current_gap >= initial_gap:
            return 0.0

        # Logarithmic progress calculation
        log_current = math.log10(max(self.current_gap, 1e-6))
        log_target = math.log10(max(self.target_gap, 1e-6))
        log_initial = math.log10(initial_gap)

        progress = (log_initial - log_current) / \
            (log_initial - log_target) * 100
        return min(max(progress, 0.0), 100.0)

    def _animate(self):
        """Internal method to run the animation in a separate thread."""
        if TQDM_AVAILABLE:
            # Use tqdm for a nice progress bar
            with tqdm(desc=self.description, total=100, unit="%",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| Gap: {postfix} | {elapsed}") as pbar:
                while self.running:
                    progress = self._calculate_progress()
                    gap_text = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                    pbar.set_postfix_str(gap_text)
                    pbar.n = progress
                    pbar.refresh()
                    time.sleep(0.5)

                # Set final state when animation stops
                final_progress = self._calculate_progress()
                final_gap = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                pbar.n = final_progress
                pbar.set_postfix_str(final_gap)
                pbar.close()

                # Show final summary
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    print(
                        f"{self.description} completed in {elapsed:.1f}s (Final gap: {final_gap})")
        else:
            # Fallback to simple text display
            spinners = ['|', '/', '-', '\\']
            i = 0
            while self.running:
                elapsed = time.time() - self.start_time if self.start_time else 0
                progress = self._calculate_progress()
                gap_text = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                print(f"\r{self.description}... {spinners[i % len(spinners)]} Progress: {progress:.1f}% | Gap: {gap_text} | ({elapsed:.1f}s)",
                      end="", flush=True)
                i += 1
                time.sleep(0.5)
            print()  # New line when done

    def start(self, solver_name="gurobi", log_file=None):
        """Start the progress indicator."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.log_file = log_file
            self.current_gap = None  # Reset gap for new run

            # Debug output
            if log_file:
                print(
                    f"Starting gap monitoring for {solver_name} with log file: {log_file}")

            # Start output monitoring thread if log file is provided
            if log_file and solver_name:
                monitor_thread = threading.Thread(
                    target=self._monitor_solver_output,
                    args=(solver_name,),
                    daemon=True,
                    name=f"GapMonitor-{solver_name}-{int(time.time())}"
                )
                monitor_thread.start()

            # Start animation thread
            self.thread = threading.Thread(
                target=self._animate,
                daemon=True,
                name=f"ProgressAnim-{int(time.time())}"
            )
            self.thread.start()

    def stop(self):
        """Stop the progress indicator."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)

            # Only show completion message if tqdm is not available
            # (tqdm animation will show its own completion message)
            if not TQDM_AVAILABLE and self.start_time:
                elapsed = time.time() - self.start_time
                final_gap = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                print(
                    f"\r{self.description} completed in {elapsed:.1f}s (Final gap: {final_gap})" + " " * 20)

    def update_gap(self, gap_value):
        """Manually update the current gap value."""
        self.current_gap = gap_value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_cli() -> ArgumentParser:
    """
    Parse command line arguments
    """
    p = ArgumentParser(
        description="Nuclear-Hydrogen Flexibility Optimization Model")
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
    logger.info("----------------------")


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
    print(f"Starting optimization for {args.iso}")

    # Print configuration using parsed args
    print_configuration(args)  # This logs the config

    model = None  # Initialize model to None
    data = None  # Initialize data to None
    solver = None  # Initialize solver to None
    results = None  # Initialize results to None

    try:
        # 1. Load data
        logger.info("Step 1: Loading hourly data...")
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

        # 2. Create model
        logger.info("Step 2: Creating optimization model...")
        # Pass simulate_dispatch flag correctly
        model = create_model(
            data, args.iso, simulate_dispatch=SIMULATE_AS_DISPATCH_EXECUTION
        )
        # Check if model creation returned None (could indicate internal error)
        if model is None:
            logger.critical(
                "Model creation failed (returned None) - terminating.")
            print("ERROR: Model creation failed.")
            sys.exit(2)
        logger.info("Step 2: Model creation complete.")

        # 3. Solve model
        logger.info(f"Step 3: Setting up solver: {args.solver}...")
        solver = SolverFactory(args.solver)
        logger.info("Step 3: Solver factory setup complete.")

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

        # Create log file for solver output monitoring
        log_dir = Path("../logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        solver_log_file = log_dir / f"{args.iso}_solver_output.log"

        # Clear/create a fresh log file for this optimization
        if solver_log_file.exists():
            solver_log_file.unlink()

        # Add logging to solver options
        if args.solver == "gurobi":
            solver_options["LogFile"] = str(solver_log_file)
            # Get target gap from solver options
            target_gap = solver_options.get("MIPGap", 0.0005)
        elif args.solver == "cplex":
            solver_options["logfile"] = str(solver_log_file)
            target_gap = solver_options.get("mipgap", 0.01)
        else:
            target_gap = 0.01  # Default gap for other solvers
            solver_log_file = None

        # Create and start progress indicator with gap monitoring
        progress = SolverProgressIndicator(
            "Solving optimization model", target_gap=target_gap)
        progress.start(args.solver, str(solver_log_file)
                       if solver_log_file else None)

        try:
            # --- MODIFICATION: Always pass the solver_options dictionary ---
            results = solver.solve(model, tee=False, options=solver_options)
        finally:
            # Always stop the progress indicator, even if solve fails
            progress.stop()

        logger.info("Step 3b: solver.solve() call finished.")

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
    solver_status = results.solver.status
    term_cond = results.solver.termination_condition
    logger.info(f"Solver Status: {solver_status}")
    logger.info(f"Termination Condition: {term_cond}")

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
            print("Optimization completed successfully")
            # Extract results only if a feasible/optimal solution was found
            results_df, summary_results = extract_results(
                model, target_iso=args.iso)
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
                    infeasible_lp_file = log_dir / \
                        f"{args.iso}_infeasible_model.lp"
                    # --- MODIFICATION: Add try-except around model.write ---
                    try:
                        model.write(
                            str(infeasible_lp_file),
                            io_options={"symbolic_solver_labels": True},
                        )
                        logger.info(
                            f"Wrote infeasible model to {infeasible_lp_file}")
                        print(
                            f"Wrote infeasible model to {infeasible_lp_file}")
                        logger.info(
                            f"Try running IIS analysis directly with your solver:"
                        )
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
                    print(
                        f"Wrote potentially unbounded model to {unbounded_lp_file}")
                except Exception as write_e:
                    logger.error(
                        f"Could not write unbounded model file: {write_e}",
                        exc_info=True,
                    )
                    print(
                        f"ERROR: Could not write unbounded model file: {write_e}")
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
    if term_cond == TerminationCondition.infeasible and args.debug_infeasibility:
        infeasible_lp_file = Path(f"../logs/{args.iso}_infeasible_model.lp")
        print(f"Infeasible model saved to {infeasible_lp_file}")
    elif term_cond == TerminationCondition.unbounded:
        unbounded_lp_file = Path(f"../logs/{args.iso}_unbounded_model.lp")
        print(f"Potentially unbounded model saved to {unbounded_lp_file}")


if __name__ == "__main__":
    try:
        main()
        logger.info("--- main.py script finished normally ---")
    except SystemExit as e:
        logger.warning(f"--- main.py script exited with code {e.code} ---")
    except Exception as e:
        logger.critical(
            f"--- main.py script encountered an unhandled top-level exception: {e} ---",
            exc_info=True,
        )
        print(f"ERROR: Unhandled exception: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
