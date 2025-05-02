"""Command‑line entry point.  Keeps the *main* script lightweight by simply
calling helper modules.

Run:
    python optimize.py --iso PJM --solver gurobi
"""
from __future__ import annotations

import sys
import timeit
from argparse import ArgumentParser

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from config import (TARGET_ISO, ENABLE_NONLINEAR_TURBINE_EFF, LOG_FILE)
from logging_setup import logger
from data_io import load_hourly_data
from model import create_model
from result_processing import extract_results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> ArgumentParser:
    p = ArgumentParser(description="Optimise nuclear‑hydrogen flexibility model")
    p.add_argument("--iso", default=TARGET_ISO, help="Target ISO (default: %(default)s)")
    p.add_argument("--solver", default="gurobi", help="Solver name registered with Pyomo")
    return p

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    args = parse_cli().parse_args(argv)

    t0 = timeit.default_timer()
    logger.info("--- optimisation run for %s ---", args.iso)

    # 1. data ----------------------------------------------------------------
    data = load_hourly_data(args.iso)
    if data is None:
        logger.critical("Data loading failed – terminating.")
        sys.exit(1)

    # 2. model ---------------------------------------------------------------
    model = create_model(data, args.iso)


    # 3. solve ---------------------------------------------------------------
    solver = SolverFactory(args.solver)
    results = solver.solve(model, tee=True)
    # process results
    results_df = extract_results(model, target_iso=args.iso)

    if results.solver.status != SolverStatus.ok:
        logger.error("Solver failed: %s", results.solver.status)
        sys.exit(2)

    if results.solver.termination_condition not in {TerminationCondition.optimal, TerminationCondition.feasible}:
        logger.warning("Solver termination: %s", results.solver.termination_condition)

    profit = pyo.value(model.TotalProfit_Objective)
    logger.info("Optimisation finished – Profit = $%.2f", profit)
    print(f"Total profit = ${profit:,.2f}")

    logger.info("Log saved to %s", LOG_FILE)
    print(f"Detailed logs: {LOG_FILE}")

if __name__ == "__main__":
    main()
