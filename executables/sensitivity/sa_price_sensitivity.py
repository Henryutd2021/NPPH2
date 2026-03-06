#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly Price Sensitivity Analysis — Optimization + TEA Pipeline

For each of 4 price scenarios (-10%, -5%, +5%, +10%), this script:
  1. Loads all nuclear plants from NPPs info.csv (remaining life >= 10 yrs)
  2. For each plant: adjusts hourly energy prices, runs optimization, saves hourly results
  3. Immediately runs TEA on those results
  4. Organizes all outputs under scenario-named subdirectories

Output layout:
  output/opt/sa_price/price_m10pct/<plant>_hourly_results.csv
  output/opt/sa_price/price_m5pct/<plant>_hourly_results.csv
  output/opt/sa_price/price_p5pct/<plant>_hourly_results.csv
  output/opt/sa_price/price_p10pct/<plant>_hourly_results.csv

  output/tea/sa_price/price_m10pct/<plant>/...
  output/tea/sa_price/price_m5pct/<plant>/...
  output/tea/sa_price/price_p5pct/<plant>/...
  output/tea/sa_price/price_p10pct/<plant>/...
"""

import logging
import os
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — must come before any src.* imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from path_setup import setup_src_paths
setup_src_paths()

# ---------------------------------------------------------------------------
# Optimization imports
# ---------------------------------------------------------------------------
try:
    from src.opt.data_io import load_hourly_data
    from src.opt.model import create_model
    from src.opt.result_processing import extract_results
    from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
    _OPT_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Cannot import optimization modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# TEA imports
# ---------------------------------------------------------------------------
try:
    from src.tea.tea_engine import run_complete_tea_analysis
    _TEA_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Cannot import TEA modules: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Price scenario definitions
# ---------------------------------------------------------------------------
PRICE_SCENARIOS = {
    "price_m10pct": -0.10,
    "price_m5pct":  -0.05,
    "price_p5pct":  +0.05,
    "price_p10pct": +0.10,
    "price_p20pct": +0.20,
    "price_p30pct": +0.30,
}

# ---------------------------------------------------------------------------
# Paths (relative to this script's location: executables/sensitivity/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_HOURLY_DIR = PROJECT_ROOT / "input" / "hourly_data"
NPP_INFO_FILE = INPUT_HOURLY_DIR / "NPPs info.csv"
OPT_SA_BASE_DIR = PROJECT_ROOT / "output" / "opt" / "sa_price"
TEA_SA_BASE_DIR = PROJECT_ROOT / "output" / "tea" / "sa_price"
LOG_SA_BASE_DIR = PROJECT_ROOT / "output" / "logs" / "sa_price"

MIN_REMAINING_YEARS = 10


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def setup_logger(log_file: Path, name: str = "sa_price") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Utility: Capital Recovery Factor
# ---------------------------------------------------------------------------
def calculate_crf(discount_rate: float, lifetime_years: float) -> float:
    if lifetime_years <= 0:
        return 0.0
    if discount_rate == 0:
        return 1.0 / lifetime_years
    try:
        factor = (1 + discount_rate) ** lifetime_years
        if abs(factor - 1.0) < 1e-9:
            return 1.0 / lifetime_years
        return (discount_rate * factor) / (factor - 1)
    except (OverflowError, ValueError):
        return discount_rate


# ---------------------------------------------------------------------------
# Utility: Adjust system params for remaining plant life (mirrors opt_cs1.py)
# ---------------------------------------------------------------------------
def adjust_system_params(df_system: pd.DataFrame, plant_params: dict) -> pd.DataFrame:
    df = df_system.copy()
    remaining_years = plant_params["remaining_years"]
    thermal_efficiency = plant_params["thermal_efficiency"]

    discount_rate = float(df.loc["discount_rate", "Value"])
    df.loc["plant_lifetime_years", "Value"] = remaining_years

    equipment_params = {
        "LTE": {
            "default_lifetime": 20,
            "capex_usd_per_kw": 2000.0,
            "params_to_update": {
                "cost_electrolyzer_capacity_USD_per_MW_year_LTE": lambda crf: 2000.0 * 1000.0 * crf
            },
        },
        "HTE": {
            "default_lifetime": 20,
            "capex_usd_per_kw": 2500.0,
            "params_to_update": {
                "cost_electrolyzer_capacity_USD_per_MW_year_HTE": lambda crf: 2500.0 * 1000.0 * crf
            },
        },
        "Battery": {
            "default_lifetime": 15,
            "params_to_update": {
                "BatteryCapex_USD_per_MWh_year": lambda crf: 236.0 * 1000.0 * crf,
                "BatteryCapex_USD_per_MW_year": lambda crf: 236.0 * 1000.0 * 4.0 * crf,
                "BatteryFixedOM_USD_per_MWh_year": lambda crf: 236.0 * 0.01 * 1000.0,
            },
        },
    }

    for equip_data in equipment_params.values():
        eff_life = min(equip_data["default_lifetime"], remaining_years)
        crf = calculate_crf(discount_rate, eff_life)
        for param_name, calc_func in equip_data["params_to_update"].items():
            if param_name in df.index:
                df.loc[param_name, "Value"] = calc_func(crf)

    df.loc["hydrogen_subsidy_duration_years", "Value"] = min(10, remaining_years)

    # Plant-specific capacities
    thermal_capacity_mwt = plant_params["thermal_capacity_mwt"]
    nameplate_capacity_mw = plant_params["nameplate_capacity_mw"]

    df.loc["qSteam_Total_MWth", "Value"] = thermal_capacity_mwt
    df.loc["pTurbine_max_MW", "Value"] = nameplate_capacity_mw
    df.loc["Turbine_Thermal_Elec_Efficiency_Const", "Value"] = thermal_efficiency

    min_steam_mwt = 100.0
    df.loc["qSteam_Turbine_max_MWth", "Value"] = thermal_capacity_mwt
    df.loc["pTurbine_min_MW", "Value"] = min_steam_mwt * thermal_efficiency
    df.loc["qSteam_Turbine_min_MWth", "Value"] = min_steam_mwt

    # Update turbine breakpoints
    if "qSteam_Turbine_Breakpoints_MWth" in df.index:
        try:
            q_str = str(df.loc["qSteam_Turbine_Breakpoints_MWth", "Value"])
            q_bp = [float(x.strip()) for x in q_str.split(",") if x.strip()]
            if len(q_bp) < 3:
                q_bp = [min_steam_mwt, thermal_capacity_mwt * 0.5, thermal_capacity_mwt]
            else:
                q_bp[0] = min(q_bp[0], min_steam_mwt)
                q_bp[-1] = max(q_bp[-1], thermal_capacity_mwt)
            q_bp = sorted(set(q_bp))
            p_out = [q * thermal_efficiency for q in q_bp]
            df.loc["qSteam_Turbine_Breakpoints_MWth", "Value"] = ", ".join(f"{q:.2f}" for q in q_bp)
            df.loc["pTurbine_Outputs_at_Breakpoints_MW", "Value"] = ", ".join(f"{p:.4f}" for p in p_out)
        except Exception as exc:
            print(f"  Warning: could not update turbine breakpoints: {exc}")

    return df


# ---------------------------------------------------------------------------
# Core: Run optimization for a single plant + price scenario
# ---------------------------------------------------------------------------
def run_optimization(plant_params: dict, price_factor: float,
                     opt_output_dir: Path, logger: logging.Logger) -> Path | None:
    """
    Run the MILP optimization for *plant_params* after scaling energy prices by
    *(1 + price_factor)*.  Returns the path to the saved hourly CSV, or None.
    """
    iso = plant_params["iso_region"]
    plant_name = plant_params["plant_name"]
    generator_id = plant_params["generator_id"]
    remaining_years = plant_params["remaining_years"]

    label = f"{plant_name}_{generator_id}_{iso}_{int(remaining_years)}"
    logger.info(f"[OPT] Starting: {label}  (price_factor={price_factor:+.0%})")

    # 1. Load hourly data
    hourly_data = load_hourly_data(iso, base_dir=str(INPUT_HOURLY_DIR))
    if hourly_data is None:
        logger.error(f"[OPT] Failed to load hourly data for ISO {iso}")
        return None

    # 2. Adjust energy prices
    df_price = hourly_data["df_price_hourly"].copy()
    df_price["Price ($/MWh)"] = df_price["Price ($/MWh)"] * (1.0 + price_factor)
    hourly_data["df_price_hourly"] = df_price

    # 3. Adjust system parameters for plant-specific values + remaining life
    df_system = hourly_data.get("df_system")
    if df_system is None:
        logger.error("[OPT] df_system not found in hourly data")
        return None
    hourly_data["df_system"] = adjust_system_params(df_system, plant_params)

    # 4. Build model
    model = create_model(hourly_data, iso, simulate_dispatch=True)
    if model is None:
        logger.error(f"[OPT] create_model returned None for {label}")
        return None

    # 5. Select solver
    solver = None
    solver_name = None
    for candidate in ["gurobi", "cplex", "glpk", "cbc"]:
        try:
            _s = SolverFactory(candidate)
            solver = _s
            solver_name = candidate
            break
        except Exception:
            pass
    if solver is None:
        logger.error("[OPT] No solver available")
        return None
    logger.info(f"[OPT] Using solver: {solver_name}")

    solver_options: dict = {}
    if solver_name == "gurobi":
        log_dir = opt_output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        solver_log = str(log_dir / f"{label}_solver.log")
        solver_options = {"MIPGap": 0.0005, "LogFile": solver_log}

    # 6. Solve
    try:
        result = solver.solve(model, tee=False, options=solver_options)
    except Exception as exc:
        logger.error(f"[OPT] Solver raised exception: {exc}")
        return None

    status = result.solver.status
    condition = result.solver.termination_condition

    if status != SolverStatus.ok or condition not in (
        TerminationCondition.optimal, TerminationCondition.feasible
    ):
        logger.error(f"[OPT] Solver failed — status={status}, condition={condition}")
        return None

    logger.info(f"[OPT] Solved successfully — condition={condition}")

    # 7. Extract and save results
    results_df, _ = extract_results(model, target_iso=iso)
    opt_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = opt_output_dir / f"{label}_hourly_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"[OPT] Saved: {csv_path}")

    return csv_path


# ---------------------------------------------------------------------------
# Core: Run TEA for a single plant result CSV
# ---------------------------------------------------------------------------
def run_tea(csv_path: Path, plant_params: dict,
            tea_output_dir: Path, logger: logging.Logger) -> bool:
    """
    Run TEA on the hourly results CSV produced by the optimizer.
    Returns True on success.
    """
    iso = plant_params["iso_region"]
    plant_name = plant_params["plant_name"]
    generator_id = plant_params["generator_id"]
    remaining_years = plant_params["remaining_years"]
    label = f"{plant_name}_{generator_id}_{iso}_{int(remaining_years)}"

    logger.info(f"[TEA] Starting: {label}")

    if not csv_path.exists():
        logger.error(f"[TEA] Input CSV not found: {csv_path}")
        return False

    # Plant-specific params passed to TEA engine
    plant_specific_params = {
        "thermal_capacity_mwt": plant_params["thermal_capacity_mwt"],
        "nameplate_capacity_mw": plant_params["nameplate_capacity_mw"],
        "thermal_efficiency": plant_params["thermal_efficiency"],
        "pTurbine_max_MW": plant_params["nameplate_capacity_mw"],
        "qSteam_Total_MWth": plant_params["thermal_capacity_mwt"],
    }

    # Enhance CSV with plant-specific columns (required by TEA engine)
    enhanced_csv = csv_path.parent / f"enhanced_{csv_path.name}"
    try:
        df_res = pd.read_csv(csv_path)
        if "Turbine_Capacity_MW" not in df_res.columns:
            df_res["Turbine_Capacity_MW"] = plant_params["nameplate_capacity_mw"]
        if "Thermal_Capacity_MWt" not in df_res.columns:
            df_res["Thermal_Capacity_MWt"] = plant_params["thermal_capacity_mwt"]
        if "Thermal_Efficiency" not in df_res.columns:
            df_res["Thermal_Efficiency"] = plant_params["thermal_efficiency"]
        df_res.to_csv(enhanced_csv, index=False)
        actual_input = enhanced_csv
    except Exception as exc:
        logger.warning(f"[TEA] CSV enhancement failed ({exc}), using original file")
        actual_input = csv_path

    reactor_output_dir = tea_output_dir / label
    reactor_output_dir.mkdir(parents=True, exist_ok=True)

    config_overrides = {
        "disable_plotting": True,
        "analysis_mode": "sa_price_sensitivity",
    }

    try:
        result = run_complete_tea_analysis(
            target_iso=iso,
            input_hourly_results_file=actual_input,
            output_dir=reactor_output_dir,
            plant_report_title=label,
            input_sys_data_dir=INPUT_HOURLY_DIR,
            plant_specific_params=plant_specific_params,
            enable_greenfield=True,
            enable_incremental=True,
            config_overrides=config_overrides,
            analysis_type="reactor-specific",
            case_type="case1_existing_retrofit",
        )
        success = bool(result)
    except Exception as exc:
        logger.error(f"[TEA] run_complete_tea_analysis raised: {exc}")
        success = False
    finally:
        # Clean up enhanced CSV
        if enhanced_csv.exists():
            try:
                enhanced_csv.unlink()
            except Exception:
                pass

    if success:
        logger.info(f"[TEA] Completed: {reactor_output_dir}")
    else:
        logger.error(f"[TEA] Failed for {label}")

    return success


# ---------------------------------------------------------------------------
# Plant data loader
# ---------------------------------------------------------------------------
def load_npp_data(npp_file: Path) -> pd.DataFrame:
    df = pd.read_csv(npp_file)
    df = df[df["ISO"].notna() & (df["ISO"] != "None")]
    for col in ["Licensed Power (MWt)", "Nameplate Capacity (MW)",
                "Summer Capacity (MW)", "Winter Capacity (MW)",
                "Minimum Load (MW)", "remaining"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    df["Thermal_Efficiency"] = df["Nameplate Capacity (MW)"] / df["Licensed Power (MWt)"]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_SA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    main_log_path = LOG_SA_BASE_DIR / f"sa_price_sensitivity_{timestamp}.log"
    logger = setup_logger(main_log_path, name="sa_price_main")

    print("=" * 70)
    print("  Hourly Price Sensitivity Analysis  (Opt + TEA Pipeline)")
    print("=" * 70)
    print(f"Scenarios : {list(PRICE_SCENARIOS.keys())}")
    print(f"OPT output: {OPT_SA_BASE_DIR}")
    print(f"TEA output: {TEA_SA_BASE_DIR}")
    print(f"Log file  : {main_log_path}\n")

    logger.info("Price sensitivity analysis started")
    logger.info(f"Scenarios: {PRICE_SCENARIOS}")

    # Load plant data
    if not NPP_INFO_FILE.exists():
        logger.error(f"NPP info file not found: {NPP_INFO_FILE}")
        sys.exit(1)

    npp_df = load_npp_data(NPP_INFO_FILE)
    plants = [
        row for _, row in npp_df.iterrows()
        if row.get("remaining", 0) >= MIN_REMAINING_YEARS
    ]
    print(f"Plants to process: {len(plants)}  (remaining >= {MIN_REMAINING_YEARS} yrs)\n")
    logger.info(f"Loaded {len(plants)} plants with remaining >= {MIN_REMAINING_YEARS} yrs")

    # Summary counters
    total_opt_ok = 0
    total_opt_fail = 0
    total_tea_ok = 0
    total_tea_fail = 0

    for scenario_name, price_factor in PRICE_SCENARIOS.items():
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {scenario_name}  ({price_factor:+.0%})")
        print(f"{'─' * 70}")
        logger.info(f"===== Scenario: {scenario_name} ({price_factor:+.0%}) =====")

        opt_out_dir = OPT_SA_BASE_DIR / scenario_name
        tea_out_dir = TEA_SA_BASE_DIR / scenario_name
        opt_out_dir.mkdir(parents=True, exist_ok=True)
        tea_out_dir.mkdir(parents=True, exist_ok=True)

        scenario_opt_ok = 0
        scenario_opt_fail = 0
        scenario_tea_ok = 0
        scenario_tea_fail = 0

        for idx, row in enumerate(plants, 1):
            plant_params = {
                "plant_id": f"{row['Plant Code']}_{row['Generator ID']}",
                "plant_name": row["Plant Name"],
                "generator_id": int(row["Generator ID"]),
                "iso_region": row["ISO"],
                "thermal_capacity_mwt": float(row["Licensed Power (MWt)"]),
                "nameplate_capacity_mw": float(row["Nameplate Capacity (MW)"]),
                "min_load_mw": float(row.get("Minimum Load (MW)", 0) or 0),
                "thermal_efficiency": float(row["Thermal_Efficiency"]),
                "remaining_years": float(row["remaining"]),
            }

            label = (f"{plant_params['plant_name']}_{plant_params['generator_id']}_"
                     f"{plant_params['iso_region']}_{int(plant_params['remaining_years'])}")
            print(f"  [{idx}/{len(plants)}] {label}")

            # ----- Optimization -----
            csv_path = run_optimization(
                plant_params=plant_params,
                price_factor=price_factor,
                opt_output_dir=opt_out_dir,
                logger=logger,
            )

            if csv_path is None:
                print(f"    OPT: FAILED")
                logger.error(f"Opt failed for {label} / {scenario_name}")
                scenario_opt_fail += 1
                total_opt_fail += 1
                continue

            print(f"    OPT: OK  -> {csv_path.name}")
            scenario_opt_ok += 1
            total_opt_ok += 1

            # ----- TEA -----
            tea_ok = run_tea(
                csv_path=csv_path,
                plant_params=plant_params,
                tea_output_dir=tea_out_dir,
                logger=logger,
            )

            if tea_ok:
                print(f"    TEA: OK")
                scenario_tea_ok += 1
                total_tea_ok += 1
            else:
                print(f"    TEA: FAILED")
                logger.error(f"TEA failed for {label} / {scenario_name}")
                scenario_tea_fail += 1
                total_tea_fail += 1

        logger.info(
            f"Scenario {scenario_name}: OPT {scenario_opt_ok} OK / {scenario_opt_fail} FAIL | "
            f"TEA {scenario_tea_ok} OK / {scenario_tea_fail} FAIL"
        )
        print(f"\n  Scenario summary — OPT: {scenario_opt_ok} ok, {scenario_opt_fail} failed  |"
              f"  TEA: {scenario_tea_ok} ok, {scenario_tea_fail} failed")

    # Final summary
    print(f"\n{'=' * 70}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  OPT  — Success: {total_opt_ok}  |  Failed: {total_opt_fail}")
    print(f"  TEA  — Success: {total_tea_ok}  |  Failed: {total_tea_fail}")
    print(f"  OPT results  : {OPT_SA_BASE_DIR}")
    print(f"  TEA results  : {TEA_SA_BASE_DIR}")
    print(f"  Log          : {main_log_path}")
    print("=" * 70)

    logger.info("Price sensitivity analysis completed")
    logger.info(f"OPT: {total_opt_ok} ok, {total_opt_fail} failed")
    logger.info(f"TEA: {total_tea_ok} ok, {total_tea_fail} failed")


if __name__ == "__main__":
    main()
