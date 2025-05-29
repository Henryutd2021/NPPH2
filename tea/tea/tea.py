# runs/tea.py
"""
Technical Economic Analysis (TEA) script for the nuclear-hydrogen optimization framework.
This script performs comprehensive lifecycle analysis including:
- Capital and operational costs (with learning rate adjustments for CAPEX)
- Revenue streams from multiple sources
- Financial metrics (NPV, IRR, LCOH, etc.)
- Sensitivity analysis
- Visualization of results
"""

import logging  # Keep for basic logging if setup_logging fails
import math  # Keep as it's used in some calculations that might remain or be called
import os
import sys
import traceback
from datetime import datetime  # Keep for generate_report
from pathlib import Path

import numpy as np  # Keep for np.array, np.cumsum, etc.
import pandas as pd  # Keep for pd.Series, pd.isna etc.

# New imports for refactored structure
from . import config  # To access and update configuration variables
from .utils import setup_logging
from .data_loader import load_tea_sys_params, load_hourly_results
from .calculations import (
    calculate_annual_metrics,
    calculate_cash_flows,
    calculate_financial_metrics,
    calculate_lcoh_breakdown,
    calculate_incremental_metrics
)
from .nuclear_calculations import (
    calculate_greenfield_nuclear_hydrogen_system,
    calculate_lifecycle_comparison_analysis
)
from .reporting import plot_results, generate_report

# Global logger, will be initialized by setup_logging in main()
logger = None


def main(
    target_iso_override: str = None,
    project_lifetime_override: int = None,
    construction_years_override: int = None,
    discount_rate_override: float = None,
    tax_rate_override: float = None,
    base_output_dir_override: Path = None,
    plant_report_title_override: str = None,  # For generate_report
    input_hourly_results_file_override: Path = None,
    input_sys_data_file_path_override: Path = None,
    enable_battery_override: bool = None,
    run_incremental_analysis_override: bool = None,
    enable_nuclear_greenfield_analysis_override: bool = None
):
    """
    Main execution function for TEA analysis.
    Orchestrates the workflow using functions from other modules.
    """
    global logger

    current_target_iso = target_iso_override if target_iso_override else config.TARGET_ISO

    # Setup Logging
    # config.LOG_DIR should be defined in tea/config.py
    # It's assumed config.LOG_DIR is correctly initialized (e.g., SCRIPT_DIR_PATH.parent / "logs")
    # SCRIPT_DIR_PATH for config.py would be tea/config.py's location.
    # If tea.py is in runs/, then config.SCRIPT_DIR_PATH (in config.py) is tea/.
    # So config.LOG_DIR would be runs/logs effectively.
    # Ensure config.LOG_DIR is correctly defined relative to the project structure.
    # For now, assuming config.LOG_DIR is a valid Path object.
    if not hasattr(config, 'LOG_DIR') or not isinstance(config.LOG_DIR, Path):
        # Fallback or error if LOG_DIR isn't properly set up in config.py
        # This might happen if config.py's SCRIPT_DIR_PATH logic isn't robust
        # to where tea.py is executed from.
        # A more robust way might be to define project root and derive paths from there.
        # For this refactor, we assume config.LOG_DIR is correctly set.
        # If not, this call would fail or log to an unintended place.
        # Fallback to current_working_dir/logs
        fallback_log_dir = Path(".") / "logs"
        logger_instance = setup_logging(
            fallback_log_dir, f"tea_{current_target_iso}_fallback")
        logger_instance.warning(
            f"config.LOG_DIR not properly configured. Using fallback: {fallback_log_dir}")
    else:
        logger_instance = setup_logging(
            config.LOG_DIR, f"tea_{current_target_iso}")

    logger = logger_instance  # Assign to global logger
    logger.info("🚀 Starting TEA analysis...")

    # Apply overrides to config module variables if provided
    if project_lifetime_override is not None:
        config.PROJECT_LIFETIME_YEARS = project_lifetime_override
        logger.info(
            f"Overridden PROJECT_LIFETIME_YEARS to: {config.PROJECT_LIFETIME_YEARS}")
    if construction_years_override is not None:
        config.CONSTRUCTION_YEARS = construction_years_override
        logger.info(
            f"Overridden CONSTRUCTION_YEARS to: {config.CONSTRUCTION_YEARS}")
    if discount_rate_override is not None:
        config.DISCOUNT_RATE = discount_rate_override
        logger.info(f"Overridden DISCOUNT_RATE to: {config.DISCOUNT_RATE}")
    if tax_rate_override is not None:
        config.TAX_RATE = tax_rate_override
        logger.info(f"Overridden TAX_RATE to: {config.TAX_RATE}")
    if enable_battery_override is not None:
        # This should update the config module's variable
        config.ENABLE_BATTERY = enable_battery_override
        logger.info(f"Overridden ENABLE_BATTERY to: {config.ENABLE_BATTERY}")

    # Determine output directory
    tea_base_output_dir = base_output_dir_override if base_output_dir_override else config.BASE_OUTPUT_DIR_DEFAULT
    os.makedirs(tea_base_output_dir, exist_ok=True)
    tea_output_file = tea_base_output_dir / \
        f"{current_target_iso}_TEA_Summary_Report.txt"
    plot_output_dir = tea_base_output_dir / f"Plots_{current_target_iso}"
    os.makedirs(plot_output_dir, exist_ok=True)
    logger.debug(
        f"Output paths: Report: {tea_output_file}, Plots: {plot_output_dir}")

    # Determine input directory for sys_data_advanced.csv
    # If input_sys_data_file_path_override is provided, use its parent. Otherwise, use default from config.
    input_dir_for_sys_params = input_sys_data_file_path_override.parent if input_sys_data_file_path_override else config.BASE_INPUT_DIR_DEFAULT

    (tea_sys_params,
     loaded_project_lifetime,
     loaded_discount_rate,
     loaded_construction_years,
     loaded_tax_rate,
     loaded_om_components,
     loaded_nuclear_config) = load_tea_sys_params(current_target_iso, input_dir_for_sys_params)

    # Update config variables with values returned from load_tea_sys_params
    config.PROJECT_LIFETIME_YEARS = loaded_project_lifetime
    config.DISCOUNT_RATE = loaded_discount_rate
    config.CONSTRUCTION_YEARS = loaded_construction_years
    config.TAX_RATE = loaded_tax_rate
    # For dictionaries, ensure we update them correctly.
    # If load_tea_sys_params returns modified copies, assign them back.
    config.OM_COMPONENTS.clear()
    config.OM_COMPONENTS.update(loaded_om_components)
    config.NUCLEAR_INTEGRATED_CONFIG.clear()
    config.NUCLEAR_INTEGRATED_CONFIG.update(loaded_nuclear_config)

    logger.info(
        f"Using Project Lifetime: {config.PROJECT_LIFETIME_YEARS} years")
    logger.info(f"Using Discount Rate: {config.DISCOUNT_RATE*100:.2f}%")
    logger.info(f"Using Construction Years: {config.CONSTRUCTION_YEARS} years")
    logger.info(f"Using Tax Rate: {config.TAX_RATE*100:.1f}%")
    logger.debug(
        f"Updated OM_COMPONENTS from load_tea_sys_params: {config.OM_COMPONENTS}")
    logger.debug(
        f"Updated NUCLEAR_INTEGRATED_CONFIG from load_tea_sys_params: {config.NUCLEAR_INTEGRATED_CONFIG}")

    # Determine results file path
    # config.SCRIPT_DIR_PATH in config.py is tea/
    # So config.SCRIPT_DIR_PATH.parent is the project root.
    results_file_path = input_hourly_results_file_override if input_hourly_results_file_override \
        else config.SCRIPT_DIR_PATH.parent / "output" / "Results_Standardized" / f"{current_target_iso}_Hourly_Results_Comprehensive.csv"

    logger.info(f"Loading results from: {results_file_path}")
    if not results_file_path.exists():
        logger.error(
            f"Optimization results file not found: {results_file_path}. Exiting TEA.")
        return False

    hourly_res_df = load_hourly_results(results_file_path)  # from data_loader
    if hourly_res_df is None:
        logger.error("Failed to load optimization results. Exiting TEA.")
        return False
    logger.info("Hourly results loaded successfully.")

    annual_metrics_results = calculate_annual_metrics(
        hourly_res_df, tea_sys_params)  # from calculations
    if annual_metrics_results is None:
        logger.error("Failed to calculate annual metrics. Exiting TEA.")
        return False
    logger.info("Annual metrics calculated.")

    optimized_caps = {
        "Electrolyzer_Capacity_MW": annual_metrics_results.get("Electrolyzer_Capacity_MW", 0),
        "H2_Storage_Capacity_kg": annual_metrics_results.get("H2_Storage_Capacity_kg", 0),
        "Battery_Capacity_MWh": annual_metrics_results.get("Battery_Capacity_MWh", 0),
        "Battery_Power_MW": annual_metrics_results.get("Battery_Power_MW", 0),
    }

    cash_flows_results = calculate_cash_flows(  # from calculations
        annual_metrics=annual_metrics_results,
        project_lifetime=config.PROJECT_LIFETIME_YEARS,
        construction_period=config.CONSTRUCTION_YEARS,
        h2_subsidy_value=float(tea_sys_params.get(
            "hydrogen_subsidy_value_usd_per_kg", 0.0)),
        h2_subsidy_duration=int(
            float(str(tea_sys_params.get("hydrogen_subsidy_duration_years", 10)))),
        capex_details=config.CAPEX_COMPONENTS,
        om_details=config.OM_COMPONENTS,
        replacement_details=config.REPLACEMENT_SCHEDULE,
        optimized_capacities=optimized_caps,
        tax_rate=config.TAX_RATE
    )
    logger.info("Cash flows calculated.")

    financial_metrics_results = calculate_financial_metrics(  # from calculations
        cash_flows_input=cash_flows_results,
        discount_rt=config.DISCOUNT_RATE,
        annual_h2_prod_kg=annual_metrics_results.get(
            "H2_Production_kg_annual", 0),
        project_lt=config.PROJECT_LIFETIME_YEARS,
        construction_p=config.CONSTRUCTION_YEARS,
    )

    total_capex_val = annual_metrics_results.get("total_capex", 0)
    npv_val = financial_metrics_results.get(
        "NPV_USD", np.nan)  # Ensure it can be nan
    if total_capex_val > 0 and npv_val is not None and not pd.isna(npv_val):
        financial_metrics_results["ROI"] = npv_val / total_capex_val
    else:
        financial_metrics_results["ROI"] = np.nan
    logger.info("Financial metrics calculated.")

    # Greenfield Nuclear-Hydrogen Analysis
    enable_greenfield = enable_nuclear_greenfield_analysis_override \
        if enable_nuclear_greenfield_analysis_override is not None \
        else config.NUCLEAR_INTEGRATED_CONFIG.get("enabled", False)

    if enable_greenfield:
        logger.info("Starting Greenfield Nuclear-Hydrogen Integrated Analysis.")

        # Use actual reactor capacity from annual metrics, not default values
        actual_nuclear_capacity = annual_metrics_results.get("Turbine_Capacity_MW",
                                                             float(tea_sys_params.get("nuclear_plant_capacity_MW",
                                                                                      config.NUCLEAR_INTEGRATED_CONFIG.get("nuclear_plant_capacity_mw", 1000))))

        logger.info(
            f"Using actual nuclear reactor capacity: {actual_nuclear_capacity:.1f} MW")

        greenfield_nuclear_metrics = calculate_greenfield_nuclear_hydrogen_system(  # from nuclear_calculations
            annual_metrics=annual_metrics_results,
            nuclear_capacity_mw=actual_nuclear_capacity,
            tea_sys_params=tea_sys_params,
            project_lifetime_config=config.NUCLEAR_INTEGRATED_CONFIG.get(
                "project_lifetime_years", 60),
            construction_period_config=config.NUCLEAR_INTEGRATED_CONFIG.get(
                "construction_years", 8),
            discount_rate_config=config.DISCOUNT_RATE,
            tax_rate_config=config.TAX_RATE,
            h2_capex_components_config=config.CAPEX_COMPONENTS,
            h2_om_components_config=config.OM_COMPONENTS,
            h2_replacement_schedule_config=config.REPLACEMENT_SCHEDULE
        )
        if greenfield_nuclear_metrics:
            annual_metrics_results["greenfield_nuclear_analysis"] = greenfield_nuclear_metrics
            logger.info("Greenfield nuclear-hydrogen analysis completed.")

            lifecycle_comparison_metrics = calculate_lifecycle_comparison_analysis(  # from nuclear_calculations
                annual_metrics=annual_metrics_results,
                nuclear_capacity_mw=actual_nuclear_capacity,
                tea_sys_params=tea_sys_params,
                discount_rate_config=config.DISCOUNT_RATE,
                tax_rate_config=config.TAX_RATE,
                h2_capex_components_config=config.CAPEX_COMPONENTS,
                h2_om_components_config=config.OM_COMPONENTS,
                h2_replacement_schedule_config=config.REPLACEMENT_SCHEDULE
            )
            if lifecycle_comparison_metrics:
                annual_metrics_results["lifecycle_comparison_analysis"] = lifecycle_comparison_metrics
                logger.info(
                    "Lifecycle comparison analysis (60yr vs 80yr) completed.")
        else:
            logger.warning(
                "Greenfield nuclear-hydrogen analysis failed or skipped.")
    else:
        logger.info(
            "Greenfield Nuclear-Hydrogen Integrated Analysis is disabled by configuration or override.")

    # LCOH Breakdown
    h2_prod_annual_lcoh = annual_metrics_results.get(
        "H2_Production_kg_annual", 0)
    if h2_prod_annual_lcoh > 0 and "capex_breakdown" in annual_metrics_results:
        lcoh_breakdown_results = calculate_lcoh_breakdown(  # from calculations
            annual_metrics=annual_metrics_results,
            capex_breakdown=annual_metrics_results["capex_breakdown"],
            project_lifetime=config.PROJECT_LIFETIME_YEARS,
            construction_period=config.CONSTRUCTION_YEARS,
            discount_rate=config.DISCOUNT_RATE,
            annual_h2_production_kg=h2_prod_annual_lcoh,
        )
        if lcoh_breakdown_results:
            annual_metrics_results["lcoh_breakdown_analysis"] = lcoh_breakdown_results
            financial_metrics_results["LCOH_USD_per_kg"] = lcoh_breakdown_results.get(
                "total_lcoh_usd_per_kg", np.nan)
            logger.info(
                f"LCOH breakdown calculated. Total LCOH: ${financial_metrics_results.get('LCOH_USD_per_kg', float('nan')):.3f}/kg")
    else:
        # Ensure it exists even if not calculated
        financial_metrics_results["LCOH_USD_per_kg"] = np.nan
        logger.warning(
            "Skipping LCOH breakdown: No H2 production or CAPEX breakdown missing.")

    # Incremental Analysis
    run_incremental = run_incremental_analysis_override \
        if run_incremental_analysis_override is not None \
        else str(tea_sys_params.get("enable_incremental_analysis", "True")).lower() in ['true', '1', 'yes']

    incremental_fin_metrics = None
    if run_incremental:
        logger.info("Starting incremental analysis.")
        inc_capex_keys = ["Electrolyzer", "H2_Storage", "NPP"]
        if config.ENABLE_BATTERY:
            inc_capex_keys.append("Battery")

        inc_capex = {k: v for k, v in config.CAPEX_COMPONENTS.items() if any(
            sub in k for sub in inc_capex_keys)}
        inc_om = {"Fixed_OM_General": config.OM_COMPONENTS.get(
            "Fixed_OM_General", {})}
        if config.ENABLE_BATTERY:
            inc_om["Fixed_OM_Battery"] = config.OM_COMPONENTS.get(
                "Fixed_OM_Battery", {})

        inc_repl_keys = ["Electrolyzer", "H2_Storage"]
        if config.ENABLE_BATTERY:
            inc_repl_keys.append("Battery")
        inc_repl = {k: v for k, v in config.REPLACEMENT_SCHEDULE.items() if any(
            sub in k for sub in inc_repl_keys)}

        baseline_revenue_val = float(tea_sys_params.get(
            "baseline_nuclear_annual_revenue_USD", 0.0))
        if baseline_revenue_val <= 0 and "Energy_Revenue" in annual_metrics_results:
            turbine_max_cap = float(tea_sys_params.get(
                "pTurbine_max_MW", annual_metrics_results.get("Turbine_Capacity_MW", 300)))
            avg_lmp = annual_metrics_results.get(
                "Avg_Electricity_Price_USD_per_MWh", 40)
            baseline_revenue_val = turbine_max_cap * config.HOURS_IN_YEAR * avg_lmp
            logger.info(
                f"Estimated baseline nuclear revenue: ${baseline_revenue_val:,.2f}")

        incremental_fin_metrics = calculate_incremental_metrics(  # from calculations
            optimized_cash_flows=cash_flows_results,
            baseline_annual_revenue=baseline_revenue_val,
            project_lifetime=config.PROJECT_LIFETIME_YEARS,
            construction_period=config.CONSTRUCTION_YEARS,
            discount_rt=config.DISCOUNT_RATE,
            tax_rt=config.TAX_RATE,
            annual_metrics_optimized=annual_metrics_results,
            capex_components_incremental=inc_capex,
            om_components_incremental=inc_om,
            replacement_schedule_incremental=inc_repl,
            h2_subsidy_val=float(tea_sys_params.get(
                "hydrogen_subsidy_value_usd_per_kg", 0.0)),
            h2_subsidy_yrs=int(
                float(str(tea_sys_params.get("hydrogen_subsidy_duration_years", 10)))),
            optimized_capacities_inc=optimized_caps,
        )
        logger.info("Incremental metrics calculated.")
    else:
        logger.info("Incremental analysis disabled.")

    # Plotting and Reporting
    report_title_to_use = plant_report_title_override if plant_report_title_override else current_target_iso

    logger.info("Generating plots...")
    plot_results(  # from reporting
        annual_metrics_data=annual_metrics_results,
        financial_metrics_data=financial_metrics_results,
        cash_flows_data=cash_flows_results,
        plot_dir=plot_output_dir,
        construction_p=config.CONSTRUCTION_YEARS,
        incremental_metrics_data=incremental_fin_metrics,
    )
    logger.info("Plots generated successfully.")

    logger.info("Generating final report...")
    generate_report(  # from reporting
        annual_metrics_rpt=annual_metrics_results,
        financial_metrics_rpt=financial_metrics_results,
        output_file_path=tea_output_file,
        target_iso_rpt=current_target_iso,
        plant_specific_title_rpt=report_title_to_use,
        capex_data=config.CAPEX_COMPONENTS,
        om_data=config.OM_COMPONENTS,
        replacement_data=config.REPLACEMENT_SCHEDULE,
        project_lt_rpt=config.PROJECT_LIFETIME_YEARS,
        construction_p_rpt=config.CONSTRUCTION_YEARS,
        discount_rt_rpt=config.DISCOUNT_RATE,
        tax_rt_rpt=config.TAX_RATE,
        incremental_metrics_rpt=incremental_fin_metrics,
    )
    logger.info("Report generation finished.")

    logger.info(
        f"--- Technical Economic Analysis completed successfully for {current_target_iso} ---")
    print(f"\nTEA Analysis completed for {current_target_iso}.")
    print(f"  Summary Report: {tea_output_file}")
    print(f"  Plots: {plot_output_dir}")

    # Find the file handler and print its baseFilename
    log_file_actual_path = "N/A"
    if logger and logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_actual_path = handler.baseFilename
                break
    print(f"  Log file: {log_file_actual_path}")
    return True


if __name__ == "__main__":
    try:
        main_success = main()  # Example: target_iso_override="SPP"
        if not main_success:
            # Logger might not be initialized if error is very early
            if logger:
                logger.error("TEA analysis failed in main execution.")
            else:
                print("TEA analysis failed. Logger not available.")
            sys.exit(1)
        sys.exit(0)
    except Exception as e_main:
        if logger:
            logger.critical(
                f"An unhandled error occurred in TEA main: {e_main}", exc_info=True)
        else:
            print(
                f"CRITICAL (no logger): An unhandled error in TEA main: {e_main}")
            traceback.print_exc()
        sys.exit(2)
