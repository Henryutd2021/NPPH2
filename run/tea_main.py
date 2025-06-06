# runs/tea_main.py
"""
Technical Economic Analysis (TEA) script for the nuclear-hydrogen optimization framework.
This is the main entry point for ISO-level TEA analysis.
"""

from src.tea.tea_engine import run_complete_tea_analysis
import src.tea.config as config
import logging
import os
import sys
import traceback
from pathlib import Path
import numpy as np  # Keep for np.array, np.cumsum, etc.
import pandas as pd  # Keep for pd.Series, pd.isna etc.

# Add src directory to Python path for relative imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import src modules AFTER adding to path

# Global logger, will be initialized by TEA engine
logger = None


def main(
    target_iso_override: str = None,
    project_lifetime_override: int = None,
    construction_years_override: int = None,
    discount_rate_override: float = None,
    tax_rate_override: float = None,
    base_output_dir_override: Path = None,
    plant_report_title_override: str = None,
    input_hourly_results_file_override: Path = None,
    input_sys_data_file_path_override: Path = None,
    enable_battery_override: bool = None,
    run_incremental_analysis_override: bool = None,
    enable_nuclear_greenfield_analysis_override: bool = None
):
    """
    Main TEA analysis function for ISO-level analysis
    """
    # Determine target ISO
    current_target_iso = target_iso_override if target_iso_override else config.TARGET_ISO

    # Prepare configuration overrides
    config_overrides = {}
    if project_lifetime_override is not None:
        config_overrides['project_lifetime'] = project_lifetime_override
    if construction_years_override is not None:
        config_overrides['construction_years'] = construction_years_override
    if discount_rate_override is not None:
        config_overrides['discount_rate'] = discount_rate_override
    if tax_rate_override is not None:
        config_overrides['tax_rate'] = tax_rate_override
    if enable_battery_override is not None:
        config_overrides['enable_battery'] = enable_battery_override

    # Determine output directory
    output_dir = base_output_dir_override if base_output_dir_override else (
        project_root / "output" / "tea" / "iso")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine log directory
    log_dir = project_root / "output" / "logs" / "tea"

    # Determine input directory for sys_data_advanced.csv
    input_sys_data_dir = input_sys_data_file_path_override.parent if input_sys_data_file_path_override else (
        project_root / "input" / "hourly_data")

    # Determine results file path - ISO-level data
    input_hourly_results_file = input_hourly_results_file_override if input_hourly_results_file_override \
        else project_root / "output" / "opt" / "Results_Standardized" / f"{current_target_iso}_Hourly_Results_Comprehensive.csv"

    # Determine report title
    plant_report_title = plant_report_title_override if plant_report_title_override else current_target_iso

    # Determine analysis flags
    enable_greenfield = enable_nuclear_greenfield_analysis_override \
        if enable_nuclear_greenfield_analysis_override is not None \
        else config.NUCLEAR_INTEGRATED_CONFIG.get("enabled", False)

    enable_incremental = run_incremental_analysis_override \
        if run_incremental_analysis_override is not None \
        else True  # Default to True for ISO-level analysis

    # Run complete TEA analysis using the engine
    success = run_complete_tea_analysis(
        target_iso=current_target_iso,
        input_hourly_results_file=input_hourly_results_file,
        output_dir=output_dir,
        plant_report_title=plant_report_title,
        input_sys_data_dir=input_sys_data_dir,
        plant_specific_params=None,  # No plant-specific params for ISO-level
        enable_greenfield=enable_greenfield,
        enable_incremental=enable_incremental,
        config_overrides=config_overrides,
        analysis_type="ISO-level",
        log_dir=log_dir
    )

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run ISO-level TEA analysis for nuclear-hydrogen optimization')
    parser.add_argument('--iso', '--target-iso', type=str, default=None,
                        help='Target ISO region (e.g., PJM, ERCOT, SPP, etc.)')
    parser.add_argument('--project-lifetime', type=int, default=None,
                        help='Project lifetime in years')
    parser.add_argument('--construction-years', type=int, default=None,
                        help='Construction period in years')
    parser.add_argument('--discount-rate', type=float, default=None,
                        help='Discount rate (decimal, e.g., 0.08 for 8%%)')
    parser.add_argument('--tax-rate', type=float, default=None,
                        help='Tax rate (decimal, e.g., 0.21 for 21%%)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Base output directory for results')
    parser.add_argument('--input-file', type=Path, default=None,
                        help='Path to hourly results CSV file')
    parser.add_argument('--enable-battery', action='store_true',
                        help='Enable battery analysis')
    parser.add_argument('--disable-battery', action='store_true',
                        help='Disable battery analysis')
    parser.add_argument('--enable-greenfield', action='store_true',
                        help='Enable greenfield nuclear analysis')
    parser.add_argument('--disable-greenfield', action='store_true',
                        help='Disable greenfield nuclear analysis')
    parser.add_argument('--enable-incremental', action='store_true',
                        help='Enable incremental analysis')
    parser.add_argument('--disable-incremental', action='store_true',
                        help='Disable incremental analysis')

    args = parser.parse_args()

    # Handle conflicting arguments
    enable_battery_override = None
    if args.enable_battery and args.disable_battery:
        print("Error: Cannot both enable and disable battery. Choose one.")
        sys.exit(1)
    elif args.enable_battery:
        enable_battery_override = True
    elif args.disable_battery:
        enable_battery_override = False

    enable_greenfield_override = None
    if args.enable_greenfield and args.disable_greenfield:
        print("Error: Cannot both enable and disable greenfield analysis. Choose one.")
        sys.exit(1)
    elif args.enable_greenfield:
        enable_greenfield_override = True
    elif args.disable_greenfield:
        enable_greenfield_override = False

    run_incremental_override = None
    if args.enable_incremental and args.disable_incremental:
        print("Error: Cannot both enable and disable incremental analysis. Choose one.")
        sys.exit(1)
    elif args.enable_incremental:
        run_incremental_override = True
    elif args.disable_incremental:
        run_incremental_override = False

    try:
        main_success = main(
            target_iso_override=args.iso,
            project_lifetime_override=args.project_lifetime,
            construction_years_override=args.construction_years,
            discount_rate_override=args.discount_rate,
            tax_rate_override=args.tax_rate,
            base_output_dir_override=args.output_dir,
            input_hourly_results_file_override=args.input_file,
            enable_battery_override=enable_battery_override,
            run_incremental_analysis_override=run_incremental_override,
            enable_nuclear_greenfield_analysis_override=enable_greenfield_override
        )
        if not main_success:
            print("TEA analysis failed.")
            sys.exit(1)
        sys.exit(0)
    except Exception as e_main:
        print(f"CRITICAL: An unhandled error in TEA main: {e_main}")
        traceback.print_exc()
        sys.exit(2)
