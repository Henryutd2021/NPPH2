"""
TEA CS1 Sensitivity Analysis Runner with total_fixed_costs_per_mw_year Parameter
"""
import os
import sys
import re
import time
import threading
from pathlib import Path
import shutil
import pandas as pd
from contextlib import contextmanager
import importlib.util
import argparse

# Add the project root to sys.path FIRST - before any src imports
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add src/logging to the path for unified logging
src_logging_path = Path(__file__).parent.parent / "src" / "logging"
sys.path.append(str(src_logging_path))

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Base directories
BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
CS1_OUTPUT_DIR = BASE_OUTPUT_DIR / 'opt' / 'cs1'
CS1_LOG_DIR = BASE_OUTPUT_DIR / 'logs' / 'cs1'

# Sensitivity analysis specific directories
SENSITIVITY_BASE_DIR = BASE_OUTPUT_DIR / 'tea' / 'cs1_sensitivity'
SENSITIVITY_LOG_DIR = CS1_LOG_DIR / 'sensitivity'

# Create base directories
os.makedirs(SENSITIVITY_BASE_DIR, exist_ok=True)
os.makedirs(SENSITIVITY_LOG_DIR, exist_ok=True)

# Enhanced filename pattern for input CSVs
FILENAME_PATTERN = re.compile(
    r'^(?:enhanced_)?(.*?)_(.*?)_(.*?)_(\d+)_hourly_results\.csv$')

# Sensitivity analysis parameters
SENSITIVITY_PARAMETERS = {
    'total_fixed_costs_per_mw_year': [170000, 200000, 260000, 290000, 320000]
}


def setup_sensitivity_logger(parameter_value):
    """Setup logger for sensitivity analysis with specific parameter value"""
    import logging
    from datetime import datetime

    # Create parameter-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = SENSITIVITY_LOG_DIR / \
        f"tea_sensitivity_{parameter_value}_{timestamp}.log"

    # Setup logger
    logger = logging.getLogger(f'tea_sensitivity_{parameter_value}')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(
        f"TEA Sensitivity Analysis started for parameter value: {parameter_value}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Input directory: {CS1_OUTPUT_DIR}")
    logger.info(f"Output directory: {SENSITIVITY_BASE_DIR}")

    return logger, log_file


@contextmanager
def enhanced_tea_logging_simple(logger):
    """Simple logging context manager for TEA analysis"""
    logger.info("🔧 Enhanced TEA logging activated (No plotting)")
    try:
        yield logger
    finally:
        logger.info("🔄 Enhanced TEA logging cleanup completed")


def extract_plant_parameters(plant_name: str, generator_id: str, logger):
    """Extract plant-specific parameters with enhanced logging"""
    logger.info(f"Looking up {plant_name} Unit {generator_id}")

    npp_data_file = Path(__file__).parent.parent / \
        "input" / "hourly_data" / "NPPs info.csv"
    plant_specific_params = {}

    if not npp_data_file.exists():
        logger.warning(f"NPPs info.csv file not found at {npp_data_file}")
        return plant_specific_params

    try:
        npp_df = pd.read_csv(npp_data_file)
        logger.info(f"✅ Loaded NPP database with {len(npp_df)} records")

        # Find the matching plant record
        plant_row = npp_df[
            (npp_df["Plant Name"] == plant_name) &
            (npp_df["Generator ID"] == int(generator_id))
        ]

        if not plant_row.empty:
            plant_data = plant_row.iloc[0]

            # Extract thermal capacity
            try:
                thermal_capacity_raw = str(plant_data["Licensed Power (MWt)"])
                thermal_capacity_mwt = float(
                    thermal_capacity_raw.replace(",", ""))
            except (ValueError, KeyError) as e:
                logger.warning(f"Using default thermal capacity: {e}")
                thermal_capacity_mwt = 1000.0

            # Extract nameplate capacity
            try:
                nameplate_capacity_raw = str(
                    plant_data["Nameplate Capacity (MW)"])
                nameplate_capacity_mw = float(
                    nameplate_capacity_raw.replace(",", ""))
            except (ValueError, KeyError) as e:
                logger.warning(f"Using default nameplate capacity: {e}")
                nameplate_capacity_mw = 300.0

            # Calculate thermal efficiency
            thermal_efficiency = nameplate_capacity_mw / thermal_capacity_mwt

            plant_specific_params = {
                "thermal_capacity_mwt": thermal_capacity_mwt,
                "nameplate_capacity_mw": nameplate_capacity_mw,
                "thermal_efficiency": thermal_efficiency,
                "pTurbine_max_MW": nameplate_capacity_mw,
                "qSteam_Total_MWth": thermal_capacity_mwt
            }

            logger.info(f"Thermal Capacity: {thermal_capacity_mwt:,.1f} MWt")
            logger.info(f"Nameplate Capacity: {nameplate_capacity_mw:,.1f} MW")
            logger.info(f"Thermal Efficiency: {thermal_efficiency:.4f}")

        else:
            logger.warning(
                f"No record found for {plant_name} Unit {generator_id}")

    except Exception as e:
        logger.error(f"Error extracting plant-specific parameters: {e}")

    return plant_specific_params


def run_tea_for_file_sensitivity(csv_path: Path, plant_name: str, generator_id: str,
                                 iso_region: str, remaining_years_str: str,
                                 parameter_value: int, output_dir: Path):
    """
    Run TEA analysis with sensitivity parameter and NO PLOTTING
    """
    # Import required modules
    import sys
    from src.tea.tea_engine import run_complete_tea_analysis
    from src.logger_utils.enhanced_logging import ReactorLogSession
    from src.logger_utils.progress_indicators import TEAProgressIndicator

    # Create reactor-specific logger
    with ReactorLogSession(plant_name, generator_id, iso_region) as logger:
        logger.info(f"🔬 Starting TEA sensitivity analysis (NO PLOTTING)")
        logger.info(f"   📁 File: {csv_path.name}")
        logger.info(f"   🏭 Plant: {plant_name}")
        logger.info(f"   🔌 Generator: {generator_id}")
        logger.info(f"   🌍 Region: {iso_region}")
        logger.info(f"   ⏰ Years: {remaining_years_str}")
        logger.info(
            f"   📊 Parameter: total_fixed_costs_per_mw_year = {parameter_value}")

        progress = TEAProgressIndicator(
            f"TEA Sensitivity for {logger.reactor_name}", logger)
        progress.start()

        try:
            # Validate input file
            if not csv_path.exists():
                logger.error(f"Input file not found: {csv_path}")
                raise FileNotFoundError(f"Input file not found: {csv_path}")

            logger.info(
                f"Input file size: {csv_path.stat().st_size / 1024:.1f} KB")

            # Extract plant parameters
            plant_specific_params = extract_plant_parameters(
                plant_name, generator_id, logger)

            # Enhance CSV file if plant parameters are available
            actual_input_file = csv_path
            enhanced_csv_path = None

            if plant_specific_params:
                logger.info("Adding plant-specific parameters to input file")
                try:
                    results_df = pd.read_csv(csv_path)

                    # Add plant-specific columns
                    if "Turbine_Capacity_MW" not in results_df.columns:
                        results_df["Turbine_Capacity_MW"] = plant_specific_params["nameplate_capacity_mw"]

                    if "Thermal_Capacity_MWt" not in results_df.columns:
                        results_df["Thermal_Capacity_MWt"] = plant_specific_params["thermal_capacity_mwt"]

                    if "Thermal_Efficiency" not in results_df.columns:
                        results_df["Thermal_Efficiency"] = plant_specific_params["thermal_efficiency"]

                    # Save enhanced file
                    enhanced_csv_path = csv_path.parent / \
                        f"enhanced_{csv_path.name}"
                    results_df.to_csv(enhanced_csv_path, index=False)
                    actual_input_file = enhanced_csv_path

                    logger.info(
                        f"Enhanced CSV with {len(plant_specific_params)} columns")

                except Exception as e:
                    logger.warning(f"CSV enhancement failed: {e}")
                    actual_input_file = csv_path

            # Create reactor output directory within parameter-specific directory
            reactor_output_dir = output_dir / \
                f"{plant_name}_{generator_id}_{iso_region}_{remaining_years_str}"
            reactor_output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare plant report title
            plant_report_title = f"{plant_name}_{generator_id}_{iso_region}_{remaining_years_str}_sensitivity_{parameter_value}"

            # Prepare input sys data directory
            project_root = Path(__file__).parent.parent
            input_sys_data_dir = project_root / "input" / "hourly_data"

            # Create config overrides for sensitivity analysis
            config_overrides = {
                'total_fixed_costs_per_mw_year': parameter_value,
                'disable_plotting': True,  # Disable all plotting functionality
                'sensitivity_analysis': True,
                'sensitivity_parameter': 'total_fixed_costs_per_mw_year',
                'sensitivity_value': parameter_value
            }

            # Use enhanced TEA logging context (no plotting)
            with enhanced_tea_logging_simple(logger):
                logger.info("Starting core TEA calculations (NO PLOTTING)")

                # Run TEA analysis using the engine with config overrides
                main_result = run_complete_tea_analysis(
                    target_iso=iso_region,
                    input_hourly_results_file=actual_input_file,
                    output_dir=reactor_output_dir,
                    plant_report_title=plant_report_title,
                    input_sys_data_dir=input_sys_data_dir,
                    plant_specific_params=plant_specific_params,
                    enable_greenfield=True,
                    enable_incremental=True,
                    config_overrides=config_overrides,
                    analysis_type="sensitivity_analysis",
                    log_dir=SENSITIVITY_LOG_DIR,
                    case_type="case1_existing_retrofit"
                )

                logger.info("Core TEA calculations completed")

            # Clean up enhanced CSV file
            if enhanced_csv_path and enhanced_csv_path.exists():
                try:
                    enhanced_csv_path.unlink()
                    logger.debug("🧹 Cleaned up enhanced CSV file")
                except:
                    pass

            # Validate results
            if not main_result:
                logger.error(
                    f"TEA sensitivity analysis failed for {plant_name}")
                return False

            # Check output files (expecting text reports only, no plots)
            if reactor_output_dir.exists():
                logger.info("Checking generated output files")

                expected_files = [
                    f"{iso_region}_TEA_Summary_Report.txt",
                    f"{iso_region}_Cash_Flow_Analysis.txt",
                    f"{iso_region}_Financial_Metrics.txt"
                ]

                found_files = []
                for file_name in expected_files:
                    file_path = reactor_output_dir / file_name
                    if file_path.exists():
                        found_files.append(file_name)
                        logger.debug(f"   📄 Found: {file_name}")

                logger.info(f"Generated {len(found_files)} reports")

                if found_files:
                    logger.info(
                        f"✅ TEA sensitivity analysis completed successfully!")
                    logger.info(f"📂 Results saved to: {reactor_output_dir}")
                    return True
                else:
                    logger.warning("⚠️  No expected output files found")
                    return False
            else:
                logger.error(
                    f"Expected results directory not found: {reactor_output_dir}")
                return False

        except Exception as e:
            logger.error(f"Error during TEA sensitivity analysis: {str(e)}")
            return False

        finally:
            progress.stop()


def run_sensitivity_analysis(parameter_value: int):
    """Run sensitivity analysis for a specific parameter value"""

    # Setup parameter-specific logger
    logger, log_file = setup_sensitivity_logger(parameter_value)

    # Create parameter-specific output directory
    param_output_dir = SENSITIVITY_BASE_DIR / f"fixed_costs_{parameter_value}"
    param_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Starting sensitivity analysis for total_fixed_costs_per_mw_year = {parameter_value}")
    logger.info(f"Output directory: {param_output_dir}")

    # Find all input files (only original files, not enhanced ones)
    all_files = list(CS1_OUTPUT_DIR.glob("*_hourly_results.csv"))
    files_to_process = [
        f for f in all_files if not f.name.startswith("enhanced_")]

    if not files_to_process:
        error_msg = f"No files found in {CS1_OUTPUT_DIR} matching pattern '*_hourly_results.csv'"
        logger.error(error_msg)
        return False

    # Clean up any existing enhanced files
    enhanced_files = [f for f in all_files if f.name.startswith("enhanced_")]
    for ef in enhanced_files:
        try:
            ef.unlink()
            logger.debug(f"Cleaned up existing enhanced file: {ef.name}")
        except:
            pass

    logger.info(f"Found {len(files_to_process)} reactor files to process")

    successful_analyses = []
    failed_analyses = []

    for i, file_path in enumerate(files_to_process, 1):
        logger.info(
            f"Processing file {i}/{len(files_to_process)}: {file_path.name}")

        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            logger.warning(
                f"Skipping file with unexpected name: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Invalid filename pattern"))
            continue

        plant_name, generator_id, iso_region, remaining_years_str = match.groups()

        if not all([plant_name, generator_id, iso_region, remaining_years_str]):
            logger.warning(
                f"Skipping file due to missing parts: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Missing filename components"))
            continue

        try:
            int(remaining_years_str)
        except ValueError:
            logger.warning(
                f"Skipping file, invalid remaining_years: {remaining_years_str}")
            failed_analyses.append(
                (file_path.name, f"Invalid remaining years: {remaining_years_str}"))
            continue

        # Run sensitivity analysis
        success = run_tea_for_file_sensitivity(
            file_path, plant_name, generator_id, iso_region, remaining_years_str,
            parameter_value, param_output_dir
        )

        if success:
            successful_analyses.append(file_path.name)
            logger.info(
                f"Successfully completed sensitivity analysis for {plant_name}")
        else:
            failed_analyses.append(
                (file_path.name, "TEA sensitivity analysis failed"))
            logger.error(f"Failed sensitivity analysis for {plant_name}")

    # Summary report
    logger.info("=" * 80)
    logger.info(
        f"TEA Sensitivity Analysis Summary for parameter value {parameter_value}")
    logger.info("=" * 80)
    logger.info(f"Successful analyses: {len(successful_analyses)}")
    logger.info(f"Failed analyses: {len(failed_analyses)}")
    logger.info(f"Total files processed: {len(files_to_process)}")

    if successful_analyses:
        logger.info("Successfully processed reactors:")
        for filename in successful_analyses:
            logger.info(f"   - {filename}")

    if failed_analyses:
        logger.warning("Failed analyses:")
        for filename, reason in failed_analyses:
            logger.warning(f"   - {filename}: {reason}")

    logger.info(f"All results stored in: {param_output_dir}")
    logger.info(f"Analysis for parameter value {parameter_value} completed!")

    return len(successful_analyses) > 0


def main():
    """Main function for sensitivity analysis"""
    parser = argparse.ArgumentParser(
        description='TEA CS1 Sensitivity Analysis')
    parser.add_argument('--parameter-value', type=int, required=True,
                        help='Value for total_fixed_costs_per_mw_year parameter')

    args = parser.parse_args()

    # Validate parameter value
    if args.parameter_value not in SENSITIVITY_PARAMETERS['total_fixed_costs_per_mw_year']:
        print(f"❌ Invalid parameter value: {args.parameter_value}")
        print(
            f"Valid values: {SENSITIVITY_PARAMETERS['total_fixed_costs_per_mw_year']}")
        return False

    print(f"\n🚀 Starting TEA Sensitivity Analysis")
    print(
        f"📊 Parameter: total_fixed_costs_per_mw_year = {args.parameter_value}")
    print(f"📂 Input: {CS1_OUTPUT_DIR}")
    print(f"📂 Output: {SENSITIVITY_BASE_DIR}")
    print(f"📋 Logs: {SENSITIVITY_LOG_DIR}")
    print(f"🚫 Plotting: DISABLED")

    # Run sensitivity analysis
    success = run_sensitivity_analysis(args.parameter_value)

    if success:
        print(f"\n✅ Sensitivity analysis completed successfully!")
        print(
            f"📂 Results: {SENSITIVITY_BASE_DIR}/fixed_costs_{args.parameter_value}")
    else:
        print(f"\n❌ Sensitivity analysis failed!")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
