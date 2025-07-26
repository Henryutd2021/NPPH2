"""
Enhanced CS1 TEA Analysis Runner with Standardized Logging and Naming
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

# Setup Python paths for importing src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from path_setup import setup_src_paths
setup_src_paths()

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Output directory for TEA results
TEA_RESULTS_DIR = Path(__file__).resolve(
).parent.parent.parent / 'output' / 'tea' / 'cs1'  # Updated path for new location
# Input directory for CS1 optimization results
CS1_OUTPUT_DIR = Path(__file__).resolve(
).parent.parent.parent / 'output' / 'opt' / 'cs1'  # Updated path for new location
# Log directory for CS1 TEA analysis
CS1_LOG_DIR = Path(__file__).resolve().parent.parent.parent / \
    'output' / 'logs' / 'cs1'  # Updated path for new location

os.makedirs(TEA_RESULTS_DIR, exist_ok=True)
os.makedirs(CS1_LOG_DIR, exist_ok=True)

# Enhanced filename pattern for input CSVs
FILENAME_PATTERN = re.compile(
    r'^(?:enhanced_)?(.*?)_(.*?)_(.*?)_(\d+)_hourly_results\.csv$')


def setup_main_logger():
    """Setup main logger for CS1 TEA analysis with minimal console output"""
    import logging
    from datetime import datetime

    # Create main log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = CS1_LOG_DIR / f"cs1_tea_batch_analysis_{timestamp}.log"

    # Setup logger
    logger = logging.getLogger('cs1_tea_main')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(
        main_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only important messages
    console_handler = logging.StreamHandler()
    # Only warnings and errors to console
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"CS1 TEA Analysis started at {datetime.now()}")
    logger.info(f"Main log file: {main_log_file}")
    logger.info(f"Input directory: {CS1_OUTPUT_DIR}")
    logger.info(f"Output directory: {TEA_RESULTS_DIR}")
    logger.info(f"Detailed logs directory: {CS1_LOG_DIR}/")

    return logger, main_log_file


@contextmanager
def enhanced_tea_logging_simple(logger):
    """Simple logging context manager for TEA analysis (NO PLOTTING)"""
    logger.info("🔧 Enhanced TEA logging activated (Plotting DISABLED)")
    try:
        yield logger
    finally:
        logger.info("🔄 Enhanced TEA logging cleanup completed")


def extract_plant_parameters(plant_name: str, generator_id: str, logger):
    """
    Extract plant-specific parameters with enhanced logging
    """
    logger.log_phase_start("Plant Parameter Extraction",
                           f"Looking up {plant_name} Unit {generator_id}")

    npp_data_file = Path(__file__).parent.parent.parent / \
        "input" / "hourly_data" / "NPPs info.csv"  # Updated path for new location
    plant_specific_params = {}

    if not npp_data_file.exists():
        logger.log_missing_data(
            component="plant_data",
            parameter="NPPs info.csv file",
            impact="high"
        )
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
                logger.log_missing_data(
                    component="thermal_capacity",
                    parameter="Licensed Power (MWt)",
                    fallback_value="1000.0 MW (default)",
                    impact="medium"
                )
                thermal_capacity_mwt = 1000.0

            # Extract nameplate capacity
            try:
                nameplate_capacity_raw = str(
                    plant_data["Nameplate Capacity (MW)"])
                nameplate_capacity_mw = float(
                    nameplate_capacity_raw.replace(",", ""))
            except (ValueError, KeyError) as e:
                logger.log_missing_data(
                    component="nameplate_capacity",
                    parameter="Nameplate Capacity (MW)",
                    fallback_value="300.0 MW (default)",
                    impact="medium"
                )
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

            logger.log_calculation_result(
                "Thermal Capacity", f"{thermal_capacity_mwt:,.1f}", "MWt")
            logger.log_calculation_result(
                "Nameplate Capacity", f"{nameplate_capacity_mw:,.1f}", "MW")
            logger.log_calculation_result(
                "Thermal Efficiency", f"{thermal_efficiency:.4f}")

        else:
            logger.log_missing_data(
                component="plant_database",
                parameter=f"Record for {plant_name} Unit {generator_id}",
                impact="high"
            )

    except Exception as e:
        logger.error(f"❌ Error extracting plant-specific parameters: {e}")
        logger.log_invalid_data(
            component="plant_data",
            parameter="NPPs info.csv processing",
            invalid_value=str(e),
            impact="high"
        )

    logger.log_phase_complete("Plant Parameter Extraction",
                              results_summary=f"Found {len(plant_specific_params)} parameters")
    return plant_specific_params


def run_tea_for_file_enhanced(csv_path: Path, plant_name: str, generator_id: str,
                              iso_region: str, remaining_years_str: str):
    """
    Run enhanced TEA analysis with comprehensive logging
    """
    # Import required modules
    import sys
    from src.tea.tea_engine import run_complete_tea_analysis
    from src.logger_utils.enhanced_logging import ReactorLogSession
    from src.logger_utils.progress_indicators import TEAProgressIndicator

    # Create reactor-specific logger
    with ReactorLogSession(plant_name, generator_id, iso_region) as logger:
        logger.info(f"🔬 Starting enhanced TEA analysis")
        logger.info(f"   📁 File: {csv_path.name}")
        logger.info(f"   🏭 Plant: {plant_name}")
        logger.info(f"   🔌 Generator: {generator_id}")
        logger.info(f"   🌍 Region: {iso_region}")
        logger.info(f"   ⏰ Years: {remaining_years_str}")

        progress = TEAProgressIndicator(
            f"Enhanced TEA for {logger.reactor_name}", logger)
        progress.start()

        try:
            # Validate input file
            if not csv_path.exists():
                logger.log_missing_data(
                    component="input_file",
                    parameter=str(csv_path),
                    impact="critical"
                )
                raise FileNotFoundError(f"Input file not found: {csv_path}")

            logger.log_calculation_result(
                "Input file size", f"{csv_path.stat().st_size / 1024:.1f}", "KB")

            # Extract plant parameters
            plant_specific_params = extract_plant_parameters(
                plant_name, generator_id, logger)

            # Enhance CSV file if plant parameters are available
            actual_input_file = csv_path
            enhanced_csv_path = None

            if plant_specific_params:
                logger.log_phase_start("CSV Enhancement",
                                       "Adding plant-specific parameters to input file")
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

                    logger.log_calculation_result(
                        "Enhanced CSV Columns", len(plant_specific_params))
                    logger.log_phase_complete("CSV Enhancement")

                except Exception as e:
                    logger.log_invalid_data(
                        component="csv_enhancement",
                        parameter="file_processing",
                        invalid_value=str(e),
                        impact="medium"
                    )
                    actual_input_file = csv_path

            # Create reactor output directory
            reactor_output_dir = TEA_RESULTS_DIR / \
                f"{plant_name}_{generator_id}_{iso_region}_{remaining_years_str}"
            reactor_output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare plant report title
            plant_report_title = f"{plant_name}_{generator_id}_{iso_region}_{remaining_years_str}"

            # Prepare input sys data directory
            # Updated path for new location
            project_root = Path(__file__).parent.parent.parent
            input_sys_data_dir = project_root / "input" / "hourly_data"

            # Create config overrides to disable plotting by default for CS1 analysis
            config_overrides = {
                'disable_plotting': True,  # Disable plotting for CS1 analysis
                'analysis_mode': 'cs1_standard'  # Mark as standard CS1 analysis
            }

            # Use enhanced TEA logging context
            with enhanced_tea_logging_simple(logger):
                logger.log_phase_start(
                    "Main TEA Analysis", "Starting core calculations (NO PLOTTING)")

                # Run TEA analysis using the engine
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
                    analysis_type="reactor-specific",
                    log_dir=CS1_LOG_DIR,
                    case_type="case1_existing_retrofit"  # CS1 is existing plant analysis
                )

                logger.log_phase_complete("Main TEA Analysis")

            # Clean up enhanced CSV file
            if enhanced_csv_path and enhanced_csv_path.exists():
                try:
                    enhanced_csv_path.unlink()
                    logger.debug("🧹 Cleaned up enhanced CSV file")
                except:
                    pass

            # Validate results
            if not main_result:
                logger.error(f"❌ TEA analysis failed for {plant_name}")
                return False

            # Check output files (text reports only, no plots expected)
            if reactor_output_dir.exists():
                logger.log_phase_start("Results Validation",
                                       "Checking generated output files (NO PLOTTING mode)")

                expected_report_files = [
                    f"{iso_region}_TEA_Summary_Report.txt",
                    f"{iso_region}_Comprehensive_TEA_Summary.txt"
                ]

                found_files = []
                for file_name in expected_report_files:
                    file_path = reactor_output_dir / file_name
                    if file_path.exists():
                        found_files.append(file_name)
                        logger.debug(f"   📄 Found: {file_name}")

                # Note: Plots directory should NOT exist since plotting is disabled
                plots_dir = reactor_output_dir / f"Plots_{iso_region}"
                if plots_dir.exists():
                    logger.warning(
                        f"⚠️  Unexpected plots directory found: {plots_dir}")
                else:
                    logger.debug(
                        "✅ No plots directory (as expected - plotting disabled)")

                logger.log_calculation_result(
                    "Generated Reports", len(found_files))
                # Always 0 in no-plotting mode
                logger.log_calculation_result("Generated Plots", 0)
                logger.log_phase_complete("Results Validation",
                                          results_summary=f"{len(found_files)} reports, 0 plots (plotting disabled)")

                if found_files:
                    logger.info(f"✅ TEA analysis completed successfully!")
                    logger.info(f"📂 Results saved to: {reactor_output_dir}")
                    logger.info(f"🚫 Plotting disabled - text reports only")
                    return True
                else:
                    logger.warning("⚠️  No expected output files found")
                    return False
            else:
                logger.error(
                    f"❌ Expected results directory not found: {reactor_output_dir}")
                return False

        except Exception as e:
            logger.error(f"❌ Error during TEA analysis: {str(e)}")
            return False

        finally:
            progress.stop()


def main():
    """Main function to run enhanced TEA analysis for all reactors"""
    # Setup main logger with minimal console output
    main_logger, main_log_file = setup_main_logger()

    # Only essential information to console
    print(f"\n🚀 Starting Enhanced CS1 TEA Analysis")
    print(f"📂 Input: {CS1_OUTPUT_DIR}")
    print(f"📂 Output: {TEA_RESULTS_DIR}")
    print(f"📋 Main log: {main_log_file}")
    print(f"📋 Detailed logs: {CS1_LOG_DIR}/")
    print(f"🚫 Plotting: DISABLED (text reports only)")

    # CRITICAL FIX: Only process original files, not enhanced ones
    all_files = list(CS1_OUTPUT_DIR.glob("*_hourly_results.csv"))
    files_to_process = [
        f for f in all_files if not f.name.startswith("enhanced_")]

    if not files_to_process:
        error_msg = f"No original files found in {CS1_OUTPUT_DIR} matching pattern '*_hourly_results.csv' (excluding enhanced_ files)"
        print(f"❌ {error_msg}")
        main_logger.error(error_msg)
        return

    # Log information about excluded enhanced files
    enhanced_files = [f for f in all_files if f.name.startswith("enhanced_")]
    if enhanced_files:
        main_logger.info(
            f"Found {len(enhanced_files)} enhanced files that will be skipped:")
        for ef in enhanced_files:
            main_logger.info(f"   - {ef.name} (will be recreated if needed)")
            # Clean up any existing enhanced files to avoid conflicts
            try:
                ef.unlink()
                main_logger.debug(
                    f"Cleaned up existing enhanced file: {ef.name}")
            except:
                pass

    print(f"📊 Found {len(files_to_process)} reactor files to process")
    main_logger.info(
        f"Found {len(files_to_process)} reactor files to process for enhanced TEA analysis")
    print(f"=" * 60)

    successful_analyses = []
    failed_analyses = []

    for i, file_path in enumerate(files_to_process, 1):
        # Minimal console output - just progress
        print(f"📍 [{i}/{len(files_to_process)}] {file_path.name}", end=" ... ")

        # Detailed logging to file
        main_logger.info(
            f"Processing file {i}/{len(files_to_process)}: {file_path.name}")

        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            print("⚠️  SKIPPED (invalid name)")
            main_logger.warning(
                f"Skipping file with unexpected name: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Invalid filename pattern"))
            continue

        plant_name, generator_id, iso_region, remaining_years_str = match.groups()

        if not all([plant_name, generator_id, iso_region, remaining_years_str]):
            print("⚠️  SKIPPED (missing components)")
            main_logger.warning(
                f"Skipping file due to missing parts in filename: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Missing filename components"))
            continue

        try:
            int(remaining_years_str)
        except ValueError:
            print("⚠️  SKIPPED (invalid years)")
            main_logger.warning(
                f"Skipping file, invalid remaining_years: {remaining_years_str}")
            failed_analyses.append(
                (file_path.name, f"Invalid remaining years: {remaining_years_str}"))
            continue

        # Run enhanced TEA analysis
        main_logger.info(
            f"Starting TEA analysis for {plant_name} Unit {generator_id}")
        success = run_tea_for_file_enhanced(file_path, plant_name, generator_id,
                                            iso_region, remaining_years_str)

        if success:
            successful_analyses.append(file_path.name)
            print("✅ SUCCESS")
            main_logger.info(
                f"Successfully completed enhanced TEA for {plant_name}")
        else:
            failed_analyses.append((file_path.name, "TEA analysis failed"))
            print("❌ FAILED")
            main_logger.error(f"Failed enhanced TEA analysis for {plant_name}")

    # Summary report - concise console output
    print(f"\n" + "=" * 60)
    print(f"🎯 CS1 TEA Analysis Summary")
    print(f"=" * 60)
    print(f"✅ Successful: {len(successful_analyses)}")
    print(f"❌ Failed: {len(failed_analyses)}")
    print(f"📊 Total: {len(files_to_process)}")

    # Detailed summary to log file
    main_logger.info("=" * 80)
    main_logger.info("Enhanced CS1 TEA Analysis Summary")
    main_logger.info("=" * 80)
    main_logger.info(f"Successful analyses: {len(successful_analyses)}")
    main_logger.info(f"Failed analyses: {len(failed_analyses)}")
    main_logger.info(f"Total files processed: {len(files_to_process)}")

    if successful_analyses:
        main_logger.info("Successfully processed reactors:")
        for filename in successful_analyses:
            main_logger.info(f"   - {filename}")

    if failed_analyses:
        main_logger.warning("Failed analyses:")
        for filename, reason in failed_analyses:
            main_logger.warning(f"   - {filename}: {reason}")

    print(f"\n📂 Results: {TEA_RESULTS_DIR}")
    print(f"📋 Logs: {main_log_file}")
    print(f"🎉 Analysis completed!")

    # Final log entries
    main_logger.info(f"All TEA results are stored in: {TEA_RESULTS_DIR}")
    main_logger.info(f"All enhanced logs are stored in: {CS1_LOG_DIR}/")
    main_logger.info("Enhanced CS1 TEA Analysis completed successfully!")


if __name__ == "__main__":
    main()
