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

# Output directory for TEA results
TEA_RESULTS_DIR = Path(__file__).resolve(
).parent.parent / 'output' / 'tea' / 'cs1'
# Input directory for CS1 optimization results
CS1_OUTPUT_DIR = Path(__file__).resolve(
).parent.parent / 'output' / 'opt' / 'cs1'
# Log directory for CS1 TEA analysis
CS1_LOG_DIR = Path(__file__).resolve().parent.parent / 'logs' / 'cs1_tea'

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
    main_log_file = CS1_LOG_DIR / f"cs1_tea_main_{timestamp}.log"

    # Setup logger
    logger = logging.getLogger('cs1_tea_main')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - captures everything
    file_handler = logging.FileHandler(main_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"CS1 TEA Analysis started at {datetime.now()}")
    logger.info(f"Main log file: {main_log_file}")
    logger.info(f"Input directory: {CS1_OUTPUT_DIR}")
    logger.info(f"Output directory: {TEA_RESULTS_DIR}")
    logger.info(f"Detailed logs directory: logs/cs1/")

    return logger, main_log_file


@contextmanager
def enhanced_tea_logging_simple(logger):
    """Simple logging context manager for TEA analysis"""
    logger.info("🔧 Enhanced TEA logging activated")
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

    npp_data_file = Path(__file__).parent.parent / \
        "input" / "hourly_data" / "NPPs info.csv"
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


def enhanced_load_tea_sys_params_for_cs1(iso_target, input_base_dir, plant_specific_params, logger):
    """
    Enhanced parameter loading function with logging integration
    """
    # Import required modules
    import src.tea.data_loader as data_loader

    logger.log_phase_start("System Parameters Loading",
                           f"Target ISO: {iso_target}")

    # Call the original function to get base params
    original_params = data_loader.load_tea_sys_params(
        iso_target, input_base_dir)
    enhanced_params = original_params.copy()

    # Add plant-specific parameters
    if plant_specific_params:
        enhanced_params.update(plant_specific_params)
        logger.info(
            f"✅ Added {len(plant_specific_params)} plant-specific parameters")

    # Enhanced CS1 settings
    cs1_enhancements = {
        # Battery Configuration
        'enable_battery': True,
        'battery_hours': 6.0,
        'battery_round_trip_efficiency': 0.92,
        'battery_degrade_annual_percent': 2.0,
        'battery_soc_initial_percent': 50.0,
        'battery_soc_target_percent': 50.0,
        'battery_min_soc_percent': 10.0,
        'battery_max_soc_percent': 95.0,
        'enable_battery_degradation': True,
        'vom_battery_per_mwh_cycled': 2.0,

        # Economic Parameters
        'discount_rate': 0.08,
        'project_lifetime_years': 30,
        'construction_period_years': 3,
        'tax_rate': 0.21,
        'inflation_rate': 0.025,

        # Advanced Controls
        'optimization_tolerance': 1e-6,
        'max_optimization_iterations': 1000,
        'convergence_criteria': 'strict',
        'enable_advanced_control_logic': True,

        # Incremental Analysis
        'enable_incremental_analysis': True,
        'baseline_annual_revenue': 0.0,  # Will be calculated automatically if not provided
    }

    enhanced_params.update(cs1_enhancements)
    logger.log_calculation_result(
        "Enhanced Parameters Count", len(cs1_enhancements))

    logger.log_phase_complete("System Parameters Loading",
                              results_summary=f"Loaded {len(enhanced_params)} total parameters")
    return enhanced_params


def run_tea_for_file_enhanced(csv_path: Path, plant_name: str, generator_id: str,
                              iso_region: str, remaining_years_str: str):
    """
    Run enhanced TEA analysis with comprehensive logging
    """
    # Import required modules
    import src.tea.data_loader as data_loader
    import src.tea.tea as tea
    from src.logging.enhanced_logging import ReactorLogSession
    from src.logging.progress_indicators import TEAProgressIndicator

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

            # Enhanced parameter loading function
            def patched_load_tea_sys_params(iso_target, input_base_dir):
                return enhanced_load_tea_sys_params_for_cs1(
                    iso_target, input_base_dir, plant_specific_params, logger)

            # Store original function for restoration
            original_load_tea_sys_params = data_loader.load_tea_sys_params

            try:
                # Temporarily replace the function
                data_loader.load_tea_sys_params = patched_load_tea_sys_params

                # Enhance CSV file if plant parameters are available
                actual_input_file = csv_path
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

                # Use enhanced TEA logging context
                with enhanced_tea_logging_simple(logger):
                    logger.log_phase_start("Main TEA Analysis",
                                           "Starting core calculations")

                    # Run main TEA calculation
                    main_result = tea.main(
                        target_iso_override=iso_region,
                        plant_report_title_override=f"{plant_name}_{generator_id}_{iso_region}_{remaining_years_str}",
                        input_hourly_results_file_override=actual_input_file,
                        base_output_dir_override=reactor_output_dir,
                        enable_nuclear_greenfield_analysis_override=True,
                        run_incremental_analysis_override=True,  # Explicitly enable incremental analysis
                    )

                    logger.log_phase_complete("Main TEA Analysis")

            finally:
                # Always restore original function
                data_loader.load_tea_sys_params = original_load_tea_sys_params

                # Clean up enhanced CSV file
                if 'enhanced_csv_path' in locals() and enhanced_csv_path.exists():
                    try:
                        enhanced_csv_path.unlink()
                        logger.debug("🧹 Cleaned up enhanced CSV file")
                    except:
                        pass

            # Validate results
            if main_result is None:
                logger.error(f"❌ TEA analysis failed for {plant_name}")
                return False

            # Check output files
            if reactor_output_dir.exists():
                logger.log_phase_start("Results Validation",
                                       "Checking generated output files")

                key_files = [
                    f"{iso_region}_TEA_Summary_Report.txt",
                    f"{iso_region}_Cash_Flow_Chart.png",
                    f"{iso_region}_CAPEX_Breakdown_Chart.png",
                    f"{iso_region}_LCOH_Analysis_Dashboard.png",
                    f"{iso_region}_Nuclear_Greenfield_Analysis_Dashboard.png"
                ]

                found_files = []
                for file_name in key_files:
                    file_path = reactor_output_dir / file_name
                    if file_path.exists():
                        found_files.append(file_name)
                        logger.debug(f"   📄 Found: {file_name}")

                plots_dir = reactor_output_dir / f"Plots_{iso_region}"
                plots_count = 0
                if plots_dir.exists():
                    plots_count = len(list(plots_dir.glob("*.png")))

                logger.log_calculation_result(
                    "Generated Reports", len(found_files))
                logger.log_calculation_result("Generated Plots", plots_count)
                logger.log_phase_complete("Results Validation",
                                          results_summary=f"{len(found_files)} reports, {plots_count} plots")

                if found_files or plots_count > 0:
                    logger.info(f"✅ TEA analysis completed successfully!")
                    logger.info(f"📂 Results saved to: {reactor_output_dir}")
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
    print(f"📋 Detailed logs: logs/cs1/")

    files_to_process = list(CS1_OUTPUT_DIR.glob("*_hourly_results.csv"))
    if not files_to_process:
        error_msg = f"No files found in {CS1_OUTPUT_DIR} matching pattern '*_hourly_results.csv'"
        print(f"❌ {error_msg}")
        main_logger.error(error_msg)
        return

    print(f"📊 Found {len(files_to_process)} reactor files to process")
    main_logger.info(f"Found {len(files_to_process)} reactor files to process for enhanced TEA analysis")
    print(f"=" * 60)

    successful_analyses = []
    failed_analyses = []

    for i, file_path in enumerate(files_to_process, 1):
        # Minimal console output - just progress
        print(f"📍 [{i}/{len(files_to_process)}] {file_path.name}", end=" ... ")

        # Detailed logging to file
        main_logger.info(f"Processing file {i}/{len(files_to_process)}: {file_path.name}")

        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            print("⚠️  SKIPPED (invalid name)")
            main_logger.warning(f"Skipping file with unexpected name: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Invalid filename pattern"))
            continue

        plant_name, generator_id, iso_region, remaining_years_str = match.groups()

        if not all([plant_name, generator_id, iso_region, remaining_years_str]):
            print("⚠️  SKIPPED (missing components)")
            main_logger.warning(f"Skipping file due to missing parts in filename: {file_path.name}")
            failed_analyses.append(
                (file_path.name, "Missing filename components"))
            continue

        try:
            int(remaining_years_str)
        except ValueError:
            print("⚠️  SKIPPED (invalid years)")
            main_logger.warning(f"Skipping file, invalid remaining_years: {remaining_years_str}")
            failed_analyses.append(
                (file_path.name, f"Invalid remaining years: {remaining_years_str}"))
            continue

        # Run enhanced TEA analysis
        main_logger.info(f"Starting TEA analysis for {plant_name} Unit {generator_id}")
        success = run_tea_for_file_enhanced(file_path, plant_name, generator_id,
                                            iso_region, remaining_years_str)

        if success:
            successful_analyses.append(file_path.name)
            print("✅ SUCCESS")
            main_logger.info(f"Successfully completed enhanced TEA for {plant_name}")
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
    main_logger.info(f"All enhanced logs are stored in: logs/cs1/")
    main_logger.info("Enhanced CS1 TEA Analysis completed successfully!")


if __name__ == "__main__":
    main()
