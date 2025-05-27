import os
import re
import sys
from pathlib import Path
import shutil
import threading
import time

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Path to tea.py
TEA_SCRIPT_PATH = Path(__file__).parent / 'tea.py'
# Output directory for TEA results
TEA_RESULTS_DIR = Path(__file__).parent.parent / 'TEA_results' / 'cs1_tea'
# Directory containing optimization results
CS1_OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'cs1'

os.makedirs(TEA_RESULTS_DIR, exist_ok=True)

# Match filename format: {plant_name}_{generator_id}_{iso_region}_{remaining_years}_hourly_results.csv
FILENAME_PATTERN = re.compile(r'^(.*?)_(.*?)_(.*?)_(\d+)_hourly_results\.csv$')

# Import tea.py main function
sys.path.insert(0, str(Path(__file__).parent))
try:
    # Force fresh import of tea module
    if 'tea' in sys.modules:
        del sys.modules['tea']
    import tea

    # Verify that the required function exists
    if not hasattr(tea, 'load_tea_sys_params'):
        # Try reloading the module
        import importlib
        importlib.reload(tea)
        if not hasattr(tea, 'load_tea_sys_params'):
            print('Error: tea.load_tea_sys_params function not found after reload.')
            print(
                f"Available tea attributes: {[attr for attr in dir(tea) if not attr.startswith('_')]}")
            sys.exit(1)

    print("Successfully imported tea module with load_tea_sys_params function")
except ImportError as e:
    print(f'Error: tea.py not found or import failed: {e}')
    sys.exit(1)


class TEAProgressIndicator:
    """
    A simple progress indicator for TEA analysis process.
    """

    def __init__(self, description="Running TEA analysis"):
        self.description = description
        self.running = False
        self.thread = None
        self.start_time = None

    def _animate(self):
        """Internal method to run the animation in a separate thread."""
        if TQDM_AVAILABLE:
            # Use tqdm for a nice progress bar
            with tqdm(desc=self.description, unit="", bar_format="{desc}: {elapsed} | {rate_fmt}") as pbar:
                while self.running:
                    pbar.update(0)  # Keep the bar alive
                    time.sleep(0.1)
        else:
            # Fallback to simple spinner
            spinners = ['|', '/', '-', '\\']
            i = 0
            while self.running:
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(
                    f"\r{self.description}... {spinners[i % len(spinners)]} ({elapsed:.1f}s)", end="", flush=True)
                i += 1
                time.sleep(0.5)
            print()  # New line when done

    def start(self):
        """Start the progress indicator."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the progress indicator."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)

            # Show final completion message
            if self.start_time:
                elapsed = time.time() - self.start_time

                # Only show completion message if tqdm is not available
                # (tqdm will show its own completion message)
                if not TQDM_AVAILABLE:
                    print(
                        f"\r{self.description} completed in {elapsed:.1f}s" + " " * 20)


def run_tea_for_file(csv_path, plant_name, generator_id, iso_region, remaining_years):
    """
    Call the main process of tea.py, specifying the input file and output directory.
    """
    # Construct output directory for this plant
    output_dir = TEA_RESULTS_DIR / \
        f"{plant_name}_{generator_id}_{iso_region}_{remaining_years}"
    os.makedirs(output_dir, exist_ok=True)

    # Inject parameters into tea.py (if necessary, can be passed via env or arguments)
    # Here we assume tea.py supports specifying input/output paths, otherwise use monkey patch or modify tea.py directly
    # We call the main function directly and temporarily override path variables in tea.py
    import importlib
    import types
    import pandas as pd

    # Extract plant-specific parameters from filename and NPP data
    npp_data_file = Path(__file__).parent.parent / \
        "input" / "hourly_data" / "NPPs info.csv"
    plant_specific_params = {}

    if npp_data_file.exists():
        try:
            npp_df = pd.read_csv(npp_data_file)
            # Find the matching plant record
            plant_row = npp_df[
                (npp_df["Plant Name"] == plant_name) &
                (npp_df["Generator ID"] == int(generator_id))
            ]

            if not plant_row.empty:
                plant_data = plant_row.iloc[0]
                # Extract thermal capacity and nameplate capacity
                thermal_capacity_mwt = float(
                    str(plant_data["Licensed Power (MWt)"]).replace(",", ""))
                nameplate_capacity_mw = float(
                    str(plant_data["Nameplate Capacity (MW)"]).replace(",", ""))
                thermal_efficiency = nameplate_capacity_mw / thermal_capacity_mwt

                plant_specific_params = {
                    "thermal_capacity_mwt": thermal_capacity_mwt,
                    "nameplate_capacity_mw": nameplate_capacity_mw,
                    "thermal_efficiency": thermal_efficiency,
                    "pTurbine_max_MW": nameplate_capacity_mw,
                    "qSteam_Total_MWth": thermal_capacity_mwt
                }

        except Exception as e:
            print(f"Error extracting plant-specific parameters: {e}")

    # Backup original paths
    old_base_output = getattr(tea, 'BASE_OUTPUT_DIR_DEFAULT', None)
    old_script_dir = getattr(tea, 'SCRIPT_DIR_PATH', None)
    old_target_iso = getattr(tea, 'TARGET_ISO', None)
    old_project_lifetime = getattr(
        tea, 'PROJECT_LIFETIME_YEARS', None)  # Backup project lifetime

    # Convert remaining_years to integer
    remaining_years_int = int(remaining_years)

    # Temporarily override
    tea.BASE_OUTPUT_DIR_DEFAULT = output_dir
    tea.TARGET_ISO = iso_region
    tea.SCRIPT_DIR_PATH = Path(__file__).parent
    # Set project lifetime to remaining_years
    tea.PROJECT_LIFETIME_YEARS = remaining_years_int

    # Copy the csv to the location and filename expected by tea.py
    # tea.py defaults to reading output/Results_Standardized/{iso}_Hourly_Results_Comprehensive.csv
    # We copy the current csv to that location
    std_dir = Path(__file__).parent.parent / 'output' / 'Results_Standardized'
    os.makedirs(std_dir, exist_ok=True)
    std_csv = std_dir / f"{iso_region}_Hourly_Results_Comprehensive.csv"

    # **ENHANCEMENT: Add plant-specific parameters to the results file**
    if plant_specific_params:
        try:
            # Read the original results file
            results_df = pd.read_csv(csv_path)

            # Add plant-specific capacity columns if they don't exist
            if "Turbine_Capacity_MW" not in results_df.columns:
                results_df["Turbine_Capacity_MW"] = plant_specific_params["nameplate_capacity_mw"]

            if "Thermal_Capacity_MWt" not in results_df.columns:
                results_df["Thermal_Capacity_MWt"] = plant_specific_params["thermal_capacity_mwt"]

            if "Thermal_Efficiency" not in results_df.columns:
                results_df["Thermal_Efficiency"] = plant_specific_params["thermal_efficiency"]

            # Save the enhanced results file
            results_df.to_csv(std_csv, index=False)

        except Exception as e:
            print(
                f"Warning: Could not enhance results file with plant-specific parameters: {e}")
            # Fallback: just copy the original file
            shutil.copy(csv_path, std_csv)
    else:
        shutil.copy(csv_path, std_csv)

    plot_dir_final = output_dir / f"Plots_{iso_region}"
    os.makedirs(plot_dir_final, exist_ok=True)

    # **ENHANCEMENT: Monkey patch tea.py to use plant-specific parameters**
    if plant_specific_params:
        # Create a custom load_tea_sys_params function that includes plant-specific parameters
        original_load_tea_sys_params = tea.load_tea_sys_params

        def enhanced_load_tea_sys_params(iso_target, input_base_dir):
            # Call the original function
            params = original_load_tea_sys_params(iso_target, input_base_dir)

            # Add plant-specific parameters
            params.update(plant_specific_params)

            return params

        # Temporarily replace the function
        tea.load_tea_sys_params = enhanced_load_tea_sys_params

        # **NEW: Set plant-specific report title**
        plant_report_title = f"{plant_name} Unit {generator_id}"
        tea.PLANT_REPORT_TITLE = plant_report_title

    # Call the main function of tea.py
    print(f"Running TEA for {plant_name} Unit {generator_id}...")

    # Create and start progress indicator
    progress = TEAProgressIndicator(
        f"TEA analysis: {plant_name} Unit {generator_id}")
    progress.start()

    try:
        tea.main()
        success = True
    except Exception as e:
        print(f"TEA failed for {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        # Always stop the progress indicator
        progress.stop()

    if success:
        print(f"TEA completed successfully")

        tea_report_src = tea.BASE_OUTPUT_DIR_DEFAULT / \
            f"{iso_region}_TEA_Summary_Report.txt"
        tea_report_dst = output_dir / f"{iso_region}_TEA_Summary_Report.txt"

        if tea_report_src.exists():
            if tea_report_src != tea_report_dst:
                if tea_report_dst.exists():
                    os.remove(tea_report_dst)
                shutil.copy2(tea_report_src, tea_report_dst)

        plot_dir_src = tea.BASE_OUTPUT_DIR_DEFAULT / f"Plots_{iso_region}"

        if plot_dir_src.exists():
            plot_files = list(plot_dir_src.glob('*'))

            for src_file in plot_files:
                dst_file = plot_dir_final / src_file.name
                # **FIX: Avoid copying to the same file**
                if src_file != dst_file:
                    shutil.copy2(src_file, dst_file)

        else:
            os.makedirs(plot_dir_final, exist_ok=True)

    # Restore global variables in tea.py
    if old_base_output is not None:
        tea.BASE_OUTPUT_DIR_DEFAULT = old_base_output
    if old_script_dir is not None:
        tea.SCRIPT_DIR_PATH = old_script_dir
    if old_target_iso is not None:
        tea.TARGET_ISO = old_target_iso
    if old_project_lifetime is not None:
        # Restore original project lifetime
        tea.PROJECT_LIFETIME_YEARS = old_project_lifetime

    # **ENHANCEMENT: Restore original load_tea_sys_params function**
    if plant_specific_params and 'original_load_tea_sys_params' in locals():
        tea.load_tea_sys_params = original_load_tea_sys_params

    # **NEW: Restore original report title if set**
    if plant_specific_params and hasattr(tea, 'PLANT_REPORT_TITLE'):
        delattr(tea, 'PLANT_REPORT_TITLE')


def main():
    # Run TEA for each file in the optimization results directory
    for file in CS1_OUTPUT_DIR.glob("*_hourly_results.csv"):
        m = FILENAME_PATTERN.match(file.name)
        if not m:
            print(f"Skipping file with unexpected name: {file.name}")
            continue
        plant_name, generator_id, iso_region, remaining_years = m.groups()
        run_tea_for_file(file, plant_name, generator_id,
                         iso_region, remaining_years)
    print(f"All TEA results saved in {TEA_RESULTS_DIR}")


if __name__ == "__main__":
    main()
