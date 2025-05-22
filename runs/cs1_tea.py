import os
import re
import sys
from pathlib import Path
import shutil

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
sys.path.append(str(Path(__file__).parent))
try:
    import tea
except ImportError:
    print('Error: tea.py not found or import failed.')
    sys.exit(1)


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

    print(
        f"Setting project lifetime to {remaining_years_int} years for TEA calculation")

    # Copy the csv to the location and filename expected by tea.py
    # tea.py defaults to reading output/Results_Standardized/{iso}_Hourly_Results_Comprehensive.csv
    # We copy the current csv to that location
    std_dir = Path(__file__).parent.parent / 'output' / 'Results_Standardized'
    os.makedirs(std_dir, exist_ok=True)
    std_csv = std_dir / f"{iso_region}_Hourly_Results_Comprehensive.csv"
    shutil.copy(csv_path, std_csv)

    plot_dir_final = output_dir / f"Plots_{iso_region}"
    os.makedirs(plot_dir_final, exist_ok=True)
    print(f"Created final plot directory at {plot_dir_final}")

    # Call the main function of tea.py
    print(f"Running TEA for {csv_path.name} ...")
    try:
        tea.main()
        print(f"TEA main function completed successfully for {csv_path.name}")

        tea_report_src = tea.BASE_OUTPUT_DIR_DEFAULT / \
            f"{iso_region}_TEA_Summary_Report.txt"
        tea_report_dst = output_dir / f"{iso_region}_TEA_Summary_Report.txt"

        if tea_report_src.exists():
            print(
                f"Report file exists at {tea_report_src}, moving to {tea_report_dst}")
            if tea_report_src != tea_report_dst:  
                if tea_report_dst.exists():
                    os.remove(tea_report_dst)
                shutil.copy2(tea_report_src, tea_report_dst)
                print(f"Report file copied successfully")
        else:
            print(f"Warning: Report file not found at {tea_report_src}")

        plot_dir_src = tea.BASE_OUTPUT_DIR_DEFAULT / f"Plots_{iso_region}"

        if plot_dir_src.exists():
            print(f"Plot directory exists at {plot_dir_src}")

            plot_files = list(plot_dir_src.glob('*'))
            print(
                f"Found {len(plot_files)} files in plot directory: {[f.name for f in plot_files]}")

            for src_file in plot_files:
                dst_file = plot_dir_final / src_file.name
                print(f"Copying {src_file} to {dst_file}")
                shutil.copy2(src_file, dst_file)

            print(f"All plot files copied successfully")

            copied_files = list(plot_dir_final.glob('*'))
            print(
                f"Final plot directory now contains {len(copied_files)} files: {[f.name for f in copied_files]}")
        else:
            print(
                f"Warning: Plot directory not found at {plot_dir_src}, no plots to copy")
            os.makedirs(plot_dir_final, exist_ok=True)

    except Exception as e:
        print(f"TEA failed for {csv_path.name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
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
        print(f"Finished processing {csv_path.name}, all variables restored")


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
