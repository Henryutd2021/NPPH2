import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("../input/hourly_data")
HOURS_IN_YEAR = 8760 # 目标小时数 (365 * 24)
np.random.seed(42)  # Consistent randomness

# --- ISO Service Definitions (MUST BE CONSISTENT WITH model.py) ---
ISO_SERVICE_MAP = {
    'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
    'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
    'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
    'PJM': ['RegUp', 'RegDown', 'Syn', 'Rse', 'TMR'], # Syn=Spin, Rse=Secondary(30min), TMR=Tertiary(30min)
    'NYISO': ['RegUp', 'RegDown', 'Spin10', 'NSpin10', 'Res30'],
    'ISONE': ['Spin10', 'NSpin10', 'OR30'],
    'MISO': ['RegUp', 'RegDown', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
}

# --- Data from PDFs for realistic generation ---

# Mileage values based on "Average Regulation Mileage (...).pdf"
ISO_MILEAGE_MAP = {
    'PJM': {'RegUp': 5.4, 'RegDown': 5.4},
    'CAISO': {'RegU': 12.5, 'RegD': 12.5},
    'MISO': {'RegUp': 7.0, 'RegDown': 7.0},
    'ERCOT': {'RegU': 7.5, 'RegD': 7.5},
    'NYISO': {'RegUp': 13.0, 'RegDown': 13.0},
    'ISONE': {},
    'SPP': {'RegU': 0.2, 'RegD': 0.2}
}

# Winning rates based on "Ancillary Service Markets (...).pdf" (Table 2 and Section 5, 8)
WINNING_RATE_CATEGORIES = {
    'REGULATION': [0.70, 1.0],
    'SPIN': [0.20, 0.50],
    'NONSPIN': [0.10, 0.30],
    'SUPPLEMENTAL': [0.10, 0.30],
    'RESERVE_30MIN': [0.10, 0.40],
    'ECRS': [0.30, 0.70],
    'RAMPING_UNCERTAINTY': [0.40, 0.80]
}
SERVICE_KEY_TO_WINNING_RATE_CATEGORY = {
    'RegU': 'REGULATION', 'RegD': 'REGULATION', 'RegUp': 'REGULATION', 'RegDown': 'REGULATION',
    'Spin': 'SPIN', 'Syn': 'SPIN', 'Spin10': 'SPIN',
    'NSpin': 'NONSPIN', 'NSpin10': 'NONSPIN',
    'Sup': 'SUPPLEMENTAL',
    'Res30': 'RESERVE_30MIN', 'OR30': 'RESERVE_30MIN', 'TMR': 'RESERVE_30MIN', 'Rse': 'RESERVE_30MIN',
    'ECRS': 'ECRS',
    'RamU': 'RAMPING_UNCERTAINTY', 'RamD': 'RAMPING_UNCERTAINTY',
    'RMU': 'RAMPING_UNCERTAINTY', 'RMD': 'RAMPING_UNCERTAINTY',
    'UncU': 'RAMPING_UNCERTAINTY', 'STR': 'RAMPING_UNCERTAINTY'
}

# Deployment factors based on "Ancillary Service Markets (...).pdf" (Table 2 and Section 5, 8)
DEPLOYMENT_FACTOR_CATEGORIES = {
    'SPIN': {'prob_range': [0.001, 0.005], 'mag_range': [0.2, 0.8]},
    'NONSPIN': {'prob_range': [0.0001, 0.001], 'mag_range': [0.2, 0.7]},
    'SUPPLEMENTAL': {'prob_range': [0.0001, 0.001], 'mag_range': [0.2, 0.7]},
    'RESERVE_30MIN': {'prob_range': [0.00001, 0.0001], 'mag_range': [0.1, 0.6]},
    'ECRS': {'prob_range': [0.01, 0.05], 'mag_range': [0.1, 0.8]},
    'RAMPING_UNCERTAINTY': {'prob_range': [0.05, 0.15], 'mag_range': [0.1, 0.8]}
}
SERVICE_KEY_TO_DEPLOYMENT_CATEGORY = {
    'Spin': 'SPIN', 'Syn': 'SPIN', 'Spin10': 'SPIN',
    'NSpin': 'NONSPIN', 'NSpin10': 'NONSPIN',
    'Sup': 'SUPPLEMENTAL',
    'Res30': 'RESERVE_30MIN', 'OR30': 'RESERVE_30MIN', 'TMR': 'RESERVE_30MIN', 'Rse': 'RESERVE_30MIN',
    'ECRS': 'ECRS',
    'RamU': 'RAMPING_UNCERTAINTY', 'RamD': 'RAMPING_UNCERTAINTY',
    'RMU': 'RAMPING_UNCERTAINTY', 'RMD': 'RAMPING_UNCERTAINTY',
    'UncU': 'RAMPING_UNCERTAINTY', 'STR': 'RAMPING_UNCERTAINTY'
}


# --- 1. Generate Hourly Deployment Factor Data (DeploymentFactor_hourly.csv) ---
def generate_deploy_factor_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates the hourly deployment factor CSV file for reserve services,
       using realistic probabilities from PDF data."""
    print(f"Generating hourly deployment factors for {iso}: {output_path}")
    df_deploy = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])
    generated_factors = False

    for service_key in iso_services:
        is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                "RegUp" in service_key or "RegDown" in service_key
        if is_regulation_service:
            continue

        factor_col_name = f'deploy_factor_{service_key}_{iso}'
        category_key = SERVICE_KEY_TO_DEPLOYMENT_CATEGORY.get(service_key)

        if category_key and category_key in DEPLOYMENT_FACTOR_CATEGORIES:
            cat_data = DEPLOYMENT_FACTOR_CATEGORIES[category_key]
            prob_min, prob_max = cat_data['prob_range']
            mag_min, mag_max = cat_data['mag_range']
            
            hourly_event_probs = np.random.uniform(prob_min, prob_max, num_hours)
            deploy_mask = np.random.rand(num_hours) < hourly_event_probs
            
            deploy_magnitudes = np.random.uniform(mag_min, mag_max, num_hours)
            deploy_values = deploy_magnitudes * deploy_mask
            
            df_deploy[factor_col_name] = np.round(deploy_values, 4)
            generated_factors = True
        else:
            print(f"Warning: Deployment category not defined for {service_key} in {iso}. Using very low default.")
            deploy_prob = 0.0001 
            deploy_mask = np.random.rand(num_hours) < deploy_prob
            deploy_magnitude = np.random.uniform(0.01, 0.1, num_hours) 
            deploy_values = deploy_magnitude * deploy_mask
            df_deploy[factor_col_name] = np.round(deploy_values, 4)
            generated_factors = True
            
    if generated_factors:
        df_deploy.index.name = 'Timestamp'
        df_deploy.to_csv(output_path)
        print(f"Hourly deployment factors for {iso} saved to {output_path}")
    else:
        print(f"No reserve deployment factors generated for {iso}. Skipping file save for {output_path.name}")


# --- 2. Generate Hourly Mileage/Performance Factor Data (MileageMultiplier_hourly.csv) ---
def generate_mileage_multiplier_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates hourly mileage and performance factors for regulation services,
       using realistic mileage from PDF data."""
    print(f"Generating hourly mileage/performance factors for {iso}: {output_path}")
    df_mileage_perf = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])
    generated_factors = False

    iso_specific_mileage_map = ISO_MILEAGE_MAP.get(iso, {})

    for service_key in iso_services:
        is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                "RegUp" in service_key or "RegDown" in service_key
        if is_regulation_service:
            mileage_col_name = f'mileage_factor_{service_key}_{iso}'
            perf_col_name = f'performance_factor_{service_key}_{iso}'

            base_mileage = iso_specific_mileage_map.get(service_key)
            if base_mileage is None:
                print(f"Warning: Mileage not defined for {service_key} in {iso}. Using default 1.0-2.5.")
                mileage_values = np.maximum(0.5, 1.5 + np.random.normal(0, 0.3, num_hours))
            else:
                noise_scale = 0.1 * base_mileage if base_mileage > 0.5 else 0.05 * base_mileage 
                mileage_values = np.maximum(0.01, base_mileage + np.random.normal(0, noise_scale, num_hours))
            
            df_mileage_perf[mileage_col_name] = np.round(mileage_values, 4)
            
            perf_values = np.clip(0.95 + np.random.normal(0, 0.05, num_hours), 0.7, 1.0) 
            df_mileage_perf[perf_col_name] = np.round(perf_values, 4)
            generated_factors = True

    if generated_factors:
        df_mileage_perf.index.name = 'Timestamp'
        df_mileage_perf.to_csv(output_path)
        print(f"Hourly mileage/performance factors for {iso} saved to {output_path}")
    else:
         print(f"No regulation mileage/performance factors generated for {iso}. Skipping file save for {output_path.name}")


# --- 3. Generate Hourly Winning Rate Data (WinningRate_hourly.csv) ---
def generate_winning_rate_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS winning rate CSV file for all services,
       using realistic rates from PDF data."""
    print(f"Generating hourly AS winning rates for {iso}: {output_path}")
    df_winrate = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])

    for service_key in iso_services:
        rate_col_name = f'winning_rate_{service_key}_{iso}'
        category_key = SERVICE_KEY_TO_WINNING_RATE_CATEGORY.get(service_key)

        if category_key and category_key in WINNING_RATE_CATEGORIES:
            min_rate, max_rate = WINNING_RATE_CATEGORIES[category_key]
            win_rates = np.random.uniform(min_rate, max_rate, num_hours)
        else:
            print(f"Warning: Winning rate category not defined for {service_key} in {iso}. Using default 0.5-0.8.")
            win_rates = np.random.uniform(0.5, 0.8, num_hours) 
        
        df_winrate[rate_col_name] = np.round(np.clip(win_rates, 0.01, 1.0), 4) 

    df_winrate.index.name = 'Timestamp'
    df_winrate.to_csv(output_path)
    print(f"Hourly AS winning rates for {iso} saved to {output_path}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Create a time index for the year 2024, excluding February 29th, for 8760 hours
    # Generate all hours for 2024
    full_2024_index = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h')
    # Filter out February 29th
    time_index_hourly = full_2024_index[~((full_2024_index.month == 2) & (full_2024_index.day == 29))]

    # Ensure the resulting index has exactly 8760 hours
    if len(time_index_hourly) != HOURS_IN_YEAR:
        # Fallback or alternative way to ensure 8760 hours for 2024, non-leap
        # This can happen if there are issues with timezone localization with the above method
        # Or if a simpler non-leap year representation for 2024 is desired.
        # Forcing 8760 periods from Jan 1st.
        print(f"Warning: Filtered 2024 index had {len(time_index_hourly)} hours. Forcing 8760 hours from 2024-01-01.")
        time_index_hourly = pd.date_range(start='2024-01-01', periods=HOURS_IN_YEAR, freq='h')
        # If you specifically need to represent a non-leap 2024,
        # the above line creates a generic 8760 hour sequence starting Jan 1, 2024.
        # This will not have a Feb 29 and will end before Dec 31 if HOURS_IN_YEAR is 8760.
        # If the goal is a "typical" year of 8760 hours *labeled* as 2024, this is fine.

    print(f"Generated time index for {len(time_index_hourly)} hours, starting {time_index_hourly[0]}, ending {time_index_hourly[-1]}")

    for target_iso in ISO_SERVICE_MAP.keys():
        print(f"\n--- Generating hourly files for {target_iso} ---")
        iso_output_dir = BASE_OUTPUT_DIR / target_iso
        iso_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate Deployment Factors
        deploy_factor_path = iso_output_dir / "DeploymentFactor_hourly.csv"
        generate_deploy_factor_hourly(deploy_factor_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

        # Generate Mileage and Performance Factors
        mileage_perf_path = iso_output_dir / "MileageMultiplier_hourly.csv" 
        generate_mileage_multiplier_hourly(mileage_perf_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

        # Generate Winning Rates
        winrate_path = iso_output_dir / "WinningRate_hourly.csv"
        generate_winning_rate_hourly(winrate_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

    print("\n--- Data generation complete for specified ancillary service factors ---")