import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List # Corrected import

# --- Configuration ---
# Match HOURS_IN_YEAR and TARGET_ISO with your config.py
HOURS_IN_YEAR = 8760
TARGET_ISO = "ERCOT"  # Change this to the ISO you want to generate data for
BASE_OUTPUT_DIR = Path("../input/hourly_data")
np.random.seed(42)  # Set seed at the beginning

# Define ISO-specific service abbreviations (match those used in revenue_cost.py)
# Type: 'R' = Regulation (Capacity +/- Perf/Mileage), 'E' = Energy Reserve (Capacity + Deployed Energy)
ISO_SERVICE_MAP = {
    'SPP': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'Sup': 'E'}, # RamU/D/UncU ignored for now
    'CAISO': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'NSpin': 'E', 'RMU': 'R', 'RMD': 'R'},
    'ERCOT': {'RegU': 'R', 'RegD': 'R', 'Spin': 'E', 'NSpin': 'E', 'ECRS': 'E'},
    'PJM': {'Reg': 'R', 'Syn': 'E', 'Rse': 'E', 'TMR': 'E'},
    'NYISO': {'RegC': 'R', 'Spin10': 'E', 'NSpin10': 'E', 'Res30': 'E'},
    'ISONE': {'Spin10': 'E', 'NSpin10': 'E', 'OR30': 'E'},
    # --- Corrected MISO Entry ---
    'MISO': {'Reg': 'R', 'Spin': 'E', 'Sup': 'E', 'STR': 'E'} # RamU/D ignored for now
}

# --- Check if TARGET_ISO is valid ---
if TARGET_ISO not in ISO_SERVICE_MAP:
    raise ValueError(f"Service definitions not found for TARGET_ISO: {TARGET_ISO}")

# Create directories if they don't exist
iso_output_dir = BASE_OUTPUT_DIR / TARGET_ISO
iso_output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory structure created/verified at: {BASE_OUTPUT_DIR}")

# --- 1. Generate System Parameters (`sys_data_advanced.csv`) ---
def generate_system_data(output_path: Path):
    """Generates the system parameters CSV file."""
    # Using the same system parameters as before
    system_params = {
        'delT_minutes': 60.0, 'qSteam_Total_MWth': 1000.0,
        'Turbine_Thermal_Elec_Efficiency_Const': 0.38, 'qSteam_Turbine_min_MWth': 100.0,
        'qSteam_Turbine_max_MWth': 1000.0, 'pTurbine_min_MW': 38.0, 'pTurbine_max_MW': 380.0,
        'Turbine_RampUp_Rate_Percent_per_Min': 2.0, 'Turbine_RampDown_Rate_Percent_per_Min': 2.0,
        'pTurbine_LTE_setpoint_MW': 380.0,
        'qSteam_Turbine_Breakpoints_MWth': '100.0, 550.0, 1000.0',
        'pTurbine_Outputs_at_Breakpoints_MW': '38.0, 215.0, 380.0',
        'pElectrolyzer_min_MW': 20.0, 'pElectrolyzer_max_upper_bound_MW': 500.0,
        'pElectrolyzer_max_lower_bound_MW': 50.0, 'Electrolyzer_RampUp_Rate_Percent_per_Min': 15.0,
        'Electrolyzer_RampDown_Rate_Percent_per_Min': 15.0,
        'qSteam_Electrolyzer_Ramp_Limit_MWth_per_Hour': 200.0,
        'pElectrolyzer_Breakpoints_MW': '20.0, 260.0, 500.0',
        'ke_H2_Values_MWh_per_kg': '55.0, 50.0, 48.0',
        'kt_H2_Values_MWth_per_kg': '10.0, 8.0, 7.0',
        'pIES_min_MW': -380.0, 'pIES_max_MW': 380.0,
        'H2_value_USD_per_kg': 3.0, 'vom_turbine_USD_per_MWh': 2.0,
        'vom_electrolyzer_USD_per_MWh': 3.0, 'cost_water_USD_per_kg_h2': 0.05,
        'cost_electrolyzer_ramping_USD_per_MW_ramp': 0.5,
        'cost_electrolyzer_capacity_USD_per_MW': 600000.0,
        'cost_startup_electrolyzer_USD_per_startup': 500.0, 'MinUpTimeElectrolyzer_hours': 2,
        'MinDownTimeElectrolyzer_hours': 1, 'uElectrolyzer_initial_status_0_or_1': 0,
        'DegradationStateInitial_Units': 0.0, 'DegradationFactorOperation_Units_per_Hour_at_MaxLoad': 0.0001,
        'DegradationFactorStartup_Units_per_Startup': 0.05,
        'H2_storage_capacity_max_kg': 20000.0, 'H2_storage_capacity_min_kg': 1000.0,
        'H2_storage_level_initial_kg': 10000.0, 'H2_storage_charge_rate_max_kg_per_hr': 1000.0,
        'H2_storage_discharge_rate_max_kg_per_hr': 1000.0, 'storage_charge_eff_fraction': 0.98,
        'storage_discharge_eff_fraction': 0.98, 'vom_storage_cycle_USD_per_kg_cycled': 0.02,
        'h2_target_capacity_factor_fraction': 0.90,
    }
    df_system = pd.DataFrame.from_dict(system_params, orient='index', columns=['Value'])
    df_system.index.name = 'Parameter'
    df_system.to_csv(output_path)
    print(f"System parameters saved to {output_path}")

# --- 2. Generate Hourly Time Series Data ---

def generate_hourly_data(hours: int, iso: str, service_map: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
    """Generates DataFrames for hourly prices, factors, etc. specific to the ISO."""

    time_index = pd.RangeIndex(start=1, stop=hours + 1, name='HourOfYear')
    iso_services = service_map[iso] # Get services for the target ISO

    # --- Generate LMP (Energy Price) ---
    base_lmp = 35.0; seasonal_amplitude = 15.0; daily_amplitude = 10.0; noise_amplitude = 5.0
    seasonal_lmp = seasonal_amplitude * -np.cos(2 * np.pi * (time_index.values - 1) / hours)
    daily_lmp = daily_amplitude * -np.cos(2 * np.pi * ((time_index.values - 1 - 16 + 12) % 24) / 24)
    noise_lmp = noise_amplitude * np.random.randn(hours)
    lmp = base_lmp + seasonal_lmp + daily_lmp + noise_lmp
    lmp[lmp < 5.0] = 5.0
    spike_indices = np.random.choice(time_index.values - 1, size=int(hours * 0.01), replace=False)
    lmp[spike_indices] *= np.random.uniform(2, 5)
    df_price = pd.DataFrame({'HourOfYear': time_index, 'Price ($/MWh)': lmp})
    print("Generated LMP data.")

    # --- Generate ISO-Specific AS Prices, Adders, Factors ---
    df_ans_price_data = {'HourOfYear': time_index}
    df_ans_mileage_data = {'HourOfYear': time_index}
    df_ans_deploy_data = {'HourOfYear': time_index}

    print(f"Generating AS data for {iso} services: {list(iso_services.keys())}")
    for service, type in iso_services.items():
        # Define column names based on service and ISO
        price_col = f'p_{service}_{iso}'
        loc_col = f'loc_{service}_{iso}'
        deploy_col = f'deploy_factor_{service}_{iso}'
        mileage_col = f'mileage_factor_{service}_{iso}' # Used by CAISO Reg
        perf_score_col = f'performance_score_{iso}' # Used by PJM Reg
        mileage_ratio_col = f'mileage_ratio_{iso}'   # Used by PJM Reg

        # --- Generate Price ---
        price = np.zeros(hours) # Initialize
        if type == 'R': # Regulation prices
            price = np.maximum(2.0, 5.0 + 0.15 * lmp + 3 * np.random.randn(hours))
            # Specific handling for ISOs with different Reg structures
            if iso == 'CAISO' and service == 'RMU': price = np.maximum(0.5, 1.0 + 0.05 * lmp + 1 * np.random.randn(hours))
            if iso == 'CAISO' and service == 'RMD': price = np.maximum(0.5, 1.0 + 0.05 * lmp + 1 * np.random.randn(hours))
            if iso == 'PJM' and service == 'Reg':
                 # Generate separate Cap/Perf prices for PJM Reg
                 price_cap_col = f'p_RegCap_{iso}'
                 price_perf_col = f'p_RegPerf_{iso}'
                 df_ans_price_data[price_cap_col] = np.maximum(3.0, 6.0 + 0.1 * lmp + 2 * np.random.randn(hours))
                 df_ans_price_data[price_perf_col] = np.maximum(1.0, 2.0 + 0.05 * lmp + 1.5 * np.random.randn(hours))
                 # PJM 'Reg' price itself might not be used directly if split Cap/Perf are used in revenue rule
                 price = df_ans_price_data[price_cap_col] # Assign Cap price as the base 'p_Reg_PJM' for consistency if needed elsewhere
            elif iso == 'MISO' and service == 'Reg':
                 price = np.maximum(3.0, 6.0 + 0.1 * lmp + 2.5 * np.random.randn(hours))
            elif iso == 'NYISO' and service == 'RegC':
                 price = np.maximum(3.0, 6.0 + 0.1 * lmp + 2 * np.random.randn(hours))
        elif type == 'E': # Energy reserves prices
            price = np.maximum(3.0, 4.0 + 0.2 * lmp + 4 * np.random.randn(hours) * (1 + 0.005 * lmp))
        # Add price column (unless handled specially like PJM split where base 'p_Reg_PJM' might be redundant)
        # We add it anyway for completeness, but ensure revenue rule uses correct Cap/Perf prices for PJM
        df_ans_price_data[price_col] = price

        # --- Generate Locational Adder ---
        df_ans_price_data[loc_col] = np.random.normal(0, 0.25, hours)

        # --- Generate Deployment Factor ---
        deploy_factor = np.zeros(hours) # Initialize
        if type == 'R':
            # Generate specific mileage/performance factors if needed
            if iso == 'PJM' and service == 'Reg':
                 perf_score_ts = np.clip(pd.Series(np.random.normal(1.0, 0.03, hours)).ewm(span=10).mean().values, 0.8, 1.1)
                 mileage_ratio_ts = np.clip(pd.Series(np.random.normal(1.0, 0.06, hours)).ewm(span=10).mean().values, 0.7, 1.3)
                 df_ans_mileage_data[perf_score_col] = perf_score_ts
                 df_ans_mileage_data[mileage_ratio_col] = mileage_ratio_ts
                 deploy_factor = np.clip(mileage_ratio_ts * perf_score_ts * np.random.normal(0.5, 0.1, hours), 0.1, 1.0) # Example relation
            elif iso == 'CAISO' and service in ['RegU', 'RegD']:
                 mileage_factor_ts = np.clip(pd.Series(np.random.normal(1.0, 0.05, hours)).ewm(span=10).mean().values, 0.7, 1.3)
                 # CAISO uses RMU/RMD prices, but deploy factor might still be separate
                 df_ans_mileage_data[mileage_col] = mileage_factor_ts
                 deploy_factor = np.clip(mileage_factor_ts * np.random.normal(0.6, 0.1, hours), 0.1, 1.0) # Example relation
            else: # Generic Regulation deployment factor
                 deploy_factor = np.clip(pd.Series(np.random.normal(0.7, 0.15, hours)).ewm(span=8).mean().values, 0.1, 1.0)
        elif type == 'E': # Energy reserves deployment factor
            deploy_factor = np.clip(pd.Series(np.random.normal(0.9, 0.1, hours)).ewm(span=6).mean().values, 0.1, 1.0)
        # Add deployment factor column
        df_ans_deploy_data[deploy_col] = deploy_factor

    df_ans_price = pd.DataFrame(df_ans_price_data)
    df_ans_mileage = pd.DataFrame(df_ans_mileage_data)
    df_ans_deploy = pd.DataFrame(df_ans_deploy_data)

    print("Generated AS Prices, Adders, Mileage, and Deployment Factors.")

    return {
        "df_price_hourly": df_price,
        "df_ANSprice_hourly": df_ans_price,
        "df_ANSmile_hourly": df_ans_mileage,
        "df_ANSdeploy_hourly": df_ans_deploy,
    }

# --- 3. Save Data to CSV ---
def save_data(data_dict: Dict[str, pd.DataFrame], iso_output_dir: Path):
    """Saves the generated DataFrames to CSV files in the specified ISO directory."""
    # Save hourly energy prices
    price_path = iso_output_dir / "Price_hourly.csv"
    data_dict["df_price_hourly"].to_csv(price_path, index=False, float_format='%.4f')
    print(f"Hourly energy prices saved to {price_path}")

    # Save hourly AS prices and adders
    ans_price_path = iso_output_dir / "Price_ANS_hourly.csv"
    data_dict["df_ANSprice_hourly"].to_csv(ans_price_path, index=False, float_format='%.4f')
    print(f"Hourly AS prices saved to {ans_price_path}")

    # Save hourly mileage/performance factors
    ans_mile_path = iso_output_dir / "MileageMultiplier_hourly.csv"
    if len(data_dict["df_ANSmile_hourly"].columns) > 1:
        data_dict["df_ANSmile_hourly"].to_csv(ans_mile_path, index=False, float_format='%.4f')
        print(f"Hourly mileage/performance factors saved to {ans_mile_path}")
    else:
        print(f"No specific mileage/performance factors generated for {iso_output_dir.name}; file MileageMultiplier_hourly.csv not saved.")

    # Save hourly deployment factors
    ans_deploy_path = iso_output_dir / "DeploymentFactor_hourly.csv"
    data_dict["df_ANSdeploy_hourly"].to_csv(ans_deploy_path, index=False, float_format='%.4f')
    print(f"Hourly deployment factors saved to {ans_deploy_path}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Generating Input Data ---")
    # Generate and save system data (common file)
    sys_data_path = BASE_OUTPUT_DIR / "sys_data_advanced.csv"
    generate_system_data(sys_data_path)
    # Generate and save hourly data (ISO-specific files)
    hourly_data = generate_hourly_data(HOURS_IN_YEAR, TARGET_ISO, ISO_SERVICE_MAP)
    save_data(hourly_data, iso_output_dir)
    print("--- Data Generation Complete ---")
    print(f"All files generated in: {BASE_OUTPUT_DIR}")
