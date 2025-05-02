# data/data_gen_v2.py

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import math # Needed for CRF calculation if converting total CAPEX

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("../input/hourly_data")
HOURS_IN_YEAR = 8760
np.random.seed(42) # Consistent randomness

# --- ISO Service Definitions (Match model.py/data_io.py needs) ---
# Define services needed per ISO based on model.py parameter loading
ISO_SERVICE_MAP = {
    'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
    'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'], # Requires mileage_factor for RegU/D
    'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
    'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'], # Requires RegCap/Perf prices, perf_score, mileage_ratio factors
    'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
    'ISONE': ['Spin10', 'NSpin10', 'OR30'],
    'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
}

# --- Helper Function ---
def calculate_crf(discount_rate, lifetime_years):
    """Calculates the Capital Recovery Factor (used for annualizing CAPEX)."""
    if lifetime_years <= 0: return 0
    if discount_rate == 0: return 1.0 / lifetime_years if lifetime_years > 0 else 0
    try:
        factor = (1 + discount_rate)**lifetime_years
        return (discount_rate * factor) / (factor - 1)
    except (OverflowError, ValueError):
        print(f"Warning: CRF calculation failed for rate={discount_rate}, life={lifetime_years}. Returning rate as approximation.")
        return discount_rate # Approximation for long lifetimes or errors

# --- 1. Generate System Parameters (sys_data_advanced.csv) ---
def generate_system_data(output_path: Path):
    """Generates the COMMON system parameters CSV file."""
    print(f"Generating common system parameters: {output_path}")

    # --- Base Parameters ---
    p_turbine_max_mw = 380.0
    q_turbine_max_mwth = 1000.0
    p_elec_max_capacity_mw = 500.0 # Default upper bound for optimization
    elec_min_load_fraction = 0.10 # 10% minimum load

    # --- Electrolyzer Efficiency (Example PEM-like values) ---
    # Breakpoints (MW DC input)
    p_elec_breakpoints_mw = [50.0, 100.0, 250.0, 500.0] # Must match max capacity if fixed
    # Efficiency (kWh/kg H2) - Higher efficiency at lower loads
    efficiency_kwh_per_kg = [58.0, 57.0, 56.0, 55.0]
    # Convert to MWh/kg H2 for the model parameter ke_H2
    ke_h2_mwh_per_kg = [e / 1000.0 for e in efficiency_kwh_per_kg]

    # Steam consumption (kt_H2) - Example for HTE (set to 0 for LTE case study)
    # Placeholder: Assume ratio relative to H2 production rate estimated at breakpoints
    # H2 rate (kg/hr) = Power (MW) / Efficiency (MWh/kg)
    # Example: kt = some_factor * H2_rate -> MWth / (kg/hr) ??? No, kt is MWth / kg
    # Let's assume a simple constant thermal input per kg H2 for HTE for now
    ht_steam_mwh_per_kg = 0.015 # Approx 15 kWh_th / kg_H2 -> MW_th / (kg/hr) ??? Needs care
    # kt = MWth / kg_H2. Let's use a simpler approach for generation:
    # Assume thermal input is a fraction of electrical input for HTE?
    hte_steam_ratio_of_power = 0.15 # e.g., Thermal input is 15% of Electrical Input (MWth/MW)
    kt_h2_values_mwth_per_kg_hte = [(p * hte_steam_ratio_of_power) / (p / ke) if ke > 1e-9 else 0 for p, ke in zip(p_elec_breakpoints_mw, ke_h2_mwh_per_kg)]

    # --- Battery Parameters ---
    battery_lifetime = 15
    discount_rate = 0.08
    batt_crf = calculate_crf(discount_rate, battery_lifetime)
    batt_capex_mwh_total = 250000.0
    batt_capex_mw_total = 150000.0
    batt_fom_mwh_yr = 5000.0

    system_params = {
        # General
        'delT_minutes': 60.0,
        'AS_Duration': 0.25, # Fraction of an hour (e.g., 15 min for reserves)
        'plant_lifetime_years': 30,
        'discount_rate': discount_rate, # General discount rate
        'pIES_min_MW': -1000.0, # Max power import
        'pIES_max_MW': 1000.0, # Max power export

        # Nuclear Generator (NPP)
        'qSteam_Total_MWth': q_turbine_max_mwth,
        'qSteam_Turbine_min_MWth': 100.0,
        'qSteam_Turbine_max_MWth': q_turbine_max_mwth,
        'pTurbine_min_MW': 38.0, # Calculated from min steam * efficiency
        'pTurbine_max_MW': p_turbine_max_mw,
        'Turbine_RampUp_Rate_Percent_per_Min': 2.0,
        'Turbine_RampDown_Rate_Percent_per_Min': 2.0,
        'pTurbine_LTE_setpoint_MW': p_turbine_max_mw, # Setpoint if LTE is active
        'Turbine_Thermal_Elec_Efficiency_Const': 0.38, # For linear fallback
        'qSteam_Turbine_Breakpoints_MWth': '100.0, 550.0, 1000.0', # Example PWL data
        'pTurbine_Outputs_at_Breakpoints_MW': '38.0, 209.0, 380.0', # Example PWL data
        'vom_turbine_USD_per_MWh': 2.0,

        # Electrolyzer (General - HTE/LTE specifics handled by LTE flag in config)
        'pElectrolyzer_min_MW': p_elec_max_capacity_mw * elec_min_load_fraction, # Min power based on % of max capacity
        'pElectrolyzer_max_upper_bound_MW': p_elec_max_capacity_mw, # Max optimizable capacity
        'pElectrolyzer_max_lower_bound_MW': 0.0, # Min optimizable capacity (can be > 0)
        'Electrolyzer_RampUp_Rate_Percent_per_Min': 10.0, # % of MAX capacity per minute
        'Electrolyzer_RampDown_Rate_Percent_per_Min': 10.0,# % of MAX capacity per minute
        'pElectrolyzer_Breakpoints_MW': ', '.join(map(str, p_elec_breakpoints_mw)),
        'ke_H2_Values_MWh_per_kg': ', '.join(map(lambda x: f"{x:.5f}", ke_h2_mwh_per_kg)), # Electrical MWh / kg H2
        'kt_H2_Values_MWth_per_kg': ', '.join(map(lambda x: f"{x:.5f}", kt_h2_values_mwth_per_kg_hte)), # Thermal MWth / kg H2 (Example for HTE, ignored if LTE flag is True)
        'Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour': 500.0, # Limit for HTE steam ramp
        'vom_electrolyzer_USD_per_MWh': 3.0, # VOM based on electricity consumption
        'cost_water_USD_per_kg_h2': 0.05,
        'cost_electrolyzer_ramping_USD_per_MW_ramp': 0.5, # Cost per MW change (up or down)
        # Annualized CAPEX ($/MW/year) - Rough estimate based on total cost / lifetime
        'cost_electrolyzer_capacity_USD_per_MW_year': 100000.0, # Example: $2M total / 20 yr lifetime = 100k/yr

        # Electrolyzer - Startup/Shutdown & Degradation
        'cost_startup_electrolyzer_USD_per_startup': 500.0,
        'MinUpTimeElectrolyzer_hours': 2,
        'MinDownTimeElectrolyzer_hours': 1,
        'uElectrolyzer_initial_status_0_or_1': 0, # Start offline
        'DegradationStateInitial_Units': 0.0,
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad': 0.0001, # Units accumulated per hour at full load
        'DegradationFactorStartup_Units_per_Startup': 1.0, # Units accumulated per startup event

        # Hydrogen Value & Target CF
        'H2_value_USD_per_kg': 3.0,
        'h2_target_capacity_factor_fraction': 0.0, # Default 0 if feature not used

        # H2 Storage (Optional)
        'H2_storage_capacity_max_kg': 100000.0,
        'H2_storage_capacity_min_kg': 1000.0,
        'H2_storage_level_initial_kg': 50000.0,
        'H2_storage_charge_rate_max_kg_per_hr': 5000.0,
        'H2_storage_discharge_rate_max_kg_per_hr': 5000.0,
        'storage_charge_eff_fraction': 0.98, # Use fraction suffix
        'storage_discharge_eff_fraction': 0.98,# Use fraction suffix
        'vom_storage_cycle_USD_per_kg_cycled': 0.01, # Cost per kg moved in OR out

        # Battery Storage (Optional)
        'BatteryCapacity_min_MWh': 10.0,   # Min optimizable capacity
        'BatteryCapacity_max_MWh': 1000.0, # Max optimizable capacity
        'BatteryPowerRatio_MW_per_MWh': 0.25, # C-rate (Power_MW / Capacity_MWh)
        'BatteryChargeEff': 0.92, # Use names from model.py
        'BatteryDischargeEff': 0.92,# Use names from model.py
        'BatterySOC_min_fraction': 0.10, # Use fraction suffix
        'BatterySOC_initial_fraction': 0.50,# Use fraction suffix
        'BatteryRequireCyclicSOC': True, # Enforce SOC matches start/end
        'BatteryRampRate_fraction_per_hour': 1.0, # Can ramp full power in 1 hour
        # Annualized CAPEX/Fixed OM ($/unit/year)
        'BatteryCapex_USD_per_MWh_year': batt_capex_mwh_total * batt_crf,
        'BatteryCapex_USD_per_MW_year': batt_capex_mw_total * batt_crf,
        'BatteryFixedOM_USD_per_MWh_year': batt_fom_mwh_yr,
        # 'vom_battery_per_mwh_cycled': 0.1 # Optional VOM cost per MWh cycled ($/MWh)
    }
    df_system = pd.DataFrame(list(system_params.items()), columns=['Parameter', 'Value'])
    df_system.set_index('Parameter', inplace=True)
    df_system.to_csv(output_path)
    print(f"System parameters saved to {output_path}")

# --- 2. Generate Hourly Price Data (Price_hourly.csv) ---
def generate_price_hourly(output_path: Path, num_hours=HOURS_IN_YEAR):
    """Generates the hourly energy price (LMP) CSV file."""
    print(f"Generating hourly energy prices: {output_path}")
    start_time = datetime.datetime(2023, 1, 1)
    time_index = pd.to_datetime([start_time + datetime.timedelta(hours=i) for i in range(num_hours)])
    df_price = pd.DataFrame(index=time_index)

    # Generate base energy price (LMP) - slightly more realistic pattern
    base_lmp = 45
    peak_factor = 1.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / HOURS_IN_YEAR), num_hours)) # Simple seasonality
    daily_variation = 20 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / 24), num_hours)) # Daily cycle
    weekly_variation = 5 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / (24*7)), num_hours)) # Weekly cycle
    # More noise, occasional spikes
    random_noise = np.random.normal(0, 8, num_hours)
    spike_prob = 0.01
    spike_magnitude = np.random.uniform(50, 200, num_hours)
    spikes = np.random.choice([0, 1], size=num_hours, p=[1-spike_prob, spike_prob]) * spike_magnitude
    lmp = base_lmp * peak_factor + daily_variation + weekly_variation + random_noise + spikes
    lmp = np.maximum(5, lmp) # Floor price

    df_price['Price ($/MWh)'] = lmp # Column name expected by data_io.py

    df_price.index.name = 'Timestamp'
    df_price.to_csv(output_path)
    print(f"Hourly energy prices saved to {output_path}")

# --- 3. Generate Hourly AS Price Data (Price_ANS_hourly.csv) ---
def generate_price_ans_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS price and adder CSV file."""
    print(f"Generating hourly AS prices/adders: {output_path}")
    df_ans = pd.DataFrame(index=lmp_series.index)

    iso_services = ISO_SERVICE_MAP[iso]
    for service in iso_services:
        # Skip factor parameters - they go in MileageMultiplier
        if any(f in service for f in ['factor', 'score', 'ratio']): continue
        # PJM Reg has separate Cap/Perf prices
        if iso == 'PJM' and service == 'Reg': continue # Handled by RegCap/RegPerf

        param_col_pattern = f"{service}_{iso}"
        price_col_name = f'p_{param_col_pattern}'
        loc_col_name = f'loc_{param_col_pattern}'

        # Generate Capacity Price (MCP) - correlated with LMP but lower, add noise/spikes
        base_mcp = np.maximum(2, lmp_series * np.random.uniform(0.05, 0.4, num_hours))
        mcp_noise = np.random.normal(0, 3, num_hours)
        mcp_spike_prob = 0.02
        mcp_spike_magnitude = np.random.uniform(10, 50, num_hours)
        mcp_spikes = np.random.choice([0, 1], size=num_hours, p=[1-mcp_spike_prob, mcp_spike_prob]) * mcp_spike_magnitude
        mcp = np.maximum(0, base_mcp + mcp_noise + mcp_spikes) # Floor at 0

        # Handle PJM specific RegCap/RegPerf if service='RegCap'/'RegPerf'
        if iso == 'PJM' and service == 'RegCap':
            df_ans[f'p_RegCap_{iso}'] = mcp # Use generated MCP for RegCap
        elif iso == 'PJM' and service == 'RegPerf':
             df_ans[f'p_RegPerf_{iso}'] = np.maximum(0, mcp * np.random.uniform(0.5, 1.5, num_hours)) # Perf price related to cap price
        else:
            df_ans[price_col_name] = mcp

        # Generate Locational Adder (mostly zero, small positive value sometimes)
        adder_mask = np.random.rand(num_hours) < 0.05 # 5% chance of non-zero adder
        adder_values = np.random.uniform(0.1, 2, num_hours) * adder_mask
        df_ans[loc_col_name] = adder_values

    df_ans.index.name = 'Timestamp'
    df_ans.to_csv(output_path)
    print(f"Hourly AS prices/adders saved to {output_path}")


# --- 4. Generate Hourly Deployment Factor Data (DeploymentFactor_hourly.csv) ---
def generate_deploy_factor_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly deployment factor CSV file (Optional)."""
    print(f"Generating hourly deployment factors: {output_path}")
    df_deploy = pd.DataFrame(index=lmp_series.index)

    iso_services = ISO_SERVICE_MAP[iso]
    for service in iso_services:
        # Only generate for services that typically have deployment (Reserves)
        # Simplified: Assume all except pure capacity products might deploy
        # This might need refinement based on specific market rules
        if service in ['RegCap', 'RegC', 'RegPerf', 'performance_score', 'mileage_ratio']: continue

        factor_col_name = f'deploy_factor_{service}_{iso}'

        # Simulate higher deployment probability/magnitude during high LMP periods?
        high_lmp_threshold = lmp_series.quantile(0.8) # Top 20% LMP
        deploy_prob = np.where(lmp_series > high_lmp_threshold, 0.15, 0.05) # Higher prob if LMP is high
        deploy_mask = np.random.rand(num_hours) < deploy_prob
        # Magnitude can also vary
        deploy_magnitude = np.random.uniform(0.01, 0.6, num_hours)
        deploy_values = deploy_magnitude * deploy_mask
        df_deploy[factor_col_name] = np.round(deploy_values, 4)

    df_deploy.index.name = 'Timestamp'
    df_deploy.to_csv(output_path)
    print(f"Hourly deployment factors saved to {output_path}")


# --- 5. Generate Hourly Mileage/Performance Factor Data (MileageMultiplier_hourly.csv) ---
def generate_mileage_multiplier_hourly(output_path: Path, iso: str, num_hours=HOURS_IN_YEAR):
    """Generates the hourly mileage/performance factor CSV file (Optional)."""
    print(f"Generating hourly mileage/performance factors: {output_path}")
    df_mileage = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=num_hours, freq='h')) # Need index

    # Only generate columns if needed by the specific ISO
    if iso == 'CAISO':
        for service in ['RegU', 'RegD']:
            # Typically around 1.0, maybe some slight noise
            values = np.maximum(0.5, 1.0 + np.random.normal(0, 0.1, num_hours))
            df_mileage[f'mileage_factor_{service}_{iso}'] = np.round(values, 4)
    elif iso == 'PJM':
        # Performance Score (often resource-specific, simulate system average around 0.9-1.0)
        perf_values = np.maximum(0.7, 0.95 + np.random.normal(0, 0.05, num_hours))
        df_mileage[f'performance_score_{iso}'] = np.round(perf_values, 4)
        # Mileage Ratio (can vary based on signal)
        mileage_values = np.maximum(0.5, 1.2 + np.random.normal(0, 0.2, num_hours))
        df_mileage[f'mileage_ratio_{iso}'] = np.round(mileage_values, 4)
    else:
        # No specific factors needed for other ISOs in this basic setup
        print(f"No specific mileage/performance factors generated for {iso}")
        # Create empty file with just index to avoid load errors? Or skip generation.
        # Let's skip file generation if no columns are added.
        if not df_mileage.columns.any():
             print(f"Skipping generation of empty {output_path}")
             return # Exit function if no columns added

    df_mileage.index.name = 'Timestamp'
    df_mileage.to_csv(output_path)
    print(f"Hourly mileage/performance factors saved to {output_path}")


# --- 6. Generate Hourly Winning Rate Data (WinningRate_hourly.csv) ---
def generate_winning_rate_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS winning rate CSV file (Optional)."""
    print(f"Generating hourly AS winning rates: {output_path}")
    df_winrate = pd.DataFrame(index=lmp_series.index)

    iso_services = ISO_SERVICE_MAP[iso]
    for service in iso_services:
         # Skip factor parameters
        if any(f in service for f in ['factor', 'score', 'ratio']): continue

        rate_col_name = f'winning_rate_{service}_{iso}'

        # Simulate winning rate between 0 and 1. Maybe higher chance when LMP is low?
        low_lmp_threshold = lmp_series.quantile(0.2) # Bottom 20% LMP
        base_rate = np.where(lmp_series < low_lmp_threshold, 0.9, 0.7) # Higher base rate if LMP low
        rate_noise = np.random.normal(0, 0.1, num_hours)
        win_rates = np.clip(base_rate + rate_noise, 0.1, 1.0) # Bound between 0.1 and 1.0
        df_winrate[rate_col_name] = np.round(win_rates, 4)

    df_winrate.index.name = 'Timestamp'
    df_winrate.to_csv(output_path)
    print(f"Hourly AS winning rates saved to {output_path}")


# --- Main Execution Logic ---

# 1. Generate the common system data file ONCE
system_data_path = BASE_OUTPUT_DIR / "sys_data_advanced.csv"
generate_system_data(system_data_path)

# 2. Iterate through ISOs to generate specific hourly files
for target_iso in ISO_SERVICE_MAP.keys():
    print(f"\n--- Generating hourly files for {target_iso} ---")
    iso_output_dir = BASE_OUTPUT_DIR / target_iso
    iso_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate Price_hourly.csv
    price_hourly_path = iso_output_dir / "Price_hourly.csv"
    generate_price_hourly(price_hourly_path, num_hours=HOURS_IN_YEAR)
    # Load the generated LMP series to use for other files
    try:
        lmp_series = pd.read_csv(price_hourly_path, index_col='Timestamp', parse_dates=True)['Price ($/MWh)']
    except Exception as e:
        print(f"ERROR: Failed to load generated LMP data for {target_iso}: {e}. Skipping remaining hourly files for this ISO.")
        continue # Skip to next ISO

    # Generate Price_ANS_hourly.csv
    price_ans_path = iso_output_dir / "Price_ANS_hourly.csv"
    generate_price_ans_hourly(price_ans_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

    # Generate DeploymentFactor_hourly.csv (Optional)
    deploy_factor_path = iso_output_dir / "DeploymentFactor_hourly.csv"
    generate_deploy_factor_hourly(deploy_factor_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

    # Generate MileageMultiplier_hourly.csv (Optional)
    mileage_path = iso_output_dir / "MileageMultiplier_hourly.csv"
    generate_mileage_multiplier_hourly(mileage_path, target_iso, num_hours=HOURS_IN_YEAR)

    # Generate WinningRate_hourly.csv (Optional)
    winrate_path = iso_output_dir / "WinningRate_hourly.csv"
    generate_winning_rate_hourly(winrate_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

print("\n--- Data generation complete for all ISOs ---")