"""
Data generation utility for nuclear-hydrogen optimization model.

This module generates synthetic data for testing and validation of the
nuclear-hydrogen optimization model, including price data, system parameters,
and ancillary service data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import math

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("../input/hourly_data")
HOURS_IN_YEAR = 8760
np.random.seed(42)  # Consistent randomness

# --- ISO Service Definitions (MUST BE CONSISTENT WITH model.py) ---
ISO_SERVICE_MAP = {
    'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
    'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
    'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
    'PJM': ['RegUp', 'RegDown', 'Syn', 'Rse', 'TMR'],
    'NYISO': ['RegUp', 'RegDown', 'Spin10', 'NSpin10', 'Res30'],
    'ISONE': ['Spin10', 'NSpin10', 'OR30'],
    'MISO': ['RegUp', 'RegDown', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
}


# --- Helper Function ---
def calculate_crf(discount_rate, lifetime_years):
    """Calculates the Capital Recovery Factor."""
    if lifetime_years <= 0:
        return 0
    if discount_rate == 0:
        return 1.0 / lifetime_years if lifetime_years > 0 else 0
    try:
        factor = (1 + discount_rate)**lifetime_years
        if abs(factor - 1.0) < 1e-9: # Avoid division by zero if factor is very close to 1
            return 0 if lifetime_years <= 0 else (1.0 / lifetime_years if discount_rate == 0 else 0)
        return (discount_rate * factor) / (factor - 1)
    except (OverflowError, ValueError):
        print(f"Warning: CRF calculation failed for rate={discount_rate}, life={lifetime_years}. Returning rate.")
        return discount_rate


# --- 1. Generate System Parameters (sys_data_advanced.csv) ---
def generate_system_data(output_path: Path):
    """Generates the COMMON system parameters CSV file."""
    print(f"Generating common system parameters: {output_path}")

    p_turbine_max_mw = 380.0
    q_turbine_max_mwth = 1000.0
    p_elec_max_capacity_mw = 500.0
    elec_min_load_fraction = 0.10
    p_elec_min_load_mw = p_elec_max_capacity_mw * elec_min_load_fraction
    discount_rate = 0.08
    # electrolyzer_economic_lifetime_years = 20 # This is used for LCOH, not directly in opt model params
    hydrogen_subsidy_duration_years = 10

    p_elec_breakpoints_mw = [p_elec_min_load_mw, 100.0, 250.0, 500.0]
    p_elec_breakpoints_str = ', '.join(map(str, p_elec_breakpoints_mw))
    cost_water_usd_per_kg_h2 = 0.03
    uElectrolyzer_initial_status_0_or_1 = 0
    DegradationStateInitial_Units = 0.0
    H2_value_USD_per_kg = 3.0
    hydrogen_subsidy_per_kg_val = 3.00
    h2_target_capacity_factor_fraction = 0.0
    aux_power_consumption_per_kg_h2_val = 0.05 # Example: 50 kWh/ton_H2 -> 0.05 kWh/kg_H2 -> 0.00005 MWh/kg_H2. This param is MW_aux per kg/hr H2.
                                             # If 0.05 kWh/kg and H2 prod rate is 1 kg/hr, then aux power is 0.05 kW = 0.00005 MW.
                                             # Let's use a slightly higher value for sensitivity, e.g. 0.5 kWh/kg -> 0.0005 MW/(kg/hr)
    aux_power_consumption_per_kg_h2_val = 0.0005 # MW_aux per (kg_H2/hr)

    # LTE (PEM Representative) Parameters
    lte_capex_usd_per_kw = 2000.0
    lte_lifetime_years = 20
    lte_crf = calculate_crf(discount_rate, lte_lifetime_years)
    lte_capex_annual_usd_per_mw_year = lte_capex_usd_per_kw * 1000.0 * lte_crf
    lte_efficiency_kwh_per_kg = [57.0, 56.0, 55.0, 54.0]
    lte_ke_h2_mwh_per_kg = [e / 1000.0 for e in lte_efficiency_kwh_per_kg]
    lte_ke_h2_values_str = ', '.join(map(lambda x: f"{x:.5f}", lte_ke_h2_mwh_per_kg))
    lte_kt_h2_values_str = ', '.join(['0.00000'] * len(p_elec_breakpoints_mw))
    lte_vom_usd_per_mwh = 8.0
    lte_cost_startup_usd = 300.0
    lte_ramp_up_pct_min = 10.0
    lte_ramp_down_pct_min = 10.0
    lte_min_uptime_hr = 2
    lte_min_downtime_hr = 1
    lte_degradation_op_units_per_hr = 0.0001
    lte_degradation_su_units = 1.0

    # HTE (SOEC Representative) Parameters
    hte_capex_usd_per_kw = 2500.0
    hte_lifetime_years = 20
    hte_crf = calculate_crf(discount_rate, hte_lifetime_years)
    hte_capex_annual_usd_per_mw_year = hte_capex_usd_per_kw * 1000.0 * hte_crf
    hte_efficiency_kwh_per_kg = [42.0, 41.0, 40.0, 39.0]
    hte_ke_h2_mwh_per_kg = [e / 1000.0 for e in hte_efficiency_kwh_per_kg]
    hte_ke_h2_values_str = ', '.join(map(lambda x: f"{x:.5f}", hte_ke_h2_mwh_per_kg))
    hte_kt_h2_mwh_per_kg = [0.015] * len(p_elec_breakpoints_mw)
    hte_kt_h2_values_str = ', '.join(map(lambda x: f"{x:.5f}", hte_kt_h2_mwh_per_kg))
    hte_vom_usd_per_mwh = 10.0
    hte_cost_startup_usd = 700.0
    hte_ramp_up_pct_min = 5.0
    hte_ramp_down_pct_min = 5.0
    hte_min_uptime_hr = 4
    hte_min_downtime_hr = 2
    hte_degradation_op_units_per_hr = 0.0003
    hte_degradation_su_units = 1.5

    # Battery Parameters
    battery_lifetime = 15
    batt_crf = calculate_crf(discount_rate, battery_lifetime)
    batt_duration_hours = 4.0
    batt_capex_mwh_total_usd_per_kwh = 236.0
    batt_capex_mwh_total = batt_capex_mwh_total_usd_per_kwh * 1000.0
    batt_capex_mw_total_usd_per_kw = batt_capex_mwh_total_usd_per_kwh * batt_duration_hours
    batt_capex_mw_total = batt_capex_mw_total_usd_per_kw * 1000.0
    batt_capex_usd_per_mwh_year = batt_capex_mwh_total * batt_crf
    batt_capex_usd_per_mw_year = batt_capex_mw_total * batt_crf # This is power capacity cost
    # Fixed O&M ($/MWh/year) - Based on NREL ATB (2.5% of $/kW CAPEX per year), then convert to per MWh-year
    fom_fraction_of_capex_per_kw = 0.025
    batt_fom_usd_per_kw_year = fom_fraction_of_capex_per_kw * batt_capex_mw_total_usd_per_kw # $/kW-year
    # To get $/MWh-year, need to divide by duration if it's related to energy capacity, or use $/MW-year and $/MWh-year separately.
    # The model uses BatteryFixedOM_USD_per_MWh_year. Let's assume this is an energy capacity related FOM.
    # If batt_fom_usd_per_mw_year is the primary, then batt_fom_mwh_yr = batt_fom_usd_per_mw_year / batt_duration_hours (incorrect)
    # Let's use a direct $/MWh-year assumption or derive from $/kW-year and assume it applies to energy capacity.
    # For simplicity, let's assume a fixed O&M cost per MWh of installed capacity per year.
    # Say, 1% of energy capex ($/kWh) per year.
    batt_fom_usd_per_mwh_year = (batt_capex_mwh_total_usd_per_kwh * 0.01) * 1000.0 # 1% of $/MWh-cap per year

    batt_power_ratio = 1.0 / batt_duration_hours
    battery_charge_eff = 0.92
    battery_discharge_eff = 0.92
    battery_soc_min_fraction = 0.10

    system_params = {
        'delT_minutes': 60.0, 'AS_Duration': 0.25, 'plant_lifetime_years': 30,
        'discount_rate': discount_rate, 'pIES_min_MW': -1000.0, 'pIES_max_MW': 1000.0,
        'qSteam_Total_MWth': q_turbine_max_mwth, 'qSteam_Turbine_min_MWth': 100.0,
        'qSteam_Turbine_max_MWth': q_turbine_max_mwth, 'pTurbine_min_MW': 38.0,
        'pTurbine_max_MW': p_turbine_max_mw, 'Turbine_RampUp_Rate_Percent_per_Min': 2.0,
        'Turbine_RampDown_Rate_Percent_per_Min': 2.0, 'pTurbine_LTE_setpoint_MW': p_turbine_max_mw,
        'Turbine_Thermal_Elec_Efficiency_Const': 0.38,
        'qSteam_Turbine_Breakpoints_MWth': '100.0, 550.0, 1000.0',
        'pTurbine_Outputs_at_Breakpoints_MW': '38.0, 209.0, 380.0',
        'vom_turbine_USD_per_MWh': 2.0,
        'pElectrolyzer_max_upper_bound_MW': p_elec_max_capacity_mw,
        'pElectrolyzer_max_lower_bound_MW': 0.0, # Ensure this is <= pElectrolyzer_min
        'pElectrolyzer_min_MW': p_elec_min_load_mw,
        'pElectrolyzer_Breakpoints_MW': p_elec_breakpoints_str,
        'ke_H2_Values_MWh_per_kg': lte_ke_h2_values_str, # Default to LTE values
        'kt_H2_Values_MWth_per_kg': lte_kt_h2_values_str, # Default to LTE values (zero)
        'cost_water_USD_per_kg_h2': cost_water_usd_per_kg_h2,
        'uElectrolyzer_initial_status_0_or_1': uElectrolyzer_initial_status_0_or_1,
        'DegradationStateInitial_Units': DegradationStateInitial_Units,
        'h2_target_capacity_factor_fraction': h2_target_capacity_factor_fraction,
        'hydrogen_subsidy_value_usd_per_kg': hydrogen_subsidy_per_kg_val,
        'hydrogen_subsidy_duration_years': hydrogen_subsidy_duration_years,
        'aux_power_consumption_per_kg_h2': aux_power_consumption_per_kg_h2_val,
        'H2_value_USD_per_kg': H2_value_USD_per_kg,
        'cost_electrolyzer_ramping_USD_per_MW_ramp': 0.5,
        'Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour': 500.0,
        'pElectrolyzer_min_MW_LTE': p_elec_min_load_mw,
        'Electrolyzer_RampUp_Rate_Percent_per_Min_LTE': lte_ramp_up_pct_min,
        'Electrolyzer_RampDown_Rate_Percent_per_Min_LTE': lte_ramp_down_pct_min,
        'pElectrolyzer_Breakpoints_MW_LTE': p_elec_breakpoints_str,
        'ke_H2_Values_MWh_per_kg_LTE': lte_ke_h2_values_str,
        'kt_H2_Values_MWth_per_kg_LTE': lte_kt_h2_values_str,
        'vom_electrolyzer_USD_per_MWh_LTE': lte_vom_usd_per_mwh,
        'cost_electrolyzer_capacity_USD_per_MW_year_LTE': lte_capex_annual_usd_per_mw_year,
        'cost_startup_electrolyzer_USD_per_startup_LTE': lte_cost_startup_usd,
        'MinUpTimeElectrolyzer_hours_LTE': lte_min_uptime_hr,
        'MinDownTimeElectrolyzer_hours_LTE': lte_min_downtime_hr,
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE': lte_degradation_op_units_per_hr,
        'DegradationFactorStartup_Units_per_Startup_LTE': lte_degradation_su_units,
        'pElectrolyzer_min_MW_HTE': p_elec_min_load_mw,
        'Electrolyzer_RampUp_Rate_Percent_per_Min_HTE': hte_ramp_up_pct_min,
        'Electrolyzer_RampDown_Rate_Percent_per_Min_HTE': hte_ramp_down_pct_min,
        'pElectrolyzer_Breakpoints_MW_HTE': p_elec_breakpoints_str,
        'ke_H2_Values_MWh_per_kg_HTE': hte_ke_h2_values_str,
        'kt_H2_Values_MWth_per_kg_HTE': hte_kt_h2_values_str,
        'vom_electrolyzer_USD_per_MWh_HTE': hte_vom_usd_per_mwh,
        'cost_electrolyzer_capacity_USD_per_MW_year_HTE': hte_capex_annual_usd_per_mw_year,
        'cost_startup_electrolyzer_USD_per_startup_HTE': hte_cost_startup_usd,
        'MinUpTimeElectrolyzer_hours_HTE': hte_min_uptime_hr,
        'MinDownTimeElectrolyzer_hours_HTE': hte_min_downtime_hr,
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_HTE': hte_degradation_op_units_per_hr,
        'DegradationFactorStartup_Units_per_Startup_HTE': hte_degradation_su_units,
        'H2_storage_capacity_max_kg': 100000.0, 'H2_storage_capacity_min_kg': 1000.0,
        'H2_storage_level_initial_kg': 50000.0, 'H2_storage_charge_rate_max_kg_per_hr': 5000.0,
        'H2_storage_discharge_rate_max_kg_per_hr': 5000.0, 'storage_charge_eff_fraction': 0.98,
        'storage_discharge_eff_fraction': 0.98, 'vom_storage_cycle_USD_per_kg_cycled': 0.01,
        'BatteryCapacity_min_MWh': 10.0, 'BatteryCapacity_max_MWh': 1000.0,
        'BatteryPowerRatio_MW_per_MWh': batt_power_ratio, 'BatteryChargeEff': battery_charge_eff,
        'BatteryDischargeEff': battery_discharge_eff, 'BatterySOC_min_fraction': battery_soc_min_fraction,
        'BatterySOC_initial_fraction': 0.50, 'BatteryRequireCyclicSOC': True,
        'BatteryRampRate_fraction_per_hour': 1.0,
        'BatteryCapex_USD_per_MWh_year': batt_capex_usd_per_mwh_year, # Annualized energy capacity cost
        'BatteryCapex_USD_per_MW_year': batt_capex_usd_per_mw_year,   # Annualized power capacity cost
        'BatteryFixedOM_USD_per_MWh_year': batt_fom_usd_per_mwh_year, # Annualized fixed O&M per MWh energy capacity
        'user_specified_electrolyzer_capacity_MW': '',
        'user_specified_battery_power_MW': '',
        'user_specified_battery_energy_MWh': '',
    }
    df = pd.DataFrame({'Parameter': system_params.keys(), 'Value': system_params.values()})
    df.set_index('Parameter', inplace=True)
    df.to_csv(output_path, float_format='%.15g')


# --- 2. Generate Hourly Price Data (Price_hourly.csv) ---
def generate_price_hourly(output_path: Path, num_hours=HOURS_IN_YEAR):
    """Generates the hourly energy price (LMP) CSV file."""
    print(f"Generating hourly energy prices: {output_path}")
    start_time = datetime.datetime(2023, 1, 1)
    time_index = pd.to_datetime([start_time + datetime.timedelta(hours=i) for i in range(num_hours)])
    df_price = pd.DataFrame(index=time_index)

    base_lmp = 45
    peak_factor = 1.5 + 0.5 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / HOURS_IN_YEAR), num_hours))
    daily_variation = 20 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / 24), num_hours))
    weekly_variation = 5 * np.sin(np.linspace(0, 2 * np.pi * (num_hours / (24 * 7)), num_hours))
    random_noise = np.random.normal(0, 8, num_hours)
    spike_prob = 0.01
    spike_magnitude = np.random.uniform(50, 200, num_hours)
    spikes = np.random.choice([0, 1], size=num_hours, p=[1 - spike_prob, spike_prob]) * spike_magnitude
    lmp = base_lmp * peak_factor + daily_variation + weekly_variation + random_noise + spikes
    lmp = np.maximum(5, lmp)

    df_price['Price ($/MWh)'] = lmp
    df_price.index.name = 'Timestamp'
    df_price.to_csv(output_path)
    print(f"Hourly energy prices saved to {output_path}")


# --- 3. Generate Hourly AS Price Data (Price_ANS_hourly.csv) ---
def generate_price_ans_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS capacity price and adder CSV file."""
    print(f"Generating hourly AS prices/adders for {iso}: {output_path}")
    df_ans = pd.DataFrame(index=lmp_series.index)
    iso_services = ISO_SERVICE_MAP[iso] # Use the corrected map

    for service_key in iso_services:
        # Construct standardized column names, e.g., p_RegUp_PJM, loc_SR_SPP
        price_col_name = f'p_{service_key}_{iso}'
        loc_col_name = f'loc_{service_key}_{iso}'

        # Generate Capacity Price (MCP)
        base_mcp = np.maximum(2, lmp_series * np.random.uniform(0.05, 0.4, num_hours))
        mcp_noise = np.random.normal(0, 3, num_hours)
        mcp_spike_prob = 0.02
        mcp_spike_magnitude = np.random.uniform(10, 50, num_hours)
        mcp_spikes = np.random.choice([0, 1], size=num_hours, p=[1 - mcp_spike_prob, mcp_spike_prob]) * mcp_spike_magnitude
        mcp = np.maximum(0, base_mcp + mcp_noise + mcp_spikes)
        df_ans[price_col_name] = np.round(mcp, 4)

        # Generate Locational Adder
        adder_mask = np.random.rand(num_hours) < 0.05
        adder_values = np.random.uniform(0.1, 2, num_hours) * adder_mask
        df_ans[loc_col_name] = np.round(adder_values, 4)

    df_ans.index.name = 'Timestamp'
    df_ans.to_csv(output_path)
    print(f"Hourly AS prices/adders for {iso} saved to {output_path}")


# --- 4. Generate Hourly Deployment Factor Data (DeploymentFactor_hourly.csv) ---
def generate_deploy_factor_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly deployment factor CSV file for reserve services."""
    print(f"Generating hourly deployment factors for {iso}: {output_path}")
    df_deploy = pd.DataFrame(index=lmp_series.index)
    iso_services = ISO_SERVICE_MAP[iso] # Use the corrected map
    generated_factors = False

    for service_key in iso_services:
        # Deployment factors are typically for reserves, not regulation
        is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                "RegUp" in service_key or "RegDown" in service_key
        if is_regulation_service:
            continue # Skip for regulation services

        factor_col_name = f'deploy_factor_{service_key}_{iso}'
        high_lmp_threshold = lmp_series.quantile(0.8)
        deploy_prob = np.where(lmp_series > high_lmp_threshold, 0.15, 0.05)
        deploy_mask = np.random.rand(num_hours) < deploy_prob
        deploy_magnitude = np.random.uniform(0.01, 0.6, num_hours) # Max 60% deployment
        deploy_values = deploy_magnitude * deploy_mask
        df_deploy[factor_col_name] = np.round(deploy_values, 4)
        generated_factors = True
    
    if not generated_factors and not df_deploy.empty: # Ensure we don't save an empty DF if no reserve services
        df_deploy.index.name = 'Timestamp'
        df_deploy.to_csv(output_path)
        print(f"Hourly deployment factors for {iso} saved to {output_path}")
    elif not generated_factors and df_deploy.empty:
        print(f"No reserve deployment factors to generate for {iso}. Skipping file save for {output_path.name}")
    elif generated_factors: # Save if any factor was generated
        df_deploy.index.name = 'Timestamp'
        df_deploy.to_csv(output_path)
        print(f"Hourly deployment factors for {iso} saved to {output_path}")


# --- 5. Generate Hourly Mileage/Performance Factor Data (MileageMultiplier_hourly.csv) ---
def generate_mileage_multiplier_hourly(output_path: Path, iso: str, num_hours=HOURS_IN_YEAR):
    """Generates hourly mileage and performance factors for regulation services."""
    print(f"Generating hourly mileage/performance factors for {iso}: {output_path}")
    df_mileage_perf = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=num_hours, freq='h'))
    iso_services = ISO_SERVICE_MAP[iso] # Use the corrected map
    generated_factors = False

    for service_key in iso_services:
        # These factors apply to regulation services
        is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                "RegUp" in service_key or "RegDown" in service_key
        if is_regulation_service:
            mileage_col_name = f'mileage_factor_{service_key}_{iso}'
            perf_col_name = f'performance_factor_{service_key}_{iso}'

            # Generate mileage factor (e.g., values around 1.0 to 2.5, can be higher for some ISOs)
            mileage_values = np.maximum(0.5, 1.5 + np.random.normal(0, 0.3, num_hours))
            df_mileage_perf[mileage_col_name] = np.round(mileage_values, 4)

            # Generate performance factor (e.g., values around 0.8 to 1.0)
            perf_values = np.clip(0.95 + np.random.normal(0, 0.05, num_hours), 0.7, 1.0) # Typically high
            df_mileage_perf[perf_col_name] = np.round(perf_values, 4)
            generated_factors = True

    if not generated_factors and df_mileage_perf.empty: # Check if DF is still empty
        print(f"No regulation mileage/performance factors to generate for {iso}. Skipping file save for {output_path.name}")
        return
    
    df_mileage_perf.index.name = 'Timestamp'
    df_mileage_perf.to_csv(output_path)
    print(f"Hourly mileage/performance factors for {iso} saved to {output_path}")


# --- 6. Generate Hourly Winning Rate Data (WinningRate_hourly.csv) ---
def generate_winning_rate_hourly(output_path: Path, iso: str, lmp_series: pd.Series, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS winning rate CSV file for all services."""
    print(f"Generating hourly AS winning rates for {iso}: {output_path}")
    df_winrate = pd.DataFrame(index=lmp_series.index)
    iso_services = ISO_SERVICE_MAP[iso] # Use the corrected map

    for service_key in iso_services:
        rate_col_name = f'winning_rate_{service_key}_{iso}'
        # Simulate higher winning probability when LMP is low (less competition for AS?)
        low_lmp_threshold = lmp_series.quantile(0.2) 
        base_rate = np.where(lmp_series < low_lmp_threshold, 0.9, 0.7) 
        rate_noise = np.random.normal(0, 0.1, num_hours)
        win_rates = np.clip(base_rate + rate_noise, 0.1, 1.0) # Ensure rates are between 0.1 and 1.0
        df_winrate[rate_col_name] = np.round(win_rates, 4)

    df_winrate.index.name = 'Timestamp'
    df_winrate.to_csv(output_path)
    print(f"Hourly AS winning rates for {iso} saved to {output_path}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    system_data_path = BASE_OUTPUT_DIR / "sys_data_advanced.csv"
    generate_system_data(system_data_path)

    for target_iso in ISO_SERVICE_MAP.keys():
        print(f"\n--- Generating hourly files for {target_iso} ---")
        iso_output_dir = BASE_OUTPUT_DIR / target_iso
        iso_output_dir.mkdir(parents=True, exist_ok=True)

        price_hourly_path = iso_output_dir / "Price_hourly.csv"
        generate_price_hourly(price_hourly_path, num_hours=HOURS_IN_YEAR)
        try:
            lmp_series = pd.read_csv(
                price_hourly_path, index_col='Timestamp', parse_dates=True)['Price ($/MWh)']
        except Exception as e:
            print(f"ERROR: Failed to load generated LMP for {target_iso}: {e}. Skipping.")
            continue

        price_ans_path = iso_output_dir / "Price_ANS_hourly.csv"
        generate_price_ans_hourly(price_ans_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

        deploy_factor_path = iso_output_dir / "DeploymentFactor_hourly.csv"
        generate_deploy_factor_hourly(deploy_factor_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

        mileage_perf_path = iso_output_dir / "MileageMultiplier_hourly.csv" 
        generate_mileage_multiplier_hourly(mileage_perf_path, target_iso, num_hours=HOURS_IN_YEAR)

        winrate_path = iso_output_dir / "WinningRate_hourly.csv"
        generate_winning_rate_hourly(winrate_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

    print("\n--- Data generation complete for all ISOs ---")
