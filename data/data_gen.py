"""Data generation utility for nuclear-hydrogen optimization model.

This module generates synthetic data for testing and validation of the
nuclear-hydrogen optimization model, including price data, system parameters,
and ancillary service data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import math  # Needed for CRF calculation if converting total CAPEX

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("../input/hourly_data")
HOURS_IN_YEAR = 8760
np.random.seed(42)  # Consistent randomness

# --- ISO Service Definitions (Match model.py/data_io.py needs) ---
ISO_SERVICE_MAP = {
    'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
    # Requires mileage_factor for RegU/D
    'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
    'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
    # Requires RegCap/Perf prices, perf_score, mileage_ratio factors
    'PJM': ['Reg', 'Syn', 'Rse', 'TMR', 'RegCap', 'RegPerf', 'performance_score', 'mileage_ratio'],
    'NYISO': ['RegC', 'Spin10', 'NSpin10', 'Res30'],
    'ISONE': ['Spin10', 'NSpin10', 'OR30'],
    'MISO': ['Reg', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
}


# --- Helper Function ---
def calculate_crf(discount_rate, lifetime_years):
    """Calculates the Capital Recovery Factor (used for annualizing CAPEX)."""
    if lifetime_years <= 0:
        return 0
    if discount_rate == 0:
        return 1.0 / lifetime_years if lifetime_years > 0 else 0
    try:
        factor = (1 + discount_rate)**lifetime_years
        # Handle potential division by zero if factor is exactly 1 (lifetime=0 or rate=-1)
        if abs(factor - 1.0) < 1e-9:
            # This case should be caught by lifetime_years <= 0 check, but added for safety
            return 0 if lifetime_years <= 0 else (1.0 / lifetime_years if discount_rate == 0 else 0)
        return (discount_rate * factor) / (factor - 1)
    except (OverflowError, ValueError):
        print(
            f"Warning: CRF calculation failed for rate={discount_rate}, life={lifetime_years}. Returning rate as approximation.")
        return discount_rate  # Approximation for long lifetimes or errors


# --- 1. Generate System Parameters (sys_data_advanced.csv) ---
def generate_system_data(output_path: Path):
    """Generates the COMMON system parameters CSV file, including HTE/LTE specific data."""
    print(
        f"Generating common system parameters (including HTE/LTE): {output_path}")

    # --- Base Parameters ---
    p_turbine_max_mw = 380.0
    q_turbine_max_mwth = 1000.0
    p_elec_max_capacity_mw = 500.0  # Upper bound for optimization
    elec_min_load_fraction = 0.10  # Min load fraction (generic)
    p_elec_min_load_mw = p_elec_max_capacity_mw * \
        elec_min_load_fraction  # Calculate the value
    discount_rate = 0.08  # General discount rate
    # Common assumption for LCOH analysis base
    electrolyzer_economic_lifetime_years = 20
    
    # --- Hydrogen Subsidy Parameters ---
    hydrogen_subsidy_duration_years = 10  # Default subsidies last 10 years

    # --- Common Electrolyzer Settings ---
    p_elec_breakpoints_mw = [p_elec_min_load_mw, 100.0,
                             250.0, 500.0]  # Start breakpoint at min load
    p_elec_breakpoints_str = ', '.join(map(str, p_elec_breakpoints_mw))
    cost_water_usd_per_kg_h2 = 0.03  # From previous estimate based on doc
    uElectrolyzer_initial_status_0_or_1 = 0
    DegradationStateInitial_Units = 0.0
    H2_value_USD_per_kg = 3.0
    hydrogen_subsidy_per_kg_val = 3.00  # Max 45V PTC
    h2_target_capacity_factor_fraction = 0.0
    # kWh/kg (Estimate for compressor + BOP)
    aux_power_consumption_per_kg_h2_val = 3.5

    # --- LTE (PEM Representative) Parameters ---
    print("... Defining LTE (PEM) parameters...")
    lte_capex_usd_per_kw = 2000.0  # Based on DOE Hub data
    lte_lifetime_years = 20  # Assume matches economic life for CRF calc
    lte_crf = calculate_crf(discount_rate, lte_lifetime_years)
    lte_capex_annual_usd_per_mw_year = lte_capex_usd_per_kw * 1000.0 * lte_crf
    # Efficiency (kWh/kg) - Define efficiency at breakpoints
    # Eff at min_load, 100, 250, 500 MW
    lte_efficiency_kwh_per_kg = [57.0, 56.0, 55.0, 54.0]
    lte_ke_h2_mwh_per_kg = [e / 1000.0 for e in lte_efficiency_kwh_per_kg]
    lte_ke_h2_values_str = ', '.join(
        map(lambda x: f"{x:.5f}", lte_ke_h2_mwh_per_kg))
    lte_kt_h2_values_str = ', '.join(
        ['0.00000'] * len(p_elec_breakpoints_mw))  # Zero for LTE
    lte_vom_usd_per_mwh = 8.0  # Refined estimate
    lte_cost_startup_usd = 300.0  # Estimate
    lte_ramp_up_pct_min = 10.0  # Estimate
    lte_ramp_down_pct_min = 10.0  # Estimate
    lte_min_uptime_hr = 2  # Estimate
    lte_min_downtime_hr = 1  # Estimate
    lte_degradation_op_units_per_hr = 0.0001  # Estimate (Lower)
    lte_degradation_su_units = 1.0  # Estimate

    # --- HTE (SOEC Representative) Parameters ---
    print("... Defining HTE (SOEC) parameters...")
    hte_capex_usd_per_kw = 2500.0  # Higher current cost estimate within $917-$4k range
    # Assume matches economic life for CRF calc (though stack life shorter)
    hte_lifetime_years = 20
    hte_crf = calculate_crf(discount_rate, hte_lifetime_years)
    hte_capex_annual_usd_per_mw_year = hte_capex_usd_per_kw * 1000.0 * hte_crf
    # Efficiency (kWh/kg) - Electrical component, defined at breakpoints
    # Eff at min_load, 100, 250, 500 MW
    hte_efficiency_kwh_per_kg = [42.0, 41.0, 40.0, 39.0]
    hte_ke_h2_mwh_per_kg = [e / 1000.0 for e in hte_efficiency_kwh_per_kg]
    hte_ke_h2_values_str = ', '.join(
        map(lambda x: f"{x:.5f}", hte_ke_h2_mwh_per_kg))
    # kt_H2: Representing thermal input ~15 kWh/kg -> 0.015 MWth/(kg/hr)
    hte_kt_h2_mwh_per_kg = [0.015] * \
        len(p_elec_breakpoints_mw)  # Using constant value
    hte_kt_h2_values_str = ', '.join(
        map(lambda x: f"{x:.5f}", hte_kt_h2_mwh_per_kg))
    hte_vom_usd_per_mwh = 10.0  # Refined estimate
    hte_cost_startup_usd = 700.0  # Estimate (Higher startup cost)
    hte_ramp_up_pct_min = 5.0  # Estimate (Slower ramps)
    hte_ramp_down_pct_min = 5.0  # Estimate
    hte_min_uptime_hr = 4  # Estimate (Longer preferred)
    hte_min_downtime_hr = 2  # Estimate
    hte_degradation_op_units_per_hr = 0.0003  # Estimate (Higher degradation)
    hte_degradation_su_units = 1.5  # Estimate

    # --- Battery Parameters (Keep as before) ---
    print("... Defining Battery parameters...")
    battery_lifetime = 15  # NREL ATB assumption
    batt_crf = calculate_crf(discount_rate, battery_lifetime)
    batt_duration_hours = 4.0
    # CAPEX ($/kWh) - Using average US turnkey cost for 2024
    batt_capex_mwh_total_usd_per_kwh = 236.0  # $/kWh
    batt_capex_mwh_total = batt_capex_mwh_total_usd_per_kwh * 1000.0
    batt_capex_mw_total_usd_per_kw = batt_capex_mwh_total_usd_per_kwh * batt_duration_hours
    batt_capex_mw_total = batt_capex_mw_total_usd_per_kw * 1000.0
    batt_capex_usd_per_mwh_year = batt_capex_mwh_total * batt_crf
    batt_capex_usd_per_mw_year = batt_capex_mw_total * batt_crf
    # Fixed O&M ($/MWh/year) - Based on NREL ATB (2.5% of $/kW CAPEX per year)
    fom_fraction_of_capex_per_kw = 0.025
    batt_fom_usd_per_mw_year = fom_fraction_of_capex_per_kw * batt_capex_mw_total
    batt_power_ratio = 1.0 / batt_duration_hours
    batt_fom_mwh_yr = batt_fom_usd_per_mw_year * batt_power_ratio
    battery_charge_eff = 0.92  # sqrt(RTE ~85%)
    battery_discharge_eff = 0.92
    battery_soc_min_fraction = 0.10

    # --- Assemble Dictionary ---
    system_params = {
        # General
        'delT_minutes': 60.0,
        'AS_Duration': 0.25,
        'plant_lifetime_years': 30,
        'discount_rate': discount_rate,
        'pIES_min_MW': -1000.0,
        'pIES_max_MW': 1000.0,

        # Nuclear Generator (NPP)
        'qSteam_Total_MWth': q_turbine_max_mwth,
        'qSteam_Turbine_min_MWth': 100.0,
        'qSteam_Turbine_max_MWth': q_turbine_max_mwth,
        'pTurbine_min_MW': 38.0,
        'pTurbine_max_MW': p_turbine_max_mw,
        'Turbine_RampUp_Rate_Percent_per_Min': 2.0,
        'Turbine_RampDown_Rate_Percent_per_Min': 2.0,
        'pTurbine_LTE_setpoint_MW': p_turbine_max_mw,
        'Turbine_Thermal_Elec_Efficiency_Const': 0.38,
        'qSteam_Turbine_Breakpoints_MWth': '100.0, 550.0, 1000.0',
        'pTurbine_Outputs_at_Breakpoints_MW': '38.0, 209.0, 380.0',
        'vom_turbine_USD_per_MWh': 2.0,

        # Electrolyzer - Common/Generic
        'pElectrolyzer_max_upper_bound_MW': p_elec_max_capacity_mw,
        'pElectrolyzer_max_lower_bound_MW': 0.0,
        'pElectrolyzer_min_MW': p_elec_min_load_mw,  # Generic min power
        'pElectrolyzer_Breakpoints_MW': p_elec_breakpoints_str,  # Generic Breakpoints
        # Generic ke (use LTE as default)
        'ke_H2_Values_MWh_per_kg': lte_ke_h2_values_str,
        # Generic kt (use LTE as default=0)
        'kt_H2_Values_MWth_per_kg': lte_kt_h2_values_str,
        'cost_water_USD_per_kg_h2': cost_water_usd_per_kg_h2,
        'uElectrolyzer_initial_status_0_or_1': uElectrolyzer_initial_status_0_or_1,
        'DegradationStateInitial_Units': DegradationStateInitial_Units,
        'h2_target_capacity_factor_fraction': h2_target_capacity_factor_fraction,
        'hydrogen_subsidy_value_usd_per_kg': hydrogen_subsidy_per_kg_val,
        'hydrogen_subsidy_duration_years': hydrogen_subsidy_duration_years,  
        'aux_power_consumption_per_kg_h2': aux_power_consumption_per_kg_h2_val,
        'H2_value_USD_per_kg': H2_value_USD_per_kg,
        # Generic, can be overridden by specific if needed
        'cost_electrolyzer_ramping_USD_per_MW_ramp': 0.5,
        # Steam ramp limit (relevant for HTE model logic)
        'Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour': 500.0,

        # Electrolyzer - LTE Specific
        'pElectrolyzer_min_MW_LTE': p_elec_min_load_mw,  # Value is same as generic here
        'Electrolyzer_RampUp_Rate_Percent_per_Min_LTE': lte_ramp_up_pct_min,
        'Electrolyzer_RampDown_Rate_Percent_per_Min_LTE': lte_ramp_down_pct_min,
        'pElectrolyzer_Breakpoints_MW_LTE': p_elec_breakpoints_str,
        'ke_H2_Values_MWh_per_kg_LTE': lte_ke_h2_values_str,
        'kt_H2_Values_MWth_per_kg_LTE': lte_kt_h2_values_str,  # Zero
        'vom_electrolyzer_USD_per_MWh_LTE': lte_vom_usd_per_mwh,
        'cost_electrolyzer_capacity_USD_per_MW_year_LTE': lte_capex_annual_usd_per_mw_year,
        'cost_startup_electrolyzer_USD_per_startup_LTE': lte_cost_startup_usd,
        'MinUpTimeElectrolyzer_hours_LTE': lte_min_uptime_hr,
        'MinDownTimeElectrolyzer_hours_LTE': lte_min_downtime_hr,
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE': lte_degradation_op_units_per_hr,
        'DegradationFactorStartup_Units_per_Startup_LTE': lte_degradation_su_units,

        # Electrolyzer - HTE Specific
        'pElectrolyzer_min_MW_HTE': p_elec_min_load_mw,  # Value is same as generic here
        'Electrolyzer_RampUp_Rate_Percent_per_Min_HTE': hte_ramp_up_pct_min,
        'Electrolyzer_RampDown_Rate_Percent_per_Min_HTE': hte_ramp_down_pct_min,
        'pElectrolyzer_Breakpoints_MW_HTE': p_elec_breakpoints_str,  # Using same breakpoints
        'ke_H2_Values_MWh_per_kg_HTE': hte_ke_h2_values_str,
        # Now non-zero based on derivation
        'kt_H2_Values_MWth_per_kg_HTE': hte_kt_h2_values_str,
        'vom_electrolyzer_USD_per_MWh_HTE': hte_vom_usd_per_mwh,
        'cost_electrolyzer_capacity_USD_per_MW_year_HTE': hte_capex_annual_usd_per_mw_year,
        'cost_startup_electrolyzer_USD_per_startup_HTE': hte_cost_startup_usd,
        'MinUpTimeElectrolyzer_hours_HTE': hte_min_uptime_hr,
        'MinDownTimeElectrolyzer_hours_HTE': hte_min_downtime_hr,
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_HTE': hte_degradation_op_units_per_hr,
        'DegradationFactorStartup_Units_per_Startup_HTE': hte_degradation_su_units,

        # H2 Storage
        'H2_storage_capacity_max_kg': 100000.0,
        'H2_storage_capacity_min_kg': 1000.0,
        'H2_storage_level_initial_kg': 50000.0,
        'H2_storage_charge_rate_max_kg_per_hr': 5000.0,
        'H2_storage_discharge_rate_max_kg_per_hr': 5000.0,
        'storage_charge_eff_fraction': 0.98,
        'storage_discharge_eff_fraction': 0.98,
        'vom_storage_cycle_USD_per_kg_cycled': 0.01,

        # Battery Storage
        'BatteryCapacity_min_MWh': 10.0,
        'BatteryCapacity_max_MWh': 1000.0,
        'BatteryPowerRatio_MW_per_MWh': batt_power_ratio,
        'BatteryChargeEff': battery_charge_eff,
        'BatteryDischargeEff': battery_discharge_eff,
        'BatterySOC_min_fraction': battery_soc_min_fraction,
        'BatterySOC_initial_fraction': 0.50,
        'BatteryRequireCyclicSOC': True,
        'BatteryRampRate_fraction_per_hour': 1.0,
        'BatteryCapex_USD_per_MWh_year': batt_capex_usd_per_mwh_year,
        'BatteryCapex_USD_per_MW_year': batt_capex_usd_per_mw_year,
        'BatteryFixedOM_USD_per_MWh_year': batt_fom_mwh_yr,

        
        'user_specified_electrolyzer_capacity_MW': '',  
        'user_specified_battery_power_MW': '',          
        'user_specified_battery_energy_MWh': '',        
    }

    # Now write out using Pandas for consistent formatting/quoting behavior
    df = pd.DataFrame({'Parameter': system_params.keys(),
                       'Value': system_params.values()})
    df.set_index('Parameter', inplace=True)

    # Write with consistent floating point format
    df.to_csv(output_path, float_format='%.15g')


# --- 2. Generate Hourly Price Data (Price_hourly.csv) ---
def generate_price_hourly(output_path: Path, num_hours=HOURS_IN_YEAR):
    """Generates the hourly energy price (LMP) CSV file."""
    print(f"Generating hourly energy prices: {output_path}")
    start_time = datetime.datetime(2023, 1, 1)
    time_index = pd.to_datetime(
        [start_time + datetime.timedelta(hours=i) for i in range(num_hours)])
    df_price = pd.DataFrame(index=time_index)

    # Generate base energy price (LMP) - slightly more realistic pattern
    base_lmp = 45
    peak_factor = 1.5 + 0.5 * \
        np.sin(np.linspace(0, 2 * np.pi * (num_hours / HOURS_IN_YEAR),
               num_hours))  # Simple seasonality
    daily_variation = 20 * \
        np.sin(np.linspace(0, 2 * np.pi * (num_hours / 24), num_hours))  # Daily cycle
    weekly_variation = 5 * \
        np.sin(np.linspace(0, 2 * np.pi * (num_hours / (24*7)),
               num_hours))  # Weekly cycle
    # More noise, occasional spikes
    random_noise = np.random.normal(0, 8, num_hours)
    spike_prob = 0.01
    spike_magnitude = np.random.uniform(50, 200, num_hours)
    spikes = np.random.choice([0, 1], size=num_hours, p=[
                              1-spike_prob, spike_prob]) * spike_magnitude
    lmp = base_lmp * peak_factor + daily_variation + \
        weekly_variation + random_noise + spikes
    lmp = np.maximum(5, lmp)  # Floor price

    df_price['Price ($/MWh)'] = lmp  # Column name expected by data_io.py

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
        if any(f in service for f in ['factor', 'score', 'ratio']):
            continue
        # PJM Reg has separate Cap/Perf prices
        if iso == 'PJM' and service == 'Reg':
            continue  # Handled by RegCap/RegPerf

        param_col_pattern = f"{service}_{iso}"
        price_col_name = f'p_{param_col_pattern}'
        loc_col_name = f'loc_{param_col_pattern}'

        # Generate Capacity Price (MCP) - correlated with LMP but lower, add noise/spikes
        base_mcp = np.maximum(
            2, lmp_series * np.random.uniform(0.05, 0.4, num_hours))
        mcp_noise = np.random.normal(0, 3, num_hours)
        mcp_spike_prob = 0.02
        mcp_spike_magnitude = np.random.uniform(10, 50, num_hours)
        mcp_spikes = np.random.choice([0, 1], size=num_hours, p=[
                                      1-mcp_spike_prob, mcp_spike_prob]) * mcp_spike_magnitude
        mcp = np.maximum(0, base_mcp + mcp_noise + mcp_spikes)  # Floor at 0

        # Handle PJM specific RegCap/RegPerf if service='RegCap'/'RegPerf'
        if iso == 'PJM' and service == 'RegCap':
            df_ans[f'p_RegCap_{iso}'] = mcp  # Use generated MCP for RegCap
        elif iso == 'PJM' and service == 'RegPerf':
            df_ans[f'p_RegPerf_{iso}'] = np.maximum(
                0, mcp * np.random.uniform(0.5, 1.5, num_hours))  # Perf price related to cap price
        else:
            df_ans[price_col_name] = mcp

        # Generate Locational Adder (mostly zero, small positive value sometimes)
        # 5% chance of non-zero adder
        adder_mask = np.random.rand(num_hours) < 0.05
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
        if service in ['RegCap', 'RegC', 'RegPerf', 'performance_score', 'mileage_ratio']:
            continue

        factor_col_name = f'deploy_factor_{service}_{iso}'

        # Simulate higher deployment probability/magnitude during high LMP periods?
        high_lmp_threshold = lmp_series.quantile(0.8)
        deploy_prob = np.where(lmp_series > high_lmp_threshold, 0.15, 0.05)
        deploy_mask = np.random.rand(num_hours) < deploy_prob
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
    df_mileage = pd.DataFrame(index=pd.date_range(
        start='2023-01-01', periods=num_hours, freq='h'))

    if iso == 'CAISO':
        for service in ['RegU', 'RegD']:
            values = np.maximum(0.5, 1.0 + np.random.normal(0, 0.1, num_hours))
            df_mileage[f'mileage_factor_{service}_{iso}'] = np.round(values, 4)
    elif iso == 'PJM':
        perf_values = np.maximum(
            0.7, 0.95 + np.random.normal(0, 0.05, num_hours))
        df_mileage[f'performance_score_{iso}'] = np.round(perf_values, 4)
        mileage_values = np.maximum(
            0.5, 1.2 + np.random.normal(0, 0.2, num_hours))
        df_mileage[f'mileage_ratio_{iso}'] = np.round(mileage_values, 4)
    else:
        print(f"No specific mileage/performance factors generated for {iso}")
        if not df_mileage.columns.any():
            print(f"Skipping generation of empty {output_path}")
            return

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
        if any(f in service for f in ['factor', 'score', 'ratio']):
            continue
        rate_col_name = f'winning_rate_{service}_{iso}'
        low_lmp_threshold = lmp_series.quantile(0.2)
        base_rate = np.where(lmp_series < low_lmp_threshold, 0.9, 0.7)
        rate_noise = np.random.normal(0, 0.1, num_hours)
        win_rates = np.clip(base_rate + rate_noise, 0.1, 1.0)
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
    try:
        lmp_series = pd.read_csv(
            price_hourly_path, index_col='Timestamp', parse_dates=True)['Price ($/MWh)']
    except Exception as e:
        print(
            f"ERROR: Failed to load generated LMP data for {target_iso}: {e}. Skipping remaining hourly files for this ISO.")
        continue

    # Generate Price_ANS_hourly.csv
    price_ans_path = iso_output_dir / "Price_ANS_hourly.csv"
    generate_price_ans_hourly(
        price_ans_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

    # Generate DeploymentFactor_hourly.csv (Optional)
    deploy_factor_path = iso_output_dir / "DeploymentFactor_hourly.csv"
    generate_deploy_factor_hourly(
        deploy_factor_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

    # Generate MileageMultiplier_hourly.csv (Optional)
    mileage_path = iso_output_dir / "MileageMultiplier_hourly.csv"
    generate_mileage_multiplier_hourly(
        mileage_path, target_iso, num_hours=HOURS_IN_YEAR)

    # Generate WinningRate_hourly.csv (Optional)
    winrate_path = iso_output_dir / "WinningRate_hourly.csv"
    generate_winning_rate_hourly(
        winrate_path, target_iso, lmp_series, num_hours=HOURS_IN_YEAR)

print("\n--- Data generation complete for all ISOs ---")
