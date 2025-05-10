"""
Shared test fixtures for the optimization framework tests.
"""

import pytest
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path

# Add src directory to Python path if running tests from a different location
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))


@pytest.fixture
def sample_hourly_data_for_iso(request):
    """
    Creates sample hourly price and AS data for testing, adaptable by ISO.
    Uses TARGET_ISO from config by default but can be overridden by test request.
    """
    from config import TARGET_ISO as DEFAULT_TARGET_ISO
    target_iso = getattr(request, "param", {}).get(
        "target_iso", DEFAULT_TARGET_ISO)
    num_hours = 24  # For sample data

    price_data_dict = {
        'hour': range(1, num_hours + 1),
        'Price ($/MWh)': [50.0 + i % 5 for i in range(num_hours)],  # LMP
    }
    # Add some AS price columns dynamically based on a simplified map
    # These names should match what data_io.py expects and model.py creates params for.
    as_price_cols = {
        'ERCOT': [f'p_RegU_{target_iso}', f'p_RegD_{target_iso}', f'p_SR_{target_iso}', f'p_NSR_{target_iso}', f'p_ECRS_{target_iso}',
                  f'loc_RegU_{target_iso}', f'loc_SR_{target_iso}'],
        'SPP':   [f'p_RegU_{target_iso}', f'p_RegD_{target_iso}', f'p_Spin_{target_iso}', f'p_Sup_{target_iso}',
                  f'loc_RegU_{target_iso}'],
        'CAISO': [f'p_RegU_{target_iso}', f'p_RegD_{target_iso}', f'p_Spin_{target_iso}', f'p_NSpin_{target_iso}',
                  f'loc_RegU_{target_iso}'],
        # Add other ISOs as needed for comprehensive testing
    }
    selected_as_cols = as_price_cols.get(target_iso, [
                                         f'p_RegU_{target_iso}', f'loc_RegU_{target_iso}'])  # Default if ISO not in map

    for i, col_name in enumerate(selected_as_cols):
        price_data_dict[col_name] = [(10.0 + i*2 + j % 3)
                                     for j in range(num_hours)]

    # For factors, mileage, winning rates - create minimal dataframes
    # These columns might also be ISO specific
    factor_cols_map = {
        'ERCOT': [f'deploy_factor_SR_{target_iso}', f'winning_rate_SR_{target_iso}', f'mileage_factor_RegU_{target_iso}'],
        'SPP':   [f'deploy_factor_Spin_{target_iso}', f'winning_rate_Spin_{target_iso}'],
        'CAISO': [f'deploy_factor_Spin_{target_iso}', f'winning_rate_Spin_{target_iso}', f'mileage_factor_RegU_{target_iso}']
    }
    selected_factor_cols = factor_cols_map.get(
        target_iso, [f'deploy_factor_RegU_{target_iso}'])

    ans_mile_data = {col: [1.0 - (i*0.01) % 0.1 for i in range(num_hours)]
                     for col in selected_factor_cols if 'mileage' in col or 'score' in col or 'ratio' in col}
    if not ans_mile_data:
        # ensure dataframe not empty
        ans_mile_data = {'hour': range(1, num_hours+1)}
    else:
        ans_mile_data['hour'] = range(1, num_hours+1)

    ans_deploy_data = {col: [0.5 + (i*0.02) % 0.2 for i in range(num_hours)]
                       for col in selected_factor_cols if 'deploy_factor' in col}
    if not ans_deploy_data:
        ans_deploy_data = {'hour': range(1, num_hours+1)}
    else:
        ans_deploy_data['hour'] = range(1, num_hours+1)

    ans_winrate_data = {col: [0.8 + (i*0.01) % 0.15 for i in range(num_hours)]
                        for col in selected_factor_cols if 'winning_rate' in col}
    if not ans_winrate_data:
        ans_winrate_data = {'hour': range(1, num_hours+1)}
    else:
        ans_winrate_data['hour'] = range(1, num_hours+1)

    return {
        "df_price_hourly": pd.DataFrame(price_data_dict),
        # Assuming AS prices are in the same file for simplicity here
        "df_ANSprice_hourly": pd.DataFrame(price_data_dict),
        "df_ANSmile_hourly": pd.DataFrame(ans_mile_data) if ans_mile_data else pd.DataFrame({'hour': range(1, num_hours + 1)}),
        "df_ANSdeploy_hourly": pd.DataFrame(ans_deploy_data) if ans_deploy_data else pd.DataFrame({'hour': range(1, num_hours + 1)}),
        "df_ANSwinrate_hourly": pd.DataFrame(ans_winrate_data) if ans_winrate_data else pd.DataFrame({'hour': range(1, num_hours + 1)})
    }


@pytest.fixture
def sample_system_data():
    """Create comprehensive sample system data for testing."""
    parameters = [
        # General
        'delT_minutes', 'AS_Duration', 'plant_lifetime_years',
        'pIES_min_MW', 'pIES_max_MW',
        # Nuclear
        'qSteam_Total_MWth', 'qSteam_Turbine_min_MWth', 'qSteam_Turbine_max_MWth',
        'pTurbine_min_MW', 'pTurbine_max_MW',
        'Turbine_RampUp_Rate_Percent_per_Min', 'Turbine_RampDown_Rate_Percent_per_Min',
        'vom_turbine_USD_per_MWh', 'Turbine_Thermal_Elec_Efficiency_Const',
        'qSteam_Turbine_Breakpoints_MWth', 'pTurbine_Outputs_at_Breakpoints_MW',
        'Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour', 'pTurbine_LTE_setpoint_MW',
        # Electrolyzer (Common & HTE/LTE specific examples)
        'hydrogen_subsidy_per_kg', 'aux_power_consumption_per_kg_h2', 'H2_value_USD_per_kg',
        'pElectrolyzer_max_upper_bound_MW', 'pElectrolyzer_max_lower_bound_MW',
        'pElectrolyzer_min_MW_HTE', 'Electrolyzer_RampUp_Rate_Percent_per_Min_HTE',
        'Electrolyzer_RampDown_Rate_Percent_per_Min_HTE',
        'vom_electrolyzer_USD_per_MWh_HTE', 'cost_water_USD_per_kg_h2',
        'cost_electrolyzer_ramping_USD_per_MW_ramp_HTE', 'cost_electrolyzer_capacity_USD_per_MW_year_HTE',
        'pElectrolyzer_Breakpoints_MW_HTE', 'ke_H2_Values_MWh_per_kg_HTE', 'kt_H2_Values_MWth_per_kg_HTE',
        'pElectrolyzer_min_MW_LTE', 'Electrolyzer_RampUp_Rate_Percent_per_Min_LTE',
        'Electrolyzer_RampDown_Rate_Percent_per_Min_LTE',
        'vom_electrolyzer_USD_per_MWh_LTE',
        'cost_electrolyzer_ramping_USD_per_MW_ramp_LTE', 'cost_electrolyzer_capacity_USD_per_MW_year_LTE',
        'pElectrolyzer_Breakpoints_MW_LTE', 'ke_H2_Values_MWh_per_kg_LTE',
        # Electrolyzer SU/SD & Degradation (can be type-specific or generic)
        'cost_startup_electrolyzer_USD_per_startup_HTE', 'MinUpTimeElectrolyzer_hours_HTE', 'MinDownTimeElectrolyzer_hours_HTE',
        'uElectrolyzer_initial_status_0_or_1', 'DegradationStateInitial_Units',
        'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_HTE', 'DegradationFactorStartup_Units_per_Startup_HTE',
        'h2_target_capacity_factor_fraction',
        # H2 Storage
        'H2_storage_capacity_max_kg', 'H2_storage_capacity_min_kg', 'H2_storage_level_initial_kg',
        'H2_storage_charge_rate_max_kg_per_hr', 'H2_storage_discharge_rate_max_kg_per_hr',
        'storage_charge_eff_fraction', 'storage_discharge_eff_fraction', 'vom_storage_cycle_USD_per_kg_cycled',
        # Battery
        'BatteryCapacity_max_MWh', 'BatteryCapacity_min_MWh', 'BatteryPowerRatio_MW_per_MWh',
        'BatteryChargeEff', 'BatteryDischargeEff', 'BatterySOC_min_fraction', 'BatterySOC_initial_fraction',
        'BatteryRequireCyclicSOC', 'BatteryRampRate_fraction_per_hour',
        'BatteryCapex_USD_per_MWh_year', 'BatteryCapex_USD_per_MW_year',
        'BatteryFixedOM_USD_per_MWh_year', 'vom_battery_per_mwh_cycled'
    ]
    values = [
        # General
        '60', '0.25', '30',
        '-500', '500',  # pIES
        # Nuclear
        '1000', '200', '900',  # qSteam
        '80', '450',  # pTurbine
        '2.0', '2.0',  # Ramp
        '2.0', '0.40',  # VOM, Eff
        '200,500,900', '80,200,360',  # Piecewise Turbine
        '500', '400',  # qSteam_Elec_Ramp, pTurbine_LTE_setpoint
        # Electrolyzer (Common & HTE/LTE)
        '0.5', '0.2', '3.0',  # subsidy, aux, H2_value
        '200', '0',  # pElec capacity bounds
        '20', '10.0', '10.0',  # pElec_min_HTE, Ramps HTE
        '1.5', '0.5',  # VOM HTE, water
        '0.1', '100000',  # Ramping cost HTE, Capacity cost HTE
        # p_bp_HTE, ke_HTE, kt_HTE (MWh/kg, MWth/kg)
        '0,50,100,150', '0.025,0.024,0.023', '0.010,0.009,0.008',
        '10', '15.0', '15.0',  # pElec_min_LTE, Ramps LTE
        '1.0',  # VOM LTE
        '0.05', '80000',  # Ramping cost LTE, Capacity cost LTE
        '0,40,80,120', '0.022,0.021,0.020',  # p_bp_LTE, ke_LTE (MWh/kg)
        # Electrolyzer SU/SD & Degradation
        '500', '2', '1',  # startup_cost_HTE, up_time_HTE, down_time_HTE
        '0', '0',  # initial_status, initial_degradation
        '0.001', '5',  # deg_op_HTE, deg_startup_HTE
        '0.0',  # target_cap_factor (usually 0 if optimising)
        # H2 Storage
        '10000', '500', '2000',  # H2_store capacity, min, initial
        '1000', '1000',  # H2_store charge/discharge rate
        '0.98', '0.95', '0.01',  # H2_store eff_charge, eff_discharge, vom_cycle
        # Battery
        '400', '0', '0.25',  # Batt_cap_max, min, power_ratio
        '0.95', '0.95', '0.1', '0.5',  # Batt eff_charge, eff_discharge, soc_min, soc_initial
        'True', '1.0',  # Batt cyclic, ramp_rate
        '20000', '15000',  # Batt capex_energy, capex_power
        '500', '0.5'  # Batt FOM, VOM
    ]
    # Check length consistency
    if len(parameters) != len(values):
        raise ValueError(
            f"Parameter list length ({len(parameters)}) and value list length ({len(values)}) mismatch in test_conftest.py")

    return pd.DataFrame({'Parameter': parameters, 'Value': values}).set_index('Parameter')


@pytest.fixture
def sample_model_config(request):
    """Fixture to manage config overrides for model creation."""
    from unittest.mock import patch
    import config as app_config  # import the actual config module

    default_config = {
        "ENABLE_NUCLEAR_GENERATOR": app_config.ENABLE_NUCLEAR_GENERATOR,
        "ENABLE_ELECTROLYZER": app_config.ENABLE_ELECTROLYZER,
        "ENABLE_LOW_TEMP_ELECTROLYZER": app_config.ENABLE_LOW_TEMP_ELECTROLYZER,
        "ENABLE_BATTERY": app_config.ENABLE_BATTERY,
        "ENABLE_H2_STORAGE": app_config.ENABLE_H2_STORAGE,
        "ENABLE_STARTUP_SHUTDOWN": app_config.ENABLE_STARTUP_SHUTDOWN,
        "ENABLE_ELECTROLYZER_DEGRADATION_TRACKING": app_config.ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
        "ENABLE_H2_CAP_FACTOR": app_config.ENABLE_H2_CAP_FACTOR,
        "ENABLE_NONLINEAR_TURBINE_EFF": app_config.ENABLE_NONLINEAR_TURBINE_EFF,
        "SIMULATE_AS_DISPATCH_EXECUTION": app_config.SIMULATE_AS_DISPATCH_EXECUTION,
        "TARGET_ISO": app_config.TARGET_ISO,
        # CAN_PROVIDE_ANCILLARY_SERVICES is derived, so we patch its constituents
    }

    # Override defaults with parameters passed from the test
    config_overrides = getattr(request, "param", {})
    current_config = {**default_config, **config_overrides}

    patched_objects = []
    for key, value in current_config.items():
        try:
            p = patch(f'config.{key}', value)
            patched_objects.append(p)
            p.start()
        except AttributeError:
            print(
                f"Warning: Could not patch config.{key}, it might not exist directly or needs module path.")

    # Re-calculate CAN_PROVIDE_ANCILLARY_SERVICES based on (potentially patched) flags
    # This is tricky because it's derived at module import.
    # A cleaner way is to make CAN_PROVIDE_ANCILLARY_SERVICES a function or ensure model.py re-evaluates it.
    # For the purpose of the model knowing, we ensure the constituent flags are set.
    # The model object `m` will have these flags set directly.

    yield current_config  # The patched config dict

    for p in reversed(patched_objects):
        p.stop()


@pytest.fixture
def sample_model(sample_hourly_data_for_iso, sample_system_data, sample_model_config):
    """
    Create a sample model for testing.
    `sample_model_config` (indirectly parameterized by tests) will apply config patches.
    `sample_hourly_data_for_iso` can also be parameterized for ISO-specific data.
    """
    from model import create_model
    # sample_model_config fixture has already patched the config module's flags

    data_inputs = {**sample_hourly_data_for_iso,
                   "df_system": sample_system_data}

    # The create_model function will use the patched config values
    # The simulate_dispatch flag for create_model comes from the (patched) config now
    model = create_model(data_inputs,
                         target_iso=sample_model_config["TARGET_ISO"],
                         simulate_dispatch=sample_model_config["SIMULATE_AS_DISPATCH_EXECUTION"])
    return model
