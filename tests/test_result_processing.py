"""
Unit tests for result_processing.py module.
Tests result processing and analysis functions.
"""

import unittest
import sys
import os
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config as app_config
from model import create_model
from result_processing import extract_results, calculate_hourly_as_revenue, get_total_deployed_as

# Path for test outputs
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class TestResultProcessing(unittest.TestCase):
    """Test cases for result processing and analysis."""

    def _create_base_data_inputs(self, iso="ERCOT", num_hours=3):
        price_data_dict = {'hour': range(1, num_hours + 1), 'Price ($/MWh)': [50.0] * num_hours}
        as_price_cols = {}
        if iso == "ERCOT":
            as_price_cols = {
                f'p_RegU_{iso}': [10]*num_hours, f'p_RegD_{iso}': [8]*num_hours,
                f'p_SR_{iso}': [5]*num_hours, f'p_NSR_{iso}': [4]*num_hours, f'p_ECRS_{iso}': [6]*num_hours,
                f'loc_RegU_{iso}': [1]*num_hours, f'loc_SR_{iso}': [0.5]*num_hours,
                f'winning_rate_RegU_{iso}': [0.9]*num_hours, f'winning_rate_SR_{iso}': [0.8]*num_hours,
                f'deploy_factor_SR_{iso}': [0.5]*num_hours # For reserves
            }
        # Add other ISOs as needed for comprehensive testing
        price_data_dict.update(as_price_cols)

        parameters = [
            'delT_minutes', 'AS_Duration', 'plant_lifetime_years', 'pIES_min_MW', 'pIES_max_MW',
            'qSteam_Total_MWth', 'qSteam_Turbine_min_MWth','pTurbine_max_MW', 'pTurbine_min_MW',
            'pElectrolyzer_max_upper_bound_MW', 'pElectrolyzer_min_MW', # Added generic pElectrolyzer_min_MW
            'pElectrolyzer_Breakpoints_MW', # Added Generic Breakpoint Fallback
            'pElectrolyzer_min_MW_LTE', 'ke_H2_Values_MWh_per_kg_LTE', 'pElectrolyzer_Breakpoints_MW_LTE',
            'cost_electrolyzer_capacity_USD_per_MW_year_LTE', 'vom_electrolyzer_USD_per_MWh_LTE',
            'H2_value_USD_per_kg', 'hydrogen_subsidy_per_kg', 'cost_water_USD_per_kg_h2',
            'vom_turbine_USD_per_MWh', 'aux_power_consumption_per_kg_h2',
            'cost_startup_electrolyzer_USD_per_startup_LTE', 'cost_electrolyzer_ramping_USD_per_MW_ramp_LTE',
            'BatteryCapacity_max_MWh', 'BatteryCapacity_min_MWh', 'BatteryPowerRatio_MW_per_MWh', 'vom_battery_per_mwh_cycled',
            'BatteryCapex_USD_per_MWh_year', 'BatteryFixedOM_USD_per_MWh_year', 'BatteryCapex_USD_per_MW_year',
            'H2_storage_capacity_max_kg', 'H2_storage_capacity_min_kg', 'vom_storage_cycle_USD_per_kg_cycled',
            'uElectrolyzer_initial_status_0_or_1', # For startup cost
            # Add any other params required by revenue/cost/summary logic in extract_results
             'Turbine_Thermal_Elec_Efficiency_Const', 'qSteam_Turbine_max_MWth',
             'Electrolyzer_RampUp_Rate_Percent_per_Min_LTE', 'Electrolyzer_RampDown_Rate_Percent_per_Min_LTE',
             'BatteryChargeEff', 'BatteryDischargeEff', 'BatterySOC_min_fraction', 'BatterySOC_initial_fraction', 'BatteryRequireCyclicSOC', 'BatteryRampRate_fraction_per_hour',
             'H2_storage_level_initial_kg', 'H2_storage_charge_rate_max_kg_per_hr', 'H2_storage_discharge_rate_max_kg_per_hr', 'storage_charge_eff_fraction', 'storage_discharge_eff_fraction',
             'DegradationStateInitial_Units', 'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE', 'DegradationFactorStartup_Units_per_Startup_LTE', # For Degradation Summary
             'pElectrolyzer_max_lower_bound_MW', 'MinUpTimeElectrolyzer_hours_LTE', 'MinDownTimeElectrolyzer_hours_LTE', # For SU/SD Summary
             'ke_H2_Values_MWh_per_kg_HTE', 'kt_H2_Values_MWth_per_kg_HTE', 'pElectrolyzer_Breakpoints_MW_HTE', 'pElectrolyzer_min_MW_HTE' # For completeness if testing HTE
        ]
        values = [
            '60', '0.25', '30', '-100', '100', # General
            '1000', '200', '450', '50', # Nuke
            '100', '0', # Elec UB/LB, Generic Min
            '0,50,100', # Generic Breakpoint Value
            '5', '0.022,0.021', '0,50,100', # LTE Min, ke, bp
            '1000', '1.0', # Elec capex, vom
            '2.0', '0.5', '0.1', # H2 val, subsidy, water
            '2.0', '0.1', # Nuke VOM, Aux
            '10', '0.05', # Elec SU cost, ramp cost
            '100', '0', '0.25', '0.5', # Batt cap, ratio, vom
            '100', '10', '10', # Batt capex, fom
            '1000', '0', '0.01', # H2 stor cap, vom
            '0', # initial status for startup
            '0.4', '900', # More nuke
            '10', '10', # Elec ramps
            '0.9', '0.9', '0.1', '0.5', 'True', '1.0', # More batt
            '100', '100', '100', '1.0', '1.0', # More H2 stor
            '0', '0.001', '1', # Deg
            '0', '1', '1', # SU/SD summary
            '0.025', '0.010', '0,60', '10' # HTE completion
        ]
        if len(parameters) != len(values):
            raise ValueError(f"Mismatch in test_result_processing parameter/value lengths ({len(parameters)} vs {len(values)}).")

        system_data = pd.DataFrame({'Parameter': parameters, 'Value': values}).set_index('Parameter')
        return {
            'df_price_hourly': pd.DataFrame(price_data_dict),
            'df_ANSprice_hourly': pd.DataFrame(price_data_dict),
            'df_ANSmile_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[1.0]*num_hours for k in as_price_cols if 'RegU' in k}}), # mileage for RegU
            'df_ANSdeploy_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[0.5]*num_hours for k in as_price_cols if 'SR' in k or 'Spin' in k}}),# deploy for SR/Spin
            'df_ANSwinrate_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[0.9]*num_hours for k in as_price_cols}}), # win rate for all
            'df_system': system_data
        }

    def _setup_model_and_fix_vars(self, target_iso="ERCOT", simulate_dispatch=False, num_hours=3, **config_flags):
        """Helper to create a model, fix variables to mock a solved state."""
        # Default flags for components needed for comprehensive result processing
        default_component_flags = {
            'ENABLE_NUCLEAR_GENERATOR': True, 'ENABLE_ELECTROLYZER': True,
            'ENABLE_LOW_TEMP_ELECTROLYZER': True, # Use LTE for simpler setup often
            'ENABLE_BATTERY': True, 'ENABLE_H2_STORAGE': True,
            'ENABLE_STARTUP_SHUTDOWN': True,
            'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING': True,
            'TARGET_ISO': target_iso,
        }
        effective_flags = {**default_component_flags, **config_flags}

        with patch.multiple('config', **effective_flags):
            importlib.reload(app_config)
            data_inputs = self._create_base_data_inputs(iso=target_iso, num_hours=num_hours)
            model = create_model(data_inputs, target_iso, simulate_dispatch=simulate_dispatch)

            # Fix variables to simulate a solved model
            # These are arbitrary values for testing calculations
            for t_idx, t in enumerate(model.TimePeriods):
                t_val = t_idx + 1
                if hasattr(model, 'pIES'): model.pIES[t].value = 20 + t_val
                if hasattr(model, 'pTurbine'): model.pTurbine[t].value = 60 + t_val
                if hasattr(model, 'qSteam_Turbine'): model.qSteam_Turbine[t].value = (60 + t_val) / 0.4 # Approx based on const eff
                if hasattr(model, 'pElectrolyzer'): model.pElectrolyzer[t].value = 30 + t_val
                if hasattr(model, 'pElectrolyzerSetpoint'): model.pElectrolyzerSetpoint[t].value = 30 + t_val # Base for AS
                if hasattr(model, 'mHydrogenProduced') and hasattr(model, 'ke_H2_values') and model.pElectrolyzer_efficiency_breakpoints:
                     # Find efficiency at the setpoint
                     # Simplified: use efficiency at first breakpoint
                     eff_key = model.pElectrolyzer_efficiency_breakpoints.first() # e.g., 0
                     mwh_per_kg = pyo.value(model.ke_H2_values.get(eff_key, 0.022)) # Use default if key missing
                     if mwh_per_kg > 1e-6 :
                          # Estimate H2 prod based on ACTUAL power pElectrolyzer
                          model.mHydrogenProduced[t].value = model.pElectrolyzer[t].value / mwh_per_kg
                     else: model.mHydrogenProduced[t].value = 0

                if hasattr(model, 'pAuxiliary') and model.pAuxiliary.is_indexed() and hasattr(model, 'mHydrogenProduced'):
                    model.pAuxiliary[t].value = model.mHydrogenProduced[t].value * pyo.value(model.aux_power_consumption_per_kg_h2) / 1000.0 if model.mHydrogenProduced[t].value is not None else 0

                if hasattr(model, 'H2_storage_level'): model.H2_storage_level[t].value = 100 + t_val*10
                if hasattr(model, 'H2_to_market'): model.H2_to_market[t].value = 5 + t_val
                if hasattr(model, 'H2_from_storage'): model.H2_from_storage[t].value = 2 + t_val*0.5
                if hasattr(model, 'H2_to_storage'): model.H2_to_storage[t].value = model.mHydrogenProduced[t].value - model.H2_to_market[t].value if model.mHydrogenProduced[t].value is not None else 0 # Input based on prod - market

                if hasattr(model, 'uElectrolyzer'): model.uElectrolyzer[t].value = 1
                if hasattr(model, 'vElectrolyzerStartup'): model.vElectrolyzerStartup[t].value = 1 if t == 1 else 0
                if hasattr(model, 'wElectrolyzerShutdown'): model.wElectrolyzerShutdown[t].value = 0
                if hasattr(model, 'DegradationState'): model.DegradationState[t].value = t_val * 0.1

                if hasattr(model, 'pElectrolyzerRampPos') and model.pElectrolyzerRampPos.is_indexed():
                    model.pElectrolyzerRampPos[t].value = 1 if t_idx > 0 else 0
                if hasattr(model, 'pElectrolyzerRampNeg') and model.pElectrolyzerRampNeg.is_indexed():
                    model.pElectrolyzerRampNeg[t].value = 0


                if hasattr(model, 'BatterySOC'): model.BatterySOC[t].value = 50 + t_val*5
                if hasattr(model, 'BatteryCharge'): model.BatteryCharge[t].value = 10 + t_val
                if hasattr(model, 'BatteryDischarge'): model.BatteryDischarge[t].value = 3 + t_val
                if hasattr(model, 'BatteryBinaryCharge'): model.BatteryBinaryCharge[t].value = 1
                if hasattr(model, 'BatteryBinaryDischarge'): model.BatteryBinaryDischarge[t].value = 0

                # Fix AS bid variables (example for ERCOT)
                if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                    # Use getattr to safely access potentially missing AS variables
                    regup_bid_var = getattr(model, 'Total_RegUp', None)
                    if regup_bid_var is not None and regup_bid_var.is_indexed(): regup_bid_var[t].value = 5 + t_val*0.1
                    regup_elec_var = getattr(model, 'RegUp_Electrolyzer', None)
                    if regup_elec_var is not None and regup_elec_var.is_indexed(): regup_elec_var[t].value = 2 + t_val*0.05
                    regup_turb_var = getattr(model, 'RegUp_Turbine', None)
                    if regup_turb_var is not None and regup_turb_var.is_indexed(): regup_turb_var[t].value = 1 + t_val*0.02
                    regup_batt_var = getattr(model, 'RegUp_Battery', None)
                    if regup_batt_var is not None and regup_batt_var.is_indexed(): regup_batt_var[t].value = 2 + t_val*0.03

                    sr_bid_var = getattr(model, 'Total_SR', None)
                    if sr_bid_var is not None and sr_bid_var.is_indexed(): sr_bid_var[t].value = 10 + t_val*0.2
                    sr_elec_var = getattr(model, 'SR_Electrolyzer', None)
                    if sr_elec_var is not None and sr_elec_var.is_indexed(): sr_elec_var[t].value = 10 + t_val*0.2


                    # If in dispatch simulation mode, also fix deployed variables
                    if simulate_dispatch:
                        regup_elec_dep_var = getattr(model, 'RegUp_Electrolyzer_Deployed', None)
                        if regup_elec_dep_var is not None and regup_elec_dep_var.is_indexed() and regup_elec_var is not None and regup_elec_var.is_indexed():
                            win_rate_param = getattr(model, f"winning_rate_RegU_{target_iso}", None)
                            win_rate_val = pyo.value(win_rate_param[t]) if win_rate_param and win_rate_param.is_indexed() and t in win_rate_param else 1.0
                            regup_elec_dep_var[t].value = regup_elec_var[t].value * win_rate_val

                        sr_elec_dep_var = getattr(model, 'SR_Electrolyzer_Deployed', None)
                        if sr_elec_dep_var is not None and sr_elec_dep_var.is_indexed() and sr_elec_var is not None and sr_elec_var.is_indexed():
                            win_rate_sr_param = getattr(model, f"winning_rate_SR_{target_iso}", None)
                            deploy_factor_sr_param = getattr(model, f"deploy_factor_SR_{target_iso}", None)
                            win_rate_sr = pyo.value(win_rate_sr_param[t]) if win_rate_sr_param and win_rate_sr_param.is_indexed() and t in win_rate_sr_param else 1.0
                            deploy_factor_sr = pyo.value(deploy_factor_sr_param[t]) if deploy_factor_sr_param and deploy_factor_sr_param.is_indexed() and t in deploy_factor_sr_param else 0.0
                            sr_elec_dep_var[t].value = sr_elec_var[t].value * win_rate_sr * deploy_factor_sr


            # Fix capacity variables (these are not indexed by time)
            elec_max_var = getattr(model, 'pElectrolyzer_max', None)
            if elec_max_var is not None: elec_max_var.value = 80 # Smaller than UB for testing
            batt_cap_var = getattr(model, 'BatteryCapacity_MWh', None)
            if batt_cap_var is not None: batt_cap_var.value = 90 # Smaller than UB
            batt_pow_var = getattr(model, 'BatteryPower_MW', None)
            batt_pow_ratio_param = getattr(model, 'BatteryPowerRatio', None)
            if batt_pow_var is not None and batt_cap_var is not None and batt_pow_ratio_param is not None:
                 batt_pow_var.value = batt_cap_var.value * pyo.value(batt_pow_ratio_param)

            # Mock the objective value if the attribute exists
            if hasattr(model, 'TotalProfit_Objective'):
                model.del_component(model.TotalProfit_Objective) # Remove old one if exists
            model.TotalProfit_Objective = pyo.Objective(expr=1000.0) # Mock objective

        return model

    def test_extract_results_bidding_mode(self):
        """Test extract_results in bidding strategy mode."""
        iso = "ERCOT"
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=False)
        df_results, summary = extract_results(model, target_iso=iso, output_dir=str(TEST_OUTPUT_DIR))

        self.assertIsInstance(df_results, pd.DataFrame)
        self.assertIsInstance(summary, dict)
        self.assertFalse(df_results.empty)

        # Check for key columns (examples)
        self.assertIn('pIES_MW', df_results.columns)
        self.assertIn('pElectrolyzer_MW', df_results.columns)
        self.assertIn('Revenue_Energy_USD', df_results.columns)
        self.assertIn('Cost_HourlyOpex_Total_USD', df_results.columns)
        self.assertIn('Profit_Hourly_USD', df_results.columns)
        if model.CAN_PROVIDE_ANCILLARY_SERVICES and hasattr(model, 'Total_RegUp'):
            self.assertIn('Total_RegUp_Bid_MW', df_results.columns) # Bid column
        # Deployed vars should not be processed into columns named this way in bidding mode
        self.assertNotIn('Total_RegUp_Deployed_MW', df_results.columns)


        # Check summary keys
        self.assertIn('Total_Revenue_USD', summary)
        self.assertIn('Total_Profit_Calculated_USD', summary)
        # Check which capacity key exists based on model setup (Var vs Param)
        if isinstance(getattr(model,'pElectrolyzer_max', None), pyo.Var):
             self.assertIn('Optimal_Electrolyzer_Capacity_MW', summary)
        else:
             self.assertIn('Fixed_Electrolyzer_Capacity_MW', summary)

        self.assertEqual(summary['Simulation_Mode'], 'Bidding Strategy')

        # Check a calculated value (example, requires careful setup of fixed vars and params)
        # Energy Revenue for first hour: pIES[1] * energy_price[1] * time_factor
        # model.pIES[1].value = 21, energy_price = 50, time_factor = 1
        if hasattr(model, 'pIES') and hasattr(model, 'energy_price'):
            expected_energy_rev_t1 = model.pIES[1].value * pyo.value(model.energy_price[1]) * 1.0
            self.assertAlmostEqual(df_results['Revenue_Energy_USD'].iloc[0], expected_energy_rev_t1, places=2)

        # Test file creation
        self.assertTrue((TEST_OUTPUT_DIR / f'{iso}_Hourly_Results_Comprehensive.csv').exists())
        self.assertTrue((TEST_OUTPUT_DIR / f'{iso}_Summary_Results.txt').exists())


    def test_extract_results_dispatch_simulation_mode(self):
        """Test extract_results in AS dispatch simulation mode."""
        iso = "ERCOT"
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=True)
        df_results, summary = extract_results(model, target_iso=iso, output_dir=str(TEST_OUTPUT_DIR))

        self.assertIsInstance(df_results, pd.DataFrame)
        self.assertFalse(df_results.empty)
        self.assertEqual(summary['Simulation_Mode'], 'Dispatch Execution')

        # Check for deployed AS columns (example for RegUp Electrolyzer)
        if model.CAN_PROVIDE_ANCILLARY_SERVICES and hasattr(model, 'RegUp_Electrolyzer') and hasattr(model, 'RegUp_Electrolyzer_Deployed'):
            self.assertIn('RegUp_Electrolyzer_Bid_MW', df_results.columns)
            self.assertIn('RegUp_Electrolyzer_Deployed_MW', df_results.columns)
            # Value check based on fixed vars:
            # RegUp_Electrolyzer_Deployed[1] should be fixed based on Bid * WinRate
            expected_deployed_val = model.RegUp_Electrolyzer_Deployed[1].value
            actual_deployed_val = df_results['RegUp_Electrolyzer_Deployed_MW'].iloc[0]
            self.assertAlmostEqual(actual_deployed_val, expected_deployed_val, places=3,
                                 msg=f"Mismatch: Expected {expected_deployed_val}, got {actual_deployed_val}")


        self.assertIn('Total_Deployed_RegUp_MWh', summary) # Summary stat for deployed AS

    def test_calculate_hourly_as_revenue_ercot_bidding(self):
        """Test hourly AS revenue calculation for ERCOT in bidding mode."""
        iso = "ERCOT"
        num_hours = 1
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=False, num_hours=num_hours)
        t = model.TimePeriods.first()

        # Relevant prices for ERCOT (from _create_base_data_inputs setup for ERCOT @ t=1)
        p_regu = pyo.value(getattr(model, f'p_RegU_{iso}')[t]) # 10
        loc_regu = pyo.value(getattr(model, f'loc_RegU_{iso}')[t]) # 1
        win_regu = pyo.value(getattr(model, f'winning_rate_RegU_{iso}')[t]) # 0.9
        p_sr = pyo.value(getattr(model, f'p_SR_{iso}')[t]) # 5
        loc_sr = pyo.value(getattr(model, f'loc_SR_{iso}')[t]) # 0.5
        win_sr_param = getattr(model, f"winning_rate_SR_{iso}", None)
        if win_sr_param and win_sr_param.is_indexed() and t in win_sr_param: win_sr = pyo.value(win_sr_param[t], default=0.8)
        else: win_sr = 0.8 # Fallback if param missing

        deploy_sr = pyo.value(getattr(model, f'deploy_factor_SR_{iso}')[t]) # 0.5
        lmp = pyo.value(model.energy_price[t]) # 50

        # Fixed bids from _setup_model_and_fix_vars @ t=1
        bid_regu = model.Total_RegUp[t].value # 5.1
        bid_sr = model.Total_SR[t].value # 10.2

        expected_regu_rev = (bid_regu * win_regu * p_regu) + (bid_regu * 1.0 * 1.0 * lmp) + (loc_regu) # Cap + Perf (Mileage*Perf*LMP) + Adder
        expected_sr_rev = (bid_sr * win_sr * p_sr) + (bid_sr * deploy_sr * lmp) + (loc_sr)    # Cap + Energy (DeployFactor*LMP) + Adder
        expected_total_hourly_as_rev_rate = expected_regu_rev + expected_sr_rev

        hourly_rev_rate = calculate_hourly_as_revenue(model, t)
        self.assertAlmostEqual(hourly_rev_rate, expected_total_hourly_as_rev_rate, places=2)


    def test_calculate_hourly_as_revenue_ercot_dispatch_sim(self):
        """Test hourly AS revenue calculation for ERCOT in dispatch simulation mode."""
        iso = "ERCOT"
        num_hours = 1
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=True, num_hours=num_hours)
        t = model.TimePeriods.first()

        # Prices @ t=1
        p_regu = pyo.value(getattr(model, f'p_RegU_{iso}')[t]) # 10
        loc_regu = pyo.value(getattr(model, f'loc_RegU_{iso}')[t]) # 1
        win_regu = pyo.value(getattr(model, f'winning_rate_RegU_{iso}')[t]) # 0.9
        p_sr = pyo.value(getattr(model, f'p_SR_{iso}')[t]) # 5
        loc_sr = pyo.value(getattr(model, f'loc_SR_{iso}')[t]) # 0.5
        win_sr_param = getattr(model, f"winning_rate_SR_{iso}", None)
        if win_sr_param and win_sr_param.is_indexed() and t in win_sr_param: win_sr = pyo.value(win_sr_param[t], default=0.8)
        else: win_sr = 0.8
        lmp = pyo.value(model.energy_price[t]) # 50

        # Fixed bids @ t=1
        bid_regu = model.Total_RegUp[t].value # 5.1
        bid_sr = model.Total_SR[t].value # 10.2

        # Fixed deployed values (assuming only electrolyzer contributes for simplicity here)
        # Need to ensure deployed vars are fixed in _setup_model_and_fix_vars
        deployed_regu = model.RegUp_Electrolyzer_Deployed[t].value # Should be 2.05 * 0.9 = 1.845 -> Updated setup: 2.1*0.9 = 1.89
        deployed_sr = model.SR_Electrolyzer_Deployed[t].value # Should be 10.2 * 0.8 * 0.5 = 4.08

        # Correct fixed values from setup:
        bid_regu = 5 + 1*0.1 # 5.1
        bid_sr = 10 + 1*0.2 # 10.2
        bid_regu_elec = 2 + 1*0.05 # 2.05
        bid_sr_elec = 10 + 1*0.2 # 10.2
        deployed_regu = bid_regu_elec * win_regu # 2.05 * 0.9 = 1.845
        deployed_sr = bid_sr_elec * win_sr * deploy_sr # 10.2 * 0.8 * 0.5 = 4.08


        # Mock get_total_deployed_as to return the sum of fixed deployed vars for the service
        with patch('result_processing.get_total_deployed_as') as mock_get_total_deployed:
            # This mock reflects that only Electrolyzer is providing these in our _setup fix
            mock_get_total_deployed.side_effect = lambda m, time_index, service_name: \
                (deployed_regu if service_name == 'RegUp' and time_index == t else
                 (deployed_sr if service_name == 'SR' and time_index == t else 0.0))


            # RegU: Cap based on Bid, Perf based on Deployed * LMP
            cap_regu = bid_regu * win_regu * p_regu
            perf_regu = deployed_regu * lmp # Deployed_RegU * LMP
            adder_regu = loc_regu
            expected_regu_rev = cap_regu + perf_regu + adder_regu

            # SR: Cap based on Bid, Energy based on Deployed * LMP
            cap_sr = bid_sr * win_sr * p_sr
            energy_sr = deployed_sr * lmp # Deployed_SR * LMP
            adder_sr = loc_sr
            expected_sr_rev = cap_sr + energy_sr + adder_sr

            expected_total_hourly_as_rev_rate = expected_regu_rev + expected_sr_rev

            hourly_rev_rate = calculate_hourly_as_revenue(model, t)
            self.assertAlmostEqual(hourly_rev_rate, expected_total_hourly_as_rev_rate, places=2)

    def test_hydrogen_revenue_split_in_results(self):
        """Test that hydrogen revenue is split into sales and subsidy in results."""
        iso = "ERCOT"
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=False, ENABLE_H2_STORAGE=True)
        t = model.TimePeriods.first()

        # Values from _setup_model_and_fix_vars & _create_base_data_inputs
        pElec_t1 = model.pElectrolyzer[t].value # 31
        eff_key = model.pElectrolyzer_efficiency_breakpoints.first() # 0
        mwh_per_kg = pyo.value(model.ke_H2_values.get(eff_key, 0.022)) # 0.022
        mHydrogenProduced_t1_val = pElec_t1 / mwh_per_kg if mwh_per_kg > 1e-6 else 0 # Approx 1409
        # Use the value fixed in setup for consistency if it was calculated there
        mHydrogenProduced_t1_val = model.mHydrogenProduced[t].value # Fixed to 5*t_val = 5

        H2_value = pyo.value(model.H2_value) # 2.0 -> Corrected to 3.0 from data setup
        H2_value = 3.0
        hydrogen_subsidy_per_kg = pyo.value(model.hydrogen_subsidy_per_kg) # 0.5 -> Corrected to 0.6 from data setup
        hydrogen_subsidy_per_kg = 0.6
        H2_to_market_t1 = model.H2_to_market[t].value # 5 + 1 = 6 -> Corrected to 2*t_val = 2
        H2_to_market_t1 = 2
        H2_from_storage_t1 = model.H2_from_storage[t].value # 2 + 0.5 = 2.5 -> Corrected to 1*t_val = 1
        H2_from_storage_t1 = 1
        time_factor = 1.0

        df_results, summary = extract_results(model, target_iso=iso, output_dir=str(TEST_OUTPUT_DIR))

        self.assertIn('Revenue_Hydrogen_Sales_USD', df_results.columns)
        self.assertIn('Revenue_Hydrogen_Subsidy_USD', df_results.columns)
        self.assertIn('Revenue_Hydrogen_USD', df_results.columns)

        expected_sales_t1 = (H2_to_market_t1 + H2_from_storage_t1) * H2_value * time_factor # (2+1)*3.0*1 = 9.0
        expected_subsidy_t1 = mHydrogenProduced_t1_val * hydrogen_subsidy_per_kg * time_factor # 5 * 0.6 * 1 = 3.0
        expected_total_h2_rev_t1 = expected_sales_t1 + expected_subsidy_t1 # 9.0 + 3.0 = 12.0

        self.assertAlmostEqual(df_results['Revenue_Hydrogen_Sales_USD'].iloc[0], expected_sales_t1, places=1)
        self.assertAlmostEqual(df_results['Revenue_Hydrogen_Subsidy_USD'].iloc[0], expected_subsidy_t1, places=1)
        self.assertAlmostEqual(df_results['Revenue_Hydrogen_USD'].iloc[0], expected_total_h2_rev_t1, places=1)

        self.assertAlmostEqual(summary['Total_Hydrogen_Sales_Revenue_USD'], df_results['Revenue_Hydrogen_Sales_USD'].sum(), places=1)
        self.assertAlmostEqual(summary['Total_Hydrogen_Subsidy_Revenue_USD'], df_results['Revenue_Hydrogen_Subsidy_USD'].sum(), places=1)

    def tearDown(self):
        """Clean up any created files after each test (optional)."""
        # Clean up files in TEST_OUTPUT_DIR
        for f_path in TEST_OUTPUT_DIR.glob("*"):
            try:
                if f_path.is_file():
                    f_path.unlink()
            except OSError as e:
                print(f"Error removing test output file {f_path}: {e}")
        pass


if __name__ == '__main__':
    unittest.main()