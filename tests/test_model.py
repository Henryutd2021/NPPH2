"""
Unit tests for model.py module.
Tests model creation and parameter loading.
"""

from model import create_model, get_sys_param
import config as app_config  # Use a distinct alias
import unittest
import sys
import os
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))

# from config import TARGET_ISO # TARGET_ISO will be patched or come from app_config

# Keep a reference to the original get_sys_param if mocking it later
original_get_sys_param = get_sys_param


class TestModel(unittest.TestCase):
    """Test cases for model creation and parameter loading."""

    def _create_base_data_inputs(self, iso="ERCOT"):
        # This is a simplified version. For full testing, it should be as comprehensive as in test_conftest.py
        num_hours = 3  # Fewer hours for faster test model creation
        price_data_dict = {'hour': range(
            1, num_hours + 1), 'Price ($/MWh)': [50.0] * num_hours}
        # Add essential AS price columns for the given ISO to avoid errors if AS is on
        if iso == "ERCOT":
            price_data_dict.update(
                {f'p_RegU_{iso}': [10]*num_hours, f'p_SR_{iso}': [5]*num_hours})
        elif iso == "SPP":
            price_data_dict.update(
                {f'p_RegU_{iso}': [10]*num_hours, f'p_Spin_{iso}': [5]*num_hours})
        # ... add other ISOs if needed for specific tests ...

        parameters = [
            'delT_minutes', 'AS_Duration', 'plant_lifetime_years', 'pIES_min_MW', 'pIES_max_MW',
            'qSteam_Total_MWth', 'qSteam_Turbine_min_MWth', 'qSteam_Turbine_max_MWth',
            'pTurbine_min_MW', 'pTurbine_max_MW',
            'Turbine_RampUp_Rate_Percent_per_Min', 'Turbine_RampDown_Rate_Percent_per_Min',
            'vom_turbine_USD_per_MWh', 'Turbine_Thermal_Elec_Efficiency_Const',
            'qSteam_Turbine_Breakpoints_MWth', 'pTurbine_Outputs_at_Breakpoints_MW',
            'pElectrolyzer_max_upper_bound_MW', 'pElectrolyzer_max_lower_bound_MW',
            'pElectrolyzer_min_MW',  # Added generic pElectrolyzer_min_MW as fallback
            'pElectrolyzer_Breakpoints_MW',  # Added Generic Breakpoint Fallback
            'pElectrolyzer_min_MW_LTE', 'ke_H2_Values_MWh_per_kg_LTE', 'pElectrolyzer_Breakpoints_MW_LTE', 'cost_electrolyzer_capacity_USD_per_MW_year_LTE',
            'pElectrolyzer_min_MW_HTE', 'ke_H2_Values_MWh_per_kg_HTE', 'kt_H2_Values_MWth_per_kg_HTE', 'pElectrolyzer_Breakpoints_MW_HTE', 'cost_electrolyzer_capacity_USD_per_MW_year_HTE',
            'H2_value_USD_per_kg', 'hydrogen_subsidy_per_kg', 'aux_power_consumption_per_kg_h2',
            'uElectrolyzer_initial_status_0_or_1', 'cost_startup_electrolyzer_USD_per_startup_LTE', 'MinUpTimeElectrolyzer_hours_LTE',
            'DegradationStateInitial_Units', 'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE',
            'BatteryCapacity_max_MWh', 'BatteryPowerRatio_MW_per_MWh', 'BatteryChargeEff', 'BatterySOC_initial_fraction',
            'H2_storage_capacity_max_kg', 'H2_storage_level_initial_kg', 'H2_storage_charge_rate_max_kg_per_hr', 'storage_charge_eff_fraction'
        ]
        values = [
            '60', '0.25', '30', '-100', '100',  # General
            '1000', '200', '900',  # Nuclear Steam
            '40', '450',  # Nuclear Power
            '2', '2', '2', '0.4',  # Nuc VOM, Ramps, Eff
            '100,500,900', '40,200,360',  # Nuc PWL
            '100', '0',  # Elec UB/LB
            '0',  # Generic pElec min
            '0,50,100',  # Generic Breakpoint Value
            '5', '0.022,0.021', '0,50,100', '1000',  # LTE Min, ke, bp, capex
            '10', '0.025,0.024', '0.010,0.009', '0,60,120', '1200',  # HTE Min, ke, kt, bp, capex
            '2.0', '0.0', '0.1',  # H2 val, subsidy, aux
            '0', '10', '1',  # SU/SD
            '0', '0.001',  # Deg
            '100', '0.25', '0.9', '0.5',  # Batt
            '1000', '100', '100', '1.0'  # H2 Stor
        ]
        if len(parameters) != len(values):
            raise ValueError(
                f"Mismatch in test_model setup parameter/value lengths ({len(parameters)} vs {len(values)}).")
        system_data = pd.DataFrame(
            {'Parameter': parameters, 'Value': values}).set_index('Parameter')

        # IMPORTANT: Need to ensure essential dataframes like df_price_hourly are present
        return {
            'df_price_hourly': pd.DataFrame(price_data_dict),
            'df_ANSprice_hourly': pd.DataFrame(price_data_dict),  # Simplified
            # Minimal
            'df_ANSmile_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            # Minimal
            'df_ANSdeploy_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            # Minimal
            'df_ANSwinrate_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            'df_system': system_data
        }

    def _setup_model_with_config(self, target_iso="ERCOT", simulate_dispatch=False, **kwargs_config_flags):
        """Helper to create a model with specific config flags patched."""
        default_flags = {
            'ENABLE_NUCLEAR_GENERATOR': True, 'ENABLE_ELECTROLYZER': True,
            'ENABLE_LOW_TEMP_ELECTROLYZER': False, 'ENABLE_BATTERY': False,
            'ENABLE_H2_STORAGE': False, 'ENABLE_STARTUP_SHUTDOWN': False,
            'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING': False, 'ENABLE_H2_CAP_FACTOR': False,
            'ENABLE_NONLINEAR_TURBINE_EFF': True, 'TARGET_ISO': target_iso,
        }
        effective_flags = {**default_flags, **kwargs_config_flags}

        with patch.multiple('config', **effective_flags):
            # Ensure config module reflects patches
            importlib.reload(app_config)
            data_inputs = self._create_base_data_inputs(iso=target_iso)
            model = create_model(data_inputs, target_iso,
                                 simulate_dispatch=simulate_dispatch)
        return model

    def test_get_sys_param_various_types(self):
        """Test get_sys_param with various data types and scenarios."""
        # Create sample data including price data to avoid error during model creation
        sample_data_sys = pd.DataFrame({
            # Added delT_minutes to fix the error in this specific test's data setup
            'Value': ['60.5', 'True', 'false', 'my_string', '100', '200,300', 'NaN', None, 'FALSE', '60']
        }, index=pd.Index(['float_param', 'bool_true_str', 'bool_false_str', 'string_param',
                           'int_param', 'bp_param', 'nan_param', 'none_param', 'bool_cap_false_str', 'delT_minutes'], name='Parameter'))
        sample_data_price = pd.DataFrame(
            {'Price ($/MWh)': [50.0]})  # Minimal price data

        sample_data_inputs = {
            'df_system': sample_data_sys,
            'df_price_hourly': sample_data_price,
            # Add other minimal required dataframes
            'df_ANSprice_hourly': sample_data_price,
            'df_ANSmile_hourly': pd.DataFrame({'hour': [1]}),
            'df_ANSdeploy_hourly': pd.DataFrame({'hour': [1]}),
            'df_ANSwinrate_hourly': pd.DataFrame({'hour': [1]}),
        }

        # get_sys_param uses a global df_system set by create_model.
        # We test it via a model instance after setting up minimal flags.
        with patch.multiple('config', ENABLE_NUCLEAR_GENERATOR=False, ENABLE_ELECTROLYZER=False, ENABLE_BATTERY=False):
            # This call should now succeed because delT_minutes is in sample_data_sys
            # This sets the global model.df_system
            model = create_model(sample_data_inputs,
                                 "ERCOT", simulate_dispatch=False)

            # Now test get_sys_param using the context set by the model
            # Use original_get_sys_param which points to the function in model.py
            self.assertEqual(original_get_sys_param('float_param'), 60.5)
            self.assertTrue(original_get_sys_param('bool_true_str'))
            self.assertFalse(original_get_sys_param('bool_false_str'))
            # Test case insensitivity for boolean
            self.assertFalse(original_get_sys_param('bool_cap_false_str'))
            self.assertEqual(original_get_sys_param(
                'string_param'), 'my_string')
            self.assertEqual(original_get_sys_param(
                'int_param'), 100.0)  # Converts to float
            # To test int conversion, the parameter name needs to be one that model.py converts to int
            # self.assertEqual(get_sys_param('int_param', param_type=int), 100) # If we had type hint
            # Returns string for breakpoints
            self.assertEqual(original_get_sys_param('bp_param'), '200,300')

            # NaN becomes None, then default (None here)
            self.assertIsNone(original_get_sys_param('nan_param'))
            self.assertEqual(original_get_sys_param(
                'nan_param', default=0.0), 0.0)
            self.assertIsNone(original_get_sys_param('none_param'))
            self.assertEqual(original_get_sys_param(
                'none_param', default='def'), 'def')

            with self.assertRaises(ValueError, msg="Required parameter missing should raise ValueError"):
                original_get_sys_param('missing_param', required=True)
            self.assertEqual(original_get_sys_param(
                'missing_param', default='fallback'), 'fallback')

    def test_model_creation_basic_structure(self):
        """Test basic model structure with default config."""
        model = self._setup_model_with_config()  # Default: HTE, Nuke, Elec ON
        self.assertIsInstance(model, pyo.ConcreteModel)
        self.assertIsNotNone(model.TimePeriods)
        self.assertTrue(hasattr(model, 'delT_minutes'))
        # Nuclear and HTE specific
        self.assertTrue(hasattr(model, 'qSteam_Total'))
        self.assertTrue(hasattr(model, 'pTurbine'))
        self.assertTrue(hasattr(model, 'pElectrolyzer'))
        self.assertTrue(hasattr(model, 'qSteam_Electrolyzer'))  # HTE
        self.assertTrue(hasattr(model, 'TotalProfit_Objective'))

    def test_model_creation_lte_mode(self):
        """Test model creation in LTE mode."""
        model = self._setup_model_with_config(
            ENABLE_LOW_TEMP_ELECTROLYZER=True, ENABLE_ELECTROLYZER=True, ENABLE_NUCLEAR_GENERATOR=True)
        self.assertTrue(model.LTE_MODE)
        self.assertTrue(hasattr(model, 'pElectrolyzer'))
        self.assertFalse(hasattr(model, 'qSteam_Electrolyzer'),
                         "qSteam_Electrolyzer should not exist for LTE")
        self.assertTrue(
            hasattr(model, 'cost_electrolyzer_capacity_USD_per_MW_year_LTE'))
        # Check if kt_H2_values exists and is zero for LTE
        self.assertTrue(hasattr(model, 'kt_H2_values'),
                        "kt_H2_values should exist even for LTE")
        if hasattr(model, 'pElectrolyzer_efficiency_breakpoints') and model.pElectrolyzer_efficiency_breakpoints:
            self.assertAlmostEqual(model.kt_H2_values[model.pElectrolyzer_efficiency_breakpoints.first(
            )], 0.0, places=5, msg="kt should be zero for LTE")

    def test_model_creation_with_battery(self):
        """Test model creation with battery enabled."""
        model = self._setup_model_with_config(ENABLE_BATTERY=True)
        self.assertTrue(hasattr(model, 'BatterySOC'))
        self.assertTrue(hasattr(model, 'BatteryCharge'))
        self.assertTrue(hasattr(model, 'BatteryDischarge'))
        self.assertTrue(hasattr(model, 'BatteryCapacity_MWh'))  # Variable
        self.assertTrue(
            hasattr(model, 'BatteryCapex_USD_per_MWh_year'))  # Parameter

    def test_model_creation_with_h2_storage(self):
        """Test model creation with H2 storage enabled."""
        model = self._setup_model_with_config(
            ENABLE_H2_STORAGE=True, ENABLE_ELECTROLYZER=True)
        self.assertTrue(hasattr(model, 'H2_storage_level'))
        self.assertTrue(hasattr(model, 'H2_to_storage'))
        self.assertTrue(hasattr(model, 'H2_from_storage'))
        self.assertTrue(hasattr(model, 'H2_storage_capacity_max'))

    def test_model_creation_with_startup_shutdown(self):
        """Test model creation with startup/shutdown enabled."""
        model = self._setup_model_with_config(
            ENABLE_STARTUP_SHUTDOWN=True, ENABLE_ELECTROLYZER=True)
        self.assertTrue(hasattr(model, 'uElectrolyzer'))
        self.assertTrue(hasattr(model, 'vElectrolyzerStartup'))
        self.assertTrue(hasattr(model, 'wElectrolyzerShutdown'))
        # or HTE depending on LTE_MODE
        self.assertTrue(
            hasattr(model, 'cost_startup_electrolyzer_USD_per_startup_LTE'))

    def test_model_creation_with_degradation(self):
        """Test model creation with degradation tracking enabled."""
        model = self._setup_model_with_config(
            ENABLE_ELECTROLYZER_DEGRADATION_TRACKING=True, ENABLE_ELECTROLYZER=True)
        self.assertTrue(hasattr(model, 'DegradationState'))
        self.assertTrue(hasattr(
            model, 'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE'))  # or HTE

    def test_ancillary_service_vars_and_params_no_as(self):
        """Test AS variable/param creation when AS is disabled."""
        # CAN_PROVIDE_ANCILLARY_SERVICES will be False
        model = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=False, ENABLE_ELECTROLYZER=False, ENABLE_BATTERY=False)
        self.assertFalse(model.CAN_PROVIDE_ANCILLARY_SERVICES)
        # Should be Param fixed to 0
        self.assertIsInstance(model.Total_RegUp, pyo.Param)
        self.assertEqual(
            pyo.value(model.Total_RegUp[model.TimePeriods.first()]), 0)
        # Deployed vars should not exist
        self.assertFalse(hasattr(model, 'RegUp_Electrolyzer_Deployed'))

    def test_ancillary_service_vars_and_params_as_enabled_bidding_mode(self):
        """Test AS variable creation when AS is enabled (bidding strategy mode)."""
        target_iso = "ERCOT"  # Example ISO
        model = self._setup_model_with_config(
            TARGET_ISO=target_iso,
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=True,
            simulate_dispatch=False
        )
        self.assertTrue(model.CAN_PROVIDE_ANCILLARY_SERVICES)
        self.assertFalse(model.SIMULATE_AS_DISPATCH_EXECUTION)

        self.assertIsInstance(model.Total_RegUp, pyo.Var)
        # Price param should exist
        self.assertTrue(hasattr(model, f'p_RegU_{target_iso}'))
        self.assertTrue(hasattr(model, 'RegUp_Turbine'))
        self.assertTrue(hasattr(model, 'RegUp_Electrolyzer'))
        self.assertTrue(hasattr(model, 'RegUp_Battery'))
        # Deployed vars should NOT exist in bidding mode
        self.assertFalse(hasattr(model, 'RegUp_Electrolyzer_Deployed'))

    def test_ancillary_service_vars_dispatch_simulation_mode(self):
        """Test AS deployed variable creation in dispatch simulation mode."""
        target_iso = "ERCOT"
        model = self._setup_model_with_config(
            TARGET_ISO=target_iso,
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=True,
            simulate_dispatch=True
        )
        self.assertTrue(model.CAN_PROVIDE_ANCILLARY_SERVICES)
        self.assertTrue(model.SIMULATE_AS_DISPATCH_EXECUTION)

        self.assertIsInstance(model.Total_RegUp, pyo.Var)
        self.assertTrue(hasattr(model, 'RegUp_Turbine'))  # Bid var
        self.assertTrue(hasattr(model, 'RegUp_Electrolyzer'))  # Bid var
        self.assertTrue(hasattr(model, 'RegUp_Battery'))  # Bid var

        # Deployed vars SHOULD exist
        self.assertTrue(hasattr(model, 'RegUp_Turbine_Deployed'))
        self.assertTrue(hasattr(model, 'RegUp_Electrolyzer_Deployed'))
        self.assertTrue(hasattr(model, 'RegUp_Battery_Deployed'))

    def test_precomputation_logic(self):
        """Test precomputation of inverse efficiency and steam values."""
        # HTE model
        model_hte = self._setup_model_with_config(
            ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=False)
        self.assertTrue(hasattr(model_hte, 'ke_H2_inv_values'))
        # Example check: if ke_H2_values_HTE was '0.025,0.024', inv should be 1/0.025 etc.
        if hasattr(model_hte, 'pElectrolyzer_efficiency_breakpoints') and model_hte.pElectrolyzer_efficiency_breakpoints:
            first_bp_hte = model_hte.pElectrolyzer_efficiency_breakpoints.first()  # e.g., 0
            # Ensure the value exists in the Param before accessing
            self.assertIn(first_bp_hte, model_hte.ke_H2_values)
            ke_val = pyo.value(model_hte.ke_H2_values[first_bp_hte])
            expected_inv_eff_hte = 1.0 / ke_val if ke_val > 1e-9 else 1e9
            self.assertAlmostEqual(
                model_hte.ke_H2_inv_values[first_bp_hte], expected_inv_eff_hte)

            if hasattr(model_hte, 'kt_H2_values'):  # kt_H2_values exists for HTE
                self.assertTrue(
                    hasattr(model_hte, 'qSteam_values_at_pElec_bp'))
                # Ensure the value exists in the Param before accessing
                self.assertIn(first_bp_hte, model_hte.kt_H2_values)
                kt_val = pyo.value(model_hte.kt_H2_values[first_bp_hte])
                expected_q_steam_hte = (
                    kt_val * model_hte.ke_H2_inv_values[first_bp_hte] * first_bp_hte)
                self.assertAlmostEqual(pyo.value(
                    model_hte.qSteam_values_at_pElec_bp[first_bp_hte]), expected_q_steam_hte)
        else:
            self.fail(
                "Missing pElectrolyzer_efficiency_breakpoints for HTE precomputation test")

        # LTE model
        model_lte = self._setup_model_with_config(
            ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=True)
        self.assertTrue(hasattr(model_lte, 'ke_H2_inv_values'))
        if hasattr(model_lte, 'pElectrolyzer_efficiency_breakpoints') and model_lte.pElectrolyzer_efficiency_breakpoints:
            first_bp_lte = model_lte.pElectrolyzer_efficiency_breakpoints.first()
            # Ensure the value exists in the Param before accessing
            self.assertIn(first_bp_lte, model_lte.ke_H2_values)
            ke_val_lte = pyo.value(model_lte.ke_H2_values[first_bp_lte])
            expected_inv_eff_lte = 1.0 / ke_val_lte if ke_val_lte > 1e-9 else 1e9
            self.assertAlmostEqual(
                model_lte.ke_H2_inv_values[first_bp_lte], expected_inv_eff_lte)
            # qSteam_values_at_pElec_bp should not be meaningfully populated for LTE or kt_H2_values is zero
            self.assertTrue(hasattr(model_lte, 'kt_H2_values'),
                            "kt_H2_values should exist for LTE (as zero)")
            # Ensure the value exists in the Param before accessing
            self.assertIn(first_bp_lte, model_lte.kt_H2_values)
            self.assertAlmostEqual(pyo.value(
                model_lte.kt_H2_values[first_bp_lte]), 0.0, places=5, msg="kt should be zero for LTE")
            self.assertFalse(hasattr(model_lte, 'qSteam_values_at_pElec_bp'),
                             "qSteam_values_at_pElec_bp should not be created for LTE")
        else:
            self.fail(
                "Missing pElectrolyzer_efficiency_breakpoints for LTE precomputation test")


if __name__ == '__main__':
    unittest.main()
