"""
Unit tests for constraints.py module.
Tests constraint generation and validation.
"""

from constraints import (
    steam_balance_rule, power_balance_rule, constant_turbine_power_rule,
    link_auxiliary_power_rule, h2_storage_balance_adj_rule, h2_prod_dispatch_rule,
    Electrolyzer_RampUp_rule, Electrolyzer_RampDown_rule, Turbine_RampUp_rule, Turbine_RampDown_rule,
    Steam_Electrolyzer_Ramp_rule, h2_CapacityFactor_rule,
    electrolyzer_on_off_logic_rule, electrolyzer_min_power_when_on_rule,
    electrolyzer_max_power_rule, electrolyzer_startup_shutdown_exclusivity_rule,
    electrolyzer_min_uptime_rule, electrolyzer_min_downtime_rule,
    electrolyzer_degradation_rule,
    battery_soc_balance_rule, battery_charge_limit_rule, battery_discharge_limit_rule,
    battery_binary_exclusivity_rule, battery_soc_max_rule, battery_soc_min_rule,
    battery_power_capacity_link_rule,
    Turbine_AS_Pmax_rule, Electrolyzer_AS_Pmin_rule, Battery_AS_SOC_Up_rule,
    link_Total_RegUp_rule,
    link_deployed_to_bid_rule, define_actual_electrolyzer_power_rule, get_as_components
)
from model import create_model
import config as app_config  # Use a distinct alias
import unittest
import sys
import os
import pandas as pd
import pyomo.environ as pyo
from pathlib import Path
from unittest.mock import patch
import importlib

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'src')))

# Import necessary modules after path append
# Import specific constraint rules to test them if they aren't directly on model


class TestConstraints(unittest.TestCase):
    """Test cases for constraint generation and validation."""

    def _create_base_data_inputs(self, iso="ERCOT"):
        # Simplified data for basic model instantiation
        num_hours = 24
        price_data_dict = {'hour': range(
            1, num_hours + 1), 'Price ($/MWh)': [50.0] * num_hours}
        # Example for ERCOT
        as_price_cols = [f'p_RegU_{iso}', f'p_RegD_{iso}', f'p_SR_{iso}']
        for col in as_price_cols:
            price_data_dict[col] = [10.0] * num_hours

        parameters = [
            'delT_minutes', 'AS_Duration', 'plant_lifetime_years', 'pIES_min_MW', 'pIES_max_MW',
            'qSteam_Total_MWth', 'qSteam_Turbine_min_MWth', 'qSteam_Turbine_max_MWth',
            'pTurbine_min_MW', 'pTurbine_max_MW',
            'Turbine_RampUp_Rate_Percent_per_Min', 'Turbine_RampDown_Rate_Percent_per_Min', 'vom_turbine_USD_per_MWh',
            'Turbine_Thermal_Elec_Efficiency_Const', 'qSteam_Turbine_Breakpoints_MWth', 'pTurbine_Outputs_at_Breakpoints_MW',
            'pElectrolyzer_max_upper_bound_MW', 'pElectrolyzer_max_lower_bound_MW',
            'pElectrolyzer_min_MW',
            'pElectrolyzer_Breakpoints_MW',  # Added Generic Breakpoint Fallback
            'pElectrolyzer_min_MW_LTE', 'ke_H2_Values_MWh_per_kg_LTE', 'pElectrolyzer_Breakpoints_MW_LTE',
            'pElectrolyzer_min_MW_HTE', 'ke_H2_Values_MWh_per_kg_HTE', 'kt_H2_Values_MWth_per_kg_HTE', 'pElectrolyzer_Breakpoints_MW_HTE',
            'cost_electrolyzer_capacity_USD_per_MW_year_LTE', 'cost_electrolyzer_capacity_USD_per_MW_year_HTE',
            'aux_power_consumption_per_kg_h2', 'H2_value_USD_per_kg', 'hydrogen_subsidy_per_kg',
            'MinUpTimeElectrolyzer_hours_LTE', 'MinDownTimeElectrolyzer_hours_LTE', 'cost_startup_electrolyzer_USD_per_startup_LTE',
            'uElectrolyzer_initial_status_0_or_1', 'DegradationStateInitial_Units',
            'DegradationFactorOperation_Units_per_Hour_at_MaxLoad_LTE', 'DegradationFactorStartup_Units_per_Startup_LTE',
            'H2_storage_capacity_max_kg', 'H2_storage_level_initial_kg', 'H2_storage_charge_rate_max_kg_per_hr', 'H2_storage_discharge_rate_max_kg_per_hr', 'storage_charge_eff_fraction', 'storage_discharge_eff_fraction',
            'BatteryCapacity_max_MWh', 'BatteryPowerRatio_MW_per_MWh', 'BatteryChargeEff', 'BatteryDischargeEff', 'BatterySOC_min_fraction', 'BatterySOC_initial_fraction',
            'BatteryRequireCyclicSOC', 'BatteryRampRate_fraction_per_hour', 'vom_battery_per_mwh_cycled', 'BatteryCapex_USD_per_MWh_year', 'BatteryCapex_USD_per_MW_year', 'BatteryFixedOM_USD_per_MWh_year',
            'cost_water_USD_per_kg_h2', 'cost_electrolyzer_ramping_USD_per_MW_ramp_LTE', 'Ramp_qSteam_Electrolyzer_limit_MWth_per_Hour', 'pTurbine_LTE_setpoint_MW'
        ]
        values = [
            '60', '0.25', '30', '-100', '100',  # General
            '1000', '200', '900',  # Nuclear Steam
            '40', '450',  # Nuclear Power
            '2', '2', '2',  # Nuc VOM, Ramps
            '0.4', '100,500,900', '40,200,360',  # Nuc eff
            '100', '0',  # Elec UB/LB
            '0',  # Generic pElec min
            '0,50,100',  # Added Generic Breakpoint Value
            '5', '0.022,0.021', '0,50,100',  # LTE Min, ke, bp
            '10', '0.025,0.024', '0.010,0.009', '0,60,120',  # HTE Min, ke, kt, bp
            '1000', '1200',  # Elec capex
            '0.1', '2.0', '0.0',  # Aux, H2 val, subsidy
            '1', '1', '10',  # SU/SD
            '0', '0', '0.001', '1',  # Deg
            '1000', '100', '100', '100', '1.0', '1.0',  # H2 Storage
            '100', '0.25', '0.9', '0.9', '0.1', '0.5',  # Battery
            'True', '1.0', '0.1', '100', '10', '10',  # Battery
            '0.1', '0.01', '1000', '300'  # water, elec_ramp, steam_ramp, lte_turb_setpoint
        ]
        if len(parameters) != len(values):
            raise ValueError(
                f"Mismatch in test_constraints setup parameter/value lengths ({len(parameters)} vs {len(values)}).")

        system_data = pd.DataFrame(
            {'Parameter': parameters, 'Value': values}).set_index('Parameter')
        return {
            'df_price_hourly': pd.DataFrame(price_data_dict),
            'df_ANSprice_hourly': pd.DataFrame(price_data_dict),  # Simplified
            'df_ANSmile_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            'df_ANSdeploy_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            'df_ANSwinrate_hourly': pd.DataFrame({'hour': range(1, num_hours+1)}),
            'df_system': system_data
        }

    def _setup_model_with_config(self, target_iso="ERCOT", simulate_dispatch=False, **kwargs_config_flags):
        """Helper to create a model with specific config flags patched."""
        # Default flags that create_model will read from config module
        default_flags = {
            'ENABLE_NUCLEAR_GENERATOR': True,
            'ENABLE_ELECTROLYZER': True,
            'ENABLE_LOW_TEMP_ELECTROLYZER': False,  # Default to HTE
            'ENABLE_BATTERY': False,
            'ENABLE_H2_STORAGE': False,
            'ENABLE_STARTUP_SHUTDOWN': False,
            'ENABLE_ELECTROLYZER_DEGRADATION_TRACKING': False,
            'ENABLE_H2_CAP_FACTOR': False,
            'ENABLE_NONLINEAR_TURBINE_EFF': True,
            'TARGET_ISO': target_iso,
            # SIMULATE_AS_DISPATCH_EXECUTION is passed directly to create_model
        }
        # Override defaults with any flags passed via kwargs_config_flags
        effective_flags = {**default_flags, **kwargs_config_flags}

        # Patch the config module for the duration of model creation
        with patch.multiple('config', **effective_flags):
            # We need to reload model if create_model uses config at module level,
            # or ensure create_model directly takes all flags.
            # Assuming create_model reads from config module during its execution.
            # Reload to pick up patched values if config is structured that way
            importlib.reload(app_config)
            data_inputs = self._create_base_data_inputs(iso=target_iso)
            # Ensure data_inputs['df_system'] is not None before creating model
            if data_inputs['df_system'] is None:
                raise ValueError("df_system is None in test setup")
            model = create_model(data_inputs, target_iso,
                                 simulate_dispatch=simulate_dispatch)
        return model

    def test_steam_balance_constraint(self):
        """Test steam_balance_rule."""
        # HTE mode (default)
        model_hte = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=False)
        self.assertTrue(hasattr(model_hte, 'steam_balance_constr'),
                        "Steam balance constraint missing for HTE")
        self.assertTrue(
            model_hte.steam_balance_constr[1].active, "Steam balance should be active for HTE")

        # LTE mode
        model_lte = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=True)
        self.assertTrue(hasattr(model_lte, 'steam_balance_constr'),
                        "Steam balance constraint missing for LTE")
        self.assertTrue(
            model_lte.steam_balance_constr[1].active, "Steam balance should be active for LTE (even if qSteam_Electrolyzer is 0)")
        # In LTE, qSteam_Electrolyzer should be effectively zero in the steam_balance_rule.
        # We'd need to fix variable values to test the expression itself.

        # No electrolyzer
        model_no_elec = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=False)
        self.assertTrue(hasattr(model_no_elec, 'steam_balance_constr'),
                        "Steam balance constraint missing")
        self.assertTrue(
            model_no_elec.steam_balance_constr[1].active, "Steam balance should be active")

    def test_power_balance_constraint(self):
        """Test power_balance_rule."""
        model = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=True)
        self.assertTrue(hasattr(model, 'power_balance_constr'))
        self.assertTrue(model.power_balance_constr[1].active)
        # To test expression: fix model.pTurbine[1], model.pIES[1] etc. and evaluate

    def test_constant_turbine_power_constraint(self):
        """Test constant_turbine_power_rule for LTE mode."""
        model_lte_on = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=True)
        self.assertTrue(hasattr(model_lte_on, 'const_turbine_power_constr'))
        self.assertTrue(model_lte_on.const_turbine_power_constr[1].active)

        model_hte = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_LOW_TEMP_ELECTROLYZER=False)
        self.assertFalse(hasattr(model_hte, 'const_turbine_power_constr'),
                         "Constant turbine power constraint should not exist for HTE")

    def test_electrolyzer_ramp_constraints(self):
        """Test electrolyzer ramp up and down constraints."""
        model = self._setup_model_with_config(ENABLE_ELECTROLYZER=True)
        self.assertTrue(hasattr(model, 'Electrolyzer_RampUp_constr'))
        self.assertTrue(hasattr(model, 'Electrolyzer_RampDown_constr'))
        # First period should skip
        self.assertEqual(
            model.Electrolyzer_RampUp_constr[1], pyo.Constraint.Skip)
        self.assertEqual(
            model.Electrolyzer_RampDown_constr[1], pyo.Constraint.Skip)
        if len(model.TimePeriods) > 1:
            self.assertTrue(model.Electrolyzer_RampUp_constr[2].active)
            self.assertTrue(model.Electrolyzer_RampDown_constr[2].active)

    def test_h2_storage_constraints(self):
        """Test H2 storage related constraints."""
        model = self._setup_model_with_config(
            ENABLE_ELECTROLYZER=True, ENABLE_H2_STORAGE=True)
        self.assertTrue(hasattr(model, 'h2_storage_balance_constr'))
        self.assertTrue(hasattr(model, 'h2_prod_dispatch_constr'))
        self.assertTrue(model.h2_storage_balance_constr[1].active)
        self.assertTrue(model.h2_prod_dispatch_constr[1].active)

    def test_startup_shutdown_constraints(self):
        """Test electrolyzer startup/shutdown constraints."""
        model = self._setup_model_with_config(
            ENABLE_ELECTROLYZER=True, ENABLE_STARTUP_SHUTDOWN=True)
        self.assertTrue(hasattr(model, 'electrolyzer_on_off_logic_constr'))
        self.assertTrue(
            hasattr(model, 'electrolyzer_min_power_when_on_constr'))
        self.assertTrue(hasattr(model, 'electrolyzer_min_uptime_constr'))
        self.assertTrue(model.electrolyzer_on_off_logic_constr[1].active)

    def test_degradation_constraint(self):
        """Test electrolyzer degradation constraint."""
        model = self._setup_model_with_config(
            ENABLE_ELECTROLYZER=True, ENABLE_ELECTROLYZER_DEGRADATION_TRACKING=True, ENABLE_STARTUP_SHUTDOWN=True)
        self.assertTrue(hasattr(model, 'electrolyzer_degradation_constr'))
        self.assertTrue(model.electrolyzer_degradation_constr[1].active)
        if len(model.TimePeriods) > 1:
            self.assertTrue(model.electrolyzer_degradation_constr[2].active)

    def test_battery_constraints(self):
        """Test battery related constraints."""
        model = self._setup_model_with_config(ENABLE_BATTERY=True)
        self.assertTrue(hasattr(model, 'battery_soc_balance_constr'))
        self.assertTrue(hasattr(model, 'battery_charge_limit_constr'))
        # This is a single constraint, not indexed by TimePeriods
        self.assertTrue(hasattr(model, 'battery_power_capacity_link_constr'))
        self.assertTrue(model.battery_soc_balance_constr[1].active)
        self.assertTrue(model.battery_power_capacity_link_constr.active)

    def test_ancillary_service_capability_constraints(self):
        """Test a sample AS capability constraint (e.g., Turbine AS Pmax)."""
        # Test when AS should be active
        model_as_active = self._setup_model_with_config(
            # Ensures CAN_PROVIDE_AS is True
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=False
        )
        self.assertTrue(model_as_active.CAN_PROVIDE_ANCILLARY_SERVICES)
        self.assertTrue(hasattr(model_as_active, 'Turbine_AS_Pmax_constr'))
        self.assertTrue(model_as_active.Turbine_AS_Pmax_constr[1].active)
        # Check another
        self.assertTrue(
            hasattr(model_as_active, 'Electrolyzer_AS_Pmin_constr'))
        self.assertTrue(model_as_active.Electrolyzer_AS_Pmin_constr[1].active)

        # Test when AS should be inactive (e.g. no nuke)
        model_as_inactive = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=False, ENABLE_ELECTROLYZER=True
        )
        self.assertFalse(model_as_inactive.CAN_PROVIDE_ANCILLARY_SERVICES)
        if hasattr(model_as_inactive, 'Turbine_AS_Pmax_constr'):  # It might not be added at all
            self.assertEqual(
                model_as_inactive.Turbine_AS_Pmax_constr[1], pyo.Constraint.Skip)

    def test_as_linking_constraints(self):
        """Test ancillary service total bid linking rules."""
        model_as_active = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=True
        )
        self.assertTrue(model_as_active.CAN_PROVIDE_ANCILLARY_SERVICES)
        self.assertTrue(hasattr(model_as_active, 'link_Total_RegUp_constr'))
        self.assertTrue(model_as_active.link_Total_RegUp_constr[1].active)
        # Ensure Total_RegUp is a Var
        self.assertIsInstance(model_as_active.Total_RegUp, pyo.Var)

        model_as_inactive = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=False)
        self.assertFalse(model_as_inactive.CAN_PROVIDE_ANCILLARY_SERVICES)
        # The constraint rule itself will skip, and Total_RegUp should be a Param
        self.assertIsInstance(model_as_inactive.Total_RegUp, pyo.Param)
        # The constraint object might not even be added if Total_RegUp is a Param.
        # The rule link_total_as_rule checks "isinstance(total_var, pyo.Var)"

    def test_dispatch_simulation_constraints(self):
        """Test constraints active only during AS dispatch simulation."""
        # Dispatch simulation OFF
        model_sim_off = self._setup_model_with_config(
            simulate_dispatch=False, ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True
        )
        self.assertFalse(model_sim_off.SIMULATE_AS_DISPATCH_EXECUTION)
        # These constraints should not be active or might not exist if their rules return Skip
        if hasattr(model_sim_off, 'link_RegUp_Electrolyzer_deployed_constr'):  # Dynamic name
            self.assertEqual(
                model_sim_off.link_RegUp_Electrolyzer_deployed_constr[1], pyo.Constraint.Skip)
        if hasattr(model_sim_off, 'define_actual_electrolyzer_power_constr'):
            self.assertEqual(
                model_sim_off.define_actual_electrolyzer_power_constr[1], pyo.Constraint.Skip)

        # Dispatch simulation ON
        model_sim_on = self._setup_model_with_config(
            simulate_dispatch=True, ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True
        )
        self.assertTrue(model_sim_on.SIMULATE_AS_DISPATCH_EXECUTION)
        self.assertTrue(model_sim_on.CAN_PROVIDE_ANCILLARY_SERVICES)

        # Constraints should be active
        # Need to ensure the deployed var exists (e.g. RegUp_Electrolyzer_Deployed)
        # and the bid var (RegUp_Electrolyzer)
        # and the relevant params (winning_rate_RegUp_ERCOT, deploy_factor_RegUp_ERCOT)

        # Test link_deployed_to_bid_rule (indirectly via model constraint)
        # Example: link_RegUp_Electrolyzer_deployed_constr
        # As defined in model.py add_deployment_link_constraints
        deployed_constr_name = 'link_RegUp_Electrolyzer_deployed_constr'
        self.assertTrue(hasattr(model_sim_on, deployed_constr_name),
                        f"{deployed_constr_name} missing")
        self.assertTrue(getattr(model_sim_on, deployed_constr_name)[1].active)

        self.assertTrue(
            hasattr(model_sim_on, 'define_actual_electrolyzer_power_constr'))
        self.assertTrue(
            model_sim_on.define_actual_electrolyzer_power_constr[1].active)

    def test_get_as_components_helper(self):
        """Test the get_as_components helper function from constraints.py."""
        model = self._setup_model_with_config(
            ENABLE_NUCLEAR_GENERATOR=True, ENABLE_ELECTROLYZER=True, ENABLE_BATTERY=True
        )
        self.assertTrue(model.CAN_PROVIDE_ANCILLARY_SERVICES)

        # Fix some bid values for testing
        t = 1
        model.RegUp_Turbine[t].fix(5)
        # Use internal name SR for Spin/Synchronized Reserve
        if hasattr(model, 'SR_Turbine'):
            model.SR_Turbine[t].fix(10)
        model.RegDown_Electrolyzer[t].fix(3)
        model.RegUp_Battery[t].fix(2)

        as_info = get_as_components(model, t)

        # 5 (RegUp) + 10 (SR)
        self.assertEqual(pyo.value(as_info['up_reserves_bid_turbine']), 15)
        self.assertEqual(pyo.value(as_info['down_reserves_bid_turbine']), 0)
        self.assertEqual(pyo.value(as_info['up_reserves_bid_h2']), 0)
        self.assertEqual(
            pyo.value(as_info['down_reserves_bid_h2']), 3)  # 3 (RegDown)
        self.assertEqual(
            pyo.value(as_info['up_reserves_bid_battery']), 2)  # 2 (RegUp)
        self.assertEqual(pyo.value(as_info['down_reserves_bid_battery']), 0)

        self.assertEqual(
            pyo.value(as_info['up_reserves_bid']), 17)  # 15 + 0 + 2
        self.assertEqual(
            pyo.value(as_info['down_reserves_bid']), 3)  # 0 + 3 + 0


if __name__ == '__main__':
    unittest.main()
