"""
Unit tests for revenue_cost.py module.
Tests revenue and cost calculations.
"""

import unittest
import sys
import os
import pandas as pd
import pyomo.environ as pyo
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config as app_config
from model import create_model
# Import rules directly to test them
from revenue_cost import (
    EnergyRevenue_rule, HydrogenRevenue_rule, OpexCost_rule,
    AncillaryRevenue_SPP_rule, AncillaryRevenue_CAISO_rule, AncillaryRevenue_ERCOT_rule,
    AncillaryRevenue_PJM_rule, AncillaryRevenue_NYISO_rule, AncillaryRevenue_ISONE_rule,
    AncillaryRevenue_MISO_rule,
    _calculate_hourly_spp_revenue, # Example of testing a private helper if needed
    get_total_deployed_as
)


class TestRevenueCost(unittest.TestCase):
    """Test cases for revenue and cost calculations."""

    def _create_base_data_inputs(self, iso="ERCOT", num_hours=3):
        # Simplified data, ensure params needed by revenue/cost rules are present
        price_data_dict = {'hour': range(1, num_hours + 1), 'Price ($/MWh)': [50.0] * num_hours}
        as_price_cols = {}
        # Add AS prices based on ISO - must match what revenue rules expect
        if iso == "ERCOT":
            as_price_cols = {
                f'p_RegU_{iso}': [10]*num_hours, f'p_RegD_{iso}': [8]*num_hours,
                f'p_SR_{iso}': [5]*num_hours, f'loc_RegU_{iso}': [1]*num_hours,
                f'winning_rate_RegU_{iso}': [0.9]*num_hours,
                # For bidding mode performance payment (mileage*perf*LMP)
                # For dispatch mode (Deployed * LMP)
                 f'winning_rate_SR_{iso}': [0.8]*num_hours, # Added for SR
                 f'deploy_factor_SR_{iso}': [0.5]*num_hours, # Added for SR
                 f'loc_SR_{iso}': [0.5]*num_hours, # Added for SR
            }
        elif iso == "SPP":
             as_price_cols = {
                f'p_RegU_{iso}': [11]*num_hours, f'p_RegD_{iso}': [9]*num_hours,
                f'p_Spin_{iso}': [6]*num_hours, f'loc_RegU_{iso}': [1.1]*num_hours,
                f'winning_rate_RegU_{iso}': [0.85]*num_hours,
                f'deploy_factor_Spin_{iso}': [0.6]*num_hours,
                # Add winning rate for Spin if needed by revenue calc
                f'winning_rate_Spin_{iso}': [0.8]*num_hours,
                f'loc_Spin_{iso}': [0.4]*num_hours,
            }
        # Add more ISOs as needed
        price_data_dict.update(as_price_cols)

        parameters = [
            'delT_minutes', 'AS_Duration', 'H2_value_USD_per_kg', 'hydrogen_subsidy_per_kg',
            'vom_turbine_USD_per_MWh', 'vom_electrolyzer_USD_per_MWh_LTE', 'vom_battery_per_mwh_cycled',
            'cost_water_USD_per_kg_h2', 'cost_electrolyzer_ramping_USD_per_MW_ramp_LTE',
            'vom_storage_cycle_USD_per_kg_cycled', 'cost_startup_electrolyzer_USD_per_startup_LTE',
            # Params needed by model to instantiate vars used in rules
            'pElectrolyzer_max_upper_bound_MW', 'pElectrolyzer_min_MW', # Added generic pElectrolyzer_min_MW
            'pElectrolyzer_Breakpoints_MW', # Added Generic Breakpoint Fallback
            'pElectrolyzer_min_MW_LTE', 'ke_H2_Values_MWh_per_kg_LTE', 'pElectrolyzer_Breakpoints_MW_LTE',
            'pTurbine_max_MW', 'qSteam_Total_MWth', 'qSteam_Turbine_min_MWth', 'qSteam_Turbine_max_MWth', # Added qSteam_Turbine_max_MWth
            'BatteryCapacity_max_MWh', 'BatteryPowerRatio_MW_per_MWh',
            'H2_storage_capacity_max_kg', 'H2_storage_level_initial_kg',
            'uElectrolyzer_initial_status_0_or_1', # for startup cost
             # Need params for AS capability rules if testing AS revenue thoroughly
             'pTurbine_min_MW', 'Electrolyzer_RampUp_Rate_Percent_per_Min_LTE', 'Electrolyzer_RampDown_Rate_Percent_per_Min_LTE',
             'BatterySOC_min_fraction', 'BatteryCapacity_min_MWh', 'BatteryRampRate_fraction_per_hour'
        ]
        values = [
            '60', '0.25', '3.0', '0.6', # General, H2 val, subsidy
            '2.0', '1.5', '0.5', # VOMs
            '0.1', '0.05', # Water, ElecRamp
            '0.01', '500', # H2StorVOM, ElecSUCost
            '100', '0', # Elec UB/LB, Generic Min
            '0,50,100', # Generic Breakpoint Value
            '5', '0.022', '0,50,100', # LTE Min, ke, bp
            '450', '1000', '200', '900', # Turbine Max, Steam, Steam Min, Steam Max
            '100', '0.25', # Batt Cap, Ratio
            '1000', '100', # H2 Stor Cap, Init
            '0', # Elec SU Init
            '50', '10', '10', # Turbine Min, Elec Ramps
            '0.1', '0', '1.0' # Batt SOC min, Cap min, Ramp Rate
        ]
        if len(parameters) != len(values):
            raise ValueError(f"Mismatch in test_revenue_cost setup parameter/value lengths ({len(parameters)} vs {len(values)}).")

        system_data = pd.DataFrame({'Parameter': parameters, 'Value': values}).set_index('Parameter')
        return {
            'df_price_hourly': pd.DataFrame(price_data_dict),
            'df_ANSprice_hourly': pd.DataFrame(price_data_dict),
            # Minimal data for other optional files if model.py loads them
            'df_ANSmile_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[1.0]*num_hours for k in as_price_cols if 'RegU' in k}}),
            'df_ANSdeploy_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[0.5]*num_hours for k in as_price_cols if 'Spin' in k or 'SR' in k}}),
            'df_ANSwinrate_hourly': pd.DataFrame({'hour': range(1,num_hours+1), **{k:[0.9]*num_hours for k in as_price_cols}}),
            'df_system': system_data
        }

    def _setup_model_and_fix_vars(self, target_iso="ERCOT", simulate_dispatch=False, num_hours=3, **config_flags):
        default_component_flags = {
            'ENABLE_NUCLEAR_GENERATOR': True, 'ENABLE_ELECTROLYZER': True,
            'ENABLE_LOW_TEMP_ELECTROLYZER': True, 'ENABLE_BATTERY': True,
            'ENABLE_H2_STORAGE': True, 'ENABLE_STARTUP_SHUTDOWN': True,
            'TARGET_ISO': target_iso,
        }
        effective_flags = {**default_component_flags, **config_flags}

        with patch.multiple('config', **effective_flags):
            importlib.reload(app_config) # Ensure config module reflects patches
            data_inputs = self._create_base_data_inputs(iso=target_iso, num_hours=num_hours)
            model = create_model(data_inputs, target_iso, simulate_dispatch=simulate_dispatch)

            # Fix variables
            for t_idx, t in enumerate(model.TimePeriods):
                t_val = t_idx + 1
                if hasattr(model, 'pIES'): model.pIES[t].value = 10 * t_val
                if hasattr(model, 'mHydrogenProduced') and model.mHydrogenProduced.is_indexed(): model.mHydrogenProduced[t].value = 5 * t_val
                if hasattr(model, 'H2_to_market') and model.H2_to_market.is_indexed(): model.H2_to_market[t].value = 2 * t_val
                if hasattr(model, 'H2_from_storage') and model.H2_from_storage.is_indexed(): model.H2_from_storage[t].value = 1 * t_val

                if hasattr(model, 'pTurbine') and model.pTurbine.is_indexed(): model.pTurbine[t].value = 20 * t_val
                if hasattr(model, 'pElectrolyzer') and model.pElectrolyzer.is_indexed(): model.pElectrolyzer[t].value = 15 * t_val # Actual power
                if hasattr(model, 'pElectrolyzerSetpoint') and model.pElectrolyzerSetpoint.is_indexed(): model.pElectrolyzerSetpoint[t].value = 15 * t_val # Setpoint

                if hasattr(model, 'BatteryCharge') and model.BatteryCharge.is_indexed(): model.BatteryCharge[t].value = 3 * t_val
                if hasattr(model, 'BatteryDischarge') and model.BatteryDischarge.is_indexed(): model.BatteryDischarge[t].value = 2 * t_val

                if hasattr(model, 'pElectrolyzerRampPos') and model.pElectrolyzerRampPos.is_indexed():
                    model.pElectrolyzerRampPos[t].value = 1 if t_idx > 0 else 0
                if hasattr(model, 'pElectrolyzerRampNeg') and model.pElectrolyzerRampNeg.is_indexed():
                    model.pElectrolyzerRampNeg[t].value = 0

                if hasattr(model, 'H2_to_storage') and model.H2_to_storage.is_indexed(): model.H2_to_storage[t].value = model.mHydrogenProduced[t].value - model.H2_to_market[t].value if model.mHydrogenProduced[t].value is not None else 0 # Input based on prod - market
                if hasattr(model, 'vElectrolyzerStartup') and model.vElectrolyzerStartup.is_indexed(): model.vElectrolyzerStartup[t].value = 1 if t_idx == 0 else 0


                # AS Bids and Deployed (example for RegUp ERCOT/SPP)
                if model.CAN_PROVIDE_ANCILLARY_SERVICES:
                    service_key = 'RegU' # Could be 'Spin' for SPP reserves etc.
                    # Bid variables
                    total_var = getattr(model, f'Total_{service_key}', None)
                    if total_var is not None and total_var.is_indexed(): total_var[t].value = 2.0 * t_val

                    elec_bid_var = getattr(model, f'{service_key}_Electrolyzer', None)
                    if elec_bid_var is not None and elec_bid_var.is_indexed(): elec_bid_var[t].value = 1.0 * t_val

                    # Example for SPP Spin (Internal Name: SR)
                    spin_key = 'SR'
                    total_spin_var = getattr(model, f'Total_{spin_key}', None)
                    if total_spin_var is not None and total_spin_var.is_indexed(): total_spin_var[t].value = 3.0 * t_val
                    elec_spin_var = getattr(model, f'{spin_key}_Electrolyzer', None)
                    if elec_spin_var is not None and elec_spin_var.is_indexed(): elec_spin_var[t].value = 1.5*t_val


                    # Deployed variables (only if simulating dispatch)
                    if simulate_dispatch:
                        elec_dep_var = getattr(model, f'{service_key}_Electrolyzer_Deployed', None)
                        if elec_dep_var is not None and elec_dep_var.is_indexed() and elec_bid_var is not None and elec_bid_var.is_indexed():
                            win_rate_param = getattr(model, f"winning_rate_{service_key}_{target_iso}", None)
                            win_rate_val = pyo.value(win_rate_param[t]) if win_rate_param and win_rate_param.is_indexed() and t in win_rate_param else 1.0
                            elec_dep_var[t].value = elec_bid_var[t].value * win_rate_val

                        elec_spin_dep_var = getattr(model, f'{spin_key}_Electrolyzer_Deployed', None)
                        if elec_spin_dep_var is not None and elec_spin_dep_var.is_indexed() and elec_spin_var is not None and elec_spin_var.is_indexed():
                             # Use the correct ISO name 'Spin' for params if that's how they are named in data
                             param_iso_spin_name = 'Spin' if target_iso == 'SPP' else spin_key
                             win_rate_param_sp = getattr(model, f"winning_rate_{param_iso_spin_name}_{target_iso}", None)
                             win_rate_val_sp = pyo.value(win_rate_param_sp[t]) if win_rate_param_sp and win_rate_param_sp.is_indexed() and t in win_rate_param_sp else 1.0
                             deploy_factor_param_sp = getattr(model, f"deploy_factor_{param_iso_spin_name}_{target_iso}", None)
                             deploy_factor_val_sp = pyo.value(deploy_factor_param_sp[t]) if deploy_factor_param_sp and deploy_factor_param_sp.is_indexed() and t in deploy_factor_param_sp else 0.0
                             elec_spin_dep_var[t].value = elec_spin_var[t].value * win_rate_val_sp * deploy_factor_val_sp


            # Fix capacity variables (these are not indexed by time)
            elec_max_var = getattr(model, 'pElectrolyzer_max', None)
            if elec_max_var is not None: elec_max_var.value = pyo.value(model.pElectrolyzer_max_upper_bound)
            batt_cap_var = getattr(model, 'BatteryCapacity_MWh', None)
            if batt_cap_var is not None: batt_cap_var.value = pyo.value(model.BatteryCapacity_max)
        return model


    def test_energy_revenue_calculation(self):
        """Test EnergyRevenue_rule with fixed values."""
        num_hours = 2
        model = self._setup_model_and_fix_vars(num_hours=num_hours)
        # pIES fixed to 10, 20 for t=1,2. energy_price is 50. delT_minutes is 60 (time_factor=1).
        expected_revenue = (10 * 50 * 1) + (20 * 50 * 1)
        actual_revenue = pyo.value(EnergyRevenue_rule(model))
        self.assertAlmostEqual(actual_revenue, expected_revenue, places=2)

    def test_hydrogen_revenue_calculation_with_storage_and_subsidy(self):
        """Test HydrogenRevenue_rule with storage and subsidy."""
        num_hours = 1
        model = self._setup_model_and_fix_vars(num_hours=num_hours, ENABLE_H2_STORAGE=True)
        t1 = model.TimePeriods.first()
        # mHydrogenProduced[t1]=5, H2_to_market[t1]=2, H2_from_storage[t1]=1
        # H2_value=3.0, subsidy=0.6, time_factor=1
        expected_sales_rev = (model.H2_to_market[t1].value + model.H2_from_storage[t1].value) * pyo.value(model.H2_value) * 1.0
        expected_subsidy_rev = model.mHydrogenProduced[t1].value * pyo.value(model.hydrogen_subsidy_per_kg) * 1.0
        expected_total_revenue = expected_sales_rev + expected_subsidy_rev
        actual_revenue = pyo.value(HydrogenRevenue_rule(model))
        self.assertAlmostEqual(actual_revenue, expected_total_revenue, places=2)

    def test_hydrogen_revenue_calculation_no_storage(self):
        """Test HydrogenRevenue_rule without storage but with subsidy."""
        num_hours = 1
        model = self._setup_model_and_fix_vars(num_hours=num_hours, ENABLE_H2_STORAGE=False)
        t1 = model.TimePeriods.first()
        # mHydrogenProduced[t1]=5. H2_value=3.0, subsidy=0.6. time_factor=1
        # Without storage, revenue from sales is mHydrogenProduced * H2_value
        # Subsidy is mHydrogenProduced * subsidy_rate
        expected_revenue = (model.mHydrogenProduced[t1].value * pyo.value(model.H2_value) +
                            model.mHydrogenProduced[t1].value * pyo.value(model.hydrogen_subsidy_per_kg)) * 1.0
        actual_revenue = pyo.value(HydrogenRevenue_rule(model))
        self.assertAlmostEqual(actual_revenue, expected_revenue, places=2)


    def test_opex_cost_calculation(self):
        """Test OpexCost_rule with fixed values."""
        num_hours = 1
        model = self._setup_model_and_fix_vars(num_hours=num_hours, ENABLE_STARTUP_SHUTDOWN=True, ENABLE_H2_STORAGE=True) # All components on
        t1 = model.TimePeriods.first()
        time_factor = pyo.value(model.delT_minutes) / 60.0

        # Values from _setup_model_and_fix_vars and _create_base_data_inputs
        # pTurbine[t1]=20, vom_turbine=2.0
        # pElectrolyzer[t1]=15, vom_electrolyzer_LTE=1.5
        # BatteryCharge[t1]=3, BatteryDischarge[t1]=2, vom_battery=0.5
        # mHydrogenProduced[t1]=5, cost_water=0.1
        # pElectrolyzerRampPos[t1]=0 (as t_idx=0), cost_elec_ramping=0.05
        # H2_to_storage[t1]=3, H2_from_storage[t1]=1, vom_storage_cycle=0.01
        # vElectrolyzerStartup[t1]=1, cost_startup_elec=500
        cost_vom_turbine = model.pTurbine[t1].value * pyo.value(model.vom_turbine) * time_factor
        cost_vom_electrolyzer = model.pElectrolyzer[t1].value * pyo.value(model.vom_electrolyzer_LTE) * time_factor
        cost_vom_battery = (model.BatteryCharge[t1].value + model.BatteryDischarge[t1].value) * pyo.value(model.vom_battery_per_mwh_cycled) * time_factor
        cost_water = model.mHydrogenProduced[t1].value * pyo.value(model.cost_water_per_kg_h2) * time_factor
        cost_ramping = (model.pElectrolyzerRampPos[t1].value + 0) * pyo.value(model.cost_electrolyzer_ramping_LTE) # RampNeg is 0
        cost_storage_cycle = (model.H2_to_storage[t1].value + model.H2_from_storage[t1].value) * pyo.value(model.vom_storage_cycle) * time_factor
        cost_startup = model.vElectrolyzerStartup[t1].value * pyo.value(model.cost_startup_electrolyzer_LTE)

        expected_opex = cost_vom_turbine + cost_vom_electrolyzer + cost_vom_battery + \
                        cost_water + cost_ramping + cost_storage_cycle + cost_startup
        actual_opex = pyo.value(OpexCost_rule(model))
        self.assertAlmostEqual(actual_opex, expected_opex, places=2)


    def test_ancillary_revenue_ercot_bidding_mode(self):
        """Test AncillaryRevenue_ERCOT_rule in bidding mode."""
        iso = "ERCOT"
        num_hours = 1
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=False, num_hours=num_hours)
        t1 = model.TimePeriods.first()
        time_factor = 1.0

        # From _setup_model_and_fix_vars & _create_base_data_inputs:
        bid_regu_t1 = model.Total_RegUp[t1].value # 2.0
        p_regu_t1 = pyo.value(getattr(model, f'p_RegU_{iso}')[t1]) # 10
        loc_regu_t1 = pyo.value(getattr(model, f'loc_RegU_{iso}')[t1]) # 1
        win_regu_t1 = pyo.value(getattr(model, f'winning_rate_RegU_{iso}')[t1]) # 0.9
        lmp_t1 = pyo.value(model.energy_price[t1]) # 50

        expected_regu_rev_t1 = ( (bid_regu_t1 * win_regu_t1 * p_regu_t1) +      # Capacity: Bid * WinRate * MCP_Cap
                                  (bid_regu_t1 * 1.0 * 1.0 * lmp_t1) + # Performance: Bid * Mileage * Perf * LMP
                                  (loc_regu_t1)                      # Adder
                                ) * time_factor
        # Assume only RegU is bid for simplicity of this test point, or other bids are zero.
        total_sr_var = getattr(model, 'Total_SR', None)
        if total_sr_var is not None and total_sr_var.is_indexed():
            for t_sr in model.TimePeriods: total_sr_var[t_sr].fix(0) # Fix other bids to 0
        # ... zero out other AS bids if they exist and contribute ...

        actual_as_revenue = pyo.value(AncillaryRevenue_ERCOT_rule(model))
        self.assertAlmostEqual(actual_as_revenue, expected_regu_rev_t1, places=2,
                             msg=f"Expected {expected_regu_rev_t1}, Got {actual_as_revenue}")


    def test_ancillary_revenue_spp_dispatch_mode(self):
        """Test AncillaryRevenue_SPP_rule in dispatch simulation mode."""
        iso = "SPP"
        num_hours = 1
        model = self._setup_model_and_fix_vars(target_iso=iso, simulate_dispatch=True, num_hours=num_hours)
        t1 = model.TimePeriods.first()
        time_factor = 1.0

        # Mock get_total_deployed_as to return specific values for this test
        # Deployed values based on _setup_model_and_fix_vars for SPP @ t=1:
        # Deployed_RegU = Bid(1.0) * WinRate(0.85) = 0.85
        # Deployed_Spin(SR) = Bid(1.5) * WinRate(0.8) * DeployFactor(0.6) = 0.72
        deployed_regu_val = model.RegU_Electrolyzer_Deployed[t1].value
        deployed_spin_val = model.SR_Electrolyzer_Deployed[t1].value # Internal name is SR

        # Prices for SPP @ t1
        p_regu_t1 = pyo.value(getattr(model, f'p_RegU_{iso}')[t1]) # 11
        loc_regu_t1 = pyo.value(getattr(model, f'loc_RegU_{iso}')[t1]) # 1.1
        win_regu_t1 = pyo.value(getattr(model, f'winning_rate_RegU_{iso}')[t1]) # 0.85
        p_spin_t1 = pyo.value(getattr(model, f'p_Spin_{iso}')[t1]) # 6
        loc_spin_t1 = pyo.value(getattr(model, f'loc_Spin_{iso}')[t1]) # 0.4
        win_spin_t1 = pyo.value(getattr(model, f'winning_rate_Spin_{iso}')[t1]) # 0.8
        lmp_t1 = pyo.value(model.energy_price[t1]) # 50

        # Fixed bids @ t1
        bid_regu_t1 = model.Total_RegU[t1].value # 2.0
        bid_spin_t1 = model.Total_SR[t1].value # 3.0 (Internal name SR)


        with patch('revenue_cost.get_total_deployed_as') as mock_get_total_deployed:
            # Configure the mock to return the sum of fixed deployed values
            mock_get_total_deployed.side_effect = lambda m, t, service_name: \
                deployed_regu_val if service_name == 'RegUp' and t == t1 else \
                (deployed_spin_val if service_name == 'SR' and t == t1 else 0.0) # Match internal name SR for Spin


            # Expected RegU revenue
            cap_regu = bid_regu_t1 * win_regu_t1 * p_regu_t1
            perf_regu = deployed_regu_val * lmp_t1 # Deployed * LMP
            adder_regu = loc_regu_t1
            expected_regu_rev_t1 = (cap_regu + perf_regu + adder_regu) * time_factor

            # Expected Spin (SR) revenue
            cap_spin = bid_spin_t1 * win_spin_t1 * p_spin_t1
            energy_spin = deployed_spin_val * lmp_t1 # Deployed * LMP
            adder_spin = loc_spin_t1
            expected_spin_rev_t1 = (cap_spin + energy_spin + adder_spin) * time_factor


            expected_total_revenue = expected_regu_rev_t1 + expected_spin_rev_t1

            actual_as_revenue = pyo.value(AncillaryRevenue_SPP_rule(model))
            self.assertAlmostEqual(actual_as_revenue, expected_total_revenue, places=2,
                                 msg=f"Expected {expected_total_revenue}, Got {actual_as_revenue}")


if __name__ == '__main__':
    unittest.main()