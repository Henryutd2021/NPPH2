"""
Test module for regulation balance constraints.
Tests that RegUp and RegDown bids are equal for each component.
"""

from model import create_model
import unittest
from unittest.mock import patch
import pyomo.environ as pyo
import pandas as pd
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRegulationBalance(unittest.TestCase):
    """Test regulation balance constraints."""

    def setUp(self):
        """Set up test data for model creation."""
        # Create minimal test data
        self.data_inputs = {
            "df_price_hourly": pd.DataFrame({
                "Price ($/MWh)": [50.0, 60.0, 70.0]
            }),
            "df_system": pd.DataFrame({
                "Parameter": [
                    "delT_minutes",
                    "qSteam_Total_MWth",
                    "qSteam_Turbine_min_MWth",
                    "qSteam_Turbine_max_MWth",
                    "pTurbine_min_MW",
                    "pTurbine_max_MW",
                    "pElectrolyzer_min_MW",
                    "pElectrolyzer_max_upper_bound_MW",
                    "pElectrolyzer_max_lower_bound_MW",
                    "pElectrolyzer_Breakpoints_MW",
                    "ke_H2_Values_MWh_per_kg",
                    "BatteryCapacity_max_MWh",
                    "BatteryCapacity_min_MWh",
                    "BatteryPowerRatio_MW_per_MWh",
                    "H2_value_USD_per_kg"
                ],
                "Value": [
                    60.0,    # delT_minutes
                    1000.0,  # qSteam_Total_MWth
                    200.0,   # qSteam_Turbine_min_MWth
                    1000.0,  # qSteam_Turbine_max_MWth
                    100.0,   # pTurbine_min_MW
                    400.0,   # pTurbine_max_MW
                    10.0,    # pElectrolyzer_min_MW
                    200.0,   # pElectrolyzer_max_upper_bound_MW
                    0.0,     # pElectrolyzer_max_lower_bound_MW
                    "10.0,50.0,100.0,200.0",  # pElectrolyzer_Breakpoints_MW
                    "0.05,0.04,0.045,0.048",  # ke_H2_Values_MWh_per_kg
                    100.0,   # BatteryCapacity_max_MWh
                    0.0,     # BatteryCapacity_min_MWh
                    0.25,    # BatteryPowerRatio_MW_per_MWh
                    3.0      # H2_value_USD_per_kg
                ]
            }).set_index("Parameter")
        }

    @patch('config.ENABLE_NUCLEAR_GENERATOR', True)
    @patch('config.ENABLE_ELECTROLYZER', True)
    @patch('config.ENABLE_BATTERY', True)
    @patch('config.CAN_PROVIDE_ANCILLARY_SERVICES', True)
    @patch('config.SIMULATE_AS_DISPATCH_EXECUTION', False)
    def test_regulation_balance_constraints_exist(self):
        """Test that regulation balance constraints are added to the model."""
        model = create_model(self.data_inputs, "PJM", False)

        # Check that regulation balance constraints exist
        self.assertTrue(hasattr(model, "battery_regulation_balance_constr"))
        self.assertTrue(
            hasattr(model, "electrolyzer_regulation_balance_constr"))
        self.assertTrue(hasattr(model, "turbine_regulation_balance_constr"))

        # Check that the constraints are active
        self.assertTrue(model.battery_regulation_balance_constr[1].active)
        self.assertTrue(model.electrolyzer_regulation_balance_constr[1].active)
        self.assertTrue(model.turbine_regulation_balance_constr[1].active)

    @patch('config.ENABLE_NUCLEAR_GENERATOR', True)
    @patch('config.ENABLE_ELECTROLYZER', True)
    @patch('config.ENABLE_BATTERY', True)
    @patch('config.CAN_PROVIDE_ANCILLARY_SERVICES', True)
    @patch('config.SIMULATE_AS_DISPATCH_EXECUTION', False)
    def test_regulation_balance_constraint_structure(self):
        """Test that the regulation balance constraints have the correct structure."""
        model = create_model(self.data_inputs, "PJM", False)

        # Test battery constraint structure
        t = 1
        constraint_body = model.battery_regulation_balance_constr[t].body
        self.assertIn("RegUp_Battery", str(constraint_body))
        self.assertIn("RegDown_Battery", str(constraint_body))

        # Test electrolyzer constraint structure
        constraint_body = model.electrolyzer_regulation_balance_constr[t].body
        self.assertIn("RegUp_Electrolyzer", str(constraint_body))
        self.assertIn("RegDown_Electrolyzer", str(constraint_body))

        # Test turbine constraint structure
        constraint_body = model.turbine_regulation_balance_constr[t].body
        self.assertIn("RegUp_Turbine", str(constraint_body))
        self.assertIn("RegDown_Turbine", str(constraint_body))

    @patch('config.ENABLE_NUCLEAR_GENERATOR', True)
    @patch('config.ENABLE_ELECTROLYZER', False)
    @patch('config.ENABLE_BATTERY', True)
    @patch('config.CAN_PROVIDE_ANCILLARY_SERVICES', True)
    @patch('config.SIMULATE_AS_DISPATCH_EXECUTION', False)
    def test_regulation_balance_only_battery_enabled(self):
        """Test regulation balance when only battery is enabled."""
        model = create_model(self.data_inputs, "PJM", False)

        # Should have battery constraint
        self.assertTrue(hasattr(model, "battery_regulation_balance_constr"))

        # Should not have electrolyzer constraint
        self.assertFalse(
            hasattr(model, "electrolyzer_regulation_balance_constr"))

        # Should not have turbine constraint (requires electrolyzer or battery + turbine)
        self.assertFalse(hasattr(model, "turbine_regulation_balance_constr"))

    @patch('config.ENABLE_NUCLEAR_GENERATOR', False)
    @patch('config.ENABLE_ELECTROLYZER', False)
    @patch('config.ENABLE_BATTERY', False)
    @patch('config.CAN_PROVIDE_ANCILLARY_SERVICES', False)
    @patch('config.SIMULATE_AS_DISPATCH_EXECUTION', False)
    def test_no_regulation_balance_when_as_disabled(self):
        """Test that no regulation balance constraints are added when AS is disabled."""
        model = create_model(self.data_inputs, "PJM", False)

        # Should not have any regulation balance constraints
        self.assertFalse(hasattr(model, "battery_regulation_balance_constr"))
        self.assertFalse(
            hasattr(model, "electrolyzer_regulation_balance_constr"))
        self.assertFalse(hasattr(model, "turbine_regulation_balance_constr"))


if __name__ == "__main__":
    unittest.main()
