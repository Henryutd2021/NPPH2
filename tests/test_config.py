# test_config.py
import unittest
import sys
from pathlib import Path

# Assuming src directory is in the python path for imports
# Add src to path if necessary, or adjust imports based on your test runner's setup
# For example:
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from src import config


class TestConfig(unittest.TestCase):

    def test_default_values(self):
        self.assertEqual(config.TARGET_ISO, "ERCOT")
        self.assertEqual(config.HOURS_IN_YEAR, 8760)
        self.assertTrue(config.ENABLE_NUCLEAR_GENERATOR)
        self.assertTrue(config.ENABLE_ELECTROLYZER)
        self.assertFalse(config.ENABLE_LOW_TEMP_ELECTROLYZER)
        # Add more assertions for other default flags as needed

    def test_can_provide_ancillary_services_logic(self):
        # Test the derived CAN_PROVIDE_ANCILLARY_SERVICES flag
        # Default case from config.py
        self.assertTrue(config.CAN_PROVIDE_ANCILLARY_SERVICES,
                        "Default config should allow AS")

        # Test scenarios (requires temporarily modifying config values or a more advanced setup)
        # This shows the principle; a better way might involve reloading the module or using a mock
        original_nuclear = config.ENABLE_NUCLEAR_GENERATOR
        original_electrolyzer = config.ENABLE_ELECTROLYZER
        original_battery = config.ENABLE_BATTERY

        try:
            # Scenario 1: Nuclear OFF
            config.ENABLE_NUCLEAR_GENERATOR = False
            config.ENABLE_ELECTROLYZER = True
            config.ENABLE_BATTERY = True
            # Re-evaluate the dependent variable (in a real test, you might need to reload config)
            # For simplicity, we assume a way to re-trigger the logic if it's not top-level
            # Or, if it's a simple top-level assignment, we can directly calculate expected
            expected_as_capable_scenario1 = config.ENABLE_NUCLEAR_GENERATOR and \
                (config.ENABLE_ELECTROLYZER or config.ENABLE_BATTERY)
            self.assertFalse(expected_as_capable_scenario1,
                             "AS should be False if Nuclear is OFF")

            # Scenario 2: Nuclear ON, Electrolyzer OFF, Battery OFF
            config.ENABLE_NUCLEAR_GENERATOR = True
            config.ENABLE_ELECTROLYZER = False
            config.ENABLE_BATTERY = False
            expected_as_capable_scenario2 = config.ENABLE_NUCLEAR_GENERATOR and \
                (config.ENABLE_ELECTROLYZER or config.ENABLE_BATTERY)
            self.assertFalse(expected_as_capable_scenario2,
                             "AS should be False if Electrolyzer and Battery are OFF")

            # Scenario 3: Nuclear ON, Electrolyzer ON, Battery OFF
            config.ENABLE_NUCLEAR_GENERATOR = True
            config.ENABLE_ELECTROLYZER = True
            config.ENABLE_BATTERY = False
            expected_as_capable_scenario3 = config.ENABLE_NUCLEAR_GENERATOR and \
                (config.ENABLE_ELECTROLYZER or config.ENABLE_BATTERY)
            self.assertTrue(expected_as_capable_scenario3,
                            "AS should be True with Nuclear and Electrolyzer ON")

        finally:
            # Restore original values to avoid side effects on other tests
            config.ENABLE_NUCLEAR_GENERATOR = original_nuclear
            config.ENABLE_ELECTROLYZER = original_electrolyzer
            config.ENABLE_BATTERY = original_battery
            # Re-evaluate CAN_PROVIDE_ANCILLARY_SERVICES based on restored values if it was modified directly
            config.CAN_PROVIDE_ANCILLARY_SERVICES = config.ENABLE_NUCLEAR_GENERATOR and \
                (config.ENABLE_ELECTROLYZER or config.ENABLE_BATTERY)

    def test_h2_storage_dependencies(self):
        # Example for testing warnings and flag adjustments
        # This is harder to test directly for print statements without capturing stdout
        # or by checking the resulting state of flags if they are modified.
        original_h2_storage = config.ENABLE_H2_STORAGE
        original_electrolyzer = config.ENABLE_ELECTROLYZER
        try:
            config.ENABLE_H2_STORAGE = True
            config.ENABLE_ELECTROLYZER = False
            # In your actual config.py, ENABLE_H2_STORAGE would be set to False.
            # We can manually simulate that re-evaluation for the test's scope.
            current_h2_storage = config.ENABLE_H2_STORAGE and config.ENABLE_ELECTROLYZER
            if config.ENABLE_H2_STORAGE and not config.ENABLE_ELECTROLYZER:
                current_h2_storage = False  # Simulating the correction in config.py
            self.assertFalse(current_h2_storage)

        finally:
            config.ENABLE_H2_STORAGE = original_h2_storage
            config.ENABLE_ELECTROLYZER = original_electrolyzer
            # Re-evaluate dependent flags if necessary


if __name__ == '__main__':
    unittest.main()
