# src/config.py
# No changes needed in this file based on the request.
# The existing flags already allow enabling/disabling components.
# The logic changes will be implemented in model.py, constraints.py, and revenue_cost.py

"""Global configuration and feature flags for the standardized
nuclear‑hydrogen optimization model.
"""
from pathlib import Path

# -----------------------------
# BASIC CONFIGURATION
# -----------------------------
TARGET_ISO: str = "ERCOT"          # 'CAISO' | 'ERCOT' | 'ISONE' | 'MISO' | 'NYISO' | 'PJM' | 'SPP'
HOURS_IN_YEAR: int = 8760        # Set to 24*7 for quick tests

# -----------------------------
# FEATURE FLAGS
# -----------------------------
# --- Core Technology Selection ---
ENABLE_NUCLEAR_GENERATOR: bool = True    # Enable the nuclear power plant (Turbine)
ENABLE_ELECTROLYZER: bool = True       # Master switch for any electrolyzer
ENABLE_LOW_TEMP_ELECTROLYZER: bool = False # If ENABLE_ELECTROLYZER is True, set True for LTE, False for HTE
ENABLE_BATTERY: bool = False          # Enable battery storage

# --- Advanced Feature Flags ---
ENABLE_H2_STORAGE: bool = False          # Enable separate hydrogen storage (requires ENABLE_ELECTROLYZER)
ENABLE_H2_CAP_FACTOR: bool = False      # Enforce H2 production target (usually disable when storage is enabled)
ENABLE_NONLINEAR_TURBINE_EFF: bool = True # Use piecewise linear turbine efficiency (requires ENABLE_NUCLEAR_GENERATOR)
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: bool = True # Track electrolyzer degradation (requires ENABLE_ELECTROLYZER)
ENABLE_STARTUP_SHUTDOWN: bool = True    # Use mixed-integer formulation for electrolyzer on/off (requires ENABLE_ELECTROLYZER)

# --- Simulation Mode ---
# Set to True to simulate AS dispatch execution affecting physical operation.
# Set to False to optimize bidding strategy based on capability (current default).
SIMULATE_AS_DISPATCH_EXECUTION: bool = False

# -----------------------------
# LOGGING
# -----------------------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / f"{TARGET_ISO}_optimization_standardized.log"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"  # Could be switched to DEBUG when needed

# --- Sanity Checks ---
# This flag derived from the config will be used elsewhere
CAN_PROVIDE_ANCILLARY_SERVICES: bool = ENABLE_NUCLEAR_GENERATOR and (ENABLE_ELECTROLYZER or ENABLE_BATTERY)
print(f"System configured to provide Ancillary Services: {CAN_PROVIDE_ANCILLARY_SERVICES}")
if ENABLE_H2_STORAGE and not ENABLE_ELECTROLYZER:
    print("Warning: ENABLE_H2_STORAGE=True but ENABLE_ELECTROLYZER=False. Disabling H2 storage.")
    ENABLE_H2_STORAGE = False
if ENABLE_LOW_TEMP_ELECTROLYZER and not ENABLE_ELECTROLYZER:
    print("Warning: ENABLE_LOW_TEMP_ELECTROLYZER=True but ENABLE_ELECTROLYZER=False. Ignoring LTE setting.")
    ENABLE_LOW_TEMP_ELECTROLYZER = False # Or handle as error
if ENABLE_H2_CAP_FACTOR and not ENABLE_ELECTROLYZER:
    print("Warning: ENABLE_H2_CAP_FACTOR=True but ENABLE_ELECTROLYZER=False. Disabling H2 cap factor.")
    ENABLE_H2_CAP_FACTOR = False
if ENABLE_ELECTROLYZER_DEGRADATION_TRACKING and not ENABLE_ELECTROLYZER:
    print("Warning: ENABLE_ELECTROLYZER_DEGRADATION_TRACKING=True but ENABLE_ELECTROLYZER=False. Disabling degradation tracking.")
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING = False
if ENABLE_STARTUP_SHUTDOWN and not ENABLE_ELECTROLYZER:
    print("Warning: ENABLE_STARTUP_SHUTDOWN=True but ENABLE_ELECTROLYZER=False. Disabling startup/shutdown logic.")
    ENABLE_STARTUP_SHUTDOWN = False
if ENABLE_NONLINEAR_TURBINE_EFF and not ENABLE_NUCLEAR_GENERATOR:
     print("Warning: ENABLE_NONLINEAR_TURBINE_EFF=True but ENABLE_NUCLEAR_GENERATOR=False. Disabling nonlinear turbine.")
     ENABLE_NONLINEAR_TURBINE_EFF = False
if SIMULATE_AS_DISPATCH_EXECUTION and not CAN_PROVIDE_ANCILLARY_SERVICES:
    print("Warning: SIMULATE_AS_DISPATCH_EXECUTION=True but CAN_PROVIDE_ANCILLARY_SERVICES=False. Dispatch simulation has no effect.")
    SIMULATE_AS_DISPATCH_EXECUTION = False # Or handle as needed

