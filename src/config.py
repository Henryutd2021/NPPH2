# src/config.py

"""Global configuration and feature flags for the standardized
nuclear‑hydrogen optimization model.
"""
from pathlib import Path

# -----------------------------
# BASIC CONFIGURATION
# -----------------------------
TARGET_ISO: str = "CAISO"          # 'CAISO' | 'ERCOT' | 'ISONE' | 'MISO' | 'NYISO' | 'PJM' | 'SPP'
HOURS_IN_YEAR: int = 8760        # Set to 24*7 for quick tests

# -----------------------------
# FEATURE FLAGS
# -----------------------------
ENABLE_H2_STORAGE: bool = True
ENABLE_H2_CAP_FACTOR: bool = False      # usually disable when storage is enabled
ENABLE_NONLINEAR_TURBINE_EFF: bool = True
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: bool = True
ENABLE_STARTUP_SHUTDOWN: bool = True    # Mixed‑integer formulation
# --- Added ---
ENABLE_LOW_TEMP_ELECTROLYZER: bool = False # Set to True to enable LTE mode

# -----------------------------
# LOGGING
# -----------------------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / f"{TARGET_ISO}_optimization_standardized.log"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"  # Could be switched to DEBUG when needed