# src/data_io.py

"""I/O utilities: load hourly price, ancillary‑service and system‑parameter
CSV files into DataFrames.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from logging_setup import logger
from config import HOURS_IN_YEAR

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def load_hourly_data(target_iso: str, base_dir: str | Path = "../input/hourly_data") -> Optional[Dict[str, pd.DataFrame]]:
    """Return a dictionary of DataFrames required by `model.create_model()`.

    Notes
    -----
    * Fails fast if any *essential* file is missing.
    * Optional mileage / performance / deployment files are included when present.
    * Requires new parameters in sys_data_advanced.csv (e.g., electrolyzer capacity bounds/cost, LTE setpoint).
    * Requires standardized AS parameters in Price_ANS_hourly.csv (e.g., RegCap_*, RegPerf_*, SR_Price_*, NSR_Price_*, Loc_* etc.).
    """

    iso_path = Path(base_dir) / target_iso
    common_path = Path(base_dir)

    # Essential files
    required = {
        "df_price_hourly": iso_path / "Price_hourly.csv",
        "df_ANSprice_hourly": iso_path / "Price_ANS_hourly.csv", # Expects standardized columns now
        "df_system": common_path / "sys_data_advanced.csv", # Expects new system params
    }

    # Optional files
    optional_files = {
         "df_ANSmile_hourly": iso_path / "MileageMultiplier_hourly.csv", # For performance/mileage factors
         "df_ANSdeploy_hourly": iso_path / "DeploymentFactor_hourly.csv" # --- Added --- For reserve deployment factors
    }

    data: Dict[str, pd.DataFrame] = {}

    # Load required files first
    for key, fpath in required.items():
        if not fpath.exists():
            logger.error("Essential file missing: %s", fpath)
            return None
        try:
            df = pd.read_csv(fpath, index_col=0 if key == "df_system" else None)
            if df.empty:
                logger.error("File %s loaded as empty DataFrame", fpath)
                return None
            if "_hourly" in key and len(df) < HOURS_IN_YEAR:
                logger.warning("%s has %d rows (expected %d)", fpath.name, len(df), HOURS_IN_YEAR)
            data[key] = df
            logger.info("Loaded required %s (%d rows) from %s", key, len(df), fpath.name)
        except Exception as e:
            logger.error(f"Failed to load or process required file {fpath}: {e}")
            return None

    # Load optional files if they exist
    for key, fpath in optional_files.items():
         if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if df.empty:
                     logger.warning("Optional file %s exists but loaded as empty DataFrame.", fpath)
                     continue # Skip if empty
                if "_hourly" in key and len(df) < HOURS_IN_YEAR:
                     logger.warning("%s has %d rows (expected %d)", fpath.name, len(df), HOURS_IN_YEAR)
                data[key] = df
                logger.info("Loaded optional %s (%d rows) from %s", key, len(df), fpath.name)
            except Exception as e:
                logger.warning(f"Failed to load or process optional file {fpath}, it will be skipped: {e}")
         else:
            logger.info("Optional file %s not found, skipping.", fpath)


    return data