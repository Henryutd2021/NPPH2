# src/data_io.py

"""
I/O utilities: load hourly price, ancillary-service parameter,
and system-parameter CSV files into DataFrames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from config import HOURS_IN_YEAR
from src.logger_utils.logging_setup import logger


def load_hourly_data(
    target_iso: str, base_dir: str | Path = "../input/hourly_data"
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Return a dictionary of DataFrames required by `model.create_model()`.

    Notes:
    - Fails fast if any *essential* file is missing.
    - Optional mileage/performance, deployment, and winning rate files are included when present.
      These files are expected to contain standardized column names. For example,
      MileageMultiplier_hourly.csv should contain columns like 'mileage_factor_RegUp_TARGETISO'
      and 'performance_factor_RegUp_TARGETISO' for all relevant ISOs.
    - sys_data_advanced.csv is expected to contain all necessary system parameters.
    - Price_ANS_hourly.csv is expected to contain standardized AS capacity prices (p_*)
      and locational adders (loc_*).
    """
    iso_path = Path(base_dir) / target_iso
    common_path = Path(base_dir)  # For files common across ISOs

    # Essential files
    required_files_spec = {
        "df_price_hourly": {
            "path": iso_path / "Price_hourly.csv",
            "index_col": None,
        },
        "df_ANSprice_hourly": {
            "path": iso_path / "Price_ANS_hourly.csv",
            "index_col": None,
        },
        "df_system": {
            "path": common_path / "sys_data_advanced.csv",
            "index_col": 0,
        },
    }

    # Optional files - these files should contain standardized column names
    # e.g., MileageMultiplier_hourly.csv should have 'mileage_factor_RegUp_PJM', 'performance_factor_RegUp_PJM'
    optional_files_spec = {
        "df_ANSmile_hourly": {
            "path": iso_path / "MileageMultiplier_hourly.csv",
            "index_col": None,
        },
        "df_ANSdeploy_hourly": {
            "path": iso_path / "DeploymentFactor_hourly.csv",
            "index_col": None,
        },
        "df_ANSwinrate_hourly": {
            "path": iso_path / "WinningRate_hourly.csv",
            "index_col": None,
        },
    }

    data: Dict[str, pd.DataFrame] = {}

    # Load required files
    for key, spec in required_files_spec.items():
        fpath = spec["path"]
        if not fpath.exists():
            logger.error(f"Essential file missing: {fpath}")
            return None
        try:
            df = pd.read_csv(fpath, index_col=spec["index_col"])
            if df.empty:
                logger.error(f"File {fpath} loaded as empty DataFrame.")
                return None

            # Validate and truncate hourly data length
            if "_hourly" in key:
                if len(df) < HOURS_IN_YEAR:
                    logger.warning(
                        f"{fpath.name} has {len(df)} rows (expected at least {HOURS_IN_YEAR}). This might be intended for shorter test runs."
                    )
                elif len(df) > HOURS_IN_YEAR:
                    logger.warning(
                        f"{fpath.name} has {len(df)} rows, truncating to {HOURS_IN_YEAR}."
                    )
                    df = df.iloc[:HOURS_IN_YEAR]

            data[key] = df
            logger.info(
                f"Loaded required {key} ({len(df)} rows) from {fpath.name}")
        except Exception as e:
            logger.error(
                f"Failed to load or process required file {fpath}: {e}",
                exc_info=True,
            )
            return None

    # Load optional files
    for key, spec in optional_files_spec.items():
        fpath = spec["path"]
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, index_col=spec["index_col"])
                if df.empty:
                    logger.warning(
                        f"Optional file {fpath} exists but loaded as empty DataFrame. Skipping."
                    )
                    continue

                if "_hourly" in key:  # All optional files here are hourly
                    if len(df) < HOURS_IN_YEAR:
                        logger.warning(
                            f"{fpath.name} has {len(df)} rows (expected at least {HOURS_IN_YEAR})."
                        )
                    elif len(df) > HOURS_IN_YEAR:
                        logger.warning(
                            f"{fpath.name} has {len(df)} rows, truncating to {HOURS_IN_YEAR}."
                        )
                        df = df.iloc[:HOURS_IN_YEAR]

                data[key] = df
                logger.info(
                    f"Loaded optional {key} ({len(df)} rows) from {fpath.name}")
            except Exception as e:
                logger.warning(
                    f"Failed to load or process optional file {fpath}, it will be skipped: {e}",
                    exc_info=True,
                )
        else:
            logger.info(f"Optional file {fpath} not found. Skipping.")

    return data
