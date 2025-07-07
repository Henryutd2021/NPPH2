"""
Data loading functions for the TEA module.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import src.tea.config as config
from typing import Any, Union, Dict, Tuple
from datetime import datetime

# Detailed logging setup
logger = logging.getLogger(__name__)

# Using fallback default values if config module doesn't have some of the required constants.
# This is a defensive programming approach to ensure the module works even if config.py doesn't have all expected values.
try:
    DEFAULT_CAPEX_COMPONENTS = getattr(config, 'CAPEX_COMPONENTS', {})
    DEFAULT_REPLACEMENT_SCHEDULE = getattr(config, 'REPLACEMENT_SCHEDULE', {})
    DEFAULT_PROJECT_LIFETIME_YEARS = getattr(
        config, 'PROJECT_LIFETIME_YEARS', 30)
    DEFAULT_DISCOUNT_RATE = getattr(config, 'DISCOUNT_RATE', 0.08)
    DEFAULT_CONSTRUCTION_YEARS = getattr(config, 'CONSTRUCTION_YEARS', 2)
    DEFAULT_TAX_RATE = getattr(config, 'TAX_RATE', 0.21)
    DEFAULT_OM_COMPONENTS = getattr(config, 'OM_COMPONENTS', {})
    DEFAULT_NUCLEAR_INTEGRATED_CONFIG = getattr(
        config, 'NUCLEAR_INTEGRATED_CONFIG', {})
    logger.info("Successfully loaded configuration from tea.config")
except ImportError as e:
    logger.warning(
        f"Could not import from tea.config. Using fallback default values: {e}")
    # If config import fails, provide sensible defaults
    DEFAULT_CAPEX_COMPONENTS = {}
    DEFAULT_REPLACEMENT_SCHEDULE = {}
    DEFAULT_PROJECT_LIFETIME_YEARS = 30
    DEFAULT_DISCOUNT_RATE = 0.08
    DEFAULT_CONSTRUCTION_YEARS = 2
    DEFAULT_TAX_RATE = 0.21
    DEFAULT_OM_COMPONENTS = {}
    DEFAULT_NUCLEAR_INTEGRATED_CONFIG = {}
except Exception as e:
    logger.error(f"Unexpected error importing from tea.config: {e}")
    # Same fallback values
    DEFAULT_CAPEX_COMPONENTS = {}
    DEFAULT_REPLACEMENT_SCHEDULE = {}
    DEFAULT_PROJECT_LIFETIME_YEARS = 30
    DEFAULT_DISCOUNT_RATE = 0.08
    DEFAULT_CONSTRUCTION_YEARS = 2
    DEFAULT_TAX_RATE = 0.21
    DEFAULT_OM_COMPONENTS = {}
    DEFAULT_NUCLEAR_INTEGRATED_CONFIG = {}

# Helper function (can be module-level or stay nested if preferred)


def _get_param_value(params_dict, key, default_val, type_converter, param_logger):
    val = params_dict.get(key)
    if val is None or pd.isna(val):
        param_logger.info(
            f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_val}"
        )
        return default_val
    try:
        return type_converter(val)
    except (ValueError, TypeError):
        param_logger.warning(
            f"Invalid value '{val}' for '{key}' in sys_data. Using default: {default_val}"
        )
        return default_val


def load_tax_incentive_policies(params: dict) -> dict:
    """
    Load and update tax incentive policy parameters from sys_data_advanced.csv.

    Args:
        params: Dictionary of parameters loaded from sys_data_advanced.csv

    Returns:
        Updated tax incentive policies dictionary
    """
    try:
        from src.tea.config import TAX_INCENTIVE_POLICIES
        tax_policies = TAX_INCENTIVE_POLICIES.copy()

        # Update 45U PTC parameters if available in CSV
        if params.get("45u_ptc_rate_per_mwh") is not None:
            tax_policies["45u_ptc"]["credit_rate_per_mwh"] = _get_param_value(
                params, "45u_ptc_rate_per_mwh",
                tax_policies["45u_ptc"]["credit_rate_per_mwh"],
                float, logger
            )

        if params.get("45u_ptc_start_year") is not None:
            tax_policies["45u_ptc"]["credit_start_year"] = _get_param_value(
                params, "45u_ptc_start_year",
                tax_policies["45u_ptc"]["credit_start_year"],
                int, logger
            )

        if params.get("45u_ptc_end_year") is not None:
            tax_policies["45u_ptc"]["credit_end_year"] = _get_param_value(
                params, "45u_ptc_end_year",
                tax_policies["45u_ptc"]["credit_end_year"],
                int, logger
            )

        # Update 45Y PTC parameters if available in CSV
        if params.get("45y_ptc_rate_per_mwh") is not None:
            tax_policies["45y_ptc"]["credit_rate_per_mwh"] = _get_param_value(
                params, "45y_ptc_rate_per_mwh",
                tax_policies["45y_ptc"]["credit_rate_per_mwh"],
                float, logger
            )

        if params.get("45y_ptc_duration_years") is not None:
            tax_policies["45y_ptc"]["credit_duration_years"] = _get_param_value(
                params, "45y_ptc_duration_years",
                tax_policies["45y_ptc"]["credit_duration_years"],
                int, logger
            )

        # Update 48E ITC parameters if available in CSV
        if params.get("48e_itc_rate") is not None:
            tax_policies["48e_itc"]["credit_rate"] = _get_param_value(
                params, "48e_itc_rate",
                tax_policies["48e_itc"]["credit_rate"],
                float, logger
            )

        if params.get("48e_itc_depreciation_basis_reduction_rate") is not None:
            tax_policies["48e_itc"]["depreciation_basis_reduction_rate"] = _get_param_value(
                params, "48e_itc_depreciation_basis_reduction_rate",
                tax_policies["48e_itc"]["depreciation_basis_reduction_rate"],
                float, logger
            )

        logger.info(
            "Successfully loaded and updated tax incentive policies from CSV parameters")
        return tax_policies

    except ImportError as e:
        logger.warning(
            f"Could not import TAX_INCENTIVE_POLICIES from config: {e}")
        # Return default policies if config import fails
        return {
            "45u_ptc": {"credit_rate_per_mwh": 15.0, "credit_start_year": 2024, "credit_end_year": 2032},
            "45y_ptc": {"credit_rate_per_mwh": 30.0, "credit_duration_years": 10},
            "48e_itc": {"credit_rate": 0.50, "depreciation_basis_reduction_rate": 0.50}
        }


def extract_plant_name_from_file_path(file_path: Union[str, Path]) -> str:
    """
    Extract plant name from hourly results file path.

    Expected format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}_hourly_results.csv

    Args:
        file_path: Path to the hourly results file

    Returns:
        Plant name extracted from file path, or None if not found
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        filename = file_path.stem  # Get filename without extension

        # Remove '_hourly_results' suffix if present
        if filename.endswith('_hourly_results'):
            filename = filename[:-len('_hourly_results')]

        # Split by underscore and reconstruct plant name
        # Format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}
        parts = filename.split('_')

        if len(parts) >= 4:
            # Remove last 3 parts (Unit_ID, ISO, Remaining_Years)
            plant_name_parts = parts[:-3]
            # FIXED: Join with spaces instead of underscores to match NPPs info.csv format
            plant_name = ' '.join(plant_name_parts)

            logger.info(f"Extracted plant name from file path: '{plant_name}'")
            return plant_name
        else:
            logger.warning(
                f"Could not parse plant name from filename: {filename}")
            return None

    except Exception as e:
        logger.warning(
            f"Error extracting plant name from file path {file_path}: {e}")
        return None


def extract_remaining_years_from_file_path(file_path: Union[str, Path]) -> int:
    """
    Extract remaining years from hourly results file path.

    Expected format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}_hourly_results.csv
    Example: Comanche Peak_2_ERCOT_28_hourly_results.csv -> 28

    Args:
        file_path: Path to the hourly results file

    Returns:
        Remaining years extracted from file path, or None if not found
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        filename = file_path.stem  # Get filename without extension

        # Remove '_hourly_results' suffix if present
        if filename.endswith('_hourly_results'):
            filename = filename[:-len('_hourly_results')]

        # Split by underscore and get remaining years
        # Format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}
        parts = filename.split('_')

        if len(parts) >= 4:
            # The last part should be the remaining years
            remaining_years_str = parts[-1]
            try:
                remaining_years = int(remaining_years_str)
                logger.info(
                    f"Extracted remaining years from file path: {remaining_years} years")
                return remaining_years
            except ValueError:
                logger.warning(
                    f"Could not parse remaining years as integer: {remaining_years_str}")
                return None
        else:
            logger.warning(
                f"Could not parse remaining years from filename: {filename}")
            return None

    except Exception as e:
        logger.warning(
            f"Error extracting remaining years from file path {file_path}: {e}")
        return None


def extract_plant_info_from_file_path(file_path: Union[str, Path]) -> dict:
    """
    Extract comprehensive plant information from hourly results file path.

    Expected format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}_hourly_results.csv
    Example: Comanche Peak_2_ERCOT_28_hourly_results.csv

    Args:
        file_path: Path to the hourly results file

    Returns:
        Dictionary containing extracted plant information
    """
    try:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        filename = file_path.stem  # Get filename without extension

        # Remove '_hourly_results' suffix if present
        if filename.endswith('_hourly_results'):
            filename = filename[:-len('_hourly_results')]

        # Split by underscore
        # Format: {Plant_Name}_{Unit_ID}_{ISO}_{Remaining_Years}
        parts = filename.split('_')

        if len(parts) >= 4:
            # Extract all components
            plant_name_parts = parts[:-3]
            unit_id = parts[-3]
            iso_region = parts[-2]
            remaining_years_str = parts[-1]

            plant_name = ' '.join(plant_name_parts)

            try:
                remaining_years = int(remaining_years_str)
            except ValueError:
                logger.warning(
                    f"Could not parse remaining years as integer: {remaining_years_str}")
                remaining_years = None

            plant_info = {
                'plant_name': plant_name,
                'unit_id': unit_id,
                'iso_region': iso_region,
                'remaining_years': remaining_years,
                'filename': filename,
                'file_path': str(file_path)
            }

            logger.info(
                f"Extracted plant info from file path: {plant_name} Unit {unit_id}, {iso_region}, {remaining_years} years remaining")
            return plant_info
        else:
            logger.warning(
                f"Could not parse plant info from filename: {filename}")
            return None

    except Exception as e:
        logger.warning(
            f"Error extracting plant info from file path {file_path}: {e}")
        return None


def load_tea_sys_params(iso_target: str, input_base_dir: Path, npps_info_path: str = None, case_type: str = None, hourly_results_file_path: str = None) -> tuple[dict, int, float, int, float, dict, dict, dict]:
    """
    Loads TEA-relevant system parameters.

    Args:
        iso_target: Target ISO region
        input_base_dir: Base input directory path
        npps_info_path: Optional path to NPPs info file for extracting actual remaining years
        case_type: Optional case type for determining project lifetime
        hourly_results_file_path: Optional path to hourly results file for extracting plant name

    Returns a tuple containing:
    - params (dict): The raw parameters loaded from CSV.
    - project_lifetime (int): Project lifetime in years (actual remaining years if available).
    - discount_rate (float): Discount rate as a fraction.
    - construction_years (int): Construction period in years.
    - tax_rate (float): Corporate tax rate as a fraction.
    - om_components (dict): O&M components dictionary, potentially updated.
    - nuclear_config (dict): Nuclear integrated config dictionary, potentially updated.
    - tax_policies (dict): Tax incentive policies dictionary, potentially updated.
    """
    logger.debug(f"load_tea_sys_params called for ISO: {iso_target}")
    params = {}

    # Use aliased default values from config
    project_lifetime_years = DEFAULT_PROJECT_LIFETIME_YEARS
    discount_rate = DEFAULT_DISCOUNT_RATE
    construction_years = DEFAULT_CONSTRUCTION_YEARS
    tax_rate = DEFAULT_TAX_RATE
    # Deep copy mutable defaults to avoid modifying them globally if they are updated
    om_components = {k: v.copy() if isinstance(
        v, dict) else v for k, v in DEFAULT_OM_COMPONENTS.items()}
    nuclear_integrated_config = {k: v.copy() if isinstance(
        v, dict) else v for k, v in DEFAULT_NUCLEAR_INTEGRATED_CONFIG.items()}

    try:
        sys_data_file_path = input_base_dir / "hourly_data" / "sys_data_advanced.csv"
        if not sys_data_file_path.exists():
            sys_data_file_path = input_base_dir / "sys_data_advanced.csv"
        logger.debug(f"Attempting to load sys_data from: {sys_data_file_path}")

        if sys_data_file_path.exists():
            df_system = pd.read_csv(sys_data_file_path, index_col=0)
            param_keys = [
                "hydrogen_subsidy_value_usd_per_kg",
                "hydrogen_subsidy_duration_years",
                "user_specified_electrolyzer_capacity_MW",
                "user_specified_h2_storage_capacity_kg",
                "user_specified_battery_capacity_MWh",
                "user_specified_battery_power_MW",
                "plant_lifetime_years",
                "baseline_nuclear_annual_revenue_USD",
                "enable_incremental_analysis",
                "discount_rate_fraction",
                "project_construction_years",
                "corporate_tax_rate_fraction",
                "BatteryFixedOM_USD_per_MW_year",
                "BatteryFixedOM_USD_per_MWh_year",
                "enable_nuclear_integrated_analysis",
                "nuclear_plant_capacity_MW",
                "nuclear_project_lifetime_years",
                "nuclear_construction_years",
                "enable_nuclear_capex_costs",
                "enable_nuclear_opex_costs",
                # Add hydrogen price parameters
                "H2_value_USD_per_kg",
                "hydrogen_price_usd_per_kg",
                "h2_price_usd_per_kg",
                # Add tax incentive policy parameters
                "45u_ptc_rate_per_mwh",
                "45u_ptc_start_year",
                "45u_ptc_end_year",
                "45y_ptc_rate_per_mwh",
                "45y_ptc_duration_years",
                "48e_itc_rate",
                "48e_itc_depreciation_basis_reduction_rate",
            ]
            for key in param_keys:
                if key in df_system.index:
                    value_series = df_system.loc[key, "Value"]
                    params[key] = (
                        value_series.iloc[0]
                        if isinstance(value_series, pd.Series)
                        else value_series
                    )
                else:
                    params[key] = None
            logger.info(
                f"Successfully loaded TEA relevant params from {sys_data_file_path}"
            )
        else:
            logger.warning(
                f"sys_data_advanced.csv not found at {sys_data_file_path}. TEA will use defaults for some parameters."
            )

    except Exception as e:
        logger.error(
            f"Error loading TEA system data from {sys_data_file_path}: {e}")
        logger.debug(f"Error in load_tea_sys_params: {e}", exc_info=True)

    # First get project lifetime from CSV parameters
    project_lifetime_years = _get_param_value(
        params,
        "plant_lifetime_years",
        project_lifetime_years,  # Default from config
        lambda x: int(float(x)),
        logger,
    )

    # Apply case-specific lifetime logic based on case_type
    from . import config

    # Determine if this is an existing project or new construction
    is_existing_project = True  # Default assumption
    if case_type:
        case_lower = case_type.lower()
        if any(case in case_lower for case in ["case4", "case5", "greenfield", "new"]):
            is_existing_project = False
        elif any(case in case_lower for case in ["case1", "case2", "case3", "existing", "retrofit"]):
            is_existing_project = True

    logger.info(
        f"Case type: {case_type}, Existing project: {is_existing_project}")

    # For existing projects (case1-3), try to get actual remaining years from hourly results filename FIRST
    remaining_years_from_filename = None
    if is_existing_project and hourly_results_file_path:
        # PRIORITY 1: Extract remaining years directly from filename (more reliable)
        remaining_years_from_filename = extract_remaining_years_from_file_path(
            hourly_results_file_path)
        if remaining_years_from_filename is not None:
            logger.info(
                f"EXISTING PROJECT: Using remaining lifetime from filename: {remaining_years_from_filename} years")
            logger.info(
                f"  (Extracted from hourly results filename: {Path(hourly_results_file_path).name})")
            project_lifetime_years = remaining_years_from_filename

    # PRIORITY 2: Always try NPPs info matching for plant identification (but only use remaining years if filename extraction failed)
    if is_existing_project and npps_info_path:
        try:
            npps_path = Path(npps_info_path)
            if npps_path.exists():
                npp_info_df = pd.read_csv(npps_path)

                # Extract plant name AND generator ID from multiple sources
                specific_plant_name = None
                specific_generator_id = None

                # First priority: Extract from hourly results file path
                if hourly_results_file_path:
                    plant_info = extract_plant_info_from_file_path(
                        hourly_results_file_path)
                    if plant_info and plant_info.get('plant_name') and plant_info.get('unit_id'):
                        specific_plant_name = plant_info['plant_name']
                        specific_generator_id = plant_info['unit_id']
                        logger.info(
                            f"Found specific plant info from hourly results file path: {specific_plant_name} Unit {specific_generator_id}")

                # Fallback to old extraction method if new method fails
                if not specific_plant_name and hourly_results_file_path:
                    specific_plant_name = extract_plant_name_from_file_path(
                        hourly_results_file_path)
                    if specific_plant_name:
                        logger.info(
                            f"Found specific plant name from hourly results file path: {specific_plant_name} (Generator ID not extracted)")

                # Second priority: Extract from tea_sys_params if file path extraction failed
                if not specific_plant_name:
                    plant_identification_keys = [
                        'plant_name', 'Plant_Name', 'original_plant_name', 'Original_Plant_Name',
                        'turbine_name', 'reactor_name'
                    ]
                    for key in plant_identification_keys:
                        if key in params and params[key]:
                            specific_plant_name = str(params[key]).strip()
                            logger.info(
                                f"Found specific plant name from params[{key}]: {specific_plant_name}")
                            break

                # Try to match specific plant first, then fall back to ISO filtering
                npp_info = None
                plant_name = 'Unknown'

                if specific_plant_name:
                    # ENHANCED PLANT MATCHING: Try multiple strategies to find the best match
                    logger.info(
                        f"Attempting to match plant: '{specific_plant_name}' in ISO: {iso_target}")

                    # Strategy 1: Exact match (case-insensitive)
                    exact_matches = npp_info_df[
                        npp_info_df['Plant Name'].str.lower(
                        ) == specific_plant_name.lower()
                    ]

                    # Strategy 2: Contains match (original logic)
                    contains_matches = npp_info_df[
                        npp_info_df['Plant Name'].str.contains(
                            specific_plant_name, case=False, na=False)
                    ]

                    # Strategy 3: Reverse contains match (plant name contains database name)
                    reverse_contains_matches = npp_info_df[
                        npp_info_df['Plant Name'].str.lower().apply(
                            lambda x: x in specific_plant_name.lower() if pd.notna(x) else False
                        )
                    ]

                    # Strategy 4: Fuzzy matching for common variations
                    def normalize_plant_name(name):
                        """Normalize plant name for fuzzy matching"""
                        if pd.isna(name):
                            return ""
                        normalized = str(name).lower()
                        # Remove common words and punctuation that might cause mismatches
                        normalized = normalized.replace(
                            'nuclear power plant', '').replace('power plant', '')
                        normalized = normalized.replace(
                            'generating station', '').replace('generation station', '')
                        normalized = normalized.replace(
                            'nuclear station', '').replace('station', '')
                        normalized = normalized.replace(
                            'nuclear', '').replace('plant', '')
                        # Remove extra spaces and punctuation
                        import re
                        normalized = re.sub(r'[^\w\s]', ' ', normalized)
                        # Remove extra whitespace
                        normalized = ' '.join(normalized.split())
                        return normalized.strip()

                    normalized_target = normalize_plant_name(
                        specific_plant_name)

                    fuzzy_matches = pd.DataFrame()
                    if normalized_target:
                        fuzzy_condition = npp_info_df['Plant Name'].apply(
                            lambda x: normalize_plant_name(x) in normalized_target or
                            normalized_target in normalize_plant_name(x)
                        )
                        fuzzy_matches = npp_info_df[fuzzy_condition]

                    # Prioritize matches: exact > contains > reverse contains > fuzzy
                    plant_matches = pd.DataFrame()
                    match_type = "none"

                    if not exact_matches.empty:
                        plant_matches = exact_matches
                        match_type = "exact"
                        logger.info(
                            f"Found exact match for '{specific_plant_name}'")
                    elif not contains_matches.empty:
                        plant_matches = contains_matches
                        match_type = "contains"
                        logger.info(
                            f"Found contains match for '{specific_plant_name}'")
                    elif not reverse_contains_matches.empty:
                        plant_matches = reverse_contains_matches
                        match_type = "reverse_contains"
                        logger.info(
                            f"Found reverse contains match for '{specific_plant_name}'")
                    elif not fuzzy_matches.empty:
                        plant_matches = fuzzy_matches
                        match_type = "fuzzy"
                        logger.info(
                            f"Found fuzzy match for '{specific_plant_name}' (normalized: '{normalized_target}')")

                    if not plant_matches.empty:
                        # CRITICAL FIX: Match by Generator ID if available
                        final_matches = plant_matches

                        # First filter by Generator ID if we have it
                        if specific_generator_id is not None:
                            try:
                                generator_id_int = int(specific_generator_id)
                                generator_matches = plant_matches[plant_matches['Generator ID']
                                                                  == generator_id_int]
                                if not generator_matches.empty:
                                    final_matches = generator_matches
                                    logger.info(
                                        f"🎯 Found Generator ID {generator_id_int} match for plant '{specific_plant_name}'")
                                else:
                                    logger.warning(
                                        f"⚠️  No Generator ID {generator_id_int} found for plant '{specific_plant_name}', using first available unit")
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"⚠️  Invalid Generator ID '{specific_generator_id}', using plant name match only")

                        # Then filter by ISO region
                        iso_matches = final_matches[final_matches['ISO']
                                                    == iso_target]
                        if not iso_matches.empty:
                            npp_info = iso_matches.iloc[0].to_dict()
                            plant_name = npp_info.get(
                                'Plant Name', specific_plant_name)
                            generator_id = npp_info.get(
                                'Generator ID', 'Unknown')
                            logger.info(
                                f"✅ Successfully matched plant: '{plant_name}' Unit {generator_id} in {iso_target} (method: {match_type})")
                        else:
                            npp_info = final_matches.iloc[0].to_dict()
                            plant_name = npp_info.get(
                                'Plant Name', specific_plant_name)
                            generator_id = npp_info.get(
                                'Generator ID', 'Unknown')
                            actual_iso = npp_info.get('ISO', 'Unknown')
                            logger.info(
                                f"⚠️  Matched plant by name and Generator ID but different ISO: '{plant_name}' Unit {generator_id} in {actual_iso} (method: {match_type})")
                    else:
                        logger.warning(
                            f"❌ No matches found for plant: '{specific_plant_name}' using any matching strategy")

                # Fallback: Filter by ISO and use first plant (original logic)
                if npp_info is None:
                    iso_plants = npp_info_df[npp_info_df['ISO'] == iso_target]
                    if not iso_plants.empty:
                        npp_info = iso_plants.iloc[0].to_dict()
                        plant_name = npp_info.get('Plant Name', 'Unknown')
                        generator_id = npp_info.get('Generator ID', 'Unknown')
                        logger.warning(
                            f"No specific plant match found, using first plant in {iso_target}: {plant_name} Unit {generator_id}")

                        # List all plants in this ISO for transparency
                        all_plants_in_iso = iso_plants['Plant Name'].tolist()
                        all_generator_ids_in_iso = iso_plants['Generator ID'].tolist(
                        )
                        remaining_years_in_iso = iso_plants['remaining'].tolist(
                        )
                        logger.info(f"Available plants in {iso_target}:")
                        for i, (p_name, p_gen_id, p_remaining) in enumerate(zip(all_plants_in_iso, all_generator_ids_in_iso, remaining_years_in_iso)):
                            prefix = "→ SELECTED: " if i == 0 else "           "
                            logger.info(
                                f"  {prefix}{p_name} Unit {p_gen_id} ({p_remaining} years remaining)")

                if npp_info:
                    # Always store plant identification info for later use (for baseline analysis)
                    params['matched_plant_name'] = plant_name
                    params['matched_generator_id'] = npp_info.get(
                        'Generator ID', 'Unknown')
                    params['plant_capacity_source'] = "NPPs info file"
                    params['plant_identification_method'] = "specific_name_and_generator_id" if (
                        specific_plant_name and specific_generator_id) else ("specific_name" if specific_plant_name else "iso_first")

                    # Only use NPPs info for remaining years if filename extraction failed
                    if remaining_years_from_filename is None:
                        remaining_years = npp_info.get('remaining', None)
                        if remaining_years is not None and pd.notna(remaining_years):
                            try:
                                actual_remaining_years = int(
                                    float(remaining_years))
                                logger.info(
                                    f"EXISTING PROJECT: Using remaining plant lifetime from NPPs info as fallback: {actual_remaining_years} years")
                                logger.info(
                                    f"  (Fallback from NPPs info file for {plant_name})")
                                logger.info(
                                    f"  (Previous value from CSV/config: {project_lifetime_years} years)")
                                project_lifetime_years = actual_remaining_years
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"Invalid remaining years value: {remaining_years}, using CSV/config value: {project_lifetime_years}")
                        else:
                            logger.warning(
                                f"No remaining years data available for {plant_name}, using default for existing projects: {config.CASE_CLASSIFICATION['existing_projects']['default_lifetime']} years")
                            project_lifetime_years = config.CASE_CLASSIFICATION[
                                'existing_projects']['default_lifetime']
                    else:
                        logger.info(
                            f"Using remaining years from filename ({remaining_years_from_filename} years), NPPs info used only for plant identification")
                else:
                    logger.warning(
                        f"No plants found for ISO {iso_target} in NPPs info file, using default for existing projects: {config.CASE_CLASSIFICATION['existing_projects']['default_lifetime']} years")
                    project_lifetime_years = config.CASE_CLASSIFICATION[
                        'existing_projects']['default_lifetime']
            else:
                logger.warning(
                    f"NPPs info file not found at {npps_path}, using default for existing projects: {config.CASE_CLASSIFICATION['existing_projects']['default_lifetime']} years")
                project_lifetime_years = config.CASE_CLASSIFICATION[
                    'existing_projects']['default_lifetime']
        except Exception as e:
            logger.error(
                f"Error loading NPPs info: {e}, using default for existing projects: {config.CASE_CLASSIFICATION['existing_projects']['default_lifetime']} years")
            project_lifetime_years = config.CASE_CLASSIFICATION['existing_projects']['default_lifetime']
    elif not is_existing_project:
        # For new construction projects (case4-5), use full lifecycle
        if case_type and "case5" in case_type.lower():
            project_lifetime_years = config.CASE_CLASSIFICATION['new_construction']['case5_lifetime']
            logger.info(
                f"NEW CONSTRUCTION (Case 5): Using 80-year lifecycle: {project_lifetime_years} years")
        else:
            project_lifetime_years = config.CASE_CLASSIFICATION['new_construction']['case4_lifetime']
            logger.info(
                f"NEW CONSTRUCTION (Case 4): Using 60-year lifecycle: {project_lifetime_years} years")
    else:
        logger.debug(
            f"No NPPs info path provided for existing project, using CSV/config value: {project_lifetime_years}")
    discount_rate = _get_param_value(
        params, "discount_rate_fraction", discount_rate, float, logger  # Default from config
    )
    construction_years = _get_param_value(
        params,
        "project_construction_years",
        construction_years,  # Default from config
        lambda x: int(float(x)),
        logger,
    )
    tax_rate = _get_param_value(
        params, "corporate_tax_rate_fraction", tax_rate, float, logger  # Default from config
    )

    # Update Battery O&M from loaded params if OM_COMPONENTS is structured as expected
    if "Fixed_OM_Battery" in om_components:
        om_components["Fixed_OM_Battery"]["base_cost_per_mw_year"] = _get_param_value(
            params, "BatteryFixedOM_USD_per_MW_year",
            om_components["Fixed_OM_Battery"].get("base_cost_per_mw_year", 0),
            float, logger
        )
        om_components["Fixed_OM_Battery"]["base_cost_per_mwh_year"] = _get_param_value(
            params, "BatteryFixedOM_USD_per_MWh_year",
            om_components["Fixed_OM_Battery"].get("base_cost_per_mwh_year", 0),
            float, logger
        )
    else:
        logger.warning(
            "OM_COMPONENTS structure missing 'Fixed_OM_Battery'. Cannot update from params.")

    # Update Nuclear Integrated System Configuration from loaded params
    nuclear_integrated_config["enabled"] = _get_param_value(
        params, "enable_nuclear_integrated_analysis",
        nuclear_integrated_config.get("enabled", False), bool, logger
    )
    nuclear_integrated_config["nuclear_plant_capacity_mw"] = _get_param_value(
        params, "nuclear_plant_capacity_MW",
        nuclear_integrated_config.get(
            "nuclear_plant_capacity_mw", 1000), float, logger
    )
    nuclear_integrated_config["project_lifetime_years"] = _get_param_value(
        params, "nuclear_project_lifetime_years",
        nuclear_integrated_config.get(
            "project_lifetime_years", 60), lambda x: int(float(x)), logger
    )
    nuclear_integrated_config["construction_years"] = _get_param_value(
        params, "nuclear_construction_years",
        nuclear_integrated_config.get(
            "construction_years", 8), lambda x: int(float(x)), logger
    )
    nuclear_integrated_config["enable_nuclear_capex"] = _get_param_value(
        params, "enable_nuclear_capex_costs",
        nuclear_integrated_config.get(
            "enable_nuclear_capex", False), bool, logger
    )
    nuclear_integrated_config["enable_nuclear_opex"] = _get_param_value(
        params, "enable_nuclear_opex_costs",
        nuclear_integrated_config.get(
            "enable_nuclear_opex", False), bool, logger
    )

    logger.debug(
        f"load_tea_sys_params finished. Project Lifetime: {project_lifetime_years}, Discount Rate: {discount_rate}"
    )
    logger.debug(f"Nuclear Integrated Config: {nuclear_integrated_config}")

    # Load tax incentive policies (with potential CSV overrides)
    tax_policies = load_tax_incentive_policies(params)
    logger.debug(f"Tax Incentive Policies: {tax_policies}")

    # Return both the raw params and the potentially updated config dictionaries/values
    return params, project_lifetime_years, discount_rate, construction_years, tax_rate, om_components, nuclear_integrated_config, tax_policies


def load_hourly_results(filepath: Path) -> pd.DataFrame | None:
    """Loads and validates hourly results from the optimization run."""
    logger.info(f"Loading hourly results from: {filepath}")
    if not filepath.exists():
        logger.error(f"Results file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)

        base_required_cols = [
            "Profit_Hourly_USD",
            "Revenue_Total_USD",
            "Cost_HourlyOpex_Total_USD",
            "mHydrogenProduced_kg_hr",
            "pElectrolyzer_MW",
            "pTurbine_MW",
            "EnergyPrice_LMP_USDperMWh",
        ]

        degradation_cols = [
            col for col in df.columns if "degradation" in col.lower()]
        if degradation_cols:
            logger.debug(f"Found degradation columns: {degradation_cols}")
        else:
            logger.debug("No degradation columns found in results file.")

        missing_base_cols = [
            col for col in base_required_cols if col not in df.columns]
        if missing_base_cols:
            logger.error(
                f"Missing essential base columns in results file: {missing_base_cols}"
            )
            return None

        capacity_cols_needed_for_capex = set()
        if DEFAULT_CAPEX_COMPONENTS:  # Check if CAPEX_COMPONENTS is not empty
            for comp_details in DEFAULT_CAPEX_COMPONENTS.values():
                cap_key = comp_details.get("applies_to_component_capacity_key")
                if cap_key:
                    capacity_cols_needed_for_capex.add(cap_key)
        else:
            logger.warning(
                "CAPEX_COMPONENTS is empty. Cannot determine capacity columns needed for CAPEX.")

        logger.debug(
            f"Capacity columns needed for CAPEX: {capacity_cols_needed_for_capex}"
        )

        capacity_related_cols = [
            col
            for col in df.columns
            if any(term in col.lower() for term in ["capacity", "mw", "mwh", "kg"])
        ]
        logger.debug(
            f"Available capacity-related columns in results file: {capacity_related_cols}"
        )

        if "Electrolyzer_Capacity_MW" in df.columns:
            unique_vals = df["Electrolyzer_Capacity_MW"].unique()
            logger.debug(
                f"Electrolyzer_Capacity_MW unique values: {unique_vals}")
        else:
            logger.warning(
                "Electrolyzer_Capacity_MW not found in results file!")
            potential_electrolyzer_cols = [
                col
                for col in df.columns
                if "electrolyzer" in col.lower() and "capacity" in col.lower()
            ]
            if potential_electrolyzer_cols:
                logger.debug(
                    f"Found potential electrolyzer capacity columns: {potential_electrolyzer_cols}"
                )
                df["Electrolyzer_Capacity_MW"] = df[potential_electrolyzer_cols[0]]
                logger.info(
                    f"Using {potential_electrolyzer_cols[0]} as fallback for Electrolyzer_Capacity_MW"
                )
            else:
                logger.warning(
                    "No alternative electrolyzer capacity columns found!")

        for cap_col_key in capacity_cols_needed_for_capex:
            if cap_col_key not in df.columns:
                logger.warning(
                    f"Capacity column '{cap_col_key}' (needed for CAPEX learning rate/scaling) "
                    f"is missing from results file '{filepath}'. "
                    f"Assuming 0 capacity for this component in this run."
                )
                df[cap_col_key] = 0.0

        all_required_cols = base_required_cols + \
            list(capacity_cols_needed_for_capex)
        all_required_cols = sorted(list(set(all_required_cols)))

        missing_final_cols = [
            col for col in all_required_cols if col not in df.columns]
        if missing_final_cols:
            logger.error(
                f"Still missing columns after attempting to add defaults: {missing_final_cols}"
            )
            return None

        return df
    except Exception as e:
        logger.error(
            f"Error loading or processing results file {filepath}: {e}",
            exc_info=True,
        )
        return None
