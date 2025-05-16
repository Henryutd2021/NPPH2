# runs/tea.py
"""
Technical Economic Analysis (TEA) script for the nuclear-hydrogen optimization framework.
This script performs comprehensive lifecycle analysis including:
- Capital and operational costs (with learning rate adjustments for CAPEX)
- Revenue streams from multiple sources
- Financial metrics (NPV, IRR, LCOH, etc.)
- Sensitivity analysis
- Visualization of results
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
from pathlib import Path
from datetime import datetime
import math
import traceback # For more detailed error printing

print("TEA_DEBUG: tea.py script started.") # DEBUG

# Add src directory to Python path
# This allows importing modules from the 'src' directory
SCRIPT_DIR_PATH = Path(__file__).resolve().parent
print(f"TEA_DEBUG: SCRIPT_DIR_PATH = {SCRIPT_DIR_PATH}") # DEBUG
SRC_PATH = SCRIPT_DIR_PATH.parent / "src"
print(f"TEA_DEBUG: SRC_PATH = {SRC_PATH}") # DEBUG
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))
    print(f"TEA_DEBUG: Added {SRC_PATH} to sys.path.") # DEBUG
else:
    print(f"TEA_DEBUG: {SRC_PATH} already in sys.path.") # DEBUG

# Framework imports
logger = None # Initialize logger to None
TARGET_ISO = "DEFAULT_ISO_FALLBACK" # Fallback
HOURS_IN_YEAR = 8760 # Fallback

try:
    print("TEA_DEBUG: Attempting to import logging_setup...") # DEBUG
    from logging_setup import logger
    print("TEA_DEBUG: Imported logging_setup successfully.") # DEBUG

    print("TEA_DEBUG: Attempting to import config...") # DEBUG
    from config import (
        TARGET_ISO,
        HOURS_IN_YEAR,
        ENABLE_BATTERY # To know if battery is generally enabled in the framework
    )
    print("TEA_DEBUG: Imported config successfully.") # DEBUG

    print("TEA_DEBUG: Attempting to import data_io...") # DEBUG
    from data_io import load_hourly_data
    print("TEA_DEBUG: Imported data_io successfully.") # DEBUG

    print("TEA_DEBUG: Attempting to import utils...") # DEBUG
    from utils import get_param
    print("TEA_DEBUG: Imported utils successfully.") # DEBUG

except ImportError as e_import:
    print(f"TEA_DEBUG: ImportError occurred: {e_import}") # DEBUG
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEA_FALLBACK - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    logger.error(f"Failed to import from optimization framework (ImportError): {e_import}. TEA script might not function correctly.")
    TARGET_ISO = "DEFAULT_ISO_IMPORT_ERROR"
    HOURS_IN_YEAR = 8760
    ENABLE_BATTERY = False # Fallback if config not loaded
except Exception as e_other: 
    print(f"TEA_DEBUG: A non-ImportError occurred during framework imports: {e_other}") # DEBUG
    traceback.print_exc() 
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEA_FALLBACK - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    logger.error(f"A non-ImportError occurred during framework imports: {e_other}. TEA script might not function correctly.")
    TARGET_ISO = "DEFAULT_ISO_OTHER_ERROR"
    HOURS_IN_YEAR = 8760
    ENABLE_BATTERY = False # Fallback

print(f"TEA_DEBUG: Framework imports section finished. TARGET_ISO set to: {TARGET_ISO}, ENABLE_BATTERY: {ENABLE_BATTERY}") # DEBUG


# --- TEA Configuration ---
BASE_OUTPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent / "TEA_results"
BASE_INPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent / "input"

# --- TEA Parameters ---
PROJECT_LIFETIME_YEARS = 25 
DISCOUNT_RATE = 0.08    
CONSTRUCTION_YEARS = 2  
TAX_RATE = 0.21         

# --- CAPEX Components (with learning rate structure) ---
CAPEX_COMPONENTS = {
    'Electrolyzer_System': {
        'total_base_cost_for_ref_size': 50_000_000,  
        'reference_total_capacity_mw': 50,         
        'applies_to_component_capacity_key': 'Electrolyzer_Capacity_MW', 
        'learning_rate_decimal': 0.15,             
        'payment_schedule_years': {-2: 0.5, -1: 0.5} 
    },
    'H2_Storage_System': {
        'total_base_cost_for_ref_size': 10_000_000,  
        'reference_total_capacity_mw': 10000, # Assuming kg
        'applies_to_component_capacity_key': 'H2_Storage_Capacity_kg', 
        'learning_rate_decimal': 0.10,             
        'payment_schedule_years': {-2: 0.5, -1: 0.5}
    },
    'Battery_System_Energy': { # Cost component for MWh capacity
        'total_base_cost_for_ref_size': 15_000_000, # e.g., for 100 MWh
        'reference_total_capacity_mw': 100, # Here 'mw' in key is a placeholder, unit is MWh
        'applies_to_component_capacity_key': 'Battery_Capacity_MWh',
        'learning_rate_decimal': 0.05, # Example learning rate for battery energy
        'payment_schedule_years': {-1: 1.0} # Example: paid in the last year of construction
    },
    'Battery_System_Power': { # Cost component for MW power
        'total_base_cost_for_ref_size': 5_000_000, # e.g., for 25 MW
        'reference_total_capacity_mw': 25, # Here unit is MW
        'applies_to_component_capacity_key': 'Battery_Power_MW',
        'learning_rate_decimal': 0.05, # Example learning rate for battery power
        'payment_schedule_years': {-1: 1.0}
    },
    'Grid_Integration': {
        'total_base_cost_for_ref_size': 5_000_000,
        'reference_total_capacity_mw': 0,
        'applies_to_component_capacity_key': None,
        'learning_rate_decimal': 0,
        'payment_schedule_years': {-1: 1.0}
    },
    'NPP_Modifications': {
        'total_base_cost_for_ref_size': 2_000_000,
        'reference_total_capacity_mw': 0,
        'applies_to_component_capacity_key': None,
        'learning_rate_decimal': 0,
        'payment_schedule_years': {-2: 1.0}
    }
}

# --- O&M Components ---
OM_COMPONENTS = {
    'Fixed_OM_General': {'base_cost': 1_000_000, 'size_dependent': False, 'inflation_rate': 0.02},
    # Battery Fixed O&M can be a percentage of its CAPEX or a fixed $/kW/yr or $/kWh/yr
    # For simplicity, we'll add a placeholder that can be populated from sys_data
    'Fixed_OM_Battery': {'base_cost_per_mw_year': 0, 'base_cost_per_mwh_year': 0, 'inflation_rate': 0.02} # Values to be loaded
}

# --- Replacement Schedule ---
REPLACEMENT_SCHEDULE = {
    'Electrolyzer_Stack': {'cost': 15_000_000, 'years': [10, 20], 'size_dependent': True},
    'H2_Storage_Components': {'cost': 5_000_000, 'years': [15], 'size_dependent': True},
    'Battery_Augmentation_Replacement': {'cost_percent_initial_capex': 0.60, 'years': [10], 'size_dependent': True} # Example: 60% of initial battery capex at year 10
}

print("TEA_DEBUG: Global configurations and constants defined.") # DEBUG

def load_tea_sys_params(iso_target: str, input_base_dir: Path) -> dict:
    """Loads TEA-relevant system parameters."""
    print(f"TEA_DEBUG: load_tea_sys_params called for ISO: {iso_target}") # DEBUG
    params = {}
    try:
        sys_data_file_path = input_base_dir / "hourly_data" / "sys_data_advanced.csv"
        if not sys_data_file_path.exists():
             sys_data_file_path = input_base_dir / "sys_data_advanced.csv" 
        print(f"TEA_DEBUG: Attempting to load sys_data from: {sys_data_file_path}") # DEBUG

        if sys_data_file_path.exists():
            df_system = pd.read_csv(sys_data_file_path, index_col=0)
            param_keys = [
                'hydrogen_subsidy_value_usd_per_kg',
                'hydrogen_subsidy_duration_years',
                'user_specified_electrolyzer_capacity_MW',
                'user_specified_h2_storage_capacity_kg',
                'user_specified_battery_capacity_MWh', # Added for battery
                'user_specified_battery_power_MW',     # Added for battery
                'plant_lifetime_years',
                'baseline_nuclear_annual_revenue_USD',
                'enable_incremental_analysis',
                'discount_rate_fraction',
                'project_construction_years',
                'corporate_tax_rate_fraction',
                'BatteryFixedOM_USD_per_MW_year', # Added for battery O&M
                'BatteryFixedOM_USD_per_MWh_year' # Added for battery O&M
            ]
            for key in param_keys:
                if key in df_system.index:
                    value_series = df_system.loc[key, 'Value']
                    params[key] = value_series.iloc[0] if isinstance(value_series, pd.Series) else value_series
                else:
                    params[key] = None 
            logger.info(f"Successfully loaded TEA relevant params from {sys_data_file_path}")
        else:
            logger.warning(f"sys_data_advanced.csv not found at {sys_data_file_path}. TEA will use defaults for some parameters.")

    except Exception as e:
        logger.error(f"Error loading TEA system data from {sys_data_file_path}: {e}")
        print(f"TEA_DEBUG: Error in load_tea_sys_params: {e}") # DEBUG

    global PROJECT_LIFETIME_YEARS, DISCOUNT_RATE, CONSTRUCTION_YEARS, TAX_RATE, OM_COMPONENTS
    
    def _get_param_value(params_dict, key, default_val, type_converter, param_logger):
        val = params_dict.get(key) 
        if val is None or pd.isna(val):
            param_logger.info(f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_val}")
            return default_val
        try:
            return type_converter(val)
        except (ValueError, TypeError):
            param_logger.warning(f"Invalid value '{val}' for '{key}' in sys_data. Using default: {default_val}")
            return default_val

    PROJECT_LIFETIME_YEARS = _get_param_value(params, 'plant_lifetime_years', PROJECT_LIFETIME_YEARS, lambda x: int(float(x)), logger)
    DISCOUNT_RATE = _get_param_value(params, 'discount_rate_fraction', DISCOUNT_RATE, float, logger)
    CONSTRUCTION_YEARS = _get_param_value(params, 'project_construction_years', CONSTRUCTION_YEARS, lambda x: int(float(x)), logger)
    TAX_RATE = _get_param_value(params, 'corporate_tax_rate_fraction', TAX_RATE, float, logger)
    
    # Update Battery O&M from loaded params
    OM_COMPONENTS['Fixed_OM_Battery']['base_cost_per_mw_year'] = _get_param_value(params, 'BatteryFixedOM_USD_per_MW_year', 0, float, logger)
    OM_COMPONENTS['Fixed_OM_Battery']['base_cost_per_mwh_year'] = _get_param_value(params, 'BatteryFixedOM_USD_per_MWh_year', 0, float, logger)

    print(f"TEA_DEBUG: load_tea_sys_params finished. Project Lifetime: {PROJECT_LIFETIME_YEARS}, Discount Rate: {DISCOUNT_RATE}") # DEBUG
    return params


def load_hourly_results(filepath: Path) -> pd.DataFrame | None:
    """Loads and validates hourly results from the optimization run."""
    logger.info(f"Loading hourly results from: {filepath}")
    if not filepath.exists():
        logger.error(f"Results file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        
        base_required_cols = [
            'Profit_Hourly_USD', 'Revenue_Total_USD', 'Cost_HourlyOpex_Total_USD',
            'mHydrogenProduced_kg_hr', 'pElectrolyzer_MW', 'pTurbine_MW',
            'EnergyPrice_LMP_USDperMWh'
        ]
        
        missing_base_cols = [col for col in base_required_cols if col not in df.columns]
        if missing_base_cols:
            logger.error(f"Missing essential base columns in results file: {missing_base_cols}")
            return None

        capacity_cols_needed_for_capex = set()
        for comp_details in CAPEX_COMPONENTS.values():
            cap_key = comp_details.get('applies_to_component_capacity_key')
            if cap_key:
                capacity_cols_needed_for_capex.add(cap_key)
        
        for cap_col_key in capacity_cols_needed_for_capex:
            if cap_col_key not in df.columns:
                logger.warning(f"Capacity column '{cap_col_key}' (needed for CAPEX learning rate/scaling) "
                               f"is missing from results file '{filepath}'. "
                               f"Assuming 0 capacity for this component in this run.")
                df[cap_col_key] = 0.0 

        all_required_cols = base_required_cols + list(capacity_cols_needed_for_capex)
        all_required_cols = sorted(list(set(all_required_cols))) 

        missing_final_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_final_cols: 
            logger.error(f"Still missing columns after attempting to add defaults: {missing_final_cols}")
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error loading or processing results file {filepath}: {e}", exc_info=True)
        return None


def calculate_annual_metrics(df: pd.DataFrame, tea_sys_params: dict) -> dict | None:
    """Calculates comprehensive annual metrics from hourly results."""
    if df is None: return None
    metrics = {}
    try:
        num_hours = len(df)
        if num_hours == 0:
            logger.error("Hourly results DataFrame is empty.")
            return None
        annualization_factor = HOURS_IN_YEAR / num_hours if num_hours > 0 and HOURS_IN_YEAR > 0 else 1.0

        metrics['Annual_Profit'] = df['Profit_Hourly_USD'].sum()
        metrics['Annual_Revenue'] = df['Revenue_Total_USD'].sum()
        metrics['Annual_Opex_Cost_from_Opt'] = df['Cost_HourlyOpex_Total_USD'].sum()
        metrics['Energy_Revenue'] = df.get('Revenue_Energy_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['AS_Revenue'] = df.get('Revenue_Ancillary_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['H2_Sales_Revenue'] = df.get('Revenue_Hydrogen_Sales_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['H2_Subsidy_Revenue'] = df.get('Revenue_Hydrogen_Subsidy_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['H2_Total_Revenue'] = metrics['H2_Sales_Revenue'] + metrics['H2_Subsidy_Revenue']
        metrics['VOM_Turbine_Cost'] = df.get('Cost_VOM_Turbine_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['VOM_Electrolyzer_Cost'] = df.get('Cost_VOM_Electrolyzer_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['VOM_Battery_Cost'] = df.get('Cost_VOM_Battery_USD', pd.Series(0.0, dtype='float64')).sum() # From optimization results
        metrics['Startup_Cost'] = df.get('Cost_Startup_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['Water_Cost'] = df.get('Cost_Water_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['Ramping_Cost'] = df.get('Cost_Ramping_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['H2_Storage_Cycle_Cost'] = df.get('Cost_Storage_Cycle_USD', pd.Series(0.0, dtype='float64')).sum()
        metrics['H2_Production_kg_annual'] = df['mHydrogenProduced_kg_hr'].sum() * annualization_factor
        
        metrics['Electrolyzer_Capacity_MW'] = df['Electrolyzer_Capacity_MW'].iloc[0] if 'Electrolyzer_Capacity_MW' in df and not df['Electrolyzer_Capacity_MW'].empty else 0
        if metrics['Electrolyzer_Capacity_MW'] > 1e-6:
            metrics['Electrolyzer_CF_percent'] = (df['pElectrolyzer_MW'].mean() / metrics['Electrolyzer_Capacity_MW']) * 100
        else:
            metrics['Electrolyzer_CF_percent'] = 0
        
        # H2 Storage Capacity
        if 'H2_Storage_Capacity_kg' in df.columns: # This column is added by load_hourly_results if needed
            metrics['H2_Storage_Capacity_kg'] = df['H2_Storage_Capacity_kg'].iloc[0] if not df['H2_Storage_Capacity_kg'].empty else 0
        else: # Should not happen if load_hourly_results works as intended
            metrics['H2_Storage_Capacity_kg'] = 0
            logger.warning("H2_Storage_Capacity_kg column unexpectedly missing in calculate_annual_metrics.")

        # Battery Capacity and Power (from results DataFrame, ensured by load_hourly_results)
        metrics['Battery_Capacity_MWh'] = df['Battery_Capacity_MWh'].iloc[0] if 'Battery_Capacity_MWh' in df and not df['Battery_Capacity_MWh'].empty else 0
        metrics['Battery_Power_MW'] = df['Battery_Power_MW'].iloc[0] if 'Battery_Power_MW' in df and not df['Battery_Power_MW'].empty else 0
        
        if metrics['Battery_Power_MW'] > 1e-6 and 'BatteryCharge_MW' in df and 'BatteryDischarge_MW' in df:
            avg_batt_usage = (df['BatteryCharge_MW'].mean() + df['BatteryDischarge_MW'].mean()) / 2
            metrics['Battery_CF_percent'] = (avg_batt_usage / metrics['Battery_Power_MW']) * 100
        else:
            metrics['Battery_CF_percent'] = 0


        metrics['Turbine_Capacity_MW'] = df.get('Turbine_Capacity_MW', pd.Series(0.0, dtype='float64')).iloc[0] if 'Turbine_Capacity_MW' in df and not df['Turbine_Capacity_MW'].empty else 0
        if metrics.get('Turbine_Capacity_MW', 0) > 1e-6:
            metrics['Turbine_CF_percent'] = (df['pTurbine_MW'].mean() / metrics['Turbine_Capacity_MW']) * 100
        else:
            metrics['Turbine_CF_percent'] = 0
        metrics['Annual_Electrolyzer_MWh'] = df['pElectrolyzer_MW'].sum() * annualization_factor if 'pElectrolyzer_MW' in df else 0
        if 'EnergyPrice_LMP_USDperMWh' in df.columns:
            metrics['Avg_Electricity_Price_USD_per_MWh'] = df['EnergyPrice_LMP_USDperMWh'].mean()
            if 'pElectrolyzer_MW' in df.columns and df['pElectrolyzer_MW'].sum() > 0:
                weighted_price = (df['EnergyPrice_LMP_USDperMWh'] * df['pElectrolyzer_MW']).sum() / df['pElectrolyzer_MW'].sum()
                metrics['Weighted_Avg_Electricity_Price_USD_per_MWh'] = weighted_price
            else:
                metrics['Weighted_Avg_Electricity_Price_USD_per_MWh'] = metrics['Avg_Electricity_Price_USD_per_MWh']
        else:
            metrics['Avg_Electricity_Price_USD_per_MWh'] = 40.0
            metrics['Weighted_Avg_Electricity_Price_USD_per_MWh'] = 40.0
    except KeyError as e:
        logger.error(f"Missing column in hourly results for annual metrics calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calculating annual metrics: {e}", exc_info=True)
        return None
    return metrics


def calculate_cash_flows(
    annual_metrics: dict, project_lifetime: int, construction_period: int,
    h2_subsidy_value: float, h2_subsidy_duration: int, capex_details: dict,
    om_details: dict, replacement_details: dict, optimized_capacities: dict
) -> np.ndarray:
    logger.info(f"Calculating cash flows for {project_lifetime} years. Construction period: {construction_period} years.")
    cash_flows_array = np.zeros(project_lifetime)
    total_capex_sum_after_learning = 0
    
    # Store initial CAPEX for battery for replacement calculation
    initial_battery_capex_energy = 0
    initial_battery_capex_power = 0

    for component_name, comp_data in capex_details.items():
        base_cost_for_ref_size = comp_data.get('total_base_cost_for_ref_size', 0)
        ref_capacity = comp_data.get('reference_total_capacity_mw', 0)
        lr_decimal = comp_data.get('learning_rate_decimal', 0)
        capacity_key = comp_data.get('applies_to_component_capacity_key')
        payment_schedule = comp_data.get('payment_schedule_years', {})
        actual_optimized_capacity = optimized_capacities.get(capacity_key, ref_capacity if capacity_key else 0)
        
        adjusted_total_component_cost = 0.0 
        if capacity_key and actual_optimized_capacity == 0 and ref_capacity > 0 :
             logger.info(f"Component '{component_name}' was sized to 0 (e.g., MW or kg). Its CAPEX will be 0.")
             adjusted_total_component_cost = 0.0
        elif lr_decimal > 0 and ref_capacity > 0 and actual_optimized_capacity > 0 and capacity_key:
            progress_ratio = 1 - lr_decimal
            b = math.log(progress_ratio) / math.log(2) if 0 < progress_ratio < 1 else 0
            scale_factor = actual_optimized_capacity / ref_capacity
            adjusted_total_component_cost = base_cost_for_ref_size * (scale_factor ** b)
            logger.info(f"Component '{component_name}': Ref Cost=${base_cost_for_ref_size:,.0f} (Ref Cap:{ref_capacity}), Optimized Cap:{actual_optimized_capacity}, LR:{lr_decimal*100}%, Adjusted Total Cost=${adjusted_total_component_cost:,.0f}")
        elif actual_optimized_capacity > 0 and ref_capacity > 0 and capacity_key: 
            scale_factor = actual_optimized_capacity / ref_capacity
            adjusted_total_component_cost = base_cost_for_ref_size * scale_factor
            logger.info(f"Component '{component_name}': Ref Cost=${base_cost_for_ref_size:,.0f} (Ref Cap:{ref_capacity}), Optimized Cap:{actual_optimized_capacity}, No LR, Linearly Scaled Total Cost=${adjusted_total_component_cost:,.0f}")
        elif not capacity_key : 
            adjusted_total_component_cost = base_cost_for_ref_size
            logger.info(f"Component '{component_name}': Fixed Cost=${adjusted_total_component_cost:,.0f} (does not scale).")
        else: 
            adjusted_total_component_cost = 0.0 
            if base_cost_for_ref_size > 0 and capacity_key: 
                 logger.info(f"Component '{component_name}' has 0 optimized capacity. Its CAPEX is 0.")
        
        if component_name == 'Battery_System_Energy':
            initial_battery_capex_energy = adjusted_total_component_cost
        if component_name == 'Battery_System_Power':
            initial_battery_capex_power = adjusted_total_component_cost

        total_capex_sum_after_learning += adjusted_total_component_cost
        for constr_year_offset, share in payment_schedule.items():
            project_year_index = construction_period + constr_year_offset
            if 0 <= project_year_index < construction_period:
                cash_flows_array[project_year_index] -= adjusted_total_component_cost * share
            else:
                logger.warning(f"Payment schedule year {constr_year_offset} for component '{component_name}' is outside construction period.")
    logger.info(f"Total CAPEX after learning rate/scaling adjustments: ${total_capex_sum_after_learning:,.2f}")
    initial_total_battery_capex = initial_battery_capex_energy + initial_battery_capex_power


    base_annual_profit_from_opt = annual_metrics.get('Annual_Revenue', 0) - annual_metrics.get('Annual_Opex_Cost_from_Opt', 0)
    for op_year_idx in range(project_lifetime - construction_period):
        current_project_year_idx = op_year_idx + construction_period
        operational_year_num = op_year_idx + 1
        current_year_profit_before_fixed_om_repl_tax = base_annual_profit_from_opt
        if operational_year_num > h2_subsidy_duration:
            current_year_profit_before_fixed_om_repl_tax -= annual_metrics.get('H2_Subsidy_Revenue', 0)
        
        # General Fixed O&M
        fixed_om_general_cost = om_details.get('Fixed_OM_General', {}).get('base_cost', 0) * \
                                ((1 + om_details.get('Fixed_OM_General', {}).get('inflation_rate', 0)) ** op_year_idx)
        current_year_profit_before_fixed_om_repl_tax -= fixed_om_general_cost

        # Battery Fixed O&M (if battery is enabled and capacity > 0)
        if ENABLE_BATTERY and optimized_capacities.get('Battery_Capacity_MWh', 0) > 0:
            batt_fixed_om_per_mw = om_details.get('Fixed_OM_Battery', {}).get('base_cost_per_mw_year', 0)
            batt_fixed_om_per_mwh = om_details.get('Fixed_OM_Battery', {}).get('base_cost_per_mwh_year', 0)
            batt_inflation = om_details.get('Fixed_OM_Battery', {}).get('inflation_rate', 0)
            
            batt_power_mw = optimized_capacities.get('Battery_Power_MW', 0)
            batt_capacity_mwh = optimized_capacities.get('Battery_Capacity_MWh', 0)

            battery_fixed_om_cost_this_year = (batt_power_mw * batt_fixed_om_per_mw + batt_capacity_mwh * batt_fixed_om_per_mwh) * \
                                              ((1 + batt_inflation) ** op_year_idx)
            current_year_profit_before_fixed_om_repl_tax -= battery_fixed_om_cost_this_year


        replacement_cost_this_year = 0
        for rep_comp_name, comp_data in replacement_details.items():
            if operational_year_num in comp_data.get('years', []):
                cost_val = comp_data.get('cost', 0)
                if rep_comp_name == 'Battery_Augmentation_Replacement' and comp_data.get('cost_percent_initial_capex', 0) > 0:
                    cost_val = initial_total_battery_capex * comp_data['cost_percent_initial_capex']
                replacement_cost_this_year += cost_val
        current_year_profit_before_fixed_om_repl_tax -= replacement_cost_this_year

        taxable_income = current_year_profit_before_fixed_om_repl_tax
        tax_amount = taxable_income * TAX_RATE if taxable_income > 0 else 0
        cash_flows_array[current_project_year_idx] = taxable_income - tax_amount
    return cash_flows_array


def calculate_financial_metrics(cash_flows_input: np.ndarray, discount_rt: float, annual_h2_prod_kg: float, project_lt: int, construction_p: int) -> dict:
    metrics_results = {}
    cf_array = np.array(cash_flows_input, dtype=float)
    try: metrics_results['NPV_USD'] = npf.npv(discount_rt, cf_array)
    except Exception: metrics_results['NPV_USD'] = np.nan
    try:
        if any(cf > 0 for cf in cf_array) and any(cf < 0 for cf in cf_array): metrics_results['IRR_percent'] = npf.irr(cf_array) * 100
        else: metrics_results['IRR_percent'] = np.nan
    except Exception: metrics_results['IRR_percent'] = np.nan
    cumulative_cash_flow = np.cumsum(cf_array)
    positive_indices = np.where(cumulative_cash_flow >= 0)[0]
    if positive_indices.size > 0:
        first_positive_idx = positive_indices[0]
        if first_positive_idx == 0 and cf_array[0] >=0 : metrics_results['Payback_Period_Years'] = 0
        elif first_positive_idx > 0 and cumulative_cash_flow[first_positive_idx -1] < 0:
            metrics_results['Payback_Period_Years'] = (first_positive_idx - 1) + abs(cumulative_cash_flow[first_positive_idx - 1]) / (cumulative_cash_flow[first_positive_idx] - cumulative_cash_flow[first_positive_idx - 1]) - construction_p +1
        else: metrics_results['Payback_Period_Years'] = first_positive_idx - construction_p + 1
    else: metrics_results['Payback_Period_Years'] = np.nan
    if annual_h2_prod_kg > 0:
        pv_total_costs = sum(abs(cf) / ((1 + discount_rt) ** i) for i, cf in enumerate(cf_array) if cf < 0)
        pv_total_h2_production = sum(annual_h2_prod_kg / ((1 + discount_rt) ** (op_idx + construction_p)) for op_idx in range(project_lt - construction_p))
        metrics_results['LCOH_USD_per_kg'] = pv_total_costs / pv_total_h2_production if pv_total_h2_production > 0 else np.nan
    else: metrics_results['LCOH_USD_per_kg'] = np.nan
    return metrics_results


def calculate_incremental_metrics(
    optimized_cash_flows: np.ndarray, baseline_annual_revenue: float, project_lifetime: int,
    construction_period: int, discount_rt: float, tax_rt: float, annual_metrics_optimized: dict,
    capex_components_incremental: dict, om_components_incremental: dict,
    replacement_schedule_incremental: dict, h2_subsidy_val: float, h2_subsidy_yrs: int,
    optimized_capacities_inc: dict
) -> dict:
    logger.info("Calculating incremental financial metrics.")
    inc_metrics = {}
    baseline_cash_flows = np.zeros(project_lifetime)
    annual_baseline_profit_before_tax = baseline_annual_revenue * (1 - 0.3) 
    for i in range(construction_period, project_lifetime):
        baseline_cash_flows[i] = annual_baseline_profit_before_tax * (1 - tax_rt if annual_baseline_profit_before_tax > 0 else 1)
    
    pure_incremental_cf = np.zeros(project_lifetime)
    total_incremental_capex_sum_after_learning = 0
    initial_inc_battery_capex_energy = 0 # For incremental battery replacement
    initial_inc_battery_capex_power = 0  # For incremental battery replacement

    for comp_name, comp_data in capex_components_incremental.items():
        base_cost = comp_data.get('total_base_cost_for_ref_size',0)
        ref_cap = comp_data.get('reference_total_capacity_mw',0)
        lr = comp_data.get('learning_rate_decimal',0)
        cap_key = comp_data.get('applies_to_component_capacity_key')
        pay_sched = comp_data.get('payment_schedule_years',{})

        actual_opt_cap_inc = optimized_capacities_inc.get(cap_key, ref_cap if cap_key else 0)
        adj_cost_inc = 0.0 
        if cap_key and actual_opt_cap_inc == 0 and ref_cap > 0:
            adj_cost_inc = 0.0
        elif lr > 0 and ref_cap > 0 and actual_opt_cap_inc > 0 and cap_key:
            pr = 1 - lr; b = math.log(pr) / math.log(2) if 0 < pr < 1 else 0
            adj_cost_inc = base_cost * ((actual_opt_cap_inc / ref_cap) ** b)
        elif actual_opt_cap_inc > 0 and ref_cap > 0 and cap_key:
            adj_cost_inc = base_cost * (actual_opt_cap_inc / ref_cap)
        elif not cap_key : 
            adj_cost_inc = base_cost
        
        if comp_name == 'Battery_System_Energy': initial_inc_battery_capex_energy = adj_cost_inc
        if comp_name == 'Battery_System_Power': initial_inc_battery_capex_power = adj_cost_inc

        total_incremental_capex_sum_after_learning += adj_cost_inc
        for constr_yr_offset, share in pay_sched.items():
            if 0 <= construction_period + constr_yr_offset < construction_period:
                pure_incremental_cf[construction_period + constr_yr_offset] -= adj_cost_inc * share
    inc_metrics['Total_Incremental_CAPEX_Learned_USD'] = total_incremental_capex_sum_after_learning
    initial_total_inc_battery_capex = initial_inc_battery_capex_energy + initial_inc_battery_capex_power


    h2_rev_annual = annual_metrics_optimized.get('H2_Total_Revenue', 0)
    as_rev_annual = annual_metrics_optimized.get('AS_Revenue', 0)
    vom_annual_inc = sum(annual_metrics_optimized.get(k,0) for k in ['VOM_Electrolyzer_Cost', 'VOM_Battery_Cost', 'Water_Cost', 'Startup_Cost', 'Ramping_Cost', 'H2_Storage_Cycle_Cost'])
    opp_cost_elec_annual = annual_metrics_optimized.get('Annual_Electrolyzer_MWh', 0) * annual_metrics_optimized.get('Avg_Electricity_Price_USD_per_MWh', 40.0)
    # Add battery charging cost to opportunity cost if battery is part of incremental system
    if ENABLE_BATTERY and optimized_capacities_inc.get('Battery_Capacity_MWh', 0) > 0:
        # Assuming VOM_Battery_Cost from optimization is for degradation/cycling, not electricity.
        # Electricity for charging battery should be an opportunity cost if not bought from grid.
        # This needs careful definition: is battery charging from NPP or grid?
        # If from NPP, it's an opportunity cost. If from grid, it's a direct cost (already in VOM_Battery_Cost if so).
        # For now, assuming VOM_Battery_Cost covers electricity if bought from grid.
        # If charged by NPP, its energy use should be an opportunity cost.
        # This part might need refinement based on how battery charging is modeled in optimization.
        pass


    for op_idx in range(project_lifetime - construction_period):
        proj_yr_idx = op_idx + construction_period; op_yr_num = op_idx + 1
        cur_h2_rev = h2_rev_annual - (annual_metrics_optimized.get('H2_Subsidy_Revenue',0) if op_yr_num > h2_subsidy_yrs else 0)
        rev_inc = cur_h2_rev + as_rev_annual
        costs_inc = vom_annual_inc + opp_cost_elec_annual
        
        # Incremental Fixed O&M (General + Battery specific)
        fixed_om_inc_general_base = om_components_incremental.get('Fixed_OM_General', {}).get('base_cost', 0) # If there's a general incremental fixed OM
        fixed_om_inc_general_inflation = om_components_incremental.get('Fixed_OM_General', {}).get('inflation_rate', 0)
        costs_inc += fixed_om_inc_general_base * ((1 + fixed_om_inc_general_inflation) ** op_idx)

        if ENABLE_BATTERY and optimized_capacities_inc.get('Battery_Capacity_MWh', 0) > 0:
            batt_fixed_om_per_mw_inc = om_components_incremental.get('Fixed_OM_Battery', {}).get('base_cost_per_mw_year', 0)
            batt_fixed_om_per_mwh_inc = om_components_incremental.get('Fixed_OM_Battery', {}).get('base_cost_per_mwh_year', 0)
            batt_inflation_inc = om_components_incremental.get('Fixed_OM_Battery', {}).get('inflation_rate', 0)
            batt_power_inc = optimized_capacities_inc.get('Battery_Power_MW',0)
            batt_capacity_inc = optimized_capacities_inc.get('Battery_Capacity_MWh',0)
            costs_inc += (batt_power_inc * batt_fixed_om_per_mw_inc + batt_capacity_inc * batt_fixed_om_per_mwh_inc) * \
                         ((1 + batt_inflation_inc) ** op_idx)

        # Incremental Replacements
        for rep_comp_name_inc, rep_data_inc in replacement_schedule_incremental.items():
            if op_yr_num in rep_data_inc.get('years', []):
                cost_val_inc = rep_data_inc.get('cost', 0)
                if rep_comp_name_inc == 'Battery_Augmentation_Replacement' and rep_data_inc.get('cost_percent_initial_capex', 0) > 0:
                    cost_val_inc = initial_total_inc_battery_capex * rep_data_inc['cost_percent_initial_capex']
                costs_inc += cost_val_inc
                
        profit_inc_pre_tax = rev_inc - costs_inc
        tax_inc = profit_inc_pre_tax * tax_rt if profit_inc_pre_tax > 0 else 0
        pure_incremental_cf[proj_yr_idx] += profit_inc_pre_tax - tax_inc
    
    inc_metrics['NPV_USD'] = npf.npv(discount_rt, pure_incremental_cf)
    try: inc_metrics['IRR_percent'] = npf.irr(pure_incremental_cf) * 100 if any(cf!=0 for cf in pure_incremental_cf) else np.nan
    except: inc_metrics['IRR_percent'] = np.nan
    
    cum_pure_inc_cf = np.cumsum(pure_incremental_cf); pos_idx_pure = np.where(cum_pure_inc_cf >=0)[0]
    if pos_idx_pure.size > 0:
        first_pos = pos_idx_pure[0]
        if first_pos == 0 and pure_incremental_cf[0] >=0 : inc_metrics['Payback_Period_Years'] = 0
        elif first_pos > 0 and cum_pure_inc_cf[first_pos-1] < 0:
            inc_metrics['Payback_Period_Years'] = (first_pos - 1) + abs(cum_pure_inc_cf[first_pos-1]) / (cum_pure_inc_cf[first_pos] - cum_pure_inc_cf[first_pos-1]) - construction_period + 1
        else: inc_metrics['Payback_Period_Years'] = first_pos - construction_period + 1
    else: inc_metrics['Payback_Period_Years'] = np.nan

    h2_prod_annual = annual_metrics_optimized.get('H2_Production_kg_annual',0)
    if h2_prod_annual > 0: # LCOH for incremental H2 project
        # Costs for LCOH are the negative cash flows of the *pure_incremental_cf*
        pv_inc_costs_for_lcoh = sum(abs(cf) / ((1+discount_rt)**i) for i, cf in enumerate(pure_incremental_cf) if cf < 0)
        pv_h2_prod_inc = sum(h2_prod_annual / ((1+discount_rt)**(op_idx+construction_period)) for op_idx in range(project_lifetime-construction_period))
        inc_metrics['LCOH_USD_per_kg'] = pv_inc_costs_for_lcoh / pv_h2_prod_inc if pv_h2_prod_inc > 0 else np.nan
    else: inc_metrics['LCOH_USD_per_kg'] = np.nan
    
    inc_metrics['pure_incremental_cash_flows'] = pure_incremental_cf
    inc_metrics['traditional_incremental_cash_flows'] = optimized_cash_flows - baseline_cash_flows
    inc_metrics['Annual_Electricity_Opportunity_Cost_USD'] = opp_cost_elec_annual
    return inc_metrics


def plot_results(annual_metrics_data: dict, financial_metrics_data: dict, cash_flows_data: np.ndarray, plot_dir: Path, construction_p: int, incremental_metrics_data: dict | None = None):
    os.makedirs(plot_dir, exist_ok=True); plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'figure.figsize': (10,6), 'font.size':10, 'axes.labelsize':11, 'axes.titlesize':13})
    years_axis = np.arange(1, len(cash_flows_data) + 1); cumulative_cf_plot = np.cumsum(cash_flows_data)
    fig, ax1 = plt.subplots(); bars = ax1.bar(years_axis, cash_flows_data, color='cornflowerblue', alpha=0.7, label='Annual Cash Flow')
    for i, val in enumerate(cash_flows_data):
        if val < 0: bars[i].set_color('salmon')
    ax2 = ax1.twinx(); ax2.plot(years_axis, cumulative_cf_plot, 'forestgreen', marker='o', markersize=4, label='Cumulative Cash Flow')
    ax1.axhline(0, color='grey', lw=0.8); ax1.set_xlabel('Project Year'); ax1.set_ylabel('Annual Cash Flow (USD)'); ax2.set_ylabel('Cumulative Cash Flow (USD)')
    if construction_p > 0: ax1.axvline(construction_p + 0.5, color='black', linestyle='--', lw=1, label='Operations Start')
    ax1.set_title('Project Cash Flow Profile', fontweight='bold'); handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best'); plt.tight_layout(); plt.savefig(plot_dir / 'cash_flow_profile.png', dpi=300); plt.close(fig)

    if incremental_metrics_data and 'pure_incremental_cash_flows' in incremental_metrics_data:
        inc_cf_data = incremental_metrics_data['pure_incremental_cash_flows']; fig_inc, ax1_inc = plt.subplots()
        inc_cumulative_cf_plot = np.cumsum(inc_cf_data); inc_bars = ax1_inc.bar(years_axis, inc_cf_data, color='mediumpurple', alpha=0.7, label='Incremental Annual CF')
        for i, val in enumerate(inc_cf_data):
            if val < 0: inc_bars[i].set_color('lightcoral')
        ax2_inc = ax1_inc.twinx(); ax2_inc.plot(years_axis, inc_cumulative_cf_plot, 'darkorange', marker='s', markersize=4, label='Cumulative Incremental CF')
        ax1_inc.axhline(0, color='grey', lw=0.8); ax1_inc.set_xlabel('Project Year'); ax1_inc.set_ylabel('Incremental Annual CF (USD)'); ax2_inc.set_ylabel('Cumulative Incremental CF (USD)')
        if construction_p > 0: ax1_inc.axvline(construction_p + 0.5, color='black', linestyle='--', lw=1, label='Operations Start')
        ax1_inc.set_title('Pure Incremental Cash Flow Profile (H2/Battery System)', fontweight='bold'); inc_handles1, inc_labels1 = ax1_inc.get_legend_handles_labels(); inc_handles2, inc_labels2 = ax2_inc.get_legend_handles_labels()
        ax1_inc.legend(inc_handles1 + inc_handles2, inc_labels1 + inc_labels2, loc='best')
        if 'Annual_Electricity_Opportunity_Cost_USD' in incremental_metrics_data:
            ax1_inc.text(0.02, 0.02, f"Annual Electricity Opportunity Cost: ${incremental_metrics_data['Annual_Electricity_Opportunity_Cost_USD']:,.0f}", transform=ax1_inc.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        plt.tight_layout(); plt.savefig(plot_dir / 'incremental_cash_flow_profile.png', dpi=300); plt.close(fig_inc)

    rev_sources = {k: annual_metrics_data.get(k,0) for k in ['Energy_Revenue', 'AS_Revenue', 'H2_Sales_Revenue', 'H2_Subsidy_Revenue']}
    rev_plot = {k.replace('_Revenue',''):v for k,v in rev_sources.items() if v > 1e-3}
    if rev_plot:
        fig_rev, ax_rev = plt.subplots(); ax_rev.pie(rev_plot.values(), labels=[f"{k}\n(${v:,.0f})" for k,v in rev_plot.items()], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(rev_plot)))
        ax_rev.set_title('Annual Revenue Breakdown', fontweight='bold'); ax_rev.axis('equal'); plt.tight_layout(); plt.savefig(plot_dir / 'revenue_breakdown.png', dpi=300); plt.close(fig_rev)

    opex_sources = {k: annual_metrics_data.get(k,0) for k in ['VOM_Turbine_Cost', 'VOM_Electrolyzer_Cost', 'VOM_Battery_Cost', 'Startup_Cost', 'Water_Cost', 'Ramping_Cost', 'H2_Storage_Cycle_Cost']}
    opex_sources['Fixed OM (General)'] = OM_COMPONENTS.get('Fixed_OM_General',{}).get('base_cost',0) # Updated key
    if ENABLE_BATTERY and annual_metrics_data.get('Battery_Capacity_MWh',0) > 0: # Add battery fixed OM if applicable
        batt_om_mw_cost = OM_COMPONENTS.get('Fixed_OM_Battery',{}).get('base_cost_per_mw_year',0) * annual_metrics_data.get('Battery_Power_MW',0)
        batt_om_mwh_cost = OM_COMPONENTS.get('Fixed_OM_Battery',{}).get('base_cost_per_mwh_year',0) * annual_metrics_data.get('Battery_Capacity_MWh',0)
        opex_sources['Fixed OM (Battery)'] = batt_om_mw_cost + batt_om_mwh_cost

    opex_plot = {k.replace('_Cost',''):v for k,v in opex_sources.items() if v > 1e-3}
    if opex_plot:
        fig_opex, ax_opex = plt.subplots(); ax_opex.pie(opex_plot.values(), labels=[f"{k}\n(${v:,.0f})" for k,v in opex_plot.items()], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("rocket", len(opex_plot)))
        ax_opex.set_title('Annual Operational Cost Breakdown (Base Year)', fontweight='bold'); ax_opex.axis('equal'); plt.tight_layout(); plt.savefig(plot_dir / 'opex_cost_breakdown.png', dpi=300); plt.close(fig_opex)

    fin_metrics = {k: financial_metrics_data.get(k,np.nan) for k in ['NPV_USD', 'IRR_percent', 'Payback_Period_Years', 'LCOH_USD_per_kg']}
    fin_valid = {k.replace('_USD',' (USD)').replace('_percent',' (%)').replace('_Years',' (Years)').replace('_per_kg',' (USD/kg)'):v for k,v in fin_metrics.items() if not pd.isna(v)}
    if fin_valid:
        fig_fin, ax_fin = plt.subplots(figsize=(8,5)); bars = ax_fin.barh(list(fin_valid.keys()), list(fin_valid.values()), color=sns.color_palette("mako", len(fin_valid)))
        ax_fin.set_xlabel('Value'); ax_fin.set_title('Key Financial Metrics', fontweight='bold')
        for i, (k,v) in enumerate(fin_valid.items()): ax_fin.text(v + 0.01*abs(v) if v!=0 else 0.01, i, f'{v:.2f}', va='center', ha='left' if v>=0 else 'right')
        plt.tight_layout(); plt.savefig(plot_dir / 'financial_metrics_summary.png', dpi=300); plt.close(fig_fin)

    cf_data = {k:annual_metrics_data.get(k,np.nan) for k in ['Electrolyzer_CF_percent', 'Turbine_CF_percent', 'Battery_CF_percent']} # Added Battery CF
    cf_valid = {k.replace('_CF_percent',' CF (%)'):v for k,v in cf_data.items() if not pd.isna(v)}
    if cf_valid:
        fig_cf, ax_cf = plt.subplots(figsize=(6,4)); ax_cf.bar(cf_valid.keys(), cf_valid.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c']) # Added color for battery
        ax_cf.set_ylabel('Capacity Factor (%)'); ax_cf.set_title('Average Capacity Factors', fontweight='bold')
        for k,v in cf_valid.items(): ax_cf.text(k, v+1, f'{v:.1f}%', ha='center')
        ax_cf.set_ylim(0,110); plt.tight_layout(); plt.savefig(plot_dir / 'capacity_factors.png', dpi=300); plt.close(fig_cf)
    logger.info(f"Plots saved to {plot_dir}")


def generate_report(annual_metrics_rpt: dict, financial_metrics_rpt: dict, output_file_path: Path, target_iso_rpt: str, capex_data: dict, om_data: dict, replacement_data: dict, project_lt_rpt: int, construction_p_rpt: int, discount_rt_rpt: float, tax_rt_rpt: float, incremental_metrics_rpt: dict | None = None):
    logger.info(f"Generating TEA report: {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Technical Economic Analysis Report - {target_iso_rpt}\n" + "="*(30+len(target_iso_rpt)) + "\n\n")
        f.write("1. Project Configuration\n" + "-"*25 + "\n")
        f.write(f"  Target ISO: {target_iso_rpt}\n  Project Lifetime: {project_lt_rpt} years\n  Construction Period: {construction_p_rpt} years\n")
        f.write(f"  Discount Rate: {discount_rt_rpt*100:.2f}%\n  Corporate Tax Rate: {tax_rt_rpt*100:.1f}%\n\n")
        f.write("2. Representative Annual Performance (from Optimization)\n" + "-"*58 + "\n")
        if annual_metrics_rpt:
            for k,v in sorted(annual_metrics_rpt.items()): f.write(f"  {k.replace('_',' ').title():<45}: {v:,.2f}\n" if isinstance(v,(int,float)) and not pd.isna(v) else f"  {k.replace('_',' ').title():<45}: {v}\n")
        else: f.write("  No annual metrics data available.\n")
        f.write("\n3. Lifecycle Financial Metrics (Total System)\n" + "-"*46 + "\n")
        if financial_metrics_rpt:
            for k,v in sorted(financial_metrics_rpt.items()):
                lbl = k.replace('_USD',' (USD)').replace('_percent',' (%)').replace('_Years',' (Years)').replace('_per_kg',' (USD/kg)').replace('_',' ').title()
                f.write(f"  {lbl:<45}: {v:,.2f}\n" if isinstance(v,(int,float)) and not pd.isna(v) else f"  {lbl:<45}: {v}\n")
        else: f.write("  No financial metrics data available.\n")
        if incremental_metrics_rpt:
            f.write("\n4. Incremental Financial Metrics (H2/Battery System vs. Baseline)\n" + "-"*68 + "\n")
            for k_inc in ['Annual_Electricity_Opportunity_Cost_USD', 'Total_Incremental_CAPEX_Learned_USD']:
                if k_inc in incremental_metrics_rpt: f.write(f"  {k_inc.replace('_',' ').title()} (USD): {incremental_metrics_rpt[k_inc]:,.2f}\n")
            for k,v in sorted(incremental_metrics_rpt.items()):
                if k in ['pure_incremental_cash_flows', 'traditional_incremental_cash_flows', 'Annual_Electricity_Opportunity_Cost_USD', 'Total_Incremental_CAPEX_Learned_USD']: continue
                lbl = k.replace('_USD',' (USD)').replace('_percent',' (%)').replace('_Years',' (Years)').replace('_per_kg',' (USD/kg)').replace('_',' ').title()
                f.write(f"  Incremental {lbl:<32}: {v:,.2f}\n" if isinstance(v,(int,float)) and not pd.isna(v) else f"  Incremental {lbl:<32}: {v}\n")
        f.write("\n5. Cost Assumptions (Base Year)\n" + "-"*32 + "\n  CAPEX Components:\n")
        for comp, det in sorted(capex_data.items()): f.write(f"    {comp:<30}: ${det.get('total_base_cost_for_ref_size',0):,.0f} (Ref Cap: {det.get('reference_total_capacity_mw',0)}, LR: {det.get('learning_rate_decimal',0)*100}%, Pay Sched: {det.get('payment_schedule_years',{})})\n")
        f.write("  O&M Components (Annual Base):\n")
        for comp, det in sorted(om_data.items()):
            if comp == 'Fixed_OM_Battery':
                 f.write(f"    {comp:<30}: ${det.get('base_cost_per_mw_year',0):,.2f}/MW/yr + ${det.get('base_cost_per_mwh_year',0):,.2f}/MWh/yr (Inflation: {det.get('inflation_rate',0)*100:.1f}%)\n")
            else:
                 f.write(f"    {comp:<30}: ${det.get('base_cost',0):,.0f} (Inflation: {det.get('inflation_rate',0)*100:.1f}%)\n")
        f.write("  Major Replacements:\n")
        for comp, det in sorted(replacement_data.items()): f.write(f"    {comp:<30}: Cost: {'{:.2f}% of Initial CAPEX'.format(det.get('cost_percent_initial_capex',0)*100) if 'cost_percent_initial_capex' in det else '${:,.0f}'.format(det.get('cost',0))} (Years: {det.get('years',[])})\n")
        f.write("\nReport generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
    logger.info(f"TEA report saved to {output_file_path}")


def main():
    """Main execution function for TEA analysis."""
    print("TEA_DEBUG: main() function started in tea.py.") 
    if logger is None: 
        print("TEA_DEBUG: Logger was None in main(), re-initializing basicConfig for logging.") 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - TEA_MAIN_FALLBACK - %(levelname)s - %(message)s')
        globals()['logger'] = logging.getLogger(__name__) 

    logger.info("--- Starting Technical Economic Analysis ---")
    current_target_iso = TARGET_ISO
    logger.info(f"Using Target ISO: {current_target_iso}")
    print(f"TEA_DEBUG: Current Target ISO in main(): {current_target_iso}") 

    tea_base_output_dir = BASE_OUTPUT_DIR_DEFAULT
    os.makedirs(tea_base_output_dir, exist_ok=True)
    tea_output_file = tea_base_output_dir / f"{current_target_iso}_TEA_Summary_Report.txt"
    plot_output_dir = tea_base_output_dir / f"Plots_{current_target_iso}"
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"TEA_DEBUG: Output paths configured. Report: {tea_output_file}, Plots: {plot_output_dir}") 

    tea_sys_params = load_tea_sys_params(current_target_iso, BASE_INPUT_DIR_DEFAULT)

    def get_float_param(params_dict, key, default_value, logger_instance):
        val = params_dict.get(key) 
        if val is None or pd.isna(val): 
            logger_instance.info(f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_value}")
            return float(default_value) 
        try:
            return float(val)
        except (ValueError, TypeError):
            logger_instance.warning(f"Invalid value for parameter '{key}': '{val}'. Using default: {default_value}")
            return float(default_value)

    def get_int_param(params_dict, key, default_value, logger_instance):
        val = params_dict.get(key)
        if val is None or pd.isna(val): 
            logger_instance.info(f"Parameter '{key}' is None or NA (likely missing or empty in CSV). Using default: {default_value}")
            return int(default_value) 
        try:
            return int(float(val)) 
        except (ValueError, TypeError):
            logger_instance.warning(f"Invalid value for parameter '{key}': '{val}'. Using default: {default_value}")
            return int(default_value)

    h2_subsidy_val = get_float_param(tea_sys_params, 'hydrogen_subsidy_value_usd_per_kg', 0.0, logger)
    h2_subsidy_yrs = get_int_param(tea_sys_params, 'hydrogen_subsidy_duration_years', PROJECT_LIFETIME_YEARS, logger)
    baseline_revenue_val = get_float_param(tea_sys_params, 'baseline_nuclear_annual_revenue_USD', 0.0, logger)
    
    run_incremental_raw = tea_sys_params.get('enable_incremental_analysis') 
    if run_incremental_raw is None or pd.isna(run_incremental_raw): 
        run_incremental = True 
        logger.info("'enable_incremental_analysis' not found or NA in sys_params. Defaulting to True.")
    else:
        try:
            if isinstance(run_incremental_raw, str):
                run_incremental = run_incremental_raw.lower() in ['true', '1', 'yes']
            else: 
                run_incremental = bool(int(float(run_incremental_raw)))
        except (ValueError, TypeError):
            run_incremental = True 
            logger.warning(f"Invalid value for 'enable_incremental_analysis': {run_incremental_raw}. Defaulting to True.")

    print("TEA_DEBUG: TEA system parameters loaded and processed in main.") 


    opt_results_dir = SCRIPT_DIR_PATH.parent / "output" / "Results_Standardized"
    results_file_path = opt_results_dir / f"{current_target_iso}_Hourly_Results_Comprehensive.csv"
    print(f"TEA_DEBUG: Attempting to load results from: {results_file_path}") 
    if not results_file_path.exists():
        logger.error(f"Optimization results file not found: {results_file_path}. Exiting TEA.")
        print(f"Error: Optimization results file not found at {results_file_path}")
        return False

    hourly_res_df = load_hourly_results(results_file_path)
    if hourly_res_df is None:
        logger.error("Failed to load optimization results. Exiting TEA.")
        return False 
    print("TEA_DEBUG: Hourly results loaded successfully.") 

    annual_metrics_results = calculate_annual_metrics(hourly_res_df, tea_sys_params)
    if annual_metrics_results is None:
        logger.error("Failed to calculate annual metrics. Exiting TEA.")
        return False
    print("TEA_DEBUG: Annual metrics calculated.") 

    optimized_caps = {
        'Electrolyzer_Capacity_MW': annual_metrics_results.get('Electrolyzer_Capacity_MW', 0),
        'H2_Storage_Capacity_kg': annual_metrics_results.get('H2_Storage_Capacity_kg', 0),
        'Battery_Capacity_MWh': annual_metrics_results.get('Battery_Capacity_MWh', 0), # Added for battery
        'Battery_Power_MW': annual_metrics_results.get('Battery_Power_MW', 0)      # Added for battery
    }
    print(f"TEA_DEBUG: Optimized capacities for cash flow: {optimized_caps}") 

    cash_flows_results = calculate_cash_flows(
        annual_metrics=annual_metrics_results, project_lifetime=PROJECT_LIFETIME_YEARS,
        construction_period=CONSTRUCTION_YEARS, h2_subsidy_value=h2_subsidy_val,
        h2_subsidy_duration=h2_subsidy_yrs, capex_details=CAPEX_COMPONENTS,
        om_details=OM_COMPONENTS, replacement_details=REPLACEMENT_SCHEDULE,
        optimized_capacities=optimized_caps
    )
    print("TEA_DEBUG: Cash flows calculated.") 

    financial_metrics_results = calculate_financial_metrics(
        cash_flows_input=cash_flows_results, discount_rt=DISCOUNT_RATE,
        annual_h2_prod_kg=annual_metrics_results.get('H2_Production_kg_annual', 0),
        project_lt=PROJECT_LIFETIME_YEARS, construction_p=CONSTRUCTION_YEARS
    )
    print("TEA_DEBUG: Financial metrics calculated.") 

    incremental_fin_metrics = None
    if run_incremental:
        print("TEA_DEBUG: Starting incremental analysis.") 
        # Incremental components now explicitly include Battery if ENABLE_BATTERY is true
        incremental_capex_keys = ["Electrolyzer", "H2_Storage"]
        if ENABLE_BATTERY: incremental_capex_keys.append("Battery")
        incremental_capex = {k:v for k,v in CAPEX_COMPONENTS.items() if any(sub in k for sub in incremental_capex_keys)}
        
        # Incremental O&M should also consider battery fixed O&M
        incremental_om = {'Fixed_OM_General': OM_COMPONENTS.get('Fixed_OM_General', {})} # General incremental fixed OM
        if ENABLE_BATTERY:
            incremental_om['Fixed_OM_Battery'] = OM_COMPONENTS.get('Fixed_OM_Battery', {}) # Battery specific fixed OM
        
        incremental_replacements_keys = ["Electrolyzer", "H2_Storage"]
        if ENABLE_BATTERY: incremental_replacements_keys.append("Battery")
        incremental_replacements = {k:v for k,v in REPLACEMENT_SCHEDULE.items() if any(sub in k for sub in incremental_replacements_keys)}

        if baseline_revenue_val <= 0 and 'Energy_Revenue' in annual_metrics_results :
             turbine_max_cap_param = tea_sys_params.get('pTurbine_max_MW') 
             turbine_max_cap = get_float_param(tea_sys_params, 'pTurbine_max_MW', annual_metrics_results.get('Turbine_Capacity_MW',300), logger)
             avg_lmp_val = annual_metrics_results.get('Avg_Electricity_Price_USD_per_MWh', 40)
             baseline_revenue_val = turbine_max_cap * HOURS_IN_YEAR * avg_lmp_val
             logger.info(f"Estimated baseline nuclear revenue: ${baseline_revenue_val:,.2f}")

        incremental_fin_metrics = calculate_incremental_metrics(
            optimized_cash_flows=cash_flows_results, baseline_annual_revenue=baseline_revenue_val,
            project_lifetime=PROJECT_LIFETIME_YEARS, construction_period=CONSTRUCTION_YEARS,
            discount_rt=DISCOUNT_RATE, tax_rt=TAX_RATE, annual_metrics_optimized=annual_metrics_results,
            capex_components_incremental=incremental_capex, om_components_incremental=incremental_om,
            replacement_schedule_incremental=incremental_replacements, h2_subsidy_val = h2_subsidy_val,
            h2_subsidy_yrs = h2_subsidy_yrs,
            optimized_capacities_inc=optimized_caps
        )
        print("TEA_DEBUG: Incremental metrics calculated.") 

    plot_results(
        annual_metrics_data=annual_metrics_results, financial_metrics_data=financial_metrics_results,
        cash_flows_data=cash_flows_results, plot_dir=plot_output_dir,
        construction_p=CONSTRUCTION_YEARS, incremental_metrics_data=incremental_fin_metrics
    )
    print("TEA_DEBUG: Plotting finished.") 
    generate_report(
        annual_metrics_rpt=annual_metrics_results, financial_metrics_rpt=financial_metrics_results,
        output_file_path=tea_output_file, target_iso_rpt=current_target_iso,
        capex_data=CAPEX_COMPONENTS, om_data=OM_COMPONENTS, replacement_data=REPLACEMENT_SCHEDULE,
        project_lt_rpt=PROJECT_LIFETIME_YEARS, construction_p_rpt=CONSTRUCTION_YEARS,
        discount_rt_rpt=DISCOUNT_RATE, tax_rt_rpt=TAX_RATE,
        incremental_metrics_rpt=incremental_fin_metrics
    )
    print("TEA_DEBUG: Report generation finished.") 
    logger.info("--- Technical Economic Analysis completed successfully ---")
    print(f"\nTEA Analysis completed for {current_target_iso}.")
    print(f"  Summary Report: {tea_output_file}")
    print(f"  Plots: {plot_output_dir}")
    return True

if __name__ == "__main__":
    print("TEA_DEBUG: tea.py is being run as the main script.") 
    try:
        main_success = main()
        print(f"TEA_DEBUG: main() returned: {main_success}") 
        sys.exit(0 if main_success else 1)
    except Exception as e_main:
        print(f"TEA_DEBUG: An unhandled error occurred in TEA __main__: {e_main}") 
        if logger: 
            logger.critical(f"An unhandled error occurred in TEA main: {e_main}", exc_info=True)
        else: 
            print(f"CRITICAL FALLBACK: An unhandled error occurred in TEA main: {e_main}")
            traceback.print_exc()
        sys.exit(2)
