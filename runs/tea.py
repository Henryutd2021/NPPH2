# src/tea.py
"""
Technical Economic Analysis (TEA) script for the nuclear-hydrogen optimization framework.
This script performs comprehensive lifecycle analysis including:
- Capital and operational costs
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
import math # Needed for isnan checks maybe

# Import optimization framework from src
# Adjust path if necessary based on your execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))) # Assuming execution from src or root

# --- Try importing necessary config and utils ---
# Wrap imports in try-except for robustness if executed standalone
try:
    from config import (
        TARGET_ISO,
    ENABLE_NONLINEAR_TURBINE_EFF,
    ENABLE_NUCLEAR_GENERATOR,
    ENABLE_ELECTROLYZER,
    ENABLE_BATTERY,
    ENABLE_H2_STORAGE,
    ENABLE_H2_CAP_FACTOR,
    ENABLE_ELECTROLYZER_DEGRADATION_TRACKING,
    ENABLE_STARTUP_SHUTDOWN,
    SIMULATE_AS_DISPATCH_EXECUTION
    )
    # Assuming LCOH calculation might be used elsewhere or needs params from here
    from lcoh import calculate_hydrogen_system_lcoh
except ImportError:
    logging.warning("Could not import from config/lcoh. Using default TARGET_ISO.")
    TARGET_ISO = "DEFAULT_ISO" # Provide a default if config cannot be imported

# --- Logging Setup ---
def setup_logging(base_output_dir, target_iso):
    """Setup logging configuration"""
    # Create TEA_results directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    log_file = base_output_dir / f"tea_analysis_{target_iso}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"TEA Log file initialized at: {log_file}")
    return log_file

# --- Configuration ---
# Define base paths relative to this script file
SCRIPT_DIR = Path(__file__).parent
BASE_OUTPUT_DIR_DEFAULT = SCRIPT_DIR.parent / "TEA_results" # Default output dir
BASE_INPUT_DIR_DEFAULT = SCRIPT_DIR.parent / "input/hourly_data" # Default input dir for sys_data

# --- TEA Parameters ---
# These could be moved to a config file or read from sys_data if needed
PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.08  # 8%
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21  # 21% corporate tax rate

# --- CAPEX Components ---
# Consider loading these from a separate file or sys_data for more flexibility
CAPEX_COMPONENTS = {
    'Electrolyzer_Year1': {
        'base_cost': 25_000_000,  # 50% of cost in first construction year
        'size_dependent': True,    # Cost scales with size
        'learning_rate': 0.15,     # Cost reduction per doubling of capacity
        'year': -2                 # First year of construction
    },
    'Electrolyzer_Year2': {
        'base_cost': 25_000_000,  # 50% of cost in second construction year
        'size_dependent': True,
        'learning_rate': 0.15,
        'year': -1                # Second year of construction
    },
    'H2_Storage_Year1': {
        'base_cost': 5_000_000,   # 50% of storage cost in first year
        'size_dependent': True,
        'learning_rate': 0.10,
        'year': -2
    },
    'H2_Storage_Year2': {
        'base_cost': 5_000_000,   # 50% of storage cost in second year
        'size_dependent': True,
        'learning_rate': 0.10,
        'year': -1
    },
    'Grid_Integration': {
        'base_cost': 5_000_000,
        'size_dependent': False,
        'year': -1                # Second year of construction
    },
    'NPP_Modifications': {
        'base_cost': 2_000_000,
        'size_dependent': False,
        'year': -2                # First year of construction
    }
}

# --- O&M Components ---
OM_COMPONENTS = {
    'Fixed_OM': {
        'base_cost': 1_000_000,    # Annual fixed cost
        'size_dependent': False,
        'inflation_rate': 0.02     # 2% annual increase
    },
    'Variable_OM': {
        'base_cost': 0,            # This is expected to be included in annual_metrics['Annual_Cost'] -> Annual_Profit
        'size_dependent': True,
        'inflation_rate': 0.02
    },
    'Water_Cost': {
        'base_cost': 0,            # Expected to be included in annual_metrics['Annual_Cost'] -> Annual_Profit
        'size_dependent': True,
        'inflation_rate': 0.03     # Higher inflation for water
    }
}

# --- Replacement Schedule ---
REPLACEMENT_SCHEDULE = {
    'Electrolyzer_Stack': {
        'cost': 15_000_000,
        'years': [10, 20],         # Replacements in operational years 10 and 20
        'size_dependent': True
    },
    'H2_Storage_Components': {
        'cost': 5_000_000,
        'years': [15],
        'size_dependent': True
    }
}

# --- Revenue Streams (Inflation/Volatility - applied if simulating future prices) ---
# Current TEA uses a single representative year's metrics, so these are less critical here
# unless future cash flows are being projected with price escalation.
REVENUE_STREAMS = {
    'Energy_Market': {
        'price_escalation': 0.02,  # 2% annual increase
        'volatility': 0.15        # 15% price volatility
    },
    'Ancillary_Services': {
        'price_escalation': 0.03,
        'volatility': 0.20
    },
    'Hydrogen_Sales': {
        'price_escalation': 0.04,
        'volatility': 0.25
    }
}

# --- Helper Function to load sys_data ---
# Minimal version to get needed TEA params
def load_tea_sys_params(sys_data_path):
    params = {}
    try:
        df = pd.read_csv(sys_data_path, index_col=0)
        param_list = [
            'hydrogen_subsidy_value_usd_per_kg',
            'hydrogen_subsidy_duration_years',
            'user_specified_electrolyzer_capacity_MW', # To check if fixed
            'plant_lifetime_years', # Optional override
            'baseline_nuclear_annual_revenue_USD', # Baseline revenue for incremental analysis
            'enable_incremental_analysis' # Whether to run incremental analysis
        ]
        for param_name in param_list:
            if param_name in df.index:
                params[param_name] = df.loc[param_name, 'Value']
            else:
                params[param_name] = None # Indicate parameter not found
        logging.info(f"Loaded TEA relevant params from {sys_data_path}")
    except FileNotFoundError:
        logging.error(f"System data file not found at {sys_data_path} for TEA.")
    except Exception as e:
        logging.error(f"Error loading system data file {sys_data_path} for TEA: {e}")
    return params

# --- Function Implementations ---

def load_hourly_results(filepath):
    """Loads and validates hourly results from optimization."""
    logging.info(f"Loading hourly results from: {filepath}")
    if not filepath.exists():
        logging.error(f"Results file not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        # Check for essential columns needed by calculate_annual_metrics
        required_cols = [
            'Profit_Hourly_USD',
            'Revenue_Total_USD',
            'Cost_HourlyOpex_Total_USD',
            'mHydrogenProduced_kg_hr',
            'pElectrolyzer_MW',
            'pTurbine_MW', # Needed for turbine power data
            'EnergyPrice_LMP_USDperMWh' # Needed for energy price data
        ]

        # Check optional revenue/cost columns used in breakdown
        optional_cols = [
            'Revenue_Energy_USD', # Changed from Market
            'Revenue_Ancillary_USD', # Changed from Service
            'Revenue_Hydrogen_USD',
            'Cost_VOM_Turbine_USD', # Changed from VOM_Total
            'Cost_VOM_Electrolyzer_USD',
            'Cost_VOM_Battery_USD',
            'Cost_Startup_USD', # Changed from Total
            'Cost_Water_USD',
            'Cost_Ramping_USD', # Changed from Total
        ]

        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            logging.error(f"Missing required columns in results file: {missing_required}")
            return None
        
        # Warn if optional columns are missing
        missing_optional = [col for col in optional_cols if col not in df.columns]
        if missing_optional:
            logging.warning(f"Missing optional breakdown columns in results file: {missing_optional}")


        return df
    except Exception as e:
        logging.error(f"Error loading results file {filepath}: {e}", exc_info=True)
        return None

def calculate_annual_metrics(df):
    """Calculates comprehensive annual metrics from hourly results."""
    if df is None:
        return None

    metrics = {}
    try:
        time_col = 'HourOfYear' if 'HourOfYear' in df.columns else df.index.name
        num_hours = len(df)
        if num_hours == 0:
            logging.error("Hourly results DataFrame is empty.")
            return None
        # Assuming results are for a full year if length is near 8760, otherwise scale?
        # For now, assume the sum represents the annual total for the simulated period.
        # time_factor_annual = 8760.0 / num_hours if num_hours > 0 else 1.0 # Scaling factor if not full year

        # Financial Metrics
        metrics['Annual_Profit'] = df['Profit_Hourly_USD'].sum() # This is Revenue - Opex - AnnualizedCapex (from model objective) if objective used
                                                                 # Or Revenue - Opex if objective calculated differently
        metrics['Annual_Revenue'] = df['Revenue_Total_USD'].sum()
        metrics['Annual_Cost'] = df['Cost_HourlyOpex_Total_USD'].sum() # This is just Opex

        # Revenue Breakdown (check if columns exist)
        metrics['Energy_Revenue'] = df['Revenue_Energy_USD'].sum() if 'Revenue_Energy_USD' in df.columns else 0
        metrics['AS_Revenue'] = df['Revenue_Ancillary_USD'].sum() if 'Revenue_Ancillary_USD' in df.columns else 0
        metrics['H2_Revenue'] = df['Revenue_Hydrogen_USD'].sum() if 'Revenue_Hydrogen_USD' in df.columns else 0
        metrics['H2_Subsidy_Revenue'] = df['Revenue_Hydrogen_Subsidy_USD'].sum() if 'Revenue_Hydrogen_Subsidy_USD' in df.columns else 0
        metrics['H2_Sales_Revenue'] = df['Revenue_Hydrogen_Sales_USD'].sum() if 'Revenue_Hydrogen_Sales_USD' in df.columns else 0


        # Cost Breakdown (check if columns exist)
        vom_cost = 0
        if 'Cost_VOM_Turbine_USD' in df.columns: vom_cost += df['Cost_VOM_Turbine_USD'].sum()
        if 'Cost_VOM_Electrolyzer_USD' in df.columns: vom_cost += df['Cost_VOM_Electrolyzer_USD'].sum()
        if 'Cost_VOM_Battery_USD' in df.columns: vom_cost += df['Cost_VOM_Battery_USD'].sum()
        metrics['VOM_Cost'] = vom_cost

        metrics['Startup_Cost'] = df['Cost_Startup_USD'].sum() if 'Cost_Startup_USD' in df.columns else 0
        metrics['Water_Cost'] = df['Cost_Water_USD'].sum() if 'Cost_Water_USD' in df.columns else 0
        metrics['Ramping_Cost'] = df['Cost_Ramping_USD'].sum() if 'Cost_Ramping_USD' in df.columns else 0
        metrics['Storage_Cycle_Cost'] = df['Cost_Storage_Cycle_USD'].sum() if 'Cost_Storage_Cycle_USD' in df.columns else 0


        # Operational Metrics
        metrics['H2_Production'] = df['mHydrogenProduced_kg_hr'].sum() * (8760.0 / num_hours) if num_hours > 0 else 0 # Scale to annual kg

        # Calculate CF based on capacity used IN THE RESULTS file, not max possible capacity
        # This requires capacity columns in the results CSV
        elec_cap_col = 'Electrolyzer_Capacity_MW'
        turbine_cap_col = 'Turbine_Capacity_MW' # Assuming this exists if turbine enabled

        if elec_cap_col in df.columns and not df[elec_cap_col].empty:
            elec_avg_cap_in_run = df[elec_cap_col].iloc[0] # Assuming capacity is constant in the run
            if elec_avg_cap_in_run > 1e-6:
                metrics['Electrolyzer_CF'] = (df['pElectrolyzer_MW'].mean() / elec_avg_cap_in_run) * 100
            else: metrics['Electrolyzer_CF'] = 0
        else: metrics['Electrolyzer_CF'] = np.nan

        if 'pTurbine_MW' in df.columns: # Check if turbine power column exists
             if turbine_cap_col in df.columns and not df[turbine_cap_col].empty:
                 turbine_avg_cap_in_run = df[turbine_cap_col].iloc[0] # Assuming constant
                 if turbine_avg_cap_in_run > 1e-6:
                     metrics['Turbine_CF'] = (df['pTurbine_MW'].mean() / turbine_avg_cap_in_run) * 100
                 else: metrics['Turbine_CF'] = 0
             else: metrics['Turbine_CF'] = np.nan # Capacity column missing
        else: metrics['Turbine_CF'] = 0 # Turbine not enabled or column missing
        
        # ---- Calculate Electrolyzer Energy Consumption and Average Electricity Price ----
        # Calculate total electrolyzer energy consumption in MWh
        if 'pElectrolyzer_MW' in df.columns:
            # Total power to electrolyzer over all hours
            total_electrolyzer_power_mwh = df['pElectrolyzer_MW'].sum()
            # Scale if not a full year
            metrics['Annual_Electrolyzer_MWh'] = total_electrolyzer_power_mwh * (8760.0 / num_hours) if num_hours > 0 else 0
            logging.info(f"Annual electrolyzer energy consumption: {metrics['Annual_Electrolyzer_MWh']:.2f} MWh")
        else:
            metrics['Annual_Electrolyzer_MWh'] = 0
            logging.warning("Electrolyzer power data not found in results file.")
        
        # Calculate average electricity price (weighted by electrolyzer consumption)
        if 'EnergyPrice_LMP_USDperMWh' in df.columns:
            # Simple average price
            metrics['Avg_Electricity_Price'] = df['EnergyPrice_LMP_USDperMWh'].mean()
            
            # Weighted average price when electrolyzer is operating
            if 'pElectrolyzer_MW' in df.columns and df['pElectrolyzer_MW'].sum() > 0:
                weighted_price = (df['EnergyPrice_LMP_USDperMWh'] * df['pElectrolyzer_MW']).sum() / df['pElectrolyzer_MW'].sum()
                metrics['Weighted_Avg_Electricity_Price'] = weighted_price
                logging.info(f"Average electricity price: ${metrics['Avg_Electricity_Price']:.2f}/MWh, " 
                           f"Weighted avg price: ${metrics['Weighted_Avg_Electricity_Price']:.2f}/MWh")
            else:
                metrics['Weighted_Avg_Electricity_Price'] = metrics['Avg_Electricity_Price']
        else:
            # Default values if price data not found
            metrics['Avg_Electricity_Price'] = 40.0  # Default $40/MWh if not available
            metrics['Weighted_Avg_Electricity_Price'] = 40.0
            logging.warning("Electricity price data not found in results file. Using default $40/MWh.")

    except KeyError as e:
        logging.error(f"Missing expected column in hourly results for annual metrics calculation: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calculating annual metrics: {e}", exc_info=True)
        return None

    return metrics

# MODIFICATION START: Updated function signature and logic
def calculate_cash_flows(annual_metrics, project_years,
                         h2_subsidy_value_usd_per_kg,
                         h2_subsidy_duration_years,
                         electrolyzer_capacity_was_fixed): # Boolean
    """Calculates detailed cash flows for the project lifetime, adjusting for subsidy duration."""
    logging.info(f"Calculating cash flows for {project_years} years. Subsidy: {h2_subsidy_value_usd_per_kg:.2f}/kg for {h2_subsidy_duration_years} years. Electrolyzer Fixed: {electrolyzer_capacity_was_fixed}")
    # cash_flows array: Indices 0 to CONSTRUCTION_YEARS-1 for construction, rest for operation
    cash_flows = np.zeros(project_years)

    # --- Construction Period Costs ---
    total_capex = 0
    for i in range(CONSTRUCTION_YEARS):
        year_for_lookup = i - CONSTRUCTION_YEARS # e.g., -2, -1
        cost_this_construction_year = 0
        for component, details in CAPEX_COMPONENTS.items():
            # Check if 'year' exists and matches or if it's not specified (assume at the end of construction)
            if details.get('year', CONSTRUCTION_YEARS-1) == year_for_lookup:
                component_cost = details.get('base_cost', 0)
                cost_this_construction_year += component_cost
                total_capex += component_cost
        
        # Apply construction period cost in the correct year
        cash_flows[i] = -cost_this_construction_year
        logging.debug(f"Construction Year Index {i} (Project Year {year_for_lookup}): CAPEX = ${-cash_flows[i]:,.2f}")
    
    # Check if construction period has any capital expenditure
    if all(cash_flows[i] == 0 for i in range(CONSTRUCTION_YEARS)):
        logging.warning("No CAPEX allocated during construction period! Redistributing costs...")
        # Find total CAPEX from all components
        total_redistributable_capex = sum(details.get('base_cost', 0) for details in CAPEX_COMPONENTS.values())
        # Redistribute evenly across construction years
        for i in range(CONSTRUCTION_YEARS):
            cash_flows[i] = -total_redistributable_capex / CONSTRUCTION_YEARS
            logging.debug(f"Redistributed CAPEX to Construction Year {i}: ${-cash_flows[i]:,.2f}")
        total_capex = total_redistributable_capex

    # Check for components with 'year' = 0 (beginning of first operational year)
    start_of_operation_capex = 0
    for component, details in CAPEX_COMPONENTS.items():
        if details.get('year', -1) == 0:  # Year 0 is beginning of first operational year
            component_cost = details.get('base_cost', 0)
            start_of_operation_capex += component_cost
            total_capex += component_cost
    
    # Apply year 0 capex to the first operational year (index=CONSTRUCTION_YEARS)
    if start_of_operation_capex > 0:
        cash_flows[CONSTRUCTION_YEARS] -= start_of_operation_capex
        logging.debug(f"Start of operation CAPEX (Year 0): ${start_of_operation_capex:,.2f}")
    
    logging.info(f"Total CAPEX: ${total_capex:,.2f}")

    # --- Calculate annual profit component and subsidy effect ---
    # Calculate base annual profit component from metrics
    base_annual_profit_component = annual_metrics.get('Annual_Revenue', 0) - annual_metrics.get('Annual_Cost', 0) # Revenue - Opex
    logging.info(f"Base annual profit component (Rev - Opex) from metrics: ${base_annual_profit_component:,.2f}")
    
    # Calculate the subsidy component in the annual profit
    annual_subsidy_component = 0
    if 'H2_Production' in annual_metrics and annual_metrics['H2_Production'] > 0 and h2_subsidy_value_usd_per_kg > 0:
        annual_subsidy_component = annual_metrics['H2_Production'] * h2_subsidy_value_usd_per_kg
        logging.info(f"Annual subsidy component calculated: ${annual_subsidy_component:,.2f}")

    # --- Operational Period Cash Flows ---
    for op_year_idx in range(project_years - CONSTRUCTION_YEARS): # 0 for 1st op_year, 1 for 2nd...
        operational_year_num = op_year_idx + 1 # 1, 2, 3...
        
        # Start with base profit component from the representative year
        annual_profit = base_annual_profit_component
        
        # Handle subsidy based on duration and whether electrolyzer was fixed
        subsidy_applicable_this_year = operational_year_num <= h2_subsidy_duration_years
        
        # If electrolyzer capacity was fixed and subsidy was included in optimization
        if electrolyzer_capacity_was_fixed:
            if not subsidy_applicable_this_year and annual_subsidy_component > 0:
                # Remove subsidy from years beyond subsidy duration
                annual_profit -= annual_subsidy_component
                logging.debug(f"Op Year {operational_year_num}: Removed subsidy (${annual_subsidy_component:,.2f}) for fixed electrolyzer case.")
        
        # --- Calculate Fixed O&M costs with inflation ---
        fixed_om_cost = 0
        fixed_om_details = OM_COMPONENTS.get('Fixed_OM')
        if fixed_om_details:
            inflated_fixed_om = fixed_om_details.get('base_cost', 0) * (1 + fixed_om_details.get('inflation_rate', 0))**op_year_idx
            fixed_om_cost = inflated_fixed_om
            annual_profit -= fixed_om_cost
            logging.debug(f"Op Year {operational_year_num}: Fixed O&M cost: ${fixed_om_cost:,.2f}")
        
        # --- Calculate Replacement Costs for this year ---
        replacement_cost = 0
        for component, details in REPLACEMENT_SCHEDULE.items():
            if operational_year_num in details.get('years', []):
                component_replacement_cost = details.get('cost', 0)
                replacement_cost += component_replacement_cost
                annual_profit -= component_replacement_cost
                logging.debug(f"Op Year {operational_year_num}: {component} replacement cost: ${component_replacement_cost:,.2f}")
        
        # --- Apply Tax ---
        # Calculate taxable income (profit before tax)
        taxable_income = annual_profit
        tax_amount = 0
        
        if taxable_income > 0:
            tax_amount = taxable_income * TAX_RATE
            annual_profit -= tax_amount
            logging.debug(f"Op Year {operational_year_num}: Tax amount: ${tax_amount:,.2f}")
        
        # Store the net cash flow for this operational year
        cash_flows[op_year_idx + CONSTRUCTION_YEARS] += annual_profit
        
        # Log detailed breakdown for this year
        logging.debug(f"Op Year {operational_year_num} (Index {op_year_idx + CONSTRUCTION_YEARS}): "
                     f"Base Profit=${base_annual_profit_component:,.2f}, "
                     f"Fixed O&M=-${fixed_om_cost:,.2f}, "
                     f"Replacement=-${replacement_cost:,.2f}, "
                     f"Tax=-${tax_amount:,.2f}, "
                     f"Net Cash Flow=${annual_profit:,.2f}")

    return cash_flows
# MODIFICATION END


def calculate_financial_metrics(cash_flows, annual_metrics):
    """Calculates comprehensive financial metrics."""
    metrics = {}

    try:
        # Ensure cash_flows is a list or array suitable for npf functions
        if not isinstance(cash_flows, (list, np.ndarray)):
            raise TypeError("cash_flows must be a list or numpy array.")
        if len(cash_flows) == 0:
            raise ValueError("cash_flows array is empty.")
        
        # Log cash flow pattern for debugging
        logging.info(f"Cash flow pattern: {['Neg' if cf < 0 else 'Pos' if cf > 0 else 'Zero' for cf in cash_flows]}")
        logging.info(f"Cash flow values: {[f'${cf:,.2f}' for cf in cash_flows]}")
        
        # Calculate sign changes in cash flow (crucial for IRR calculation)
        sign_changes = sum(1 for i in range(1, len(cash_flows)) if np.sign(cash_flows[i]) != np.sign(cash_flows[i-1]) and cash_flows[i-1] != 0 and cash_flows[i] != 0)
        logging.info(f"Number of cash flow sign changes: {sign_changes}")

        # Extract initial investment (negative cash flows during construction period)
        initial_investment = abs(sum(cf for cf in cash_flows[:CONSTRUCTION_YEARS] if cf < 0))
        operating_cash_flows = cash_flows[CONSTRUCTION_YEARS:]
        
        # NPV calculation
        # First method: Using numpy_financial directly with all cash flows
        try:
            metrics['NPV'] = npf.npv(DISCOUNT_RATE, cash_flows)
            logging.info(f"NPV calculated using all cash flows: ${metrics['NPV']:,.2f}")
        except Exception as npv_e:
            logging.warning(f"Standard NPV calculation failed: {npv_e}. Trying alternative method.")
            # Alternative method: Calculate NPV manually
            npv = -initial_investment
            for year, cf in enumerate(operating_cash_flows):
                npv += cf / ((1 + DISCOUNT_RATE) ** (year + 1))
            metrics['NPV'] = npv
            logging.info(f"NPV calculated using alternative method: ${metrics['NPV']:,.2f}")
        
        # Helper function for manual IRR calculation using binary search
        def manual_irr(cash_flows, min_rate=-0.999, max_rate=10.0, tolerance=1e-6, max_iterations=1000):
            """Calculate IRR manually using binary search approach."""
            if all(cf >= 0 for cf in cash_flows) or all(cf <= 0 for cf in cash_flows):
                logging.warning("All cash flows have the same sign, IRR calculation not possible.")
                return np.nan
                
            # Function to calculate NPV at a given rate
            def npv_at_rate(rate, cfs):
                return sum(cf / (1 + rate)**(i) for i, cf in enumerate(cfs))
            
            # Check if IRR exists between min_rate and max_rate
            npv_min = npv_at_rate(min_rate, cash_flows)
            npv_max = npv_at_rate(max_rate, cash_flows)
            
            if npv_min * npv_max > 0:
                # No sign change between min and max rate, IRR might not exist
                logging.warning(f"IRR might not exist in range [{min_rate}, {max_rate}]")
                # Try with extended range
                if min_rate > -0.999:
                    return manual_irr(cash_flows, min_rate=-0.999, max_rate=max_rate)
                if max_rate < 100.0:
                    return manual_irr(cash_flows, min_rate=min_rate, max_rate=100.0)
                return np.nan
            
            # Binary search
            current_rate = (min_rate + max_rate) / 2
            iteration = 0
            
            while iteration < max_iterations:
                npv_current = npv_at_rate(current_rate, cash_flows)
                
                if abs(npv_current) < tolerance:
                    return current_rate
                
                if npv_current * npv_min < 0:
                    max_rate = current_rate
                else:
                    min_rate = current_rate
                    npv_min = npv_current
                
                current_rate = (min_rate + max_rate) / 2
                iteration += 1
            
            logging.warning(f"IRR calculation did not converge after {max_iterations} iterations")
            return current_rate  # Return best approximation
        
        # IRR calculation
        # Only attempt IRR if there's at least one positive and one negative cash flow
        if any(cf > 0 for cf in cash_flows) and any(cf < 0 for cf in cash_flows):
            try:
                # Try standard numpy financial IRR first
                metrics['IRR'] = npf.irr(cash_flows)
                logging.info(f"IRR calculated using numpy_financial: {metrics['IRR']*100:.2f}%")
            except Exception as irr_e:
                logging.warning(f"Standard IRR calculation failed: {irr_e}. Trying manual calculation.")
                # Try manual IRR calculation
                metrics['IRR'] = manual_irr(cash_flows)
                if not math.isnan(metrics['IRR']):
                    logging.info(f"IRR calculated manually: {metrics['IRR']*100:.2f}%")
                else:
                    logging.warning("Manual IRR calculation also failed. Setting IRR to NaN.")
        else:
            logging.warning("Cash flow pattern does not allow IRR calculation (need both positive and negative values).")
            metrics['IRR'] = np.nan

        # Payback period calculation - improved to handle edge cases
        cumulative_cash_flow = np.cumsum(cash_flows)
        positive_indices = np.where(cumulative_cash_flow >= 0)[0]
        
        if len(positive_indices) > 0:
            first_positive_idx = positive_indices[0]
            # If the first positive is after construction period, calculate proper payback
            if first_positive_idx >= CONSTRUCTION_YEARS:
                # Interpolate for more accurate payback if possible
                if first_positive_idx > 0 and cumulative_cash_flow[first_positive_idx-1] < 0:
                    # Linear interpolation between years
                    prev_cf = cumulative_cash_flow[first_positive_idx-1]
                    curr_cf = cumulative_cash_flow[first_positive_idx]
                    fraction = abs(prev_cf) / (curr_cf - prev_cf)
                    payback_decimal = first_positive_idx - fraction
                    # Adjust to operational years
                    metrics['Payback_Period'] = payback_decimal - CONSTRUCTION_YEARS + 1
                else:
                    # Simple payback (first year of positive cumulative CF)
                    metrics['Payback_Period'] = first_positive_idx - CONSTRUCTION_YEARS + 1
            else:
                # Payback occurs during construction
                metrics['Payback_Period'] = 0  # Immediate payback in operational terms
        else:
            # Never pays back
            metrics['Payback_Period'] = np.nan
            logging.warning("Project never reaches payback based on provided cash flows.")

        # LCOH calculation using NPV of costs and PV of H2 production
        # This requires discounting annual H2 production.
        total_pv_h2_production_kg = 0
        annual_h2_production = annual_metrics.get('H2_Production', 0) # Annual kg

        if annual_h2_production > 0:
            for op_year_idx in range(PROJECT_LIFETIME_YEARS - CONSTRUCTION_YEARS):
                 # Discount factor for cash flow at end of operational_year_num
                 operational_year_num = op_year_idx + 1
                 # Standard LCOH discounts costs and production similarly. Using end-of-year.
                 pv_factor = (1 + DISCOUNT_RATE) ** operational_year_num
                 total_pv_h2_production_kg += annual_h2_production / pv_factor

        # Calculate LCOH properly
        # LCOH = (PV of Total Costs) / (PV of Total H2 Production)
        if total_pv_h2_production_kg > 0:
            # Calculate total PV of all costs
            # First, get the initial capital expenditure (CAPEX)
            initial_capex = 0
            for component, details in CAPEX_COMPONENTS.items():
                initial_capex += details.get('base_cost', 0)
            
            # Calculate PV of operating costs
            # Include fixed O&M, variable O&M, replacements
            pv_opex = 0
            for op_year_idx in range(PROJECT_LIFETIME_YEARS - CONSTRUCTION_YEARS):
                operational_year_num = op_year_idx + 1
                discount_factor = (1 + DISCOUNT_RATE) ** operational_year_num
                
                # Annual operating cost (from optimization)
                annual_opex = annual_metrics.get('Annual_Cost', 0)
                
                # Add fixed O&M cost with inflation
                fixed_om_details = OM_COMPONENTS.get('Fixed_OM', {})
                if fixed_om_details:
                    inflated_fixed_om = fixed_om_details.get('base_cost', 0) * (1 + fixed_om_details.get('inflation_rate', 0))**op_year_idx
                    annual_opex += inflated_fixed_om
                
                # Add replacement costs if any for this year
                replacement_cost_this_year = 0
                for component, details in REPLACEMENT_SCHEDULE.items():
                    if operational_year_num in details.get('years', []):
                        replacement_cost_this_year += details.get('cost', 0)
                
                annual_opex += replacement_cost_this_year
                
                # Discount the annual opex
                pv_opex += annual_opex / discount_factor
            
            # Total PV of all costs = initial CAPEX + PV of all operating costs
            pv_total_costs = initial_capex + pv_opex
            
            # Calculate LCOH
            metrics['LCOH'] = pv_total_costs / total_pv_h2_production_kg
            logging.info(f"LCOH calculated: ${metrics['LCOH']:.3f}/kg H2")
        else:
            metrics['LCOH'] = np.nan
            logging.warning("Could not calculate LCOH: No hydrogen production data available.")

    except Exception as e:
        logging.error(f"Error calculating financial metrics: {e}", exc_info=True)
        metrics['NPV'] = np.nan
        metrics['IRR'] = np.nan
        metrics['Payback_Period'] = np.nan
        metrics['LCOH'] = np.nan


    return metrics

def calculate_incremental_metrics(cash_flows, baseline_revenue, total_capex, annual_metrics, project_years, h2_subsidy_duration_years=0):
    """
    Calculates incremental financial metrics comparing the optimized system (with H2/battery)
    to the baseline scenario (nuclear plant selling only to grid).
    
    Parameters:
    -----------
    cash_flows : array-like
        Cash flows of the optimized system (with H2/battery)
    baseline_revenue : float
        Annual revenue from the baseline scenario (only selling to grid)
    total_capex : float
        Total capital expenditure for the H2/battery system
    annual_metrics : dict
        Annual metrics from the optimized system
    project_years : int
        Total project lifetime in years
    h2_subsidy_duration_years : int, optional
        Duration of hydrogen subsidy in years (default: 0)
    
    Returns:
    --------
    dict
        Dictionary of incremental financial metrics
    """
    logging.info(f"Calculating incremental metrics compared to baseline (nuclear-only) scenario")
    incremental_metrics = {}
    
    try:
        # Calculate baseline cash flows (simple model: revenue - O&M costs)
        baseline_cash_flows = np.zeros(project_years)
        # Assume baseline has no construction period (plant already exists)
        # Annual O&M costs for nuclear plant (typically 2-3% of capital cost)
        # These would ideally come from actual nuclear plant data
        nuclear_annual_om_percentage = 0.025  # 2.5% of capital cost
        # We don't have nuclear capital cost, so estimate from annual revenue
        # Assuming 10% annual return on nuclear capital
        estimated_nuclear_capital = baseline_revenue / 0.10
        nuclear_annual_om = estimated_nuclear_capital * nuclear_annual_om_percentage
        
        # Fill baseline cash flows (pure nuclear operation)
        for year in range(project_years):
            # Skip construction period in baseline (plant already exists)
            if year >= CONSTRUCTION_YEARS:
                baseline_cash_flows[year] = baseline_revenue - nuclear_annual_om
                # Apply tax
                taxable_income = baseline_cash_flows[year]
                if taxable_income > 0:
                    tax = taxable_income * TAX_RATE
                    baseline_cash_flows[year] -= tax
        
        logging.info(f"Baseline scenario annual cash flow: ${baseline_cash_flows[CONSTRUCTION_YEARS]:,.2f}")
        
        # Calculate opportunity cost of electricity
        # This is electricity that could have been sold to the grid but was used by electrolyzer
        annual_electricity_to_electrolyzer = annual_metrics.get('Annual_Electrolyzer_MWh', 0)
        
        # If Annual_Electrolyzer_MWh is not available, try to estimate it
        if annual_electricity_to_electrolyzer == 0 and 'pElectrolyzer_MW' in annual_metrics:
            # Estimate from average electrolyzer power if available (approximate)
            annual_electricity_to_electrolyzer = annual_metrics.get('pElectrolyzer_MW', 0) * 8760
            logging.warning(f"Estimating electrolyzer consumption from average power: {annual_electricity_to_electrolyzer:.2f} MWh")
        
        # Use weighted average electricity price when electrolyzer is operating (more accurate)
        # Fall back to simple average or default if not available
        if 'Weighted_Avg_Electricity_Price' in annual_metrics:
            avg_electricity_price = annual_metrics['Weighted_Avg_Electricity_Price']
            logging.info(f"Using weighted average electricity price: ${avg_electricity_price:.2f}/MWh")
        elif 'Avg_Electricity_Price' in annual_metrics:
            avg_electricity_price = annual_metrics['Avg_Electricity_Price']
            logging.info(f"Using simple average electricity price: ${avg_electricity_price:.2f}/MWh")
        else:
            avg_electricity_price = 40.0  # Default value
            logging.warning(f"No electricity price data available. Using default: ${avg_electricity_price:.2f}/MWh")
        
        # Calculate the annual opportunity cost
        annual_opportunity_cost = annual_electricity_to_electrolyzer * avg_electricity_price
        incremental_metrics['Annual_Electricity_Opportunity_Cost'] = annual_opportunity_cost
        logging.info(f"Annual electricity opportunity cost: ${annual_opportunity_cost:,.2f} " 
                    f"({annual_electricity_to_electrolyzer:.1f} MWh @ ${avg_electricity_price:.2f}/MWh)")
        
        # Store electricity values for reference
        incremental_metrics['Annual_Electrolyzer_MWh'] = annual_electricity_to_electrolyzer
        incremental_metrics['Avg_Electricity_Price'] = avg_electricity_price
        
        # Create two types of incremental cash flows:
        # 1. Traditional: optimized - baseline (total system comparison)
        traditional_incremental_cash_flows = cash_flows - baseline_cash_flows
        incremental_metrics['traditional_incremental_cash_flows'] = traditional_incremental_cash_flows
        
        # 2. Pure incremental: only considering the H2/battery system with opportunity cost
        pure_incremental_cash_flows = np.zeros(project_years)
        
        # Add capital expenditures in construction period
        for i in range(CONSTRUCTION_YEARS):
            pure_incremental_cash_flows[i] = cash_flows[i]  # Capital expenditures for H2/battery
        
        # For operational years, calculate incremental profit considering opportunity cost
        # Note: The traditional approach already includes opportunity cost implicitly 
        # as the difference between optimized revenue and baseline revenue
        # For pure incremental, we'll use an alternative approach to more explicitly 
        # account for the opportunity cost of electricity used by the electrolyzer
        for year in range(CONSTRUCTION_YEARS, project_years):
            # Calculate H2 system revenues and costs directly
            h2_revenue = 0
            if 'H2_Revenue' in annual_metrics:
                h2_revenue = annual_metrics['H2_Revenue']
            
            # Add H2 subsidy revenue if applicable for this year
            if year - CONSTRUCTION_YEARS < h2_subsidy_duration_years and 'H2_Subsidy_Revenue' in annual_metrics:
                h2_revenue += annual_metrics.get('H2_Subsidy_Revenue', 0)
            
            # H2 system direct costs from annual metrics
            h2_direct_costs = 0
            if 'Cost_VOM_Electrolyzer_USD' in annual_metrics:
                h2_direct_costs += annual_metrics['Cost_VOM_Electrolyzer_USD']
            if 'Cost_VOM_Battery_USD' in annual_metrics:
                h2_direct_costs += annual_metrics['Cost_VOM_Battery_USD']
            if 'Cost_Water_USD' in annual_metrics:
                h2_direct_costs += annual_metrics['Cost_Water_USD']
            
            # Get incremental O&M costs (inflation-adjusted)
            incremental_om_cost = 0
            fixed_om_details = OM_COMPONENTS.get('Fixed_OM')
            if fixed_om_details:
                op_year_idx = year - CONSTRUCTION_YEARS
                incremental_om_cost = fixed_om_details.get('base_cost', 0) * (1 + fixed_om_details.get('inflation_rate', 0))**op_year_idx
            
            # Get incremental replacement costs for this year
            replacement_cost = 0
            operational_year_num = year - CONSTRUCTION_YEARS + 1
            for component, details in REPLACEMENT_SCHEDULE.items():
                if operational_year_num in details.get('years', []):
                    replacement_cost += details.get('cost', 0)
            
            # The pure incremental approach: Revenue - Costs - Opportunity Cost
            annual_profit = h2_revenue - h2_direct_costs - incremental_om_cost - replacement_cost
            
            # Explicitly subtract the opportunity cost of electricity
            annual_profit -= annual_opportunity_cost
            
            # Apply tax if profit is positive
            if annual_profit > 0:
                tax = annual_profit * TAX_RATE
                annual_profit -= tax
            
            pure_incremental_cash_flows[year] = annual_profit
            
            logging.debug(f"Year {year}: Pure incremental cash flow=${annual_profit:,.2f}, " 
                        f"H2 Revenue=${h2_revenue:,.2f}, H2 Costs=${h2_direct_costs:,.2f}, "
                        f"Opportunity Cost=${annual_opportunity_cost:,.2f}")
        
        # Store pure incremental cash flows
        incremental_metrics['incremental_cash_flows'] = pure_incremental_cash_flows
        
        # Log incremental cash flow pattern
        logging.info(f"Pure incremental cash flow pattern: {['Neg' if cf < 0 else 'Pos' if cf > 0 else 'Zero' for cf in pure_incremental_cash_flows]}")
        
        # Calculate NPV of incremental cash flows
        incremental_metrics['NPV'] = npf.npv(DISCOUNT_RATE, pure_incremental_cash_flows)
        logging.info(f"Incremental NPV: ${incremental_metrics['NPV']:,.2f}")
        
        # Calculate IRR of incremental cash flows if applicable
        if any(cf > 0 for cf in pure_incremental_cash_flows) and any(cf < 0 for cf in pure_incremental_cash_flows):
            try:
                incremental_metrics['IRR'] = npf.irr(pure_incremental_cash_flows)
                logging.info(f"Incremental IRR: {incremental_metrics['IRR']*100:.2f}%")
            except Exception as irr_e:
                logging.warning(f"Incremental IRR calculation failed: {irr_e}")
                # Use manual IRR calculation from calculate_financial_metrics
                # This would require refactoring that function to make manual_irr accessible
                incremental_metrics['IRR'] = np.nan
        else:
            incremental_metrics['IRR'] = np.nan
            
        # Calculate payback period for incremental investment
        cumulative_incremental_cf = np.cumsum(pure_incremental_cash_flows)
        positive_indices = np.where(cumulative_incremental_cf >= 0)[0]
        
        if len(positive_indices) > 0:
            first_positive_idx = positive_indices[0]
            if first_positive_idx > 0 and cumulative_incremental_cf[first_positive_idx-1] < 0:
                # Linear interpolation for more accurate payback
                prev_cf = cumulative_incremental_cf[first_positive_idx-1]
                curr_cf = cumulative_incremental_cf[first_positive_idx]
                fraction = abs(prev_cf) / (curr_cf - prev_cf)
                payback_decimal = first_positive_idx - fraction
                incremental_metrics['Payback_Period'] = payback_decimal
            else:
                incremental_metrics['Payback_Period'] = first_positive_idx
        else:
            incremental_metrics['Payback_Period'] = np.nan
            
        # Calculate ROI (Return on Investment)
        # ROI = (Total Returns - Investment) / Investment
        total_incremental_returns = sum(cf for cf in pure_incremental_cash_flows if cf > 0)
        total_incremental_investment = abs(sum(cf for cf in pure_incremental_cash_flows if cf < 0))
        
        if total_incremental_investment > 0:
            incremental_metrics['ROI'] = (total_incremental_returns - total_incremental_investment) / total_incremental_investment
            logging.info(f"Incremental ROI: {incremental_metrics['ROI']*100:.2f}%")
        else:
            incremental_metrics['ROI'] = np.nan
            
        # Calculate incremental LCOH if hydrogen is produced
        annual_h2_production = annual_metrics.get('H2_Production', 0)
        if annual_h2_production > 0:
            # Calculate total hydrogen production over lifetime
            total_pv_h2_production_kg = 0
            for op_year_idx in range(project_years - CONSTRUCTION_YEARS):
                operational_year_num = op_year_idx + 1
                pv_factor = (1 + DISCOUNT_RATE) ** operational_year_num
                total_pv_h2_production_kg += annual_h2_production / pv_factor
                
            # LCOH = PV(Total Incremental Costs) / PV(Total H2 Production)
            # Total incremental costs = negative incremental cash flows
            pv_incremental_costs = sum(abs(cf) / (1 + DISCOUNT_RATE)**(idx+1) 
                                    for idx, cf in enumerate(pure_incremental_cash_flows) if cf < 0)
            
            if total_pv_h2_production_kg > 0:
                incremental_metrics['LCOH'] = pv_incremental_costs / total_pv_h2_production_kg
                logging.info(f"Incremental LCOH: ${incremental_metrics['LCOH']:.3f}/kg")
            else:
                incremental_metrics['LCOH'] = np.nan
        else:
            incremental_metrics['LCOH'] = np.nan
        
        # Analyze component contributions (electrolyzer vs battery)
        electrolyzer_capex = 0
        battery_capex = 0
        
        # Calculate component-specific CAPEX
        for component, details in CAPEX_COMPONENTS.items():
            cost = details.get('base_cost', 0)
            if 'Electrolyzer' in component:
                electrolyzer_capex += cost
            elif 'Battery' in component:
                battery_capex += cost
        
        # Calculate percentage contributions
        if total_capex > 0:
            electrolyzer_pct = (electrolyzer_capex / total_capex) * 100
            battery_pct = (battery_capex / total_capex) * 100
            incremental_metrics['Electrolyzer_CAPEX_Pct'] = electrolyzer_pct
            incremental_metrics['Battery_CAPEX_Pct'] = battery_pct
            incremental_metrics['Electrolyzer_CAPEX'] = electrolyzer_capex
            incremental_metrics['Battery_CAPEX'] = battery_capex
            logging.info(f"Component contributions: Electrolyzer {electrolyzer_pct:.1f}%, Battery {battery_pct:.1f}%")
            
    except Exception as e:
        logging.error(f"Error calculating incremental metrics: {e}", exc_info=True)
        incremental_metrics = {
            'NPV': np.nan,
            'IRR': np.nan,
            'Payback_Period': np.nan,
            'ROI': np.nan,
            'LCOH': np.nan
        }
        
    return incremental_metrics

def plot_results(annual_metrics, financial_metrics, cash_flows, plot_output_dir, incremental_metrics=None):
    """Generates comprehensive visualization of results."""
    os.makedirs(plot_output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')  # Use a modern style

    # Set global matplotlib parameters for better visuals
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # 1. Cash Flow Profile
    plt.figure()
    # Create years array for x-axis
    years = np.arange(1, len(cash_flows) + 1)
    
    # Create cumulative cash flow for secondary y-axis
    cumulative_cf = np.cumsum(cash_flows)
    
    # Create primary axis for annual cash flows
    ax1 = plt.gca()
    annual_bars = ax1.bar(years, cash_flows, color='royalblue', alpha=0.7, label='Annual Cash Flow')
    
    # Color negative bars differently
    for idx, cf in enumerate(cash_flows):
        if cf < 0:
            annual_bars[idx].set_color('firebrick')
    
    # Create secondary axis for cumulative cash flow
    ax2 = ax1.twinx()
    ax2.plot(years, cumulative_cf, 'darkgreen', marker='o', label='Cumulative Cash Flow')
    
    # Add zero line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add construction/operations divider
    ax1.axvline(x=CONSTRUCTION_YEARS + 0.5, color='orange', linestyle='--', linewidth=1.5, 
               label='Operations Start')
    
    # Labels and title
    ax1.set_title('Project Cash Flow Profile', fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual Cash Flow ($)')
    ax2.set_ylabel('Cumulative Cash Flow ($)')
    
    # X-axis ticks at each year
    ax1.set_xticks(years)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Format y-axis with commas for thousands
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
    
    plt.tight_layout()
    plt.savefig(plot_output_dir / 'cash_flow_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1.B. Incremental Cash Flow Profile (if available)
    if incremental_metrics is not None and 'incremental_cash_flows' in incremental_metrics:
        incremental_cash_flows = incremental_metrics['incremental_cash_flows']
        plt.figure()
        
        # Create years array for x-axis
        years = np.arange(1, len(incremental_cash_flows) + 1)
        
        # Create cumulative cash flow for secondary y-axis
        cumulative_icf = np.cumsum(incremental_cash_flows)
        
        # Create primary axis for annual cash flows
        ax1 = plt.gca()
        annual_bars = ax1.bar(years, incremental_cash_flows, color='mediumpurple', alpha=0.7, label='Incremental Annual Cash Flow')
        
        # Color negative bars differently
        for idx, cf in enumerate(incremental_cash_flows):
            if cf < 0:
                annual_bars[idx].set_color('crimson')
        
        # Create secondary axis for cumulative cash flow
        ax2 = ax1.twinx()
        ax2.plot(years, cumulative_icf, 'darkgreen', marker='o', label='Cumulative Incremental Cash Flow')
        
        # Add zero line
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Add construction/operations divider if applicable
        if CONSTRUCTION_YEARS > 0:
            ax1.axvline(x=CONSTRUCTION_YEARS + 0.5, color='orange', linestyle='--', linewidth=1.5, 
                       label='Operations Start')
        
        # Labels and title
        ax1.set_title('Incremental Cash Flow Profile (H2/Battery System Only)', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Incremental Annual Cash Flow ($)')
        ax2.set_ylabel('Cumulative Incremental Cash Flow ($)')
        
        # X-axis ticks at each year
        ax1.set_xticks(years)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Format y-axis with commas for thousands
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
        
        # Annotate opportunity cost if available
        if 'Annual_Electricity_Opportunity_Cost' in incremental_metrics:
            opportunity_cost = incremental_metrics['Annual_Electricity_Opportunity_Cost']
            textstr = f"Annual Electricity Opportunity Cost: ${opportunity_cost:,.0f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(plot_output_dir / 'incremental_cash_flow_profile.png', dpi=300, bbox_inches='tight')
        
        # 1.C. Compare Traditional and Pure Incremental Cash Flows (if both available)
        if 'traditional_incremental_cash_flows' in incremental_metrics:
            trad_incremental_cash_flows = incremental_metrics['traditional_incremental_cash_flows']
            plt.figure(figsize=(14, 8))
            
            # Create years array for x-axis
            years = np.arange(1, len(incremental_cash_flows) + 1)
            
            # Calculate cumulative cash flows for both
            cum_trad_icf = np.cumsum(trad_incremental_cash_flows)
            cum_pure_icf = np.cumsum(incremental_cash_flows)
            
            # Create subplot for annual values
            plt.subplot(2, 1, 1)
            width = 0.35
            
            # Plot traditional incremental cash flows
            trad_bars = plt.bar(years - width/2, trad_incremental_cash_flows, width, color='royalblue', 
                               alpha=0.7, label='Traditional Incremental Cash Flow')
            
            # Plot pure incremental cash flows
            pure_bars = plt.bar(years + width/2, incremental_cash_flows, width, color='darkgreen', 
                               alpha=0.7, label='Pure Incremental Cash Flow')
            
            # Color negative bars differently
            for idx, cf in enumerate(trad_incremental_cash_flows):
                if cf < 0:
                    trad_bars[idx].set_color('crimson')
            
            for idx, cf in enumerate(incremental_cash_flows):
                if cf < 0:
                    pure_bars[idx].set_color('darkred')
            
            # Add zero line and operations start
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            if CONSTRUCTION_YEARS > 0:
                plt.axvline(x=CONSTRUCTION_YEARS + 0.5, color='orange', linestyle='--', linewidth=1.5, 
                           label='Operations Start')
            
            plt.title('Comparison of Incremental Cash Flow Approaches', fontweight='bold')
            plt.ylabel('Annual Cash Flow ($)')
            plt.xticks(years)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create subplot for cumulative values
            plt.subplot(2, 1, 2)
            plt.plot(years, cum_trad_icf, 'royalblue', marker='o', linewidth=2,
                   label='Cumulative Traditional Incremental CF')
            plt.plot(years, cum_pure_icf, 'darkgreen', marker='s', linewidth=2,
                   label='Cumulative Pure Incremental CF')
            
            # Add zero line and operations start
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            if CONSTRUCTION_YEARS > 0:
                plt.axvline(x=CONSTRUCTION_YEARS + 0.5, color='orange', linestyle='--', linewidth=1.5, 
                           label='Operations Start')
            
            plt.xlabel('Year')
            plt.ylabel('Cumulative Cash Flow ($)')
            plt.xticks(years)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format y-axis with commas for thousands
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
            
            # Add annotation about opportunity cost
            if 'Annual_Electricity_Opportunity_Cost' in incremental_metrics:
                opportunity_cost = incremental_metrics['Annual_Electricity_Opportunity_Cost']
                textstr = (f"Annual Electricity Opportunity Cost: ${opportunity_cost:,.0f}\n"
                          f"Pure Incremental includes electricity that could have been sold to grid")
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                             verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(plot_output_dir / 'incremental_cash_flow_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.close()

    # 2. Revenue Breakdown (Based on representative annual metrics)
    revenue_data = {
        'Energy Market': annual_metrics.get('Energy_Revenue', 0),
        'Ancillary Services': annual_metrics.get('AS_Revenue', 0),
        'H2 Sales': annual_metrics.get('H2_Sales_Revenue', 0),
        'H2 Subsidy': annual_metrics.get('H2_Subsidy_Revenue', 0)
    }
    
    # Filter out zero/negligible values for pie chart clarity
    revenue_data_filtered = {k: v for k, v in revenue_data.items() if v > 1e-3}
    
    if revenue_data_filtered:
        plt.figure()
        
        # Use a visually appealing color palette
        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(revenue_data_filtered)))
        
        # Create pie chart with percentage and value labels
        wedges, texts, autotexts = plt.pie(
            revenue_data_filtered.values(), 
            labels=None,
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12}
        )
        
        # Adjust percentage text color for visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add legend with values
        labels_with_values = [f"{k} (${v:,.0f})" for k, v in revenue_data_filtered.items()]
        plt.legend(wedges, labels_with_values, title="Revenue Sources", 
                  loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))
        
        plt.title('Annual Revenue Breakdown', fontweight='bold')
        plt.axis('equal')  # Equal aspect ratio ensures pie is circular
        plt.tight_layout()
        plt.savefig(plot_output_dir / 'revenue_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        logging.warning("No revenue data > 0 for revenue breakdown plot.")

    # 3. Cost Breakdown (Based on representative annual Opex metrics)
    cost_data = {
        'Variable O&M': annual_metrics.get('VOM_Cost', 0),
        'Startup': annual_metrics.get('Startup_Cost', 0),
        'Water': annual_metrics.get('Water_Cost', 0),
        'Ramping': annual_metrics.get('Ramping_Cost', 0),
        'H2 Storage Cycle': annual_metrics.get('Storage_Cycle_Cost', 0)
    }
    
    # Add Fixed O&M if available in OM_COMPONENTS
    if 'Fixed_OM' in OM_COMPONENTS:
        cost_data['Fixed O&M'] = OM_COMPONENTS['Fixed_OM'].get('base_cost', 0)
    
    # Filter out zero/negligible values for pie chart clarity
    cost_data_filtered = {k: v for k, v in cost_data.items() if v > 1e-3}
    
    if cost_data_filtered:
        plt.figure()
        
        # Use a visually appealing color palette (different from revenue)
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(cost_data_filtered)))
        
        # Create pie chart with percentage and value labels
        wedges, texts, autotexts = plt.pie(
            cost_data_filtered.values(), 
            labels=None,
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12}
        )
        
        # Adjust percentage text color for visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add legend with values
        labels_with_values = [f"{k} (${v:,.0f})" for k, v in cost_data_filtered.items()]
        plt.legend(wedges, labels_with_values, title="Cost Categories", 
                  loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))
        
        plt.title('Annual Operating Cost Breakdown', fontweight='bold')
        plt.axis('equal')  # Equal aspect ratio ensures pie is circular
        plt.tight_layout()
        plt.savefig(plot_output_dir / 'opex_cost_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        logging.warning("No Opex cost data > 0 for cost breakdown plot.")

    # 4. Key Financial Metrics Summary
    # Format metrics for bar chart
    metrics_to_plot = {
        'NPV ($M)': financial_metrics.get('NPV', np.nan) / 1e6,  # Convert to millions
        'IRR (%)': financial_metrics.get('IRR', np.nan) * 100 if not math.isnan(financial_metrics.get('IRR', np.nan)) else np.nan,
        'Payback (Years)': financial_metrics.get('Payback_Period', np.nan),
        'LCOH ($/kg)': financial_metrics.get('LCOH', np.nan)
    }
    
    # Filter out NaN values before plotting
    metrics_plottable = {k: v for k, v in metrics_to_plot.items() if not math.isnan(v)}
    
    if metrics_plottable:
        plt.figure()
        
        # Use different colors for different metrics
        colors = ['royalblue', 'forestgreen', 'darkred', 'darkorange'][:len(metrics_plottable)]
        
        # Create horizontal bar chart for better label visibility
        bars = plt.barh(list(metrics_plottable.keys()), list(metrics_plottable.values()), color=colors, alpha=0.7)
        
        # Add value labels at the end of each bar
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01  # Position slightly to the right of the bar
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    va='center', ha='left', fontweight='bold')
        
        plt.title('Key Financial Metrics', fontweight='bold')
        plt.xlabel('Value')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_output_dir / 'financial_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        logging.warning("No valid financial metrics to plot.")
        
    # 5. Capacity Factors (if available)
    cf_data = {
        'Electrolyzer': annual_metrics.get('Electrolyzer_CF', np.nan),
        'Turbine': annual_metrics.get('Turbine_CF', np.nan)
    }
    
    # Filter out NaN values
    cf_data_plottable = {k: v for k, v in cf_data.items() if not math.isnan(v)}
    
    if cf_data_plottable:
        plt.figure()
        
        # Use different colors for different components
        colors = ['#3498db', '#e74c3c'][:len(cf_data_plottable)]
        
        # Create bar chart
        bars = plt.bar(list(cf_data_plottable.keys()), list(cf_data_plottable.values()), color=colors, alpha=0.8)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height * 1.01, f'{height:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Capacity Factors', fontweight='bold')
        plt.ylabel('Capacity Factor (%)')
        plt.ylim(0, max(list(cf_data_plottable.values())) * 1.2)  # Add space for labels
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_output_dir / 'capacity_factors.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Incremental Metrics Comparison (if available)
    if incremental_metrics is not None and any(not math.isnan(v) for v in incremental_metrics.values() if isinstance(v, (int, float)) and not isinstance(v, np.ndarray)):
        # Create comparison of original vs incremental metrics
        comparison_metrics = {
            'NPV ($M)': [financial_metrics.get('NPV', np.nan) / 1e6, incremental_metrics.get('NPV', np.nan) / 1e6],
            'IRR (%)': [financial_metrics.get('IRR', np.nan) * 100 if not math.isnan(financial_metrics.get('IRR', np.nan)) else np.nan,
                       incremental_metrics.get('IRR', np.nan) * 100 if not math.isnan(incremental_metrics.get('IRR', np.nan)) else np.nan],
            'Payback (Years)': [financial_metrics.get('Payback_Period', np.nan), incremental_metrics.get('Payback_Period', np.nan)],
            'LCOH ($/kg)': [financial_metrics.get('LCOH', np.nan), incremental_metrics.get('LCOH', np.nan)]
        }
        
        # Filter metrics where both values aren't NaN
        comparison_metrics_plottable = {k: v for k, v in comparison_metrics.items() 
                                        if not (math.isnan(v[0]) and math.isnan(v[1]))}
        
        if comparison_metrics_plottable:
            plt.figure(figsize=(14, 8))
            
            # Set width of bars
            bar_width = 0.35
            ind = np.arange(len(comparison_metrics_plottable))
            
            # Create grouped bar chart
            plt.bar(ind - bar_width/2, [v[0] for v in comparison_metrics_plottable.values()], 
                   bar_width, label='Total System', color='royalblue', alpha=0.8)
            plt.bar(ind + bar_width/2, [v[1] for v in comparison_metrics_plottable.values()], 
                   bar_width, label='Incremental (H2/Battery Only)', color='firebrick', alpha=0.8)
            
            # Add labels and title
            plt.xlabel('Financial Metrics')
            plt.ylabel('Value')
            plt.title('Comparison of Total vs Incremental Financial Metrics', fontweight='bold')
            plt.xticks(ind, comparison_metrics_plottable.keys(), rotation=15)
            plt.legend()
            
            # Add value labels on bars
            for i, metric_key in enumerate(comparison_metrics_plottable.keys()):
                metric_values = comparison_metrics_plottable[metric_key]
                for j, value in enumerate(metric_values):
                    if not math.isnan(value):
                        x_pos = i - bar_width/2 if j == 0 else i + bar_width/2
                        y_pos = value + (abs(value) * 0.02) if value > 0 else value - (abs(value) * 0.15)
                        plt.text(x_pos, y_pos, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_output_dir / 'incremental_metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # 7. Add ROI chart for incremental investment
        if not math.isnan(incremental_metrics.get('ROI', np.nan)):
            plt.figure(figsize=(8, 6))
            
            roi_value = incremental_metrics.get('ROI', 0) * 100  # Convert to percentage
            plt.bar(['Return on Incremental Investment'], [roi_value], color='forestgreen', alpha=0.8)
            
            plt.title('Return on Incremental Investment', fontweight='bold')
            plt.ylabel('ROI (%)')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add value label
            plt.text(0, roi_value + 1, f'{roi_value:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plot_output_dir / 'incremental_roi.png', dpi=300, bbox_inches='tight')
            plt.close()

def generate_report(annual_metrics, financial_metrics, output_file, incremental_metrics=None):
    """Generates comprehensive TEA report."""
    logging.info(f"Generating TEA report at: {output_file}")
    try:
        with open(output_file, 'w') as f:
            f.write("Technical Economic Analysis Report\n")
            f.write("================================\n\n")

            f.write("1. Project Overview\n")
            f.write("------------------\n")
            f.write(f"Project Lifetime: {PROJECT_LIFETIME_YEARS} years\n")
            f.write(f"Construction Period: {CONSTRUCTION_YEARS} years\n")
            f.write(f"Discount Rate: {DISCOUNT_RATE*100:.2f}%\n")
            f.write(f"Tax Rate: {TAX_RATE*100:.1f}%\n\n")

            f.write("2. Representative Annual Performance (based on optimization run)\n")
            f.write("-------------------------------------------------------------\n")
            # Format metrics nicely
            if annual_metrics:
                for key, value in sorted(annual_metrics.items()):
                    if value is None or (isinstance(value, float) and math.isnan(value)): 
                        formatted_value = "N/A"
                    elif isinstance(value, (int, float)):
                        if "Revenue" in key or "Cost" in key or "Profit" in key: 
                            formatted_value = f"${value:,.2f}"
                        elif "CF" in key: 
                            formatted_value = f"{value:.2f}%"
                        elif "Production" in key: 
                            formatted_value = f"{value:,.2f} kg"
                        else: 
                            formatted_value = f"{value:,.2f}"
                    else: 
                        formatted_value = str(value)
                    f.write(f"{key.replace('_', ' ')}: {formatted_value}\n")
            else:
                f.write("No annual metrics data available.\n")
            f.write("\n")

            f.write("3. Lifecycle Financial Metrics\n")
            f.write("-----------------------------\n")
            if financial_metrics:
                for key, value in sorted(financial_metrics.items()):
                    key_label = key.replace('_', ' ')
                    if value is None or (isinstance(value, float) and math.isnan(value)): 
                        formatted_value = "N/A"
                    elif key == 'IRR': 
                        formatted_value = f"{value*100:.2f}%" if value >= 0 else "Negative/N/A"
                    elif 'LCOH' in key: 
                        formatted_value = f"${value:.3f}/kg" # Increased precision for LCOH
                    elif key == 'Payback_Period': 
                        if value == 0:
                            formatted_value = "Immediate (during operations)"
                        elif value > 0:
                            formatted_value = f"{value:.1f} years (Operational)"
                        else:
                            formatted_value = "N/A (never pays back)"
                    else: 
                        formatted_value = f"${value:,.2f}"
                    f.write(f"{key_label}: {formatted_value}\n")
            else:
                f.write("No financial metrics data available.\n")
            f.write("\n")
            
            # Add incremental metrics section if available
            if incremental_metrics:
                f.write("4. Incremental Financial Metrics (H2/Battery System Only)\n")
                f.write("------------------------------------------------------\n")
                f.write("These metrics represent the financial performance of the incremental\n")
                f.write("investment in H2/battery systems compared to the baseline nuclear plant.\n\n")
                
                # First show component breakdown if available
                if 'Electrolyzer_CAPEX_Pct' in incremental_metrics and 'Battery_CAPEX_Pct' in incremental_metrics:
                    f.write("Component Investment Breakdown:\n")
                    if 'Electrolyzer_CAPEX' in incremental_metrics:
                        f.write(f"  Electrolyzer CAPEX: ${incremental_metrics['Electrolyzer_CAPEX']:,.2f} ")
                        f.write(f"({incremental_metrics['Electrolyzer_CAPEX_Pct']:.1f}% of total)\n")
                    if 'Battery_CAPEX' in incremental_metrics:
                        f.write(f"  Battery CAPEX: ${incremental_metrics['Battery_CAPEX']:,.2f} ")
                        f.write(f"({incremental_metrics['Battery_CAPEX_Pct']:.1f}% of total)\n")
                    f.write("\n")
                
                # Show opportunity cost if available
                if 'Annual_Electricity_Opportunity_Cost' in incremental_metrics:
                    opportunity_cost = incremental_metrics['Annual_Electricity_Opportunity_Cost']
                    f.write("Opportunity Cost Analysis:\n")
                    f.write(f"  Annual Electricity Opportunity Cost: ${opportunity_cost:,.2f}\n")
                    f.write("  (This is the revenue foregone by using electricity for H2/battery instead of grid sales)\n\n")
                    
                    # Add distinction between traditional and pure incremental approaches
                    f.write("Incremental Analysis Approaches:\n")
                    f.write("  1. Traditional Incremental: Compares total system to baseline nuclear operation\n")
                    f.write("  2. Pure Incremental: Isolates H2/battery contribution with opportunity cost factored in\n")
                    f.write("\n  The metrics below reflect the Pure Incremental approach, which provides a more\n")
                    f.write("  accurate assessment of the H2/battery investment value by accounting for\n")
                    f.write("  the opportunity cost of electricity that could have been sold to the grid.\n\n")
                
                # Show main incremental metrics
                for key, value in sorted(incremental_metrics.items()):
                    # Skip the special fields we handle separately
                    if key in ['incremental_cash_flows', 'traditional_incremental_cash_flows', 'Electrolyzer_CAPEX', 'Battery_CAPEX', 
                              'Electrolyzer_CAPEX_Pct', 'Battery_CAPEX_Pct', 'Annual_Electricity_Opportunity_Cost']:
                        continue
                        
                    key_label = key.replace('_', ' ')
                    if value is None or (isinstance(value, float) and math.isnan(value)): 
                        formatted_value = "N/A"
                    elif key == 'IRR': 
                        formatted_value = f"{value*100:.2f}%" if value >= 0 else "Negative/N/A"
                    elif key == 'ROI':
                        formatted_value = f"{value*100:.2f}%"
                    elif 'LCOH' in key: 
                        formatted_value = f"${value:.3f}/kg"
                    elif key == 'Payback_Period': 
                        if value == 0:
                            formatted_value = "Immediate"
                        elif value > 0:
                            formatted_value = f"{value:.1f} years"
                        else:
                            formatted_value = "N/A (never pays back)"
                    else: 
                        formatted_value = f"${value:,.2f}"
                    f.write(f"Incremental {key_label}: {formatted_value}\n")
                f.write("\n")
                f.write("Note: Incremental metrics isolate the economic performance of the\n")
                f.write("H2/battery investment by comparing against baseline nuclear operation.\n")
                f.write("The analysis accounts for:\n")
                f.write("  - Additional capital costs for H2/battery equipment\n")
                f.write("  - Additional O&M costs for the H2/battery system\n")
                f.write("  - Revenue from hydrogen production\n")
                f.write("  - Opportunity costs from diverting electricity from grid sales to H2 production\n\n")
                f.write("A positive incremental NPV indicates that the H2/battery investment adds value\n")
                f.write("beyond what could be achieved by selling all electricity to the grid.\n\n")

            f.write("5. Component Costs (Base Year Estimates)\n")
            f.write("---------------------------------------\n")
            f.write("CAPEX Components:\n")
            for component, details in sorted(CAPEX_COMPONENTS.items()):
                cost = details.get('base_cost', 0)
                unit = "/kW" if details.get('size_dependent') else "" # Add unit if needed
                year = details.get('year', 'Construction End')
                year_label = f" (Year {year})" if isinstance(year, int) else f" ({year})"
                f.write(f"  {component}: ${cost:,.0f}{unit}{year_label}\n")
            
            f.write("\nO&M Components (Base Annual):\n")
            for component, details in sorted(OM_COMPONENTS.items()):
                cost = details.get('base_cost', 0)
                inflation = details.get('inflation_rate', 0) * 100
                if component != 'Fixed_OM': 
                    cost_label = "(From Optimization)"
                else: 
                    cost_label = f"${cost:,.0f}/year"
                f.write(f"  {component}: {cost_label} (Inflation: {inflation:.1f}%/year)\n")
            
            f.write("\nReplacements:\n")
            for component, details in sorted(REPLACEMENT_SCHEDULE.items()):
                cost = details.get('cost', 0)
                years = details.get('years', [])
                f.write(f"  {component}: ${cost:,.0f} in years {years}\n")
            f.write("\n")
            
            f.write("6. Analysis Notes\n")
            f.write("----------------\n")
            f.write("- All financial metrics calculated using end-of-year cash flow convention.\n")
            f.write("- LCOH represents levelized cost over project lifetime including all capital and operating costs.\n")
            f.write("- NPV and IRR calculations account for the time value of money with the specified discount rate.\n")
            f.write("- Hydrogen subsidy effects are included according to provided parameters.\n")
            f.write("- No terminal/salvage value is assumed at the end of project lifetime.\n")
            
            if incremental_metrics:
                f.write("- Incremental analysis compares the H2/battery system investment to baseline nuclear operation.\n")
                f.write("- ROI (Return on Investment) measures the profitability of the incremental investment.\n")
                f.write("- Opportunity costs of electricity (nuclear power that could have been sold to grid) are included.\n")
                if 'Electrolyzer_CAPEX' in incremental_metrics and 'Battery_CAPEX' in incremental_metrics:
                    f.write("- Both electrolyzer and battery investments are included in the incremental analysis.\n")
            
            f.write("\n\nReport generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            
    except Exception as e:
        logging.error(f"Failed to write report file {output_file}: {e}", exc_info=True)


def main():
    """Main execution function for TEA analysis."""
    # Set explicit paths relative to this script
    tea_base_output_dir = BASE_OUTPUT_DIR_DEFAULT
    base_input_dir = BASE_INPUT_DIR_DEFAULT
    
    # Setup logging
    log_file = setup_logging(tea_base_output_dir, TARGET_ISO)
    logging.info("--- Starting Technical Economic Analysis ---")

    # Determine TARGET_ISO (from config or default)
    current_target_iso = TARGET_ISO
    logging.info(f"Using Target ISO: {current_target_iso}")

    # Define output file paths using TARGET_ISO
    tea_output_file = tea_base_output_dir / f"{current_target_iso}_TEA_Summary.txt"
    plot_output_dir = tea_base_output_dir / f"Plots_{current_target_iso}"

    # Create output directories
    os.makedirs(tea_base_output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)

    # --- Load necessary TEA parameters from sys_data ---
    sys_data_path = base_input_dir / "sys_data_advanced.csv"
    tea_params = load_tea_sys_params(sys_data_path)

    # Extract parameters with defaults if not found/loaded
    try:
        h2_subsidy_val = float(tea_params.get('hydrogen_subsidy_value_usd_per_kg', 0.0))
        h2_subsidy_yrs = int(float(tea_params.get('hydrogen_subsidy_duration_years', PROJECT_LIFETIME_YEARS))) # Default to full lifetime if missing
        
        # Determine if electrolyzer capacity was fixed
        elec_fixed_val_str = tea_params.get('user_specified_electrolyzer_capacity_MW', None)
        electrolyzer_was_fixed = False
        if elec_fixed_val_str is not None:
            try:
                if float(elec_fixed_val_str) >= 0:
                    electrolyzer_was_fixed = True
            except (ValueError, TypeError):
                pass # Keep as False if value is invalid

        # Use PROJECT_LIFETIME_YEARS as default if plant_lifetime_years not provided or invalid
        project_lifetime = PROJECT_LIFETIME_YEARS
        if 'plant_lifetime_years' in tea_params and tea_params['plant_lifetime_years'] is not None:
            try:
                lifetime_val = float(tea_params['plant_lifetime_years'])
                if lifetime_val > 0:
                    project_lifetime = int(lifetime_val)
            except (ValueError, TypeError):
                logging.warning(f"Invalid plant_lifetime_years value: {tea_params['plant_lifetime_years']}. Using default: {PROJECT_LIFETIME_YEARS}")
        
        # Get baseline nuclear annual revenue for incremental analysis
        baseline_revenue = 0.0
        run_incremental_analysis = False
        
        if 'baseline_nuclear_annual_revenue_USD' in tea_params and tea_params['baseline_nuclear_annual_revenue_USD'] is not None:
            try:
                baseline_revenue = float(tea_params['baseline_nuclear_annual_revenue_USD'])
                logging.info(f"Baseline nuclear annual revenue from sys_data: ${baseline_revenue:,.2f}")
            except (ValueError, TypeError):
                logging.warning(f"Invalid baseline_nuclear_annual_revenue_USD value: {tea_params['baseline_nuclear_annual_revenue_USD']}. Will calculate from hourly results.")
                baseline_revenue = 0.0
        
        # Check if incremental analysis is enabled
        if 'enable_incremental_analysis' in tea_params and tea_params['enable_incremental_analysis'] is not None:
            try:
                run_incremental_analysis = bool(int(float(tea_params['enable_incremental_analysis'])))
                logging.info(f"Incremental analysis enabled: {run_incremental_analysis}")
            except (ValueError, TypeError):
                logging.warning(f"Invalid enable_incremental_analysis value: {tea_params['enable_incremental_analysis']}. Defaulting to True.")
                run_incremental_analysis = True
        else:
            # Always enable incremental analysis by default
            run_incremental_analysis = True
            logging.info("Incremental analysis enabled by default.")

    except (ValueError, TypeError) as e:
        logging.error(f"Invalid format for TEA parameters in sys_data: {e}. Using defaults.")
        h2_subsidy_val = 0.0
        h2_subsidy_yrs = PROJECT_LIFETIME_YEARS
        electrolyzer_was_fixed = False
        project_lifetime = PROJECT_LIFETIME_YEARS
        baseline_revenue = 0.0
        run_incremental_analysis = True

    logging.info(f"TEA Params - H2 Subsidy: {h2_subsidy_val}/kg, Duration: {h2_subsidy_yrs} yrs, Elec Fixed: {electrolyzer_was_fixed}, Lifetime: {project_lifetime} yrs")

    # Load optimization results
    # Ensure this path points to the correct output location of your main optimization
    results_file = SCRIPT_DIR.parent / f"output/Results_Standardized/{current_target_iso}_Hourly_Results_Comprehensive.csv"
    
    # Check if file exists, if not try alternative locations
    if not results_file.exists():
        alternate_paths = [
            SCRIPT_DIR.parent / f"output/{current_target_iso}_Hourly_Results_Comprehensive.csv",
            SCRIPT_DIR.parent / f"output/Results/{current_target_iso}_Hourly_Results_Comprehensive.csv",
            SCRIPT_DIR.parent / f"output/Results_{current_target_iso}_Hourly_Results.csv"
        ]
        
        for alt_path in alternate_paths:
            if alt_path.exists():
                results_file = alt_path
                logging.info(f"Using alternative results file: {alt_path}")
                break
    
    hourly_results = load_hourly_results(results_file)

    if hourly_results is None:
        logging.error("Failed to load optimization results. Exiting TEA.")
        print(f"Error: Could not find or load results file. Checked: {results_file}")
        print("Please ensure the optimization has been run and results are available.")
        return False

    # Calculate metrics
    annual_metrics = calculate_annual_metrics(hourly_results)
    if annual_metrics is None:
        logging.error("Failed to calculate annual metrics. Exiting TEA.")
        return False

    # Calculate cash flows - passing the updated TEA parameters
    cash_flows = calculate_cash_flows(annual_metrics, project_lifetime,
                                      h2_subsidy_value_usd_per_kg=h2_subsidy_val,
                                      h2_subsidy_duration_years=h2_subsidy_yrs,
                                      electrolyzer_capacity_was_fixed=electrolyzer_was_fixed)

    # Calculate financial metrics
    financial_metrics = calculate_financial_metrics(cash_flows, annual_metrics)
    
    # Calculate incremental metrics if enabled
    incremental_metrics = None
    if run_incremental_analysis:
        # If baseline revenue not provided, calculate from hourly results
        if baseline_revenue <= 0:
            try:
                # For nuclear-only operation, we need to estimate the revenue from selling all electricity to the grid
                # We can use the hourly electricity price and turbine capacity
                turbine_max_capacity = float(tea_params.get('pTurbine_max_MW', 380.0))  # Default based on sys_data_advanced.csv
                logging.info(f"Using turbine maximum capacity: {turbine_max_capacity} MW")
                
                # Calculate baseline revenue based on full power operation * electricity price
                if 'EnergyPrice_LMP_USDperMWh' in hourly_results.columns and 'pTurbine_MW' in hourly_results.columns:
                    # Two options:
                    # 1. Use average electricity price and assume full capacity operation
                    hours_in_year = len(hourly_results)
                    
                    # Get the most appropriate electricity price metric
                    if 'Weighted_Avg_Electricity_Price' in annual_metrics:
                        avg_energy_price = annual_metrics['Weighted_Avg_Electricity_Price']
                        logging.info(f"Using weighted average electricity price: ${avg_energy_price:.2f}/MWh")
                    else:
                        avg_energy_price = hourly_results['EnergyPrice_LMP_USDperMWh'].mean()
                        logging.info(f"Using simple average electricity price: ${avg_energy_price:.2f}/MWh")
                    
                    baseline_revenue_option1 = avg_energy_price * turbine_max_capacity * hours_in_year
                    
                    # 2. Use the actual turbine output but assume all power goes to grid (no electrolyzer/battery)
                    # This may not be accurate if the turbine output was optimized WITH electrolyzer/battery
                    actual_turbine_output = hourly_results['pTurbine_MW'].sum()
                    weighted_avg_price = (hourly_results['EnergyPrice_LMP_USDperMWh'] * hourly_results['pTurbine_MW']).sum() / actual_turbine_output if actual_turbine_output > 0 else 0
                    baseline_revenue_option2 = weighted_avg_price * actual_turbine_output
                    
                    # Use option 1 as it better represents "full power to grid" scenario
                    baseline_revenue = baseline_revenue_option1
                    logging.info(f"Calculated baseline revenue from full capacity operation: ${baseline_revenue:,.2f}")
                    logging.info(f"Average electricity price: ${avg_energy_price:.2f}/MWh, Turbine capacity: {turbine_max_capacity} MW, Hours: {hours_in_year}")
                    
                    # Add turbine power data to annual metrics if missing
                    if 'Annual_Turbine_MWh' not in annual_metrics:
                        annual_metrics['Annual_Turbine_MWh'] = turbine_max_capacity * hours_in_year
                        logging.info(f"Added turbine annual generation: {annual_metrics['Annual_Turbine_MWh']:.2f} MWh")
                else:
                    logging.warning("Cannot calculate baseline revenue: required columns missing from hourly results.")
                    logging.warning("Using total energy revenue as baseline revenue.")
                    baseline_revenue = annual_metrics.get('Energy_Revenue', 0)
            except Exception as e:
                logging.error(f"Error calculating baseline revenue: {e}")
                baseline_revenue = annual_metrics.get('Energy_Revenue', 0)
                logging.warning(f"Using total energy revenue as baseline revenue: ${baseline_revenue:,.2f}")
        
        # Sum up total CAPEX for H2/battery system
        total_capex = sum(details.get('base_cost', 0) for details in CAPEX_COMPONENTS.values())
        logging.info(f"Total incremental CAPEX: ${total_capex:,.2f}")
        
        # Calculate incremental metrics
        incremental_metrics = calculate_incremental_metrics(
            cash_flows=cash_flows,
            baseline_revenue=baseline_revenue,
            total_capex=total_capex,
            annual_metrics=annual_metrics,
            project_years=project_lifetime,
            h2_subsidy_duration_years=h2_subsidy_yrs
        )
        logging.info("Incremental analysis completed.")

    # Generate visualizations
    plot_results(annual_metrics, financial_metrics, cash_flows, plot_output_dir, incremental_metrics)

    # Generate report
    generate_report(annual_metrics, financial_metrics, tea_output_file, incremental_metrics)

    logging.info("--- Technical Economic Analysis completed successfully ---")
    print(f"\nTEA Analysis completed.")
    print(f"  Summary Report: {tea_output_file}")
    print(f"  Plots: {plot_output_dir}")
    print(f"  Log file: {log_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)