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

# Import optimization framework from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
from lcoh import calculate_hydrogen_system_lcoh

# --- Logging Setup ---
def setup_logging():
    """Setup logging configuration"""
    # Create TEA_results directory if it doesn't exist
    os.makedirs(TEA_BASE_OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = TEA_BASE_OUTPUT_DIR / f"tea_analysis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

# --- Configuration ---
TEA_BASE_OUTPUT_DIR = Path("../TEA_results")
TEA_OUTPUT_FILE = TEA_BASE_OUTPUT_DIR / f"{TARGET_ISO}_TEA_Summary.txt"
PLOT_OUTPUT_DIR = TEA_BASE_OUTPUT_DIR / f"Plots_{TARGET_ISO}"

# --- TEA Parameters ---
PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.08  # 8%
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21  # 21% corporate tax rate

# --- CAPEX Components ---
CAPEX_COMPONENTS = {
    'Electrolyzer': {
        'base_cost': 50_000_000,  # Base cost for reference size
        'size_dependent': True,    # Cost scales with size
        'learning_rate': 0.15,     # Cost reduction per doubling of capacity
        'year': 0                  # When cost is incurred
    },
    'H2_Storage': {
        'base_cost': 10_000_000,
        'size_dependent': True,
        'learning_rate': 0.10,
        'year': 0
    },
    'Grid_Integration': {
        'base_cost': 5_000_000,
        'size_dependent': False,
        'year': 0
    },
    'NPP_Modifications': {
        'base_cost': 2_000_000,
        'size_dependent': False,
        'year': 0
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
        'base_cost': 0,            # Will be calculated from optimization results
        'size_dependent': True,
        'inflation_rate': 0.02
    },
    'Water_Cost': {
        'base_cost': 0,            # Will be calculated from optimization results
        'size_dependent': True,
        'inflation_rate': 0.03     # Higher inflation for water
    }
}

# --- Replacement Schedule ---
REPLACEMENT_SCHEDULE = {
    'Electrolyzer_Stack': {
        'cost': 15_000_000,
        'years': [10, 20],         # Replacements in years 10 and 20
        'size_dependent': True
    },
    'H2_Storage_Components': {
        'cost': 5_000_000,
        'years': [15],
        'size_dependent': True
    }
}

# --- Revenue Streams ---
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

def load_hourly_results(filepath):
    """Loads and validates hourly results from optimization."""
    logging.info(f"Loading hourly results from: {filepath}")
    if not filepath.exists():
        logging.error(f"Results file not found: {filepath}")
        return None
        
    try:
        df = pd.read_csv(filepath)
        required_cols = [
            'Profit_Hourly_USD',
            'Revenue_Total_USD',
            'Cost_HourlyOpex_Total_USD',
            'mHydrogenProduced_kg_hr',
            'pElectrolyzer_MW',
            'pTurbine_MW'
        ]
        
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logging.error(f"Missing required columns: {missing_cols}")
            return None
            
        return df
    except Exception as e:
        logging.error(f"Error loading results: {e}", exc_info=True)
        return None

def calculate_annual_metrics(df):
    """Calculates comprehensive annual metrics from hourly results."""
    if df is None:
        return None
        
    metrics = {}
    
    # Financial Metrics
    metrics['Annual_Profit'] = df['Profit_Hourly_USD'].sum()
    metrics['Annual_Revenue'] = df['Revenue_Total_USD'].sum()
    metrics['Annual_Cost'] = df['Cost_HourlyOpex_Total_USD'].sum()
    
    # Revenue Breakdown
    revenue_cols = {
        'Revenue_Energy_Market_USD': 'Energy_Revenue',
        'Revenue_Ancillary_Service_USD': 'AS_Revenue',
        'Revenue_Hydrogen_USD': 'H2_Revenue'
    }
    
    for col, name in revenue_cols.items():
        if col in df.columns:
            metrics[name] = df[col].sum()
        else:
            metrics[name] = 0
            
    # Cost Breakdown
    cost_cols = {
        'Cost_VOM_Total_USD': 'VOM_Cost',
        'Cost_Startup_Total_USD': 'Startup_Cost',
        'Cost_Water_USD': 'Water_Cost',
        'Cost_Ramping_Total_USD': 'Ramping_Cost'
    }
    
    for col, name in cost_cols.items():
        if col in df.columns:
            metrics[name] = df[col].sum()
        else:
            metrics[name] = 0
            
    # Operational Metrics
    metrics['H2_Production'] = df['mHydrogenProduced_kg_hr'].sum()
    metrics['Electrolyzer_CF'] = (df['pElectrolyzer_MW'].sum() / 
                                (df['pElectrolyzer_MW'].max() * len(df))) * 100
    metrics['Turbine_CF'] = (df['pTurbine_MW'].sum() / 
                           (df['pTurbine_MW'].max() * len(df))) * 100
    
    return metrics

def calculate_cash_flows(annual_metrics, project_years):
    """Calculates detailed cash flows for the project lifetime."""
    cash_flows = np.zeros(project_years)
    
    # Construction period (negative years)
    for year in range(-CONSTRUCTION_YEARS, 0):
        capex_year = 0
        for component, details in CAPEX_COMPONENTS.items():
            if details['year'] == year:
                capex_year += details['base_cost']
        cash_flows[year + CONSTRUCTION_YEARS] = -capex_year
    
    # Operational years
    for year in range(project_years - CONSTRUCTION_YEARS):
        operational_year = year + 1
        
        # Base operational cash flow
        cash_flow = annual_metrics['Annual_Profit']
        
        # Add inflation to costs
        for component, details in OM_COMPONENTS.items():
            if component == 'Fixed_OM':
                cash_flow -= details['base_cost'] * (1 + details['inflation_rate'])**year
            else:
                # Variable costs are already in annual_metrics
                pass
        
        # Add replacements
        for component, details in REPLACEMENT_SCHEDULE.items():
            if operational_year in details['years']:
                cash_flow -= details['cost']
        
        # Apply tax
        if cash_flow > 0:
            cash_flow *= (1 - TAX_RATE)
            
        cash_flows[year + CONSTRUCTION_YEARS] = cash_flow
    
    return cash_flows

def calculate_financial_metrics(cash_flows, annual_metrics):
    """Calculates comprehensive financial metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['NPV'] = npf.npv(DISCOUNT_RATE, cash_flows)
    metrics['IRR'] = npf.irr(cash_flows)
    
    # Payback period
    cumulative_cash_flow = np.cumsum(cash_flows)
    payback_idx = np.where(cumulative_cash_flow >= 0)[0]
    metrics['Payback_Period'] = payback_idx[0] if len(payback_idx) > 0 else np.nan
    
    # LCOH calculation
    total_h2_production = annual_metrics['H2_Production'] * PROJECT_LIFETIME_YEARS
    metrics['LCOH'] = -metrics['NPV'] / total_h2_production if total_h2_production > 0 else np.nan
    
    return metrics

def plot_results(annual_metrics, financial_metrics, cash_flows):
    """Generates comprehensive visualization of results."""
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    # 1. Cash Flow Profile
    plt.figure(figsize=(12, 6))
    years = np.arange(-CONSTRUCTION_YEARS, PROJECT_LIFETIME_YEARS - CONSTRUCTION_YEARS)
    plt.plot(years, cash_flows, 'b-', label='Annual Cash Flow')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Project Cash Flow Profile')
    plt.xlabel('Year')
    plt.ylabel('Cash Flow ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOT_OUTPUT_DIR / 'cash_flow_profile.png')
    plt.close()
    
    # 2. Revenue Breakdown
    revenue_data = {
        'Energy Market': annual_metrics['Energy_Revenue'],
        'Ancillary Services': annual_metrics['AS_Revenue'],
        'Hydrogen Sales': annual_metrics['H2_Revenue']
    }
    
    plt.figure(figsize=(10, 6))
    plt.pie(revenue_data.values(), labels=revenue_data.keys(), autopct='%1.1f%%')
    plt.title('Annual Revenue Breakdown')
    plt.savefig(PLOT_OUTPUT_DIR / 'revenue_breakdown.png')
    plt.close()
    
    # 3. Cost Breakdown
    cost_data = {
        'VOM': annual_metrics['VOM_Cost'],
        'Startup': annual_metrics['Startup_Cost'],
        'Water': annual_metrics['Water_Cost'],
        'Ramping': annual_metrics['Ramping_Cost']
    }
    
    plt.figure(figsize=(10, 6))
    plt.pie(cost_data.values(), labels=cost_data.keys(), autopct='%1.1f%%')
    plt.title('Annual Cost Breakdown')
    plt.savefig(PLOT_OUTPUT_DIR / 'cost_breakdown.png')
    plt.close()
    
    # 4. Key Metrics Summary
    metrics_data = {
        'NPV': financial_metrics['NPV'],
        'IRR': financial_metrics['IRR'] * 100,
        'Payback Period': financial_metrics['Payback_Period'],
        'LCOH': financial_metrics['LCOH']
    }
    
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_data.keys(), metrics_data.values())
    plt.title('Key Financial Metrics')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig(PLOT_OUTPUT_DIR / 'financial_metrics.png')
    plt.close()

def generate_report(annual_metrics, financial_metrics, output_file):
    """Generates comprehensive TEA report."""
    with open(output_file, 'w') as f:
        f.write("Technical Economic Analysis Report\n")
        f.write("================================\n\n")
        
        f.write("1. Project Overview\n")
        f.write("------------------\n")
        f.write(f"Project Lifetime: {PROJECT_LIFETIME_YEARS} years\n")
        f.write(f"Construction Period: {CONSTRUCTION_YEARS} years\n")
        f.write(f"Discount Rate: {DISCOUNT_RATE*100}%\n")
        f.write(f"Tax Rate: {TAX_RATE*100}%\n\n")
        
        f.write("2. Annual Performance\n")
        f.write("--------------------\n")
        for key, value in annual_metrics.items():
            f.write(f"{key}: ${value:,.2f}\n")
        f.write("\n")
        
        f.write("3. Financial Metrics\n")
        f.write("-------------------\n")
        for key, value in financial_metrics.items():
            if key == 'IRR':
                f.write(f"{key}: {value*100:.2f}%\n")
            elif key == 'LCOH':
                f.write(f"{key}: ${value:.2f}/kg\n")
            else:
                f.write(f"{key}: ${value:,.2f}\n")
        f.write("\n")
        
        f.write("4. Component Costs\n")
        f.write("-----------------\n")
        for component, details in CAPEX_COMPONENTS.items():
            f.write(f"{component}: ${details['base_cost']:,.2f}\n")
        f.write("\n")
        
        f.write("5. O&M Costs\n")
        f.write("------------\n")
        for component, details in OM_COMPONENTS.items():
            f.write(f"{component}: ${details['base_cost']:,.2f}/year\n")
        f.write("\n")

def main():
    """Main execution function for TEA analysis."""
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting Technical Economic Analysis")
    
    # Create output directories
    os.makedirs(TEA_BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    # Load optimization results
    results_file = Path(f"../Results_Standardized/{TARGET_ISO}_Hourly_Results_Comprehensive.csv")
    hourly_results = load_hourly_results(results_file)
    
    if hourly_results is None:
        logging.error("Failed to load optimization results. Exiting.")
        return
    
    # Calculate metrics
    annual_metrics = calculate_annual_metrics(hourly_results)
    if annual_metrics is None:
        logging.error("Failed to calculate annual metrics. Exiting.")
        return
    
    # Calculate cash flows
    cash_flows = calculate_cash_flows(annual_metrics, PROJECT_LIFETIME_YEARS)
    
    # Calculate financial metrics
    financial_metrics = calculate_financial_metrics(cash_flows, annual_metrics)
    
    # Generate visualizations
    plot_results(annual_metrics, financial_metrics, cash_flows)
    
    # Generate report
    generate_report(annual_metrics, financial_metrics, TEA_OUTPUT_FILE)
    
    logging.info("Technical Economic Analysis completed successfully")
    print(f"\nAnalysis completed. Results saved to {TEA_BASE_OUTPUT_DIR}")
    print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()