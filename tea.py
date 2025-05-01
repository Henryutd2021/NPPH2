# tea.py
import pandas as pd
import numpy as np
import numpy_financial as npf # Requires: pip install numpy-financial
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import sys
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Configuration ---
# TODO: Update these values based on your specific project and ISO
TARGET_ISO = 'ERCOT' # Specify the ISO for which results are being analyzed
HOURLY_RESULTS_FILE = Path(f"./Results_Standardized/{TARGET_ISO}_Hourly_Results_Comprehensive.csv") # Path to the optimization output

# --- Output Configuration ---
TEA_BASE_OUTPUT_DIR = Path("./tea") # Base directory for all TEA outputs
TEA_OUTPUT_FILE = TEA_BASE_OUTPUT_DIR / f"{TARGET_ISO}_TEA_Summary.txt"
PLOT_OUTPUT_DIR = TEA_BASE_OUTPUT_DIR / f"Plots_{TARGET_ISO}"

# --- TEA Parameters ---
# ***** ALL VALUES BELOW ARE EXAMPLES - PROVIDE YOUR PROJECT'S ACTUAL VALUES *****
PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.08 # Example: 8%
CONSTRUCTION_YEARS = 2 # Example: Time before operation starts

# --- CAPEX (Total Installed Costs in Year 0) ---
# Provide total installed costs ($). These occur *before* year 1 of operation.
CAPEX_ELECTROLYZER = 50_000_000 # Example: Based on electrolyzer size
CAPEX_STORAGE = 10_000_000    # Example: H2 storage system cost
CAPEX_INTEGRATION = 5_000_000  # Example: Grid connection, BOP modification, control systems
CAPEX_NPP_MODS = 2_000_000     # Example: Steam extraction mods, piping
# -------------------------------------
TOTAL_CAPEX = CAPEX_ELECTROLYZER + CAPEX_STORAGE + CAPEX_INTEGRATION + CAPEX_NPP_MODS

# --- Fixed O&M ($ per year) ---
# Annual costs regardless of operation level
FIXED_OM_PER_YEAR = 1_000_000 # Example: Staff, insurance, regular maintenance

# --- Major Replacements / Refurbishments --- 
# Define as a dictionary: {year_of_replacement: cost_in_that_year}
REPLACEMENT_COSTS = {
    10: 15_000_000, # Example: Electrolyzer stack replacement Year 10
    20: 15_000_000, # Example: Electrolyzer stack replacement Year 20
} 

# --- Tax Rate (Optional) ---
# Apply corporate tax rate if needed (e.g., 0.21 for 21%)
TAX_RATE = 0.0 # Example: 0% tax rate

# ==============================================================================
# --- DATA LOADING AND PREPARATION --- 
# ==============================================================================
def load_hourly_results(filepath):
    """Loads the hourly results CSV file."""
    logging.info(f"Attempting to load hourly results from: {filepath}")
    if not filepath.exists():
        logging.error(f"Hourly results file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded {filepath}. Shape: {df.shape}")
        # Basic validation - check for essential columns
        required_cols = ['Profit_Hourly_USD'] # Add more as needed for analysis
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns in {filepath}. Expected: {required_cols}")
            return None
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {filepath}: {e}", exc_info=True)
        return None

# ==============================================================================
# --- ANNUAL CALCULATIONS --- 
# ==============================================================================
def calculate_annual_aggregates(df):
    """Calculates annual totals from hourly results."""
    if df is None:
        return None
    
    aggregates = {}
    # Assuming 8760 hours in the simulation year
    hours_in_year = len(df)
    if hours_in_year != 8760:
        logging.warning(f"Input data has {hours_in_year} hours, not 8760. Annualization based on this number.")
        if hours_in_year == 0:
            logging.error("Input dataframe is empty.")
            return None
            
    # --- Key Financial Metrics ---
    aggregates['Annual Profit Before Tax ($)'] = df['Profit_Hourly_USD'].sum()
    aggregates['Annual Revenue Total ($)'] = df['Revenue_Total_USD'].sum()
    aggregates['Annual OpEx Total ($)'] = df['Cost_HourlyOpex_Total_USD'].sum()

    # --- Revenue Breakdown (Requires these columns in CSV) ---
    revenue_cols = {
        'Revenue_Energy_Market_USD': 'Annual Revenue Energy ($)',
        'Revenue_Ancillary_Service_USD': 'Annual Revenue AS ($)',
        'Revenue_Hydrogen_USD': 'Annual Revenue H2 ($)',
        # Add other specific revenue columns if they exist
    }
    for col, agg_name in revenue_cols.items():
        if col in df.columns:
            aggregates[agg_name] = df[col].sum()
        else:
            logging.warning(f"Revenue column '{col}' not found in results.")
            aggregates[agg_name] = 0

    # --- Cost Breakdown (Requires these columns in CSV) ---
    cost_cols = {
        'Cost_VOM_Total_USD': 'Annual VOM Cost ($)',
        'Cost_Startup_Total_USD': 'Annual Startup Cost ($)',
        'Cost_Water_USD': 'Annual Water Cost ($)',
        'Cost_Ramping_Total_USD': 'Annual Ramping Cost ($)',
        # Add other specific cost columns if they exist
    }
    for col, agg_name in cost_cols.items():
        if col in df.columns:
            aggregates[agg_name] = df[col].sum()
        else:
            logging.warning(f"Cost column '{col}' not found in results.")
            aggregates[agg_name] = 0
            
    # --- Operational Metrics ---
    op_cols = {
        'mH2_produced_kg': 'Annual H2 Production (kg)',
        'pElectrolyzer_Input_MW': 'Average Electrolyzer Power (MW)', # Calculate average usage
        'pTurbine_Output_MW': 'Average Turbine Power (MW)',
    }
    if 'mH2_produced_kg' in df.columns:
        aggregates['Annual H2 Production (kg)'] = df['mH2_produced_kg'].sum()
    if 'pElectrolyzer_Input_MW' in df.columns:
        aggregates['Average Electrolyzer Power (MW)'] = df[df['pElectrolyzer_Input_MW'] > 1e-3]['pElectrolyzer_Input_MW'].mean() # Avg when running
        aggregates['Electrolyzer Capacity Factor (%)'] = (df['pElectrolyzer_Input_MW'].sum() / (df['pElectrolyzer_Input_MW'].max() * hours_in_year)) * 100 if df['pElectrolyzer_Input_MW'].max() > 0 else 0
    if 'pTurbine_Output_MW' in df.columns:
        aggregates['Average Turbine Power (MW)'] = df[df['pTurbine_Output_MW'] > 1e-3]['pTurbine_Output_MW'].mean() # Avg when running

    logging.info(f"Calculated annual aggregates: {aggregates}")
    return aggregates

# ==============================================================================
# --- TEA CALCULATIONS (NPV, IRR, Payback) ---
# ==============================================================================
def calculate_tea_metrics(annual_aggregates):
    """Calculates NPV, IRR, and Payback Period."""
    if annual_aggregates is None:
        return None

    # --- Construct Cash Flow Series ---
    # Year 0 (and potentially negative years for construction) = CAPEX outflow
    # Operational Years = Annual Profit - Fixed O&M + Replacements
    
    # Determine the starting year index (negative for construction)
    start_year = -CONSTRUCTION_YEARS + 1
    total_years = PROJECT_LIFETIME_YEARS + CONSTRUCTION_YEARS
    years = np.arange(start_year, PROJECT_LIFETIME_YEARS + 1)
    cash_flows = np.zeros(total_years)

    # Add CAPEX spread over construction years (simplified: assumes equal spread)
    # More accurate would be a specific spending profile
    capex_per_construction_year = TOTAL_CAPEX / CONSTRUCTION_YEARS if CONSTRUCTION_YEARS > 0 else TOTAL_CAPEX
    for i in range(CONSTRUCTION_YEARS):
        cash_flows[i] = -capex_per_construction_year
    
    # Add Operational Cash Flows
    annual_profit_before_tax = annual_aggregates.get('Annual Profit Before Tax ($)', 0)
    for year_idx in range(CONSTRUCTION_YEARS, total_years):
        operational_year = year_idx - CONSTRUCTION_YEARS + 1 # Year 1, 2, ... 25
        
        # Start with annual profit (from simulation)
        profit = annual_profit_before_tax
        
        # Subtract Fixed O&M
        profit -= FIXED_OM_PER_YEAR
        
        # Account for Taxes (Simplified: applied to profit after Fixed O&M)
        taxable_income = profit
        taxes = taxable_income * TAX_RATE if taxable_income > 0 else 0
        profit_after_tax = profit - taxes
        
        # Subtract Replacement Costs (occur as outflows in specific years)
        replacement_outflow = REPLACEMENT_COSTS.get(operational_year, 0)
        
        cash_flows[year_idx] = profit_after_tax - replacement_outflow
        
    # --- Calculate Metrics ---
    tea_results = {}
    # Adjust cash flows for npf.npv which assumes first value is t=0
    # Our cash_flows array starts from the first construction year, need to shift index
    try:
        # NPV calculation needs flows starting from t=0 (end of construction / start of op year 1)
        operational_cash_flows = cash_flows[CONSTRUCTION_YEARS:]
        initial_investment = -sum(cash_flows[:CONSTRUCTION_YEARS]) # Total CAPEX as positive number
        
        npv_op_flows = npf.npv(DISCOUNT_RATE, operational_cash_flows)
        tea_results['NPV ($)'] = npv_op_flows - initial_investment
        
    except Exception as e:
        logging.warning(f"Could not calculate NPV: {e}")
        tea_results['NPV ($)'] = np.nan

    try:
        # IRR calculation needs flows starting from t=0, including initial investment
        irr_cash_flows = np.concatenate(([-initial_investment], operational_cash_flows))
        tea_results['IRR (%)'] = npf.irr(irr_cash_flows) * 100
    except Exception as e:
        logging.warning(f"Could not calculate IRR: {e}. Might be no sign change in cash flows.")
        tea_results['IRR (%)'] = np.nan

    # --- Payback Period (Simple) ---
    cumulative_cash_flow = np.cumsum(cash_flows)
    try:
        # Find first year where cumulative cash flow is positive (after construction)
        payback_idx = np.where(cumulative_cash_flow[CONSTRUCTION_YEARS:] > 0)[0][0]
        year_before_payback = payback_idx + CONSTRUCTION_YEARS -1 # Index in the full cash_flows array
        
        # Linear interpolation for fractional year
        cumulative_at_year_before = cumulative_cash_flow[year_before_payback]
        cash_flow_in_payback_year = cash_flows[year_before_payback + 1]
        
        fractional_year = abs(cumulative_at_year_before) / cash_flow_in_payback_year if cash_flow_in_payback_year > 0 else 0
        payback_year_float = (year_before_payback - CONSTRUCTION_YEARS + 1) + fractional_year # Operational year
        tea_results['Simple Payback Period (Years)'] = payback_year_float
    except IndexError:
        logging.warning("Project does not reach payback within its lifetime.")
        tea_results['Simple Payback Period (Years)'] = np.nan
        
    tea_results['Cash Flow Series'] = cash_flows # Store for plotting
    tea_results['Years'] = np.arange(start_year, PROJECT_LIFETIME_YEARS + 1) # Years corresponding to cash_flows

    logging.info(f"Calculated TEA metrics: {tea_results}")
    return tea_results

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================
def plot_cash_flows(tea_results, plot_dir):
    """Plots annual and cumulative cash flows."""
    if tea_results is None or 'Cash Flow Series' not in tea_results:
        logging.warning("Cannot plot cash flows, data missing.")
        return
        
    years = tea_results['Years']
    cash_flows = tea_results['Cash Flow Series']
    cumulative_cash_flows = np.cumsum(cash_flows)

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for annual cash flow
    colors = ['red' if x < 0 else 'green' for x in cash_flows]
    ax1.bar(years, cash_flows, color=colors, alpha=0.7, label='Annual Cash Flow')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual Cash Flow ($)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, axis='y')
    # Format y-axis for currency
    from matplotlib.ticker import FuncFormatter
    def millions_formatter(x, pos): return f'${x/1e6:.1f}M'
    ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Line chart for cumulative cash flow
    ax2 = ax1.twinx()
    ax2.plot(years, cumulative_cash_flows, color='blue', marker='o', linestyle='-', linewidth=2, label='Cumulative Cash Flow')
    ax2.set_ylabel('Cumulative Cash Flow ($)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(False)
    ax2.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Add payback period marker if available
    payback = tea_results.get('Simple Payback Period (Years)')
    if payback is not np.nan:
        payback_year_abs = payback + CONSTRUCTION_YEARS # Adjust to absolute year index for plotting
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax2.axvline(payback, color='purple', linestyle=':', linewidth=2, label=f'Payback: {payback:.1f} yrs')

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.title(f'{TARGET_ISO} - Annual and Cumulative Cash Flows', pad=20)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.savefig(plot_dir / f'{TARGET_ISO}_Cash_Flows.png', dpi=300)
    logging.info(f"Saved cash flow plot to {plot_dir}")
    plt.close(fig)

def plot_revenue_breakdown(annual_aggregates, plot_dir):
    """Plots the annual revenue breakdown as a pie chart."""
    if annual_aggregates is None:
        return
        
    revenue_sources = {
        'Energy': annual_aggregates.get('Annual Revenue Energy ($)', 0),
        'Ancillary Services': annual_aggregates.get('Annual Revenue AS ($)', 0),
        'Hydrogen': annual_aggregates.get('Annual Revenue H2 ($)', 0),
        # Add others if they exist
    }
    # Filter out zero-value sources
    revenue_data = {k: v for k, v in revenue_sources.items() if v > 1e-3}
    
    if not revenue_data:
        logging.warning("No significant revenue sources found to plot breakdown.")
        return
        
    labels = revenue_data.keys()
    sizes = revenue_data.values()
    total_revenue = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(sizes, autopct=lambda p: f'${p*total_revenue/100/1e6:.1f}M\n({p:.1f}%)',
                                      startangle=90, pctdistance=0.80)
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'{TARGET_ISO} - Annual Revenue Breakdown (Total: ${total_revenue/1e6:.2f}M)', pad=20)
    ax.legend(wedges, labels, title="Revenue Sources", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{TARGET_ISO}_Revenue_Breakdown.png', dpi=300)
    logging.info(f"Saved revenue breakdown plot to {plot_dir}")
    plt.close(fig)
    
# Add more plotting functions as needed (e.g., cost breakdown, hourly profiles)

# ==============================================================================
# --- MAIN EXECUTION --- 
# ==============================================================================
if __name__ == "__main__":
    logging.info("--- Starting TEA Analysis ---")
    logging.info(f"Target ISO: {TARGET_ISO}")
    logging.info(f"Hourly Results File: {HOURLY_RESULTS_FILE}")

    # Ensure output directories exist
    TEA_BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_hourly = load_hourly_results(HOURLY_RESULTS_FILE)

    if df_hourly is not None:
        # 2. Calculate Annual Aggregates
        annual_data = calculate_annual_aggregates(df_hourly)

        if annual_data is not None:
            # 3. Calculate TEA Metrics
            tea_results = calculate_tea_metrics(annual_data)

            if tea_results is not None:
                # 4. Save Summary Output
                with open(TEA_OUTPUT_FILE, 'w') as f:
                    f.write(f"--- Techno-Economic Analysis Summary for {TARGET_ISO} ---\n\n")
                    f.write("*** Configuration ***\n")
                    f.write(f"Project Lifetime: {PROJECT_LIFETIME_YEARS} years\n")
                    f.write(f"Discount Rate: {DISCOUNT_RATE*100:.1f}%\n")
                    f.write(f"Construction Period: {CONSTRUCTION_YEARS} years\n")
                    f.write(f"Total CAPEX: ${TOTAL_CAPEX:,.0f}\n")
                    f.write(f"Annual Fixed O&M: ${FIXED_OM_PER_YEAR:,.0f}\n")
                    f.write(f"Tax Rate: {TAX_RATE*100:.1f}%\n")
                    f.write(f"Replacement Costs: {REPLACEMENT_COSTS}\n\n")
                    
                    f.write("*** Annual Operational Performance (Based on Simulation Year) ***\n")
                    for key, value in annual_data.items():
                        if isinstance(value, (int, float)):
                             unit = "$" if "$" in key else ("kg" if "kg" in key else ("%" if "%" in key else "MW" if "MW" in key else ""))
                             f.write(f"{key}: {value:,.2f} {unit}\n")
                        else:
                             f.write(f"{key}: {value}\n") # Should not happen often
                    f.write("\n")

                    f.write("*** Key Financial Metrics ***\n")
                    f.write(f"NPV (Net Present Value): ${tea_results.get('NPV ($)', 'N/A'):,.0f}\n")
                    f.write(f"IRR (Internal Rate of Return): {tea_results.get('IRR (%)', 'N/A'):.2f}%\n")
                    f.write(f"Simple Payback Period: {tea_results.get('Simple Payback Period (Years)', 'N/A'):.2f} years\n")
                logging.info(f"TEA summary saved to {TEA_OUTPUT_FILE}")
                print(f"TEA summary saved to {TEA_OUTPUT_FILE}")

                # 5. Generate Plots
                plot_cash_flows(tea_results, PLOT_OUTPUT_DIR)
                plot_revenue_breakdown(annual_data, PLOT_OUTPUT_DIR)
                # Add calls to other plotting functions here
                logging.info(f"Plots saved to directory: {PLOT_OUTPUT_DIR}")
                print(f"Plots saved to directory: {PLOT_OUTPUT_DIR}")

            else:
                logging.error("Failed to calculate TEA metrics.")
        else:
            logging.error("Failed to calculate annual aggregates.")
    else:
        logging.error("Failed to load hourly results. Cannot proceed with TEA.")

    logging.info("--- TEA Analysis Finished ---")