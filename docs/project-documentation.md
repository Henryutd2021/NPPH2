# TEA (Techno-Economic Analysis) Framework Documentation

## 1. Framework Overview

This Techno-Economic Analysis (TEA) framework is designed for a comprehensive economic assessment of nuclear-hydrogen production and related integrated energy systems. It supports various analysis scenarios, including baseline operations of existing nuclear reactors, retrofit analysis with integrated hydrogen/battery storage systems, financial evaluation of incremental investments, and full lifecycle analysis of new-build nuclear-hydrogen integrated projects (including comparisons of different operational lifespans).

Core functionalities of the framework include:

- Loading detailed technical and cost parameters from CSV and Python configuration files.
- Calculating annual performance metrics from hourly system operation simulation results (typically from upstream optimization models).
- Constructing detailed annual cash flows, considering initial investment (CAPEX), operational costs (OPEX), equipment replacement costs, taxes (including MACRS depreciation), and various tax credit policies (e.g., 45U, 45Y PTC, 48E ITC).
- Calculating key financial indicators such as Net Present Value (NPV), Internal Rate of Return (IRR), and Payback Period.
- Calculating levelized cost metrics like Levelized Cost of Hydrogen (LCOH), Levelized Cost of Electricity (LCOE), and Levelized Cost of Storage (LCOS), with detailed cost breakdown analysis.
- Generating text reports and visualizations to summarize analysis results.

## 2. Core Modules and Their Responsibilities

The framework consists of multiple Python modules, each with specific responsibilities:

- **`tea_engine.py`**:
  - Acts as the main engine for TEA analysis.
  - Drives the analysis workflow, invoking functions from other modules based on the selected analysis case.
  - Integrates calculation results to form the final analysis report.

- **`data_loader.py`**:
  - `load_tea_sys_params()`: Loads system-level parameters from `sys_data_advanced.csv`.
  - `load_hourly_results()`: Loads hourly operational data from optimization result files (e.g., `hourly_results.csv`).

- **`config.py`**:
  - Serves as a global configuration file with default parameters and cost data.
  - **Hydrogen/Battery System Costs**: `CAPEX_COMPONENTS`, `OM_COMPONENTS`, `REPLACEMENT_SCHEDULE`.
  - **New-Build Nuclear Power Costs**: `NUCLEAR_INTEGRATED_CONFIG`, `NUCLEAR_CAPEX_COMPONENTS`, `NUCLEAR_COST_PARAMETERS`.
  - **Taxation and Depreciation**: `TAX_RATE`, `MACRS_CONFIG`.
  - **Tax Incentives**: `TAX_INCENTIVE_PARAMETERS` for policies like 45U, 45Y PTC, and 48E ITC.

- **`calculations.py`**:
  - `calculate_annual_metrics()`: Calculates annual summary metrics from hourly data.
  - `calculate_cash_flows()`: Constructs the project's annual cash flows.
  - `calculate_financial_metrics()`: Calculates NPV, IRR, and Payback Period.
  - `calculate_lcoh_breakdown()`: Performs a detailed LCOH breakdown.
  - `calculate_lcos_breakdown()`: Performs a detailed LCOS breakdown for battery systems.
  - `calculate_incremental_metrics()`: Calculates financial metrics for incremental investments.

- **`nuclear_calculations.py`**:
  - `calculate_nuclear_baseline_financial_analysis()` (Case 1): Analyzes the financial status of existing nuclear reactors.
  - `calculate_nuclear_integrated_financial_metrics()` (Case 2): Analyzes existing nuclear plants retrofitted with H2/battery systems.
  - `calculate_greenfield_nuclear_hydrogen_system()` (Core for Case 4/5 Baseline): Calculates economics for new-build integrated projects.
  - `calculate_greenfield_nuclear_hydrogen_with_tax_incentives()` (Case 4/5 with Tax): Analyzes the impact of federal tax policies on greenfield projects.
  - `calculate_lifecycle_comparison_analysis()` (Case 5): Compares financial performance over different project lifecycles.
  - `calculate_45u_nuclear_ptc_benefits()`: Calculates annual 45U PTC benefits.
  - `calculate_nuclear_capex_breakdown()`: Calculates nuclear CAPEX.

- **`macrs.py`**:
  - `get_macrs_schedule()`: Provides MACRS depreciation percentage tables.
  - `calculate_total_macrs_depreciation()`: Calculates annual MACRS depreciation.

- **`tax_incentives.py`**:
  - `run_comprehensive_tax_incentive_analysis()`: Calculates adjusted cash flows and financial metrics under 45Y PTC and 48E ITC policies.
  - `calculate_45y_ptc_benefits()`: Calculates annual benefits from the 45Y PTC.
  - `calculate_48e_itc_benefits()`: Calculates the 48E ITC credit amount.

- **`reporting.py`, `summary_reporting.py`, `tax_incentive_reporting.py`**:
  - Responsible for generating formatted text analysis reports.

- **`visualization.py`**:
  - Provides functions to generate charts and dashboards.

- **`utils.py`**:
  - `setup_logging()`: Configures the logging system.

## 3. Key Data Sources Explained

- **`hourly_results.csv` (Annual Operational Data)**:
  - **Source**: Upstream optimization models.
  - **Content**: Hourly electricity prices, system profit, revenues, costs, H2 production, equipment power, storage levels, and ancillary service data.

- **`sys_data_advanced.csv` (System Parameters and Configuration)**:
  - **Source**: User inputs or predefined system parameters.
  - **Content**: Hydrogen system parameters, user-specified capacities, financial parameters, and overrides for tax incentive policies.

- **`NPPs info.csv` (Existing Nuclear Power Plant Information)**:
  - **Source**: Public or user-provided data.
  - **Content**: Plant name, ISO, capacity factors, thermal capacity, and remaining operational years.

- **`config.py` (Global Configuration and Default Cost Library)**:
  - **Source**: Developer-predefined parameters.
  - **Content**: Default costs for H2/Battery systems, new-build nuclear components, tax policies, and general financial parameters.

## 4. Analysis Cases (Case Studies) Explained

The framework supports several predefined analysis cases:

- **Case 1: Existing Reactor Operations (Baseline)**
  - **Objective**: Evaluate the financial performance of an existing nuclear reactor.
  - **Core Module**: `nuclear_calculations.py` -> `calculate_nuclear_baseline_financial_analysis`.

- **Case 2: Existing Reactor Retrofit (H2/Battery)**
  - **Objective**: Evaluate the economics of retrofitting an existing reactor with H2/battery systems.
  - **Core Modules**: `tea_engine.py` coordinating `calculations.py` and `nuclear_calculations.py`.

- **Case 3: Incremental Investment Financial Metrics**
  - **Objective**: Evaluate the ROI for new H2/battery systems, excluding the existing nuclear plant's economics.
  - **Core Module**: `calculations.py` -> `calculate_incremental_metrics`.

- **Case 4: New-Build Nuclear-Hydrogen Integrated Project (60yr Lifetime)**
  - **Objective**: Evaluate the economics of a new nuclear plant integrated with H2 production over a 60-year life.
  - **Core Module**: `nuclear_calculations.py` -> `calculate_greenfield_nuclear_hydrogen_with_tax_incentives`.

- **Case 5: New-Build Nuclear-Hydrogen Integrated Project (80yr vs. 60yr Lifetime)**
  - **Objective**: Compare the impact of an 80-year vs. 60-year lifetime on new-build project economics.
  - **Core Module**: `nuclear_calculations.py` -> `calculate_lifecycle_comparison_analysis`.

## 5. Core Calculation Logic

- **Annual Metrics (`calculate_annual_metrics`)**: Aggregates hourly data to annual summaries and calculates key performance indicators.
- **Cash Flow (`calculate_cash_flows`, etc.)**:
  - **Construction Period**: Negative cash flow from CAPEX.
  - **Operational Period**:
    - **Revenue**: H2 sales, electricity sales, ancillary services, subsidies.
    - **Costs**: VOM, FOM, replacements, heat opportunity cost.
    - **Taxes**: Calculated after MACRS depreciation.
    - **Credits**: PTC/ITC are applied to cash flow.
- **Financial Metrics (`calculate_financial_metrics`)**: Standard NPV, IRR, and Payback Period calculations.
- **LCOH (`calculate_lcoh_breakdown`, etc.)**:
  - **Formula**: `LCOH = PV(Total Costs - AS Revenue) / PV(Total H2 Production)`
  - **Cost Components**: CAPEX, O&M, electricity, heat opportunity cost, replacements.
- **LCOE (Nuclear)**:
  - **Formula**: `LCOE = PV(Total Nuclear Costs - Nuclear AS Revenue) / PV(Total Nuclear Generation)`
- **LCOS (Battery)**:
  - **Formula**: `LCOS = PV(Total Battery Costs - Battery AS Revenue) / PV(Total Battery Discharged Energy)`

## 6. Tax Incentive Policy Handling

- **Configuration**: Policy parameters (rates, durations) are defined in `config.py` and can be overridden by `sys_data_advanced.csv`.
- **MACRS**: Handled by `macrs.py` based on asset classifications in `config.MACRS_CONFIG`.
- **45U PTC (Existing Nuclear)**: Applied in Cases 1 and 2.
- **45Y PTC (New Clean Electricity)**: Applied in tax scenarios for Cases 4 and 5.
- **48E ITC (New Clean Investment)**: Applied in tax scenarios for Cases 4 and 5; reduces the depreciable basis for MACRS.

## 7. Reporting and Visualization

- **Reporting**: `reporting.py` and `summary_reporting.py` generate structured text reports. `tax_incentive_reporting.py` creates comparative reports for tax policies.
- **Visualization**: `visualization.py` uses Matplotlib/Seaborn to create charts, dashboards, and other visualizations.
