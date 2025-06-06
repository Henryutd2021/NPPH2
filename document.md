TEA (Technical Economic Analysis) Framework Documentation
1. Framework Overview
This TEA (Technical Economic Analysis) framework is designed for a comprehensive economic assessment of nuclear-hydrogen production and related integrated energy systems. It can handle various analysis scenarios, including baseline operations of existing nuclear reactors, retrofit analysis with integrated hydrogen/battery storage systems, financial evaluation of incremental investments, and full lifecycle analysis of new-build nuclear-hydrogen integrated projects (including comparisons of different operational lifespans).

Core functionalities of the framework include:

Loading detailed technical and cost parameters from CSV files and Python configuration files.
Calculating annual performance metrics based on hourly system operation simulation results (typically from upstream optimization models).
Constructing detailed annual cash flows, considering initial investment (CAPEX), operational costs (OPEX), equipment replacement costs, taxes (including MACRS depreciation), and various tax credit policies (e.g., 45U, 45Y PTC, 48E ITC).
Calculating key financial indicators such as Net Present Value (NPV), Internal Rate of Return (IRR), and Payback Period.
Calculating levelized cost metrics, such as Levelized Cost of Hydrogen (LCOH), Levelized Cost of Electricity (LCOE), and Levelized Cost of Storage (LCOS), with detailed cost breakdown analysis.
Generating text reports and visualizations to summarize analysis results.
2. Core Modules and Their Responsibilities
The framework consists of multiple Python modules, each with specific responsibilities:

tea_engine.py:

Acts as the main engine or coordinator for TEA analysis.
Drives the entire analysis workflow, invoking functions from other modules based on the user-selected analysis case.
Integrates calculation results from different modules to form the final analysis report.
data_loader.py:

load_tea_sys_params(): Loads system-level parameters from sys_data_advanced.csv, such as project lifetime, discount rate, tax rate, hydrogen subsidies, user-specified equipment capacities, and parameters for tax incentive policies (e.g., PTC/ITC rates and durations). These parameters override or supplement default values in config.py.
load_hourly_results(): Loads hourly system operational data from hourly_results.csv (or other specified optimization result files), such as electricity prices, equipment power, and hydrogen production.
config.py:

Serves as a global configuration file, defining numerous default parameters and cost data structures.
Hydrogen/Battery System Costs:
CAPEX_COMPONENTS: Reference capacities, baseline costs, learning rates, and payment schedules for various hydrogen and battery components (electrolyzer, H2 storage, battery energy/power, grid integration).
OM_COMPONENTS: Fixed operational and maintenance (O&M) costs (as a percentage of CAPEX or per unit capacity) and inflation rates.
REPLACEMENT_SCHEDULE: Replacement cycles and costs for various components.
New-Build Nuclear Power Costs:
NUCLEAR_INTEGRATED_CONFIG: Global switch, default lifetime (e.g., 60 years), and construction period for new-build nuclear-hydrogen projects.
NUCLEAR_CAPEX_COMPONENTS: Detailed cost parameters (base cost, reference capacity, learning rate, payment schedule) for major sections of a new nuclear plant. The greenfield nuclear CAPEX calculation utilizes these detailed component costs.
NUCLEAR_COST_PARAMETERS: A centralized dictionary defining standardized nuclear power cost parameters, including:
nuclear_capex_per_mw: Can be used as a fallback or for simplified CAPEX estimation if detailed components are not used.
opex_parameters: For nuclear OPEX calculations in all scenarios (existing and new-build), covering fixed O&M, variable O&M, fuel costs, etc..
replacement_costs_per_mw: For component replacement costs in existing nuclear plants.
Taxation and Depreciation:
TAX_RATE: Corporate income tax rate.
MACRS_CONFIG: MACRS depreciation enablement switch, and depreciation periods and classifications for different asset types (nuclear, hydrogen, battery, grid).
TAX_INCENTIVE_PARAMETERS (New/Updated): A dictionary holding parameters for various tax incentives, such as PTC_45Y_RATE_USD_PER_MWH, PTC_45Y_DURATION_YEARS, ITC_48E_RATE_FRACTION, PTC_45U_RATE_USD_PER_MWH, PTC_45U_START_YEAR, PTC_45U_END_YEAR. These can be overridden by sys_data_advanced.csv.
Other General Parameters: Default values for PROJECT_LIFETIME_YEARS, DISCOUNT_RATE, CONSTRUCTION_YEARS.
calculations.py:

calculate_annual_metrics(): Calculates annual summary metrics from hourly operational data, such as annual profit, revenue, generation, hydrogen production, equipment utilization rates, and heat opportunity cost (HTE_Heat_Opportunity_Cost_Annual_USD). It uses config.NUCLEAR_COST_PARAMETERS to calculate the nuclear power portion of annual OPEX.
calculate_cash_flows(): Constructs the project's annual cash flows. It considers:
Initial CAPEX for hydrogen/battery systems and payment schedules, based on config.CAPEX_COMPONENTS and optimized/specified capacities.
Annual fixed O&M costs for hydrogen/battery systems, based on config.OM_COMPONENTS.
Equipment replacement costs for hydrogen/battery systems, based on config.REPLACEMENT_SCHEDULE.
Annual operating profit (from annual_metrics_results, which includes hourly variable costs and nuclear OPEX).
Hydrogen subsidy revenue (subsidy value and duration potentially loaded from sys_data_advanced.csv via tea_sys_params).
MACRS depreciation calculated by macrs.py, and subsequently, income tax.
calculate_financial_metrics(): Calculates NPV, IRR, Payback Period based on cash flows.
calculate_lcoh_breakdown(): Performs a detailed LCOH breakdown, including CAPEX, fixed O&M, electricity costs (using market average price for retrofit projects, or nuclear LCOE for greenfield), heat opportunity cost, variable O&M, and equipment replacement. It deducts ancillary service revenue associated with the hydrogen system (annual_hydrogen_as_revenue).
calculate_lcos_breakdown(): Performs a detailed LCOS breakdown for battery systems, similar to LCOH, including CAPEX, O&M, electricity charging costs, and replacements. It deducts ancillary service revenue associated with the battery system (annual_battery_as_revenue).
calculate_incremental_metrics(): Specifically calculates financial metrics for incremental investments (hydrogen and battery systems). It computes incremental CAPEX, OPEX, revenues, and considers MACRS depreciation for incremental assets. Electricity and heat opportunity costs are calculated. It calls calculate_lcoh_breakdown for incremental LCOH, passing the relevant incremental AS revenue.
nuclear_calculations.py:

calculate_nuclear_baseline_financial_analysis() (Case 1): Analyzes the baseline financial status of existing nuclear reactors. Uses NPPs info.csv for remaining lifetime and power factors, config.NUCLEAR_COST_PARAMETERS for nuclear OPEX and component replacement costs, and integrates 45U PTC benefits (rates and duration now configurable via config.TAX_INCENTIVE_PARAMETERS and sys_data_advanced.csv).
calculate_nuclear_integrated_financial_metrics() (Case 2): Analyzes existing nuclear plant retrofitted with H2/battery. Uses calculations.calculate_cash_flows for the H2/battery system and integrates nuclear OPEX. Incorporates 45U PTC benefits for the nuclear portion, with policy parameters being configurable.
calculate_greenfield_nuclear_hydrogen_system() (Core for Case 4/5 Baseline): Calculates economics for new-build nuclear-hydrogen integrated projects. Nuclear CAPEX is calculated using calculate_nuclear_capex_breakdown(use_detailed_components=True), leveraging config.NUCLEAR_CAPEX_COMPONENTS. Nuclear OPEX uses config.NUCLEAR_COST_PARAMETERS["opex_parameters"]. Hydrogen/battery costs come from config.py. This function uses an independent accounting method to calculate LCOE, LCOH, and LCOS, deducting corresponding system AS revenues in their respective calculations. Includes MACRS depreciation.
calculate_greenfield_nuclear_hydrogen_with_tax_incentives() (Case 4/5 with Tax): Builds upon calculate_greenfield_nuclear_hydrogen_system by invoking tax_incentives.py to analyze the impact of 45Y PTC and 48E ITC federal tax policies (policy parameters configurable via config.TAX_INCENTIVE_PARAMETERS and sys_data_advanced.csv) on greenfield project financial metrics.
calculate_lifecycle_comparison_analysis() (Case 5): Compares the financial performance of new-build nuclear-hydrogen integrated projects over different lifecycles (e.g., 60 vs. 80 years) by calling calculate_greenfield_nuclear_hydrogen_with_tax_incentives twice with different lifetime overrides.
calculate_45u_nuclear_ptc_benefits(): Calculates annual 45U PTC benefits for existing nuclear plants, using configurable rates and duration.
calculate_nuclear_capex_breakdown(): Calculates nuclear CAPEX, with an option to use detailed component costs from config.NUCLEAR_CAPEX_COMPONENTS (including learning rates) or a simplified $/MW approach.
macrs.py:

get_macrs_schedule(): Provides MACRS depreciation percentage tables for different periods (e.g., 7-year, 15-year).
calculate_total_macrs_depreciation(): Calculates annual MACRS depreciation amounts for the entire project or specific incremental parts, based on asset classifications and depreciation periods defined in config.MACRS_CONFIG and individual component CAPEX.
tax_incentives.py:

run_comprehensive_tax_incentive_analysis(): Core function that takes baseline cash flows, CAPEX breakdown, etc., and calculates adjusted cash flows and financial metrics under 45Y PTC and 48E ITC policies. Policy parameters (rates, durations) are now sourced from config.TAX_INCENTIVE_PARAMETERS (potentially overridden by sys_data_advanced.csv via plant_specific_params or tea_sys_params passed through the call stack).
calculate_45y_ptc_benefits(): Calculates annual benefits from the 45Y PTC using configurable rates and duration.
calculate_48e_itc_benefits(): Calculates the 48E ITC credit amount (using configurable rate) and handles its impact on reducing the depreciable basis by half.
reporting.py / summary_reporting.py / tax_incentive_reporting.py:

Responsible for generating text-formatted analysis reports.
generate_report() (from reporting.py) and generate_comprehensive_tea_summary_report() (from summary_reporting.py) consolidate calculation results from various modules and generate detailed TEA reports. summary_reporting.py provides a more structured Case 1-5 report.
generate_tax_incentive_comparative_report() specifically generates reports comparing financial results under different federal tax incentive policies.
visualization.py:

Provides functions to generate various charts, such as cash flow diagrams, cost breakdown pie/bar charts, LCOH dashboards, etc..
plot_results(): Coordinates calls to various plotting functions.
create_lcoh_comprehensive_dashboard(): Generates detailed LCOH composition and benchmarking charts.
create_tax_incentive_visualizations(): Generates comparative charts for federal tax incentive analysis.
utils.py:

setup_logging(): Configures the logging system.
3. Key Data Sources Explained
hourly_results.csv (Typical Annual Operational Data):

Source: Typically from upstream production simulation or optimization models.
Content: Hourly electricity prices (EnergyPrice_LMP_USDperMWh), total system profit (Profit_Hourly_USD), revenues (total, energy, ancillary, H2 sales, H2 subsidy), operational costs (Cost_HourlyOpex_Total_USD), H2 production (mHydrogenProduced_kg_hr), equipment power (e.g., pElectrolyzer_MW, pTurbine_MW, BatteryCharge_MW, BatteryDischarge_MW), storage levels (H2_Storage_Level_kg, BatterySOC_MWh), ancillary service provision and revenue components.
Usage: calculate_annual_metrics uses this for annual summaries. LCOH/LCOS electricity costs and AS revenue details are also derived from here.
sys_data_advanced.csv (System Parameters and Configuration):

Source: User inputs or predefined system parameters.
Content: Hydrogen system parameters (sales price, subsidy value & duration), user-specified capacities, financial parameters (lifetime, discount rate, tax rate, construction period), incremental analysis parameters, greenfield project parameters, and now also can include parameters for tax incentives (e.g., ptc_45y_rate_usd_per_mwh, itc_48e_rate_fraction) to override config.py defaults.
Usage: data_loader.py loads these, overriding config.py defaults. Used throughout calculations.
NPPs info.csv (Existing Nuclear Power Plant Information):

Source: Public data or user-provided data.
Content: Plant name, ISO, nameplate power factor, thermal capacity, summer capacity, remaining operational years.
Usage: calculate_nuclear_baseline_financial_analysis (Case 1) uses this, especially remaining and Nameplate Power Factor, to calibrate analysis for existing plants.
config.py (Global Configuration and Default Cost Library):

Source: Developer-predefined parameters.
Content:
H2/Battery System Costs: CAPEX_COMPONENTS, OM_COMPONENTS, REPLACEMENT_SCHEDULE.
New-Build Nuclear Costs: NUCLEAR_INTEGRATED_CONFIG, NUCLEAR_CAPEX_COMPONENTS (detailed component costs now primary for greenfield CAPEX calculation), NUCLEAR_COST_PARAMETERS (opex, fallback capex/MW, existing plant replacements).
Taxation and Depreciation: TAX_RATE, MACRS_CONFIG, and now TAX_INCENTIVE_PARAMETERS holding default rates and durations for 45U, 45Y, 48E, etc..
General Parameters: Defaults for PROJECT_LIFETIME_YEARS, DISCOUNT_RATE, etc..
Usage: Provides baseline parameters; sys_data_advanced.csv can override many of these.
4. Analysis Cases (Case Studies) Explained
The framework supports several predefined analysis cases:

Case 1: Existing Reactor Operations (Baseline Nuclear Operations)

Objective: Evaluate financial performance of an existing nuclear reactor.
Core Module: nuclear_calculations.py -> calculate_nuclear_baseline_financial_analysis.
Key Inputs: NPPs info.csv, hourly_results.csv (for market prices), config.NUCLEAR_COST_PARAMETERS (OPEX, replacements), config.TAX_INCENTIVE_PARAMETERS (for 45U details).
Taxation: Considers 45U PTC (details now configurable).
Output: NPV, IRR, Payback (with/without 45U PTC).
Case 2: Existing Reactor Retrofit (Retrofit with H2/Battery)

Objective: Evaluate overall economics of retrofitting an existing reactor with H2/battery systems.
Core Modules: tea_engine.py coordinating calculations.py and nuclear_calculations.py (calculate_nuclear_integrated_financial_metrics).
Key Inputs: NPPs info.csv, hourly_results.csv, config.py (H2/battery costs, nuclear OPEX from NUCLEAR_COST_PARAMETERS), sys_data_advanced.csv (H2 subsidy, tax incentive params for 45U).
Taxation: New H2/battery assets eligible for MACRS. Nuclear portion considers 45U PTC (details configurable).
Output: NPV, IRR, Payback for integrated system. LCOH (via calculate_lcoh_breakdown, electricity at market average, H2 system AS revenue deducted). LCOS (via calculate_lcos_breakdown, battery AS revenue deducted).
Case 3: Incremental Investment Financial Metrics (Incremental H2/Battery Analysis)

Objective: Evaluate ROI for new H2/battery systems, excluding existing nuclear plant's economics.
Core Module: calculations.py -> calculate_incremental_metrics.
Key Inputs: hourly_results.csv, config.py (H2/battery costs), sys_data_advanced.csv (H2 subsidy, market average electricity price for opportunity cost).
Taxation: Incremental assets eligible for MACRS.
Output: NPV, IRR, Payback for incremental project. Incremental LCOH calculated by calling calculate_lcoh_breakdown (electricity at market price, H2 system AS revenue deducted). Note: A separate LCOS for only the incremental battery part is not explicitly calculated by this function if an incremental battery is included; overall system LCOS is calculated if a battery exists.
Case 4: New-Build Nuclear-Hydrogen Integrated Project - 60yr Lifetime (Greenfield Nuclear-Hydrogen - 60yr)

Objective: Evaluate economics of a new nuclear plant integrated with H2 production (60-year life).
Core Module: nuclear_calculations.py -> calculate_greenfield_nuclear_hydrogen_with_tax_incentives (calls calculate_greenfield_nuclear_hydrogen_system and tax_incentives.py).
Key Inputs: hourly_results.csv, config.py (H2/battery costs; nuclear CAPEX from detailed NUCLEAR_CAPEX_COMPONENTS; nuclear OPEX from NUCLEAR_COST_PARAMETERS; MACRS; default tax incentive params from TAX_INCENTIVE_PARAMETERS), sys_data_advanced.csv (H2 subsidy, overrides for tax incentive params).
Taxation: All new assets eligible for MACRS. Compares: no tax incentives, 45Y PTC, 48E ITC (details configurable). 48E ITC reduces depreciable basis by half.
Output: NPV, IRR, Payback for each tax scenario. LCOE, LCOH, LCOS from calculate_greenfield_nuclear_hydrogen_system using independent accounting (LCOH electricity: NPP self-use at nuclear LCOE, grid at market; AS revenues deducted from respective costs).
Case 5: New-Build Nuclear-Hydrogen Integrated Project - 80yr vs. 60yr Lifetime (Greenfield Nuclear-Hydrogen - 80yr vs 60yr)

Objective: Compare impact of 80-year vs. 60-year lifetime on new-build project economics.
Core Module: nuclear_calculations.py -> calculate_lifecycle_comparison_analysis.
Implementation: Calls calculate_greenfield_nuclear_hydrogen_with_tax_incentives twice (once default 60yr, once override 80yr). Replacement counts adjusted for longer life.
Taxation & LCOx: Same logic as Case 4, applied to respective lifetimes.
5. Core Calculation Logic
Annual Metrics (calculate_annual_metrics):

Aggregates hourly data to annual sums/averages (revenue, OPEX from optimization, H2 production, generation).
Calculates equipment utilization/capacity factors.
Calculates HTE_Heat_Opportunity_Cost_Annual_USD = (Annual Steam Consumption / Nuclear Thermal Efficiency) * Avg Market Electricity Price.
Calculates standardized annual nuclear OPEX using config.NUCLEAR_COST_PARAMETERS.
Cash Flow (calculate_cash_flows, calculate_incremental_metrics, calculate_greenfield_nuclear_hydrogen_system):

Construction Period: Negative cash flow from CAPEX, distributed per payment_schedule_years (for H2/Battery in calculate_cash_flows; nuclear CAPEX in calculate_greenfield_nuclear_hydrogen_system currently spread evenly but uses detailed component costs for total).
Operational Period:
Revenue: H2 sales, electricity sales, ancillary services, H2 subsidies (limited duration).
Costs:
Variable O&M (VOM): From hourly_results (e.g., electrolyzer, battery, water).
Fixed O&M (FOM): From config.OM_COMPONENTS (H2/Battery) and config.NUCLEAR_COST_PARAMETERS (Nuclear), inflation-adjusted.
Equipment Replacements: From config.REPLACEMENT_SCHEDULE (H2/Battery) and config.NUCLEAR_COST_PARAMETERS (Existing Nuclear) or specific logic in greenfield (Nuclear).
Heat Opportunity Cost (HTE): From annual metrics.
Profit Before Tax (PBT): Revenue - Costs.
MACRS Depreciation: Annual amount from macrs.py.
Taxable Income: PBT - MACRS Depreciation.
Income Tax: Taxable Income * TAX_RATE (if Taxable Income > 0).
Cash Flow After Tax: PBT - Income Tax (MACRS is non-cash but reduces tax).
Federal Tax Credits (PTC/ITC):
PTC (45U, 45Y): Directly increases after-tax cash flow.
ITC (48E): One-time credit increases early cash flow; reduces depreciable basis by half the ITC amount.
Financial Metrics (calculate_financial_metrics):

NPV: npf.npv(discount_rate, cash_flows_array).
IRR: npf.irr(cash_flows_array).
Payback Period: Year cumulative cash flow turns positive.
LCOH (calculate_lcoh_breakdown, calculate_greenfield_nuclear_hydrogen_system):

General Formula: LCOH = PV(Total H2 System Costs - H2 System AS Revenue) / PV(Total Hydrogen Production).
Cost Components (Present Value):
PV of H2 system initial CAPEX and all its replacement costs.
PV of annual H2 system fixed O&M.
PV of annual H2 system variable O&M.
PV of annual electricity costs for H2 production:
Retrofit/Incremental (Cases 2/3): Market average electricity price * consumption.
Greenfield (Cases 4/5): Nuclear LCOE * NPP self-consumption + Market Price * Grid Purchases.
PV of annual heat opportunity costs (for HTE).
AS Revenue Deduction: PV of annual AS revenues specifically attributed to the H2 system is deducted from the total H2 system costs.
LCOE (Nuclear, calculate_greenfield_nuclear_hydrogen_system):

LCOE = PV(Total Nuclear Costs - Nuclear AS Revenue) / PV(Total Nuclear Generation).
Total Nuclear Costs include PV of nuclear CAPEX and lifecycle nuclear OPEX.
LCOS (Battery, calculate_lcos_breakdown, calculate_greenfield_nuclear_hydrogen_system):

LCOS = PV(Total Battery System Costs - Battery AS Revenue) / PV(Total Battery Discharged Energy/Throughput).
Total Battery System Costs include PV of battery CAPEX, replacement costs, and lifecycle OPEX (including charging electricity costs).
Charging electricity cost for LCOS:
Retrofit/Incremental (Cases 2/3): Market average electricity price * consumption.
Greenfield (Cases 4/5): Nuclear LCOE * NPP self-consumption for charging + Market Price * Grid Purchases for charging.
6. Tax Incentive Policy Handling
Policy parameters (rates, durations) for 45U, 45Y, and 48E are primarily defined in config.py under TAX_INCENTIVE_PARAMETERS and can be overridden by values loaded from sys_data_advanced.csv.

MACRS (Modified Accelerated Cost Recovery System):

Handled by macrs.py.
config.MACRS_CONFIG defines depreciation periods and classifications (e.g., nuclear-15yr, hydrogen-7yr, battery-7yr, grid-15yr).
calculate_total_macrs_depreciation computes annual depreciation, reducing taxable income.
45U PTC (Production Tax Credit for Existing Nuclear):

Calculated by nuclear_calculations.py -> calculate_45u_nuclear_ptc_benefits using configurable parameters.
Applied in Case 1 (Baseline Nuclear) and Case 2 (Retrofit).
Directly increases after-tax cash flow.
45Y PTC (Clean Electricity Production Credit - for new zero-emission facilities):

Calculated by tax_incentives.py -> calculate_45y_ptc_benefits using configurable parameters.
Applied in tax scenarios for Case 4 and Case 5 (Greenfield).
Directly increases after-tax cash flow.
48E ITC (Clean Electricity Investment Credit - for new zero-emission facilities & energy storage):

Calculated by tax_incentives.py -> calculate_48e_itc_benefits using configurable parameters.
Applied in tax scenarios for Case 4 and Case 5 (Greenfield).
Provides a one-time investment credit; reduces depreciable basis for MACRS by half the ITC amount.
7. Reporting and Visualization
Reporting:
reporting.py and summary_reporting.py consolidate results into structured text reports. summary_reporting.py provides a Case 1-5 structure.
tax_incentive_reporting.py generates specific comparative reports for federal tax incentive policies.
Visualization:
visualization.py uses Matplotlib/Seaborn for charts (cash flows, cost breakdowns, LCOH dashboards, tax impact comparisons).