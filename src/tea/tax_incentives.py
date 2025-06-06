"""
Federal Tax Incentive Calculations for Nuclear-Hydrogen Integrated Systems.

This module implements comprehensive calculations for federal tax incentive scenarios:
- Scenario A: 45Y Production Tax Credit (PTC)
- Scenario B: 48E Investment Tax Credit (ITC)

Both scenarios are compared against a baseline scenario (no incentives) to evaluate
the impact of tax incentives on project economics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import numpy_financial as npf

# Import MACRS depreciation functions
from src.tea.macrs import calculate_total_macrs_depreciation

logger = logging.getLogger(__name__)

# Constants
HOURS_IN_YEAR = 8760


class TaxIncentiveAnalyzer:
    """Comprehensive tax incentive analyzer for nuclear-hydrogen systems."""

    def __init__(self,
                 project_lifetime_years: int,
                 construction_period_years: int,
                 discount_rate: float,
                 tax_rate: float):
        """
        Initialize the tax incentive analyzer.

        Args:
            project_lifetime_years: Total project operational lifetime
            construction_period_years: Construction period duration
            discount_rate: Discount rate for NPV calculations
            tax_rate: Corporate tax rate
        """
        self.project_lifetime_years = project_lifetime_years
        self.construction_period_years = construction_period_years
        self.discount_rate = discount_rate
        self.tax_rate = tax_rate
        self.total_years = construction_period_years + project_lifetime_years

        logger.info(f"Initialized Tax Incentive Analyzer:")
        logger.info(f"  Project Lifetime: {project_lifetime_years} years")
        logger.info(
            f"  Construction Period: {construction_period_years} years")
        logger.info(f"  Discount Rate: {discount_rate:.1%}")
        logger.info(f"  Tax Rate: {tax_rate:.1%}")


def calculate_45y_ptc_benefits(
    hourly_results_df: pd.DataFrame,
    ptc_rate_usd_per_mwh: float = None,
    ptc_duration_years: int = None,
    project_lifetime_years: int = 60,
    construction_period_years: int = 8,
    plant_specific_params: Dict = None,
    tax_policies: Dict = None
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate 45Y Production Tax Credit benefits based on actual hourly generation data.

    Args:
        hourly_results_df: DataFrame with hourly optimization results
        ptc_rate_usd_per_mwh: PTC rate in $/MWh (if None, uses config default)
        ptc_duration_years: Duration of PTC eligibility (if None, uses config default)
        project_lifetime_years: Total project lifetime
        construction_period_years: Construction period
        plant_specific_params: Plant-specific parameters
        tax_policies: Tax incentive policies configuration dictionary

    Returns:
        Tuple of (annual_ptc_benefits_array, ptc_analysis_dict)
    """

    # Load default values from tax policies configuration if not provided
    if tax_policies is None:
        try:
            from src.tea.config import TAX_INCENTIVE_POLICIES
            tax_policies = TAX_INCENTIVE_POLICIES
        except ImportError:
            logger.warning("Could not import TAX_INCENTIVE_POLICIES, using hardcoded defaults")
            tax_policies = {
                "45y_ptc": {
                    "credit_rate_per_mwh": 30.0,
                    "credit_duration_years": 10
                }
            }

    # Use configuration values if parameters not explicitly provided
    if ptc_rate_usd_per_mwh is None:
        ptc_rate_usd_per_mwh = tax_policies.get("45y_ptc", {}).get("credit_rate_per_mwh", 30.0)
    if ptc_duration_years is None:
        ptc_duration_years = tax_policies.get("45y_ptc", {}).get("credit_duration_years", 10)
    logger.info("Calculating 45Y Production Tax Credit (PTC) benefits...")

    total_years = construction_period_years + project_lifetime_years
    annual_ptc_benefits = np.zeros(total_years)

    # Extract electricity generation data from hourly results
    # Look for total electricity generation (both grid sales and hydrogen production)
    electricity_gen_cols = [
        'pTurbine_MW',  # Direct turbine power output - most common in hourly results
        'Nuclear_Generation_MWh_hr',
        'Total_Electricity_Generation_MWh_hr',
        'Power_Generation_MWh_hr',
        'Electricity_Generation_Total_MWh_hr'
    ]

    generation_col = None
    annual_generation_mwh = 0

    for col in electricity_gen_cols:
        if col in hourly_results_df.columns:
            generation_col = col
            if col == 'pTurbine_MW':
                # Convert MW to MWh (sum hourly MW values to get annual MWh)
                annual_generation_mwh = hourly_results_df[col].sum()
                logger.info(
                    f"Annual electricity generation from {col}: {annual_generation_mwh:,.0f} MWh")
            else:
                # For MWh_hr columns, sum directly
                annual_generation_mwh = hourly_results_df[col].sum()
                logger.info(
                    f"Annual electricity generation from {col}: {annual_generation_mwh:,.0f} MWh")
            break

    if generation_col is None:
        logger.warning(
            "No electricity generation column found in hourly results. Using fallback calculation.")
        # Fallback: estimate from nuclear capacity if available
        try:
            # First priority: check plant_specific_params for capacity
            nuclear_capacity_mw = None
            if plant_specific_params:
                capacity_keys = ['nameplate_capacity_mw',
                                 'pTurbine_max_MW', 'turbine_capacity_mw']
                for key in capacity_keys:
                    if key in plant_specific_params and plant_specific_params[key]:
                        nuclear_capacity_mw = float(plant_specific_params[key])
                        logger.info(
                            f"Using nuclear capacity from plant_specific_params[{key}]: {nuclear_capacity_mw} MW")
                        break

            # Second priority: check for various capacity columns in hourly results
            if nuclear_capacity_mw is None:
                capacity_cols = ['Nuclear_Capacity_MW', 'Turbine_Capacity_MW',
                                 'nameplate_capacity_mw', 'pTurbine_max_MW']
                for cap_col in capacity_cols:
                    if cap_col in hourly_results_df.columns:
                        nuclear_capacity_mw = hourly_results_df[cap_col].iloc[0]
                        logger.info(
                            f"Using nuclear capacity from {cap_col}: {nuclear_capacity_mw} MW")
                        break

            if nuclear_capacity_mw is None:
                # Use a default nuclear capacity if no data available
                nuclear_capacity_mw = 1000  # MW - typical small modular reactor
                logger.warning(
                    f"No nuclear capacity data found. Using default {nuclear_capacity_mw} MW")

            capacity_factor = 0.9  # Typical nuclear capacity factor
            annual_generation_mwh = nuclear_capacity_mw * 8760 * \
                capacity_factor  # Use explicit HOURS_IN_YEAR
            logger.info(
                f"Estimated annual generation: {annual_generation_mwh:,.0f} MWh (fallback calculation)")
        except Exception as e:
            logger.error(f"Error in fallback calculation: {e}")
            # Last resort: use a reasonable default
            annual_generation_mwh = 7884000  # 1000 MW * 8760 hr * 0.9 CF
            logger.warning(
                f"Using default annual generation: {annual_generation_mwh:,.0f} MWh")
    else:
        # Calculate annual generation from hourly data
        annual_generation_mwh = hourly_results_df[generation_col].sum()
        logger.info(
            f"Annual electricity generation from {generation_col}: {annual_generation_mwh:,.0f} MWh")

    # Calculate annual PTC benefits
    annual_ptc_benefit = annual_generation_mwh * ptc_rate_usd_per_mwh

    # Apply PTC benefits for eligible years (starting from first operational year)
    ptc_start_year = construction_period_years
    ptc_end_year = min(ptc_start_year + ptc_duration_years, total_years)

    for year in range(ptc_start_year, ptc_end_year):
        annual_ptc_benefits[year] = annual_ptc_benefit

    # Calculate total PTC value
    total_ptc_value = np.sum(annual_ptc_benefits)
    total_ptc_npv = np.sum([
        annual_ptc_benefits[year] / (1 + 0.08) ** year
        for year in range(total_years)
    ])

    ptc_analysis = {
        "ptc_rate_usd_per_mwh": ptc_rate_usd_per_mwh,
        "ptc_duration_years": ptc_duration_years,
        "annual_generation_mwh": annual_generation_mwh,
        "annual_ptc_benefit_usd": annual_ptc_benefit,
        "total_ptc_value_usd": total_ptc_value,
        "total_ptc_npv_usd": total_ptc_npv,
        "ptc_eligible_years": list(range(ptc_start_year, ptc_end_year)),
        "generation_data_source": generation_col or "estimated"
    }

    logger.info(f"PTC Analysis Results:")
    logger.info(f"  Annual Generation: {annual_generation_mwh:,.0f} MWh")
    logger.info(f"  Annual PTC Benefit: ${annual_ptc_benefit:,.0f}")
    logger.info(f"  Total PTC Value: ${total_ptc_value:,.0f}")
    logger.info(f"  Total PTC NPV: ${total_ptc_npv:,.0f}")

    return annual_ptc_benefits, ptc_analysis


def calculate_48e_itc_benefits(
    capex_breakdown: Dict[str, float],
    itc_rate: float = None,
    project_lifetime_years: int = 60,
    construction_period_years: int = 8,
    tax_policies: Dict = None
) -> Tuple[float, float, Dict]:
    """
    Calculate 48E Investment Tax Credit benefits based on qualified capital expenditures.

    The 48E ITC is specifically for nuclear power facilities that generate electricity.
    Only nuclear power equipment qualifies for this credit. Hydrogen production equipment
    (electrolyzers, hydrogen storage) and battery storage equipment are NOT eligible.

    Args:
        capex_breakdown: Dictionary of component CAPEX values
        itc_rate: ITC rate (if None, uses config default)
        project_lifetime_years: Total project lifetime
        construction_period_years: Construction period
        tax_policies: Tax incentive policies configuration dictionary

    Returns:
        Tuple of (itc_credit_amount, reduced_depreciation_basis, itc_analysis_dict)
    """

    # Load default values from tax policies configuration if not provided
    if tax_policies is None:
        try:
            from src.tea.config import TAX_INCENTIVE_POLICIES
            tax_policies = TAX_INCENTIVE_POLICIES
        except ImportError:
            logger.warning("Could not import TAX_INCENTIVE_POLICIES, using hardcoded defaults")
            tax_policies = {
                "48e_itc": {
                    "credit_rate": 0.50,
                    "depreciation_basis_reduction_rate": 0.50
                }
            }

    # Use configuration values if parameters not explicitly provided
    if itc_rate is None:
        itc_rate = tax_policies.get("48e_itc", {}).get("credit_rate", 0.50)
    logger.info("Calculating 48E Investment Tax Credit (ITC) benefits...")

    # Define eligible equipment categories for 48E ITC
    # 48E ITC is specifically for nuclear power facilities that generate electricity
    # Hydrogen production equipment is NOT eligible for 48E ITC
    eligible_components = {
        "Nuclear_Power_Plant": 1.0,  # 100% eligible - nuclear electricity generation
        "Nuclear_Island": 1.0,  # 100% eligible - nuclear electricity generation
        "Nuclear_Site_Preparation": 1.0,  # 100% eligible - nuclear electricity generation
        "Nuclear_Safety_Systems": 1.0,  # 100% eligible - nuclear electricity generation
        "Nuclear_Grid_Connection": 1.0,  # 100% eligible - nuclear electricity generation
        "Electrolyzer_System": 0.0,  # NOT eligible for 48E - hydrogen production equipment
        "H2_Storage_System": 0.0,  # NOT eligible for 48E - hydrogen storage equipment
        # 80% eligible (electrical infrastructure for nuclear)
        "Grid_Integration": 0.8,
        "Battery_System_Energy": 0.0,  # Not eligible for 48E - energy storage
        "Battery_System_Power": 0.0,  # Not eligible for 48E - energy storage
        "NPP_Modifications": 0.0,  # 100% eligible - nuclear power plant modifications
    }

    # Calculate qualified CAPEX
    total_qualified_capex = 0
    component_qualified_capex = {}

    for component, capex_value in capex_breakdown.items():
        if component in eligible_components:
            qualified_amount = capex_value * eligible_components[component]
            component_qualified_capex[component] = qualified_amount
            total_qualified_capex += qualified_amount

            if qualified_amount > 0:
                logger.debug(
                    f"  {component}: ${capex_value:,.0f} * {eligible_components[component]:.0%} = ${qualified_amount:,.0f}")
        else:
            logger.debug(
                f"  {component}: ${capex_value:,.0f} (not qualified for ITC)")

    # Calculate ITC credit amount
    itc_credit_amount = total_qualified_capex * itc_rate

    # Calculate reduced depreciation basis (IRS requirement)
    # Depreciation basis must be reduced by 50% of the ITC amount claimed
    depreciation_basis_reduction = itc_credit_amount * 0.50

    itc_analysis = {
        "itc_rate": itc_rate,
        "total_qualified_capex_usd": total_qualified_capex,
        "component_qualified_capex": component_qualified_capex,
        "itc_credit_amount_usd": itc_credit_amount,
        "depreciation_basis_reduction_usd": depreciation_basis_reduction,
        "eligible_components": eligible_components
    }

    logger.info(f"ITC Analysis Results:")
    logger.info(f"  Total Qualified CAPEX: ${total_qualified_capex:,.0f}")
    logger.info(f"  ITC Credit Amount: ${itc_credit_amount:,.0f}")
    logger.info(
        f"  Depreciation Basis Reduction: ${depreciation_basis_reduction:,.0f}")

    return itc_credit_amount, depreciation_basis_reduction, itc_analysis


def calculate_scenario_cash_flows(
    base_cash_flows: np.ndarray,
    annual_metrics: Dict,
    capex_breakdown: Dict,
    scenario_type: str,
    hourly_results_df: pd.DataFrame = None,
    macrs_config: Dict = None,
    tax_rate: float = 0.21,
    project_lifetime_years: int = 60,
    construction_period_years: int = 8,
    plant_specific_params: Dict = None
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate cash flows for a specific tax incentive scenario.

    Args:
        base_cash_flows: Baseline cash flows without tax incentives
        annual_metrics: Dictionary of annual financial metrics
        capex_breakdown: Dictionary of component CAPEX values
        scenario_type: 'baseline', 'ptc', or 'itc'
        hourly_results_df: Hourly optimization results (required for PTC)
        macrs_config: MACRS configuration (required for ITC)
        tax_rate: Corporate tax rate
        project_lifetime_years: Total project lifetime
        construction_period_years: Construction period
        plant_specific_params: Plant-specific parameters

    Returns:
        Tuple of (scenario_cash_flows, scenario_analysis_dict)
    """
    logger.info(
        f"Calculating cash flows for {scenario_type.upper()} scenario...")

    # Start with baseline cash flows
    scenario_cash_flows = base_cash_flows.copy()
    total_years = len(scenario_cash_flows)

    scenario_analysis = {
        "scenario_type": scenario_type,
        "base_cash_flows_total": np.sum(base_cash_flows),
        "tax_benefits": {}
    }

    if scenario_type == "baseline":
        # No additional tax incentives
        scenario_analysis["description"] = "Baseline scenario without federal tax incentives"

    elif scenario_type == "ptc":
        # Apply 45Y Production Tax Credit
        if hourly_results_df is None:
            raise ValueError(
                "Hourly results data required for PTC scenario analysis")

        ptc_benefits, ptc_analysis = calculate_45y_ptc_benefits(
            hourly_results_df=hourly_results_df,
            project_lifetime_years=project_lifetime_years,
            construction_period_years=construction_period_years,
            plant_specific_params=plant_specific_params
        )

        # Add PTC benefits to cash flows
        scenario_cash_flows += ptc_benefits

        scenario_analysis[
            "description"] = "45Y Production Tax Credit scenario ($30/MWh for 10 years)"
        scenario_analysis["tax_benefits"]["ptc"] = ptc_analysis
        scenario_analysis["total_ptc_benefits"] = np.sum(ptc_benefits)

    elif scenario_type == "itc":
        # Apply 48E Investment Tax Credit
        if macrs_config is None:
            raise ValueError(
                "MACRS configuration required for ITC scenario analysis")

        itc_credit, depreciation_reduction, itc_analysis = calculate_48e_itc_benefits(
            capex_breakdown=capex_breakdown,
            project_lifetime_years=project_lifetime_years,
            construction_period_years=construction_period_years
        )

        # Apply ITC credit in first operational year
        first_operational_year = construction_period_years
        if first_operational_year < total_years:
            scenario_cash_flows[first_operational_year] += itc_credit

        # Calculate adjusted MACRS depreciation with reduced basis
        adjusted_capex_breakdown = capex_breakdown.copy()
        for component in adjusted_capex_breakdown:
            if component in itc_analysis["component_qualified_capex"]:
                # Reduce depreciation basis by 50% of allocated ITC
                component_itc = itc_analysis["component_qualified_capex"][component] * 0.50
                component_reduction = component_itc * 0.50
                adjusted_capex_breakdown[component] -= component_reduction

        # Recalculate MACRS depreciation with reduced basis
        adjusted_depreciation, _ = calculate_total_macrs_depreciation(
            capex_breakdown=adjusted_capex_breakdown,
            construction_period_years=construction_period_years,
            project_lifetime_years=project_lifetime_years,
            macrs_config=macrs_config
        )

        # Calculate original MACRS depreciation
        original_depreciation, _ = calculate_total_macrs_depreciation(
            capex_breakdown=capex_breakdown,
            construction_period_years=construction_period_years,
            project_lifetime_years=project_lifetime_years,
            macrs_config=macrs_config
        )

        # Calculate the difference in tax benefits from reduced depreciation
        depreciation_tax_impact = (
            original_depreciation - adjusted_depreciation) * tax_rate
        # Reduce cash flows due to lower depreciation
        scenario_cash_flows -= depreciation_tax_impact

        scenario_analysis[
            "description"] = "48E Investment Tax Credit scenario (50% of qualified CAPEX)"
        scenario_analysis["tax_benefits"]["itc"] = itc_analysis
        scenario_analysis["total_itc_credit"] = itc_credit
        scenario_analysis["depreciation_impact"] = np.sum(
            depreciation_tax_impact)
        scenario_analysis["net_itc_benefit"] = itc_credit - \
            np.sum(depreciation_tax_impact)

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    scenario_analysis["scenario_cash_flows_total"] = np.sum(
        scenario_cash_flows)
    scenario_analysis["incremental_value"] = scenario_analysis["scenario_cash_flows_total"] - \
        scenario_analysis["base_cash_flows_total"]

    logger.info(f"{scenario_type.upper()} scenario analysis complete:")
    logger.info(
        f"  Base cash flows total: ${scenario_analysis['base_cash_flows_total']:,.0f}")
    logger.info(
        f"  Scenario cash flows total: ${scenario_analysis['scenario_cash_flows_total']:,.0f}")
    logger.info(
        f"  Incremental value: ${scenario_analysis['incremental_value']:,.0f}")

    return scenario_cash_flows, scenario_analysis


def calculate_financial_metrics_for_scenario(
    cash_flows: np.ndarray,
    discount_rate: float,
    annual_h2_production_kg: float,
    project_lifetime_years: int,
    construction_period_years: int,
    scenario_name: str = "",
    baseline_lcoe_nuclear: float = None,
    baseline_lcos_battery: float = None
) -> Dict:
    """
    Calculate comprehensive financial metrics for a scenario.

    Args:
        cash_flows: Array of annual cash flows
        discount_rate: Discount rate for calculations
        annual_h2_production_kg: Annual hydrogen production in kg
        project_lifetime_years: Total project lifetime
        construction_period_years: Construction period
        scenario_name: Name of the scenario for logging

    Returns:
        Dictionary of financial metrics
    """
    logger.debug(
        f"Calculating financial metrics for {scenario_name} scenario...")

    # NPV calculation
    npv = npf.npv(discount_rate, cash_flows)

    # IRR calculation
    try:
        irr = npf.irr(cash_flows)
        if np.isnan(irr) or np.isinf(irr):
            irr = None
    except:
        irr = None

    # Simple payback period calculation
    cumulative_cash_flows = np.cumsum(cash_flows)
    payback_period = None
    for i, cum_cf in enumerate(cumulative_cash_flows):
        if cum_cf > 0:
            payback_period = i
            break

    # Discounted payback period
    discounted_cash_flows = cash_flows / \
        (1 + discount_rate) ** np.arange(len(cash_flows))
    cumulative_discounted_cf = np.cumsum(discounted_cash_flows)
    discounted_payback_period = None
    for i, cum_dcf in enumerate(cumulative_discounted_cf):
        if cum_dcf > 0:
            discounted_payback_period = i
            break

    # ROI calculation (simple return on investment)
    # Total negative cash flows
    total_investment = -np.sum(cash_flows[cash_flows < 0])
    # Total positive cash flows
    total_returns = np.sum(cash_flows[cash_flows > 0])
    roi_simple = (total_returns - total_investment) / \
        total_investment if total_investment > 0 else 0

    # LCOH calculation (if hydrogen production is available)
    lcoh = None
    if annual_h2_production_kg > 0:
        total_discounted_costs = -npv  # Negative NPV represents net cost
        total_discounted_h2_production = annual_h2_production_kg * sum(
            1 / (1 + discount_rate) ** year
            for year in range(construction_period_years, construction_period_years + project_lifetime_years)
        )
        lcoh = total_discounted_costs / \
            total_discounted_h2_production if total_discounted_h2_production > 0 else None

    metrics = {
        "npv_usd": npv,
        "irr": irr,
        "irr_percent": irr * 100 if irr is not None else None,
        "payback_period_years": payback_period,
        "discounted_payback_period_years": discounted_payback_period,
        "roi_simple": roi_simple,
        "roi_percent": roi_simple * 100,
        "lcoh_usd_per_kg": lcoh,
        "total_investment_usd": total_investment,
        "total_returns_usd": total_returns,
        "cumulative_cash_flows": cumulative_cash_flows,
        "discounted_cash_flows": discounted_cash_flows,
        "cumulative_discounted_cash_flows": cumulative_discounted_cf,
        # Add LCOE/LCOS values from baseline greenfield results
        "lcoe_nuclear_usd_per_mwh": baseline_lcoe_nuclear,
        "lcos_battery_usd_per_mwh": baseline_lcos_battery
    }

    return metrics


def run_comprehensive_tax_incentive_analysis(
    annual_metrics: Dict,
    base_cash_flows: np.ndarray,
    capex_breakdown: Dict,
    hourly_results_df: pd.DataFrame,
    macrs_config: Dict,
    project_lifetime_years: int = 60,
    construction_period_years: int = 8,
    discount_rate: float = 0.08,
    tax_rate: float = 0.21,
    plant_specific_params: Dict = None,
    baseline_lcoe_nuclear: float = None,
    baseline_lcos_battery: float = None
) -> Dict:
    """
    Run comprehensive analysis of all tax incentive scenarios.

    Args:
        annual_metrics: Dictionary of annual financial metrics
        base_cash_flows: Baseline cash flows without tax incentives
        capex_breakdown: Dictionary of component CAPEX values
        hourly_results_df: Hourly optimization results
        macrs_config: MACRS configuration
        project_lifetime_years: Total project lifetime
        construction_period_years: Construction period
        discount_rate: Discount rate for calculations
        tax_rate: Corporate tax rate
        plant_specific_params: Plant-specific parameters

    Returns:
        Comprehensive analysis dictionary with all scenarios
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE FEDERAL TAX INCENTIVE ANALYSIS")
    logger.info("Greenfield Nuclear-Hydrogen System")
    logger.info("=" * 80)

    # Validate input parameters
    expected_total_years = construction_period_years + project_lifetime_years

    if len(base_cash_flows) == 0:
        logger.error(
            "Base cash flows array is empty. Cannot proceed with tax incentive analysis.")
        raise ValueError("Base cash flows array is empty")

    if len(base_cash_flows) != expected_total_years:
        logger.warning(
            f"Base cash flows length ({len(base_cash_flows)}) does not match expected total years ({expected_total_years})")
        # Try to resize or pad the array
        if len(base_cash_flows) < expected_total_years:
            # Pad with zeros
            logger.info(
                f"Padding base cash flows from {len(base_cash_flows)} to {expected_total_years} years with zeros")
            padded_cash_flows = np.zeros(expected_total_years)
            padded_cash_flows[:len(base_cash_flows)] = base_cash_flows
            base_cash_flows = padded_cash_flows
        else:
            # Truncate
            logger.info(
                f"Truncating base cash flows from {len(base_cash_flows)} to {expected_total_years} years")
            base_cash_flows = base_cash_flows[:expected_total_years]

    logger.info(f"Base cash flows shape validated: {base_cash_flows.shape}")
    logger.info(
        f"Project parameters: {construction_period_years} construction + {project_lifetime_years} operational = {expected_total_years} total years")

    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)

    # Initialize results dictionary
    analysis_results = {
        "analysis_parameters": {
            "project_lifetime_years": project_lifetime_years,
            "construction_period_years": construction_period_years,
            "discount_rate": discount_rate,
            "tax_rate": tax_rate,
            "annual_h2_production_kg": annual_h2_production,
            "total_capex_usd": sum(capex_breakdown.values())
        },
        "scenarios": {}
    }

    # Scenario A: Baseline (no tax incentives)
    baseline_cash_flows, baseline_analysis = calculate_scenario_cash_flows(
        base_cash_flows=base_cash_flows,
        annual_metrics=annual_metrics,
        capex_breakdown=capex_breakdown,
        scenario_type="baseline",
        tax_rate=tax_rate,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        plant_specific_params=plant_specific_params
    )

    baseline_metrics = calculate_financial_metrics_for_scenario(
        cash_flows=baseline_cash_flows,
        discount_rate=discount_rate,
        annual_h2_production_kg=annual_h2_production,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        scenario_name="Baseline",
        baseline_lcoe_nuclear=baseline_lcoe_nuclear,
        baseline_lcos_battery=baseline_lcos_battery
    )

    analysis_results["scenarios"]["baseline"] = {
        "cash_flows": baseline_cash_flows,
        "analysis": baseline_analysis,
        "financial_metrics": baseline_metrics
    }

    # Scenario B: 45Y Production Tax Credit
    ptc_cash_flows, ptc_analysis = calculate_scenario_cash_flows(
        base_cash_flows=base_cash_flows,
        annual_metrics=annual_metrics,
        capex_breakdown=capex_breakdown,
        scenario_type="ptc",
        hourly_results_df=hourly_results_df,
        tax_rate=tax_rate,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        plant_specific_params=plant_specific_params
    )

    ptc_metrics = calculate_financial_metrics_for_scenario(
        cash_flows=ptc_cash_flows,
        discount_rate=discount_rate,
        annual_h2_production_kg=annual_h2_production,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        scenario_name="45Y PTC",
        baseline_lcoe_nuclear=baseline_lcoe_nuclear,
        baseline_lcos_battery=baseline_lcos_battery
    )

    analysis_results["scenarios"]["ptc"] = {
        "cash_flows": ptc_cash_flows,
        "analysis": ptc_analysis,
        "financial_metrics": ptc_metrics
    }

    # Scenario C: 48E Investment Tax Credit
    itc_cash_flows, itc_analysis = calculate_scenario_cash_flows(
        base_cash_flows=base_cash_flows,
        annual_metrics=annual_metrics,
        capex_breakdown=capex_breakdown,
        scenario_type="itc",
        macrs_config=macrs_config,
        tax_rate=tax_rate,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        plant_specific_params=plant_specific_params
    )

    itc_metrics = calculate_financial_metrics_for_scenario(
        cash_flows=itc_cash_flows,
        discount_rate=discount_rate,
        annual_h2_production_kg=annual_h2_production,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        scenario_name="48E ITC",
        baseline_lcoe_nuclear=baseline_lcoe_nuclear,
        baseline_lcos_battery=baseline_lcos_battery
    )

    analysis_results["scenarios"]["itc"] = {
        "cash_flows": itc_cash_flows,
        "analysis": itc_analysis,
        "financial_metrics": itc_metrics
    }

    # Generate comparative analysis
    analysis_results["comparative_analysis"] = generate_comparative_analysis(
        analysis_results)

    # Generate sensitivity analysis
    analysis_results["sensitivity_analysis"] = generate_sensitivity_analysis(
        annual_metrics=annual_metrics,
        base_cash_flows=base_cash_flows,
        capex_breakdown=capex_breakdown,
        hourly_results_df=hourly_results_df,
        macrs_config=macrs_config,
        project_lifetime_years=project_lifetime_years,
        construction_period_years=construction_period_years,
        discount_rate=discount_rate,
        tax_rate=tax_rate
    )

    logger.info("Comprehensive tax incentive analysis completed successfully")

    return analysis_results


def generate_comparative_analysis(analysis_results: Dict) -> Dict:
    """Generate comparative analysis between scenarios."""

    baseline = analysis_results["scenarios"]["baseline"]
    ptc = analysis_results["scenarios"]["ptc"]
    itc = analysis_results["scenarios"]["itc"]

    comparative = {
        "npv_comparison": {
            "baseline_npv": baseline["financial_metrics"]["npv_usd"],
            "ptc_npv": ptc["financial_metrics"]["npv_usd"],
            "itc_npv": itc["financial_metrics"]["npv_usd"],
            "ptc_npv_improvement": ptc["financial_metrics"]["npv_usd"] - baseline["financial_metrics"]["npv_usd"],
            "itc_npv_improvement": itc["financial_metrics"]["npv_usd"] - baseline["financial_metrics"]["npv_usd"]
        },
        "irr_comparison": {
            "baseline_irr": baseline["financial_metrics"]["irr_percent"],
            "ptc_irr": ptc["financial_metrics"]["irr_percent"],
            "itc_irr": itc["financial_metrics"]["irr_percent"]
        },
        "payback_comparison": {
            "baseline_payback": baseline["financial_metrics"]["payback_period_years"],
            "ptc_payback": ptc["financial_metrics"]["payback_period_years"],
            "itc_payback": itc["financial_metrics"]["payback_period_years"]
        },
        "best_scenario": None
    }

    # Determine best scenario based on NPV
    npv_values = {
        "baseline": baseline["financial_metrics"]["npv_usd"],
        "ptc": ptc["financial_metrics"]["npv_usd"],
        "itc": itc["financial_metrics"]["npv_usd"]
    }

    comparative["best_scenario"] = max(npv_values, key=npv_values.get)

    return comparative


def generate_sensitivity_analysis(
    annual_metrics: Dict,
    base_cash_flows: np.ndarray,
    capex_breakdown: Dict,
    hourly_results_df: pd.DataFrame,
    macrs_config: Dict,
    project_lifetime_years: int,
    construction_period_years: int,
    discount_rate: float,
    tax_rate: float
) -> Dict:
    """Generate sensitivity analysis for key parameters."""

    logger.info("Generating sensitivity analysis...")

    sensitivity_results = {
        "discount_rate_sensitivity": {},
        "ptc_rate_sensitivity": {},
        "itc_rate_sensitivity": {}
    }

    # Discount rate sensitivity (±2%)
    for dr_delta in [-0.02, -0.01, 0.01, 0.02]:
        adj_discount_rate = discount_rate + dr_delta

        # Recalculate metrics for baseline scenario
        baseline_metrics = calculate_financial_metrics_for_scenario(
            cash_flows=base_cash_flows,
            discount_rate=adj_discount_rate,
            annual_h2_production_kg=annual_metrics.get(
                "H2_Production_kg_annual", 0),
            project_lifetime_years=project_lifetime_years,
            construction_period_years=construction_period_years,
            scenario_name=f"Baseline_DR_{adj_discount_rate:.1%}"
        )

        sensitivity_results["discount_rate_sensitivity"][f"discount_rate_{adj_discount_rate:.1%}"] = {
            "npv": baseline_metrics["npv_usd"],
            "irr": baseline_metrics["irr_percent"]
        }

    logger.info("Sensitivity analysis completed")

    return sensitivity_results
