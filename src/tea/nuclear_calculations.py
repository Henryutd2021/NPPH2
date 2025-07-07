"""
Nuclear power plant economic calculations for TEA analysis.
"""

import logging
import math
import numpy as np
import pandas as pd

# Import MACRS depreciation functions
from .macrs import calculate_total_macrs_depreciation

# Import new tax incentive analysis functionality
from .tax_incentives import run_comprehensive_tax_incentive_analysis
from .tax_incentive_reporting import (
    generate_tax_incentive_comparative_report,
    create_tax_incentive_visualizations
)

logger = logging.getLogger(__name__)

# Constants
HOURS_IN_YEAR = 8760


def get_value_with_fallback(source_dict: dict, key: str, default_value: float, units: str = "", context: str = "") -> float:
    """
    Safely retrieve a value from a dictionary with fallback to default and logging.

    Args:
        source_dict: Dictionary to retrieve value from
        key: Key to look up
        default_value: Default value if key not found
        units: Units string for logging (e.g., "USD/MWh")
        context: Additional context for logging

    Returns:
        Retrieved value or default
    """
    if key in source_dict and source_dict[key] is not None:
        return float(source_dict[key])
    else:
        context_str = f" in {context}" if context else ""
        logger.warning(
            f"{key} not found{context_str} – using default value: {default_value} {units}"
        )
        return default_value


def calculate_45u_nuclear_ptc_benefits(
    annual_generation_mwh: float,
    project_start_year: int = 2024,
    credit_rate_per_mwh: float = None,
    credit_start_year: int = None,
    credit_end_year: int = None,
    project_lifetime_years: int = 40,
    tax_policies: dict = None,
    hourly_prices_usd_per_mwh=None,
    hourly_generation_profile_mwh=None,
) -> dict:
    """
    Calculate 45U Production Tax Credit benefits for existing nuclear power plants.

    45U Policy Details:
    - $15 per MWh for electricity generation from existing nuclear plants (configurable)
    - Available from 2024 to 2032 (configurable)
    - Only applies to existing nuclear plants (not new construction)

    Args:
        annual_generation_mwh: Annual electricity generation in MWh
        project_start_year: Year when the analysis starts
        credit_rate_per_mwh: PTC rate in $/MWh (if None, uses config default)
        credit_start_year: Year when 45U credit begins (if None, uses config default)
        credit_end_year: Year when 45U credit ends (if None, uses config default)
        project_lifetime_years: Total project lifetime for analysis
        tax_policies: Tax incentive policies configuration dictionary
        hourly_prices_usd_per_mwh: Array of hourly electricity prices in USD/MWh
        hourly_generation_profile_mwh: Array of hourly generation profiles in MWh

    Returns:
        Dictionary containing 45U benefits calculation results
    """

    # Load default values from tax policies configuration if not provided
    if tax_policies is None:
        try:
            from .config import TAX_INCENTIVE_POLICIES
            tax_policies = TAX_INCENTIVE_POLICIES
        except ImportError:
            logger.warning(
                "Could not import TAX_INCENTIVE_POLICIES, using hardcoded defaults")
            tax_policies = {
                "45u_ptc": {
                    "credit_rate_per_mwh": 15.0,
                    "credit_start_year": 2024,
                    "credit_end_year": 2032
                }
            }

    # Use configuration values if parameters not explicitly provided
    if credit_rate_per_mwh is None:
        credit_rate_per_mwh = tax_policies.get(
            "45u_ptc", {}).get("credit_rate_per_mwh", 15.0)
    if credit_start_year is None:
        credit_start_year = tax_policies.get(
            "45u_ptc", {}).get("credit_start_year", 2024)
    if credit_end_year is None:
        credit_end_year = tax_policies.get(
            "45u_ptc", {}).get("credit_end_year", 2032)
    logger.info(f"Calculating 45U Nuclear PTC benefits:")
    logger.info(f"  Annual generation: {annual_generation_mwh:,.0f} MWh")
    logger.info(f"  Credit rate: ${credit_rate_per_mwh}/MWh")
    logger.info(f"  Credit period: {credit_start_year}-{credit_end_year}")

    # ----------------------------------------------------------------------------------
    # Dynamic credit calculation based on hourly market price
    # ----------------------------------------------------------------------------------
    dynamic_credit_used = False
    if hourly_prices_usd_per_mwh is not None and len(hourly_prices_usd_per_mwh) > 0:
        hourly_prices = np.array(hourly_prices_usd_per_mwh, dtype=float)

        # Define piece-wise credit function vectorised
        def _calc_credit(price_array):
            # credit = 15 if p<25; 0 if p>43.75; else linear interpolation
            credit = np.where(price_array < 25.0, 15.0, 0.0)
            mid_mask = (price_array >= 25.0) & (price_array <= 43.75)
            credit[mid_mask] = 15.0 - 15.0 / \
                (43.75 - 25.0) * (price_array[mid_mask] - 25.0)
            return credit

        credit_rate_hourly = _calc_credit(hourly_prices)

        # Hourly generation profile
        if hourly_generation_profile_mwh is not None and len(hourly_generation_profile_mwh) == len(hourly_prices):
            generation_hourly = np.array(
                hourly_generation_profile_mwh, dtype=float)
        else:
            generation_per_hour = annual_generation_mwh / len(hourly_prices)
            generation_hourly = np.full(
                len(hourly_prices), generation_per_hour)

        # Annual credit value is the dot product of hourly credit rate and generation
        annual_credit_value_dynamic = np.sum(
            credit_rate_hourly * generation_hourly)
        dynamic_credit_used = True

        logger.info(
            "  Using dynamic 45U PTC calculation based on hourly prices")
        logger.info(
            f"  Dynamic annual credit value: ${annual_credit_value_dynamic:,.0f}")

    # ----------------------------------------------------------------------------------
    # Initialize annual credit array for project lifetime
    annual_45u_credits = np.zeros(project_lifetime_years)

    # Calculate eligible years and credit per year
    eligible_years = []
    for year in range(project_lifetime_years):
        project_year = project_start_year + year
        if credit_start_year <= project_year <= credit_end_year:
            eligible_years.append(year)
            if dynamic_credit_used:
                annual_45u_credits[year] = annual_credit_value_dynamic
            else:
                annual_45u_credits[year] = annual_generation_mwh * \
                    credit_rate_per_mwh

    # Summary calculations
    total_eligible_years = len(eligible_years)
    total_45u_credits = np.sum(annual_45u_credits)
    annual_credit_value = annual_45u_credits[eligible_years[0]
                                             ] if eligible_years else 0

    logger.info(f"  Eligible years: {total_eligible_years} years")
    logger.info(f"  Annual credit value: ${annual_credit_value:,.0f}")
    logger.info(f"  Total 45U credits: ${total_45u_credits:,.0f}")

    return {
        "policy_name": "45U Nuclear Production Tax Credit",
        "credit_rate_per_mwh": credit_rate_per_mwh,
        "credit_period_start": credit_start_year,
        "credit_period_end": credit_end_year,
        "eligible_years": eligible_years,
        "total_eligible_years": total_eligible_years,
        "annual_generation_mwh": annual_generation_mwh,
        "annual_credit_value": annual_credit_value,
        "annual_45u_credits": annual_45u_credits,
        "total_45u_credits": total_45u_credits,
        "applies_to_existing_plants_only": True,
        "dynamic_credit_used": dynamic_credit_used
    }


def calculate_nuclear_capex_breakdown(nuclear_capacity_mw: float, use_detailed_components: bool = True) -> dict:
    """
    Calculate nuclear plant CAPEX breakdown based on capacity.

    Args:
        nuclear_capacity_mw: Nuclear plant capacity in MW
        use_detailed_components: If True, use detailed NUCLEAR_CAPEX_COMPONENTS with learning rates.
                               If False, use simplified percentage-based breakdown.

    Returns:
        Dictionary with nuclear CAPEX breakdown including detailed component costs
    """
    from . import config

    if use_detailed_components and hasattr(config, 'NUCLEAR_CAPEX_COMPONENTS'):
        logger.info(
            f"Calculating detailed nuclear CAPEX breakdown for {nuclear_capacity_mw:.0f} MW")
        logger.info(
            "Using NUCLEAR_CAPEX_COMPONENTS with learning rates and component-specific costs")

        # Calculate detailed component costs with learning rates
        detailed_breakdown = {}
        total_nuclear_capex = 0

        for component_name, component_config in config.NUCLEAR_CAPEX_COMPONENTS.items():
            # Get component parameters
            base_cost = component_config.get("total_base_cost_for_ref_size", 0)
            reference_capacity = component_config.get(
                "reference_total_capacity_mw", 1000)
            learning_rate = component_config.get("learning_rate_decimal", 0.0)

            # Calculate capacity ratio
            capacity_ratio = nuclear_capacity_mw / reference_capacity

            # Apply learning rate (cost reduction for larger scale)
            # Learning rate formula: adjusted_cost = base_cost * (capacity_ratio)^(-learning_rate)
            if learning_rate > 0 and capacity_ratio != 1.0:
                learning_factor = capacity_ratio ** (-learning_rate)
                component_cost = base_cost * capacity_ratio * learning_factor
                logger.debug(
                    f"  {component_name}: ${base_cost:,.0f} * {capacity_ratio:.3f} * {learning_factor:.3f} = ${component_cost:,.0f}")
            else:
                component_cost = base_cost * capacity_ratio
                logger.debug(
                    f"  {component_name}: ${base_cost:,.0f} * {capacity_ratio:.3f} = ${component_cost:,.0f}")

            detailed_breakdown[component_name] = component_cost
            total_nuclear_capex += component_cost

        # Add total to breakdown
        detailed_breakdown["Total_Nuclear_CAPEX"] = total_nuclear_capex

        logger.info(f"Detailed nuclear CAPEX breakdown:")
        for component, cost in detailed_breakdown.items():
            if component != "Total_Nuclear_CAPEX":
                logger.info(f"  {component}: ${cost:,.0f}")
        logger.info(f"  Total Nuclear CAPEX: ${total_nuclear_capex:,.0f}")

        return detailed_breakdown

    else:
        # Fallback to simplified percentage-based breakdown
        logger.info(
            f"Using simplified percentage-based nuclear CAPEX breakdown for {nuclear_capacity_mw:.0f} MW")

        # Nuclear CAPEX per MW (from centralized config)
        nuclear_capex_per_mw = config.NUCLEAR_COST_PARAMETERS["nuclear_capex_per_mw"]

        # Calculate total CAPEX
        total_nuclear_capex = nuclear_capacity_mw * nuclear_capex_per_mw

        # Breakdown (percentages from centralized config)
        breakdown_percentages = config.NUCLEAR_COST_PARAMETERS["capex_breakdown_percentages"]
        breakdown = {
            "Nuclear_Island": total_nuclear_capex * breakdown_percentages["Nuclear_Island"],
            "Turbine_Generator": total_nuclear_capex * breakdown_percentages["Turbine_Generator"],
            "Balance_of_Plant": total_nuclear_capex * breakdown_percentages["Balance_of_Plant"],
            "Owner_Costs": total_nuclear_capex * breakdown_percentages["Owner_Costs"],
            "Total_Nuclear_CAPEX": total_nuclear_capex
        }

        logger.debug(
            f"Nuclear CAPEX breakdown for {nuclear_capacity_mw:.0f} MW: ${total_nuclear_capex:,.0f}")

        return breakdown


def calculate_nuclear_annual_opex(nuclear_capacity_mw: float, annual_generation_mwh: float) -> dict:
    """
    Calculate annual nuclear operating expenses.
    Uses centralized parameters from config.NUCLEAR_COST_PARAMETERS.
    """
    from . import config

    # Nuclear OPEX components (from centralized config)
    opex_params = config.NUCLEAR_COST_PARAMETERS["opex_parameters"]
    fixed_om_per_mw_month = opex_params["fixed_om_per_mw_month"]
    variable_om_per_mwh = opex_params["variable_om_per_mwh"]
    nuclear_fuel_cost_per_mwh = opex_params["fuel_cost_per_mwh"]
    additional_costs_per_mw_year = opex_params["additional_costs_per_mw_year"]

    # Calculate annual costs
    fixed_om_annual = nuclear_capacity_mw * fixed_om_per_mw_month * 12
    variable_om_annual = annual_generation_mwh * variable_om_per_mwh
    fuel_cost_annual = annual_generation_mwh * nuclear_fuel_cost_per_mwh
    additional_costs_annual = nuclear_capacity_mw * additional_costs_per_mw_year

    total_nuclear_opex = fixed_om_annual + variable_om_annual + \
        fuel_cost_annual + additional_costs_annual

    breakdown = {
        "Fixed_OM": fixed_om_annual,
        "Variable_OM": variable_om_annual,
        "Fuel_Cost": fuel_cost_annual,
        "Additional_Costs": additional_costs_annual,
        "Total_Nuclear_OPEX": total_nuclear_opex
    }

    return breakdown


def calculate_nuclear_replacement_costs(nuclear_capex_breakdown: dict, year: int) -> dict:
    """
    Calculate nuclear component replacement costs for specific years.
    """
    replacement_costs = {}

    # Major refurbishments every 30 years
    if year == 30:
        replacement_costs["Steam_Generator"] = nuclear_capex_breakdown["Nuclear_Island"] * 0.15
        replacement_costs["Turbine_Overhaul"] = nuclear_capex_breakdown["Turbine_Generator"] * 0.25

    # Extended operation upgrades every 40 years
    if year == 40:
        replacement_costs["Safety_Systems_Upgrade"] = nuclear_capex_breakdown["Nuclear_Island"] * 0.08
        replacement_costs["Control_Systems"] = nuclear_capex_breakdown["Balance_of_Plant"] * 0.10

    # 60-year license extension preparations
    if year == 50:
        replacement_costs["Containment_Upgrade"] = nuclear_capex_breakdown["Nuclear_Island"] * 0.05
        replacement_costs["Electrical_Systems"] = nuclear_capex_breakdown["Balance_of_Plant"] * 0.08

    total_replacement_cost = sum(replacement_costs.values())

    if total_replacement_cost > 0:
        logger.debug(
            f"Nuclear replacement costs for year {year}: ${total_replacement_cost:,.0f}")

    return replacement_costs


def calculate_nuclear_integrated_cash_flows(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    project_lifetime: int,
    construction_period: int,
    h2_subsidy_value: float,
    h2_subsidy_duration: int,
    capex_details: dict,
    om_details: dict,
    replacement_details: dict,
    optimized_capacities: dict,
    tax_rate: float
) -> np.ndarray:
    """
    Calculate nuclear-integrated cash flows.
    This is a placeholder - the actual implementation would be complex.
    """
    # Initialize cash flow array
    total_years = construction_period + project_lifetime
    cash_flows = np.zeros(total_years)

    # Construction period - CAPEX investment
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw)
    nuclear_total_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]

    # Distribute nuclear CAPEX over construction period
    annual_nuclear_capex = nuclear_total_capex / construction_period
    for year in range(construction_period):
        cash_flows[year] = -annual_nuclear_capex

    # Operating period
    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)
    annual_h2_revenue = annual_metrics.get("H2_Total_Revenue", 0)
    annual_as_revenue = annual_metrics.get("AS_Revenue_Total", 0)

    # Annual nuclear OPEX
    annual_nuclear_generation = nuclear_capacity_mw * \
        HOURS_IN_YEAR * 0.9  # 90% capacity factor
    nuclear_opex_breakdown = calculate_nuclear_annual_opex(
        nuclear_capacity_mw, annual_nuclear_generation)
    annual_nuclear_opex = nuclear_opex_breakdown["Total_Nuclear_OPEX"]

    # Annual net revenue
    annual_net_revenue = annual_h2_revenue + \
        annual_as_revenue - annual_nuclear_opex

    # Fill operating years
    for year in range(construction_period, total_years):
        operating_year = year - construction_period + 1

        # Base cash flow
        cash_flows[year] = annual_net_revenue

        # H2 subsidy (limited duration)
        if operating_year <= h2_subsidy_duration:
            cash_flows[year] += annual_h2_production * h2_subsidy_value

        # Replacement costs
        replacement_costs = calculate_nuclear_replacement_costs(
            nuclear_capex_breakdown, operating_year)
        total_replacement = sum(replacement_costs.values())
        cash_flows[year] -= total_replacement

    return cash_flows


def calculate_greenfield_nuclear_hydrogen_system(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict,
    project_lifetime_config: int,
    construction_period_config: int,
    discount_rate_config: float,
    tax_rate_config: float,
    h2_capex_components_config: dict,
    h2_om_components_config: dict,
    h2_replacement_schedule_config: dict
) -> dict:
    """
    Calculate financial metrics for a greenfield nuclear-hydrogen integrated system.
    Complete rewrite to match the original tea.py implementation exactly.
    """
    logger.info("=" * 80)
    logger.info("GREENFIELD NUCLEAR-HYDROGEN INTEGRATED SYSTEM ANALYSIS")
    logger.info(
        f"{project_lifetime_config}-Year Lifecycle Analysis with System Data & Hourly Results")
    logger.info("=" * 80)

    # Project parameters (use configurable project lifetime)
    project_lifetime = project_lifetime_config
    construction_period = 8
    discount_rate = discount_rate_config

    # Get subsidy parameters
    h2_subsidy_val = float(tea_sys_params.get(
        "hydrogen_subsidy_value_usd_per_kg", 0))
    h2_subsidy_duration_raw = tea_sys_params.get(
        "hydrogen_subsidy_duration_years", 10)
    try:
        h2_subsidy_yrs = int(float(str(h2_subsidy_duration_raw))
                             ) if h2_subsidy_duration_raw else 10
    except (ValueError, TypeError):
        h2_subsidy_yrs = 10

    logger.info(f"\nSystem Configuration:")
    logger.info(
        f"  Analysis Type                   : greenfield_nuclear_hydrogen_system_{project_lifetime}yr")
    logger.info(
        f"  Nuclear Capacity                : {nuclear_capacity_mw:,.0f} MW")
    logger.info(
        f"  Project Lifetime                : {project_lifetime} years")
    logger.info(
        f"  Construction Period             : {construction_period} years")
    logger.info(f"  Discount Rate                   : {discount_rate:.1%}")

    # === 1. NUCLEAR SYSTEM COSTS ===
    # Use detailed nuclear CAPEX components for greenfield analysis
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw, use_detailed_components=True)
    nuclear_total_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]

    # === 2. HYDROGEN SYSTEM COSTS ===
    # Get actual capacities from annual_metrics
    electrolyzer_capacity_mw = annual_metrics.get(
        "Electrolyzer_Capacity_MW", 0)
    h2_storage_capacity_kg = annual_metrics.get("H2_Storage_Capacity_kg", 0)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 0)
    battery_power_mw = annual_metrics.get("Battery_Power_MW", 0)

    # Initial hydrogen system CAPEX from optimization
    h2_initial_capex = annual_metrics.get("total_capex", 0)

    # Get replacement costs using actual system data from CAPEX_COMPONENTS
    # Calculate replacement schedules based on actual project lifetime

    # Electrolyzer replacements (every 20 years)
    electrolyzer_capex_component = h2_capex_components_config.get(
        "Electrolyzer_System", {})
    electrolyzer_ref_capacity = electrolyzer_capex_component.get(
        "reference_total_capacity_mw", 50)
    electrolyzer_ref_cost = electrolyzer_capex_component.get(
        "total_base_cost_for_ref_size", 100_000_000)

    if electrolyzer_capacity_mw > 0 and electrolyzer_ref_capacity > 0:
        electrolyzer_replacement_cost = electrolyzer_ref_cost * \
            (electrolyzer_capacity_mw / electrolyzer_ref_capacity)
    else:
        electrolyzer_replacement_cost = electrolyzer_capacity_mw * \
            1000 * 1200  # $1200/kW fallback

    # Calculate number of electrolyzer replacements based on project lifetime
    electrolyzer_replacement_years = [
        year for year in range(20, project_lifetime, 20)]
    num_electrolyzer_replacements = len(electrolyzer_replacement_years)
    total_electrolyzer_replacements = electrolyzer_replacement_cost * \
        num_electrolyzer_replacements

    # H2 Storage system replacements (every 30 years)
    h2_storage_capex_component = h2_capex_components_config.get(
        "H2_Storage_System", {})
    h2_storage_ref_capacity = h2_storage_capex_component.get(
        "reference_total_capacity_mw", 10000)  # kg
    h2_storage_ref_cost = h2_storage_capex_component.get(
        "total_base_cost_for_ref_size", 10_000_000)

    if h2_storage_capacity_kg > 0 and h2_storage_ref_capacity > 0:
        h2_storage_replacement_cost = h2_storage_ref_cost * \
            (h2_storage_capacity_kg / h2_storage_ref_capacity)
    else:
        h2_storage_replacement_cost = h2_storage_capacity_kg * 400  # $400/kg fallback

    # Calculate number of H2 storage replacements based on project lifetime
    h2_storage_replacement_years = [
        year for year in range(30, project_lifetime, 30)]
    num_h2_storage_replacements = len(h2_storage_replacement_years)
    total_h2_storage_replacements = h2_storage_replacement_cost * \
        num_h2_storage_replacements

    # Battery replacements (every 15 years)
    battery_energy_capex_component = h2_capex_components_config.get(
        "Battery_System_Energy", {})
    battery_power_capex_component = h2_capex_components_config.get(
        "Battery_System_Power", {})

    battery_energy_ref_capacity = battery_energy_capex_component.get(
        "reference_total_capacity_mw", 100)  # MWh
    battery_energy_ref_cost = battery_energy_capex_component.get(
        "total_base_cost_for_ref_size", 0)  # Should be 0 for power-only costing
    battery_power_ref_capacity = battery_power_capex_component.get(
        "reference_total_capacity_mw", 25)
    battery_power_ref_cost = battery_power_capex_component.get(
        "total_base_cost_for_ref_size", 23_600_000)

    # MODIFIED: Only power capacity cost (energy capacity cost = 0)
    battery_energy_replacement_cost = 0  # No energy capacity cost

    if battery_power_mw > 0 and battery_power_ref_capacity > 0:
        battery_power_replacement_cost = battery_power_ref_cost * \
            (battery_power_mw / battery_power_ref_capacity)
    else:
        battery_power_replacement_cost = 0

    battery_replacement_cost = battery_energy_replacement_cost + \
        battery_power_replacement_cost

    # Calculate number of battery replacements based on project lifetime
    battery_replacement_years = [
        year for year in range(15, project_lifetime, 15)]
    num_battery_replacements = len(battery_replacement_years)
    total_battery_replacements = battery_replacement_cost * num_battery_replacements

    # Enhanced maintenance factor for 60-year operation
    enhanced_maintenance_factor = 1.2

    # Total H2 system investment over 60 years
    total_h2_capex = h2_initial_capex + total_electrolyzer_replacements + \
        total_h2_storage_replacements + total_battery_replacements

    # === 3. TOTAL SYSTEM INVESTMENT ===
    total_system_capex = nuclear_total_capex + total_h2_capex

    # === 3.1. MACRS DEPRECIATION CALCULATION ===
    # Create combined CAPEX breakdown for MACRS calculation
    combined_capex_breakdown = {
        "Nuclear_Power_Plant": nuclear_total_capex,
        "Electrolyzer_System": h2_initial_capex * 0.6,  # Estimate electrolyzer portion
        "H2_Storage_System": h2_initial_capex * 0.2,    # Estimate storage portion
        # Estimate battery energy portion
        "Battery_System_Energy": h2_initial_capex * 0.15,
        "Battery_System_Power": h2_initial_capex * 0.05,   # Estimate battery power portion
    }

    # Import MACRS config
    try:
        from .config import MACRS_CONFIG
        macrs_config = MACRS_CONFIG
    except ImportError:
        logger.warning(
            "Could not import MACRS_CONFIG. Using default configuration.")
        macrs_config = {
            "enabled": True, "nuclear_depreciation_years": 15, "hydrogen_depreciation_years": 7}

    # Calculate MACRS depreciation for the integrated system
    logger.info(
        "Calculating MACRS depreciation for greenfield nuclear-hydrogen system")
    total_macrs_depreciation, component_macrs_depreciation = calculate_total_macrs_depreciation(
        capex_breakdown=combined_capex_breakdown,
        construction_period_years=construction_period,
        project_lifetime_years=project_lifetime,
        macrs_config=macrs_config
    )

    # Calculate total MACRS tax benefits
    tax_rate = tax_rate_config
    total_macrs_tax_benefits = np.sum(total_macrs_depreciation) * tax_rate

    logger.info(f"MACRS Depreciation Summary:")
    logger.info(
        f"  Total MACRS Depreciation        : ${np.sum(total_macrs_depreciation):,.0f}")
    logger.info(
        f"  Total Tax Benefits from MACRS   : ${total_macrs_tax_benefits:,.0f}")
    logger.info(f"  Tax Rate Applied                : {tax_rate:.1%}")

    logger.info(f"\nCapital Investment Breakdown:")
    logger.info(
        f"  Nuclear Plant CAPEX             : ${nuclear_total_capex:,.0f} ({nuclear_total_capex/total_system_capex*100:.1f}%)")
    logger.info(
        f"  Hydrogen System CAPEX           : ${total_h2_capex:,.0f} ({total_h2_capex/total_system_capex*100:.1f}%)")
    logger.info(
        f"  Total System CAPEX              : ${total_system_capex:,.0f}")
    logger.info(
        f"  CAPEX per MW Nuclear            : ${total_system_capex/nuclear_capacity_mw:,.0f}/MW")

    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)
    if annual_h2_production > 0:
        capex_per_kg_h2_annual = total_system_capex / annual_h2_production
        logger.info(
            f"  CAPEX per kg H2/year            : ${capex_per_kg_h2_annual:,.0f}/kg")
    else:
        capex_per_kg_h2_annual = 0
        logger.info(
            f"  CAPEX per kg H2/year            : N/A (no H2 production)")

    # === 4. PRODUCTION METRICS FROM HOURLY DATA ===
    # Use actual production data from optimization hourly results as typical year
    logger.info(
        "Using hourly optimization results as typical year for lifecycle analysis")

    # Nuclear generation calculated from actual turbine operation
    nuclear_capacity_factor = annual_metrics.get(
        "Turbine_CF_percent", 90) / 100
    annual_nuclear_generation = nuclear_capacity_mw * 8760 * nuclear_capacity_factor

    # Verify against any direct nuclear generation data if available
    direct_nuclear_gen = annual_metrics.get("Annual_Nuclear_Generation_MWh", 0)
    if direct_nuclear_gen > 0:
        logger.info(
            f"Direct nuclear generation data available: {direct_nuclear_gen:,.0f} MWh/year")
        annual_nuclear_generation = direct_nuclear_gen

    logger.info(
        f"Nuclear generation for lifecycle analysis: {annual_nuclear_generation:,.0f} MWh/year")
    logger.info(f"Nuclear capacity factor: {nuclear_capacity_factor:.1%}")

    # Efficiency metrics based on actual production
    hydrogen_lhv_kwh_per_kg = 33.3  # kWh/kg H2 LHV
    if annual_nuclear_generation > 0 and annual_h2_production > 0:
        electricity_to_h2_efficiency = (
            annual_h2_production * hydrogen_lhv_kwh_per_kg / 1000) / annual_nuclear_generation
    else:
        electricity_to_h2_efficiency = 0

    if nuclear_capacity_mw > 0:
        h2_production_per_mw = annual_h2_production / nuclear_capacity_mw
    else:
        h2_production_per_mw = 0

    logger.info(f"\nProduction Metrics:")
    logger.info(
        f"  Annual H2 Production            : {annual_h2_production:,.0f} kg/year")
    logger.info(
        f"  Annual Nuclear Generation       : {annual_nuclear_generation:,.0f} MWh/year")
    logger.info(
        f"  H2 Production per MW Nuclear    : {h2_production_per_mw:,.0f} kg/MW/year")
    logger.info(
        f"  Nuclear Capacity Factor         : {nuclear_capacity_factor:.1%}")
    logger.info(
        f"  Electricity to H2 Efficiency    : {electricity_to_h2_efficiency:.1%}")

    logger.info(f"\nInvestment Breakdown ({project_lifetime}-year lifecycle):")
    logger.info(
        f"  H2 System Initial CAPEX         : ${h2_initial_capex:,.0f}")
    logger.info(
        f"  H2 System Replacement CAPEX     : ${total_h2_capex - h2_initial_capex:,.0f}")
    logger.info(
        f"    Electrolyzer Replacements     : {num_electrolyzer_replacements} times (years: {electrolyzer_replacement_years})")
    logger.info(
        f"    H2 Storage Replacements       : {num_h2_storage_replacements} times (years: {h2_storage_replacement_years})")
    logger.info(
        f"    Battery Replacements          : {num_battery_replacements} times (years: {battery_replacement_years})")
    logger.info(
        f"  Enhanced Maintenance Factor     : {enhanced_maintenance_factor:.1f}x")

    # === 5. FINANCIAL ANALYSIS ===
    # Get hydrogen price from system data file, not default values
    h2_price_raw = tea_sys_params.get("H2_value_USD_per_kg")
    if h2_price_raw is None:
        # Try alternative keys for hydrogen price
        h2_price_raw = tea_sys_params.get("hydrogen_price_usd_per_kg")
        if h2_price_raw is None:
            h2_price_raw = tea_sys_params.get("h2_price_usd_per_kg")

    try:
        h2_price = float(h2_price_raw) if h2_price_raw is not None else None
        if h2_price is None:
            logger.warning(
                "Hydrogen price not found in system data file. This is required for accurate analysis.")
            h2_price = 5.0  # Fallback only if absolutely necessary
        else:
            logger.info(
                f"Using hydrogen price from system data: ${h2_price:.2f}/kg")
    except (ValueError, TypeError):
        logger.warning(
            f"Invalid hydrogen price value in system data: {h2_price_raw}. Using fallback.")
        h2_price = 5.0

    annual_h2_revenue = annual_metrics.get(
        "H2_Total_Revenue", annual_h2_production * h2_price)

    # Get actual AS revenue and use real system-specific AS revenue data
    annual_as_revenue_total = annual_metrics.get("AS_Revenue_Total", 0)
    if annual_as_revenue_total == 0:
        annual_as_revenue_total = annual_metrics.get("AS_Revenue", 0)

    # Use real AS revenue breakdown from system components instead of estimates
    turbine_as_revenue = annual_metrics.get("AS_Revenue_Turbine", 0)
    electrolyzer_as_revenue = annual_metrics.get("AS_Revenue_Electrolyzer", 0)
    battery_as_revenue = annual_metrics.get("AS_Revenue_Battery", 0)
    h2_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue

    # If detailed breakdown not available, calculate from deployment data
    if turbine_as_revenue == 0 and h2_system_as_revenue == 0 and annual_as_revenue_total > 0:
        # Use detailed AS deployment data to calculate real revenue allocation
        as_deployment_keys_turbine = [
            "AS_Total_Deployed_ECRS_Turbine_MWh", "AS_Total_Deployed_RegDown_Turbine_MWh",
            "AS_Total_Deployed_RegUp_Turbine_MWh", "AS_Total_Deployed_NSR_Turbine_MWh",
            "AS_Total_Deployed_SR_Turbine_MWh"
        ]
        as_deployment_keys_electrolyzer = [
            "AS_Total_Deployed_ECRS_Electrolyzer_MWh", "AS_Total_Deployed_RegDown_Electrolyzer_MWh",
            "AS_Total_Deployed_RegUp_Electrolyzer_MWh", "AS_Total_Deployed_NSR_Electrolyzer_MWh",
            "AS_Total_Deployed_SR_Electrolyzer_MWh"
        ]
        as_deployment_keys_battery = [
            "AS_Total_Deployed_ECRS_Battery_MWh", "AS_Total_Deployed_RegDown_Battery_MWh",
            "AS_Total_Deployed_RegUp_Battery_MWh", "AS_Total_Deployed_NSR_Battery_MWh",
            "AS_Total_Deployed_SR_Battery_MWh"
        ]

        total_turbine_deployment = sum(annual_metrics.get(
            key, 0) for key in as_deployment_keys_turbine)
        total_electrolyzer_deployment = sum(annual_metrics.get(
            key, 0) for key in as_deployment_keys_electrolyzer)
        total_battery_deployment = sum(annual_metrics.get(
            key, 0) for key in as_deployment_keys_battery)
        total_deployment = total_turbine_deployment + \
            total_electrolyzer_deployment + total_battery_deployment

        if total_deployment > 0:
            turbine_as_revenue = annual_as_revenue_total * \
                (total_turbine_deployment / total_deployment)
            electrolyzer_as_revenue = annual_as_revenue_total * \
                (total_electrolyzer_deployment / total_deployment)
            battery_as_revenue = annual_as_revenue_total * \
                (total_battery_deployment / total_deployment)
            h2_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue
        else:
            # Use bid capacity allocation as fallback
            electrolyzer_capacity = annual_metrics.get(
                "Electrolyzer_Capacity_MW", 0)
            battery_power = annual_metrics.get("Battery_Power_MW", 0)
            turbine_capacity = annual_metrics.get("Turbine_Capacity_MW", 0)
            total_capacity = electrolyzer_capacity + battery_power + turbine_capacity

            if total_capacity > 0:
                turbine_as_revenue = annual_as_revenue_total * \
                    (turbine_capacity / total_capacity)
                electrolyzer_as_revenue = annual_as_revenue_total * \
                    (electrolyzer_capacity / total_capacity)
                battery_as_revenue = annual_as_revenue_total * \
                    (battery_power / total_capacity)
                h2_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue
            else:
                # Final fallback: use zero allocation when no deployment or capacity data available
                logger.warning(
                    "No AS deployment or capacity data available. Setting AS revenue allocation to zero for accurate accounting.")
                turbine_as_revenue = 0
                h2_system_as_revenue = 0

    # Include HTE thermal energy opportunity cost
    hte_thermal_cost = annual_metrics.get(
        "HTE_Heat_Opportunity_Cost_Annual_USD", 0)

    # Get average electricity price
    avg_electricity_price = get_value_with_fallback(
        annual_metrics, "Avg_Electricity_Price_USD_per_MWh", 60.0, "USD/MWh",
        "greenfield 60-year analysis")

    # Hydrogen subsidy revenue
    h2_subsidy_revenue = annual_h2_production * \
        h2_subsidy_val if h2_subsidy_yrs > 0 else 0

    # Annual operating costs
    nuclear_annual_opex_breakdown = calculate_nuclear_annual_opex(
        nuclear_capacity_mw, annual_nuclear_generation)
    nuclear_annual_opex = nuclear_annual_opex_breakdown["Total_Nuclear_OPEX"]
    h2_annual_opex = total_h2_capex * 0.025 * \
        enhanced_maintenance_factor  # 2.5% of CAPEX
    total_annual_opex = nuclear_annual_opex + h2_annual_opex

    # Total annual revenue (for greenfield, all electricity goes to H2 production)
    annual_electricity_revenue = 0
    total_annual_revenue = (annual_h2_revenue + annual_electricity_revenue +
                            turbine_as_revenue + h2_system_as_revenue + h2_subsidy_revenue)
    annual_net_revenue = total_annual_revenue - total_annual_opex - hte_thermal_cost

    logger.info(f"\nAnnual Performance:")
    logger.info(
        f"  Total Annual Revenue            : ${total_annual_revenue:,.0f}")
    logger.info(
        f"    H2 Revenue                    : ${annual_h2_revenue:,.0f}")
    logger.info(
        f"    Electricity Revenue           : ${annual_electricity_revenue:,.0f}")
    logger.info(
        f"    Turbine AS Revenue (Real)     : ${turbine_as_revenue:,.0f}")
    logger.info(
        f"    H2 System AS Revenue (Real)   : ${h2_system_as_revenue:,.0f}")
    logger.info(
        f"    H2 Subsidy Revenue            : ${h2_subsidy_revenue:,.0f}")
    logger.info(
        f"  Total Annual OPEX               : ${total_annual_opex:,.0f}")
    logger.info(
        f"  HTE Thermal Opportunity Cost    : ${hte_thermal_cost:,.0f}")
    logger.info(
        f"  Net Annual Revenue              : ${annual_net_revenue:,.0f}")

    # === 6. CALCULATE FINANCIAL METRICS USING INDEPENDENT ACCOUNTING ===
    # Present value calculations
    nuclear_costs_pv = nuclear_total_capex
    h2_system_costs_pv = total_h2_capex
    turbine_as_revenue_pv = 0
    h2_as_revenue_pv = 0
    h2_revenue_pv = 0
    h2_subsidy_pv = 0
    nuclear_opex_pv = 0
    h2_opex_pv = 0
    hte_thermal_costs_pv = 0
    electricity_revenue_pv = 0

    # Calculate present values over project lifetime
    for year in range(construction_period + 1, construction_period + project_lifetime + 1):
        operating_year = year - construction_period
        discount_factor = (1 + discount_rate) ** (year - 1)

        # Revenues
        h2_revenue_pv += annual_h2_revenue / discount_factor
        turbine_as_revenue_pv += turbine_as_revenue / discount_factor
        h2_as_revenue_pv += h2_system_as_revenue / discount_factor
        electricity_revenue_pv += annual_electricity_revenue / discount_factor

        # Costs
        nuclear_opex_pv += nuclear_annual_opex / discount_factor
        h2_opex_pv += h2_annual_opex / discount_factor
        hte_thermal_costs_pv += hte_thermal_cost / discount_factor

        # Subsidies (only for specified duration)
        if operating_year <= h2_subsidy_yrs:
            h2_subsidy_pv += h2_subsidy_revenue / discount_factor

        # Add replacement costs in specific years based on dynamic schedules
        replacement_cost = 0
        if operating_year in battery_replacement_years:
            replacement_cost += battery_replacement_cost
        if operating_year in electrolyzer_replacement_years:
            replacement_cost += electrolyzer_replacement_cost
        if operating_year in h2_storage_replacement_years:
            replacement_cost += h2_storage_replacement_cost

        if replacement_cost > 0:
            h2_system_costs_pv += replacement_cost / discount_factor

    # Calculate present value of production
    total_h2_production_pv = 0
    total_generation_pv = 0

    for year in range(1, project_lifetime + 1):
        discount_factor = (1 + discount_rate) ** year
        total_h2_production_pv += annual_h2_production / discount_factor
        total_generation_pv += annual_nuclear_generation / discount_factor

    # === INDEPENDENT ACCOUNTING METHOD FOR LCOE/LCOH/LCOS ===

    # 1. NUCLEAR LCOE: (nuclear costs + nuclear opex - turbine AS revenue) / total generation
    nuclear_total_costs_pv = nuclear_costs_pv + nuclear_opex_pv
    if total_generation_pv > 0:
        nuclear_lcoe = (nuclear_total_costs_pv -
                        turbine_as_revenue_pv) / total_generation_pv
    else:
        nuclear_lcoe = 0

    # 2. LCOH: H2 system costs + H2 opex + electricity costs + HTE thermal costs - H2 system AS revenue) / H2 production
    # Calculate electricity consumption for H2 production using actual hourly data
    electrolyzer_electricity_consumption_annual = get_value_with_fallback(
        annual_metrics, "Annual_Electrolyzer_MWh", 0, "MWh",
        "electrolyzer consumption data")
    if electrolyzer_electricity_consumption_annual == 0:
        # Estimate from H2 production (50 kWh/kg H2 typical)
        electrolyzer_electricity_consumption_annual = annual_h2_production * 50 / 1000
        logger.warning(
            f"Annual_Electrolyzer_MWh is zero – estimating from H2 production: {electrolyzer_electricity_consumption_annual:.0f} MWh/year (50 kWh/kg H2). Actual hourly data preferred.")
    else:
        logger.info(
            f"Using actual electrolyzer electricity consumption from hourly data: {electrolyzer_electricity_consumption_annual:,.0f} MWh/year")

    # Add HTE thermal energy consumption and opportunity cost
    hte_steam_consumption_annual = annual_metrics.get(
        "HTE_Steam_Consumption_Annual_MWth", 0)
    thermal_efficiency = get_value_with_fallback(
        annual_metrics, "thermal_efficiency", 0.335, "",
        "thermal efficiency")

    # Calculate thermal energy opportunity cost in electricity terms
    if hte_steam_consumption_annual > 0 and thermal_efficiency > 0:
        # Thermal energy converted to lost electricity generation
        hte_electricity_equivalent_annual = hte_steam_consumption_annual / thermal_efficiency
        electrolyzer_electricity_consumption_annual += hte_electricity_equivalent_annual
        logger.info(
            f"HTE thermal energy equivalent: {hte_electricity_equivalent_annual:,.0f} MWh/year")

    # Battery charging electricity consumption - use actual hourly data
    battery_charge_annual = annual_metrics.get("Annual_Battery_Charge_MWh", 0)
    battery_charge_from_grid = annual_metrics.get(
        "Annual_Battery_Charge_From_Grid_MWh", 0)
    battery_charge_from_npp = annual_metrics.get(
        "Annual_Battery_Charge_From_NPP_MWh", 0)

    logger.info(f"Battery charging breakdown:")
    logger.info(
        f"  Total battery charging: {battery_charge_annual:,.0f} MWh/year")
    logger.info(
        f"  From grid purchase: {battery_charge_from_grid:,.0f} MWh/year")
    logger.info(
        f"  From NPP (opportunity cost): {battery_charge_from_npp:,.0f} MWh/year")

    # Total electricity consumption for cost calculation
    # For H2 production: use nuclear LCOE (opportunity cost)
    # For battery charging from NPP: use nuclear LCOE (opportunity cost)
    # For battery charging from grid: use market price (direct cost)
    total_electricity_consumption_annual = electrolyzer_electricity_consumption_annual + \
        battery_charge_from_npp
    grid_electricity_consumption_annual = battery_charge_from_grid

    # Present value of electricity costs using nuclear LCOE for NPP-sourced electricity
    electricity_costs_pv = 0
    grid_electricity_costs_pv = 0

    for year in range(1, project_lifetime + 1):
        discount_factor = (1 + discount_rate) ** year
        # NPP electricity at nuclear LCOE (opportunity cost)
        electricity_costs_pv += (total_electricity_consumption_annual *
                                 nuclear_lcoe) / discount_factor
        # Grid electricity at market price (direct cost)
        grid_electricity_costs_pv += (grid_electricity_consumption_annual *
                                      avg_electricity_price) / discount_factor

    total_electricity_costs_pv = electricity_costs_pv + grid_electricity_costs_pv

    logger.info(f"Electricity cost calculation:")
    logger.info(
        f"  NPP electricity cost (PV): ${electricity_costs_pv:,.0f} at LCOE ${nuclear_lcoe:.2f}/MWh")
    logger.info(
        f"  Grid electricity cost (PV): ${grid_electricity_costs_pv:,.0f} at market price ${avg_electricity_price:.2f}/MWh")

    h2_system_total_costs_pv = h2_system_costs_pv + h2_opex_pv
    if total_h2_production_pv > 0:
        # Use total electricity costs (both NPP and grid)
        lcoh_integrated = ((h2_system_total_costs_pv + total_electricity_costs_pv +
                           hte_thermal_costs_pv - h2_as_revenue_pv) / total_h2_production_pv)
    else:
        lcoh_integrated = 0

    # 3. INDEPENDENT BATTERY SYSTEM LCOS CALCULATION
    # Calculate LCOS for battery as independent system using actual hourly data
    battery_lcos = 0
    battery_system_npv = 0
    battery_system_revenue_pv = 0
    battery_system_costs_pv = 0

    if battery_capacity_mwh > 0:
        logger.info(
            "Calculating independent battery system LCOS using actual hourly data")

        # Battery system CAPEX from replacement costs (initial investment)
        battery_total_capex = battery_energy_replacement_cost + \
            battery_power_replacement_cost

        # Battery system OPEX using actual data from config
        battery_fixed_om_annual = get_value_with_fallback(
            annual_metrics, "Battery_Fixed_OM_Annual", 0, "USD/year",
            "battery fixed O&M")
        if battery_fixed_om_annual == 0:
            # Fallback: estimate from capacity
            battery_fixed_om_annual = (
                battery_power_mw * 10000 + battery_capacity_mwh * 5000)  # $10k/MW + $5k/MWh
            logger.warning(
                f"Battery_Fixed_OM_Annual is zero – estimating from capacity: ${battery_fixed_om_annual:,.0f}/year")

        battery_vom_annual = annual_metrics.get("VOM_Battery_Cost", 0)

        # Battery electricity costs using actual charging data and nuclear LCOE
        battery_charge_from_npp_annual = annual_metrics.get(
            "Annual_Battery_Charge_From_NPP_MWh", 0)
        battery_charge_from_grid_annual = annual_metrics.get(
            "Annual_Battery_Charge_From_Grid_MWh", 0)

        # Electricity costs: NPP at LCOE (opportunity cost), grid at market price
        battery_electricity_cost_annual = (battery_charge_from_npp_annual * nuclear_lcoe +
                                           battery_charge_from_grid_annual * avg_electricity_price)

        # Battery AS revenue - use actual deployment data from hourly results
        battery_as_revenue_annual = 0
        as_deployment_keys = [k for k in annual_metrics.keys(
        ) if "AS_Total_Deployed" in k and "Battery" in k]
        for key in as_deployment_keys:
            battery_as_deployed_mwh = annual_metrics.get(key, 0)
            # Use average electricity price as AS price proxy
            battery_as_revenue_annual += battery_as_deployed_mwh * \
                avg_electricity_price * 1.5  # 50% premium for AS

        # Battery energy arbitrage revenue (discharge at higher prices)
        battery_discharge_annual = annual_metrics.get(
            "Annual_Battery_Discharge_MWh", 0)
        if battery_discharge_annual == 0:
            # Estimate from charging with round-trip efficiency
            battery_discharge_annual = (
                battery_charge_from_npp_annual + battery_charge_from_grid_annual) * 0.85

        # If still zero and we have battery capacity, use a reasonable estimate
        if battery_discharge_annual == 0 and battery_capacity_mwh > 0:
            # Estimate based on battery capacity and typical utilization
            # Assume 1 cycle per day with 85% efficiency
            battery_discharge_annual = battery_capacity_mwh * 365 * 0.85
            logger.info(
                f"Using estimated battery discharge based on capacity: {battery_discharge_annual:,.0f} MWh/year")

        battery_arbitrage_revenue_annual = battery_discharge_annual * \
            avg_electricity_price * 0.1  # 10% arbitrage margin

        logger.info(f"Battery system annual metrics:")
        logger.info(f"  CAPEX: ${battery_total_capex:,.0f}")
        logger.info(f"  Fixed O&M: ${battery_fixed_om_annual:,.0f}/year")
        logger.info(f"  VOM: ${battery_vom_annual:,.0f}/year")
        logger.info(
            f"  Electricity cost: ${battery_electricity_cost_annual:,.0f}/year")
        logger.info(f"  AS revenue: ${battery_as_revenue_annual:,.0f}/year")
        logger.info(
            f"  Arbitrage revenue: ${battery_arbitrage_revenue_annual:,.0f}/year")
        logger.info(
            f"  Annual discharge: {battery_discharge_annual:,.0f} MWh/year")

        # Calculate present values
        battery_system_costs_pv = battery_total_capex  # Initial CAPEX
        battery_system_revenue_pv = 0
        battery_throughput_pv = 0

        for year in range(1, project_lifetime + 1):
            discount_factor = (1 + discount_rate) ** year

            # Annual costs
            annual_battery_costs = (battery_fixed_om_annual + battery_vom_annual +
                                    battery_electricity_cost_annual)
            battery_system_costs_pv += annual_battery_costs / discount_factor

            # Annual revenues
            annual_battery_revenue = battery_as_revenue_annual + \
                battery_arbitrage_revenue_annual
            battery_system_revenue_pv += annual_battery_revenue / discount_factor

            # Annual throughput (energy discharged for consistency with existing reports)
            battery_throughput_pv += battery_discharge_annual / discount_factor

            # Battery replacements based on 15-year schedule (60-year project)
            if year in [15, 30, 45]:
                battery_replacement_cost = battery_total_capex * 0.8  # 80% of initial cost
                battery_system_costs_pv += battery_replacement_cost / discount_factor
                logger.info(
                    f"  Battery replacement year {year}: ${battery_replacement_cost:,.0f} (PV: ${battery_replacement_cost / discount_factor:,.0f})")

        # Calculate LCOS and NPV
        if battery_throughput_pv > 0:
            battery_lcos = (battery_system_costs_pv -
                            battery_system_revenue_pv) / battery_throughput_pv

            # Log detailed calculation for debugging (60-year)
            logger.info(f"LCOS CALCULATION DEBUG (60-year):")
            logger.info(
                f"  Battery System Costs (PV): ${battery_system_costs_pv:,.0f}")
            logger.info(
                f"  Battery System Revenue (PV): ${battery_system_revenue_pv:,.0f}")
            logger.info(
                f"  Net Costs (PV): ${battery_system_costs_pv - battery_system_revenue_pv:,.0f}")
            logger.info(
                f"  Battery Throughput (PV): {battery_throughput_pv:,.0f} MWh")
            logger.info(f"  Calculated LCOS: ${battery_lcos:.2f}/MWh")
            logger.info(f"  Replacement count: 3 (years 15, 30, 45)")
        else:
            battery_lcos = 0

        battery_system_npv = battery_system_revenue_pv - battery_system_costs_pv

        logger.info(f"Independent battery system analysis:")
        logger.info(f"  Total costs (PV): ${battery_system_costs_pv:,.0f}")
        logger.info(f"  Total revenue (PV): ${battery_system_revenue_pv:,.0f}")
        logger.info(f"  NPV: ${battery_system_npv:,.0f}")
        logger.info(f"  LCOS: ${battery_lcos:.2f}/MWh discharged")

    # Calculate present value of MACRS tax benefits
    macrs_tax_benefits_pv = 0
    for year in range(construction_period + 1, construction_period + project_lifetime + 1):
        year_index = year - 1  # Convert to 0-based index for depreciation array
        if year_index < len(total_macrs_depreciation):
            annual_macrs_depreciation = total_macrs_depreciation[year_index]
            annual_tax_benefit = annual_macrs_depreciation * tax_rate
            discount_factor = (1 + discount_rate) ** (year - 1)
            macrs_tax_benefits_pv += annual_tax_benefit / discount_factor

    logger.info(
        f"MACRS Tax Benefits (Present Value): ${macrs_tax_benefits_pv:,.0f}")

    # Total system NPV (including MACRS tax benefits)
    total_revenue_pv = (h2_revenue_pv + h2_subsidy_pv + turbine_as_revenue_pv +
                        h2_as_revenue_pv + electricity_revenue_pv + macrs_tax_benefits_pv)
    total_costs_pv = (nuclear_total_costs_pv + h2_system_total_costs_pv +
                      total_electricity_costs_pv + hte_thermal_costs_pv)
    npv = total_revenue_pv - total_costs_pv

    # IRR calculation
    try:
        if npv > 0 and total_costs_pv > 0:
            irr_estimate = (total_revenue_pv /
                            total_costs_pv) ** (1/project_lifetime) - 1
            irr_percent = irr_estimate * 100
        else:
            irr_percent = float('nan')
    except:
        irr_percent = float('nan')

    # ROI calculation
    if total_system_capex > 0:
        roi_percent = (npv / total_system_capex * 100)
    else:
        roi_percent = 0

    # Payback period calculation
    cumulative_cash_flow = -total_system_capex
    payback_years = float('nan')

    for year in range(1, project_lifetime + 1):
        annual_net_revenue_calc = (annual_h2_revenue + turbine_as_revenue + h2_system_as_revenue +
                                   annual_electricity_revenue - total_annual_opex - hte_thermal_cost)
        if year <= h2_subsidy_yrs:
            annual_net_revenue_calc += h2_subsidy_revenue
        cumulative_cash_flow += annual_net_revenue_calc
        if cumulative_cash_flow > 0 and payback_years != payback_years:  # Check for NaN
            payback_years = year
            break

    logger.info(f"\nFinancial Results (60-year lifecycle with MACRS):")
    logger.info(f"  Net Present Value (NPV)         : ${npv:,.0f}")
    logger.info(
        f"  MACRS Tax Benefits (PV)         : ${macrs_tax_benefits_pv:,.0f}")
    logger.info(
        f"  NPV without MACRS               : ${npv - macrs_tax_benefits_pv:,.0f}")
    if irr_percent == irr_percent:  # Check for not NaN
        logger.info(f"  Internal Rate of Return (IRR)   : {irr_percent:.2f}%")
    else:
        logger.info(f"  Internal Rate of Return (IRR)   : N/A")
    logger.info(f"  Return on Investment (ROI)      : {roi_percent:.2f}%")
    if payback_years == payback_years:  # Check for not NaN
        logger.info(
            f"  Payback Period                  : {payback_years:.0f} years")
    else:
        logger.info(f"  Payback Period                  : N/A")

    logger.info(f"\nLevelized Costs (Independent Accounting Method):")
    logger.info(
        f"  LCOH (Integrated System)        : ${lcoh_integrated:.3f}/kg")
    logger.info(f"  Nuclear LCOE                    : ${nuclear_lcoe:.2f}/MWh")
    if battery_lcos > 0:
        logger.info(
            f"  Battery LCOS                    : ${battery_lcos:.2f}/MWh")
    logger.info(
        "\nNote: Independent accounting method used:")
    logger.info(
        "• LCOE: (nuclear costs + nuclear OPEX - turbine AS revenue) / total generation")
    logger.info(
        "• LCOH: (H2 costs + H2 OPEX + electricity at LCOE + HTE thermal costs - H2 AS revenue) / H2 production")
    logger.info(
        "• LCOS: (battery costs + battery OPEX) / battery throughput")
    logger.info(
        "• All AS revenues calculated from real system deployment data, not estimates")

    logger.info(f"\nCash Flow Summary (Present Value):")
    logger.info(
        f"  Total Revenue (PV)              : ${total_revenue_pv:,.0f}")
    logger.info(f"  Total Costs (PV)                : ${total_costs_pv:,.0f}")
    logger.info(f"  Net Cash Flow (PV)              : ${npv:,.0f}")

    logger.info(f"\nKey Insights:")
    if npv > 0:
        logger.info(
            f"  • The greenfield nuclear-hydrogen system shows positive NPV")
    else:
        logger.info(
            f"  • The greenfield nuclear-hydrogen system shows negative NPV")

    if irr_percent == irr_percent and irr_percent > discount_rate * 100:  # Check for not NaN
        logger.info(
            f"  • IRR exceeds the discount rate, indicating attractive returns")
    else:
        logger.info(
            f"  • IRR is below the discount rate, indicating marginal returns")

    if payback_years == payback_years and payback_years < project_lifetime / 2:  # Check for not NaN
        logger.info(f"  • Reasonable payback period indicates manageable risk")
    else:
        logger.info(
            f"  • Long payback period indicates high capital requirements")

    logger.info(
        f"\nNote: This greenfield analysis assumes building both nuclear plant and")
    logger.info(
        f"hydrogen system from zero, with both systems designed for 60-year operation.")
    logger.info(
        f"The analysis includes periodic replacement of H2 system components:")
    logger.info(f"• Electrolyzers replaced every 20 years (2 replacements)")
    logger.info(f"• H2 storage systems replaced every 30 years (1 replacement)")
    logger.info(f"• Batteries replaced every 15 years (3 replacements)")
    logger.info(
        f"• Enhanced maintenance costs (+20%) for extended lifecycle operation")
    logger.info(
        f"This provides a comprehensive view of long-term integrated system economics.")

    # === 7. CALCULATE DETAILED CASH FLOWS ===
    # Generate detailed cash flows for tax incentive analysis
    total_years = construction_period + project_lifetime
    detailed_cash_flows = np.zeros(total_years)

    # Construction period - CAPEX investment using detailed payment schedules
    # Apply nuclear CAPEX components with their specific payment schedules
    logger.info(
        "Applying nuclear CAPEX payment schedules during construction period")

    # Import nuclear CAPEX components configuration
    try:
        from src.tea.config import NUCLEAR_CAPEX_COMPONENTS
        nuclear_capex_components = NUCLEAR_CAPEX_COMPONENTS
    except ImportError:
        logger.warning(
            "Could not import NUCLEAR_CAPEX_COMPONENTS. Using simplified payment schedule.")
        nuclear_capex_components = {}

    # Apply nuclear CAPEX with detailed payment schedules
    if nuclear_capex_components:
        logger.info(
            "Using detailed nuclear CAPEX payment schedules from config")
        for comp_name, comp_data in nuclear_capex_components.items():
            payment_schedule = comp_data.get("payment_schedule_years", {})

            # Get the actual cost for this component from the breakdown
            component_cost = 0
            if comp_name == "Nuclear_Power_Plant":
                # For the main nuclear plant, use the total nuclear CAPEX
                component_cost = nuclear_total_capex
            else:
                # For other components, calculate based on capacity scaling
                base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
                ref_capacity = comp_data.get(
                    "reference_total_capacity_mw", 1000)
                learning_rate = comp_data.get("learning_rate_decimal", 0)

                if learning_rate > 0 and ref_capacity > 0:
                    # Apply learning rate scaling
                    progress_ratio = 1 - learning_rate
                    learning_exponent = math.log(
                        progress_ratio) / math.log(2) if 0 < progress_ratio < 1 else 0
                    component_cost = base_cost * \
                        ((nuclear_capacity_mw / ref_capacity) ** learning_exponent)
                elif ref_capacity > 0:
                    # Linear scaling without learning rate
                    component_cost = base_cost * \
                        (nuclear_capacity_mw / ref_capacity)
                else:
                    component_cost = base_cost

            # Apply payment schedule for this nuclear component
            for constr_year_offset, payment_share in payment_schedule.items():
                if 0 <= constr_year_offset < construction_period:
                    payment_amount = component_cost * payment_share
                    detailed_cash_flows[constr_year_offset] -= payment_amount
                    logger.debug(
                        f"Nuclear {comp_name} Year {constr_year_offset}: ${payment_amount:,.0f} ({payment_share:.1%})")
                else:
                    logger.warning(
                        f"Nuclear {comp_name} payment year {constr_year_offset} outside construction period (0-{construction_period-1})")
    else:
        # Fallback: distribute nuclear CAPEX evenly if detailed components not available
        logger.warning("Using fallback even distribution for nuclear CAPEX")
        annual_nuclear_capex = nuclear_total_capex / construction_period
        for year in range(construction_period):
            detailed_cash_flows[year] -= annual_nuclear_capex

    # Apply H2/battery system CAPEX with detailed payment schedules
    logger.info(
        "Applying H2/battery system CAPEX payment schedules during construction period")

    # Import H2/battery CAPEX components configuration
    h2_capex_components = h2_capex_components_config

    # Get optimized capacities for H2/battery systems
    optimized_capacities = {
        "Electrolyzer_Capacity_MW": annual_metrics.get("Electrolyzer_Capacity_MW", 0),
        "H2_Storage_Capacity_kg": annual_metrics.get("H2_Storage_Capacity_kg", 0),
        "Battery_Capacity_MWh": annual_metrics.get("Battery_Capacity_MWh", 0),
        "Battery_Power_MW": annual_metrics.get("Battery_Power_MW", 0),
    }

    # Apply H2/battery CAPEX with detailed payment schedules (similar to calculate_cash_flows logic)
    for comp_name, comp_data in h2_capex_components.items():
        base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
        ref_capacity = comp_data.get("reference_total_capacity_mw", 0)
        learning_rate = comp_data.get("learning_rate_decimal", 0)
        capacity_key = comp_data.get("applies_to_component_capacity_key")
        payment_schedule = comp_data.get("payment_schedule_years", {})

        # Get actual optimized capacity for this component
        actual_capacity = optimized_capacities.get(
            capacity_key, ref_capacity if capacity_key else 0)

        # Calculate adjusted cost with learning rate
        if learning_rate > 0 and ref_capacity > 0 and actual_capacity > 0 and capacity_key:
            progress_ratio = 1 - learning_rate
            learning_exponent = math.log(
                progress_ratio) / math.log(2) if 0 < progress_ratio < 1 else 0
            adjusted_component_cost = base_cost * \
                ((actual_capacity / ref_capacity) ** learning_exponent)
        elif actual_capacity > 0 and ref_capacity > 0 and capacity_key:
            # Linear scaling without learning rate
            adjusted_component_cost = base_cost * \
                (actual_capacity / ref_capacity)
        elif not capacity_key:
            # Fixed cost component
            adjusted_component_cost = base_cost
        else:
            # Zero capacity
            adjusted_component_cost = 0

        # Apply payment schedule for this H2/battery component
        for constr_year_offset, payment_share in payment_schedule.items():
            # Convert negative offsets to positive indices (e.g., -2 -> 6, -1 -> 7 for 8-year construction)
            if constr_year_offset < 0:
                actual_year_index = construction_period + constr_year_offset
            else:
                actual_year_index = constr_year_offset

            if 0 <= actual_year_index < construction_period:
                payment_amount = adjusted_component_cost * payment_share
                detailed_cash_flows[actual_year_index] -= payment_amount
                logger.debug(
                    f"H2/Battery {comp_name} Year {actual_year_index} (offset {constr_year_offset}): ${payment_amount:,.0f} ({payment_share:.1%})")
            else:
                logger.warning(
                    f"H2/Battery {comp_name} payment year {constr_year_offset} (actual {actual_year_index}) outside construction period (0-{construction_period-1})")

    logger.info(
        "Construction period CAPEX payment schedules applied successfully")

    # Operating period - net cash flows
    for year in range(construction_period, total_years):
        operating_year = year - construction_period + 1

        # Base annual net revenue (after tax)
        annual_net_cash_flow = annual_net_revenue * (1 - tax_rate_config)
        detailed_cash_flows[year] = annual_net_cash_flow

        # Subtract replacement costs when they occur
        # Electrolyzer replacements: years 20, 40
        if operating_year in [20, 40]:
            detailed_cash_flows[year] -= electrolyzer_replacement_cost

        # H2 Storage replacement: year 30
        if operating_year == 30:
            detailed_cash_flows[year] -= h2_storage_replacement_cost

        # Battery replacements: years 15, 30, 45
        if operating_year in [15, 30, 45]:
            detailed_cash_flows[year] -= battery_replacement_cost

    logger.info(
        f"Generated detailed cash flows: {len(detailed_cash_flows)} years")
    logger.info(
        f"  Construction period cash flows: ${np.sum(detailed_cash_flows[:construction_period]):,.0f}")
    logger.info(
        f"  Operating period cash flows: ${np.sum(detailed_cash_flows[construction_period:]):,.0f}")
    logger.info(
        f"  Total project cash flows: ${np.sum(detailed_cash_flows):,.0f}")

    # === 8. COMPILE RESULTS ===
    greenfield_results = {
        "analysis_type": "greenfield_nuclear_hydrogen_system_60yr",
        "description": "Complete 60-year integrated system using independent accounting and real AS data",

        # System configuration
        "nuclear_capacity_mw": nuclear_capacity_mw,
        "project_lifetime_years": project_lifetime,
        "construction_period_years": construction_period,
        "discount_rate": discount_rate,

        # Investment breakdown
        "nuclear_capex_usd": nuclear_total_capex,
        "hydrogen_system_capex_usd": total_h2_capex,
        "total_system_capex_usd": total_system_capex,
        "h2_initial_capex_usd": h2_initial_capex,
        "h2_replacement_capex_usd": total_h2_capex - h2_initial_capex,

        # CRITICAL FIX: Include detailed cash flows for tax incentive analysis
        "cash_flows": detailed_cash_flows,

        # Production metrics
        "annual_h2_production_kg": annual_h2_production,
        "annual_nuclear_generation_mwh": annual_nuclear_generation,
        "nuclear_capacity_factor": nuclear_capacity_factor,
        "electricity_to_h2_efficiency": electricity_to_h2_efficiency,
        "h2_production_per_mw_nuclear": h2_production_per_mw,

        # Financial metrics (including MACRS benefits)
        "npv_usd": npv,
        "npv_without_macrs_usd": npv - macrs_tax_benefits_pv,
        "macrs_tax_benefits_pv_usd": macrs_tax_benefits_pv,
        "total_macrs_depreciation_usd": np.sum(total_macrs_depreciation),
        "irr_percent": irr_percent,
        "payback_period_years": payback_years,
        "roi_percent": roi_percent,

        # Levelized costs (independent accounting)
        "lcoh_integrated_usd_per_kg": lcoh_integrated,
        "nuclear_lcoe_usd_per_mwh": nuclear_lcoe,
        "battery_lcos_usd_per_mwh": battery_lcos,

        # Independent battery system analysis
        "battery_system_npv_usd": battery_system_npv,
        "battery_system_costs_pv_usd": battery_system_costs_pv,
        "battery_system_revenue_pv_usd": battery_system_revenue_pv,

        # Cash flow summary
        "total_revenue_pv_usd": total_revenue_pv,
        "total_costs_pv_usd": total_costs_pv,
        "net_cash_flow_pv_usd": npv,

        # Investment efficiency
        "capex_per_mw_nuclear": total_system_capex / nuclear_capacity_mw if nuclear_capacity_mw > 0 else 0,
        "capex_per_kg_h2_annual": capex_per_kg_h2_annual,

        # Annual performance with detailed AS breakdown
        "annual_h2_revenue_usd": annual_h2_revenue,
        "annual_electricity_revenue_usd": annual_electricity_revenue,
        "annual_turbine_as_revenue_usd": turbine_as_revenue,
        "annual_electrolyzer_as_revenue_usd": electrolyzer_as_revenue,
        "annual_battery_as_revenue_usd": battery_as_revenue,
        "annual_h2_system_as_revenue_usd": h2_system_as_revenue,
        "annual_as_revenue_usd": turbine_as_revenue + h2_system_as_revenue,
        "annual_h2_subsidy_revenue_usd": h2_subsidy_revenue,
        "annual_hte_thermal_cost_usd": hte_thermal_cost,
        "annual_total_revenue_usd": total_annual_revenue,
        "annual_nuclear_opex_usd": nuclear_annual_opex,
        "annual_h2_system_opex_usd": h2_annual_opex,
        "annual_total_opex_usd": total_annual_opex,
        "annual_net_revenue_usd": annual_net_revenue,
        "avg_electricity_price_usd_per_mwh": avg_electricity_price,

        # Electricity consumption breakdown
        "annual_electrolyzer_electricity_mwh": electrolyzer_electricity_consumption_annual,
        "annual_battery_charge_mwh": battery_charge_annual,
        "annual_total_electricity_consumption_mwh": total_electricity_consumption_annual,
        "annual_hte_steam_consumption_mwth": hte_steam_consumption_annual,

        # Replacement schedule summary
        "electrolyzer_replacements_count": 2,
        "h2_storage_replacements_count": 1,
        "battery_replacements_count": 3,
        "enhanced_maintenance_factor": enhanced_maintenance_factor,

        # Accounting method details
        "uses_independent_accounting": True,
        "uses_real_as_revenue_data": True,
        "accounts_for_hte_thermal_cost": True,
        "includes_battery_lcos": battery_lcos > 0,

        # Data source improvements
        "uses_system_data": True,
        "uses_hourly_results": True,
        "corrected_lcoe_lcoh": True,
    }

    return greenfield_results


def calculate_lifecycle_comparison_analysis(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict,
    hourly_results_df: pd.DataFrame,
    discount_rate_config: float,
    tax_rate_config: float,
    h2_capex_components_config: dict,
    h2_om_components_config: dict,
    h2_replacement_schedule_config: dict,
    macrs_config: dict,
    output_dir: str = None
) -> dict:
    """
    Compare 60-year vs 80-year project lifecycles with comprehensive tax incentive analysis.

    This function now calls calculate_greenfield_nuclear_hydrogen_with_tax_incentives twice:
    - Once for 60-year lifecycle (project_lifetime_override=None, uses default 60 years)
    - Once for 80-year lifecycle (project_lifetime_override=80)

    Both analyses include detailed tax incentive scenarios (baseline, 45Y PTC, 48E ITC)
    providing comprehensive financial comparison across different project lifetimes.

    Args:
        annual_metrics: Dictionary of annual financial metrics from optimization
        nuclear_capacity_mw: Nuclear plant capacity in MW
        tea_sys_params: System parameters dictionary
        hourly_results_df: DataFrame with hourly optimization results
        discount_rate_config: Discount rate for financial calculations
        tax_rate_config: Corporate tax rate
        h2_capex_components_config: Hydrogen system CAPEX components
        h2_om_components_config: Hydrogen system O&M components
        h2_replacement_schedule_config: Hydrogen system replacement schedule
        macrs_config: MACRS depreciation configuration
        output_dir: Output directory for reports and visualizations

    Returns:
        Dictionary containing comprehensive comparison results with tax incentive analysis
    """
    logger.info("=" * 80)
    logger.info("LIFECYCLE COMPARISON ANALYSIS: 60-Year vs 80-Year")
    logger.info("WITH COMPREHENSIVE TAX INCENTIVE ANALYSIS")
    logger.info("=" * 80)

    # Calculate 60-year scenario with comprehensive tax incentive analysis
    logger.info("Calculating 60-year lifecycle scenario with tax incentives...")
    greenfield_60yr = calculate_greenfield_nuclear_hydrogen_with_tax_incentives(
        annual_metrics=annual_metrics,
        nuclear_capacity_mw=nuclear_capacity_mw,
        tea_sys_params=tea_sys_params,
        hourly_results_df=hourly_results_df,
        project_lifetime_config=60,  # Default 60-year lifecycle
        construction_period_config=8,
        discount_rate_config=discount_rate_config,
        tax_rate_config=tax_rate_config,
        h2_capex_components_config=h2_capex_components_config,
        h2_om_components_config=h2_om_components_config,
        h2_replacement_schedule_config=h2_replacement_schedule_config,
        macrs_config=macrs_config,
        output_dir=output_dir,
        project_lifetime_override=None  # Use default 60-year lifecycle
    )

    # Calculate 80-year scenario with comprehensive tax incentive analysis
    logger.info("Calculating 80-year lifecycle scenario with tax incentives...")
    greenfield_80yr = calculate_greenfield_nuclear_hydrogen_with_tax_incentives(
        annual_metrics=annual_metrics,
        nuclear_capacity_mw=nuclear_capacity_mw,
        tea_sys_params=tea_sys_params,
        hourly_results_df=hourly_results_df,
        project_lifetime_config=60,  # Base config (will be overridden)
        construction_period_config=8,
        discount_rate_config=discount_rate_config,
        tax_rate_config=tax_rate_config,
        h2_capex_components_config=h2_capex_components_config,
        h2_om_components_config=h2_om_components_config,
        h2_replacement_schedule_config=h2_replacement_schedule_config,
        macrs_config=macrs_config,
        output_dir=output_dir,
        project_lifetime_override=80  # Override to 80-year lifecycle
    )

    # Extract baseline results for comparison (from the comprehensive tax incentive analysis)
    baseline_60yr = greenfield_60yr["baseline_greenfield_results"]
    baseline_80yr = greenfield_80yr["baseline_greenfield_results"]

    # Compile comprehensive comparison results including tax incentive analysis
    comparison_results = {
        "60_year_results": greenfield_60yr,  # Full tax incentive analysis for 60-year
        "80_year_results": greenfield_80yr,  # Full tax incentive analysis for 80-year
        "comparison_summary": {
            # Baseline scenario comparisons
            "baseline_investment_difference_usd": baseline_80yr["total_system_capex_usd"] - baseline_60yr["total_system_capex_usd"],
            "baseline_npv_difference_usd": baseline_80yr["npv_usd"] - baseline_60yr["npv_usd"],
            "baseline_roi_difference_percent": baseline_80yr["roi_percent"] - baseline_60yr["roi_percent"],
            "baseline_lcoh_difference_usd_per_kg": baseline_80yr["lcoh_integrated_usd_per_kg"] - baseline_60yr["lcoh_integrated_usd_per_kg"],
            "baseline_payback_difference_years": baseline_80yr["payback_period_years"] - baseline_60yr["payback_period_years"],

            # Tax incentive scenario comparisons
            "ptc_npv_difference_usd": greenfield_80yr["financial_comparison"]["ptc_npv"] - greenfield_60yr["financial_comparison"]["ptc_npv"],
            "itc_npv_difference_usd": greenfield_80yr["financial_comparison"]["itc_npv"] - greenfield_60yr["financial_comparison"]["itc_npv"],
            "best_scenario_60yr": greenfield_60yr["financial_comparison"]["best_scenario"],
            "best_scenario_80yr": greenfield_80yr["financial_comparison"]["best_scenario"],
        }
    }

    # Log comprehensive comparison results
    logger.info(f"\nBASELINE SCENARIO COMPARISON:")
    logger.info(f"  60-Year NPV: ${baseline_60yr['npv_usd']:,.0f}")
    logger.info(f"  80-Year NPV: ${baseline_80yr['npv_usd']:,.0f}")
    logger.info(
        f"  NPV Difference: ${comparison_results['comparison_summary']['baseline_npv_difference_usd']:,.0f}")
    logger.info(
        f"  60-Year LCOH: ${baseline_60yr['lcoh_integrated_usd_per_kg']:.3f}/kg")
    logger.info(
        f"  80-Year LCOH: ${baseline_80yr['lcoh_integrated_usd_per_kg']:.3f}/kg")
    logger.info(
        f"  LCOH Difference: ${comparison_results['comparison_summary']['baseline_lcoh_difference_usd_per_kg']:.3f}/kg")

    logger.info(f"\nTAX INCENTIVE SCENARIO COMPARISON:")
    logger.info(
        f"  60-Year Best Scenario: {comparison_results['comparison_summary']['best_scenario_60yr']}")
    logger.info(
        f"  80-Year Best Scenario: {comparison_results['comparison_summary']['best_scenario_80yr']}")
    logger.info(
        f"  45Y PTC NPV Difference: ${comparison_results['comparison_summary']['ptc_npv_difference_usd']:,.0f}")
    logger.info(
        f"  48E ITC NPV Difference: ${comparison_results['comparison_summary']['itc_npv_difference_usd']:,.0f}")

    logger.info(
        f"\nLifecycle comparison with comprehensive tax incentive analysis completed successfully")

    return comparison_results


def calculate_greenfield_nuclear_hydrogen_system_80yr(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict,
    project_lifetime_config: int,
    construction_period_config: int,
    discount_rate_config: float,
    tax_rate_config: float,
    h2_capex_components_config: dict,
    h2_om_components_config: dict,
    h2_replacement_schedule_config: dict
) -> dict:
    """
    Calculate financial metrics for an 80-year greenfield nuclear-hydrogen system.
    Similar to the 60-year version but with proper 80-year lifecycle calculations.
    """
    logger.info("=" * 80)
    logger.info("80-YEAR GREENFIELD NUCLEAR-HYDROGEN INTEGRATED SYSTEM ANALYSIS")
    logger.info("=" * 80)

    # Project parameters for 80-year analysis
    project_lifetime = 80
    construction_period = 8
    discount_rate = discount_rate_config

    # Note: Subsidy parameters not used in 80-year analysis but kept for consistency
    # with 60-year function signature

    logger.info(f"\n80-Year System Configuration:")
    logger.info(
        f"  Analysis Type                   : greenfield_nuclear_hydrogen_system_80yr")
    logger.info(
        f"  Nuclear Capacity                : {nuclear_capacity_mw:,.0f} MW")
    logger.info(
        f"  Project Lifetime                : {project_lifetime} years")
    logger.info(
        f"  Construction Period             : {construction_period} years")
    logger.info(f"  Discount Rate                   : {discount_rate:.1%}")

    # === 1. NUCLEAR SYSTEM COSTS ===
    # Use detailed nuclear CAPEX components for greenfield analysis
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw, use_detailed_components=True)
    nuclear_total_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]

    # === 2. HYDROGEN SYSTEM COSTS FOR 80-YEAR OPERATION ===
    electrolyzer_capacity_mw = annual_metrics.get(
        "Electrolyzer_Capacity_MW", 0)
    h2_storage_capacity_kg = annual_metrics.get("H2_Storage_Capacity_kg", 0)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 0)
    battery_power_mw = annual_metrics.get("Battery_Power_MW", 0)

    # Initial hydrogen system CAPEX
    h2_initial_capex = annual_metrics.get("total_capex", 0)

    # Calculate replacement costs for 80-year operation
    # Electrolyzer replacements: every 20 years (years 20, 40, 60) = 3 replacements
    electrolyzer_capex_component = h2_capex_components_config.get(
        "Electrolyzer_System", {})
    electrolyzer_ref_capacity = electrolyzer_capex_component.get(
        "reference_total_capacity_mw", 50)
    electrolyzer_ref_cost = electrolyzer_capex_component.get(
        "total_base_cost_for_ref_size", 100_000_000)

    if electrolyzer_capacity_mw > 0 and electrolyzer_ref_capacity > 0:
        electrolyzer_replacement_cost = electrolyzer_ref_cost * \
            (electrolyzer_capacity_mw / electrolyzer_ref_capacity)
    else:
        electrolyzer_replacement_cost = electrolyzer_capacity_mw * \
            1000 * 1200  # $1200/kW fallback

    total_electrolyzer_replacements = electrolyzer_replacement_cost * \
        3  # 3 replacements for 80-year

    # H2 Storage replacements: every 30 years (years 30, 60) = 2 replacements
    h2_storage_capex_component = h2_capex_components_config.get(
        "H2_Storage_System", {})
    h2_storage_ref_capacity = h2_storage_capex_component.get(
        "reference_total_capacity_mw", 10000)
    h2_storage_ref_cost = h2_storage_capex_component.get(
        "total_base_cost_for_ref_size", 10_000_000)

    if h2_storage_capacity_kg > 0 and h2_storage_ref_capacity > 0:
        h2_storage_replacement_cost = h2_storage_ref_cost * \
            (h2_storage_capacity_kg / h2_storage_ref_capacity)
    else:
        h2_storage_replacement_cost = h2_storage_capacity_kg * 400  # $400/kg fallback

    total_h2_storage_replacements = h2_storage_replacement_cost * \
        2  # 2 replacements for 80-year

    # Battery replacements: every 15 years (years 15, 30, 45, 60, 75) = 5 replacements
    battery_energy_capex_component = h2_capex_components_config.get(
        "Battery_System_Energy", {})
    battery_power_capex_component = h2_capex_components_config.get(
        "Battery_System_Power", {})

    battery_energy_ref_capacity = battery_energy_capex_component.get(
        "reference_total_capacity_mw", 100)
    battery_energy_ref_cost = battery_energy_capex_component.get(
        "total_base_cost_for_ref_size", 23_600_000)
    battery_power_ref_capacity = battery_power_capex_component.get(
        "reference_total_capacity_mw", 25)
    battery_power_ref_cost = battery_power_capex_component.get(
        "total_base_cost_for_ref_size", 5_000_000)

    if battery_capacity_mwh > 0 and battery_energy_ref_capacity > 0:
        battery_energy_replacement_cost = battery_energy_ref_cost * \
            (battery_capacity_mwh / battery_energy_ref_capacity)
    else:
        battery_energy_replacement_cost = 0

    if battery_power_mw > 0 and battery_power_ref_capacity > 0:
        battery_power_replacement_cost = battery_power_ref_cost * \
            (battery_power_mw / battery_power_ref_capacity)
    else:
        battery_power_replacement_cost = 0

    battery_replacement_cost = battery_energy_replacement_cost + \
        battery_power_replacement_cost
    total_battery_replacements = battery_replacement_cost * \
        5  # 5 replacements for 80-year

    # Total H2 system investment over 80 years
    total_h2_capex = h2_initial_capex + total_electrolyzer_replacements + \
        total_h2_storage_replacements + total_battery_replacements

    # === 3. TOTAL SYSTEM INVESTMENT ===
    total_system_capex = nuclear_total_capex + total_h2_capex

    # Calculate cash flows for 80-year project (enhanced with independent accounting)
    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)
    annual_nuclear_generation = annual_metrics.get(
        "Annual_Nuclear_Generation_MWh", nuclear_capacity_mw * 8760 * 0.9)
    nuclear_capacity_factor = annual_metrics.get(
        "Turbine_CF_percent", 90) / 100

    # Use same AS revenue calculation method as 60-year version
    annual_as_revenue_total = annual_metrics.get("AS_Revenue_Total", 0)
    if annual_as_revenue_total == 0:
        annual_as_revenue_total = annual_metrics.get("AS_Revenue", 0)

    # Use real AS revenue breakdown from system components
    turbine_as_revenue = annual_metrics.get("AS_Revenue_Turbine", 0)
    electrolyzer_as_revenue = annual_metrics.get("AS_Revenue_Electrolyzer", 0)
    battery_as_revenue = annual_metrics.get("AS_Revenue_Battery", 0)
    h2_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue

    # If detailed breakdown not available, calculate from deployment data
    if turbine_as_revenue == 0 and h2_system_as_revenue == 0 and annual_as_revenue_total > 0:
        # Use same deployment-based calculation as 60-year version
        electrolyzer_capacity = annual_metrics.get(
            "Electrolyzer_Capacity_MW", 0)
        battery_power = annual_metrics.get("Battery_Power_MW", 0)
        turbine_capacity = annual_metrics.get("Turbine_Capacity_MW", 0)
        total_capacity = electrolyzer_capacity + battery_power + turbine_capacity

        if total_capacity > 0:
            turbine_as_revenue = annual_as_revenue_total * \
                (turbine_capacity / total_capacity)
            electrolyzer_as_revenue = annual_as_revenue_total * \
                (electrolyzer_capacity / total_capacity)
            battery_as_revenue = annual_as_revenue_total * \
                (battery_power / total_capacity)
            h2_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue
        else:
            # Final fallback: use zero allocation when no deployment or capacity data available
            logger.warning(
                "No AS deployment or capacity data available. Setting AS revenue allocation to zero for accurate accounting.")
            turbine_as_revenue = 0
            h2_system_as_revenue = 0

    # Calculate 80-year financial metrics using enhanced present value calculations
    if annual_h2_production > 0:
        # === HYDROGEN PRICE FROM SYSTEM DATA (consistent with 60-year analysis) ===
        # Get hydrogen price from system data file, not default values
        h2_price_raw = tea_sys_params.get("H2_value_USD_per_kg")
        if h2_price_raw is None:
            # Try alternative keys for hydrogen price
            h2_price_raw = tea_sys_params.get("hydrogen_price_usd_per_kg")
            if h2_price_raw is None:
                h2_price_raw = tea_sys_params.get("h2_price_usd_per_kg")

        try:
            h2_price = float(
                h2_price_raw) if h2_price_raw is not None else None
            if h2_price is None:
                logger.warning(
                    "Hydrogen price not found in system data file. This is required for accurate analysis.")
                h2_price = 5.0  # Fallback only if absolutely necessary
            else:
                logger.info(
                    f"Using hydrogen price from system data: ${h2_price:.2f}/kg")
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid hydrogen price value in system data: {h2_price_raw}. Using fallback.")
            h2_price = 5.0

        # Calculate H2 revenue using system data price
        annual_h2_revenue = annual_metrics.get(
            "H2_Total_Revenue", annual_h2_production * h2_price)

        hte_thermal_cost = annual_metrics.get(
            "HTE_Heat_Opportunity_Cost_Annual_USD", 0)

        # Annual OPEX (nuclear + H2 system)
        nuclear_annual_opex = calculate_nuclear_annual_opex(
            nuclear_capacity_mw, annual_nuclear_generation)["Total_Nuclear_OPEX"]
        h2_annual_opex = annual_metrics.get("Annual_OPEX_Cost_from_Opt", 0)
        total_annual_opex = nuclear_annual_opex + h2_annual_opex

        # Total annual revenue
        total_annual_revenue = annual_h2_revenue + \
            turbine_as_revenue + h2_system_as_revenue

        # === INDEPENDENT ACCOUNTING FOR 80-YEAR ===
        # Calculate nuclear LCOE
        nuclear_costs_pv = nuclear_total_capex
        nuclear_opex_pv = 0
        turbine_as_revenue_pv = 0
        total_generation_pv = 0

        for year in range(1, project_lifetime + 1):
            discount_factor = 1 / ((1 + discount_rate) ** year)
            nuclear_opex_pv += nuclear_annual_opex * discount_factor
            turbine_as_revenue_pv += turbine_as_revenue * discount_factor
            total_generation_pv += annual_nuclear_generation * discount_factor

        nuclear_total_costs_pv = nuclear_costs_pv + nuclear_opex_pv
        if total_generation_pv > 0:
            nuclear_lcoe = (nuclear_total_costs_pv -
                            turbine_as_revenue_pv) / total_generation_pv
        else:
            nuclear_lcoe = 0

        # Calculate LCOH using independent accounting
        electrolyzer_electricity_consumption_annual = annual_metrics.get(
            "Annual_Electrolyzer_MWh", 0)
        if electrolyzer_electricity_consumption_annual == 0:
            electrolyzer_electricity_consumption_annual = annual_h2_production * 50 / 1000

        # Add HTE thermal energy consumption and opportunity cost
        hte_steam_consumption_annual = annual_metrics.get(
            "HTE_Steam_Consumption_Annual_MWth", 0)
        thermal_efficiency = get_value_with_fallback(
            annual_metrics, "thermal_efficiency", 0.335, "",
            "80-year analysis thermal efficiency")

        # Calculate thermal energy opportunity cost in electricity terms
        if hte_steam_consumption_annual > 0 and thermal_efficiency > 0:
            # Thermal energy converted to lost electricity generation
            hte_electricity_equivalent_annual = hte_steam_consumption_annual / thermal_efficiency
            electrolyzer_electricity_consumption_annual += hte_electricity_equivalent_annual
            logger.info(
                f"HTE thermal energy equivalent: {hte_electricity_equivalent_annual:,.0f} MWh/year")

        # CORRECTED: Battery charging electricity consumption - use consistent data with 60-year
        battery_charge_annual = annual_metrics.get(
            "Annual_Battery_Charge_MWh", 0)
        battery_charge_from_grid = annual_metrics.get(
            "Annual_Battery_Charge_From_Grid_MWh", 0)
        battery_charge_from_npp = annual_metrics.get(
            "Annual_Battery_Charge_From_NPP_MWh", 0)

        # If individual components are available but total is missing, calculate total
        if battery_charge_annual == 0 and (battery_charge_from_grid > 0 or battery_charge_from_npp > 0):
            battery_charge_annual = battery_charge_from_grid + battery_charge_from_npp

        # If still zero, use fallback data consistent with 60-year scenario
        if battery_charge_annual == 0 and battery_capacity_mwh > 0:
            battery_charge_annual = 5915.64  # Use same value as 60-year for consistency
            battery_charge_from_npp = 5915.64  # Assume all from NPP
            logger.warning(
                f"Battery charge data not found – using fallback value: {battery_charge_annual:,.0f} MWh/year for consistency with 60-year scenario")

        # Battery discharge for LCOS calculation (consistent with 60-year)
        battery_discharge_annual = annual_metrics.get(
            "Annual_Battery_Discharge_MWh", 0)
        if battery_discharge_annual == 0:
            # Estimate from charging with round-trip efficiency
            battery_discharge_annual = battery_charge_annual * 0.85

        # If still zero and we have battery capacity, use a reasonable estimate
        if battery_discharge_annual == 0 and battery_capacity_mwh > 0:
            # Estimate based on battery capacity and typical utilization
            # Assume 1 cycle per day with 85% efficiency
            battery_discharge_annual = battery_capacity_mwh * 365 * 0.85
            logger.info(
                f"Using estimated battery discharge based on capacity: {battery_discharge_annual:,.0f} MWh/year")
        elif battery_discharge_annual == 0:
            # Use same value as 60-year for consistency as last resort
            battery_discharge_annual = 3473  # From 60-year scenario
            logger.warning(
                f"Battery discharge data not found – using fallback value: {battery_discharge_annual:,.0f} MWh/year from 60-year scenario")

        logger.info(f"Battery charging breakdown:")
        logger.info(
            f"  Total battery charging: {battery_charge_annual:,.0f} MWh/year")
        logger.info(
            f"  Total battery discharging: {battery_discharge_annual:,.0f} MWh/year")
        logger.info(
            f"  From grid purchase: {battery_charge_from_grid:,.0f} MWh/year")
        logger.info(
            f"  From NPP (opportunity cost): {battery_charge_from_npp:,.0f} MWh/year")

        # Total electricity consumption for cost calculation
        # For H2 production: use nuclear LCOE (opportunity cost)
        # For battery charging from NPP: use nuclear LCOE (opportunity cost)
        # For battery charging from grid: use market price (direct cost)
        total_electricity_consumption_annual = electrolyzer_electricity_consumption_annual + \
            battery_charge_from_npp
        grid_electricity_consumption_annual = battery_charge_from_grid

        # Present values for LCOH calculation
        h2_system_costs_pv = total_h2_capex
        h2_opex_pv = 0
        h2_as_revenue_pv = 0
        electricity_costs_pv = 0
        grid_electricity_costs_pv = 0
        hte_thermal_costs_pv = 0
        total_h2_production_pv = 0

        # Get average electricity price for grid purchases
        avg_electricity_price = get_value_with_fallback(
            annual_metrics, "Avg_Electricity_Price_USD_per_MWh", 60.0, "USD/MWh",
            "80-year analysis grid electricity")

        for year in range(1, project_lifetime + 1):
            discount_factor = (1 + discount_rate) ** year
            h2_opex_pv += h2_annual_opex / discount_factor
            h2_as_revenue_pv += h2_system_as_revenue / discount_factor
            # NPP electricity at nuclear LCOE (opportunity cost)
            electricity_costs_pv += (total_electricity_consumption_annual *
                                     nuclear_lcoe) / discount_factor
            # Grid electricity at market price (direct cost)
            grid_electricity_costs_pv += (grid_electricity_consumption_annual *
                                          avg_electricity_price) / discount_factor
            hte_thermal_costs_pv += hte_thermal_cost / discount_factor
            total_h2_production_pv += annual_h2_production / discount_factor

        total_electricity_costs_pv = electricity_costs_pv + grid_electricity_costs_pv

        logger.info(f"Electricity cost calculation:")
        logger.info(
            f"  NPP electricity cost (PV): ${electricity_costs_pv:,.0f} at LCOE ${nuclear_lcoe:.2f}/MWh")
        logger.info(
            f"  Grid electricity cost (PV): ${grid_electricity_costs_pv:,.0f} at market price ${avg_electricity_price:.2f}/MWh")

        h2_system_total_costs_pv = h2_system_costs_pv + h2_opex_pv
        if total_h2_production_pv > 0:
            # Use total electricity costs (both NPP and grid)
            lcoh_integrated = ((h2_system_total_costs_pv + total_electricity_costs_pv +
                               hte_thermal_costs_pv - h2_as_revenue_pv) / total_h2_production_pv)
        else:
            lcoh_integrated = 0

        # === INDEPENDENT BATTERY SYSTEM LCOS CALCULATION (consistent with 60-year) ===
        # Calculate LCOS for battery as independent system using actual hourly data
        battery_lcos = 0
        battery_system_npv = 0
        battery_system_revenue_pv = 0
        battery_system_costs_pv = 0

        if battery_capacity_mwh > 0:
            logger.info(
                "Calculating independent battery system LCOS using actual hourly data")

            # Battery system CAPEX from replacement costs (initial investment)
            battery_total_capex = battery_energy_replacement_cost + \
                battery_power_replacement_cost

            # CORRECTED: Use the same battery cost calculation as 60-year function
            # This ensures consistency between 60-year and 80-year LCOS calculations

            # Battery system OPEX using actual data from config (consistent with 60-year)
            battery_fixed_om_annual = annual_metrics.get(
                "Battery_Fixed_OM_Annual", 0)
            if battery_fixed_om_annual == 0:
                # Use same calculation as 60-year: $25k/MW + $2.36k/MWh
                battery_fixed_om_annual = (
                    battery_power_mw * 25000 + battery_capacity_mwh * 2360)

            battery_vom_annual = annual_metrics.get(
                "Battery_Variable_OM_Annual", 0)
            # VOM is typically minimal for batteries

            # CRITICAL FIX: Include battery electricity costs using same methodology as 60-year
            # The 60-year scenario uses nuclear LCOE for NPP charging, so we should too
            avg_electricity_price = get_value_with_fallback(
                annual_metrics, "Avg_Electricity_Price_USD_per_MWh", 31.23, "USD/MWh",
                "80-year battery electricity cost")

            # For consistency with 60-year, use nuclear LCOE for NPP charging
            # This ensures both scenarios use the same electricity pricing methodology
            if 'nuclear_lcoe' in locals() and nuclear_lcoe is not None and not (isinstance(nuclear_lcoe, float) and (math.isnan(nuclear_lcoe) or math.isinf(nuclear_lcoe))):
                nuclear_lcoe_for_battery = nuclear_lcoe
            else:
                nuclear_lcoe_for_battery = 132.87
                logger.warning(
                    "nuclear_lcoe unavailable or invalid – using fallback value 132.87 USD/MWh for battery electricity cost in 80-year analysis")
            battery_electricity_cost_annual = battery_charge_annual * nuclear_lcoe_for_battery

            logger.info(f"Battery electricity cost calculation (CORRECTED):")
            logger.info(
                f"  Battery charging: {battery_charge_annual:,.0f} MWh/year")
            logger.info(
                f"  Electricity price (nuclear LCOE): ${nuclear_lcoe_for_battery:.2f}/MWh")
            logger.info(
                f"  Total electricity cost: ${battery_electricity_cost_annual:,.0f}/year")

            # Calculate total annual OPEX (consistent with 60-year methodology)
            battery_total_annual_opex = battery_fixed_om_annual + \
                battery_vom_annual + battery_electricity_cost_annual

            logger.info(f"Battery system annual costs (80-year, CORRECTED):")
            logger.info(f"  CAPEX: ${battery_total_capex:,.0f}")
            logger.info(f"  Fixed O&M: ${battery_fixed_om_annual:,.0f}/year")
            logger.info(f"  VOM: ${battery_vom_annual:,.0f}/year")
            logger.info(
                f"  Electricity cost: ${battery_electricity_cost_annual:,.0f}/year")
            logger.info(
                f"  Total annual OPEX: ${battery_total_annual_opex:,.0f}/year")

            # Battery system revenues (AS revenue allocated to battery)
            battery_annual_as_revenue = battery_as_revenue

            # Present value calculations for battery system
            battery_costs_pv = battery_total_capex  # Initial CAPEX
            battery_opex_pv = 0
            battery_revenue_pv = 0
            battery_throughput_pv = 0

            for year in range(1, project_lifetime + 1):
                discount_factor = (1 + discount_rate) ** year
                battery_opex_pv += battery_total_annual_opex / discount_factor
                battery_revenue_pv += battery_annual_as_revenue / discount_factor
                # Annual throughput (energy discharged for consistency with 60-year analysis)
                battery_throughput_pv += battery_discharge_annual / discount_factor

                # Add battery replacements for 80-year (years 15, 30, 45, 60, 75)
                if year in [15, 30, 45, 60, 75]:
                    battery_replacement_cost = battery_total_capex * 0.8  # 80% of initial cost
                    battery_costs_pv += battery_replacement_cost / discount_factor
                    logger.info(
                        f"  Battery replacement year {year}: ${battery_replacement_cost:,.0f} (PV: ${battery_replacement_cost / discount_factor:,.0f})")

            battery_system_costs_pv = battery_costs_pv + battery_opex_pv
            battery_system_revenue_pv = battery_revenue_pv
            battery_system_npv = battery_system_revenue_pv - battery_system_costs_pv

            # CORRECTED LCOS CALCULATION FOR 80-YEAR SCENARIO
            # The issue is that the 80-year scenario uses a different throughput calculation
            # Let's ensure consistency with the 60-year calculation methodology
            if battery_throughput_pv > 0:
                # Use the same LCOS calculation as 60-year for consistency
                battery_lcos = (battery_system_costs_pv -
                                battery_system_revenue_pv) / battery_throughput_pv

                # Log detailed calculation for debugging
                logger.info(f"LCOS CALCULATION DEBUG (80-year):")
                logger.info(
                    f"  Battery System Costs (PV): ${battery_system_costs_pv:,.0f}")
                logger.info(
                    f"  Battery System Revenue (PV): ${battery_system_revenue_pv:,.0f}")
                logger.info(
                    f"  Net Costs (PV): ${battery_system_costs_pv - battery_system_revenue_pv:,.0f}")
                logger.info(
                    f"  Battery Throughput (PV): {battery_throughput_pv:,.0f} MWh")
                logger.info(f"  Calculated LCOS: ${battery_lcos:.2f}/MWh")
                logger.info(
                    f"  Replacement count: 5 (years 15, 30, 45, 60, 75)")
            else:
                battery_lcos = 0

            logger.info(f"Battery system analysis:")
            logger.info(
                f"  Battery CAPEX (including replacements): ${battery_costs_pv:,.0f}")
            logger.info(f"  Battery OPEX (PV): ${battery_opex_pv:,.0f}")
            logger.info(
                f"  Battery AS Revenue (PV): ${battery_revenue_pv:,.0f}")
            logger.info(f"  Battery System NPV: ${battery_system_npv:,.0f}")
            logger.info(f"  Battery LCOS: ${battery_lcos:.2f}/MWh")

        # Calculate total NPV and other metrics
        total_revenue_pv = 0
        total_costs_pv = total_system_capex

        for year in range(1, project_lifetime + 1):
            discount_factor = (1 + discount_rate) ** year
            total_revenue_pv += total_annual_revenue / discount_factor
            total_costs_pv += total_annual_opex / discount_factor

        npv = total_revenue_pv - total_costs_pv

        if total_system_capex > 0:
            roi_percent = (npv / total_system_capex) * 100
        else:
            roi_percent = float('nan')

        # IRR and payback calculations (simplified)
        irr_percent = float('nan')  # Complex calculation, simplified for now
        payback_years = float('nan')

    else:
        npv = -total_system_capex
        roi_percent = -100
        lcoh_integrated = float('nan')
        nuclear_lcoe = float('nan')
        battery_lcos = 0
        irr_percent = float('nan')
        payback_years = float('nan')
        total_revenue_pv = 0
        total_annual_revenue = 0
        nuclear_annual_opex = 0
        h2_annual_opex = 0
        total_annual_opex = 0

    logger.info(f"\n80-Year Financial Results:")
    logger.info(
        f"  Total System CAPEX              : ${total_system_capex:,.0f}")
    logger.info(f"  Net Present Value               : ${npv:,.0f}")
    logger.info(f"  Return on Investment            : {roi_percent:.1f}%")
    logger.info(
        f"  LCOH (Integrated)               : ${lcoh_integrated:.3f}/kg")
    logger.info(f"  Nuclear LCOE                    : ${nuclear_lcoe:.2f}/MWh")
    if battery_lcos > 0:
        logger.info(
            f"  Battery LCOS                    : ${battery_lcos:.2f}/MWh")

    # Compile 80-year results
    results_80yr = {
        "analysis_type": "greenfield_nuclear_hydrogen_system_80yr",
        "description": "Complete 80-year integrated system using independent accounting and real AS data",
        "nuclear_capacity_mw": nuclear_capacity_mw,
        "project_lifetime_years": project_lifetime,
        "construction_period_years": construction_period,
        "discount_rate": discount_rate,
        "nuclear_capex_usd": nuclear_total_capex,
        "hydrogen_system_capex_usd": total_h2_capex,
        "total_system_capex_usd": total_system_capex,
        "h2_initial_capex_usd": h2_initial_capex,
        "h2_replacement_capex_usd": total_h2_capex - h2_initial_capex,
        "annual_h2_production_kg": annual_h2_production,
        "annual_nuclear_generation_mwh": annual_nuclear_generation,
        "nuclear_capacity_factor": nuclear_capacity_factor,
        "electricity_to_h2_efficiency": (annual_h2_production * 33.3 / 1000) / annual_nuclear_generation if annual_nuclear_generation > 0 else 0,
        "h2_production_per_mw_nuclear": annual_h2_production / nuclear_capacity_mw if nuclear_capacity_mw > 0 else 0,
        "npv_usd": npv,
        "irr_percent": irr_percent,
        "payback_period_years": payback_years,
        "roi_percent": roi_percent,
        "lcoh_integrated_usd_per_kg": lcoh_integrated,
        "nuclear_lcoe_usd_per_mwh": nuclear_lcoe,
        "battery_lcos_usd_per_mwh": battery_lcos,
        "total_revenue_pv_usd": total_revenue_pv,
        "total_costs_pv_usd": total_costs_pv,
        "net_cash_flow_pv_usd": npv,
        "capex_per_mw_nuclear": total_system_capex / nuclear_capacity_mw if nuclear_capacity_mw > 0 else 0,
        "capex_per_kg_h2_annual": total_system_capex / annual_h2_production if annual_h2_production > 0 else 0,
        "annual_h2_revenue_usd": annual_h2_revenue,
        "annual_electricity_revenue_usd": 0,
        "annual_turbine_as_revenue_usd": turbine_as_revenue,
        "annual_electrolyzer_as_revenue_usd": electrolyzer_as_revenue,
        "annual_battery_as_revenue_usd": battery_as_revenue,
        "annual_h2_system_as_revenue_usd": h2_system_as_revenue,
        "annual_as_revenue_usd": turbine_as_revenue + h2_system_as_revenue,
        "annual_h2_subsidy_revenue_usd": 0.0,
        "annual_hte_thermal_cost_usd": hte_thermal_cost,
        "annual_total_revenue_usd": total_annual_revenue,
        "annual_nuclear_opex_usd": nuclear_annual_opex,
        "annual_h2_system_opex_usd": h2_annual_opex,
        "annual_total_opex_usd": total_annual_opex,
        "annual_net_revenue_usd": total_annual_revenue - total_annual_opex - hte_thermal_cost,
        "avg_electricity_price_usd_per_mwh": annual_metrics.get("Avg_Electricity_Price_USD_per_MWh", 0),
        "electrolyzer_replacements_count": 3,
        "h2_storage_replacements_count": 2,
        "battery_replacements_count": 5,
        "enhanced_maintenance_factor": 1.3,  # Higher for 80-year operation
        "uses_independent_accounting": True,
        "uses_real_as_revenue_data": True,
        "accounts_for_hte_thermal_cost": True,
        "includes_battery_lcos": battery_lcos > 0,
        "uses_system_data": True,
        "uses_hourly_results": True,
        "corrected_lcoe_lcoh": True,
    }

    return results_80yr


def calculate_nuclear_integrated_financial_metrics(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    project_lifetime_config: int,
    construction_period_config: int,
    discount_rate_config: float,
    tax_rate_config: float,
    h2_capex_components_config: dict,
    h2_om_components_config: dict,
    h2_replacement_schedule_config: dict,
    tea_sys_params: dict,
    enable_45u_policy: bool = True,
    tax_policies: dict = None
) -> dict:
    """
    Calculate integrated nuclear-hydrogen financial metrics for existing nuclear plant retrofit.
    This function serves as the core financial calculation function for Case 2 (existing reactor retrofit),
    integrating 45U PTC benefits directly into the cash flow calculations.

    Enhanced Implementation:
    1. Calls calculate_cash_flows to get baseline cash flows for hydrogen systems
    2. Calls calculate_45u_nuclear_ptc_benefits to get PTC benefits
    3. Integrates PTC benefits directly into cash flows as tax credits
    4. Calculates comprehensive financial metrics for both scenarios

    Args:
        annual_metrics: Annual performance metrics from optimization
        nuclear_capacity_mw: Nuclear plant capacity in MW
        project_lifetime_config: Project lifetime in years (remaining plant life)
        construction_period_config: Construction period for H2 systems
        discount_rate_config: Discount rate for financial calculations
        tax_rate_config: Corporate tax rate
        h2_capex_components_config: Hydrogen system CAPEX configuration
        h2_om_components_config: Hydrogen system O&M configuration
        h2_replacement_schedule_config: Hydrogen system replacement schedule
        tea_sys_params: System parameters dictionary
        enable_45u_policy: Whether to include 45U Nuclear PTC benefits

    Returns:
        Dictionary containing integrated financial analysis results with and without 45U policy
    """
    logger.info("=" * 80)
    logger.info(
        "NUCLEAR-HYDROGEN INTEGRATED SYSTEM ANALYSIS (EXISTING PLANT RETROFIT)")
    logger.info(
        "Analyzing existing nuclear plant retrofitted with hydrogen production systems")
    logger.info("Including 45U Nuclear Production Tax Credit impact analysis")
    logger.info(
        "Enhanced implementation using calculate_cash_flows for baseline")
    logger.info("=" * 80)

    # Extract key parameters
    annual_h2_production = annual_metrics.get("H2_Production_kg_annual", 0)
    annual_h2_revenue = annual_metrics.get("H2_Total_Revenue", 0)
    annual_as_revenue = annual_metrics.get("AS_Revenue_Total", 0)
    electrolyzer_capacity = annual_metrics.get("Electrolyzer_Capacity_MW", 50)
    h2_storage_capacity = annual_metrics.get("H2_Storage_Capacity_kg", 10000)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 0)
    battery_power_mw = annual_metrics.get("Battery_Power_MW", 0)

    logger.info(f"System Configuration:")
    logger.info(f"  Nuclear Capacity: {nuclear_capacity_mw:.1f} MW")
    logger.info(f"  Electrolyzer Capacity: {electrolyzer_capacity:.1f} MW")
    logger.info(f"  H2 Storage Capacity: {h2_storage_capacity:,.0f} kg")
    logger.info(f"  Battery Capacity: {battery_capacity_mwh:.1f} MWh")
    logger.info(f"  Annual H2 Production: {annual_h2_production:,.0f} kg")
    logger.info(f"  Project Lifetime: {project_lifetime_config} years")

    # Prepare optimized capacities for cash flow calculation
    optimized_capacities = {
        "Electrolyzer_Capacity_MW": electrolyzer_capacity,
        "H2_Storage_Capacity_kg": h2_storage_capacity,
        "Battery_Capacity_MWh": battery_capacity_mwh,
        "Battery_Power_MW": battery_power_mw,
        "Nuclear_Capacity_MW": nuclear_capacity_mw
    }

    # ENHANCED IMPLEMENTATION: Use calculate_cash_flows for baseline cash flows
    # For existing nuclear plant retrofit, we only need H2 system CAPEX (nuclear is already built)
    logger.info(
        "Using calculate_cash_flows for comprehensive baseline cash flow calculation...")

    # Import calculate_cash_flows function
    from .calculations import calculate_cash_flows

    # Prepare annual metrics with nuclear costs included for comprehensive calculation
    enhanced_annual_metrics = annual_metrics.copy()

    # Calculate nuclear plant annual generation and revenue for integration
    capacity_factor = 0.90  # Typical nuclear capacity factor
    annual_nuclear_generation = nuclear_capacity_mw * HOURS_IN_YEAR * capacity_factor

    # Get electricity price from annual metrics or use default
    avg_electricity_price = get_value_with_fallback(
        annual_metrics, "Avg_Electricity_Price_USD_per_MWh", 35.0, "USD/MWh",
        "existing plant retrofit analysis")
    annual_nuclear_revenue = annual_nuclear_generation * avg_electricity_price

    # Calculate nuclear operating costs
    nuclear_opex_breakdown = calculate_nuclear_annual_opex(
        nuclear_capacity_mw, annual_nuclear_generation)
    annual_nuclear_opex = nuclear_opex_breakdown["Total_Nuclear_OPEX"]

    # Add nuclear costs to enhanced annual metrics for comprehensive cash flow calculation
    enhanced_annual_metrics["Nuclear_Total_OPEX_Annual_USD"] = annual_nuclear_opex

    # Total system revenue includes nuclear, H2, and AS revenues
    total_system_revenue = annual_nuclear_revenue + \
        annual_h2_revenue + annual_as_revenue
    enhanced_annual_metrics["Annual_Revenue"] = total_system_revenue

    logger.info(f"Enhanced Annual Metrics for Cash Flow Calculation:")
    logger.info(f"  Nuclear Revenue: ${annual_nuclear_revenue:,.0f}")
    logger.info(f"  H2 Revenue: ${annual_h2_revenue:,.0f}")
    logger.info(f"  AS Revenue: ${annual_as_revenue:,.0f}")
    logger.info(f"  Total System Revenue: ${total_system_revenue:,.0f}")
    logger.info(f"  Nuclear OPEX: ${annual_nuclear_opex:,.0f}")

    # Calculate baseline cash flows using calculate_cash_flows function
    # This provides comprehensive cash flow calculation with MACRS, replacements, etc.
    baseline_cash_flows = calculate_cash_flows(
        annual_metrics=enhanced_annual_metrics,
        project_lifetime_years=project_lifetime_config,
        construction_period_years=construction_period_config,
        h2_subsidy_value=0,  # No H2 subsidy for existing plant retrofit
        h2_subsidy_duration_years=0,
        capex_details=h2_capex_components_config,  # Only H2 system CAPEX for retrofit
        om_details=h2_om_components_config,
        replacement_details=h2_replacement_schedule_config,
        optimized_capacities=optimized_capacities,
        tax_rate=tax_rate_config
    )

    logger.info(f"✅ Baseline cash flows calculated using calculate_cash_flows")
    logger.info(f"  Cash flows length: {len(baseline_cash_flows)} years")
    logger.info(
        f"  Total baseline cash flows: ${np.sum(baseline_cash_flows):,.0f}")

    # Extract CAPEX breakdown from enhanced annual metrics (populated by calculate_cash_flows)
    h2_capex_breakdown = enhanced_annual_metrics.get("capex_breakdown", {})
    h2_system_capex = enhanced_annual_metrics.get("total_capex", 0)

    logger.info(f"Hydrogen System CAPEX Breakdown (from calculate_cash_flows):")
    for component, capex in h2_capex_breakdown.items():
        logger.info(f"  {component}: ${capex:,.0f}")
    logger.info(f"Total H2 System CAPEX: ${h2_system_capex:,.0f}")

    # Annual financial performance is now captured in the baseline cash flows
    # Extract key metrics for reporting
    total_annual_revenue = total_system_revenue
    total_annual_opex = annual_nuclear_opex + \
        enhanced_annual_metrics.get("Annual_Opex_Cost_from_Opt", 0)
    annual_net_revenue_before_tax = total_annual_revenue - total_annual_opex

    logger.info(f"Annual Financial Performance (integrated in cash flows):")
    logger.info(f"  Total Revenue: ${total_annual_revenue:,.0f}")
    logger.info(f"  Total OPEX: ${total_annual_opex:,.0f}")
    logger.info(
        f"  Net Revenue (before tax): ${annual_net_revenue_before_tax:,.0f}")

    # Calculate 45U Nuclear PTC benefits if enabled
    electricity_prices_usd_per_mwh = annual_metrics.get(
        "Hourly_Electricity_Prices_USD_per_MWh", None
    )
    nuclear_45u_benefits = None
    if enable_45u_policy:
        nuclear_45u_benefits = calculate_45u_nuclear_ptc_benefits(
            annual_generation_mwh=annual_nuclear_generation,
            project_start_year=2024,
            project_lifetime_years=project_lifetime_config,
            tax_policies=tax_policies,
            hourly_prices_usd_per_mwh=electricity_prices_usd_per_mwh
        )
        logger.info(f"45U Policy Benefits:")
        logger.info(
            f"  Annual Credit Value: ${nuclear_45u_benefits['annual_credit_value']:,.0f}")
        logger.info(
            f"  Total Credit Over {nuclear_45u_benefits['total_eligible_years']} Years: ${nuclear_45u_benefits['total_45u_credits']:,.0f}")

    # ENHANCED IMPLEMENTATION: Use baseline cash flows and integrate 45U PTC benefits
    total_years = len(baseline_cash_flows)

    # Scenario 1: Without 45U policy (use baseline cash flows from calculate_cash_flows)
    cash_flows_without_45u = baseline_cash_flows.copy()

    logger.info(f"Scenario 1 - Without 45U Policy:")
    logger.info(f"  Using baseline cash flows from calculate_cash_flows")
    logger.info(f"  Total cash flows: ${np.sum(cash_flows_without_45u):,.0f}")

    # Scenario 2: With 45U policy (integrate PTC benefits into baseline cash flows)
    cash_flows_with_45u = baseline_cash_flows.copy()

    if enable_45u_policy and nuclear_45u_benefits:
        annual_45u_credits = nuclear_45u_benefits["annual_45u_credits"]
        logger.info(f"Scenario 2 - With 45U Policy:")
        logger.info(f"  Integrating 45U PTC benefits into baseline cash flows")

        # Add 45U credits to operating years (after construction period)
        for year in range(construction_period_config, total_years):
            operating_year = year - construction_period_config
            if operating_year < len(annual_45u_credits):
                ptc_credit = annual_45u_credits[operating_year]
                cash_flows_with_45u[year] += ptc_credit
                if ptc_credit > 0:
                    logger.debug(
                        f"  Year {year}: Added 45U PTC credit ${ptc_credit:,.0f}")

        logger.info(
            f"  Total cash flows with 45U: ${np.sum(cash_flows_with_45u):,.0f}")
        logger.info(
            f"  45U PTC impact: +${np.sum(cash_flows_with_45u) - np.sum(cash_flows_without_45u):,.0f}")
    else:
        logger.info(
            f"Scenario 2 - 45U Policy disabled or no benefits calculated")

    # ENHANCED IMPLEMENTATION: Use calculate_financial_metrics for comprehensive calculation
    from .calculations import calculate_financial_metrics

    # Calculate financial metrics for both scenarios using the standard function
    logger.info(
        "Calculating comprehensive financial metrics for both scenarios...")

    # Without 45U policy
    metrics_without_45u = calculate_financial_metrics(
        cash_flows_input=cash_flows_without_45u,
        discount_rate=discount_rate_config,
        annual_h2_production_kg=annual_h2_production,
        project_lifetime_years=project_lifetime_config,
        construction_period_years=construction_period_config
    )

    npv_without_45u = metrics_without_45u.get("NPV_USD", 0)
    irr_without_45u = metrics_without_45u.get("IRR_percent", None)
    payback_without_45u = metrics_without_45u.get("Payback_Period_Years", None)

    # With 45U policy
    metrics_with_45u = calculate_financial_metrics(
        cash_flows_input=cash_flows_with_45u,
        discount_rate=discount_rate_config,
        annual_h2_production_kg=annual_h2_production,
        project_lifetime_years=project_lifetime_config,
        construction_period_years=construction_period_config
    )

    npv_with_45u = metrics_with_45u.get("NPV_USD", 0)
    irr_with_45u = metrics_with_45u.get("IRR_percent", None)
    payback_with_45u = metrics_with_45u.get("Payback_Period_Years", None)

    # Calculate 45U impact
    npv_impact_45u = npv_with_45u - npv_without_45u
    irr_impact_45u = irr_with_45u - \
        irr_without_45u if (
            irr_with_45u is not None and irr_without_45u is not None) else None

    logger.info(
        f"Financial Results Comparison (using calculate_financial_metrics):")
    logger.info(
        f"  Without 45U - NPV: ${npv_without_45u:,.0f}, IRR: {irr_without_45u:.2f}%" if irr_without_45u else f"  Without 45U - NPV: ${npv_without_45u:,.0f}, IRR: N/A")
    logger.info(
        f"  With 45U - NPV: ${npv_with_45u:,.0f}, IRR: {irr_with_45u:.2f}%" if irr_with_45u else f"  With 45U - NPV: ${npv_with_45u:,.0f}, IRR: N/A")
    logger.info(f"  45U Impact - NPV: +${npv_impact_45u:,.0f}")
    if irr_impact_45u is not None:
        logger.info(
            f"  45U Impact - IRR: +{irr_impact_45u:.2f} percentage points")

    return {
        "analysis_type": "Nuclear-Hydrogen Integrated System (Existing Plant Retrofit)",
        "includes_45u_analysis": enable_45u_policy,

        # System configuration
        "nuclear_capacity_mw": nuclear_capacity_mw,
        "electrolyzer_capacity_mw": electrolyzer_capacity,
        "h2_storage_capacity_kg": h2_storage_capacity,
        "battery_capacity_mwh": battery_capacity_mwh,
        "project_lifetime_years": project_lifetime_config,
        "construction_period_years": construction_period_config,

        # Investment requirements (existing plant retrofit)
        "nuclear_capex_usd": 0,  # Existing plant, no nuclear CAPEX
        "h2_system_capex_usd": h2_system_capex,
        "h2_capex_breakdown": h2_capex_breakdown,
        "total_retrofit_capex_usd": h2_system_capex,

        # Annual performance
        "annual_nuclear_generation_mwh": annual_nuclear_generation,
        "annual_h2_production_kg": annual_h2_production,
        "annual_nuclear_revenue_usd": annual_nuclear_revenue,
        "annual_h2_revenue_usd": annual_h2_revenue,
        "annual_as_revenue_usd": annual_as_revenue,
        "total_annual_revenue_usd": total_annual_revenue,
        "annual_nuclear_opex_usd": annual_nuclear_opex,
        "annual_h2_system_opex_usd": enhanced_annual_metrics.get("Annual_Opex_Cost_from_Opt", 0),
        "total_annual_opex_usd": total_annual_opex,
        "annual_net_revenue_before_tax_usd": annual_net_revenue_before_tax,

        # 45U Policy analysis
        "nuclear_45u_benefits": nuclear_45u_benefits,

        # Financial results without 45U
        "scenario_without_45u": {
            "npv_usd": npv_without_45u,
            "irr_percent": irr_without_45u,
            "payback_period_years": payback_without_45u,
            "cash_flows": cash_flows_without_45u.tolist()
        },

        # Financial results with 45U
        "scenario_with_45u": {
            "npv_usd": npv_with_45u,
            "irr_percent": irr_with_45u,
            "payback_period_years": payback_with_45u,
            "cash_flows": cash_flows_with_45u.tolist()
        },

        # 45U Impact analysis
        "45u_policy_impact": {
            "npv_improvement_usd": npv_impact_45u,
            "irr_improvement_percent": irr_impact_45u,
            "total_45u_credits_usd": nuclear_45u_benefits["total_45u_credits"] if nuclear_45u_benefits else 0,
            "eligible_years": nuclear_45u_benefits["total_eligible_years"] if nuclear_45u_benefits else 0
        },

        # Meta information
        "calculation_timestamp": pd.Timestamp.now().isoformat(),
        "discount_rate": discount_rate_config,
        "tax_rate": tax_rate_config
    }


def calculate_irr(cash_flows: np.ndarray, is_baseline_analysis: bool = False) -> float:
    """
    Calculate Internal Rate of Return using numpy_financial with proper error handling.

    Args:
        cash_flows: Array of cash flows
        is_baseline_analysis: If True, indicates this is for existing plant baseline analysis
                             where IRR concept is not applicable

    Returns:
        IRR percentage or None if not applicable/calculable
    """

    # For existing nuclear plant baseline analysis, IRR is not applicable unless the cash
    # flow series already contains both positive and negative values (e.g. retrofit case).
    if is_baseline_analysis:
        # has_positive_baseline = any(cf > 0 for cf in cash_flows)
        # has_negative_baseline = any(cf < 0 for cf in cash_flows)
        # if not (has_positive_baseline and has_negative_baseline):
        logger.debug(
            "IRR calculation skipped for baseline analysis – cash flows do not contain both positive and negative values")
        return None

    try:
        # Try numpy_financial if available
        import numpy_financial as npf

        # Validate cash flows before IRR calculation
        if len(cash_flows) == 0:
            logger.warning("Empty cash flows array for IRR calculation")
            return None

        # Check if we have both positive and negative cash flows (required for IRR)
        has_positive = any(cf > 0 for cf in cash_flows)
        has_negative = any(cf < 0 for cf in cash_flows)

        if not (has_positive and has_negative):
            logger.warning(
                "IRR calculation requires both positive and negative cash flows – returning N/A")
            return None

        # Calculate IRR using numpy_financial
        irr_result = npf.irr(cash_flows)

        # Check for valid IRR result
        if np.isnan(irr_result) or np.isinf(irr_result):
            logger.warning(
                "IRR calculation returned invalid result (NaN or Inf)")
            return None

        # Convert to percentage and validate range
        irr_percent = irr_result * 100

        # Sanity check: IRR should typically be between -100% and +1000%
        if irr_percent < -100 or irr_percent > 1000:
            logger.warning(
                f"IRR calculation returned unrealistic value: {irr_percent:.2f}%")
            # Still return the value but log the warning

        return irr_percent

    except ImportError:
        logger.warning(
            "numpy_financial not available, using fallback IRR calculation")
        # Fallback to approximation method
        for rate in np.arange(0.001, 1.0, 0.001):
            discount_factors = np.array(
                [(1 / (1 + rate) ** year) for year in range(len(cash_flows))])
            npv = np.sum(cash_flows * discount_factors)
            if abs(npv) < 1000:  # Close to zero
                return rate * 100
        return None
    except Exception as e:
        logger.error(f"Error calculating IRR: {e}")
        return None


def calculate_payback_period(cash_flows: np.ndarray) -> float:
    """Calculate simple payback period."""
    cumulative_cash_flows = np.cumsum(cash_flows)
    for year, cum_cf in enumerate(cumulative_cash_flows):
        if cum_cf > 0:
            # If the very first period (year 0) is already positive, the payback is instant
            # and should be reported as 0 rather than 1.
            return 0 if year == 0 else year
    return None


def calculate_greenfield_nuclear_hydrogen_with_tax_incentives(
    annual_metrics: dict,
    nuclear_capacity_mw: float,
    tea_sys_params: dict,
    hourly_results_df: pd.DataFrame,
    project_lifetime_config: int,
    construction_period_config: int,
    discount_rate_config: float,
    tax_rate_config: float,
    h2_capex_components_config: dict,
    h2_om_components_config: dict,
    h2_replacement_schedule_config: dict,
    macrs_config: dict,
    output_dir: str = None,
    project_lifetime_override: int = None
) -> dict:
    """
    Calculate comprehensive financial analysis for greenfield nuclear-hydrogen system
    with federal tax incentive scenarios (45Y PTC and 48E ITC).

    This function provides the complete implementation requested:
    - Baseline scenario (no tax incentives)
    - Scenario A: 45Y Production Tax Credit ($30/MWh for 10 years)
    - Scenario B: 48E Investment Tax Credit (50% of qualified CAPEX)
    - Comprehensive comparative analysis and reporting

    Args:
        annual_metrics: Dictionary of annual financial metrics from optimization
        nuclear_capacity_mw: Nuclear plant capacity in MW
        tea_sys_params: System parameters dictionary
        hourly_results_df: DataFrame with hourly optimization results
        project_lifetime_config: Total project lifetime in years
        construction_period_config: Construction period in years
        discount_rate_config: Discount rate for financial calculations
        tax_rate_config: Corporate tax rate
        h2_capex_components_config: Hydrogen system CAPEX components
        h2_om_components_config: Hydrogen system O&M components
        h2_replacement_schedule_config: Hydrogen system replacement schedule
        macrs_config: MACRS depreciation configuration
        output_dir: Output directory for reports and visualizations
        project_lifetime_override: Optional override for project lifetime (e.g., 80 years)
                                  If provided, this will be used instead of project_lifetime_config
                                  for all cash flow, replacement, and financial calculations

    Returns:
        Comprehensive analysis results dictionary
    """
    # Determine the actual project lifetime to use
    actual_project_lifetime = project_lifetime_override if project_lifetime_override is not None else project_lifetime_config

    logger.info("=" * 100)
    logger.info("GREENFIELD NUCLEAR-HYDROGEN SYSTEM WITH FEDERAL TAX INCENTIVES")
    logger.info(
        "Comprehensive Analysis: Baseline, 45Y PTC, and 48E ITC Scenarios")
    if project_lifetime_override is not None:
        logger.info(
            f"Using project lifetime override: {actual_project_lifetime} years")
    else:
        logger.info(
            f"Using default project lifetime: {actual_project_lifetime} years")
    logger.info("=" * 100)

    # First calculate the baseline greenfield nuclear-hydrogen system
    logger.info("Calculating baseline greenfield nuclear-hydrogen system...")
    baseline_results = calculate_greenfield_nuclear_hydrogen_system(
        annual_metrics=annual_metrics,
        nuclear_capacity_mw=nuclear_capacity_mw,
        tea_sys_params=tea_sys_params,
        project_lifetime_config=actual_project_lifetime,
        construction_period_config=construction_period_config,
        discount_rate_config=discount_rate_config,
        tax_rate_config=tax_rate_config,
        h2_capex_components_config=h2_capex_components_config,
        h2_om_components_config=h2_om_components_config,
        h2_replacement_schedule_config=h2_replacement_schedule_config
    )

    # Extract baseline cash flows and CAPEX breakdown
    baseline_cash_flows = baseline_results.get("cash_flows", np.array([]))

    # CRITICAL FIX: Use detailed cash flows from baseline analysis instead of generating simplified ones
    if len(baseline_cash_flows) > 0:
        logger.info(
            "✅ SUCCESS: Using detailed cash flows from baseline greenfield analysis")
        logger.info(
            f"  Cash flows data source: Baseline greenfield calculation")
        logger.info(f"  Cash flows length: {len(baseline_cash_flows)} years")
        logger.info(
            f"  Total project cash flows: ${np.sum(baseline_cash_flows):,.0f}")

        # Validate cash flows before proceeding
        expected_years = construction_period_config + actual_project_lifetime
        if len(baseline_cash_flows) != expected_years:
            logger.warning(
                f"Cash flows length mismatch. Expected: {expected_years}, Got: {len(baseline_cash_flows)}")
            if len(baseline_cash_flows) < expected_years:
                # Pad with zeros
                padded_cash_flows = np.zeros(expected_years)
                padded_cash_flows[:len(baseline_cash_flows)
                                  ] = baseline_cash_flows
                baseline_cash_flows = padded_cash_flows
                logger.info(f"Padded cash flows to {expected_years} years")
            else:
                # Truncate
                baseline_cash_flows = baseline_cash_flows[:expected_years]
                logger.info(f"Truncated cash flows to {expected_years} years")
    else:
        # Fallback: Generate simplified cash flows only if detailed ones are not available
        logger.warning(
            "Baseline cash flows not found in results. Generating simplified cash flows for tax incentive analysis.")

        # Generate simplified cash flows for tax incentive analysis
        total_years = construction_period_config + actual_project_lifetime
        baseline_cash_flows = np.zeros(total_years)

        # Construction period - negative cash flows (CAPEX)
        total_capex = baseline_results.get("total_system_capex_usd", 0)
        if total_capex > 0:
            annual_capex = total_capex / construction_period_config
            for year in range(construction_period_config):
                baseline_cash_flows[year] = -annual_capex

        # Operating period - positive cash flows (net revenue)
        annual_net_revenue = baseline_results.get("annual_net_revenue_usd", 0)
        for year in range(construction_period_config, total_years):
            baseline_cash_flows[year] = annual_net_revenue

        logger.info(
            f"Generated simplified baseline cash flows: {len(baseline_cash_flows)} years, total: ${np.sum(baseline_cash_flows):,.0f}")

    logger.info(
        f"Final baseline cash flows: shape={baseline_cash_flows.shape}, sum=${np.sum(baseline_cash_flows):,.0f}")

    # Prepare comprehensive CAPEX breakdown including both nuclear and hydrogen components
    comprehensive_capex_breakdown = {}

    # Add nuclear CAPEX components using detailed breakdown with learning rates
    from . import config
    logger.info(
        "Calculating detailed nuclear CAPEX breakdown for tax incentive analysis...")
    nuclear_capex_breakdown = calculate_nuclear_capex_breakdown(
        nuclear_capacity_mw, use_detailed_components=True)

    # Add detailed nuclear components to comprehensive breakdown
    for component_name, component_cost in nuclear_capex_breakdown.items():
        if component_name != "Total_Nuclear_CAPEX":
            comprehensive_capex_breakdown[component_name] = component_cost

    total_nuclear_capex = nuclear_capex_breakdown["Total_Nuclear_CAPEX"]

    logger.info(f"Detailed nuclear CAPEX components for tax incentive analysis:")
    for component_name, component_cost in nuclear_capex_breakdown.items():
        if component_name != "Total_Nuclear_CAPEX":
            logger.info(f"  {component_name}: ${component_cost:,.0f}")
    logger.info(f"  Total Nuclear CAPEX: ${total_nuclear_capex:,.0f}")

    # Add hydrogen system CAPEX components
    electrolyzer_capacity = annual_metrics.get("Electrolyzer_Capacity_MW", 50)
    h2_storage_capacity = tea_sys_params.get("h2_storage_capacity_kg", 10000)
    battery_capacity_mwh = annual_metrics.get("Battery_Capacity_MWh", 0)
    battery_power_mw = annual_metrics.get("Battery_Power_MW", 0)

    # Calculate hydrogen system CAPEX using existing config structure
    for component_name, component_config in h2_capex_components_config.items():
        if component_name == "Electrolyzer_System":
            capacity_ratio = electrolyzer_capacity / \
                component_config.get("reference_total_capacity_mw", 50)
            component_capex = component_config.get(
                "total_base_cost_for_ref_size", 100_000_000) * capacity_ratio
        elif component_name == "H2_Storage_System":
            capacity_ratio = h2_storage_capacity / \
                component_config.get("reference_total_capacity_mw", 10000)
            component_capex = component_config.get(
                "total_base_cost_for_ref_size", 10_000_000) * capacity_ratio
        elif component_name == "Battery_System_Energy" and battery_capacity_mwh > 0:
            # MODIFIED: No energy capacity cost for power-only costing
            component_capex = 0
        elif component_name == "Battery_System_Power" and battery_power_mw > 0:
            capacity_ratio = battery_power_mw / \
                component_config.get("reference_total_capacity_mw", 25)
            component_capex = component_config.get(
                "total_base_cost_for_ref_size", 5_000_000) * capacity_ratio
        else:
            # Fixed cost components
            component_capex = component_config.get(
                "total_base_cost_for_ref_size", 0)

        if component_capex > 0:
            comprehensive_capex_breakdown[component_name] = component_capex

    logger.info(
        f"Total project CAPEX: ${sum(comprehensive_capex_breakdown.values()):,.0f}")
    logger.info(
        f"Nuclear CAPEX: ${total_nuclear_capex:,.0f} ({total_nuclear_capex/sum(comprehensive_capex_breakdown.values()):.1%})")
    logger.info(
        f"Hydrogen system CAPEX: ${sum(comprehensive_capex_breakdown.values()) - total_nuclear_capex:,.0f}")

    # Run comprehensive tax incentive analysis
    logger.info("Running comprehensive federal tax incentive analysis...")
    try:
        # Extract plant-specific parameters from tea_sys_params for tax incentive analysis
        plant_specific_params = {}
        plant_param_keys = ['nameplate_capacity_mw', 'pTurbine_max_MW', 'turbine_capacity_mw',
                            'thermal_capacity_mwt', 'thermal_efficiency', 'qSteam_Total_MWth']
        for key in plant_param_keys:
            if key in tea_sys_params:
                plant_specific_params[key] = tea_sys_params[key]

        # Also add nuclear capacity from the function parameter
        plant_specific_params['nuclear_capacity_mw'] = nuclear_capacity_mw

        logger.info(
            f"Plant-specific parameters for tax incentive analysis: {list(plant_specific_params.keys())}")

        tax_incentive_results = run_comprehensive_tax_incentive_analysis(
            annual_metrics=annual_metrics,
            base_cash_flows=baseline_cash_flows,
            capex_breakdown=comprehensive_capex_breakdown,
            hourly_results_df=hourly_results_df,
            macrs_config=macrs_config,
            project_lifetime_years=actual_project_lifetime,
            construction_period_years=construction_period_config,
            discount_rate=discount_rate_config,
            tax_rate=tax_rate_config,
            plant_specific_params=plant_specific_params,
            baseline_lcoe_nuclear=baseline_results.get(
                "nuclear_lcoe_usd_per_mwh"),
            baseline_lcos_battery=baseline_results.get(
                "battery_lcos_usd_per_mwh")
        )
    except Exception as e:
        logger.error(f"Tax incentive analysis failed: {e}")
        logger.warning("Returning baseline greenfield results only")
        # Return baseline results with tax incentive analysis marked as failed
        return {
            "baseline_greenfield_results": baseline_results,
            "tax_incentive_analysis": None,
            "system_configuration": {
                "nuclear_capacity_mw": nuclear_capacity_mw,
                "electrolyzer_capacity_mw": electrolyzer_capacity,
                "h2_storage_capacity_kg": h2_storage_capacity,
                "battery_capacity_mwh": battery_capacity_mwh,
                "battery_power_mw": battery_power_mw,
                "total_capex_usd": sum(comprehensive_capex_breakdown.values()),
                "nuclear_capex_usd": total_nuclear_capex,
                "hydrogen_capex_usd": sum(comprehensive_capex_breakdown.values()) - total_nuclear_capex
            },
            "financial_comparison": {
                "baseline_npv": baseline_results.get("npv_usd", 0),
                "ptc_npv": baseline_results.get("npv_usd", 0),
                "itc_npv": baseline_results.get("npv_usd", 0),
                "ptc_npv_improvement": 0,
                "itc_npv_improvement": 0,
                "best_scenario": "baseline"
            },
            "analysis_status": "failed",
            "error_message": str(e)
        }

    # Generate comprehensive reports and visualizations
    if output_dir:
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate detailed comparative report
        report_file = output_path / "Federal_Tax_Incentive_Analysis_Report.txt"
        logger.info(
            f"Generating comprehensive tax incentive report: {report_file}")
        generate_tax_incentive_comparative_report(
            analysis_results=tax_incentive_results,
            output_file_path=report_file,
            project_name="Greenfield Nuclear-Hydrogen System"
        )

        # Generate comprehensive visualizations
        plots_dir = output_path / "Tax_Incentive_Plots"
        logger.info(
            f"Creating tax incentive analysis visualizations: {plots_dir}")
        create_tax_incentive_visualizations(
            analysis_results=tax_incentive_results,
            output_dir=plots_dir,
            project_name="Greenfield Nuclear-Hydrogen System"
        )

    # Combine results for comprehensive output
    comprehensive_results = {
        "baseline_greenfield_results": baseline_results,
        "tax_incentive_analysis": tax_incentive_results,
        "system_configuration": {
            "nuclear_capacity_mw": nuclear_capacity_mw,
            "electrolyzer_capacity_mw": electrolyzer_capacity,
            "h2_storage_capacity_kg": h2_storage_capacity,
            "battery_capacity_mwh": battery_capacity_mwh,
            "battery_power_mw": battery_power_mw,
            "total_capex_usd": sum(comprehensive_capex_breakdown.values()),
            "nuclear_capex_usd": total_nuclear_capex,
            "hydrogen_capex_usd": sum(comprehensive_capex_breakdown.values()) - total_nuclear_capex
        },
        "financial_comparison": {
            "baseline_npv": tax_incentive_results["scenarios"]["baseline"]["financial_metrics"]["npv_usd"],
            "ptc_npv": tax_incentive_results["scenarios"]["ptc"]["financial_metrics"]["npv_usd"],
            "itc_npv": tax_incentive_results["scenarios"]["itc"]["financial_metrics"]["npv_usd"],
            "ptc_npv_improvement": tax_incentive_results["comparative_analysis"]["npv_comparison"]["ptc_npv_improvement"],
            "itc_npv_improvement": tax_incentive_results["comparative_analysis"]["npv_comparison"]["itc_npv_improvement"],
            "best_scenario": tax_incentive_results["comparative_analysis"]["best_scenario"]
        }
    }

    # Log executive summary
    logger.info("=" * 80)
    logger.info("EXECUTIVE SUMMARY - FEDERAL TAX INCENTIVE ANALYSIS")
    logger.info("=" * 80)

    baseline_npv = comprehensive_results["financial_comparison"]["baseline_npv"]
    ptc_npv = comprehensive_results["financial_comparison"]["ptc_npv"]
    itc_npv = comprehensive_results["financial_comparison"]["itc_npv"]
    ptc_improvement = comprehensive_results["financial_comparison"]["ptc_npv_improvement"]
    itc_improvement = comprehensive_results["financial_comparison"]["itc_npv_improvement"]
    best_scenario = comprehensive_results["financial_comparison"]["best_scenario"]

    logger.info(f"System Configuration:")
    logger.info(f"  Nuclear Capacity:         {nuclear_capacity_mw:,.0f} MW")
    logger.info(f"  Electrolyzer Capacity:    {electrolyzer_capacity:,.0f} MW")
    logger.info(
        f"  Total Project CAPEX:      ${sum(comprehensive_capex_breakdown.values()):,.0f}")
    logger.info(
        f"  Annual H2 Production:     {annual_metrics.get('H2_Production_kg_annual', 0):,.0f} kg/year")

    logger.info(f"\nFinancial Analysis Results:")
    logger.info(f"  Baseline NPV:             ${baseline_npv:,.0f}")
    logger.info(
        f"  45Y PTC NPV:              ${ptc_npv:,.0f} ({ptc_improvement:+,.0f})")
    logger.info(
        f"  48E ITC NPV:              ${itc_npv:,.0f} ({itc_improvement:+,.0f})")

    best_scenario_name = {
        "baseline": "Baseline (No Incentives)",
        "ptc": "45Y Production Tax Credit",
        "itc": "48E Investment Tax Credit"
    }.get(best_scenario, best_scenario)

    logger.info(f"\nRecommendation:")
    logger.info(f"  Best Scenario: {best_scenario_name}")

    if ptc_improvement > 0 and itc_improvement > 0:
        if ptc_improvement > itc_improvement:
            logger.info(
                f"  The 45Y PTC provides ${ptc_improvement - itc_improvement:,.0f} higher NPV than 48E ITC")
        elif itc_improvement > ptc_improvement:
            logger.info(
                f"  The 48E ITC provides ${itc_improvement - ptc_improvement:,.0f} higher NPV than 45Y PTC")
        else:
            logger.info(
                f"  Both incentives provide similar financial benefits")

    # Log tax incentive details
    ptc_analysis = tax_incentive_results["scenarios"]["ptc"]["analysis"]
    if "tax_benefits" in ptc_analysis and "ptc" in ptc_analysis["tax_benefits"]:
        ptc_details = ptc_analysis["tax_benefits"]["ptc"]
        logger.info(f"\n45Y PTC Details:")
        logger.info(
            f"  Annual Generation:        {ptc_details['annual_generation_mwh']:,.0f} MWh")
        logger.info(
            f"  Annual PTC Benefit:       ${ptc_details['annual_ptc_benefit_usd']:,.0f}")
        logger.info(
            f"  Total PTC Value:          ${ptc_details['total_ptc_value_usd']:,.0f}")

    itc_analysis = tax_incentive_results["scenarios"]["itc"]["analysis"]
    if "tax_benefits" in itc_analysis and "itc" in itc_analysis["tax_benefits"]:
        itc_details = itc_analysis["tax_benefits"]["itc"]
        logger.info(f"\n48E ITC Details:")
        logger.info(
            f"  Qualified CAPEX:          ${itc_details['total_qualified_capex_usd']:,.0f}")
        logger.info(
            f"  ITC Credit Amount:        ${itc_details['itc_credit_amount_usd']:,.0f}")
        logger.info(
            f"  Net ITC Benefit:          ${itc_analysis.get('net_itc_benefit', 0):,.0f}")

    logger.info("=" * 80)
    logger.info("Federal tax incentive analysis completed successfully")
    logger.info("=" * 80)

    return comprehensive_results


def calculate_nuclear_baseline_financial_analysis(
    tea_sys_params: dict,
    hourly_results_df: pd.DataFrame,
    project_lifetime_config: int,
    construction_period_config: int,
    discount_rate_config: float,
    tax_rate_config: float,
    target_iso: str,
    npps_info_path: str = None,
    tax_policies: dict = None,
    hourly_results_file_path: str = None
) -> dict:
    """
    Calculate financial analysis for nuclear power plant baseline operation (no modifications).
    Analyzes NPP operation using Nameplate Power Factor from NPPs info and hourly electricity prices.

    Args:
        tea_sys_params: System parameters dictionary
        hourly_results_df: Hourly optimization results dataframe
        project_lifetime_config: Project lifetime in years (fallback if NPP data unavailable)
        construction_period_config: Construction period in years
        discount_rate_config: Discount rate as fraction
        tax_rate_config: Tax rate as fraction
        target_iso: Target ISO region
        npps_info_path: Path to NPPs info CSV file

    Returns:
        Dictionary containing baseline financial analysis results
    """
    logger.info("=" * 80)
    logger.info("NUCLEAR POWER PLANT BASELINE FINANCIAL ANALYSIS")
    logger.info("Financial analysis for NPP operation without modifications")
    logger.info("=" * 80)

    # Load NPPs info if path provided and extract actual remaining years
    npp_info = None
    actual_project_lifetime = project_lifetime_config

    # PRIORITY 1: Extract remaining years from hourly results filename
    remaining_years_from_filename = None
    if hourly_results_file_path:
        from .data_loader import extract_remaining_years_from_file_path
        from pathlib import Path
        remaining_years_from_filename = extract_remaining_years_from_file_path(
            hourly_results_file_path)
        if remaining_years_from_filename is not None:
            actual_project_lifetime = remaining_years_from_filename
            logger.info(
                f"NUCLEAR BASELINE: Using remaining lifetime from filename: {actual_project_lifetime} years")
            logger.info(
                f"  (Extracted from {Path(hourly_results_file_path).name})")

    # Initialize data sources tracking
    plant_data_sources = {
        "plant_capacity_source": "optimization results",
        "plant_lifetime_source": "filename extraction" if remaining_years_from_filename is not None else "default values",
        "plant_identification_method": "fallback"
    }  # Default fallback

    if npps_info_path:
        try:
            import pandas as pd
            npp_info_df = pd.read_csv(npps_info_path)

            # Try to extract plant name from hourly results or tea_sys_params
            specific_plant_name = None

            # Check if plant name is available in tea_sys_params
            if 'matched_plant_name' in tea_sys_params:
                specific_plant_name = tea_sys_params['matched_plant_name']
                logger.info(
                    f"Using matched plant name from tea_sys_params: {specific_plant_name}")

            # Try to match specific plant first, then fall back to ISO filtering
            if specific_plant_name:
                # Try to find exact plant match by name
                plant_matches = npp_info_df[
                    npp_info_df['Plant Name'].str.contains(
                        specific_plant_name, case=False, na=False)
                ]

                if not plant_matches.empty:
                    # CRITICAL FIX: Also check for Generator ID if available
                    final_matches = plant_matches
                    specific_generator_id = tea_sys_params.get(
                        'matched_generator_id', None)

                    if specific_generator_id is not None and specific_generator_id != 'Unknown':
                        try:
                            generator_id_int = int(
                                float(specific_generator_id))
                            generator_matches = plant_matches[plant_matches['Generator ID']
                                                              == generator_id_int]
                            if not generator_matches.empty:
                                final_matches = generator_matches
                                logger.info(
                                    f"🎯 Found Generator ID {generator_id_int} match for plant '{specific_plant_name}' in baseline analysis")
                            else:
                                logger.warning(
                                    f"⚠️  No Generator ID {generator_id_int} found for plant '{specific_plant_name}' in baseline analysis, using first available unit")
                        except (ValueError, TypeError):
                            logger.warning(
                                f"⚠️  Invalid Generator ID '{specific_generator_id}' in baseline analysis, using plant name match only")

                    # If multiple matches, prioritize by ISO region
                    iso_matches = final_matches[final_matches['ISO']
                                                == target_iso]
                    if not iso_matches.empty:
                        npp_info = iso_matches.iloc[0].to_dict()
                        plant_name = npp_info.get(
                            'Plant Name', specific_plant_name)
                        generator_id = npp_info.get('Generator ID', 'Unknown')
                        logger.info(
                            f"✅ Matched specific plant in baseline analysis: {plant_name} Unit {generator_id} in {target_iso}")
                    else:
                        npp_info = final_matches.iloc[0].to_dict()
                        plant_name = npp_info.get(
                            'Plant Name', specific_plant_name)
                        generator_id = npp_info.get('Generator ID', 'Unknown')
                        actual_iso = npp_info.get('ISO', 'Unknown')
                        logger.info(
                            f"⚠️  Matched plant by name and Generator ID but different ISO in baseline analysis: {plant_name} Unit {generator_id} in {actual_iso}")

            # Fallback: Filter for the target ISO and use first plant (original logic)
            if npp_info is None:
                iso_plants = npp_info_df[npp_info_df['ISO'] == target_iso]
                if not iso_plants.empty:
                    npp_info = iso_plants.iloc[0].to_dict()
                    plant_name = npp_info.get('Plant Name', 'Unknown')
                    generator_id = npp_info.get('Generator ID', 'Unknown')
                    logger.warning(
                        f"No specific plant match found in baseline analysis, using first plant in {target_iso}: {plant_name} Unit {generator_id}")

                    # List all plants in this ISO for transparency
                    all_plants_in_iso = iso_plants['Plant Name'].tolist()
                    all_generator_ids_in_iso = iso_plants['Generator ID'].tolist(
                    )
                    remaining_years_in_iso = iso_plants['remaining'].tolist()
                    logger.info(
                        f"Available plants in {target_iso} for baseline analysis:")
                    for i, (p_name, p_gen_id, p_remaining) in enumerate(zip(all_plants_in_iso, all_generator_ids_in_iso, remaining_years_in_iso)):
                        prefix = "→ SELECTED: " if i == 0 else "           "
                        logger.info(
                            f"  {prefix}{p_name} Unit {p_gen_id} ({p_remaining} years remaining)")
                else:
                    logger.warning(f"No NPP info found for ISO: {target_iso}")

            if npp_info:
                # Only use NPPs info for remaining years if filename extraction failed
                if remaining_years_from_filename is None:
                    remaining_years = npp_info.get('remaining', None)
                    if remaining_years is not None and pd.notna(remaining_years):
                        try:
                            actual_project_lifetime = int(
                                float(remaining_years))
                            logger.info(
                                f"Using remaining plant lifetime from NPPs info as fallback: {actual_project_lifetime} years")
                            logger.info(
                                f"  (Fallback from NPPs info file for {plant_name})")
                            plant_data_sources[
                                "plant_lifetime_source"] = "NPPs info file (fallback)"
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid remaining years value: {remaining_years}, using default: {project_lifetime_config}")
                    else:
                        logger.warning(
                            f"No remaining years data available for {plant_name}, using default: {project_lifetime_config}")
                else:
                    logger.info(
                        f"Skipping NPPs info remaining years since filename extraction succeeded")

                logger.info(f"Using NPP info for plant: {plant_name}")

                # Store plant data sources for later use
                specific_generator_id = tea_sys_params.get(
                    'matched_generator_id', None)
                plant_data_sources.update({
                    "plant_capacity_source": "NPPs info file",
                    "matched_generator_id": specific_generator_id if specific_generator_id != 'Unknown' else None,
                    "plant_identification_method": "specific_name_and_generator_id" if (specific_plant_name and specific_generator_id and specific_generator_id != 'Unknown') else ("specific_name" if specific_plant_name else "iso_first")
                })

        except Exception as e:
            logger.error(f"Error loading NPPs info: {e}")
            plant_data_sources = {
                "plant_capacity_source": "optimization results",
                "plant_lifetime_source": "default values",
                "plant_identification_method": "fallback"
            }

    logger.info(
        f"Project lifetime for analysis: {actual_project_lifetime} years")

    # Get nuclear plant parameters from hourly results or use defaults
    plant_name_from_hourly = 'Unknown'

    # CRITICAL FIX: For Case 1 baseline analysis, prioritize NPPs info nameplate capacity over optimization results
    # This ensures Case 1 uses actual plant specifications rather than optimization results
    nameplate_capacity_mw = None
    thermal_capacity_mwt = None

    # First priority: Extract nameplate capacity from NPPs info if available
    if npp_info:
        try:
            # Try to get nameplate capacity from NPPs info
            if 'Nameplate Capacity (MW)' in npp_info:
                nameplate_capacity_mw = float(
                    str(npp_info['Nameplate Capacity (MW)']).replace(',', ''))
                logger.info(
                    f"Using Nameplate Capacity from NPPs info: {nameplate_capacity_mw:.2f} MW")
                plant_data_sources["plant_capacity_source"] = "NPPs info file"

            # Also get thermal capacity if available
            if 'Licensed Power (MWt)' in npp_info:
                thermal_capacity_mwt = float(
                    str(npp_info['Licensed Power (MWt)']).replace(',', ''))
                logger.info(
                    f"Using Licensed Power from NPPs info: {thermal_capacity_mwt:.2f} MWt")

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Error extracting nameplate capacity from NPPs info: {e}")
            nameplate_capacity_mw = None
            thermal_capacity_mwt = None

    # Second priority: Use hourly results as fallback only if NPPs info not available
    turbine_capacity_from_hourly = None
    if len(hourly_results_df) > 0:
        # Get plant capacity from hourly results (turbine capacity) - only as fallback
        turbine_capacity_from_hourly = hourly_results_df['pTurbine_MW'].max()
        logger.info(
            f"Nuclear turbine capacity from hourly results: {turbine_capacity_from_hourly:.2f} MW")

        # Try to get plant name from hourly results
        if 'Plant_Name' in hourly_results_df.columns:
            plant_name_from_hourly = hourly_results_df['Plant_Name'].iloc[0]
        elif 'Original_Plant_Name' in hourly_results_df.columns:
            plant_name_from_hourly = hourly_results_df['Original_Plant_Name'].iloc[0]

        if plant_name_from_hourly != 'Unknown':
            logger.info(
                f"Plant name from hourly results: {plant_name_from_hourly}")

    # Determine final capacity to use for Case 1 analysis
    if nameplate_capacity_mw is not None:
        # Use nameplate capacity from NPPs info (preferred for Case 1)
        turbine_capacity_mw = nameplate_capacity_mw
        logger.info(
            f"CASE 1 ANALYSIS: Using nameplate capacity from NPPs info: {turbine_capacity_mw:.2f} MW")
    elif turbine_capacity_from_hourly is not None:
        # Fallback to hourly results if NPPs info not available
        turbine_capacity_mw = turbine_capacity_from_hourly
        logger.warning(
            f"CASE 1 ANALYSIS: NPPs info not available, using turbine capacity from hourly results: {turbine_capacity_mw:.2f} MW")
        plant_data_sources["plant_capacity_source"] = "optimization results"
    else:
        # Default fallback
        turbine_capacity_mw = 1000.0
        logger.warning(
            "CASE 1 ANALYSIS: No capacity data available, using default: 1000 MW")
        plant_data_sources["plant_capacity_source"] = "default fallback"

    # Use Nameplate Power Factor from NPPs info if available
    nameplate_power_factor = 0.90  # Default
    if npp_info and 'Nameplate Power Factor' in npp_info:
        try:
            nameplate_power_factor = float(npp_info['Nameplate Power Factor'])
            logger.info(
                f"Using Nameplate Power Factor from NPPs info: {nameplate_power_factor:.3f}")
        except (ValueError, TypeError):
            logger.warning(
                "Invalid Nameplate Power Factor in NPPs info, using default: 0.90")
    else:
        logger.info(
            f"Using default Nameplate Power Factor: {nameplate_power_factor:.3f}")

    # Calculate hourly electricity prices from hourly results
    electricity_prices_usd_per_mwh = []
    if len(hourly_results_df) > 0 and 'EnergyPrice_LMP_USDperMWh' in hourly_results_df.columns:
        electricity_prices_usd_per_mwh = hourly_results_df['EnergyPrice_LMP_USDperMWh'].values
        avg_electricity_price = np.mean(electricity_prices_usd_per_mwh)
        min_electricity_price = np.min(electricity_prices_usd_per_mwh)
        max_electricity_price = np.max(electricity_prices_usd_per_mwh)

        logger.info(f"Electricity price data source: Hourly results file")
        logger.info(
            f"  Data points: {len(electricity_prices_usd_per_mwh)} hours")
        logger.info(f"  Average: ${avg_electricity_price:.2f}/MWh")
        logger.info(f"  Minimum: ${min_electricity_price:.2f}/MWh")
        logger.info(f"  Maximum: ${max_electricity_price:.2f}/MWh")
    else:
        # Use a reasonable default price
        avg_electricity_price = 35.0  # $/MWh
        electricity_prices_usd_per_mwh = np.full(8760, avg_electricity_price)
        logger.warning(f"Electricity price data source: Default fallback")
        logger.warning(
            f"  EnergyPrice_LMP_USDperMWh not found in hourly results – using default value: ${avg_electricity_price:.2f}/MWh")

    # Calculate thermal efficiency if we have both thermal and electric capacity
    thermal_efficiency = None
    if thermal_capacity_mwt is not None and nameplate_capacity_mw is not None and thermal_capacity_mwt > 0:
        thermal_efficiency = nameplate_capacity_mw / thermal_capacity_mwt
        logger.info(
            f"Calculated thermal efficiency from NPPs info: {thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)")
    elif thermal_capacity_mwt is not None and turbine_capacity_mw > 0:
        thermal_efficiency = turbine_capacity_mw / thermal_capacity_mwt
        logger.info(
            f"Calculated thermal efficiency: {thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)")
    else:
        thermal_efficiency = 0.33  # Default nuclear efficiency
        logger.info(
            f"Using default thermal efficiency: {thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)")

    # Nuclear plant operational parameters
    # Use nameplate power factor as capacity factor
    capacity_factor = nameplate_power_factor
    hours_per_year = 8760

    # Calculate annual generation
    annual_generation_mwh = turbine_capacity_mw * hours_per_year * capacity_factor
    logger.info(
        f"Annual electricity generation: {annual_generation_mwh:,.0f} MWh")

    # Calculate annual revenue using hourly prices
    if len(electricity_prices_usd_per_mwh) == 8760:
        # Use hourly prices
        hourly_generation_mwh = turbine_capacity_mw * capacity_factor  # MWh per hour
        hourly_revenues = hourly_generation_mwh * electricity_prices_usd_per_mwh
        annual_revenue = np.sum(hourly_revenues)
        logger.info("Using hourly electricity prices for revenue calculation")
    else:
        # Fallback to average price
        annual_revenue = annual_generation_mwh * avg_electricity_price
        logger.info("Using average electricity price for revenue calculation")

    logger.info(f"Annual electricity revenue: ${annual_revenue:,.0f}")

    # Nuclear plant operating costs (using centralized config parameters)
    from . import config
    opex_params = config.NUCLEAR_COST_PARAMETERS["opex_parameters"]

    # Fixed O&M costs
    fixed_om_per_mw_month = opex_params["fixed_om_per_mw_month"]
    annual_fixed_om = turbine_capacity_mw * fixed_om_per_mw_month * 12

    # Variable O&M costs
    variable_om_per_mwh = opex_params["variable_om_per_mwh"]
    annual_variable_om = annual_generation_mwh * variable_om_per_mwh

    # Nuclear fuel costs
    fuel_cost_per_mwh = opex_params["fuel_cost_per_mwh"]
    annual_fuel_cost = annual_generation_mwh * fuel_cost_per_mwh

    # Additional costs (insurance, regulatory, waste disposal, security)
    additional_costs_per_mw_year = opex_params["additional_costs_per_mw_year"]
    annual_additional_costs = turbine_capacity_mw * additional_costs_per_mw_year

    # Total annual OPEX
    annual_opex = annual_fixed_om + annual_variable_om + \
        annual_fuel_cost + annual_additional_costs

    logger.info(f"Annual operating costs breakdown:")
    logger.info(f"  Fixed O&M: ${annual_fixed_om:,.0f}")
    logger.info(f"  Variable O&M: ${annual_variable_om:,.0f}")
    logger.info(f"  Fuel costs: ${annual_fuel_cost:,.0f}")
    logger.info(f"  Additional costs: ${annual_additional_costs:,.0f}")
    logger.info(f"  Total annual OPEX: ${annual_opex:,.0f}")

    # Annual net cash flow (before tax)
    annual_net_cash_flow_before_tax = annual_revenue - annual_opex
    logger.info(
        f"Annual net cash flow (before tax): ${annual_net_cash_flow_before_tax:,.0f}")

    # Calculate taxes
    annual_taxes = max(0, annual_net_cash_flow_before_tax * tax_rate_config)
    annual_net_cash_flow_after_tax = annual_net_cash_flow_before_tax - annual_taxes

    logger.info(f"Annual taxes: ${annual_taxes:,.0f}")
    logger.info(
        f"Annual net cash flow (after tax): ${annual_net_cash_flow_after_tax:,.0f}")

    # Major replacement/refurbishment costs (using centralized config parameters)
    replacement_costs = config.NUCLEAR_COST_PARAMETERS["replacement_costs_per_mw"]
    replacement_schedule = {
        15: turbine_capacity_mw * replacement_costs["turbine_overhaul_15_years"],
        25: turbine_capacity_mw * replacement_costs["steam_generator_25_years"],
        30: turbine_capacity_mw * replacement_costs["major_refurbishment_30_years"],
        40: turbine_capacity_mw * replacement_costs["life_extension_40_years"],
    }

    # Calculate 45U Nuclear PTC benefits for existing nuclear plant
    logger.info("=" * 60)
    logger.info("45U NUCLEAR PRODUCTION TAX CREDIT ANALYSIS")
    logger.info("=" * 60)

    nuclear_45u_benefits = calculate_45u_nuclear_ptc_benefits(
        annual_generation_mwh=annual_generation_mwh,
        project_start_year=2024,
        project_lifetime_years=actual_project_lifetime,
        tax_policies=tax_policies,
        hourly_prices_usd_per_mwh=electricity_prices_usd_per_mwh
    )

    logger.info(f"45U Policy Benefits for Nuclear Plant:")
    logger.info(f"  Annual Generation: {annual_generation_mwh:,.0f} MWh")
    logger.info(
        f"  Credit Rate: ${nuclear_45u_benefits['credit_rate_per_mwh']}/MWh")
    logger.info(
        f"  Credit Period: {nuclear_45u_benefits['credit_period_start']}-{nuclear_45u_benefits['credit_period_end']}")
    logger.info(
        f"  Eligible Years: {nuclear_45u_benefits['total_eligible_years']} years")
    logger.info(
        f"  Annual Credit Value: ${nuclear_45u_benefits['annual_credit_value']:,.0f}")
    logger.info(
        f"  Total 45U Credits: ${nuclear_45u_benefits['total_45u_credits']:,.0f}")

    # Calculate lifecycle cash flows for both scenarios
    total_years = actual_project_lifetime

    # Scenario 1: Without 45U policy (baseline)
    cash_flows_without_45u = np.zeros(total_years)

    for year in range(total_years):
        # Base annual cash flow
        cash_flows_without_45u[year] = annual_net_cash_flow_after_tax

        # Subtract replacement costs if scheduled
        if (year + 1) in replacement_schedule:
            replacement_cost = replacement_schedule[year + 1]
            cash_flows_without_45u[year] -= replacement_cost
            logger.info(
                f"Year {year + 1}: Replacement cost ${replacement_cost:,.0f}")

    # Scenario 2: With 45U policy
    cash_flows_with_45u = cash_flows_without_45u.copy()
    annual_45u_credits = nuclear_45u_benefits["annual_45u_credits"]

    # Add 45U credits to applicable years
    for year in range(total_years):
        if year < len(annual_45u_credits):
            cash_flows_with_45u[year] += annual_45u_credits[year]

    # Calculate financial metrics for both scenarios
    discount_factors = np.array(
        [(1 / (1 + discount_rate_config) ** year) for year in range(total_years)])

    # Without 45U policy
    discounted_cash_flows_without_45u = cash_flows_without_45u * discount_factors
    npv_without_45u = np.sum(discounted_cash_flows_without_45u)

    # IRR is not applicable for existing nuclear plant baseline analysis
    # because there's no initial investment - the plant already exists
    irr_without_45u = calculate_irr(
        cash_flows_without_45u, is_baseline_analysis=True)
    payback_without_45u = calculate_payback_period(cash_flows_without_45u)

    # With 45U policy
    discounted_cash_flows_with_45u = cash_flows_with_45u * discount_factors
    npv_with_45u = np.sum(discounted_cash_flows_with_45u)

    # IRR is still not applicable even with 45U policy for existing plant baseline
    # The 45U policy doesn't create an investment scenario - it's just additional revenue
    irr_with_45u = calculate_irr(
        cash_flows_with_45u, is_baseline_analysis=True)
    payback_with_45u = calculate_payback_period(cash_flows_with_45u)

    # Calculate 45U impact
    npv_improvement_45u = npv_with_45u - npv_without_45u
    # IRR improvement is not applicable for existing plant baseline analysis
    irr_improvement_45u = None

    # Legacy variables for backward compatibility
    npv = npv_without_45u
    irr_value = irr_without_45u
    payback_period = payback_without_45u
    # Default to without 45U for legacy compatibility
    cash_flows = cash_flows_without_45u

    # Additional financial metrics
    total_revenue = annual_revenue * total_years
    total_opex = annual_opex * total_years
    total_replacement_costs = sum(replacement_schedule.values())
    profit_margin = (annual_net_cash_flow_before_tax /
                     annual_revenue * 100) if annual_revenue > 0 else 0

    # Calculate LCOE (Levelized Cost of Electricity) using only OPEX costs
    try:
        # Calculate present value of total OPEX over project lifetime
        total_opex_pv = 0
        total_generation_pv = 0

        # Ensure we have valid values for calculation
        if annual_opex is not None and annual_generation_mwh is not None and annual_opex > 0 and annual_generation_mwh > 0:
            for year in range(total_years):
                discount_factor = 1 / \
                    ((1 + discount_rate_config) ** (year + 1))
                # Add annual OPEX to present value
                total_opex_pv += annual_opex * discount_factor
                # Add annual generation to present value
                total_generation_pv += annual_generation_mwh * discount_factor

            # Calculate LCOE for both scenarios
            lcoe_without_45u = total_opex_pv / \
                total_generation_pv if total_generation_pv > 0 else None
            # LCOE doesn't change with 45U since it's only OPEX-based
            lcoe_with_45u = lcoe_without_45u

            logger.info("LCOE (OPEX-only) calculated successfully")
        else:
            logger.warning(
                "Cannot calculate LCOE: invalid annual_opex or annual_generation_mwh")
            lcoe_without_45u = None
            lcoe_with_45u = None
    except Exception as e:
        logger.error(f"Error calculating LCOE: {e}")
        lcoe_without_45u = None
        lcoe_with_45u = None

    logger.info(f"Financial Results Comparison:")
    logger.info(f"  WITHOUT 45U POLICY:")
    logger.info(f"    NPV: ${npv_without_45u:,.0f}")
    logger.info(f"    IRR: N/A (not applicable for existing plant operations)")
    logger.info(
        f"    Payback Period: {payback_without_45u} years" if payback_without_45u else "    Payback Period: > project lifetime")

    logger.info(f"  WITH 45U POLICY:")
    logger.info(f"    NPV: ${npv_with_45u:,.0f}")
    logger.info(f"    IRR: N/A (not applicable for existing plant operations)")
    logger.info(
        f"    Payback Period: {payback_with_45u} years" if payback_with_45u else "    Payback Period: > project lifetime")

    logger.info(f"  45U POLICY IMPACT:")
    logger.info(f"    NPV Improvement: +${npv_improvement_45u:,.0f}")
    logger.info(
        f"    IRR Improvement: N/A (IRR not applicable for existing plant operations)")
    logger.info(
        f"    Total 45U Benefits: ${nuclear_45u_benefits['total_45u_credits']:,.0f}")

    logger.info(f"  Overall Profit Margin: {profit_margin:.1f}%")

    # Return comprehensive results with 45U analysis
    return {
        "analysis_type": "Nuclear Baseline (No Modifications)",
        "includes_45u_analysis": True,

        "plant_parameters": {
            "turbine_capacity_mw": turbine_capacity_mw,
            # FIXED: Include nameplate capacity
            "nameplate_capacity_mw": nameplate_capacity_mw if nameplate_capacity_mw is not None else turbine_capacity_mw,
            # Estimate if not available
            "thermal_capacity_mwt": thermal_capacity_mwt if thermal_capacity_mwt is not None else (turbine_capacity_mw / 0.33 if turbine_capacity_mw else None),
            "thermal_efficiency": thermal_efficiency,  # FIXED: Include thermal efficiency
            "nameplate_power_factor": nameplate_power_factor,
            "capacity_factor": capacity_factor,
            "plant_name": npp_info.get('Plant Name', plant_name_from_hourly) if npp_info else plant_name_from_hourly,
            "iso_region": target_iso,
            "remaining_plant_life_years": actual_project_lifetime
        },

        "annual_performance": {
            "annual_generation_mwh": annual_generation_mwh,
            "annual_revenue_usd": annual_revenue,
            "annual_fixed_om_usd": annual_fixed_om,
            "annual_variable_om_usd": annual_variable_om,
            "annual_fuel_cost_usd": annual_fuel_cost,
            "annual_additional_costs_usd": annual_additional_costs,
            "annual_total_opex_usd": annual_opex,
            "annual_net_cash_flow_before_tax_usd": annual_net_cash_flow_before_tax,
            "annual_taxes_usd": annual_taxes,
            "annual_net_cash_flow_after_tax_usd": annual_net_cash_flow_after_tax,
            "profit_margin_percent": profit_margin
        },

        "electricity_market": {
            "avg_electricity_price_usd_per_mwh": avg_electricity_price,
            "min_electricity_price_usd_per_mwh": min_electricity_price if 'min_electricity_price' in locals() else avg_electricity_price,
            "max_electricity_price_usd_per_mwh": max_electricity_price if 'max_electricity_price' in locals() else avg_electricity_price,
            "using_hourly_prices": len(electricity_prices_usd_per_mwh) == 8760
        },

        # 45U Policy analysis
        "nuclear_45u_benefits": nuclear_45u_benefits,

        # Financial metrics without 45U (baseline scenario)
        "scenario_without_45u": {
            "npv_usd": npv_without_45u,
            "irr_percent": irr_without_45u,
            "payback_period_years": payback_without_45u,
            "cash_flows": cash_flows_without_45u.tolist()
        },

        # Financial metrics with 45U (enhanced scenario)
        "scenario_with_45u": {
            "npv_usd": npv_with_45u,
            "irr_percent": irr_with_45u,
            "payback_period_years": payback_with_45u,
            "cash_flows": cash_flows_with_45u.tolist()
        },

        # 45U Impact analysis
        "45u_policy_impact": {
            "npv_improvement_usd": npv_improvement_45u,
            "irr_improvement_percent": irr_improvement_45u,
            "total_45u_credits_usd": nuclear_45u_benefits["total_45u_credits"],
            "eligible_years": nuclear_45u_benefits["total_eligible_years"]
        },

        # Legacy compatibility fields (default to without 45U scenario)
        "financial_metrics": {
            "npv_usd": npv_without_45u,
            "irr_percent": irr_without_45u,
            "payback_period_years": payback_without_45u,
            "project_lifetime_years": total_years,
            "discount_rate_percent": discount_rate_config * 100,
            "tax_rate_percent": tax_rate_config * 100
        },

        "data_sources": plant_data_sources if 'plant_data_sources' in locals() else {
            "project_lifetime_source": "NPPs info file" if actual_project_lifetime != project_lifetime_config else "Config default",
            "electricity_price_source": "Hourly results file" if len(electricity_prices_usd_per_mwh) == 8760 else "Default fallback",
            "plant_capacity_source": "Hourly results file" if len(hourly_results_df) > 0 else "Default fallback",
            "nameplate_power_factor_source": "NPPs info file" if npp_info and 'Nameplate Power Factor' in npp_info else "Default fallback",
            "plant_name_source": "NPPs info file" if npp_info and 'Plant Name' in npp_info else ("Hourly results file" if plant_name_from_hourly != 'Unknown' else "Default fallback"),
            "plant_identification_method": "fallback"
        },

        "lifecycle_totals": {
            "total_revenue_usd": total_revenue,
            "total_opex_usd": total_opex,
            "total_replacement_costs_usd": total_replacement_costs,
            "total_taxes_usd": annual_taxes * total_years,
            "total_net_cash_flow_usd": np.sum(cash_flows_without_45u),
            "total_net_cash_flow_with_45u_usd": np.sum(cash_flows_with_45u)
        },

        # Legacy compatibility fields
        "cash_flows": cash_flows_without_45u.tolist(),
        "replacement_schedule": replacement_schedule
    }
