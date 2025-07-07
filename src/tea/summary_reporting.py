"""
Comprehensive TEA Summary Report Generator
Generates detailed technical-economic analysis reports following standardized format.
"""

import matplotlib
import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Set matplotlib backend for server environments
matplotlib.use('Agg')


def format_aligned_line(name: str, value: str, min_width: int = 50, indent: str = "  ") -> str:
    """
    Format a name-value pair with consistent colon alignment.
    """
    effective_width = max(min_width, len(name))
    return f"{indent}{name:<{effective_width}} : {value}\n"


def format_aligned_section(items: dict, min_width: int = 50, indent: str = "  ") -> str:
    """
    Format a dictionary of name-value pairs with consistent alignment.
    """
    if not items:
        return ""

    max_name_length = 0
    if items:
        string_keys = [str(name) for name in items.keys()]
        if string_keys:
            max_name_length = max(len(name) for name in string_keys)

    effective_width = max(min_width, max_name_length)

    result_lines = []
    for name, value in items.items():
        result_lines.append(format_aligned_line(
            str(name), str(value), effective_width, indent))

    return "".join(result_lines)


def _try_format_currency(value: Any, default_na: bool = False) -> str:
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"${value:,.0f}"
    return "N/A" if default_na else "$0"


def _try_format_percentage(value: Any, decimals: int = 1) -> str:
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"{value:.{decimals}f}%"
    return "N/A"


def _try_format_number(value: Any, decimals: int = 1, default_na: bool = False) -> str:
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"{value:,.{decimals}f}"
    return "N/A" if default_na else ("0" if decimals == 0 else f"{0:.{decimals}f}")


def _format_currency(value: Optional[float], default_na: bool = False) -> str:
    return _try_format_currency(value, default_na)


def _format_percentage(value: Optional[float], decimals: int = 1) -> str:
    return _try_format_percentage(value, decimals)


def _format_number(value: Optional[float], decimals: int = 1, default_na: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return _try_format_number(value, decimals, default_na)


def _format_lcoh(value: Optional[float]) -> str:
    if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
        return f"${value:.3f}/kg"
    return "N/A"


def _format_lcoe_lcos(value: Optional[float]) -> str:
    if isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
        return f"${value:.2f}/MWh"
    return "N/A"


def _calculate_case1_lcoe(annual_perf: dict, plant_params: dict, financial_metrics: dict) -> Optional[float]:
    """
    Calculate LCOE for Case 1 using only OPEX costs over remaining lifetime
    and total electricity generation over remaining lifetime.

    LCOE = Present Value of Total OPEX / Present Value of Total Generation
    """
    try:
        # Get required data
        annual_generation_mwh = annual_perf.get('annual_generation_mwh')
        annual_opex_usd = annual_perf.get('annual_total_opex_usd')
        remaining_lifetime = plant_params.get(
            'remaining_plant_life_years') or financial_metrics.get('project_lifetime_years')

        # Default discount rate (8% as commonly used in the system)
        discount_rate = 0.08

        # Validate inputs
        if not all([
            annual_generation_mwh is not None,
            annual_opex_usd is not None,
            remaining_lifetime is not None,
            annual_generation_mwh > 0,
            annual_opex_usd > 0,
            remaining_lifetime > 0
        ]):
            logger.warning("Cannot calculate LCOE: missing or invalid data")
            return None

        # Calculate present values over remaining lifetime
        total_opex_pv = 0
        total_generation_pv = 0

        for year in range(1, int(remaining_lifetime) + 1):
            discount_factor = 1 / ((1 + discount_rate) ** year)
            total_opex_pv += annual_opex_usd * discount_factor
            total_generation_pv += annual_generation_mwh * discount_factor

        # Calculate LCOE
        if total_generation_pv > 0:
            lcoe = total_opex_pv / total_generation_pv
            logger.info(
                f"Case 1 LCOE calculated: ${lcoe:.2f}/MWh (OPEX-only over {remaining_lifetime} years)")
            return lcoe
        else:
            logger.warning(
                "Cannot calculate LCOE: total generation PV is zero")
            return None

    except Exception as e:
        logger.error(f"Error calculating Case 1 LCOE: {e}")
        return None


def generate_comprehensive_tea_summary_report(
    annual_metrics_rpt: dict,
    financial_metrics_rpt: dict,
    output_file_path: Path,
    target_iso_rpt: str,
    plant_specific_title_rpt: str,
    capex_data: dict,
    om_data: dict,
    replacement_data: dict,
    project_lifetime_years_rpt: int,
    construction_period_years_rpt: int,
    discount_rate_rpt: float,
    tax_rate_rpt: float,
    incremental_metrics_rpt: Optional[dict] = None,
) -> bool:
    """
    Generate comprehensive TEA summary report following standardized format.
    """
    logger.info(
        f"Generating comprehensive TEA summary report: {output_file_path}")

    annual_metrics_rpt = annual_metrics_rpt or {}
    financial_metrics_rpt = financial_metrics_rpt or {}
    incremental_metrics_rpt = incremental_metrics_rpt or {}

    try:
        report_title = plant_specific_title_rpt if plant_specific_title_rpt else f"{target_iso_rpt} Region"

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(f"Techno-Economic Analysis Report - {report_title}\n")
            f.write("=" * (len("Techno-Economic Analysis Report - ") +
                    len(report_title)) + "\n\n")

            _write_project_overview_section(
                f, annual_metrics_rpt, target_iso_rpt, project_lifetime_years_rpt,
                discount_rate_rpt, tax_rate_rpt
            )

            nuclear_baseline_analysis_rpt = annual_metrics_rpt.get(
                "nuclear_baseline_analysis", {})
            _write_existing_reactor_baseline_section(
                f, nuclear_baseline_analysis_rpt
            )

            nuclear_integrated_analysis_rpt = annual_metrics_rpt.get(
                "nuclear_integrated_analysis", {})
            lcoh_breakdown_analysis_rpt = annual_metrics_rpt.get(
                "lcoh_breakdown_analysis", {})
            _write_existing_reactor_retrofit_section(
                f, annual_metrics_rpt, financial_metrics_rpt,
                nuclear_integrated_analysis_rpt, lcoh_breakdown_analysis_rpt
            )

            _write_incremental_system_analysis_section(
                f, incremental_metrics_rpt, annual_metrics_rpt, tax_rate_rpt
            )

            greenfield_60yr_tax_analysis_main = annual_metrics_rpt.get(
                "comprehensive_tax_incentive_analysis", {})
            _write_greenfield_60yr_analysis_section(
                f, greenfield_60yr_tax_analysis_main
            )

            lifecycle_comparison_rpt = annual_metrics_rpt.get(
                "lifecycle_comparison_analysis", {})
            _write_greenfield_lifecycle_comparison_section(
                f, lifecycle_comparison_rpt
            )

            # Add lifecycle-specific cash flow analysis
            _write_lifecycle_cash_flow_analysis_section(
                f, lifecycle_comparison_rpt, project_lifetime_years_rpt
            )

            _write_detailed_performance_cost_section(
                f, annual_metrics_rpt, tax_rate_rpt
            )

            _write_core_assumptions_section(
                f, capex_data, om_data, replacement_data,
                project_lifetime_years_rpt, construction_period_years_rpt,
                discount_rate_rpt, tax_rate_rpt, annual_metrics_rpt
            )

            f.write("\nReport generated successfully.\n")

        logger.info(
            f"Comprehensive TEA summary report saved to {output_file_path}")
        return True

    except Exception as e:
        logger.error(
            f"Error generating comprehensive TEA summary report: {str(e)}", exc_info=True)
        return False


def _write_project_overview_section(
    f, annual_metrics_rpt: dict, target_iso_rpt: str, project_lifetime_years_rpt: int,
    discount_rate_rpt: float, tax_rate_rpt: float
):
    f.write("1. Project Overview & Nuclear Unit Characteristics\n")
    f.write("-" * 50 + "\n")

    actual_lifetime_years = project_lifetime_years_rpt
    lifetime_note = "(Note: Specific cases may use different lifetimes)"

    nuclear_baseline_analysis_data = annual_metrics_rpt.get(
        "nuclear_baseline_analysis", {})
    nuclear_baseline_fin_metrics = nuclear_baseline_analysis_data.get(
        "financial_metrics", {})
    nuclear_baseline_plant_params = nuclear_baseline_analysis_data.get(
        "plant_parameters", {})

    specific_lifetime_fm = nuclear_baseline_fin_metrics.get(
        "project_lifetime_years")
    specific_lifetime_pp = nuclear_baseline_plant_params.get(
        'remaining_plant_life_years')

    if specific_lifetime_fm is not None and isinstance(specific_lifetime_fm, (int, float)) and specific_lifetime_fm > 0:
        actual_lifetime_years = specific_lifetime_fm
        lifetime_note = f"(Plant Remaining Lifetime from Case 1 financial_metrics: {_format_number(actual_lifetime_years,0)} yrs)"
    elif specific_lifetime_pp is not None and isinstance(specific_lifetime_pp, (int, float)) and specific_lifetime_pp > 0:
        actual_lifetime_years = specific_lifetime_pp
        lifetime_note = f"(Plant Remaining Lifetime from Case 1 plant_parameters: {_format_number(actual_lifetime_years,0)} yrs)"

    overview_items = {
        "ISO Region": target_iso_rpt,
        "Project Lifecycle": f"{_format_number(actual_lifetime_years,0)} years {lifetime_note}",
        "Discount Rate": _format_percentage(discount_rate_rpt * 100),
        "Tax Rate": _format_percentage(tax_rate_rpt * 100)
    }

    # Enhanced nuclear capacity reporting with data source transparency
    nuclear_capacity_actual = nuclear_baseline_plant_params.get(
        "turbine_capacity_mw")
    if nuclear_capacity_actual is None:
        nuclear_capacity_actual = annual_metrics_rpt.get("Turbine_Capacity_MW")

    # Try to get nameplate capacity from NPPs info if available
    nuclear_capacity_nameplate = None
    data_sources = nuclear_baseline_analysis_data.get("data_sources", {})
    npp_info_available = data_sources.get(
        "plant_capacity_source") == "NPPs info file"

    if npp_info_available and nuclear_baseline_plant_params:
        # If we have NPPs info, check for nameplate capacity fields
        nuclear_capacity_nameplate = nuclear_baseline_plant_params.get(
            "nameplate_capacity_mw")

    # FIXED: Display capacity information with source transparency - prioritize nameplate capacity for Case 1
    if nuclear_capacity_nameplate is not None and nuclear_capacity_actual is not None:
        # Both nameplate and actual available - always prioritize nameplate for Case 1 display
        if abs(nuclear_capacity_nameplate - nuclear_capacity_actual) > 1.0:  # Significant difference
            overview_items["Nuclear Unit Capacity (MW)"] = f"{_format_number(nuclear_capacity_nameplate, 2)} (from NPPs info file)"
            overview_items["Nuclear Unit Operating Capacity (MW)"] = f"{_format_number(nuclear_capacity_actual, 2)} (from optimization results)"
        else:
            # Even if values are close, prioritize nameplate capacity
            overview_items["Nuclear Unit Capacity (MW)"] = f"{_format_number(nuclear_capacity_nameplate, 2)} (from NPPs info file)"
    elif nuclear_capacity_nameplate is not None:
        # Only nameplate capacity available (preferred)
        overview_items["Nuclear Unit Capacity (MW)"] = f"{_format_number(nuclear_capacity_nameplate, 2)} (from NPPs info file)"
    elif nuclear_capacity_actual is not None:
        # Only actual capacity available (fallback)
        capacity_source = data_sources.get(
            "plant_capacity_source", "optimization results")
        overview_items["Nuclear Unit Capacity (MW)"] = f"{_format_number(nuclear_capacity_actual, 2)} (from {capacity_source})"

    thermal_capacity_mwt = nuclear_baseline_plant_params.get("thermal_capacity_mwt",
                                                             annual_metrics_rpt.get("Thermal_Capacity_MWt",
                                                                                    annual_metrics_rpt.get("thermal_capacity_mwt")))
    thermal_efficiency = nuclear_baseline_plant_params.get("thermal_efficiency",
                                                           annual_metrics_rpt.get("thermal_efficiency",
                                                                                  annual_metrics_rpt.get("Thermal_Efficiency")))

    if thermal_capacity_mwt is not None and isinstance(thermal_capacity_mwt, (int, float)) and thermal_capacity_mwt > 0:
        overview_items["Nuclear Unit Thermal Capacity (MWt)"] = _format_number(
            thermal_capacity_mwt, 2)

    if thermal_efficiency is not None and isinstance(thermal_efficiency, (int, float)) and thermal_efficiency > 0:
        overview_items["Nuclear Unit Thermal Efficiency"] = f"{_format_number(thermal_efficiency, 4)} ({_format_percentage(thermal_efficiency * 100, 2)})"

    overview_items["Other Key Configuration Parameters"] = "Refer to specific case sections and assumptions"

    # Add data source transparency note if there are capacity discrepancies
    if nuclear_capacity_nameplate is not None and nuclear_capacity_actual is not None:
        if abs(nuclear_capacity_nameplate - nuclear_capacity_actual) > 1.0:
            overview_items["Data Source Note"] = "Capacity values reflect both design specifications and operational constraints"

    f.write(format_aligned_section(overview_items, min_width=40))
    f.write("\n")


def _write_existing_reactor_baseline_section(f, nuclear_baseline_analysis_rpt: dict):
    f.write("2. Case 1: Existing Reactor Baseline Operation Analysis\n")
    f.write("-" * 55 + "\n")

    f.write("  Objective & Description:\n")
    f.write(
        "    Analysis of an existing nuclear reactor's baseline operations without\n")
    f.write(
        "    modifications for hydrogen production. This case evaluates the financial\n")
    f.write("    performance of the nuclear plant operating as-is, considering its\n")
    f.write(
        "    remaining lifetime and typical operational parameters. It serves as a\n")
    f.write(
        "    benchmark for comparison with retrofit or integrated system scenarios.\n\n")

    if not nuclear_baseline_analysis_rpt or not nuclear_baseline_analysis_rpt.get("plant_parameters"):
        f.write(
            "  Status: Existing Reactor Baseline Analysis data not available or incomplete.\n\n")
        return

    plant_params = nuclear_baseline_analysis_rpt.get("plant_parameters", {})
    annual_perf = nuclear_baseline_analysis_rpt.get("annual_performance", {})
    scenario_no_45u = nuclear_baseline_analysis_rpt.get(
        "scenario_without_45u", {})
    scenario_45u = nuclear_baseline_analysis_rpt.get("scenario_with_45u", {})
    impact_45u = nuclear_baseline_analysis_rpt.get("45u_policy_impact", {})
    benefits_45u = nuclear_baseline_analysis_rpt.get(
        "nuclear_45u_benefits", {})
    financial_metrics_no_45u = nuclear_baseline_analysis_rpt.get(
        "financial_metrics", scenario_no_45u)

    f.write("  Key Assumptions:\n")
    remaining_lifetime_val = plant_params.get(
        'remaining_plant_life_years', financial_metrics_no_45u.get('project_lifetime_years'))
    assumptions = {
        "Remaining Lifetime": f"{_format_number(remaining_lifetime_val, 0, default_na=True)} years",
        "Nameplate Power Factor": _format_number(plant_params.get('nameplate_power_factor'), 3, default_na=True)
    }
    f.write(format_aligned_section(assumptions, min_width=35, indent="    "))

    f.write("\n  Financial Metrics (NPV, IRR, Payback - without 45U):\n")
    fin_metrics_no_45u_items = {
        "NPV": _format_currency(scenario_no_45u.get("npv_usd")),
        "IRR": _format_percentage(scenario_no_45u.get("irr_percent")),
        "Payback": f"{_format_number(scenario_no_45u.get('payback_period_years'), default_na=True)} years"
    }
    f.write(format_aligned_section(
        fin_metrics_no_45u_items, min_width=25, indent="    "))

    # Calculate and display LCOE (Levelized Cost of Electricity) for Case 1
    f.write(
        "\n  LCOE (Levelized Cost of Electricity - OPEX only over remaining lifetime):\n")
    lcoe_value = _calculate_case1_lcoe(
        annual_perf, plant_params, financial_metrics_no_45u)
    lcoe_items = {
        "LCOE (Nuclear OPEX only)": _format_lcoe_lcos(lcoe_value) if lcoe_value is not None else "N/A"
    }
    f.write(format_aligned_section(lcoe_items, min_width=25, indent="    "))

    if benefits_45u and benefits_45u.get('total_45u_credits', 0) > 0:
        f.write("\n  45U PTC Policy Impact:\n")
        f.write("    Financial Metrics (NPV, IRR, Payback - with 45U):\n")
        fin_metrics_45u_items = {
            "NPV": _format_currency(scenario_45u.get("npv_usd")),
            "IRR": _format_percentage(scenario_45u.get("irr_percent")),
            "Payback": f"{_format_number(scenario_45u.get('payback_period_years'), default_na=True)} years"
        }
        f.write(format_aligned_section(
            fin_metrics_45u_items, min_width=25, indent="      "))

        f.write("\n    NPV Improvement etc.:\n")
        npv_improvement_items = {
            "NPV Improvement": _format_currency(impact_45u.get("npv_improvement_usd")),
            "Total 45U Credits": _format_currency(impact_45u.get("total_45u_credits_usd"))
        }
        f.write(format_aligned_section(
            npv_improvement_items, min_width=25, indent="      "))
    else:
        f.write(
            "\n  45U PTC Policy Impact: Not applicable or no benefits calculated for this scenario.\n")

    f.write("\n  Key Annual Operating Data (Generation, Revenue, OPEX etc.):\n")
    op_data = {
        "Annual Generation": f"{_format_number(annual_perf.get('annual_generation_mwh'), 0)} MWh",
        "Annual Revenue": _format_currency(annual_perf.get('annual_revenue_usd')),
        "Total Annual OPEX": _format_currency(annual_perf.get('annual_total_opex_usd'))
    }
    f.write(format_aligned_section(op_data, min_width=30, indent="    "))
    f.write("\n")


def _write_existing_reactor_retrofit_section(
    f, annual_metrics_rpt: dict, financial_metrics_rpt: dict,
    nuclear_integrated_analysis_rpt: dict, lcoh_breakdown_analysis_rpt: dict
):
    f.write("3. Case 2: Existing Reactor Retrofit Analysis (Integrated Hydrogen and Battery Systems)\n")
    f.write("-" * 80 + "\n")
    f.write("  Objective & Description:\n")
    f.write("    Analysis of an existing nuclear reactor retrofitted with integrated hydrogen\n")
    f.write(
        "    production and battery storage systems. This case evaluates the combined\n")
    f.write(
        "    financial and operational performance of the nuclear plant augmented with\n")
    f.write("    these new energy system components.\n\n")

    f.write("  New System Capacities (Electrolyzer, H2 Storage, Battery):\n")
    capacities = {
        "Electrolyzer Capacity": f"{_format_number(annual_metrics_rpt.get('Electrolyzer_Capacity_MW'), 2)} MW",
        "H2 Storage Capacity": f"{_format_number(annual_metrics_rpt.get('H2_Storage_Capacity_kg'), 0)} kg",
        "Battery Energy Capacity": f"{_format_number(annual_metrics_rpt.get('Battery_Capacity_MWh'), 2)} MWh",
        "Battery Power Capacity": f"{_format_number(annual_metrics_rpt.get('Battery_Power_MW'), 2)} MW"
    }
    f.write(format_aligned_section(capacities, min_width=35, indent="    "))

    f.write("\n  Financial Metrics (Integrated System - NPV, IRR, Payback - nuclear part without 45U):\n")
    fin_metrics_integrated_no_45U = {
        "NPV": _format_currency(financial_metrics_rpt.get("NPV_USD")),
        "IRR": _format_percentage(financial_metrics_rpt.get("IRR_percent")),
        "Payback": f"{_format_number(financial_metrics_rpt.get('Payback_Period_Years'), default_na=True)} years"
    }
    f.write(format_aligned_section(
        fin_metrics_integrated_no_45U, min_width=30, indent="    "))

    if nuclear_integrated_analysis_rpt and nuclear_integrated_analysis_rpt.get("includes_45u_analysis"):
        scenario_with_45u = nuclear_integrated_analysis_rpt.get(
            "scenario_with_45u", {})
        f.write("\n  45U PTC Policy Impact on Nuclear Portion (Integrated System Financials with 45U for Nuclear):\n")
        fin_metrics_integrated_with_45U = {
            "NPV": _format_currency(scenario_with_45u.get("npv_usd")),
            "IRR": _format_percentage(scenario_with_45u.get("irr_percent")),
            "Payback": f"{_format_number(scenario_with_45u.get('payback_period_years'), default_na=True)} years"
        }
        f.write(format_aligned_section(
            fin_metrics_integrated_with_45U, min_width=30, indent="    "))
    else:
        f.write(
            "\n  45U PTC Policy Impact on Nuclear Portion: Data not available for this configuration.\n")

    f.write("\n  LCOH (Detailed Composition):\n")
    if lcoh_breakdown_analysis_rpt and "total_lcoh_usd_per_kg" in lcoh_breakdown_analysis_rpt:
        total_lcoh = lcoh_breakdown_analysis_rpt.get("total_lcoh_usd_per_kg")
        f.write(f"    Total LCOH: {_format_lcoh(total_lcoh)}\n")
        lcoh_breakdown = lcoh_breakdown_analysis_rpt.get(
            "lcoh_breakdown_usd_per_kg", {})
        lcoh_percentages = lcoh_breakdown_analysis_rpt.get(
            "lcoh_percentages", {})
        if lcoh_breakdown:
            sorted_components = sorted(lcoh_breakdown.items(
            ), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0, reverse=True)
            for comp, cost_val in sorted_components:
                display_comp_name = comp.replace(
                    "CAPEX_", "").replace("_", " ").title()
                if isinstance(cost_val, (int, float)) and cost_val > 0.0001:
                    f.write(
                        f"      {display_comp_name:<40}: {_format_lcoh(cost_val)} ({_format_percentage(lcoh_percentages.get(comp,0))})\n")
    else:
        f.write("    LCOH breakdown data not available.\n")

    f.write("\n  Key Annual Operating Data (Total Revenue, Total OPEX, H2 Production, AS Revenue etc.):\n")
    op_data_retrofit = {
        "Total Annual Revenue": _format_currency(annual_metrics_rpt.get("Annual_Revenue")),
        "Total System OPEX": _format_currency(annual_metrics_rpt.get("Total_System_OPEX_Annual_USD")),
        "Annual H2 Production": f"{_format_number(annual_metrics_rpt.get('H2_Production_kg_annual'), 0)} kg",
        "Ancillary Services Revenue": _format_currency(annual_metrics_rpt.get("AS_Revenue"))
    }
    f.write(format_aligned_section(
        op_data_retrofit, min_width=35, indent="    "))
    f.write("\n")


def _write_lifecycle_cash_flow_analysis_section(f, lifecycle_comparison_rpt: dict, project_lifetime_years: int):
    """Write lifecycle-specific cash flow analysis section"""
    if not lifecycle_comparison_rpt:
        return

    f.write("\n6. Lifecycle Cash Flow Analysis\n")
    f.write("-" * 32 + "\n")
    f.write("  Analysis of cash flow patterns across different project lifecycles,\n")
    f.write("  comparing existing plant retrofit scenarios with new construction projects.\n\n")

    # Extract cash flow data
    results_60yr = lifecycle_comparison_rpt.get("60_year_results", {})
    results_80yr = lifecycle_comparison_rpt.get("80_year_results", {})

    if results_60yr and results_80yr:
        baseline_60yr = results_60yr.get("baseline_scenario", {})
        baseline_80yr = results_80yr.get("baseline_scenario", {})

        f.write("  A. New Construction Lifecycle Comparison:\n")

        # Financial metrics comparison
        npv_60 = baseline_60yr.get("npv_usd", 0)
        npv_80 = baseline_80yr.get("npv_usd", 0)
        irr_60 = baseline_60yr.get("irr_percent", 0)
        irr_80 = baseline_80yr.get("irr_percent", 0)

        comparison_items = {
            "60-Year Project NPV": _format_currency(npv_60),
            "80-Year Project NPV": _format_currency(npv_80),
            "NPV Improvement (80yr vs 60yr)": _format_currency(npv_80 - npv_60),
            "60-Year Project IRR": f"{irr_60:.2f}%" if irr_60 else "N/A",
            "80-Year Project IRR": f"{irr_80:.2f}%" if irr_80 else "N/A",
            "IRR Improvement (80yr vs 60yr)": f"{irr_80 - irr_60:.2f}%" if irr_60 and irr_80 else "N/A"
        }

        f.write(format_aligned_section(
            comparison_items, min_width=35, indent="    "))

        # Cash flow characteristics
        cf_60yr = baseline_60yr.get("cash_flows", [])
        cf_80yr = baseline_80yr.get("cash_flows", [])

        if cf_60yr and cf_80yr:
            f.write("\n  B. Cash Flow Characteristics:\n")

            # Calculate payback periods
            payback_60 = baseline_60yr.get("payback_period_years", 0)
            payback_80 = baseline_80yr.get("payback_period_years", 0)

            # Calculate average annual cash flows during operations
            construction_period = 8  # New construction period
            if len(cf_60yr) > construction_period:
                avg_annual_cf_60 = np.mean(cf_60yr[construction_period:])
            else:
                avg_annual_cf_60 = 0

            if len(cf_80yr) > construction_period:
                avg_annual_cf_80 = np.mean(cf_80yr[construction_period:])
            else:
                avg_annual_cf_80 = 0

            cf_characteristics = {
                "60-Year Payback Period": f"{payback_60:.1f} years" if payback_60 else "N/A",
                "80-Year Payback Period": f"{payback_80:.1f} years" if payback_80 else "N/A",
                "60-Year Avg Annual Operating CF": _format_currency(avg_annual_cf_60),
                "80-Year Avg Annual Operating CF": _format_currency(avg_annual_cf_80),
                "Extended Operations Benefit": _format_currency((avg_annual_cf_80 * 20)) if avg_annual_cf_80 > 0 else "N/A"
            }

            f.write(format_aligned_section(
                cf_characteristics, min_width=35, indent="    "))

    # Compare with existing plant scenarios
    f.write(f"\n  C. Existing Plant vs New Construction Comparison:\n")
    f.write(
        f"    Existing Plant Remaining Life: {project_lifetime_years} years\n")
    f.write(f"    New Construction Options: 60 years (Case 4) or 80 years (Case 5)\n")
    f.write(f"    \n")
    f.write(f"    Key Insights:\n")
    f.write(f"    • Existing plants have limited remaining lifetime but lower initial investment\n")
    f.write(f"    • New construction requires higher CAPEX but offers full lifecycle benefits\n")
    f.write(
        f"    • 80-year lifecycle provides additional 20 years of revenue generation\n")
    f.write(
        f"    • Lifecycle extension significantly improves long-term financial metrics\n")

    f.write("\n  Note: Cash flow visualizations are available in the plots directory,\n")
    f.write("  including lifecycle comparison charts for detailed analysis.\n")


def _write_incremental_system_analysis_section(f, incremental_metrics_rpt: dict, annual_metrics_rpt: dict, tax_rate_rpt: float):
    f.write("4. Case 3: Incremental System Financial Analysis (Hydrogen and Battery Systems)\n")
    f.write("-" * 75 + "\n")
    f.write("  Objective & Description:\n")
    f.write("    Financial analysis focusing solely on the incremental investment in hydrogen\n")
    f.write(
        "    and battery systems, assuming they are added to an existing operational\n")
    f.write(
        "    nuclear plant. This evaluates the standalone economics of the H2/battery\n")
    f.write(
        "    project, considering its own CAPEX, revenues, and opportunity costs.\n\n")

    if not incremental_metrics_rpt:
        f.write(
            "  Status: Incremental System Financial Analysis data not available.\n\n")
        return

    f.write("  Incremental Investment CAPEX:\n")
    inc_capex_total = incremental_metrics_rpt.get(
        "Total_Incremental_CAPEX_Learned_USD", 0)
    f.write(
        f"    Total Incremental CAPEX: {_format_currency(inc_capex_total)}\n")
    inc_capex_breakdown = incremental_metrics_rpt.get(
        "incremental_capex_breakdown", {})
    if inc_capex_breakdown:
        f.write("    CAPEX Breakdown:\n")
        sorted_capex_breakdown = sorted(inc_capex_breakdown.items(
        ), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0, reverse=True)
        for comp, cost_val in sorted_capex_breakdown:
            display_comp_name = comp.replace("_System", "").replace("_", " ")
            if isinstance(cost_val, (int, float)) and cost_val > 0:
                f.write(
                    f"      {display_comp_name:<35}: {_format_currency(cost_val)} ({_format_percentage((cost_val / inc_capex_total * 100) if inc_capex_total else 0)})\n")

    f.write("\n  Incremental Financial Metrics (NPV, IRR, Payback):\n")
    inc_fin_metrics = {
        "NPV": _format_currency(incremental_metrics_rpt.get("NPV_USD")),
        "IRR": _format_percentage(incremental_metrics_rpt.get("IRR_percent")),
        "Payback": f"{_format_number(incremental_metrics_rpt.get('Payback_Period_Years'), default_na=True)} years"
    }
    f.write(format_aligned_section(
        inc_fin_metrics, min_width=30, indent="    "))

    f.write("\n  MACRS Depreciation Impact on Incremental Project:\n")
    inc_macrs_total_dep_arr = incremental_metrics_rpt.get(
        "incremental_macrs_total_depreciation")

    if inc_macrs_total_dep_arr is not None and isinstance(inc_macrs_total_dep_arr, (list, np.ndarray)) and len(inc_macrs_total_dep_arr) > 0:
        total_dep_val = np.sum(inc_macrs_total_dep_arr)
        tax_shield_val = total_dep_val * tax_rate_rpt
        macrs_impact = {
            "Total Incremental MACRS Depreciation": _format_currency(total_dep_val),
            "Tax Shield Effect (Value)": _format_currency(tax_shield_val),
            "NPV Contribution from MACRS (Approximation)": _format_currency(incremental_metrics_rpt.get("Incremental_MACRS_Total_Tax_Benefit_USD", tax_shield_val))
        }
        f.write(format_aligned_section(
            macrs_impact, min_width=45, indent="    "))
    else:
        f.write(
            "    MACRS depreciation data not available or not applicable for incremental project.\n")

    f.write("\n  Key Annual Incremental Operating Data (Incremental Revenue, Cost, Opportunity Costs):\n")
    inc_op_data = {
        "Incremental Revenue": _format_currency(incremental_metrics_rpt.get("Annual_Incremental_Revenue_USD", 0)),
        "Incremental Costs": _format_currency(incremental_metrics_rpt.get("Annual_Incremental_Costs_USD", 0)),
        "Electricity Opportunity Cost": _format_currency(incremental_metrics_rpt.get("Annual_Electricity_Opportunity_Cost_USD")),
        "Thermal Energy Opportunity Cost": _format_currency(incremental_metrics_rpt.get("Annual_HTE_Thermal_Opportunity_Cost_USD"))
    }
    f.write(format_aligned_section(inc_op_data, min_width=35, indent="    "))
    f.write("\n")


def _write_greenfield_60yr_analysis_section(f, greenfield_60yr_tax_analysis_main_dict: dict):
    f.write("5. Case 4: Greenfield Nuclear-Hydrogen Project (60-year) - Federal Tax Incentive Analysis\n")
    f.write("-" * 90 + "\n")
    f.write("  Objective & Description:\n")
    f.write(
        "    Analysis of a new, purpose-built nuclear-hydrogen integrated system with a\n")
    f.write("    60-year project lifetime. This case evaluates the project's financial viability\n")
    f.write("    under different federal tax incentive scenarios: a baseline (no incentives),\n")
    f.write("    the 45Y Production Tax Credit (PTC), and the 48E Investment Tax Credit (ITC).\n\n")

    tax_analysis_content = greenfield_60yr_tax_analysis_main_dict.get(
        "tax_incentive_analysis", {})

    if not tax_analysis_content or "scenarios" not in tax_analysis_content:
        f.write("  Status: Greenfield Nuclear-Hydrogen (60yr) Tax Incentive Analysis data not available or incomplete.\n")
        f.write("          (Expected under 'comprehensive_tax_incentive_analysis' -> 'tax_incentive_analysis' -> 'scenarios' in input data)\n\n")
        return

    system_config = greenfield_60yr_tax_analysis_main_dict.get(
        "system_configuration", tax_analysis_content.get("system_configuration", {}))

    f.write("  System Configuration (Nuclear Capacity, Hydrogen/Battery Capacity):\n")
    config_items = {
        "Nuclear Capacity": f"{_format_number(system_config.get('nuclear_capacity_mw'), 0)} MW",
        "Hydrogen System Capacity (Electrolyzer)": f"{_format_number(system_config.get('electrolyzer_capacity_mw'),0)} MW",
        "Battery Capacity (Energy)": f"{_format_number(system_config.get('battery_capacity_mwh'),0)} MWh"
    }
    f.write(format_aligned_section(config_items, min_width=45, indent="    "))

    scenarios = tax_analysis_content.get("scenarios", {})
    baseline_scenario_fin_metrics = scenarios.get(
        "baseline", {}).get("financial_metrics", {})

    f.write("\n  Tax Incentive Scenario Comparison:\n")
    for scenario_key, scenario_name in [("baseline", "Baseline (No Incentives)"),
                                        ("ptc", "45Y PTC Production Tax Credit"),
                                        ("itc", "48E ITC Investment Tax Credit")]:

        scenario_fin_metrics = scenarios.get(
            scenario_key, {}).get("financial_metrics", {})
        f.write(f"    {scenario_name}:\n")
        metrics_to_display = {
            "NPV": _format_currency(scenario_fin_metrics.get("npv_usd")),
            "IRR": _format_percentage(scenario_fin_metrics.get("irr_percent")),
            "Payback": f"{_format_number(scenario_fin_metrics.get('payback_period_years'), default_na=True)} years",
        }

        metrics_to_display["LCOH (Hydrogen)"] = _format_lcoh(
            scenario_fin_metrics.get("lcoh_usd_per_kg"))

        # Use LCOE/LCOS from the 'baseline' tax scenario's financial_metrics as the reference for the physical plant design
        metrics_to_display["LCOE (Nuclear)"] = _format_lcoe_lcos(scenario_fin_metrics.get("lcoe_nuclear_usd_per_mwh",
                                                                                          baseline_scenario_fin_metrics.get("lcoe_nuclear_usd_per_mwh")))
        metrics_to_display["LCOS (Battery, if applicable)"] = _format_lcoe_lcos(scenario_fin_metrics.get("lcos_battery_usd_per_mwh",
                                                                                baseline_scenario_fin_metrics.get("lcos_battery_usd_per_mwh")))

        f.write(format_aligned_section(
            metrics_to_display, min_width=30, indent="      "))

        if scenario_key == "ptc":
            ptc_details = scenarios.get("ptc", {}).get(
                "analysis", {}).get("tax_benefits", {}).get("ptc", {})
            ptc_value = ptc_details.get("total_ptc_value_usd", 0)
            f.write(
                f"        PTC Value Assessment: {_format_currency(ptc_value)} (Total over 10 years)\n")
        elif scenario_key == "itc":
            itc_details = scenarios.get("itc", {}).get(
                "analysis", {}).get("tax_benefits", {}).get("itc", {})
            itc_credit = itc_details.get("itc_credit_amount_usd", 0)
            dep_reduction = itc_details.get(
                "depreciation_basis_reduction_usd", 0)
            f.write(
                f"        ITC Value Assessment: {_format_currency(itc_credit)} (Upfront credit)\n")
            f.write(
                f"        ITC Impact on Depreciation Basis: -{_format_currency(dep_reduction)}\n")
        f.write("\n")

    financial_comp = tax_analysis_content.get("financial_comparison",
                                              greenfield_60yr_tax_analysis_main_dict.get("financial_comparison", {}))
    f.write("  Recommended Incentive Scheme & Key Findings:\n")
    best_scenario_key = financial_comp.get("best_scenario", "baseline")
    best_scenario_name_map = {
        "baseline": "Baseline (No Incentives)", "ptc": "45Y PTC", "itc": "48E ITC"}
    f.write(
        f"    Recommended Scenario: {best_scenario_name_map.get(best_scenario_key, best_scenario_key)}\n")

    baseline_npv_comp = financial_comp.get("baseline_npv", 0)
    ptc_npv_comp = financial_comp.get("ptc_npv", 0)
    itc_npv_comp = financial_comp.get("itc_npv", 0)

    findings = []
    if best_scenario_key == "ptc" and ptc_npv_comp > baseline_npv_comp:
        findings.append(
            f"45Y PTC significantly improves NPV to {_format_currency(ptc_npv_comp)} from {_format_currency(baseline_npv_comp)} (Baseline).")
    elif best_scenario_key == "itc" and itc_npv_comp > baseline_npv_comp:
        findings.append(
            f"48E ITC significantly improves NPV to {_format_currency(itc_npv_comp)} from {_format_currency(baseline_npv_comp)} (Baseline).")
    elif best_scenario_key == "baseline" and baseline_npv_comp >= ptc_npv_comp and baseline_npv_comp >= itc_npv_comp:
        findings.append(
            f"Baseline scenario (NPV: {_format_currency(baseline_npv_comp)}) is currently the most favorable or equal to tax incentive scenarios.")
    else:
        findings.append(
            f"Review NPVs: Baseline ({_format_currency(baseline_npv_comp)}), PTC ({_format_currency(ptc_npv_comp)}), ITC ({_format_currency(itc_npv_comp)}).")

    for finding in findings:
        f.write(f"    Key Finding: {finding}\n")
    f.write("\n")


def _write_greenfield_lifecycle_comparison_section(f, lifecycle_comparison_rpt: dict):
    f.write("6. Case 5: Greenfield Nuclear-Hydrogen Project Lifecycle Comparison (60 years vs. 80 years) - Federal Tax Incentive Analysis\n")
    f.write("-" * 120 + "\n")
    f.write("  Objective & Description:\n")
    f.write("    Comparative analysis of new nuclear-hydrogen integrated systems with varying\n")
    f.write("    project lifecycles (60 years vs. 80 years). This evaluation includes the impact\n")
    f.write(
        "    of federal tax incentives (Baseline, 45Y PTC, 48E ITC) on the financial\n")
    f.write("    viability of these long-term projects.\n\n")

    results_60yr = lifecycle_comparison_rpt.get("60_year_results", {})
    results_80yr = lifecycle_comparison_rpt.get("80_year_results", {})

    data_missing_60yr = not results_60yr or \
        "tax_incentive_analysis" not in results_60yr or \
        not results_60yr["tax_incentive_analysis"].get("scenarios") or \
        "baseline_greenfield_results" not in results_60yr

    data_missing_80yr = not results_80yr or \
        "tax_incentive_analysis" not in results_80yr or \
        not results_80yr["tax_incentive_analysis"].get("scenarios") or \
        "baseline_greenfield_results" not in results_80yr

    if data_missing_60yr or data_missing_80yr:
        f.write("  Status: Lifecycle Comparison Analysis data not available or incomplete for one or both lifecycles.\n")
        if data_missing_60yr:
            f.write(
                "          - Missing or incomplete data for 60-year lifecycle results.\n")
        if data_missing_80yr:
            f.write(
                "          - Missing or incomplete data for 80-year lifecycle results.\n")
        f.write(
            "          (Expected under 'lifecycle_comparison_analysis' -> '[60/80]_year_results' -> \n")
        f.write("           both 'baseline_greenfield_results' AND 'tax_incentive_analysis' with 'scenarios' must be present).\n\n")
        return

    f.write("  Comparison Summary Table:\n")
    header_parts = ["Lifecycle", "Tax Scenario",
                    "NPV ($M)", "IRR (%)", "Payback (yrs)", "LCOH ($/kg)", "LCOE ($/MWh)", "LCOS ($/MWh)"]
    col_widths = [9, 12, 10, 7, 13, 11, 12, 12]
    header_format = " | ".join(
        [f"{{:<{w}}}" if i < 2 else f"{{:>{w}}}" for i, w in enumerate(col_widths)])
    header_str = header_format.format(*header_parts)

    f.write("  " + "-" * len(header_str) + "\n")
    f.write("  " + header_str + "\n")
    f.write("  " + "-" * len(header_str) + "\n")

    scenarios_data_map = {
        "60 years": results_60yr,
        "80 years": results_80yr
    }
    tax_scenario_map = {
        "baseline": "Baseline",
        "ptc": "45Y PTC",
        "itc": "48E ITC"
    }

    for life_name, life_data_main in scenarios_data_map.items():
        # baseline_greenfield_res_life contains LCOE/LCOS for this lifecycle's baseline physical plant
        baseline_greenfield_res_life = life_data_main.get(
            "baseline_greenfield_results", {})
        # tax_scenarios_overall_life contains financial_metrics for baseline, ptc, itc tax cases
        tax_scenarios_overall_life = life_data_main.get(
            "tax_incentive_analysis", {}).get("scenarios", {})

        for tax_key, tax_name in tax_scenario_map.items():
            # Get the specific financial_metrics for the current tax_key (baseline, ptc, or itc)
            current_scenario_fin_metrics = tax_scenarios_overall_life.get(
                tax_key, {}).get("financial_metrics", {})

            if not current_scenario_fin_metrics:
                continue

            npv_m = current_scenario_fin_metrics.get("npv_usd", 0) / 1e6
            irr = current_scenario_fin_metrics.get("irr_percent")
            payback = current_scenario_fin_metrics.get("payback_period_years")

            # LCOH should be in each tax scenario's metrics
            lcoh = current_scenario_fin_metrics.get("lcoh_usd_per_kg")

            # For LCOE/LCOS, default to the values from this lifecycle's baseline_greenfield_results,
            # but allow override if the specific tax scenario's metrics contain them.
            lcoe = current_scenario_fin_metrics.get("lcoe_nuclear_usd_per_mwh",
                                                    baseline_greenfield_res_life.get("nuclear_lcoe_usd_per_mwh"))
            lcos = current_scenario_fin_metrics.get("lcos_battery_usd_per_mwh",
                                                    baseline_greenfield_res_life.get("battery_lcos_usd_per_mwh"))

            row_values = [
                life_name, tax_name,
                f"{npv_m:.1f}",
                _format_percentage(irr, 1) if irr is not None else "N/A",
                _format_number(
                    payback, 1, default_na=True) if payback is not None else "N/A",
                _format_lcoh(lcoh) if lcoh is not None else "N/A",
                _format_lcoe_lcos(lcoe) if isinstance(
                    lcoe, (int, float)) else lcoe,
                _format_lcoe_lcos(lcos) if isinstance(
                    lcos, (int, float)) else lcos
            ]
            f.write("  " + header_format.format(*row_values) + "\n")

    f.write("  " + "-" * len(header_str) + "\n\n")

    f.write("  Different Lifecycle Key Differences Analysis and Conclusions:\n")
    f.write("    - Longer lifecycles generally improve NPV due to extended revenue generation and longer amortization of initial CAPEX.\n")
    f.write("    - LCOH and LCOE may decrease with longer amortization periods for fixed costs, though more replacement cycles are also incurred.\n")
    f.write("    - The relative benefit of tax incentives (PTC vs. ITC) may shift with different project lifetimes and cash flow profiles.\n")
    f.write("    - Decision-making should weigh improved long-term returns against potential increased risks and O&M over extended periods.\n")
    f.write("\n")


def _write_detailed_performance_cost_section(f, annual_metrics_rpt: dict, tax_rate_rpt: float):
    f.write("7. Detailed Performance & Cost Analysis\n")
    f.write("-" * 40 + "\n")

    f.write("  A. Annual System Performance Indicators (from typical year `annual_metrics_rpt`):\n")
    if not annual_metrics_rpt:
        f.write("    Performance data not available.\n")
    else:
        perf_indicator_lines = [
            ("Total Generation/H2 Production/Battery Throughput", None),
            ("  Total Nuclear Generation",
             f"{_format_number(annual_metrics_rpt.get('Annual_Nuclear_Generation_MWh'), 0)} MWh"),
            ("  Annual H2 Production",
             f"{_format_number(annual_metrics_rpt.get('H2_Production_kg_annual'), 0)} kg"),
            ("  Battery Annual Throughput",
             f"{_format_number(annual_metrics_rpt.get('Annual_Battery_Throughput_MWh', annual_metrics_rpt.get('Annual_Battery_Charge_MWh',0) + annual_metrics_rpt.get('Annual_Battery_Discharge_MWh',0)), 0)} MWh"),
            ("Capacity Factors", None),
            ("  Electrolyzer Capacity Factor", _format_percentage(
                annual_metrics_rpt.get("Electrolyzer_CF_percent"))),
            ("  Turbine Capacity Factor", _format_percentage(
                annual_metrics_rpt.get("Turbine_CF_percent"))),
            ("  Battery Capacity Factor", _format_percentage(
                annual_metrics_rpt.get("Battery_CF_percent"))),
            ("Average Prices", None),
            ("  Average Electricity Price",
             f"${_format_number(annual_metrics_rpt.get('Avg_Electricity_Price_USD_per_MWh'), 2)}/MWh"),
            ("  Average Hydrogen Price (if set)",
             f"${_format_number(annual_metrics_rpt.get('Avg_H2_Price_USD_per_kg'), 2)}/kg"),
            ("Ancillary Services", None),
            ("  Total AS Revenue", _format_currency(
                annual_metrics_rpt.get("AS_Revenue"))),
            ("Thermal Energy Utilization", None),
            ("  HTE Thermal Efficiency", _format_number(annual_metrics_rpt.get(
                "thermal_efficiency", annual_metrics_rpt.get("Thermal_Efficiency")), 4)),
            ("  HTE Heat Opportunity Cost", _format_currency(
                annual_metrics_rpt.get("HTE_Heat_Opportunity_Cost_Annual_USD")))
        ]
        for name, value_str in perf_indicator_lines:
            if value_str is None:
                f.write(f"    {name}\n")
            else:
                f.write(format_aligned_line(name.strip(),
                        value_str, min_width=45, indent="    "))

    f.write("\n  B. Detailed Cost Composition (based on primary analysis case, e.g., Case 2 or Case 4 Baseline):\n")
    capex_breakdown = annual_metrics_rpt.get("capex_breakdown", {})
    total_capex_val = annual_metrics_rpt.get("total_capex")
    if total_capex_val is None and capex_breakdown:
        total_capex_val = sum(
            v for v in capex_breakdown.values() if isinstance(v, (int, float)))

    f.write("    Total CAPEX Composition Details:\n")
    if capex_breakdown and total_capex_val is not None and isinstance(total_capex_val, (int, float)) and total_capex_val > 0:
        sorted_capex_breakdown = sorted(capex_breakdown.items(
        ), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0, reverse=True)
        for comp, cost_val in sorted_capex_breakdown:
            display_comp_name = comp.replace("_System", "").replace("_", " ")
            if isinstance(cost_val, (int, float)) and cost_val > 0:
                f.write(
                    f"      {display_comp_name:<40}: {_format_currency(cost_val)} ({_format_percentage(cost_val / total_capex_val * 100)})\n")
        f.write(
            f"      {'Total CAPEX':<40}: {_format_currency(total_capex_val)}\n")
    else:
        f.write("      CAPEX breakdown data not available.\n")

    f.write(
        "\n    Total Annual OPEX Composition Details (Fixed, Variable, Fuel, etc.):\n")
    opex_items_detail = {
        "Nuclear Fixed O&M": annual_metrics_rpt.get("Nuclear_Fixed_OM_Annual_USD"),
        "Nuclear Variable O&M": annual_metrics_rpt.get("Nuclear_Variable_OM_Annual_USD"),
        "Nuclear Fuel Cost": annual_metrics_rpt.get("Nuclear_Fuel_Cost_Annual_USD"),
        "Nuclear Additional Costs": annual_metrics_rpt.get("Nuclear_Additional_Costs_Annual_USD"),
        "H2/Battery Fixed O&M (General)": annual_metrics_rpt.get("Fixed_OM_General", annual_metrics_rpt.get("annual_fixed_om_costs_total")),
        "H2/Battery VOM (Electrolyzer)": annual_metrics_rpt.get("VOM_Electrolyzer_Cost"),
        "H2/Battery VOM (Battery)": annual_metrics_rpt.get("VOM_Battery_Cost"),
        "Water Cost (for H2)": annual_metrics_rpt.get("Water_Cost"),
        "Startup Costs": annual_metrics_rpt.get("Startup_Cost"),
        "Ramping Costs": annual_metrics_rpt.get("Ramping_Cost"),
        "H2 Storage Cycle Cost": annual_metrics_rpt.get("H2_Storage_Cycle_Cost"),
    }
    total_annual_opex_calc = sum(v for v in opex_items_detail.values(
    ) if isinstance(v, (int, float)) and not np.isnan(v) and v is not None)

    has_opex_data = False
    for item, value in opex_items_detail.items():
        if value is not None and isinstance(value, (int, float)) and not np.isnan(value) and value != 0:
            f.write(f"      {item:<40}: {_format_currency(value)}\n")
            has_opex_data = True
    if has_opex_data:
        f.write(
            f"      {'Calculated Total Annual OPEX':<40}: {_format_currency(total_annual_opex_calc)}\n")
        reported_total_opex = annual_metrics_rpt.get(
            "Total_System_OPEX_Annual_USD", annual_metrics_rpt.get("Annual_Opex_Cost_from_Opt"))
        if reported_total_opex is not None:
            f.write(
                f"      {'Reported Total System OPEX':<40}: {_format_currency(reported_total_opex)}\n")
    else:
        f.write("      Annual OPEX breakdown data not available.\n")

    f.write("\n    LCOH Detailed Composition (if H2 produced):\n")
    lcoh_analysis = annual_metrics_rpt.get("lcoh_breakdown_analysis", {})
    if lcoh_analysis and "total_lcoh_usd_per_kg" in lcoh_analysis:
        total_lcoh_val = lcoh_analysis.get("total_lcoh_usd_per_kg")
        f.write(f"      Total LCOH: {_format_lcoh(total_lcoh_val)}\n")
        lcoh_breakdown_detail = lcoh_analysis.get(
            "lcoh_breakdown_usd_per_kg", {})
        lcoh_percentages_detail = lcoh_analysis.get("lcoh_percentages", {})
        if lcoh_breakdown_detail:
            sorted_lcoh_components = sorted(lcoh_breakdown_detail.items(
            ), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0, reverse=True)
            for comp, cost_val in sorted_lcoh_components:
                display_comp_name = comp.replace(
                    "CAPEX_", "").replace("_", " ").title()
                if isinstance(cost_val, (int, float)) and cost_val > 0.0001:
                    f.write(
                        f"        {display_comp_name:<40}: {_format_lcoh(cost_val)} ({_format_percentage(lcoh_percentages_detail.get(comp,0))})\n")
    else:
        f.write("      LCOH detailed composition data not available.\n")

    f.write("\n  C. MACRS Depreciation Summary:\n")
    macrs_enabled = annual_metrics_rpt.get("macrs_enabled", False)
    if macrs_enabled:
        total_depreciation_arr = annual_metrics_rpt.get(
            "macrs_total_depreciation", np.array([]))
        total_depreciation_sum = np.sum(total_depreciation_arr) if isinstance(
            total_depreciation_arr, (list, np.ndarray)) and len(total_depreciation_arr) > 0 else 0
        total_tax_shield = total_depreciation_sum * tax_rate_rpt

        macrs_summary = {
            "Total MACRS Depreciation Over Lifetime": _format_currency(total_depreciation_sum),
            "Total Tax Shield Value (at current tax rate)": _format_currency(total_tax_shield)
        }
        f.write(format_aligned_section(
            macrs_summary, min_width=45, indent="    "))

        component_macrs_dep = annual_metrics_rpt.get(
            "macrs_component_depreciation", {})
        if component_macrs_dep:
            f.write("\n    MACRS Depreciation by Asset Type/Component Category:\n")
            asset_depreciation = {}
            for comp_name, dep_array_val in component_macrs_dep.items():
                if isinstance(dep_array_val, (list, np.ndarray)):
                    comp_sum = np.sum(dep_array_val)
                    asset_depreciation[comp_name.replace(
                        "_", " ")] = _format_currency(comp_sum)
                else:
                    asset_depreciation[comp_name.replace(
                        "_", " ")] = "N/A (Data error)"
            f.write(format_aligned_section(
                asset_depreciation, min_width=45, indent="      "))
    else:
        f.write(
            "    MACRS Depreciation not enabled or data not available for this analysis.\n")
    f.write("\n")


def _write_core_assumptions_section(
    f, capex_data: dict, om_data: dict, replacement_data: dict,
    project_lifetime_years_rpt: int, construction_period_years_rpt: int,
    discount_rate_rpt: float, tax_rate_rpt: float, annual_metrics_rpt: dict
):
    f.write("8. Core Economic & Cost Assumptions\n")
    f.write("-" * 35 + "\n")

    f.write("  Global Parameters (used unless overridden by specific analysis case):\n")
    global_assumptions = {
        "Project Lifetime (General)": f"{project_lifetime_years_rpt} years",
        "Construction Period (General)": f"{construction_period_years_rpt} years",
        "Discount Rate": _format_percentage(discount_rate_rpt * 100),
        "Corporate Tax Rate": _format_percentage(tax_rate_rpt * 100)
    }
    f.write(format_aligned_section(
        global_assumptions, min_width=40, indent="    "))

    f.write(
        "\n  Main Equipment CAPEX Cost Parameters (from `config.CAPEX_COMPONENTS`):\n")
    if capex_data:
        capex_assumptions = {}
        for comp, details in capex_data.items():
            ref_cap_key = details.get('applies_to_component_capacity_key', '')
            unit_val = details.get('reference_total_capacity_mw', 'N/A')
            unit_suffix = ref_cap_key.split(
                '_')[-1] if ref_cap_key and '_' in ref_cap_key else 'units'

            cost_str = f"${details.get('total_base_cost_for_ref_size', 0):,.0f} for {_try_format_number(unit_val,0,default_na=True)} {unit_suffix}"
            if details.get('learning_rate_decimal', 0) > 0:
                cost_str += f", LR: {_format_percentage(details.get('learning_rate_decimal',0)*100)}"
            capex_assumptions[comp.replace("_", " ")] = cost_str
        f.write(format_aligned_section(
            capex_assumptions, min_width=40, indent="    "))
    else:
        f.write("    CAPEX component data not available.\n")

    f.write("\n  Main Operations & Maintenance Cost Parameters (from `config.OM_COMPONENTS`):\n")
    if om_data:
        om_assumptions = {}
        for comp, details in om_data.items():
            cost_info_parts = []
            if "base_cost_percent_of_capex" in details:
                cost_info_parts.append(
                    f"{_format_percentage(details['base_cost_percent_of_capex']*100)} of CAPEX")

            bc_mw_yr = details.get("base_cost_per_mw_year")
            if bc_mw_yr is not None and bc_mw_yr > 0:
                cost_info_parts.append(f"${bc_mw_yr:,.0f}/MW/year")

            bc_mwh_yr = details.get("base_cost_per_mwh_year")
            if bc_mwh_yr is not None and bc_mwh_yr > 0:
                cost_info_parts.append(f"${bc_mwh_yr:,.0f}/MWh/year")

            bc_mwh = details.get("base_cost_per_mwh")
            if bc_mwh is not None:
                cost_info_parts.append(f"${bc_mwh:,.2f}/MWh")

            bc_kg_cycled = details.get("base_cost_per_kg_cycled")
            if bc_kg_cycled is not None:
                cost_info_parts.append(f"${bc_kg_cycled:,.2f}/kg cycled")

            if not cost_info_parts and "base_cost" in details:
                cost_info_parts.append(f"${details['base_cost']:,.0f}/year")

            cost_str = " + ".join(cost_info_parts) if cost_info_parts else "N/A"

            if details.get("inflation_rate", 0) > 0:
                cost_str += f" (Inflation: {_format_percentage(details.get('inflation_rate',0)*100)})"
            om_assumptions[comp.replace("_", " ")] = cost_str
        f.write(format_aligned_section(
            om_assumptions, min_width=40, indent="    "))
    else:
        f.write("    O&M component data not available.\n")

    f.write(
        "\n  Main Replacement Cost Parameters (from `config.REPLACEMENT_SCHEDULE`):\n")
    if replacement_data:
        replacement_assumptions = {}
        for comp, details in replacement_data.items():
            cost_str = ""
            if "cost_percent_initial_capex" in details:
                cost_str = f"{_format_percentage(details['cost_percent_initial_capex']*100)} of initial CAPEX"
            elif "cost" in details:
                cost_str = _format_currency(details['cost'])

            years_str = ", ".join(map(str, details.get("years", [])))
            replacement_assumptions[comp.replace(
                "_", " ")] = f"{cost_str} in years: {years_str}"
        f.write(format_aligned_section(
            replacement_assumptions, min_width=40, indent="    "))
    else:
        f.write("    Replacement schedule data not available.\n")

    f.write("\n  Nuclear Cost Parameters (from `config.NUCLEAR_COST_PARAMETERS` or calculation outputs):\n")
    nuclear_cost_params_items = {}

    fixed_om_val_am = annual_metrics_rpt.get(
        'Nuclear_Fixed_OM_USD_per_MW_month')
    variable_om_val_am = annual_metrics_rpt.get(
        'Nuclear_Variable_OM_USD_per_MWh')
    fuel_cost_val_am = annual_metrics_rpt.get('Nuclear_Fuel_Cost_USD_per_MWh')
    additional_costs_val_am = annual_metrics_rpt.get(
        'Nuclear_Additional_Costs_USD_per_MW_year')
    nuclear_capex_val_am = annual_metrics_rpt.get('Nuclear_CAPEX_per_MW')

    populated_from_annual_metrics = any(v is not None for v in [
        fixed_om_val_am, variable_om_val_am, fuel_cost_val_am,
        additional_costs_val_am, nuclear_capex_val_am
    ])

    if populated_from_annual_metrics:
        logger.debug(
            "Using nuclear cost unit parameters from annual_metrics_rpt.")
        nuclear_cost_params_items[
            "Nuclear Fixed O&M"] = f"{_try_format_currency(fixed_om_val_am, default_na=True)}/MW/month"
        nuclear_cost_params_items[
            "Nuclear Variable O&M"] = f"{_try_format_number(variable_om_val_am, 2, default_na=True)}/MWh"
        nuclear_cost_params_items[
            "Nuclear Fuel Cost"] = f"{_try_format_number(fuel_cost_val_am, 2, default_na=True)}/MWh"
        nuclear_cost_params_items[
            "Nuclear Additional Costs"] = f"{_try_format_currency(additional_costs_val_am, default_na=True)}/MW/year"
        nuclear_cost_params_items[
            "Nuclear CAPEX (if greenfield)"] = f"{_try_format_currency(nuclear_capex_val_am, default_na=True)}/MW"
    else:
        logger.debug(
            "Nuclear cost unit parameters not in annual_metrics_rpt, attempting to load from tea.config.")
        try:
            from . import config as tea_config
            if hasattr(tea_config, 'NUCLEAR_COST_PARAMETERS'):
                nc_params = tea_config.NUCLEAR_COST_PARAMETERS
                opex_p = nc_params.get("opex_parameters", {})
                nuclear_cost_params_items[
                    "Nuclear Fixed O&M"] = f"{_try_format_currency(opex_p.get('fixed_om_per_mw_month'), default_na=True)}/MW/month"
                nuclear_cost_params_items[
                    "Nuclear Variable O&M"] = f"{_try_format_number(opex_p.get('variable_om_per_mwh'), 2, default_na=True)}/MWh"
                nuclear_cost_params_items[
                    "Nuclear Fuel Cost"] = f"{_try_format_number(opex_p.get('fuel_cost_per_mwh'), 2, default_na=True)}/MWh"
                nuclear_cost_params_items[
                    "Nuclear Additional Costs"] = f"{_try_format_currency(opex_p.get('additional_costs_per_mw_year'), default_na=True)}/MW/year"
                nuclear_cost_params_items[
                    "Nuclear CAPEX (if greenfield)"] = f"{_try_format_currency(nc_params.get('nuclear_capex_per_mw'), default_na=True)}/MW"
            else:
                nuclear_cost_params_items["Note"] = "NUCLEAR_COST_PARAMETERS not found in config."
                for key in ["Nuclear Fixed O&M", "Nuclear Variable O&M", "Nuclear Fuel Cost", "Nuclear Additional Costs", "Nuclear CAPEX (if greenfield)"]:
                    nuclear_cost_params_items[key] = "N/A"
        except ImportError:
            logger.warning(
                "tea.config could not be imported for NUCLEAR_COST_PARAMETERS.")
            nuclear_cost_params_items["Note"] = "tea.config not found or NUCLEAR_COST_PARAMETERS missing."
            for key in ["Nuclear Fixed O&M", "Nuclear Variable O&M", "Nuclear Fuel Cost", "Nuclear Additional Costs", "Nuclear CAPEX (if greenfield)"]:
                nuclear_cost_params_items[key] = "N/A"
    f.write(format_aligned_section(
        nuclear_cost_params_items, min_width=40, indent="    "))

    f.write(
        "\n  MACRS Depreciation Year Classifications (from `config.MACRS_CONFIG`):\n")
    macrs_classifications = {}
    try:
        from . import config as tea_config
        if hasattr(tea_config, 'MACRS_CONFIG'):
            mc_config = tea_config.MACRS_CONFIG
            comp_class = mc_config.get("component_classification", {})
            class_to_years = {
                "nuclear": mc_config.get("nuclear_depreciation_years", 15),
                "hydrogen": mc_config.get("hydrogen_depreciation_years", 7),
                "battery": mc_config.get("battery_depreciation_years", 7),
                "grid": mc_config.get("grid_depreciation_years", 15),
            }
            years_to_comps = {val: [] for val in class_to_years.values()}
            for comp, classification_type in comp_class.items():
                years = class_to_years.get(classification_type)
                if years:
                    if years not in years_to_comps:
                        years_to_comps[years] = []
                    years_to_comps[years].append(comp.replace("_", " "))

            for years_val, comps_list in sorted(years_to_comps.items()):
                if comps_list:
                    macrs_classifications[f"{years_val}-year MACRS"] = ", ".join(
                        comps_list)
        else:
            macrs_classifications["Note"] = "MACRS_CONFIG not found in config."
    except ImportError:
        macrs_classifications["Note"] = "tea.config not found or MACRS_CONFIG missing."

    if not macrs_classifications or "Note" in macrs_classifications:
        macrs_classifications = {
            "Nuclear Equipment": "15-year",
            "Hydrogen Equipment": "7-year",
            "Battery Systems": "7-year",
            "Grid Infrastructure": "15-year"
        }
        if "Note" not in macrs_classifications:
            f.write("    (Using default classifications if config not available)\n")

    f.write(format_aligned_section(
        macrs_classifications, min_width=40, indent="    "))
    f.write("\n")


def generate_comprehensive_case_comparison_charts(
    annual_metrics_rpt: dict,
    financial_metrics_rpt: dict,
    incremental_metrics_rpt: Optional[dict] = None,
    output_dir: Path = None,
    target_iso_rpt: str = "Unknown",
    plant_specific_title_rpt: str = "Nuclear Plant"
) -> bool:
    """
    Generate comprehensive comparison charts for the 5 TEA cases across multiple dimensions.

    Args:
        annual_metrics_rpt: Annual metrics data containing all case results
        financial_metrics_rpt: Financial metrics data
        incremental_metrics_rpt: Incremental system metrics (Case 3)
        output_dir: Directory to save charts
        target_iso_rpt: ISO region name
        plant_specific_title_rpt: Plant name for titles

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Generating comprehensive case comparison charts...")

    if output_dir is None:
        output_dir = Path("./comparison_charts")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Extract data for all 5 cases
        case_data = _extract_case_data(
            annual_metrics_rpt, financial_metrics_rpt, incremental_metrics_rpt)

        # Generate individual chart types
        _create_financial_metrics_comparison(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_cost_breakdown_comparison(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_lcoh_lcoe_comparison(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_capacity_utilization_comparison(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_tax_incentive_impact_comparison(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_lifecycle_comparison_chart(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)
        _create_comprehensive_dashboard(
            case_data, output_dir, target_iso_rpt, plant_specific_title_rpt)

        logger.info(
            f"Comprehensive case comparison charts saved to {output_dir}")
        return True

    except Exception as e:
        logger.error(
            f"Error generating comprehensive case comparison charts: {str(e)}", exc_info=True)
        return False


def _extract_case_data(annual_metrics_rpt: dict, financial_metrics_rpt: dict, incremental_metrics_rpt: dict) -> dict:
    """Extract and organize data for all 5 cases."""
    case_data = {
        'case1': {},  # Existing Reactor Baseline
        'case2': {},  # Existing Reactor Retrofit
        'case3': {},  # Incremental System
        'case4': {},  # Greenfield 60-year
        'case5': {}   # Greenfield 80-year
    }

    # Case 1: Existing Reactor Baseline
    nuclear_baseline = annual_metrics_rpt.get("nuclear_baseline_analysis", {})
    if nuclear_baseline:
        case_data['case1'] = {
            'name': 'Case 1: Existing Reactor Baseline',
            'npv': nuclear_baseline.get("scenario_without_45u", {}).get("npv_usd", 0),
            'npv_with_45u': nuclear_baseline.get("scenario_with_45u", {}).get("npv_usd", 0),
            'irr': nuclear_baseline.get("scenario_without_45u", {}).get("irr_percent", 0),
            'payback': nuclear_baseline.get("scenario_without_45u", {}).get("payback_period_years", 0),
            'annual_revenue': nuclear_baseline.get("annual_performance", {}).get("annual_revenue_usd", 0),
            'annual_opex': nuclear_baseline.get("annual_performance", {}).get("annual_total_opex_usd", 0),
            'annual_generation': nuclear_baseline.get("annual_performance", {}).get("annual_generation_mwh", 0),
            'capacity_factor': nuclear_baseline.get("plant_parameters", {}).get("nameplate_power_factor", 0),
            'remaining_lifetime': nuclear_baseline.get("plant_parameters", {}).get("remaining_plant_life_years", 25),
            'lcoe': None,  # Will be calculated
            'lcoh': 0,  # No hydrogen production
            'h2_production': 0,
            'capex': 0,  # Existing plant
            'type': 'existing'
        }

    # Case 2: Existing Reactor Retrofit (Integrated System)
    case_data['case2'] = {
        'name': 'Case 2: Existing Reactor Retrofit',
        'npv': financial_metrics_rpt.get("NPV_USD", 0),
        'irr': financial_metrics_rpt.get("IRR_percent", 0),
        'payback': financial_metrics_rpt.get("Payback_Period_Years", 0),
        'annual_revenue': annual_metrics_rpt.get("Annual_Revenue", 0),
        'annual_opex': annual_metrics_rpt.get("Total_System_OPEX_Annual_USD", 0),
        'annual_generation': annual_metrics_rpt.get("Annual_Nuclear_Generation_MWh", 0),
        'capacity_factor': annual_metrics_rpt.get("Turbine_CF_percent", 0),
        'electrolyzer_cf': annual_metrics_rpt.get("Electrolyzer_CF_percent", 0),
        'battery_cf': annual_metrics_rpt.get("Battery_CF_percent", 0),
        'lcoh': annual_metrics_rpt.get("lcoh_breakdown_analysis", {}).get("total_lcoh_usd_per_kg", 0),
        'h2_production': annual_metrics_rpt.get("H2_Production_kg_annual", 0),
        'capex': annual_metrics_rpt.get("total_capex", 0),
        'electrolyzer_capacity': annual_metrics_rpt.get("Electrolyzer_Capacity_MW", 0),
        'battery_capacity': annual_metrics_rpt.get("Battery_Capacity_MWh", 0),
        'h2_storage_capacity': annual_metrics_rpt.get("H2_Storage_Capacity_kg", 0),
        'type': 'retrofit'
    }

    # Case 3: Incremental System
    if incremental_metrics_rpt:
        case_data['case3'] = {
            'name': 'Case 3: Incremental System',
            'npv': incremental_metrics_rpt.get("NPV_USD", 0),
            'irr': incremental_metrics_rpt.get("IRR_percent", 0),
            'payback': incremental_metrics_rpt.get("Payback_Period_Years", 0),
            'annual_revenue': incremental_metrics_rpt.get("Annual_Incremental_Revenue_USD", 0),
            'annual_opex': incremental_metrics_rpt.get("Annual_Incremental_Costs_USD", 0),
            'capex': incremental_metrics_rpt.get("Total_Incremental_CAPEX_Learned_USD", 0),
            'opportunity_cost_elec': incremental_metrics_rpt.get("Annual_Electricity_Opportunity_Cost_USD", 0),
            'opportunity_cost_thermal': incremental_metrics_rpt.get("Annual_HTE_Thermal_Opportunity_Cost_USD", 0),
            'type': 'incremental'
        }

    # Cases 4 & 5: Greenfield projects
    lifecycle_comparison = annual_metrics_rpt.get(
        "lifecycle_comparison_analysis", {})
    if lifecycle_comparison:
        # Case 4: 60-year
        results_60yr = lifecycle_comparison.get("60_year_results", {})
        if results_60yr:
            baseline_60 = results_60yr.get("baseline_scenario", {})
            case_data['case4'] = {
                'name': 'Case 4: Greenfield 60-Year',
                'npv': baseline_60.get("npv_usd", 0),
                'irr': baseline_60.get("irr_percent", 0),
                'payback': baseline_60.get("payback_period_years", 0),
                'lcoh': results_60yr.get("tax_incentive_analysis", {}).get("scenarios", {}).get("baseline", {}).get("financial_metrics", {}).get("lcoh_usd_per_kg", 0),
                'lcoe': results_60yr.get("baseline_greenfield_results", {}).get("nuclear_lcoe_usd_per_mwh", 0),
                'lcos': results_60yr.get("baseline_greenfield_results", {}).get("battery_lcos_usd_per_mwh", 0),
                'lifetime': 60,
                'type': 'greenfield'
            }

        # Case 5: 80-year
        results_80yr = lifecycle_comparison.get("80_year_results", {})
        if results_80yr:
            baseline_80 = results_80yr.get("baseline_scenario", {})
            case_data['case5'] = {
                'name': 'Case 5: Greenfield 80-Year',
                'npv': baseline_80.get("npv_usd", 0),
                'irr': baseline_80.get("irr_percent", 0),
                'payback': baseline_80.get("payback_period_years", 0),
                'lcoh': results_80yr.get("tax_incentive_analysis", {}).get("scenarios", {}).get("baseline", {}).get("financial_metrics", {}).get("lcoh_usd_per_kg", 0),
                'lcoe': results_80yr.get("baseline_greenfield_results", {}).get("nuclear_lcoe_usd_per_mwh", 0),
                'lcos': results_80yr.get("baseline_greenfield_results", {}).get("battery_lcos_usd_per_mwh", 0),
                'lifetime': 80,
                'type': 'greenfield'
            }

    # Calculate Case 1 LCOE if possible
    if case_data['case1'] and case_data['case1'].get('annual_generation', 0) > 0:
        annual_perf = nuclear_baseline.get("annual_performance", {})
        plant_params = nuclear_baseline.get("plant_parameters", {})
        financial_metrics = nuclear_baseline.get("financial_metrics", {})
        case_data['case1']['lcoe'] = _calculate_case1_lcoe(
            annual_perf, plant_params, financial_metrics)

    return case_data


def _create_financial_metrics_comparison(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create financial metrics comparison chart (NPV, IRR, Payback)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    cases = []
    npvs = []
    irrs = []
    paybacks = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (case_key, data) in enumerate(case_data.items()):
        if data:  # Only include cases with data
            cases.append(data.get('name', case_key))
            npvs.append(data.get('npv', 0) / 1e6)  # Convert to millions
            irrs.append(data.get('irr', 0))
            paybacks.append(data.get('payback', 0)
                            if data.get('payback', 0) > 0 else None)

    # NPV Comparison
    bars1 = ax1.bar(cases, npvs, color=colors[:len(cases)], alpha=0.8)
    ax1.set_ylabel('NPV (Million USD)')
    ax1.set_title('Net Present Value Comparison', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars1, npvs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (max(npvs) - min(npvs)) * 0.01,
                 f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')

    # IRR Comparison
    valid_irrs = [irr for irr in irrs if irr > 0]
    valid_cases_irr = [case for case, irr in zip(cases, irrs) if irr > 0]
    valid_colors_irr = [colors[i] for i, irr in enumerate(irrs) if irr > 0]

    if valid_irrs:
        bars2 = ax2.bar(valid_cases_irr, valid_irrs,
                        color=valid_colors_irr, alpha=0.8)
        ax2.set_ylabel('IRR (%)')
        ax2.set_title('Internal Rate of Return Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars2, valid_irrs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(valid_irrs) * 0.01,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Positive IRR Data Available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, fontweight='bold')
        ax2.set_title('Internal Rate of Return Comparison', fontweight='bold')

    # Payback Period Comparison
    valid_paybacks = [pb for pb in paybacks if pb is not None and pb > 0]
    valid_cases_pb = [case for case, pb in zip(
        cases, paybacks) if pb is not None and pb > 0]
    valid_colors_pb = [colors[i] for i, pb in enumerate(
        paybacks) if pb is not None and pb > 0]

    if valid_paybacks:
        bars3 = ax3.bar(valid_cases_pb, valid_paybacks,
                        color=valid_colors_pb, alpha=0.8)
        ax3.set_ylabel('Payback Period (Years)')
        ax3.set_title('Payback Period Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars3, valid_paybacks):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(valid_paybacks) * 0.01,
                     f'{value:.1f}y', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Valid Payback Data Available', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold')
        ax3.set_title('Payback Period Comparison', fontweight='bold')

    # Financial Performance Summary (Radar Chart)
    if len(valid_irrs) >= 2:
        # Normalize metrics for radar chart
        normalized_npv = [(npv - min(npvs)) / (max(npvs) - min(npvs))
                          * 100 if max(npvs) != min(npvs) else 50 for npv in npvs]
        normalized_irr = [irr / max(valid_irrs) *
                          100 if irr > 0 else 0 for irr in irrs]

        # Create simple performance score
        performance_scores = []
        for i, case in enumerate(cases):
            score = 0
            if npvs[i] > 0:
                score += 40
            if irrs[i] > 8:  # Above discount rate
                score += 40
            if paybacks[i] and paybacks[i] < 10:
                score += 20
            performance_scores.append(score)

        bars4 = ax4.bar(cases, performance_scores,
                        color=colors[:len(cases)], alpha=0.8)
        ax4.set_ylabel('Performance Score (0-100)')
        ax4.set_title('Overall Financial Performance Score', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 100)

        for bar, value in zip(bars4, performance_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{value}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data for\nPerformance Scoring', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.set_title('Overall Financial Performance Score', fontweight='bold')

    plt.suptitle(f'Financial Metrics Comparison - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'financial_metrics_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_cost_breakdown_comparison(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create cost breakdown comparison chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract cost data
    cases = []
    capex_values = []
    opex_values = []
    revenue_values = []

    for case_key, data in case_data.items():
        if data and data.get('type') != 'existing':  # Skip Case 1 for CAPEX
            cases.append(data.get('name', case_key))
            capex_values.append(data.get('capex', 0) /
                                1e6)  # Convert to millions
            opex_values.append(data.get('annual_opex', 0) / 1e6)
            revenue_values.append(data.get('annual_revenue', 0) / 1e6)

    colors = ['#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd']  # Skip blue for Case 1

    # CAPEX Comparison
    if capex_values:
        bars1 = ax1.bar(cases, capex_values,
                        color=colors[:len(cases)], alpha=0.8)
        ax1.set_ylabel('CAPEX (Million USD)')
        ax1.set_title('Capital Expenditure Comparison', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars1, capex_values):
            if value > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(capex_values) * 0.01,
                         f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

    # Annual OPEX Comparison
    if opex_values:
        bars2 = ax2.bar(cases, opex_values,
                        color=colors[:len(cases)], alpha=0.8)
        ax2.set_ylabel('Annual OPEX (Million USD)')
        ax2.set_title('Annual Operating Expenditure Comparison',
                      fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars2, opex_values):
            if value > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(opex_values) * 0.01,
                         f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

    # Annual Revenue Comparison
    if revenue_values:
        bars3 = ax3.bar(cases, revenue_values,
                        color=colors[:len(cases)], alpha=0.8)
        ax3.set_ylabel('Annual Revenue (Million USD)')
        ax3.set_title('Annual Revenue Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars3, revenue_values):
            if value > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(revenue_values) * 0.01,
                         f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

    # Cost-Revenue Ratio
    if revenue_values and opex_values:
        ratios = [opex/rev if rev > 0 else 0 for opex,
                  rev in zip(opex_values, revenue_values)]
        bars4 = ax4.bar(cases, ratios, color=colors[:len(cases)], alpha=0.8)
        ax4.set_ylabel('OPEX/Revenue Ratio')
        ax4.set_title('Operating Cost Efficiency', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=1.0, color='red', linestyle='--',
                    alpha=0.7, label='Break-even')
        ax4.legend()

        for bar, value in zip(bars4, ratios):
            if value > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(ratios) * 0.01,
                         f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Cost Breakdown Comparison - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cost_breakdown_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_lcoh_lcoe_comparison(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create LCOH and LCOE comparison chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract LCOH data
    cases_h2 = []
    lcoh_values = []
    h2_production = []

    # Extract LCOE data
    cases_lcoe = []
    lcoe_values = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (case_key, data) in enumerate(case_data.items()):
        if data:
            case_name = data.get('name', case_key)

            # LCOH data (Cases 2, 4, 5)
            if data.get('lcoh', 0) > 0:
                cases_h2.append(case_name)
                lcoh_values.append(data.get('lcoh', 0))
                # Convert to million kg
                h2_production.append(data.get('h2_production', 0) / 1e6)

            # LCOE data (Cases 1, 4, 5)
            if data.get('lcoe') is not None and data.get('lcoe') > 0:
                cases_lcoe.append(case_name)
                lcoe_values.append(data.get('lcoe', 0))

    # LCOH Comparison
    if lcoh_values:
        colors_h2 = [colors[i] for i, (_, data) in enumerate(case_data.items())
                     if data and data.get('lcoh', 0) > 0]
        bars1 = ax1.bar(cases_h2, lcoh_values, color=colors_h2, alpha=0.8)
        ax1.set_ylabel('LCOH (USD/kg)')
        ax1.set_title('Levelized Cost of Hydrogen Comparison',
                      fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)

        # Add DOE target line
        ax1.axhline(y=2.0, color='red', linestyle='--',
                    alpha=0.7, label='DOE 2030 Target ($2/kg)')
        ax1.legend()

        for bar, value in zip(bars1, lcoh_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(lcoh_values) * 0.01,
                     f'${value:.2f}/kg', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No LCOH Data Available', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold')
        ax1.set_title('Levelized Cost of Hydrogen Comparison',
                      fontweight='bold')

    # LCOE Comparison
    if lcoe_values:
        colors_lcoe = [colors[i] for i, (_, data) in enumerate(case_data.items())
                       if data and data.get('lcoe') is not None and data.get('lcoe') > 0]
        bars2 = ax2.bar(cases_lcoe, lcoe_values, color=colors_lcoe, alpha=0.8)
        ax2.set_ylabel('LCOE (USD/MWh)')
        ax2.set_title('Levelized Cost of Electricity Comparison',
                      fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars2, lcoe_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(lcoe_values) * 0.01,
                     f'${value:.1f}/MWh', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No LCOE Data Available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, fontweight='bold')
        ax2.set_title('Levelized Cost of Electricity Comparison',
                      fontweight='bold')

    # H2 Production Volume
    if h2_production:
        bars3 = ax3.bar(cases_h2, h2_production, color=colors_h2, alpha=0.8)
        ax3.set_ylabel('Annual H2 Production (Million kg)')
        ax3.set_title('Annual Hydrogen Production Comparison',
                      fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars3, h2_production):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(h2_production) * 0.01,
                     f'{value:.1f}M kg', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No H2 Production Data Available', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold')
        ax3.set_title('Annual Hydrogen Production Comparison',
                      fontweight='bold')

    # Cost Efficiency (LCOH vs Production)
    if lcoh_values and h2_production:
        efficiency = [prod/lcoh if lcoh > 0 else 0 for prod,
                      lcoh in zip(h2_production, lcoh_values)]
        bars4 = ax4.bar(cases_h2, efficiency, color=colors_h2, alpha=0.8)
        ax4.set_ylabel('Production Efficiency (Million kg per $/kg)')
        ax4.set_title('Hydrogen Production Cost Efficiency', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars4, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(efficiency) * 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data for\nEfficiency Analysis', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.set_title('Hydrogen Production Cost Efficiency', fontweight='bold')

    plt.suptitle(f'LCOH & LCOE Comparison - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'lcoh_lcoe_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_capacity_utilization_comparison(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create capacity utilization comparison chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract capacity factor data
    cases = []
    nuclear_cf = []
    electrolyzer_cf = []
    battery_cf = []
    capacities = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (case_key, data) in enumerate(case_data.items()):
        if data:
            case_name = data.get('name', case_key)
            cases.append(case_name)
            nuclear_cf.append(data.get('capacity_factor', 0)
                              * 100)  # Convert to percentage
            electrolyzer_cf.append(data.get('electrolyzer_cf', 0))
            battery_cf.append(data.get('battery_cf', 0))

            # System capacities
            elec_cap = data.get('electrolyzer_capacity', 0)
            batt_cap = data.get('battery_capacity', 0)
            capacities.append({'electrolyzer': elec_cap, 'battery': batt_cap})

    # Nuclear Capacity Factor
    bars1 = ax1.bar(cases, nuclear_cf, color=colors[:len(cases)], alpha=0.8)
    ax1.set_ylabel('Nuclear Capacity Factor (%)')
    ax1.set_title('Nuclear Plant Capacity Factor Comparison',
                  fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 100)

    for bar, value in zip(bars1, nuclear_cf):
        if value > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Electrolyzer Capacity Factor
    valid_elec_cf = [cf for cf in electrolyzer_cf if cf > 0]
    valid_cases_elec = [case for case, cf in zip(
        cases, electrolyzer_cf) if cf > 0]
    valid_colors_elec = [colors[i]
                         for i, cf in enumerate(electrolyzer_cf) if cf > 0]

    if valid_elec_cf:
        bars2 = ax2.bar(valid_cases_elec, valid_elec_cf,
                        color=valid_colors_elec, alpha=0.8)
        ax2.set_ylabel('Electrolyzer Capacity Factor (%)')
        ax2.set_title('Electrolyzer Utilization Comparison', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)

        for bar, value in zip(bars2, valid_elec_cf):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Electrolyzer Data Available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, fontweight='bold')
        ax2.set_title('Electrolyzer Utilization Comparison', fontweight='bold')

    # System Capacity Overview
    elec_capacities = [cap['electrolyzer']
                       for cap in capacities if cap['electrolyzer'] > 0]
    elec_cases = [case for case, cap in zip(
        cases, capacities) if cap['electrolyzer'] > 0]
    elec_colors = [colors[i]
                   for i, cap in enumerate(capacities) if cap['electrolyzer'] > 0]

    if elec_capacities:
        bars3 = ax3.bar(elec_cases, elec_capacities,
                        color=elec_colors, alpha=0.8)
        ax3.set_ylabel('Electrolyzer Capacity (MW)')
        ax3.set_title('Electrolyzer Capacity Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars3, elec_capacities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(elec_capacities) * 0.01,
                     f'{value:.0f} MW', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Electrolyzer Capacity Data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12, fontweight='bold')
        ax3.set_title('Electrolyzer Capacity Comparison', fontweight='bold')

    # Utilization Efficiency (Production per MW)
    if valid_elec_cf and elec_capacities:
        # Calculate annual production per MW of capacity
        production_per_mw = []
        prod_cases = []
        for case_key, data in case_data.items():
            if data and data.get('h2_production', 0) > 0 and data.get('electrolyzer_capacity', 0) > 0:
                prod_per_mw = data.get(
                    'h2_production', 0) / data.get('electrolyzer_capacity', 1) / 1000  # kg per kW
                production_per_mw.append(prod_per_mw)
                prod_cases.append(data.get('name', case_key))

        if production_per_mw:
            prod_colors = [colors[i] for i, (_, data) in enumerate(case_data.items())
                           if data and data.get('h2_production', 0) > 0 and data.get('electrolyzer_capacity', 0) > 0]
            bars4 = ax4.bar(prod_cases, production_per_mw,
                            color=prod_colors, alpha=0.8)
            ax4.set_ylabel('H2 Production per kW (kg/kW/year)')
            ax4.set_title('Electrolyzer Productivity Comparison',
                          fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)

            for bar, value in zip(bars4, production_per_mw):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(production_per_mw) * 0.01,
                         f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data for\nProductivity Analysis', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=12, fontweight='bold')
            ax4.set_title('Electrolyzer Productivity Comparison',
                          fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data for\nProductivity Analysis', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.set_title('Electrolyzer Productivity Comparison',
                      fontweight='bold')

    plt.suptitle(f'Capacity Utilization Comparison - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'capacity_utilization_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_tax_incentive_impact_comparison(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create tax incentive impact comparison chart."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # For this chart, we'll focus on Case 1 (45U impact) and Cases 4/5 (tax incentive scenarios)
    # Case 1: 45U Impact
    case1_data = case_data.get('case1', {})
    if case1_data:
        npv_without_45u = case1_data.get('npv', 0) / 1e6
        npv_with_45u = case1_data.get('npv_with_45u', 0) / 1e6

        categories = ['Without 45U', 'With 45U']
        values = [npv_without_45u, npv_with_45u]
        colors = ['#ff7f0e', '#2ca02c']

        bars1 = ax1.bar(categories, values, color=colors, alpha=0.8)
        ax1.set_ylabel('NPV (Million USD)')
        ax1.set_title(
            'Case 1: 45U PTC Impact on Existing Nuclear', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for bar, value in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max(values) - min(values)) * 0.02,
                     f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

        # Add improvement annotation
        improvement = npv_with_45u - npv_without_45u
        ax1.text(0.5, 0.95, f'NPV Improvement: ${improvement:.0f}M',
                 transform=ax1.transAxes, ha='center', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    else:
        ax1.text(0.5, 0.5, 'No Case 1 Data Available', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold')
        ax1.set_title(
            'Case 1: 45U PTC Impact on Existing Nuclear', fontweight='bold')

    # Cases 4/5: Tax Incentive Scenarios (placeholder - would need detailed tax scenario data)
    ax2.text(0.5, 0.5, 'Tax Incentive Scenarios\n(Cases 4 & 5)\nDetailed data needed',
             ha='center', va='center', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax2.set_title('Cases 4 & 5: Federal Tax Incentive Impact',
                  fontweight='bold')

    # Project Lifetime Comparison
    lifetimes = []
    lifetime_cases = []
    lifetime_npvs = []

    for case_key, data in case_data.items():
        if data and data.get('type') in ['existing', 'greenfield']:
            lifetime_cases.append(data.get('name', case_key))
            if case_key == 'case1':
                lifetimes.append(data.get('remaining_lifetime', 25))
            else:
                lifetimes.append(data.get('lifetime', 60))
            lifetime_npvs.append(data.get('npv', 0) / 1e6)

    if lifetimes:
        colors_lifetime = ['#1f77b4', '#d62728', '#9467bd'][:len(lifetimes)]
        bars3 = ax3.bar(lifetime_cases, lifetimes,
                        color=colors_lifetime, alpha=0.8)
        ax3.set_ylabel('Project Lifetime (Years)')
        ax3.set_title('Project Lifetime Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars3, lifetimes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(lifetimes) * 0.01,
                     f'{value}y', ha='center', va='bottom', fontweight='bold')

    # NPV per Year of Operation
    if lifetimes and lifetime_npvs:
        npv_per_year = [npv/lifetime if lifetime > 0 else 0 for npv,
                        lifetime in zip(lifetime_npvs, lifetimes)]
        bars4 = ax4.bar(lifetime_cases, npv_per_year,
                        color=colors_lifetime, alpha=0.8)
        ax4.set_ylabel('NPV per Year (Million USD/year)')
        ax4.set_title('Annual NPV Efficiency', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for bar, value in zip(bars4, npv_per_year):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (max(npv_per_year) - min(npv_per_year)) * 0.02,
                     f'${value:.1f}M/y', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'Tax Incentive & Lifetime Impact - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'tax_incentive_impact_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_lifecycle_comparison_chart(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create lifecycle comparison chart focusing on Cases 4 and 5."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract Cases 4 and 5 data
    case4_data = case_data.get('case4', {})
    case5_data = case_data.get('case5', {})

    if case4_data and case5_data:
        # Lifecycle comparison
        lifecycles = ['60-Year (Case 4)', '80-Year (Case 5)']
        npvs = [case4_data.get('npv', 0) / 1e6, case5_data.get('npv', 0) / 1e6]
        lcoh_values = [case4_data.get('lcoh', 0), case5_data.get('lcoh', 0)]
        lcoe_values = [case4_data.get('lcoe', 0), case5_data.get('lcoe', 0)]

        colors = ['#2ca02c', '#d62728']

        # NPV Comparison
        bars1 = ax1.bar(lifecycles, npvs, color=colors, alpha=0.8)
        ax1.set_ylabel('NPV (Million USD)')
        ax1.set_title('NPV: 60-Year vs 80-Year Lifecycle', fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        for bar, value in zip(bars1, npvs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (max(npvs) - min(npvs)) * 0.02,
                     f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

        # LCOH Comparison
        if all(lcoh > 0 for lcoh in lcoh_values):
            bars2 = ax2.bar(lifecycles, lcoh_values, color=colors, alpha=0.8)
            ax2.set_ylabel('LCOH (USD/kg)')
            ax2.set_title('LCOH: 60-Year vs 80-Year Lifecycle',
                          fontweight='bold')
            ax2.axhline(y=2.0, color='red', linestyle='--',
                        alpha=0.7, label='DOE Target')
            ax2.legend()

            for bar, value in zip(bars2, lcoh_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(lcoh_values) * 0.01,
                         f'${value:.2f}/kg', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'LCOH Data Not Available', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12, fontweight='bold')
            ax2.set_title('LCOH: 60-Year vs 80-Year Lifecycle',
                          fontweight='bold')

        # LCOE Comparison
        if all(lcoe > 0 for lcoe in lcoe_values):
            bars3 = ax3.bar(lifecycles, lcoe_values, color=colors, alpha=0.8)
            ax3.set_ylabel('LCOE (USD/MWh)')
            ax3.set_title('LCOE: 60-Year vs 80-Year Lifecycle',
                          fontweight='bold')

            for bar, value in zip(bars3, lcoe_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(lcoe_values) * 0.01,
                         f'${value:.1f}/MWh', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'LCOE Data Not Available', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12, fontweight='bold')
            ax3.set_title('LCOE: 60-Year vs 80-Year Lifecycle',
                          fontweight='bold')

        # Lifecycle Benefits Analysis
        npv_improvement = npvs[1] - npvs[0]  # 80-year vs 60-year
        lcoh_improvement = lcoh_values[0] - lcoh_values[1] if all(
            lcoh > 0 for lcoh in lcoh_values) else 0

        benefits = ['NPV Improvement\n(80y vs 60y)',
                    'LCOH Reduction\n(60y vs 80y)']
        # Scale LCOH for visibility
        benefit_values = [npv_improvement, lcoh_improvement * 1000]
        benefit_colors = ['green' if val >
                          0 else 'red' for val in benefit_values]

        bars4 = ax4.bar(benefits, benefit_values,
                        color=benefit_colors, alpha=0.8)
        ax4.set_ylabel('Improvement Value')
        ax4.set_title('Lifecycle Extension Benefits', fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Add value labels
        ax4.text(0, benefit_values[0] + abs(benefit_values[0]) * 0.05,
                 f'${npv_improvement:.0f}M', ha='center', va='bottom', fontweight='bold')
        if lcoh_improvement != 0:
            ax4.text(1, benefit_values[1] + abs(benefit_values[1]) * 0.05,
                     f'${lcoh_improvement:.3f}/kg\n(×1000 scale)', ha='center', va='bottom', fontweight='bold')

    else:
        # No lifecycle data available
        for ax, title in zip([ax1, ax2, ax3, ax4],
                             ['NPV Comparison', 'LCOH Comparison', 'LCOE Comparison', 'Benefits Analysis']):
            ax.text(0.5, 0.5, 'Lifecycle Comparison Data\nNot Available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.set_title(title, fontweight='bold')

    plt.suptitle(f'Lifecycle Comparison (60y vs 80y) - {plant_name} ({target_iso})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'lifecycle_comparison_chart.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_comprehensive_dashboard(case_data: dict, output_dir: Path, target_iso: str, plant_name: str):
    """Create comprehensive dashboard summarizing all cases."""
    fig = plt.figure(figsize=(20, 16))

    # Create a 3x3 grid for the dashboard
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract summary data
    cases = []
    npvs = []
    lcoh_values = []
    lcoe_values = []
    capacities = []
    lifetimes = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (case_key, data) in enumerate(case_data.items()):
        if data:
            cases.append(data.get('name', case_key).replace(
                'Case ', '').replace(': ', ':\n'))
            npvs.append(data.get('npv', 0) / 1e6)
            lcoh_values.append(data.get('lcoh', 0))
            lcoe_values.append(data.get('lcoe', 0) if data.get('lcoe') else 0)
            capacities.append(data.get('electrolyzer_capacity', 0))

            if case_key == 'case1':
                lifetimes.append(data.get('remaining_lifetime', 25))
            else:
                lifetimes.append(data.get('lifetime', 60))

    # 1. NPV Overview (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(len(cases)), npvs,
                    color=colors[:len(cases)], alpha=0.8)
    ax1.set_ylabel('NPV (Million USD)')
    ax1.set_title('NPV Comparison Across All Cases',
                  fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(cases)))
    ax1.set_xticklabels(cases, rotation=45, ha='right', fontsize=9)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # 2. LCOH Comparison (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    valid_lcoh = [lcoh for lcoh in lcoh_values if lcoh > 0]
    valid_cases_lcoh = [case for case, lcoh in zip(
        cases, lcoh_values) if lcoh > 0]
    valid_colors_lcoh = [colors[i]
                         for i, lcoh in enumerate(lcoh_values) if lcoh > 0]

    if valid_lcoh:
        bars2 = ax2.bar(range(len(valid_cases_lcoh)), valid_lcoh,
                        color=valid_colors_lcoh, alpha=0.8)
        ax2.set_ylabel('LCOH (USD/kg)')
        ax2.set_title('LCOH Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(range(len(valid_cases_lcoh)))
        ax2.set_xticklabels(valid_cases_lcoh, rotation=45,
                            ha='right', fontsize=9)
        ax2.axhline(y=2.0, color='red', linestyle='--',
                    alpha=0.7, label='DOE Target')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No LCOH Data', ha='center',
                 va='center', transform=ax2.transAxes)
        ax2.set_title('LCOH Comparison', fontweight='bold', fontsize=12)

    # 3. System Capacity (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    valid_cap = [cap for cap in capacities if cap > 0]
    valid_cases_cap = [case for case, cap in zip(cases, capacities) if cap > 0]
    valid_colors_cap = [colors[i]
                        for i, cap in enumerate(capacities) if cap > 0]

    if valid_cap:
        bars3 = ax3.bar(range(len(valid_cases_cap)), valid_cap,
                        color=valid_colors_cap, alpha=0.8)
        ax3.set_ylabel('Electrolyzer Capacity (MW)')
        ax3.set_title('System Capacity', fontweight='bold', fontsize=12)
        ax3.set_xticks(range(len(valid_cases_cap)))
        ax3.set_xticklabels(valid_cases_cap, rotation=45,
                            ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Capacity Data', ha='center',
                 va='center', transform=ax3.transAxes)
        ax3.set_title('System Capacity', fontweight='bold', fontsize=12)

    # 4. Project Lifetime (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(range(len(cases)), lifetimes,
                    color=colors[:len(cases)], alpha=0.8)
    ax4.set_ylabel('Project Lifetime (Years)')
    ax4.set_title('Project Lifetime Comparison',
                  fontweight='bold', fontsize=12)
    ax4.set_xticks(range(len(cases)))
    ax4.set_xticklabels(cases, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Case Type Distribution (middle-center) - Pie chart
    ax5 = fig.add_subplot(gs[1, 1])
    case_types = []
    type_counts = {}

    for data in case_data.values():
        if data:
            case_type = data.get('type', 'unknown')
            type_counts[case_type] = type_counts.get(case_type, 0) + 1

    if type_counts:
        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(labels)]

        ax5.pie(sizes, labels=labels, colors=colors_pie,
                autopct='%1.0f%%', startangle=90)
        ax5.set_title('Case Type Distribution', fontweight='bold', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'No Type Data', ha='center',
                 va='center', transform=ax5.transAxes)
        ax5.set_title('Case Type Distribution', fontweight='bold', fontsize=12)

    # 6. Financial Performance Matrix (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])

    # Create performance matrix
    performance_data = []
    for i, (case_key, data) in enumerate(case_data.items()):
        if data:
            npv_score = 1 if data.get('npv', 0) > 0 else 0
            irr_score = 1 if data.get(
                'irr', 0) > 8 else 0  # Above discount rate
            payback_score = 1 if data.get(
                'payback', 0) > 0 and data.get('payback', 0) < 10 else 0
            performance_data.append([npv_score, irr_score, payback_score])

    if performance_data:
        performance_array = np.array(performance_data)
        im = ax6.imshow(performance_array, cmap='RdYlGn',
                        aspect='auto', vmin=0, vmax=1)

        ax6.set_xticks(range(3))
        ax6.set_xticklabels(
            ['NPV > 0', 'IRR > 8%', 'Payback < 10y'], rotation=45, ha='right', fontsize=9)
        ax6.set_yticks(range(len(cases)))
        ax6.set_yticklabels(cases, fontsize=9)
        ax6.set_title('Financial Performance Matrix',
                      fontweight='bold', fontsize=12)

        # Add text annotations
        for i in range(len(cases)):
            for j in range(3):
                text = '✓' if performance_array[i, j] == 1 else '✗'
                ax6.text(j, i, text, ha='center', va='center',
                         fontweight='bold', fontsize=14)
    else:
        ax6.text(0.5, 0.5, 'No Performance Data', ha='center',
                 va='center', transform=ax6.transAxes)
        ax6.set_title('Financial Performance Matrix',
                      fontweight='bold', fontsize=12)

    # 7. Summary Statistics Table (bottom span)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    # Create summary table
    table_data = []
    headers = ['Case', 'NPV ($M)', 'LCOH ($/kg)',
               'LCOE ($/MWh)', 'Capacity (MW)', 'Lifetime (y)']

    for i, (case_key, data) in enumerate(case_data.items()):
        if data:
            row = [
                data.get('name', case_key),
                f"${data.get('npv', 0)/1e6:.0f}M" if data.get('npv') else 'N/A',
                f"${data.get('lcoh', 0):.2f}/kg" if data.get('lcoh',
                                                             0) > 0 else 'N/A',
                f"${data.get('lcoe', 0):.1f}/MWh" if data.get('lcoe',
                                                              0) > 0 else 'N/A',
                f"{data.get('electrolyzer_capacity', 0):.0f} MW" if data.get(
                    'electrolyzer_capacity', 0) > 0 else 'N/A',
                f"{lifetimes[i]:.0f}y"
            ]
            table_data.append(row)

    if table_data:
        table = ax7.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

    ax7.set_title('Summary Statistics Table',
                  fontweight='bold', fontsize=14, pad=20)

    # Add overall title and metadata
    fig.suptitle(f'Comprehensive TEA Analysis Dashboard\n{plant_name} ({target_iso})',
                 fontsize=20, fontweight='bold', y=0.98)

    # Add metadata text
    metadata_text = f"""
    Analysis Overview: Comparison of 5 TEA cases for nuclear-hydrogen integration
    • Case 1: Existing reactor baseline operation
    • Case 2: Existing reactor retrofit with H2/battery systems
    • Case 3: Incremental system financial analysis
    • Case 4: Greenfield nuclear-hydrogen (60-year lifecycle)
    • Case 5: Greenfield nuclear-hydrogen (80-year lifecycle)

    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    fig.text(0.02, 0.02, metadata_text, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.savefig(output_dir / 'comprehensive_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Comprehensive dashboard created successfully")
