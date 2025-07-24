"""
Reporting and plotting coordination functions for the TEA module.
Main entry point for generating reports and coordinating visualizations.
"""

from src.tea.visualization import (
    get_component_color,
    create_cash_flow_plots,
    create_capex_breakdown_plots,
    create_lcoh_comprehensive_dashboard,
    create_lcoh_benchmarking_analysis,
    create_lifecycle_comparison_cash_flow_plots
)
import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Import visualization functions from the separate module

# Initialize logger for this module
logger = logging.getLogger(__name__)


def format_aligned_line(name: str, value: str, min_width: int = 50, indent: str = "  ") -> str:
    """
    Format a name-value pair with consistent colon alignment.

    Args:
        name: The parameter/metric name
        value: The formatted value string
        min_width: Minimum width for name field (default 50)
        indent: Indentation prefix (default "  ")

    Returns:
        Formatted string with aligned colons
    """
    # Calculate the actual width needed for this set of names
    effective_width = max(min_width, len(name))
    return f"{indent}{name:<{effective_width}} : {value}\n"


def format_aligned_section(items: dict, min_width: int = 50, indent: str = "  ") -> str:
    """
    Format a dictionary of name-value pairs with consistent alignment.

    Args:
        items: Dictionary of name -> value pairs
        min_width: Minimum width for name field
        indent: Indentation prefix

    Returns:
        Formatted string with all items aligned
    """
    if not items:
        return ""

    # Calculate the maximum name length for this section
    max_name_length = max(len(str(name)) for name in items.keys())
    effective_width = max(min_width, max_name_length)

    result = ""
    for name, value in items.items():
        result += format_aligned_line(str(name),
                                      str(value), effective_width, indent)

    return result


def plot_results(
    annual_metrics_data: dict,
    financial_metrics_data: dict,
    cash_flows_data: np.ndarray,
    plot_dir: Path,
    construction_period_years: int,
    incremental_metrics_data: dict | None = None,
    case_type: str = None,
    project_lifetime_years: int = None,
):
    """Generate all visualization charts by coordinating specialized visualization functions"""
    logger.info(f"Generating visualization charts...")

    os.makedirs(plot_dir, exist_ok=True)

    try:
        plt.style.use("seaborn")
    except:
        try:
            plt.style.use("ggplot")
        except:
            plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.3
        }
    )

    # Generate cash flow plots with case-specific enhancements
    create_cash_flow_plots(
        cash_flows_data,
        plot_dir,
        construction_period_years,
        incremental_metrics_data,
        case_type,
        project_lifetime_years
    )

    # Generate CAPEX breakdown plots
    create_capex_breakdown_plots(
        annual_metrics_data,
        cash_flows_data,
        plot_dir
    )

    # Generate additional standard plots
    _create_revenue_breakdown_plot(annual_metrics_data, plot_dir)
    _create_opex_breakdown_plot(annual_metrics_data, plot_dir)
    _create_financial_metrics_plot(financial_metrics_data, plot_dir)
    _create_performance_metrics_plot(annual_metrics_data, plot_dir)

    logger.info(f"Base plots saved to {plot_dir}")

    # Create LCOH comprehensive analysis if data is available
    if annual_metrics_data and "lcoh_breakdown_analysis" in annual_metrics_data:
        create_lcoh_comprehensive_dashboard(annual_metrics_data, plot_dir)
        create_lcoh_benchmarking_analysis(annual_metrics_data, plot_dir)

    # Create lifecycle comparison plots if data is available
    if annual_metrics_data and "lifecycle_comparison_analysis" in annual_metrics_data:
        lifecycle_comparison_data = annual_metrics_data["lifecycle_comparison_analysis"]
        create_lifecycle_comparison_cash_flow_plots(
            lifecycle_comparison_data, plot_dir)


def _create_revenue_breakdown_plot(annual_metrics_data: dict, plot_dir: Path):
    """Create revenue breakdown pie chart"""
    logger.debug("Creating revenue breakdown plot...")

    rev_sources = {
        k: annual_metrics_data.get(k, 0)
        for k in [
            "Energy_Revenue",
            "AS_Revenue",
            "H2_Sales_Revenue",
            "H2_Subsidy_Revenue",
        ]
    }
    rev_plot = {
        k.replace("_Revenue", ""): v for k, v in rev_sources.items() if v > 1e-3
    }

    if rev_plot:
        fig_rev, ax_rev = plt.subplots()
        ax_rev.pie(
            rev_plot.values(),
            labels=[f"{k}\n(${v:,.0f})" for k, v in rev_plot.items()],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("viridis", len(rev_plot)),
        )
        ax_rev.set_title("Annual Revenue Breakdown", fontweight="bold")
        ax_rev.axis("equal")
        plt.tight_layout()
        plt.savefig(plot_dir / "revenue_breakdown.png", dpi=300)
        plt.close(fig_rev)


def _create_opex_breakdown_plot(annual_metrics_data: dict, plot_dir: Path):
    """Create OPEX cost breakdown pie chart"""
    logger.debug("Creating OPEX breakdown plot...")

    opex_sources = {
        k: annual_metrics_data.get(k, 0)
        for k in [
            "VOM_Turbine_Cost",
            "VOM_Electrolyzer_Cost",
            "VOM_Battery_Cost",
            "Startup_Cost",
            "Water_Cost",
            "Ramping_Cost",
            "H2_Storage_Cycle_Cost",
        ]
    }

    # Add Fixed O&M from available data
    fixed_om_general = annual_metrics_data.get("Fixed_OM_General", 0)
    if fixed_om_general > 0:
        opex_sources["Fixed OM (General)"] = fixed_om_general

    # Add battery fixed OM if applicable
    battery_capacity = annual_metrics_data.get("Battery_Capacity_MWh", 0)
    if battery_capacity > 0:
        battery_fixed_om = annual_metrics_data.get("Fixed_OM_Battery", 0)
        if battery_fixed_om > 0:
            opex_sources["Fixed OM (Battery)"] = battery_fixed_om

    opex_plot = {k.replace("_Cost", ""): v for k,
                 v in opex_sources.items() if v > 1e-3}

    if opex_plot:
        fig_opex, ax_opex = plt.subplots()
        ax_opex.pie(
            opex_plot.values(),
            labels=[f"{k}\n(${v:,.0f})" for k, v in opex_plot.items()],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("rocket", len(opex_plot)),
        )
        ax_opex.set_title(
            "Annual Operational Cost Breakdown (Base Year)", fontweight="bold"
        )
        ax_opex.axis("equal")
        plt.tight_layout()
        plt.savefig(plot_dir / "opex_cost_breakdown.png", dpi=300)
        plt.close(fig_opex)


def _create_financial_metrics_plot(financial_metrics_data: dict, plot_dir: Path):
    """Create financial metrics summary plot"""
    logger.debug("Creating financial metrics plot...")

    fin_metrics = {
        k: financial_metrics_data.get(k, np.nan)
        for k in [
            "NPV_USD",
            "IRR_percent",
            "Payback_Period_Years",
            "LCOH_USD_per_kg",
        ]
    }
    fin_valid = {
        k.replace("_USD", " (USD)")
        .replace("_percent", " (%)")
        .replace("_Years", " (Years)")
        .replace("_per_kg", " (USD/kg)"): v
        for k, v in fin_metrics.items()
        if not pd.isna(v)
    }

    if fin_valid:
        npv_key = "NPV (USD)" if "NPV (USD)" in fin_valid else None
        npv_value = fin_valid.get("NPV (USD)", None)
        other_metrics = {k: v for k,
                         v in fin_valid.items() if k != "NPV (USD)"}

        fig_fin = plt.figure(figsize=(12, 8))

        if npv_key and npv_value is not None:
            ax_npv = plt.subplot(2, 1, 1)
            npv_bar = ax_npv.barh(
                [npv_key],
                [npv_value],
                color=sns.color_palette("mako", 1)[0]
            )
            ax_npv.set_xlabel("Value (USD)")
            ax_npv.set_title("Net Present Value (NPV)", fontweight="bold")

            ax_npv.text(
                npv_value + 0.01 * abs(npv_value) if npv_value != 0 else 0.01,
                0,
                f"${npv_value:,.2f}",
                va="center",
                ha="left" if npv_value >= 0 else "right",
            )

        if other_metrics:
            ax_other = plt.subplot(2, 1, 2)
            other_bars = ax_other.barh(
                list(other_metrics.keys()),
                list(other_metrics.values()),
                color=sns.color_palette("mako", len(other_metrics))
            )
            ax_other.set_xlabel("Value")
            ax_other.set_title("Other Financial Metrics", fontweight="bold")

            for i, (k, v) in enumerate(other_metrics.items()):
                if "IRR" in k:
                    label_text = f"{v:.2f}%"
                elif "LCOH" in k:
                    label_text = f"${v:.2f}/kg"
                elif "Payback" in k:
                    label_text = f"{v:.2f} years"
                else:
                    label_text = f"${v:,.2f}"

                ax_other.text(
                    v + 0.01 * abs(v) if v != 0 else 0.01,
                    i,
                    label_text,
                    va="center",
                    ha="left" if v >= 0 else "right",
                )

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.savefig(plot_dir / "financial_metrics_summary.png", dpi=300)
        plt.close(fig_fin)


def _create_performance_metrics_plot(annual_metrics_data: dict, plot_dir: Path):
    """Create system performance indicators plot"""
    logger.debug("Creating performance metrics plot...")

    cf_data = {
        "Electrolyzer_CF_percent": annual_metrics_data.get("Electrolyzer_CF_percent", np.nan),
        "Turbine_CF_percent": annual_metrics_data.get("Turbine_CF_percent", np.nan),
        "Battery_SOC_percent": annual_metrics_data.get("Battery_SOC_percent", np.nan),
        "H2_Storage_SOC_percent": annual_metrics_data.get("H2_Storage_SOC_percent", np.nan),
    }

    plot_labels = {
        "Electrolyzer_CF_percent": "Electrolyzer CF (%)",
        "Turbine_CF_percent": "Turbine CF (%)",
        "Battery_SOC_percent": "Battery Avg SOC (%)",
        "H2_Storage_SOC_percent": "H2 Storage Avg SOC (%)",
    }

    cf_valid = {
        plot_labels[k]: v
        for k, v in cf_data.items()
        if not pd.isna(v) and v is not None
    }

    if cf_valid:
        fig_cf, ax_cf = plt.subplots(figsize=(10, 6))
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        ]
        bars = ax_cf.bar(
            range(len(cf_valid)),
            list(cf_valid.values()),
            color=colors[:len(cf_valid)],
        )
        ax_cf.set_ylabel("Percentage (%)")
        ax_cf.set_title("System Performance Metrics", fontweight="bold")

        for i, (k, v) in enumerate(cf_valid.items()):
            ax_cf.text(
                i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9
            )

        ax_cf.set_xticks(range(len(cf_valid)))
        ax_cf.set_xticklabels(list(cf_valid.keys()), rotation=45, ha="right")
        ax_cf.set_ylim(0, max(cf_valid.values()) * 1.1)
        plt.tight_layout()
        plt.savefig(plot_dir / "capacity_factors.png", dpi=300)
        plt.close(fig_cf)


def generate_report(
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
    incremental_metrics_rpt: dict | None = None,
):
    """Generate comprehensive TEA report with improved structure and organization"""
    logger.info(f"Generating TEA report: {output_file_path}")

    # **ENHANCEMENT: Use plant-specific title if available**
    current_module = sys.modules[__name__]
    if hasattr(current_module, 'PLANT_REPORT_TITLE'):
        report_title = getattr(current_module, 'PLANT_REPORT_TITLE')
        subtitle_info = f"ISO Region: {target_iso_rpt}"
        logger.info(f"Using plant-specific report title: {report_title}")
    else:
        report_title = plant_specific_title_rpt if plant_specific_title_rpt else target_iso_rpt
        subtitle_info = f"Target ISO: {target_iso_rpt}"
        logger.info(f"Using default ISO report title: {report_title}")

    with open(output_file_path, "w", encoding="utf-8") as f:
        # ========================================
        # REPORT HEADER
        # ========================================
        f.write(
            f"Technical Economic Analysis Report - {report_title}\n"
            + "=" * (30 + len(report_title))
            + "\n\n"
        )

        # ========================================
        # 1. EXECUTIVE SUMMARY & CONFIGURATION
        # ========================================
        f.write("1. Executive Summary & Configuration\n" + "=" * 37 + "\n\n")

        f.write("1.1 Project Configuration\n" + "-" * 25 + "\n")
        config_items = {
            "ISO Region": target_iso_rpt,
            "Project Lifetime": f"{project_lifetime_years_rpt} years",
            "Construction Period": f"{construction_period_years_rpt} years",
            "Discount Rate": f"{discount_rate_rpt*100:.2f}%",
            "Corporate Tax Rate": f"{tax_rate_rpt*100:.1f}%",
        }

        # Add plant-specific technical parameters
        if annual_metrics_rpt:
            turbine_capacity = annual_metrics_rpt.get("Turbine_Capacity_MW", 0)
            if turbine_capacity > 0:
                config_items["Turbine Capacity"] = f"{turbine_capacity:,.2f} MW"

            thermal_capacity = annual_metrics_rpt.get(
                "thermal_capacity_mwt", 0)
            if thermal_capacity == 0:
                thermal_capacity = annual_metrics_rpt.get(
                    "Thermal_Capacity_MWt", 0)
            if thermal_capacity > 0:
                config_items["Thermal Capacity"] = f"{thermal_capacity:,.2f} MWt"

            thermal_efficiency = annual_metrics_rpt.get(
                "thermal_efficiency", 0)
            if thermal_efficiency == 0:
                thermal_efficiency = annual_metrics_rpt.get(
                    "Thermal_Efficiency", 0)
            if thermal_efficiency > 0:
                config_items[
                    "Thermal Efficiency"] = f"{thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)"

        f.write(format_aligned_section(config_items, min_width=30))
        f.write("\n")

        f.write("1.2 System Capacities\n" + "-" * 18 + "\n")
        if annual_metrics_rpt:
            capacity_items = {}
            capacity_metrics = {
                "Electrolyzer Capacity": annual_metrics_rpt.get("Electrolyzer_Capacity_MW", 0),
                "Hydrogen Storage Capacity": annual_metrics_rpt.get("H2_Storage_Capacity_kg", 0),
                "Battery Energy Capacity": annual_metrics_rpt.get("Battery_Capacity_MWh", 0),
                "Battery Power Capacity": annual_metrics_rpt.get("Battery_Power_MW", 0),
            }
            capacity_units = {
                "Electrolyzer Capacity": "MW",
                "Hydrogen Storage Capacity": "kg",
                "Battery Energy Capacity": "MWh",
                "Battery Power Capacity": "MW",
            }

            for name, value in capacity_metrics.items():
                unit = capacity_units.get(name, "")
                capacity_items[name] = f"{value:,.2f} {unit}"

            # Add hydrogen constant sales rate if available
            h2_constant_sales_rate = annual_metrics_rpt.get(
                "Optimal_H2_Constant_Sales_Rate_kg_hr", 0)
            if h2_constant_sales_rate == 0:
                h2_constant_sales_rate = annual_metrics_rpt.get(
                    "H2_Constant_Sales_Rate_kg_hr", 0)

            if h2_constant_sales_rate > 0:
                capacity_items["Optimal H2 Constant Sales Rate"] = f"{h2_constant_sales_rate:,.2f} kg/hr"
                daily_sales = h2_constant_sales_rate * 24
                annual_sales = daily_sales * 365
                capacity_items["Optimal H2 Daily Sales Rate"] = f"{daily_sales:,.2f} kg/day"
                capacity_items["Optimal H2 Annual Sales Rate"] = f"{annual_sales:,.0f} kg/year"

            f.write(format_aligned_section(capacity_items, min_width=35))
        else:
            f.write("  No capacity data available.\n")
        f.write("\n")

        # ========================================
        # 2. FINANCIAL PERFORMANCE SUMMARY
        # ========================================
        f.write("2. Financial Performance Summary\n" + "=" * 32 + "\n\n")

        f.write("2.1 Key Financial Metrics\n" + "-" * 25 + "\n")
        if financial_metrics_rpt:
            financial_items = {}
            metrics_order = ["IRR_percent", "LCOH_USD_per_kg",
                             "NPV_USD", "Payback_Period_Years", "Roi"]

            for metric in metrics_order:
                if metric in financial_metrics_rpt:
                    v = financial_metrics_rpt[metric]
                    if metric == "IRR_percent":
                        financial_items["IRR (%)"] = f"{v:.2f}%"
                    elif metric == "LCOH_USD_per_kg":
                        financial_items["LCOH (USD/kg)"] = f"${v:.3f}"
                    elif metric == "NPV_USD":
                        financial_items["NPV (USD)"] = f"${v:,.2f}"
                    elif metric == "Payback_Period_Years":
                        financial_items["Payback Period (Years)"] = f"{v:.2f}"
                    elif metric == "Roi":
                        financial_items["Return on Investment (ROI)"] = f"{v:.4f}"

            f.write(format_aligned_section(financial_items, min_width=45))
        else:
            f.write("  No financial metrics data available.\n")
        f.write("\n")

        f.write("2.2 Revenue Breakdown\n" + "-" * 18 + "\n")
        if annual_metrics_rpt:
            # Revenue Mix Analysis
            energy_revenue = annual_metrics_rpt.get("Energy_Revenue", 0)
            h2_sales_revenue = annual_metrics_rpt.get("H2_Sales_Revenue", 0)
            h2_subsidy_revenue = annual_metrics_rpt.get(
                "H2_Subsidy_Revenue", 0)
            as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)
            total_revenue = annual_metrics_rpt.get("Annual_Revenue", 0)

            revenue_items = {
                "Total Annual Revenue": f"${total_revenue:,.2f}"
            }

            if total_revenue > 0:
                energy_pct = (energy_revenue / total_revenue) * 100
                h2_sales_pct = (h2_sales_revenue / total_revenue) * 100
                h2_subsidy_pct = (h2_subsidy_revenue / total_revenue) * 100
                as_pct = (as_revenue / total_revenue) * 100

                revenue_items["Energy Revenue"] = f"${energy_revenue:,.2f} ({energy_pct:.1f}%)"
                revenue_items["H2 Sales Revenue"] = f"${h2_sales_revenue:,.2f} ({h2_sales_pct:.1f}%)"
                revenue_items[
                    "H2 Subsidy Revenue"] = f"${h2_subsidy_revenue:,.2f} ({h2_subsidy_pct:.1f}%)"
                revenue_items["Ancillary Services Revenue"] = f"${as_revenue:,.2f} ({as_pct:.1f}%)"

            f.write(format_aligned_section(revenue_items, min_width=45))
        else:
            f.write("  No revenue data available.\n")
        f.write("\n")

        # ========================================
        # 3. SYSTEM PERFORMANCE ANALYSIS
        # ========================================
        f.write("3. System Performance Analysis\n" + "=" * 30 + "\n\n")

        f.write("3.1 Production Metrics\n" + "-" * 19 + "\n")
        if annual_metrics_rpt:
            production_items = {}

            # Key production metrics
            h2_production = annual_metrics_rpt.get(
                "H2_Production_kg_annual", 0)
            nuclear_generation = annual_metrics_rpt.get(
                "Annual_Nuclear_Generation_MWh", 0)
            electrolyzer_mwh = annual_metrics_rpt.get(
                "Annual_Electrolyzer_MWh", 0)

            if h2_production > 0:
                production_items["Annual H2 Production"] = f"{h2_production:,.0f} kg/year"
                production_items["Daily H2 Production"] = f"{h2_production/365:,.0f} kg/day"

            if nuclear_generation > 0:
                production_items["Annual Nuclear Generation"] = f"{nuclear_generation:,.0f} MWh/year"

            if electrolyzer_mwh > 0:
                production_items["Annual Electrolyzer Consumption"] = f"{electrolyzer_mwh:,.0f} MWh/year"

            # Efficiency metrics
            if h2_production > 0 and electrolyzer_mwh > 0:
                h2_efficiency = h2_production / electrolyzer_mwh  # kg H2 per MWh
                production_items["H2 Production Efficiency"] = f"{h2_efficiency:.2f} kg/MWh"

            f.write(format_aligned_section(production_items, min_width=35))
        else:
            f.write("  No production data available.\n")
        f.write("\n")

        f.write("3.2 Capacity Factors & Utilization\n" + "-" * 31 + "\n")
        if annual_metrics_rpt:
            cf_items = {}

            # Capacity factors
            electrolyzer_cf = annual_metrics_rpt.get(
                "Electrolyzer_CF_percent", 0)
            turbine_cf = annual_metrics_rpt.get("Turbine_CF_percent", 0)
            battery_soc = annual_metrics_rpt.get("Battery_SOC_percent", 0)
            h2_storage_soc = annual_metrics_rpt.get(
                "H2_Storage_SOC_percent", 0)

            if electrolyzer_cf > 0:
                cf_items["Electrolyzer Capacity Factor"] = f"{electrolyzer_cf:.2f}%"
            if turbine_cf > 0:
                cf_items["Turbine Capacity Factor"] = f"{turbine_cf:.2f}%"
            if battery_soc > 0:
                cf_items["Battery Average SOC"] = f"{battery_soc:.2f}%"
            if h2_storage_soc > 0:
                cf_items["H2 Storage Average SOC"] = f"{h2_storage_soc:.2f}%"

            f.write(format_aligned_section(cf_items, min_width=35))
        else:
            f.write("  No capacity factor data available.\n")
        f.write("\n")

        f.write("3.3 Additional Performance Metrics\n" + "-" * 31 + "\n")
        if annual_metrics_rpt:
            # Show other relevant performance metrics, excluding complex data structures
            metrics_to_skip = [
                "capex_breakdown", "total_capex", "lcoh_breakdown_analysis",
                "greenfield_nuclear_analysis", "lifecycle_comparison_analysis",
                "nuclear_baseline_analysis", "nuclear_integrated_analysis",
                "comprehensive_tax_incentive_analysis", "annual_fixed_om_costs",
                "annual_other_replacement_costs", "annual_stack_replacement_costs",
                "electrolyzer_capex", "thermal_capacity_mwt", "thermal_efficiency",
                # Capacity metrics already shown
                "Electrolyzer_Capacity_MW", "H2_Storage_Capacity_kg",
                "Battery_Capacity_MWh", "Battery_Power_MW", "Turbine_Capacity_MW",
                # CF metrics already shown
                "Electrolyzer_CF_percent", "Turbine_CF_percent",
                "Battery_SOC_percent", "H2_Storage_SOC_percent",
                # Production metrics already shown
                "H2_Production_kg_annual", "Annual_Nuclear_Generation_MWh",
                "Annual_Electrolyzer_MWh",
                # Revenue metrics (shown in section 2)
                "Annual_Revenue", "Energy_Revenue", "H2_Sales_Revenue",
                "H2_Subsidy_Revenue", "AS_Revenue",
                # Battery charging metrics (will be shown in section 5)
                "Annual_Battery_Charge_MWh", "Annual_Battery_Charge_From_Grid_MWh",
                "Annual_Battery_Charge_From_NPP_MWh",
            ]

            performance_items = {}
            for k, v in sorted(annual_metrics_rpt.items()):
                # Skip excluded metrics and complex data structures
                if k in metrics_to_skip or isinstance(v, (dict, list)):
                    continue

                # Skip AS-specific detailed metrics
                if any(x in k for x in ["AS_Max_Bid_", "AS_Avg_Bid_", "AS_Total_Deployed_",
                                        "AS_Avg_Deployed_", "AS_Deployment_Efficiency_"]):
                    continue

                # Format and include relevant metrics
                if isinstance(v, (int, float)) and not pd.isna(v) and v != 0:
                    display_name = k.replace('_', ' ').replace('Cf ', 'Capacity Factor ').replace(
                        'Soc ', 'SOC ').replace('Vom ', 'VOM ').replace('Opex ', 'OPEX ').replace(
                        'Capex', 'CAPEX').replace('Mw', 'MW').replace('Mwh', 'MWh').replace(
                        'Usd', 'USD').replace('As ', 'AS ')

                    if 'Revenue' in k or 'Cost' in k or 'USD' in k:
                        formatted_value = f"${v:,.2f}"
                    elif k.endswith('_percent') or 'Percent' in display_name:
                        formatted_value = f"{v:.2f}%"
                    elif 'Hours' in k:
                        formatted_value = f"{v:,.0f}"
                    elif 'Price' in k and 'USD' in k:
                        formatted_value = f"${v:.2f}"
                    else:
                        formatted_value = f"{v:,.2f}"

                    performance_items[display_name] = formatted_value

            if performance_items:
                f.write(format_aligned_section(
                    performance_items, min_width=45))
            else:
                f.write("  No additional performance metrics available.\n")
        else:
            f.write("  No performance data available.\n")
        f.write("\n")

        # ========================================
        # 4. COST ANALYSIS
        # ========================================
        f.write("4. Cost Analysis\n" + "=" * 16 + "\n\n")

        f.write("4.1 Capital Expenditure (CAPEX) Breakdown\n" + "-" * 42 + "\n")
        if annual_metrics_rpt and "capex_breakdown" in annual_metrics_rpt:
            capex_breakdown = annual_metrics_rpt["capex_breakdown"]
            total_capex = annual_metrics_rpt.get(
                "total_capex", sum(capex_breakdown.values()))

            # Sort by values in descending order
            for component, cost in sorted(capex_breakdown.items(), key=lambda x: x[1], reverse=True):
                if cost > 0:
                    percentage = (cost / total_capex *
                                  100) if total_capex > 0 else 0
                    f.write(
                        f"  {component:<30}: ${cost:,.0f} ({percentage:.1f}%)\n")

            f.write(f"\n  Total CAPEX: ${total_capex:,.0f}\n")
        else:
            f.write("  No CAPEX breakdown data available.\n")
        f.write("\n")

        f.write("4.2 Operating Expenditure (OPEX) Breakdown\n" + "-" * 43 + "\n")
        if annual_metrics_rpt:
            opex_items = {}

            # Variable O&M costs
            vom_turbine = annual_metrics_rpt.get("VOM_Turbine_Cost", 0)
            vom_electrolyzer = annual_metrics_rpt.get(
                "VOM_Electrolyzer_Cost", 0)
            vom_battery = annual_metrics_rpt.get("VOM_Battery_Cost", 0)
            startup_cost = annual_metrics_rpt.get("Startup_Cost", 0)
            water_cost = annual_metrics_rpt.get("Water_Cost", 0)
            ramping_cost = annual_metrics_rpt.get("Ramping_Cost", 0)
            h2_storage_cost = annual_metrics_rpt.get(
                "H2_Storage_Cycle_Cost", 0)

            # Fixed O&M costs
            fixed_om_general = annual_metrics_rpt.get("Fixed_OM_General", 0)
            fixed_om_battery = annual_metrics_rpt.get("Fixed_OM_Battery", 0)

            # Nuclear operating costs
            nuclear_total_opex = annual_metrics_rpt.get(
                "Nuclear_Total_OPEX_Annual_USD", 0)

            if vom_turbine > 0:
                opex_items["VOM Turbine"] = f"${vom_turbine:,.0f}"
            if vom_electrolyzer > 0:
                opex_items["VOM Electrolyzer"] = f"${vom_electrolyzer:,.0f}"
            if vom_battery > 0:
                opex_items["VOM Battery"] = f"${vom_battery:,.0f}"
            if startup_cost > 0:
                opex_items["Startup Cost"] = f"${startup_cost:,.0f}"
            if water_cost > 0:
                opex_items["Water Cost"] = f"${water_cost:,.0f}"
            if ramping_cost > 0:
                opex_items["Ramping Cost"] = f"${ramping_cost:,.0f}"
            if h2_storage_cost > 0:
                opex_items["H2 Storage Cycle Cost"] = f"${h2_storage_cost:,.0f}"
            if fixed_om_general > 0:
                opex_items["Fixed O&M (General)"] = f"${fixed_om_general:,.0f}"
            if fixed_om_battery > 0:
                opex_items["Fixed O&M (Battery)"] = f"${fixed_om_battery:,.0f}"
            if nuclear_total_opex > 0:
                opex_items["Nuclear Plant OPEX"] = f"${nuclear_total_opex:,.0f}"

            # Calculate total
            total_opex = sum([vom_turbine, vom_electrolyzer, vom_battery, startup_cost,
                              water_cost, ramping_cost, h2_storage_cost, fixed_om_general,
                              fixed_om_battery, nuclear_total_opex])

            if total_opex > 0:
                opex_items["Total Annual OPEX"] = f"${total_opex:,.0f}"

            if opex_items:
                f.write(format_aligned_section(opex_items, min_width=35))
            else:
                f.write("  No OPEX data available.\n")
        else:
            f.write("  No OPEX data available.\n")
        f.write("\n")

        f.write("4.3 Cost Assumptions\n" + "-" * 17 + "\n")
        f.write("  CAPEX Components (Base Cost for Reference Size):\n")
        capex_items = {}
        for comp, det in sorted(capex_data.items()):
            ref_cap = det.get('reference_total_capacity_mw', 0)
            base_cost = det.get('total_base_cost_for_ref_size', 0)
            learning_rate = det.get('learning_rate_decimal', 0) * 100
            payment_schedule = det.get('payment_schedule_years', {})
            capex_items[comp] = f"${base_cost:,.0f} (Ref: {ref_cap}MW, LR: {learning_rate}%)"

        f.write(format_aligned_section(
            capex_items, min_width=50, indent="    "))

        f.write("\n  O&M Components (Annual Base):\n")
        om_items = {}
        for comp, det in sorted(om_data.items()):
            if comp == "Fixed_OM_Battery":
                base_cost_mw = det.get('base_cost_per_mw_year', 0)
                base_cost_mwh = det.get('base_cost_per_mwh_year', 0)
                inflation_rate = det.get('inflation_rate', 0) * 100
                om_items[
                    comp] = f"${base_cost_mw:,.2f}/MW/yr + ${base_cost_mwh:,.2f}/MWh/yr (Inflation: {inflation_rate:.1f}%)"
            else:
                base_cost = det.get('base_cost', 0)
                inflation_rate = det.get('inflation_rate', 0) * 100
                om_items[comp] = f"${base_cost:,.0f} (Inflation: {inflation_rate:.1f}%)"

        f.write(format_aligned_section(om_items, min_width=50, indent="    "))

        f.write("\n  Major Replacements:\n")
        replacement_items = {}
        for comp, det in sorted(replacement_data.items()):
            if 'cost_percent_initial_capex' in det:
                cost_info = f"{det.get('cost_percent_initial_capex', 0)*100:.2f}% of Initial CAPEX"
            else:
                cost_info = f"${det.get('cost', 0):,.0f}"
            years = det.get('years', [])
            replacement_items[comp] = f"Cost: {cost_info} (Years: {years})"

        f.write(format_aligned_section(
            replacement_items, min_width=50, indent="    "))
        f.write("\n")

        # ========================================
        # 5. REVENUE ANALYSIS
        # ========================================
        f.write("5. Revenue Analysis\n" + "=" * 19 + "\n\n")

        f.write("5.1 Ancillary Services Performance\n" + "-" * 35 + "\n")
        if annual_metrics_rpt:
            as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)
            as_items = {
                "Total Ancillary Services Revenue": f"${as_revenue:,.2f}"
            }

            # AS Revenue as percentage of total revenue
            total_revenue = annual_metrics_rpt.get("Annual_Revenue", 0)
            if total_revenue > 0:
                as_pct = (as_revenue / total_revenue) * 100
                as_items["AS Revenue as % of Total Revenue"] = f"{as_pct:.2f}%"

            f.write(format_aligned_section(as_items, min_width=45))

            # System utilization that affects AS capability
            f.write("\n  System Utilization (affects AS capability):\n")
            utilization_items = {}
            electrolyzer_cf = annual_metrics_rpt.get(
                "Electrolyzer_CF_percent", 0)
            turbine_cf = annual_metrics_rpt.get("Turbine_CF_percent", 0)
            battery_soc = annual_metrics_rpt.get("Battery_SOC_percent", 0)

            if electrolyzer_cf > 0:
                utilization_items["Electrolyzer Capacity Factor"] = f"{electrolyzer_cf:.2f}%"
            if turbine_cf > 0:
                utilization_items["Turbine Capacity Factor"] = f"{turbine_cf:.2f}%"
            if battery_soc > 0:
                utilization_items["Battery SOC"] = f"{battery_soc:.2f}%"

            f.write(format_aligned_section(
                utilization_items, min_width=45, indent="    "))
        else:
            f.write("  No ancillary services data available.\n")
        f.write("\n")

        f.write("5.2 Battery Performance and Charging Analysis\n" + "-" * 47 + "\n")
        battery_capacity = annual_metrics_rpt.get(
            "Battery_Capacity_MWh", 0) if annual_metrics_rpt else 0
        if battery_capacity > 0:
            # Battery charging breakdown
            total_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_MWh", 0)
            grid_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_Grid_MWh", 0)
            npp_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_NPP_MWh", 0)

            f.write("  Battery Charging Electricity Consumption:\n")
            charging_items = {
                "Total Annual Charging": f"{total_charge:,.2f} MWh/year"
            }

            if total_charge > 0:
                grid_pct = (grid_charge / total_charge) * 100
                npp_pct = (npp_charge / total_charge) * 100
                charging_items[
                    "From Grid Purchase"] = f"{grid_charge:,.2f} MWh/year ({grid_pct:.1f}%)"
                charging_items["From NPP (Opportunity Cost)"] = f"{npp_charge:,.2f} MWh/year ({npp_pct:.1f}%)"

            f.write(format_aligned_section(
                charging_items, min_width=40, indent="    "))

            # Battery utilization metrics
            battery_cf = annual_metrics_rpt.get("Battery_CF_percent", 0)
            battery_soc = annual_metrics_rpt.get("Battery_SOC_percent", 0)

            f.write("\n  Battery Utilization:\n")
            utilization_items = {
                "Capacity Factor": f"{battery_cf:.2f}%",
                "Average State of Charge": f"{battery_soc:.2f}%"
            }
            f.write(format_aligned_section(
                utilization_items, min_width=40, indent="    "))

            # Economic implications
            avg_price = annual_metrics_rpt.get(
                "Avg_Electricity_Price_USD_per_MWh", 0)
            if total_charge > 0 and avg_price > 0:
                total_charging_cost = total_charge * avg_price
                grid_cost = grid_charge * avg_price
                opportunity_cost = npp_charge * avg_price

                f.write(
                    f"\n  Economic Impact (at avg price ${avg_price:.2f}/MWh):\n")
                economic_items = {
                    "Total Charging Cost": f"${total_charging_cost:,.2f}/year",
                    "Direct Cost (Grid)": f"${grid_cost:,.2f}/year",
                    "Opportunity Cost (NPP)": f"${opportunity_cost:,.2f}/year"
                }
                f.write(format_aligned_section(
                    economic_items, min_width=40, indent="    "))
        else:
            f.write("  No battery system present in this configuration.\n")
        f.write("\n")

        # ========================================
        # 6. ADVANCED FINANCIAL ANALYSIS
        # ========================================
        f.write("6. Advanced Financial Analysis\n" + "=" * 30 + "\n\n")

        # **NEW: 45U Nuclear PTC Policy Impact Analysis for Integrated System**
        if annual_metrics_rpt and "nuclear_integrated_analysis" in annual_metrics_rpt:
            integrated_analysis = annual_metrics_rpt["nuclear_integrated_analysis"]
            if integrated_analysis.get("includes_45u_analysis", False):
                f.write(
                    "\n6.1. 45U Nuclear PTC Policy Impact Analysis (Existing Plant Retrofit)\n" + "-" * 72 + "\n")
                f.write(
                    "Analysis of 45U Nuclear Production Tax Credit impact on existing nuclear\n")
                f.write(
                    "plant retrofitted with hydrogen production systems. The 45U policy provides\n")
                f.write(
                    "$15/MWh tax credit for existing nuclear plants from 2024-2032.\n\n")

                # 45U Policy Details
                nuclear_45u_benefits = integrated_analysis.get(
                    "nuclear_45u_benefits", {})
                if nuclear_45u_benefits:
                    f.write("45U Policy Configuration:\n")
                    policy_config = {
                        "Credit Rate": f"${nuclear_45u_benefits.get('credit_rate_per_mwh', 15)}/MWh",
                        "Credit Period": f"{nuclear_45u_benefits.get('credit_period_start', 2024)}-{nuclear_45u_benefits.get('credit_period_end', 2032)}",
                        "Eligible Years": f"{nuclear_45u_benefits.get('total_eligible_years', 9)} years",
                        "Annual Generation": f"{nuclear_45u_benefits.get('annual_generation_mwh', 0):,.0f} MWh",
                        "Annual Credit Value": f"${nuclear_45u_benefits.get('annual_credit_value', 0):,.0f}",
                        "Total Credits Over Lifetime": f"${nuclear_45u_benefits.get('total_45u_credits', 0):,.0f}"
                    }
                    f.write(format_aligned_section(
                        policy_config, min_width=35, indent="  "))

                # Financial Impact Comparison
                scenario_without_45u = integrated_analysis.get(
                    "scenario_without_45u", {})
                scenario_with_45u = integrated_analysis.get(
                    "scenario_with_45u", {})
                policy_impact = integrated_analysis.get(
                    "45u_policy_impact", {})

                f.write("\nFinancial Impact Comparison:\n")
                f.write("  WITHOUT 45U Policy (Baseline Retrofit):\n")
                without_45u_items = {
                    "NPV": f"${scenario_without_45u.get('npv_usd', 0):,.0f}",
                    "IRR": f"{scenario_without_45u.get('irr_percent', 0):.2f}%" if scenario_without_45u.get('irr_percent') is not None else "N/A",
                    "Payback Period": f"{scenario_without_45u.get('payback_period_years', 0):.1f} years" if scenario_without_45u.get('payback_period_years') is not None else "N/A"
                }
                f.write(format_aligned_section(
                    without_45u_items, min_width=25, indent="    "))

                f.write("\n  WITH 45U Policy (Enhanced Retrofit):\n")
                with_45u_items = {
                    "NPV": f"${scenario_with_45u.get('npv_usd', 0):,.0f}",
                    "IRR": f"{scenario_with_45u.get('irr_percent', 0):.2f}%" if scenario_with_45u.get('irr_percent') is not None else "N/A",
                    "Payback Period": f"{scenario_with_45u.get('payback_period_years', 0):.1f} years" if scenario_with_45u.get('payback_period_years') is not None else "N/A"
                }
                f.write(format_aligned_section(
                    with_45u_items, min_width=25, indent="    "))

                f.write("\n  45U Policy Financial Impact:\n")
                npv_improvement = policy_impact.get('npv_improvement_usd', 0)
                irr_improvement = policy_impact.get(
                    'irr_improvement_percent', 0)
                impact_items = {
                    "NPV Improvement": f"+${npv_improvement:,.0f}",
                    "IRR Improvement": f"+{irr_improvement:.2f}%" if irr_improvement is not None else "N/A",
                    "Total 45U Credits": f"${policy_impact.get('total_45u_credits_usd', 0):,.0f}",
                    "Credit Eligible Years": f"{policy_impact.get('eligible_years', 0)} years"
                }
                f.write(format_aligned_section(
                    impact_items, min_width=25, indent="    "))

                # Investment Analysis
                f.write("\nRetrofit Investment Analysis:\n")
                retrofit_items = {
                    "Nuclear Plant CAPEX": f"${integrated_analysis.get('nuclear_capex_usd', 0):,.0f} (Existing plant)",
                    "H2 System CAPEX": f"${integrated_analysis.get('h2_system_capex_usd', 0):,.0f}",
                    "Total Retrofit Investment": f"${integrated_analysis.get('total_retrofit_capex_usd', 0):,.0f}",
                    "45U Credits as % of Investment": f"{(policy_impact.get('total_45u_credits_usd', 0) / max(integrated_analysis.get('total_retrofit_capex_usd', 1), 1) * 100):.1f}%"
                }
                f.write(format_aligned_section(
                    retrofit_items, min_width=35, indent="  "))

                # Economic Benefit Analysis
                if npv_improvement > 0:
                    f.write("\nEconomic Benefits:\n")
                    f.write(
                        f"  â€¢ The 45U Nuclear PTC policy improves project NPV by ${npv_improvement:,.0f}\n")
                    f.write(
                        f"  â€¢ This represents a {(npv_improvement / max(abs(scenario_without_45u.get('npv_usd', 1)), 1) * 100):.1f}% improvement over baseline\n")
                    if irr_improvement and irr_improvement > 0:
                        f.write(
                            f"  â€¢ IRR increases by {irr_improvement:.2f} percentage points\n")
                    f.write(
                        f"  â€¢ Total tax credits of ${policy_impact.get('total_45u_credits_usd', 0):,.0f} over {policy_impact.get('eligible_years', 0)} years\n")
                else:
                    f.write(
                        "\nNote: The 45U policy provides financial benefits but may not be sufficient\n")
                    f.write(
                        "to make the project profitable without additional incentives or improvements.\n")

                f.write(
                    "\nNote: This analysis compares existing nuclear plant retrofit scenarios\n")
                f.write(
                    "with and without the 45U Nuclear Production Tax Credit. The credit applies\n")
                f.write(
                    "only to existing nuclear facilities and does not affect new construction.\n\n")

        # **NEW: Nuclear Operating Costs Analysis (Existing Plant)**
        # Display standardized nuclear operating costs for existing nuclear plant
        if annual_metrics_rpt and annual_metrics_rpt.get("Nuclear_Total_OPEX_Annual_USD", 0) > 0:
            f.write(
                "\n6.2. Nuclear Power Plant Operating Costs (Existing Plant)\n" + "-" * 58 + "\n")
            f.write(
                "Analysis of nuclear power plant operating costs using standardized\n")
            f.write(
                "industry parameters. These costs apply to the existing nuclear\n")
            f.write("facility and do not include tax incentives (ITC/PTC/MACRS).\n\n")

            # Nuclear plant configuration
            nuclear_capacity = annual_metrics_rpt.get("Turbine_Capacity_MW", 0)
            annual_nuclear_generation = annual_metrics_rpt.get(
                "Annual_Nuclear_Generation_MWh", 0)
            capacity_factor = (annual_nuclear_generation /
                               (nuclear_capacity * 8760) * 100) if nuclear_capacity > 0 else 0

            f.write("Nuclear Plant Configuration:\n")
            nuclear_config_items = {
                "Nuclear Capacity": f"{nuclear_capacity:,.1f} MW",
                "Annual Generation": f"{annual_nuclear_generation:,.0f} MWh",
                "Capacity Factor": f"{capacity_factor:.1f}%"
            }
            f.write(format_aligned_section(
                nuclear_config_items, min_width=45, indent="  "))

            # Standardized nuclear operating costs
            f.write("\nStandardized Nuclear Operating Costs:\n")
            nuclear_cost_items = {}

            # Unit costs (using centralized config as fallback)
            from . import config
            opex_params = config.NUCLEAR_COST_PARAMETERS["opex_parameters"]

            fixed_om_per_mw_month = annual_metrics_rpt.get(
                "Nuclear_Fixed_OM_USD_per_MW_month", opex_params["fixed_om_per_mw_month"])
            variable_om_per_mwh = annual_metrics_rpt.get(
                "Nuclear_Variable_OM_USD_per_MWh", opex_params["variable_om_per_mwh"])
            fuel_cost_per_mwh = annual_metrics_rpt.get(
                "Nuclear_Fuel_Cost_USD_per_MWh", opex_params["fuel_cost_per_mwh"])
            additional_costs_per_mw_year = annual_metrics_rpt.get(
                "Nuclear_Additional_Costs_USD_per_MW_year", opex_params["additional_costs_per_mw_year"])

            nuclear_cost_items["Fixed O&M Rate"] = f"${fixed_om_per_mw_month:,.0f}/MW/month"
            nuclear_cost_items["Variable O&M Rate"] = f"${variable_om_per_mwh:.1f}/MWh"
            nuclear_cost_items["Fuel Cost Rate"] = f"${fuel_cost_per_mwh:.1f}/MWh"
            nuclear_cost_items["Additional Costs Rate"] = f"${additional_costs_per_mw_year:,.0f}/MW/year"

            f.write(format_aligned_section(
                nuclear_cost_items, min_width=45, indent="  "))

            # Annual costs
            f.write("\nAnnual Nuclear Operating Costs:\n")
            annual_nuclear_costs = {}

            fixed_om_annual = annual_metrics_rpt.get(
                "Nuclear_Fixed_OM_Annual_USD", 0)
            variable_om_annual = annual_metrics_rpt.get(
                "Nuclear_Variable_OM_Annual_USD", 0)
            fuel_cost_annual = annual_metrics_rpt.get(
                "Nuclear_Fuel_Cost_Annual_USD", 0)
            additional_costs_annual = annual_metrics_rpt.get(
                "Nuclear_Additional_Costs_Annual_USD", 0)
            total_nuclear_opex = annual_metrics_rpt.get(
                "Nuclear_Total_OPEX_Annual_USD", 0)

            annual_nuclear_costs["Fixed O&M"] = f"${fixed_om_annual:,.0f}"
            annual_nuclear_costs["Variable O&M"] = f"${variable_om_annual:,.0f}"
            annual_nuclear_costs["Fuel Costs"] = f"${fuel_cost_annual:,.0f}"
            annual_nuclear_costs["Additional Costs"] = f"${additional_costs_annual:,.0f}"
            annual_nuclear_costs["Total Nuclear OPEX"] = f"${total_nuclear_opex:,.0f}"

            f.write(format_aligned_section(
                annual_nuclear_costs, min_width=45, indent="  "))

            # Cost breakdown percentages
            if total_nuclear_opex > 0:
                f.write("\nCost Breakdown by Category:\n")
                cost_breakdown = {}
                cost_breakdown["Fixed O&M"] = f"{(fixed_om_annual/total_nuclear_opex)*100:.1f}%"
                cost_breakdown["Variable O&M"] = f"{(variable_om_annual/total_nuclear_opex)*100:.1f}%"
                cost_breakdown["Fuel Costs"] = f"{(fuel_cost_annual/total_nuclear_opex)*100:.1f}%"
                cost_breakdown["Additional Costs"] = f"{(additional_costs_annual/total_nuclear_opex)*100:.1f}%"

                f.write(format_aligned_section(
                    cost_breakdown, min_width=45, indent="  "))

                # Unit cost analysis
                unit_cost_per_mwh = total_nuclear_opex / \
                    annual_nuclear_generation if annual_nuclear_generation > 0 else 0
                unit_cost_per_mw_year = total_nuclear_opex / \
                    nuclear_capacity if nuclear_capacity > 0 else 0

                f.write("\nUnit Cost Analysis:\n")
                unit_costs = {
                    "Total OPEX per MWh": f"${unit_cost_per_mwh:.2f}/MWh",
                    "Total OPEX per MW/year": f"${unit_cost_per_mw_year:,.0f}/MW/year"
                }
                f.write(format_aligned_section(
                    unit_costs, min_width=45, indent="  "))

            f.write("\nTotal System Operating Costs:\n")
            total_system_opex = annual_metrics_rpt.get(
                "Total_System_OPEX_Annual_USD", 0)
            h2_battery_opex = annual_metrics_rpt.get(
                "H2_Battery_OPEX_Annual_USD", 0)

            if total_system_opex > 0:
                system_opex_items = {
                    "Nuclear Plant OPEX": f"${total_nuclear_opex:,.0f}",
                    "H2/Battery System OPEX": f"${h2_battery_opex:,.0f}",
                    "Total System OPEX": f"${total_system_opex:,.0f}"
                }
                f.write(format_aligned_section(
                    system_opex_items, min_width=45, indent="  "))

                f.write("\nSystem OPEX Breakdown:\n")
                system_breakdown = {
                    "Nuclear Plant": f"{(total_nuclear_opex/total_system_opex)*100:.1f}%",
                    "H2/Battery Systems": f"{(h2_battery_opex/total_system_opex)*100:.1f}%"
                }
                f.write(format_aligned_section(
                    system_breakdown, min_width=45, indent="  "))

            f.write(
                "\nNote: These costs use standardized industry parameters consistent\n")
            f.write("with nuclear baseline analysis (Section 9) and tax incentive\n")
            f.write(
                "analysis (Section 11). No tax incentives are applied to existing\n")
            f.write(
                "nuclear facilities. Nuclear costs are now included in total system\n")
            f.write("financial metrics (IRR, NPV, LCOH) calculations.\n")

        # **NEW: MACRS Depreciation Analysis Section**
        if annual_metrics_rpt and annual_metrics_rpt.get("macrs_enabled", False):
            f.write(
                "\n6.1. MACRS Depreciation Tax Benefits Analysis\n" + "-" * 44 + "\n")

            # Get MACRS data from annual metrics
            total_macrs_depreciation = annual_metrics_rpt.get(
                "macrs_total_depreciation", np.array([]))
            component_macrs_depreciation = annual_metrics_rpt.get(
                "macrs_component_depreciation", {})
            total_capex = annual_metrics_rpt.get("total_capex", 0)

            # Calculate total MACRS depreciation and tax benefits
            total_depreciation_amount = np.sum(total_macrs_depreciation) if len(
                total_macrs_depreciation) > 0 else 0
            total_tax_benefits = total_depreciation_amount * tax_rate_rpt

            f.write("  MACRS Depreciation Summary:\n")
            macrs_summary = {
                "Total CAPEX Subject to MACRS": f"${total_capex:,.0f}",
                "Total MACRS Depreciation": f"${total_depreciation_amount:,.0f}",
                "MACRS Coverage": f"{(total_depreciation_amount/total_capex*100):.1f}%" if total_capex > 0 else "N/A",
                "Corporate Tax Rate": f"{tax_rate_rpt*100:.1f}%",
                "Total Tax Benefits from MACRS": f"${total_tax_benefits:,.0f}",
                "Tax Benefits as % of CAPEX": f"{(total_tax_benefits/total_capex*100):.1f}%" if total_capex > 0 else "N/A"
            }
            f.write(format_aligned_section(
                macrs_summary, min_width=45, indent="    "))

            # Component-wise MACRS breakdown
            if component_macrs_depreciation:
                f.write("\n  MACRS Depreciation by Component Category:\n")

                # Group components by MACRS classification
                nuclear_components = []
                hydrogen_components = []
                battery_components = []
                grid_components = []

                for component_name, depreciation_array in component_macrs_depreciation.items():
                    total_component_depreciation = np.sum(depreciation_array)
                    if total_component_depreciation > 0:
                        if "Nuclear" in component_name or "NPP" in component_name:
                            nuclear_components.append(
                                (component_name, total_component_depreciation))
                        elif "Electrolyzer" in component_name or "H2" in component_name:
                            hydrogen_components.append(
                                (component_name, total_component_depreciation))
                        elif "Battery" in component_name:
                            battery_components.append(
                                (component_name, total_component_depreciation))
                        elif "Grid" in component_name:
                            grid_components.append(
                                (component_name, total_component_depreciation))

                # Display by category
                categories = [
                    ("Nuclear Equipment (15-year MACRS)", nuclear_components),
                    ("Hydrogen Equipment (7-year MACRS)", hydrogen_components),
                    ("Battery Systems (7-year MACRS)", battery_components),
                    ("Grid Infrastructure (15-year MACRS)", grid_components)
                ]

                for category_name, components in categories:
                    if components:
                        category_total = sum(comp[1] for comp in components)
                        category_tax_benefit = category_total * tax_rate_rpt
                        f.write(f"\n    {category_name}:\n")
                        f.write(
                            f"      Total Depreciation: ${category_total:,.0f}\n")
                        f.write(
                            f"      Tax Benefits: ${category_tax_benefit:,.0f}\n")

                        for comp_name, comp_depreciation in sorted(components, key=lambda x: x[1], reverse=True):
                            comp_tax_benefit = comp_depreciation * tax_rate_rpt
                            f.write(
                                f"        {comp_name}: ${comp_depreciation:,.0f} (Tax Benefit: ${comp_tax_benefit:,.0f})\n")

            # Year-by-year MACRS depreciation schedule (first 10 years)
            if len(total_macrs_depreciation) > 0:
                f.write("\n  Annual MACRS Depreciation Schedule (First 10 Years):\n")
                f.write(
                    "    Year    Annual Depreciation    Tax Benefit    Cumulative Depreciation\n")
                f.write(
                    "    ----    ------------------    -----------    ----------------------\n")

                cumulative_depreciation = 0
                for year in range(min(10, len(total_macrs_depreciation))):
                    annual_depreciation = total_macrs_depreciation[year]
                    annual_tax_benefit = annual_depreciation * tax_rate_rpt
                    cumulative_depreciation += annual_depreciation

                    if annual_depreciation > 0:
                        f.write(
                            f"    {year+1:4d}    ${annual_depreciation:>15,.0f}    ${annual_tax_benefit:>10,.0f}    ${cumulative_depreciation:>19,.0f}\n")

                # Show total for remaining years if applicable
                if len(total_macrs_depreciation) > 10:
                    remaining_depreciation = np.sum(
                        total_macrs_depreciation[10:])
                    remaining_tax_benefit = remaining_depreciation * tax_rate_rpt
                    f.write(
                        f"    11+     ${remaining_depreciation:>15,.0f}    ${remaining_tax_benefit:>10,.0f}    (Years 11-{len(total_macrs_depreciation)})\n")

                f.write(
                    f"    Total   ${total_depreciation_amount:>15,.0f}    ${total_tax_benefits:>10,.0f}\n")

            # NPV impact of MACRS
            if financial_metrics_rpt and "NPV_USD" in financial_metrics_rpt:
                current_npv = financial_metrics_rpt["NPV_USD"]

                # Calculate present value of MACRS tax benefits
                macrs_pv = 0
                discount_rate = discount_rate_rpt
                for year in range(len(total_macrs_depreciation)):
                    annual_tax_benefit = total_macrs_depreciation[year] * \
                        tax_rate_rpt
                    if annual_tax_benefit > 0:
                        pv_factor = 1 / ((1 + discount_rate) ** year)
                        macrs_pv += annual_tax_benefit * pv_factor

                npv_without_macrs = current_npv - macrs_pv

                f.write("\n  MACRS Impact on Financial Metrics:\n")
                macrs_impact = {
                    "NPV with MACRS": f"${current_npv:,.0f}",
                    "NPV without MACRS": f"${npv_without_macrs:,.0f}",
                    "MACRS Tax Benefits (Present Value)": f"${macrs_pv:,.0f}",
                    "MACRS Improvement in NPV": f"${macrs_pv:,.0f}",
                    "MACRS as % of Total NPV": f"{(macrs_pv/current_npv*100):.1f}%" if current_npv != 0 else "N/A"
                }
                f.write(format_aligned_section(
                    macrs_impact, min_width=45, indent="    "))

        # Removed duplicate LCOH analysis section - now in 6.2
            pass  # Removed duplicate LCOH section

            lcoh_analysis = annual_metrics_rpt["lcoh_breakdown_analysis"]
            lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
            lcoh_percentages = lcoh_analysis.get("lcoh_percentages", {})
            total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

            f.write(f"  Total LCOH: ${total_lcoh:.3f}/kg H2\n\n")

            # LCOH Component Breakdown
            f.write("  LCOH Component Breakdown:\n")
            sorted_components = sorted(
                lcoh_breakdown.items(), key=lambda x: x[1], reverse=True)

            for component, cost in sorted_components:
                if abs(cost) > 0.001:  # Show significant components (including negative values)
                    percentage = lcoh_percentages.get(component, 0)
                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Electricity Opportunity Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"

                    f.write(
                        f"    {clean_name:<35}: ${cost:>8.3f}/kg ({percentage:>5.1f}%)\n")

            # Cost Category Analysis
            f.write("\n  Cost Category Analysis:\n")

            # Calculate category totals
            capex_total = sum(
                v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
            electricity_total = sum(v for k, v in lcoh_breakdown.items()
                                    if "Opportunity_Cost" in k or "Direct_Cost" in k)
            fixed_om_total = sum(
                v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k)
            variable_opex_total = sum(v for k, v in lcoh_breakdown.items()
                                      if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost",
                                               "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"])
            replacement_total = sum(
                v for k, v in lcoh_breakdown.items() if "Replacement" in k)

            categories = [
                ("Capital Recovery (CAPEX)", capex_total),
                ("Electricity Costs", electricity_total),
                ("Fixed O&M", fixed_om_total),
                ("Variable OPEX", variable_opex_total),
                ("Equipment Replacements", replacement_total)
            ]

            for cat_name, cat_cost in categories:
                if cat_cost > 0.001:
                    cat_percentage = (cat_cost / total_lcoh) * \
                        100 if total_lcoh > 0 else 0
                    f.write(
                        f"    {cat_name:<35}: ${cat_cost:>8.3f}/kg ({cat_percentage:>5.1f}%)\n")

            # LCOH Benchmarking
            f.write("\n  LCOH Benchmarking:\n")
            benchmarks = [
                ("DOE 2030 Target", 2.0),
                ("Steam Methane Reforming (typical)", 1.5),
                ("Grid Electrolysis (typical)", 5.0),
                ("Renewable Electrolysis (typical)", 3.5)
            ]

            for bench_name, bench_value in benchmarks:
                comparison = "âœ“ Below" if total_lcoh < bench_value else "âœ— Above"
                difference = abs(total_lcoh - bench_value)
                f.write(
                    f"    vs {bench_name:<30}: {comparison} by ${difference:.3f}/kg\n")

            # Cost Efficiency Metrics
            if annual_metrics_rpt:
                f.write("\n  Cost Efficiency Metrics:\n")

                annual_h2_production = annual_metrics_rpt.get(
                    "H2_Production_kg_annual", 0)
                electrolyzer_capacity = annual_metrics_rpt.get(
                    "Electrolyzer_Capacity_MW", 0)
                capacity_factor = annual_metrics_rpt.get(
                    "Electrolyzer_CF_percent", 0) / 100

                if annual_h2_production > 0:
                    daily_production = annual_h2_production / 365
                    f.write(
                        f"    Daily H2 Production Rate            : {daily_production:,.0f} kg/day\n")

                if capacity_factor > 0:
                    f.write(
                        f"    Electrolyzer Capacity Factor        : {capacity_factor*100:.1f}%\n")
                    lcoh_per_cf = total_lcoh / \
                        (capacity_factor * 100) if capacity_factor > 0 else 0
                    f.write(
                        f"    LCOH per Capacity Factor Point      : ${lcoh_per_cf:.4f}/kg per 1%\n")

                if electrolyzer_capacity > 0 and annual_h2_production > 0:
                    specific_production = annual_h2_production / electrolyzer_capacity
                    f.write(
                        f"    Specific H2 Production              : {specific_production:,.0f} kg/MW/year\n")

        f.write("6.1 Incremental Financial Analysis\n" + "-" * 35 + "\n")
        f.write("Analysis of H2/Battery System vs. Nuclear Baseline\n\n")

        if incremental_metrics_rpt:
            # Core incremental financial metrics
            f.write("  Incremental Financial Results:\n")
            core_metrics = {
                "NPV_USD": ("NPV (USD)", "currency"),
                "IRR_percent": ("IRR (%)", "percent"),
                "Payback_Period_Years": ("Payback Period (Years)", "number"),
                "Total_Incremental_CAPEX_Learned_USD": ("Total Incremental CAPEX (USD)", "currency"),
                "Annual_Electricity_Opportunity_Cost_USD": ("Annual Electricity Opportunity Cost (USD)", "currency"),
            }

            incremental_items = {}
            for metric_key, (metric_label, format_type) in core_metrics.items():
                if metric_key in incremental_metrics_rpt:
                    value = incremental_metrics_rpt[metric_key]
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if format_type == "currency":
                            formatted_value = f"${value:,.2f}"
                        elif format_type == "percent":
                            formatted_value = f"{value:.2f}%"
                        else:
                            formatted_value = f"{value:.2f}"
                        incremental_items[metric_label] = formatted_value

            # Calculate and show incremental ROI
            inc_npv = incremental_metrics_rpt.get("NPV_USD")
            inc_total_capex = incremental_metrics_rpt.get(
                "Total_Incremental_CAPEX_Learned_USD")
            if inc_npv is not None and inc_total_capex and inc_total_capex > 0:
                inc_roi = inc_npv / inc_total_capex
                incremental_items["Incremental ROI"] = f"{inc_roi:.4f}"

            f.write(format_aligned_section(
                incremental_items, min_width=45, indent="    "))

            # Baseline nuclear plant analysis
            f.write("\n  Baseline Nuclear Plant Analysis:\n")
            baseline_revenue = incremental_metrics_rpt.get(
                "Annual_Electricity_Opportunity_Cost_USD", 0)
            if baseline_revenue > 0:
                # Estimate baseline values from available data
                turbine_capacity = annual_metrics_rpt.get(
                    "Turbine_Capacity_MW", 0) if annual_metrics_rpt else 0
                turbine_cf = annual_metrics_rpt.get(
                    "Turbine_CF_percent", 0) if annual_metrics_rpt else 0
                avg_lmp = annual_metrics_rpt.get(
                    "Avg_Electricity_Price_USD_per_MWh", 0) if annual_metrics_rpt else 0

                baseline_items = {}
                # Calculate baseline generation and revenue
                if turbine_capacity > 0 and turbine_cf > 0:
                    baseline_annual_generation = turbine_capacity * \
                        8760 * (turbine_cf / 100)
                    calculated_baseline_revenue = baseline_annual_generation * avg_lmp
                    baseline_items[
                        "Annual Baseline Revenue (Electricity Sales)"] = f"${calculated_baseline_revenue:,.2f}"
                else:
                    baseline_items[
                        "Annual Baseline Revenue (Electricity Sales)"] = f"${baseline_revenue:,.2f}"

                # Estimate baseline OPEX (mainly VOM)
                baseline_opex = annual_metrics_rpt.get(
                    "VOM_Turbine_Cost", 0) if annual_metrics_rpt else 0
                baseline_profit = (calculated_baseline_revenue if 'calculated_baseline_revenue' in locals(
                ) else baseline_revenue) - baseline_opex
                baseline_margin = (baseline_profit / (calculated_baseline_revenue if 'calculated_baseline_revenue' in locals() else baseline_revenue)
                                   * 100) if (calculated_baseline_revenue if 'calculated_baseline_revenue' in locals() else baseline_revenue) > 0 else 0

                baseline_items["Annual Baseline OPEX (Turbine VOM)"] = f"${baseline_opex:,.2f}"
                baseline_items["Annual Baseline Profit"] = f"${baseline_profit:,.2f}"
                baseline_items["Baseline Profit Margin"] = f"{baseline_margin:.1f}%"

                f.write(format_aligned_section(
                    baseline_items, min_width=45, indent="    "))
            else:
                f.write("    Baseline analysis data not available\n")

            # **ENHANCED: Electricity cost analysis using nuclear LCOE (mandatory per memory)**
            f.write("\n  Electricity Cost Analysis (Using Nuclear LCOE):\n")

            # Electrolyzer electricity consumption
            electrolyzer_mwh = annual_metrics_rpt.get(
                "Annual_Electrolyzer_MWh", 0) if annual_metrics_rpt else 0
            nuclear_lcoe = annual_metrics_rpt.get(
                "Nuclear_LCOE_USD_per_MWh", 0) if annual_metrics_rpt else 0
            avg_price = annual_metrics_rpt.get(
                "Avg_Electricity_Price_USD_per_MWh", 0) if annual_metrics_rpt else 0

            # In incremental analysis, Nuclear LCOE may be zero (since nuclear plant already exists)
            # Use market price as the opportunity cost for electricity in incremental analysis
            if nuclear_lcoe <= 0 and avg_price > 0:
                effective_electricity_price = avg_price
                price_note = f"market price (incremental analysis)"
            else:
                effective_electricity_price = nuclear_lcoe
                price_note = f"Nuclear LCOE"

            electrolyzer_cost = electrolyzer_mwh * effective_electricity_price
            f.write(
                f"    Electrolyzer Electricity Consumption: {electrolyzer_mwh:,.0f} MWh/year\n")
            f.write(
                f"    Electrolyzer Electricity Cost (at {price_note}): ${electrolyzer_cost:,.2f}/year\n")
            f.write(f"    Nuclear LCOE: ${nuclear_lcoe:.2f}/MWh\n")
            if nuclear_lcoe <= 0:
                f.write(
                    f"    Note: Nuclear LCOE is zero in incremental analysis (existing plant)\n")
                f.write(
                    f"    Note: Using market price ${avg_price:.2f}/MWh for opportunity cost calculation\n")

            # **ENHANCED: Battery charging cost analysis**
            if annual_metrics_rpt and annual_metrics_rpt.get("Battery_Capacity_MWh", 0) > 0:
                f.write("\n  Battery Charging Cost Analysis:\n")

                grid_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_Grid_MWh", 0)
                npp_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_NPP_MWh", 0)

                # Use appropriate pricing for battery charging costs
                # In incremental analysis, use market price for both grid and NPP charging
                if nuclear_lcoe <= 0 and avg_price > 0:
                    # Incremental analysis: use market price for NPP opportunity cost
                    npp_price = avg_price
                    npp_price_note = "market price (incremental analysis)"
                else:
                    # Greenfield analysis: use Nuclear LCOE for NPP opportunity cost
                    npp_price = nuclear_lcoe
                    npp_price_note = "Nuclear LCOE"

                direct_cost = grid_charge * avg_price
                opportunity_cost = npp_charge * npp_price
                total_charging_cost = direct_cost + opportunity_cost

                f.write(f"    Grid Charging: {grid_charge:,.0f} MWh/year\n")
                f.write(f"    NPP Charging: {npp_charge:,.0f} MWh/year\n")
                f.write(
                    f"    Direct Operating Cost (Grid at Market Price): ${direct_cost:,.2f}/year\n")
                f.write(
                    f"    Opportunity Cost (NPP at {npp_price_note}): ${opportunity_cost:,.2f}/year\n")
                f.write(
                    f"    Total Battery Charging Cost: ${total_charging_cost:,.2f}/year\n")

                if total_charging_cost > 0:
                    direct_pct = (direct_cost / total_charging_cost) * 100
                    opp_pct = (opportunity_cost / total_charging_cost) * 100
                    f.write(
                        f"    Cost Breakdown: {direct_pct:.1f}% Direct, {opp_pct:.1f}% Opportunity\n")

                f.write(
                    f"    Note: NPP charging uses {npp_price_note} (${npp_price:.2f}/MWh) as opportunity cost\n")
                f.write(
                    f"    Note: Grid charging uses market price (${avg_price:.2f}/MWh) as direct cost\n")

            # **ENHANCED: AS opportunity cost analysis**
            if annual_metrics_rpt:
                as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)
                if as_revenue > 0:
                    f.write("\n  Ancillary Services Opportunity Cost Analysis:\n")

                    # Estimate AS opportunity cost (lost electricity sales due to AS provision)
                    # This is a simplified calculation based on available data
                    electrolyzer_capacity = annual_metrics_rpt.get(
                        "Electrolyzer_Capacity_MW", 0)
                    avg_lmp = annual_metrics_rpt.get(
                        "Avg_Electricity_Price_USD_per_MWh", 0)

                    # Rough estimate: AS provision reduces electricity sales
                    # This could be refined with more detailed AS deployment data
                    estimated_as_opportunity_cost = as_revenue * 0.4  # Rough approximation
                    net_as_benefit = as_revenue - estimated_as_opportunity_cost
                    net_as_margin = (net_as_benefit /
                                     as_revenue * 100) if as_revenue > 0 else 0

                    f.write(f"    AS Revenue: ${as_revenue:,.2f}/year\n")
                    f.write(
                        f"    AS Opportunity Cost (Lost Electricity Sales): ${estimated_as_opportunity_cost:,.2f}/year\n")
                    f.write(
                        f"    Net AS Benefit: ${net_as_benefit:,.2f}/year\n")
                    f.write(f"    Net AS Margin: {net_as_margin:.1f}%\n")

            # **ENHANCED: HTE thermal opportunity cost analysis (mandatory per memory)**
            if annual_metrics_rpt:
                hte_thermal_cost = incremental_metrics_rpt.get(
                    "Annual_HTE_Thermal_Opportunity_Cost_USD", 0)
                # Always show HTE analysis section, even if cost is zero
                f.write("\n  HTE Thermal Energy Opportunity Cost Analysis:\n")

                # Get thermal efficiency and steam consumption data
                thermal_efficiency = annual_metrics_rpt.get(
                    "thermal_efficiency", 0)
                steam_consumption = annual_metrics_rpt.get(
                    "HTE_Steam_Consumption_Annual_MWth", 0)
                avg_price = annual_metrics_rpt.get(
                    "Avg_Electricity_Price_USD_per_MWh", 0)
                lost_generation = annual_metrics_rpt.get(
                    "HTE_Lost_Electricity_Generation_Annual_MWh", 0)
                hte_mode = annual_metrics_rpt.get("HTE_Mode_Detected", False)

                f.write(
                    f"    HTE Mode Detected: {'Yes' if hte_mode else 'No (LTE Mode)'}\n")
                f.write(
                    f"    Annual Steam Consumption: {steam_consumption:,.1f} MWth/year\n")
                f.write(
                    f"    Thermal Efficiency: {thermal_efficiency:.4f} ({thermal_efficiency*100:.2f}%)\n")
                f.write(
                    f"    Lost Electricity Generation: {lost_generation:,.1f} MWh/year\n")
                f.write(
                    f"    Average Electricity Price: ${avg_price:.2f}/MWh\n")
                f.write(
                    f"    HTE Thermal Opportunity Cost: ${hte_thermal_cost:,.2f}/year\n")

                # Calculate cost per kg H2 if H2 production data is available
                h2_production = annual_metrics_rpt.get(
                    "H2_Production_kg_annual", 0)
                if h2_production > 0:
                    hte_cost_per_kg = hte_thermal_cost / h2_production
                    f.write(
                        f"    HTE Thermal Cost per kg H2: ${hte_cost_per_kg:.3f}/kg\n")

                # Show impact on LCOH
                if hte_thermal_cost > 0:
                    f.write(
                        f"    Note: HTE thermal opportunity cost included in LCOH calculation\n")
                else:
                    f.write(
                        f"    Note: No thermal opportunity cost (LTE mode or no steam consumption)\n")

            f.write("\n")
        else:
            # Show status when incremental analysis is not available
            f.write("  Status: Incremental analysis not performed\n")
            f.write("  \n")
            f.write("  Current System Overview (without incremental comparison):\n")
            if annual_metrics_rpt:
                total_revenue = annual_metrics_rpt.get("Annual_Revenue", 0)
                energy_revenue = annual_metrics_rpt.get("Energy_Revenue", 0)
                h2_revenue = annual_metrics_rpt.get("H2_Sales_Revenue", 0)
                as_revenue = annual_metrics_rpt.get("AS_Revenue", 0)

                overview_items = {
                    "Total Annual Revenue": f"${total_revenue:,.2f}",
                    "Energy Sales": f"${energy_revenue:,.2f}",
                    "H2 Sales": f"${h2_revenue:,.2f}",
                    "Ancillary Services": f"${as_revenue:,.2f}"
                }
                f.write(format_aligned_section(
                    overview_items, min_width=35, indent="    "))
        f.write("\n")

        # LCOH Detailed Analysis
        if annual_metrics_rpt and "lcoh_breakdown_analysis" in annual_metrics_rpt:
            f.write(
                "6.2 Detailed Levelized Cost of Hydrogen (LCOH) Analysis\n" + "-" * 58 + "\n")

            lcoh_analysis = annual_metrics_rpt["lcoh_breakdown_analysis"]
            lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
            lcoh_percentages = lcoh_analysis.get("lcoh_percentages", {})
            total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

            f.write(f"  Total LCOH: ${total_lcoh:.3f}/kg H2\n\n")

            # LCOH Component Breakdown
            f.write("  LCOH Component Breakdown:\n")
            sorted_components = sorted(
                lcoh_breakdown.items(), key=lambda x: x[1], reverse=True)

            lcoh_items = {}
            for component, cost in sorted_components:
                if abs(cost) > 0.001:  # Show significant components (including negative values)
                    percentage = lcoh_percentages.get(component, 0)
                    clean_name = component.replace(
                        "CAPEX_", "").replace("_", " ").title()
                    if "Electricity Opportunity Cost" in clean_name:
                        clean_name = "Electricity Opportunity Cost"
                    elif "Npp Modifications" in clean_name:
                        clean_name = "NPP Modifications"

                    lcoh_items[clean_name] = f"${cost:.3f}/kg ({percentage:.1f}%)"

            f.write(format_aligned_section(
                lcoh_items, min_width=35, indent="    "))

            # Cost Category Analysis
            f.write("\n  Cost Category Analysis:\n")
            capex_total = sum(
                v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
            electricity_total = sum(v for k, v in lcoh_breakdown.items(
            ) if "Opportunity_Cost" in k or "Direct_Cost" in k)
            fixed_om_total = sum(
                v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k)
            variable_opex_total = sum(v for k, v in lcoh_breakdown.items() if k in [
                                      "VOM_Electrolyzer", "VOM_Battery", "Water_Cost", "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"])
            replacement_total = sum(
                v for k, v in lcoh_breakdown.items() if "Replacement" in k)

            category_items = {}
            categories = [
                ("Capital Recovery (CAPEX)", capex_total),
                ("Electricity Costs", electricity_total),
                ("Fixed O&M", fixed_om_total),
                ("Variable OPEX", variable_opex_total),
                ("Equipment Replacements", replacement_total)
            ]

            for cat_name, cat_cost in categories:
                if cat_cost > 0.001:
                    cat_percentage = (cat_cost / total_lcoh) * \
                        100 if total_lcoh > 0 else 0
                    category_items[cat_name] = f"${cat_cost:.3f}/kg ({cat_percentage:.1f}%)"

            f.write(format_aligned_section(
                category_items, min_width=35, indent="    "))

            # LCOH Benchmarking
            f.write("\n  LCOH Benchmarking:\n")
            benchmarks = [
                ("DOE 2030 Target", 2.0),
                ("Steam Methane Reforming (typical)", 1.5),
                ("Grid Electrolysis (typical)", 5.0),
                ("Renewable Electrolysis (typical)", 3.5)
            ]

            benchmark_items = {}
            for bench_name, bench_value in benchmarks:
                comparison = "✓ Below" if total_lcoh < bench_value else "✗ Above"
                difference = abs(total_lcoh - bench_value)
                benchmark_items[f"vs {bench_name}"] = f"{comparison} by ${difference:.3f}/kg"

            f.write(format_aligned_section(
                benchmark_items, min_width=35, indent="    "))
        f.write("\n")

        # ========================================
        # 7. COMPARATIVE ANALYSIS
        # ========================================
        f.write("7. Comparative Analysis\n" + "=" * 23 + "\n\n")

        f.write("7.1 Nuclear Power Plant Baseline Analysis\n" + "-" * 42 + "\n")
        if annual_metrics_rpt and "nuclear_baseline_analysis" in annual_metrics_rpt:
            baseline_analysis = annual_metrics_rpt["nuclear_baseline_analysis"]
            f.write(
                "Analysis of the nuclear power plant's financial performance under\n")
            f.write(
                "baseline operation (no modifications for hydrogen production).\n")
            f.write(
                "Uses Nameplate Power Factor from NPP specifications and hourly\n")
            f.write("electricity prices from market data.\n\n")

            # Plant Configuration
            plant_params = baseline_analysis.get("plant_parameters", {})
            f.write("Plant Configuration:\n")
            plant_config_items = {
                "Plant Name": plant_params.get("plant_name", "Unknown"),
                "ISO Region": plant_params.get("iso_region", "Unknown"),
                "Turbine Capacity": f"{plant_params.get('turbine_capacity_mw', 0):.1f} MW",
                "Nameplate Power Factor": f"{plant_params.get('nameplate_power_factor', 0):.3f}",
                "Operational Capacity Factor": f"{plant_params.get('capacity_factor', 0):.1%}"
            }
            f.write(format_aligned_section(
                plant_config_items, min_width=35, indent="  "))

            # Market Conditions
            market_data = baseline_analysis.get("electricity_market", {})
            f.write("\nElectricity Market Conditions:\n")
            market_items = {
                "Average Electricity Price": f"${market_data.get('avg_electricity_price_usd_per_mwh', 0):.2f}/MWh",
                "Minimum Electricity Price": f"${market_data.get('min_electricity_price_usd_per_mwh', 0):.2f}/MWh",
                "Maximum Electricity Price": f"${market_data.get('max_electricity_price_usd_per_mwh', 0):.2f}/MWh",
                "Hourly Price Data Used": "Yes" if market_data.get('using_hourly_prices', False) else "No (Average Price)"
            }
            f.write(format_aligned_section(
                market_items, min_width=35, indent="  "))

            # Annual Performance
            annual_perf = baseline_analysis.get("annual_performance", {})
            f.write("\nAnnual Performance Metrics:\n")
            annual_items = {
                "Annual Generation": f"{annual_perf.get('annual_generation_mwh', 0):,.0f} MWh",
                "Annual Revenue": f"${annual_perf.get('annual_revenue_usd', 0):,.0f}",
                "Annual Fixed O&M": f"${annual_perf.get('annual_fixed_om_usd', 0):,.0f}",
                "Annual Variable O&M": f"${annual_perf.get('annual_variable_om_usd', 0):,.0f}",
                "Annual Fuel Costs": f"${annual_perf.get('annual_fuel_cost_usd', 0):,.0f}",
                "Annual Additional Costs": f"${annual_perf.get('annual_additional_costs_usd', 0):,.0f}",
                "Total Annual OPEX": f"${annual_perf.get('annual_total_opex_usd', 0):,.0f}",
                "Annual Net Cash Flow (Before Tax)": f"${annual_perf.get('annual_net_cash_flow_before_tax_usd', 0):,.0f}",
                "Annual Taxes": f"${annual_perf.get('annual_taxes_usd', 0):,.0f}",
                "Annual Net Cash Flow (After Tax)": f"${annual_perf.get('annual_net_cash_flow_after_tax_usd', 0):,.0f}",
                "Profit Margin": f"{annual_perf.get('profit_margin_percent', 0):.1f}%"
            }
            f.write(format_aligned_section(
                annual_items, min_width=45, indent="  "))

            # Financial Metrics
            financial_metrics = baseline_analysis.get("financial_metrics", {})
            f.write("\nLifecycle Financial Analysis:\n")
            financial_items = {
                f"Project Lifetime": f"{financial_metrics.get('project_lifetime_years', 0)} years",
                "Discount Rate": f"{financial_metrics.get('discount_rate_percent', 0):.1f}%",
                "Tax Rate": f"{financial_metrics.get('tax_rate_percent', 0):.1f}%",
                "Net Present Value (NPV)": f"${financial_metrics.get('npv_usd', 0):,.0f}"
            }

            # Handle IRR and Payback which might be None
            irr_value = financial_metrics.get('irr_percent')
            if irr_value is not None:
                financial_items["Internal Rate of Return (IRR)"] = f"{irr_value:.2f}%"
            else:
                financial_items["Internal Rate of Return (IRR)"] = "N/A"

            payback_value = financial_metrics.get('payback_period_years')
            if payback_value is not None:
                financial_items["Payback Period"] = f"{payback_value:.1f} years"
            else:
                financial_items["Payback Period"] = "N/A"

            f.write(format_aligned_section(
                financial_items, min_width=45, indent="  "))

            # Lifecycle Totals
            lifecycle_totals = baseline_analysis.get("lifecycle_totals", {})
            f.write("\nLifecycle Totals:\n")
            lifecycle_items = {
                "Total Revenue": f"${lifecycle_totals.get('total_revenue_usd', 0):,.0f}",
                "Total OPEX": f"${lifecycle_totals.get('total_opex_usd', 0):,.0f}",
                "Total Replacement Costs": f"${lifecycle_totals.get('total_replacement_costs_usd', 0):,.0f}",
                "Total Taxes": f"${lifecycle_totals.get('total_taxes_usd', 0):,.0f}",
                "Total Net Cash Flow": f"${lifecycle_totals.get('total_net_cash_flow_usd', 0):,.0f}"
            }
            f.write(format_aligned_section(
                lifecycle_items, min_width=35, indent="  "))

            # Replacement Schedule
            replacement_schedule = baseline_analysis.get(
                "replacement_schedule", {})
            if replacement_schedule:
                f.write("\nMajor Replacement/Refurbishment Schedule:\n")
                for year, cost in sorted(replacement_schedule.items()):
                    f.write(
                        f"  Year {year:2d}: ${cost:,.0f} (Major component replacement/upgrade)\n")

            # Performance Indicators
            f.write("\nKey Performance Indicators:\n")
            revenue_per_mwh = (annual_perf.get('annual_revenue_usd', 0) /
                               annual_perf.get('annual_generation_mwh', 1)) if annual_perf.get('annual_generation_mwh', 0) > 0 else 0
            opex_per_mwh = (annual_perf.get('annual_total_opex_usd', 0) /
                            annual_perf.get('annual_generation_mwh', 1)) if annual_perf.get('annual_generation_mwh', 0) > 0 else 0
            capacity_mw = plant_params.get('turbine_capacity_mw', 0)
            revenue_per_mw = annual_perf.get(
                'annual_revenue_usd', 0) / capacity_mw if capacity_mw > 0 else 0

            kpi_items = {
                "Revenue per MWh": f"${revenue_per_mwh:.2f}/MWh",
                "OPEX per MWh": f"${opex_per_mwh:.2f}/MWh",
                "Annual Revenue per MW": f"${revenue_per_mw:,.0f}/MW",
                "Capacity Utilization": f"{plant_params.get('capacity_factor', 0):.1%}",
                "Economic Efficiency": f"Operating Profitably" if annual_perf.get('profit_margin_percent', 0) > 0 else "Operating at Loss"
            }
            f.write(format_aligned_section(
                kpi_items, min_width=35, indent="  "))

            # **NEW: 45U Nuclear PTC Policy Impact on Baseline Operations**
            if baseline_analysis.get("includes_45u_analysis", False):
                f.write("\n45U Nuclear PTC Policy Impact Analysis:\n")
                f.write(
                    "The following analysis compares nuclear plant baseline operations\n")
                f.write(
                    "with and without the 45U Nuclear Production Tax Credit.\n\n")

                # 45U Policy Benefits
                nuclear_45u_benefits = baseline_analysis.get(
                    "nuclear_45u_benefits", {})
                if nuclear_45u_benefits:
                    f.write("45U Policy Details:\n")
                    policy_details = {
                        "Credit Rate": f"${nuclear_45u_benefits.get('credit_rate_per_mwh', 15)}/MWh",
                        "Credit Period": f"{nuclear_45u_benefits.get('credit_period_start', 2024)}-{nuclear_45u_benefits.get('credit_period_end', 2032)}",
                        "Eligible Years": f"{nuclear_45u_benefits.get('total_eligible_years', 9)} years",
                        "Annual Credit Value": f"${nuclear_45u_benefits.get('annual_credit_value', 0):,.0f}",
                        "Total Credits": f"${nuclear_45u_benefits.get('total_45u_credits', 0):,.0f}"
                    }
                    f.write(format_aligned_section(
                        policy_details, min_width=25, indent="  "))

                # Comparative Financial Results
                scenario_without_45u = baseline_analysis.get(
                    "scenario_without_45u", {})
                scenario_with_45u = baseline_analysis.get(
                    "scenario_with_45u", {})
                policy_impact = baseline_analysis.get("45u_policy_impact", {})

                f.write("\nComparative Financial Results:\n")
                f.write("  WITHOUT 45U Policy (Current Baseline):\n")
                without_45u_metrics = {
                    "NPV": f"${scenario_without_45u.get('npv_usd', 0):,.0f}",
                    "IRR": f"{scenario_without_45u.get('irr_percent', 0):.2f}%" if scenario_without_45u.get('irr_percent') is not None else "N/A",
                    "Payback": f"{scenario_without_45u.get('payback_period_years', 0):.1f} years" if scenario_without_45u.get('payback_period_years') is not None else "N/A"
                }
                f.write(format_aligned_section(
                    without_45u_metrics, min_width=15, indent="    "))

                f.write("\n  WITH 45U Policy (Enhanced Baseline):\n")
                with_45u_metrics = {
                    "NPV": f"${scenario_with_45u.get('npv_usd', 0):,.0f}",
                    "IRR": f"{scenario_with_45u.get('irr_percent', 0):.2f}%" if scenario_with_45u.get('irr_percent') is not None else "N/A",
                    "Payback": f"{scenario_with_45u.get('payback_period_years', 0):.1f} years" if scenario_with_45u.get('payback_period_years') is not None else "N/A"
                }
                f.write(format_aligned_section(
                    with_45u_metrics, min_width=15, indent="    "))

                f.write("\n  45U Policy Impact:\n")
                npv_improvement = policy_impact.get('npv_improvement_usd', 0)
                irr_improvement = policy_impact.get(
                    'irr_improvement_percent', 0)
                impact_metrics = {
                    "NPV Improvement": f"+${npv_improvement:,.0f}",
                    "IRR Improvement": f"+{irr_improvement:.2f}%" if irr_improvement is not None else "N/A",
                    "Total Tax Credits": f"${policy_impact.get('total_45u_credits_usd', 0):,.0f}"
                }
                f.write(format_aligned_section(
                    impact_metrics, min_width=25, indent="    "))

                # Policy Benefits Analysis
                if npv_improvement > 0:
                    f.write("\n45U Policy Benefits Assessment:\n")
                    annual_revenue = annual_perf.get('annual_revenue_usd', 0)
                    if annual_revenue > 0:
                        revenue_improvement = (nuclear_45u_benefits.get(
                            'annual_credit_value', 0) / annual_revenue) * 100
                        f.write(
                            f"  â€¢ 45U credits increase annual revenue by {revenue_improvement:.1f}%\n")

                    if scenario_without_45u.get('npv_usd', 0) != 0:
                        npv_percentage_improvement = (
                            npv_improvement / abs(scenario_without_45u.get('npv_usd', 1))) * 100
                        f.write(
                            f"  â€¢ NPV improvement of {npv_percentage_improvement:.1f}% over baseline\n")

                    f.write(
                        f"  â€¢ Policy provides ${nuclear_45u_benefits.get('total_45u_credits', 0):,.0f} in tax benefits over {nuclear_45u_benefits.get('total_eligible_years', 0)} years\n")

                    if irr_improvement and irr_improvement > 0:
                        f.write(
                            f"  â€¢ Return on investment improves by {irr_improvement:.2f} percentage points\n")
                else:
                    f.write(
                        "\nNote: While the 45U policy provides financial benefits, additional\n")
                    f.write(
                        "market or policy support may be needed for optimal plant economics.\n")

                f.write(
                    "\nPolicy Eligibility Note: The 45U Nuclear Production Tax Credit applies\n")
                f.write(
                    "exclusively to existing nuclear facilities and is available from 2024\n")
                f.write(
                    "through 2032. New nuclear construction is not eligible for this credit.\n\n")

            f.write(
                "\nNote: This baseline analysis represents the nuclear plant's financial\n")
            f.write(
                "performance under current market conditions without any modifications\n")
            f.write("for hydrogen production or other alternative applications.\n\n")

        f.write("7.2 Greenfield Nuclear-Hydrogen System Analysis\n" + "-" * 49 + "\n")
        if annual_metrics_rpt and "greenfield_nuclear_analysis" in annual_metrics_rpt:
            greenfield_results = annual_metrics_rpt["greenfield_nuclear_analysis"]
            f.write(
                "This analysis calculates the economics of building both nuclear plant\n")
            f.write(
                "and hydrogen production system from scratch (greenfield development).\n\n")

            f.write("System Configuration:\n")
            f.write(
                f"  Analysis Type                   : {greenfield_results.get('analysis_type', 'N/A')}\n")
            f.write(
                f"  Nuclear Capacity                : {greenfield_results.get('nuclear_capacity_mw', 0):,.0f} MW\n")
            f.write(
                f"  Project Lifetime                : {greenfield_results.get('project_lifetime_years', 0)} years\n")
            f.write(
                f"  Construction Period             : {greenfield_results.get('construction_period_years', 0)} years\n")
            f.write(
                f"  Discount Rate                   : {greenfield_results.get('discount_rate', 0)*100:.1f}%\n")

            f.write("\nCapital Investment Breakdown:\n")
            nuclear_capex = greenfield_results.get('nuclear_capex_usd', 0)
            h2_capex = greenfield_results.get('hydrogen_system_capex_usd', 0)
            total_capex = greenfield_results.get('total_system_capex_usd', 0)

            if total_capex > 0:
                nuclear_pct = nuclear_capex / total_capex * 100
                h2_pct = h2_capex / total_capex * 100
            else:
                nuclear_pct = h2_pct = 0

            f.write(
                f"  Nuclear Plant CAPEX             : ${nuclear_capex:,.0f} ({nuclear_pct:.1f}%)\n")
            f.write(
                f"  Hydrogen System CAPEX           : ${h2_capex:,.0f} ({h2_pct:.1f}%)\n")
            f.write(
                f"  Total System CAPEX              : ${total_capex:,.0f}\n")
            f.write(
                f"  CAPEX per MW Nuclear            : ${greenfield_results.get('capex_per_mw_nuclear', 0):,.0f}/MW\n")
            f.write(
                f"  CAPEX per kg H2/year            : ${greenfield_results.get('capex_per_kg_h2_annual', 0):,.0f}/kg\n")

            f.write("\nFinancial Results (60-year lifecycle):\n")
            f.write(
                f"  Net Present Value (NPV)         : ${greenfield_results.get('npv_usd', 0):,.0f}\n")

            # Handle NaN values for IRR and Payback
            import math
            irr_value = greenfield_results.get('irr_percent', 0)
            payback_value = greenfield_results.get('payback_period_years', 0)

            if isinstance(irr_value, float) and math.isnan(irr_value):
                f.write(f"  Internal Rate of Return (IRR)   : N/A\n")
            else:
                f.write(
                    f"  Internal Rate of Return (IRR)   : {irr_value:.2f}%\n")

            f.write(
                f"  Return on Investment (ROI)      : {greenfield_results.get('roi_percent', 0):.2f}%\n")

            if isinstance(payback_value, float) and math.isnan(payback_value):
                f.write(f"  Payback Period                  : N/A\n")
            else:
                f.write(
                    f"  Payback Period                  : {payback_value:.1f} years\n")

            f.write(
                f"  LCOH (Integrated System)        : ${greenfield_results.get('lcoh_integrated_usd_per_kg', 0):.3f}/kg\n")

            # **NEW: MACRS Tax Benefits for Greenfield System**
            if greenfield_results.get('macrs_tax_benefits_pv_usd', 0) > 0:
                f.write("\nMACRS Tax Benefits Analysis:\n")
                macrs_benefits_pv = greenfield_results.get(
                    'macrs_tax_benefits_pv_usd', 0)
                total_macrs_depreciation = greenfield_results.get(
                    'total_macrs_depreciation_usd', 0)
                npv_without_macrs = greenfield_results.get(
                    'npv_without_macrs_usd', 0)
                current_npv = greenfield_results.get('npv_usd', 0)

                f.write(
                    f"  Total MACRS Depreciation        : ${total_macrs_depreciation:,.0f}\n")
                f.write(
                    f"  MACRS Tax Benefits (Present Value): ${macrs_benefits_pv:,.0f}\n")
                f.write(
                    f"  NPV with MACRS                  : ${current_npv:,.0f}\n")
                f.write(
                    f"  NPV without MACRS               : ${npv_without_macrs:,.0f}\n")
                f.write(
                    f"  MACRS Improvement in NPV        : ${macrs_benefits_pv:,.0f}\n")

                if total_capex > 0:
                    macrs_as_pct_capex = (
                        macrs_benefits_pv / total_capex) * 100
                    f.write(
                        f"  MACRS Benefits as % of CAPEX    : {macrs_as_pct_capex:.1f}%\n")

                if current_npv != 0:
                    macrs_as_pct_npv = (macrs_benefits_pv / current_npv) * 100
                    f.write(
                        f"  MACRS Benefits as % of NPV      : {macrs_as_pct_npv:.1f}%\n")

            f.write("\nProduction Metrics:\n")
            f.write(
                f"  Annual H2 Production            : {greenfield_results.get('annual_h2_production_kg', 0):,.0f} kg/year\n")
            f.write(
                f"  Annual Nuclear Generation       : {greenfield_results.get('annual_nuclear_generation_mwh', 0):,.0f} MWh/year\n")
            f.write(
                f"  H2 Production per MW Nuclear    : {greenfield_results.get('h2_production_per_mw_nuclear', 0):,.0f} kg/MW/year\n")
            f.write(
                f"  Nuclear Capacity Factor         : {greenfield_results.get('nuclear_capacity_factor', 0)*100:.1f}%\n")
            f.write(
                f"  Electricity to H2 Efficiency    : {greenfield_results.get('electricity_to_h2_efficiency', 0)*100:.1f}%\n")

            # Investment breakdown with replacements
            f.write("\nInvestment Breakdown (60-year lifecycle):\n")
            h2_initial = greenfield_results.get('h2_initial_capex_usd', 0)
            h2_replacement = greenfield_results.get(
                'h2_replacement_capex_usd', 0)
            f.write(
                f"  H2 System Initial CAPEX         : ${h2_initial:,.0f}\n")
            f.write(
                f"  H2 System Replacement CAPEX     : ${h2_replacement:,.0f}\n")
            f.write(
                f"    Electrolyzer Replacements     : {greenfield_results.get('electrolyzer_replacements_count', 0)} times\n")
            f.write(
                f"    H2 Storage Replacements       : {greenfield_results.get('h2_storage_replacements_count', 0)} times\n")
            f.write(
                f"    Battery Replacements          : {greenfield_results.get('battery_replacements_count', 0)} times\n")
            f.write(
                f"  Enhanced Maintenance Factor     : {greenfield_results.get('enhanced_maintenance_factor', 1.0):.1f}x\n")

            f.write("\nAnnual Performance:\n")
            f.write(
                f"  Total Annual Revenue            : ${greenfield_results.get('annual_total_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    H2 Revenue                    : ${greenfield_results.get('annual_h2_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Electricity Revenue           : ${greenfield_results.get('annual_electricity_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Turbine AS Revenue (Real)     : ${greenfield_results.get('annual_turbine_as_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Electrolyzer AS Revenue (Real): ${greenfield_results.get('annual_electrolyzer_as_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    Battery AS Revenue (Real)     : ${greenfield_results.get('annual_battery_as_revenue_usd', 0):,.0f}\n")
            f.write(
                f"    H2 Subsidy Revenue            : ${greenfield_results.get('annual_h2_subsidy_revenue_usd', 0):,.0f}\n")
            f.write(
                f"  Total Annual OPEX               : ${greenfield_results.get('annual_total_opex_usd', 0):,.0f}\n")
            f.write(
                f"  HTE Thermal Opportunity Cost    : ${greenfield_results.get('annual_hte_thermal_cost_usd', 0):,.0f}\n")
            f.write(
                f"  Net Annual Revenue              : ${greenfield_results.get('annual_net_revenue_usd', 0):,.0f}\n")

            f.write("\nLevelized Costs:\n")
            f.write(
                f"  LCOH (Integrated System)        : ${greenfield_results.get('lcoh_integrated_usd_per_kg', 0):.3f}/kg\n")
            f.write(
                f"  Nuclear LCOE                    : ${greenfield_results.get('nuclear_lcoe_usd_per_mwh', 0):.2f}/MWh\n")

            # **NEW: Add LCOS calculation for Greenfield analysis**
            battery_lcos = greenfield_results.get(
                'battery_lcos_usd_per_mwh', 0)
            if battery_lcos > 0:
                f.write(
                    f"  Battery LCOS                    : ${battery_lcos:.2f}/MWh\n")

            f.write("\nCash Flow Summary (Present Value):\n")
            f.write(
                f"  Total Revenue (PV)              : ${greenfield_results.get('total_revenue_pv_usd', 0):,.0f}\n")
            f.write(
                f"  Total Costs (PV)                : ${greenfield_results.get('total_costs_pv_usd', 0):,.0f}\n")
            f.write(
                f"  Net Cash Flow (PV)              : ${greenfield_results.get('net_cash_flow_pv_usd', 0):,.0f}\n")

            f.write("\nKey Insights:\n")
            npv_value = greenfield_results.get('npv_usd', 0)
            if npv_value > 0:
                f.write(
                    "  â€¢ The greenfield nuclear-hydrogen system shows positive NPV\n")
            else:
                f.write(
                    "  â€¢ The greenfield nuclear-hydrogen system shows negative NPV\n")

            irr_value = greenfield_results.get('irr_percent', 0)
            discount_rate = greenfield_results.get('discount_rate', 0.08) * 100
            if not (isinstance(irr_value, float) and math.isnan(irr_value)):
                if irr_value > discount_rate:
                    f.write(
                        "  â€¢ IRR exceeds the discount rate, indicating attractive returns\n")
                else:
                    f.write(
                        "  â€¢ IRR is below the discount rate, indicating marginal returns\n")
            else:
                f.write(
                    "  â€¢ IRR cannot be calculated (negative cash flows throughout project)\n")

            payback = greenfield_results.get('payback_period_years', 999)
            if isinstance(payback, float) and math.isnan(payback):
                f.write(
                    "  â€¢ Payback period cannot be calculated (negative cash flows)\n")
            elif payback < 15:
                f.write(
                    "  â€¢ Relatively short payback period suggests good investment recovery\n")
            elif payback < 25:
                f.write(
                    "  â€¢ Moderate payback period typical for large infrastructure projects\n")
            else:
                f.write(
                    "  â€¢ Long payback period indicates high capital requirements\n")

            # **NEW: Add information about independent accounting method**
            f.write(
                "\nAccounting Method:\n")
            if greenfield_results.get('uses_independent_accounting', False):
                f.write(
                    "  â€¢ Independent accounting method used for LCOE/LCOH/LCOS calculations\n")
                f.write(
                    "  â€¢ LCOE: (nuclear costs + OPEX - turbine AS revenue) / total generation\n")
                f.write(
                    "  â€¢ LCOH: (H2 costs + OPEX + electricity at LCOE + HTE thermal - H2 AS revenue) / H2 production\n")
                if greenfield_results.get('includes_battery_lcos', False):
                    f.write(
                        "  â€¢ LCOS: (battery costs + OPEX) / battery throughput\n")

            if greenfield_results.get('uses_real_as_revenue_data', False):
                f.write(
                    "  â€¢ AS revenues calculated from real system deployment data, not estimates\n")

            if greenfield_results.get('accounts_for_hte_thermal_cost', False):
                f.write(
                    "  â€¢ HTE thermal energy opportunity cost included in LCOH calculation\n")

            f.write(
                "\nNote: This greenfield analysis assumes building both nuclear plant and\n")
            f.write(
                "hydrogen system from zero, with both systems designed for 60-year operation.\n")
            f.write(
                "The analysis includes periodic replacement of H2 system components:\n")
            f.write("â€¢ Electrolyzers replaced every 20 years (2 replacements)\n")
            f.write(
                "â€¢ H2 storage systems replaced every 30 years (1 replacement)\n")
            f.write("â€¢ Batteries replaced every 15 years (3 replacements)\n")
            f.write(
                "â€¢ Enhanced maintenance costs (+20%) for extended lifecycle operation\n")
            f.write(
                "This provides a comprehensive view of long-term integrated system economics.\n\n")

        f.write("7.3 Lifecycle Comparison Analysis\n" + "-" * 34 + "\n")
        if annual_metrics_rpt and "lifecycle_comparison_analysis" in annual_metrics_rpt:
            comparison_results = annual_metrics_rpt["lifecycle_comparison_analysis"]
            f.write(
                "This section compares the financial performance of 60-year vs 80-year project lifecycles\n")
            f.write(
                "to evaluate the impact of extending project duration on investment returns.\n\n")

            # Extract results for both scenarios
            lifecycle_60 = comparison_results.get("60_year_results", {})
            lifecycle_80 = comparison_results.get("80_year_results", {})

            f.write("Financial Performance Comparison:\n")
            f.write(
                f"{'Metric':<35} {'60-Year':<20} {'80-Year':<20} {'Difference':<15}\n")
            f.write("-" * 90 + "\n")

            # Key metrics comparison with LCOE and LCOS
            metrics_to_compare = [
                ('NPV (USD)', 'npv_usd', 'npv_usd'),
                ('IRR (%)', 'irr_percent', 'irr_percent'),
                ('ROI (%)', 'roi_percent', 'roi_percent'),
                ('Payback (Years)', 'payback_period_years', 'payback_period_years'),
                ('LCOH (USD/kg)', 'lcoh_integrated_usd_per_kg',
                 'lcoh_integrated_usd_per_kg'),
                ('LCOE (USD/MWh)', 'nuclear_lcoe_usd_per_mwh',
                 'nuclear_lcoe_usd_per_mwh'),
                ('Battery LCOS (USD/MWh)', 'battery_lcos_usd_per_mwh',
                 'battery_lcos_usd_per_mwh')
            ]

            for metric_name, key_60, key_80 in metrics_to_compare:
                val_60 = lifecycle_60.get(key_60, 0)
                val_80 = lifecycle_80.get(key_80, 0)

                # Handle NaN values
                import math
                val_60_is_nan = isinstance(
                    val_60, float) and math.isnan(val_60)
                val_80_is_nan = isinstance(
                    val_80, float) and math.isnan(val_80)

                if val_60_is_nan or val_80_is_nan:
                    # If either value is NaN, display N/A for all
                    f.write(
                        f"{metric_name:<35} {'N/A':>20} {'N/A':>20} {'N/A':>15}\n")
                else:
                    # Both values are valid, calculate difference and format
                    difference = val_80 - val_60

                    if 'USD' in metric_name:
                        f.write(
                            f"{metric_name:<35} ${val_60:>15,.0f} ${val_80:>15,.0f} ${difference:>12,.0f}\n")
                    elif '%' in metric_name or 'ROI' in metric_name:
                        f.write(
                            f"{metric_name:<35} {val_60:>16.2f}% {val_80:>16.2f}% {difference:>13.2f}%\n")
                    elif 'Years' in metric_name:
                        f.write(
                            f"{metric_name:<35} {val_60:>19.1f} {val_80:>19.1f} {difference:>16.1f}\n")
                    else:
                        f.write(
                            f"{metric_name:<35} {val_60:>19.3f} {val_80:>19.3f} {difference:>16.3f}\n")

            # **NEW: Add LCOS comparison for battery systems**
            if annual_metrics_rpt and annual_metrics_rpt.get("Battery_Capacity_MWh", 0) > 0:
                f.write("\nBattery Storage Cost Analysis:\n")

                # Get LCOS from both scenarios with improved error handling
                lcos_60 = lifecycle_60.get('battery_lcos_usd_per_mwh', 0)
                lcos_80 = lifecycle_80.get('battery_lcos_usd_per_mwh', 0)

                # Check if battery systems are present in the scenarios
                battery_capacity_60 = lifecycle_60.get(
                    'battery_capacity_mwh', 0)
                battery_capacity_80 = lifecycle_80.get(
                    'battery_capacity_mwh', 0)

                # Only validate LCOS if battery systems are present
                if battery_capacity_60 > 0 and lcos_60 <= 0:
                    logger.warning(
                        f"60-year LCOS is zero or negative despite battery presence: {lcos_60}")
                if battery_capacity_80 > 0 and lcos_80 <= 0:
                    logger.warning(
                        f"80-year LCOS is zero or negative despite battery presence: {lcos_80}")

                lcos_difference = lcos_80 - lcos_60

                # Display LCOS values if either scenario has valid LCOS or battery systems
                if (lcos_60 > 0 or lcos_80 > 0) or (battery_capacity_60 > 0 or battery_capacity_80 > 0):
                    if lcos_60 > 0 and lcos_80 > 0:
                        f.write(
                            f"{'LCOS (USD/MWh)':<35} ${lcos_60:>15.2f} ${lcos_80:>15.2f} ${lcos_difference:>12.2f}\n")
                        logger.info(
                            f"LCOS comparison: 60yr=${lcos_60:.2f}/MWh, 80yr=${lcos_80:.2f}/MWh, diff=${lcos_difference:.2f}/MWh")
                    else:
                        # Show individual values even if one is zero
                        lcos_60_str = f"${lcos_60:.2f}" if lcos_60 > 0 else "N/A"
                        lcos_80_str = f"${lcos_80:.2f}" if lcos_80 > 0 else "N/A"
                        lcos_diff_str = f"${lcos_difference:.2f}" if lcos_60 > 0 and lcos_80 > 0 else "N/A"
                        f.write(
                            f"{'LCOS (USD/MWh)':<35} {lcos_60_str:>20} {lcos_80_str:>20} {lcos_diff_str:>15}\n")
                        logger.info(
                            f"LCOS comparison (partial data): 60yr={lcos_60_str}/MWh, 80yr={lcos_80_str}/MWh")
                else:
                    f.write(
                        f"{'LCOS (USD/MWh)':<35} {'N/A':>20} {'N/A':>20} {'N/A':>15}\n")
                    logger.debug(
                        "LCOS not applicable - no battery systems in lifecycle scenarios")

        f.write("7.4 Federal Tax Incentive Analysis\n" + "-" * 35 + "\n")
        if annual_metrics_rpt and "comprehensive_tax_incentive_analysis" in annual_metrics_rpt:
            tax_incentive_results = annual_metrics_rpt["comprehensive_tax_incentive_analysis"]
            f.write("Comprehensive analysis of federal tax incentive scenarios for\n")
            f.write("greenfield nuclear-hydrogen integrated systems including:\n")
            f.write("- Baseline scenario (no tax incentives)\n")
            f.write("- 45Y Production Tax Credit scenario ($30/MWh for 10 years)\n")
            f.write(
                "- 48E Investment Tax Credit scenario (50% of qualified CAPEX)\n\n")

            # Check if analysis was successful or failed
            if tax_incentive_results is None or tax_incentive_results.get("analysis_status") == "failed":
                f.write(
                    "STATUS: Tax incentive analysis was not completed successfully.\n")
                if tax_incentive_results and "error_message" in tax_incentive_results:
                    f.write(
                        f"Error: {tax_incentive_results['error_message']}\n")
                f.write("\nNote: This analysis requires:\n")
                f.write(
                    "- Valid hourly optimization results with electricity generation data\n")
                f.write("- Proper cash flow data from baseline analysis\n")
                f.write("- Compatible CAPEX breakdown for MACRS depreciation\n\n")

                # Still show baseline results if available
                if tax_incentive_results and "financial_comparison" in tax_incentive_results:
                    baseline_npv = tax_incentive_results["financial_comparison"]["baseline_npv"]
                    f.write(f"Baseline System NPV: ${baseline_npv:,.0f}\n")
                    f.write("(Tax incentive scenarios could not be calculated)\n\n")
            else:
                # Analysis was successful - show full results
                financial_comparison = tax_incentive_results.get(
                    "financial_comparison", {})

                f.write("FINANCIAL COMPARISON SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Baseline NPV (No Incentives):     ${financial_comparison.get('baseline_npv', 0):>15,.0f}\n")
                f.write(
                    f"45Y PTC NPV:                      ${financial_comparison.get('ptc_npv', 0):>15,.0f}\n")
                f.write(
                    f"48E ITC NPV:                      ${financial_comparison.get('itc_npv', 0):>15,.0f}\n\n")

                f.write("NPV IMPROVEMENTS:\n")
                f.write("-" * 40 + "\n")
                ptc_improvement = financial_comparison.get(
                    'ptc_npv_improvement', 0)
                itc_improvement = financial_comparison.get(
                    'itc_npv_improvement', 0)
                f.write(
                    f"45Y PTC Improvement:              ${ptc_improvement:>15,.0f}\n")
                f.write(
                    f"48E ITC Improvement:              ${itc_improvement:>15,.0f}\n\n")

                best_scenario = financial_comparison.get(
                    'best_scenario', 'baseline')
                scenario_names = {
                    'baseline': 'Baseline (No Incentives)',
                    'ptc': '45Y Production Tax Credit',
                    'itc': '48E Investment Tax Credit'
                }
                f.write(
                    f"Recommended Scenario: {scenario_names.get(best_scenario, best_scenario)}\n\n")

                # Add detailed tax incentive breakdown if available
                if "tax_incentive_analysis" in tax_incentive_results and tax_incentive_results["tax_incentive_analysis"]:
                    tax_analysis = tax_incentive_results["tax_incentive_analysis"]

                    # PTC Details
                    if "scenarios" in tax_analysis and "ptc" in tax_analysis["scenarios"]:
                        ptc_scenario = tax_analysis["scenarios"]["ptc"]
                        if "analysis" in ptc_scenario and "tax_benefits" in ptc_scenario["analysis"]:
                            ptc_benefits = ptc_scenario["analysis"]["tax_benefits"].get(
                                "ptc", {})
                            f.write("45Y PRODUCTION TAX CREDIT DETAILS:\n")
                            f.write("-" * 40 + "\n")
                            f.write(
                                f"Annual Generation:                {ptc_benefits.get('annual_generation_mwh', 0):>10,.0f} MWh\n")
                            f.write(
                                f"PTC Rate:                         ${ptc_benefits.get('ptc_rate_usd_per_mwh', 30):>10.0f}/MWh\n")
                            f.write(
                                f"Annual PTC Benefit:               ${ptc_benefits.get('annual_ptc_benefit_usd', 0):>15,.0f}\n")
                            f.write(
                                f"Total PTC Value (10 years):       ${ptc_benefits.get('total_ptc_value_usd', 0):>15,.0f}\n")
                            f.write(
                                f"PTC NPV:                          ${ptc_benefits.get('total_ptc_npv_usd', 0):>15,.0f}\n\n")

                    # ITC Details
                    if "scenarios" in tax_analysis and "itc" in tax_analysis["scenarios"]:
                        itc_scenario = tax_analysis["scenarios"]["itc"]
                        if "analysis" in itc_scenario and "tax_benefits" in itc_scenario["analysis"]:
                            itc_benefits = itc_scenario["analysis"]["tax_benefits"].get(
                                "itc", {})
                            f.write("48E INVESTMENT TAX CREDIT DETAILS:\n")
                            f.write("-" * 40 + "\n")
                            # Get total CAPEX from available data
                            total_project_capex = annual_metrics_rpt.get(
                                "total_capex", 0) if annual_metrics_rpt else 0
                            f.write(
                                f"Total Project CAPEX:              ${total_project_capex:>15,.0f}\n")
                            f.write(
                                f"Qualified CAPEX:                  ${itc_benefits.get('total_qualified_capex_usd', 0):>15,.0f}\n")
                            f.write(
                                f"ITC Rate:                         {itc_benefits.get('itc_rate', 0.5):>10.1%}\n")
                            f.write(
                                f"ITC Credit Amount:                ${itc_benefits.get('itc_credit_amount_usd', 0):>15,.0f}\n")
                            f.write(
                                f"Depreciation Basis Reduction:     ${itc_benefits.get('depreciation_basis_reduction_usd', 0):>15,.0f}\n")
                            if "net_itc_benefit" in itc_scenario["analysis"]:
                                f.write(
                                    f"Net ITC Benefit:                  ${itc_scenario['analysis']['net_itc_benefit']:>15,.0f}\n")
                            f.write("\n")

                f.write("FINANCIAL METRICS COMPARISON:\n")
                f.write("-" * 60 + "\n")
                f.write(
                    f"{'Metric':<25} {'Baseline':<15} {'45Y PTC':<15} {'48E ITC':<15}\n")
                f.write("-" * 60 + "\n")

                if "tax_incentive_analysis" in tax_incentive_results and tax_incentive_results["tax_incentive_analysis"]:
                    scenarios = tax_incentive_results["tax_incentive_analysis"].get(
                        "scenarios", {})

                    # NPV comparison
                    baseline_npv = scenarios.get("baseline", {}).get(
                        "financial_metrics", {}).get("npv_usd", 0)
                    ptc_npv = scenarios.get("ptc", {}).get(
                        "financial_metrics", {}).get("npv_usd", 0)
                    itc_npv = scenarios.get("itc", {}).get(
                        "financial_metrics", {}).get("npv_usd", 0)
                    f.write(
                        f"{'NPV ($M)':<25} ${baseline_npv/1e6:<14.1f} ${ptc_npv/1e6:<14.1f} ${itc_npv/1e6:<14.1f}\n")

                    # IRR comparison
                    baseline_irr = scenarios.get("baseline", {}).get(
                        "financial_metrics", {}).get("irr_percent")
                    ptc_irr = scenarios.get("ptc", {}).get(
                        "financial_metrics", {}).get("irr_percent")
                    itc_irr = scenarios.get("itc", {}).get(
                        "financial_metrics", {}).get("irr_percent")

                    if baseline_irr is not None:
                        f.write(
                            f"{'IRR (%)':<25} {baseline_irr:<14.1f} {ptc_irr if ptc_irr is not None else 'N/A':<14} {itc_irr if itc_irr is not None else 'N/A':<14}\n")

                    # Payback period comparison
                    baseline_payback = scenarios.get("baseline", {}).get(
                        "financial_metrics", {}).get("payback_period_years")
                    ptc_payback = scenarios.get("ptc", {}).get(
                        "financial_metrics", {}).get("payback_period_years")
                    itc_payback = scenarios.get("itc", {}).get(
                        "financial_metrics", {}).get("payback_period_years")

                    if baseline_payback is not None and not np.isnan(baseline_payback):
                        f.write(f"{'Payback (years)':<25} {baseline_payback:<14.1f} {ptc_payback if ptc_payback is not None and not np.isnan(ptc_payback) else 'N/A':<14} {itc_payback if itc_payback is not None and not np.isnan(itc_payback) else 'N/A':<14}\n")

                f.write("\n")
                f.write("ANALYSIS METHODOLOGY:\n")
                f.write(
                    "- Independent system accounting for nuclear and hydrogen components\n")
                f.write(
                    "- Actual hourly generation data from optimization results\n")
                f.write("- MACRS depreciation (15-year nuclear, 7-year hydrogen)\n")
                f.write(
                    "- 50% depreciation basis reduction for ITC per IRS requirements\n")
                f.write(
                    "- Present value calculations using system discount rate\n\n")
        else:
            # Greenfield analysis exists but no tax incentive analysis
            if annual_metrics_rpt and ("greenfield_nuclear_analysis" in annual_metrics_rpt or
                                       "comprehensive_tax_incentive_analysis" in annual_metrics_rpt):
                f.write(
                    "STATUS: Tax incentive analysis was not performed for this configuration.\n\n")
                f.write(
                    "Note: Tax incentive analysis is available for greenfield nuclear-hydrogen\n")
                f.write(
                    "systems when both nuclear and hydrogen components are included.\n\n")

        # ========================================
        # 8. TECHNICAL APPENDIX
        # ========================================
        f.write("8. Technical Appendix\n" + "=" * 21 + "\n\n")

        f.write("8.1 Cost Assumptions and Parameters\n" + "-" * 36 + "\n")
        f.write("  CAPEX Components (Base Cost for Reference Size):\n")
        capex_items = {}
        for comp, det in sorted(capex_data.items()):
            ref_cap = det.get('reference_total_capacity_mw', 0)
            base_cost = det.get('total_base_cost_for_ref_size', 0)
            learning_rate = det.get('learning_rate_decimal', 0) * 100
            capex_items[comp] = f"${base_cost:,.0f} (Ref: {ref_cap}MW, LR: {learning_rate}%)"

        f.write(format_aligned_section(
            capex_items, min_width=50, indent="    "))

        f.write("\n  O&M Components (Annual Base):\n")
        om_items = {}
        for comp, det in sorted(om_data.items()):
            if comp == "Fixed_OM_Battery":
                base_cost_mw = det.get('base_cost_per_mw_year', 0)
                base_cost_mwh = det.get('base_cost_per_mwh_year', 0)
                inflation_rate = det.get('inflation_rate', 0) * 100
                om_items[
                    comp] = f"${base_cost_mw:,.2f}/MW/yr + ${base_cost_mwh:,.2f}/MWh/yr (Inflation: {inflation_rate:.1f}%)"
            else:
                base_cost = det.get('base_cost', 0)
                inflation_rate = det.get('inflation_rate', 0) * 100
                om_items[comp] = f"${base_cost:,.0f} (Inflation: {inflation_rate:.1f}%)"

        f.write(format_aligned_section(om_items, min_width=50, indent="    "))

        f.write("\n  Major Replacements:\n")
        replacement_items = {}
        for comp, det in sorted(replacement_data.items()):
            if 'cost_percent_initial_capex' in det:
                cost_info = f"{det.get('cost_percent_initial_capex', 0)*100:.2f}% of Initial CAPEX"
            else:
                cost_info = f"${det.get('cost', 0):,.0f}"
            years = det.get('years', [])
            replacement_items[comp] = f"Cost: {cost_info} (Years: {years})"

        f.write(format_aligned_section(
            replacement_items, min_width=50, indent="    "))

        f.write("\n8.2 Methodology Notes\n" + "-" * 18 + "\n")
        f.write("  • Financial calculations use discounted cash flow analysis\n")
        f.write("  • LCOH calculations include all system costs and revenues\n")
        f.write("  • Capacity factors based on hourly optimization results\n")
        f.write("  • Tax incentive analysis uses independent system accounting\n")
        f.write("  • Nuclear costs standardized across all analysis sections\n")
        f.write("  • Battery LCOS includes full lifecycle costs and throughput\n")
        f.write("  • HTE thermal opportunity costs included where applicable\n\n")

        f.write("\nReport generated successfully.\n")

    logger.info(f"TEA report saved to {output_file_path}")
