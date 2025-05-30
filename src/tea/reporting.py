"""
Reporting and plotting coordination functions for the TEA module.
Main entry point for generating reports and coordinating visualizations.
"""

from src.tea.visualization import (
    get_component_color,
    create_cash_flow_plots,
    create_capex_breakdown_plots,
    create_lcoh_comprehensive_dashboard,
    create_lcoh_benchmarking_analysis
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
    construction_p: int,
    incremental_metrics_data: dict | None = None,
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

    # Generate cash flow plots
    create_cash_flow_plots(
        cash_flows_data,
        plot_dir,
        construction_p,
        incremental_metrics_data
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
    project_lt_rpt: int,
    construction_p_rpt: int,
    discount_rt_rpt: float,
    tax_rt_rpt: float,
    incremental_metrics_rpt: dict | None = None,
):
    """Generate comprehensive TEA report exactly matching main tea.py"""
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
        f.write(
            f"Technical Economic Analysis Report - {report_title}\n"
            + "=" * (30 + len(report_title))
            + "\n\n"
        )
        f.write("1. Project Configuration\n" + "-" * 25 + "\n")

        # **ENHANCEMENT: Use consistent alignment formatting**
        config_items = {
            "ISO Region": target_iso_rpt,
            "Project Lifetime": f"{project_lt_rpt} years",
            "Construction Period": f"{construction_p_rpt} years",
            "Discount Rate": f"{discount_rt_rpt*100:.2f}%",
            "Corporate Tax Rate": f"{tax_rt_rpt*100:.1f}%",
        }

        # **ENHANCEMENT: Add plant-specific technical parameters**
        if annual_metrics_rpt:
            # Add Turbine Capacity (moved from System Capacities section)
            turbine_capacity = annual_metrics_rpt.get("Turbine_Capacity_MW", 0)
            if turbine_capacity > 0:
                config_items["Turbine Capacity"] = f"{turbine_capacity:,.2f} MW"

            # Add Thermal Capacity if available
            thermal_capacity = annual_metrics_rpt.get(
                "thermal_capacity_mwt", 0)
            if thermal_capacity == 0:
                thermal_capacity = annual_metrics_rpt.get(
                    "Thermal_Capacity_MWt", 0)
            if thermal_capacity > 0:
                config_items["Thermal Capacity"] = f"{thermal_capacity:,.2f} MWt"

            # Add Thermal Efficiency if available
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

        # Add CAPEX breakdown section
        f.write("2. Capital Expenditure (CAPEX) Breakdown\n" + "-" * 42 + "\n")
        if annual_metrics_rpt and "capex_breakdown" in annual_metrics_rpt:
            capex_breakdown = annual_metrics_rpt["capex_breakdown"]
            total_capex = annual_metrics_rpt.get(
                "total_capex", sum(capex_breakdown.values())
            )

            # Sort by values in descending order
            for component, cost in sorted(
                capex_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                if cost > 0:
                    percentage = (cost / total_capex *
                                  100) if total_capex > 0 else 0
                    f.write(
                        f"  {component:<30}             : ${cost:,.0f} ({percentage:.1f}%)\n")

            f.write(
                f"  \n  Total CAPEX                     : ${total_capex:,.0f}\n\n")
        else:
            f.write("  No CAPEX breakdown data available.\n\n")

        # Add a new section for actual capacity values used in calculations
        f.write("3. Optimization Results - System Capacities\n" + "-" * 45 + "\n")
        if annual_metrics_rpt:
            # Show the actual capacity values that were used for calculations
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

            # Add hydrogen constant sales rate if available (for optimal storage sizing mode)
            h2_constant_sales_rate = annual_metrics_rpt.get(
                "Optimal_H2_Constant_Sales_Rate_kg_hr", 0)
            if h2_constant_sales_rate == 0:
                h2_constant_sales_rate = annual_metrics_rpt.get(
                    "H2_Constant_Sales_Rate_kg_hr", 0)

            if h2_constant_sales_rate > 0:
                capacity_items["Optimal H2 Constant Sales Rate"] = f"{h2_constant_sales_rate:,.2f} kg/hr"

                # Calculate and show daily/annual production rates
                daily_sales = h2_constant_sales_rate * 24
                annual_sales = daily_sales * 365
                capacity_items["Optimal H2 Daily Sales Rate"] = f"{daily_sales:,.2f} kg/day"
                capacity_items["Optimal H2 Annual Sales Rate"] = f"{annual_sales:,.0f} kg/year"

            f.write(format_aligned_section(capacity_items, min_width=35))
            f.write("\n")

        # 4. Representative Annual Performance (from Optimization)
        f.write(
            "4. Representative Annual Performance (from Optimization)\n"
            + "-" * 58
            + "\n"
        )
        if annual_metrics_rpt:
            metrics_to_skip = [
                "capex_breakdown",
                "total_capex",
                "Electrolyzer_Capacity_MW",
                "H2_Storage_Capacity_kg",
                "Battery_Capacity_MWh",
                "Battery_Power_MW",
                "Turbine_Capacity_MW",
                "lcoh_breakdown_analysis",
                # Skip AS-specific metrics here as they will be shown in the AS section
                "AS_Revenue_Total",
                "AS_Revenue_Average_Hourly",
                "AS_Revenue_Maximum_Hourly",
                "AS_Revenue_Utilization_Rate",
                "AS_Revenue_per_MW_Electrolyzer",
                "AS_Revenue_per_MW_Battery",
                "AS_Total_Bid_Services",
                "AS_Total_Max_Bid_Capacity_MW",
                "AS_Bid_Utilization_vs_Electrolyzer",
                "AS_Total_Deployed_Energy_MWh",
                # Skip battery charging metrics here as they will be shown in battery section
                "Annual_Battery_Charge_MWh",
                "Annual_Battery_Charge_From_Grid_MWh",
                "Annual_Battery_Charge_From_NPP_MWh",
                # Skip complex data structures that are not user-friendly for display
                "greenfield_nuclear_analysis",
                "lifecycle_comparison_analysis",
                "annual_fixed_om_costs",
                "annual_other_replacement_costs",
                "annual_stack_replacement_costs",
                "electrolyzer_capex",
                "thermal_capacity_mwt",
                "thermal_efficiency",
            ]

            for k, v in sorted(annual_metrics_rpt.items()):
                # Skip AS-specific detailed metrics and other excluded items
                if any(x in k for x in ["AS_Max_Bid_", "AS_Avg_Bid_", "AS_Total_Deployed_", "AS_Avg_Deployed_", "AS_Deployment_Efficiency_"]) or k in metrics_to_skip:
                    continue

                # **ENHANCEMENT: Skip complex data structures that are not suitable for simple display**
                if isinstance(v, (dict, list)):
                    continue

                # Format metric names properly
                display_name = k.replace('_', ' ').replace('Cf ', 'Capacity Factor ').replace('Soc ', 'SOC ').replace('Vom ', 'VOM ').replace(
                    'Opex ', 'OPEX ').replace('Capex', 'CAPEX').replace('Mw', 'MW').replace('Mwh', 'MWh').replace('Usd', 'USD').replace('As ', 'AS ')

                if isinstance(v, (int, float)) and not pd.isna(v):
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
                else:
                    # **ENHANCEMENT: Better handling of non-numeric values**
                    if isinstance(v, str) and len(v) > 100:
                        # Skip very long strings that are likely not user-friendly
                        continue
                    formatted_value = str(v)

                f.write(f"  {display_name:<45}: {formatted_value}\n")
        else:
            f.write("  No annual metrics data available.\n")

        # **NEW: Battery Performance and Charging Analysis Section**
        battery_capacity = annual_metrics_rpt.get(
            "Battery_Capacity_MWh", 0) if annual_metrics_rpt else 0
        if battery_capacity > 0:
            f.write(
                "\n4.1. Battery Performance and Charging Analysis\n" + "-" * 47 + "\n")

            # Battery charging breakdown
            total_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_MWh", 0)
            grid_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_Grid_MWh", 0)
            npp_charge = annual_metrics_rpt.get(
                "Annual_Battery_Charge_From_NPP_MWh", 0)

            f.write(f"  Battery Charging Electricity Consumption:\n")
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

            f.write(f"\n  Battery Utilization:\n")
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

        # 5. Ancillary Services Performance
        f.write("\n5. Ancillary Services Performance\n" + "-" * 35 + "\n")
        as_revenue = annual_metrics_rpt.get(
            "AS_Revenue", 0) if annual_metrics_rpt else 0

        as_items = {
            "Total Ancillary Services Revenue": f"${as_revenue:,.2f}"
        }

        if annual_metrics_rpt:
            # Revenue Mix Analysis
            energy_revenue = annual_metrics_rpt.get("Energy_Revenue", 0)
            h2_sales_revenue = annual_metrics_rpt.get("H2_Sales_Revenue", 0)
            h2_subsidy_revenue = annual_metrics_rpt.get(
                "H2_Subsidy_Revenue", 0)
            total_revenue = annual_metrics_rpt.get("Annual_Revenue", 0)

            f.write(format_aligned_section(as_items, min_width=45))

            f.write("\n  Revenue Mix:\n")
            revenue_mix_items = {}
            if total_revenue > 0:
                energy_pct = (energy_revenue / total_revenue) * 100
                h2_sales_pct = (h2_sales_revenue / total_revenue) * 100
                h2_subsidy_pct = (h2_subsidy_revenue / total_revenue) * 100
                as_pct = (as_revenue / total_revenue) * 100

                revenue_mix_items["Energy Revenue"] = f"${energy_revenue:,.2f} ({energy_pct:.1f}%)"
                revenue_mix_items[
                    "H2 Sales Revenue"] = f"${h2_sales_revenue:,.2f} ({h2_sales_pct:.1f}%)"
                revenue_mix_items[
                    "H2 Subsidy Revenue"] = f"${h2_subsidy_revenue:,.2f} ({h2_subsidy_pct:.1f}%)"
                revenue_mix_items[
                    "Ancillary Services Revenue"] = f"${as_revenue:,.2f} ({as_pct:.1f}%)"

                f.write(format_aligned_section(
                    revenue_mix_items, min_width=45, indent="    "))

                f.write(
                    f"\n  AS Revenue as % of Total Revenue            : {as_pct:.2f}%\n")

            # System utilization that affects AS capability
            f.write("\n  System Utilization (affects AS capability):\n")
            utilization_items = {}
            electrolyzer_cf = annual_metrics_rpt.get(
                "Electrolyzer_CF_percent", 0)
            turbine_cf = annual_metrics_rpt.get("Turbine_CF_percent", 0)
            battery_soc = annual_metrics_rpt.get("Battery_SOC_percent", 0)

            utilization_items["Electrolyzer Capacity Factor"] = f"{electrolyzer_cf:.2f}%"
            utilization_items["Turbine Capacity Factor"] = f"{turbine_cf:.2f}%"
            utilization_items["Battery SOC"] = f"{battery_soc:.2f}%"

            f.write(format_aligned_section(
                utilization_items, min_width=45, indent="    "))
        else:
            f.write(format_aligned_section(as_items, min_width=45))

        # 6. Lifecycle Financial Metrics (Total System)
        f.write("\n6. Lifecycle Financial Metrics (Total System)\n" + "-" * 46 + "\n")
        if financial_metrics_rpt:
            # **FIXED: Avoid duplicate ROI reporting**
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

        # **NEW: Detailed LCOH Analysis Section**
        if annual_metrics_rpt and "lcoh_breakdown_analysis" in annual_metrics_rpt:
            f.write(
                "\n6.1. Detailed Levelized Cost of Hydrogen (LCOH) Analysis\n" + "-" * 58 + "\n")

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
                if cost > 0.001:  # Only show significant components
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
                comparison = "✓ Below" if total_lcoh < bench_value else "✗ Above"
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

        # **ENHANCEMENT: Always show Incremental Analysis section with improved format**
        # 7. Incremental Financial Metrics (always show, even if disabled/unavailable)
        f.write(
            "\n7. Incremental Financial Metrics (H2/Battery System vs. Baseline)\n" + "-" * 68 + "\n")

        if incremental_metrics_rpt:
            # **CORE INCREMENTAL FINANCIAL METRICS**
            f.write("  Incremental Financial Results:\n")

            # Primary financial metrics
            core_metrics = {
                "NPV_USD": ("NPV (USD)", "currency"),
                "IRR_percent": ("IRR (%)", "percent"),
                "Payback_Period_Years": ("Payback Period (Years)", "number"),
                "Total_Incremental_CAPEX_Learned_USD": ("Total Incremental CAPEX (USD)", "currency"),
                "Annual_Electricity_Opportunity_Cost_USD": ("Annual Electricity Opportunity Cost (USD)", "currency"),
            }

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
                        f.write(f"    {metric_label:<45}: {formatted_value}\n")

            # Calculate and show incremental ROI
            inc_npv = incremental_metrics_rpt.get("NPV_USD")
            inc_total_capex = incremental_metrics_rpt.get(
                "Total_Incremental_CAPEX_Learned_USD")
            if inc_npv is not None and inc_total_capex and inc_total_capex > 0:
                inc_roi = inc_npv / inc_total_capex
                f.write(f"    {'Incremental ROI':<45}: {inc_roi:.4f}\n")

            # **ENHANCED: Show comprehensive baseline analysis**
            f.write("\n  Baseline Nuclear Plant Analysis:\n")

            # Calculate baseline metrics if available
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

                # Calculate baseline generation and revenue
                if turbine_capacity > 0 and turbine_cf > 0:
                    baseline_annual_generation = turbine_capacity * \
                        8760 * (turbine_cf / 100)
                    calculated_baseline_revenue = baseline_annual_generation * avg_lmp
                    f.write(
                        f"    Annual Baseline Revenue (Electricity Sales): ${calculated_baseline_revenue:,.2f}\n")
                else:
                    f.write(
                        f"    Annual Baseline Revenue (Electricity Sales): ${baseline_revenue:,.2f}\n")

                # Estimate baseline OPEX (mainly VOM)
                baseline_opex = annual_metrics_rpt.get(
                    "VOM_Turbine_Cost", 0) if annual_metrics_rpt else 0
                baseline_profit = calculated_baseline_revenue - \
                    baseline_opex if 'calculated_baseline_revenue' in locals() else baseline_revenue - \
                    baseline_opex
                baseline_margin = (baseline_profit / calculated_baseline_revenue *
                                   100) if 'calculated_baseline_revenue' in locals() and calculated_baseline_revenue > 0 else 0

                f.write(
                    f"    Annual Baseline OPEX (Turbine VOM): ${baseline_opex:,.2f}\n")
                f.write(
                    f"    Annual Baseline Profit: ${baseline_profit:,.2f}\n")
                f.write(
                    f"    Baseline Profit Margin: {baseline_margin:.1f}%\n")
            else:
                f.write("    Baseline analysis data not available\n")

            # **ENHANCED: Battery charging cost analysis**
            if annual_metrics_rpt and annual_metrics_rpt.get("Battery_Capacity_MWh", 0) > 0:
                f.write("\n  Battery Charging Cost Analysis:\n")

                grid_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_Grid_MWh", 0)
                npp_charge = annual_metrics_rpt.get(
                    "Annual_Battery_Charge_From_NPP_MWh", 0)
                avg_price = annual_metrics_rpt.get(
                    "Avg_Electricity_Price_USD_per_MWh", 0)

                direct_cost = grid_charge * avg_price
                opportunity_cost = npp_charge * avg_price
                total_charging_cost = direct_cost + opportunity_cost

                f.write(
                    f"    Direct Operating Cost (Grid Charging): ${direct_cost:,.2f}/year\n")
                f.write(
                    f"    Opportunity Cost (NPP Charging): ${opportunity_cost:,.2f}/year\n")
                f.write(
                    f"    Total Battery Charging Cost: ${total_charging_cost:,.2f}/year\n")

                if total_charging_cost > 0:
                    direct_pct = (direct_cost / total_charging_cost) * 100
                    opp_pct = (opportunity_cost / total_charging_cost) * 100
                    f.write(
                        f"    Cost Breakdown: {direct_pct:.1f}% Direct, {opp_pct:.1f}% Opportunity\n")

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

                f.write(f"    Total Annual Revenue: ${total_revenue:,.2f}\n")
                f.write(f"      Energy Sales: ${energy_revenue:,.2f}\n")
                f.write(f"      H2 Sales: ${h2_revenue:,.2f}\n")
                f.write(f"      Ancillary Services: ${as_revenue:,.2f}\n")

        # 8. Cost Assumptions
        f.write("\n8. Cost Assumptions\n" + "-" * 19 + "\n")
        f.write("  CAPEX Components (Base Cost for Reference Size):\n")

        capex_items = {}
        for comp, det in sorted(capex_data.items()):
            ref_cap = det.get('reference_total_capacity_mw', 0)
            base_cost = det.get('total_base_cost_for_ref_size', 0)
            learning_rate = det.get('learning_rate_decimal', 0) * 100
            payment_schedule = det.get('payment_schedule_years', {})
            capex_items[comp] = f"${base_cost:,.0f} (Ref Cap: {ref_cap}, LR: {learning_rate}%, Pay Sched: {payment_schedule})"

        f.write(format_aligned_section(
            capex_items, min_width=50, indent="    "))

        f.write("    \n  O&M Components (Annual Base):\n")
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

        f.write("    \n  Major Replacements:\n")
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

        # 9. Greenfield Nuclear-Hydrogen System Analysis (if available)
        if annual_metrics_rpt and "greenfield_nuclear_analysis" in annual_metrics_rpt:
            greenfield_results = annual_metrics_rpt["greenfield_nuclear_analysis"]
            f.write("\n9. Greenfield Nuclear-Hydrogen System\n" + "-" * 38 + "\n")
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
                    "  • The greenfield nuclear-hydrogen system shows positive NPV\n")
            else:
                f.write(
                    "  • The greenfield nuclear-hydrogen system shows negative NPV\n")

            irr_value = greenfield_results.get('irr_percent', 0)
            discount_rate = greenfield_results.get('discount_rate', 0.08) * 100
            if not (isinstance(irr_value, float) and math.isnan(irr_value)):
                if irr_value > discount_rate:
                    f.write(
                        "  • IRR exceeds the discount rate, indicating attractive returns\n")
                else:
                    f.write(
                        "  • IRR is below the discount rate, indicating marginal returns\n")
            else:
                f.write(
                    "  • IRR cannot be calculated (negative cash flows throughout project)\n")

            payback = greenfield_results.get('payback_period_years', 999)
            if isinstance(payback, float) and math.isnan(payback):
                f.write(
                    "  • Payback period cannot be calculated (negative cash flows)\n")
            elif payback < 15:
                f.write(
                    "  • Relatively short payback period suggests good investment recovery\n")
            elif payback < 25:
                f.write(
                    "  • Moderate payback period typical for large infrastructure projects\n")
            else:
                f.write(
                    "  • Long payback period indicates high capital requirements\n")

            # **NEW: Add information about independent accounting method**
            f.write(
                "\nAccounting Method:\n")
            if greenfield_results.get('uses_independent_accounting', False):
                f.write(
                    "  • Independent accounting method used for LCOE/LCOH/LCOS calculations\n")
                f.write(
                    "  • LCOE: (nuclear costs + OPEX - turbine AS revenue) / total generation\n")
                f.write(
                    "  • LCOH: (H2 costs + OPEX + electricity at LCOE + HTE thermal - H2 AS revenue) / H2 production\n")
                if greenfield_results.get('includes_battery_lcos', False):
                    f.write(
                        "  • LCOS: (battery costs + OPEX) / battery throughput\n")

            if greenfield_results.get('uses_real_as_revenue_data', False):
                f.write(
                    "  • AS revenues calculated from real system deployment data, not estimates\n")

            if greenfield_results.get('accounts_for_hte_thermal_cost', False):
                f.write(
                    "  • HTE thermal energy opportunity cost included in LCOH calculation\n")

            f.write(
                "\nNote: This greenfield analysis assumes building both nuclear plant and\n")
            f.write(
                "hydrogen system from zero, with both systems designed for 60-year operation.\n")
            f.write(
                "The analysis includes periodic replacement of H2 system components:\n")
            f.write("• Electrolyzers replaced every 20 years (2 replacements)\n")
            f.write("• H2 storage systems replaced every 30 years (1 replacement)\n")
            f.write("• Batteries replaced every 15 years (3 replacements)\n")
            f.write(
                "• Enhanced maintenance costs (+20%) for extended lifecycle operation\n")
            f.write(
                "This provides a comprehensive view of long-term integrated system economics.\n\n")

        # 10. Lifecycle Comparison Analysis (if available)
        if annual_metrics_rpt and "lifecycle_comparison_analysis" in annual_metrics_rpt:
            comparison_results = annual_metrics_rpt["lifecycle_comparison_analysis"]
            f.write(
                "\n10. Lifecycle Comparison Analysis: 60-Year vs 80-Year\n" + "-" * 54 + "\n")
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

                # Get LCOS from both scenarios
                lcos_60 = lifecycle_60.get('battery_lcos_usd_per_mwh', 0)
                lcos_80 = lifecycle_80.get('battery_lcos_usd_per_mwh', 0)
                lcos_difference = lcos_80 - lcos_60

                if lcos_60 > 0 or lcos_80 > 0:
                    f.write(
                        f"{'LCOS (USD/MWh)':<35} ${lcos_60:>15.2f} ${lcos_80:>15.2f} ${lcos_difference:>12.2f}\n")
                else:
                    f.write(
                        f"{'LCOS (USD/MWh)':<35} {'N/A':>20} {'N/A':>20} {'N/A':>15}\n")

        f.write("\nReport generated successfully.\n")

    logger.info(f"TEA report saved to {output_file_path}")
