"""
Visualization functions for the TEA module.
Contains detailed plotting implementations and color schemes.
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Initialize logger for this module
logger = logging.getLogger(__name__)


def get_component_color(component_name):
    """Define color scheme for different cost categories exactly as tea.py"""
    if component_name.startswith("CAPEX_"):
        if "Electrolyzer" in component_name:
            return '#1f77b4'  # Blue for Electrolyzer CAPEX
        elif "H2 Storage" in component_name or "H2_Storage" in component_name:
            return '#2ca02c'  # Green for H2 Storage CAPEX
        elif "Battery" in component_name:
            return '#ff7f0e'  # Orange for Battery CAPEX
        elif "NPP" in component_name or "Npp" in component_name:
            return '#d62728'  # Red for NPP Modifications
        else:
            return '#9467bd'  # Purple for other CAPEX
    elif "Opportunity_Cost" in component_name or "Direct_Cost" in component_name:
        return '#ff9999'  # Light red for electricity costs
    elif "Fixed_OM" in component_name:
        return '#90EE90'  # Light green for O&M
    elif "Replacement" in component_name:
        return '#FFB6C1'  # Light pink for replacements
    else:
        return '#DDA0DD'  # Plum for other OPEX


def create_cash_flow_plots(cash_flows_data, plot_dir, construction_period_years, incremental_metrics_data=None):
    """Create cash flow profile plots exactly matching tea.py"""
    logger.info("Creating cash flow plots...")

    years_axis = np.arange(1, len(cash_flows_data) + 1)
    cumulative_cf_plot = np.cumsum(cash_flows_data)

    # Main cash flow plot
    fig, ax1 = plt.subplots()
    bars = ax1.bar(
        years_axis,
        cash_flows_data,
        color="cornflowerblue",
        alpha=0.7,
        label="Annual Cash Flow",
    )
    for i, val in enumerate(cash_flows_data):
        if val < 0:
            bars[i].set_color("salmon")

    ax2 = ax1.twinx()
    ax2.plot(
        years_axis,
        cumulative_cf_plot,
        "forestgreen",
        marker="o",
        markersize=4,
        label="Cumulative Cash Flow",
    )

    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_xlabel("Project Year")
    ax1.set_ylabel("Annual Cash Flow (USD)")
    ax2.set_ylabel("Cumulative Cash Flow (USD)")

    if construction_period_years > 0:
        ax1.axvline(
            construction_period_years + 0.5,
            color="black",
            linestyle="--",
            lw=1,
            label="Operations Start",
        )

    ax1.set_title("Project Cash Flow Profile", fontweight="bold")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(plot_dir / "cash_flow_profile.png", dpi=300)
    plt.close(fig)

    # Incremental cash flow plot (if available)
    if (
        incremental_metrics_data
        and "pure_incremental_cash_flows" in incremental_metrics_data
    ):
        inc_cf_data = incremental_metrics_data["pure_incremental_cash_flows"]
        fig_inc, ax1_inc = plt.subplots()
        inc_cumulative_cf_plot = np.cumsum(inc_cf_data)
        inc_bars = ax1_inc.bar(
            years_axis,
            inc_cf_data,
            color="mediumpurple",
            alpha=0.7,
            label="Incremental Annual CF",
        )
        for i, val in enumerate(inc_cf_data):
            if val < 0:
                inc_bars[i].set_color("lightcoral")

        ax2_inc = ax1_inc.twinx()
        ax2_inc.plot(
            years_axis,
            inc_cumulative_cf_plot,
            "darkorange",
            marker="s",
            markersize=4,
            label="Cumulative Incremental CF",
        )

        ax1_inc.axhline(0, color="grey", lw=0.8)
        ax1_inc.set_xlabel("Project Year")
        ax1_inc.set_ylabel("Incremental Annual CF (USD)")
        ax2_inc.set_ylabel("Cumulative Incremental CF (USD)")

        if construction_period_years > 0:
            ax1_inc.axvline(
                construction_period_years + 0.5,
                color="black",
                linestyle="--",
                lw=1,
                label="Operations Start",
            )

        ax1_inc.set_title(
            "Pure Incremental Cash Flow Profile (H2/Battery System)",
            fontweight="bold",
        )
        inc_handles1, inc_labels1 = ax1_inc.get_legend_handles_labels()
        inc_handles2, inc_labels2 = ax2_inc.get_legend_handles_labels()
        ax1_inc.legend(
            inc_handles1 + inc_handles2, inc_labels1 + inc_labels2, loc="best"
        )

        if "Annual_Electricity_Opportunity_Cost_USD" in incremental_metrics_data:
            ax1_inc.text(
                0.02,
                0.02,
                f"Annual Electricity Opportunity Cost: ${incremental_metrics_data['Annual_Electricity_Opportunity_Cost_USD']:,.0f}",
                transform=ax1_inc.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
            )

        plt.tight_layout()
        plt.savefig(plot_dir / "incremental_cash_flow_profile.png", dpi=300)
        plt.close(fig_inc)


def create_capex_breakdown_plots(annual_metrics_data, cash_flows_data, plot_dir):
    """Create CAPEX breakdown visualization plots exactly matching tea.py"""
    logger.info("Creating CAPEX breakdown plots...")

    if (
        hasattr(annual_metrics_data, "capex_breakdown")
        and annual_metrics_data["capex_breakdown"]
    ):
        capex_data = annual_metrics_data["capex_breakdown"]
        capex_filtered = {k: v for k, v in capex_data.items() if v > 1e-3}

        if capex_filtered:
            # CAPEX Pie Chart
            fig_capex_pie, ax_capex_pie = plt.subplots()
            ax_capex_pie.pie(
                capex_filtered.values(),
                labels=[f"{k}\n(${v:,.0f})" for k,
                        v in capex_filtered.items()],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("crest", len(capex_filtered)),
            )
            ax_capex_pie.set_title(
                "CAPEX Breakdown by Component", fontweight="bold")
            ax_capex_pie.axis("equal")
            plt.tight_layout()
            plt.savefig(plot_dir / "capex_breakdown_pie.png", dpi=300)
            plt.close(fig_capex_pie)

            # CAPEX Bar Chart
            fig_capex_bar, ax_capex_bar = plt.subplots()
            capex_items = list(capex_filtered.items())
            capex_items.sort(key=lambda x: x[1], reverse=True)

            bar_labels = [k for k, v in capex_items]
            bar_values = [v for k, v in capex_items]

            bars = ax_capex_bar.bar(
                bar_labels,
                bar_values,
                color=sns.color_palette("crest", len(capex_items)),
            )
            ax_capex_bar.set_ylabel("Cost (USD)")
            ax_capex_bar.set_title("CAPEX by Component", fontweight="bold")

            for bar in bars:
                height = bar.get_height()
                ax_capex_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01 * max(bar_values),
                    f"${height:,.0f}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(plot_dir / "capex_breakdown_bar.png", dpi=300)
            plt.close(fig_capex_bar)

            # Total Cost Structure
            if "total_capex" in annual_metrics_data:
                total_capex = annual_metrics_data["total_capex"]
                fig_total, ax_total = plt.subplots()

                total_cost = abs(sum(cf for cf in cash_flows_data if cf < 0))
                opex_replacements = (
                    total_cost - total_capex if total_cost > total_capex else 0
                )

                categories = ["Total Project Cost"]
                capex_bar = ax_total.bar(
                    categories, [total_capex], label="CAPEX", color="steelblue"
                )
                opex_bar = ax_total.bar(
                    categories,
                    [opex_replacements],
                    bottom=[total_capex],
                    label="OPEX & Replacements",
                    color="lightcoral",
                )

                ax_total.set_ylabel("Cost (USD)")
                ax_total.set_title("Project Cost Structure", fontweight="bold")
                ax_total.legend()

                for bar in [capex_bar, opex_bar]:
                    for rect in bar:
                        height = rect.get_height()
                        if height > 0:
                            ax_total.text(
                                rect.get_x() + rect.get_width() / 2.0,
                                rect.get_y() + height / 2.0,
                                f"${height:,.0f}\n({height/total_cost*100:.1f}%)",
                                ha="center",
                                va="center",
                                color="white",
                                fontweight="bold",
                            )

                plt.tight_layout()
                plt.savefig(plot_dir / "total_cost_structure.png", dpi=300)
                plt.close(fig_total)


def create_lcoh_comprehensive_dashboard(annual_metrics_data: dict, plot_dir: Path):
    """Create comprehensive LCOH analysis dashboard exactly matching main tea.py"""
    logger.info("Creating comprehensive LCOH analysis dashboard...")

    if not annual_metrics_data or "lcoh_breakdown_analysis" not in annual_metrics_data:
        logger.warning(
            "No LCOH breakdown analysis data available for dashboard")
        return

    lcoh_analysis = annual_metrics_data["lcoh_breakdown_analysis"]
    lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
    total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

    if not lcoh_breakdown or total_lcoh <= 0:
        logger.warning("Invalid or empty LCOH breakdown data")
        return

    # Create a comprehensive 2x3 subplot figure (exactly matching tea.py)
    fig_lcoh_dashboard = plt.figure(figsize=(24, 16))

    # Filter out very small components (< 1% of total)
    significant_components = {
        k: v for k, v in lcoh_breakdown.items()
        if (v / total_lcoh) >= 0.01
    }

    # Group small components together
    small_components_total = sum(
        v for k, v in lcoh_breakdown.items()
        if (v / total_lcoh) < 0.01
    )

    if small_components_total > 0:
        significant_components["Other (< 1%)"] = small_components_total

    # Sort by value for better visualization
    sorted_components = sorted(
        significant_components.items(), key=lambda x: x[1], reverse=True)

    # Subplot 1: High-Level Breakdown (CAPEX vs OPEX)
    ax1 = plt.subplot(2, 3, 1)
    total_capex_lcoh = sum(
        v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))
    total_opex_lcoh = total_lcoh - total_capex_lcoh

    categories = ['CAPEX\n(Capital Recovery)', 'OPEX\n(Operating Costs)']
    values = [total_capex_lcoh, total_opex_lcoh]
    colors = ['#1f77b4', '#ff7f0e']

    bars1 = ax1.bar(categories, values, color=colors, alpha=0.8)
    for bar, value in zip(bars1, values):
        percentage = (value / total_lcoh) * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2.0,
            f"${value:.3f}/kg\n({percentage:.1f}%)",
            ha="center", va="center", fontweight="bold", color="white", fontsize=11
        )

    ax1.set_ylabel("Cost (USD/kg H2)", fontsize=12)
    ax1.set_title("LCOH High-Level Breakdown", fontweight="bold", fontsize=14)
    ax1.set_ylim(0, max(values) * 1.1)

    # Subplot 2: Detailed Component Breakdown (Pie Chart)
    ax2 = plt.subplot(2, 3, 2)

    pie_labels = []
    pie_values = []
    pie_colors = []

    for component, value in sorted_components:
        percentage = (value / total_lcoh) * 100
        display_name = component.replace(
            "CAPEX_", "").replace("_", " ").title()
        if "Electricity Opportunity Cost" in display_name:
            display_name = display_name.replace(
                "Electricity Opportunity Cost", "Elec. Opp. Cost")
        elif "Npp Modifications" in display_name:
            display_name = "NPP Modifications"

        pie_labels.append(
            f"{display_name}\n${value:.3f}/kg\n({percentage:.1f}%)")
        pie_values.append(value)
        pie_colors.append(get_component_color(component))

    wedges, texts, autotexts = ax2.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct='',
        startangle=90,
        textprops={'fontsize': 9}
    )

    ax2.set_title("LCOH Component Breakdown", fontweight="bold", fontsize=14)

    # Subplot 3: CAPEX Components Detail
    ax3 = plt.subplot(2, 3, 4)

    capex_components = {
        k.replace("CAPEX_", ""): v for k, v in sorted_components
        if k.startswith("CAPEX_")
    }

    if capex_components:
        capex_names = list(capex_components.keys())
        capex_values = list(capex_components.values())
        capex_colors = [get_component_color(
            f"CAPEX_{name}") for name in capex_names]

        bars3 = ax3.bar(range(len(capex_names)), capex_values,
                        color=capex_colors, alpha=0.8)

        for i, (bar, value) in enumerate(zip(bars3, capex_values)):
            percentage = (value / total_lcoh) * 100
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(capex_values) * 0.01,
                f"${value:.3f}\n({percentage:.1f}%)",
                ha="center", va="bottom", fontsize=9
            )

        ax3.set_ylabel("Cost (USD/kg H2)", fontsize=12)
        ax3.set_title("CAPEX Components", fontweight="bold", fontsize=14)
        ax3.set_xticks(range(len(capex_names)))
        ax3.set_xticklabels([name.replace("_", " ").title() for name in capex_names],
                            rotation=45, ha="right", fontsize=10)

    # Subplot 4: OPEX Components Detail
    ax4 = plt.subplot(2, 3, 5)

    opex_components = {
        k: v for k, v in sorted_components
        if not k.startswith("CAPEX_")
    }

    if opex_components:
        opex_names = list(opex_components.keys())
        opex_values = list(opex_components.values())
        opex_colors = [get_component_color(name) for name in opex_names]

        bars4 = ax4.bar(range(len(opex_names)), opex_values,
                        color=opex_colors, alpha=0.8)

        for i, (bar, value) in enumerate(zip(bars4, opex_values)):
            percentage = (value / total_lcoh) * 100
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(opex_values) * 0.01,
                f"${value:.3f}\n({percentage:.1f}%)",
                ha="center", va="bottom", fontsize=9
            )

        ax4.set_ylabel("Cost (USD/kg H2)", fontsize=12)
        ax4.set_title("OPEX Components", fontweight="bold", fontsize=14)
        ax4.set_xticks(range(len(opex_names)))
        ax4.set_xticklabels([name.replace("_", " ").title() for name in opex_names],
                            rotation=45, ha="right", fontsize=10)

    # Subplot 5: Cost Efficiency Analysis
    ax5 = plt.subplot(2, 3, 3)

    annual_h2_production = annual_metrics_data.get(
        "Annual_H2_Production_kg", 1)
    electrolyzer_capacity = annual_metrics_data.get(
        "Electrolyzer_Capacity_MW", 1)
    capacity_factor = annual_metrics_data.get(
        "Electrolyzer_Capacity_Factor", 0.5)

    efficiency_metrics = {
        'H2 Production\nRate (kg/day)': annual_h2_production / 365,
        'Capacity\nFactor (%)': capacity_factor * 100,
        'LCOH per\nCapacity Factor': total_lcoh / (capacity_factor * 100) if capacity_factor > 0 else 0,
        'Cost per\nMW Capacity': total_lcoh * annual_h2_production / electrolyzer_capacity if electrolyzer_capacity > 0 else 0,
        'Energy\nEfficiency (%)': annual_metrics_data.get("Electrolyzer_Efficiency", 0.7) * 100
    }

    metrics_names = list(efficiency_metrics.keys())
    metrics_values = list(efficiency_metrics.values())

    # Normalize values for better visualization (scale to 0-100)
    max_val = max(metrics_values) if metrics_values else 1
    normalized_values = [v / max_val * 100 for v in metrics_values]

    colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB']
    bars5 = ax5.bar(range(len(metrics_names)), normalized_values,
                    color=colors[:len(metrics_names)], alpha=0.8)

    for i, (bar, orig_val) in enumerate(zip(bars5, metrics_values)):
        if 'kg/day' in metrics_names[i]:
            label = f'{orig_val:.0f}'
        elif '%' in metrics_names[i]:
            label = f'{orig_val:.1f}%'
        elif 'LCOH per' in metrics_names[i]:
            label = f'${orig_val:.4f}'
        elif 'Cost per' in metrics_names[i]:
            label = f'${orig_val:.0f}'
        else:
            label = f'{orig_val:.2f}'

        ax5.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 2,
                 label, ha="center", va="bottom", fontsize=9)

    ax5.set_ylabel("Normalized Score (0-100)", fontsize=10)
    ax5.set_title("Production Efficiency Metrics",
                  fontweight="bold", fontsize=12)
    ax5.set_xticks(range(len(metrics_names)))
    ax5.set_xticklabels(metrics_names, rotation=45, ha="right", fontsize=9)
    ax5.set_ylim(0, 110)
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Cost Structure Comparison
    ax6 = plt.subplot(2, 3, 6)

    cost_categories = {
        'Capital\nRecovery': sum(v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_")),
        'Electricity\nCosts': sum(v for k, v in lcoh_breakdown.items()
                                  if "Opportunity_Cost" in k or "Direct_Cost" in k),
        'Fixed\nO&M': sum(v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k),
        'Variable\nOPEX': sum(v for k, v in lcoh_breakdown.items()
                              if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost",
                                       "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"]),
        'Replacements': sum(v for k, v in lcoh_breakdown.items() if "Replacement" in k),
        'Other': sum(v for k, v in lcoh_breakdown.items()
                     if not any(term in k for term in ["CAPEX_", "Opportunity_Cost", "Direct_Cost",
                                                       "Fixed_OM", "VOM_", "Water_Cost", "Startup_Cost",
                                                       "Ramping_Cost", "H2_Storage_Cycle_Cost", "Replacement"]))
    }

    # Filter out zero values
    cost_categories = {k: v for k, v in cost_categories.items() if v > 1e-6}

    if cost_categories:
        categories = list(cost_categories.keys())
        values = list(cost_categories.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b'][:len(categories)]

        bars6 = ax6.bar(categories, values, color=colors, alpha=0.8)

        for bar, value in zip(bars6, values):
            percentage = (value / total_lcoh) * 100
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(values) * 0.01,
                f"${value:.3f}\n({percentage:.1f}%)",
                ha="center", va="bottom", fontsize=9
            )

        ax6.set_ylabel("Cost (USD/kg H2)", fontsize=10)
        ax6.set_title("Cost Structure by Category",
                      fontweight="bold", fontsize=12)
        ax6.set_xticks(range(len(categories)))
        ax6.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)

    # Add overall title and summary information
    fig_lcoh_dashboard.suptitle(
        f"Levelized Cost of Hydrogen (LCOH) Analysis\nTotal LCOH: ${total_lcoh:.3f}/kg H2",
        fontsize=18, fontweight="bold", y=0.98
    )

    # Add summary text box
    summary_text = f"""
Key Insights:
• Total LCOH: ${total_lcoh:.3f}/kg H2
• CAPEX Contribution: ${total_capex_lcoh:.3f}/kg ({total_capex_lcoh/total_lcoh*100:.1f}%)
• OPEX Contribution: ${total_opex_lcoh:.3f}/kg ({total_opex_lcoh/total_lcoh*100:.1f}%)
• Number of Cost Components: {len(lcoh_breakdown)}
    """

    fig_lcoh_dashboard.text(0.02, 0.02, summary_text.strip(),
                            fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                                                   facecolor="lightgray", alpha=0.8),
                            verticalalignment='bottom')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15, hspace=0.3, wspace=0.3)
    plt.savefig(plot_dir / "lcoh_comprehensive_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig_lcoh_dashboard)

    logger.info("Comprehensive LCOH analysis dashboard created successfully.")


def create_lcoh_benchmarking_analysis(annual_metrics_data: dict, plot_dir: Path):
    """Create LCOH benchmarking and trends analysis exactly matching main tea.py"""
    logger.info("Creating LCOH benchmarking analysis...")

    if not annual_metrics_data or "lcoh_breakdown_analysis" not in annual_metrics_data:
        logger.warning(
            "No LCOH breakdown analysis data available for benchmarking")
        return

    lcoh_analysis = annual_metrics_data["lcoh_breakdown_analysis"]
    lcoh_breakdown = lcoh_analysis.get("lcoh_breakdown_usd_per_kg", {})
    total_lcoh = lcoh_analysis.get("total_lcoh_usd_per_kg", 0)

    if not lcoh_breakdown or total_lcoh <= 0:
        logger.warning("Invalid or empty LCOH breakdown data for benchmarking")
        return

    # Create a larger figure with 2x3 subplots
    fig_benchmark, ((ax_bench, ax_waterfall, ax_tornado), (ax_trends, ax_breakdown_pie, ax_heatmap)) = plt.subplots(
        2, 3, figsize=(24, 12))

    # Subplot 1: LCOH Benchmarking against industry standards
    benchmark_data = {
        'Current LCOH': total_lcoh,
        'DOE 2030 Target': 2.0,  # DOE hydrogen target
        'Steam Methane\nReforming': 1.5,  # Typical SMR cost
        'Grid Electrolysis': 5.0,  # Typical grid electrolysis
        'Renewable\nElectrolysis': 3.5   # Renewable electrolysis
    }

    bench_names = list(benchmark_data.keys())
    bench_values = list(benchmark_data.values())
    bench_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    bars_bench = ax_bench.bar(
        bench_names, bench_values, color=bench_colors, alpha=0.8)

    for bar, value in zip(bars_bench, bench_values):
        ax_bench.text(bar.get_x() + bar.get_width() / 2.0,
                      bar.get_height() + 0.05,
                      f'${value:.2f}', ha="center", va="bottom", fontsize=10, fontweight='bold')

    ax_bench.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
    ax_bench.set_title("LCOH Benchmarking Analysis",
                       fontweight="bold", fontsize=14)
    ax_bench.set_xticks(range(len(bench_names)))
    ax_bench.set_xticklabels(bench_names, rotation=45, ha="right", fontsize=10)
    ax_bench.grid(True, alpha=0.3, axis='y')
    ax_bench.axhline(y=2.0, color='red', linestyle='--',
                     alpha=0.7, label='DOE 2030 Target')
    ax_bench.legend()

    # Subplot 2: LCOH Waterfall Chart
    waterfall_components = [
        ('Base', 0),
        ('CAPEX', sum(v for k, v in lcoh_breakdown.items() if k.startswith("CAPEX_"))),
        ('Electricity', sum(v for k, v in lcoh_breakdown.items()
         if "Opportunity_Cost" in k or "Direct_Cost" in k)),
        ('Fixed O&M', sum(v for k, v in lcoh_breakdown.items() if "Fixed_OM" in k)),
        ('Variable OPEX', sum(v for k, v in lcoh_breakdown.items()
                              if k in ["VOM_Electrolyzer", "VOM_Battery", "Water_Cost", "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"])),
        ('Replacements', sum(v for k, v in lcoh_breakdown.items() if "Replacement" in k)),
        ('Total', total_lcoh)
    ]

    # Calculate cumulative values for waterfall
    cumulative = 0
    x_pos = range(len(waterfall_components))
    heights = []
    bottoms = []
    colors_waterfall = []

    for i, (name, value) in enumerate(waterfall_components):
        if name == 'Base':
            heights.append(0)
            bottoms.append(0)
            colors_waterfall.append('lightgray')
        elif name == 'Total':
            heights.append(total_lcoh)
            bottoms.append(0)
            colors_waterfall.append('darkgreen')
        else:
            heights.append(value)
            bottoms.append(cumulative)
            colors_waterfall.append('steelblue')
            cumulative += value

    bars_waterfall = ax_waterfall.bar(x_pos, heights, bottom=bottoms,
                                      color=colors_waterfall, alpha=0.8)

    # Add connecting lines
    for i in range(1, len(waterfall_components)-1):
        if waterfall_components[i][1] > 0:
            ax_waterfall.plot([i-0.4, i+0.4], [bottoms[i], bottoms[i]],
                              'k--', alpha=0.5, linewidth=1)

    # Add value labels
    for i, (bar, (name, value)) in enumerate(zip(bars_waterfall, waterfall_components)):
        if name not in ['Base', 'Total'] and value > 0:
            ax_waterfall.text(bar.get_x() + bar.get_width() / 2.0,
                              bar.get_height() + bottoms[i] + 0.02,
                              f'${value:.3f}', ha="center", va="bottom", fontsize=9)

    ax_waterfall.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
    ax_waterfall.set_title("LCOH Waterfall Analysis",
                           fontweight="bold", fontsize=14)
    ax_waterfall.set_xticks(x_pos)
    ax_waterfall.set_xticklabels([comp[0] for comp in waterfall_components],
                                 rotation=45, ha="right", fontsize=10)
    ax_waterfall.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Cost trends and projections
    years = list(range(2024, 2035))
    capex_reduction = [1.0 * (0.95 ** (year - 2024))
                       for year in years]  # 5% annual reduction
    electricity_cost_trend = [1.0 * (0.97 ** (year - 2024))
                              for year in years]  # 3% annual reduction

    current_capex = sum(v for k, v in lcoh_breakdown.items()
                        if k.startswith("CAPEX_"))
    current_elec = sum(v for k, v in lcoh_breakdown.items()
                       if "Opportunity_Cost" in k or "Direct_Cost" in k)
    current_other = total_lcoh - current_capex - current_elec

    projected_lcoh = [current_capex * capex_red + current_elec * elec_red + current_other
                      for capex_red, elec_red in zip(capex_reduction, electricity_cost_trend)]

    ax_trends.plot(years, projected_lcoh, 'b-', linewidth=3,
                   marker='o', markersize=6, label='Projected LCOH')
    ax_trends.axhline(y=2.0, color='red', linestyle='--',
                      alpha=0.7, label='DOE 2030 Target')
    ax_trends.axhline(y=total_lcoh, color='green',
                      linestyle='-', alpha=0.7, label='Current LCOH')

    ax_trends.set_xlabel("Year", fontsize=12)
    ax_trends.set_ylabel("LCOH (USD/kg H2)", fontsize=12)
    ax_trends.set_title("LCOH Projection (2024-2034)",
                        fontweight="bold", fontsize=14)
    ax_trends.grid(True, alpha=0.3)
    ax_trends.legend()
    ax_trends.set_ylim(0, max(projected_lcoh) * 1.1)

    # Subplot 4: Enhanced pie chart with cost efficiency indicators
    significant_components_pie = {
        k: v for k, v in lcoh_breakdown.items() if v/total_lcoh >= 0.05}
    other_components_sum = sum(
        v for k, v in lcoh_breakdown.items() if v/total_lcoh < 0.05)

    if other_components_sum > 0:
        significant_components_pie['Other Components'] = other_components_sum

    pie_labels_enhanced = []
    pie_values_enhanced = list(significant_components_pie.values())
    pie_colors_enhanced = []

    for component in significant_components_pie.keys():
        clean_name = component.replace("CAPEX_", "").replace("_", " ").title()
        if "Electricity Opportunity Cost" in clean_name:
            clean_name = "Electricity Cost"
        elif "Npp Modifications" in clean_name:
            clean_name = "NPP Modifications"

        percentage = (significant_components_pie[component] / total_lcoh) * 100
        pie_labels_enhanced.append(
            f"{clean_name}\n${significant_components_pie[component]:.3f}\n({percentage:.1f}%)")
        pie_colors_enhanced.append(get_component_color(component))

    wedges_enhanced, texts_enhanced, autotexts_enhanced = ax_breakdown_pie.pie(
        pie_values_enhanced,
        labels=pie_labels_enhanced,
        colors=pie_colors_enhanced,
        autopct='',
        startangle=90,
        textprops={'fontsize': 9}
    )

    ax_breakdown_pie.set_title(
        "Major LCOH Components\n(>5% of total)", fontweight="bold", fontsize=12)

    # Subplot 5: Tornado Chart for LCOH Sensitivity (if available)
    sensitivity_data = lcoh_analysis.get("sensitivity_analysis", {})
    if sensitivity_data:
        # Create simplified tornado chart
        tornado_data = []
        # Top 5
        for component, sensitivity in list(sensitivity_data.items())[:5]:
            neg_impact = abs(sensitivity.get("-20%", {}).get("lcoh_change", 0))
            pos_impact = sensitivity.get("+20%", {}).get("lcoh_change", 0)

            clean_name = component.replace(
                "CAPEX_", "").replace("_", " ").title()
            if "Electricity Opportunity Cost" in clean_name:
                clean_name = "Electricity Opportunity Cost"
            elif "Npp Modifications" in clean_name:
                clean_name = "NPP Modifications"

            tornado_data.append({
                'component': clean_name,
                'neg_impact': -neg_impact,
                'pos_impact': pos_impact,
            })

        if tornado_data:
            y_pos = range(len(tornado_data))
            components = [item['component'] for item in tornado_data]
            neg_impacts = [item['neg_impact'] for item in tornado_data]
            pos_impacts = [item['pos_impact'] for item in tornado_data]

            bars_neg = ax_tornado.barh(
                y_pos, neg_impacts, color='#FF6B6B', alpha=0.8, label='-20% Cost')
            bars_pos = ax_tornado.barh(
                y_pos, pos_impacts, color='#4ECDC4', alpha=0.8, label='+20% Cost')

            ax_tornado.set_yticks(y_pos)
            ax_tornado.set_yticklabels(components, fontsize=9)
            ax_tornado.set_xlabel('LCOH Change (USD/kg H2)', fontsize=10)
            ax_tornado.set_title('LCOH Sensitivity Analysis\n(Top 5 Cost Drivers)',
                                 fontweight='bold', fontsize=12)
            ax_tornado.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax_tornado.legend(loc='lower right', fontsize=8)
            ax_tornado.grid(True, alpha=0.3, axis='x')
    else:
        ax_tornado.text(0.5, 0.5, 'No Sensitivity\nData Available',
                        transform=ax_tornado.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold')
        ax_tornado.set_title('LCOH Sensitivity Analysis',
                             fontweight='bold', fontsize=12)

    # Subplot 6: Multi-Parameter Sensitivity Heatmap (if available)
    if sensitivity_data:
        # Create simplified heatmap
        parameters = list(sensitivity_data.keys())[:5]
        change_levels = ['-20%', '0%', '+20%']

        sensitivity_matrix = []
        parameter_labels = []

        for param in parameters:
            clean_name = param.replace("CAPEX_", "").replace("_", " ").title()
            if "Electricity Opportunity Cost" in clean_name:
                clean_name = "Elec. Opp. Cost"
            elif "Npp Modifications" in clean_name:
                clean_name = "NPP Modifications"
            elif len(clean_name) > 15:
                clean_name = clean_name[:12] + "..."
            parameter_labels.append(clean_name)

            param_sensitivity = sensitivity_data[param]
            row_values = []
            for change in change_levels:
                if change == '0%':
                    row_values.append(total_lcoh)
                elif change in param_sensitivity:
                    row_values.append(param_sensitivity[change].get(
                        'new_total_lcoh', total_lcoh))
                else:
                    row_values.append(total_lcoh)
            sensitivity_matrix.append(row_values)

        if sensitivity_matrix:
            sensitivity_array = np.array(sensitivity_matrix)

            from matplotlib.colors import TwoSlopeNorm
            vmin = sensitivity_array.min()
            vmax = sensitivity_array.max()
            vcenter = total_lcoh
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            im = ax_heatmap.imshow(
                sensitivity_array, cmap='RdYlBu_r', norm=norm, aspect='auto')

            cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
            cbar.set_label('LCOH (USD/kg)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            ax_heatmap.set_xticks(range(len(change_levels)))
            ax_heatmap.set_xticklabels(change_levels, fontsize=9)
            ax_heatmap.set_yticks(range(len(parameter_labels)))
            ax_heatmap.set_yticklabels(parameter_labels, fontsize=9)

            for i in range(len(parameter_labels)):
                for j in range(len(change_levels)):
                    value = sensitivity_array[i, j]
                    text_color = 'white' if abs(
                        value - vcenter) > (vmax - vmin) * 0.3 else 'black'
                    ax_heatmap.text(j, i, f'${value:.3f}', ha='center', va='center',
                                    color=text_color, fontsize=8, fontweight='bold')

            ax_heatmap.set_xlabel('Parameter Change', fontsize=10)
            ax_heatmap.set_ylabel('Cost Parameters', fontsize=10)
            ax_heatmap.set_title('LCOH Sensitivity Heatmap\n(Parameter Impact)',
                                 fontweight='bold', fontsize=12)
    else:
        ax_heatmap.text(0.5, 0.5, 'No Sensitivity\nData Available',
                        transform=ax_heatmap.transAxes, ha='center', va='center',
                        fontsize=12, fontweight='bold')
        ax_heatmap.set_title('LCOH Sensitivity Heatmap',
                             fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(plot_dir / "lcoh_benchmarking_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig_benchmark)

    logger.info("LCOH benchmarking analysis created successfully.")
