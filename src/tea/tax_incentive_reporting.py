"""
Tax Incentive Analysis Reporting Module.

This module generates comprehensive reports and visualizations for federal tax incentive scenarios,
including detailed comparison tables, cash flow charts, and financial metric analysis.
"""

import textwrap
from typing import Dict, List
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)


def generate_tax_incentive_comparative_report(
    analysis_results: Dict,
    output_file_path: Path,
    project_name: str = "Greenfield Nuclear-Hydrogen System"
) -> bool:
    """
    Generate comprehensive comparative report for tax incentive scenarios.

    Args:
        analysis_results: Results from run_comprehensive_tax_incentive_analysis
        output_file_path: Path for the output report file
        project_name: Name of the project for the report header

    Returns:
        Boolean indicating success
    """
    logger.info(
        f"Generating comprehensive tax incentive comparative report: {output_file_path}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Report Header
            f.write("=" * 100 + "\n")
            f.write(f"FEDERAL TAX INCENTIVE ANALYSIS REPORT\n")
            f.write(f"{project_name}\n")
            f.write("=" * 100 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 50 + "\n\n")

            comparative = analysis_results["comparative_analysis"]
            params = analysis_results["analysis_parameters"]

            # Key findings
            best_scenario = comparative["best_scenario"]
            best_scenario_name = {
                "baseline": "Baseline (No Incentives)",
                "ptc": "45Y Production Tax Credit",
                "itc": "48E Investment Tax Credit"
            }.get(best_scenario, best_scenario)

            f.write(f"Best Performing Scenario: {best_scenario_name}\n\n")

            # NPV comparison
            baseline_npv = comparative["npv_comparison"]["baseline_npv"]
            ptc_npv = comparative["npv_comparison"]["ptc_npv"]
            itc_npv = comparative["npv_comparison"]["itc_npv"]
            ptc_improvement = comparative["npv_comparison"]["ptc_npv_improvement"]
            itc_improvement = comparative["npv_comparison"]["itc_npv_improvement"]

            f.write("Net Present Value (NPV) Summary:\n")
            f.write(f"  Baseline Scenario:        ${baseline_npv:>15,.0f}\n")
            f.write(
                f"  45Y PTC Scenario:         ${ptc_npv:>15,.0f} (${ptc_improvement:+15,.0f})\n")
            f.write(
                f"  48E ITC Scenario:         ${itc_npv:>15,.0f} (${itc_improvement:+15,.0f})\n\n")

            # Key metrics comparison
            f.write("Key Financial Metrics Comparison:\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Metric':<30} {'Baseline':<15} {'45Y PTC':<15} {'48E ITC':<15}\n")
            f.write("-" * 75 + "\n")

            baseline_metrics = analysis_results["scenarios"]["baseline"]["financial_metrics"]
            ptc_metrics = analysis_results["scenarios"]["ptc"]["financial_metrics"]
            itc_metrics = analysis_results["scenarios"]["itc"]["financial_metrics"]

            # Format IRR values
            baseline_irr = baseline_metrics.get("irr_percent", "N/A")
            ptc_irr = ptc_metrics.get("irr_percent", "N/A")
            itc_irr = itc_metrics.get("irr_percent", "N/A")

            baseline_irr_str = f"{baseline_irr:.1f}%" if baseline_irr != "N/A" and baseline_irr is not None else "N/A"
            ptc_irr_str = f"{ptc_irr:.1f}%" if ptc_irr != "N/A" and ptc_irr is not None else "N/A"
            itc_irr_str = f"{itc_irr:.1f}%" if itc_irr != "N/A" and itc_irr is not None else "N/A"

            f.write(
                f"{'NPV (Million USD)':<30} ${baseline_npv/1e6:<14.1f} ${ptc_npv/1e6:<14.1f} ${itc_npv/1e6:<14.1f}\n")
            f.write(
                f"{'IRR':<30} {baseline_irr_str:<15} {ptc_irr_str:<15} {itc_irr_str:<15}\n")

            # Payback periods
            baseline_payback = baseline_metrics.get(
                "payback_period_years", "N/A")
            ptc_payback = ptc_metrics.get("payback_period_years", "N/A")
            itc_payback = itc_metrics.get("payback_period_years", "N/A")

            baseline_payback_str = f"{baseline_payback:.1f}" if baseline_payback != "N/A" and baseline_payback is not None else "N/A"
            ptc_payback_str = f"{ptc_payback:.1f}" if ptc_payback != "N/A" and ptc_payback is not None else "N/A"
            itc_payback_str = f"{itc_payback:.1f}" if itc_payback != "N/A" and itc_payback is not None else "N/A"

            f.write(
                f"{'Payback Period (years)':<30} {baseline_payback_str:<15} {ptc_payback_str:<15} {itc_payback_str:<15}\n\n")

            # Project Parameters
            f.write("PROJECT PARAMETERS\n")
            f.write("-" * 50 + "\n\n")
            f.write(
                f"Project Lifetime:           {params['project_lifetime_years']} years\n")
            f.write(
                f"Construction Period:        {params['construction_period_years']} years\n")
            f.write(
                f"Discount Rate:              {params['discount_rate']:.1%}\n")
            f.write(f"Corporate Tax Rate:         {params['tax_rate']:.1%}\n")
            f.write(
                f"Total Project CAPEX:        ${params['total_capex_usd']:,.0f}\n")
            f.write(
                f"Annual H2 Production:       {params['annual_h2_production_kg']:,.0f} kg/year\n\n")

            # Detailed Scenario Analysis
            f.write("DETAILED SCENARIO ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # Scenario A: Baseline
            _write_scenario_details(
                f, analysis_results["scenarios"]["baseline"], "BASELINE SCENARIO (No Tax Incentives)")

            # Scenario B: 45Y PTC
            _write_scenario_details(
                f, analysis_results["scenarios"]["ptc"], "45Y PRODUCTION TAX CREDIT SCENARIO")

            # Scenario C: 48E ITC
            _write_scenario_details(
                f, analysis_results["scenarios"]["itc"], "48E INVESTMENT TAX CREDIT SCENARIO")

            # Tax Incentive Value Analysis
            f.write("TAX INCENTIVE VALUE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # PTC Analysis
            ptc_analysis = analysis_results["scenarios"]["ptc"]["analysis"]
            if "tax_benefits" in ptc_analysis and "ptc" in ptc_analysis["tax_benefits"]:
                ptc_details = ptc_analysis["tax_benefits"]["ptc"]
                f.write("45Y Production Tax Credit Details:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"PTC Rate:                   ${ptc_details['ptc_rate_usd_per_mwh']:.2f}/MWh\n")
                f.write(
                    f"Eligible Duration:          {ptc_details['ptc_duration_years']} years\n")
                f.write(
                    f"Annual Generation:          {ptc_details['annual_generation_mwh']:,.0f} MWh\n")
                f.write(
                    f"Annual PTC Benefit:         ${ptc_details['annual_ptc_benefit_usd']:,.0f}\n")
                f.write(
                    f"Total PTC Value:            ${ptc_details['total_ptc_value_usd']:,.0f}\n")
                f.write(
                    f"Total PTC NPV:              ${ptc_details['total_ptc_npv_usd']:,.0f}\n")
                f.write(
                    f"Data Source:                {ptc_details['generation_data_source']}\n\n")

            # ITC Analysis
            itc_analysis = analysis_results["scenarios"]["itc"]["analysis"]
            if "tax_benefits" in itc_analysis and "itc" in itc_analysis["tax_benefits"]:
                itc_details = itc_analysis["tax_benefits"]["itc"]
                f.write("48E Investment Tax Credit Details:\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"ITC Rate:                   {itc_details['itc_rate']:.0%}\n")
                f.write(
                    f"Qualified CAPEX:            ${itc_details['total_qualified_capex_usd']:,.0f}\n")
                f.write(
                    f"ITC Credit Amount:          ${itc_details['itc_credit_amount_usd']:,.0f}\n")
                f.write(
                    f"Depreciation Reduction:     ${itc_details['depreciation_basis_reduction_usd']:,.0f}\n")

                if "net_itc_benefit" in itc_analysis:
                    f.write(
                        f"Net ITC Benefit:            ${itc_analysis['net_itc_benefit']:,.0f}\n")

                f.write("\nQualified Equipment Categories:\n")
                for component, qualification in itc_details['eligible_components'].items():
                    if component in itc_details['component_qualified_capex']:
                        qualified_amount = itc_details['component_qualified_capex'][component]
                        if qualified_amount > 0:
                            f.write(
                                f"  {component:<25}: {qualification:>6.0%} (${qualified_amount:>12,.0f})\n")
                f.write("\n")

            # Cash Flow Impact Summary
            f.write("ANNUAL CASH FLOW IMPACT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Year-by-year comparison table for first 20 years
            f.write("Cash Flow Comparison (First 20 Years, Million USD):\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Year':<6} {'Baseline':<12} {'45Y PTC':<12} {'48E ITC':<12} {'PTC Delta':<12} {'ITC Delta':<12}\n")
            f.write("-" * 80 + "\n")

            baseline_cf = analysis_results["scenarios"]["baseline"]["cash_flows"]
            ptc_cf = analysis_results["scenarios"]["ptc"]["cash_flows"]
            itc_cf = analysis_results["scenarios"]["itc"]["cash_flows"]

            for year in range(min(20, len(baseline_cf))):
                baseline_val = baseline_cf[year] / 1e6
                ptc_val = ptc_cf[year] / 1e6
                itc_val = itc_cf[year] / 1e6
                ptc_delta = (ptc_cf[year] - baseline_cf[year]) / 1e6
                itc_delta = (itc_cf[year] - baseline_cf[year]) / 1e6

                f.write(
                    f"{year:<6} {baseline_val:>11.1f} {ptc_val:>11.1f} {itc_val:>11.1f} {ptc_delta:>+11.1f} {itc_delta:>+11.1f}\n")

            f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")

            # Determine which incentive provides better value
            if ptc_improvement > itc_improvement:
                better_incentive = "45Y Production Tax Credit"
                improvement_amount = ptc_improvement
                worse_incentive = "48E Investment Tax Credit"
            else:
                better_incentive = "48E Investment Tax Credit"
                improvement_amount = itc_improvement
                worse_incentive = "45Y Production Tax Credit"

            f.write(
                f"1. The {better_incentive} provides superior financial returns with an NPV\n")
            f.write(
                f"   improvement of ${improvement_amount:,.0f} compared to the baseline scenario.\n\n")

            f.write(
                f"2. Both tax incentive scenarios improve project economics significantly:\n")
            f.write(f"   - 45Y PTC improves NPV by ${ptc_improvement:,.0f}\n")
            f.write(
                f"   - 48E ITC improves NPV by ${itc_improvement:,.0f}\n\n")

            f.write(
                f"3. Consider the following factors when choosing between incentives:\n")
            f.write(f"   - PTC provides steady cash flows over 10 years\n")
            f.write(
                f"   - ITC provides immediate cash benefit but reduces depreciation\n")
            f.write(f"   - Risk tolerance and financing requirements\n")
            f.write(f"   - Regulatory and policy stability considerations\n\n")

            # Report Footer
            f.write("=" * 100 + "\n")
            f.write("End of Tax Incentive Analysis Report\n")
            f.write("=" * 100 + "\n")

        logger.info(
            f"Tax incentive comparative report generated successfully: {output_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error generating tax incentive report: {e}")
        return False


def _write_scenario_details(f, scenario_data: Dict, scenario_title: str):
    """Write detailed scenario analysis to report file."""
    f.write(f"{scenario_title}\n")
    f.write("-" * len(scenario_title) + "\n\n")

    analysis = scenario_data["analysis"]
    metrics = scenario_data["financial_metrics"]

    f.write(f"Description: {analysis['description']}\n\n")

    # Financial metrics
    f.write("Financial Metrics:\n")
    npv = metrics.get("npv_usd", 0)
    irr = metrics.get("irr_percent")
    payback = metrics.get("payback_period_years")
    roi = metrics.get("roi_percent", 0)
    lcoh = metrics.get("lcoh_usd_per_kg")

    f.write(f"  Net Present Value:        ${npv:,.0f}\n")

    if irr is not None and not np.isnan(irr):
        f.write(f"  Internal Rate of Return:  {irr:.2f}%\n")
    else:
        f.write(f"  Internal Rate of Return:  N/A\n")

    if payback is not None:
        f.write(f"  Payback Period:           {payback:.1f} years\n")
    else:
        f.write(f"  Payback Period:           N/A\n")

    f.write(f"  Return on Investment:     {roi:.2f}%\n")

    if lcoh is not None:
        f.write(f"  LCOH:                     ${lcoh:.2f}/kg\n")
    else:
        f.write(f"  LCOH:                     N/A\n")

    # Cash flow totals
    total_investment = metrics.get("total_investment_usd", 0)
    total_returns = metrics.get("total_returns_usd", 0)

    f.write(f"  Total Investment:         ${total_investment:,.0f}\n")
    f.write(f"  Total Returns:            ${total_returns:,.0f}\n")

    # Incremental value
    if "incremental_value" in analysis:
        f.write(
            f"  Incremental Value:        ${analysis['incremental_value']:,.0f}\n")

    f.write("\n")


def create_tax_incentive_visualizations(
    analysis_results: Dict,
    output_dir: Path,
    project_name: str = "Nuclear-Hydrogen System"
) -> bool:
    """
    Create comprehensive visualizations for tax incentive analysis.

    Args:
        analysis_results: Results from run_comprehensive_tax_incentive_analysis
        output_dir: Directory for output plots
        project_name: Project name for plot titles

    Returns:
        Boolean indicating success
    """
    logger.info(
        f"Creating tax incentive analysis visualizations: {output_dir}")

    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 13,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

        # 1. NPV Comparison Chart
        _create_npv_comparison_chart(
            analysis_results, output_dir, project_name)

        # 2. Cash Flow Comparison Chart
        _create_cash_flow_comparison_chart(
            analysis_results, output_dir, project_name)

        # 3. Tax Incentive Value Breakdown
        _create_tax_incentive_value_chart(
            analysis_results, output_dir, project_name)

        # 4. Financial Metrics Dashboard
        _create_financial_metrics_dashboard(
            analysis_results, output_dir, project_name)

        # 5. Annual Cash Flow Heatmap
        _create_annual_cash_flow_heatmap(
            analysis_results, output_dir, project_name)

        # 6. Cumulative Cash Flow Chart
        _create_cumulative_cash_flow_chart(
            analysis_results, output_dir, project_name)

        logger.info("Tax incentive visualizations created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating tax incentive visualizations: {e}")
        return False


def _create_npv_comparison_chart(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create NPV comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['Baseline\n(No Incentives)',
                 '45Y PTC\n($30/MWh)', '48E ITC\n(50% Credit)']
    npv_values = [
        analysis_results["scenarios"]["baseline"]["financial_metrics"]["npv_usd"] / 1e6,
        analysis_results["scenarios"]["ptc"]["financial_metrics"]["npv_usd"] / 1e6,
        analysis_results["scenarios"]["itc"]["financial_metrics"]["npv_usd"] / 1e6
    ]

    colors = ['lightcoral', 'lightgreen', 'lightblue']
    bars = ax.bar(scenarios, npv_values, color=colors,
                  alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, value in zip(bars, npv_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(npv_values) * 0.02),
                f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

    ax.set_title(f'{project_name}\nNet Present Value Comparison by Tax Incentive Scenario',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Net Present Value (Million USD)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement annotations
    baseline_npv = npv_values[0]
    ptc_improvement = npv_values[1] - baseline_npv
    itc_improvement = npv_values[2] - baseline_npv

    if ptc_improvement > 0:
        ax.annotate(f'+${ptc_improvement:.0f}M', xy=(1, npv_values[1]),
                    xytext=(1, npv_values[1] + max(npv_values) * 0.1),
                    ha='center', fontweight='bold', color='green',
                    arrowprops=dict(arrowstyle='->', color='green'))

    if itc_improvement > 0:
        ax.annotate(f'+${itc_improvement:.0f}M', xy=(2, npv_values[2]),
                    xytext=(2, npv_values[2] + max(npv_values) * 0.1),
                    ha='center', fontweight='bold', color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue'))

    plt.tight_layout()
    plt.savefig(output_dir / 'npv_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_cash_flow_comparison_chart(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create annual cash flow comparison chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    baseline_cf = analysis_results["scenarios"]["baseline"]["cash_flows"] / 1e6
    ptc_cf = analysis_results["scenarios"]["ptc"]["cash_flows"] / 1e6
    itc_cf = analysis_results["scenarios"]["itc"]["cash_flows"] / 1e6

    years = np.arange(len(baseline_cf))

    # Plot cash flows
    ax.plot(years, baseline_cf, label='Baseline (No Incentives)',
            linewidth=2, color='red', alpha=0.8)
    ax.plot(years, ptc_cf, label='45Y PTC ($30/MWh)',
            linewidth=2, color='green', alpha=0.8)
    ax.plot(years, itc_cf, label='48E ITC (50% Credit)',
            linewidth=2, color='blue', alpha=0.8)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Highlight construction period
    construction_years = analysis_results["analysis_parameters"]["construction_period_years"]
    ax.axvspan(0, construction_years-1, alpha=0.2,
               color='gray', label='Construction Period')

    ax.set_title(f'{project_name}\nAnnual Cash Flow Comparison by Tax Incentive Scenario',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Cash Flow (Million USD)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cash_flow_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_tax_incentive_value_chart(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create tax incentive value breakdown chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PTC Value Breakdown
    ptc_analysis = analysis_results["scenarios"]["ptc"]["analysis"]
    if "tax_benefits" in ptc_analysis and "ptc" in ptc_analysis["tax_benefits"]:
        ptc_details = ptc_analysis["tax_benefits"]["ptc"]

        # Annual PTC benefits over time
        ptc_years = ptc_details["ptc_eligible_years"]
        ptc_annual_benefit = ptc_details["annual_ptc_benefit_usd"] / 1e6

        ax1.bar(range(len(ptc_years)), [ptc_annual_benefit] * len(ptc_years),
                color='green', alpha=0.7, label=f'${ptc_annual_benefit:.1f}M/year')
        ax1.set_title('45Y PTC Benefits Over Time', fontweight='bold')
        ax1.set_xlabel('Eligible Years')
        ax1.set_ylabel('Annual Benefit (Million USD)')
        ax1.set_xticks(range(len(ptc_years)))
        ax1.set_xticklabels(
            [f'Year {y+1}' for y in range(len(ptc_years))], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # ITC Value Breakdown
    itc_analysis = analysis_results["scenarios"]["itc"]["analysis"]
    if "tax_benefits" in itc_analysis and "itc" in itc_analysis["tax_benefits"]:
        itc_details = itc_analysis["tax_benefits"]["itc"]

        # Component breakdown
        components = []
        values = []
        for comp, value in itc_details["component_qualified_capex"].items():
            if value > 0:
                components.append(comp.replace('_', ' '))
                values.append(value / 1e6)

        bars = ax2.barh(components, values, color='blue', alpha=0.7)
        ax2.set_title('48E ITC Qualified CAPEX by Component',
                      fontweight='bold')
        ax2.set_xlabel('Qualified CAPEX (Million USD)')

        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax2.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                     f'${value:.0f}M', ha='left', va='center')

        ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'tax_incentive_value_breakdown.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_financial_metrics_dashboard(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create comprehensive financial metrics dashboard."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    scenarios = ['Baseline', '45Y PTC', '48E ITC']
    colors = ['lightcoral', 'lightgreen', 'lightblue']

    # NPV Comparison
    npv_values = [
        analysis_results["scenarios"]["baseline"]["financial_metrics"]["npv_usd"] / 1e6,
        analysis_results["scenarios"]["ptc"]["financial_metrics"]["npv_usd"] / 1e6,
        analysis_results["scenarios"]["itc"]["financial_metrics"]["npv_usd"] / 1e6
    ]

    bars1 = ax1.bar(scenarios, npv_values, color=colors, alpha=0.8)
    ax1.set_title('Net Present Value (NPV)', fontweight='bold')
    ax1.set_ylabel('NPV (Million USD)')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars1, npv_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(npv_values) * 0.02,
                 f'${value:.0f}M', ha='center', va='bottom', fontweight='bold')

    # IRR Comparison
    irr_values = []
    for scenario in ["baseline", "ptc", "itc"]:
        irr = analysis_results["scenarios"][scenario]["financial_metrics"].get(
            "irr_percent")
        irr_values.append(irr if irr is not None and not np.isnan(irr) else 0)

    bars2 = ax2.bar(scenarios, irr_values, color=colors, alpha=0.8)
    ax2.set_title('Internal Rate of Return (IRR)', fontweight='bold')
    ax2.set_ylabel('IRR (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars2, irr_values):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(irr_values) * 0.02,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Payback Period Comparison
    payback_values = []
    for scenario in ["baseline", "ptc", "itc"]:
        payback = analysis_results["scenarios"][scenario]["financial_metrics"].get(
            "payback_period_years")
        payback_values.append(payback if payback is not None else 0)

    bars3 = ax3.bar(scenarios, payback_values, color=colors, alpha=0.8)
    ax3.set_title('Payback Period', fontweight='bold')
    ax3.set_ylabel('Years')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars3, payback_values):
        if value > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(payback_values) * 0.02,
                     f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # ROI Comparison
    roi_values = [
        analysis_results["scenarios"]["baseline"]["financial_metrics"].get(
            "roi_percent", 0),
        analysis_results["scenarios"]["ptc"]["financial_metrics"].get(
            "roi_percent", 0),
        analysis_results["scenarios"]["itc"]["financial_metrics"].get(
            "roi_percent", 0)
    ]

    bars4 = ax4.bar(scenarios, roi_values, color=colors, alpha=0.8)
    ax4.set_title('Return on Investment (ROI)', fontweight='bold')
    ax4.set_ylabel('ROI (%)')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars4, roi_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(roi_values) * 0.02,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'{project_name}\nFinancial Metrics Dashboard',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'financial_metrics_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_annual_cash_flow_heatmap(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create annual cash flow heatmap showing differences between scenarios."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepare data
    baseline_cf = analysis_results["scenarios"]["baseline"]["cash_flows"]
    ptc_cf = analysis_results["scenarios"]["ptc"]["cash_flows"]
    itc_cf = analysis_results["scenarios"]["itc"]["cash_flows"]

    # Calculate differences from baseline
    ptc_diff = (ptc_cf - baseline_cf) / 1e6
    itc_diff = (itc_cf - baseline_cf) / 1e6

    # Create data matrix (show first 30 years)
    max_years = min(30, len(baseline_cf))
    data = np.array([
        baseline_cf[:max_years] / 1e6,
        ptc_diff[:max_years],
        itc_diff[:max_years]
    ])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    # Set labels
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(
        ['Baseline Cash Flow', 'PTC Difference', 'ITC Difference'])
    ax.set_xlabel('Year')
    ax.set_title(f'{project_name}\nAnnual Cash Flow Analysis (Million USD)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cash Flow (Million USD)', rotation=270, labelpad=20)

    # Add text annotations for key values
    for i in range(data.shape[0]):
        # Show values for first 20 years
        for j in range(min(20, data.shape[1])):
            if abs(data[i, j]) > 0.1:  # Only show significant values
                text = ax.text(j, i, f'{data[i, j]:.0f}', ha="center", va="center",
                               color="white" if abs(data[i, j]) > np.max(
                                   abs(data)) * 0.5 else "black",
                               fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'annual_cash_flow_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _create_cumulative_cash_flow_chart(analysis_results: Dict, output_dir: Path, project_name: str):
    """Create cumulative cash flow chart for all scenarios."""
    fig, ax = plt.subplots(figsize=(14, 8))

    baseline_cf = analysis_results["scenarios"]["baseline"]["cash_flows"]
    ptc_cf = analysis_results["scenarios"]["ptc"]["cash_flows"]
    itc_cf = analysis_results["scenarios"]["itc"]["cash_flows"]

    # Calculate cumulative cash flows
    baseline_cumulative = np.cumsum(baseline_cf) / 1e6
    ptc_cumulative = np.cumsum(ptc_cf) / 1e6
    itc_cumulative = np.cumsum(itc_cf) / 1e6

    years = np.arange(len(baseline_cf))

    # Plot cumulative cash flows
    ax.plot(years, baseline_cumulative, label='Baseline (No Incentives)',
            linewidth=3, color='red', alpha=0.8)
    ax.plot(years, ptc_cumulative, label='45Y PTC ($30/MWh)',
            linewidth=3, color='green', alpha=0.8)
    ax.plot(years, itc_cumulative, label='48E ITC (50% Credit)',
            linewidth=3, color='blue', alpha=0.8)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Highlight construction period
    construction_years = analysis_results["analysis_parameters"]["construction_period_years"]
    ax.axvspan(0, construction_years-1, alpha=0.2,
               color='gray', label='Construction Period')

    # Mark breakeven points
    for name, cumulative, color in [('Baseline', baseline_cumulative, 'red'),
                                    ('PTC', ptc_cumulative, 'green'),
                                    ('ITC', itc_cumulative, 'blue')]:
        breakeven_year = None
        for i, value in enumerate(cumulative):
            if value > 0:
                breakeven_year = i
                break
        if breakeven_year is not None:
            ax.plot(breakeven_year, 0, 'o', color=color, markersize=8,
                    label=f'{name} Breakeven: Year {breakeven_year}')

    ax.set_title(f'{project_name}\nCumulative Cash Flow Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Cash Flow (Million USD)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_cash_flow.png',
                dpi=300, bbox_inches='tight')
    plt.close()
