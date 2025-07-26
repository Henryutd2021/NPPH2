#!/usr/bin/env python3
"""
TEA CS1 Sensitivity Analysis Results Analyzer
Analyzes and summarizes results from sensitivity analysis runs
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from datetime import datetime
import argparse

# Setup Python paths for importing src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from path_setup import setup_src_paths
setup_src_paths()

# Configuration
SENSITIVITY_BASE_DIR = Path(__file__).parent.parent / \
    'output' / 'tea' / 'cs1_sensitivity'
ANALYSIS_OUTPUT_DIR = Path(__file__).parent.parent / \
    'output' / 'tea' / 'cs1_sensitivity_analysis'

# Parameter values tested
PARAMETER_VALUES = [170000, 200000, 260000, 290000, 320000]


def create_analysis_directories():
    """Create directories for analysis results"""
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_OUTPUT_DIR / 'summary_tables', exist_ok=True)
    os.makedirs(ANALYSIS_OUTPUT_DIR / 'comparison_reports', exist_ok=True)


def extract_metrics_from_report(report_file: Path) -> dict:
    """Extract key metrics from TEA summary report"""
    metrics = {}

    if not report_file.exists():
        return metrics

    try:
        with open(report_file, 'r') as f:
            content = f.read()

        # Extract key financial metrics using regex
        patterns = {
            'npv': r'NPV\s*:\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            'irr': r'IRR\s*:\s*([-+]?\d+(?:\.\d+)?)',
            'payback_period': r'Payback Period\s*:\s*([-+]?\d+(?:\.\d+)?)',
            'lcoh': r'LCOH\s*:\s*\$?([-+]?\d+(?:\.\d+)?)',
            'total_capex': r'Total CAPEX\s*:\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            'total_opex': r'Total OPEX\s*:\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            'electrolyzer_capacity': r'Electrolyzer Capacity\s*:\s*([-+]?\d+(?:\.\d+)?)',
            'h2_production': r'H2 Production\s*:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            'energy_revenue': r'Energy Revenue\s*:\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            'h2_revenue': r'H2 Revenue\s*:\s*\$?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    metrics[key] = float(value_str)
                except ValueError:
                    metrics[key] = None
            else:
                metrics[key] = None

    except Exception as e:
        print(f"Error reading report {report_file}: {e}")

    return metrics


def collect_sensitivity_results():
    """Collect results from all sensitivity analysis runs"""
    results_data = []

    for param_value in PARAMETER_VALUES:
        param_dir = SENSITIVITY_BASE_DIR / f"fixed_costs_{param_value}"

        if not param_dir.exists():
            print(
                f"Warning: Directory not found for parameter {param_value}: {param_dir}")
            continue

        # Find all reactor directories
        reactor_dirs = [d for d in param_dir.iterdir() if d.is_dir()]

        for reactor_dir in reactor_dirs:
            # Extract reactor info from directory name
            reactor_match = re.match(
                r'(.+?)_(\d+)_(.+?)_(\d+)', reactor_dir.name)
            if not reactor_match:
                continue

            plant_name, generator_id, iso_region, remaining_years = reactor_match.groups()

            # Look for summary reports
            report_patterns = [
                f"{iso_region}_TEA_Summary_Report.txt",
                f"{iso_region}_Comprehensive_TEA_Summary.txt"
            ]

            metrics = {}
            for pattern in report_patterns:
                report_file = reactor_dir / pattern
                if report_file.exists():
                    metrics = extract_metrics_from_report(report_file)
                    break

            # Add reactor and parameter info
            result_record = {
                'parameter_value': param_value,
                'plant_name': plant_name,
                'generator_id': generator_id,
                'iso_region': iso_region,
                'remaining_years': int(remaining_years),
                'reactor_name': f"{plant_name}_{generator_id}",
                'result_directory': str(reactor_dir),
                **metrics
            }

            results_data.append(result_record)

    return pd.DataFrame(results_data)


def generate_summary_tables(results_df: pd.DataFrame):
    """Generate summary tables for sensitivity analysis"""

    # 1. Parameter sensitivity summary
    param_summary = results_df.groupby('parameter_value').agg({
        'npv': ['count', 'mean', 'std', 'min', 'max'],
        'irr': ['mean', 'std', 'min', 'max'],
        'lcoh': ['mean', 'std', 'min', 'max'],
        'payback_period': ['mean', 'std', 'min', 'max']
    }).round(2)

    param_summary.columns = ['_'.join(col).strip()
                             for col in param_summary.columns]
    param_summary.to_csv(ANALYSIS_OUTPUT_DIR /
                         'summary_tables' / 'parameter_sensitivity_summary.csv')

    # 2. Reactor performance by parameter
    reactor_performance = results_df.pivot_table(
        values=['npv', 'irr', 'lcoh', 'payback_period'],
        index=['reactor_name', 'iso_region'],
        columns='parameter_value',
        aggfunc='mean'
    ).round(2)

    reactor_performance.to_csv(
        ANALYSIS_OUTPUT_DIR / 'summary_tables' / 'reactor_performance_by_parameter.csv')

    # 3. ISO region analysis
    iso_analysis = results_df.groupby(['iso_region', 'parameter_value']).agg({
        'npv': ['count', 'mean', 'std'],
        'irr': ['mean', 'std'],
        'lcoh': ['mean', 'std']
    }).round(2)

    iso_analysis.columns = ['_'.join(col).strip()
                            for col in iso_analysis.columns]
    iso_analysis.to_csv(ANALYSIS_OUTPUT_DIR /
                        'summary_tables' / 'iso_region_analysis.csv')

    # 4. Best and worst performing reactors
    best_worst = []
    for param in PARAMETER_VALUES:
        param_data = results_df[results_df['parameter_value'] == param]
        if not param_data.empty:
            best_npv = param_data.loc[param_data['npv'].idxmax()]
            worst_npv = param_data.loc[param_data['npv'].idxmin()]

            best_worst.append({
                'parameter_value': param,
                'metric': 'NPV',
                'best_reactor': best_npv['reactor_name'],
                'best_value': best_npv['npv'],
                'worst_reactor': worst_npv['reactor_name'],
                'worst_value': worst_npv['npv']
            })

    best_worst_df = pd.DataFrame(best_worst)
    best_worst_df.to_csv(ANALYSIS_OUTPUT_DIR / 'summary_tables' /
                         'best_worst_performers.csv', index=False)

    print(
        f"✅ Summary tables generated in: {ANALYSIS_OUTPUT_DIR / 'summary_tables'}")


def generate_comparison_report(results_df: pd.DataFrame):
    """Generate detailed comparison report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = ANALYSIS_OUTPUT_DIR / 'comparison_reports' / \
        'sensitivity_analysis_comparison.txt'

    with open(report_file, 'w') as f:
        f.write("TEA CS1 Sensitivity Analysis Comparison Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Parameter: total_fixed_costs_per_mw_year\n")
        f.write(f"Parameter Values: {PARAMETER_VALUES}\n")
        f.write(f"Total Results: {len(results_df)} reactor analyses\n\n")

        # Overall statistics
        f.write("Overall Statistics by Parameter Value:\n")
        f.write("-" * 40 + "\n")

        for param in PARAMETER_VALUES:
            param_data = results_df[results_df['parameter_value'] == param]
            if not param_data.empty:
                f.write(f"\nParameter Value: ${param:,}/MW/year\n")
                f.write(f"  Number of Reactors: {len(param_data)}\n")
                f.write(f"  Average NPV: ${param_data['npv'].mean():,.0f}\n")
                f.write(f"  Average IRR: {param_data['irr'].mean():.2f}%\n")
                f.write(
                    f"  Average LCOH: ${param_data['lcoh'].mean():.2f}/kg\n")
                f.write(
                    f"  Average Payback: {param_data['payback_period'].mean():.1f} years\n")

                # Count positive/negative NPV
                positive_npv = len(param_data[param_data['npv'] > 0])
                negative_npv = len(param_data[param_data['npv'] <= 0])
                f.write(
                    f"  Positive NPV: {positive_npv} ({positive_npv/len(param_data)*100:.1f}%)\n")
                f.write(
                    f"  Negative NPV: {negative_npv} ({negative_npv/len(param_data)*100:.1f}%)\n")

        # Sensitivity analysis
        f.write("\n" + "=" * 60 + "\n")
        f.write("Sensitivity Analysis Results:\n")
        f.write("-" * 30 + "\n")

        # Calculate percentage changes relative to base case (230,000)
        base_param = 230000
        if base_param in results_df['parameter_value'].values:
            base_data = results_df[results_df['parameter_value'] == base_param]
            base_avg_npv = base_data['npv'].mean()
            base_avg_lcoh = base_data['lcoh'].mean()

            f.write(f"\nBase Case (${base_param:,}/MW/year):\n")
            f.write(f"  Average NPV: ${base_avg_npv:,.0f}\n")
            f.write(f"  Average LCOH: ${base_avg_lcoh:.2f}/kg\n\n")

            f.write("Sensitivity to Parameter Changes:\n")
            for param in sorted(PARAMETER_VALUES):
                if param != base_param:
                    param_data = results_df[results_df['parameter_value'] == param]
                    if not param_data.empty:
                        param_avg_npv = param_data['npv'].mean()
                        param_avg_lcoh = param_data['lcoh'].mean()

                        npv_change = (
                            (param_avg_npv - base_avg_npv) / abs(base_avg_npv)) * 100
                        lcoh_change = (
                            (param_avg_lcoh - base_avg_lcoh) / base_avg_lcoh) * 100

                        f.write(
                            f"  ${param:,}/MW/year: NPV {npv_change:+.1f}%, LCOH {lcoh_change:+.1f}%\n")

        # ISO region breakdown
        f.write("\n" + "=" * 60 + "\n")
        f.write("Results by ISO Region:\n")
        f.write("-" * 25 + "\n")

        for iso in sorted(results_df['iso_region'].unique()):
            iso_data = results_df[results_df['iso_region'] == iso]
            f.write(f"\n{iso} Region:\n")
            f.write(f"  Total Reactors: {len(iso_data)}\n")

            for param in PARAMETER_VALUES:
                param_iso_data = iso_data[iso_data['parameter_value'] == param]
                if not param_iso_data.empty:
                    avg_npv = param_iso_data['npv'].mean()
                    avg_lcoh = param_iso_data['lcoh'].mean()
                    f.write(
                        f"    ${param:,}/MW/year: NPV=${avg_npv:,.0f}, LCOH=${avg_lcoh:.2f}/kg\n")

        # Top performers
        f.write("\n" + "=" * 60 + "\n")
        f.write("Top Performing Reactors (by NPV):\n")
        f.write("-" * 35 + "\n")

        top_reactors = results_df.nlargest(10, 'npv')
        for i, row in top_reactors.iterrows():
            f.write(
                f"  {row['reactor_name']} ({row['iso_region']}) - ${row['parameter_value']:,}/MW/year\n")
            f.write(
                f"    NPV: ${row['npv']:,.0f}, IRR: {row['irr']:.2f}%, LCOH: ${row['lcoh']:.2f}/kg\n")

    print(f"✅ Comparison report generated: {report_file}")


def generate_json_summary(results_df: pd.DataFrame):
    """Generate JSON summary for programmatic access"""

    summary_data = {
        'analysis_info': {
            'generated_timestamp': datetime.now().isoformat(),
            'parameter_name': 'total_fixed_costs_per_mw_year',
            'parameter_values': PARAMETER_VALUES,
            'total_analyses': len(results_df),
            'unique_reactors': len(results_df['reactor_name'].unique()),
            'iso_regions': sorted(results_df['iso_region'].unique().tolist())
        },
        'parameter_summary': {},
        'reactor_summary': {},
        'iso_summary': {}
    }

    # Parameter summary
    for param in PARAMETER_VALUES:
        param_data = results_df[results_df['parameter_value'] == param]
        if not param_data.empty:
            summary_data['parameter_summary'][str(param)] = {
                'count': len(param_data),
                'avg_npv': float(param_data['npv'].mean()),
                'avg_irr': float(param_data['irr'].mean()),
                'avg_lcoh': float(param_data['lcoh'].mean()),
                'avg_payback': float(param_data['payback_period'].mean()),
                'positive_npv_count': int(len(param_data[param_data['npv'] > 0])),
                'negative_npv_count': int(len(param_data[param_data['npv'] <= 0]))
            }

    # Reactor summary (best performance for each reactor)
    for reactor in results_df['reactor_name'].unique():
        reactor_data = results_df[results_df['reactor_name'] == reactor]
        best_result = reactor_data.loc[reactor_data['npv'].idxmax()]
        summary_data['reactor_summary'][reactor] = {
            'iso_region': best_result['iso_region'],
            'best_parameter_value': int(best_result['parameter_value']),
            'best_npv': float(best_result['npv']),
            'best_irr': float(best_result['irr']),
            'best_lcoh': float(best_result['lcoh'])
        }

    # ISO summary
    for iso in results_df['iso_region'].unique():
        iso_data = results_df[results_df['iso_region'] == iso]
        summary_data['iso_summary'][iso] = {
            'reactor_count': len(iso_data['reactor_name'].unique()),
            'total_analyses': len(iso_data),
            'avg_npv': float(iso_data['npv'].mean()),
            'avg_lcoh': float(iso_data['lcoh'].mean())
        }

    # Save JSON summary
    json_file = ANALYSIS_OUTPUT_DIR / 'sensitivity_analysis_summary.json'
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"✅ JSON summary generated: {json_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Analyze TEA CS1 sensitivity analysis results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    print("🔍 Starting TEA CS1 Sensitivity Analysis Results Analysis")
    print(f"📂 Input directory: {SENSITIVITY_BASE_DIR}")
    print(f"📂 Output directory: {ANALYSIS_OUTPUT_DIR}")
    print(f"📊 Parameter values: {PARAMETER_VALUES}")

    # Create analysis directories
    create_analysis_directories()

    # Collect results
    print("\n📊 Collecting sensitivity analysis results...")
    results_df = collect_sensitivity_results()

    if results_df.empty:
        print("❌ No sensitivity analysis results found!")
        return False

    print(f"✅ Collected {len(results_df)} analysis results")
    print(f"   - {len(results_df['reactor_name'].unique())} unique reactors")
    print(f"   - {len(results_df['iso_region'].unique())} ISO regions")
    print(
        f"   - {len(results_df['parameter_value'].unique())} parameter values")

    # Generate outputs
    print("\n📈 Generating analysis outputs...")

    # Save raw results
    results_df.to_csv(ANALYSIS_OUTPUT_DIR /
                      'raw_sensitivity_results.csv', index=False)
    print(
        f"✅ Raw results saved: {ANALYSIS_OUTPUT_DIR / 'raw_sensitivity_results.csv'}")

    # Generate summary tables
    generate_summary_tables(results_df)

    # Generate comparison report
    generate_comparison_report(results_df)

    # Generate JSON summary
    generate_json_summary(results_df)

    print(f"\n🎉 Analysis completed successfully!")
    print(f"📂 All results available in: {ANALYSIS_OUTPUT_DIR}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
