#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for Ancillary Services Analysis Results
Creates charts and plots to showcase system flexibility and AS capabilities
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_analysis_results(csv_file="../../output/ancillary/analysis_results.csv"):
    """Load the analysis results from CSV"""
    try:
        df = pd.read_csv(csv_file)
        print(
            f"Loaded analysis results: {len(df)} reactors, {len(df.columns)} metrics")
        return df
    except FileNotFoundError:
        print(
            f"Error: {csv_file} not found. Please run the analysis script first.")
        return None


def create_visualizations(df):
    """Create comprehensive visualizations of the analysis results"""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Service Diversity by ISO
    plt.subplot(3, 3, 1)
    iso_diversity = df.groupby('iso_region')['service_diversity_ratio'].agg([
        'mean', 'std']).reset_index()
    bars = plt.bar(iso_diversity['iso_region'], iso_diversity['mean'],
                   yerr=iso_diversity['std'], capsize=5, alpha=0.7)
    plt.title('Service Diversity Ratio by ISO Region',
              fontsize=12, fontweight='bold')
    plt.xlabel('ISO Region')
    plt.ylabel('Average Service Diversity Ratio')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Ancillary Revenue Share by ISO
    plt.subplot(3, 3, 2)
    iso_revenue = df.groupby('iso_region')['ancillary_revenue_share'].agg(
        ['mean', 'std']).reset_index()
    bars = plt.bar(iso_revenue['iso_region'], iso_revenue['mean'],
                   yerr=iso_revenue['std'], capsize=5, alpha=0.7, color='orange')
    plt.title('Ancillary Revenue Share by ISO Region',
              fontsize=12, fontweight='bold')
    plt.xlabel('ISO Region')
    plt.ylabel('Average Ancillary Revenue Share')
    plt.xticks(rotation=45)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 3. Overall Flexibility Score Distribution
    plt.subplot(3, 3, 3)
    plt.hist(df['overall_flexibility_score'], bins=15,
             alpha=0.7, color='green', edgecolor='black')
    plt.title('Overall Flexibility Score Distribution',
              fontsize=12, fontweight='bold')
    plt.xlabel('Flexibility Score')
    plt.ylabel('Number of Reactors')
    plt.axvline(df['overall_flexibility_score'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["overall_flexibility_score"].mean():.3f}')
    plt.legend()

    # 4. Grid Output Range vs Flexibility Score
    plt.subplot(3, 3, 4)
    scatter = plt.scatter(df['grid_output_range_MW'], df['overall_flexibility_score'],
                          c=df['iso_region'].astype('category').cat.codes, alpha=0.7, s=50)
    plt.title('Grid Output Range vs Flexibility Score',
              fontsize=12, fontweight='bold')
    plt.xlabel('Grid Output Range (MW)')
    plt.ylabel('Overall Flexibility Score')

    # Add ISO legend
    iso_regions = df['iso_region'].unique()
    for i, iso in enumerate(iso_regions):
        iso_data = df[df['iso_region'] == iso]
        plt.scatter([], [], label=iso, s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. Ramping Capabilities by ISO
    plt.subplot(3, 3, 5)
    ramping_data = df[['iso_region', 'max_ramp_up_MW_hr', 'max_ramp_down_MW_hr']].melt(
        id_vars='iso_region', var_name='Ramp_Type', value_name='Ramp_Rate')
    sns.boxplot(data=ramping_data, x='iso_region',
                y='Ramp_Rate', hue='Ramp_Type')
    plt.title('Ramping Capabilities by ISO Region',
              fontsize=12, fontweight='bold')
    plt.xlabel('ISO Region')
    plt.ylabel('Ramp Rate (MW/hr)')
    plt.xticks(rotation=45)
    plt.legend(title='Ramp Type')

    # 6. Electrolyzer vs Turbine Flexibility
    plt.subplot(3, 3, 6)
    plt.scatter(df['electrolyzer_capacity_factor'], df['turbine_flexibility_factor'],
                alpha=0.7, s=50, c='purple')
    plt.title('Electrolyzer vs Turbine Flexibility',
              fontsize=12, fontweight='bold')
    plt.xlabel('Electrolyzer Capacity Factor')
    plt.ylabel('Turbine Flexibility Factor')

    # Add correlation coefficient
    corr = df['electrolyzer_capacity_factor'].corr(
        df['turbine_flexibility_factor'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 7. Revenue Composition
    plt.subplot(3, 3, 7)
    revenue_cols = ['energy_revenue_avg_hourly',
                    'hydrogen_revenue_avg_hourly', 'ancillary_revenue_avg_hourly']
    revenue_means = [df[col].mean() for col in revenue_cols]
    revenue_labels = ['Energy', 'Hydrogen', 'Ancillary']
    colors = ['blue', 'green', 'orange']

    wedges, texts, autotexts = plt.pie(revenue_means, labels=revenue_labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    plt.title('Average Revenue Composition', fontsize=12, fontweight='bold')

    # 8. ISO Region Reactor Count and Performance
    plt.subplot(3, 3, 8)
    iso_stats = df.groupby('iso_region').agg({
        'reactor_name': 'count',
        'overall_flexibility_score': 'mean',
        'ancillary_revenue_share': 'mean'
    }).reset_index()

    x = np.arange(len(iso_stats))
    width = 0.25

    bars1 = plt.bar(
        x - width, iso_stats['reactor_name'], width, label='Reactor Count', alpha=0.7)
    bars2 = plt.bar(x, iso_stats['overall_flexibility_score'] * 100, width,
                    label='Flexibility Score (×100)', alpha=0.7)
    bars3 = plt.bar(x + width, iso_stats['ancillary_revenue_share'] * 100, width,
                    label='AS Revenue Share (×100)', alpha=0.7)

    plt.xlabel('ISO Region')
    plt.ylabel('Value')
    plt.title('ISO Region Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, iso_stats['iso_region'], rotation=45)
    plt.legend()

    # 9. Top Performers Analysis
    plt.subplot(3, 3, 9)
    # Create a composite score
    df['composite_score'] = (df['overall_flexibility_score'] * 0.4 +
                             df['service_diversity_ratio'] * 0.3 +
                             df['ancillary_revenue_share'] * 0.3)

    top_performers = df.nlargest(10, 'composite_score')[
        ['reactor_name', 'composite_score', 'iso_region']]

    bars = plt.barh(range(len(top_performers)), top_performers['composite_score'],
                    color=sns.color_palette("viridis", len(top_performers)))
    plt.yticks(range(len(top_performers)),
               [f"{name[:20]}..." if len(name) > 20 else name
                for name in top_performers['reactor_name']], fontsize=8)
    plt.xlabel('Composite Performance Score')
    plt.title('Top 10 Performing Reactors', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add ISO labels on bars
    for i, (idx, row) in enumerate(top_performers.iterrows()):
        plt.text(row['composite_score'] + 0.001, i, f"({row['iso_region']})",
                 va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('../../output/ancillary/charts.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def create_summary_table(df):
    """Create a summary table of key metrics"""

    print("\n" + "="*80)
    print("ANCILLARY SERVICES ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nDataset Overview:")
    print(f"  • Total Reactors Analyzed: {len(df)}")
    print(
        f"  • ISO Regions Covered: {len(df['iso_region'].unique())} ({', '.join(df['iso_region'].unique())})")
    print(f"  • Total Analysis Metrics: {len(df.columns)}")

    print(f"\nSystem Flexibility Metrics:")
    print(
        f"  • Average Flexibility Score: {df['overall_flexibility_score'].mean():.3f} ± {df['overall_flexibility_score'].std():.3f}")
    print(
        f"  • Average Service Diversity: {df['service_diversity_ratio'].mean():.3f} ({df['service_diversity_ratio'].mean()*100:.1f}% of available services)")
    print(
        f"  • Average Grid Output Range: {df['grid_output_range_MW'].mean():.1f} MW")
    print(
        f"  • Average Ramp Rate: {df['avg_ramp_rate_MW_hr'].mean():.1f} MW/hr")

    print(f"\nRevenue and Economic Metrics:")
    print(
        f"  • Average Ancillary Revenue Share: {df['ancillary_revenue_share'].mean():.1%}")
    print(
        f"  • Average Total Revenue: ${df['total_revenue_avg_hourly'].mean():,.0f}/hr")
    print(
        f"  • Average Ancillary Revenue: ${df['ancillary_revenue_avg_hourly'].mean():,.0f}/hr")

    print(f"\nISO-Specific Performance:")
    iso_summary = df.groupby('iso_region').agg({
        'reactor_name': 'count',
        'overall_flexibility_score': 'mean',
        'service_diversity_ratio': 'mean',
        'ancillary_revenue_share': 'mean'
    }).round(3)
    iso_summary.columns = ['Reactors', 'Flexibility Score',
                           'Service Diversity', 'AS Revenue Share']
    print(iso_summary.to_string())

    print(f"\nTop 5 Most Flexible Reactors:")
    top_flexible = df.nlargest(5, 'overall_flexibility_score')[
        ['reactor_name', 'iso_region', 'overall_flexibility_score']]
    for i, (_, row) in enumerate(top_flexible.iterrows(), 1):
        print(
            f"  {i}. {row['reactor_name']} ({row['iso_region']}) - Score: {row['overall_flexibility_score']:.3f}")

    print(f"\nTop 5 Highest AS Revenue Share:")
    top_revenue = df.nlargest(5, 'ancillary_revenue_share')[
        ['reactor_name', 'iso_region', 'ancillary_revenue_share']]
    for i, (_, row) in enumerate(top_revenue.iterrows(), 1):
        print(
            f"  {i}. {row['reactor_name']} ({row['iso_region']}) - Share: {row['ancillary_revenue_share']:.1%}")

    print("\n" + "="*80)


def main():
    """Main function to create visualizations and summaries"""

    # Load the analysis results
    df = load_analysis_results()
    if df is None:
        return

    print("Creating visualizations...")

    # Create visualizations
    fig = create_visualizations(df)

    # Create summary table
    create_summary_table(df)

    print(f"\nVisualization saved as: output/ancillary/charts.png")
    print(f"Analysis complete! Check the generated charts and summary above.")


if __name__ == "__main__":
    main()
