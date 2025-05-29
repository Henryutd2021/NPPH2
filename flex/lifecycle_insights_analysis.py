#!/usr/bin/env python3
"""
Comprehensive Lifecycle Insights Analysis
Deep dive into project lifetime relationships and optimization opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_tea_analyzer import EnhancedTEAAnalyzer
from pathlib import Path


def analyze_lifecycle_insights():
    """Comprehensive lifecycle analysis with deep insights"""

    print("🔍 COMPREHENSIVE LIFECYCLE INSIGHTS ANALYSIS")
    print("=" * 60)

    # Load data
    analyzer = EnhancedTEAAnalyzer('../output/tea')
    df = analyzer.collect_comprehensive_data()
    df = analyzer.calculate_enhanced_metrics(df)

    print(f"📊 Total projects analyzed: {len(df)}")
    print(
        f"📊 Projects with lifetime data: {df['project_lifetime_years'].notna().sum()}")

    # Create lifetime groups for analysis
    df['lifetime_group'] = pd.cut(df['project_lifetime_years'],
                                  bins=[0, 15, 20, 25, 35],
                                  labels=['Short (≤15y)', 'Medium (16-20y)', 'Long (21-25y)', 'Very Long (>25y)'])

    # Calculate integration ratio
    if 'electrolyzer_capacity_mw' in df.columns and 'turbine_capacity_mw' in df.columns:
        df['integration_ratio'] = df['electrolyzer_capacity_mw'] / \
            df['turbine_capacity_mw']

    # === KEY CORRELATIONS ===
    print("\n🔗 KEY CORRELATIONS WITH PROJECT LIFETIME:")
    print("-" * 50)

    correlation_metrics = [
        ('lcoh_usd_per_kg', 'LCOH (USD/kg)'),
        ('electrolyzer_cf_percent', 'Electrolyzer Capacity Factor (%)'),
        ('irr_percent', 'IRR (%)'),
        ('npv_usd', 'NPV (USD)'),
        ('as_revenue_share', 'AS Revenue Share (%)'),
        ('integration_ratio', 'Integration Ratio'),
        ('turbine_capacity_mw', 'Nuclear Capacity (MW)'),
        ('h2_annual_production_kg', 'H2 Production (kg/year)')
    ]

    correlations = {}
    for metric, label in correlation_metrics:
        if metric in df.columns:
            corr = df['project_lifetime_years'].corr(df[metric])
            correlations[metric] = corr
            strength = "Strong" if abs(
                corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
            direction = "Positive" if corr > 0 else "Negative"
            print(f"  {label:30}: {corr:6.3f} ({strength} {direction})")

    # === LIFETIME GROUP ANALYSIS ===
    print("\n📈 PERFORMANCE BY LIFETIME GROUPS:")
    print("-" * 50)

    key_metrics = ['lcoh_usd_per_kg', 'electrolyzer_cf_percent',
                   'irr_percent', 'integration_ratio']
    available_metrics = [m for m in key_metrics if m in df.columns]

    for metric in available_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        group_stats = df.groupby('lifetime_group')[
            metric].agg(['mean', 'std', 'count'])
        for group in group_stats.index:
            if not pd.isna(group_stats.loc[group, 'mean']):
                print(
                    f"  {group:15}: {group_stats.loc[group, 'mean']:6.3f} ± {group_stats.loc[group, 'std']:5.3f} (n={group_stats.loc[group, 'count']})")

    # === OPTIMIZATION INSIGHTS ===
    print("\n💡 OPTIMIZATION INSIGHTS:")
    print("-" * 50)

    # Find optimal lifetime for LCOH
    if 'lcoh_usd_per_kg' in df.columns:
        optimal_lifetime_lcoh = df.loc[df['lcoh_usd_per_kg'].idxmin(
        ), 'project_lifetime_years']
        min_lcoh = df['lcoh_usd_per_kg'].min()
        print(
            f"  🎯 Optimal lifetime for LCOH: {optimal_lifetime_lcoh} years (LCOH: ${min_lcoh:.3f}/kg)")

        # LCOH improvement potential
        short_term_avg_lcoh = df[df['project_lifetime_years']
                                 <= 15]['lcoh_usd_per_kg'].mean()
        long_term_avg_lcoh = df[df['project_lifetime_years']
                                >= 25]['lcoh_usd_per_kg'].mean()
        if pd.notna(short_term_avg_lcoh) and pd.notna(long_term_avg_lcoh):
            improvement = (short_term_avg_lcoh -
                           long_term_avg_lcoh) / short_term_avg_lcoh * 100
            print(f"  📉 LCOH improvement (long vs short): {improvement:.1f}%")

    # Find optimal lifetime for IRR
    if 'irr_percent' in df.columns:
        optimal_lifetime_irr = df.loc[df['irr_percent'].idxmax(
        ), 'project_lifetime_years']
        max_irr = df['irr_percent'].max()
        print(
            f"  🎯 Optimal lifetime for IRR: {optimal_lifetime_irr} years (IRR: {max_irr:.1f}%)")

    # Integration ratio insights
    if 'integration_ratio' in df.columns:
        avg_integration = df['integration_ratio'].mean()
        print(f"  ⚙️  Average integration ratio: {avg_integration:.1%}")

        # Correlation with lifetime
        int_corr = df['project_lifetime_years'].corr(df['integration_ratio'])
        if pd.notna(int_corr):
            print(f"  ⚙️  Integration-lifetime correlation: {int_corr:.3f}")

    # === ECONOMIC INSIGHTS ===
    print("\n💰 ECONOMIC INSIGHTS:")
    print("-" * 50)

    # Revenue diversification by lifetime
    if 'as_revenue_share' in df.columns:
        short_as = df[df['project_lifetime_years']
                      <= 15]['as_revenue_share'].mean()
        long_as = df[df['project_lifetime_years']
                     >= 25]['as_revenue_share'].mean()
        if pd.notna(short_as) and pd.notna(long_as):
            print(f"  📊 AS revenue share - Short projects: {short_as:.1f}%")
            print(f"  📊 AS revenue share - Long projects: {long_as:.1f}%")
            print(f"  📊 AS revenue difference: {long_as - short_as:+.1f}%")

    # NPV analysis
    if 'npv_usd' in df.columns:
        short_npv = df[df['project_lifetime_years'] <= 15]['npv_usd'].mean()
        long_npv = df[df['project_lifetime_years'] >= 25]['npv_usd'].mean()
        if pd.notna(short_npv) and pd.notna(long_npv):
            print(f"  💵 NPV - Short projects: ${short_npv/1e9:.2f}B")
            print(f"  💵 NPV - Long projects: ${long_npv/1e9:.2f}B")
            print(f"  💵 NPV improvement: ${(long_npv - short_npv)/1e9:+.2f}B")

    # === TECHNOLOGY INSIGHTS ===
    print("\n🔬 TECHNOLOGY INSIGHTS:")
    print("-" * 50)

    # Capacity factor trends
    if 'electrolyzer_cf_percent' in df.columns:
        cf_corr = df['project_lifetime_years'].corr(
            df['electrolyzer_cf_percent'])
        print(f"  ⚡ Electrolyzer CF-lifetime correlation: {cf_corr:.3f}")

        high_cf_projects = df[df['electrolyzer_cf_percent'] > 95]
        if len(high_cf_projects) > 0:
            avg_lifetime_high_cf = high_cf_projects['project_lifetime_years'].mean(
            )
            print(
                f"  ⚡ Average lifetime of high-CF projects (>95%): {avg_lifetime_high_cf:.1f} years")

    # System size trends
    if 'turbine_capacity_mw' in df.columns:
        size_corr = df['project_lifetime_years'].corr(
            df['turbine_capacity_mw'])
        print(f"  🏭 Nuclear capacity-lifetime correlation: {size_corr:.3f}")

        large_plants = df[df['turbine_capacity_mw'] > 1000]
        if len(large_plants) > 0:
            avg_lifetime_large = large_plants['project_lifetime_years'].mean()
            print(
                f"  🏭 Average lifetime of large plants (>1000MW): {avg_lifetime_large:.1f} years")

    # === REGIONAL INSIGHTS ===
    print("\n🌍 REGIONAL INSIGHTS:")
    print("-" * 50)

    if 'iso' in df.columns:
        regional_lifetime = df.groupby(
            'iso')['project_lifetime_years'].agg(['mean', 'count'])
        regional_lcoh = df.groupby('iso')['lcoh_usd_per_kg'].mean(
        ) if 'lcoh_usd_per_kg' in df.columns else None

        print("  Region-wise lifetime and LCOH:")
        for iso in regional_lifetime.index:
            lifetime_avg = regional_lifetime.loc[iso, 'mean']
            count = regional_lifetime.loc[iso, 'count']
            lcoh_avg = regional_lcoh.loc[iso] if regional_lcoh is not None else None

            lcoh_str = f", LCOH: ${lcoh_avg:.3f}/kg" if pd.notna(
                lcoh_avg) else ""
            print(
                f"    {iso:6}: {lifetime_avg:5.1f} years (n={count}){lcoh_str}")

    # === RECOMMENDATIONS ===
    print("\n🎯 STRATEGIC RECOMMENDATIONS:")
    print("-" * 50)

    recommendations = []

    # Based on LCOH correlation
    if 'lcoh_usd_per_kg' in correlations and correlations['lcoh_usd_per_kg'] < -0.5:
        recommendations.append(
            "✅ Prioritize longer project lifetimes (>25 years) for significant LCOH reduction")

    # Based on integration ratio
    if 'integration_ratio' in df.columns:
        optimal_integration = df.loc[df['lcoh_usd_per_kg'].idxmin(
        ), 'integration_ratio'] if 'lcoh_usd_per_kg' in df.columns else None
        if pd.notna(optimal_integration):
            recommendations.append(
                f"✅ Target integration ratio around {optimal_integration:.1%} for optimal economics")

    # Based on capacity factors
    if 'electrolyzer_cf_percent' in correlations and correlations['electrolyzer_cf_percent'] > 0.2:
        recommendations.append(
            "✅ Longer projects tend to have better electrolyzer utilization - plan for sustained operations")

    # Based on regional analysis
    if 'iso' in df.columns and 'lcoh_usd_per_kg' in df.columns:
        best_region = df.groupby('iso')['lcoh_usd_per_kg'].mean().idxmin()
        recommendations.append(
            f"✅ Consider {best_region} region for deployment - shows best LCOH performance")

    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 60)
    print("🎉 LIFECYCLE INSIGHTS ANALYSIS COMPLETED")

    return df, correlations


if __name__ == "__main__":
    df, correlations = analyze_lifecycle_insights()
