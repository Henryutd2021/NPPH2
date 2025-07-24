#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEA Breakdown数据提取分析脚本
专门分析各种breakdown类数据的提取完整性
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_breakdown_data(excel_file: str = "output/tea/tea_summary_analysis.xlsx"):
    """分析各种breakdown数据的提取情况"""

    logger.info(f"分析TEA Breakdown数据提取结果: {excel_file}")

    print("TEA Summary Report - Breakdown 数据提取分析")
    print("=" * 70)

    # 1. CAPEX Breakdown 分析
    print("\n🏗️ CAPEX Breakdown 分析")
    print("-" * 50)

    cost_df = pd.read_excel(excel_file, sheet_name="Cost_Analysis")
    capex_components = [
        'electrolyzer_system_capex',
        'h2_storage_system_capex',
        'grid_integration_capex',
        'npp_modifications_capex',
        'total_capex'
    ]

    print("CAPEX 组件提取情况:")
    total_reactors = len(cost_df)
    for comp in capex_components:
        if comp in cost_df.columns:
            missing = cost_df[comp].isnull().sum()
            success_rate = (total_reactors - missing) / total_reactors * 100
            print(
                f"  {comp}: {success_rate:.1f}% ({total_reactors-missing}/{total_reactors})")

    # 显示CAPEX分解示例
    print("\n💡 CAPEX Breakdown 示例 (前3个反应堆):")
    for i in range(min(3, len(cost_df))):
        reactor = cost_df.iloc[i]
        reactor_name = reactor['reactor_name']
        total = reactor['total_capex']
        print(f"\n  {reactor_name}:")
        print(f"    总CAPEX: ${total:,.0f}")

        for comp in capex_components[:-1]:  # 不包括total
            if comp in cost_df.columns:
                value = reactor[comp]
                percentage = (value / total * 100) if total > 0 else 0
                comp_name = comp.replace(
                    '_capex', '').replace('_', ' ').title()
                print(f"    {comp_name}: ${value:,.0f} ({percentage:.1f}%)")

    # 2. OPEX Breakdown 分析
    print(f"\n\n💰 OPEX Breakdown 分析")
    print("-" * 50)

    opex_components = [
        'vom_turbine_cost',
        'vom_electrolyzer_cost',
        'water_cost',
        'ramping_cost',
        'h2_storage_cycle_cost',
        'nuclear_plant_opex',
        'total_annual_opex'
    ]

    print("OPEX 组件提取情况:")
    for comp in opex_components:
        if comp in cost_df.columns:
            missing = cost_df[comp].isnull().sum()
            success_rate = (total_reactors - missing) / total_reactors * 100
            print(
                f"  {comp}: {success_rate:.1f}% ({total_reactors-missing}/{total_reactors})")

    # 显示OPEX分解示例
    print("\n💡 OPEX Breakdown 示例 (Seabrook_1_ISONE_25):")
    seabrook_row = cost_df[cost_df['reactor_name'] == 'Seabrook_1_ISONE_25']
    if not seabrook_row.empty:
        reactor = seabrook_row.iloc[0]
        total_opex = reactor['total_annual_opex']
        print(f"  总年度OPEX: ${total_opex:,.0f}")

        for comp in opex_components[:-1]:  # 不包括total
            if comp in cost_df.columns:
                value = reactor[comp]
                percentage = (value / total_opex *
                              100) if total_opex > 0 else 0
                comp_name = comp.replace('_cost', '').replace('_', ' ').title()
                print(f"  {comp_name}: ${value:,.0f} ({percentage:.1f}%)")

    # 3. LCOH Breakdown 分析
    print(f"\n\n⚡ LCOH Breakdown 分析")
    print("-" * 50)

    advanced_df = pd.read_excel(excel_file, sheet_name="Advanced_Financial")

    # LCOH 组件
    lcoh_components = [
        'total_lcoh_detailed',
        'lcoh_electricity_opportunity_cost',
        'lcoh_electrolyzer_system',
        'lcoh_vom_electrolyzer',
        'lcoh_fixed_om',
        'lcoh_hte_heat_opportunity_cost',
        'lcoh_water_cost',
        'lcoh_h2_storage_system',
        'lcoh_h2_storage_cycle_cost',
        'lcoh_grid_integration',
        'lcoh_npp_modifications',
        'lcoh_ramping_cost'
    ]

    print("LCOH 组件提取情况:")
    for comp in lcoh_components:
        if comp in advanced_df.columns:
            missing = advanced_df[comp].isnull().sum()
            success_rate = (total_reactors - missing) / total_reactors * 100
            print(
                f"  {comp}: {success_rate:.1f}% ({total_reactors-missing}/{total_reactors})")

    # LCOH 类别分析
    lcoh_categories = [
        'lcoh_capital_recovery_capex',
        'lcoh_electricity_costs',
        'lcoh_fixed_om_category',
        'lcoh_variable_opex',
        'lcoh_equipment_replacements'
    ]

    print("\nLCOH 成本类别提取情况:")
    for comp in lcoh_categories:
        if comp in advanced_df.columns:
            missing = advanced_df[comp].isnull().sum()
            success_rate = (total_reactors - missing) / total_reactors * 100
            print(
                f"  {comp}: {success_rate:.1f}% ({total_reactors-missing}/{total_reactors})")

    # 显示LCOH分解示例
    print("\n💡 LCOH Breakdown 示例 (Seabrook_1_ISONE_25):")
    seabrook_advanced = advanced_df[advanced_df['reactor_name']
                                    == 'Seabrook_1_ISONE_25']
    if not seabrook_advanced.empty:
        reactor = seabrook_advanced.iloc[0]
        total_lcoh = reactor['total_lcoh_detailed']
        print(f"  总LCOH: ${total_lcoh:.3f}/kg")

        # 显示主要组件
        main_components = [
            'lcoh_electricity_opportunity_cost',
            'lcoh_electrolyzer_system',
            'lcoh_vom_electrolyzer',
            'lcoh_fixed_om',
            'lcoh_hte_heat_opportunity_cost'
        ]

        for comp in main_components:
            if comp in advanced_df.columns:
                value = reactor[comp]
                if not pd.isna(value):
                    percentage = (value / total_lcoh *
                                  100) if total_lcoh > 0 else 0
                    comp_name = comp.replace(
                        'lcoh_', '').replace('_', ' ').title()
                    print(f"  {comp_name}: ${value:.3f}/kg ({percentage:.1f}%)")

    # 4. Revenue Breakdown 分析
    print(f"\n\n💵 Revenue Breakdown 分析")
    print("-" * 50)

    financial_df = pd.read_excel(
        excel_file, sheet_name="Financial_Performance")

    revenue_components = [
        'total_annual_revenue',
        'energy_revenue',
        'h2_sales_revenue',
        'h2_subsidy_revenue',
        'ancillary_services_revenue'
    ]

    print("Revenue 组件提取情况:")
    for comp in revenue_components:
        if comp in financial_df.columns:
            missing = financial_df[comp].isnull().sum()
            success_rate = (total_reactors - missing) / total_reactors * 100
            print(
                f"  {comp}: {success_rate:.1f}% ({total_reactors-missing}/{total_reactors})")

    # 显示Revenue分解示例
    print("\n💡 Revenue Breakdown 示例 (Seabrook_1_ISONE_25):")
    seabrook_financial = financial_df[financial_df['reactor_name']
                                      == 'Seabrook_1_ISONE_25']
    if not seabrook_financial.empty:
        reactor = seabrook_financial.iloc[0]
        total_revenue = reactor['total_annual_revenue']
        print(f"  总年度收入: ${total_revenue:,.0f}")

        for comp in revenue_components[1:]:  # 不包括total
            if comp in financial_df.columns:
                value = reactor[comp]
                percentage = (value / total_revenue *
                              100) if total_revenue > 0 else 0
                comp_name = comp.replace(
                    '_revenue', '').replace('_', ' ').title()
                print(f"  {comp_name}: ${value:,.0f} ({percentage:.1f}%)")

    # 5. 总体 Breakdown 质量评估
    print(f"\n\n📊 总体 Breakdown 数据质量评估")
    print("=" * 50)

    # 统计完全成功提取的breakdown类型
    breakdown_success = {
        'CAPEX Breakdown': all(
            (cost_df[comp].isnull().sum() ==
             0) if comp in cost_df.columns else False
            for comp in capex_components
        ),
        'OPEX Breakdown': all(
            (cost_df[comp].isnull().sum() ==
             0) if comp in cost_df.columns else False
            for comp in opex_components
        ),
        'Revenue Breakdown': all(
            (financial_df[comp].isnull().sum() ==
             0) if comp in financial_df.columns else False
            for comp in revenue_components
        ),
        'LCOH Component Breakdown': sum(
            (advanced_df[comp].isnull().sum() ==
             0) if comp in advanced_df.columns else 0
            for comp in lcoh_components
        ) >= 10,  # 至少10个组件成功
        'LCOH Category Breakdown': all(
            (advanced_df[comp].isnull().sum() ==
             0) if comp in advanced_df.columns else False
            for comp in lcoh_categories
        )
    }

    print("Breakdown 类型完整性:")
    for breakdown_type, is_complete in breakdown_success.items():
        status = "✅ 完整" if is_complete else "⚠️  部分缺失"
        print(f"  {breakdown_type}: {status}")

    successful_breakdowns = sum(breakdown_success.values())
    total_breakdowns = len(breakdown_success)

    print(f"\n总体评分: {successful_breakdowns}/{total_breakdowns} ({successful_breakdowns/total_breakdowns*100:.1f}%) breakdown类型完整提取")

    if successful_breakdowns >= 4:
        print("🎉 TEA Breakdown 数据提取质量优秀！")
    elif successful_breakdowns >= 3:
        print("👍 TEA Breakdown 数据提取质量良好！")
    else:
        print("🔧 TEA Breakdown 数据提取需要进一步改进")


def main():
    """主函数"""
    analyze_breakdown_data()


if __name__ == "__main__":
    main()
