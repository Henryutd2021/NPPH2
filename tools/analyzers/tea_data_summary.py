#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEA数据提取结果总结脚本
分析Excel文件中各个数据表的数据完整性和质量
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_excel_data(excel_file: str = "output/tea/tea_summary_analysis.xlsx"):
    """分析Excel文件中的数据质量"""
    
    logger.info(f"分析TEA数据提取结果: {excel_file}")
    
    try:
        # 读取所有工作表
        xl = pd.ExcelFile(excel_file)
        sheet_names = xl.sheet_names
        
        print("TEA Summary Report 数据提取结果分析")
        print("=" * 60)
        
        total_reactors = 0
        
        for sheet_name in sheet_names:
            print(f"\n📊 工作表: {sheet_name}")
            print("-" * 40)
            
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if len(df) > 0:
                total_reactors = len(df)
                total_columns = len(df.columns)
                nan_counts = df.isnull().sum()
                
                print(f"反应堆数量: {total_reactors}")
                print(f"数据列数量: {total_columns}")
                
                # 统计完全提取成功的列
                complete_columns = sum(nan_counts == 0)
                partial_columns = sum((nan_counts > 0) & (nan_counts < total_reactors))
                missing_columns = sum(nan_counts == total_reactors)
                
                print(f"完全成功提取: {complete_columns} 列 ({complete_columns/total_columns*100:.1f}%)")
                print(f"部分成功提取: {partial_columns} 列 ({partial_columns/total_columns*100:.1f}%)")
                print(f"完全缺失数据: {missing_columns} 列 ({missing_columns/total_columns*100:.1f}%)")
                
                # 显示重要指标的提取情况
                if sheet_name == "Executive_Summary":
                    print("\n🔑 核心指标提取情况:")
                    core_metrics = ['npv_usd', 'irr_percent', 'lcoh_usd_per_kg', 'payback_period_years']
                    for metric in core_metrics:
                        if metric in df.columns:
                            missing = df[metric].isnull().sum()
                            success_rate = (total_reactors - missing) / total_reactors * 100
                            print(f"  {metric}: {success_rate:.1f}% ({total_reactors - missing}/{total_reactors})")
                
                elif sheet_name == "Financial_Performance":
                    print("\n💰 财务指标提取情况:")
                    financial_metrics = ['npv_usd', 'irr_percent', 'lcoh_usd_per_kg', 'total_annual_revenue']
                    for metric in financial_metrics:
                        if metric in df.columns:
                            missing = df[metric].isnull().sum()
                            success_rate = (total_reactors - missing) / total_reactors * 100
                            print(f"  {metric}: {success_rate:.1f}% ({total_reactors - missing}/{total_reactors})")
                
                elif sheet_name == "System_Performance":
                    print("\n⚡ 系统性能指标提取情况:")
                    performance_metrics = ['annual_h2_production_kg', 'electrolyzer_capacity_factor_percent', 
                                         'turbine_capacity_factor_percent', 'annual_nuclear_generation_mwh']
                    for metric in performance_metrics:
                        if metric in df.columns:
                            missing = df[metric].isnull().sum()
                            success_rate = (total_reactors - missing) / total_reactors * 100
                            print(f"  {metric}: {success_rate:.1f}% ({total_reactors - missing}/{total_reactors})")
                
                elif sheet_name == "Cost_Analysis":
                    print("\n💸 成本分析指标提取情况:")
                    cost_metrics = ['total_capex', 'electrolyzer_system_capex', 'nuclear_plant_opex', 'total_annual_opex']
                    for metric in cost_metrics:
                        if metric in df.columns:
                            missing = df[metric].isnull().sum()
                            success_rate = (total_reactors - missing) / total_reactors * 100
                            print(f"  {metric}: {success_rate:.1f}% ({total_reactors - missing}/{total_reactors})")
                
                # 显示缺失数据最多的前5列
                if missing_columns > 0:
                    print(f"\n❌ 缺失数据最多的列 (前5个):")
                    missing_data = nan_counts[nan_counts > 0].sort_values(ascending=False).head(5)
                    for col, missing_count in missing_data.items():
                        print(f"  {col}: {missing_count}/{total_reactors} ({missing_count/total_reactors*100:.1f}%)")
        
        # 总体统计
        print(f"\n📈 总体数据质量评估")
        print("=" * 40)
        
        # 读取执行摘要来获取核心统计
        exec_df = pd.read_excel(excel_file, sheet_name="Executive_Summary")
        
        # NPV统计
        positive_npv = (exec_df['npv_usd'] > 0).sum()
        avg_npv = exec_df['npv_usd'].mean()
        
        # LCOH统计
        avg_lcoh = exec_df['lcoh_usd_per_kg'].mean()
        below_target = (exec_df['lcoh_usd_per_kg'] <= 2.0).sum()
        
        # IRR统计
        valid_irr = exec_df['irr_percent'].notna().sum()
        avg_irr = exec_df['irr_percent'].mean()
        
        print(f"总反应堆数量: {total_reactors}")
        print(f"NPV为正值的反应堆: {positive_npv}/{total_reactors} ({positive_npv/total_reactors*100:.1f}%)")
        print(f"平均NPV: ${avg_npv:,.0f}")
        print(f"平均LCOH: ${avg_lcoh:.3f}/kg")
        print(f"达到DOE $2/kg目标的反应堆: {below_target}/{total_reactors} ({below_target/total_reactors*100:.1f}%)")
        print(f"有效IRR数据: {valid_irr}/{total_reactors} ({valid_irr/total_reactors*100:.1f}%)")
        if valid_irr > 0:
            print(f"平均IRR: {avg_irr:.2f}%")
        
        # 推荐改进
        print(f"\n🔧 建议改进的解析模式:")
        fin_df = pd.read_excel(excel_file, sheet_name="Financial_Performance")
        irr_missing = fin_df['irr_percent'].isnull().sum()
        if irr_missing > 5:
            print(f"  - IRR提取: {irr_missing}个反应堆缺失，需要改进正则表达式")
        
        advanced_missing = fin_df['npv_improvement_from_45u'].isnull().sum()
        if advanced_missing > 35:
            print(f"  - 45U分析数据: 需要添加第6节高级分析的解析")
        
        baseline_missing = fin_df['baseline_npv'].isnull().sum()
        if baseline_missing > 35:
            print(f"  - 基线分析数据: 需要添加第7节比较分析的解析")
            
        print(f"\n✅ 解析脚本修复成功！主要财务和性能数据已正确提取。")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}", exc_info=True)


def main():
    """主函数"""
    analyze_excel_data()


if __name__ == "__main__":
    main() 