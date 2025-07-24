#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Results Script
Analyzes sensitivity analysis results from parametric studies
Based on ancillary_analyzer.py structure
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import json
warnings.filterwarnings('ignore')

# Go up three levels: analyzers -> tools -> project_root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.logger_utils.logging_setup import logger
except ImportError:
    # Fallback to basic logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SensitivityAnalysisAnalyzer:
    """
    Analyzes sensitivity analysis results from parametric studies
    """

    def __init__(self, results_dir: str = "../../output/sensitivity_analysis"):
        """
        Initialize the analyzer

        Args:
            results_dir: Directory containing sensitivity analysis results
        """
        self.results_dir = Path(results_dir)
        self.sensitivity_data = {}
        self.analysis_results = []

        # Define sensitivity parameter categories
        self.parameter_categories = {
            "electrolyzer_capacity": "Electrolyzer Capacity (MW)",
            "electrolyzer_ramp": "Electrolyzer Ramp Rate",
            "turbine_ramp": "Turbine Ramp Rate"
        }

        logger.info(
            f"Initialized sensitivity analyzer for directory: {self.results_dir}")

    def load_sensitivity_results(self) -> Dict[str, Dict]:
        """
        Load all sensitivity analysis results files

        Returns:
            Dictionary mapping parameter_value combinations to their data
        """
        sensitivity_dirs = [
            d for d in self.results_dir.iterdir() if d.is_dir() and d.name != 'logs']
        logger.info(
            f"Found {len(sensitivity_dirs)} sensitivity analysis directories")

        for dir_path in sensitivity_dirs:
            try:
                # Extract parameter type and value from directory name
                dir_name = dir_path.name

                # Skip non-parametric directories
                if not any(param in dir_name for param in self.parameter_categories.keys()):
                    continue

                # Parse parameter and value
                for param_type in self.parameter_categories.keys():
                    if param_type in dir_name:
                        # Extract value (everything after the parameter type)
                        value_str = dir_name.replace(f"{param_type}_", "")
                        try:
                            param_value = float(value_str)
                        except ValueError:
                            logger.warning(
                                f"Could not parse value from {dir_name}")
                            continue

                        # Look for the main results CSV file
                        expected_csv = dir_path / f"{dir_name}_results.csv"
                        if expected_csv.exists():
                            # Load data
                            df = pd.read_csv(expected_csv)
                            logger.info(
                                f"Loaded {len(df)} hours of data for {param_type}={param_value}")

                            # Also load summary JSON if available
                            summary_json_path = dir_path / \
                                f"{dir_name}_summary.json"
                            summary_data = {}
                            if summary_json_path.exists():
                                with open(summary_json_path, 'r') as f:
                                    summary_data = json.load(f)

                            self.sensitivity_data[f"{param_type}_{param_value}"] = {
                                'data': df,
                                'parameter_type': param_type,
                                'parameter_value': param_value,
                                'directory': dir_name,
                                'summary': summary_data
                            }
                        else:
                            logger.warning(
                                f"Expected CSV file not found: {expected_csv}")
                        break

            except Exception as e:
                logger.error(f"Error loading {dir_path}: {e}")
                continue

        return self.sensitivity_data

    def analyze_capacity_factors(self, df: pd.DataFrame, case_name: str) -> Dict:
        """
        Analyze capacity factors for hydrogen storage and generators

        Args:
            df: Hourly results DataFrame
            case_name: Name of the sensitivity case

        Returns:
            Dictionary with capacity factor metrics
        """
        metrics = {}

        # Hydrogen storage capacity factor
        if 'H2_Storage_Level_kg' in df.columns and 'H2_Storage_Capacity_kg' in df.columns:
            storage_level = df['H2_Storage_Level_kg'].dropna()
            storage_capacity = df['H2_Storage_Capacity_kg'].dropna()

            if len(storage_level) > 0 and len(storage_capacity) > 0:
                # Calculate average utilization
                avg_capacity = storage_capacity.iloc[0] if len(
                    storage_capacity.unique()) == 1 else storage_capacity.mean()
                if avg_capacity > 0:
                    h2_capacity_factor = storage_level.mean() / avg_capacity
                    metrics['h2_storage_capacity_factor'] = h2_capacity_factor
                    metrics['h2_storage_max_utilization'] = storage_level.max() / \
                        avg_capacity
                    metrics['h2_storage_min_utilization'] = storage_level.min() / \
                        avg_capacity

                    # Additional H2 storage metrics
                    metrics['h2_storage_avg_level_kg'] = storage_level.mean()
                    metrics['h2_storage_capacity_kg'] = avg_capacity

                    # Calculate cycling frequency
                    storage_changes = storage_level.diff().dropna()
                    significant_changes = storage_changes[abs(
                        storage_changes) > avg_capacity * 0.01]  # 1% of capacity
                    metrics['h2_storage_cycling_frequency'] = len(
                        significant_changes) / len(storage_level)

        # Generator (Turbine) capacity factor
        if 'pTurbine_MW' in df.columns:
            turbine_output = df['pTurbine_MW'].dropna()

            if len(turbine_output) > 0:
                # Estimate nameplate capacity from max output
                estimated_nameplate = turbine_output.max()
                if estimated_nameplate > 0:
                    turbine_capacity_factor = turbine_output.mean() / estimated_nameplate
                    metrics['turbine_capacity_factor'] = turbine_capacity_factor
                    metrics['turbine_max_output_MW'] = estimated_nameplate
                    metrics['turbine_avg_output_MW'] = turbine_output.mean()
                    metrics['turbine_min_output_MW'] = turbine_output.min()

                    # Operating hours analysis
                    operating_threshold = estimated_nameplate * \
                        0.05  # 5% of max as operating threshold
                    operating_hours = len(
                        turbine_output[turbine_output > operating_threshold])
                    metrics['turbine_operating_hours'] = operating_hours
                    metrics['turbine_operating_frequency'] = operating_hours / \
                        len(turbine_output)

        # Electrolyzer capacity factor
        if 'pElectrolyzer_MW' in df.columns and 'Electrolyzer_Capacity_MW' in df.columns:
            electrolyzer_output = df['pElectrolyzer_MW'].dropna()
            electrolyzer_capacity = df['Electrolyzer_Capacity_MW'].dropna()

            if len(electrolyzer_output) > 0 and len(electrolyzer_capacity) > 0:
                avg_capacity = electrolyzer_capacity.iloc[0] if len(
                    electrolyzer_capacity.unique()) == 1 else electrolyzer_capacity.mean()
                if avg_capacity > 0:
                    electrolyzer_capacity_factor = electrolyzer_output.mean() / avg_capacity
                    metrics['electrolyzer_capacity_factor'] = electrolyzer_capacity_factor
                    metrics['electrolyzer_nameplate_MW'] = avg_capacity
                    metrics['electrolyzer_avg_output_MW'] = electrolyzer_output.mean()

                    # Operating analysis
                    operating_hours = len(
                        electrolyzer_output[electrolyzer_output > 0])
                    metrics['electrolyzer_operating_hours'] = operating_hours
                    metrics['electrolyzer_operating_frequency'] = operating_hours / \
                        len(electrolyzer_output)

        # Battery capacity factor if available
        if 'Battery_SOC_MWh' in df.columns and 'Battery_Capacity_MWh' in df.columns:
            battery_soc = df['Battery_SOC_MWh'].dropna()
            battery_capacity = df['Battery_Capacity_MWh'].dropna()

            if len(battery_soc) > 0 and len(battery_capacity) > 0:
                avg_capacity = battery_capacity.iloc[0] if len(
                    battery_capacity.unique()) == 1 else battery_capacity.mean()
                if avg_capacity > 0:
                    battery_capacity_factor = battery_soc.mean() / avg_capacity
                    metrics['battery_capacity_factor'] = battery_capacity_factor
                    metrics['battery_max_utilization'] = battery_soc.max() / \
                        avg_capacity

                    # Battery cycling analysis
                    if 'Battery_Charge_MW' in df.columns and 'Battery_Discharge_MW' in df.columns:
                        charge_power = df['Battery_Charge_MW'].fillna(0)
                        discharge_power = df['Battery_Discharge_MW'].fillna(0)

                        # Count cycles (charge/discharge events)
                        charge_events = len(charge_power[charge_power > 0])
                        discharge_events = len(
                            discharge_power[discharge_power > 0])

                        metrics['battery_charge_events'] = charge_events
                        metrics['battery_discharge_events'] = discharge_events
                        metrics['battery_cycling_frequency'] = min(
                            charge_events, discharge_events) / len(battery_soc)

        return metrics

    def analyze_system_performance(self, df: pd.DataFrame, case_name: str) -> Dict:
        """
        Analyze overall system performance metrics

        Args:
            df: Hourly results DataFrame  
            case_name: Name of the sensitivity case

        Returns:
            Dictionary with system performance metrics
        """
        metrics = {}

        # Grid interaction analysis
        if 'pIES_MW' in df.columns:
            grid_output = df['pIES_MW'].dropna()
            if len(grid_output) > 0:
                metrics['grid_output_mean_MW'] = grid_output.mean()
                metrics['grid_output_std_MW'] = grid_output.std()
                metrics['grid_output_max_MW'] = grid_output.max()
                metrics['grid_output_min_MW'] = grid_output.min()
                metrics['grid_output_range_MW'] = grid_output.max() - \
                    grid_output.min()

                # Grid export vs import analysis
                export_hours = len(grid_output[grid_output > 0])
                import_hours = len(grid_output[grid_output < 0])
                metrics['grid_export_hours'] = export_hours
                metrics['grid_import_hours'] = import_hours
                metrics['grid_export_fraction'] = export_hours / \
                    len(grid_output)

        # Hydrogen production analysis
        if 'mHydrogenProduced_kg_hr' in df.columns:
            h2_production = df['mHydrogenProduced_kg_hr'].dropna()
            if len(h2_production) > 0:
                metrics['h2_production_total_kg'] = h2_production.sum()
                metrics['h2_production_rate_avg_kg_hr'] = h2_production.mean()
                metrics['h2_production_rate_max_kg_hr'] = h2_production.max()

                # Production consistency
                production_hours = len(h2_production[h2_production > 0])
                metrics['h2_production_hours'] = production_hours
                metrics['h2_production_frequency'] = production_hours / \
                    len(h2_production)

        # System flexibility analysis
        if 'pElectrolyzer_MW' in df.columns and 'pTurbine_MW' in df.columns:
            electrolyzer_power = df['pElectrolyzer_MW'].fillna(0)
            turbine_power = df['pTurbine_MW'].fillna(0)

            # Calculate ramping capabilities
            electrolyzer_ramp = electrolyzer_power.diff().dropna()
            turbine_ramp = turbine_power.diff().dropna()

            if len(electrolyzer_ramp) > 0:
                metrics['electrolyzer_max_ramp_up_MW_hr'] = electrolyzer_ramp.max()
                metrics['electrolyzer_max_ramp_down_MW_hr'] = abs(
                    electrolyzer_ramp.min())
                metrics['electrolyzer_avg_ramp_rate_MW_hr'] = abs(
                    electrolyzer_ramp).mean()

            if len(turbine_ramp) > 0:
                metrics['turbine_max_ramp_up_MW_hr'] = turbine_ramp.max()
                metrics['turbine_max_ramp_down_MW_hr'] = abs(
                    turbine_ramp.min())
                metrics['turbine_avg_ramp_rate_MW_hr'] = abs(
                    turbine_ramp).mean()

        # Ancillary services provision (simplified from ancillary analyzer)
        ancillary_cols = [col for col in df.columns if '_Bid_MW' in col]
        if ancillary_cols:
            total_ancillary_provision = 0
            for col in ancillary_cols:
                service_bids = df[col].fillna(0)
                provision_hours = len(service_bids[service_bids > 0])
                if provision_hours > 0:
                    total_ancillary_provision += provision_hours

            metrics['ancillary_service_provision_hours'] = total_ancillary_provision
            metrics['ancillary_service_diversity'] = len([col for col in ancillary_cols
                                                          if df[col].fillna(0).sum() > 0])

        return metrics

    def analyze_economic_performance(self, df: pd.DataFrame, summary_data: Dict, case_name: str) -> Dict:
        """
        Analyze economic performance metrics

        Args:
            df: Hourly results DataFrame
            summary_data: Summary data from JSON file
            case_name: Name of the sensitivity case

        Returns:
            Dictionary with economic performance metrics
        """
        metrics = {}

        # Extract economic data from summary if available
        if summary_data:
            # Look for common economic metrics in summary
            economic_keys = ['NPV', 'IRR', 'LCOE',
                             'Total_Revenue', 'Total_Cost', 'Payback_Period']
            for key in economic_keys:
                if key in summary_data:
                    metrics[f'economic_{key.lower()}'] = summary_data[key]

        # If hourly revenue/cost data is available in the main DataFrame
        revenue_cols = [
            col for col in df.columns if 'Revenue' in col or 'revenue' in col]
        cost_cols = [
            col for col in df.columns if 'Cost' in col or 'cost' in col]

        for col in revenue_cols:
            if col in df.columns:
                revenue_data = df[col].dropna()
                if len(revenue_data) > 0:
                    col_name = col.lower().replace('revenue_', '').replace('_usd', '')
                    metrics[f'{col_name}_revenue_total'] = revenue_data.sum()
                    metrics[f'{col_name}_revenue_avg_hourly'] = revenue_data.mean()

        for col in cost_cols:
            if col in df.columns:
                cost_data = df[col].dropna()
                if len(cost_data) > 0:
                    col_name = col.lower().replace('cost_', '').replace('_usd', '')
                    metrics[f'{col_name}_cost_total'] = cost_data.sum()
                    metrics[f'{col_name}_cost_avg_hourly'] = cost_data.mean()

        return metrics

    def analyze_sensitivity_case(self, case_key: str) -> Dict:
        """
        Complete analysis for a single sensitivity case

        Args:
            case_key: Key identifying the sensitivity case

        Returns:
            Dictionary with all analysis results
        """
        if case_key not in self.sensitivity_data:
            logger.error(
                f"Sensitivity case {case_key} not found in loaded data")
            return {}

        case_info = self.sensitivity_data[case_key]
        df = case_info['data']
        parameter_type = case_info['parameter_type']
        parameter_value = case_info['parameter_value']
        summary_data = case_info['summary']

        logger.info(f"Analyzing {case_key} - {len(df)} hours of data")

        # Combine all analysis results
        results = {
            'case_name': case_key,
            'parameter_type': parameter_type,
            'parameter_value': parameter_value,
            'data_hours': len(df),
            'directory': case_info['directory']
        }

        # Run all analysis modules
        results.update(self.analyze_capacity_factors(df, case_key))
        results.update(self.analyze_system_performance(df, case_key))
        results.update(self.analyze_economic_performance(
            df, summary_data, case_key))

        return results

    def analyze_all_cases(self) -> pd.DataFrame:
        """
        Analyze all loaded sensitivity cases and compile results

        Returns:
            DataFrame with analysis results for all cases
        """
        logger.info("Starting analysis of all sensitivity cases...")

        self.analysis_results = []

        for case_key in self.sensitivity_data.keys():
            try:
                results = self.analyze_sensitivity_case(case_key)
                if results:
                    self.analysis_results.append(results)
                    logger.info(f"Completed analysis for {case_key}")
            except Exception as e:
                logger.error(f"Error analyzing {case_key}: {e}")
                continue

        if self.analysis_results:
            results_df = pd.DataFrame(self.analysis_results)
            logger.info(
                f"Analysis completed for {len(results_df)} sensitivity cases")
            return results_df
        else:
            logger.warning("No analysis results generated")
            return pd.DataFrame()

    def create_sensitivity_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Create summary statistics by parameter type

        Args:
            results_df: DataFrame with analysis results

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        for param_type in results_df['parameter_type'].unique():
            param_data = results_df[results_df['parameter_type']
                                    == param_type].copy()
            param_data = param_data.sort_values('parameter_value')

            param_summary = {
                'parameter_range': f"{param_data['parameter_value'].min()} - {param_data['parameter_value'].max()}",
                'cases_count': len(param_data),
                'parameter_values': param_data['parameter_value'].tolist()
            }

            # Key metrics trends
            key_metrics = [
                'h2_storage_capacity_factor', 'turbine_capacity_factor', 'electrolyzer_capacity_factor',
                'h2_production_total_kg', 'grid_output_mean_MW', 'electrolyzer_max_ramp_up_MW_hr'
            ]

            for metric in key_metrics:
                if metric in param_data.columns:
                    param_summary[f'{metric}_trend'] = {
                        'min': param_data[metric].min(),
                        'max': param_data[metric].max(),
                        'values': param_data[metric].tolist()
                    }

            summary[param_type] = param_summary

        return summary

    def save_results(self, results_df: pd.DataFrame, output_file: str = "sensitivity_analysis_results.csv"):
        """
        Save analysis results to CSV file and create summary reports

        Args:
            results_df: DataFrame with analysis results
            output_file: Output filename
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("../../output/sensitivity_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            output_path = output_dir / output_file
            results_df.to_csv(output_path, index=False)
            logger.info(f"Detailed results saved to {output_path}")

            # Create parameter-specific summary files
            for param_type in results_df['parameter_type'].unique():
                param_data = results_df[results_df['parameter_type'] == param_type].copy(
                )
                param_data = param_data.sort_values('parameter_value')

                param_file = output_dir / \
                    f"{param_type}_sensitivity_results.csv"
                param_data.to_csv(param_file, index=False)
                logger.info(
                    f"Parameter-specific results saved to {param_file}")

            # Create comprehensive summary report
            summary_path = output_dir / \
                output_file.replace('.csv', '_comprehensive_summary.txt')
            self._create_comprehensive_summary_report(results_df, summary_path)

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _create_comprehensive_summary_report(self, results_df: pd.DataFrame, summary_file: str):
        """
        Create a comprehensive summary report of the sensitivity analysis

        Args:
            results_df: DataFrame with analysis results
            summary_file: Summary report filename
        """
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== Sensitivity Analysis Comprehensive Summary ===\n\n")
                f.write(f"Total Cases Analyzed: {len(results_df)}\n")
                f.write(
                    f"Parameter Types: {', '.join(results_df['parameter_type'].unique())}\n\n")

                # Analysis by parameter type
                for param_type in results_df['parameter_type'].unique():
                    param_data = results_df[results_df['parameter_type'] == param_type].copy(
                    )
                    param_data = param_data.sort_values('parameter_value')

                    f.write(
                        f"=== {param_type.replace('_', ' ').title()} Sensitivity ===\n")
                    f.write(
                        f"Parameter Range: {param_data['parameter_value'].min()} - {param_data['parameter_value'].max()}\n")
                    f.write(f"Number of Cases: {len(param_data)}\n\n")

                    # Key capacity factor analysis
                    capacity_metrics = [
                        'h2_storage_capacity_factor', 'turbine_capacity_factor', 'electrolyzer_capacity_factor']
                    f.write("Capacity Factor Analysis:\n")
                    for metric in capacity_metrics:
                        if metric in param_data.columns:
                            f.write(f"  {metric.replace('_', ' ').title()}:\n")
                            f.write(
                                f"    Range: {param_data[metric].min():.3f} - {param_data[metric].max():.3f}\n")
                            f.write(
                                f"    Average: {param_data[metric].mean():.3f}\n")
                            f.write(
                                f"    Std Dev: {param_data[metric].std():.3f}\n")

                    # Performance metrics
                    f.write("\nPerformance Metrics:\n")
                    performance_metrics = [
                        'h2_production_total_kg', 'grid_output_mean_MW', 'electrolyzer_operating_frequency']
                    for metric in performance_metrics:
                        if metric in param_data.columns:
                            f.write(f"  {metric.replace('_', ' ').title()}:\n")
                            f.write(
                                f"    Range: {param_data[metric].min():.3f} - {param_data[metric].max():.3f}\n")
                            f.write(
                                f"    Average: {param_data[metric].mean():.3f}\n")

                    f.write("\n" + "="*50 + "\n")

                # Overall summary statistics
                f.write("\n=== Overall Summary Statistics ===\n")
                numeric_cols = results_df.select_dtypes(
                    include=[np.number]).columns
                key_metrics = [col for col in numeric_cols if any(keyword in col.lower()
                                                                  for keyword in ['capacity_factor', 'production', 'output', 'ramp'])]

                for metric in key_metrics[:10]:  # Top 10 key metrics
                    if metric in results_df.columns:
                        f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                        f.write(
                            f"  Overall Range: {results_df[metric].min():.3f} - {results_df[metric].max():.3f}\n")
                        f.write(
                            f"  Overall Average: {results_df[metric].mean():.3f}\n")
                        f.write(
                            f"  Overall Std Dev: {results_df[metric].std():.3f}\n")

                f.write(
                    f"\nDetailed results saved to: {summary_file.replace('_comprehensive_summary.txt', '.csv')}\n")

            logger.info(
                f"Comprehensive summary report saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error creating comprehensive summary report: {e}")


def main():
    """
    Main function to run the sensitivity analysis
    """
    # Setup logging
    log_dir = Path('../../output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'sensitivity_analysis.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("Starting Sensitivity Analysis")

    try:
        # Initialize analyzer
        analyzer = SensitivityAnalysisAnalyzer()

        # Load data
        logger.info("Loading sensitivity analysis data...")
        sensitivity_data = analyzer.load_sensitivity_results()

        if not sensitivity_data:
            logger.error(
                "No sensitivity data loaded. Please check the results directory.")
            return

        # Run analysis
        logger.info("Running comprehensive sensitivity analysis...")
        results_df = analyzer.analyze_all_cases()

        if results_df.empty:
            logger.error("No analysis results generated.")
            return

        # Save results
        analyzer.save_results(results_df)

        logger.info("Sensitivity analysis completed successfully!")
        logger.info(
            f"Analyzed {len(results_df)} cases across {len(results_df['parameter_type'].unique())} parameter types")

        # Print summary statistics
        print("\n=== Sensitivity Analysis Summary ===")
        print(f"Total Cases: {len(results_df)}")
        print(
            f"Parameter Types: {', '.join(results_df['parameter_type'].unique())}")

        # Print key findings
        for param_type in results_df['parameter_type'].unique():
            param_data = results_df[results_df['parameter_type'] == param_type]
            print(f"\n{param_type.replace('_', ' ').title()}:")
            print(f"  Cases: {len(param_data)}")
            print(
                f"  Parameter Range: {param_data['parameter_value'].min()} - {param_data['parameter_value'].max()}")

            if 'h2_storage_capacity_factor' in param_data.columns:
                print(
                    f"  H2 Storage CF Range: {param_data['h2_storage_capacity_factor'].min():.3f} - {param_data['h2_storage_capacity_factor'].max():.3f}")
            if 'turbine_capacity_factor' in param_data.columns:
                print(
                    f"  Turbine CF Range: {param_data['turbine_capacity_factor'].min():.3f} - {param_data['turbine_capacity_factor'].max():.3f}")

    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
