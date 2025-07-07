#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ancillary Services Analysis Script for OPT Hourly Results
Analyzes reactor flexibility and ancillary service capabilities from hourly optimization results
"""

from src.logger_utils.logging_setup import logger
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))



class AncillaryServicesAnalyzer:
    """
    Analyzes ancillary services and system flexibility from OPT hourly results
    """

    def __init__(self, results_dir: str = "../output/opt/cs1"):
        """
        Initialize the analyzer

        Args:
            results_dir: Directory containing hourly results CSV files
        """
        self.results_dir = Path(results_dir)
        self.reactor_data = {}
        self.analysis_results = []

        # ISO-specific ancillary service mappings
        self.iso_services = {
            "SPP": ["RegU", "RegD", "Spin", "Sup", "RamU", "RamD", "UncU"],
            "CAISO": ["RegU", "RegD", "Spin", "NSpin", "RMU", "RMD"],
            "ERCOT": ["RegU", "RegD", "Spin", "NSpin", "ECRS"],
            "PJM": ["RegUp", "RegDown", "Syn", "Rse", "TMR"],
            "NYISO": ["RegUp", "RegDown", "Spin10", "NSpin10", "Res30"],
            "ISONE": ["RegUp", "RegDown", "Spin10", "NSpin10", "OR30"],
            "MISO": ["RegUp", "RegDown", "Spin", "Sup", "STR", "RamU", "RamD"]
        }

        # Service type categorization
        self.service_categories = {
            "regulation": ["RegU", "RegD", "RegUp", "RegDown"],
            "spinning_reserve": ["Spin", "Spin10", "Syn"],
            "non_spinning_reserve": ["Sup", "NSpin", "NSpin10", "Rse"],
            "ramping": ["RamU", "RamD", "RMU", "RMD"],
            "other": ["UncU", "ECRS", "TMR", "Res30", "OR30", "STR"]
        }

        logger.info(f"Initialized analyzer for directory: {self.results_dir}")

    def load_hourly_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all hourly results files

        Returns:
            Dictionary mapping reactor names to their hourly data
        """
        result_files = list(self.results_dir.glob("*_hourly_results.csv"))
        logger.info(f"Found {len(result_files)} hourly results files")

        for file_path in result_files:
            try:
                # Extract reactor name and ISO from filename
                filename = file_path.stem
                # Format: "Reactor Name_Unit_ISO_XX_hourly_results"
                parts = filename.replace("_hourly_results", "").split("_")

                # Find ISO (assuming it's one of the known ISOs)
                iso = None
                for potential_iso in self.iso_services.keys():
                    if potential_iso in parts:
                        iso = potential_iso
                        break

                if not iso:
                    logger.warning(
                        f"Could not identify ISO for file: {filename}")
                    continue

                # Reconstruct reactor name (everything before ISO)
                iso_index = parts.index(iso)
                reactor_parts = parts[:iso_index]
                reactor_name = " ".join(reactor_parts)

                # Load data
                df = pd.read_csv(file_path)
                logger.info(
                    f"Loaded {len(df)} hours of data for {reactor_name} ({iso})")

                self.reactor_data[reactor_name] = {
                    'data': df,
                    'iso': iso,
                    'filename': filename
                }

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        return self.reactor_data

    def analyze_grid_output_flexibility(self, df: pd.DataFrame, reactor_name: str) -> Dict:
        """
        Analyze grid output flexibility parameters

        Args:
            df: Hourly results DataFrame
            reactor_name: Name of the reactor

        Returns:
            Dictionary with grid output flexibility metrics
        """
        metrics = {}

        # Grid power output analysis
        if 'pIES_MW' in df.columns:
            grid_output = df['pIES_MW'].dropna()

            metrics.update({
                'grid_output_mean_MW': grid_output.mean(),
                'grid_output_std_MW': grid_output.std(),
                'grid_output_min_MW': grid_output.min(),
                'grid_output_max_MW': grid_output.max(),
                'grid_output_range_MW': grid_output.max() - grid_output.min(),
                'grid_output_cv': grid_output.std() / grid_output.mean() if grid_output.mean() > 0 else 0,
            })

            # Calculate ramping capabilities
            grid_ramp = grid_output.diff().dropna()
            metrics.update({
                'max_ramp_up_MW_hr': grid_ramp.max(),
                'max_ramp_down_MW_hr': abs(grid_ramp.min()),
                'avg_ramp_rate_MW_hr': abs(grid_ramp).mean(),
                'ramp_events_count': len(grid_ramp[abs(grid_ramp) > grid_output.std()]),
            })

        # Turbine-specific flexibility
        if 'pTurbine_MW' in df.columns:
            turbine_output = df['pTurbine_MW'].dropna()
            turbine_ramp = turbine_output.diff().dropna()

            metrics.update({
                'turbine_output_range_MW': turbine_output.max() - turbine_output.min(),
                'turbine_max_ramp_MW_hr': abs(turbine_ramp).max(),
                'turbine_flexibility_factor': (turbine_output.max() - turbine_output.min()) / turbine_output.max() if turbine_output.max() > 0 else 0,
            })

        # Electrolyzer flexibility
        if 'pElectrolyzer_MW' in df.columns:
            electrolyzer_output = df['pElectrolyzer_MW'].dropna()

            # Calculate operational flexibility
            on_hours = len(electrolyzer_output[electrolyzer_output > 0])
            off_hours = len(electrolyzer_output[electrolyzer_output == 0])

            metrics.update({
                'electrolyzer_capacity_factor': electrolyzer_output.mean() / electrolyzer_output.max() if electrolyzer_output.max() > 0 else 0,
                'electrolyzer_on_hours': on_hours,
                'electrolyzer_off_hours': off_hours,
                'electrolyzer_cycling_frequency': len(electrolyzer_output.diff()[abs(electrolyzer_output.diff()) > 1]) / len(electrolyzer_output),
            })

        return metrics

    def analyze_ancillary_service_provision(self, df: pd.DataFrame, iso: str, reactor_name: str) -> Dict:
        """
        Analyze ancillary service provision capabilities

        Args:
            df: Hourly results DataFrame
            iso: ISO region
            reactor_name: Name of the reactor

        Returns:
            Dictionary with ancillary service metrics
        """
        metrics = {}

        # Get available services for this ISO
        available_services = self.iso_services.get(iso, [])

        for service in available_services:
            # Analyze bid quantities
            bid_cols = [
                col for col in df.columns if f'{service}_' in col and '_Bid_MW' in col]
            deployed_cols = [
                col for col in df.columns if f'{service}_' in col and '_Deployed_MW' in col]

            if bid_cols:
                # Combine all bid sources for this service
                total_bids = pd.DataFrame()
                for col in bid_cols:
                    if col in df.columns:
                        total_bids[col] = df[col].fillna(0)

                if not total_bids.empty:
                    service_bid_total = total_bids.sum(axis=1)

                    # Service provision frequency and characteristics
                    provision_hours = len(
                        service_bid_total[service_bid_total > 0])
                    provision_frequency = provision_hours / \
                        len(service_bid_total) if len(
                            service_bid_total) > 0 else 0

                    metrics[f'{service}_provision_frequency'] = provision_frequency
                    metrics[f'{service}_avg_bid_MW'] = service_bid_total[service_bid_total > 0].mean(
                    ) if provision_hours > 0 else 0
                    metrics[f'{service}_max_bid_MW'] = service_bid_total.max()
                    metrics[f'{service}_total_bid_MWh'] = service_bid_total.sum()

                    # Calculate service variability
                    if provision_hours > 0:
                        active_bids = service_bid_total[service_bid_total > 0]
                        metrics[f'{service}_bid_variability'] = active_bids.std(
                        ) / active_bids.mean() if active_bids.mean() > 0 else 0

                        # Consecutive hours analysis
                        consecutive_provision = self._analyze_consecutive_hours(
                            service_bid_total > 0)
                        metrics[f'{service}_avg_consecutive_hours'] = consecutive_provision['avg_consecutive']
                        metrics[f'{service}_max_consecutive_hours'] = consecutive_provision['max_consecutive']

            # Analyze deployed amounts if available
            if deployed_cols:
                total_deployed = pd.DataFrame()
                for col in deployed_cols:
                    if col in df.columns:
                        total_deployed[col] = df[col].fillna(0)

                if not total_deployed.empty:
                    service_deployed_total = total_deployed.sum(axis=1)
                    deployed_hours = len(
                        service_deployed_total[service_deployed_total > 0])

                    if deployed_hours > 0:
                        metrics[f'{service}_deployment_frequency'] = deployed_hours / \
                            len(service_deployed_total)
                        metrics[f'{service}_avg_deployed_MW'] = service_deployed_total[service_deployed_total > 0].mean(
                        )
                        metrics[f'{service}_max_deployed_MW'] = service_deployed_total.max(
                        )

        # Calculate service diversity and portfolio metrics
        total_services_provided = sum(1 for service in available_services
                                      if metrics.get(f'{service}_provision_frequency', 0) > 0)
        metrics['service_diversity_count'] = total_services_provided
        metrics['service_diversity_ratio'] = total_services_provided / \
            len(available_services) if available_services else 0

        return metrics

    def analyze_revenue_and_performance(self, df: pd.DataFrame, reactor_name: str) -> Dict:
        """
        Analyze revenue and performance metrics

        Args:
            df: Hourly results DataFrame
            reactor_name: Name of the reactor

        Returns:
            Dictionary with revenue and performance metrics
        """
        metrics = {}

        # Revenue analysis
        revenue_cols = [col for col in df.columns if 'Revenue_' in col]
        for col in revenue_cols:
            if col in df.columns:
                revenue_data = df[col].dropna()
                if len(revenue_data) > 0:
                    col_name = col.replace(
                        'Revenue_', '').replace('_USD', '').lower()
                    metrics[f'{col_name}_revenue_total'] = revenue_data.sum()
                    metrics[f'{col_name}_revenue_avg_hourly'] = revenue_data.mean()
                    metrics[f'{col_name}_revenue_variability'] = revenue_data.std(
                    ) / revenue_data.mean() if revenue_data.mean() > 0 else 0

        # Ancillary services revenue share
        if 'Revenue_Ancillary_USD' in df.columns and 'Revenue_Total_USD' in df.columns:
            ancillary_rev = df['Revenue_Ancillary_USD'].sum()
            total_rev = df['Revenue_Total_USD'].sum()
            metrics['ancillary_revenue_share'] = ancillary_rev / \
                total_rev if total_rev > 0 else 0

        # Cost analysis
        cost_cols = [col for col in df.columns if 'Cost_' in col]
        for col in cost_cols:
            if col in df.columns:
                cost_data = df[col].dropna()
                if len(cost_data) > 0:
                    col_name = col.replace(
                        'Cost_', '').replace('_USD', '').lower()
                    metrics[f'{col_name}_cost_total'] = cost_data.sum()
                    metrics[f'{col_name}_cost_avg_hourly'] = cost_data.mean()

        # Profit analysis
        if 'Profit_Hourly_USD' in df.columns:
            profit_data = df['Profit_Hourly_USD'].dropna()
            if len(profit_data) > 0:
                metrics['profit_total'] = profit_data.sum()
                metrics['profit_avg_hourly'] = profit_data.mean()
                metrics['profit_volatility'] = profit_data.std()
                metrics['profitable_hours'] = len(profit_data[profit_data > 0])
                metrics['profit_margin'] = profit_data.mean() / df['Revenue_Total_USD'].mean(
                ) if 'Revenue_Total_USD' in df.columns and df['Revenue_Total_USD'].mean() > 0 else 0

        return metrics

    def analyze_system_flexibility_metrics(self, df: pd.DataFrame, reactor_name: str) -> Dict:
        """
        Analyze comprehensive system flexibility metrics

        Args:
            df: Hourly results DataFrame
            reactor_name: Name of the reactor

        Returns:
            Dictionary with system flexibility metrics
        """
        metrics = {}

        # Storage utilization if available
        if 'H2_Storage_Level_kg' in df.columns:
            storage_level = df['H2_Storage_Level_kg'].dropna()
            if len(storage_level) > 0:
                metrics['h2_storage_utilization'] = storage_level.mean(
                ) / storage_level.max() if storage_level.max() > 0 else 0
                metrics['h2_storage_cycling_frequency'] = len(storage_level.diff()[abs(
                    storage_level.diff()) > storage_level.std()]) / len(storage_level)

        if 'Battery_SOC_MWh' in df.columns:
            battery_soc = df['Battery_SOC_MWh'].dropna()
            if len(battery_soc) > 0:
                metrics['battery_utilization'] = battery_soc.mean(
                ) / battery_soc.max() if battery_soc.max() > 0 else 0
                metrics['battery_cycling_frequency'] = len(battery_soc.diff()[abs(
                    battery_soc.diff()) > battery_soc.std()]) / len(battery_soc)

        # System responsiveness metrics
        if 'pElectrolyzer_MW' in df.columns and 'pElectrolyzerSetpoint_MW' in df.columns:
            actual = df['pElectrolyzer_MW'].dropna()
            setpoint = df['pElectrolyzerSetpoint_MW'].dropna()
            if len(actual) > 0 and len(setpoint) > 0:
                tracking_error = abs(actual - setpoint).mean()
                metrics['electrolyzer_tracking_accuracy'] = 1 - \
                    (tracking_error / setpoint.mean()
                     ) if setpoint.mean() > 0 else 0

        # Overall system flexibility score
        flexibility_components = []

        # Grid output flexibility
        if 'pIES_MW' in df.columns:
            grid_output = df['pIES_MW'].dropna()
            if len(grid_output) > 0:
                grid_flexibility = (grid_output.max(
                ) - grid_output.min()) / grid_output.max() if grid_output.max() > 0 else 0
                flexibility_components.append(grid_flexibility)

        # Service provision diversity
        service_diversity = metrics.get('service_diversity_ratio', 0)
        flexibility_components.append(service_diversity)

        # Calculate composite flexibility score
        if flexibility_components:
            metrics['overall_flexibility_score'] = np.mean(
                flexibility_components)

        return metrics

    def _analyze_consecutive_hours(self, boolean_series: pd.Series) -> Dict:
        """
        Analyze consecutive hours of service provision

        Args:
            boolean_series: Boolean series indicating service provision

        Returns:
            Dictionary with consecutive hours statistics
        """
        consecutive_lengths = []
        current_length = 0

        for value in boolean_series:
            if value:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                current_length = 0

        # Add final sequence if it ends with True
        if current_length > 0:
            consecutive_lengths.append(current_length)

        if consecutive_lengths:
            return {
                'avg_consecutive': np.mean(consecutive_lengths),
                'max_consecutive': max(consecutive_lengths),
                'min_consecutive': min(consecutive_lengths),
                'total_sequences': len(consecutive_lengths)
            }
        else:
            return {
                'avg_consecutive': 0,
                'max_consecutive': 0,
                'min_consecutive': 0,
                'total_sequences': 0
            }

    def analyze_reactor(self, reactor_name: str) -> Dict:
        """
        Complete analysis for a single reactor

        Args:
            reactor_name: Name of the reactor to analyze

        Returns:
            Dictionary with all analysis results
        """
        if reactor_name not in self.reactor_data:
            logger.error(f"Reactor {reactor_name} not found in loaded data")
            return {}

        reactor_info = self.reactor_data[reactor_name]
        df = reactor_info['data']
        iso = reactor_info['iso']

        logger.info(
            f"Analyzing {reactor_name} ({iso}) - {len(df)} hours of data")

        # Combine all analysis results
        results = {
            'reactor_name': reactor_name,
            'iso_region': iso,
            'data_hours': len(df),
            'filename': reactor_info['filename']
        }

        # Run all analysis modules
        results.update(self.analyze_grid_output_flexibility(df, reactor_name))
        results.update(self.analyze_ancillary_service_provision(
            df, iso, reactor_name))
        results.update(self.analyze_revenue_and_performance(df, reactor_name))
        results.update(
            self.analyze_system_flexibility_metrics(df, reactor_name))

        return results

    def analyze_all_reactors(self) -> pd.DataFrame:
        """
        Analyze all loaded reactors and compile results

        Returns:
            DataFrame with analysis results for all reactors
        """
        logger.info("Starting analysis of all reactors...")

        self.analysis_results = []

        for reactor_name in self.reactor_data.keys():
            try:
                results = self.analyze_reactor(reactor_name)
                if results:
                    self.analysis_results.append(results)
                    logger.info(f"Completed analysis for {reactor_name}")
            except Exception as e:
                logger.error(f"Error analyzing {reactor_name}: {e}")
                continue

        if self.analysis_results:
            results_df = pd.DataFrame(self.analysis_results)
            logger.info(f"Analysis completed for {len(results_df)} reactors")
            return results_df
        else:
            logger.warning("No analysis results generated")
            return pd.DataFrame()

    def save_results(self, results_df: pd.DataFrame, output_file: str = "analysis_results.csv"):
        """
        Save analysis results to CSV file

        Args:
            results_df: DataFrame with analysis results
            output_file: Output filename
        """
        try:
            # Save to organized output directory
            output_path = f"../output/ancillary/{output_file}"
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

            # Create a summary report
            summary_path = f"../output/ancillary/{output_file.replace('.csv', '_summary.txt')}"
            self._create_summary_report(results_df, summary_path)

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _create_summary_report(self, results_df: pd.DataFrame, summary_file: str):
        """
        Create a summary report of the analysis

        Args:
            results_df: DataFrame with analysis results
            summary_file: Summary report filename
        """
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== Ancillary Services Analysis Summary ===\n\n")
                f.write(f"Total Reactors Analyzed: {len(results_df)}\n")
                f.write(
                    f"ISO Regions: {', '.join(results_df['iso_region'].unique())}\n\n")

                # High-level statistics
                f.write("=== Key Flexibility Metrics ===\n")

                numeric_cols = results_df.select_dtypes(
                    include=[np.number]).columns
                key_metrics = [col for col in numeric_cols if any(keyword in col.lower()
                                                                  for keyword in ['flexibility', 'diversity', 'revenue_share', 'ramping'])]

                for metric in key_metrics:
                    if metric in results_df.columns:
                        f.write(f"{metric}:\n")
                        f.write(f"  Mean: {results_df[metric].mean():.3f}\n")
                        f.write(f"  Max: {results_df[metric].max():.3f}\n")
                        f.write(f"  Min: {results_df[metric].min():.3f}\n")
                        f.write(f"  Std: {results_df[metric].std():.3f}\n\n")

                # ISO-specific analysis
                f.write("=== ISO-Specific Analysis ===\n")
                for iso in results_df['iso_region'].unique():
                    iso_data = results_df[results_df['iso_region'] == iso]
                    f.write(f"\n{iso} ({len(iso_data)} reactors):\n")

                    if 'service_diversity_ratio' in iso_data.columns:
                        f.write(
                            f"  Avg Service Diversity: {iso_data['service_diversity_ratio'].mean():.3f}\n")
                    if 'ancillary_revenue_share' in iso_data.columns:
                        f.write(
                            f"  Avg Ancillary Revenue Share: {iso_data['ancillary_revenue_share'].mean():.3f}\n")
                    if 'overall_flexibility_score' in iso_data.columns:
                        f.write(
                            f"  Avg Flexibility Score: {iso_data['overall_flexibility_score'].mean():.3f}\n")

                f.write(
                    f"\nDetailed results saved to: {summary_file.replace('_summary.txt', '.csv')}\n")

            logger.info(f"Summary report saved to {summary_file}")

        except Exception as e:
            logger.error(f"Error creating summary report: {e}")


def main():
    """
    Main function to run the ancillary services analysis
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/ancillary_analysis.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("Starting Ancillary Services Analysis")

    try:
        # Initialize analyzer
        analyzer = AncillaryServicesAnalyzer()

        # Load data
        logger.info("Loading hourly results data...")
        reactor_data = analyzer.load_hourly_results()

        if not reactor_data:
            logger.error(
                "No reactor data loaded. Please check the results directory.")
            return

        # Run analysis
        logger.info("Running comprehensive analysis...")
        results_df = analyzer.analyze_all_reactors()

        if results_df.empty:
            logger.error("No analysis results generated.")
            return

        # Save results
        analyzer.save_results(results_df)

        logger.info("Analysis completed successfully!")
        logger.info(
            f"Analyzed {len(results_df)} reactors across {len(results_df['iso_region'].unique())} ISO regions")

        # Print summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Total Reactors: {len(results_df)}")
        print(f"ISO Regions: {', '.join(results_df['iso_region'].unique())}")

        if 'service_diversity_ratio' in results_df.columns:
            print(
                f"Average Service Diversity: {results_df['service_diversity_ratio'].mean():.3f}")
        if 'ancillary_revenue_share' in results_df.columns:
            print(
                f"Average Ancillary Revenue Share: {results_df['ancillary_revenue_share'].mean():.3f}")
        if 'overall_flexibility_score' in results_df.columns:
            print(
                f"Average Flexibility Score: {results_df['overall_flexibility_score'].mean():.3f}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
