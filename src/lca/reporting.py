"""
Nuclear Power Plant LCA Reporting
Comprehensive reporting system for life cycle assessment results

This module provides detailed analysis reports, summary statistics,
and integration with TEA and optimization results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import logging
import json
from jinja2 import Template

from .models import (
    ComprehensiveLCAResults, LifecycleEmissions, LCAResults,
    NuclearPlantParameters, HydrogenProductionData, ReactorType
)
from .config import config, LCA_REPORTS_DIR, LCA_DATA_DIR
from .calculator import NuclearLCACalculator

# Set up logging
logger = logging.getLogger(__name__)


class LCAReporter:
    """Comprehensive LCA reporting system"""

    def __init__(self, output_dir: Union[str, Path] = None):
        """
        Initialize LCA reporter

        Args:
            output_dir: Directory to save reports (defaults to configured LCA reports directory)
        """
        if output_dir is None:
            output_dir = LCA_REPORTS_DIR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize calculator for additional analyses
        self.calculator = NuclearLCACalculator()

        logger.info(
            f"LCA Reporter initialized, output directory: {self.output_dir}")

    def generate_comprehensive_report(self,
                                      comprehensive_results: ComprehensiveLCAResults,
                                      report_name: str = "comprehensive_lca_report") -> Dict[str, Any]:
        """
        Generate comprehensive LCA analysis report

        Args:
            comprehensive_results: Comprehensive LCA results
            report_name: Name for the report files

        Returns:
            Dictionary with report data and file paths
        """
        logger.info(f"Generating comprehensive LCA report: {report_name}")

        # Generate all report sections
        report_data = {
            'metadata': self._generate_metadata(comprehensive_results),
            'executive_summary': self._generate_executive_summary(comprehensive_results),
            'nuclear_analysis': self._generate_nuclear_analysis(comprehensive_results),
            'comparative_analysis': self._generate_comparative_analysis(comprehensive_results),
            'uncertainty_analysis': self._generate_uncertainty_analysis(comprehensive_results),
            'economic_analysis': self._generate_economic_analysis(comprehensive_results),
            'policy_implications': self._generate_policy_implications(comprehensive_results),
            'conclusions': self._generate_conclusions(comprehensive_results)
        }

        # Add hydrogen analysis if applicable
        if comprehensive_results.lca_results.integrated_system_emissions:
            report_data['hydrogen_analysis'] = self._generate_hydrogen_analysis(
                comprehensive_results)

        # Save detailed data as JSON
        json_path = self.output_dir / f"{report_name}_data.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate summary tables
        summary_tables = self._generate_summary_tables(comprehensive_results)
        excel_path = self.output_dir / f"{report_name}_tables.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in summary_tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)

        # Generate text report
        text_report = self._generate_text_report(report_data)
        text_path = self.output_dir / f"{report_name}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        logger.info(
            f"Comprehensive report generated with {len(report_data)} sections")

        return {
            'report_data': report_data,
            'file_paths': {
                'json': json_path,
                'excel': excel_path,
                'text': text_path,
            }
        }

    def generate_multi_plant_comparison(self,
                                        results_list: List[ComprehensiveLCAResults],
                                        comparison_name: str = "multi_plant_comparison") -> Dict[str, Any]:
        """
        Generate comparison report for multiple nuclear plants

        Args:
            results_list: List of comprehensive LCA results for different plants
            comparison_name: Name for the comparison report

        Returns:
            Dictionary with comparison data and file paths
        """
        logger.info(f"Generating multi-plant comparison: {comparison_name}")

        # Create comparison data
        comparison_data = []
        for result in results_list:
            plant_data = {
                'plant_name': result.lca_results.plant_parameters.plant_name,
                'reactor_type': result.lca_results.plant_parameters.reactor_type.value,
                'capacity_mw': result.lca_results.plant_parameters.electric_power_mw,
                'nuclear_emissions_gco2_per_kwh': result.lca_results.nuclear_carbon_intensity,
                'emission_reduction_vs_coal_percent': result.emission_reduction_vs_coal_percent,
                'emission_reduction_vs_gas_percent': result.emission_reduction_vs_gas_percent,
                'has_hydrogen': result.lca_results.integrated_system_emissions is not None
            }

            # Add economic data if available
            if result.tea_data:
                plant_data.update({
                    'lcoe_usd_per_mwh': result.tea_data.lcoe_usd_per_mwh,
                    'carbon_abatement_cost': result.carbon_abatement_cost_vs_coal
                })

            # Add optimization data if available
            if result.optimization_data:
                plant_data.update({
                    'flexibility_utilization': result.optimization_data.flexibility_utilization_factor,
                    'revenue_improvement_percent': result.optimization_data.revenue_improvement_percent
                })

            comparison_data.append(plant_data)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Generate statistics
        stats_summary = {
            'total_plants': len(comparison_data),
            'reactor_types': comparison_df['reactor_type'].value_counts().to_dict(),
            'emissions_statistics': {
                'mean_gco2_per_kwh': comparison_df['nuclear_emissions_gco2_per_kwh'].mean(),
                'median_gco2_per_kwh': comparison_df['nuclear_emissions_gco2_per_kwh'].median(),
                'min_gco2_per_kwh': comparison_df['nuclear_emissions_gco2_per_kwh'].min(),
                'max_gco2_per_kwh': comparison_df['nuclear_emissions_gco2_per_kwh'].max(),
                'std_gco2_per_kwh': comparison_df['nuclear_emissions_gco2_per_kwh'].std()
            },
            'best_performers': {
                'lowest_emissions': comparison_df.loc[comparison_df['nuclear_emissions_gco2_per_kwh'].idxmin(), 'plant_name'],
                'highest_coal_reduction': comparison_df.loc[comparison_df['emission_reduction_vs_coal_percent'].idxmax(), 'plant_name']
            }
        }

        # Save comparison data
        excel_path = self.output_dir / f"{comparison_name}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            comparison_df.to_excel(
                writer, sheet_name='Plant Comparison', index=False)

            # Create statistics sheet
            stats_df = pd.DataFrame([stats_summary['emissions_statistics']])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # Generate comparison report
        comparison_report = self._generate_comparison_text_report(
            comparison_data, stats_summary)
        text_path = self.output_dir / f"{comparison_name}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(comparison_report)

        logger.info(
            f"Multi-plant comparison generated for {len(results_list)} plants")

        return {
            'comparison_data': comparison_data,
            'statistics': stats_summary,
            'file_paths': {
                'excel': excel_path,
                'text': text_path
            }
        }

    def generate_sensitivity_analysis_report(self,
                                             base_results: ComprehensiveLCAResults,
                                             sensitivity_parameters: Dict[str, List[float]],
                                             report_name: str = "sensitivity_analysis") -> Dict[str, Any]:
        """
        Generate sensitivity analysis report

        Args:
            base_results: Base case LCA results
            sensitivity_parameters: Dictionary of parameters and their test values
            report_name: Name for the sensitivity report

        Returns:
            Dictionary with sensitivity analysis data
        """
        logger.info(f"Generating sensitivity analysis report: {report_name}")

        sensitivity_results = []
        base_plant_params = base_results.lca_results.plant_parameters

        # Perform sensitivity analysis for each parameter
        for param_name, param_values in sensitivity_parameters.items():
            for value in param_values:
                # Create modified plant parameters
                modified_params = base_plant_params
                if hasattr(modified_params, param_name):
                    setattr(modified_params, param_name, value)

                    # Calculate emissions with modified parameters
                    modified_emissions = self.calculator.calculate_nuclear_only_emissions(
                        modified_params)

                    sensitivity_results.append({
                        'parameter': param_name,
                        'value': value,
                        'emissions_gco2_per_kwh': modified_emissions.total_nuclear_only,
                        'change_from_base_percent': ((modified_emissions.total_nuclear_only -
                                                      base_results.lca_results.nuclear_carbon_intensity) /
                                                     base_results.lca_results.nuclear_carbon_intensity) * 100
                    })

        # Create sensitivity DataFrame
        sensitivity_df = pd.DataFrame(sensitivity_results)

        # Calculate sensitivity indices
        sensitivity_indices = {}
        for param in sensitivity_parameters.keys():
            param_data = sensitivity_df[sensitivity_df['parameter'] == param]
            if len(param_data) > 1:
                sensitivity_indices[param] = {
                    'range_percent': param_data['change_from_base_percent'].max() - param_data['change_from_base_percent'].min(),
                    'std_percent': param_data['change_from_base_percent'].std()
                }

        # Save sensitivity analysis
        excel_path = self.output_dir / f"{report_name}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            sensitivity_df.to_excel(
                writer, sheet_name='Sensitivity Results', index=False)

            indices_df = pd.DataFrame(sensitivity_indices).T
            indices_df.to_excel(
                writer, sheet_name='Sensitivity Indices', index=True)

        logger.info(
            f"Sensitivity analysis completed for {len(sensitivity_parameters)} parameters")

        return {
            'sensitivity_results': sensitivity_results,
            'sensitivity_indices': sensitivity_indices,
            'file_paths': {
                'excel': excel_path
            }
        }

    def _generate_metadata(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'report_generation_time': datetime.now().isoformat(),
            'analysis_id': results.lca_results.analysis_id,
            'plant_name': results.lca_results.plant_parameters.plant_name,
            'reactor_type': results.lca_results.plant_parameters.reactor_type.value,
            'analysis_scope': 'Nuclear-only' if not results.lca_results.integrated_system_emissions else 'Integrated nuclear-hydrogen',
            'methodology': 'ISO 14040/14044 compliant LCA',
            'functional_unit': config.methodology_parameters['functional_unit_electricity'],
            'system_boundary': config.methodology_parameters['system_boundary'],
            'reference_emission_factor': config.get_default_nuclear_emissions()
        }

    def _generate_executive_summary(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate executive summary"""
        nuclear_emissions = results.lca_results.nuclear_carbon_intensity

        summary = {
            'key_findings': {
                'nuclear_carbon_intensity_gco2_per_kwh': nuclear_emissions,
                'emission_reduction_vs_coal_percent': results.emission_reduction_vs_coal_percent,
                'emission_reduction_vs_gas_percent': results.emission_reduction_vs_gas_percent,
                'lifetime_coal_avoidance_ktco2': self._calculate_lifetime_avoidance(results, 'coal'),
                'lifetime_gas_avoidance_ktco2': self._calculate_lifetime_avoidance(results, 'gas')
            },
            'technology_ranking': self._get_technology_ranking(nuclear_emissions),
            'main_contributors': self._identify_main_emission_contributors(results.lca_results.nuclear_only_emissions)
        }

        if results.lca_results.integrated_system_emissions:
            integrated_emissions = results.lca_results.integrated_system_emissions.total_integrated_system
            summary['key_findings']['integrated_system_emissions_gco2_per_kwh'] = integrated_emissions
            summary['key_findings']['hydrogen_system_impact_gco2_per_kwh'] = (
                integrated_emissions - nuclear_emissions
            )

        return summary

    def _generate_nuclear_analysis(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate detailed nuclear power analysis"""
        emissions = results.lca_results.nuclear_only_emissions
        plant_params = results.lca_results.plant_parameters

        return {
            'plant_specifications': {
                'thermal_power_mw': plant_params.thermal_power_mw,
                'electric_power_mw': plant_params.electric_power_mw,
                'thermal_efficiency': plant_params.thermal_efficiency,
                'capacity_factor': plant_params.capacity_factor,
                'plant_lifetime_years': plant_params.plant_lifetime_years,
                'annual_generation_gwh': plant_params.annual_electricity_generation_mwh / 1000,
                'lifetime_generation_twh': plant_params.lifetime_electricity_generation_mwh / 1000000
            },
            'fuel_cycle_analysis': {
                'front_end_total_gco2_per_kwh': emissions.total_front_end,
                'uranium_mining_milling_gco2_per_kwh': emissions.uranium_mining_milling,
                'uranium_conversion_gco2_per_kwh': emissions.uranium_conversion,
                'uranium_enrichment_gco2_per_kwh': emissions.uranium_enrichment,
                'fuel_fabrication_gco2_per_kwh': emissions.fuel_fabrication,
                'back_end_total_gco2_per_kwh': emissions.total_back_end,
                'waste_management_gco2_per_kwh': emissions.waste_management,
                'decommissioning_gco2_per_kwh': emissions.decommissioning
            },
            'plant_lifecycle_analysis': {
                'plant_total_gco2_per_kwh': emissions.total_plant,
                'construction_gco2_per_kwh': emissions.plant_construction,
                'operation_gco2_per_kwh': emissions.plant_operation,
                'construction_materials': {
                    'concrete_tonnes': plant_params.concrete_tonnes,
                    'steel_tonnes': plant_params.steel_tonnes
                }
            },
            'emission_breakdown_percentages': {
                'front_end_percent': (emissions.total_front_end / emissions.total_nuclear_only) * 100,
                'plant_percent': (emissions.total_plant / emissions.total_nuclear_only) * 100,
                'back_end_percent': (emissions.total_back_end / emissions.total_nuclear_only) * 100
            }
        }

    def _generate_comparative_analysis(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate comparative analysis with other technologies"""
        nuclear_emissions = results.lca_results.nuclear_carbon_intensity

        # Get avoided emissions for all comparison technologies
        avoided_emissions = {}
        emission_factors = {}

        for tech_name, tech_data in config.comparison_technologies.items():
            avoided = self.calculator.calculate_avoided_emissions(
                nuclear_emissions, tech_name)
            avoided_emissions[tech_name] = avoided
            emission_factors[tech_name] = tech_data.value

        return {
            'emission_factors_gco2_per_kwh': emission_factors,
            'avoided_emissions': avoided_emissions,
            'technology_comparison': {
                'nuclear_rank': self._get_nuclear_rank_among_technologies(nuclear_emissions),
                'technologies_with_lower_emissions': self._get_technologies_with_lower_emissions(nuclear_emissions),
                'emission_reduction_potential': self._calculate_emission_reduction_potential(results)
            }
        }

    def _generate_uncertainty_analysis(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate uncertainty analysis (placeholder for future implementation)"""
        return {
            'uncertainty_sources': [
                'Uranium ore grade variability',
                'Mining method selection',
                'Enrichment technology choice',
                'Plant lifetime assumptions',
                'Capacity factor variations'
            ],
            'recommended_sensitivity_parameters': {
                'capacity_factor': [0.80, 0.85, 0.90, 0.95],
                'plant_lifetime_years': [40, 50, 60, 70, 80],
                'fuel_enrichment_percent': [3.5, 4.0, 4.2, 4.5, 5.0]
            },
            'data_quality_assessment': {
                'temporal_coverage': 'Excellent (2014-2024 data)',
                'geographical_coverage': 'Good (Global/OECD)',
                'technology_coverage': 'Excellent (Current commercial PWR)',
                'completeness': 'Very good (>95% of impacts covered)'
            }
        }

    def _generate_economic_analysis(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate economic analysis"""
        analysis = {
            'carbon_value_analysis': {
                'social_cost_of_carbon_usd_per_tonne': 51,  # US EPA 2021 estimate
                'carbon_value_usd_per_mwh': self._calculate_carbon_value(results, 51)
            }
        }

        if results.tea_data:
            analysis.update({
                'cost_analysis': {
                    'lcoe_usd_per_mwh': results.tea_data.lcoe_usd_per_mwh,
                    'carbon_abatement_cost_usd_per_tonne': results.carbon_abatement_cost_vs_coal,
                    'annual_revenue_usd': results.tea_data.total_annual_revenue_usd
                }
            })

        if results.optimization_data:
            analysis.update({
                'optimization_benefits': {
                    'revenue_improvement_percent': results.optimization_data.revenue_improvement_percent,
                    'flexibility_utilization_factor': results.optimization_data.flexibility_utilization_factor,
                    'annual_optimization_value_usd': results.optimization_data.optimized_annual_revenue_usd
                }
            })

        return analysis

    def _generate_hydrogen_analysis(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate hydrogen system analysis"""
        if not results.lca_results.hydrogen_data:
            return {}

        hydrogen_data = results.lca_results.hydrogen_data
        integrated_emissions = results.lca_results.integrated_system_emissions

        return {
            'hydrogen_production': {
                'annual_production_kg': hydrogen_data.annual_hydrogen_production_kg,
                'electrolyzer_capacity_mw': hydrogen_data.electrolyzer_capacity_mw,
                'electrolyzer_efficiency': hydrogen_data.electrolyzer_efficiency,
                'electricity_consumption_mwh_per_kg': hydrogen_data.electricity_consumption_mwh_per_kg
            },
            'hydrogen_emissions': {
                'production_emissions_kg_co2_per_kg_h2': results.lca_results.hydrogen_emissions_kg_co2_per_kg,
                'electrolyzer_construction_gco2_per_kwh': integrated_emissions.electrolyzer_construction,
                'electrolyzer_operation_gco2_per_kwh': integrated_emissions.electrolyzer_operation,
                'storage_transport_gco2_per_kwh': integrated_emissions.hydrogen_storage_transport
            },
            'comparison_with_conventional_hydrogen': {
                'conventional_smr_kg_co2_per_kg_h2': config.hydrogen_production_emissions['steam_methane_reforming'].value,
                'emission_reduction_vs_smr_kg_co2_per_kg_h2': (
                    config.hydrogen_production_emissions['steam_methane_reforming'].value -
                    results.lca_results.hydrogen_emissions_kg_co2_per_kg
                ),
                'emission_reduction_vs_smr_percent': (
                    (config.hydrogen_production_emissions['steam_methane_reforming'].value -
                     results.lca_results.hydrogen_emissions_kg_co2_per_kg) /
                    config.hydrogen_production_emissions['steam_methane_reforming'].value
                ) * 100
            }
        }

    def _generate_policy_implications(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate policy implications and recommendations"""
        return {
            'climate_policy_alignment': {
                'paris_agreement_contribution': 'Significant emission reductions support 1.5°C target',
                'net_zero_compatibility': 'Fully compatible with net-zero electricity systems',
                'carbon_pricing_impact': 'Becomes more competitive with higher carbon prices'
            },
            'energy_security_benefits': {
                'domestic_fuel_supply': 'Reduces dependence on fossil fuel imports',
                'price_stability': 'Provides stable long-term electricity costs',
                'grid_stability': 'Offers reliable baseload power and grid services'
            },
            'recommendations': [
                'Include nuclear power in clean energy portfolios',
                'Recognize nuclear\'s role in carbon abatement strategies',
                'Support lifecycle assessment in energy planning',
                'Consider nuclear-hydrogen integration for deep decarbonization'
            ]
        }

    def _generate_conclusions(self, results: ComprehensiveLCAResults) -> Dict[str, Any]:
        """Generate conclusions and key takeaways"""
        nuclear_emissions = results.lca_results.nuclear_carbon_intensity

        conclusions = {
            'main_conclusions': [
                f'Nuclear power lifecycle emissions: {nuclear_emissions:.1f} gCO₂-eq/kWh',
                f'Achieves {results.emission_reduction_vs_coal_percent:.1f}% emission reduction vs coal',
                f'Achieves {results.emission_reduction_vs_gas_percent:.1f}% emission reduction vs natural gas',
                'Ranks among the lowest-carbon electricity generation technologies'
            ],
            'key_insights': [
                'Front-end fuel cycle dominates lifecycle emissions',
                'Construction and operation phases contribute minimally',
                'Technology choice (reactor type) has limited impact on total emissions',
                'Integration with hydrogen production maintains low carbon intensity'
            ],
            'future_research_needs': [
                'Advanced reactor technology assessment',
                'Regional variation analysis',
                'Waste management technology improvements',
                'Nuclear-renewable hybrid system optimization'
            ]
        }

        if results.lca_results.integrated_system_emissions:
            integrated_emissions = results.lca_results.integrated_system_emissions.total_integrated_system
            conclusions['main_conclusions'].append(
                f'Integrated nuclear-hydrogen system: {integrated_emissions:.1f} gCO₂-eq/kWh'
            )

        return conclusions

    def _generate_summary_tables(self, results: ComprehensiveLCAResults) -> Dict[str, pd.DataFrame]:
        """Generate summary tables for Excel export"""
        tables = {}

        # Emission breakdown table
        emissions = results.lca_results.nuclear_only_emissions
        emission_data = {
            'Lifecycle Stage': [
                'Uranium Mining & Milling', 'Uranium Conversion', 'Uranium Enrichment',
                'Fuel Fabrication', 'Plant Construction', 'Plant Operation',
                'Waste Management', 'Decommissioning'
            ],
            'Emissions (gCO₂-eq/kWh)': [
                emissions.uranium_mining_milling, emissions.uranium_conversion,
                emissions.uranium_enrichment, emissions.fuel_fabrication,
                emissions.plant_construction, emissions.plant_operation,
                emissions.waste_management, emissions.decommissioning
            ]
        }

        # Add percentages
        total_emissions = emissions.total_nuclear_only
        emission_data['Percentage (%)'] = [
            (value / total_emissions) * 100 for value in emission_data['Emissions (gCO₂-eq/kWh)']
        ]

        tables['Emission Breakdown'] = pd.DataFrame(emission_data)

        # Technology comparison table
        comparison_data = config.get_comparison_data()
        comparison_data['Nuclear (This Study)'] = results.lca_results.nuclear_carbon_intensity

        tech_df = pd.DataFrame([
            {'Technology': tech, 'Emissions (gCO₂-eq/kWh)': emissions}
            for tech, emissions in comparison_data.items()
        ])
        tech_df = tech_df.sort_values('Emissions (gCO₂-eq/kWh)')
        tables['Technology Comparison'] = tech_df

        # Plant specifications table
        plant_params = results.lca_results.plant_parameters
        spec_data = {
            'Parameter': [
                'Plant Name', 'Reactor Type', 'Electric Power (MW)', 'Thermal Power (MW)',
                'Thermal Efficiency (%)', 'Capacity Factor (%)', 'Plant Lifetime (years)',
                'Annual Generation (GWh)', 'Lifetime Generation (TWh)'
            ],
            'Value': [
                plant_params.plant_name, plant_params.reactor_type.value,
                plant_params.electric_power_mw, plant_params.thermal_power_mw,
                plant_params.thermal_efficiency * 100, plant_params.capacity_factor * 100,
                plant_params.plant_lifetime_years,
                plant_params.annual_electricity_generation_mwh / 1000,
                plant_params.lifetime_electricity_generation_mwh / 1000000
            ]
        }
        tables['Plant Specifications'] = pd.DataFrame(spec_data)

        return tables

    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive text report"""
        lines = []

        # Header
        lines.append("="*80)
        lines.append("NUCLEAR POWER PLANT LIFE CYCLE ASSESSMENT REPORT")
        lines.append("="*80)
        lines.append("")

        # Metadata
        metadata = report_data['metadata']
        lines.append("REPORT METADATA")
        lines.append("-" * 40)
        lines.append(f"Plant Name: {metadata['plant_name']}")
        lines.append(f"Reactor Type: {metadata['reactor_type']}")
        lines.append(f"Analysis Scope: {metadata['analysis_scope']}")
        lines.append(f"Report Generated: {metadata['report_generation_time']}")
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        summary = report_data['executive_summary']
        key_findings = summary['key_findings']
        lines.append(
            f"Nuclear Carbon Intensity: {key_findings['nuclear_carbon_intensity_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"Emission Reduction vs Coal: {key_findings['emission_reduction_vs_coal_percent']:.1f}%")
        lines.append(
            f"Emission Reduction vs Natural Gas: {key_findings['emission_reduction_vs_gas_percent']:.1f}%")
        lines.append(
            f"Lifetime Coal Avoidance: {key_findings['lifetime_coal_avoidance_ktco2']:.0f} ktCO₂")
        lines.append("")

        # Main Contributors
        lines.append("MAIN EMISSION CONTRIBUTORS")
        lines.append("-" * 40)
        contributors = summary['main_contributors']
        for i, (stage, value) in enumerate(contributors.items(), 1):
            lines.append(
                f"{i}. {stage.replace('_', ' ').title()}: {value:.2f} gCO₂-eq/kWh")
        lines.append("")

        # Nuclear Analysis
        lines.append("DETAILED NUCLEAR ANALYSIS")
        lines.append("-" * 40)
        nuclear_analysis = report_data['nuclear_analysis']

        lines.append("Fuel Cycle Breakdown:")
        fuel_cycle = nuclear_analysis['fuel_cycle_analysis']
        lines.append(
            f"  Front-end Total: {fuel_cycle['front_end_total_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"  Back-end Total: {fuel_cycle['back_end_total_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append("")

        lines.append("Plant Lifecycle:")
        plant_lifecycle = nuclear_analysis['plant_lifecycle_analysis']
        lines.append(
            f"  Construction: {plant_lifecycle['construction_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"  Operation: {plant_lifecycle['operation_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append("")

        # Conclusions
        lines.append("CONCLUSIONS")
        lines.append("-" * 40)
        conclusions = report_data['conclusions']
        for conclusion in conclusions['main_conclusions']:
            lines.append(f"• {conclusion}")
        lines.append("")

        lines.append("Key Insights:")
        for insight in conclusions['key_insights']:
            lines.append(f"• {insight}")
        lines.append("")

        lines.append("="*80)

        return "\n".join(lines)

    def _generate_comparison_text_report(self, comparison_data: List[Dict], stats: Dict) -> str:
        """Generate text report for multi-plant comparison"""
        lines = []

        lines.append("="*80)
        lines.append("MULTI-PLANT NUCLEAR LCA COMPARISON REPORT")
        lines.append("="*80)
        lines.append("")

        lines.append(f"Total Plants Analyzed: {stats['total_plants']}")
        lines.append("")

        lines.append("EMISSION STATISTICS")
        lines.append("-" * 40)
        emission_stats = stats['emissions_statistics']
        lines.append(
            f"Mean Emissions: {emission_stats['mean_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"Median Emissions: {emission_stats['median_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"Range: {emission_stats['min_gco2_per_kwh']:.2f} - {emission_stats['max_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append(
            f"Standard Deviation: {emission_stats['std_gco2_per_kwh']:.2f} gCO₂-eq/kWh")
        lines.append("")

        lines.append("BEST PERFORMERS")
        lines.append("-" * 40)
        best = stats['best_performers']
        lines.append(f"Lowest Emissions: {best['lowest_emissions']}")
        lines.append(
            f"Highest Coal Reduction: {best['highest_coal_reduction']}")
        lines.append("")

        lines.append("REACTOR TYPE DISTRIBUTION")
        lines.append("-" * 40)
        for reactor_type, count in stats['reactor_types'].items():
            lines.append(f"{reactor_type}: {count} plants")
        lines.append("")

        return "\n".join(lines)

    # Helper methods for calculations

    def _calculate_lifetime_avoidance(self, results: ComprehensiveLCAResults, comparison_tech: str) -> float:
        """Calculate lifetime emission avoidance"""
        plant_params = results.lca_results.plant_parameters
        nuclear_emissions = results.lca_results.nuclear_carbon_intensity

        if comparison_tech == 'coal':
            comparison_emissions = 820  # gCO2-eq/kWh
        elif comparison_tech == 'gas':
            comparison_emissions = 490  # gCO2-eq/kWh
        else:
            return 0

        annual_generation = plant_params.annual_electricity_generation_mwh
        lifetime_generation = annual_generation * plant_params.plant_lifetime_years

        avoided_per_kwh = (comparison_emissions -
                           nuclear_emissions) / 1000  # kgCO2-eq/kWh
        total_avoided = avoided_per_kwh * lifetime_generation / 1000  # ktCO2

        return total_avoided

    def _get_technology_ranking(self, nuclear_emissions: float) -> int:
        """Get nuclear power ranking among all technologies"""
        comparison_data = config.get_comparison_data()
        all_emissions = list(comparison_data.values()) + [nuclear_emissions]
        all_emissions.sort()

        return all_emissions.index(nuclear_emissions) + 1

    def _identify_main_emission_contributors(self, emissions: LifecycleEmissions) -> Dict[str, float]:
        """Identify top 3 emission contributors"""
        contributors = {
            'uranium_mining_milling': emissions.uranium_mining_milling,
            'uranium_conversion': emissions.uranium_conversion,
            'uranium_enrichment': emissions.uranium_enrichment,
            'fuel_fabrication': emissions.fuel_fabrication,
            'plant_construction': emissions.plant_construction,
            'plant_operation': emissions.plant_operation,
            'waste_management': emissions.waste_management,
            'decommissioning': emissions.decommissioning
        }

        # Sort by value and return top 3
        sorted_contributors = dict(
            sorted(contributors.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_contributors.items())[:3])

    def _get_nuclear_rank_among_technologies(self, nuclear_emissions: float) -> int:
        """Get nuclear rank among all technologies"""
        return self._get_technology_ranking(nuclear_emissions)

    def _get_technologies_with_lower_emissions(self, nuclear_emissions: float) -> List[str]:
        """Get list of technologies with lower emissions than nuclear"""
        comparison_data = config.get_comparison_data()
        lower_emission_techs = [
            tech for tech, emissions in comparison_data.items()
            if emissions < nuclear_emissions
        ]
        return lower_emission_techs

    def _calculate_emission_reduction_potential(self, results: ComprehensiveLCAResults) -> Dict[str, float]:
        """Calculate emission reduction potential for different scenarios"""
        nuclear_emissions = results.lca_results.nuclear_carbon_intensity
        plant_params = results.lca_results.plant_parameters
        annual_generation = plant_params.annual_electricity_generation_mwh

        return {
            'vs_coal_tco2_per_year': ((820 - nuclear_emissions) / 1000) * annual_generation / 1000,
            'vs_gas_tco2_per_year': ((490 - nuclear_emissions) / 1000) * annual_generation / 1000,
            'vs_world_avg_grid_tco2_per_year': ((475 - nuclear_emissions) / 1000) * annual_generation / 1000
        }

    def _calculate_carbon_value(self, results: ComprehensiveLCAResults, carbon_price: float) -> float:
        """Calculate carbon value based on avoided emissions"""
        coal_avoided = self.calculator.calculate_avoided_emissions(
            results.lca_results.nuclear_carbon_intensity, 'coal_pc'
        )
        return coal_avoided['avoided_emissions_kg_co2_per_mwh'] * carbon_price / 1000
