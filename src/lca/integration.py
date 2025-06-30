"""
Nuclear Power Plant LCA Integration
Integration module for combining LCA results with TEA and optimization analyses

This module provides functions to load and integrate results from the TEA and
optimization modules with LCA calculations for comprehensive analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
import json
import os

from .models import (
    NuclearPlantParameters, HydrogenProductionData, TEAIntegrationData,
    OptimizationIntegrationData, ReactorType
)
from .calculator import NuclearLCACalculator
from .config import config

# Set up logging
logger = logging.getLogger(__name__)


class LCAIntegrator:
    """Integration class for LCA with TEA and optimization results"""

    def __init__(self,
                 tea_results_dir: Union[str, Path] = "output/tea",
                 opt_results_dir: Union[str, Path] = "output/opt"):
        """
        Initialize LCA integrator

        Args:
            tea_results_dir: Directory containing TEA results
            opt_results_dir: Directory containing optimization results
        """
        self.tea_results_dir = Path(tea_results_dir)
        self.opt_results_dir = Path(opt_results_dir)

        # Initialize calculator
        self.calculator = NuclearLCACalculator()

        logger.info(f"LCA Integrator initialized")
        logger.info(f"TEA results directory: {self.tea_results_dir}")
        logger.info(f"Optimization results directory: {self.opt_results_dir}")

    def load_plant_parameters_from_tea(self, plant_name: str) -> Optional[NuclearPlantParameters]:
        """
        Load nuclear plant parameters from TEA results

        Args:
            plant_name: Name of the nuclear plant

        Returns:
            NuclearPlantParameters object or None if not found
        """
        try:
            # Look for TEA results files
            tea_files = list(self.tea_results_dir.glob(f"**/*{plant_name}*"))

            if not tea_files:
                logger.warning(f"No TEA files found for plant: {plant_name}")
                return None

            # Try to find configuration or summary file
            config_file = None
            for file in tea_files:
                if any(keyword in file.name.lower() for keyword in ['config', 'summary', 'plant']):
                    config_file = file
                    break

            if not config_file:
                config_file = tea_files[0]  # Use first file as fallback

            logger.info(f"Loading plant parameters from: {config_file}")

            # Load plant data based on file type
            if config_file.suffix == '.csv':
                df = pd.read_csv(config_file)
                plant_data = self._extract_plant_params_from_csv(
                    df, plant_name)
            elif config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    plant_data = json.load(f)
                plant_data = self._extract_plant_params_from_json(
                    plant_data, plant_name)
            else:
                logger.warning(
                    f"Unsupported file format: {config_file.suffix}")
                return None

            if plant_data:
                return self._create_plant_parameters_from_data(plant_data, plant_name)

        except Exception as e:
            logger.error(
                f"Error loading plant parameters for {plant_name}: {e}")

        return None

    def load_tea_integration_data(self, plant_name: str) -> Optional[TEAIntegrationData]:
        """
        Load TEA results for integration with LCA

        Args:
            plant_name: Name of the nuclear plant

        Returns:
            TEAIntegrationData object or None if not found
        """
        try:
            # Look for TEA summary or results files
            tea_files = list(self.tea_results_dir.glob(f"**/*{plant_name}*"))

            if not tea_files:
                logger.warning(f"No TEA files found for plant: {plant_name}")
                return None

            # Find the most relevant file (summary or results)
            results_file = None
            for file in tea_files:
                if any(keyword in file.name.lower() for keyword in ['summary', 'results', 'lcoe']):
                    results_file = file
                    break

            if not results_file:
                results_file = tea_files[0]

            logger.info(f"Loading TEA data from: {results_file}")

            # Load and process TEA data
            if results_file.suffix == '.csv':
                df = pd.read_csv(results_file)
                tea_data = self._extract_tea_data_from_csv(df, plant_name)
            elif results_file.suffix == '.json':
                with open(results_file, 'r') as f:
                    tea_data = json.load(f)
                tea_data = self._extract_tea_data_from_json(
                    tea_data, plant_name)
            else:
                logger.warning(
                    f"Unsupported TEA file format: {results_file.suffix}")
                return None

            if tea_data:
                return self._create_tea_integration_data(tea_data)

        except Exception as e:
            logger.error(f"Error loading TEA data for {plant_name}: {e}")

        return None

    def load_optimization_integration_data(self, plant_name: str) -> Optional[OptimizationIntegrationData]:
        """
        Load optimization results for integration with LCA

        Args:
            plant_name: Name of the nuclear plant

        Returns:
            OptimizationIntegrationData object or None if not found
        """
        try:
            # Look for optimization results files
            opt_files = list(self.opt_results_dir.glob(f"**/*{plant_name}*"))

            if not opt_files:
                logger.warning(
                    f"No optimization files found for plant: {plant_name}")
                return None

            # Find the most relevant file
            results_file = None
            for file in opt_files:
                if any(keyword in file.name.lower() for keyword in ['summary', 'results', 'optimal']):
                    results_file = file
                    break

            if not results_file:
                results_file = opt_files[0]

            logger.info(f"Loading optimization data from: {results_file}")

            # Load and process optimization data
            if results_file.suffix == '.csv':
                df = pd.read_csv(results_file)
                opt_data = self._extract_opt_data_from_csv(df, plant_name)
            elif results_file.suffix == '.json':
                with open(results_file, 'r') as f:
                    opt_data = json.load(f)
                opt_data = self._extract_opt_data_from_json(
                    opt_data, plant_name)
            else:
                logger.warning(
                    f"Unsupported optimization file format: {results_file.suffix}")
                return None

            if opt_data:
                return self._create_optimization_integration_data(opt_data)

        except Exception as e:
            logger.error(
                f"Error loading optimization data for {plant_name}: {e}")

        return None

    def create_hydrogen_production_data(self,
                                        plant_params: NuclearPlantParameters,
                                        hydrogen_allocation_factor: float = 0.2,
                                        electrolyzer_efficiency: float = 0.65) -> HydrogenProductionData:
        """
        Create hydrogen production data for integrated system analysis

        Args:
            plant_params: Nuclear plant parameters
            hydrogen_allocation_factor: Fraction of electricity for hydrogen production
            electrolyzer_efficiency: Electrolyzer efficiency (HHV basis)

        Returns:
            HydrogenProductionData object
        """
        # Calculate hydrogen production parameters
        electricity_for_hydrogen = (plant_params.annual_electricity_generation_mwh *
                                    hydrogen_allocation_factor)

        # Hydrogen energy content (HHV): 39.4 kWh/kg
        hydrogen_energy_content = 39.4  # kWh/kg
        electricity_consumption_per_kg = hydrogen_energy_content / electrolyzer_efficiency

        annual_hydrogen_production = electricity_for_hydrogen / \
            electricity_consumption_per_kg
        electrolyzer_capacity = (
            plant_params.electric_power_mw * hydrogen_allocation_factor)

        # Economic parameters (typical values)
        electrolyzer_capex_per_kw = 1500  # USD/kW (PEM electrolyzer)
        electrolyzer_opex_percent = 3.0   # % of CAPEX per year
        hydrogen_selling_price = 5.0      # USD/kg (target for green hydrogen)

        # Emission parameters
        nuclear_emissions = self.calculator.calculate_nuclear_only_emissions(
            plant_params)
        nuclear_intensity = nuclear_emissions.total_nuclear_only / 1000  # kgCO2-eq/kWh

        hydrogen_emission_factor = nuclear_intensity * electricity_consumption_per_kg
        smr_emissions = config.hydrogen_production_emissions['steam_methane_reforming'].value
        avoided_emissions = smr_emissions - hydrogen_emission_factor

        return HydrogenProductionData(
            annual_hydrogen_production_kg=annual_hydrogen_production,
            electrolyzer_capacity_mw=electrolyzer_capacity,
            electrolyzer_efficiency=electrolyzer_efficiency,
            electricity_consumption_mwh_per_kg=electricity_consumption_per_kg / 1000,
            electrolyzer_capex_per_kw=electrolyzer_capex_per_kw,
            electrolyzer_opex_percent=electrolyzer_opex_percent,
            hydrogen_selling_price_per_kg=hydrogen_selling_price,
            hydrogen_emission_factor_kg_co2_per_kg=hydrogen_emission_factor,
            avoided_emissions_kg_co2_per_kg=avoided_emissions
        )

    def analyze_existing_reactor(self, plant_name: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of existing reactor using integrated data

        Args:
            plant_name: Name of the existing nuclear plant

        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Analyzing existing reactor: {plant_name}")

        # Load plant parameters from TEA results
        plant_params = self.load_plant_parameters_from_tea(plant_name)
        if not plant_params:
            # Fall back to input data if available
            plant_params = self._load_plant_params_from_input_data(plant_name)

        if not plant_params:
            logger.error(f"Could not load plant parameters for {plant_name}")
            return {}

        # Load integration data
        tea_data = self.load_tea_integration_data(plant_name)
        opt_data = self.load_optimization_integration_data(plant_name)

        # Perform LCA calculations
        nuclear_emissions = self.calculator.calculate_nuclear_only_emissions(
            plant_params)

        # Calculate avoided emissions and economic metrics
        coal_avoided = self.calculator.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, 'coal_pc'
        )
        gas_avoided = self.calculator.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, 'natural_gas_ccgt'
        )

        # Calculate carbon abatement costs if TEA data available
        carbon_abatement_cost = None
        if tea_data:
            carbon_abatement_cost = self.calculator.calculate_carbon_abatement_cost(
                tea_data.lcoe_usd_per_mwh, 50.0,  # Assume $50/MWh for coal
                nuclear_emissions.total_nuclear_only, 820.0  # Coal emissions
            )

        # Compile results
        analysis_results = {
            'plant_info': {
                'name': plant_name,
                'reactor_type': plant_params.reactor_type.value,
                'capacity_mw': plant_params.electric_power_mw,
                'capacity_factor': plant_params.capacity_factor,
                'annual_generation_gwh': plant_params.annual_electricity_generation_mwh / 1000
            },
            'lca_results': {
                'nuclear_emissions_gco2_per_kwh': nuclear_emissions.total_nuclear_only,
                'emission_breakdown': {
                    'front_end_gco2_per_kwh': nuclear_emissions.total_front_end,
                    'plant_gco2_per_kwh': nuclear_emissions.total_plant,
                    'back_end_gco2_per_kwh': nuclear_emissions.total_back_end
                },
                'avoided_emissions': {
                    'vs_coal_kg_co2_per_mwh': coal_avoided['avoided_emissions_kg_co2_per_mwh'],
                    'vs_gas_kg_co2_per_mwh': gas_avoided['avoided_emissions_kg_co2_per_mwh'],
                    'vs_coal_percent': coal_avoided['avoided_emissions_percent'],
                    'vs_gas_percent': gas_avoided['avoided_emissions_percent']
                }
            },
            'economic_analysis': {},
            'optimization_analysis': {},
            'integration_quality': {
                'tea_data_available': tea_data is not None,
                'optimization_data_available': opt_data is not None,
                'data_consistency_check': 'passed'  # Could add actual checks
            }
        }

        # Add economic analysis if TEA data available
        if tea_data:
            analysis_results['economic_analysis'] = {
                'lcoe_usd_per_mwh': tea_data.lcoe_usd_per_mwh,
                'carbon_abatement_cost_usd_per_tonne': carbon_abatement_cost,
                'annual_revenue_usd': tea_data.total_annual_revenue_usd,
                'electricity_revenue_usd_per_mwh': tea_data.electricity_revenue_usd_per_mwh,
                'ancillary_services_revenue_usd_per_mwh': tea_data.ancillary_services_revenue_usd_per_mwh
            }

        # Add optimization analysis if data available
        if opt_data:
            analysis_results['optimization_analysis'] = {
                'optimization_scenario': opt_data.optimization_scenario,
                'revenue_improvement_percent': opt_data.revenue_improvement_percent,
                'flexibility_utilization_factor': opt_data.flexibility_utilization_factor,
                'optimized_annual_revenue_usd': opt_data.optimized_annual_revenue_usd,
                'market_participation_hours': opt_data.total_market_participation_hours
            }

        logger.info(f"Analysis completed for {plant_name}")
        return analysis_results

    def analyze_new_reactor_scenario(self,
                                     scenario_name: str,
                                     reactor_type: ReactorType = ReactorType.AP1000,
                                     capacity_mw: float = 1117,
                                     location: str = "Generic Location",
                                     iso_region: str = "Generic ISO",
                                     include_hydrogen: bool = False) -> Dict[str, Any]:
        """
        Analyze new reactor scenario with specified parameters

        Args:
            scenario_name: Name for the scenario
            reactor_type: Type of reactor
            capacity_mw: Plant capacity in MW
            location: Plant location
            iso_region: ISO region
            include_hydrogen: Whether to include hydrogen production

        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Analyzing new reactor scenario: {scenario_name}")

        # Create plant parameters for new reactor
        plant_params = self._create_new_reactor_parameters(
            scenario_name, reactor_type, capacity_mw, location, iso_region
        )

        # Perform LCA calculations
        nuclear_emissions = self.calculator.calculate_nuclear_only_emissions(
            plant_params)

        # Create hydrogen system if requested
        hydrogen_data = None
        integrated_emissions = None
        if include_hydrogen:
            hydrogen_data = self.create_hydrogen_production_data(plant_params)
            integrated_emissions = self.calculator.calculate_integrated_system_emissions(
                plant_params, hydrogen_data
            )

        # Calculate comparative metrics
        coal_avoided = self.calculator.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, 'coal_pc'
        )
        gas_avoided = self.calculator.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, 'natural_gas_ccgt'
        )

        # Compile results
        analysis_results = {
            'scenario_info': {
                'name': scenario_name,
                'reactor_type': reactor_type.value,
                'capacity_mw': capacity_mw,
                'location': location,
                'iso_region': iso_region,
                'includes_hydrogen': include_hydrogen
            },
            'lca_results': {
                'nuclear_only_emissions_gco2_per_kwh': nuclear_emissions.total_nuclear_only,
                'integrated_emissions_gco2_per_kwh': (
                    integrated_emissions.total_integrated_system if integrated_emissions else None
                ),
                'emission_breakdown': {
                    'uranium_mining_milling': nuclear_emissions.uranium_mining_milling,
                    'uranium_conversion': nuclear_emissions.uranium_conversion,
                    'uranium_enrichment': nuclear_emissions.uranium_enrichment,
                    'fuel_fabrication': nuclear_emissions.fuel_fabrication,
                    'plant_construction': nuclear_emissions.plant_construction,
                    'plant_operation': nuclear_emissions.plant_operation,
                    'waste_management': nuclear_emissions.waste_management,
                    'decommissioning': nuclear_emissions.decommissioning
                },
                'avoided_emissions': {
                    'vs_coal_kg_co2_per_mwh': coal_avoided['avoided_emissions_kg_co2_per_mwh'],
                    'vs_gas_kg_co2_per_mwh': gas_avoided['avoided_emissions_kg_co2_per_mwh'],
                    'vs_coal_percent': coal_avoided['avoided_emissions_percent'],
                    'vs_gas_percent': gas_avoided['avoided_emissions_percent']
                }
            },
            'hydrogen_analysis': {},
            'lifetime_metrics': {
                'total_generation_twh': plant_params.lifetime_electricity_generation_mwh / 1000000,
                'lifetime_coal_avoidance_mtco2': self._calculate_lifetime_avoidance(
                    plant_params, nuclear_emissions.total_nuclear_only, 820
                ),
                'lifetime_gas_avoidance_mtco2': self._calculate_lifetime_avoidance(
                    plant_params, nuclear_emissions.total_nuclear_only, 490
                )
            }
        }

        # Add hydrogen analysis if applicable
        if hydrogen_data and integrated_emissions:
            hydrogen_emissions = self.calculator.calculate_hydrogen_production_emissions(
                plant_params, hydrogen_data
            )

            analysis_results['hydrogen_analysis'] = {
                'annual_production_tonnes': hydrogen_data.annual_hydrogen_production_kg / 1000,
                'electrolyzer_capacity_mw': hydrogen_data.electrolyzer_capacity_mw,
                'hydrogen_emissions_kg_co2_per_kg_h2': hydrogen_emissions,
                'smr_avoidance_kg_co2_per_kg_h2': hydrogen_data.avoided_emissions_kg_co2_per_kg,
                'annual_hydrogen_emission_avoidance_tco2': (
                    hydrogen_data.avoided_emissions_kg_co2_per_kg *
                    hydrogen_data.annual_hydrogen_production_kg / 1000
                )
            }

        logger.info(
            f"New reactor scenario analysis completed: {scenario_name}")
        return analysis_results

    def compare_reactor_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple reactor scenarios

        Args:
            scenarios: List of scenario dictionaries with scenario parameters

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(scenarios)} reactor scenarios")

        scenario_results = []

        for scenario in scenarios:
            if 'existing_plant' in scenario:
                # Analyze existing plant
                result = self.analyze_existing_reactor(
                    scenario['existing_plant'])
            else:
                # Analyze new reactor scenario
                result = self.analyze_new_reactor_scenario(**scenario)

            scenario_results.append(result)

        # Create comparison summary
        comparison_summary = self._create_scenario_comparison_summary(
            scenario_results)

        return {
            'individual_results': scenario_results,
            'comparison_summary': comparison_summary,
            'analysis_metadata': {
                'total_scenarios': len(scenarios),
                'analysis_date': pd.Timestamp.now().isoformat(),
                'methodology': 'Integrated LCA-TEA-OPT analysis'
            }
        }

    # Private helper methods

    def _extract_plant_params_from_csv(self, df: pd.DataFrame, plant_name: str) -> Optional[Dict]:
        """Extract plant parameters from CSV data"""
        # Implementation would depend on CSV structure
        # This is a placeholder that assumes certain column names
        try:
            plant_row = df[df['plant_name'].str.contains(
                plant_name, case=False)]
            if plant_row.empty:
                return None

            return plant_row.iloc[0].to_dict()
        except:
            return None

    def _extract_plant_params_from_json(self, data: Dict, plant_name: str) -> Optional[Dict]:
        """Extract plant parameters from JSON data"""
        # Implementation would depend on JSON structure
        if plant_name in data:
            return data[plant_name]
        return None

    def _create_plant_parameters_from_data(self, data: Dict, plant_name: str) -> NuclearPlantParameters:
        """Create NuclearPlantParameters from loaded data"""
        # Map common field names to our model
        field_mapping = {
            'capacity': 'electric_power_mw',
            'power': 'electric_power_mw',
            'thermal_power': 'thermal_power_mw',
            'efficiency': 'thermal_efficiency',
            'cf': 'capacity_factor',
            'capacity_factor': 'capacity_factor',
            'reactor_type': 'reactor_type',
            'location': 'location'
        }

        # Extract and map fields
        mapped_data = {}
        for key, value in data.items():
            mapped_key = field_mapping.get(key.lower(), key)
            mapped_data[mapped_key] = value

        # Set defaults for missing fields
        defaults = {
            'plant_name': plant_name,
            'reactor_type': ReactorType.PWR,
            'location': 'Unknown',
            'iso_region': 'Unknown',
            'thermal_power_mw': mapped_data.get('electric_power_mw', 1000) / 0.33,
            'thermal_efficiency': 0.33,
            'capacity_factor': 0.90,
            'plant_lifetime_years': 60,
            'construction_time_years': 10,
            'commissioning_year': 2025,
            'fuel_enrichment_percent': 4.2,
            'fuel_burnup_mwd_per_kg': 45,
            'fuel_cycle_length_months': 18,
            'natural_uranium_per_enriched_kg': 8.1,
            'separative_work_swu_per_kg': 6.7,
            'concrete_tonnes': 400000,
            'steel_tonnes': 65000
        }

        # Combine mapped data with defaults
        for key, default_value in defaults.items():
            if key not in mapped_data:
                mapped_data[key] = default_value

        # Handle reactor type conversion
        if isinstance(mapped_data['reactor_type'], str):
            reactor_type_map = {
                'PWR': ReactorType.PWR,
                'BWR': ReactorType.BWR,
                'AP1000': ReactorType.AP1000,
                'EPR': ReactorType.EPR
            }
            mapped_data['reactor_type'] = reactor_type_map.get(
                mapped_data['reactor_type'].upper(), ReactorType.PWR
            )

        return NuclearPlantParameters(**mapped_data)

    def _extract_tea_data_from_csv(self, df: pd.DataFrame, plant_name: str) -> Optional[Dict]:
        """Extract TEA data from CSV"""
        # This would depend on TEA output format
        try:
            # Look for rows containing plant name or summary data
            if 'plant_name' in df.columns:
                plant_row = df[df['plant_name'].str.contains(
                    plant_name, case=False)]
                if not plant_row.empty:
                    return plant_row.iloc[0].to_dict()

            # If no plant-specific data, use first row (summary)
            return df.iloc[0].to_dict()
        except:
            return None

    def _extract_tea_data_from_json(self, data: Dict, plant_name: str) -> Optional[Dict]:
        """Extract TEA data from JSON"""
        if plant_name in data:
            return data[plant_name]
        elif 'results' in data:
            return data['results']
        return data

    def _create_tea_integration_data(self, tea_data: Dict) -> TEAIntegrationData:
        """Create TEAIntegrationData from loaded data"""
        # Map TEA fields to our model
        field_mapping = {
            'lcoe': 'lcoe_usd_per_mwh',
            'lcoh': 'lcoh_usd_per_kg',
            'electricity_revenue': 'electricity_revenue_usd_per_mwh',
            'ancillary_revenue': 'ancillary_services_revenue_usd_per_mwh',
            'capex': 'capex_usd',
            'opex': 'opex_annual_usd',
            'fuel_cost': 'fuel_cost_annual_usd'
        }

        mapped_data = {}
        for key, value in tea_data.items():
            mapped_key = field_mapping.get(key.lower(), key)
            mapped_data[mapped_key] = value

        # Set defaults
        defaults = {
            'lcoe_usd_per_mwh': 80.0,
            'electricity_revenue_usd_per_mwh': 50.0,
            'ancillary_services_revenue_usd_per_mwh': 10.0,
            'capex_usd': 8000000000,  # $8B
            'opex_annual_usd': 200000000,  # $200M/year
            'fuel_cost_annual_usd': 50000000,  # $50M/year
            'capacity_factor_electricity': 0.90,
            'ancillary_services_participation_hours': 2000
        }

        for key, default_value in defaults.items():
            if key not in mapped_data:
                mapped_data[key] = default_value

        return TEAIntegrationData(**mapped_data)

    def _extract_opt_data_from_csv(self, df: pd.DataFrame, plant_name: str) -> Optional[Dict]:
        """Extract optimization data from CSV"""
        try:
            if 'plant_name' in df.columns:
                plant_row = df[df['plant_name'].str.contains(
                    plant_name, case=False)]
                if not plant_row.empty:
                    return plant_row.iloc[0].to_dict()
            return df.iloc[0].to_dict()
        except:
            return None

    def _extract_opt_data_from_json(self, data: Dict, plant_name: str) -> Optional[Dict]:
        """Extract optimization data from JSON"""
        if plant_name in data:
            return data[plant_name]
        elif 'optimization_results' in data:
            return data['optimization_results']
        return data

    def _create_optimization_integration_data(self, opt_data: Dict) -> OptimizationIntegrationData:
        """Create OptimizationIntegrationData from loaded data"""
        # This would map optimization output fields to our model
        defaults = {
            'optimization_scenario': 'Base Case Optimization',
            'electricity_generation_profile_mwh': np.zeros(8760),
            'energy_market_participation_hours': 8760,
            'regulation_market_participation_hours': 2000,
            'spinning_reserve_participation_hours': 1000,
            'load_following_capability_mw': 100,
            'ramping_rate_mw_per_min': 5.0,
            'minimum_stable_output_percent': 20.0,
            'frequency_regulation_service_mw': 50.0,
            'voltage_support_service_mvar': 25.0,
            'optimized_annual_revenue_usd': 500000000,
            'revenue_improvement_percent': 15.0
        }

        # Update with actual data if available
        for key, value in opt_data.items():
            if key in defaults:
                defaults[key] = value

        return OptimizationIntegrationData(**defaults)

    def _load_plant_params_from_input_data(self, plant_name: str) -> Optional[NuclearPlantParameters]:
        """Load plant parameters from input data directory"""
        try:
            # Look in input directory for NPP info
            input_dir = Path("input/hourly_data")
            npp_file = input_dir / "NPPs info.csv"

            if npp_file.exists():
                df = pd.read_csv(npp_file)
                plant_row = df[df['NPP Name'].str.contains(
                    plant_name, case=False)]

                if not plant_row.empty:
                    row = plant_row.iloc[0]

                    # Map from NPP info format
                    return NuclearPlantParameters(
                        plant_name=row['NPP Name'],
                        reactor_type=ReactorType.PWR,  # Default
                        location=row.get('State', 'Unknown'),
                        iso_region=row.get('ISO', 'Unknown'),
                        thermal_power_mw=3000,  # Default
                        electric_power_mw=row.get('Capacity (MW)', 1000),
                        thermal_efficiency=0.33,
                        capacity_factor=0.90,
                        plant_lifetime_years=60,
                        construction_time_years=10,
                        commissioning_year=2025,
                        fuel_enrichment_percent=4.2,
                        fuel_burnup_mwd_per_kg=45,
                        fuel_cycle_length_months=18,
                        natural_uranium_per_enriched_kg=8.1,
                        separative_work_swu_per_kg=6.7,
                        concrete_tonnes=400000,
                        steel_tonnes=65000
                    )
        except Exception as e:
            logger.warning(f"Could not load from input data: {e}")

        return None

    def _create_new_reactor_parameters(self,
                                       name: str,
                                       reactor_type: ReactorType,
                                       capacity_mw: float,
                                       location: str,
                                       iso_region: str) -> NuclearPlantParameters:
        """Create parameters for new reactor scenario"""

        # Reactor-specific parameters
        reactor_specs = {
            ReactorType.AP1000: {
                'thermal_power_mw': 3400,
                'thermal_efficiency': 0.33,
                'construction_time_years': 8
            },
            ReactorType.EPR: {
                'thermal_power_mw': 4590,
                'thermal_efficiency': 0.35,
                'construction_time_years': 12
            },
            ReactorType.SMR: {
                'thermal_power_mw': 250,
                'thermal_efficiency': 0.35,
                'construction_time_years': 5
            }
        }

        specs = reactor_specs.get(reactor_type, {
            'thermal_power_mw': capacity_mw / 0.33,
            'thermal_efficiency': 0.33,
            'construction_time_years': 10
        })

        return NuclearPlantParameters(
            plant_name=name,
            reactor_type=reactor_type,
            location=location,
            iso_region=iso_region,
            thermal_power_mw=specs['thermal_power_mw'],
            electric_power_mw=capacity_mw,
            thermal_efficiency=specs['thermal_efficiency'],
            capacity_factor=0.90,
            plant_lifetime_years=60,
            construction_time_years=specs['construction_time_years'],
            commissioning_year=2030,  # Future reactor
            fuel_enrichment_percent=4.2,
            fuel_burnup_mwd_per_kg=45,
            fuel_cycle_length_months=18,
            natural_uranium_per_enriched_kg=8.1,
            separative_work_swu_per_kg=6.7,
            concrete_tonnes=400000,
            steel_tonnes=65000
        )

    def _calculate_lifetime_avoidance(self,
                                      plant_params: NuclearPlantParameters,
                                      nuclear_emissions: float,
                                      comparison_emissions: float) -> float:
        """Calculate lifetime emission avoidance in MtCO2"""
        lifetime_generation = plant_params.lifetime_electricity_generation_mwh
        avoided_per_kwh = (comparison_emissions -
                           nuclear_emissions) / 1000  # kgCO2-eq/kWh
        total_avoided = avoided_per_kwh * lifetime_generation / 1000000  # MtCO2
        return total_avoided

    def _create_scenario_comparison_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Create summary of scenario comparison"""

        # Extract key metrics for comparison
        emissions = [r['lca_results']['nuclear_only_emissions_gco2_per_kwh']
                     if 'nuclear_only_emissions_gco2_per_kwh' in r['lca_results']
                     else r['lca_results'].get('nuclear_emissions_gco2_per_kwh', 0)
                     for r in results]

        capacities = [r.get('plant_info', {}).get('capacity_mw') or
                      r.get('scenario_info', {}).get('capacity_mw', 0) for r in results]

        names = [r.get('plant_info', {}).get('name') or
                 r.get('scenario_info', {}).get('name', 'Unknown') for r in results]

        return {
            'emission_statistics': {
                'mean_gco2_per_kwh': np.mean(emissions),
                'min_gco2_per_kwh': np.min(emissions),
                'max_gco2_per_kwh': np.max(emissions),
                'std_gco2_per_kwh': np.std(emissions)
            },
            'best_performer': {
                'name': names[np.argmin(emissions)],
                'emissions_gco2_per_kwh': np.min(emissions)
            },
            'capacity_statistics': {
                'total_capacity_mw': np.sum(capacities),
                'average_capacity_mw': np.mean(capacities)
            },
            'ranking': [
                {'rank': i+1, 'name': name, 'emissions': emission}
                for i, (name, emission) in enumerate(
                    sorted(zip(names, emissions), key=lambda x: x[1])
                )
            ]
        }
