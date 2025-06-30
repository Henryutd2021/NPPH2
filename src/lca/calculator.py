"""
Nuclear Power Plant LCA Calculator
Main calculation engine for life cycle assessment

This module contains the core calculation logic for nuclear power plant LCA,
including standalone nuclear plants and nuclear-hydrogen integrated systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
from pathlib import Path

from .config import config, EmissionFactors
from .models import (
    NuclearPlantParameters, LifecycleEmissions, HydrogenProductionData,
    LCAResults, TEAIntegrationData, OptimizationIntegrationData,
    ComprehensiveLCAResults, ReactorType
)

# Set up logging
logger = logging.getLogger(__name__)


class NuclearLCACalculator:
    """Main LCA calculator for nuclear power plants"""

    def __init__(self, use_conservative_estimates: bool = False):
        """
        Initialize LCA calculator

        Args:
            use_conservative_estimates: If True, use conservative (higher) emission estimates
        """
        self.config = config
        self.use_conservative_estimates = use_conservative_estimates

        # Load external data integration paths
        self.tea_results_path = None
        self.optimization_results_path = None

        logger.info("Nuclear LCA Calculator initialized")

    def set_tea_results_path(self, path: Union[str, Path]) -> None:
        """Set path to TEA results for integration"""
        self.tea_results_path = Path(path)
        logger.info(f"TEA results path set to: {self.tea_results_path}")

    def set_optimization_results_path(self, path: Union[str, Path]) -> None:
        """Set path to optimization results for integration"""
        self.optimization_results_path = Path(path)
        logger.info(
            f"Optimization results path set to: {self.optimization_results_path}")

    def calculate_nuclear_only_emissions(self,
                                         plant_params: NuclearPlantParameters) -> LifecycleEmissions:
        """
        Calculate lifecycle emissions for standalone nuclear power plant

        Args:
            plant_params: Nuclear plant parameters

        Returns:
            LifecycleEmissions object with detailed breakdown
        """
        logger.info(
            f"Calculating nuclear-only emissions for {plant_params.plant_name}")

        # Get base emission factors from configuration
        base_emissions = self.config.lifecycle_breakdown

        # Apply reactor-specific adjustments
        emission_multipliers = self._get_reactor_specific_multipliers(
            plant_params.reactor_type)

        # Apply plant-specific adjustments
        capacity_adjustment = self._get_capacity_adjustment_factor(
            plant_params)
        fuel_adjustment = self._get_fuel_cycle_adjustment_factor(plant_params)

        # Calculate adjusted emissions for each stage
        emissions = LifecycleEmissions(
            # Front-end fuel cycle
            uranium_mining_milling=self._adjust_emission(
                base_emissions["uranium_mining_milling"].value,
                emission_multipliers.get("mining", 1.0) * fuel_adjustment
            ),
            uranium_conversion=self._adjust_emission(
                base_emissions["uranium_conversion"].value,
                emission_multipliers.get("conversion", 1.0) * fuel_adjustment
            ),
            uranium_enrichment=self._adjust_emission(
                base_emissions["uranium_enrichment"].value,
                emission_multipliers.get("enrichment", 1.0) * fuel_adjustment
            ),
            fuel_fabrication=self._adjust_emission(
                base_emissions["fuel_fabrication"].value,
                emission_multipliers.get("fabrication", 1.0) * fuel_adjustment
            ),

            # Plant lifecycle
            plant_construction=self._adjust_emission(
                base_emissions["plant_construction"].value,
                emission_multipliers.get(
                    "construction", 1.0) * capacity_adjustment
            ),
            plant_operation=self._adjust_emission(
                base_emissions["plant_operation"].value,
                emission_multipliers.get("operation", 1.0)
            ),

            # Back-end fuel cycle
            waste_management=self._adjust_emission(
                base_emissions["waste_management"].value,
                emission_multipliers.get("waste", 1.0) * fuel_adjustment
            ),
            decommissioning=self._adjust_emission(
                base_emissions["decommissioning"].value,
                emission_multipliers.get(
                    "decommissioning", 1.0) * capacity_adjustment
            )
        )

        logger.info(
            f"Nuclear-only total emissions: {emissions.total_nuclear_only:.2f} gCO2-eq/kWh")
        return emissions

    def calculate_hydrogen_production_emissions(self,
                                                plant_params: NuclearPlantParameters,
                                                hydrogen_data: HydrogenProductionData) -> float:
        """
        Calculate emissions from hydrogen production via nuclear electrolysis

        Args:
            plant_params: Nuclear plant parameters
            hydrogen_data: Hydrogen production data

        Returns:
            Hydrogen production emissions in kgCO2-eq/kg H2
        """
        logger.info("Calculating hydrogen production emissions")

        # Base nuclear electricity emissions (gCO2-eq/kWh)
        nuclear_emissions = self.calculate_nuclear_only_emissions(plant_params)
        nuclear_intensity = nuclear_emissions.total_nuclear_only

        # Electricity consumption for hydrogen production (kWh/kg H2)
        electricity_per_kg_h2 = hydrogen_data.electricity_consumption_mwh_per_kg * 1000

        # Direct emissions from nuclear electricity use (gCO2-eq/kg H2)
        electricity_emissions = nuclear_intensity * electricity_per_kg_h2

        # Electrolyzer construction emissions (amortized, gCO2-eq/kg H2)
        electrolyzer_construction_emissions = self._calculate_electrolyzer_construction_emissions(
            hydrogen_data
        )

        # Electrolyzer operation emissions (maintenance, cooling, etc.)
        electrolyzer_operation_emissions = 50  # gCO2-eq/kg H2 (typical value)

        # Total hydrogen production emissions (convert to kgCO2-eq/kg H2)
        total_emissions = (electricity_emissions + electrolyzer_construction_emissions +
                           electrolyzer_operation_emissions) / 1000

        logger.info(
            f"Hydrogen production emissions: {total_emissions:.3f} kgCO2-eq/kg H2")
        return total_emissions

    def calculate_integrated_system_emissions(self,
                                              plant_params: NuclearPlantParameters,
                                              hydrogen_data: HydrogenProductionData,
                                              allocation_method: str = "energy") -> LifecycleEmissions:
        """
        Calculate emissions for nuclear-hydrogen integrated system

        Args:
            plant_params: Nuclear plant parameters
            hydrogen_data: Hydrogen production data
            allocation_method: Allocation method ("energy", "economic", "mass")

        Returns:
            LifecycleEmissions for integrated system (per kWh electricity)
        """
        logger.info("Calculating integrated system emissions")

        # Calculate nuclear-only emissions
        nuclear_emissions = self.calculate_nuclear_only_emissions(plant_params)

        # Calculate allocation factors
        electricity_allocation, hydrogen_allocation = self._calculate_allocation_factors(
            plant_params, hydrogen_data, allocation_method
        )

        # Calculate electrolyzer lifecycle emissions (amortized to electricity output)
        electrolyzer_emissions = self._calculate_electrolyzer_lifecycle_emissions(
            plant_params, hydrogen_data, electricity_allocation
        )

        # Create integrated system emissions
        integrated_emissions = LifecycleEmissions(
            # Nuclear plant emissions (allocated to electricity)
            uranium_mining_milling=nuclear_emissions.uranium_mining_milling * electricity_allocation,
            uranium_conversion=nuclear_emissions.uranium_conversion * electricity_allocation,
            uranium_enrichment=nuclear_emissions.uranium_enrichment * electricity_allocation,
            fuel_fabrication=nuclear_emissions.fuel_fabrication * electricity_allocation,
            plant_construction=nuclear_emissions.plant_construction * electricity_allocation,
            plant_operation=nuclear_emissions.plant_operation * electricity_allocation,
            waste_management=nuclear_emissions.waste_management * electricity_allocation,
            decommissioning=nuclear_emissions.decommissioning * electricity_allocation,

            # Electrolyzer-specific emissions
            electrolyzer_construction=electrolyzer_emissions["construction"],
            electrolyzer_operation=electrolyzer_emissions["operation"],
            hydrogen_storage_transport=electrolyzer_emissions["storage_transport"]
        )

        logger.info(
            f"Integrated system emissions: {integrated_emissions.total_integrated_system:.2f} gCO2-eq/kWh")
        return integrated_emissions

    def calculate_avoided_emissions(self,
                                    nuclear_emissions: float,
                                    comparison_technology: str = "natural_gas_ccgt") -> Dict[str, float]:
        """
        Calculate avoided emissions compared to alternative technologies

        Args:
            nuclear_emissions: Nuclear plant emissions (gCO2-eq/kWh)
            comparison_technology: Technology to compare against

        Returns:
            Dictionary with avoided emissions data
        """
        comparison_emissions = self.config.comparison_technologies[comparison_technology].value

        avoided_per_mwh = (comparison_emissions -
                           nuclear_emissions) / 1000  # kgCO2-eq/MWh
        avoided_percent = (
            (comparison_emissions - nuclear_emissions) / comparison_emissions) * 100

        return {
            "comparison_technology": comparison_technology,
            "comparison_emissions_gco2_per_kwh": comparison_emissions,
            "nuclear_emissions_gco2_per_kwh": nuclear_emissions,
            "avoided_emissions_kg_co2_per_mwh": avoided_per_mwh,
            "avoided_emissions_percent": avoided_percent,
            "emission_reduction_factor": comparison_emissions / nuclear_emissions
        }

    def calculate_carbon_abatement_cost(self,
                                        nuclear_lcoe: float,
                                        alternative_lcoe: float,
                                        nuclear_emissions: float,
                                        alternative_emissions: float) -> float:
        """
        Calculate carbon abatement cost

        Args:
            nuclear_lcoe: Nuclear LCOE (USD/MWh)
            alternative_lcoe: Alternative technology LCOE (USD/MWh)
            nuclear_emissions: Nuclear emissions (gCO2-eq/kWh)
            alternative_emissions: Alternative emissions (gCO2-eq/kWh)

        Returns:
            Carbon abatement cost (USD/tonne CO2)
        """
        cost_difference = nuclear_lcoe - alternative_lcoe  # USD/MWh
        emission_difference = (alternative_emissions -
                               nuclear_emissions) / 1000  # kgCO2-eq/kWh

        if emission_difference <= 0:
            return 0.0  # No emission reduction

        # Convert to USD/tonne CO2
        abatement_cost = (cost_difference * 1000) / \
            emission_difference  # USD/tonne CO2

        return abatement_cost

    def perform_uncertainty_analysis(self,
                                     plant_params: NuclearPlantParameters,
                                     n_iterations: int = 1000) -> Dict[str, float]:
        """
        Perform Monte Carlo uncertainty analysis

        Args:
            plant_params: Nuclear plant parameters
            n_iterations: Number of Monte Carlo iterations

        Returns:
            Dictionary with uncertainty statistics
        """
        logger.info(
            f"Performing uncertainty analysis with {n_iterations} iterations")

        results = []

        for i in range(n_iterations):
            # Sample uncertain parameters
            sampled_params = self._sample_uncertain_parameters()

            # Calculate emissions with sampled parameters
            emissions = self._calculate_emissions_with_uncertainty(
                plant_params, sampled_params)
            results.append(emissions.total_nuclear_only)

        results = np.array(results)

        uncertainty_stats = {
            "mean": np.mean(results),
            "median": np.median(results),
            "std": np.std(results),
            "min": np.min(results),
            "max": np.max(results),
            "percentile_5": np.percentile(results, 5),
            "percentile_95": np.percentile(results, 95),
            "confidence_interval_90": (np.percentile(results, 5), np.percentile(results, 95))
        }

        logger.info(
            f"Uncertainty analysis complete. Mean: {uncertainty_stats['mean']:.2f} ± {uncertainty_stats['std']:.2f} gCO2-eq/kWh")
        return uncertainty_stats

    def create_comprehensive_analysis(self,
                                      plant_params: NuclearPlantParameters,
                                      hydrogen_data: Optional[HydrogenProductionData] = None,
                                      monte_carlo_runs: int = 0) -> ComprehensiveLCAResults:
        """
        Create comprehensive LCA analysis with all integrations

        Args:
            plant_params: Nuclear plant parameters
            hydrogen_data: Hydrogen production data (if applicable)
            monte_carlo_runs: Number of runs for uncertainty analysis. If 0, skipped.

        Returns:
            ComprehensiveLCAResults object
        """
        logger.info(
            f"Creating comprehensive LCA analysis for {plant_params.plant_name}")

        # Calculate nuclear-only emissions
        nuclear_emissions = self.calculate_nuclear_only_emissions(plant_params)

        # Calculate integrated system emissions if hydrogen is included
        integrated_emissions = None
        hydrogen_production_emissions = None
        if hydrogen_data:
            integrated_emissions = self.calculate_integrated_system_emissions(
                plant_params, hydrogen_data
            )
            hydrogen_production_emissions = self.calculate_hydrogen_production_emissions(
                plant_params, hydrogen_data
            )

        # Create base LCA results
        lca_results = LCAResults(
            analysis_id="",  # Will be auto-generated
            timestamp=datetime.now(),
            plant_parameters=plant_params,
            nuclear_only_emissions=nuclear_emissions,
            integrated_system_emissions=integrated_emissions,
            hydrogen_data=hydrogen_data,
            hydrogen_emissions_kg_co2_per_kg=hydrogen_production_emissions
        )

        # Perform uncertainty analysis if requested
        uncertainty_stats = None
        if monte_carlo_runs > 0:
            uncertainty_stats = self.perform_uncertainty_analysis(
                plant_params, n_iterations=monte_carlo_runs
            )

        # Calculate comparative metrics
        coal_avoided = self.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, "coal_pc"
        )
        gas_avoided = self.calculate_avoided_emissions(
            nuclear_emissions.total_nuclear_only, "natural_gas_ccgt"
        )

        lca_results.avoided_emissions_vs_coal_kg_co2_per_mwh = coal_avoided[
            "avoided_emissions_kg_co2_per_mwh"]
        lca_results.avoided_emissions_vs_gas_kg_co2_per_mwh = gas_avoided[
            "avoided_emissions_kg_co2_per_mwh"]

        # Integrate TEA data if requested
        tea_data = None
        if self.tea_results_path:
            tea_data = self._load_tea_integration_data(plant_params.plant_name)
            if tea_data:
                lca_results.carbon_abatement_cost_usd_per_tonne_co2 = self.calculate_carbon_abatement_cost(
                    tea_data.lcoe_usd_per_mwh, 50.0,  # Assume $50/MWh for coal
                    nuclear_emissions.total_nuclear_only, 820.0  # Coal emissions
                )

        # Integrate optimization data if requested
        optimization_data = None
        if self.optimization_results_path:
            optimization_data = self._load_optimization_integration_data(
                plant_params.plant_name)

        # Create comprehensive results
        comprehensive_results = ComprehensiveLCAResults(
            lca_results=lca_results,
            uncertainty_stats=uncertainty_stats,
            tea_data=tea_data,
            optimization_data=optimization_data
        )

        logger.info("Comprehensive LCA analysis complete")
        return comprehensive_results

    # Private helper methods

    def _get_reactor_specific_multipliers(self, reactor_type: ReactorType) -> Dict[str, float]:
        """Get reactor-specific emission multipliers"""
        multipliers = {
            ReactorType.PWR: {"construction": 1.0, "fuel": 1.0},
            ReactorType.BWR: {"construction": 1.05, "fuel": 1.02},
            # Uses natural uranium
            ReactorType.CANDU: {"construction": 1.1, "fuel": 0.8},
            # More efficient
            ReactorType.AP1000: {"construction": 0.95, "fuel": 0.98},
            # Larger but more efficient
            ReactorType.EPR: {"construction": 1.1, "fuel": 0.96},
            # Higher per-unit construction
            ReactorType.SMR: {"construction": 1.2, "fuel": 1.1},
        }

        base_multiplier = multipliers.get(
            reactor_type, {"construction": 1.0, "fuel": 1.0})

        return {
            "mining": base_multiplier["fuel"],
            "conversion": base_multiplier["fuel"],
            "enrichment": base_multiplier["fuel"],
            "fabrication": base_multiplier["fuel"],
            "construction": base_multiplier["construction"],
            "operation": 1.0,
            "waste": base_multiplier["fuel"],
            "decommissioning": base_multiplier["construction"]
        }

    def _get_capacity_adjustment_factor(self, plant_params: NuclearPlantParameters) -> float:
        """Calculate capacity-based adjustment factor"""
        # Economies of scale: larger plants have lower per-kWh construction emissions
        reference_capacity = 1000  # MW
        capacity_factor = (reference_capacity /
                           plant_params.electric_power_mw) ** 0.1
        return min(max(capacity_factor, 0.8), 1.3)  # Limit adjustment to ±30%

    def _get_fuel_cycle_adjustment_factor(self, plant_params: NuclearPlantParameters) -> float:
        """Calculate fuel cycle adjustment factor"""
        # Higher enrichment or longer cycles may affect fuel cycle emissions
        reference_enrichment = 4.2  # %
        reference_burnup = 45  # MWd/kg

        enrichment_factor = plant_params.fuel_enrichment_percent / reference_enrichment
        burnup_factor = reference_burnup / plant_params.fuel_burnup_mwd_per_kg

        return (enrichment_factor * burnup_factor) ** 0.5

    def _adjust_emission(self, base_emission: float, adjustment_factor: float) -> float:
        """Apply adjustment factor with conservative bias if requested"""
        adjusted = base_emission * adjustment_factor

        if self.use_conservative_estimates:
            adjusted *= 1.2  # 20% conservative bias

        return adjusted

    def _calculate_allocation_factors(self,
                                      plant_params: NuclearPlantParameters,
                                      hydrogen_data: HydrogenProductionData,
                                      allocation_method: str) -> Tuple[float, float]:
        """Calculate allocation factors between electricity and hydrogen"""

        if allocation_method == "energy":
            # Energy-based allocation
            electricity_energy = plant_params.annual_electricity_generation_mwh
            hydrogen_energy = hydrogen_data.annual_hydrogen_production_kg * \
                0.0393  # MWh (HHV)
            total_energy = electricity_energy + hydrogen_energy

            electricity_allocation = electricity_energy / total_energy
            hydrogen_allocation = hydrogen_energy / total_energy

        elif allocation_method == "economic":
            # Economic value-based allocation
            electricity_value = (plant_params.annual_electricity_generation_mwh *
                                 hydrogen_data.hydrogen_selling_price_per_kg * 10)  # Simplified
            hydrogen_value = (hydrogen_data.annual_hydrogen_production_kg *
                              hydrogen_data.hydrogen_selling_price_per_kg)
            total_value = electricity_value + hydrogen_value

            electricity_allocation = electricity_value / total_value
            hydrogen_allocation = hydrogen_value / total_value

        else:  # Default to energy allocation
            return self._calculate_allocation_factors(plant_params, hydrogen_data, "energy")

        return electricity_allocation, hydrogen_allocation

    def _calculate_electrolyzer_lifecycle_emissions(self,
                                                    plant_params: NuclearPlantParameters,
                                                    hydrogen_data: HydrogenProductionData,
                                                    electricity_allocation: float) -> Dict[str, float]:
        """Calculate electrolyzer lifecycle emissions allocated to electricity"""

        # Electrolyzer construction emissions (gCO2-eq/kW)
        electrolyzer_construction_emission_factor = 150  # Typical value
        total_construction_emissions = (hydrogen_data.electrolyzer_capacity_mw * 1000 *
                                        electrolyzer_construction_emission_factor)

        # Amortize over electrolyzer lifetime and electricity production
        electrolyzer_lifetime = 20  # years
        annual_electricity = plant_params.annual_electricity_generation_mwh * \
            electricity_allocation
        lifetime_electricity = annual_electricity * electrolyzer_lifetime

        construction_per_kwh = total_construction_emissions / \
            (lifetime_electricity * 1000)

        return {
            "construction": construction_per_kwh,
            "operation": 0.05,  # gCO2-eq/kWh for O&M
            "storage_transport": 0.02  # gCO2-eq/kWh for H2 handling
        }

    def _calculate_electrolyzer_construction_emissions(self,
                                                       hydrogen_data: HydrogenProductionData) -> float:
        """Calculate electrolyzer construction emissions per kg H2"""

        # Electrolyzer construction emission factor (gCO2-eq/kW)
        construction_ef = 150  # Based on PEM electrolyzer LCA studies

        # Total construction emissions
        total_emissions = hydrogen_data.electrolyzer_capacity_mw * 1000 * construction_ef

        # Amortize over lifetime hydrogen production
        lifetime_h2_production = hydrogen_data.lifetime_hydrogen_production_kg

        return total_emissions / lifetime_h2_production

    def _sample_uncertain_parameters(self) -> Dict[str, float]:
        """Sample uncertain parameters for Monte Carlo analysis"""
        sampled = {}

        uncertainty_params = self.config.uncertainty_parameters

        for param, config in uncertainty_params.items():
            if param == "monte_carlo_runs" or param == "confidence_interval":
                continue

            if config["distribution"] == "normal":
                sampled[param] = np.random.normal(
                    config["mean"], config["std"])
            elif config["distribution"] == "lognormal":
                sampled[param] = np.random.lognormal(0, config["cv"])
            elif config["distribution"] == "triangular":
                sampled[param] = np.random.triangular(
                    config["min"], config["mode"], config["max"])
            elif config["distribution"] == "discrete":
                sampled[param] = np.random.choice(config["options"])

        return sampled

    def _calculate_emissions_with_uncertainty(self,
                                              plant_params: NuclearPlantParameters,
                                              sampled_params: Dict[str, float]) -> LifecycleEmissions:
        """Calculate emissions with uncertain parameters"""
        # Create modified plant parameters based on sampled values
        modified_params = plant_params

        if "capacity_factor" in sampled_params:
            modified_params.capacity_factor = max(
                0.5, min(0.98, sampled_params["capacity_factor"]))

        if "plant_lifetime" in sampled_params:
            modified_params.plant_lifetime_years = int(
                sampled_params["plant_lifetime"])

        # Calculate emissions with modifications
        return self.calculate_nuclear_only_emissions(modified_params)

    def _load_tea_integration_data(self, plant_name: str) -> Optional[TEAIntegrationData]:
        """Load TEA results for integration"""
        try:
            # This would load from actual TEA results files
            # Placeholder implementation
            logger.info(f"Loading TEA data for {plant_name}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load TEA data: {e}")
            return None

    def _load_optimization_integration_data(self, plant_name: str) -> Optional[OptimizationIntegrationData]:
        """Load optimization results for integration"""
        try:
            # This would load from actual optimization results files
            # Placeholder implementation
            logger.info(f"Loading optimization data for {plant_name}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load optimization data: {e}")
            return None

# Convenience functions for common calculations


def calculate_simple_nuclear_lca(plant_name: str,
                                 electric_power_mw: float,
                                 capacity_factor: float = 0.9,
                                 reactor_type: ReactorType = ReactorType.PWR) -> float:
    """
    Simple nuclear LCA calculation with minimal parameters

    Args:
        plant_name: Name of the nuclear plant
        electric_power_mw: Electric power capacity (MW)
        capacity_factor: Plant capacity factor
        reactor_type: Type of nuclear reactor

    Returns:
        Nuclear lifecycle emissions (gCO2-eq/kWh)
    """
    from .models import create_sample_plant_parameters

    # Create basic plant parameters
    plant_params = create_sample_plant_parameters()
    plant_params.plant_name = plant_name
    plant_params.electric_power_mw = electric_power_mw
    plant_params.capacity_factor = capacity_factor
    plant_params.reactor_type = reactor_type

    # Calculate emissions
    calculator = NuclearLCACalculator()
    emissions = calculator.calculate_nuclear_only_emissions(plant_params)

    return emissions.total_nuclear_only
