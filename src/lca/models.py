"""
Nuclear Power Plant LCA Data Models
Data models and classes for life cycle assessment calculations

This module defines the data structures used throughout the LCA analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum


class ReactorType(Enum):
    """Nuclear reactor types supported in LCA analysis"""
    PWR = "Pressurized Water Reactor"
    BWR = "Boiling Water Reactor"
    CANDU = "Canada Deuterium Uranium"
    AGR = "Advanced Gas-Cooled Reactor"
    VVER = "Vodo-Vodyanoi Energetichesky Reactor"
    AP1000 = "AP1000 Advanced PWR"
    EPR = "European Pressurized Reactor"
    SMR = "Small Modular Reactor"


class LCAPhase(Enum):
    """LCA phases defined according to ISO 14040/14044"""
    GOAL_SCOPE = "Goal and Scope Definition"
    INVENTORY = "Life Cycle Inventory"
    IMPACT_ASSESSMENT = "Life Cycle Impact Assessment"
    INTERPRETATION = "Life Cycle Interpretation"


class SystemBoundary(Enum):
    """System boundary definitions"""
    CRADLE_TO_GATE = "Cradle-to-Gate"
    CRADLE_TO_GRAVE = "Cradle-to-Grave"
    GATE_TO_GATE = "Gate-to-Gate"
    GATE_TO_GRAVE = "Gate-to-Grave"


@dataclass
class NuclearPlantParameters:
    """Nuclear power plant technical and operational parameters"""

    # Basic plant identification
    plant_name: str
    reactor_type: ReactorType
    location: str
    iso_region: str

    # Technical specifications
    thermal_power_mw: float  # Thermal power (MWth)
    electric_power_mw: float  # Net electric power (MWe)
    thermal_efficiency: float  # Thermal efficiency (0-1)
    capacity_factor: float  # Capacity factor (0-1)

    # Operational parameters
    plant_lifetime_years: int  # Design lifetime
    construction_time_years: int  # Construction period
    commissioning_year: int  # Year of commercial operation

    # Fuel cycle parameters
    fuel_enrichment_percent: float  # U-235 enrichment (%)
    fuel_burnup_mwd_per_kg: float  # Fuel burnup (MWd/kg)
    fuel_cycle_length_months: int  # Refueling cycle
    natural_uranium_per_enriched_kg: float  # Natural uranium requirement
    separative_work_swu_per_kg: float  # Separative work requirement

    # Construction materials
    concrete_tonnes: float  # Concrete usage
    steel_tonnes: float  # Steel usage

    # Optional hydrogen production parameters
    hydrogen_production_enabled: bool = False
    electrolyzer_capacity_mw: Optional[float] = None
    electrolyzer_efficiency: Optional[float] = None
    # Fraction of electricity for hydrogen
    hydrogen_allocation_factor: Optional[float] = None

    def __post_init__(self):
        """Validate parameters after initialization"""
        if self.thermal_efficiency <= 0 or self.thermal_efficiency > 1:
            raise ValueError("Thermal efficiency must be between 0 and 1")
        if self.capacity_factor <= 0 or self.capacity_factor > 1:
            raise ValueError("Capacity factor must be between 0 and 1")
        if self.hydrogen_production_enabled and self.electrolyzer_capacity_mw is None:
            raise ValueError(
                "Electrolyzer capacity required when hydrogen production is enabled")

    @property
    def annual_electricity_generation_mwh(self) -> float:
        """Calculate annual electricity generation"""
        hours_per_year = 8760
        return self.electric_power_mw * self.capacity_factor * hours_per_year

    @property
    def lifetime_electricity_generation_mwh(self) -> float:
        """Calculate lifetime electricity generation"""
        return self.annual_electricity_generation_mwh * self.plant_lifetime_years


@dataclass
class LifecycleEmissions:
    """Lifecycle emission data for different stages"""

    # Front-end fuel cycle (gCO2-eq/kWh)
    uranium_mining_milling: float
    uranium_conversion: float
    uranium_enrichment: float
    fuel_fabrication: float

    # Plant lifecycle (gCO2-eq/kWh)
    plant_construction: float
    plant_operation: float

    # Back-end fuel cycle (gCO2-eq/kWh)
    waste_management: float
    decommissioning: float

    # Additional hydrogen-specific emissions (if applicable)
    electrolyzer_construction: float = 0.0
    electrolyzer_operation: float = 0.0
    hydrogen_storage_transport: float = 0.0

    @property
    def total_front_end(self) -> float:
        """Total front-end fuel cycle emissions"""
        return (self.uranium_mining_milling + self.uranium_conversion +
                self.uranium_enrichment + self.fuel_fabrication)

    @property
    def total_plant(self) -> float:
        """Total plant-related emissions"""
        return self.plant_construction + self.plant_operation

    @property
    def total_back_end(self) -> float:
        """Total back-end fuel cycle emissions"""
        return self.waste_management + self.decommissioning

    @property
    def total_nuclear_only(self) -> float:
        """Total nuclear power emissions (excluding hydrogen)"""
        return self.total_front_end + self.total_plant + self.total_back_end

    @property
    def total_hydrogen_related(self) -> float:
        """Total hydrogen-related emissions"""
        return (self.electrolyzer_construction + self.electrolyzer_operation +
                self.hydrogen_storage_transport)

    @property
    def total_integrated_system(self) -> float:
        """Total integrated system emissions (nuclear + hydrogen)"""
        return self.total_nuclear_only + self.total_hydrogen_related


@dataclass
class HydrogenProductionData:
    """Hydrogen production data and parameters"""

    # Production parameters
    annual_hydrogen_production_kg: float  # Annual hydrogen production
    electrolyzer_capacity_mw: float  # Electrolyzer capacity
    electrolyzer_efficiency: float  # Electrolyzer efficiency (HHV basis)
    electricity_consumption_mwh_per_kg: float  # Electricity consumption per kg H2

    # Economic parameters
    electrolyzer_capex_per_kw: float  # Electrolyzer capital cost
    electrolyzer_opex_percent: float  # O&M cost as % of CAPEX
    hydrogen_selling_price_per_kg: float  # Hydrogen selling price

    # Emission parameters
    hydrogen_emission_factor_kg_co2_per_kg: float  # Direct H2 production emissions
    avoided_emissions_kg_co2_per_kg: float  # Avoided emissions vs conventional H2

    @property
    def lifetime_hydrogen_production_kg(self) -> float:
        """Calculate lifetime hydrogen production"""
        return self.annual_hydrogen_production_kg * 20  # Assume 20-year electrolyzer lifetime


@dataclass
class LCAResults:
    """Complete LCA results for nuclear power plant analysis"""

    # Basic information
    analysis_id: str
    timestamp: datetime
    plant_parameters: NuclearPlantParameters

    # Emission results (gCO2-eq/kWh electricity)
    nuclear_only_emissions: LifecycleEmissions
    integrated_system_emissions: Optional[LifecycleEmissions] = None

    # Hydrogen-specific results (if applicable)
    hydrogen_data: Optional[HydrogenProductionData] = None
    hydrogen_emissions_kg_co2_per_kg: Optional[float] = None

    # Comparative analysis
    avoided_emissions_vs_coal_kg_co2_per_mwh: Optional[float] = None
    avoided_emissions_vs_gas_kg_co2_per_mwh: Optional[float] = None
    carbon_intensity_reduction_percent: Optional[float] = None

    # Economic metrics
    carbon_abatement_cost_usd_per_tonne_co2: Optional[float] = None

    # Uncertainty analysis
    uncertainty_range_low_gco2_per_kwh: Optional[float] = None
    uncertainty_range_high_gco2_per_kwh: Optional[float] = None
    confidence_interval: Optional[float] = None

    # Additional metrics
    land_use_m2_per_mwh: Optional[float] = None
    water_consumption_m3_per_mwh: Optional[float] = None

    def __post_init__(self):
        """Generate analysis ID if not provided"""
        if not self.analysis_id:
            self.analysis_id = f"LCA_{self.plant_parameters.plant_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

    @property
    def nuclear_carbon_intensity(self) -> float:
        """Nuclear-only carbon intensity"""
        return self.nuclear_only_emissions.total_nuclear_only

    @property
    def integrated_carbon_intensity(self) -> Optional[float]:
        """Integrated system carbon intensity (if applicable)"""
        if self.integrated_system_emissions:
            return self.integrated_system_emissions.total_integrated_system
        return None

    @property
    def carbon_intensity_reduction_absolute(self) -> Optional[float]:
        """Absolute carbon intensity reduction (gCO2-eq/kWh)"""
        if self.integrated_system_emissions:
            return (self.nuclear_only_emissions.total_nuclear_only -
                    self.integrated_system_emissions.total_integrated_system)
        return None


@dataclass
class TEAIntegrationData:
    """Data structure for integrating TEA results with LCA"""

    # --- Fields without default values ---
    # TEA results
    lcoe_usd_per_mwh: float  # Levelized cost of electricity

    # Revenue data
    electricity_revenue_usd_per_mwh: float
    ancillary_services_revenue_usd_per_mwh: float

    # Cost breakdown
    capex_usd: float
    opex_annual_usd: float
    fuel_cost_annual_usd: float

    # Market participation data
    capacity_factor_electricity: float
    ancillary_services_participation_hours: float

    # --- Fields with default values (Optional) ---
    lcoh_usd_per_kg: Optional[float] = None  # Levelized cost of hydrogen
    hydrogen_revenue_usd_per_kg: Optional[float] = None
    capacity_factor_hydrogen: Optional[float] = None

    @property
    def total_annual_revenue_usd(self) -> float:
        """Calculate total annual revenue"""
        electricity_revenue = (self.electricity_revenue_usd_per_mwh +
                               self.ancillary_services_revenue_usd_per_mwh) * 8760 * self.capacity_factor_electricity

        hydrogen_revenue = 0.0
        if self.hydrogen_revenue_usd_per_kg and self.capacity_factor_hydrogen:
            # Simplified calculation - would need actual hydrogen production data
            hydrogen_revenue = self.hydrogen_revenue_usd_per_kg * 1000  # Placeholder

        return electricity_revenue + hydrogen_revenue


@dataclass
class OptimizationIntegrationData:
    """Data structure for integrating optimization results with LCA"""

    # --- Fields without default values ---
    # Optimization strategy
    optimization_scenario: str  # Description of optimization scenario

    # Operational profile
    electricity_generation_profile_mwh: np.ndarray  # Hourly electricity generation

    # Market participation
    energy_market_participation_hours: int
    regulation_market_participation_hours: int
    spinning_reserve_participation_hours: int

    # Flexibility utilization
    load_following_capability_mw: float
    ramping_rate_mw_per_min: float
    minimum_stable_output_percent: float

    # Grid services
    frequency_regulation_service_mw: float
    voltage_support_service_mvar: float

    # Economic performance
    optimized_annual_revenue_usd: float
    revenue_improvement_percent: float

    # --- Fields with default values (Optional) ---
    # Hourly hydrogen production
    hydrogen_production_profile_kg: Optional[np.ndarray] = None

    @property
    def total_market_participation_hours(self) -> int:
        """Total hours of market participation"""
        return (self.energy_market_participation_hours +
                self.regulation_market_participation_hours +
                self.spinning_reserve_participation_hours)

    @property
    def flexibility_utilization_factor(self) -> float:
        """Calculate flexibility utilization factor"""
        total_hours = 8760
        return self.total_market_participation_hours / total_hours


@dataclass
class ComprehensiveLCAResults:
    """Comprehensive LCA results integrating TEA and optimization data"""

    # Core LCA results
    lca_results: LCAResults

    # Uncertainty analysis results
    uncertainty_stats: Optional[Dict[str, float]] = None

    # Integrated data
    tea_data: Optional[TEAIntegrationData] = None
    optimization_data: Optional[OptimizationIntegrationData] = None

    # Comparative analysis against baseline
    baseline_coal_emissions_gco2_per_kwh: float = 820.0
    baseline_gas_emissions_gco2_per_kwh: float = 490.0

    # Regional grid emission factors
    regional_grid_emission_factor_gco2_per_kwh: Optional[float] = None

    # Policy and incentive impacts
    carbon_price_usd_per_tonne: Optional[float] = None
    renewable_energy_credits_value_usd_per_mwh: Optional[float] = None

    @property
    def emission_reduction_vs_coal_percent(self) -> float:
        """Emission reduction percentage vs coal baseline"""
        nuclear_emissions = self.lca_results.nuclear_carbon_intensity
        return ((self.baseline_coal_emissions_gco2_per_kwh - nuclear_emissions) /
                self.baseline_coal_emissions_gco2_per_kwh) * 100

    @property
    def emission_reduction_vs_gas_percent(self) -> float:
        """Emission reduction percentage vs natural gas baseline"""
        nuclear_emissions = self.lca_results.nuclear_carbon_intensity
        return ((self.baseline_gas_emissions_gco2_per_kwh - nuclear_emissions) /
                self.baseline_gas_emissions_gco2_per_kwh) * 100

    @property
    def carbon_abatement_cost_vs_coal(self) -> Optional[float]:
        """Carbon abatement cost vs coal (USD/tonne CO2)"""
        if not self.tea_data:
            return None

        emission_reduction = (self.baseline_coal_emissions_gco2_per_kwh -
                              self.lca_results.nuclear_carbon_intensity) / 1000  # Convert to kg

        cost_difference = self.tea_data.lcoe_usd_per_mwh - 50  # Assume $50/MWh for coal

        if emission_reduction > 0:
            return cost_difference / emission_reduction
        return None

    def to_dict(self) -> Dict:
        """Convert results to dictionary for export"""
        result_dict = {
            'analysis_id': self.lca_results.analysis_id,
            'timestamp': self.lca_results.timestamp.isoformat(),
            'plant_name': self.lca_results.plant_parameters.plant_name,
            'reactor_type': self.lca_results.plant_parameters.reactor_type.value,
            'nuclear_carbon_intensity_gco2_per_kwh': self.lca_results.nuclear_carbon_intensity,
            'emission_reduction_vs_coal_percent': self.emission_reduction_vs_coal_percent,
            'emission_reduction_vs_gas_percent': self.emission_reduction_vs_gas_percent,
        }

        if self.tea_data:
            result_dict.update({
                'lcoe_usd_per_mwh': self.tea_data.lcoe_usd_per_mwh,
                'carbon_abatement_cost_vs_coal': self.carbon_abatement_cost_vs_coal,
            })

        if self.optimization_data:
            result_dict.update({
                'optimization_scenario': self.optimization_data.optimization_scenario,
                'flexibility_utilization_factor': self.optimization_data.flexibility_utilization_factor,
                'revenue_improvement_percent': self.optimization_data.revenue_improvement_percent,
            })

        return result_dict

# Utility functions for data validation and conversion


def validate_plant_parameters(params: NuclearPlantParameters) -> List[str]:
    """Validate nuclear plant parameters and return list of issues"""
    issues = []

    if params.thermal_efficiency <= 0.25 or params.thermal_efficiency > 0.40:
        issues.append("Thermal efficiency outside realistic range (25-40%)")

    if params.capacity_factor < 0.5 or params.capacity_factor > 0.98:
        issues.append("Capacity factor outside realistic range (50-98%)")

    if params.fuel_enrichment_percent < 2.0 or params.fuel_enrichment_percent > 5.0:
        issues.append("Fuel enrichment outside typical range (2-5%)")

    if params.plant_lifetime_years < 30 or params.plant_lifetime_years > 80:
        issues.append("Plant lifetime outside realistic range (30-80 years)")

    return issues


def create_sample_plant_parameters() -> NuclearPlantParameters:
    """Create sample plant parameters for testing"""
    return NuclearPlantParameters(
        plant_name="Sample Nuclear Plant",
        reactor_type=ReactorType.PWR,
        location="Sample Location",
        iso_region="SAMPLE_ISO",
        thermal_power_mw=3000,
        electric_power_mw=1000,
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
        steel_tonnes=65000,
        hydrogen_production_enabled=False
    )
