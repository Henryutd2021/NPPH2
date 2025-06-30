"""
Nuclear-Hydrogen System LCA Analysis
Comprehensive carbon footprint analysis of nuclear-hydrogen integrated systems

This module provides specialized LCA analysis for nuclear-hydrogen systems,
integrating results from TEA and optimization frameworks to assess:
1. Pure nuclear power baseline carbon footprint
2. Nuclear-hydrogen integrated system carbon footprint
3. Carbon reduction benefits from ancillary services
4. Comparative analysis across different ISOs
5. System flexibility carbon benefits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime

from .models import (
    NuclearPlantParameters, LCAResults, LifecycleEmissions,
    HydrogenProductionData, ReactorType
)
from .calculator import NuclearLCACalculator
from .config import NuclearLCAConfig

logger = logging.getLogger(__name__)


@dataclass
class NuclearHydrogenSystemConfig:
    """Configuration for nuclear-hydrogen system LCA analysis"""

    # Emission factors for hydrogen production pathways (gCO2-eq/kg H2)
    conventional_hydrogen_emissions: float = 9500.0  # Steam methane reforming
    # Grid average (varies by region)
    grid_electrolysis_emissions: float = 26000.0

    # Grid emission factors by ISO region (gCO2-eq/kWh)
    iso_grid_emission_factors: Dict[str, float] = field(default_factory=lambda: {
        'PJM': 450.0,     # Coal/gas heavy
        'ERCOT': 400.0,   # Gas heavy, some wind
        'CAISO': 250.0,   # Lower carbon grid
        'NYISO': 300.0,   # Mixed grid
        'ISONE': 280.0,   # Nuclear + gas
        'MISO': 480.0,    # Coal heavy
        'SPP': 520.0      # Coal heavy
    })

    # Ancillary services carbon benefits (gCO2-eq/MWh avoided)
    gas_turbine_emission_factor: float = 450.0      # Fast ramping gas units
    coal_plant_emission_factor: float = 820.0       # Coal units

    # Electrolyzer lifecycle emissions (gCO2-eq/kW installed)
    electrolyzer_manufacturing_emissions: float = 240.0  # Per kW capacity
    electrolyzer_lifetime_years: float = 15.0

    # Battery system emissions (gCO2-eq/kWh installed)
    battery_manufacturing_emissions: float = 150.0  # Li-ion batteries
    battery_lifetime_years: float = 15.0


@dataclass
class HydrogenSystemEmissions:
    """Hydrogen production system emissions breakdown"""

    # Direct emissions (gCO2-eq/kg H2)
    electricity_emissions: float = 0.0         # From electricity consumption
    thermal_energy_emissions: float = 0.0      # From thermal integration (HTE)
    electrolyzer_manufacturing: float = 0.0    # Amortized manufacturing
    water_treatment_emissions: float = 0.0     # Water processing

    # Indirect emissions
    grid_displacement_emissions: float = 0.0   # Grid electricity displaced

    # Total system emissions
    total_hydrogen_emissions: float = 0.0      # gCO2-eq/kg H2

    # Avoided emissions (negative values = benefits)
    avoided_conventional_h2: float = 0.0       # vs steam methane reforming
    avoided_grid_electrolysis: float = 0.0     # vs grid electrolysis


@dataclass
class AncillaryServicesEmissions:
    """Ancillary services carbon footprint analysis"""

    # Service provision (annual values in MWh)
    regulation_service_mwh: float = 0.0
    spinning_reserve_mwh: float = 0.0
    load_following_mwh: float = 0.0

    # Detailed service breakdown
    regulation_up_mwh: float = 0.0
    regulation_down_mwh: float = 0.0
    spinning_reserve_actual_mwh: float = 0.0
    non_spinning_reserve_mwh: float = 0.0
    ramp_up_mwh: float = 0.0
    ramp_down_mwh: float = 0.0
    ecrs_mwh: float = 0.0
    thirty_min_reserve_mwh: float = 0.0

    # Avoided emissions from displaced generation (kg CO2-eq/year)
    avoided_gas_turbine_emissions: float = 0.0
    avoided_coal_emissions: float = 0.0
    avoided_grid_emissions: float = 0.0         # From load following
    total_avoided_emissions: float = 0.0

    # Specific emission rates
    avoided_emissions_per_mwh: float = 0.0      # kg CO2-eq/MWh service


@dataclass
class NuclearHydrogenLCAResults:
    """Comprehensive LCA results for nuclear-hydrogen systems"""

    # System identification
    plant_name: str
    iso_region: str
    analysis_timestamp: datetime

    # System configuration
    nuclear_capacity_mw: float
    electrolyzer_capacity_mw: float = 0.0
    battery_capacity_mwh: float = 0.0
    annual_hydrogen_production_kg: float = 0.0

    # Nuclear baseline emissions (pure nuclear operation)
    nuclear_baseline_emissions: LifecycleEmissions = None

    # Integrated system emissions
    hydrogen_system_emissions: HydrogenSystemEmissions = None
    ancillary_services_emissions: AncillaryServicesEmissions = None

    # Comparative metrics
    nuclear_only_carbon_intensity: float = 0.0     # gCO2-eq/kWh
    integrated_carbon_intensity: float = 0.0       # gCO2-eq/kWh
    carbon_intensity_reduction: float = 0.0        # gCO2-eq/kWh
    carbon_reduction_percentage: float = 0.0       # %

    # Annual impact metrics
    annual_electricity_generation_mwh: float = 0.0
    annual_carbon_footprint_nuclear_kg: float = 0.0
    annual_carbon_footprint_integrated_kg: float = 0.0
    annual_carbon_reduction_kg: float = 0.0

    # Net system impact metrics
    net_annual_carbon_impact_kg: float = 0.0
    net_equivalent_carbon_intensity: float = 0.0

    # Economic-carbon metrics
    carbon_abatement_cost_usd_per_tonne: float = 0.0

    def calculate_summary_metrics(self):
        """Calculate derived summary metrics"""
        if self.nuclear_baseline_emissions:
            self.nuclear_only_carbon_intensity = self.nuclear_baseline_emissions.total_nuclear_only

            # Default integrated intensity to nuclear-only
            self.integrated_carbon_intensity = self.nuclear_only_carbon_intensity

            # Calculate integrated system carbon intensity if hydrogen is produced
            if self.annual_hydrogen_production_kg > 0 and self.hydrogen_system_emissions:
                # Use energy allocation method
                electricity_for_h2_mwh = self.annual_hydrogen_production_kg * 50.0 / 1000  # ~50 kWh/kg

                # Check for valid generation data
                if self.annual_electricity_generation_mwh > 0:

                    total_nuclear_emissions_kg = self.nuclear_only_carbon_intensity * \
                        self.annual_electricity_generation_mwh
                    total_h2_emissions_kg = self.hydrogen_system_emissions.total_hydrogen_emissions * \
                        self.annual_hydrogen_production_kg / 1000

                    # Total emissions from the integrated system
                    total_integrated_emissions_kg = total_nuclear_emissions_kg + total_h2_emissions_kg

                    self.integrated_carbon_intensity = total_integrated_emissions_kg / \
                        self.annual_electricity_generation_mwh

            self.carbon_intensity_reduction = self.nuclear_only_carbon_intensity - \
                self.integrated_carbon_intensity
            if self.nuclear_only_carbon_intensity > 0:
                self.carbon_reduction_percentage = (
                    self.carbon_intensity_reduction / self.nuclear_only_carbon_intensity) * 100

            # Annual footprint calculations
            self.annual_carbon_footprint_nuclear_kg = (
                self.nuclear_only_carbon_intensity * self.annual_electricity_generation_mwh)
            self.annual_carbon_footprint_integrated_kg = (
                self.integrated_carbon_intensity * self.annual_electricity_generation_mwh)
            self.annual_carbon_reduction_kg = self.annual_carbon_footprint_nuclear_kg - \
                self.annual_carbon_footprint_integrated_kg

            # Calculate Net System Carbon Impact
            self._calculate_net_system_impact()

    def _calculate_net_system_impact(self):
        """Calculate the holistic net carbon impact of the integrated system."""

        direct_emissions_kg = self.annual_carbon_footprint_integrated_kg

        # Calculate avoided emissions from displacing conventional hydrogen
        h2_avoided_emissions_kg = 0
        if self.hydrogen_system_emissions and self.annual_hydrogen_production_kg > 0:
            # This represents the emissions that WOULD have happened using SMR
            conventional_h2_emissions_factor = 9500.0  # g/kg H2, from config
            h2_avoided_emissions_kg = (
                self.annual_hydrogen_production_kg * conventional_h2_emissions_factor) / 1000

        # Get avoided emissions from ancillary services
        as_avoided_emissions_kg = 0
        if self.ancillary_services_emissions:
            as_avoided_emissions_kg = self.ancillary_services_emissions.total_avoided_emissions

        total_avoided_emissions_kg = h2_avoided_emissions_kg + as_avoided_emissions_kg

        # Net impact is the direct footprint minus the benefits (avoided emissions)
        self.net_annual_carbon_impact_kg = direct_emissions_kg - total_avoided_emissions_kg

        # Convert net annual impact to an equivalent intensity for comparison
        if self.annual_electricity_generation_mwh > 0:
            self.net_equivalent_carbon_intensity = self.net_annual_carbon_impact_kg / \
                self.annual_electricity_generation_mwh
        else:
            self.net_equivalent_carbon_intensity = 0


class NuclearHydrogenLCAAnalyzer:
    """Advanced LCA analyzer for nuclear-hydrogen integrated systems"""

    def __init__(self,
                 config: Optional[NuclearHydrogenSystemConfig] = None,
                 tea_results_dir: Union[str, Path] = "output/tea",
                 opt_results_dir: Union[str, Path] = "output/opt"):
        """
        Initialize nuclear-hydrogen LCA analyzer

        Args:
            config: System configuration for emissions factors
            tea_results_dir: Directory containing TEA results
            opt_results_dir: Directory containing optimization results
        """
        self.config = config or NuclearHydrogenSystemConfig()
        self.tea_results_dir = Path(tea_results_dir)
        self.opt_results_dir = Path(opt_results_dir)

        # Initialize nuclear LCA calculator
        self.nuclear_calculator = NuclearLCACalculator()
        self.nuclear_config = NuclearLCAConfig()

        logger.info("Nuclear-Hydrogen LCA Analyzer initialized")

    def load_tea_results(self, plant_name: str) -> Optional[Dict]:
        """
        Load TEA results for specified plant

        Args:
            plant_name: Name of nuclear plant

        Returns:
            Dictionary containing TEA results or None
        """
        try:
            logger.info(f"🔍 Looking for TEA results for plant: {plant_name}")
            logger.info(f"📂 TEA results directory: {self.tea_results_dir}")
            logger.info(
                f"✅ TEA directory exists: {self.tea_results_dir.exists()}")

            # Try multiple search patterns
            search_patterns = [
                f"**/*{plant_name}*/*Summary*.txt",
                f"**/cs1/{plant_name}/*Summary*.txt",
                f"cs1/{plant_name}/*Summary*.txt",
                f"**/{plant_name}/*Summary*.txt"
            ]

            tea_files = []
            for pattern in search_patterns:
                logger.info(f"🔍 Trying pattern: {pattern}")
                found_files = list(self.tea_results_dir.glob(pattern))
                logger.info(f"📄 Found {len(found_files)} files with pattern")
                if found_files:
                    tea_files.extend(found_files)
                    break

            # Remove duplicates
            tea_files = list(set(tea_files))

            if not tea_files:
                logger.warning(
                    f"❌ No TEA summary files found for {plant_name}")

                # Debug: show what's actually in the plant directory
                plant_dir = self.tea_results_dir / "cs1" / plant_name
                if plant_dir.exists():
                    logger.info(f"📁 Files in {plant_dir}:")
                    for file in plant_dir.iterdir():
                        logger.info(f"  - {file.name}")
                else:
                    logger.warning(
                        f"❌ Plant directory doesn't exist: {plant_dir}")

                return None

            logger.info(
                f"📄 Found {len(tea_files)} TEA files: {[f.name for f in tea_files]}")

            # Use the comprehensive summary if available
            summary_file = None
            for file in tea_files:
                if "Comprehensive" in file.name:
                    summary_file = file
                    break

            if not summary_file:
                summary_file = tea_files[0]

            logger.info(f"✅ Loading TEA results from: {summary_file}")

            # Parse TEA summary file
            tea_data = self._parse_tea_summary_file(summary_file)
            return tea_data

        except Exception as e:
            logger.error(f"❌ Error loading TEA results for {plant_name}: {e}")
            return None

    def _parse_tea_summary_file(self, file_path: Path) -> Dict:
        """Parse TEA summary text file and extract key metrics"""
        tea_data = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract key metrics using text parsing
            lines = content.split('\n')

            for i, line in enumerate(lines):
                line = line.strip()

                # Extract nuclear capacity
                if "Nuclear Unit Capacity (MW)" in line or "Turbine Capacity" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            ',', '').split()[0]
                        try:
                            tea_data['nuclear_capacity_mw'] = float(value_str)
                        except ValueError:
                            pass

                # Extract electrolyzer capacity
                if "Electrolyzer Capacity" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            ',', '').split()[0]
                        try:
                            tea_data['electrolyzer_capacity_mw'] = float(
                                value_str)
                        except ValueError:
                            pass

                # Extract hydrogen production
                if "Annual H2 Production" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            ',', '').split()[0]
                        try:
                            tea_data['annual_h2_production_kg'] = float(
                                value_str)
                        except ValueError:
                            pass

                # Extract electricity generation
                if "Annual Nuclear Generation" in line or "Annual Generation" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            ',', '').split()[0]
                        try:
                            tea_data['annual_electricity_generation_mwh'] = float(
                                value_str)
                        except ValueError:
                            pass

                # Extract ancillary services data
                if "Ancillary Services Revenue" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            '$', '').replace(',', '').split()[0]
                        try:
                            tea_data['ancillary_services_revenue_usd'] = float(
                                value_str)
                        except ValueError:
                            pass

                # Extract LCOH
                if "LCOH" in line and "USD/kg" in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_str = parts[1].strip().replace(
                            '$', '').split('/')[0]
                        try:
                            tea_data['lcoh_usd_per_kg'] = float(value_str)
                        except ValueError:
                            pass

            return tea_data

        except Exception as e:
            logger.error(f"Error parsing TEA file {file_path}: {e}")
            return {}

    def calculate_hydrogen_system_emissions(self,
                                            tea_data: Dict,
                                            iso_region: str) -> HydrogenSystemEmissions:
        """
        Calculate hydrogen production system emissions

        Args:
            tea_data: TEA analysis results
            iso_region: ISO region for grid factors

        Returns:
            HydrogenSystemEmissions object
        """
        emissions = HydrogenSystemEmissions()

        # Get system parameters
        annual_h2_production = tea_data.get('annual_h2_production_kg', 0)
        electrolyzer_capacity = tea_data.get('electrolyzer_capacity_mw', 0)

        if annual_h2_production == 0:
            return emissions

        # 1. Electricity emissions for hydrogen production
        # Assume ~50 kWh/kg H2 (higher heating value basis)
        electricity_per_kg_h2 = 50.0  # kWh/kg
        nuclear_emission_factor = self.nuclear_config.get_default_nuclear_emissions()  # gCO2-eq/kWh

        emissions.electricity_emissions = electricity_per_kg_h2 * nuclear_emission_factor

        # 2. Electrolyzer manufacturing emissions (amortized)
        if electrolyzer_capacity > 0:
            manufacturing_emissions_total = (electrolyzer_capacity * 1000 *
                                             self.config.electrolyzer_manufacturing_emissions)  # gCO2-eq
            annual_manufacturing_emissions = manufacturing_emissions_total / \
                self.config.electrolyzer_lifetime_years
            emissions.electrolyzer_manufacturing = annual_manufacturing_emissions / \
                annual_h2_production

        # 3. Water treatment emissions (minimal for nuclear plants)
        emissions.water_treatment_emissions = 50.0  # gCO2-eq/kg H2 (estimated)

        # 4. Thermal energy emissions (for HTE mode)
        # If thermal integration reduces electricity needs
        # 10% efficiency improvement with thermal integration
        thermal_efficiency_benefit = 0.10
        thermal_electricity_savings = electricity_per_kg_h2 * thermal_efficiency_benefit
        emissions.thermal_energy_emissions = - \
            thermal_electricity_savings * nuclear_emission_factor

        # 5. Total hydrogen system emissions
        emissions.total_hydrogen_emissions = (emissions.electricity_emissions +
                                              emissions.electrolyzer_manufacturing +
                                              emissions.water_treatment_emissions +
                                              emissions.thermal_energy_emissions)

        # 6. Avoided emissions compared to conventional hydrogen
        emissions.avoided_conventional_h2 = (self.config.conventional_hydrogen_emissions -
                                             emissions.total_hydrogen_emissions)

        # 7. Avoided emissions compared to grid electrolysis
        grid_factor = self.config.iso_grid_emission_factors.get(
            iso_region, 500.0)
        grid_electrolysis_emissions = electricity_per_kg_h2 * \
            grid_factor + emissions.electrolyzer_manufacturing
        emissions.avoided_grid_electrolysis = grid_electrolysis_emissions - \
            emissions.total_hydrogen_emissions

        logger.info(
            f"Hydrogen system emissions calculated: {emissions.total_hydrogen_emissions:.1f} gCO2-eq/kg H2")

        return emissions

    def load_optimization_ancillary_services(self, plant_name: str) -> Optional[Dict]:
        """
        Load actual ancillary services data from optimization results

        Args:
            plant_name: Name of nuclear plant

        Returns:
            Dictionary containing actual deployed ancillary services or None
        """
        try:
            # Search for optimization results file with actual format
            # Format: Arkansas Nuclear One_2_SPP_15_hourly_results.csv
            # Or: enhanced_Plant Name_X_ISO_Y_hourly_results.csv
            opt_dir = self.opt_results_dir / "cs1"
            opt_file = None

            # Search for files that match the plant name pattern
            for file in opt_dir.glob("*_hourly_results.csv"):
                filename = file.stem
                if '_hourly_results' in filename:
                    # Remove '_hourly_results' suffix first
                    base_name = filename.replace('_hourly_results', '')

                    # Handle 'enhanced_' prefix
                    if base_name.startswith('enhanced_'):
                        base_name = base_name[9:]

                    # Extract reactor name (remove ISO info - last 2 parts)
                    parts = base_name.split('_')
                    if len(parts) >= 3:
                        extracted_plant_name = '_'.join(parts[:-2])
                        if extracted_plant_name == plant_name:
                            opt_file = file
                            break

            if not opt_file or not opt_file.exists():
                logger.warning(
                    f"Optimization results file not found for: {plant_name}")
                return None

            logger.info(
                f"Loading optimization ancillary services from: {opt_file}")

            # Read optimization results
            opt_df = pd.read_csv(opt_file)

            # Sum deployed ancillary services across all sources (electrolyzer + battery + turbine)
            # Regulation services (MW deployed per hour -> convert to MWh)
            reg_up_total = (opt_df['RegUp_Electrolyzer_Deployed_MW'].fillna(0) +
                            opt_df['RegUp_Battery_Deployed_MW'].fillna(0) +
                            opt_df['RegUp_Turbine_Deployed_MW'].fillna(0)).sum()

            reg_down_total = (opt_df['RegDown_Electrolyzer_Deployed_MW'].fillna(0) +
                              opt_df['RegDown_Battery_Deployed_MW'].fillna(0) +
                              opt_df['RegDown_Turbine_Deployed_MW'].fillna(0)).sum()

            # Spinning and non-spinning reserves
            sr_total = (opt_df['SR_Electrolyzer_Deployed_MW'].fillna(0) +
                        opt_df['SR_Battery_Deployed_MW'].fillna(0) +
                        opt_df['SR_Turbine_Deployed_MW'].fillna(0)).sum()

            nsr_total = (opt_df['NSR_Electrolyzer_Deployed_MW'].fillna(0) +
                         opt_df['NSR_Battery_Deployed_MW'].fillna(0) +
                         opt_df['NSR_Turbine_Deployed_MW'].fillna(0)).sum()

            # Other ancillary services (if available)
            ecrs_total = 0
            thirty_min_total = 0
            ramp_up_total = 0
            ramp_down_total = 0

            # Check if columns exist before accessing
            if 'ECRS_Electrolyzer_Deployed_MW' in opt_df.columns:
                ecrs_total = (opt_df['ECRS_Electrolyzer_Deployed_MW'].fillna(0) +
                              opt_df['ECRS_Battery_Deployed_MW'].fillna(0) +
                              opt_df['ECRS_Turbine_Deployed_MW'].fillna(0)).sum()

            if 'ThirtyMin_Electrolyzer_Deployed_MW' in opt_df.columns:
                thirty_min_total = (opt_df['ThirtyMin_Electrolyzer_Deployed_MW'].fillna(0) +
                                    opt_df['ThirtyMin_Battery_Deployed_MW'].fillna(0) +
                                    opt_df['ThirtyMin_Turbine_Deployed_MW'].fillna(0)).sum()

            if 'RampUp_Electrolyzer_Deployed_MW' in opt_df.columns:
                ramp_up_total = (opt_df['RampUp_Electrolyzer_Deployed_MW'].fillna(0) +
                                 opt_df['RampUp_Battery_Deployed_MW'].fillna(0) +
                                 opt_df['RampUp_Turbine_Deployed_MW'].fillna(0)).sum()

                ramp_down_total = (opt_df['RampDown_Electrolyzer_Deployed_MW'].fillna(0) +
                                   opt_df['RampDown_Battery_Deployed_MW'].fillna(0) +
                                   opt_df['RampDown_Turbine_Deployed_MW'].fillna(0)).sum()

            ancillary_services = {
                'regulation_up_mwh': reg_up_total,
                'regulation_down_mwh': reg_down_total,
                'spinning_reserve_mwh': sr_total,
                'non_spinning_reserve_mwh': nsr_total,
                'ecrs_mwh': ecrs_total,
                'thirty_min_reserve_mwh': thirty_min_total,
                'ramp_up_mwh': ramp_up_total,
                'ramp_down_mwh': ramp_down_total,
                'total_ancillary_services_mwh': (reg_up_total + reg_down_total + sr_total +
                                                 nsr_total + ecrs_total + thirty_min_total +
                                                 ramp_up_total + ramp_down_total)
            }

            logger.info(f"Loaded ancillary services for {plant_name}:")
            logger.info(f"  Regulation Up: {reg_up_total:.1f} MWh")
            logger.info(f"  Regulation Down: {reg_down_total:.1f} MWh")
            logger.info(f"  Spinning Reserve: {sr_total:.1f} MWh")
            logger.info(f"  Non-Spinning Reserve: {nsr_total:.1f} MWh")
            logger.info(
                f"  Total: {ancillary_services['total_ancillary_services_mwh']:.1f} MWh")

            return ancillary_services

        except Exception as e:
            logger.error(
                f"Error loading optimization ancillary services for {plant_name}: {e}")
            return None

    def calculate_ancillary_services_emissions(self,
                                               plant_name: str,
                                               iso_region: str) -> AncillaryServicesEmissions:
        """
        Calculate carbon benefits from ancillary services using actual optimization results

        Args:
            plant_name: Name of nuclear plant  
            iso_region: ISO region

        Returns:
            AncillaryServicesEmissions object
        """
        as_emissions = AncillaryServicesEmissions()

        # Load actual ancillary services data from optimization results
        opt_as_data = self.load_optimization_ancillary_services(plant_name)

        if not opt_as_data or opt_as_data['total_ancillary_services_mwh'] == 0:
            logger.info(f"No ancillary services found for {plant_name}")
            return as_emissions

        # Fill in detailed service breakdown
        as_emissions.regulation_up_mwh = opt_as_data['regulation_up_mwh']
        as_emissions.regulation_down_mwh = opt_as_data['regulation_down_mwh']
        as_emissions.spinning_reserve_actual_mwh = opt_as_data['spinning_reserve_mwh']
        as_emissions.non_spinning_reserve_mwh = opt_as_data['non_spinning_reserve_mwh']
        as_emissions.ramp_up_mwh = opt_as_data['ramp_up_mwh']
        as_emissions.ramp_down_mwh = opt_as_data['ramp_down_mwh']
        as_emissions.ecrs_mwh = opt_as_data['ecrs_mwh']
        as_emissions.thirty_min_reserve_mwh = opt_as_data['thirty_min_reserve_mwh']

        # Use actual deployed amounts from optimization for aggregated values
        as_emissions.regulation_service_mwh = (opt_as_data['regulation_up_mwh'] +
                                               opt_as_data['regulation_down_mwh'])
        as_emissions.spinning_reserve_mwh = (opt_as_data['spinning_reserve_mwh'] +
                                             opt_as_data['non_spinning_reserve_mwh'])
        as_emissions.load_following_mwh = (opt_as_data['ramp_up_mwh'] +
                                           opt_as_data['ramp_down_mwh'])

        # Calculate avoided emissions based on actual service provision
        # Nuclear flexibility displaces fast-ramping fossil units

        # Regulation services typically displace gas turbines (fast response)
        regulation_avoided = (as_emissions.regulation_service_mwh *
                              self.config.gas_turbine_emission_factor / 1000)  # Convert g to kg

        # Spinning reserves may displace mix of gas and coal units
        spinning_avoided = (as_emissions.spinning_reserve_mwh *
                            (self.config.gas_turbine_emission_factor + self.config.coal_plant_emission_factor) / 2 / 1000)

        # Load following (ramping) displaces mix of generation based on grid
        grid_factor = self.config.iso_grid_emission_factors.get(
            iso_region, 450.0)
        load_following_avoided = as_emissions.load_following_mwh * \
            grid_factor / 1000  # Convert g to kg

        as_emissions.avoided_gas_turbine_emissions = regulation_avoided
        as_emissions.avoided_coal_emissions = spinning_avoided
        as_emissions.avoided_grid_emissions = load_following_avoided
        as_emissions.total_avoided_emissions = (regulation_avoided +
                                                spinning_avoided +
                                                load_following_avoided)

        # Calculate specific emission rate
        total_as_mwh = opt_as_data['total_ancillary_services_mwh']
        if total_as_mwh > 0:
            as_emissions.avoided_emissions_per_mwh = as_emissions.total_avoided_emissions / total_as_mwh

        logger.info(
            f"Ancillary services avoided emissions for {plant_name}: {as_emissions.total_avoided_emissions:,.0f} kg CO2-eq/year")
        logger.info(
            f"  From regulation services: {regulation_avoided:,.0f} kg CO2-eq/year")
        logger.info(
            f"  From spinning reserves: {spinning_avoided:,.0f} kg CO2-eq/year")
        logger.info(
            f"  From load following: {load_following_avoided:,.0f} kg CO2-eq/year")

        return as_emissions

    def analyze_plant(self, plant_name: str, iso_region: str = None) -> Optional[NuclearHydrogenLCAResults]:
        """
        Perform comprehensive LCA analysis for a nuclear-hydrogen plant

        Args:
            plant_name: Name of nuclear plant
            iso_region: ISO region (extracted from plant name if not provided)

        Returns:
            NuclearHydrogenLCAResults object or None
        """
        try:
            # Extract ISO region from plant name if not provided
            if not iso_region:
                iso_region = self._extract_iso_from_plant_name(plant_name)

            logger.info(
                f"Starting LCA analysis for {plant_name} in {iso_region}")

            # Load TEA results
            tea_data = self.load_tea_results(plant_name)
            if not tea_data:
                logger.error(f"Failed to load TEA data for {plant_name}")
                return None

            # Create results object
            results = NuclearHydrogenLCAResults(
                plant_name=plant_name,
                iso_region=iso_region,
                analysis_timestamp=datetime.now(),
                nuclear_capacity_mw=tea_data.get(
                    'nuclear_capacity_mw', 1000.0),
                electrolyzer_capacity_mw=tea_data.get(
                    'electrolyzer_capacity_mw', 0.0),
                annual_hydrogen_production_kg=tea_data.get(
                    'annual_h2_production_kg', 0.0),
                annual_electricity_generation_mwh=tea_data.get(
                    'annual_electricity_generation_mwh', 0.0)
            )

            # Calculate nuclear baseline emissions
            results.nuclear_baseline_emissions = self._calculate_nuclear_baseline_emissions()

            # Calculate hydrogen system emissions
            if results.annual_hydrogen_production_kg > 0:
                results.hydrogen_system_emissions = self.calculate_hydrogen_system_emissions(
                    tea_data, iso_region)

            # Calculate ancillary services emissions (regardless of hydrogen production)
            results.ancillary_services_emissions = self.calculate_ancillary_services_emissions(
                plant_name, iso_region)

            # Calculate summary metrics
            results.calculate_summary_metrics()

            logger.info(f"LCA analysis completed for {plant_name}")
            logger.info(
                f"Nuclear-only carbon intensity: {results.nuclear_only_carbon_intensity:.1f} gCO2-eq/kWh")
            logger.info(
                f"Integrated carbon intensity: {results.integrated_carbon_intensity:.1f} gCO2-eq/kWh")
            logger.info(
                f"Carbon reduction: {results.carbon_reduction_percentage:.1f}%")

            return results

        except Exception as e:
            logger.error(f"Error in LCA analysis for {plant_name}: {e}")
            return None

    def _extract_iso_from_plant_name(self, plant_name: str) -> str:
        """Extract ISO region from plant name"""
        iso_mapping = {
            'PJM': ['PJM'],
            'ERCOT': ['ERCOT'],
            'CAISO': ['CAISO'],
            'NYISO': ['NYISO'],
            'ISONE': ['ISONE'],
            'MISO': ['MISO'],
            'SPP': ['SPP']
        }

        for iso, keywords in iso_mapping.items():
            for keyword in keywords:
                if keyword in plant_name.upper():
                    return iso

        return 'PJM'  # Default

    def _calculate_nuclear_baseline_emissions(self) -> LifecycleEmissions:
        """Calculate nuclear baseline lifecycle emissions"""
        # Use IPCC 2014 median values as baseline
        baseline_factor = self.nuclear_config.get_default_nuclear_emissions()

        # Use detailed breakdown from config
        emissions = LifecycleEmissions(
            uranium_mining_milling=2.8,
            uranium_conversion=0.4,
            uranium_enrichment=0.9,
            fuel_fabrication=0.3,
            plant_construction=2.0,
            plant_operation=0.0,  # Direct operational emissions
            waste_management=1.0,
            decommissioning=0.5
        )

        return emissions

    def compare_iso_regions(self, plant_results: List[NuclearHydrogenLCAResults]) -> Dict:
        """
        Compare carbon footprint results across different ISO regions

        Args:
            plant_results: List of plant analysis results

        Returns:
            Dictionary with comparative analysis
        """
        if not plant_results:
            return {}

        comparison = {
            'summary': {},
            'by_iso': {},
            'ranking': {}
        }

        # Group results by ISO
        iso_results = {}
        for result in plant_results:
            iso = result.iso_region
            if iso not in iso_results:
                iso_results[iso] = []
            iso_results[iso].append(result)

        # Calculate ISO-level statistics
        for iso, results in iso_results.items():
            iso_stats = {
                'plant_count': len(results),
                'avg_nuclear_intensity': np.mean([r.nuclear_only_carbon_intensity for r in results]),
                'avg_integrated_intensity': np.mean([r.integrated_carbon_intensity for r in results]),
                'avg_carbon_reduction_pct': np.mean([r.carbon_reduction_percentage for r in results]),
                'total_annual_h2_production': sum([r.annual_hydrogen_production_kg for r in results]),
                'total_annual_carbon_reduction': sum([r.annual_carbon_reduction_kg for r in results])
            }
            comparison['by_iso'][iso] = iso_stats

        # Overall summary
        all_results = plant_results
        comparison['summary'] = {
            'total_plants': len(all_results),
            'avg_nuclear_intensity': np.mean([r.nuclear_only_carbon_intensity for r in all_results]),
            'avg_integrated_intensity': np.mean([r.integrated_carbon_intensity for r in all_results]),
            'avg_carbon_reduction_pct': np.mean([r.carbon_reduction_percentage for r in all_results]),
            'total_annual_h2_production': sum([r.annual_hydrogen_production_kg for r in all_results]),
            'total_annual_carbon_reduction': sum([r.annual_carbon_reduction_kg for r in all_results])
        }

        # Rank ISOs by carbon performance
        iso_performance = []
        for iso, stats in comparison['by_iso'].items():
            iso_performance.append({
                'iso': iso,
                'carbon_reduction_pct': stats['avg_carbon_reduction_pct'],
                'integrated_intensity': stats['avg_integrated_intensity']
            })

        # Sort by carbon reduction percentage (descending)
        iso_performance.sort(
            key=lambda x: x['carbon_reduction_pct'], reverse=True)
        comparison['ranking'] = iso_performance

        return comparison

    def generate_summary_report(self,
                                results: List[NuclearHydrogenLCAResults],
                                output_file: Path) -> bool:
        """
        Generate comprehensive LCA summary report

        Args:
            results: List of plant analysis results
            output_file: Path for output report

        Returns:
            True if successful
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("Nuclear-Hydrogen System LCA Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Plants Analyzed: {len(results)}\n\n")

                # Executive Summary
                f.write("Executive Summary\n")
                f.write("-" * 20 + "\n")
                if results:
                    avg_nuclear = np.mean(
                        [r.nuclear_only_carbon_intensity for r in results])
                    avg_integrated = np.mean(
                        [r.integrated_carbon_intensity for r in results])
                    avg_reduction = np.mean(
                        [r.carbon_reduction_percentage for r in results])
                    total_h2 = sum(
                        [r.annual_hydrogen_production_kg for r in results])
                    total_reduction = sum(
                        [r.annual_carbon_reduction_kg for r in results])

                    f.write(
                        f"Average Nuclear-Only Carbon Intensity: {avg_nuclear:.1f} gCO2-eq/kWh\n")
                    f.write(
                        f"Average Integrated System Carbon Intensity: {avg_integrated:.1f} gCO2-eq/kWh\n")
                    f.write(
                        f"Average Carbon Reduction: {avg_reduction:.1f}%\n")
                    f.write(
                        f"Total Annual Hydrogen Production: {total_h2:,.0f} kg/year\n")
                    f.write(
                        f"Total Annual Carbon Reduction: {total_reduction:,.0f} kg CO2-eq/year\n\n")

                # ISO Comparison
                iso_comparison = self.compare_iso_regions(results)
                if iso_comparison:
                    f.write("ISO Region Comparison\n")
                    f.write("-" * 25 + "\n")
                    for iso_data in iso_comparison['ranking']:
                        iso = iso_data['iso']
                        f.write(
                            f"{iso}: {iso_data['carbon_reduction_pct']:.1f}% carbon reduction, ")
                        f.write(
                            f"{iso_data['integrated_intensity']:.1f} gCO2-eq/kWh\n")
                    f.write("\n")

                # Individual Plant Results
                f.write("Individual Plant Results\n")
                f.write("-" * 30 + "\n")
                for result in results:
                    f.write(f"\nPlant: {result.plant_name}\n")
                    f.write(f"ISO Region: {result.iso_region}\n")
                    f.write(
                        f"Nuclear Capacity: {result.nuclear_capacity_mw:.1f} MW\n")
                    f.write(
                        f"Electrolyzer Capacity: {result.electrolyzer_capacity_mw:.1f} MW\n")
                    f.write(
                        f"Annual H2 Production: {result.annual_hydrogen_production_kg:,.0f} kg/year\n")
                    f.write(
                        f"Nuclear-Only Carbon Intensity: {result.nuclear_only_carbon_intensity:.1f} gCO2-eq/kWh\n")
                    f.write(
                        f"Integrated Carbon Intensity: {result.integrated_carbon_intensity:.1f} gCO2-eq/kWh\n")
                    f.write(
                        f"Carbon Reduction: {result.carbon_reduction_percentage:.1f}%\n")
                    f.write(
                        f"Annual Carbon Reduction: {result.annual_carbon_reduction_kg:,.0f} kg CO2-eq/year\n")

                    if result.hydrogen_system_emissions:
                        f.write(
                            f"H2 System Emissions: {result.hydrogen_system_emissions.total_hydrogen_emissions:.1f} gCO2-eq/kg H2\n")
                        f.write(
                            f"Avoided vs Conventional H2: {result.hydrogen_system_emissions.avoided_conventional_h2:.1f} gCO2-eq/kg H2\n")

                    if result.ancillary_services_emissions:
                        f.write(
                            f"AS Avoided Emissions: {result.ancillary_services_emissions.total_avoided_emissions:,.0f} kg CO2-eq/year\n")

                    f.write("\n--- Holistic System Impact ---\n")
                    f.write(
                        f"Net Annual Carbon Impact: {result.net_annual_carbon_impact_kg:,.0f} kg CO2-eq/year\n")
                    f.write(
                        f"Net Equivalent Carbon Intensity: {result.net_equivalent_carbon_intensity:.1f} gCO2-eq/kWh\n")

                    f.write("-" * 50 + "\n")

            logger.info(f"LCA summary report generated: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return False


def run_batch_nuclear_hydrogen_lca(plants: List[str],
                                   output_dir: Path,
                                   config: Optional[NuclearHydrogenSystemConfig] = None) -> bool:
    """
    Run batch LCA analysis for multiple nuclear-hydrogen plants

    Args:
        plants: List of plant names to analyze
        output_dir: Output directory for results
        config: System configuration

    Returns:
        True if successful
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = NuclearHydrogenLCAAnalyzer(config)
        results = []

        for plant_name in plants:
            logger.info(f"Analyzing plant: {plant_name}")
            result = analyzer.analyze_plant(plant_name)
            if result:
                results.append(result)

                # Save individual plant report
                plant_report_file = output_dir / \
                    f"{plant_name}_LCA_Analysis.txt"
                analyzer.generate_summary_report([result], plant_report_file)

        # Generate comprehensive report
        if results:
            summary_report_file = output_dir / "Nuclear_Hydrogen_LCA_Summary.txt"
            analyzer.generate_summary_report(results, summary_report_file)

            logger.info(
                f"Batch LCA analysis completed: {len(results)} plants analyzed")
            return True
        else:
            logger.warning("No successful plant analyses completed")
            return False

    except Exception as e:
        logger.error(f"Error in batch LCA analysis: {e}")
        return False
