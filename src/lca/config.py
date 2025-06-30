"""
Nuclear Power Plant LCA Configuration
Life Cycle Assessment configuration file for nuclear power plants

Based on latest real data from IPCC 2014, UNECE 2022, and other authoritative sources
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =================================================================
# LCA Output Configuration
# =================================================================

# Get project root directory (go up from config.py -> lca -> src -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
LCA_OUTPUT_DIR = PROJECT_ROOT / "output" / "lca"
LCA_LOG_DIR = PROJECT_ROOT / "output" / "logs" / "lca"
LCA_REPORTS_DIR = LCA_OUTPUT_DIR / "reports"
LCA_DATA_DIR = LCA_OUTPUT_DIR / "data"

# =================================================================
# Logging Configuration
# =================================================================


def setup_lca_logging(log_level=logging.INFO, log_to_file=True):
    """
    Setup LCA-specific logging configuration

    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file (default: True)
    """
    import os
    from datetime import datetime

    # Create log directory if it doesn't exist
    LCA_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler (WARNING and above only for cleaner output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        # Add file handler for detailed logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LCA_LOG_DIR / f"lca_analysis_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        print(f"📋 Detailed logs will be saved to: {log_file}")

    return root_logger


@dataclass
class EmissionFactors:
    """Emission factors data class"""
    value: float  # gCO2-eq/kWh or other units
    source: str   # Data source
    year: int     # Data year
    uncertainty_range: Optional[Tuple[float, float]
                                ] = None  # Uncertainty range
    description: str = ""


class NuclearLCAConfig:
    """Nuclear power plant LCA configuration class"""

    def __init__(self):
        """Initialize configuration based on latest international authoritative data"""

        # =================================================================
        # 1. Nuclear lifecycle emission factors (gCO2-eq/kWh)
        # =================================================================

        # Main data sources and their emission factors
        self.nuclear_lifecycle_emissions = {
            # IPCC 2014 data (most authoritative international reference)
            "ipcc_2014_median": EmissionFactors(
                value=12.0,
                source="IPCC AR5 Working Group III (2014) - Annex III",
                year=2014,
                uncertainty_range=(3.7, 110.0),
                description="IPCC 2014 Fifth Assessment Report, median from hundreds of studies"
            ),

            # UNECE 2022 data (latest UN data)
            "unece_2022_nuclear": EmissionFactors(
                value=5.1,
                source="UN Economic Commission for Europe (2022)",
                year=2022,
                uncertainty_range=(5.1, 6.4),
                description="UN ECE 2022 latest assessment, lowest among all low-carbon technologies"
            ),

            # French EDF 2022 latest study
            "edf_2022_france": EmissionFactors(
                value=4.0,
                source="Électricité de France (EDF) - ISO 14040 LCA Study",
                year=2022,
                uncertainty_range=(3.5, 4.5),
                description="EDF 2022 detailed LCA study following ISO 14040 standard"
            ),

            # Country-specific data (based on official national reports)
            "france_official": EmissionFactors(
                value=6.0,
                source="Base Carbone (ADEME) France",
                year=2023,
                description="French Environment and Energy Management Agency official carbon database"
            ),

            "uk_official": EmissionFactors(
                value=6.4,
                source="UK Parliamentary Office of Science and Technology",
                year=2011,
                description="UK Parliamentary Office of Science and Technology"
            ),

            "switzerland_psi": EmissionFactors(
                value=15.0,
                source="Paul Scherrer Institute (PSI) Switzerland",
                year=2017,
                uncertainty_range=(10.0, 20.0),
                description="Paul Scherrer Institute Switzerland"
            ),

            "us_nrel": EmissionFactors(
                value=13.0,
                source="National Renewable Energy Laboratory (NREL) USA",
                year=2021,
                description="US National Renewable Energy Laboratory"
            ),
        }

        # =================================================================
        # 2. Detailed emission breakdown by lifecycle stages
        # =================================================================

        # Based on Gibon & Menacho (2023) parametric LCA study detailed breakdown
        self.lifecycle_breakdown = {
            # Front-end fuel cycle
            "uranium_mining_milling": EmissionFactors(
                value=2.8,  # 46% of 6.1 total
                source="Gibon & Menacho (2023) - Parametric LCA Nuclear Power",
                year=2023,
                uncertainty_range=(1.5, 4.5),
                description="Uranium mining and milling, including in-situ leaching (ISL) and conventional mining"
            ),

            "uranium_conversion": EmissionFactors(
                value=0.4,
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.3, 0.6),
                description="Uranium conversion (yellowcake to uranium hexafluoride)"
            ),

            "uranium_enrichment": EmissionFactors(
                value=0.9,  # Based on centrifuge method, much lower than gas diffusion
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.5, 2.5),
                description="Uranium enrichment (modern centrifuge method, replacing high-energy gas diffusion)"
            ),

            "fuel_fabrication": EmissionFactors(
                value=0.3,
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.2, 0.5),
                description="Fuel assembly fabrication"
            ),

            # Construction phase
            "plant_construction": EmissionFactors(
                value=0.8,  # 13% of 6.1 total
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.4, 1.5),
                description="Nuclear plant construction, including concrete, steel and construction processes"
            ),

            # Operation phase
            "plant_operation": EmissionFactors(
                value=0.3,  # 5% of 6.1 total
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.2, 0.5),
                description="Plant operation and maintenance, including cooling water, spare parts replacement"
            ),

            # Back-end processing
            "waste_management": EmissionFactors(
                value=0.6,  # Including interim storage and final disposal
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.3, 1.0),
                description="Nuclear waste management, including interim storage and geological disposal"
            ),

            "decommissioning": EmissionFactors(
                value=0.2,
                source="Gibon & Menacho (2023)",
                year=2023,
                uncertainty_range=(0.1, 0.4),
                description="Plant decommissioning and dismantling"
            ),
        }

        # =================================================================
        # 3. Hydrogen production emission factors
        # =================================================================

        self.hydrogen_production_emissions = {
            # Nuclear-based hydrogen production via electrolysis
            "nuclear_electrolysis": EmissionFactors(
                value=0.5,  # kgCO2-eq/kgH2 (based on nuclear electricity)
                source="IEA Hydrogen Roadmap (2021) + Nuclear LCA",
                year=2021,
                uncertainty_range=(0.3, 0.8),
                description="Hydrogen production via electrolysis using nuclear electricity"
            ),

            # Conventional hydrogen production for comparison
            "steam_methane_reforming": EmissionFactors(
                value=9.3,  # kgCO2-eq/kgH2
                source="IEA Hydrogen Report (2021)",
                year=2021,
                uncertainty_range=(8.5, 10.5),
                description="Conventional hydrogen production via steam methane reforming"
            ),

            "coal_gasification": EmissionFactors(
                value=19.3,  # kgCO2-eq/kgH2
                source="IEA Hydrogen Report (2021)",
                year=2021,
                uncertainty_range=(17.0, 22.0),
                description="Hydrogen production via coal gasification"
            ),

            # Grid-based electrolysis (varies by grid mix)
            "grid_electrolysis_world_avg": EmissionFactors(
                value=26.0,  # kgCO2-eq/kgH2 (based on world average grid)
                source="IEA Hydrogen Report (2021)",
                year=2021,
                uncertainty_range=(15.0, 40.0),
                description="Hydrogen production via electrolysis using world average grid electricity"
            ),

            "grid_electrolysis_renewable": EmissionFactors(
                value=0.8,  # kgCO2-eq/kgH2 (based on renewable electricity)
                source="IEA Hydrogen Report (2021)",
                year=2021,
                uncertainty_range=(0.5, 1.2),
                description="Hydrogen production via electrolysis using renewable electricity"
            ),
        }

        # =================================================================
        # 4. Technical parameters (typical PWR parameters)
        # =================================================================

        self.technical_parameters = {
            # Basic parameters
            "reactor_type": "PWR",  # Pressurized Water Reactor
            "thermal_power_mw": 3000,  # Thermal power (MWth)
            "electric_power_mw": 1000,  # Electric power (MWe)
            "thermal_efficiency": 0.33,  # Thermal efficiency
            "capacity_factor": 0.90,  # Capacity factor
            "plant_lifetime_years": 60,  # Design lifetime

            # Fuel cycle parameters
            "fuel_enrichment_percent": 4.2,  # Fuel enrichment
            "fuel_burnup_mwd_per_kg": 42,  # Fuel burnup
            "fuel_cycle_length_months": 18,  # Refueling cycle
            "natural_uranium_per_enriched_kg": 8.1,  # Natural uranium requirement
            "separative_work_swu_per_kg": 6.7,  # Separative work requirement

            # Construction parameters
            "construction_time_years": 10,  # Construction period
            "concrete_tonnes": 400000,  # Concrete usage
            "steel_tonnes": 65000,  # Steel usage

            # Hydrogen production parameters
            "electrolyzer_efficiency": 0.65,  # Electrolyzer efficiency (HHV)
            "electrolyzer_lifetime_years": 20,  # Electrolyzer lifetime
            # Hydrogen energy content (HHV)
            "hydrogen_energy_content_mwh_per_kg": 0.0393,
        }

        # =================================================================
        # 5. Comparison technologies emission factors
        # =================================================================

        self.comparison_technologies = {
            # Fossil fuels (IPCC 2014 & UNECE 2022)
            "coal_pc": EmissionFactors(
                value=820,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(740, 910),
                description="Coal power - pulverized coal boiler"
            ),

            "natural_gas_ccgt": EmissionFactors(
                value=490,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(410, 650),
                description="Natural gas combined cycle power generation"
            ),

            # Renewable energy
            "wind_onshore": EmissionFactors(
                value=11,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(7, 56),
                description="Onshore wind power"
            ),

            "wind_offshore": EmissionFactors(
                value=12,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(8, 35),
                description="Offshore wind power"
            ),

            "solar_pv_utility": EmissionFactors(
                value=48,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(18, 180),
                description="Utility-scale ground-mounted PV"
            ),

            "solar_pv_rooftop": EmissionFactors(
                value=41,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(26, 60),
                description="Rooftop PV"
            ),

            "hydropower": EmissionFactors(
                value=24,
                source="IPCC 2014",
                year=2014,
                # Highly variable depending on reservoir type
                uncertainty_range=(1, 2200),
                description="Hydroelectric power"
            ),

            "geothermal": EmissionFactors(
                value=38,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(6, 79),
                description="Geothermal power"
            ),

            "biomass_dedicated": EmissionFactors(
                value=230,
                source="IPCC 2014",
                year=2014,
                uncertainty_range=(130, 420),
                description="Dedicated biomass power"
            ),
        }

        # =================================================================
        # 6. System boundaries and methodology parameters
        # =================================================================

        self.methodology_parameters = {
            "functional_unit_electricity": "1 kWh electricity delivered to grid",
            "functional_unit_hydrogen": "1 kg hydrogen produced",
            "system_boundary": "cradle-to-grave",
            "time_horizon_years": 100,  # GWP time horizon
            "cutoff_criteria": "1% of total impact",
            "allocation_method": "economic",
            "impact_categories": [
                "climate_change",
                "ozone_depletion",
                "acidification",
                "eutrophication",
                "human_toxicity",
                "ionizing_radiation",
                "land_use",
                "water_use",
                "mineral_resource_depletion"
            ],

            # Data quality requirements
            "data_quality": {
                "temporal_scope": "2014-2024",  # Time range
                "geographical_scope": "Global/OECD",  # Geographic scope
                "technology_scope": "Current commercial PWR",  # Technology scope
                "precision": 2,  # Result decimal places
            }
        }

        # =================================================================
        # 7. Uncertainty and sensitivity analysis parameters
        # =================================================================

        self.uncertainty_parameters = {
            # Main uncertainty sources
            "ore_grade": {"distribution": "lognormal", "cv": 0.5},
            "mining_method": {"distribution": "discrete", "options": ["ISL", "underground", "open_pit"]},
            "enrichment_method": {"distribution": "discrete", "options": ["centrifuge", "diffusion"]},
            "plant_lifetime": {"distribution": "triangular", "min": 40, "mode": 60, "max": 80},
            "capacity_factor": {"distribution": "normal", "mean": 0.90, "std": 0.05},

            # Monte Carlo parameters
            "monte_carlo_runs": 10000,
            "confidence_interval": 0.95,
        }

        logger.info("Nuclear LCA configuration loaded successfully")
        logger.info(
            f"Default nuclear lifecycle emissions: {self.get_default_nuclear_emissions():.1f} gCO2-eq/kWh")

    def get_default_nuclear_emissions(self) -> float:
        """Get default nuclear emission factor (using latest UNECE data)"""
        return self.nuclear_lifecycle_emissions["unece_2022_nuclear"].value

    def get_lifecycle_total(self) -> float:
        """Calculate total lifecycle stage emissions"""
        total = sum(stage.value for stage in self.lifecycle_breakdown.values())
        return round(total, 2)

    def validate_data_consistency(self) -> bool:
        """Validate data consistency"""
        total_calculated = self.get_lifecycle_total()
        reference_value = self.get_default_nuclear_emissions()

        # Allow 10% difference
        if abs(total_calculated - reference_value) / reference_value > 0.1:
            logger.warning(
                f"Data inconsistency detected: calculated {total_calculated}, reference {reference_value}")
            return False

        return True

    def get_comparison_data(self) -> Dict[str, float]:
        """Get comparison data"""
        comparison = {}
        comparison["Nuclear"] = self.get_default_nuclear_emissions()

        for tech, data in self.comparison_technologies.items():
            comparison[tech.replace('_', ' ').title()] = data.value

        return comparison


# Create global configuration instance
config = NuclearLCAConfig()

# Validate data consistency
if not config.validate_data_consistency():
    logger.warning("Configuration data may have inconsistencies")
