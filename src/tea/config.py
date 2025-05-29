"""
Global configuration variables, constants, and parameter dictionaries for the TEA module.
"""

import os
from pathlib import Path

# Base file paths
SCRIPT_DIR_PATH = Path(__file__).resolve().parent
# Updated: Point to new output/logs directory structure
LOG_DIR = SCRIPT_DIR_PATH.parent.parent / "output" / "logs"
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists

# Add src directory to Python path (though this might be better handled at runtime in main script)
SRC_PATH = SCRIPT_DIR_PATH.parent / "src"
# The following sys.path.append should ideally be in the main script or managed by the execution environment
# import sys
# if str(SRC_PATH) not in sys.path:
#     sys.path.append(str(SRC_PATH))

# TEA Configuration
BASE_OUTPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent.parent / "output" / "tea"
BASE_INPUT_DIR_DEFAULT = SCRIPT_DIR_PATH.parent.parent / "input"

# TEA Parameters
PROJECT_LIFETIME_YEARS = 30
DISCOUNT_RATE = 0.08
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21
HOURS_IN_YEAR = 8760

# Logging Configuration
LOG_LEVEL = "DEBUG"  # Changed from INFO to DEBUG for troubleshooting

# CAPEX Components (with learning rate structure)
CAPEX_COMPONENTS = {
    "Electrolyzer_System": {
        "total_base_cost_for_ref_size": 100_000_000,  # 50MW * 1000 * $2000
        "reference_total_capacity_mw": 50,
        "applies_to_component_capacity_key": "Electrolyzer_Capacity_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "H2_Storage_System": {
        "total_base_cost_for_ref_size": 10_000_000,  # 10,000kg * $1000
        "reference_total_capacity_mw": 10000,  # Assuming kg
        "applies_to_component_capacity_key": "H2_Storage_Capacity_kg",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "Battery_System_Energy": {  # Cost component for MWh capacity
        "total_base_cost_for_ref_size": 23_600_000,  # 100MWh * 1000 * $236
        "reference_total_capacity_mw": 100,
        "applies_to_component_capacity_key": "Battery_Capacity_MWh",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-1: 1.0},
    },
    "Battery_System_Power": {  # Cost component for MW power
        "total_base_cost_for_ref_size": 5_000_000,
        "reference_total_capacity_mw": 25,  # Here unit is MW
        "applies_to_component_capacity_key": "Battery_Power_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-1: 1.0},
    },
    "Grid_Integration": {
        "total_base_cost_for_ref_size": 5_000_000,
        "reference_total_capacity_mw": 0,
        "applies_to_component_capacity_key": None,
        "learning_rate_decimal": 0,
        "payment_schedule_years": {-1: 1.0},
    },
    "NPP_Modifications": {
        "total_base_cost_for_ref_size": 2_000_000,
        "reference_total_capacity_mw": 0,
        "applies_to_component_capacity_key": None,
        "learning_rate_decimal": 0,
        "payment_schedule_years": {-2: 1.0},
    },
}

# O&M Components
OM_COMPONENTS = {
    "Fixed_OM_General": {
        "base_cost_percent_of_capex": 0.02,
        "size_dependent": True,
        "inflation_rate": 0.02,
    },
    "Fixed_OM_Battery": {
        "base_cost_per_mw_year": 25_000,
        "base_cost_per_mwh_year": 0,
        "inflation_rate": 0.02,
    },
}

# Replacement Schedule
REPLACEMENT_SCHEDULE = {
    "Electrolyzer_Stack": {
        "cost_percent_initial_capex": 0.30,
        "years": [10, 20],
        "size_dependent": True,
    },
    "H2_Storage_Components": {
        "cost": 5_000_000,
        "years": [15],
        "size_dependent": True,
    },
    "Battery_Augmentation_Replacement": {
        "cost_percent_initial_capex": 0.60,
        "years": [10],
        "size_dependent": True,
    },
}

# Nuclear Power Plant Integrated System Configuration
# For comprehensive nuclear + hydrogen analysis (60-year lifecycle)
NUCLEAR_INTEGRATED_CONFIG = {
    "enabled": True,  # Default enabled for greenfield nuclear-hydrogen analysis
    "project_lifetime_years": 60,  # Nuclear plant lifetime
    "construction_years": 8,  # Nuclear construction period
    # Default nuclear plant capacity (will be overridden by actual reactor size)
    "nuclear_plant_capacity_mw": 1000,
    # Include nuclear construction costs in greenfield analysis
    "enable_nuclear_capex": True,
    "enable_nuclear_opex": True,  # Include nuclear O&M costs in greenfield analysis
}

# Nuclear Plant CAPEX Components (separate from existing components)
NUCLEAR_CAPEX_COMPONENTS = {
    "Nuclear_Power_Plant": {
        # $10B for 1000MW plant (more realistic)
        "total_base_cost_for_ref_size": 10_000_000_000,
        "reference_total_capacity_mw": 1000,  # Reference capacity in MW
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.05,  # 5% learning rate for nuclear construction
        # 8-year construction (year 0-7)
        "payment_schedule_years": {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.10, 7: 0.05},
    },
    "Nuclear_Site_Preparation": {
        "total_base_cost_for_ref_size": 300_000_000,  # Reduced site preparation costs
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.02,
        "payment_schedule_years": {0: 0.8, 1: 0.2},  # Front-loaded
    },
    "Nuclear_Safety_Systems": {
        "total_base_cost_for_ref_size": 1_500_000_000,  # Reduced safety systems cost
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.03,
        "payment_schedule_years": {3: 0.3, 4: 0.4, 5: 0.3},
    },
    "Nuclear_Grid_Connection": {
        "total_base_cost_for_ref_size": 200_000_000,  # Reduced grid connection cost
        "reference_total_capacity_mw": 1000,
        "applies_to_component_capacity_key": "Nuclear_Plant_Capacity_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {6: 0.6, 7: 0.4},
    },
}

# Nuclear Plant O&M Components (separate from existing components)
NUCLEAR_OM_COMPONENTS = {
    "Nuclear_Fixed_OM": {
        "base_cost_per_mw_year": 120_000,  # $120k/MW/year for nuclear O&M
        "inflation_rate": 0.025,  # Slightly higher inflation for nuclear O&M
    },
    "Nuclear_Fuel_Cost": {
        "base_cost_per_mwh": 8.0,  # $8/MWh fuel cost (from PDF data)
        "inflation_rate": 0.02,
    },
    "Nuclear_Security": {
        "base_cost_per_mw_year": 15_000,  # $15k/MW/year for security
        "inflation_rate": 0.03,
    },
    "Nuclear_Regulatory": {
        "base_cost_per_mw_year": 8_000,  # $8k/MW/year for regulatory compliance
        "inflation_rate": 0.025,
    },
    "Nuclear_Waste_Management": {
        "base_cost_per_mwh": 1.0,  # $1/MWh for waste management
        "inflation_rate": 0.02,
    },
}

# Nuclear Plant Major Refurbishments/Replacements
NUCLEAR_REPLACEMENT_SCHEDULE = {
    "Steam_Generator_Replacement": {
        "cost_percent_initial_capex": 0.08,  # 8% of initial nuclear CAPEX
        "years": [25],  # Mid-life steam generator replacement
        "size_dependent": True,
    },
    "Reactor_Pressure_Vessel_Head": {
        "cost_percent_initial_capex": 0.03,  # 3% of initial nuclear CAPEX
        "years": [20, 40],  # Periodic replacements
        "size_dependent": True,
    },
    "Major_Maintenance_Outage": {
        "cost_percent_initial_capex": 0.02,  # 2% of initial nuclear CAPEX
        "years": [15, 30, 45],  # Major outages every 15 years
        "size_dependent": True,
    },
    "Control_Rod_Drive_Mechanisms": {
        "cost_percent_initial_capex": 0.015,  # 1.5% of initial nuclear CAPEX
        "years": [18, 36, 54],  # Replacement every 18 years
        "size_dependent": True,
    },
}

# Fallback/default values that might be overridden by framework imports
# These are included here for completeness, but the main script tea.py
# will handle the try-except block for framework imports.
# This will be overridden by framework's config.py
TARGET_ISO = "DEFAULT_ISO_FALLBACK"
ENABLE_BATTERY = False  # This will be overridden by framework's config.py
