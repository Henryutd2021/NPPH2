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
DISCOUNT_RATE = 0.06
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21
HOURS_IN_YEAR = 8760

# Case Classification Configuration
CASE_CLASSIFICATION = {
    "existing_projects": {
        "cases": ["case1", "case2", "case3"],
        "description": "Existing nuclear plant retrofit/modification scenarios",
        "lifetime_source": "remaining_years",  # Use actual remaining plant lifetime
        "default_lifetime": 10,  # Fallback if remaining years not available
        "construction_period": 2,  # Retrofit construction period
        "include_nuclear_capex": False,  # Existing plant, no new nuclear CAPEX
        "tax_incentives": {
            "45u_eligible": True,  # 45U applies to existing plants
            "itc_ptc_eligible": False,  # ITC/PTC for new construction only
            "macrs_eligible": True  # MACRS applies to retrofit equipment
        }
    },
    "new_construction": {
        "cases": ["case4", "case5"],
        "description": "New nuclear-hydrogen integrated plant construction",
        "lifetime_source": "full_lifecycle",  # Use full plant lifetime
        "case4_lifetime": 60,  # Case 4: 60-year analysis
        "case5_lifetime": 80,  # Case 5: 80-year analysis
        "construction_period": 8,  # New nuclear construction period
        "include_nuclear_capex": True,  # New plant, include nuclear CAPEX
        "tax_incentives": {
            "45u_eligible": False,  # 45U for existing plants only
            "itc_ptc_eligible": True,  # ITC/PTC for new construction
            "macrs_eligible": True  # MACRS applies to all equipment
        }
    }
}

# Logging Configuration
LOG_LEVEL = "DEBUG"  # Changed from INFO to DEBUG for troubleshooting

# CAPEX Components (updated with sys_data_advanced.csv data)
CAPEX_COMPONENTS = {
    "Electrolyzer_System": {
        # Based on data_gen.py original costs: HTE = $1,500/kW = $1,500,000/MW
        # HTE (High Temperature Electrolysis) is more suitable for nuclear-hydrogen integration
        # due to thermal integration capabilities and higher efficiency
        # Total cost for 50MW reference: 50 * $1,500,000 = $75,000,000
        # HTE original CAPEX from data_gen.py
        "total_base_cost_for_ref_size": 75_000_000,
        "reference_total_capacity_mw": 50,
        "applies_to_component_capacity_key": "Electrolyzer_Capacity_MW",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "H2_Storage_System": {
        # Estimated based on industry standards: ~$1000-1500/kg storage capacity
        # For 100,000kg reference capacity: 100,000 * $160 = $16M
        "total_base_cost_for_ref_size": 16_000_000,  # Updated estimate
        "reference_total_capacity_mw": 100000,  # kg capacity
        "applies_to_component_capacity_key": "H2_Storage_Capacity_kg",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-2: 0.5, -1: 0.5},
    },
    "Battery_System_Energy": {  # Cost component for MWh capacity - DISABLED (power-only costing)
        # MODIFIED: Set to 0 to implement power-only battery costing strategy
        "total_base_cost_for_ref_size": 0,  # No energy capacity cost
        "reference_total_capacity_mw": 100,
        "applies_to_component_capacity_key": "Battery_Capacity_MWh",
        "learning_rate_decimal": 0.0,
        "payment_schedule_years": {-1: 1.0},
    },
    "Battery_System_Power": {  # Cost component for MW power
        # Based on data_gen.py original costs: $236/kWh * 4h = $944/kW = $944,000/MW
        # For 25MW reference: 25 * $944,000 = $23,600,000
        # Battery power CAPEX from data_gen.py
        "total_base_cost_for_ref_size": 23_600_000,
        "reference_total_capacity_mw": 25,  # MW power capacity
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

# O&M Components (updated with sys_data_advanced.csv data)
OM_COMPONENTS = {
    "Fixed_OM_General": {
        "base_cost_percent_of_capex": 0.02,
        "size_dependent": True,
        "inflation_rate": 0.02,
    },
    "Fixed_OM_Battery": {
        # MODIFIED: Switch to power-based O&M to align with power-only costing strategy
        # 1% of power CAPEX per year = 0.01 * $944,000 = $9,440/MW/year
        "base_cost_per_mw_year": 48_168.0,  # Power-based O&M cost
        "base_cost_per_mwh_year": 0,  # No energy-based O&M cost
        "inflation_rate": 0.02,
    },
    "Variable_OM_Electrolyzer": {
        # Based on data_gen.py: HTE variable O&M = $10.0/MWh
        # Using HTE (High Temperature Electrolysis) for nuclear-hydrogen integration
        "base_cost_per_mwh": 6.9,  # HTE VOM from data_gen.py
        "inflation_rate": 0.02,
    },
    "Variable_OM_H2_Storage": {
        # Based on data_gen.py: hydrogen storage VOM = $0.01/kg cycled
        "base_cost_per_kg_cycled": 0.01,  # H2 storage VOM from data_gen.py
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
        # $8B for 1000MW plant (more realistic)
        "total_base_cost_for_ref_size": 6_000_000_000,
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

# Centralized Nuclear Cost Parameters (for standardized calculations across all modules)
NUCLEAR_COST_PARAMETERS = {
    # CAPEX Parameters ($/MW, industry standard for new nuclear plants)
    # $/MW (standardized from industry data)
    "nuclear_capex_per_mw": 8_000_000,

    # CAPEX Breakdown Percentages (for detailed analysis)
    "capex_breakdown_percentages": {
        "Nuclear_Island": 0.45,      # 45% - reactor and primary systems
        "Turbine_Generator": 0.25,   # 25% - turbine and generator systems
        "Balance_of_Plant": 0.20,    # 20% - supporting systems
        "Owner_Costs": 0.10,         # 10% - owner costs and contingency
    },

    # OPEX Parameters (standardized across all calculations)
    "opex_parameters": {
        "fixed_om_per_mw_month": 20_000,      # $/MW/month (industry standard)
        "fixed_om_per_mw_year": 240_000,      # $/MW/year (20,000 * 12)
        # $/MWh (operations & maintenance)
        "variable_om_per_mwh": 1.0,
        "fuel_cost_per_mwh": 10.0,             # $/MWh (nuclear fuel costs)
        # $/MW/year (insurance, regulatory, waste, security)
        "additional_costs_per_mw_year": 90_000.0,
        "total_fixed_costs_per_mw_year": 330_000,  # $/MW/year (240,000 + 90,000)
    },

    # Operational Parameters
    "operational_parameters": {
        "typical_capacity_factor": 0.90,      # 90% capacity factor
        "hours_per_year": 8760,               # Hours in a year
        "typical_annual_generation_factor": 7884,  # hours/year * capacity_factor
    },

    # Nuclear Plant Replacement/Refurbishment Costs (for existing plants)
    "replacement_costs_per_mw": {
        "turbine_overhaul_15_years": 30_000,      # $/MW at year 15
        "steam_generator_25_years": 50_000,       # $/MW at year 25
        "major_refurbishment_30_years": 80_000,   # $/MW at year 30
        "life_extension_40_years": 120_000,       # $/MW at year 40
    },

    # Inflation Rates for different cost components
    "inflation_rates": {
        "fixed_om": 0.025,          # 2.5% for fixed O&M
        "variable_om": 0.02,        # 2.0% for variable O&M
        "fuel_costs": 0.02,         # 2.0% for fuel costs
        "additional_costs": 0.025,  # 2.5% for additional costs
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

# Construction Financing Configuration
CONSTRUCTION_FINANCING = {
    "interest_rate_during_construction": 0.06,  # 6% annual interest rate
    "financing_method": "compound",  # compound or simple
    # end_of_period or beginning_of_period
    "payment_schedule_type": "end_of_period",
}

# MACRS Depreciation Configuration
MACRS_CONFIG = {
    "enabled": True,
    "nuclear_depreciation_years": 15,
    "hydrogen_depreciation_years": 7,
    "battery_depreciation_years": 7,
    "grid_depreciation_years": 15,
    "component_classification": {
        # Nuclear components (15-year MACRS)
        "Nuclear_Power_Plant": "nuclear",
        "Nuclear_Island": "nuclear",
        "Nuclear_Site_Preparation": "nuclear",
        "Nuclear_Safety_Systems": "nuclear",
        "Nuclear_Grid_Connection": "grid",

        # Hydrogen components (7-year MACRS)
        "Electrolyzer_System": "hydrogen",
        "H2_Storage_System": "hydrogen",

        # Battery components (7-year MACRS)
        "Battery_System_Energy": "battery",
        "Battery_System_Power": "battery",

        # Infrastructure components
        "Grid_Integration": "grid",
        "NPP_Modifications": "nuclear"
    }
}

# Federal Tax Incentive Policy Configuration
TAX_INCENTIVE_POLICIES = {
    # 45U Production Tax Credit for Existing Nuclear Plants
    "45u_ptc": {
        "credit_rate_per_mwh": 15.0,              # $/MWh credit rate
        "credit_start_year": 2024,                # Policy start year
        "credit_end_year": 2032,                  # Policy end year
        "applies_to_existing_plants_only": True,  # Only for existing nuclear plants
        "description": "45U Nuclear Production Tax Credit for existing nuclear power plants"
    },

    # 45Y Production Tax Credit for New Nuclear Plants
    "45y_ptc": {
        # $/MWh credit rate (up to $30/MWh)
        "credit_rate_per_mwh": 30.0,
        "credit_duration_years": 10,              # Duration of credit eligibility
        "applies_to_new_plants": True,            # For new nuclear construction
        "description": "45Y Production Tax Credit for new nuclear power facilities"
    },

    # 48E Investment Tax Credit for Nuclear Plants
    "48e_itc": {
        "credit_rate": 0.50,                      # 50% ITC rate (up to 50%)
        # 50% of ITC amount reduces depreciation basis
        "depreciation_basis_reduction_rate": 0.50,
        "applies_to_nuclear_only": True,          # Only nuclear equipment qualifies
        "description": "48E Investment Tax Credit for nuclear power facilities"
    },

    # Policy sensitivity analysis parameters
    "sensitivity_analysis": {
        # $/MWh range for sensitivity analysis
        "45u_rate_range": [10.0, 15.0, 20.0],
        # $/MWh range for sensitivity analysis
        "45y_rate_range": [20.0, 30.0, 40.0],
        # ITC rate range for sensitivity analysis
        "48e_rate_range": [0.30, 0.50, 0.60],
        # Years range for PTC duration analysis
        "duration_range": [5, 10, 15],
    }
}

# Integrated System Cost Parameters (based on data_gen.py original costs)
SYSTEM_COST_PARAMETERS = {
    # Electrolyzer Parameters (from data_gen.py original costs)
    "electrolyzer": {
        # Original CAPEX costs (not annualized)
        "lte_capex_usd_per_mw": 1_000_000,        # LTE: $1,000/kW = $1M/MW
        "hte_capex_usd_per_mw": 1_500_000,        # HTE: $1,500/kW = $1.5M/MW
        "lifetime_years": 20,                     # Equipment lifetime
        "discount_rate": 0.06,                    # Discount rate for annualization

        # Annualized costs (for reference, calculated from original costs)
        "lte_capex_usd_per_mw_year": 87_275,  # LTE annualized
        "hte_capex_usd_per_mw_year": 130_912.5,  # HTE annualized

        # Operating costs
        "lte_vom_usd_per_mwh": 6.0,               # LTE variable O&M
        "hte_vom_usd_per_mwh": 6.9,              # HTE variable O&M
        "water_cost_usd_per_kg_h2": 0.03,         # Water cost
        "aux_power_per_kg_h2": 0.0005,            # Auxiliary power consumption

        # Efficiency parameters
        # LTE efficiency curve
        "lte_efficiency_kwh_per_kg": [57.0, 56.0, 55.0, 54.0],
        # HTE efficiency curve
        "hte_efficiency_kwh_per_kg": [44.0, 43.0, 42.0, 41.0],
    },

    # Battery Parameters (MODIFIED for power-only costing strategy)
    "battery": {
        # Original CAPEX costs (not annualized) - ENERGY COST DISABLED
        "energy_capex_usd_per_mwh": 0,            # No energy capacity cost
        # NOTE: power_capex_usd_per_mw includes 4-hour duration cost ($2.1M/MW for 4h storage)
        "power_capex_usd_per_mw": 2_100_000,
        "lifetime_years": 20,                     # Equipment lifetime
        "discount_rate": 0.06,                    # Discount rate for annualization

        # Annualized costs (for reference, calculated from original costs)
        "energy_capex_usd_per_mwh_year": 0,       # No energy capacity cost
        # Power capacity annualized (includes 4h duration, no double-counting)
        "power_capex_usd_per_mw_year": 183_277.5,

        # Operating costs
        # No energy-based O&M
        "fixed_om_usd_per_mwh_year": 0,
        # Power-based O&M (1% of power CAPEX, includes 4h duration consideration)
        "fixed_om_usd_per_mw_year": 48_168.0,

        # Technical parameters
        "duration_hours": 4.0,                       # Battery duration
        # Power to energy ratio (1/4h)
        "power_ratio_mw_per_mwh": 0.25,
        "charge_efficiency": 0.92,                   # Charge efficiency
        "discharge_efficiency": 0.92,                # Discharge efficiency
        "min_soc_fraction": 0.1,                     # Minimum state of charge
    },

    # H2 Storage Parameters (from sys_data_advanced.csv)
    "h2_storage": {
        "vom_usd_per_kg_cycled": 0.01,               # Variable O&M per kg cycled
        "charge_efficiency": 0.98,                   # Storage charge efficiency
        "discharge_efficiency": 0.98,                # Storage discharge efficiency
        "max_capacity_kg": 100_000.0,                # Maximum storage capacity
        "min_capacity_kg": 5_000.0,                  # Minimum storage capacity
    },

    # Turbine Parameters (from sys_data_advanced.csv)
    "turbine": {
        "vom_usd_per_mwh": 1.0,                      # Variable O&M
        "thermal_efficiency": 0.38,                  # Thermal to electric efficiency
        "ramp_up_rate_percent_per_min": 2.0,         # Ramp up rate
        "ramp_down_rate_percent_per_min": 2.0,       # Ramp down rate
    },

    # Economic Parameters (from sys_data_advanced.csv)
    "economic": {
        "h2_value_usd_per_kg": 3.0,                  # Hydrogen market value
        "hydrogen_subsidy_usd_per_kg": 3.0,          # Hydrogen subsidy
        "hydrogen_subsidy_duration_years": 10,       # Subsidy duration
        "discount_rate": 0.06,                       # Discount rate
    },
}

# Fallback/default values that might be overridden by framework imports
# These are included here for completeness, but the main script tea.py
# will handle the try-except block for framework imports.
# This will be overridden by framework's config.py
TARGET_ISO = "DEFAULT_ISO_FALLBACK"

# Import ENABLE_BATTERY from optimization config to ensure consistency
try:
    from src.opt.config import ENABLE_BATTERY
except ImportError:
    ENABLE_BATTERY = False  # Fallback if optimization config is not available
