"""
MACRS (Modified Accelerated Cost Recovery System) depreciation calculations for TEA module.

This module implements MACRS depreciation schedules for nuclear-hydrogen integrated systems:
- Nuclear equipment: 15-year MACRS
- Hydrogen equipment: 7-year MACRS  
- Battery systems: 7-year MACRS
- Grid infrastructure: 15-year MACRS
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def get_macrs_schedule(depreciation_years: int) -> List[float]:
    """
    Get MACRS depreciation schedule for specified number of years.
    
    Args:
        depreciation_years: Number of years for depreciation (7 or 15)
        
    Returns:
        List of annual depreciation percentages
    """
    if depreciation_years == 7:
        # 7-year MACRS schedule (half-year convention)
        return [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
    elif depreciation_years == 15:
        # 15-year MACRS schedule (half-year convention)
        return [
            0.05, 0.095, 0.0855, 0.077, 0.0693, 0.0623, 0.059, 0.059,
            0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.0295
        ]
    else:
        logger.error(f"Unsupported MACRS depreciation period: {depreciation_years} years")
        return []


def classify_component_for_macrs(component_name: str, macrs_config: dict) -> str:
    """
    Classify a component for MACRS depreciation based on configuration.
    
    Args:
        component_name: Name of the component
        macrs_config: MACRS configuration dictionary
        
    Returns:
        Component classification ('nuclear', 'hydrogen', 'battery', 'grid')
    """
    classification_map = macrs_config.get("component_classification", {})
    return classification_map.get(component_name, "hydrogen")  # Default to hydrogen (7-year)


def get_depreciation_years_for_component(component_name: str, macrs_config: dict) -> int:
    """
    Get depreciation years for a specific component.
    
    Args:
        component_name: Name of the component
        macrs_config: MACRS configuration dictionary
        
    Returns:
        Number of depreciation years
    """
    classification = classify_component_for_macrs(component_name, macrs_config)
    
    if classification == "nuclear":
        return macrs_config.get("nuclear_depreciation_years", 15)
    elif classification == "hydrogen":
        return macrs_config.get("hydrogen_depreciation_years", 7)
    elif classification == "battery":
        return macrs_config.get("battery_depreciation_years", 7)
    elif classification == "grid":
        return macrs_config.get("grid_depreciation_years", 15)
    else:
        logger.warning(f"Unknown component classification: {classification}. Using 7-year default.")
        return 7


def calculate_component_macrs_depreciation(
    component_name: str,
    component_capex: float,
    construction_period_years: int,
    project_lifetime_years: int,
    macrs_config: dict
) -> np.ndarray:
    """
    Calculate MACRS depreciation schedule for a single component.
    
    Args:
        component_name: Name of the component
        component_capex: Total CAPEX for the component
        construction_period_years: Number of construction years
        project_lifetime_years: Total project lifetime
        macrs_config: MACRS configuration dictionary
        
    Returns:
        Array of annual depreciation amounts for each year of the project
    """
    if not macrs_config.get("enabled", True):
        logger.info("MACRS depreciation is disabled")
        return np.zeros(construction_period_years + project_lifetime_years)
    
    if component_capex <= 0:
        return np.zeros(construction_period_years + project_lifetime_years)
    
    # Get depreciation parameters for this component
    depreciation_years = get_depreciation_years_for_component(component_name, macrs_config)
    macrs_schedule = get_macrs_schedule(depreciation_years)
    
    if not macrs_schedule:
        logger.error(f"Could not get MACRS schedule for {component_name}")
        return np.zeros(construction_period_years + project_lifetime_years)
    
    # Initialize depreciation array
    total_years = construction_period_years + project_lifetime_years
    depreciation_array = np.zeros(total_years)
    
    # MACRS depreciation starts in the first operational year (after construction)
    start_year = construction_period_years
    
    # Apply MACRS schedule
    for i, percentage in enumerate(macrs_schedule):
        year_index = start_year + i
        if year_index < total_years:
            depreciation_array[year_index] = component_capex * percentage
            logger.debug(f"{component_name} Year {year_index}: ${depreciation_array[year_index]:,.0f} "
                        f"({percentage:.1%} of ${component_capex:,.0f})")
    
    total_depreciation = np.sum(depreciation_array)
    logger.info(f"{component_name} total MACRS depreciation: ${total_depreciation:,.0f} "
               f"({total_depreciation/component_capex:.1%} of CAPEX)")
    
    return depreciation_array


def calculate_total_macrs_depreciation(
    capex_breakdown: dict,
    construction_period_years: int,
    project_lifetime_years: int,
    macrs_config: dict
) -> Tuple[np.ndarray, dict]:
    """
    Calculate total MACRS depreciation for all components.
    
    Args:
        capex_breakdown: Dictionary of component CAPEX values
        construction_period_years: Number of construction years
        project_lifetime_years: Total project lifetime
        macrs_config: MACRS configuration dictionary
        
    Returns:
        Tuple of (total_depreciation_array, component_depreciation_breakdown)
    """
    if not macrs_config.get("enabled", True):
        logger.info("MACRS depreciation is disabled")
        total_years = construction_period_years + project_lifetime_years
        return np.zeros(total_years), {}
    
    logger.info("Calculating MACRS depreciation for all components")
    
    total_years = construction_period_years + project_lifetime_years
    total_depreciation = np.zeros(total_years)
    component_depreciation = {}
    
    for component_name, component_capex in capex_breakdown.items():
        if component_capex > 0:
            component_depreciation_array = calculate_component_macrs_depreciation(
                component_name=component_name,
                component_capex=component_capex,
                construction_period_years=construction_period_years,
                project_lifetime_years=project_lifetime_years,
                macrs_config=macrs_config
            )
            
            component_depreciation[component_name] = component_depreciation_array
            total_depreciation += component_depreciation_array
            
            logger.debug(f"Added {component_name} depreciation: "
                        f"${np.sum(component_depreciation_array):,.0f} total")
    
    total_depreciation_sum = np.sum(total_depreciation)
    total_capex = sum(capex_breakdown.values())
    
    logger.info(f"Total MACRS depreciation across all components: ${total_depreciation_sum:,.0f}")
    logger.info(f"Total CAPEX: ${total_capex:,.0f}")
    logger.info(f"Depreciation coverage: {total_depreciation_sum/total_capex:.1%}")
    
    return total_depreciation, component_depreciation


def calculate_macrs_tax_benefit(
    depreciation_array: np.ndarray,
    tax_rate: float
) -> np.ndarray:
    """
    Calculate tax benefits from MACRS depreciation.
    
    Args:
        depreciation_array: Array of annual depreciation amounts
        tax_rate: Corporate tax rate
        
    Returns:
        Array of annual tax benefits (depreciation * tax_rate)
    """
    tax_benefits = depreciation_array * tax_rate
    total_tax_benefit = np.sum(tax_benefits)
    
    logger.info(f"Total MACRS tax benefit: ${total_tax_benefit:,.0f} "
               f"(${np.sum(depreciation_array):,.0f} depreciation × {tax_rate:.1%} tax rate)")
    
    return tax_benefits
