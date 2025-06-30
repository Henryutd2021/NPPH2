"""
Nuclear Power Plant Life Cycle Assessment (LCA) Framework

This package provides comprehensive tools for conducting life cycle assessment
of nuclear power plants, including standalone nuclear plants and nuclear-hydrogen
integrated systems.

Main Components:
- NuclearLCACalculator: Core calculation engine
- LCAReporter: Report generation in multiple formats
- Configuration: Emission factors and methodology settings

Usage:
    from src.lca import NuclearLCACalculator, LCAReporter
    from src.lca.models import NuclearPlantParameters, ReactorType
    
    # Create plant parameters
    plant = NuclearPlantParameters(...)
    
    # Run analysis
    calculator = NuclearLCACalculator()
    results = calculator.create_comprehensive_analysis(plant)
    
    # Generate reports
    reporter = LCAReporter()
    reporter.generate_comprehensive_report(results)
"""

from .calculator import NuclearLCACalculator
from .reporting import LCAReporter
from .models import (
    NuclearPlantParameters, HydrogenProductionData, ReactorType,
    LifecycleEmissions, LCAResults, ComprehensiveLCAResults
)
from .config import config

__all__ = [
    "NuclearLCACalculator",
    "LCAReporter",
    "NuclearPlantParameters",
    "HydrogenProductionData",
    "ReactorType",
    "LifecycleEmissions",
    "LCAResults",
    "ComprehensiveLCAResults",
    "config"
]
