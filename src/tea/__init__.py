"""
TEA (Technical Economic Analysis) module for nuclear-hydrogen optimization.

The main TEA entry points have been moved to the run directory:
- run/tea_main.py: ISO-level TEA analysis
- run/tea_cs1.py: Reactor-specific TEA analysis
"""

# Import key functions that other modules might need
from .calculations import (
    calculate_annual_metrics,
    calculate_cash_flows,
    calculate_financial_metrics,
    calculate_lcoh_breakdown,
    calculate_lcos_breakdown,
    calculate_incremental_metrics
)
from .nuclear_calculations import (
    calculate_greenfield_nuclear_hydrogen_system,
    calculate_lifecycle_comparison_analysis,
    calculate_greenfield_nuclear_hydrogen_with_tax_incentives,
    calculate_nuclear_baseline_financial_analysis
)
from .reporting import plot_results, generate_report
from .summary_reporting import generate_comprehensive_tea_summary_report
from .data_loader import load_tea_sys_params, load_hourly_results
from .utils import setup_logging
from .tea_engine import TEAEngine, run_complete_tea_analysis

__all__ = [
    # Calculations
    'calculate_annual_metrics',
    'calculate_cash_flows',
    'calculate_financial_metrics',
    'calculate_lcoh_breakdown',
    'calculate_incremental_metrics',
    # Nuclear calculations
    'calculate_greenfield_nuclear_hydrogen_system',
    'calculate_lifecycle_comparison_analysis',
    'calculate_greenfield_nuclear_hydrogen_with_tax_incentives',
    'calculate_nuclear_baseline_financial_analysis',
    # Reporting
    'plot_results',
    'generate_report',
    'generate_comprehensive_tea_summary_report',
    # Data loading
    'load_tea_sys_params',
    'load_hourly_results',
    # Utils
    'setup_logging',
    # TEA Engine
    'TEAEngine',
    'run_complete_tea_analysis',
]
