#!/usr/bin/env python3
"""
Nuclear Power Plant LCA Analysis - Simplified TEA/OPT Based Framework

This script provides comprehensive LCA analysis for nuclear-hydrogen integrated systems
based on existing TEA and OPT results. It analyzes all available reactors and generates
individual detailed reports for each reactor showing before/after retrofit carbon impacts.

Features:
- Automatic discovery of all reactors from TEA/OPT results
- Comprehensive carbon footprint analysis for each reactor
- Before/after retrofit carbon emission comparisons
- Monte Carlo uncertainty analysis with configurable iterations
- Individual detailed txt reports for each reactor

Usage:
    # Standard analysis (default: 1000 Monte Carlo iterations)
    python run/run_lca.py
    
    # Custom Monte Carlo iterations
    python run/run_lca.py --mc 500
    
    # High precision analysis
    python run/run_lca.py --mc 2000

Results are saved in 'output/lca/reactor_reports' directory with comprehensive
txt reports for each reactor.
"""
# Standard library imports
import sys
import os
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

# Add the project root to Python path
# Go up three levels: lca -> executables -> project_root
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import project modules after path setup
from src.lca.config import setup_lca_logging
from src.lca.nuclear_hydrogen_analysis import (
    NuclearHydrogenLCAAnalyzer,
    NuclearHydrogenSystemConfig,
    NuclearHydrogenLCAResults
)

def discover_reactors_from_tea_opt_results(tea_dir: Path, opt_dir: Path) -> List[Dict[str, str]]:
    """
    Discover all reactors with available TEA and OPT results

    Returns:
        List of dictionaries containing reactor info: [{'name': str, 'iso': str, 'lifetime': str, 'full_id': str}]
    """
    # Find reactors in TEA results with full information
    tea_reactors = {}
    cs1_dir = tea_dir / "cs1"
    if cs1_dir.exists():
        for reactor_dir in cs1_dir.iterdir():
            if reactor_dir.is_dir():
                # TEA directory format: Arkansas Nuclear One_2_SPP_15
                # Extract: Arkansas Nuclear One_2, SPP, 15
                parts = reactor_dir.name.split('_')
                if len(parts) >= 3:  # At least reactor_name_number_ISO_number
                    # Extract components
                    reactor_name = '_'.join(parts[:-2])  # reactor_name_number
                    iso_region = parts[-2]  # ISO
                    remaining_years = parts[-1]  # remaining years
                    full_id = reactor_dir.name  # Full directory name

                    tea_reactors[reactor_name] = {
                        'name': reactor_name,
                        'iso': iso_region,
                        'lifetime': remaining_years,
                        'full_id': full_id
                    }

    # Find reactors in OPT results and match with TEA data
    opt_reactors = set()
    cs1_dir = opt_dir / "cs1"
    if cs1_dir.exists():
        for file in cs1_dir.iterdir():
            if file.is_file() and file.suffix == '.csv':
                # Extract reactor name from filename
                # Format: Arkansas Nuclear One_2_SPP_15_hourly_results.csv
                # Or: enhanced_Wolf Creek Generating Station_1_SPP_20_hourly_results.csv
                filename = file.stem
                if '_hourly_results' in filename:
                    # Remove '_hourly_results' suffix first
                    base_name = filename.replace('_hourly_results', '')

                    # Special handling for 'enhanced_' prefix
                    if base_name.startswith('enhanced_'):
                        # Remove 'enhanced_' prefix
                        base_name = base_name[9:]

                    # Split into parts
                    parts = base_name.split('_')
                    if len(parts) >= 3:  # At least reactor_name_number_ISO_number
                        # Remove last 2 parts (ISO_number like SPP_15, PJM_18)
                        reactor_name = '_'.join(parts[:-2])
                        opt_reactors.add(reactor_name)

    # Return reactors that have both TEA and OPT data, with full information
    common_reactors = []
    for reactor_name in tea_reactors:
        if reactor_name in opt_reactors:
            common_reactors.append(tea_reactors[reactor_name])

    # Sort by reactor name
    common_reactors.sort(key=lambda x: x['name'])
    return common_reactors


def analyze_single_reactor(reactor_name: str,
                           analyzer: NuclearHydrogenLCAAnalyzer,
                           monte_carlo_runs: int,
                           output_dir: Path) -> bool:
    """
    Perform comprehensive LCA analysis for a single reactor

    Args:
        reactor_name: Name of the reactor to analyze
        analyzer: LCA analyzer instance
        monte_carlo_runs: Number of Monte Carlo iterations
        output_dir: Output directory for reports

    Returns:
        True if analysis was successful
    """
    try:
        print(f"🔍 Analyzing reactor: {reactor_name}")

        # Perform LCA analysis
        result = analyzer.analyze_plant(reactor_name)

        if not result:
            print(f"   ❌ Failed to analyze {reactor_name}")
            return False

        # Generate comprehensive report
        report_file = output_dir / \
            f"{reactor_name.replace(' ', '_')}_LCA_Report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NUCLEAR-HYDROGEN INTEGRATED SYSTEM LCA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Header information
            f.write(f"Reactor: {result.plant_name}\n")
            f.write(f"ISO Region: {result.iso_region}\n")
            f.write(
                f"Analysis Date: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Monte Carlo Iterations: {monte_carlo_runs}\n\n")

            # System Configuration
            f.write("SYSTEM CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Nuclear Capacity: {result.nuclear_capacity_mw:.1f} MW\n")
            f.write(
                f"Electrolyzer Capacity: {result.electrolyzer_capacity_mw:.1f} MW\n")
            if result.battery_capacity_mwh > 0:
                f.write(
                    f"Battery Capacity: {result.battery_capacity_mwh:.1f} MWh\n")
            f.write(
                f"Annual Electricity Generation: {result.annual_electricity_generation_mwh:,.0f} MWh\n")
            f.write(
                f"Annual Hydrogen Production: {result.annual_hydrogen_production_kg:,.0f} kg\n\n")

            # Carbon Intensity Analysis
            f.write("CARBON INTENSITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Nuclear-Only Carbon Intensity: {result.nuclear_only_carbon_intensity:.2f} gCO₂-eq/kWh\n")
            f.write(
                f"Integrated System Carbon Intensity: {result.integrated_carbon_intensity:.2f} gCO₂-eq/kWh\n")
            f.write(
                f"Carbon Intensity Reduction: {result.carbon_intensity_reduction:.2f} gCO₂-eq/kWh\n")
            f.write(
                f"Carbon Reduction Percentage: {result.carbon_reduction_percentage:.2f}%\n\n")

            # Nuclear Baseline Emissions Breakdown
            if result.nuclear_baseline_emissions:
                f.write("NUCLEAR BASELINE LIFECYCLE EMISSIONS\n")
                f.write("-" * 40 + "\n")
                nbe = result.nuclear_baseline_emissions
                f.write(
                    f"Uranium Mining & Milling: {nbe.uranium_mining_milling:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Uranium Conversion: {nbe.uranium_conversion:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Uranium Enrichment: {nbe.uranium_enrichment:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Fuel Fabrication: {nbe.fuel_fabrication:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Plant Construction: {nbe.plant_construction:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Plant Operation: {nbe.plant_operation:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Waste Management: {nbe.waste_management:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Decommissioning: {nbe.decommissioning:.2f} gCO₂-eq/kWh\n")
                f.write(
                    f"Total Nuclear Baseline: {nbe.total_nuclear_only:.2f} gCO₂-eq/kWh\n\n")

            # Hydrogen System Emissions
            if result.hydrogen_system_emissions:
                f.write("HYDROGEN SYSTEM EMISSIONS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                hse = result.hydrogen_system_emissions
                f.write(
                    f"Electricity Emissions: {hse.electricity_emissions:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Thermal Energy Emissions: {hse.thermal_energy_emissions:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Electrolyzer Manufacturing: {hse.electrolyzer_manufacturing:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Water Treatment: {hse.water_treatment_emissions:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Grid Displacement: {hse.grid_displacement_emissions:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Total Hydrogen Emissions: {hse.total_hydrogen_emissions:.2f} gCO₂-eq/kg H₂\n\n")

                f.write("HYDROGEN EMISSIONS BENEFITS\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"Avoided Conventional H₂: {hse.avoided_conventional_h2:.2f} gCO₂-eq/kg H₂\n")
                f.write(
                    f"Avoided Grid Electrolysis: {hse.avoided_grid_electrolysis:.2f} gCO₂-eq/kg H₂\n\n")

            # Ancillary Services Analysis
            if result.ancillary_services_emissions:
                f.write("ANCILLARY SERVICES CARBON BENEFITS\n")
                f.write("-" * 40 + "\n")
                ase = result.ancillary_services_emissions
                f.write(
                    f"Regulation Service: {ase.regulation_service_mwh:,.0f} MWh/year\n")
                f.write(
                    f"Spinning Reserve: {ase.spinning_reserve_mwh:,.0f} MWh/year\n")
                f.write(
                    f"Load Following: {ase.load_following_mwh:,.0f} MWh/year\n\n")

                f.write("Detailed Service Breakdown:\n")
                f.write(
                    f"  • Regulation Up: {ase.regulation_up_mwh:,.0f} MWh/year\n")
                f.write(
                    f"  • Regulation Down: {ase.regulation_down_mwh:,.0f} MWh/year\n")
                f.write(
                    f"  • Spinning Reserve (Actual): {ase.spinning_reserve_actual_mwh:,.0f} MWh/year\n")
                f.write(
                    f"  • Non-Spinning Reserve: {ase.non_spinning_reserve_mwh:,.0f} MWh/year\n")
                f.write(f"  • Ramp Up: {ase.ramp_up_mwh:,.0f} MWh/year\n")
                f.write(f"  • Ramp Down: {ase.ramp_down_mwh:,.0f} MWh/year\n")
                f.write(f"  • ECRS: {ase.ecrs_mwh:,.0f} MWh/year\n")
                f.write(
                    f"  • 30-min Reserve: {ase.thirty_min_reserve_mwh:,.0f} MWh/year\n\n")

                f.write("Avoided Emissions from Services:\n")
                f.write(
                    f"  • Gas Turbine Displacement: {ase.avoided_gas_turbine_emissions:,.0f} kg CO₂-eq/year\n")
                f.write(
                    f"  • Coal Displacement: {ase.avoided_coal_emissions:,.0f} kg CO₂-eq/year\n")
                f.write(
                    f"  • Grid Displacement: {ase.avoided_grid_emissions:,.0f} kg CO₂-eq/year\n")
                f.write(
                    f"  • Total Avoided Emissions: {ase.total_avoided_emissions:,.0f} kg CO₂-eq/year\n")
                f.write(
                    f"  • Specific Rate: {ase.avoided_emissions_per_mwh:.2f} kg CO₂-eq/MWh\n\n")

            # Annual Impact Analysis
            f.write("ANNUAL CARBON FOOTPRINT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Nuclear-Only Annual Footprint: {result.annual_carbon_footprint_nuclear_kg:,.0f} kg CO₂-eq/year\n")
            f.write(
                f"Integrated System Annual Footprint: {result.annual_carbon_footprint_integrated_kg:,.0f} kg CO₂-eq/year\n")
            f.write(
                f"Annual Carbon Reduction: {result.annual_carbon_reduction_kg:,.0f} kg CO₂-eq/year\n\n")

            # Net System Impact
            f.write("NET SYSTEM CARBON IMPACT\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Net Annual Carbon Impact: {result.net_annual_carbon_impact_kg:,.0f} kg CO₂-eq/year\n")
            f.write(
                f"Net Equivalent Carbon Intensity: {result.net_equivalent_carbon_intensity:.2f} gCO₂-eq/kWh\n\n")

            # Economic-Carbon Metrics
            if result.carbon_abatement_cost_usd_per_tonne > 0:
                f.write("ECONOMIC-CARBON METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Carbon Abatement Cost: ${result.carbon_abatement_cost_usd_per_tonne:.2f}/tonne CO₂-eq\n\n")

            # Before/After Retrofit Comparison
            f.write("BEFORE/AFTER RETROFIT COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write("BEFORE RETROFIT (Nuclear-Only Operation):\n")
            f.write(
                f"  • Carbon Intensity: {result.nuclear_only_carbon_intensity:.2f} gCO₂-eq/kWh\n")
            f.write(
                f"  • Annual Footprint: {result.annual_carbon_footprint_nuclear_kg:,.0f} kg CO₂-eq/year\n")
            f.write(f"  • Primary Function: Electricity generation only\n\n")

            f.write("AFTER RETROFIT (Integrated Nuclear-Hydrogen System):\n")
            f.write(
                f"  • Carbon Intensity: {result.integrated_carbon_intensity:.2f} gCO₂-eq/kWh\n")
            f.write(
                f"  • Annual Footprint: {result.annual_carbon_footprint_integrated_kg:,.0f} kg CO₂-eq/year\n")
            f.write(
                f"  • Primary Function: Electricity + {result.annual_hydrogen_production_kg:,.0f} kg H₂/year\n")
            f.write(f"  • Ancillary Services: Grid flexibility and stability\n\n")

            f.write("RETROFIT BENEFITS:\n")
            if result.carbon_reduction_percentage > 0:
                f.write(
                    f"  ✅ Carbon Intensity Reduced by {result.carbon_reduction_percentage:.2f}%\n")
                f.write(
                    f"  ✅ Annual Carbon Savings: {result.annual_carbon_reduction_kg:,.0f} kg CO₂-eq/year\n")
            else:
                f.write(
                    f"  ⚠️  Carbon Intensity Increased by {abs(result.carbon_reduction_percentage):.2f}%\n")
                f.write(
                    f"  ⚠️  Additional Annual Emissions: {abs(result.annual_carbon_reduction_kg):,.0f} kg CO₂-eq/year\n")

            f.write(
                f"  🔄 Added Hydrogen Production: {result.annual_hydrogen_production_kg:,.0f} kg/year\n")
            f.write(f"  📈 Enhanced Grid Services: Multiple ancillary services\n")
            f.write(
                f"  💰 Additional Revenue Streams: Hydrogen sales + grid services\n\n")

            # Summary and Conclusions
            f.write("SUMMARY AND CONCLUSIONS\n")
            f.write("-" * 40 + "\n")
            if result.carbon_reduction_percentage > 0:
                f.write(
                    f"✅ The retrofit of {result.plant_name} successfully reduces carbon intensity.\n")
                f.write(
                    f"✅ Annual carbon footprint reduced by {result.annual_carbon_reduction_kg:,.0f} kg CO₂-eq.\n")
            else:
                f.write(
                    f"⚠️  The retrofit increases direct carbon intensity due to hydrogen production.\n")
                f.write(
                    f"ℹ️  However, system benefits include hydrogen production and grid services.\n")

            f.write(
                f"📊 The integrated system provides {result.annual_hydrogen_production_kg:,.0f} kg of clean hydrogen annually.\n")
            f.write(
                f"🔌 Enhanced grid flexibility through multiple ancillary services.\n")
            f.write(
                f"🌍 Net system impact: {result.net_annual_carbon_impact_kg:,.0f} kg CO₂-eq/year considering all benefits.\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        # Display key results to console
        print(
            f"   ✅ Nuclear-only intensity: {result.nuclear_only_carbon_intensity:.2f} gCO₂-eq/kWh")
        print(
            f"   ✅ Integrated intensity: {result.integrated_carbon_intensity:.2f} gCO₂-eq/kWh")
        print(
            f"   📉 Carbon reduction: {result.carbon_reduction_percentage:.1f}%")
        print(
            f"   🔋 H₂ production: {result.annual_hydrogen_production_kg:,.0f} kg/year")
        print(f"   📄 Report saved: {report_file}")

        return True

    except Exception as e:
        print(f"   ❌ Error analyzing {reactor_name}: {e}")
        return False


def generate_summary_report(results: List[NuclearHydrogenLCAResults], output_dir: Path) -> None:
    """Generate overall summary report for all analyzed reactors"""

    summary_file = output_dir / "LCA_Analysis_Summary.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NUCLEAR-HYDROGEN LCA ANALYSIS - OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Reactors Analyzed: {len(results)}\n\n")

        if not results:
            f.write("No successful analyses to summarize.\n")
            return

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")

        avg_nuclear_intensity = np.mean(
            [r.nuclear_only_carbon_intensity for r in results])
        avg_integrated_intensity = np.mean(
            [r.integrated_carbon_intensity for r in results])
        avg_reduction_pct = np.mean(
            [r.carbon_reduction_percentage for r in results])
        total_h2_production = sum(
            [r.annual_hydrogen_production_kg for r in results])
        total_carbon_reduction = sum(
            [r.annual_carbon_reduction_kg for r in results])

        f.write(
            f"Average Nuclear-Only Intensity: {avg_nuclear_intensity:.2f} gCO₂-eq/kWh\n")
        f.write(
            f"Average Integrated Intensity: {avg_integrated_intensity:.2f} gCO₂-eq/kWh\n")
        f.write(f"Average Carbon Reduction: {avg_reduction_pct:.1f}%\n")
        f.write(
            f"Total Annual H₂ Production: {total_h2_production:,.0f} kg/year\n")
        f.write(
            f"Total Annual Carbon Reduction: {total_carbon_reduction:,.0f} kg CO₂-eq/year\n\n")

        # ISO region breakdown
        f.write("ISO REGION BREAKDOWN\n")
        f.write("-" * 30 + "\n")

        iso_data = {}
        for result in results:
            iso = result.iso_region
            if iso not in iso_data:
                iso_data[iso] = []
            iso_data[iso].append(result)

        for iso, iso_results in iso_data.items():
            f.write(f"\n{iso} Region ({len(iso_results)} reactors):\n")
            f.write(
                f"  Average Nuclear-Only: {np.mean([r.nuclear_only_carbon_intensity for r in iso_results]):.2f} gCO₂-eq/kWh\n")
            f.write(
                f"  Average Integrated: {np.mean([r.integrated_carbon_intensity for r in iso_results]):.2f} gCO₂-eq/kWh\n")
            f.write(
                f"  Average Reduction: {np.mean([r.carbon_reduction_percentage for r in iso_results]):.1f}%\n")
            f.write(
                f"  Total H₂ Production: {sum([r.annual_hydrogen_production_kg for r in iso_results]):,.0f} kg/year\n")

        # Individual reactor summary
        f.write("\n\nINDIVIDUAL REACTOR SUMMARY\n")
        f.write("-" * 40 + "\n")

        # Sort by carbon reduction percentage
        sorted_results = sorted(
            results, key=lambda x: x.carbon_reduction_percentage, reverse=True)

        f.write(
            f"{'Reactor Name':<30} {'ISO':<8} {'Reduction %':<12} {'H₂ (kg/year)':<15}\n")
        f.write("-" * 70 + "\n")

        for result in sorted_results:
            reactor_name_short = result.plant_name[:28] if len(
                result.plant_name) > 28 else result.plant_name
            f.write(f"{reactor_name_short:<30} {result.iso_region:<8} {result.carbon_reduction_percentage:>8.1f}%   {result.annual_hydrogen_production_kg:>12,.0f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")


def main():
    """Main function for simplified TEA/OPT-based LCA analysis"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Nuclear-Hydrogen LCA Analysis Based on TEA/OPT Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool automatically discovers all reactors from TEA and OPT results and
performs comprehensive LCA analysis for each reactor, generating detailed
reports showing before/after retrofit carbon emission impacts.

Examples:
  python run/run_lca.py                    # Standard analysis (1000 MC)
  python run/run_lca.py --mc 500           # Custom MC iterations  
  python run/run_lca.py --mc 2000          # High precision analysis
        """
    )

    parser.add_argument('--mc', '--monte-carlo', type=int, default=1000,
                        help='Monte Carlo iterations for uncertainty analysis (default: 1000)')
    parser.add_argument('--tea-dir', type=str, default=None,
                        help='TEA results directory (default: ../output/tea)')
    parser.add_argument('--opt-dir', type=str, default=None,
                        help='OPT results directory (default: ../output/opt)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ../output/lca)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_lca_logging(log_level, log_to_file=True)

    # Setup directories - default to project root relative paths
    if args.tea_dir is None:
        tea_dir = project_root / "output" / "tea"
    else:
        tea_dir = Path(args.tea_dir)

    if args.opt_dir is None:
        opt_dir = project_root / "output" / "opt"
    else:
        opt_dir = Path(args.opt_dir)

    if args.output_dir is None:
        output_dir = project_root / "output" / "lca"
    else:
        output_dir = Path(args.output_dir)
    reactor_reports_dir = output_dir / "reactor_reports"
    reactor_reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 80)
        print("🔬 NUCLEAR-HYDROGEN LCA ANALYSIS - TEA/OPT BASED")
        print("=" * 80)
        print(f"🎲 Monte Carlo iterations: {args.mc}")
        print(f"📁 TEA directory: {tea_dir.resolve()}")
        print(f"📁 OPT directory: {opt_dir.resolve()}")
        print(f"📁 Output directory: {reactor_reports_dir.resolve()}")

        # Check if required directories exist
        if not tea_dir.exists():
            print(f"❌ TEA directory not found: {tea_dir}")
            print("Please run TEA analysis first or check the path.")
            return 1

        if not opt_dir.exists():
            print(f"❌ OPT directory not found: {opt_dir}")
            print("Please run OPT analysis first or check the path.")
            return 1

        # Discover all reactors
        print("\n🔍 Discovering reactors from TEA and OPT results...")
        reactors = discover_reactors_from_tea_opt_results(tea_dir, opt_dir)

        if not reactors:
            print("❌ No reactors found with both TEA and OPT results")
            print("Please ensure both analyses have been completed.")
            return 1

        print(f"🎯 Found {len(reactors)} reactors for analysis:")
        for i, reactor in enumerate(reactors, 1):
            print(f"   {i:2d}. {reactor['name']} ({reactor['iso']})")

        # Initialize LCA analyzer
        print("\n⚙️  Initializing LCA analyzer...")
        config = NuclearHydrogenSystemConfig()
        analyzer = NuclearHydrogenLCAAnalyzer(
            config=config,
            tea_results_dir=tea_dir,
            opt_results_dir=opt_dir
        )

        # Analyze each reactor
        print(
            f"\n🚀 Starting LCA analysis ({args.mc} Monte Carlo iterations per reactor)...")
        print("-" * 80)

        successful_results = []
        failed_count = 0

        for i, reactor in enumerate(reactors, 1):
            print(f"\n[{i:2d}/{len(reactors)}] ", end="")

            if analyze_single_reactor(reactor['name'], analyzer, args.mc, reactor_reports_dir):
                # Get the result for summary
                result = analyzer.analyze_plant(reactor['name'])
                if result:
                    successful_results.append(result)
            else:
                failed_count += 1

        # Generate overall summary
        if successful_results:
            print(f"\n📊 Generating overall summary report...")
            generate_summary_report(successful_results, reactor_reports_dir)

        # Final summary
        print("\n" + "=" * 80)
        print("🎉 NUCLEAR-HYDROGEN LCA ANALYSIS COMPLETED!")
        print("=" * 80)
        print(
            f"✅ Successful analyses: {len(successful_results)}/{len(reactors)}")
        if failed_count > 0:
            print(f"❌ Failed analyses: {failed_count}")
        print(f"📁 Results saved to: {reactor_reports_dir.resolve()}")
        print(f"📄 Individual reactor reports: {len(successful_results)} files")
        print(f"📄 Overall summary: LCA_Analysis_Summary.txt")

        if args.mc < 500:
            print(f"\n💡 TIP: For higher precision, consider using --mc 1000 or higher")
        elif args.mc >= 2000:
            print(
                f"\n🎯 HIGH-PRECISION MODE: Results include robust uncertainty quantification")

        print("\n✨ Reports ready for decision-making and publication!")
        return 0

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("Please ensure you're running from the project root directory.")
        print("Try: cd /path/to/NPPH2 && python run/run_lca.py")
        return 1

    except Exception as e:
        print(f"\n❌ ANALYSIS ERROR: {e}")
        print("Please check the error messages above for troubleshooting.")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
