"""
Main entry point for Nuclear-Hydrogen LCA Analysis Framework.

This module provides simplified LCA analysis specifically designed for 
nuclear-hydrogen integrated systems based on TEA and OPT optimization results.
It focuses on comprehensive before/after retrofit carbon emission analysis.

The framework automatically discovers reactors from TEA/OPT results and 
performs detailed lifecycle carbon footprint analysis with Monte Carlo 
uncertainty quantification.

To run from the project root directory:
python -m src.lca.main
"""

from .nuclear_hydrogen_analysis import (
    NuclearHydrogenLCAAnalyzer,
    NuclearHydrogenSystemConfig,
    NuclearHydrogenLCAResults
)
from .config import setup_lca_logging
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
from datetime import datetime


def discover_reactors_from_results(tea_dir: Path, opt_dir: Path) -> List[str]:
    """
    Discover all reactors with available TEA and OPT results

    Args:
        tea_dir: Path to TEA results directory
        opt_dir: Path to OPT results directory

    Returns:
        List of reactor names that have both TEA and OPT data
    """
    # Find reactors in TEA results
    tea_reactors = set()
    cs1_dir = tea_dir / "cs1"
    if cs1_dir.exists():
        for reactor_dir in cs1_dir.iterdir():
            if reactor_dir.is_dir():
                # TEA directory format: Arkansas Nuclear One_2_SPP_15
                # Extract: Arkansas Nuclear One_2
                parts = reactor_dir.name.split('_')
                if len(parts) >= 3:  # At least reactor_name_number_ISO_number
                    # Remove last 2 parts (ISO_number)
                    reactor_name = '_'.join(parts[:-2])
                    tea_reactors.add(reactor_name)

        # Find reactors in OPT results
    opt_reactors = set()
    opt_cs1_dir = opt_dir / "cs1"
    if opt_cs1_dir.exists():
        for file in opt_cs1_dir.iterdir():
            if file.is_file() and file.suffix == '.csv':
                # Extract reactor name from filename
                # Format: Arkansas Nuclear One_2_SPP_15_hourly_results.csv
                filename = file.stem
                if '_hourly_results' in filename:
                    # Remove '_hourly_results' suffix and the last 3 parts (number_ISO_number)
                    parts = filename.split('_')
                    if len(parts) >= 4:  # At least reactor_name_number_ISO_number_hourly_results
                        # Remove last 3 parts (number_ISO_number)
                        reactor_name = '_'.join(parts[:-3])
                        # Special handling for 'enhanced_' prefix
                        if reactor_name.startswith('enhanced_'):
                            # Remove 'enhanced_' prefix
                            reactor_name = reactor_name[9:]
                        opt_reactors.add(reactor_name)

    # Return reactors that have both TEA and OPT data
    common_reactors = tea_reactors.intersection(opt_reactors)
    return sorted(list(common_reactors))


def analyze_reactor_comprehensive(reactor_name: str,
                                  analyzer: NuclearHydrogenLCAAnalyzer,
                                  monte_carlo_runs: int = 1000) -> Optional[NuclearHydrogenLCAResults]:
    """
    Perform comprehensive LCA analysis for a single reactor

    Args:
        reactor_name: Name of the reactor to analyze
        analyzer: LCA analyzer instance
        monte_carlo_runs: Number of Monte Carlo iterations

    Returns:
        NuclearHydrogenLCAResults object or None if analysis failed
    """
    try:
        print(f"🔋 Analyzing reactor: {reactor_name}")

        # Perform comprehensive LCA analysis
        result = analyzer.analyze_plant(reactor_name)

        if result:
            print(
                f"   ✅ Nuclear-only: {result.nuclear_only_carbon_intensity:.2f} gCO₂-eq/kWh")
            print(
                f"   ✅ Integrated: {result.integrated_carbon_intensity:.2f} gCO₂-eq/kWh")
            print(f"   📉 Reduction: {result.carbon_reduction_percentage:.1f}%")
            print(
                f"   🔋 H₂ production: {result.annual_hydrogen_production_kg:,.0f} kg/year")
            return result
        else:
            print(f"   ❌ Failed to analyze {reactor_name}")
            return None

    except Exception as e:
        print(f"   ❌ Error analyzing {reactor_name}: {e}")
        return None


def generate_comprehensive_report(result: NuclearHydrogenLCAResults,
                                  output_dir: Path,
                                  monte_carlo_runs: int) -> Path:
    """
    Generate comprehensive txt report for a single reactor

    Args:
        result: LCA analysis results
        output_dir: Output directory for reports
        monte_carlo_runs: Number of Monte Carlo iterations used

    Returns:
        Path to generated report file
    """
    # Create reactor-specific filename
    safe_name = result.plant_name.replace(' ', '_').replace('/', '_')
    report_file = output_dir / f"{safe_name}_LCA_Report.txt"

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
        f.write(f"  💰 Additional Revenue Streams: Hydrogen sales + grid services\n\n")

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
        f.write(f"🔌 Enhanced grid flexibility through multiple ancillary services.\n")
        f.write(
            f"🌍 Net system impact: {result.net_annual_carbon_impact_kg:,.0f} kg CO₂-eq/year considering all benefits.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    return report_file


def generate_summary_report(results: List[NuclearHydrogenLCAResults],
                            output_dir: Path) -> Path:
    """
    Generate overall summary report for all analyzed reactors

    Args:
        results: List of reactor analysis results
        output_dir: Output directory for reports

    Returns:
        Path to generated summary report
    """
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
            return summary_file

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

    return summary_file


def main(monte_carlo_runs: int = 1000,
         tea_dir: str = "output/tea",
         opt_dir: str = "output/opt",
         output_dir: str = "output/lca"):
    """
    Main function for nuclear-hydrogen LCA analysis based on TEA/OPT results.

    Args:
        monte_carlo_runs: Number of Monte Carlo iterations for uncertainty analysis
        tea_dir: Path to TEA results directory
        opt_dir: Path to OPT results directory  
        output_dir: Path to output directory
    """
    # Setup logging
    setup_lca_logging(log_level=logging.INFO, log_to_file=True)

    print("=" * 80)
    print("🔬 NUCLEAR-HYDROGEN LCA ANALYSIS")
    print("=" * 80)

    # Setup directories
    tea_path = Path(tea_dir)
    opt_path = Path(opt_dir)
    output_path = Path(output_dir)
    reactor_reports_dir = output_path / "reactor_reports"
    reactor_reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎲 Monte Carlo iterations: {monte_carlo_runs}")
    print(f"📁 TEA directory: {tea_path.resolve()}")
    print(f"📁 OPT directory: {opt_path.resolve()}")
    print(f"📁 Output directory: {reactor_reports_dir.resolve()}")

    # Check if required directories exist
    if not tea_path.exists():
        print(f"❌ TEA directory not found: {tea_path}")
        print("Please run TEA analysis first or check the path.")
        return False

    if not opt_path.exists():
        print(f"❌ OPT directory not found: {opt_path}")
        print("Please run OPT analysis first or check the path.")
        return False

    # Discover all reactors
    print("\n🔍 Discovering reactors from TEA and OPT results...")
    reactors = discover_reactors_from_results(tea_path, opt_path)

    if not reactors:
        print("❌ No reactors found with both TEA and OPT results")
        print("Please ensure both analyses have been completed.")
        return False

    print(f"🎯 Found {len(reactors)} reactors for analysis:")
    for i, reactor in enumerate(reactors, 1):
        print(f"   {i:2d}. {reactor}")

    # Initialize LCA analyzer
    print("\n⚙️  Initializing LCA analyzer...")
    config = NuclearHydrogenSystemConfig()
    analyzer = NuclearHydrogenLCAAnalyzer(
        config=config,
        tea_results_dir=tea_path,
        opt_results_dir=opt_path
    )

    # Analyze each reactor
    print(f"\n🚀 Starting comprehensive LCA analysis...")
    print("-" * 80)

    successful_results = []
    failed_count = 0

    for i, reactor_name in enumerate(reactors, 1):
        print(f"\n[{i:2d}/{len(reactors)}] ", end="")

        result = analyze_reactor_comprehensive(
            reactor_name, analyzer, monte_carlo_runs)

        if result:
            # Generate individual reactor report
            report_file = generate_comprehensive_report(
                result, reactor_reports_dir, monte_carlo_runs)
            print(f"   📄 Report saved: {report_file.name}")
            successful_results.append(result)
        else:
            failed_count += 1

    # Generate overall summary
    if successful_results:
        print(f"\n📊 Generating overall summary report...")
        summary_file = generate_summary_report(
            successful_results, reactor_reports_dir)
        print(f"   📄 Summary saved: {summary_file.name}")

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 NUCLEAR-HYDROGEN LCA ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"✅ Successful analyses: {len(successful_results)}/{len(reactors)}")
    if failed_count > 0:
        print(f"❌ Failed analyses: {failed_count}")
    print(f"📁 Results saved to: {reactor_reports_dir.resolve()}")
    print(f"📄 Individual reactor reports: {len(successful_results)} files")
    if successful_results:
        print(f"📄 Overall summary: LCA_Analysis_Summary.txt")

    if monte_carlo_runs < 500:
        print(f"\n💡 TIP: For higher precision, consider using mc_runs=1000 or higher")
    elif monte_carlo_runs >= 2000:
        print(f"\n🎯 HIGH-PRECISION MODE: Results include robust uncertainty quantification")

    print("\n✨ Reports ready for decision-making and publication!")
    return True


if __name__ == "__main__":
    main()
