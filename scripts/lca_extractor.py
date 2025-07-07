#!/usr/bin/env python3
"""
Nuclear Power Plant LCA Results Extraction to CSV

This script extracts key LCA analysis results from all 42 reactor report files
and compiles them into a comprehensive CSV file for further analysis.

Features:
- Automatic discovery of all reactor LCA report files
- Extraction of key metrics from each report
- Comprehensive CSV output with all major LCA indicators
- Error handling for missing or corrupted report files

Usage:
    python extract_lca_results_to_csv.py

Output:
    - nuclear_lca_results_comprehensive.csv: Complete LCA results for all reactors
    - extraction_log.txt: Log of extraction process and any issues
"""

# Standard library imports
# from src.logger_utils.main import setup_logger  # Not needed, using standard logging
import sys
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import project modules after path setup


class LCAResultsExtractor:
    """Extract LCA results from reactor report files to CSV"""

    def __init__(self, reports_dir: Path, output_dir: Path = None):
        """
        Initialize the extractor

        Args:
            reports_dir: Directory containing LCA report files
            output_dir: Output directory for CSV files (default: current directory)
        """
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir) if output_dir else Path(".")

        # Setup simple logging
        self.logger = self._setup_logger()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup simple file and console logger"""
        logger = logging.getLogger("lca_extractor")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler
        log_file = Path("../logs/lca_extraction.log")
        file_handler = logging.FileHandler(
            log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def discover_report_files(self) -> List[Path]:
        """
        Discover all LCA report files in the reports directory

        Returns:
            List of paths to LCA report files
        """
        if not self.reports_dir.exists():
            self.logger.error(
                f"Reports directory does not exist: {self.reports_dir}")
            return []

        # Find all LCA report files
        report_files = list(self.reports_dir.glob("*_LCA_Report.txt"))

        # Filter out summary files
        report_files = [f for f in report_files if "Summary" not in f.name]

        self.logger.info(f"Found {len(report_files)} LCA report files")
        return sorted(report_files)

    def extract_value_from_line(self, line: str, pattern: str) -> Optional[float]:
        """
        Extract numerical value from a text line using regex pattern

        Args:
            line: Text line to search
            pattern: Regex pattern to match

        Returns:
            Extracted float value or None if not found
        """
        match = re.search(pattern, line)
        if match:
            try:
                # Remove commas and convert to float
                value_str = match.group(1).replace(',', '')
                return float(value_str)
            except (ValueError, IndexError):
                return None
        return None

    def parse_report_file(self, file_path: Path) -> Dict:
        """
        Parse a single LCA report file and extract key metrics

        Args:
            file_path: Path to the report file

        Returns:
            Dictionary with extracted metrics
        """
        self.logger.info(f"Parsing report: {file_path.name}")

        # Initialize result dictionary with default values
        result = {
            'reactor_name': '',
            'iso_region': '',
            'analysis_date': '',
            'monte_carlo_iterations': 0,

            # System Configuration
            'nuclear_capacity_mw': 0.0,
            'electrolyzer_capacity_mw': 0.0,
            'annual_electricity_generation_mwh': 0.0,
            'annual_hydrogen_production_kg': 0.0,

            # Carbon Intensity Analysis
            'nuclear_only_carbon_intensity': 0.0,
            'integrated_carbon_intensity': 0.0,
            'carbon_intensity_reduction': 0.0,
            'carbon_reduction_percentage': 0.0,

            # Nuclear Baseline Lifecycle Emissions
            'uranium_mining_milling': 0.0,
            'uranium_conversion': 0.0,
            'uranium_enrichment': 0.0,
            'fuel_fabrication': 0.0,
            'plant_construction': 0.0,
            'plant_operation': 0.0,
            'waste_management': 0.0,
            'decommissioning': 0.0,
            'total_nuclear_baseline': 0.0,

            # Hydrogen System Emissions
            'h2_electricity_emissions': 0.0,
            'h2_thermal_energy_emissions': 0.0,
            'h2_electrolyzer_manufacturing': 0.0,
            'h2_water_treatment': 0.0,
            'h2_grid_displacement': 0.0,
            'h2_total_emissions': 0.0,

            # Hydrogen Benefits
            'h2_avoided_conventional': 0.0,
            'h2_avoided_grid_electrolysis': 0.0,

            # Ancillary Services
            'as_regulation_service_mwh': 0.0,
            'as_spinning_reserve_mwh': 0.0,
            'as_load_following_mwh': 0.0,
            'as_total_avoided_emissions_kg': 0.0,
            'as_specific_rate': 0.0,

            # Annual Carbon Footprint
            'nuclear_only_annual_footprint_kg': 0.0,
            'integrated_annual_footprint_kg': 0.0,
            'annual_carbon_reduction_kg': 0.0,

            # Net System Impact
            'net_annual_carbon_impact_kg': 0.0,
            'net_equivalent_carbon_intensity': 0.0,
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Parse each line for relevant data
            for line in lines:
                line = line.strip()

                # Basic information
                if line.startswith('Reactor:'):
                    result['reactor_name'] = line.split(':', 1)[1].strip()
                elif line.startswith('ISO Region:'):
                    result['iso_region'] = line.split(':', 1)[1].strip()
                elif line.startswith('Analysis Date:'):
                    result['analysis_date'] = line.split(':', 1)[1].strip()
                elif line.startswith('Monte Carlo Iterations:'):
                    value = self.extract_value_from_line(line, r':\s*(\d+)')
                    if value is not None:
                        result['monte_carlo_iterations'] = int(value)

                # System Configuration
                elif 'Nuclear Capacity:' in line and 'MW' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MW')
                    if value is not None:
                        result['nuclear_capacity_mw'] = value
                elif 'Electrolyzer Capacity:' in line and 'MW' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MW')
                    if value is not None:
                        result['electrolyzer_capacity_mw'] = value
                elif 'Annual Electricity Generation:' in line and 'MWh' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MWh')
                    if value is not None:
                        result['annual_electricity_generation_mwh'] = value
                elif 'Annual Hydrogen Production:' in line and 'kg' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*kg')
                    if value is not None:
                        result['annual_hydrogen_production_kg'] = value

                # Carbon Intensity Analysis
                elif 'Nuclear-Only Carbon Intensity:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['nuclear_only_carbon_intensity'] = value
                elif 'Integrated System Carbon Intensity:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['integrated_carbon_intensity'] = value
                elif 'Carbon Intensity Reduction:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['carbon_intensity_reduction'] = value
                elif 'Carbon Reduction Percentage:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)%')
                    if value is not None:
                        result['carbon_reduction_percentage'] = value

                # Nuclear Baseline Lifecycle Emissions
                elif 'Uranium Mining & Milling:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['uranium_mining_milling'] = value
                elif 'Uranium Conversion:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['uranium_conversion'] = value
                elif 'Uranium Enrichment:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['uranium_enrichment'] = value
                elif 'Fuel Fabrication:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['fuel_fabrication'] = value
                elif 'Plant Construction:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['plant_construction'] = value
                elif 'Plant Operation:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['plant_operation'] = value
                elif 'Waste Management:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['waste_management'] = value
                elif 'Decommissioning:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['decommissioning'] = value
                elif 'Total Nuclear Baseline:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['total_nuclear_baseline'] = value

                # Hydrogen System Emissions
                elif 'Electricity Emissions:' in line and 'gCO₂-eq/kg H₂' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_electricity_emissions'] = value
                elif 'Thermal Energy Emissions:' in line and 'gCO₂-eq/kg H₂' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_thermal_energy_emissions'] = value
                elif 'Electrolyzer Manufacturing:' in line and 'gCO₂-eq/kg H₂' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_electrolyzer_manufacturing'] = value
                elif 'Water Treatment:' in line and 'gCO₂-eq/kg H₂' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_water_treatment'] = value
                elif 'Grid Displacement:' in line and 'gCO₂-eq/kg H₂' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_grid_displacement'] = value
                elif 'Total Hydrogen Emissions:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_total_emissions'] = value

                # Hydrogen Benefits
                elif 'Avoided Conventional H₂:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_avoided_conventional'] = value
                elif 'Avoided Grid Electrolysis:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['h2_avoided_grid_electrolysis'] = value

                # Ancillary Services
                elif 'Regulation Service:' in line and 'MWh/year' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MWh')
                    if value is not None:
                        result['as_regulation_service_mwh'] = value
                elif 'Spinning Reserve:' in line and 'MWh/year' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MWh')
                    if value is not None:
                        result['as_spinning_reserve_mwh'] = value
                elif 'Load Following:' in line and 'MWh/year' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*MWh')
                    if value is not None:
                        result['as_load_following_mwh'] = value
                elif 'Total Avoided Emissions:' in line and 'kg CO₂-eq/year' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*kg')
                    if value is not None:
                        result['as_total_avoided_emissions_kg'] = value
                elif 'Specific Rate:' in line and 'kg CO₂-eq/MWh' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*kg')
                    if value is not None:
                        result['as_specific_rate'] = value

                # Annual Carbon Footprint
                elif 'Nuclear-Only Annual Footprint:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*kg')
                    if value is not None:
                        result['nuclear_only_annual_footprint_kg'] = value
                elif 'Integrated System Annual Footprint:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.]+)\s*kg')
                    if value is not None:
                        result['integrated_annual_footprint_kg'] = value
                elif 'Annual Carbon Reduction:' in line and 'kg CO₂-eq/year' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*kg')
                    if value is not None:
                        result['annual_carbon_reduction_kg'] = value

                # Net System Impact
                elif 'Net Annual Carbon Impact:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*kg')
                    if value is not None:
                        result['net_annual_carbon_impact_kg'] = value
                elif 'Net Equivalent Carbon Intensity:' in line:
                    value = self.extract_value_from_line(
                        line, r':\s*([\d,\.\-]+)\s*gCO')
                    if value is not None:
                        result['net_equivalent_carbon_intensity'] = value

            # Extract reactor name from filename if not found in content
            if not result['reactor_name']:
                # Remove '_LCA_Report.txt' and replace underscores with spaces
                reactor_name = file_path.stem.replace(
                    '_LCA_Report', '').replace('_', ' ')
                result['reactor_name'] = reactor_name

            self.logger.info(f"Successfully parsed {file_path.name}")
            return result

        except Exception as e:
            self.logger.error(f"Error parsing {file_path.name}: {e}")
            # Return empty result with reactor name from filename
            result['reactor_name'] = file_path.stem.replace(
                '_LCA_Report', '').replace('_', ' ')
            return result

    def extract_all_results(self) -> pd.DataFrame:
        """
        Extract results from all LCA report files

        Returns:
            DataFrame with all extracted results
        """
        report_files = self.discover_report_files()

        if not report_files:
            self.logger.error("No report files found to process")
            return pd.DataFrame()

        self.logger.info(
            f"Starting extraction from {len(report_files)} report files")

        results = []
        successful_extractions = 0

        for file_path in report_files:
            try:
                result = self.parse_report_file(file_path)
                results.append(result)
                if result['reactor_name']:  # Check if extraction was successful
                    successful_extractions += 1
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                continue

        self.logger.info(
            f"Successfully extracted data from {successful_extractions}/{len(report_files)} files")

        if not results:
            self.logger.error("No results extracted from any files")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(results)

        # Sort by reactor name for consistent ordering
        df = df.sort_values('reactor_name').reset_index(drop=True)

        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str = "nuclear_results.csv") -> Path:
        """
        Save extracted results to CSV file

        Args:
            df: DataFrame with extracted results
            filename: Output filename

        Returns:
            Path to saved CSV file
        """
        output_path = self.output_dir / filename

        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            self.logger.info(f"Results saved to {output_path}")

            # Create summary statistics
            self._create_summary_report(df)

            return output_path

        except Exception as e:
            self.logger.error(f"Error saving CSV file: {e}")
            raise

    def _create_summary_report(self, df: pd.DataFrame):
        """Create a summary report of the extraction results"""
        summary_path = Path("../logs/lca_summary.txt")

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("LCA RESULTS EXTRACTION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Reactors Processed: {len(df)}\n\n")

                if len(df) > 0:
                    # Basic statistics
                    f.write("BASIC STATISTICS\n")
                    f.write("-" * 20 + "\n")
                    f.write(
                        f"Reactors with Hydrogen Production: {(df['annual_hydrogen_production_kg'] > 0).sum()}\n")
                    f.write(
                        f"Average Nuclear Capacity: {df['nuclear_capacity_mw'].mean():.1f} MW\n")
                    f.write(
                        f"Average Nuclear Carbon Intensity: {df['nuclear_only_carbon_intensity'].mean():.2f} gCO₂-eq/kWh\n")
                    f.write(
                        f"Average Integrated Carbon Intensity: {df['integrated_carbon_intensity'].mean():.2f} gCO₂-eq/kWh\n\n")

                    # ISO region breakdown
                    f.write("ISO REGION BREAKDOWN\n")
                    f.write("-" * 20 + "\n")
                    iso_counts = df['iso_region'].value_counts()
                    for iso, count in iso_counts.items():
                        f.write(f"{iso}: {count} reactors\n")
                    f.write("\n")

                    # Top performers
                    f.write("TOP 5 PERFORMERS (by carbon reduction %)\n")
                    f.write("-" * 40 + "\n")
                    top_performers = df.nlargest(
                        5, 'carbon_reduction_percentage')
                    for _, row in top_performers.iterrows():
                        f.write(
                            f"{row['reactor_name']}: {row['carbon_reduction_percentage']:.1f}%\n")

                self.logger.info(f"Summary report saved to {summary_path}")

        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")


def main():
    """Main execution function"""
    print("🔍 Nuclear LCA Results Extraction Script")
    print("=" * 50)

    # Set up paths
    reports_dir = Path("../output/lca/reactor_reports")
    output_dir = Path("../output/lca")

    # Create extractor
    extractor = LCAResultsExtractor(reports_dir, output_dir)

    print(f"📁 Reports directory: {reports_dir}")
    print(f"📤 Output directory: {output_dir}")

    # Extract results
    print("\n🚀 Starting extraction process...")
    df = extractor.extract_all_results()

    if df.empty:
        print("❌ No data extracted. Check log file for details.")
        return False

    print(f"✅ Successfully extracted data from {len(df)} reactors")

    # Save to CSV
    print("\n💾 Saving results to CSV...")
    csv_path = extractor.save_to_csv(df)

    print(f"✅ Results saved to: {csv_path}")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"📈 Total rows: {len(df)}")

    # Display basic statistics
    print("\n📋 BASIC STATISTICS")
    print("-" * 30)
    print(
        f"Reactors with H₂ production: {(df['annual_hydrogen_production_kg'] > 0).sum()}")
    print(
        f"Average nuclear capacity: {df['nuclear_capacity_mw'].mean():.1f} MW")
    print(
        f"Avg nuclear carbon intensity: {df['nuclear_only_carbon_intensity'].mean():.2f} gCO₂-eq/kWh")
    print(
        f"Avg integrated carbon intensity: {df['integrated_carbon_intensity'].mean():.2f} gCO₂-eq/kWh")

    # ISO region distribution
    print(f"\n🌍 ISO REGION DISTRIBUTION")
    print("-" * 30)
    iso_counts = df['iso_region'].value_counts()
    for iso, count in iso_counts.items():
        print(f"{iso}: {count} reactors")

    print(f"\n🎉 Extraction completed successfully!")
    print(f"📄 Check 'logs/lca_extraction.log' for detailed processing log")
    print(f"📄 Check 'logs/lca_summary.txt' for detailed summary")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
