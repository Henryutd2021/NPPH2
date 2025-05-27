#!/usr/bin/env python3
"""
Nuclear Flexibility Enhancement Analysis Runner
Nuclear Flexibility Enhancement Analysis Main Program

Purpose: 
Execute comprehensive nuclear flexibility enhancement techno-economic analysis, including data collection, analysis, visualization and report generation
Provide decision support for large-scale nuclear development

Key Focus Areas:
1. High investment cost problem solution analysis
2. Technical solution path for lack of flexibility issues
3. Economic value assessment of multi-revenue stream model
4. Applicability analysis across all 7 ISOs in the United States
5. Feasibility demonstration of large-scale deployment
"""

import sys
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔄 Importing modules...")

try:
    from nuclear_flexibility_analysis import NuclearFlexibilityAnalyzer
    print("✅ Successfully imported NuclearFlexibilityAnalyzer")
except Exception as e:
    print(f"❌ Failed to import NuclearFlexibilityAnalyzer: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from nuclear_flexibility_plots import NuclearFlexibilityPlotter
    print("✅ Successfully imported NuclearFlexibilityPlotter")
except Exception as e:
    print(f"❌ Failed to import NuclearFlexibilityPlotter: {e}")
    traceback.print_exc()
    sys.exit(1)


def print_banner():
    """Print program banner"""
    print("=" * 80)
    print("🚀 NUCLEAR FLEXIBILITY ENHANCEMENT ANALYSIS")
    print("   Nuclear Flexibility Enhancement Techno-Economic Analysis System")
    print("=" * 80)
    print()
    print("📋 Analysis Objectives:")
    print("   1. Evaluate techno-economic feasibility of enhancing nuclear flexibility through hydrogen + battery systems")
    print("   2. Analyze improvement effects of multi-revenue stream model on nuclear project financial performance")
    print("   3. Quantify additional revenue value from ancillary services participation")
    print("   4. Demonstrate applicability of this solution across all 7 ISOs in the United States")
    print("   5. Assess economic potential and policy requirements for large-scale deployment")
    print()


def run_comprehensive_analysis():
    """Run comprehensive analysis"""

    print("🔬 Starting nuclear flexibility enhancement comprehensive analysis...")
    print()

    try:
        # 1. Initialize analyzer
        print("📊 Step 1/4: Initialize analysis framework...")
        analyzer = NuclearFlexibilityAnalyzer()
        plotter = NuclearFlexibilityPlotter()
        print("   ✅ Analyzer initialization completed")

        # 2. Data collection and preprocessing
        print("📈 Step 2/4: Collect TEA data and calculate flexibility metrics...")
        df = analyzer._collect_flexibility_data()

        if df.empty:
            print("❌ Error: No TEA analysis data found")
            print(
                "Please ensure TEA analysis has been run and result files have been generated")
            return False

        print(f"   ✅ Successfully collected data for {len(df)} projects")
        print(
            f"   📍 Covering {df['iso'].nunique() if 'iso' in df.columns else 0} ISO regions")

        # 3. Execute nuclear flexibility analysis
        print("🔍 Step 3/4: Execute multi-dimensional flexibility analysis...")
        results = analyzer.analyze_nuclear_flexibility_enhancement()
        print("   ✅ Multi-dimensional analysis completed")

        # Print key findings
        print_key_findings(results)

        # 4. Generate reports and visualizations
        print("📝 Step 4/4: Generate analysis reports and visualizations...")

        # Generate text report
        report_file = "flex_results/Nuclear_Flexibility_Analysis_Report.md"
        analyzer.generate_flexibility_report(report_file)
        print(f"   ✅ Detailed report generated: {report_file}")

        # Generate visualization dashboard
        plotter.create_nuclear_flexibility_dashboard(df)
        print(
            f"   ✅ Visualization dashboard generated: {plotter.output_dir}/nuclear_flexibility_dashboard.png")

        print()
        print("🎉 Nuclear flexibility analysis completed!")
        print("📁 Output files:")
        print(f"   📄 Detailed report: {report_file}")
        print(
            f"   📊 Visualization dashboard: {plotter.output_dir}/nuclear_flexibility_dashboard.png")

        return True

    except Exception as e:
        print(f"❌ Error occurred during analysis: {str(e)}")
        print("📋 Error details:")
        traceback.print_exc()
        return False


def print_key_findings(results: dict):
    """Print key findings"""
    print()
    print("🔑 Key Findings Preview:")
    print("-" * 50)

    # Technology adoption status
    if 'flexibility_impact' in results:
        tech_adoption = results['flexibility_impact'].get(
            'technology_adoption', {})
        if tech_adoption:
            print(
                f"   🔧 Technology adoption: {tech_adoption.get('full_adoption_projects', 0)} projects fully adopted flexibility technologies")

    # Economic value creation
    if 'economic_value' in results:
        value_creation = results['economic_value'].get(
            'value_creation_sources', {})
        if value_creation:
            print(
                f"   💰 Revenue sources: Successfully achieved multi-revenue stream model")

    # Regional feasibility
    if 'regional_feasibility' in results:
        regional = results['regional_feasibility'].get(
            'iso_performance_ranking', {})
        top_regions = regional.get('top_3_regions', [])
        if top_regions:
            print(f"   🌍 Advantageous regions: {', '.join(top_regions[:3])}")

    # Scalability potential
    if 'scalability_potential' in results:
        scaling = results['scalability_potential'].get(
            'deployment_scenarios', {})
        if scaling:
            current_size = scaling.get('current_sample_size', 0)
            capacity = scaling.get('total_capacity_analyzed_mw', 0)
            print(
                f"   📈 Scalability: Verified expansion potential based on {current_size} projects ({capacity:.0f} MW)")

    print()


def main():
    """Main function"""
    print_banner()

    try:
        success = run_comprehensive_analysis()

        print()
        if success:
            print("✅ Nuclear flexibility analysis fully completed!")
            print(
                "🎯 Analysis results provide important techno-economic basis for large-scale nuclear development")
            print(
                "📊 Recommend reviewing generated reports and visualization results for in-depth understanding")
        else:
            print(
                "❌ Analysis could not be completed, please check error messages and retry")

    except Exception as e:
        print(f"❌ Main program execution error: {e}")
        traceback.print_exc()

    print("=" * 80)


if __name__ == "__main__":
    main()
