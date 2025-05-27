"""
TEA Statistical Analysis Module
Comprehensive analysis of TEA results across multiple dimensions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

sns.set_palette("husl")


class TEAAnalyzer:
    """Comprehensive TEA results analyzer with multi-dimensional analysis capabilities"""

    def __init__(self, results_dir: str = "TEA_results"):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.summary_stats = {}
        self.analysis_results = {}

    def collect_all_data(self) -> pd.DataFrame:
        """Collect and parse all TEA results from the directory structure"""
        print("🔍 Collecting TEA data from all sources...")

        all_data = []

        # Collect from cs1_tea directory (individual plant results)
        cs1_dir = self.results_dir / "cs1_tea"
        if cs1_dir.exists():
            all_data.extend(self._parse_cs1_results(cs1_dir))

        # Collect from ISO summary reports
        iso_reports = list(self.results_dir.glob("*_TEA_Summary_Report.txt"))
        for report in iso_reports:
            all_data.extend(self._parse_iso_summary(report))

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        if not df.empty:
            # Data cleaning and type conversion
            df = self._clean_and_process_data(df)
            print(
                f"✅ Collected data for {len(df)} projects across {df['iso'].nunique()} ISO regions")
        else:
            print("⚠️ No data found in the specified directory")

        self.data['raw'] = df
        return df

    def _parse_cs1_results(self, cs1_dir: Path) -> List[Dict]:
        """Parse individual plant results from cs1_tea directory"""
        results = []

        for plant_dir in cs1_dir.iterdir():
            if plant_dir.is_dir():
                # Extract plant info from directory name
                plant_info = self._extract_plant_info(plant_dir.name)

                # Look for TEA summary report
                report_files = list(plant_dir.glob("*_TEA_Summary_Report.txt"))
                if report_files:
                    plant_data = self._parse_tea_report(report_files[0])
                    plant_data.update(plant_info)
                    results.append(plant_data)

        return results

    def _extract_plant_info(self, dir_name: str) -> Dict:
        """Extract plant information from directory name"""
        # Pattern: PlantName_Unit_ISO_ProjectLifetime
        parts = dir_name.split('_')

        if len(parts) >= 4:
            plant_name = '_'.join(parts[:-3])
            unit = parts[-3]
            iso = parts[-2]
            project_lifetime = parts[-1]
        else:
            plant_name = dir_name
            unit = "Unknown"
            iso = "Unknown"
            project_lifetime = "Unknown"

        return {
            'plant_name': plant_name,
            'unit': unit,
            'iso': iso,
            'project_lifetime_years': project_lifetime,
            'source': 'cs1_tea'
        }

    def _parse_tea_report(self, report_path: Path) -> Dict:
        """Parse TEA summary report and extract key metrics"""
        data = {}

        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract key metrics using regex patterns
            patterns = {
                # Financial metrics
                'irr_percent': r'IRR \(%\)\s*:\s*([\d.]+)%?',
                'lcoh_usd_per_kg': r'LCOH \(USD/kg\)\s*:\s*\$?([\d.]+)',
                'npv_usd': r'NPV \(USD\)\s*:\s*\$?([\d,.-]+)',
                'payback_period_years': r'Payback Period \(Years\)\s*:\s*([\d.]+)',
                'roi': r'Return on Investment \(ROI\)\s*:\s*([\d.]+)',

                # System configuration
                'turbine_capacity_mw': r'Turbine Capacity\s*:\s*([\d,.]+)\s*MW',
                'thermal_capacity_mwt': r'Thermal Capacity\s*:\s*([\d,.]+)\s*MWt',
                'thermal_efficiency': r'Thermal Efficiency\s*:\s*([\d.]+)',
                'discount_rate': r'Discount Rate\s*:\s*([\d.]+)%',
                'tax_rate': r'Corporate Tax Rate\s*:\s*([\d.]+)%',

                # Optimization results
                'electrolyzer_capacity_mw': r'Electrolyzer Capacity\s*:\s*([\d.]+)\s*MW',
                'h2_storage_capacity_kg': r'Hydrogen Storage Capacity\s*:\s*([\d,.]+)\s*kg',
                'battery_energy_capacity_mwh': r'Battery Energy Capacity\s*:\s*([\d.]+)\s*MWh',
                'battery_power_capacity_mw': r'Battery Power Capacity\s*:\s*([\d.]+)\s*MW',
                'h2_annual_production_kg': r'H2 Production kg annual\s*:\s*([\d,.]+)',

                # Performance metrics
                'electrolyzer_cf_percent': r'Electrolyzer CF percent\s*:\s*([\d.]+)%',
                'turbine_cf_percent': r'Turbine CF percent\s*:\s*([\d.]+)%',
                'battery_cf_percent': r'Battery CF percent\s*:\s*([\d.]+)%',

                # Revenue breakdown
                'total_revenue': r'Annual Revenue\s*:\s*\$?([\d,.]+)',
                'energy_revenue': r'Energy Revenue\s*:\s*\$?([\d,.]+)',
                'h2_sales_revenue': r'H2 Sales Revenue\s*:\s*\$?([\d,.]+)',
                'h2_subsidy_revenue': r'H2 Subsidy Revenue\s*:\s*\$?([\d,.]+)',
                'as_revenue': r'AS Revenue Total\s*:\s*\$?([\d,.]+)',

                # AS performance
                'as_avg_hourly_revenue': r'AS Revenue Average Hourly\s*:\s*\$?([\d,.]+)',
                'as_max_hourly_revenue': r'AS Revenue Maximum Hourly\s*:\s*\$?([\d,.]+)',
                'as_revenue_per_mw_electrolyzer': r'AS Revenue per MW Electrolyzer\s*:\s*\$?([\d,.]+)',
                'as_bid_utilization': r'AS Bid Utilization vs Electrolyzer\s*:\s*([\d.]+)',

                # Costs
                'total_capex': r'Total CAPEX\s*:\s*\$?([\d,.]+)',
                'annual_opex': r'Annual OPEX Cost from Opt\s*:\s*\$?([\d,.]+)',
                'electrolyzer_capex': r'Electrolyzer System\s*:\s*\$?([\d,.]+)',
                'h2_storage_capex': r'H2 Storage System\s*:\s*\$?([\d,.]+)',

                # Electricity metrics
                'avg_electricity_price': r'Avg Electricity Price USD per MWh\s*:\s*\$?([\d.]+)',
                'weighted_avg_electricity_price': r'Weighted Avg Electricity Price USD per MWh\s*:\s*\$?([\d.]+)',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value_str = match.group(1).replace(',', '')
                    try:
                        data[key] = float(value_str)
                    except ValueError:
                        data[key] = value_str
                else:
                    data[key] = None

            # Extract LCOH breakdown if available
            lcoh_breakdown = self._extract_lcoh_breakdown(content)
            if lcoh_breakdown:
                data.update(lcoh_breakdown)

        except Exception as e:
            print(f"⚠️ Error parsing {report_path}: {e}")

        return data

    def _extract_lcoh_breakdown(self, content: str) -> Dict:
        """Extract LCOH component breakdown from report content"""
        breakdown = {}

        # Look for LCOH breakdown section
        lcoh_section = re.search(r'LCOH Component Breakdown:(.*?)(?=\n\n|\nCost Category|\Z)',
                                 content, re.DOTALL | re.IGNORECASE)

        if lcoh_section:
            breakdown_text = lcoh_section.group(1)

            # Extract individual components
            component_pattern = r'(\w+(?:\s+\w+)*)\s*:\s*\$\s*([\d.]+)/kg\s*\(\s*([\d.]+)%\)'
            matches = re.findall(component_pattern, breakdown_text)

            for component, cost, percentage in matches:
                component_clean = component.strip().replace(' ', '_').lower()
                breakdown[f'lcoh_{component_clean}_usd_per_kg'] = float(cost)
                breakdown[f'lcoh_{component_clean}_percent'] = float(
                    percentage)

        return breakdown

    def _parse_iso_summary(self, report_path: Path) -> List[Dict]:
        """Parse ISO-level summary reports"""
        # For now, return empty list as these seem to be aggregate reports
        # Can be extended based on actual content structure
        return []

    def _clean_and_process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the collected data"""
        # Convert string numbers to float
        numeric_columns = [col for col in df.columns if any(keyword in col.lower()
                                                            for keyword in ['revenue', 'cost', 'capex', 'opex', 'price', 'capacity',
                                                                            'production', 'lcoh', 'npv', 'irr', 'roi', 'cf'])]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create derived metrics
        if 'total_revenue' in df.columns and 'annual_opex' in df.columns:
            df['annual_profit'] = df['total_revenue'] - df['annual_opex']

        if 'as_revenue' in df.columns and 'total_revenue' in df.columns:
            df['as_revenue_share'] = (
                df['as_revenue'] / df['total_revenue'] * 100).round(2)

        if 'electrolyzer_capacity_mw' in df.columns and 'h2_annual_production_kg' in df.columns:
            df['h2_production_per_mw'] = (
                df['h2_annual_production_kg'] / df['electrolyzer_capacity_mw']).round(0)

        # Categorize plants by size
        if 'turbine_capacity_mw' in df.columns:
            df['plant_size_category'] = pd.cut(df['turbine_capacity_mw'],
                                               bins=[0, 500, 1000,
                                                     1500, float('inf')],
                                               labels=['Small (<500MW)', 'Medium (500-1000MW)',
                                                       'Large (1000-1500MW)', 'Very Large (>1500MW)'])

        return df

    def perform_comprehensive_analysis(self) -> Dict:
        """Perform comprehensive multi-dimensional analysis"""
        if 'raw' not in self.data or self.data['raw'].empty:
            print(
                "❌ No data available for analysis. Please run collect_all_data() first.")
            return {}

        df = self.data['raw']
        results = {}

        print("📊 Performing comprehensive TEA analysis...")

        # 1. ISO-based analysis
        results['iso_analysis'] = self._analyze_by_iso(df)

        # 2. Plant size analysis
        results['size_analysis'] = self._analyze_by_plant_size(df)

        # 3. Financial performance analysis
        results['financial_analysis'] = self._analyze_financial_performance(df)

        # 4. AS performance analysis
        results['as_analysis'] = self._analyze_as_performance(df)

        # 5. LCOH analysis
        results['lcoh_analysis'] = self._analyze_lcoh_components(df)

        # 6. Technology optimization analysis
        results['optimization_analysis'] = self._analyze_optimization_results(
            df)

        # 7. Correlation analysis
        results['correlation_analysis'] = self._perform_correlation_analysis(
            df)

        # 8. Efficiency analysis
        results['efficiency_analysis'] = self._analyze_system_efficiency(df)

        self.analysis_results = results
        print("✅ Comprehensive analysis completed!")

        return results

    def _analyze_by_iso(self, df: pd.DataFrame) -> Dict:
        """Analyze results by ISO region"""
        if 'iso' not in df.columns:
            return {}

        iso_stats = {}

        for iso in df['iso'].unique():
            if pd.isna(iso):
                continue

            iso_data = df[df['iso'] == iso]

            iso_stats[iso] = {
                'count': len(iso_data),
                'avg_lcoh': iso_data['lcoh_usd_per_kg'].mean() if 'lcoh_usd_per_kg' in iso_data.columns else None,
                'avg_irr': iso_data['irr_percent'].mean() if 'irr_percent' in iso_data.columns else None,
                'avg_as_revenue_share': iso_data['as_revenue_share'].mean() if 'as_revenue_share' in iso_data.columns else None,
                'avg_electrolyzer_cf': iso_data['electrolyzer_cf_percent'].mean() if 'electrolyzer_cf_percent' in iso_data.columns else None,
                'avg_electricity_price': iso_data['avg_electricity_price'].mean() if 'avg_electricity_price' in iso_data.columns else None,
                'total_h2_production': iso_data['h2_annual_production_kg'].sum() if 'h2_annual_production_kg' in iso_data.columns else None,
            }

        return iso_stats

    def _analyze_by_plant_size(self, df: pd.DataFrame) -> Dict:
        """Analyze results by plant size category"""
        if 'plant_size_category' not in df.columns:
            return {}

        size_stats = {}

        for category in df['plant_size_category'].cat.categories:
            category_data = df[df['plant_size_category'] == category]

            if len(category_data) > 0:
                size_stats[category] = {
                    'count': len(category_data),
                    'avg_lcoh': category_data['lcoh_usd_per_kg'].mean() if 'lcoh_usd_per_kg' in category_data.columns else None,
                    'avg_irr': category_data['irr_percent'].mean() if 'irr_percent' in category_data.columns else None,
                    'avg_capacity_factor': category_data['electrolyzer_cf_percent'].mean() if 'electrolyzer_cf_percent' in category_data.columns else None,
                    'economies_of_scale': category_data['h2_production_per_mw'].mean() if 'h2_production_per_mw' in category_data.columns else None,
                }

        return size_stats

    def _analyze_financial_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze financial performance metrics"""
        financial_cols = ['lcoh_usd_per_kg', 'irr_percent',
                          'npv_usd', 'roi', 'payback_period_years']
        available_cols = [col for col in financial_cols if col in df.columns]

        if not available_cols:
            return {}

        stats = {}

        for col in available_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                }

        # Performance ranking
        if 'lcoh_usd_per_kg' in df.columns:
            df_ranked = df.copy()
            df_ranked['lcoh_rank'] = df_ranked['lcoh_usd_per_kg'].rank()
            stats['top_performers'] = df_ranked.nsmallest(5, 'lcoh_usd_per_kg')[
                ['plant_name', 'iso', 'lcoh_usd_per_kg']].to_dict('records')
            stats['bottom_performers'] = df_ranked.nlargest(5, 'lcoh_usd_per_kg')[
                ['plant_name', 'iso', 'lcoh_usd_per_kg']].to_dict('records')

        return stats

    def _analyze_as_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze Ancillary Services performance"""
        as_cols = ['as_revenue', 'as_revenue_share', 'as_avg_hourly_revenue',
                   'as_max_hourly_revenue', 'as_revenue_per_mw_electrolyzer', 'as_bid_utilization']
        available_cols = [col for col in as_cols if col in df.columns]

        if not available_cols:
            return {}

        stats = {}

        # Basic statistics for AS metrics
        for col in available_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                }

        # AS performance by ISO
        if 'iso' in df.columns and 'as_revenue_share' in df.columns:
            stats['as_by_iso'] = df.groupby('iso')['as_revenue_share'].agg(
                ['mean', 'std', 'count']).to_dict('index')

        # Correlation between AS performance and other metrics
        if 'as_revenue_share' in df.columns and 'electrolyzer_cf_percent' in df.columns:
            correlation = df[['as_revenue_share',
                              'electrolyzer_cf_percent']].corr().iloc[0, 1]
            stats['as_cf_correlation'] = correlation

        return stats

    def _analyze_lcoh_components(self, df: pd.DataFrame) -> Dict:
        """Analyze LCOH component breakdown"""
        lcoh_component_cols = [col for col in df.columns if col.startswith(
            'lcoh_') and col.endswith('_usd_per_kg')]

        if not lcoh_component_cols:
            return {}

        stats = {}

        # Average component contributions
        component_averages = {}
        for col in lcoh_component_cols:
            component_name = col.replace(
                'lcoh_', '').replace('_usd_per_kg', '')
            data = df[col].dropna()
            if len(data) > 0:
                component_averages[component_name] = {
                    'avg_cost': data.mean(),
                    'std_cost': data.std(),
                    'min_cost': data.min(),
                    'max_cost': data.max(),
                }

        stats['component_breakdown'] = component_averages

        # Identify major cost drivers
        if component_averages:
            sorted_components = sorted(component_averages.items(),
                                       key=lambda x: x[1]['avg_cost'], reverse=True)
            stats['major_cost_drivers'] = sorted_components[:5]

        return stats

    def _analyze_optimization_results(self, df: pd.DataFrame) -> Dict:
        """Analyze system optimization results"""
        optimization_cols = ['electrolyzer_capacity_mw', 'h2_storage_capacity_kg',
                             'battery_energy_capacity_mwh', 'battery_power_capacity_mw',
                             'h2_annual_production_kg', 'h2_production_per_mw']
        available_cols = [
            col for col in optimization_cols if col in df.columns]

        if not available_cols:
            return {}

        stats = {}

        # Capacity utilization analysis
        for col in available_cols:
            data = df[col].dropna()
            if len(data) > 0:
                stats[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'coefficient_of_variation': data.std() / data.mean() if data.mean() != 0 else 0,
                }

        # Optimization efficiency metrics
        if 'electrolyzer_capacity_mw' in df.columns and 'h2_annual_production_kg' in df.columns:
            df_clean = df.dropna(
                subset=['electrolyzer_capacity_mw', 'h2_annual_production_kg'])
            if len(df_clean) > 0:
                stats['production_efficiency'] = {
                    'avg_production_per_mw': df_clean['h2_annual_production_kg'].mean() / df_clean['electrolyzer_capacity_mw'].mean(),
                    'efficiency_range': {
                        'min': (df_clean['h2_annual_production_kg'] / df_clean['electrolyzer_capacity_mw']).min(),
                        'max': (df_clean['h2_annual_production_kg'] / df_clean['electrolyzer_capacity_mw']).max(),
                    }
                }

        return stats

    def _perform_correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform correlation analysis between key metrics"""
        # Select numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Focus on key metrics
        key_metrics = ['lcoh_usd_per_kg', 'irr_percent', 'as_revenue_share',
                       'electrolyzer_cf_percent', 'avg_electricity_price',
                       'h2_production_per_mw', 'turbine_capacity_mw']

        available_metrics = [col for col in key_metrics if col in numeric_cols]

        if len(available_metrics) < 2:
            return {}

        correlation_matrix = df[available_metrics].corr()

        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5 and not pd.isna(corr_value):
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'available_metrics': available_metrics
        }

    def _analyze_system_efficiency(self, df: pd.DataFrame) -> Dict:
        """Analyze overall system efficiency metrics"""
        efficiency_metrics = ['electrolyzer_cf_percent', 'turbine_cf_percent',
                              'battery_cf_percent', 'thermal_efficiency']
        available_metrics = [
            col for col in efficiency_metrics if col in df.columns]

        if not available_metrics:
            return {}

        stats = {}

        for metric in available_metrics:
            data = df[metric].dropna()
            if len(data) > 0:
                stats[metric] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'efficiency_distribution': {
                        'high_efficiency': len(data[data > data.quantile(0.75)]),
                        'medium_efficiency': len(data[(data >= data.quantile(0.25)) & (data <= data.quantile(0.75))]),
                        'low_efficiency': len(data[data < data.quantile(0.25)])
                    }
                }

        # Overall system efficiency score
        if len(available_metrics) > 1:
            efficiency_scores = df[available_metrics].mean(axis=1, skipna=True)
            stats['overall_efficiency'] = {
                'mean_score': efficiency_scores.mean(),
                'top_performers': df.loc[efficiency_scores.nlargest(5).index, ['plant_name', 'iso'] + available_metrics].to_dict('records')
            }

        return stats

    def generate_comprehensive_plots(self, output_dir: str = "analysis_plots") -> None:
        """Generate comprehensive visualization plots"""
        if 'raw' not in self.data or self.data['raw'].empty:
            print(
                "❌ No data available for plotting. Please run collect_all_data() first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        df = self.data['raw']

        print("📈 Generating comprehensive visualization plots...")

        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        # 1. ISO Comparison Plots
        self._plot_iso_comparison(df, output_path)

        # 2. Financial Performance Plots
        self._plot_financial_performance(df, output_path)

        # 3. AS Performance Plots
        self._plot_as_performance(df, output_path)

        # 4. LCOH Analysis Plots
        self._plot_lcoh_analysis(df, output_path)

        # 5. System Optimization Plots
        self._plot_optimization_results(df, output_path)

        # 6. Correlation Heatmap
        self._plot_correlation_heatmap(df, output_path)

        # 7. Efficiency Analysis Plots
        self._plot_efficiency_analysis(df, output_path)

        # 8. Comprehensive Dashboard
        self._create_dashboard(df, output_path)

        print(f"✅ All plots saved to {output_path}")

    def _plot_iso_comparison(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create ISO comparison plots"""
        if 'iso' not in df.columns:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ISO Region Comparison Analysis',
                     fontsize=16, fontweight='bold')

        # LCOH by ISO
        if 'lcoh_usd_per_kg' in df.columns:
            sns.boxplot(data=df, x='iso', y='lcoh_usd_per_kg', ax=axes[0, 0])
            axes[0, 0].set_title('LCOH Distribution by ISO')
            axes[0, 0].set_ylabel('LCOH (USD/kg)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # AS Revenue Share by ISO
        if 'as_revenue_share' in df.columns:
            sns.boxplot(data=df, x='iso', y='as_revenue_share', ax=axes[0, 1])
            axes[0, 1].set_title('AS Revenue Share by ISO')
            axes[0, 1].set_ylabel('AS Revenue Share (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # Electrolyzer Capacity Factor by ISO
        if 'electrolyzer_cf_percent' in df.columns:
            sns.boxplot(data=df, x='iso',
                        y='electrolyzer_cf_percent', ax=axes[1, 0])
            axes[1, 0].set_title('Electrolyzer Capacity Factor by ISO')
            axes[1, 0].set_ylabel('Capacity Factor (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Average Electricity Price by ISO
        if 'avg_electricity_price' in df.columns:
            iso_avg_price = df.groupby(
                'iso')['avg_electricity_price'].mean().sort_values()
            iso_avg_price.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Average Electricity Price by ISO')
            axes[1, 1].set_ylabel('Price (USD/MWh)')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'iso_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_financial_performance(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create financial performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Financial Performance Analysis',
                     fontsize=16, fontweight='bold')

        # LCOH Distribution
        if 'lcoh_usd_per_kg' in df.columns:
            df['lcoh_usd_per_kg'].hist(
                bins=20, ax=axes[0, 0], alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(df['lcoh_usd_per_kg'].mean(), color='red', linestyle='--',
                               label=f'Mean: ${df["lcoh_usd_per_kg"].mean():.2f}/kg')
            axes[0, 0].set_title('LCOH Distribution')
            axes[0, 0].set_xlabel('LCOH (USD/kg)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()

        # IRR vs LCOH
        if 'irr_percent' in df.columns and 'lcoh_usd_per_kg' in df.columns:
            scatter = axes[0, 1].scatter(df['lcoh_usd_per_kg'], df['irr_percent'],
                                         c=df['turbine_capacity_mw'] if 'turbine_capacity_mw' in df.columns else 'blue',
                                         alpha=0.6, s=60)
            axes[0, 1].set_title('IRR vs LCOH')
            axes[0, 1].set_xlabel('LCOH (USD/kg)')
            axes[0, 1].set_ylabel('IRR (%)')
            if 'turbine_capacity_mw' in df.columns:
                plt.colorbar(scatter, ax=axes[0, 1],
                             label='Turbine Capacity (MW)')

        # NPV Distribution
        if 'npv_usd' in df.columns:
            df['npv_usd'].hist(bins=20, ax=axes[1, 0],
                               alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(df['npv_usd'].mean(), color='red', linestyle='--',
                               label=f'Mean: ${df["npv_usd"].mean()/1e6:.1f}M')
            axes[1, 0].set_title('NPV Distribution')
            axes[1, 0].set_xlabel('NPV (USD)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()

        # Payback Period vs Plant Size
        if 'payback_period_years' in df.columns and 'turbine_capacity_mw' in df.columns:
            axes[1, 1].scatter(df['turbine_capacity_mw'],
                               df['payback_period_years'], alpha=0.6)
            axes[1, 1].set_title('Payback Period vs Plant Size')
            axes[1, 1].set_xlabel('Turbine Capacity (MW)')
            axes[1, 1].set_ylabel('Payback Period (Years)')

        plt.tight_layout()
        plt.savefig(output_path / 'financial_performance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_as_performance(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create AS performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ancillary Services Performance Analysis',
                     fontsize=16, fontweight='bold')

        # AS Revenue Share Distribution
        if 'as_revenue_share' in df.columns:
            df['as_revenue_share'].hist(
                bins=20, ax=axes[0, 0], alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(df['as_revenue_share'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["as_revenue_share"].mean():.1f}%')
            axes[0, 0].set_title('AS Revenue Share Distribution')
            axes[0, 0].set_xlabel('AS Revenue Share (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()

        # AS Revenue vs Electrolyzer CF
        if 'as_revenue_share' in df.columns and 'electrolyzer_cf_percent' in df.columns:
            axes[0, 1].scatter(df['electrolyzer_cf_percent'],
                               df['as_revenue_share'], alpha=0.6)
            axes[0, 1].set_title(
                'AS Revenue Share vs Electrolyzer Capacity Factor')
            axes[0, 1].set_xlabel('Electrolyzer CF (%)')
            axes[0, 1].set_ylabel('AS Revenue Share (%)')

            # Add trend line
            cf_data = df['electrolyzer_cf_percent'].dropna()
            as_data = df['as_revenue_share'].dropna()
            if len(cf_data) > 1 and len(as_data) > 1:
                z = np.polyfit(cf_data.values, as_data.values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(cf_data.min(), cf_data.max(), 100)
                axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8)

        # AS Revenue per MW by ISO
        if 'as_revenue_per_mw_electrolyzer' in df.columns and 'iso' in df.columns:
            sns.boxplot(data=df, x='iso',
                        y='as_revenue_per_mw_electrolyzer', ax=axes[1, 0])
            axes[1, 0].set_title('AS Revenue per MW Electrolyzer by ISO')
            axes[1, 0].set_ylabel('AS Revenue per MW (USD/MW)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # AS Bid Utilization
        if 'as_bid_utilization' in df.columns:
            df['as_bid_utilization'].hist(
                bins=20, ax=axes[1, 1], alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('AS Bid Utilization Distribution')
            axes[1, 1].set_xlabel('Bid Utilization')
            axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_path / 'as_performance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_lcoh_analysis(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create LCOH component analysis plots"""
        # Find LCOH component columns
        lcoh_cols = [col for col in df.columns if col.startswith(
            'lcoh_') and col.endswith('_usd_per_kg')]

        if not lcoh_cols:
            return

        # Calculate average component costs
        component_averages = {}
        for col in lcoh_cols:
            component_name = col.replace('lcoh_', '').replace(
                '_usd_per_kg', '').replace('_', ' ').title()
            avg_cost = df[col].mean()
            if not pd.isna(avg_cost):
                component_averages[component_name] = avg_cost

        if not component_averages:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LCOH Component Analysis', fontsize=16, fontweight='bold')

        # LCOH Component Breakdown (Pie Chart)
        sorted_components = sorted(
            component_averages.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_components[:8])  # Top 8 components

        axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Average LCOH Component Breakdown')

        # LCOH Component Breakdown (Bar Chart)
        component_df = pd.DataFrame(list(component_averages.items()),
                                    columns=['Component', 'Cost'])
        component_df = component_df.sort_values('Cost', ascending=True)

        component_df.plot(x='Component', y='Cost',
                          kind='barh', ax=axes[0, 1], legend=False)
        axes[0, 1].set_title('LCOH Components (USD/kg)')
        axes[0, 1].set_xlabel('Cost (USD/kg)')

        # LCOH vs Major Components
        if len(lcoh_cols) >= 2 and 'lcoh_usd_per_kg' in df.columns:
            # Find the two largest components
            major_components = sorted(
                component_averages.items(), key=lambda x: x[1], reverse=True)[:2]

            if len(major_components) >= 2:
                comp1_col = f"lcoh_{major_components[0][0].lower().replace(' ', '_')}_usd_per_kg"
                comp2_col = f"lcoh_{major_components[1][0].lower().replace(' ', '_')}_usd_per_kg"

                if comp1_col in df.columns:
                    axes[1, 0].scatter(
                        df[comp1_col], df['lcoh_usd_per_kg'], alpha=0.6)
                    axes[1, 0].set_title(
                        f'Total LCOH vs {major_components[0][0]}')
                    axes[1, 0].set_xlabel(f'{major_components[0][0]} (USD/kg)')
                    axes[1, 0].set_ylabel('Total LCOH (USD/kg)')

                if comp2_col in df.columns:
                    axes[1, 1].scatter(
                        df[comp2_col], df['lcoh_usd_per_kg'], alpha=0.6)
                    axes[1, 1].set_title(
                        f'Total LCOH vs {major_components[1][0]}')
                    axes[1, 1].set_xlabel(f'{major_components[1][0]} (USD/kg)')
                    axes[1, 1].set_ylabel('Total LCOH (USD/kg)')

        plt.tight_layout()
        plt.savefig(output_path / 'lcoh_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_optimization_results(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create system optimization results plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('System Optimization Results',
                     fontsize=16, fontweight='bold')

        # Electrolyzer Capacity vs H2 Production
        if 'electrolyzer_capacity_mw' in df.columns and 'h2_annual_production_kg' in df.columns:
            axes[0, 0].scatter(df['electrolyzer_capacity_mw'],
                               df['h2_annual_production_kg']/1e6, alpha=0.6)
            axes[0, 0].set_title('H2 Production vs Electrolyzer Capacity')
            axes[0, 0].set_xlabel('Electrolyzer Capacity (MW)')
            axes[0, 0].set_ylabel('H2 Production (Million kg/year)')

        # H2 Production Efficiency
        if 'h2_production_per_mw' in df.columns:
            df['h2_production_per_mw'].hist(
                bins=20, ax=axes[0, 1], alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('H2 Production Efficiency Distribution')
            axes[0, 1].set_xlabel('H2 Production per MW (kg/MW/year)')
            axes[0, 1].set_ylabel('Frequency')

        # Battery vs Electrolyzer Capacity
        if 'battery_power_capacity_mw' in df.columns and 'electrolyzer_capacity_mw' in df.columns:
            axes[1, 0].scatter(df['electrolyzer_capacity_mw'],
                               df['battery_power_capacity_mw'], alpha=0.6)
            axes[1, 0].set_title('Battery vs Electrolyzer Capacity')
            axes[1, 0].set_xlabel('Electrolyzer Capacity (MW)')
            axes[1, 0].set_ylabel('Battery Power Capacity (MW)')

        # Storage Capacity vs Production
        if 'h2_storage_capacity_kg' in df.columns and 'h2_annual_production_kg' in df.columns:
            storage_ratio = df['h2_storage_capacity_kg'] / \
                df['h2_annual_production_kg'] * 365
            storage_ratio.hist(
                bins=20, ax=axes[1, 1], alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('H2 Storage Days of Production')
            axes[1, 1].set_xlabel('Storage Capacity (Days of Production)')
            axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_path / 'optimization_results.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create correlation heatmap"""
        # Select key numeric columns
        key_metrics = ['lcoh_usd_per_kg', 'irr_percent', 'as_revenue_share',
                       'electrolyzer_cf_percent', 'avg_electricity_price',
                       'h2_production_per_mw', 'turbine_capacity_mw', 'npv_usd']

        available_metrics = [col for col in key_metrics if col in df.columns]

        if len(available_metrics) < 3:
            return

        correlation_matrix = df[available_metrics].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Key TEA Metrics',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_efficiency_analysis(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create efficiency analysis plots"""
        efficiency_cols = ['electrolyzer_cf_percent',
                           'turbine_cf_percent', 'thermal_efficiency']
        available_cols = [col for col in efficiency_cols if col in df.columns]

        if not available_cols:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('System Efficiency Analysis',
                     fontsize=16, fontweight='bold')

        # Capacity Factor Comparison
        if len(available_cols) >= 2:
            cf_data = df[available_cols].dropna()
            cf_data.boxplot(ax=axes[0, 0])
            axes[0, 0].set_title('Capacity Factor Comparison')
            axes[0, 0].set_ylabel('Capacity Factor (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # Efficiency vs LCOH
        if 'electrolyzer_cf_percent' in df.columns and 'lcoh_usd_per_kg' in df.columns:
            axes[0, 1].scatter(df['electrolyzer_cf_percent'],
                               df['lcoh_usd_per_kg'], alpha=0.6)
            axes[0, 1].set_title('LCOH vs Electrolyzer Efficiency')
            axes[0, 1].set_xlabel('Electrolyzer CF (%)')
            axes[0, 1].set_ylabel('LCOH (USD/kg)')

        # Efficiency by ISO
        if 'electrolyzer_cf_percent' in df.columns and 'iso' in df.columns:
            sns.boxplot(data=df, x='iso',
                        y='electrolyzer_cf_percent', ax=axes[1, 0])
            axes[1, 0].set_title('Electrolyzer Efficiency by ISO')
            axes[1, 0].set_ylabel('Electrolyzer CF (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # Overall Efficiency Score
        if len(available_cols) >= 2:
            efficiency_score = df[available_cols].mean(axis=1, skipna=True)
            efficiency_score.hist(
                bins=20, ax=axes[1, 1], alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Overall System Efficiency Score')
            axes[1, 1].set_xlabel('Average Efficiency Score')
            axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_path / 'efficiency_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create a comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        fig.suptitle('TEA Comprehensive Analysis Dashboard',
                     fontsize=20, fontweight='bold')

        # Key metrics summary
        ax1 = fig.add_subplot(gs[0, :2])
        metrics_text = self._generate_summary_text(df)
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Key Metrics Summary', fontweight='bold')

        # LCOH by ISO
        if 'iso' in df.columns and 'lcoh_usd_per_kg' in df.columns:
            ax2 = fig.add_subplot(gs[0, 2:])
            iso_lcoh = df.groupby(
                'iso')['lcoh_usd_per_kg'].mean().sort_values()
            iso_lcoh.plot(kind='bar', ax=ax2, color='skyblue')
            ax2.set_title('Average LCOH by ISO')
            ax2.set_ylabel('LCOH (USD/kg)')
            ax2.tick_params(axis='x', rotation=45)

        # Financial performance scatter
        if 'lcoh_usd_per_kg' in df.columns and 'irr_percent' in df.columns:
            ax3 = fig.add_subplot(gs[1, :2])
            scatter = ax3.scatter(df['lcoh_usd_per_kg'], df['irr_percent'],
                                  c=df['as_revenue_share'] if 'as_revenue_share' in df.columns else 'blue',
                                  alpha=0.6, s=60)
            ax3.set_xlabel('LCOH (USD/kg)')
            ax3.set_ylabel('IRR (%)')
            ax3.set_title('Financial Performance Overview')
            if 'as_revenue_share' in df.columns:
                plt.colorbar(scatter, ax=ax3, label='AS Revenue Share (%)')

        # AS Revenue distribution
        if 'as_revenue_share' in df.columns:
            ax4 = fig.add_subplot(gs[1, 2:])
            df['as_revenue_share'].hist(
                bins=15, ax=ax4, alpha=0.7, edgecolor='black', color='orange')
            ax4.set_title('AS Revenue Share Distribution')
            ax4.set_xlabel('AS Revenue Share (%)')
            ax4.set_ylabel('Frequency')

        # System capacity optimization
        if 'electrolyzer_capacity_mw' in df.columns and 'h2_annual_production_kg' in df.columns:
            ax5 = fig.add_subplot(gs[2, :2])
            ax5.scatter(df['electrolyzer_capacity_mw'],
                        df['h2_annual_production_kg']/1e6, alpha=0.6, color='green')
            ax5.set_xlabel('Electrolyzer Capacity (MW)')
            ax5.set_ylabel('H2 Production (Million kg/year)')
            ax5.set_title('System Capacity vs Production')

        # Efficiency comparison
        efficiency_cols = ['electrolyzer_cf_percent', 'turbine_cf_percent']
        available_eff_cols = [
            col for col in efficiency_cols if col in df.columns]
        if available_eff_cols:
            ax6 = fig.add_subplot(gs[2, 2:])
            df[available_eff_cols].boxplot(ax=ax6)
            ax6.set_title('System Efficiency Comparison')
            ax6.set_ylabel('Capacity Factor (%)')

        # Top performers table
        ax7 = fig.add_subplot(gs[3, :])
        if 'lcoh_usd_per_kg' in df.columns:
            top_performers = df.nsmallest(5, 'lcoh_usd_per_kg')[
                ['plant_name', 'iso', 'lcoh_usd_per_kg', 'irr_percent']]
            table_data = []
            for _, row in top_performers.iterrows():
                table_data.append([
                    row['plant_name'][:20] +
                    '...' if len(str(row['plant_name'])) > 20 else str(
                        row['plant_name']),
                    str(row['iso']),
                    f"${row['lcoh_usd_per_kg']:.2f}" if pd.notna(
                        row['lcoh_usd_per_kg']) else 'N/A',
                    f"{row['irr_percent']:.1f}%" if pd.notna(
                        row['irr_percent']) else 'N/A'
                ])

            table = ax7.table(cellText=table_data,
                              colLabels=['Plant Name', 'ISO',
                                         'LCOH (USD/kg)', 'IRR (%)'],
                              cellLoc='center',
                              loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax7.axis('off')
            ax7.set_title('Top 5 Performers (Lowest LCOH)', fontweight='bold')

        plt.savefig(output_path / 'comprehensive_dashboard.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_text(self, df: pd.DataFrame) -> str:
        """Generate summary text for dashboard"""
        summary_lines = []

        summary_lines.append(f"Total Projects Analyzed: {len(df)}")

        if 'iso' in df.columns:
            summary_lines.append(f"ISO Regions: {df['iso'].nunique()}")

        if 'lcoh_usd_per_kg' in df.columns:
            summary_lines.append(
                f"Average LCOH: ${df['lcoh_usd_per_kg'].mean():.2f}/kg")
            summary_lines.append(
                f"LCOH Range: ${df['lcoh_usd_per_kg'].min():.2f} - ${df['lcoh_usd_per_kg'].max():.2f}/kg")

        if 'irr_percent' in df.columns:
            summary_lines.append(
                f"Average IRR: {df['irr_percent'].mean():.1f}%")

        if 'as_revenue_share' in df.columns:
            summary_lines.append(
                f"Average AS Revenue Share: {df['as_revenue_share'].mean():.1f}%")

        if 'electrolyzer_cf_percent' in df.columns:
            summary_lines.append(
                f"Average Electrolyzer CF: {df['electrolyzer_cf_percent'].mean():.1f}%")

        return '\n'.join(summary_lines)

    def generate_comprehensive_report(self, output_file: str = "TEA_Comprehensive_Analysis_Report.txt") -> None:
        """Generate a comprehensive text report"""
        if not self.analysis_results:
            print(
                "❌ No analysis results available. Please run perform_comprehensive_analysis() first.")
            return

        print("📝 Generating comprehensive analysis report...")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TEA COMPREHENSIVE STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*40 + "\n")
            self._write_executive_summary(f)
            f.write("\n\n")

            # ISO Analysis
            if 'iso_analysis' in self.analysis_results:
                f.write("1. ISO REGION ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_iso_analysis(
                    f, self.analysis_results['iso_analysis'])
                f.write("\n\n")

            # Financial Performance Analysis
            if 'financial_analysis' in self.analysis_results:
                f.write("2. FINANCIAL PERFORMANCE ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_financial_analysis(
                    f, self.analysis_results['financial_analysis'])
                f.write("\n\n")

            # AS Performance Analysis
            if 'as_analysis' in self.analysis_results:
                f.write("3. ANCILLARY SERVICES PERFORMANCE ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_as_analysis(
                    f, self.analysis_results['as_analysis'])
                f.write("\n\n")

            # LCOH Analysis
            if 'lcoh_analysis' in self.analysis_results:
                f.write("4. LEVELIZED COST OF HYDROGEN (LCOH) ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_lcoh_analysis(
                    f, self.analysis_results['lcoh_analysis'])
                f.write("\n\n")

            # System Optimization Analysis
            if 'optimization_analysis' in self.analysis_results:
                f.write("5. SYSTEM OPTIMIZATION ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_optimization_analysis(
                    f, self.analysis_results['optimization_analysis'])
                f.write("\n\n")

            # Correlation Analysis
            if 'correlation_analysis' in self.analysis_results:
                f.write("6. CORRELATION ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_correlation_analysis(
                    f, self.analysis_results['correlation_analysis'])
                f.write("\n\n")

            # Efficiency Analysis
            if 'efficiency_analysis' in self.analysis_results:
                f.write("7. SYSTEM EFFICIENCY ANALYSIS\n")
                f.write("-"*40 + "\n")
                self._write_efficiency_analysis(
                    f, self.analysis_results['efficiency_analysis'])
                f.write("\n\n")

            # Recommendations
            f.write("8. RECOMMENDATIONS AND INSIGHTS\n")
            f.write("-"*40 + "\n")
            self._write_recommendations(f)
            f.write("\n\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"✅ Comprehensive report saved to {output_file}")

    def _write_executive_summary(self, f) -> None:
        """Write executive summary section"""
        if 'raw' not in self.data:
            return

        df = self.data['raw']

        f.write(
            f"This report analyzes {len(df)} nuclear-hydrogen projects across ")
        if 'iso' in df.columns:
            f.write(f"{df['iso'].nunique()} ISO regions.\n\n")
        else:
            f.write("multiple regions.\n\n")

        f.write("Key Findings:\n")

        if 'lcoh_usd_per_kg' in df.columns:
            f.write(
                f"• Average LCOH: ${df['lcoh_usd_per_kg'].mean():.2f}/kg (Range: ${df['lcoh_usd_per_kg'].min():.2f} - ${df['lcoh_usd_per_kg'].max():.2f}/kg)\n")

        if 'irr_percent' in df.columns:
            f.write(
                f"• Average IRR: {df['irr_percent'].mean():.1f}% (Range: {df['irr_percent'].min():.1f}% - {df['irr_percent'].max():.1f}%)\n")

        if 'as_revenue_share' in df.columns:
            f.write(
                f"• Average AS Revenue Share: {df['as_revenue_share'].mean():.1f}% of total revenue\n")

        if 'electrolyzer_cf_percent' in df.columns:
            f.write(
                f"• Average Electrolyzer CF: {df['electrolyzer_cf_percent'].mean():.1f}%\n")

    def _write_iso_analysis(self, f, iso_analysis: Dict) -> None:
        """Write ISO analysis section"""
        f.write("Regional Performance Comparison:\n\n")

        for iso, stats in iso_analysis.items():
            f.write(f"{iso} Region:\n")
            f.write(f"  • Number of projects: {stats['count']}\n")
            if stats['avg_lcoh']:
                f.write(f"  • Average LCOH: ${stats['avg_lcoh']:.2f}/kg\n")
            if stats['avg_irr']:
                f.write(f"  • Average IRR: {stats['avg_irr']:.1f}%\n")
            if stats['avg_as_revenue_share']:
                f.write(
                    f"  • Average AS Revenue Share: {stats['avg_as_revenue_share']:.1f}%\n")
            if stats['avg_electricity_price']:
                f.write(
                    f"  • Average Electricity Price: ${stats['avg_electricity_price']:.2f}/MWh\n")
            f.write("\n")

    def _write_financial_analysis(self, f, financial_analysis: Dict) -> None:
        """Write financial analysis section"""
        for metric, stats in financial_analysis.items():
            if metric in ['top_performers', 'bottom_performers']:
                continue

            if isinstance(stats, dict) and 'mean' in stats:
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  • Mean: {stats['mean']:.2f}\n")
                f.write(f"  • Median: {stats['median']:.2f}\n")
                f.write(f"  • Standard Deviation: {stats['std']:.2f}\n")
                f.write(
                    f"  • Range: {stats['min']:.2f} - {stats['max']:.2f}\n\n")

        if 'top_performers' in financial_analysis:
            f.write("Top 5 Performers (Lowest LCOH):\n")
            for i, performer in enumerate(financial_analysis['top_performers'], 1):
                f.write(
                    f"  {i}. {performer['plant_name']} ({performer['iso']}): ${performer['lcoh_usd_per_kg']:.2f}/kg\n")
            f.write("\n")

    def _write_as_analysis(self, f, as_analysis: Dict) -> None:
        """Write AS analysis section"""
        f.write("Ancillary Services Performance Metrics:\n\n")

        for metric, stats in as_analysis.items():
            if metric == 'as_by_iso':
                f.write("AS Performance by ISO:\n")
                for iso, iso_stats in stats.items():
                    f.write(
                        f"  • {iso}: {iso_stats['mean']:.1f}% ± {iso_stats['std']:.1f}% (n={iso_stats['count']})\n")
                f.write("\n")
            elif isinstance(stats, dict) and 'mean' in stats:
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  • Average: {stats['mean']:.2f}\n")
                f.write(
                    f"  • Range: {stats['min']:.2f} - {stats['max']:.2f}\n\n")

    def _write_lcoh_analysis(self, f, lcoh_analysis: Dict) -> None:
        """Write LCOH analysis section"""
        if 'major_cost_drivers' in lcoh_analysis:
            f.write("Major LCOH Cost Drivers:\n")
            for i, (component, stats) in enumerate(lcoh_analysis['major_cost_drivers'], 1):
                f.write(
                    f"  {i}. {component.replace('_', ' ').title()}: ${stats['avg_cost']:.3f}/kg\n")
            f.write("\n")

        if 'component_breakdown' in lcoh_analysis:
            f.write("Detailed Component Analysis:\n")
            for component, stats in lcoh_analysis['component_breakdown'].items():
                f.write(f"  • {component.replace('_', ' ').title()}:\n")
                f.write(f"    - Average: ${stats['avg_cost']:.3f}/kg\n")
                f.write(
                    f"    - Range: ${stats['min_cost']:.3f} - ${stats['max_cost']:.3f}/kg\n")
            f.write("\n")

    def _write_optimization_analysis(self, f, optimization_analysis: Dict) -> None:
        """Write optimization analysis section"""
        f.write("System Optimization Results:\n\n")

        for metric, stats in optimization_analysis.items():
            if isinstance(stats, dict) and 'mean' in stats:
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  • Average: {stats['mean']:.2f}\n")
                f.write(
                    f"  • Coefficient of Variation: {stats['coefficient_of_variation']:.2f}\n\n")

    def _write_correlation_analysis(self, f, correlation_analysis: Dict) -> None:
        """Write correlation analysis section"""
        if 'strong_correlations' in correlation_analysis:
            f.write("Strong Correlations (|r| > 0.5):\n\n")
            for corr in correlation_analysis['strong_correlations']:
                f.write(
                    f"• {corr['metric1']} vs {corr['metric2']}: r = {corr['correlation']:.3f} ({corr['strength']})\n")
            f.write("\n")

    def _write_efficiency_analysis(self, f, efficiency_analysis: Dict) -> None:
        """Write efficiency analysis section"""
        f.write("System Efficiency Metrics:\n\n")

        for metric, stats in efficiency_analysis.items():
            if isinstance(stats, dict) and 'mean' in stats:
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  • Average: {stats['mean']:.1f}%\n")
                f.write(f"  • Standard Deviation: {stats['std']:.1f}%\n\n")

    def _write_recommendations(self, f) -> None:
        """Write recommendations section"""
        f.write(
            "Based on the comprehensive analysis, the following recommendations are made:\n\n")

        f.write("1. Regional Strategy:\n")
        f.write("   • Focus deployment in regions with favorable electricity pricing\n")
        f.write("   • Leverage regional AS market opportunities\n\n")

        f.write("2. Technology Optimization:\n")
        f.write("   • Optimize electrolyzer sizing for maximum capacity factor\n")
        f.write("   • Balance storage capacity with production requirements\n\n")

        f.write("3. Financial Performance:\n")
        f.write("   • Target projects with LCOH below industry benchmarks\n")
        f.write("   • Maximize AS revenue participation\n\n")

        f.write("4. System Integration:\n")
        f.write("   • Improve overall system efficiency through better integration\n")
        f.write("   • Consider economies of scale for larger deployments\n")


def main():
    """Main function to run comprehensive TEA analysis"""
    print("🚀 Starting Comprehensive TEA Analysis...")

    # Initialize analyzer
    analyzer = TEAAnalyzer()

    # Collect all data
    df = analyzer.collect_all_data()

    if df.empty:
        print("❌ No data found. Please check the TEA_results directory.")
        return

    # Perform comprehensive analysis
    results = analyzer.perform_comprehensive_analysis()

    # Generate plots
    analyzer.generate_comprehensive_plots()

    # Generate report
    analyzer.generate_comprehensive_report()

    print("🎉 Comprehensive TEA analysis completed successfully!")
    print("📊 Check the 'analysis_plots' directory for visualizations")
    print("📝 Check 'TEA_Comprehensive_Analysis_Report.txt' for detailed report")


if __name__ == "__main__":
    main()
