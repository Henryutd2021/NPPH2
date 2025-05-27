"""
Nuclear Flexibility Visualization Module
Nuclear Flexibility Visualization Module
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class NuclearFlexibilityPlotter:
    """Nuclear Flexibility Visualizer"""

    def __init__(self, output_dir: str = "flex_results/nuclear_flexibility_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)

    def create_nuclear_flexibility_dashboard(self, df: pd.DataFrame) -> None:
        """Create nuclear flexibility dashboard"""

        # Create enhanced dashboard with more panels
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)

        fig.suptitle('Nuclear Flexibility Enhancement Analysis Dashboard',
                     fontsize=18, fontweight='bold')

        # Row 1: Original core analysis
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_technology_adoption(df, ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_economic_comparison(df, ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_regional_feasibility(df, ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_revenue_diversification(df, ax4)

        # Row 2: Performance and scaling
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_grid_services_value(df, ax5)

        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_scalability_potential(df, ax6)

        ax7 = fig.add_subplot(gs[1, 2:])
        self._plot_decarbonization_impact(df, ax7)

        # Row 3: Enhanced analysis - Lifecycle and deployment strategy
        ax8 = fig.add_subplot(gs[2, :2])
        self._plot_lifecycle_comparison(df, ax8)

        ax9 = fig.add_subplot(gs[2, 2:])
        self._plot_greenfield_vs_retrofit(df, ax9)

        # Row 4: Cost analysis
        ax10 = fig.add_subplot(gs[3, :2])
        self._plot_cost_breakdown(df, ax10)

        ax11 = fig.add_subplot(gs[3, 2:])
        self._plot_technology_learning(df, ax11)

        # Row 5: System integration and optimization
        ax12 = fig.add_subplot(gs[4, :])
        self._plot_system_integration_analysis(df, ax12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'nuclear_flexibility_dashboard.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_technology_adoption(self, df: pd.DataFrame, ax) -> None:
        """Plot technology adoption status"""
        if 'flexibility_technology_adoption' not in df.columns:
            ax.text(0.5, 0.5, 'No flexibility data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Technology Adoption Status')
            return

        adoption_counts = df['flexibility_technology_adoption'].value_counts()
        labels = ['No Adoption (0)', 'Partial (0.5)', 'Full (1.0)']
        values = [adoption_counts.get(0, 0), adoption_counts.get(
            0.5, 0), adoption_counts.get(1.0, 0)]

        colors = ['lightcoral', 'gold', 'lightgreen']
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax.set_title('Nuclear Flexibility Technology Adoption')

    def _plot_economic_comparison(self, df: pd.DataFrame, ax) -> None:
        """Plot economic value comparison"""
        if 'lcoh_usd_per_kg' not in df.columns:
            ax.text(0.5, 0.5, 'No economic data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Economic Performance Comparison')
            return

        # Use different criteria for comparison - based on IRR or revenue diversification
        if 'new_revenue_share' in df.columns:
            # Compare based on revenue diversification
            traditional = df[df['new_revenue_share'] < 20]  # Lower threshold
            enhanced = df[df['new_revenue_share'] >= 20]    # Higher threshold
        elif 'irr_percent' in df.columns:
            # Compare based on IRR performance
            median_irr = df['irr_percent'].median()
            traditional = df[df['irr_percent'] < median_irr]
            enhanced = df[df['irr_percent'] >= median_irr]
        else:
            # Split data in half for comparison
            median_lcoh = df['lcoh_usd_per_kg'].median()
            traditional = df[df['lcoh_usd_per_kg'] >=
                             median_lcoh]  # Higher cost = traditional
            # Lower cost = enhanced
            enhanced = df[df['lcoh_usd_per_kg'] < median_lcoh]

        if len(traditional) > 0 and len(enhanced) > 0:
            # Filter out NaN values for each group
            traditional_lcoh = traditional['lcoh_usd_per_kg'].dropna()
            enhanced_lcoh = enhanced['lcoh_usd_per_kg'].dropna()

            if len(traditional_lcoh) > 0 and len(enhanced_lcoh) > 0:
                categories = ['Traditional\nMode', 'Enhanced\nMode']
                lcoh_values = [traditional_lcoh.mean(), enhanced_lcoh.mean()]

                # Check for finite values
                if all(np.isfinite(lcoh_values)):
                    bars = ax.bar(categories, lcoh_values, color=[
                                  'lightcoral', 'lightgreen'], alpha=0.7)
                    ax.set_ylabel('LCOH (USD/kg)')
                    ax.set_title('Economic Performance Comparison')

                    # Add value labels on bars
                    for bar, value in zip(bars, lcoh_values):
                        if np.isfinite(value):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lcoh_values) * 0.01,
                                    f'${value:.2f}/kg', ha='center', va='bottom')
                    return

        # Show overall LCOH distribution instead
        lcoh_data = df['lcoh_usd_per_kg'].dropna()
        if len(lcoh_data) > 0:
            ax.hist(lcoh_data, bins=10, alpha=0.7,
                    color='skyblue', edgecolor='black')
            mean_lcoh = lcoh_data.mean()
            if np.isfinite(mean_lcoh):
                ax.axvline(mean_lcoh, color='red', linestyle='--',
                           label=f'Mean: ${mean_lcoh:.2f}/kg')
            ax.set_xlabel('LCOH (USD/kg)')
            ax.set_ylabel('Number of Projects')
            ax.set_title('LCOH Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid LCOH data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Economic Performance Comparison')

    def _plot_regional_feasibility(self, df: pd.DataFrame, ax) -> None:
        """Plot regional feasibility"""
        if 'iso' not in df.columns or 'lcoh_usd_per_kg' not in df.columns:
            ax.text(0.5, 0.5, 'No regional data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Regional Feasibility')
            return

        # Filter out NaN values before grouping
        valid_data = df[['iso', 'lcoh_usd_per_kg']].dropna()

        if len(valid_data) == 0:
            ax.text(0.5, 0.5, 'No valid regional data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Regional Feasibility')
            return

        iso_lcoh = valid_data.groupby(
            'iso')['lcoh_usd_per_kg'].mean().sort_values()

        # Check for finite values
        finite_values = iso_lcoh[np.isfinite(iso_lcoh)]

        if len(finite_values) == 0:
            ax.text(0.5, 0.5, 'No valid regional LCOH data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Regional Feasibility')
            return

        bars = ax.barh(range(len(finite_values)), finite_values.values,
                       color='skyblue', alpha=0.7)
        ax.set_yticks(range(len(finite_values)))
        ax.set_yticklabels(finite_values.index)
        ax.set_xlabel('Average LCOH (USD/kg)')
        ax.set_title('Regional Performance Ranking')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, finite_values.values)):
            if np.isfinite(value):
                ax.text(value + max(finite_values.values) *
                        0.01, i, f'${value:.2f}', va='center')

    def _plot_revenue_diversification(self, df: pd.DataFrame, ax) -> None:
        """Plot revenue diversification"""
        revenue_sources = ['energy_revenue', 'h2_sales_revenue',
                           'h2_subsidy_revenue', 'as_revenue']
        available_sources = [
            col for col in revenue_sources if col in df.columns]

        if len(available_sources) < 2:
            ax.text(0.5, 0.5, 'Insufficient revenue data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Revenue Diversification')
            return

        # Calculate average revenue by source
        avg_revenues = []
        labels = []
        for col in available_sources:
            avg_revenues.append(df[col].fillna(0).mean())
            labels.append(col.replace('_', ' ').title())

        # Create pie chart
        colors = ['lightblue', 'lightgreen', 'gold',
                  'lightcoral'][:len(available_sources)]
        wedges, texts, autotexts = ax.pie(avg_revenues, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax.set_title('Revenue Stream Diversification')

    def _plot_grid_services_value(self, df: pd.DataFrame, ax) -> None:
        """Plot grid services value"""
        if 'as_revenue_share' not in df.columns:
            ax.text(0.5, 0.5, 'No AS data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Grid Services Value')
            return

        # Filter out NaN values
        as_data = df['as_revenue_share'].dropna()

        if len(as_data) == 0:
            ax.text(0.5, 0.5, 'No valid AS revenue data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Grid Services Value')
            return

        # Create histogram of AS revenue share
        ax.hist(as_data, bins=15, alpha=0.7,
                color='orange', edgecolor='black')

        mean_as = as_data.mean()
        if np.isfinite(mean_as):
            ax.axvline(mean_as, color='red', linestyle='--',
                       label=f'Mean: {mean_as:.1f}%')
        ax.set_xlabel('AS Revenue Share (%)')
        ax.set_ylabel('Number of Projects')
        ax.set_title('Ancillary Services Value Distribution')
        ax.legend()

    def _plot_scalability_potential(self, df: pd.DataFrame, ax) -> None:
        """Plot scalability potential"""
        if 'turbine_capacity_mw' not in df.columns or 'h2_annual_production_kg' not in df.columns:
            ax.text(0.5, 0.5, 'No scalability data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scalability Potential')
            return

        # Filter out rows with missing data
        valid_data = df[['turbine_capacity_mw',
                         'h2_annual_production_kg']].dropna()

        if len(valid_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for scaling analysis',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scalability Potential')
            return

        x_data = valid_data['turbine_capacity_mw']
        y_data = valid_data['h2_annual_production_kg'] / \
            1e6  # Convert to millions

        scatter = ax.scatter(x_data, y_data, alpha=0.6, s=60, color='green')
        ax.set_xlabel('Nuclear Capacity (MW)')
        ax.set_ylabel('H2 Production (Million kg/year)')
        ax.set_title('Capacity vs Production Scaling')

        # Add trend line only if we have enough valid data points
        if len(valid_data) > 1:
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)
            except Exception as e:
                print(f"Warning: Could not fit trend line: {e}")
                # Continue without trend line

    def _plot_decarbonization_impact(self, df: pd.DataFrame, ax) -> None:
        """Plot decarbonization impact"""
        if 'h2_annual_production_kg' not in df.columns:
            ax.text(0.5, 0.5, 'No H2 production data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decarbonization Impact')
            return

        # Filter out NaN values
        h2_data = df['h2_annual_production_kg'].dropna()

        if len(h2_data) == 0:
            ax.text(0.5, 0.5, 'No valid H2 production data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decarbonization Impact')
            return

        # Calculate cumulative H2 production and CO2 avoidance
        total_h2 = h2_data.sum()

        if not np.isfinite(total_h2) or total_h2 <= 0:
            ax.text(0.5, 0.5, 'Invalid H2 production data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decarbonization Impact')
            return

        co2_avoidance = total_h2 * 9.3 / 1000  # kg CO2 per kg H2, convert to tonnes

        # Create scaling scenarios
        scales = [1, 10, 50, 100, 500]
        h2_production = [total_h2 * scale /
                         1000 for scale in scales]  # Convert to tonnes
        co2_reduction = [co2_avoidance * scale /
                         1000 for scale in scales]  # Convert to kt

        # Ensure all values are finite
        h2_production = [v if np.isfinite(v) else 0 for v in h2_production]
        co2_reduction = [v if np.isfinite(v) else 0 for v in co2_reduction]

        x = np.arange(len(scales))
        width = 0.35

        bars1 = ax.bar(x - width/2, h2_production, width, label='H2 Production (kt/year)',
                       color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, co2_reduction, width, label='CO2 Avoidance (kt/year)',
                       color='lightgreen', alpha=0.7)

        ax.set_xlabel('Deployment Scale (x current sample)')
        ax.set_ylabel('Annual Impact (kt)')
        ax.set_title('Decarbonization Impact at Scale')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{scale}x' for scale in scales])
        ax.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0 and np.isfinite(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{height:.0f}', ha='center', va='bottom', fontsize=8)

    def _plot_lifecycle_comparison(self, df: pd.DataFrame, ax) -> None:
        """Plot comprehensive lifecycle analysis"""
        # Check if we have project lifetime data
        if 'project_lifetime_years' not in df.columns:
            ax.text(0.5, 0.5, 'No project lifetime data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lifecycle Analysis')
            return

        # Filter out NaN values
        lifetime_data = df['project_lifetime_years'].dropna()

        if len(lifetime_data) == 0:
            ax.text(0.5, 0.5, 'No valid lifetime data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lifecycle Analysis')
            return

        # Create comprehensive lifecycle analysis
        # 1. Lifetime distribution
        unique_lifetimes = sorted(lifetime_data.unique())
        lifetime_counts = [len(df[df['project_lifetime_years'] == lt])
                           for lt in unique_lifetimes]

        # Create subplot within the main plot
        # Main bar chart for lifetime distribution
        bars = ax.bar(range(len(unique_lifetimes)), lifetime_counts,
                      color='skyblue', alpha=0.7, width=0.6)

        ax.set_xticks(range(len(unique_lifetimes)))
        ax.set_xticklabels([f'{int(lt)}y' for lt in unique_lifetimes])
        ax.set_ylabel('Number of Projects')
        ax.set_title('Project Lifetime Distribution & Economic Impact')

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, lifetime_counts)):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lifetime_counts) * 0.02,
                        f'{count}', ha='center', va='bottom', fontweight='bold')

        # Add secondary y-axis for economic metrics
        if 'lcoh_usd_per_kg' in df.columns:
            ax2 = ax.twinx()

            # Calculate average LCOH for each lifetime
            avg_lcoh_by_lifetime = []
            for lt in unique_lifetimes:
                lt_data = df[df['project_lifetime_years']
                             == lt]['lcoh_usd_per_kg'].dropna()
                if len(lt_data) > 0:
                    avg_lcoh_by_lifetime.append(lt_data.mean())
                else:
                    avg_lcoh_by_lifetime.append(0)

            # Filter out zero values for plotting
            valid_indices = [i for i, lcoh in enumerate(
                avg_lcoh_by_lifetime) if lcoh > 0]
            if valid_indices:
                valid_lifetimes = [unique_lifetimes[i] for i in valid_indices]
                valid_lcoh = [avg_lcoh_by_lifetime[i] for i in valid_indices]
                valid_x = [i for i in valid_indices]

                line = ax2.plot(valid_x, valid_lcoh, 'ro-', linewidth=2, markersize=6,
                                label='Avg LCOH', color='red')
                ax2.set_ylabel('Average LCOH (USD/kg)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')

                # Add LCOH value labels
                for x, lcoh in zip(valid_x, valid_lcoh):
                    if np.isfinite(lcoh):
                        ax2.text(x, lcoh + max(valid_lcoh) * 0.02, f'${lcoh:.2f}',
                                 ha='center', va='bottom', color='red', fontsize=8)

        # Add insights text box
        insights = []
        if len(unique_lifetimes) > 1:
            most_common_lifetime = lifetime_data.mode(
            ).iloc[0] if not lifetime_data.mode().empty else unique_lifetimes[0]
            insights.append(f"Most common: {int(most_common_lifetime)} years")

            if 'lcoh_usd_per_kg' in df.columns:
                # Find optimal lifetime (lowest LCOH)
                lifetime_lcoh_pairs = []
                for lt in unique_lifetimes:
                    lt_lcoh = df[df['project_lifetime_years']
                                 == lt]['lcoh_usd_per_kg'].mean()
                    if np.isfinite(lt_lcoh):
                        lifetime_lcoh_pairs.append((lt, lt_lcoh))

                if lifetime_lcoh_pairs:
                    optimal_lifetime, min_lcoh = min(
                        lifetime_lcoh_pairs, key=lambda x: x[1])
                    insights.append(
                        f"Optimal (lowest LCOH): {int(optimal_lifetime)} years")
                    insights.append(
                        f"LCOH range: ${min([x[1] for x in lifetime_lcoh_pairs]):.2f}-${max([x[1] for x in lifetime_lcoh_pairs]):.2f}/kg")

        if insights:
            insight_text = '\n'.join(insights)
            ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8)

    def _plot_greenfield_vs_retrofit(self, df: pd.DataFrame, ax) -> None:
        """Plot greenfield vs retrofit comparison"""
        if 'greenfield_lcoh' not in df.columns or 'lcoh_usd_per_kg' not in df.columns:
            ax.text(0.5, 0.5, 'No greenfield comparison data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Greenfield vs Retrofit Comparison')
            return

        # Filter out NaN values
        greenfield_data = df['greenfield_lcoh'].dropna()
        retrofit_data = df['lcoh_usd_per_kg'].dropna()

        if len(greenfield_data) == 0 or len(retrofit_data) == 0:
            ax.text(0.5, 0.5, 'No valid deployment strategy data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Greenfield vs Retrofit Comparison')
            return

        # Create comparison chart
        categories = ['Greenfield\nDevelopment', 'Retrofit\nExisting Plant']
        lcoh_values = [greenfield_data.mean(), retrofit_data.mean()]

        # Check for finite values
        if not all(np.isfinite(lcoh_values)):
            ax.text(0.5, 0.5, 'Invalid deployment strategy data',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Greenfield vs Retrofit Comparison')
            return

        bars = ax.bar(categories, lcoh_values, color=[
                      'orange', 'green'], alpha=0.7)
        ax.set_ylabel('LCOH (USD/kg)')
        ax.set_title('Deployment Strategy Economic Comparison')

        # Add value labels
        for bar, value in zip(bars, lcoh_values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lcoh_values) * 0.02,
                        f'${value:.2f}/kg', ha='center', va='bottom')

        # Add cost savings annotation
        if greenfield_data.mean() > retrofit_data.mean():
            savings = greenfield_data.mean() - retrofit_data.mean()
            if np.isfinite(savings):
                ax.annotate(f'Retrofit saves\n${savings:.2f}/kg',
                            xy=(1, retrofit_data.mean()),
                            xytext=(0.5, max(lcoh_values) * 0.8),
                            arrowprops=dict(arrowstyle='->', color='red'),
                            ha='center', fontsize=10, color='red')

    def _plot_cost_breakdown(self, df: pd.DataFrame, ax) -> None:
        """Plot detailed cost breakdown"""
        # Look for LCOH component columns
        lcoh_components = [col for col in df.columns if col.startswith(
            'lcoh_') and col != 'lcoh_usd_per_kg']

        if not lcoh_components:
            ax.text(0.5, 0.5, 'No cost breakdown data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cost Breakdown Analysis')
            return

        # Calculate average costs for each component
        component_costs = {}
        for col in lcoh_components:
            component_name = col.replace('lcoh_', '').replace('_', ' ').title()
            component_costs[component_name] = df[col].mean()

        # Sort by cost (descending)
        sorted_components = sorted(
            component_costs.items(), key=lambda x: x[1], reverse=True)

        # Take top 8 components for readability
        top_components = sorted_components[:8]

        if top_components:
            labels, values = zip(*top_components)

            # Create horizontal bar chart
            bars = ax.barh(range(len(labels)), values,
                           color='skyblue', alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('Cost (USD/kg H2)')
            ax.set_title('LCOH Component Breakdown (Top 8)')

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 0.01, i, f'${value:.3f}', va='center')

    def _plot_technology_learning(self, df: pd.DataFrame, ax) -> None:
        """Plot technology learning rates and performance relationships"""

        # Check for project lifetime and performance relationship
        if 'project_lifetime_years' in df.columns and 'electrolyzer_cf_percent' in df.columns:
            # Create scatter plot showing lifetime vs electrolyzer performance
            lifetime_data = df['project_lifetime_years'].dropna()
            cf_data = df['electrolyzer_cf_percent'].dropna()

            # Get common indices for both datasets
            common_indices = df.index[df['project_lifetime_years'].notna(
            ) & df['electrolyzer_cf_percent'].notna()]

            if len(common_indices) > 1:
                x_data = df.loc[common_indices, 'project_lifetime_years']
                y_data = df.loc[common_indices, 'electrolyzer_cf_percent']

                # Create scatter plot
                scatter = ax.scatter(
                    x_data, y_data, alpha=0.6, s=60, c='green', edgecolors='darkgreen')
                ax.set_xlabel('Project Lifetime (Years)')
                ax.set_ylabel('Electrolyzer Capacity Factor (%)')
                ax.set_title('Technology Performance vs Project Lifetime')

                # Add trend line if we have enough data points
                if len(x_data) > 2:
                    try:
                        # Calculate correlation
                        correlation = x_data.corr(y_data)

                        # Fit trend line
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                        ax.plot(x_trend, p(x_trend), "r--",
                                alpha=0.8, linewidth=2)

                        # Add correlation info
                        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                                transform=ax.transAxes, bbox=dict(
                                    boxstyle='round', facecolor='lightblue', alpha=0.8),
                                verticalalignment='top', fontsize=9)

                        # Add performance insights
                        insights = []
                        avg_cf = y_data.mean()
                        insights.append(f'Avg CF: {avg_cf:.1f}%')

                        high_performance = len(y_data[y_data > 95])
                        insights.append(
                            f'High perf (>95%): {high_performance}/{len(y_data)}')

                        # Lifetime analysis
                        short_lifetime = x_data[x_data <= 20]
                        long_lifetime = x_data[x_data > 20]

                        if len(short_lifetime) > 0 and len(long_lifetime) > 0:
                            short_cf = df.loc[df['project_lifetime_years'].isin(
                                short_lifetime), 'electrolyzer_cf_percent'].mean()
                            long_cf = df.loc[df['project_lifetime_years'].isin(
                                long_lifetime), 'electrolyzer_cf_percent'].mean()

                            if np.isfinite(short_cf) and np.isfinite(long_cf):
                                cf_diff = long_cf - short_cf
                                insights.append(
                                    f'Long vs Short: {cf_diff:+.1f}%')

                        insight_text = '\n'.join(insights)
                        ax.text(0.05, 0.75, insight_text, transform=ax.transAxes,
                                bbox=dict(boxstyle='round',
                                          facecolor='wheat', alpha=0.8),
                                verticalalignment='top', fontsize=8)

                    except Exception as e:
                        print(f"Warning: Could not fit trend line: {e}")

                return

        # Fallback to original capacity factor analysis if lifetime data not available
        learning_cols = [col for col in df.columns if 'learning_rate' in col]

        if not learning_cols:
            # Plot capacity factors as technology maturity indicator
            cf_cols = ['electrolyzer_cf_percent',
                       'turbine_cf_percent', 'battery_cf_percent']
            available_cf_cols = [col for col in cf_cols if col in df.columns]

            if available_cf_cols:
                cf_data = []
                labels = []
                colors = ['lightgreen', 'lightblue', 'orange']

                for i, col in enumerate(available_cf_cols):
                    # Filter out NaN values
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        mean_value = valid_data.mean()
                        if np.isfinite(mean_value):
                            cf_data.append(mean_value)
                            labels.append(col.replace(
                                '_cf_percent', '').replace('_', ' ').title())

                if cf_data:
                    bars = ax.bar(labels, cf_data,
                                  color=colors[:len(labels)], alpha=0.7)
                    ax.set_ylabel('Capacity Factor (%)')
                    ax.set_title('Technology Maturity (Capacity Factors)')
                    ax.set_ylim(0, 100)

                    # Add value labels
                    for bar, value in zip(bars, cf_data):
                        if np.isfinite(value):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                    f'{value:.1f}%', ha='center', va='bottom')

                    # Add performance benchmarks
                    ax.axhline(y=90, color='red', linestyle='--',
                               alpha=0.5, label='Good Performance (90%)')
                    ax.axhline(y=95, color='green', linestyle='--',
                               alpha=0.5, label='Excellent Performance (95%)')
                    ax.legend(loc='upper right', fontsize=8)

                else:
                    ax.text(0.5, 0.5, 'No valid capacity factor data available',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Technology Learning & Maturity')
            else:
                ax.text(0.5, 0.5, 'No technology maturity data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Technology Learning & Maturity')
            return

        # Plot learning rates if available
        learning_data = []
        labels = []
        for col in learning_cols:
            # Filter out NaN values
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    learning_data.append(mean_value)
                    labels.append(col.replace('_learning_rate',
                                  '').replace('_', ' ').title())

        if learning_data:
            bars = ax.bar(labels, learning_data, color='lightcoral', alpha=0.7)
            ax.set_ylabel('Learning Rate (%)')
            ax.set_title('Technology Learning Rates')

            # Add value labels
            for bar, value in zip(bars, learning_data):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(learning_data) * 0.02,
                            f'{value:.1f}%', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No valid learning rate data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Technology Learning Rates')

    def _plot_system_integration_analysis(self, df: pd.DataFrame, ax) -> None:
        """Plot comprehensive system integration analysis with multi-dimensional insights"""

        # Check for key integration metrics
        key_metrics = ['project_lifetime_years', 'electrolyzer_capacity_mw', 'turbine_capacity_mw',
                       'lcoh_usd_per_kg', 'irr_percent', 'as_revenue_share']
        available_metrics = [col for col in key_metrics if col in df.columns]

        if len(available_metrics) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for system integration analysis',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Integration Analysis')
            return

        # Create multi-metric integration analysis
        # Focus on electrolyzer-to-nuclear ratio and its impacts
        if 'electrolyzer_capacity_mw' in df.columns and 'turbine_capacity_mw' in df.columns:
            # Calculate integration ratio
            valid_data = df[['electrolyzer_capacity_mw',
                             'turbine_capacity_mw']].dropna()
            if len(valid_data) > 0:
                integration_ratios = valid_data['electrolyzer_capacity_mw'] / \
                    valid_data['turbine_capacity_mw']

                # Create bins for integration levels
                ratio_bins = [0, 0.1, 0.2, 0.3, 1.0]
                ratio_labels = [
                    'Low\n(<10%)', 'Medium\n(10-20%)', 'High\n(20-30%)', 'Very High\n(>30%)']

                # Categorize projects
                integration_categories = pd.cut(
                    integration_ratios, bins=ratio_bins, labels=ratio_labels, include_lowest=True)
                category_counts = integration_categories.value_counts().sort_index()

                # Create main bar chart
                bars = ax.bar(range(len(category_counts)), category_counts.values,
                              color=['lightcoral', 'gold', 'lightgreen', 'lightblue'], alpha=0.7)

                ax.set_xticks(range(len(category_counts)))
                ax.set_xticklabels(category_counts.index)
                ax.set_ylabel('Number of Projects')
                ax.set_title(
                    'System Integration Levels & Economic Performance')

                # Add count labels
                for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
                    if count > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts.values) * 0.02,
                                f'{count}', ha='center', va='bottom', fontweight='bold')

                # Add economic performance overlay if LCOH data available
                if 'lcoh_usd_per_kg' in df.columns:
                    ax2 = ax.twinx()

                    # Calculate average LCOH for each integration category
                    avg_lcoh_by_category = []
                    for category in category_counts.index:
                        # Get projects in this category
                        category_mask = integration_categories == category
                        category_indices = valid_data.index[category_mask]

                        if len(category_indices) > 0:
                            category_lcoh = df.loc[category_indices,
                                                   'lcoh_usd_per_kg'].dropna()
                            if len(category_lcoh) > 0:
                                avg_lcoh_by_category.append(
                                    category_lcoh.mean())
                            else:
                                avg_lcoh_by_category.append(np.nan)
                        else:
                            avg_lcoh_by_category.append(np.nan)

                    # Filter out NaN values for plotting
                    valid_lcoh_indices = [i for i, lcoh in enumerate(
                        avg_lcoh_by_category) if np.isfinite(lcoh)]
                    if valid_lcoh_indices:
                        valid_lcoh_values = [avg_lcoh_by_category[i]
                                             for i in valid_lcoh_indices]

                        line = ax2.plot(valid_lcoh_indices, valid_lcoh_values, 'ro-',
                                        linewidth=2, markersize=6, label='Avg LCOH', color='red')
                        ax2.set_ylabel('Average LCOH (USD/kg)', color='red')
                        ax2.tick_params(axis='y', labelcolor='red')

                        # Add LCOH value labels
                        for x, lcoh in zip(valid_lcoh_indices, valid_lcoh_values):
                            if np.isfinite(lcoh):
                                ax2.text(x, lcoh + max(valid_lcoh_values) * 0.02, f'${lcoh:.2f}',
                                         ha='center', va='bottom', color='red', fontsize=8)

                # Add comprehensive insights
                insights = []

                # Integration ratio statistics
                avg_ratio = integration_ratios.mean()
                insights.append(f'Avg Integration Ratio: {avg_ratio:.1%}')

                # Find optimal integration level (lowest LCOH)
                if 'lcoh_usd_per_kg' in df.columns and len(valid_lcoh_indices) > 0:
                    min_lcoh_idx = valid_lcoh_indices[np.argmin(
                        valid_lcoh_values)]
                    optimal_category = category_counts.index[min_lcoh_idx]
                    min_lcoh_value = min(valid_lcoh_values)
                    insights.append(f'Optimal Level: {optimal_category}')
                    insights.append(f'Best LCOH: ${min_lcoh_value:.2f}/kg')

                # Performance correlation with lifetime if available
                if 'project_lifetime_years' in df.columns:
                    # Calculate correlation between integration ratio and lifetime
                    common_indices = df.index[df['electrolyzer_capacity_mw'].notna() &
                                              df['turbine_capacity_mw'].notna() &
                                              df['project_lifetime_years'].notna()]

                    if len(common_indices) > 2:
                        ratios_for_corr = (df.loc[common_indices, 'electrolyzer_capacity_mw'] /
                                           df.loc[common_indices, 'turbine_capacity_mw'])
                        lifetimes_for_corr = df.loc[common_indices,
                                                    'project_lifetime_years']

                        correlation = ratios_for_corr.corr(lifetimes_for_corr)
                        if np.isfinite(correlation):
                            insights.append(
                                f'Lifetime Correlation: {correlation:.3f}')

                # Revenue diversification impact
                if 'as_revenue_share' in df.columns:
                    common_indices = df.index[df['electrolyzer_capacity_mw'].notna() &
                                              df['turbine_capacity_mw'].notna() &
                                              df['as_revenue_share'].notna()]

                    if len(common_indices) > 2:
                        ratios_for_as = (df.loc[common_indices, 'electrolyzer_capacity_mw'] /
                                         df.loc[common_indices, 'turbine_capacity_mw'])
                        as_revenue_for_corr = df.loc[common_indices,
                                                     'as_revenue_share']

                        as_correlation = ratios_for_as.corr(
                            as_revenue_for_corr)
                        if np.isfinite(as_correlation):
                            insights.append(
                                f'AS Revenue Correlation: {as_correlation:.3f}')

                # Add insights text box
                if insights:
                    insight_text = '\n'.join(insights)
                    ax.text(0.02, 0.98, insight_text, transform=ax.transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                            fontsize=8)

                return

        # Fallback to original metrics if integration ratio cannot be calculated
        metrics = {}

        # Integration ratio
        if 'integration_ratio' in df.columns:
            valid_data = df['integration_ratio'].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    metrics['Integration\nRatio'] = mean_value

        # System utilization
        if 'system_utilization_score' in df.columns:
            valid_data = df['system_utilization_score'].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    metrics['System\nUtilization (%)'] = mean_value

        # Economic resilience
        if 'economic_resilience' in df.columns:
            valid_data = df['economic_resilience'].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    metrics['Economic\nResilience'] = mean_value

        # Revenue diversification
        if 'revenue_diversification_score' in df.columns:
            valid_data = df['revenue_diversification_score'].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    metrics['Revenue\nDiversification'] = mean_value * \
                        100  # Convert to percentage

        # AS efficiency
        if 'as_efficiency_per_mw' in df.columns:
            valid_data = df['as_efficiency_per_mw'].dropna()
            if len(valid_data) > 0:
                mean_value = valid_data.mean()
                if np.isfinite(mean_value):
                    # Convert to thousands
                    metrics['AS Efficiency\n(k$/MW)'] = mean_value / 1000

        if not metrics:
            ax.text(0.5, 0.5, 'No system integration data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Integration Analysis')
            return

        # Create radar-like comparison using bar chart
        labels = list(metrics.keys())
        values = list(metrics.values())

        # Normalize values for comparison (0-100 scale)
        normalized_values = []
        for i, (label, value) in enumerate(zip(labels, values)):
            if 'Utilization' in label or 'Diversification' in label:
                # Already in percentage, cap at 100
                normalized_values.append(min(value, 100))
            elif 'Ratio' in label:
                # Convert ratio to percentage (assuming typical range 0-1)
                normalized_values.append(min(value * 100, 100))
            elif 'Resilience' in label:
                # Scale resilience (assuming typical range 0-10)
                normalized_values.append(min(value * 10, 100))
            else:
                # For other metrics, normalize to 0-100 scale
                if max(values) > 0:
                    normalized_values.append(
                        min(value / max(values) * 100, 100))
                else:
                    normalized_values.append(0)

        # Ensure all values are finite
        normalized_values = [v if np.isfinite(
            v) else 0 for v in normalized_values]

        if all(v == 0 for v in normalized_values):
            ax.text(0.5, 0.5, 'No valid system integration metrics',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Integration Analysis')
            return

        bars = ax.bar(labels, normalized_values, color=[
                      'lightblue', 'lightgreen', 'orange', 'lightcoral', 'gold'][:len(labels)], alpha=0.7)
        ax.set_ylabel('Performance Score (0-100)')
        ax.set_title('System Integration Performance Metrics')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, orig_value, norm_value in zip(bars, values, normalized_values):
            if np.isfinite(norm_value) and np.isfinite(orig_value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{norm_value:.1f}', ha='center', va='bottom', fontweight='bold')
                # Add original value below
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'({orig_value:.2f})', ha='center', va='center', fontsize=8)


def main():
    """Main function"""
    print("🎨 Generating nuclear flexibility visualization charts...")

    from nuclear_flexibility_analysis import NuclearFlexibilityAnalyzer

    # Initialize analyzer and plotter
    analyzer = NuclearFlexibilityAnalyzer()
    plotter = NuclearFlexibilityPlotter()

    # Collect data
    df = analyzer._collect_flexibility_data()

    if not df.empty:
        # Generate dashboard
        plotter.create_nuclear_flexibility_dashboard(df)
        print("✅ Nuclear flexibility dashboard generated")
        print(
            f"📁 View charts: {plotter.output_dir}/nuclear_flexibility_dashboard.png")
    else:
        print("❌ No data found, unable to generate charts")


if __name__ == "__main__":
    main()
