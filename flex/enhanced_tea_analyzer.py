"""
Enhanced TEA Data Analyzer
Enhanced TEA Data Collection and Analysis Module

Purpose: Extract comprehensive data from TEA summary reports including:
- Lifecycle analysis data
- Greenfield nuclear-hydrogen system analysis
- Detailed cost breakdowns (LCOH components)
- Ancillary services performance
- Battery performance metrics
- Replacement schedules and lifecycle comparisons
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class EnhancedTEAAnalyzer:
    """Enhanced TEA Data Analyzer with comprehensive data extraction"""

    def __init__(self, results_dir: str = "output/tea"):
        self.results_dir = Path(results_dir)
        self.data = {}

    def collect_comprehensive_data(self) -> pd.DataFrame:
        """Collect comprehensive data from all TEA reports"""
        print("🔍 Collecting comprehensive TEA data from all sources...")

        all_data = []

        # Search for all TEA summary reports
        tea_files = list(self.results_dir.rglob("*TEA_Summary_Report.txt"))

        if not tea_files:
            print("⚠️ No TEA summary reports found in the specified directory")
            return pd.DataFrame()

        print(f"📁 Found {len(tea_files)} TEA summary reports")

        for file_path in tea_files:
            try:
                project_data = self._extract_project_data(file_path)
                if project_data:
                    all_data.append(project_data)
                    print(
                        f"✅ Processed: {project_data.get('project_name', 'Unknown')}")
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
                continue

        if not all_data:
            print("❌ No valid data extracted from TEA reports")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        print(
            f"✅ Collected data for {len(df)} projects across {df['iso'].nunique() if 'iso' in df.columns else 0} ISO regions")

        return df

    def _extract_project_data(self, file_path: Path) -> Dict:
        """Extract comprehensive data from a single TEA report"""

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        data = {}

        # Extract basic project information
        data.update(self._extract_basic_info(content, file_path))

        # Extract system capacities and configuration
        data.update(self._extract_system_capacities(content))

        # Extract financial metrics
        data.update(self._extract_financial_metrics(content))

        # Extract performance metrics
        data.update(self._extract_performance_metrics(content))

        # Extract ancillary services data
        data.update(self._extract_ancillary_services_data(content))

        # Extract battery performance data
        data.update(self._extract_battery_performance(content))

        # Extract LCOH breakdown
        data.update(self._extract_lcoh_breakdown(content))

        # Extract lifecycle analysis
        data.update(self._extract_lifecycle_analysis(content))

        # Extract greenfield analysis
        data.update(self._extract_greenfield_analysis(content))

        # Extract replacement schedules
        data.update(self._extract_replacement_schedules(content))

        # Extract cost assumptions
        data.update(self._extract_cost_assumptions(content))

        return data

    def _extract_basic_info(self, content: str, file_path: Path) -> Dict:
        """Extract basic project information"""
        data = {}

        # Extract project name from file path or content
        project_match = re.search(
            r'Technical Economic Analysis Report - (.+)', content)
        if project_match:
            data['project_name'] = project_match.group(1).strip()
        else:
            data['project_name'] = file_path.parent.name

        # Extract ISO region
        iso_match = re.search(r'ISO Region\s*:\s*(\w+)', content)
        data['iso'] = iso_match.group(1) if iso_match else 'Unknown'

        # Extract project configuration
        lifetime_match = re.search(
            r'Project Lifetime\s*:\s*(\d+)\s*years', content)
        data['project_lifetime_years'] = int(
            lifetime_match.group(1)) if lifetime_match else None

        construction_match = re.search(
            r'Construction Period\s*:\s*(\d+)\s*years', content)
        data['construction_period_years'] = int(
            construction_match.group(1)) if construction_match else None

        discount_match = re.search(r'Discount Rate\s*:\s*([\d.]+)%', content)
        data['discount_rate_percent'] = float(
            discount_match.group(1)) if discount_match else None

        tax_match = re.search(r'Corporate Tax Rate\s*:\s*([\d.]+)%', content)
        data['corporate_tax_rate_percent'] = float(
            tax_match.group(1)) if tax_match else None

        return data

    def _extract_system_capacities(self, content: str) -> Dict:
        """Extract system capacities and configuration"""
        data = {}

        # Nuclear plant capacities
        turbine_match = re.search(
            r'Turbine Capacity\s*:\s*([\d,]+\.?\d*)\s*MW', content)
        data['turbine_capacity_mw'] = float(turbine_match.group(
            1).replace(',', '')) if turbine_match else None

        thermal_match = re.search(
            r'Thermal Capacity\s*:\s*([\d,]+\.?\d*)\s*MWt', content)
        data['thermal_capacity_mwt'] = float(thermal_match.group(
            1).replace(',', '')) if thermal_match else None

        efficiency_match = re.search(
            r'Thermal Efficiency\s*:\s*([\d.]+)', content)
        data['thermal_efficiency'] = float(
            efficiency_match.group(1)) if efficiency_match else None

        # Hydrogen system capacities
        electrolyzer_match = re.search(
            r'Electrolyzer Capacity\s*:\s*([\d,]+\.?\d*)\s*MW', content)
        data['electrolyzer_capacity_mw'] = float(electrolyzer_match.group(
            1).replace(',', '')) if electrolyzer_match else None

        h2_storage_match = re.search(
            r'Hydrogen Storage Capacity\s*:\s*([\d,]+\.?\d*)\s*kg', content)
        data['h2_storage_capacity_kg'] = float(h2_storage_match.group(
            1).replace(',', '')) if h2_storage_match else None

        # Battery system capacities
        battery_energy_match = re.search(
            r'Battery Energy Capacity\s*:\s*([\d,]+\.?\d*)\s*MWh', content)
        data['battery_energy_capacity_mwh'] = float(battery_energy_match.group(
            1).replace(',', '')) if battery_energy_match else None

        battery_power_match = re.search(
            r'Battery Power Capacity\s*:\s*([\d,]+\.?\d*)\s*MW', content)
        data['battery_power_capacity_mw'] = float(battery_power_match.group(
            1).replace(',', '')) if battery_power_match else None

        # H2 production rates
        h2_annual_match = re.search(
            r'Optimal H2 Annual Sales Rate\s*:\s*([\d,]+\.?\d*)\s*kg/year', content)
        data['h2_annual_production_kg'] = float(h2_annual_match.group(
            1).replace(',', '')) if h2_annual_match else None

        return data

    def _extract_financial_metrics(self, content: str) -> Dict:
        """Extract financial performance metrics"""
        data = {}

        # Main financial metrics
        irr_match = re.search(r'IRR \(%\)\s*:\s*([\d.]+)%', content)
        data['irr_percent'] = float(irr_match.group(1)) if irr_match else None

        lcoh_match = re.search(r'LCOH \(USD/kg\)\s*:\s*\$([\d.]+)', content)
        data['lcoh_usd_per_kg'] = float(
            lcoh_match.group(1)) if lcoh_match else None

        npv_match = re.search(r'NPV \(USD\)\s*:\s*\$([\d,]+\.?\d*)', content)
        data['npv_usd'] = float(npv_match.group(
            1).replace(',', '')) if npv_match else None

        payback_match = re.search(
            r'Payback Period \(Years\)\s*:\s*([\d.]+)', content)
        data['payback_period_years'] = float(
            payback_match.group(1)) if payback_match else None

        roi_match = re.search(
            r'Return on Investment \(ROI\)\s*:\s*([\d.]+)', content)
        data['roi'] = float(roi_match.group(1)) if roi_match else None

        # Revenue breakdown
        energy_revenue_match = re.search(
            r'Energy Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['energy_revenue'] = float(energy_revenue_match.group(
            1).replace(',', '')) if energy_revenue_match else None

        h2_sales_match = re.search(
            r'H2 Sales Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['h2_sales_revenue'] = float(h2_sales_match.group(
            1).replace(',', '')) if h2_sales_match else None

        h2_subsidy_match = re.search(
            r'H2 Subsidy Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['h2_subsidy_revenue'] = float(h2_subsidy_match.group(
            1).replace(',', '')) if h2_subsidy_match else None

        as_revenue_match = re.search(
            r'Total Ancillary Services Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['as_revenue'] = float(as_revenue_match.group(
            1).replace(',', '')) if as_revenue_match else None

        total_revenue_match = re.search(
            r'Annual Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['total_revenue'] = float(total_revenue_match.group(
            1).replace(',', '')) if total_revenue_match else None

        # Calculate revenue shares
        if data.get('as_revenue') and data.get('total_revenue'):
            data['as_revenue_share'] = (
                data['as_revenue'] / data['total_revenue']) * 100

        # CAPEX breakdown
        total_capex_match = re.search(r'Total CAPEX\s*:\s*\$([\d,]+)', content)
        data['total_capex_usd'] = float(total_capex_match.group(
            1).replace(',', '')) if total_capex_match else None

        return data

    def _extract_performance_metrics(self, content: str) -> Dict:
        """Extract system performance metrics"""
        data = {}

        # Capacity factors
        electrolyzer_cf_match = re.search(
            r'Electrolyzer CF percent\s*:\s*([\d.]+)%', content)
        data['electrolyzer_cf_percent'] = float(
            electrolyzer_cf_match.group(1)) if electrolyzer_cf_match else None

        turbine_cf_match = re.search(
            r'Turbine CF percent\s*:\s*([\d.]+)%', content)
        data['turbine_cf_percent'] = float(
            turbine_cf_match.group(1)) if turbine_cf_match else None

        battery_cf_match = re.search(
            r'Battery CF percent\s*:\s*([\d.]+)%', content)
        data['battery_cf_percent'] = float(
            battery_cf_match.group(1)) if battery_cf_match else None

        # State of charge
        battery_soc_match = re.search(
            r'Battery SOC percent\s*:\s*([\d.]+)%', content)
        data['battery_soc_percent'] = float(
            battery_soc_match.group(1)) if battery_soc_match else None

        h2_soc_match = re.search(
            r'H2 Storage SOC percent\s*:\s*([\d.]+)%', content)
        data['h2_storage_soc_percent'] = float(
            h2_soc_match.group(1)) if h2_soc_match else None

        # Electricity prices
        avg_price_match = re.search(
            r'Avg Electricity Price USD per MWh\s*:\s*\$([\d.]+)', content)
        data['avg_electricity_price'] = float(
            avg_price_match.group(1)) if avg_price_match else None

        weighted_price_match = re.search(
            r'Weighted Avg Electricity Price USD per MWh\s*:\s*\$([\d.]+)', content)
        data['weighted_avg_electricity_price'] = float(
            weighted_price_match.group(1)) if weighted_price_match else None

        return data

    def _extract_ancillary_services_data(self, content: str) -> Dict:
        """Extract detailed ancillary services performance data"""
        data = {}

        # AS revenue metrics
        as_hourly_avg_match = re.search(
            r'Average Hourly AS Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['as_hourly_avg_revenue'] = float(as_hourly_avg_match.group(
            1).replace(',', '')) if as_hourly_avg_match else None

        as_hourly_max_match = re.search(
            r'Maximum Hourly AS Revenue\s*:\s*\$([\d,]+\.?\d*)', content)
        data['as_hourly_max_revenue'] = float(as_hourly_max_match.group(
            1).replace(',', '')) if as_hourly_max_match else None

        as_utilization_match = re.search(
            r'AS Revenue Utilization Rate\s*:\s*([\d.]+)%', content)
        data['as_utilization_rate'] = float(
            as_utilization_match.group(1)) if as_utilization_match else None

        # AS capacity metrics
        as_per_mw_electrolyzer_match = re.search(
            r'AS Revenue per MW Electrolyzer\s*:\s*\$([\d,]+\.?\d*)/MW', content)
        data['as_revenue_per_mw_electrolyzer'] = float(as_per_mw_electrolyzer_match.group(
            1).replace(',', '')) if as_per_mw_electrolyzer_match else None

        as_per_mw_battery_match = re.search(
            r'AS Revenue per MW Battery Power\s*:\s*\$([\d,]+\.?\d*)/MW', content)
        data['as_revenue_per_mw_battery'] = float(as_per_mw_battery_match.group(
            1).replace(',', '')) if as_per_mw_battery_match else None

        # AS bidding performance
        as_services_count_match = re.search(
            r'Number of AS Services Bid\s*:\s*(\d+)', content)
        data['as_services_count'] = int(as_services_count_match.group(
            1)) if as_services_count_match else None

        as_max_bid_capacity_match = re.search(
            r'Total Maximum Bid Capacity\s*:\s*([\d,]+\.?\d*)\s*MW', content)
        data['as_max_bid_capacity_mw'] = float(as_max_bid_capacity_match.group(
            1).replace(',', '')) if as_max_bid_capacity_match else None

        return data

    def _extract_battery_performance(self, content: str) -> Dict:
        """Extract detailed battery performance data"""
        data = {}

        # Battery charging analysis
        battery_charging_match = re.search(
            r'Total Annual Charging:\s*([\d,]+\.?\d*)\s*MWh/year', content)
        data['battery_annual_charging_mwh'] = float(battery_charging_match.group(
            1).replace(',', '')) if battery_charging_match else None

        grid_charging_match = re.search(
            r'From Grid Purchase:\s*([\d,]+\.?\d*)\s*MWh/year', content)
        data['battery_grid_charging_mwh'] = float(grid_charging_match.group(
            1).replace(',', '')) if grid_charging_match else None

        npp_charging_match = re.search(
            r'From NPP \(Opportunity Cost\):\s*([\d,]+\.?\d*)\s*MWh/year', content)
        data['battery_npp_charging_mwh'] = float(npp_charging_match.group(
            1).replace(',', '')) if npp_charging_match else None

        # Battery costs
        charging_cost_match = re.search(
            r'Total Charging Cost:\s*\$([\d,]+\.?\d*)/year', content)
        data['battery_charging_cost_annual'] = float(charging_cost_match.group(
            1).replace(',', '')) if charging_cost_match else None

        return data

    def _extract_lcoh_breakdown(self, content: str) -> Dict:
        """Extract detailed LCOH component breakdown"""
        data = {}

        # Find LCOH breakdown section
        lcoh_section = re.search(
            r'LCOH Component Breakdown:(.*?)(?=\n\n|\nCost Category Analysis)', content, re.DOTALL)
        if lcoh_section:
            breakdown_text = lcoh_section.group(1)

            # Extract individual components
            components = {
                'lcoh_electrolyzer_system': r'Electrolyzer System\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_electricity_opportunity_cost': r'Electricity Opportunity Cost\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_vom_electrolyzer': r'Vom Electrolyzer\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_h2_storage_system': r'H2 Storage System\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_fixed_om': r'Fixed Om\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_stack_replacement': r'Stack Replacement\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_heat_opportunity_cost': r'Hte Heat Opportunity Cost\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_water_cost': r'Water Cost\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_storage_cycle_cost': r'H2 Storage Cycle Cost\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_grid_integration': r'Grid Integration\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_battery_energy': r'Battery System Energy\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_npp_modifications': r'NPP Modifications\s*:\s*\$\s*([\d.]+)/kg',
                'lcoh_battery_power': r'Battery System Power\s*:\s*\$\s*([\d.]+)/kg'
            }

            for key, pattern in components.items():
                match = re.search(pattern, breakdown_text)
                data[key] = float(match.group(1)) if match else None

        return data

    def _extract_lifecycle_analysis(self, content: str) -> Dict:
        """Extract lifecycle comparison analysis"""
        data = {}

        # Find lifecycle comparison section
        lifecycle_section = re.search(
            r'Lifecycle Comparison Analysis:(.*?)(?=\n\n|\nReport generated)', content, re.DOTALL)
        if lifecycle_section:
            lifecycle_text = lifecycle_section.group(1)

            # Extract 60-year metrics
            lifecycle_60_match = re.search(
                r"'lifecycle_60':\s*{([^}]+)}", lifecycle_text)
            if lifecycle_60_match:
                data['lifecycle_60_npv'] = self._extract_float_from_text(
                    lifecycle_60_match.group(1), r"'npv_usd':\s*np\.float64\(([-\d.]+)\)")
                data['lifecycle_60_roi'] = self._extract_float_from_text(
                    lifecycle_60_match.group(1), r"'roi_percent':\s*np\.float64\(([-\d.]+)\)")
                data['lifecycle_60_lcoh'] = self._extract_float_from_text(
                    lifecycle_60_match.group(1), r"'lcoh_usd_per_kg':\s*np\.float64\(([\d.]+)\)")
                data['lifecycle_60_payback'] = self._extract_float_from_text(
                    lifecycle_60_match.group(1), r"'payback_years':\s*(\d+)")

            # Extract 80-year metrics
            lifecycle_80_match = re.search(
                r"'lifecycle_80':\s*{([^}]+)}", lifecycle_text)
            if lifecycle_80_match:
                data['lifecycle_80_npv'] = self._extract_float_from_text(
                    lifecycle_80_match.group(1), r"'npv_usd':\s*np\.float64\(([-\d.]+)\)")
                data['lifecycle_80_roi'] = self._extract_float_from_text(
                    lifecycle_80_match.group(1), r"'roi_percent':\s*np\.float64\(([-\d.]+)\)")
                data['lifecycle_80_lcoh'] = self._extract_float_from_text(
                    lifecycle_80_match.group(1), r"'lcoh_usd_per_kg':\s*np\.float64\(([\d.]+)\)")
                data['lifecycle_80_payback'] = self._extract_float_from_text(
                    lifecycle_80_match.group(1), r"'payback_years':\s*(\d+)")

        return data

    def _extract_greenfield_analysis(self, content: str) -> Dict:
        """Extract greenfield nuclear-hydrogen system analysis"""
        data = {}

        # Find greenfield analysis section
        greenfield_section = re.search(
            r'greenfield nuclear analysis\s*:\s*{(.*?)}', content, re.DOTALL)
        if greenfield_section:
            greenfield_text = greenfield_section.group(1)

            # Extract key greenfield metrics
            data['greenfield_nuclear_capex'] = self._extract_float_from_text(
                greenfield_text, r"'nuclear_capex_usd':\s*([\d.]+)")
            data['greenfield_h2_capex'] = self._extract_float_from_text(
                greenfield_text, r"'hydrogen_system_capex_usd':\s*np\.float64\(([\d.]+)\)")
            data['greenfield_total_capex'] = self._extract_float_from_text(
                greenfield_text, r"'total_system_capex_usd':\s*np\.float64\(([\d.]+)\)")
            data['greenfield_npv'] = self._extract_float_from_text(
                greenfield_text, r"'npv_usd':\s*np\.float64\(([-\d.]+)\)")
            data['greenfield_irr'] = self._extract_float_from_text(
                greenfield_text, r"'irr_percent':\s*(\d+)")
            data['greenfield_lcoh'] = self._extract_float_from_text(
                greenfield_text, r"'lcoh_integrated_usd_per_kg':\s*np\.float64\(([\d.]+)\)")
            data['greenfield_payback'] = self._extract_float_from_text(
                greenfield_text, r"'payback_period_years':\s*(\d+)")
            data['greenfield_nuclear_lcoe'] = self._extract_float_from_text(
                greenfield_text, r"'nuclear_lcoe_usd_per_mwh':\s*([\d.]+)")
            data['greenfield_project_lifetime'] = self._extract_float_from_text(
                greenfield_text, r"'project_lifetime_years':\s*(\d+)")
            data['greenfield_construction_period'] = self._extract_float_from_text(
                greenfield_text, r"'construction_period_years':\s*(\d+)")

        return data

    def _extract_replacement_schedules(self, content: str) -> Dict:
        """Extract equipment replacement schedules"""
        data = {}

        # Extract replacement information from greenfield analysis
        greenfield_section = re.search(
            r'greenfield nuclear analysis\s*:\s*{(.*?)}', content, re.DOTALL)
        if greenfield_section:
            greenfield_text = greenfield_section.group(1)

            data['electrolyzer_replacements_count'] = self._extract_float_from_text(
                greenfield_text, r"'electrolyzer_replacements_count':\s*(\d+)")
            data['h2_storage_replacements_count'] = self._extract_float_from_text(
                greenfield_text, r"'h2_storage_replacements_count':\s*(\d+)")
            data['battery_replacements_count'] = self._extract_float_from_text(
                greenfield_text, r"'battery_replacements_count':\s*(\d+)")

        return data

    def _extract_cost_assumptions(self, content: str) -> Dict:
        """Extract cost assumptions and learning rates"""
        data = {}

        # Find cost assumptions section
        cost_section = re.search(
            r'Cost Assumptions \(Base Year\)(.*?)(?=\n\d+\.|\nReport generated)', content, re.DOTALL)
        if cost_section:
            cost_text = cost_section.group(1)

            # Extract learning rates and reference capacities
            electrolyzer_lr_match = re.search(
                r'Electrolyzer_System.*?LR:\s*([\d.]+)%', cost_text)
            data['electrolyzer_learning_rate'] = float(
                electrolyzer_lr_match.group(1)) if electrolyzer_lr_match else None

            battery_lr_match = re.search(
                r'Battery_System.*?LR:\s*([\d.]+)%', cost_text)
            data['battery_learning_rate'] = float(
                battery_lr_match.group(1)) if battery_lr_match else None

            h2_storage_lr_match = re.search(
                r'H2_Storage_System.*?LR:\s*([\d.]+)%', cost_text)
            data['h2_storage_learning_rate'] = float(
                h2_storage_lr_match.group(1)) if h2_storage_lr_match else None

        return data

    def _extract_float_from_text(self, text: str, pattern: str) -> Optional[float]:
        """Helper function to extract float values from text using regex"""
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    def calculate_enhanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced metrics from the comprehensive data"""

        # 1. Lifecycle efficiency metrics
        if 'lifecycle_60_npv' in df.columns and 'lifecycle_80_npv' in df.columns:
            df['lifecycle_efficiency'] = df['lifecycle_60_npv'] / \
                df['lifecycle_80_npv']

        # 2. Greenfield vs retrofit comparison
        if 'greenfield_lcoh' in df.columns and 'lcoh_usd_per_kg' in df.columns:
            df['retrofit_advantage'] = df['greenfield_lcoh'] - \
                df['lcoh_usd_per_kg']
            df['retrofit_advantage_percent'] = (
                df['retrofit_advantage'] / df['greenfield_lcoh']) * 100

        # 3. Technology integration efficiency
        if 'electrolyzer_capacity_mw' in df.columns and 'turbine_capacity_mw' in df.columns:
            df['integration_ratio'] = df['electrolyzer_capacity_mw'] / \
                df['turbine_capacity_mw']

        # 4. Revenue diversification index (enhanced)
        revenue_cols = ['energy_revenue', 'h2_sales_revenue',
                        'h2_subsidy_revenue', 'as_revenue']
        available_revenue_cols = [
            col for col in revenue_cols if col in df.columns]

        if len(available_revenue_cols) >= 3:
            # Calculate Herfindahl-Hirschman Index for revenue concentration
            revenue_matrix = df[available_revenue_cols].fillna(0)
            revenue_shares = revenue_matrix.div(
                revenue_matrix.sum(axis=1), axis=0)
            df['revenue_concentration_hhi'] = (revenue_shares ** 2).sum(axis=1)
            df['revenue_diversification_score'] = 1 - \
                df['revenue_concentration_hhi']

        # 5. Cost efficiency metrics
        if 'total_capex_usd' in df.columns and 'h2_annual_production_kg' in df.columns:
            df['capex_per_kg_h2_annual'] = df['total_capex_usd'] / \
                df['h2_annual_production_kg']

        if 'total_capex_usd' in df.columns and 'turbine_capacity_mw' in df.columns:
            df['capex_per_mw_nuclear'] = df['total_capex_usd'] / \
                df['turbine_capacity_mw']

        # 6. Ancillary services efficiency
        if 'as_revenue' in df.columns and 'electrolyzer_capacity_mw' in df.columns:
            df['as_efficiency_per_mw'] = df['as_revenue'] / \
                df['electrolyzer_capacity_mw']

        # 7. System utilization score
        utilization_cols = ['electrolyzer_cf_percent',
                            'turbine_cf_percent', 'battery_cf_percent']
        available_util_cols = [
            col for col in utilization_cols if col in df.columns]
        if available_util_cols:
            df['system_utilization_score'] = df[available_util_cols].mean(
                axis=1)

        # 8. Economic resilience indicator
        if 'payback_period_years' in df.columns and 'irr_percent' in df.columns:
            df['economic_resilience'] = df['irr_percent'] / \
                df['payback_period_years']

        return df


def main():
    """Main function for testing"""
    print("🚀 Testing Enhanced TEA Analyzer...")

    analyzer = EnhancedTEAAnalyzer()
    df = analyzer.collect_comprehensive_data()

    if not df.empty:
        df = analyzer.calculate_enhanced_metrics(df)
        print(f"✅ Enhanced analysis completed for {len(df)} projects")
        print(f"📊 Total columns: {len(df.columns)}")
        print(f"📋 Key metrics available: {list(df.columns[:10])}")
    else:
        print("❌ No data collected")


if __name__ == "__main__":
    main()
