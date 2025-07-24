#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEA Results Parsing and Organization Script
Parse comprehensive TEA results from 42 reactors based on summary_reporting.py output format
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_value(value_str):
    """Universal value parsing function, handles numbers, percentages, currency and N/A"""
    if value_str is None:
        return np.nan

    # First handle explicit "N/A"
    s = value_str.strip()
    if s == 'N/A':
        return np.nan

    # Remove trailing parentheses explanations using string methods to avoid regex issues
    if s.endswith(')'):
        start_paren_index = s.rfind('(')
        if start_paren_index != -1:
            s = s[:start_paren_index].strip()

    # Remove currency symbols, commas, units etc.
    s = s.replace('$', '').replace(',', '').strip()
    s = re.sub(r'/kg|/MWh|years?|MWh|kg|MW|%', '', s).strip()

    try:
        # Handle negative numbers
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]

        if '.' in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return value_str  # If conversion fails, return original value


def _extract_from_section(section_content, patterns):
    """Extract data from given text block"""
    data = {}
    for key, pattern in patterns.items():
        # Ensure regex pattern is valid
        try:
            re.compile(pattern)
        except re.error as e:
            logger.error(
                f"Invalid regex pattern for key '{key}': {pattern} - {e}")
            data[key] = np.nan
            continue

        match = re.search(pattern, section_content, re.DOTALL | re.MULTILINE)
        if match:
            # Find first non-empty capture group
            value = next((g for g in match.groups() if g is not None), None)
            data[key] = _parse_value(value)
        else:
            data[key] = np.nan
    return data


class TEAResultParser:
    def __init__(self, tea_output_dir="../output/tea/cs1_sensitivity/fixed_costs_320000"):
        self.tea_output_dir = Path(tea_output_dir)
        self.all_reactor_data = {}

    def parse_all_reactors(self):
        """Parse all reactor folders"""
        reactor_folders = [
            d for d in self.tea_output_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(reactor_folders)} reactor folders")

        for folder in reactor_folders:
            reactor_name = folder.name
            logger.info(f"Processing: {reactor_name}")
            tea_file = next(folder.glob(
                "*_Comprehensive_TEA_Summary.txt"), None)

            if tea_file:
                try:
                    with open(tea_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    parsed_data = self._parse_file_content(content)
                    parsed_data['reactor_name'] = reactor_name
                    self.all_reactor_data[reactor_name] = parsed_data
                except Exception as e:
                    logger.error(
                        f"Failed to parse file {tea_file}: {e}", exc_info=True)
            else:
                logger.warning(
                    f"Comprehensive TEA file not found: {reactor_name}")

        logger.info(
            f"Successfully parsed {len(self.all_reactor_data)} reactors")
        return self.all_reactor_data

    def _parse_file_content(self, content):
        """Parse complete file content using more robust section splitting"""
        data = {}

        sections_raw = re.split(r'\n(?=\d+\.\s)', content)

        sections = {}
        for s in sections_raw:
            s = s.strip()
            if not s:
                continue

            match = re.match(r'(\d+)\.\s(.*?)\n', s)
            if match:
                sec_num = int(match.group(1))
                sections[sec_num] = s

        data.update(self._parse_project_overview(sections.get(1, "")))
        data.update(self._parse_case1(sections.get(2, "")))
        data.update(self._parse_case2(sections.get(3, "")))
        data.update(self._parse_case3(sections.get(4, "")))
        # Search for greenfield data in the entire content, not just specific sections
        data.update(self._parse_greenfield_cases(content))
        data.update(self._parse_detailed_performance(sections.get(7, "")))

        return data

    def _parse_project_overview(self, section_content):
        if not section_content:
            return {}
        # Parentheses in patterns are for capturing, not literals, so no imbalance issues
        patterns = {
            'iso_region': r'ISO Region\s*:\s*(\S+)',
            'project_lifecycle_years': r'Project Lifecycle\s*:\s*([\d.]+)',
            'discount_rate_percent': r'Discount Rate\s*:\s*([\d.]+)',
            'tax_rate_percent': r'Tax Rate\s*:\s*([\d.]+)',
            'nuclear_capacity_mw': r'Nuclear Unit Capacity \(MW\)\s*:\s*([\d,]+\.?\d*)',
            'nuclear_thermal_capacity_mwt': r'Nuclear Unit Thermal Capacity \(MWt\)\s*:\s*([\d,]+\.?\d*)',
            'thermal_efficiency': r'Nuclear Unit Thermal Efficiency\s*:\s*([\d.]+)',
        }
        return _extract_from_section(section_content, patterns)

    def _parse_case1(self, section_content):
        if not section_content:
            return {}
        prefix = 'case1_'
        # Each pattern is more specific and only captures the value we want
        patterns = {
            f'{prefix}remaining_lifetime_years': r'Remaining Lifetime\s*:\s*([\d,]+\.?\d*)',
            f'{prefix}nameplate_power_factor': r'Nameplate Power Factor\s*:\s*([\d,]+\.?\d*)',
            f'{prefix}npv_usd_no_45u': r'without 45U\s*\):\s*\n\s*NPV\s*:\s*(\S+)',
            f'{prefix}irr_percent_no_45u': r'without 45U\s*\):\s*.*?IRR\s*:\s*(\S+)',
            f'{prefix}payback_years_no_45u': r'without 45U\s*\):\s*.*?Payback\s*:\s*(\S+)',
            f'{prefix}lcoe_usd_per_mwh': r'LCOE \(Nuclear OPEX only\)\s*:\s*(\S+)',
            f'{prefix}npv_usd_with_45u': r'with 45U\s*\):\s*\n\s*NPV\s*:\s*(\S+)',
            f'{prefix}irr_percent_with_45u': r'with 45U\s*\):\s*.*?IRR\s*:\s*(\S+)',
            f'{prefix}payback_years_with_45u': r'with 45U\s*\):\s*.*?Payback\s*:\s*(\S+)',
            f'{prefix}npv_improvement_usd': r'NPV Improvement\s*:\s*(\S+)',
            f'{prefix}total_45u_credits_usd': r'Total 45U Credits\s*:\s*(\S+)',
            f'{prefix}annual_generation_mwh': r'Annual Generation\s*:\s*(\S+)',
            f'{prefix}annual_revenue_usd': r'Annual Revenue\s*:\s*(\S+)',
            f'{prefix}total_annual_opex_usd': r'Total Annual OPEX\s*:\s*(\S+)',
        }
        return _extract_from_section(section_content, patterns)

    def _parse_case2(self, section_content):
        if not section_content:
            return {}
        prefix = 'case2_'
        patterns = {
            f'{prefix}electrolyzer_capacity_mw': r'Electrolyzer Capacity\s*:\s*(\S+)',
            f'{prefix}h2_storage_capacity_kg': r'H2 Storage Capacity\s*:\s*(\S+)',
            f'{prefix}battery_energy_capacity_mwh': r'Battery Energy Capacity\s*:\s*(\S+)',
            f'{prefix}battery_power_capacity_mw': r'Battery Power Capacity\s*:\s*(\S+)',
            f'{prefix}npv_usd_no_45u': r'nuclear part without 45U\s*\):\s*\n\s*NPV\s*:\s*(\S+)',
            f'{prefix}irr_percent_no_45u': r'nuclear part without 45U\s*\):\s*.*?IRR\s*:\s*(\S+)',
            f'{prefix}payback_years_no_45u': r'nuclear part without 45U\s*\):\s*.*?Payback\s*:\s*(\S+)',
            f'{prefix}npv_usd_with_45u': r'with 45U for Nuclear\s*\):\s*\n\s*NPV\s*:\s*(\S+)',
            f'{prefix}irr_percent_with_45u': r'with 45U for Nuclear\s*\):\s*.*?IRR\s*:\s*(\S+)',
            f'{prefix}payback_years_with_45u': r'with 45U for Nuclear\s*\):\s*.*?Payback\s*:\s*(\S+)',
            f'{prefix}total_lcoh_usd_per_kg': r'Total LCOH:\s*(\S+)',
            f'{prefix}total_annual_revenue_usd': r'Total Annual Revenue\s*:\s*(\S+)',
            f'{prefix}total_system_opex_usd': r'Total System OPEX\s*:\s*(\S+)',
            f'{prefix}annual_h2_production_kg': r'Annual H2 Production\s*:\s*(\S+)',
            f'{prefix}ancillary_services_revenue_usd': r'Ancillary Services Revenue\s*:\s*(\S+)',
        }

        data = _extract_from_section(section_content, patterns)
        # LCOH parsing remains unchanged as it appears valid
        return data

    def _parse_case3(self, section_content):
        if not section_content:
            return {}
        prefix = 'case3_'
        patterns = {
            f'{prefix}total_incremental_capex_usd': r'Total Incremental CAPEX:\s*(\S+)',
            f'{prefix}npv_usd': r'Incremental Financial Metrics.*?NPV\s*:\s*(\S+)',
            f'{prefix}irr_percent': r'Incremental Financial Metrics.*?IRR\s*:\s*(\S+)',
            f'{prefix}payback_years': r'Incremental Financial Metrics.*?Payback\s*:\s*(\S+)',
            f'{prefix}total_macrs_depreciation_usd': r'Total Incremental MACRS Depreciation\s*:\s*(\S+)',
            f'{prefix}tax_shield_value_usd': r'Tax Shield Effect \(Value\)\s*:\s*(\S+)',
            f'{prefix}npv_contribution_from_macrs_usd': r'NPV Contribution from MACRS \(Approximation\)\s*:\s*(\S+)',
            f'{prefix}incremental_revenue_usd': r'Incremental Revenue\s*:\s*(\S+)',
            f'{prefix}incremental_costs_usd': r'Incremental Costs\s*:\s*(\S+)',
            f'{prefix}electricity_opportunity_cost_usd': r'Electricity Opportunity Cost\s*:\s*(\S+)',
            f'{prefix}thermal_opportunity_cost_usd': r'Thermal Energy Opportunity Cost\s*:\s*(\S+)',
        }
        return _extract_from_section(section_content, patterns)

    def _parse_greenfield_cases(self, section_content):
        """Enhanced parsing for greenfield cases with better table structure handling"""
        if not section_content:
            return {}

        data = {}

        # Look for the comparison summary table - more flexible pattern
        table_pattern = r'Comparison Summary Table:\s*\n\s*-+.*?\n.*?Lifecycle.*?\n\s*-+.*?\n(.*?)\n\s*-+'
        table_match = re.search(table_pattern, section_content, re.DOTALL)

        if table_match:
            table_content = table_match.group(1)
            lines = [line.strip()
                     for line in table_content.strip().split('\n') if line.strip()]

            for line in lines:
                # Split by pipe and clean up, keeping all non-empty parts
                parts = [p.strip() for p in line.split('|')]
                # Filter out empty parts but keep those that contain data
                parts = [p for p in parts if p.strip()]

                # Expect exactly 8 columns: Lifecycle, Tax Scenario, NPV, IRR, Payback, LCOH, LCOE, LCOS
                if len(parts) == 8:
                    lifecycle = parts[0].strip()
                    tax_scenario = parts[1].strip()

                    # Skip if this looks like a header or doesn't contain "years"
                    if 'Lifecycle' in lifecycle or 'years' not in lifecycle.lower():
                        continue

                    # Create more readable column names
                    lifecycle_clean = lifecycle.replace(
                        ' ', '_').replace('years', 'yr')
                    tax_clean = tax_scenario.replace(' ', '_').replace(
                        'Y', 'y').replace('E', 'e').lower()
                    prefix = f"greenfield_{lifecycle_clean}_{tax_clean}_"

                    # Parse each metric with better error handling
                    data[f'{prefix}npv_m_usd'] = _parse_value(parts[2])
                    data[f'{prefix}irr_percent'] = _parse_value(parts[3])
                    data[f'{prefix}payback_years'] = _parse_value(parts[4])
                    data[f'{prefix}lcoh_usd_per_kg'] = _parse_value(parts[5])
                    data[f'{prefix}lcoe_usd_per_mwh'] = _parse_value(parts[6])
                    data[f'{prefix}lcos_usd_per_mwh'] = _parse_value(parts[7])

                    logger.debug(
                        f"Parsed row: {lifecycle} | {tax_scenario} -> {prefix}")
        else:
            logger.warning(
                "Greenfield comparison table not found or format changed")

        return data

    def _parse_detailed_performance(self, section_content):
        if not section_content:
            return {}
        prefix = 'detailed_'
        patterns = {
            f'{prefix}total_nuclear_generation_mwh': r'Total Nuclear Generation\s*:\s*(\S+)',
            f'{prefix}annual_h2_production_kg': r'Annual H2 Production\s*:\s*(\S+)',
            f'{prefix}electrolyzer_cf_percent': r'Electrolyzer Capacity Factor\s*:\s*(\S+)',
            f'{prefix}turbine_cf_percent': r'Turbine Capacity Factor\s*:\s*(\S+)',
            f'{prefix}avg_electricity_price_usd_per_mwh': r'Average Electricity Price\s*:\s*(\S+)',
            f'{prefix}total_as_revenue_usd': r'Total AS Revenue\s*:\s*(\S+)',
            f'{prefix}total_capex_usd': r'Total CAPEX\s*:\s*(\S+)',
            f'{prefix}nuclear_fixed_om_usd': r'Nuclear Fixed O&M\s*:\s*(\S+)',
            f'{prefix}nuclear_variable_om_usd': r'Nuclear Variable O&M\s*:\s*(\S+)',
            f'{prefix}nuclear_fuel_cost_usd': r'Nuclear Fuel Cost\s*:\s*(\S+)',
            f'{prefix}h2_battery_vom_electrolyzer_usd': r'H2/Battery VOM \(Electrolyzer\)\s*:\s*(\S+)',
            f'{prefix}reported_total_opex_usd': r'Reported Total System OPEX\s*:\s*(\S+)',
            f'{prefix}total_macrs_depreciation_usd': r'Total MACRS Depreciation Over Lifetime\s*:\s*(\S+)',
            f'{prefix}total_tax_shield_value_usd': r'Total Tax Shield Value \(at current tax rate\)\s*:\s*(\S+)',
        }
        return _extract_from_section(section_content, patterns)

    def create_excel_output(self, output_file="../output/tea/cs1_sensitivity/fixed_costs_320000/summary32.xlsx"):
        """Create enhanced Excel output with improved greenfield data organization"""
        if not self.all_reactor_data:
            logger.error("No data available to write to Excel file")
            return

        full_df = pd.DataFrame.from_dict(self.all_reactor_data, orient='index')
        full_df.reset_index(drop=True, inplace=True)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Overview Sheet
            overview_cols = [
                'reactor_name', 'iso_region', 'nuclear_capacity_mw',
                'case1_npv_usd_with_45u', 'case1_lcoe_usd_per_mwh',
                'case2_npv_usd_with_45u', 'case2_irr_percent_with_45u', 'case2_total_lcoh_usd_per_kg',
                'case3_npv_usd', 'case3_irr_percent'
            ]
            self._write_sheet(writer, 'Overview', full_df, overview_cols)

            # Case Sheets
            self._write_sheet(writer, 'Case1_Baseline', full_df, [
                              c for c in full_df.columns if c.startswith('case1_')])
            self._write_sheet(writer, 'Case2_Retrofit', full_df, [
                              c for c in full_df.columns if c.startswith('case2_')])
            self._write_sheet(writer, 'Case3_Incremental', full_df, [
                              c for c in full_df.columns if c.startswith('case3_')])

            # Enhanced Greenfield sheet with organized columns
            greenfield_cols = [
                c for c in full_df.columns if c.startswith('greenfield_')]
            self._write_sheet(
                writer, 'Greenfield_Lifecycle_Analysis', full_df, greenfield_cols)

            # Create separate sheets for each lifecycle and tax scenario combination
            self._create_greenfield_summary_sheets(writer, full_df)

            self._write_sheet(writer, 'Detailed_Performance', full_df, [
                              c for c in full_df.columns if c.startswith('detailed_')])
            self._write_sheet(writer, 'Project_Info', full_df, [c for c in full_df.columns if not (
                c.startswith('case') or c.startswith('detailed') or c.startswith('greenfield'))])
            self._write_sheet(writer, 'All_Data', full_df,
                              full_df.columns.tolist())

        logger.info(f"Excel file created: {output_file}")

    def _create_greenfield_summary_sheets(self, writer, df):
        """Create organized summary sheets for greenfield analysis"""
        # Extract all greenfield columns
        greenfield_cols = [
            c for c in df.columns if c.startswith('greenfield_')]

        if not greenfield_cols:
            return

        # Create summary sheet with organized structure
        summary_data = []

        for _, row in df.iterrows():
            reactor_name = row['reactor_name']

            # Extract data for each lifecycle/tax scenario combination
            for lifecycle in ['60_yr', '80_yr']:
                for tax in ['baseline', '45y_ptc', '48e_itc']:
                    prefix = f"greenfield_{lifecycle}_{tax}_"

                    summary_row = {
                        'reactor_name': reactor_name,
                        'lifecycle': lifecycle.replace('_', ' '),
                        'tax_scenario': tax.replace('_', ' ').upper(),
                        'npv_m_usd': row.get(f'{prefix}npv_m_usd', np.nan),
                        'irr_percent': row.get(f'{prefix}irr_percent', np.nan),
                        'payback_years': row.get(f'{prefix}payback_years', np.nan),
                        'lcoh_usd_per_kg': row.get(f'{prefix}lcoh_usd_per_kg', np.nan),
                        'lcoe_usd_per_mwh': row.get(f'{prefix}lcoe_usd_per_mwh', np.nan),
                        'lcos_usd_per_mwh': row.get(f'{prefix}lcos_usd_per_mwh', np.nan),
                    }
                    summary_data.append(summary_row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(
                writer, sheet_name='Greenfield_Summary', index=False)

    def _write_sheet(self, writer, sheet_name, df, columns):
        """Helper function to filter columns and write to sheet"""
        cols_to_use = [c for c in columns if c in df.columns]
        if 'reactor_name' not in cols_to_use:
            cols_to_use = ['reactor_name'] + cols_to_use

        # Ensure reactor_name is unique and move to front
        unique_cols = list(dict.fromkeys(cols_to_use))
        if 'reactor_name' in unique_cols:
            unique_cols.insert(0, unique_cols.pop(
                unique_cols.index('reactor_name')))

        sheet_df = df[unique_cols].copy()
        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    """Main function"""
    parser = TEAResultParser()
    parser.parse_all_reactors()
    parser.create_excel_output()


if __name__ == "__main__":
    main()
