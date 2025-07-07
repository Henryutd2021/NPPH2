"""
TEA Engine - Core Technical Economic Analysis functionality
This module contains the main TEA analysis logic that can be used by both
ISO-level and reactor-specific analysis scripts.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .data_loader import load_tea_sys_params, load_hourly_results
from .calculations import (
    calculate_annual_metrics,
    calculate_cash_flows,
    calculate_financial_metrics,
    calculate_lcoh_breakdown,
    calculate_incremental_metrics
)
from .nuclear_calculations import (
    calculate_greenfield_nuclear_hydrogen_system,
    calculate_lifecycle_comparison_analysis,
    calculate_greenfield_nuclear_hydrogen_with_tax_incentives,
    calculate_nuclear_baseline_financial_analysis,
    calculate_nuclear_integrated_financial_metrics
)
from .reporting import plot_results, generate_report
from .summary_reporting import generate_comprehensive_tea_summary_report
from .utils import setup_logging, setup_tea_module_logger
from . import config


class TEAEngine:
    """
    Core TEA analysis engine that can be used by different entry points
    """

    def __init__(self, analysis_type: str = "general"):
        """
        Initialize TEA engine

        Args:
            analysis_type: Type of analysis ("iso", "reactor", "general")
        """
        self.analysis_type = analysis_type
        self.logger = None
        self.case_type = None
        self.project_lifetime_years = None

    def safe_float_from_params(self, tea_sys_params: Dict, key: str, default_value: float = 0.0) -> float:
        """Safely convert tea_sys_params value to float with fallback"""
        try:
            value = tea_sys_params.get(key)
            if value is None:
                return default_value
            if isinstance(value, str):
                # Handle string representations that might be empty or 'nan'
                value = value.strip()
                if not value or value.lower() in ['nan', 'none', '']:
                    return default_value
            return float(value)
        except (ValueError, TypeError):
            return default_value

    def setup_logging(self, log_dir: Path, log_name: str, analysis_type: str = None):
        """Setup logging for TEA analysis"""
        if analysis_type:
            log_message = f"🚀 Starting {analysis_type} TEA analysis..."
        else:
            log_message = "🚀 Starting TEA analysis..."

        # Use improved logging without timestamp by default for reactor-based logging
        self.logger = setup_logging(log_dir, log_name, add_timestamp=False)
        self.logger.info(log_message)

        # Reduce verbose logging for cleaner console output
        logging.getLogger('src.tea.calculations').setLevel(logging.WARNING)
        logging.getLogger('src.tea.nuclear_calculations').setLevel(
            logging.WARNING)
        logging.getLogger('src.tea.visualization').setLevel(logging.WARNING)
        logging.getLogger('src.tea.reporting').setLevel(logging.WARNING)

        return self.logger

    def apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        if not self.logger:
            return

        for key, value in overrides.items():
            if key == 'project_lifetime' and value is not None:
                config.PROJECT_LIFETIME_YEARS = value
                self.logger.info(
                    f"Overridden PROJECT_LIFETIME_YEARS to: {config.PROJECT_LIFETIME_YEARS}")
            elif key == 'construction_years' and value is not None:
                config.CONSTRUCTION_YEARS = value
                self.logger.info(
                    f"Overridden CONSTRUCTION_YEARS to: {config.CONSTRUCTION_YEARS}")
            elif key == 'discount_rate' and value is not None:
                config.DISCOUNT_RATE = value
                self.logger.info(
                    f"Overridden DISCOUNT_RATE to: {config.DISCOUNT_RATE}")
            elif key == 'tax_rate' and value is not None:
                config.TAX_RATE = value
                self.logger.info(f"Overridden TAX_RATE to: {config.TAX_RATE}")
            elif key == 'enable_battery' and value is not None:
                config.ENABLE_BATTERY = value
                self.logger.info(
                    f"Overridden ENABLE_BATTERY to: {config.ENABLE_BATTERY}")

    def load_and_prepare_data(self,
                              target_iso: str,
                              input_hourly_results_file: Path,
                              input_sys_data_dir: Path,
                              plant_specific_params: Dict = None,
                              case_type: str = None) -> Tuple[pd.DataFrame, Dict, bool]:
        """
        Load and prepare all input data for TEA analysis

        Returns:
            Tuple of (hourly_results_df, tea_sys_params, success)
        """
        if not self.logger:
            raise RuntimeError(
                "Logger not initialized. Call setup_logging first.")

        # Store case type for later use
        self.case_type = case_type

        # Find NPPs info file path for extracting actual remaining years
        project_root = Path(__file__).parent.parent.parent
        npps_info_path = project_root / "input" / "hourly_data" / "NPPs info.csv"
        if not npps_info_path.exists():
            npps_info_path = None
            self.logger.warning(
                "NPPs info.csv not found, will use CSV/config values for project lifetime")
        else:
            self.logger.info(f"Found NPPs info file: {npps_info_path}")

        # Load system parameters with NPPs info path for actual remaining years
        # and hourly results file path for plant identification
        (tea_sys_params,
         loaded_project_lifetime,
         loaded_discount_rate,
         loaded_construction_years,
         loaded_tax_rate,
         loaded_om_components,
         loaded_nuclear_config,
         loaded_tax_policies) = load_tea_sys_params(
            target_iso,
            input_sys_data_dir,
            str(npps_info_path) if npps_info_path else None,
            case_type,
            # Pass hourly results file path for plant identification
            str(input_hourly_results_file)
        )

        # Add plant-specific parameters if provided
        if plant_specific_params:
            tea_sys_params.update(plant_specific_params)
            self.logger.info(
                f"Added {len(plant_specific_params)} plant-specific parameters")

        # Update config variables with values returned from load_tea_sys_params
        config.PROJECT_LIFETIME_YEARS = loaded_project_lifetime
        config.DISCOUNT_RATE = loaded_discount_rate
        config.CONSTRUCTION_YEARS = loaded_construction_years
        config.TAX_RATE = loaded_tax_rate

        # Store project lifetime for later use in plotting
        self.project_lifetime_years = loaded_project_lifetime

        # For dictionaries, ensure we update them correctly
        config.OM_COMPONENTS.clear()
        config.OM_COMPONENTS.update(loaded_om_components)
        config.NUCLEAR_INTEGRATED_CONFIG.clear()
        config.NUCLEAR_INTEGRATED_CONFIG.update(loaded_nuclear_config)

        self.logger.info(
            f"Using Project Lifetime: {config.PROJECT_LIFETIME_YEARS} years")
        self.logger.info(
            f"Using Discount Rate: {config.DISCOUNT_RATE*100:.2f}%")
        self.logger.info(
            f"Using Construction Years: {config.CONSTRUCTION_YEARS} years")
        self.logger.info(f"Using Tax Rate: {config.TAX_RATE*100:.1f}%")
        self.logger.info(
            f"Loaded tax incentive policies: {list(loaded_tax_policies.keys())}")

        # Load hourly results
        self.logger.info(
            f"Loading hourly results from: {input_hourly_results_file}")
        if not input_hourly_results_file.exists():
            self.logger.error(
                f"Optimization results file not found: {input_hourly_results_file}")
            return None, None, False

        # Store the hourly results file path for remaining years extraction
        self.hourly_results_file_path = str(input_hourly_results_file)

        hourly_res_df = load_hourly_results(input_hourly_results_file)
        if hourly_res_df is None:
            self.logger.error("Failed to load optimization results.")
            return None, None, False

        self.logger.info("Hourly results loaded successfully.")
        return hourly_res_df, tea_sys_params, True

    def run_core_analysis(self,
                          hourly_res_df: pd.DataFrame,
                          tea_sys_params: Dict,
                          enable_greenfield: bool = False,
                          enable_incremental: bool = False) -> Tuple[Dict, Dict, Dict, bool]:
        """
        Run core TEA analysis calculations

        Returns:
            Tuple of (annual_metrics, financial_metrics, cash_flows, success)
        """
        if not self.logger:
            raise RuntimeError(
                "Logger not initialized. Call setup_logging first.")

        # Calculate annual metrics
        annual_metrics_results = calculate_annual_metrics(
            hourly_res_df, tea_sys_params)
        if annual_metrics_results is None:
            self.logger.error("Failed to calculate annual metrics.")
            return None, None, None, False
        self.logger.info("Annual metrics calculated.")

        # Prepare optimized capacities
        optimized_caps = {
            "Electrolyzer_Capacity_MW": annual_metrics_results.get("Electrolyzer_Capacity_MW", 0),
            "H2_Storage_Capacity_kg": annual_metrics_results.get("H2_Storage_Capacity_kg", 0),
            "Battery_Capacity_MWh": annual_metrics_results.get("Battery_Capacity_MWh", 0),
            "Battery_Power_MW": annual_metrics_results.get("Battery_Power_MW", 0),
        }

        # Calculate cash flows
        cash_flows_results = calculate_cash_flows(
            annual_metrics=annual_metrics_results,
            project_lifetime_years=config.PROJECT_LIFETIME_YEARS,
            construction_period_years=config.CONSTRUCTION_YEARS,
            h2_subsidy_value=self.safe_float_from_params(
                tea_sys_params, "hydrogen_subsidy_value_usd_per_kg", 0.0),
            h2_subsidy_duration_years=int(self.safe_float_from_params(
                tea_sys_params, "hydrogen_subsidy_duration_years", 10.0)),
            capex_details=config.CAPEX_COMPONENTS,
            om_details=config.OM_COMPONENTS,
            replacement_details=config.REPLACEMENT_SCHEDULE,
            optimized_capacities=optimized_caps,
            tax_rate=config.TAX_RATE,
            macrs_config=config.MACRS_CONFIG
        )
        self.logger.info("Cash flows calculated.")

        # Calculate financial metrics
        financial_metrics_results = calculate_financial_metrics(
            cash_flows_input=cash_flows_results,
            discount_rate=config.DISCOUNT_RATE,
            annual_h2_production_kg=annual_metrics_results.get(
                "H2_Production_kg_annual", 0),
            project_lifetime_years=config.PROJECT_LIFETIME_YEARS,
            construction_period_years=config.CONSTRUCTION_YEARS,
        )

        # Calculate ROI
        total_capex_val = annual_metrics_results.get("total_capex", 0)
        npv_val = financial_metrics_results.get("NPV_USD", np.nan)
        if total_capex_val > 0 and npv_val is not None and not pd.isna(npv_val):
            financial_metrics_results["ROI"] = npv_val / total_capex_val
        else:
            financial_metrics_results["ROI"] = np.nan
        self.logger.info("Financial metrics calculated.")

        return annual_metrics_results, financial_metrics_results, cash_flows_results, True

    def run_advanced_analysis(self,
                              annual_metrics_results: Dict,
                              financial_metrics_results: Dict,
                              cash_flows_results: Dict,
                              tea_sys_params: Dict,
                              hourly_res_df: pd.DataFrame,
                              target_iso: str,
                              enable_greenfield: bool = False,
                              enable_incremental: bool = False,
                              output_dir: Path = None) -> bool:
        """
        Run advanced analysis modules (greenfield, incremental, etc.)
        """
        if not self.logger:
            raise RuntimeError(
                "Logger not initialized. Call setup_logging first.")

        # Greenfield Nuclear-Hydrogen Analysis
        if enable_greenfield:
            self.logger.info(
                "Starting Greenfield Nuclear-Hydrogen Integrated Analysis with Federal Tax Incentives.")

            # Use actual reactor capacity from annual metrics, not default values
            nuclear_capacity_from_params = tea_sys_params.get(
                "nuclear_plant_capacity_MW")
            nuclear_capacity_from_config = config.NUCLEAR_INTEGRATED_CONFIG.get(
                "nuclear_plant_capacity_mw", 1000)

            # Ensure we have a valid float value, with proper fallback chain
            if nuclear_capacity_from_params is not None:
                try:
                    nuclear_capacity_fallback = float(
                        nuclear_capacity_from_params)
                except (ValueError, TypeError):
                    nuclear_capacity_fallback = nuclear_capacity_from_config
            else:
                nuclear_capacity_fallback = nuclear_capacity_from_config

            actual_nuclear_capacity = annual_metrics_results.get(
                "Turbine_Capacity_MW", nuclear_capacity_fallback)
            self.logger.info(
                f"Using actual nuclear reactor capacity: {actual_nuclear_capacity:.1f} MW")

            # Run comprehensive tax incentive analysis - only for new construction projects
            is_new_construction = False
            if self.case_type:
                case_lower = self.case_type.lower()
                if any(case in case_lower for case in ["case4", "case5", "greenfield", "new"]):
                    is_new_construction = True

            if is_new_construction:
                self.logger.info(
                    "Running comprehensive tax incentive analysis for new construction project...")
                try:
                    # Determine project lifetime based on case type
                    if self.case_type and "case5" in self.case_type.lower():
                        project_lifetime_for_analysis = 80
                        construction_period_for_analysis = 8
                        self.logger.info(
                            "Using 80-year lifecycle for Case 5 analysis")
                    else:
                        project_lifetime_for_analysis = 60
                        construction_period_for_analysis = 8
                        self.logger.info(
                            "Using 60-year lifecycle for Case 4 analysis")

                    comprehensive_tax_incentive_results = calculate_greenfield_nuclear_hydrogen_with_tax_incentives(
                        annual_metrics=annual_metrics_results,
                        nuclear_capacity_mw=actual_nuclear_capacity,
                        tea_sys_params=tea_sys_params,
                        hourly_results_df=hourly_res_df,
                        project_lifetime_config=project_lifetime_for_analysis,
                        construction_period_config=construction_period_for_analysis,
                        discount_rate_config=config.DISCOUNT_RATE,
                        tax_rate_config=config.TAX_RATE,
                        h2_capex_components_config=config.CAPEX_COMPONENTS,
                        h2_om_components_config=config.OM_COMPONENTS,
                        h2_replacement_schedule_config=config.REPLACEMENT_SCHEDULE,
                        macrs_config=config.MACRS_CONFIG,
                        output_dir=str(output_dir) if output_dir else None
                    )

                    if comprehensive_tax_incentive_results:
                        annual_metrics_results["comprehensive_tax_incentive_analysis"] = comprehensive_tax_incentive_results
                        self.logger.info(
                            "Comprehensive tax incentive analysis completed successfully.")
                    else:
                        self.logger.warning(
                            "Comprehensive tax incentive analysis failed or returned no results.")

                except Exception as e:
                    self.logger.error(
                        f"Error in comprehensive tax incentive analysis: {e}")
            else:
                self.logger.info(
                    "Skipping comprehensive tax incentive analysis for existing project (Case 1-3)")
                self.logger.info(
                    "Tax incentive analysis for existing projects is handled in nuclear integrated analysis")

        # LCOH Breakdown
        h2_prod_annual_lcoh = annual_metrics_results.get(
            "H2_Production_kg_annual", 0)
        if h2_prod_annual_lcoh > 0 and "capex_breakdown" in annual_metrics_results:
            # Calculate hydrogen system AS revenue for LCOH deduction
            # This includes AS revenue from electrolyzer and battery systems
            total_as_revenue = annual_metrics_results.get(
                "AS_Revenue_Total", 0)
            electrolyzer_as_revenue = annual_metrics_results.get(
                "AS_Revenue_Electrolyzer", 0)
            battery_as_revenue = annual_metrics_results.get(
                "AS_Revenue_Battery", 0)

            # Use component-specific AS revenue if available, otherwise allocate based on capacity
            if electrolyzer_as_revenue > 0 or battery_as_revenue > 0:
                hydrogen_system_as_revenue = electrolyzer_as_revenue + battery_as_revenue
            else:
                # Fallback: allocate total AS revenue based on hydrogen system capacity
                electrolyzer_capacity = annual_metrics_results.get(
                    "Electrolyzer_Capacity_MW", 0)
                battery_power = annual_metrics_results.get(
                    "Battery_Power_MW", 0)
                turbine_capacity = annual_metrics_results.get(
                    "Turbine_Capacity_MW", 0)

                total_capacity = electrolyzer_capacity + battery_power + turbine_capacity
                if total_capacity > 0:
                    h2_system_capacity = electrolyzer_capacity + battery_power
                    hydrogen_system_as_revenue = total_as_revenue * \
                        (h2_system_capacity / total_capacity)
                else:
                    hydrogen_system_as_revenue = 0

            self.logger.info(
                f"Hydrogen system AS revenue for LCOH calculation: ${hydrogen_system_as_revenue:,.0f}/year")

            lcoh_breakdown_results = calculate_lcoh_breakdown(
                annual_metrics=annual_metrics_results,
                capex_breakdown=annual_metrics_results["capex_breakdown"],
                project_lifetime_years=config.PROJECT_LIFETIME_YEARS,
                construction_period_years=config.CONSTRUCTION_YEARS,
                discount_rate=config.DISCOUNT_RATE,
                annual_h2_production_kg=h2_prod_annual_lcoh,
                annual_hydrogen_as_revenue=hydrogen_system_as_revenue,
            )
            if lcoh_breakdown_results:
                annual_metrics_results["lcoh_breakdown_analysis"] = lcoh_breakdown_results
                financial_metrics_results["LCOH_USD_per_kg"] = lcoh_breakdown_results.get(
                    "total_lcoh_usd_per_kg", np.nan)
                self.logger.info(
                    f"LCOH breakdown calculated. Total LCOH: ${financial_metrics_results.get('LCOH_USD_per_kg', float('nan')):.3f}/kg")
        else:
            financial_metrics_results["LCOH_USD_per_kg"] = np.nan
            self.logger.warning(
                "Skipping LCOH breakdown: No H2 production or CAPEX breakdown missing.")

        # LCOS Breakdown (for battery systems)
        battery_capacity_mwh = annual_metrics_results.get(
            "Battery_Capacity_MWh", 0)
        if battery_capacity_mwh > 0 and "capex_breakdown" in annual_metrics_results:
            # Calculate annual battery throughput (charge + discharge)
            annual_battery_charge = annual_metrics_results.get(
                "Annual_Battery_Charge_MWh", 0)
            annual_battery_discharge = annual_metrics_results.get(
                "Annual_Battery_Discharge_MWh", 0)
            annual_battery_throughput = annual_battery_charge + annual_battery_discharge

            if annual_battery_throughput > 0:
                # Calculate battery system AS revenue for LCOS deduction
                battery_as_revenue = annual_metrics_results.get(
                    "AS_Revenue_Battery", 0)

                # If component-specific AS revenue not available, allocate based on capacity
                if battery_as_revenue == 0:
                    total_as_revenue = annual_metrics_results.get(
                        "AS_Revenue_Total", 0)
                    battery_power = annual_metrics_results.get(
                        "Battery_Power_MW", 0)
                    electrolyzer_capacity = annual_metrics_results.get(
                        "Electrolyzer_Capacity_MW", 0)
                    turbine_capacity = annual_metrics_results.get(
                        "Turbine_Capacity_MW", 0)

                    total_capacity = battery_power + electrolyzer_capacity + turbine_capacity
                    if total_capacity > 0:
                        battery_as_revenue = total_as_revenue * \
                            (battery_power / total_capacity)

                self.logger.info(
                    f"Battery system AS revenue for LCOS calculation: ${battery_as_revenue:,.0f}/year")

                try:
                    from .calculations import calculate_lcos_breakdown
                    lcos_breakdown_results = calculate_lcos_breakdown(
                        annual_metrics=annual_metrics_results,
                        capex_breakdown=annual_metrics_results["capex_breakdown"],
                        project_lifetime_years=config.PROJECT_LIFETIME_YEARS,
                        construction_period_years=config.CONSTRUCTION_YEARS,
                        discount_rate=config.DISCOUNT_RATE,
                        annual_battery_throughput_mwh=annual_battery_throughput,
                        annual_battery_as_revenue=battery_as_revenue,
                    )

                    if lcos_breakdown_results:
                        annual_metrics_results["lcos_breakdown_analysis"] = lcos_breakdown_results
                        financial_metrics_results["LCOS_USD_per_MWh"] = lcos_breakdown_results.get(
                            "total_lcos_usd_per_mwh", np.nan)
                        self.logger.info(
                            f"LCOS breakdown calculated. Total LCOS: ${financial_metrics_results.get('LCOS_USD_per_MWh', float('nan')):.3f}/MWh")
                    else:
                        financial_metrics_results["LCOS_USD_per_MWh"] = np.nan
                        self.logger.warning(
                            "Failed to calculate LCOS breakdown")
                except Exception as e:
                    self.logger.error(f"Error calculating LCOS breakdown: {e}")
                    financial_metrics_results["LCOS_USD_per_MWh"] = np.nan
            else:
                financial_metrics_results["LCOS_USD_per_MWh"] = np.nan
                self.logger.warning(
                    "No battery throughput data available for LCOS calculation")
        else:
            financial_metrics_results["LCOS_USD_per_MWh"] = np.nan
            self.logger.debug(
                "Skipping LCOS breakdown: No battery capacity or CAPEX breakdown missing.")

        # Nuclear Power Plant Baseline Financial Analysis
        self.logger.info(
            "Starting Nuclear Power Plant Baseline Financial Analysis...")
        try:
            # Find NPPs info file path
            project_root = Path(__file__).parent.parent.parent
            npps_info_path = project_root / "input" / "hourly_data" / "NPPs info.csv"
            if not npps_info_path.exists():
                npps_info_path = None
                self.logger.warning(
                    "NPPs info.csv not found, will use default values")

            baseline_nuclear_analysis = calculate_nuclear_baseline_financial_analysis(
                tea_sys_params=tea_sys_params,
                hourly_results_df=hourly_res_df,
                project_lifetime_config=config.PROJECT_LIFETIME_YEARS,
                construction_period_config=config.CONSTRUCTION_YEARS,
                discount_rate_config=config.DISCOUNT_RATE,
                tax_rate_config=config.TAX_RATE,
                target_iso=target_iso,
                npps_info_path=str(npps_info_path) if npps_info_path else None,
                tax_policies=config.TAX_INCENTIVE_POLICIES,
                hourly_results_file_path=getattr(
                    self, 'hourly_results_file_path', None)
            )

            if baseline_nuclear_analysis:
                annual_metrics_results["nuclear_baseline_analysis"] = baseline_nuclear_analysis
                self.logger.info(
                    "Nuclear baseline financial analysis completed successfully.")

        except Exception as e:
            self.logger.error(
                f"Error in nuclear baseline financial analysis: {e}")

        # Nuclear-Hydrogen Integrated System Analysis (Existing Plant Retrofit with 45U Policy)
        self.logger.info(
            "Starting Nuclear-Hydrogen Integrated System Analysis (Existing Plant Retrofit)...")
        try:
            # Use actual reactor capacity from annual metrics
            nuclear_capacity_from_params = tea_sys_params.get(
                "nuclear_plant_capacity_MW")
            nuclear_capacity_from_config = config.NUCLEAR_INTEGRATED_CONFIG.get(
                "nuclear_plant_capacity_mw", 1000)

            # Ensure we have a valid float value, with proper fallback chain
            if nuclear_capacity_from_params is not None:
                try:
                    nuclear_capacity_fallback = float(
                        nuclear_capacity_from_params)
                except (ValueError, TypeError):
                    nuclear_capacity_fallback = nuclear_capacity_from_config
            else:
                nuclear_capacity_fallback = nuclear_capacity_from_config

            actual_nuclear_capacity = annual_metrics_results.get(
                "Turbine_Capacity_MW", nuclear_capacity_fallback)
            self.logger.info(
                f"Using actual nuclear reactor capacity for retrofit analysis: {actual_nuclear_capacity:.1f} MW")

            # Use actual remaining years from tea_sys_params (extracted from filename or NPPs info)
            # This ensures Case 2 uses the correct plant lifetime for existing reactor retrofit
            actual_project_lifetime = tea_sys_params.get(
                'project_lifetime_years', config.PROJECT_LIFETIME_YEARS)
            if actual_project_lifetime != config.PROJECT_LIFETIME_YEARS:
                self.logger.info(
                    f"Case 2 Retrofit: Using actual remaining lifetime: {actual_project_lifetime} years")
                self.logger.info(
                    f"  (Override from filename/NPPs info, config default was: {config.PROJECT_LIFETIME_YEARS} years)")
            else:
                self.logger.info(
                    f"Case 2 Retrofit: Using config default lifetime: {actual_project_lifetime} years")

            integrated_nuclear_analysis = calculate_nuclear_integrated_financial_metrics(
                annual_metrics=annual_metrics_results,
                nuclear_capacity_mw=actual_nuclear_capacity,
                project_lifetime_config=actual_project_lifetime,  # Use actual remaining years
                construction_period_config=config.CONSTRUCTION_YEARS,
                discount_rate_config=config.DISCOUNT_RATE,
                tax_rate_config=config.TAX_RATE,
                h2_capex_components_config=config.CAPEX_COMPONENTS,
                h2_om_components_config=config.OM_COMPONENTS,
                h2_replacement_schedule_config=config.REPLACEMENT_SCHEDULE,
                tea_sys_params=tea_sys_params,
                enable_45u_policy=True  # Enable 45U policy analysis
            )

            if integrated_nuclear_analysis:
                annual_metrics_results["nuclear_integrated_analysis"] = integrated_nuclear_analysis
                self.logger.info(
                    "Nuclear-Hydrogen integrated system analysis completed successfully.")
                self.logger.info(
                    f"45U Policy Impact: NPV improvement of ${integrated_nuclear_analysis.get('45u_policy_impact', {}).get('npv_improvement_usd', 0):,.0f}")

        except Exception as e:
            self.logger.error(
                f"Error in nuclear integrated financial analysis: {e}")

        # Lifecycle Comparison Analysis (60-year vs 80-year) - Case 5 Analysis
        # This analysis compares new nuclear-hydrogen projects with 60-year vs 80-year lifecycles
        # It runs independently when greenfield analysis is enabled, regardless of current case type
        if enable_greenfield:
            self.logger.info(
                f"Lifecycle comparison check: enable_greenfield={enable_greenfield}")
            self.logger.info(
                "Starting Lifecycle Comparison Analysis (60-year vs 80-year) for greenfield nuclear-hydrogen projects...")

            try:
                lifecycle_comparison_results = calculate_lifecycle_comparison_analysis(
                    annual_metrics=annual_metrics_results,
                    nuclear_capacity_mw=actual_nuclear_capacity,
                    tea_sys_params=tea_sys_params,
                    hourly_results_df=hourly_res_df,
                    discount_rate_config=config.DISCOUNT_RATE,
                    tax_rate_config=config.TAX_RATE,
                    h2_capex_components_config=config.CAPEX_COMPONENTS,
                    h2_om_components_config=config.OM_COMPONENTS,
                    h2_replacement_schedule_config=config.REPLACEMENT_SCHEDULE,
                    macrs_config=config.MACRS_CONFIG,
                    output_dir=str(output_dir) if output_dir else None
                )

                if lifecycle_comparison_results:
                    annual_metrics_results["lifecycle_comparison_analysis"] = lifecycle_comparison_results
                    self.logger.info(
                        "Lifecycle comparison analysis (60-year vs 80-year) completed successfully.")

                    # Log key comparison metrics
                    comparison_summary = lifecycle_comparison_results.get(
                        "comparison_summary", {})
                    baseline_npv_diff = comparison_summary.get(
                        "baseline_npv_difference_usd", 0)
                    baseline_lcoh_diff = comparison_summary.get(
                        "baseline_lcoh_difference_usd_per_kg", 0)
                    best_60yr = comparison_summary.get(
                        "best_scenario_60yr", "N/A")
                    best_80yr = comparison_summary.get(
                        "best_scenario_80yr", "N/A")

                    self.logger.info(
                        f"Baseline NPV difference (80yr-60yr): ${baseline_npv_diff:,.0f}")
                    self.logger.info(
                        f"Baseline LCOH difference (80yr-60yr): ${baseline_lcoh_diff:.3f} USD/kg")
                    self.logger.info(
                        f"Best tax scenario for 60-year: {best_60yr}")
                    self.logger.info(
                        f"Best tax scenario for 80-year: {best_80yr}")
                else:
                    self.logger.warning(
                        "Lifecycle comparison analysis failed or returned no results.")

            except Exception as e:
                self.logger.error(
                    f"Error in lifecycle comparison analysis: {e}")
                # Log the full traceback for debugging
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
        else:
            self.logger.info(
                "Skipping lifecycle comparison analysis - greenfield analysis not enabled")

        # Incremental Analysis
        if enable_incremental:
            self.logger.info("Starting incremental analysis.")
            inc_capex_keys = ["Electrolyzer", "H2_Storage", "NPP"]
            if config.ENABLE_BATTERY:
                inc_capex_keys.append("Battery")
                self.logger.debug("Added 'Battery' to incremental CAPEX keys")

            inc_capex = {k: v for k, v in config.CAPEX_COMPONENTS.items() if any(
                sub in k for sub in inc_capex_keys)}

            # Log incremental CAPEX components for verification
            self.logger.debug(
                f"Incremental CAPEX filter keys: {inc_capex_keys}")
            self.logger.debug(
                f"Filtered incremental CAPEX components: {list(inc_capex.keys())}")
            for comp_name, comp_data in inc_capex.items():
                capacity_key = comp_data.get(
                    "applies_to_component_capacity_key")
                base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
                self.logger.debug(
                    f"  {comp_name}: capacity_key={capacity_key}, base_cost=${base_cost:,.0f}")

            inc_om = {"Fixed_OM_General": config.OM_COMPONENTS.get(
                "Fixed_OM_General", {})}
            if config.ENABLE_BATTERY:
                inc_om["Fixed_OM_Battery"] = config.OM_COMPONENTS.get(
                    "Fixed_OM_Battery", {})

            inc_repl = config.REPLACEMENT_SCHEDULE
            baseline_revenue_val = self.safe_float_from_params(
                tea_sys_params, "baseline_annual_revenue", 0.0)

            # Enhanced baseline revenue calculation
            if baseline_revenue_val <= 0:
                turbine_max_cap = self.safe_float_from_params(
                    tea_sys_params, "pTurbine_max_MW", annual_metrics_results.get("Turbine_Capacity_MW", 300.0))
                avg_lmp = annual_metrics_results.get(
                    "Avg_Electricity_Price_USD_per_MWh", 40)
                turbine_cf = annual_metrics_results.get(
                    "Turbine_CF_percent", 90) / 100

                if turbine_max_cap > 0 and avg_lmp > 0 and turbine_cf > 0:
                    baseline_revenue_val = turbine_max_cap * \
                        config.HOURS_IN_YEAR * turbine_cf * avg_lmp
                    self.logger.info(
                        f"Baseline nuclear revenue calculated from turbine capacity: ${baseline_revenue_val:,.2f}")

            if baseline_revenue_val > 0:
                optimized_caps = {
                    "Electrolyzer_Capacity_MW": annual_metrics_results.get("Electrolyzer_Capacity_MW", 0),
                    "H2_Storage_Capacity_kg": annual_metrics_results.get("H2_Storage_Capacity_kg", 0),
                    "Battery_Capacity_MWh": annual_metrics_results.get("Battery_Capacity_MWh", 0),
                    "Battery_Power_MW": annual_metrics_results.get("Battery_Power_MW", 0),
                }

                # Log optimized capacities for incremental analysis
                self.logger.debug(
                    f"Optimized capacities for incremental analysis: {optimized_caps}")

                try:
                    incremental_fin_metrics = calculate_incremental_metrics(
                        optimized_cash_flows=cash_flows_results,
                        baseline_annual_revenue=baseline_revenue_val,
                        project_lifetime_years=config.PROJECT_LIFETIME_YEARS,
                        construction_period_years=config.CONSTRUCTION_YEARS,
                        discount_rate=config.DISCOUNT_RATE,
                        tax_rate=config.TAX_RATE,
                        annual_metrics_optimized=annual_metrics_results,
                        capex_components_incremental=inc_capex,
                        om_components_incremental=inc_om,
                        replacement_schedule_incremental=inc_repl,
                        h2_subsidy_value=self.safe_float_from_params(
                            tea_sys_params, "hydrogen_subsidy_value_usd_per_kg", 0.0),
                        h2_subsidy_duration_years=int(self.safe_float_from_params(
                            tea_sys_params, "hydrogen_subsidy_duration_years", 10.0)),
                        optimized_capacities_inc=optimized_caps,
                    )
                    self.logger.info(
                        "Incremental metrics calculated successfully.")
                    return incremental_fin_metrics
                except Exception as e:
                    self.logger.error(
                        f"Error calculating incremental metrics: {e}")

        return None

    def generate_outputs(self,
                         annual_metrics_results: Dict,
                         financial_metrics_results: Dict,
                         cash_flows_results: Dict,
                         target_iso: str,
                         plant_report_title: str,
                         output_dir: Path,
                         incremental_metrics: Dict = None) -> bool:
        """
        Generate all output files and reports
        """
        if not self.logger:
            raise RuntimeError(
                "Logger not initialized. Call setup_logging first.")

        try:
            # Generate plots
            plot_output_dir = output_dir / f"Plots_{target_iso}"
            plot_output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Generating plots...")
            plot_results(
                annual_metrics_data=annual_metrics_results,
                financial_metrics_data=financial_metrics_results,
                cash_flows_data=cash_flows_results,
                plot_dir=plot_output_dir,
                construction_period_years=config.CONSTRUCTION_YEARS,
                incremental_metrics_data=incremental_metrics,
                case_type=self.case_type,
                project_lifetime_years=self.project_lifetime_years,
            )
            self.logger.info("Plots generated successfully.")

            # Generate original detailed report
            report_file = output_dir / f"{target_iso}_TEA_Summary_Report.txt"
            self.logger.info("Generating detailed TEA report...")
            generate_report(
                annual_metrics_rpt=annual_metrics_results,
                financial_metrics_rpt=financial_metrics_results,
                output_file_path=report_file,
                target_iso_rpt=target_iso,
                plant_specific_title_rpt=plant_report_title,
                capex_data=config.CAPEX_COMPONENTS,
                om_data=config.OM_COMPONENTS,
                replacement_data=config.REPLACEMENT_SCHEDULE,
                project_lifetime_years_rpt=config.PROJECT_LIFETIME_YEARS,
                construction_period_years_rpt=config.CONSTRUCTION_YEARS,
                discount_rate_rpt=config.DISCOUNT_RATE,
                tax_rate_rpt=config.TAX_RATE,
                incremental_metrics_rpt=incremental_metrics,
            )
            self.logger.info("Detailed report generated successfully.")

            # Generate comprehensive summary report (NEW)
            comprehensive_report_file = output_dir / \
                f"{target_iso}_Comprehensive_TEA_Summary.txt"
            self.logger.info("Generating comprehensive TEA summary report...")
            try:
                generate_comprehensive_tea_summary_report(
                    annual_metrics_rpt=annual_metrics_results,
                    financial_metrics_rpt=financial_metrics_results,
                    output_file_path=comprehensive_report_file,
                    target_iso_rpt=target_iso,
                    plant_specific_title_rpt=plant_report_title,
                    capex_data=config.CAPEX_COMPONENTS,
                    om_data=config.OM_COMPONENTS,
                    replacement_data=config.REPLACEMENT_SCHEDULE,
                    project_lifetime_years_rpt=config.PROJECT_LIFETIME_YEARS,
                    construction_period_years_rpt=config.CONSTRUCTION_YEARS,
                    discount_rate_rpt=config.DISCOUNT_RATE,
                    tax_rate_rpt=config.TAX_RATE,
                    incremental_metrics_rpt=incremental_metrics,
                )
                self.logger.info(
                    "Comprehensive summary report generated successfully.")
            except Exception as e:
                self.logger.warning(
                    f"Failed to generate comprehensive summary report: {e}")
                self.logger.info("Continuing with original report only...")

            self.logger.info("Report generation finished.")

            self.logger.info(
                f"--- Technical Economic Analysis completed successfully for {target_iso} ---")
            self.logger.info(f"Detailed Report: {report_file}")
            if comprehensive_report_file.exists():
                self.logger.info(
                    f"Comprehensive Summary: {comprehensive_report_file}")
            self.logger.info(f"Plots: {plot_output_dir}")

            return True

        except Exception as e:
            self.logger.error(f"Error generating outputs: {e}")
            return False


def run_complete_tea_analysis(
    target_iso: str,
    input_hourly_results_file: Path,
    output_dir: Path,
    plant_report_title: str = None,
    input_sys_data_dir: Path = None,
    plant_specific_params: Dict = None,
    enable_greenfield: bool = False,
    enable_incremental: bool = False,
    config_overrides: Dict = None,
    analysis_type: str = "general",
    log_dir: Path = None,
    case_type: str = None
) -> bool:
    """
    Complete TEA analysis workflow - convenience function

    Args:
        target_iso: Target ISO region
        input_hourly_results_file: Path to hourly results CSV
        output_dir: Output directory for results
        plant_report_title: Title for reports
        input_sys_data_dir: Directory containing system data files
        plant_specific_params: Plant-specific parameters
        enable_greenfield: Enable greenfield analysis
        enable_incremental: Enable incremental analysis
        config_overrides: Configuration overrides
        analysis_type: Type of analysis ("iso", "reactor", "general")
        log_dir: Directory for log files

    Returns:
        True if successful, False otherwise
    """

    # Initialize TEA engine
    engine = TEAEngine(analysis_type)

    # Setup logging
    if not log_dir:
        log_dir = output_dir.parent / "logs" / analysis_type
    log_dir.mkdir(parents=True, exist_ok=True)

    engine.setup_logging(log_dir, f"tea_{target_iso}", analysis_type)

    # Apply config overrides
    if config_overrides:
        engine.apply_config_overrides(config_overrides)

    # Determine input directory if not provided
    if not input_sys_data_dir:
        project_root = Path(__file__).parent.parent.parent
        input_sys_data_dir = project_root / "input" / "hourly_data"

    # Load and prepare data
    hourly_res_df, tea_sys_params, success = engine.load_and_prepare_data(
        target_iso, input_hourly_results_file, input_sys_data_dir, plant_specific_params, case_type
    )

    if not success:
        return False

    # Run core analysis
    annual_metrics, financial_metrics, cash_flows, success = engine.run_core_analysis(
        hourly_res_df, tea_sys_params, enable_greenfield, enable_incremental
    )

    if not success:
        return False

    # Run advanced analysis
    incremental_metrics = engine.run_advanced_analysis(
        annual_metrics, financial_metrics, cash_flows, tea_sys_params, hourly_res_df,
        target_iso, enable_greenfield, enable_incremental, output_dir
    )

    # Generate outputs
    if not plant_report_title:
        plant_report_title = target_iso

    success = engine.generate_outputs(
        annual_metrics, financial_metrics, cash_flows,
        target_iso, plant_report_title, output_dir, incremental_metrics
    )

    return success
