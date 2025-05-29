"""
Calculation functions for the TEA module.
"""

import logging
import math
import numpy as np
import numpy_financial as npf
import pandas as pd

# Attempt to import from tea.config, handling potential circular imports or module not found
try:
    from src.tea.config import (  # Changed to absolute import
        HOURS_IN_YEAR,
        ENABLE_BATTERY,
        TAX_RATE,
        # PROJECT_LIFETIME_YEARS, # These are passed as arguments or determined in main script
        # CONSTRUCTION_YEARS, # These are passed as arguments or determined in main script
        # DISCOUNT_RATE, # These are passed as arguments or determined in main script
        CAPEX_COMPONENTS,
        OM_COMPONENTS,
        REPLACEMENT_SCHEDULE,
        CONSTRUCTION_FINANCING
        # NUCLEAR_INTEGRATED_CONFIG, # This is used in nuclear specific calculations, not directly here
        # NUCLEAR_CAPEX_COMPONENTS, # This is used in nuclear specific calculations
        # NUCLEAR_OM_COMPONENTS, # This is used in nuclear specific calculations
        # NUCLEAR_REPLACEMENT_SCHEDULE # This is used in nuclear specific calculations
    )
except ImportError:
    # Fallback values if tea.config is not available during initial setup or specific contexts
    # This is a safeguard, ideally tea.config should always be resolvable
    HOURS_IN_YEAR = 8760
    ENABLE_BATTERY = False  # Default, will be overridden by actual config
    TAX_RATE = 0.21  # Default
    CAPEX_COMPONENTS = {}
    OM_COMPONENTS = {}
    REPLACEMENT_SCHEDULE = {}
    logger = logging.getLogger(__name__)
    logger.warning(
        "Could not import from tea.config. Using fallback default values for calculations.py")


logger = logging.getLogger(__name__)


def calculate_annual_metrics(df: pd.DataFrame, tea_sys_params: dict) -> dict | None:
    """Calculates comprehensive annual metrics from hourly results."""
    if df is None:
        logger.error(
            "Input DataFrame is None. Cannot calculate annual metrics.")
        return None
    metrics = {}
    try:
        num_hours = len(df)
        if num_hours == 0:
            logger.error("Hourly results DataFrame is empty.")
            return None

        # Ensure HOURS_IN_YEAR is available, either from import or local default
        try:
            current_hours_in_year = HOURS_IN_YEAR
        except NameError:
            logger.warning(
                "HOURS_IN_YEAR not imported from config, using fallback 8760 for annualization_factor")
            current_hours_in_year = 8760

        annualization_factor = (
            current_hours_in_year / num_hours if num_hours > 0 and current_hours_in_year > 0 else 1.0
        )

        metrics["Annual_Profit"] = df["Profit_Hourly_USD"].sum()
        metrics["Annual_Revenue"] = df["Revenue_Total_USD"].sum()
        metrics["Annual_Opex_Cost_from_Opt"] = df["Cost_HourlyOpex_Total_USD"].sum()
        metrics["Energy_Revenue"] = df.get(
            "Revenue_Energy_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["AS_Revenue"] = df.get(
            "Revenue_Ancillary_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Sales_Revenue"] = df.get(
            "Revenue_Hydrogen_Sales_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Subsidy_Revenue"] = df.get(
            "Revenue_Hydrogen_Subsidy_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Total_Revenue"] = (
            metrics["H2_Sales_Revenue"] + metrics["H2_Subsidy_Revenue"]
        )
        metrics["VOM_Turbine_Cost"] = df.get(
            "Cost_VOM_Turbine_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["VOM_Electrolyzer_Cost"] = df.get(
            "Cost_VOM_Electrolyzer_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["VOM_Battery_Cost"] = df.get(
            "Cost_VOM_Battery_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["Startup_Cost"] = df.get(
            "Cost_Startup_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["Water_Cost"] = df.get(
            "Cost_Water_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["Ramping_Cost"] = df.get(
            "Cost_Ramping_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Storage_Cycle_Cost"] = df.get(
            "Cost_Storage_Cycle_USD", pd.Series(0.0, dtype="float64")
        ).sum()
        metrics["H2_Production_kg_annual"] = (
            df["mHydrogenProduced_kg_hr"].sum() * annualization_factor
        )

        degradation_cols = [
            col for col in df.columns if "degradation" in col.lower()]
        if degradation_cols:
            for col in degradation_cols:
                metrics[f"{col}_avg"] = df[col].mean()
                metrics[f"{col}_end"] = df[col].iloc[-1] if not df[col].empty else 0
                logger.debug(
                    f"Added degradation metric - {col}_avg: {metrics[f'{col}_avg']}"
                )
                logger.debug(
                    f"Added degradation metric - {col}_end: {metrics[f'{col}_end']}"
                )

        user_spec_electrolyzer_cap = None
        if (
            "user_specified_electrolyzer_capacity_MW" in tea_sys_params
            and tea_sys_params["user_specified_electrolyzer_capacity_MW"] is not None
        ):
            try:
                user_spec_electrolyzer_cap = float(
                    tea_sys_params["user_specified_electrolyzer_capacity_MW"]
                )
                if user_spec_electrolyzer_cap > 0:
                    logger.debug(
                        f"Found user-specified electrolyzer capacity: {user_spec_electrolyzer_cap} MW"
                    )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid user-specified electrolyzer capacity value: {tea_sys_params['user_specified_electrolyzer_capacity_MW']}"
                )

        if (
            "Electrolyzer_Capacity_MW" in df.columns
            and not df["Electrolyzer_Capacity_MW"].empty
        ):
            opt_electrolyzer_cap = df["Electrolyzer_Capacity_MW"].iloc[0]
            logger.debug(
                f"Found electrolyzer capacity in results file: {opt_electrolyzer_cap} MW"
            )
            if (
                user_spec_electrolyzer_cap is not None
                and user_spec_electrolyzer_cap > 0
            ):
                if abs(opt_electrolyzer_cap - user_spec_electrolyzer_cap) > 1:
                    logger.warning(
                        f"Warning - electrolyzer capacity from results ({opt_electrolyzer_cap} MW) differs significantly from user-specified value ({user_spec_electrolyzer_cap} MW)"
                    )
                    metrics["Electrolyzer_Capacity_MW"] = user_spec_electrolyzer_cap
                    logger.debug(
                        f"Using user-specified electrolyzer capacity: {user_spec_electrolyzer_cap} MW"
                    )
                else:
                    metrics["Electrolyzer_Capacity_MW"] = opt_electrolyzer_cap
            else:
                metrics["Electrolyzer_Capacity_MW"] = opt_electrolyzer_cap
        elif user_spec_electrolyzer_cap is not None and user_spec_electrolyzer_cap > 0:
            metrics["Electrolyzer_Capacity_MW"] = user_spec_electrolyzer_cap
            logger.debug(
                f"Using user-specified electrolyzer capacity as fallback: {user_spec_electrolyzer_cap} MW"
            )
        else:
            default_cap = 0
            metrics["Electrolyzer_Capacity_MW"] = default_cap
            logger.debug(
                f"No electrolyzer capacity found in results or user parameters. Using default: {default_cap} MW"
            )
        logger.debug(
            f"Final electrolyzer capacity used for calculations: {metrics['Electrolyzer_Capacity_MW']} MW"
        )

        if "H2_Storage_Capacity_kg" in df.columns:
            metrics["H2_Storage_Capacity_kg"] = (
                df["H2_Storage_Capacity_kg"].iloc[0]
                if not df["H2_Storage_Capacity_kg"].empty
                else 0
            )
            logger.debug(
                f"H2 storage capacity from results: {metrics['H2_Storage_Capacity_kg']} kg"
            )
        else:
            user_spec_h2_storage = tea_sys_params.get(
                "user_specified_h2_storage_capacity_kg"
            )
            if user_spec_h2_storage is not None and not pd.isna(user_spec_h2_storage):
                try:
                    metrics["H2_Storage_Capacity_kg"] = float(
                        user_spec_h2_storage)
                    logger.debug(
                        f"Using user-specified H2 storage capacity: {metrics['H2_Storage_Capacity_kg']} kg"
                    )
                except (ValueError, TypeError):
                    metrics["H2_Storage_Capacity_kg"] = 0
                    logger.debug(
                        f"Invalid user-specified H2 storage value: {user_spec_h2_storage}. Using 0 kg"
                    )
            else:
                metrics["H2_Storage_Capacity_kg"] = 0
                logger.debug(
                    "H2_Storage_Capacity_kg column unexpectedly missing and no user value."
                )

        if "H2_Constant_Sales_Rate_kg_hr" in df.columns:
            metrics["H2_Constant_Sales_Rate_kg_hr"] = (
                df["H2_Constant_Sales_Rate_kg_hr"].iloc[0]
                if not df["H2_Constant_Sales_Rate_kg_hr"].empty
                else 0
            )
            logger.debug(
                f"H2 constant sales rate from results: {metrics['H2_Constant_Sales_Rate_kg_hr']} kg/hr"
            )
        else:
            metrics["H2_Constant_Sales_Rate_kg_hr"] = 0
            logger.debug("H2 constant sales rate not found in results")

        for summary_col in df.columns:
            if "Optimal_H2_Constant_Sales_Rate_kg_hr" in summary_col:
                optimal_rate = df[summary_col].iloc[0] if not df[summary_col].empty else 0
                if optimal_rate > 0:
                    metrics["Optimal_H2_Constant_Sales_Rate_kg_hr"] = optimal_rate
                    logger.debug(
                        f"Found optimal H2 constant sales rate: {optimal_rate} kg/hr")

        metrics["Battery_Capacity_MWh"] = (
            df["Battery_Capacity_MWh"].iloc[0]
            if "Battery_Capacity_MWh" in df and not df["Battery_Capacity_MWh"].empty
            else 0
        )
        metrics["Battery_Power_MW"] = (
            df["Battery_Power_MW"].iloc[0]
            if "Battery_Power_MW" in df and not df["Battery_Power_MW"].empty
            else 0
        )
        logger.debug(
            f"Battery capacity: {metrics['Battery_Capacity_MWh']} MWh, power: {metrics['Battery_Power_MW']} MW"
        )

        battery_soc_col = None
        possible_battery_soc_cols = [
            "BatterySOC_MWh", "Battery_SOC_MWh", "BatterySOC", "Battery_SOC", "SOC_Battery_MWh",
        ]
        battery_cols = [
            col for col in df.columns if "battery" in col.lower() or "soc" in col.lower()]
        logger.debug(f"Found battery-related columns: {battery_cols}")
        for col_name in possible_battery_soc_cols:
            if col_name in df.columns:
                battery_soc_col = col_name
                logger.debug(f"Found battery SOC column: {col_name}")
                break
        if battery_soc_col is not None and metrics["Battery_Capacity_MWh"] > 1e-6:
            metrics["Battery_SOC_percent"] = (
                df[battery_soc_col].mean() / metrics["Battery_Capacity_MWh"]
            ) * 100
            logger.debug(
                f"Battery average SOC calculated: {metrics['Battery_SOC_percent']}% from column {battery_soc_col}"
            )
        else:
            metrics["Battery_SOC_percent"] = 0
            logger.debug(
                "Battery SOC set to 0 (capacity is zero or SOC data missing)")

        h2_storage_soc_col = None
        possible_h2_soc_cols = [
            "H2_Storage_Level_kg", "H2_Storage_SOC_kg", "H2StorageSOC_kg", "H2StorageSOC", "H2_Storage_SOC", "mH2Storage_kg",
        ]
        h2_storage_cols = [col for col in df.columns if "h2" in col.lower() and (
            "storage" in col.lower() or "inventory" in col.lower())]
        logger.debug(f"Found H2 storage-related columns: {h2_storage_cols}")
        for col_name in possible_h2_soc_cols:
            if col_name in df.columns:
                h2_storage_soc_col = col_name
                logger.debug(f"Found H2 storage SOC column: {col_name}")
                break
        if h2_storage_soc_col is not None and metrics["H2_Storage_Capacity_kg"] > 1e-6:
            metrics["H2_Storage_SOC_percent"] = (
                df[h2_storage_soc_col].mean() / metrics["H2_Storage_Capacity_kg"]
            ) * 100
            logger.debug(
                f"H2 Storage average SOC calculated: {metrics['H2_Storage_SOC_percent']}% from column {h2_storage_soc_col}"
            )
        else:
            metrics["H2_Storage_SOC_percent"] = 0
            logger.debug(
                "H2 Storage SOC set to 0 (capacity is zero or SOC data missing)")

        if (
            metrics["Battery_Power_MW"] > 1e-6
            and "BatteryCharge_MW" in df
            and "BatteryDischarge_MW" in df
        ):
            avg_batt_usage = (df["BatteryCharge_MW"].mean(
            ) + df["BatteryDischarge_MW"].mean()) / 2
            metrics["Battery_CF_percent"] = (
                avg_batt_usage / metrics["Battery_Power_MW"]) * 100
        else:
            metrics["Battery_CF_percent"] = 0

        battery_charge_col = None
        possible_battery_charge_cols = [
            "BatteryCharge_MW", "Battery_Charge_MW", "BatteryCharge"]
        for col_name in possible_battery_charge_cols:
            if col_name in df.columns:
                battery_charge_col = col_name
                break
        if battery_charge_col is not None and len(df) > 0:
            battery_charge_mwh_annual = df[battery_charge_col].sum(
            ) * annualization_factor
            metrics["Annual_Battery_Charge_MWh"] = battery_charge_mwh_annual
            grid_purchase_col = None
            possible_grid_purchase_cols = [
                "pGridPurchase_MW", "pGridPurchase", "GridPurchase_MW"]
            for col_name in possible_grid_purchase_cols:
                if col_name in df.columns:
                    grid_purchase_col = col_name
                    break
            if grid_purchase_col is not None:
                battery_grid_charge_mask = (df[battery_charge_col] > 0) & (
                    df[grid_purchase_col] > 0)
                battery_charge_from_grid_mwh = df.loc[battery_grid_charge_mask, battery_charge_col].sum(
                ) * annualization_factor
                battery_npp_charge_mask = (df[battery_charge_col] > 0) & (
                    df[grid_purchase_col] <= 1e-6)
                battery_charge_from_npp_mwh = df.loc[battery_npp_charge_mask, battery_charge_col].sum(
                ) * annualization_factor
                metrics["Annual_Battery_Charge_From_Grid_MWh"] = battery_charge_from_grid_mwh
                metrics["Annual_Battery_Charge_From_NPP_MWh"] = battery_charge_from_npp_mwh
                logger.debug(
                    f"Battery charging breakdown: Total: {battery_charge_mwh_annual:.2f} MWh/year, From Grid: {battery_charge_from_grid_mwh:.2f} MWh/year, From NPP: {battery_charge_from_npp_mwh:.2f} MWh/year")
            else:
                metrics["Annual_Battery_Charge_From_Grid_MWh"] = 0.0
                metrics["Annual_Battery_Charge_From_NPP_MWh"] = battery_charge_mwh_annual
                logger.warning(
                    "No pGridPurchase_MW data. Assuming all battery charging from NPP.")
        else:
            metrics["Annual_Battery_Charge_MWh"] = 0.0
            metrics["Annual_Battery_Charge_From_Grid_MWh"] = 0.0
            metrics["Annual_Battery_Charge_From_NPP_MWh"] = 0.0

        metrics["Turbine_Capacity_MW"] = (
            df.get("Turbine_Capacity_MW", pd.Series(
                0.0, dtype="float64")).iloc[0]
            if "Turbine_Capacity_MW" in df and not df["Turbine_Capacity_MW"].empty
            else 0
        )
        if metrics["Turbine_Capacity_MW"] <= 1e-6:
            if ("pTurbine_max_MW" in tea_sys_params and tea_sys_params["pTurbine_max_MW"] is not None):
                try:
                    user_spec_turbine_cap = float(
                        tea_sys_params["pTurbine_max_MW"])
                    if user_spec_turbine_cap > 0:
                        metrics["Turbine_Capacity_MW"] = user_spec_turbine_cap
                        logger.debug(
                            f"Using pTurbine_max_MW from sys_params as Turbine capacity: {user_spec_turbine_cap} MW")
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid pTurbine_max_MW value: {tea_sys_params['pTurbine_max_MW']}")

        if "pTurbine_MW" in df.columns:
            logger.debug(
                f"pTurbine_MW data available - Min: {df['pTurbine_MW'].min()}, Max: {df['pTurbine_MW'].max()}, Mean: {df['pTurbine_MW'].mean()}")
            if metrics["Turbine_Capacity_MW"] <= 1e-6:
                if df["pTurbine_MW"].max() > 0:
                    metrics["Turbine_Capacity_MW"] = df["pTurbine_MW"].max()
                    logger.debug(
                        f"Using maximum observed pTurbine_MW as capacity: {metrics['Turbine_Capacity_MW']} MW")
                elif ("pTurbine_max_MW" in tea_sys_params and tea_sys_params["pTurbine_max_MW"] is not None):
                    try:
                        turbine_max = float(tea_sys_params["pTurbine_max_MW"])
                        if turbine_max > 0:
                            metrics["Turbine_Capacity_MW"] = turbine_max
                            logger.debug(
                                f"Forced setting of Turbine capacity to pTurbine_max_MW: {turbine_max} MW")
                    except (ValueError, TypeError):
                        pass  # Already warned above
            if metrics["Turbine_Capacity_MW"] > 1e-6:
                metrics["Turbine_CF_percent"] = (
                    df["pTurbine_MW"].mean() / metrics["Turbine_Capacity_MW"]) * 100
                logger.debug(
                    f"Turbine CF calculated: {metrics['Turbine_CF_percent']}% (Capacity: {metrics['Turbine_Capacity_MW']} MW)")
            else:
                metrics["Turbine_CF_percent"] = 0
                logger.debug(
                    "Turbine CF set to 0 (valid capacity couldn't be determined)")
        else:
            metrics["Turbine_CF_percent"] = 0
            logger.debug("Turbine CF set to 0 (pTurbine_MW column not found)")

        metrics["Annual_Electrolyzer_MWh"] = (df["pElectrolyzer_MW"].sum(
        ) * annualization_factor if "pElectrolyzer_MW" in df else 0)
        if ("pElectrolyzer_MW" in df.columns and metrics["Electrolyzer_Capacity_MW"] > 1e-6):
            metrics["Electrolyzer_CF_percent"] = (
                df["pElectrolyzer_MW"].mean() / metrics["Electrolyzer_Capacity_MW"]) * 100
            logger.debug(
                f"Electrolyzer CF calculated: {metrics['Electrolyzer_CF_percent']}%")
        else:
            metrics["Electrolyzer_CF_percent"] = 0
            logger.debug(
                "Electrolyzer CF set to 0 (capacity or power data missing)")

        if "EnergyPrice_LMP_USDperMWh" in df.columns:
            metrics["Avg_Electricity_Price_USD_per_MWh"] = df["EnergyPrice_LMP_USDperMWh"].mean()
            if "pElectrolyzer_MW" in df.columns and df["pElectrolyzer_MW"].sum() > 0:
                weighted_price = (df["EnergyPrice_LMP_USDperMWh"] *
                                  df["pElectrolyzer_MW"]).sum() / df["pElectrolyzer_MW"].sum()
                metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = weighted_price
            else:
                metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = metrics["Avg_Electricity_Price_USD_per_MWh"]
        else:
            metrics["Avg_Electricity_Price_USD_per_MWh"] = 40.0
            metrics["Weighted_Avg_Electricity_Price_USD_per_MWh"] = 40.0

        if "Thermal_Capacity_MWt" in df.columns and not df["Thermal_Capacity_MWt"].empty:
            metrics["thermal_capacity_mwt"] = df["Thermal_Capacity_MWt"].iloc[0]
            logger.debug(
                f"Thermal capacity from results: {metrics['thermal_capacity_mwt']} MWt")
        elif "thermal_capacity_mwt" in tea_sys_params and tea_sys_params["thermal_capacity_mwt"] is not None:
            try:
                metrics["thermal_capacity_mwt"] = float(
                    tea_sys_params["thermal_capacity_mwt"])
                logger.debug(
                    f"Thermal capacity from sys_params: {metrics['thermal_capacity_mwt']} MWt")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid thermal capacity value in sys_params: {tea_sys_params['thermal_capacity_mwt']}")
                metrics["thermal_capacity_mwt"] = 0
        else:
            metrics["thermal_capacity_mwt"] = 0

        if "Thermal_Efficiency" in df.columns and not df["Thermal_Efficiency"].empty:
            metrics["thermal_efficiency"] = df["Thermal_Efficiency"].iloc[0]
            logger.debug(
                f"Thermal efficiency from results: {metrics['thermal_efficiency']:.4f}")
        elif "thermal_efficiency" in tea_sys_params and tea_sys_params["thermal_efficiency"] is not None:
            try:
                metrics["thermal_efficiency"] = float(
                    tea_sys_params["thermal_efficiency"])
                logger.debug(
                    f"Thermal efficiency from sys_params: {metrics['thermal_efficiency']:.4f}")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid thermal efficiency value in sys_params: {tea_sys_params['thermal_efficiency']}")
                metrics["thermal_efficiency"] = 0
        else:
            metrics["thermal_efficiency"] = 0

        if "Revenue_Ancillary_USD" in df.columns:
            as_revenue_hourly = df["Revenue_Ancillary_USD"]
            metrics["AS_Revenue_Total"] = as_revenue_hourly.sum()
            metrics["AS_Revenue_Average_Hourly"] = as_revenue_hourly.mean()
            metrics["AS_Revenue_Maximum_Hourly"] = as_revenue_hourly.max()
            metrics["AS_Revenue_Hours_Positive"] = (
                as_revenue_hourly > 0).sum()
            metrics["AS_Revenue_Utilization_Rate"] = (metrics["AS_Revenue_Hours_Positive"] / len(
                as_revenue_hourly) * 100 if len(as_revenue_hourly) > 0 else 0)
            if metrics["Electrolyzer_Capacity_MW"] > 0:
                metrics["AS_Revenue_per_MW_Electrolyzer"] = metrics["AS_Revenue_Total"] / \
                    metrics["Electrolyzer_Capacity_MW"]
            if metrics["Battery_Power_MW"] > 0:
                metrics["AS_Revenue_per_MW_Battery"] = metrics["AS_Revenue_Total"] / \
                    metrics["Battery_Power_MW"]
            logger.debug(
                f"AS Revenue statistics calculated: Total ${metrics['AS_Revenue_Total']:,.2f}, Utilization {metrics['AS_Revenue_Utilization_Rate']:.1f}%")

        as_bid_columns = [
            col for col in df.columns if "_Bid_MW" in col and "Total_" in col]
        if as_bid_columns:
            metrics["AS_Total_Bid_Services"] = len(as_bid_columns)
            total_bid_capacity = 0
            for bid_col in as_bid_columns:
                service_max_bid = df[bid_col].max()
                total_bid_capacity += service_max_bid
                service_name = bid_col.replace(
                    "Total_", "").replace("_Bid_MW", "")
                metrics[f"AS_Max_Bid_{service_name}_MW"] = service_max_bid
                metrics[f"AS_Avg_Bid_{service_name}_MW"] = df[bid_col].mean()
            metrics["AS_Total_Max_Bid_Capacity_MW"] = total_bid_capacity
            if metrics["Electrolyzer_Capacity_MW"] > 0:
                metrics["AS_Bid_Utilization_vs_Electrolyzer"] = (
                    total_bid_capacity / metrics["Electrolyzer_Capacity_MW"] * 100)

        as_deployed_columns = [
            col for col in df.columns if "_Deployed_MW" in col]
        if as_deployed_columns:
            total_deployed_energy = 0
            for deployed_col in as_deployed_columns:
                service_deployed_total = df[deployed_col].sum(
                ) * annualization_factor
                total_deployed_energy += service_deployed_total
                service_name = deployed_col.replace("_Deployed_MW", "")
                metrics[f"AS_Total_Deployed_{service_name}_MWh"] = service_deployed_total
                metrics[f"AS_Avg_Deployed_{service_name}_MW"] = df[deployed_col].mean(
                )
            metrics["AS_Total_Deployed_Energy_MWh"] = total_deployed_energy
            for deployed_col in as_deployed_columns:
                service_base = deployed_col.replace("_Deployed_MW", "")
                corresponding_bid_col = f"Total_{service_base}_Bid_MW"
                if corresponding_bid_col in df.columns:
                    avg_deployed = df[deployed_col].mean()
                    avg_bid = df[corresponding_bid_col].mean()
                    if avg_bid > 0:
                        metrics[f"AS_Deployment_Efficiency_{service_base}_percent"] = (
                            avg_deployed / avg_bid) * 100

        metrics["HTE_Heat_Opportunity_Cost_Annual_USD"] = 0.0
        metrics["HTE_Steam_Consumption_Annual_MWth"] = 0.0
        metrics["HTE_Mode_Detected"] = False
        if "qSteam_Electrolyzer_MWth" in df.columns:
            steam_consumption_hourly = df["qSteam_Electrolyzer_MWth"]
            total_steam_consumption_annual = steam_consumption_hourly.sum() * \
                annualization_factor
            if total_steam_consumption_annual > 1.0:
                metrics["HTE_Mode_Detected"] = True
                metrics["HTE_Steam_Consumption_Annual_MWth"] = total_steam_consumption_annual
                thermal_efficiency = metrics.get("thermal_efficiency", 0.0)
                avg_electricity_price = metrics.get(
                    "Avg_Electricity_Price_USD_per_MWh", 40.0)
                if thermal_efficiency > 0:
                    lost_electricity_generation_mwh = total_steam_consumption_annual * thermal_efficiency
                    heat_opportunity_cost_annual = lost_electricity_generation_mwh * avg_electricity_price
                    metrics["HTE_Heat_Opportunity_Cost_Annual_USD"] = heat_opportunity_cost_annual
                    metrics["HTE_Lost_Electricity_Generation_Annual_MWh"] = lost_electricity_generation_mwh
                    logger.info(
                        f"HTE Heat Opportunity Cost: Annual steam {total_steam_consumption_annual:,.1f} MWth, Efficiency {thermal_efficiency:.4f}, Avg price ${avg_electricity_price:.2f}/MWh, Lost gen {lost_electricity_generation_mwh:,.1f} MWh, Cost ${heat_opportunity_cost_annual:,.2f}/year")
                    h2_production_annual = metrics.get(
                        "H2_Production_kg_annual", 0)
                    if h2_production_annual > 0:
                        metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = heat_opportunity_cost_annual / \
                            h2_production_annual
                        logger.info(
                            f"  Heat opportunity cost per kg H2: ${metrics['HTE_Heat_Opportunity_Cost_USD_per_kg_H2']:.3f}/kg")
                    else:
                        metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = 0.0
                else:
                    logger.warning(
                        "HTE detected but thermal efficiency is 0. Cannot calculate heat opportunity cost.")
                    metrics["HTE_Heat_Opportunity_Cost_USD_per_kg_H2"] = 0.0
                    metrics["HTE_Lost_Electricity_Generation_Annual_MWh"] = 0.0
            else:
                logger.debug(
                    "LTE mode detected (no significant steam consumption)")
        else:
            logger.debug("No steam consumption data found - likely LTE mode")

    except KeyError as e:
        logger.error(
            f"Missing column in hourly results for annual metrics: {e}")
        return None
    except Exception as e:
        logger.error(f"Error calculating annual metrics: {e}", exc_info=True)
        return None
    return metrics


def calculate_lcoh_breakdown(
    annual_metrics: dict,
    capex_breakdown: dict,
    project_lifetime: int,
    construction_period: int,
    discount_rate: float,
    annual_h2_production_kg: float,
) -> dict:
    logger.info("Calculating detailed LCOH breakdown by cost factors")
    if annual_h2_production_kg <= 0:
        logger.warning(
            "Annual H2 production is zero or negative. Cannot calculate LCOH breakdown.")
        return {}

    pv_total_h2_production = 0
    for op_idx in range(project_lifetime - construction_period):
        year_idx = op_idx + construction_period
        pv_factor = (1 + discount_rate) ** year_idx
        pv_total_h2_production += annual_h2_production_kg / pv_factor

    if pv_total_h2_production <= 0:
        logger.warning(
            "PV of H2 production is zero or negative. Cannot calculate LCOH breakdown.")
        return {}

    lcoh_breakdown = {}
    if discount_rate > 0:
        crf = (discount_rate * (1 + discount_rate) ** project_lifetime) / \
            ((1 + discount_rate) ** project_lifetime - 1)
    else:
        crf = 1 / project_lifetime

    capex_lcoh_components = {}
    for component, capex_cost in capex_breakdown.items():
        if capex_cost > 0:
            annualized_capex = capex_cost * crf
            capex_lcoh_per_kg = annualized_capex / annual_h2_production_kg
            capex_lcoh_components[f"CAPEX_{component}"] = capex_lcoh_per_kg
            logger.debug(
                f"   {component}: ${capex_cost:,.0f} -> ${capex_lcoh_per_kg:.3f}/kg H2")

    annual_fixed_om_costs = annual_metrics.get("annual_fixed_om_costs", [])
    if annual_fixed_om_costs:
        pv_fixed_om_total = sum(cost / (1 + discount_rate)**(idx + construction_period)
                                for idx, cost in enumerate(annual_fixed_om_costs))
        lcoh_breakdown["Fixed_OM"] = pv_fixed_om_total / pv_total_h2_production
        logger.debug(
            f"   Fixed O&M: PV ${pv_fixed_om_total:,.0f} -> ${lcoh_breakdown['Fixed_OM']:.3f}/kg H2")
    else:  # Fallback
        total_capex_val = annual_metrics.get(
            "total_capex", 0)  # Ensure total_capex is a float
        fixed_om_rate = 0.02
        annual_fixed_om = float(total_capex_val) * fixed_om_rate
        lcoh_breakdown["Fixed_OM"] = annual_fixed_om / annual_h2_production_kg
        logger.debug(
            f"   Fixed O&M (estimated): ${annual_fixed_om:,.0f}/year -> ${lcoh_breakdown['Fixed_OM']:.3f}/kg H2")

    variable_opex_components = {
        "VOM_Electrolyzer": annual_metrics.get("VOM_Electrolyzer_Cost", 0),
        "VOM_Battery": annual_metrics.get("VOM_Battery_Cost", 0),
        "Water_Cost": annual_metrics.get("Water_Cost", 0),
        "Startup_Cost": annual_metrics.get("Startup_Cost", 0),
        "Ramping_Cost": annual_metrics.get("Ramping_Cost", 0),
        "H2_Storage_Cycle_Cost": annual_metrics.get("H2_Storage_Cycle_Cost", 0),
    }
    for component, annual_cost in variable_opex_components.items():
        if annual_cost > 0:
            cost_per_kg = annual_cost / annual_h2_production_kg
            lcoh_breakdown[component] = cost_per_kg
            logger.debug(
                f"   {component}: ${annual_cost:,.0f}/year -> ${cost_per_kg:.3f}/kg H2")

    annual_electrolyzer_mwh = annual_metrics.get("Annual_Electrolyzer_MWh", 0)
    avg_electricity_price = annual_metrics.get(
        "Avg_Electricity_Price_USD_per_MWh", 40.0)
    electrolyzer_opportunity_cost = annual_electrolyzer_mwh * avg_electricity_price
    if electrolyzer_opportunity_cost > 0:
        lcoh_breakdown["Electricity_Opportunity_Cost_Electrolyzer"] = electrolyzer_opportunity_cost / \
            annual_h2_production_kg
        logger.debug(
            f"   Electrolyzer Elec Opp Cost: ${electrolyzer_opportunity_cost:,.0f}/yr -> ${lcoh_breakdown['Electricity_Opportunity_Cost_Electrolyzer']:.3f}/kg H2")

    battery_npp_charge_mwh = annual_metrics.get(
        "Annual_Battery_Charge_From_NPP_MWh", 0)
    battery_opportunity_cost = battery_npp_charge_mwh * avg_electricity_price
    if battery_opportunity_cost > 0:
        lcoh_breakdown["Electricity_Opportunity_Cost_Battery"] = battery_opportunity_cost / \
            annual_h2_production_kg
        logger.debug(
            f"   Battery Elec Opp Cost: ${battery_opportunity_cost:,.0f}/yr -> ${lcoh_breakdown['Electricity_Opportunity_Cost_Battery']:.3f}/kg H2")

    battery_grid_charge_mwh = annual_metrics.get(
        "Annual_Battery_Charge_From_Grid_MWh", 0)
    battery_direct_cost = battery_grid_charge_mwh * avg_electricity_price
    if battery_direct_cost > 0:
        lcoh_breakdown["Electricity_Direct_Cost_Battery"] = battery_direct_cost / \
            annual_h2_production_kg
        logger.debug(
            f"   Battery Direct Elec Cost: ${battery_direct_cost:,.0f}/yr -> ${lcoh_breakdown['Electricity_Direct_Cost_Battery']:.3f}/kg H2")

    hte_heat_opportunity_cost = annual_metrics.get(
        "HTE_Heat_Opportunity_Cost_Annual_USD", 0)
    if hte_heat_opportunity_cost > 0:
        lcoh_breakdown["HTE_Heat_Opportunity_Cost"] = hte_heat_opportunity_cost / \
            annual_h2_production_kg
        logger.debug(
            f"   HTE Heat Opp Cost: ${hte_heat_opportunity_cost:,.0f}/yr -> ${lcoh_breakdown['HTE_Heat_Opportunity_Cost']:.3f}/kg H2")

    annual_stack_replacement_costs = annual_metrics.get(
        "annual_stack_replacement_costs", [])
    if annual_stack_replacement_costs:
        pv_stack_replacement_total = sum(cost / (1 + discount_rate)**(idx + construction_period)
                                         for idx, cost in enumerate(annual_stack_replacement_costs) if cost > 0)
        if pv_stack_replacement_total > 0:
            lcoh_breakdown["Stack_Replacement"] = pv_stack_replacement_total / \
                pv_total_h2_production
            logger.debug(
                f"   Stack Replacement: PV ${pv_stack_replacement_total:,.0f} -> ${lcoh_breakdown['Stack_Replacement']:.3f}/kg H2")

    annual_other_replacement_costs = annual_metrics.get(
        "annual_other_replacement_costs", [])
    if annual_other_replacement_costs:
        pv_other_replacement_total = sum(cost / (1 + discount_rate)**(idx + construction_period)
                                         for idx, cost in enumerate(annual_other_replacement_costs) if cost > 0)
        if pv_other_replacement_total > 0:
            lcoh_breakdown["Other_Replacements"] = pv_other_replacement_total / \
                pv_total_h2_production
            logger.debug(
                f"   Other Replacements: PV ${pv_other_replacement_total:,.0f} -> ${lcoh_breakdown['Other_Replacements']:.3f}/kg H2")

    lcoh_breakdown.update(capex_lcoh_components)
    total_lcoh = sum(lcoh_breakdown.values())
    lcoh_percentages = {comp: (cost / total_lcoh) * 100 for comp,
                        cost in lcoh_breakdown.items()} if total_lcoh > 0 else {}

    lcoh_analysis = {
        "total_lcoh_usd_per_kg": total_lcoh,
        "lcoh_breakdown_usd_per_kg": lcoh_breakdown,
        "lcoh_percentages": lcoh_percentages,
        "pv_total_h2_production_kg": pv_total_h2_production,
    }

    logger.info(
        f"LCOH Breakdown: Total ${total_lcoh:.3f}/kg H2. Components: {len(lcoh_breakdown)}")
    sorted_components = sorted(
        lcoh_breakdown.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"   Top 5 LCOH contributors:")
    for i, (component, cost) in enumerate(sorted_components[:5]):
        logger.info(
            f"     {i+1}. {component}: ${cost:.3f}/kg ({lcoh_percentages.get(component, 0):.1f}%)")

    sensitivity_range = 0.20
    sensitivity_analysis = {}
    for component, base_cost in sorted_components[:5]:  # Sensitivity for top 5
        comp_sensitivity = {}
        for change_pct in [-sensitivity_range, sensitivity_range]:
            adjusted_cost = base_cost * (1 + change_pct)
            cost_diff = adjusted_cost - base_cost
            new_total_lcoh = total_lcoh + cost_diff
            lcoh_change = new_total_lcoh - total_lcoh
            comp_sensitivity[f"{change_pct*100:+.0f}%"] = {
                "lcoh_change": lcoh_change, "new_total_lcoh": new_total_lcoh,
                "impact_percentage": (lcoh_change / total_lcoh) * 100 if total_lcoh else 0
            }
        sensitivity_analysis[component] = comp_sensitivity
    lcoh_analysis["sensitivity_analysis"] = sensitivity_analysis
    logger.debug(
        f"Sensitivity analysis for {len(sensitivity_analysis)} components completed.")
    return lcoh_analysis


def calculate_cash_flows(
    annual_metrics: dict,
    project_lifetime: int,  # project_lifetime from config or params
    construction_period: int,  # construction_period from config or params
    h2_subsidy_value: float,
    h2_subsidy_duration: int,
    capex_details: dict,  # CAPEX_COMPONENTS from config
    om_details: dict,  # OM_COMPONENTS from config
    replacement_details: dict,  # REPLACEMENT_SCHEDULE from config
    optimized_capacities: dict,
    tax_rate: float  # TAX_RATE from config
) -> np.ndarray:
    logger.info(
        f"Calculating cash flows for {project_lifetime} years. Construction: {construction_period} years.")
    logger.debug(f"Optimized capacities for cash flow: {optimized_capacities}")

    cash_flows_array = np.zeros(project_lifetime + construction_period)
    total_capex_sum_after_learning = 0
    initial_battery_capex_energy = 0
    initial_battery_capex_power = 0
    initial_electrolyzer_capex = 0
    capex_breakdown = {}

    for comp_name, comp_data in capex_details.items():
        base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
        ref_cap = comp_data.get("reference_total_capacity_mw", 0)
        lr_dec = comp_data.get("learning_rate_decimal", 0)
        cap_key = comp_data.get("applies_to_component_capacity_key")
        pay_sched = comp_data.get("payment_schedule_years", {})
        actual_opt_cap = optimized_capacities.get(
            cap_key, ref_cap if cap_key else 0)

        adj_total_comp_cost = 0.0
        if cap_key and actual_opt_cap == 0 and ref_cap > 0:
            adj_total_comp_cost = 0.0
        elif lr_dec > 0 and ref_cap > 0 and actual_opt_cap > 0 and cap_key:
            prog_ratio = 1 - lr_dec
            b_exp = math.log(prog_ratio) / \
                math.log(2) if 0 < prog_ratio < 1 else 0
            scale_f = actual_opt_cap / ref_cap
            adj_total_comp_cost = base_cost * (scale_f**b_exp)
        elif actual_opt_cap > 0 and ref_cap > 0 and cap_key:
            adj_total_comp_cost = base_cost * (actual_opt_cap / ref_cap)
        elif not cap_key:  # Fixed cost component
            adj_total_comp_cost = base_cost

        capex_breakdown[comp_name.replace("_", " ")] = adj_total_comp_cost
        if comp_name == "Battery_System_Energy":
            initial_battery_capex_energy = adj_total_comp_cost
        if comp_name == "Battery_System_Power":
            initial_battery_capex_power = adj_total_comp_cost
        if comp_name == "Electrolyzer_System":
            initial_electrolyzer_capex = adj_total_comp_cost

        total_capex_sum_after_learning += adj_total_comp_cost
        for constr_yr_offset, share in pay_sched.items():
            proj_yr_idx = construction_period + constr_yr_offset
            if 0 <= proj_yr_idx < construction_period:
                cash_flows_array[proj_yr_idx] -= (adj_total_comp_cost * share)
            else:
                logger.warning(
                    f"Payment year {constr_yr_offset} for {comp_name} outside construction period.")

    annual_metrics["capex_breakdown"] = capex_breakdown
    annual_metrics["total_capex"] = total_capex_sum_after_learning
    annual_metrics["electrolyzer_capex"] = initial_electrolyzer_capex
    logger.info(
        f"Total CAPEX after learning/scaling: ${total_capex_sum_after_learning:,.2f}")
    initial_total_battery_capex = initial_battery_capex_energy + \
        initial_battery_capex_power

    base_annual_profit = annual_metrics.get(
        "Annual_Revenue", 0) - annual_metrics.get("Annual_Opex_Cost_from_Opt", 0)
    annual_fixed_om_costs = []
    annual_stack_replacement_costs = []
    annual_other_replacement_costs = []

    for op_yr_idx in range(project_lifetime - construction_period):
        curr_proj_yr_idx = op_yr_idx + construction_period
        op_yr_num = op_yr_idx + 1
        curr_yr_profit_pre_fixed_om_tax = base_annual_profit
        if op_yr_num > h2_subsidy_duration:
            curr_yr_profit_pre_fixed_om_tax -= annual_metrics.get(
                "H2_Subsidy_Revenue", 0)

        hte_opp_cost_annual = annual_metrics.get(
            "HTE_Heat_Opportunity_Cost_Annual_USD", 0.0)
        if hte_opp_cost_annual > 0:
            curr_yr_profit_pre_fixed_om_tax -= hte_opp_cost_annual

        fixed_om_general_cost = 0
        if om_details.get("Fixed_OM_General", {}).get("base_cost_percent_of_capex", 0) > 0:
            fixed_om_pct = om_details["Fixed_OM_General"].get(
                "base_cost_percent_of_capex", 0.02)
            fixed_om_general_cost = total_capex_sum_after_learning * fixed_om_pct * \
                ((1 + om_details["Fixed_OM_General"].get("inflation_rate", 0))**op_yr_idx)
        else:  # Fallback to old method if percent of capex not defined
            fixed_om_general_cost = om_details.get("Fixed_OM_General", {}).get(
                "base_cost", 0) * ((1 + om_details.get("Fixed_OM_General", {}).get("inflation_rate", 0))**op_yr_idx)

        current_fixed_om_total_for_year = fixed_om_general_cost  # Start with general OM
        curr_yr_profit_pre_fixed_om_tax -= fixed_om_general_cost

        # Use ENABLE_BATTERY from config
        enable_battery_flag = ENABLE_BATTERY
        if enable_battery_flag and optimized_capacities.get("Battery_Capacity_MWh", 0) > 0:
            batt_fixed_om_mw = om_details.get(
                "Fixed_OM_Battery", {}).get("base_cost_per_mw_year", 0)
            batt_fixed_om_mwh = om_details.get(
                "Fixed_OM_Battery", {}).get("base_cost_per_mwh_year", 0)
            batt_inflation = om_details.get(
                "Fixed_OM_Battery", {}).get("inflation_rate", 0)
            batt_power_mw_opt = optimized_capacities.get("Battery_Power_MW", 0)
            batt_cap_mwh_opt = optimized_capacities.get(
                "Battery_Capacity_MWh", 0)
            batt_fixed_om_cost_yr = (batt_power_mw_opt * batt_fixed_om_mw +
                                     batt_cap_mwh_opt * batt_fixed_om_mwh) * ((1 + batt_inflation)**op_yr_idx)
            current_fixed_om_total_for_year += batt_fixed_om_cost_yr
            curr_yr_profit_pre_fixed_om_tax -= batt_fixed_om_cost_yr

        # Store total fixed OM for the year
        annual_fixed_om_costs.append(current_fixed_om_total_for_year)

        replacement_cost_yr = 0
        stack_replacement_cost_yr = 0
        other_replacement_cost_yr = 0
        for rep_comp, rep_data in replacement_details.items():
            if op_yr_num in rep_data.get("years", []):
                cost_val = 0
                if rep_comp == "Electrolyzer_Stack" and "cost_percent_initial_capex" in rep_data:
                    cost_val = initial_electrolyzer_capex * \
                        rep_data.get("cost_percent_initial_capex", 0.30)
                    stack_replacement_cost_yr += cost_val
                elif rep_comp == "Battery_Augmentation_Replacement" and rep_data.get("cost_percent_initial_capex", 0) > 0:
                    cost_val = initial_total_battery_capex * \
                        rep_data["cost_percent_initial_capex"]
                    other_replacement_cost_yr += cost_val
                else:  # Fixed cost replacement
                    cost_val = rep_data.get("cost", 0)
                    other_replacement_cost_yr += cost_val
                replacement_cost_yr += cost_val

        annual_stack_replacement_costs.append(stack_replacement_cost_yr)
        annual_other_replacement_costs.append(other_replacement_cost_yr)
        curr_yr_profit_pre_fixed_om_tax -= replacement_cost_yr

        taxable_income = curr_yr_profit_pre_fixed_om_tax
        tax_amt = taxable_income * \
            tax_rate if taxable_income > 0 else 0  # Use tax_rate argument
        cash_flows_array[curr_proj_yr_idx] = taxable_income - tax_amt

    annual_metrics["annual_fixed_om_costs"] = annual_fixed_om_costs
    annual_metrics["annual_stack_replacement_costs"] = annual_stack_replacement_costs
    annual_metrics["annual_other_replacement_costs"] = annual_other_replacement_costs
    return cash_flows_array


def calculate_financial_metrics(
    cash_flows_input: np.ndarray,
    discount_rt: float,  # discount_rt from config or params
    annual_h2_prod_kg: float,
    project_lt: int,  # project_lt from config or params
    construction_p: int,  # construction_p from config or params
) -> dict:
    metrics_results = {}
    cf_array = np.array(cash_flows_input, dtype=float)
    try:
        metrics_results["NPV_USD"] = npf.npv(discount_rt, cf_array)
    except Exception:
        # Changed from 0 to np.nan for consistency
        metrics_results["NPV_USD"] = np.nan
    try:
        # IRR requires at least one positive and one negative cash flow
        if any(cf > 0 for cf in cf_array) and any(cf < 0 for cf in cf_array):
            metrics_results["IRR_percent"] = npf.irr(cf_array) * 100
        else:
            # Or some indicator that IRR is not applicable
            metrics_results["IRR_percent"] = np.nan
    except Exception:  # Catch any error from npf.irr
        metrics_results["IRR_percent"] = np.nan

    cumulative_cash_flow = np.cumsum(cf_array)
    positive_indices = np.where(cumulative_cash_flow >= 0)[0]
    if positive_indices.size > 0:
        first_positive_idx = positive_indices[0]
        # Payback in year 0 if CF starts positive
        if first_positive_idx == 0 and cf_array[0] >= 0:
            metrics_results["Payback_Period_Years"] = 0
        # Ensure previous CF was negative
        elif first_positive_idx > 0 and cumulative_cash_flow[first_positive_idx - 1] < 0:
            metrics_results["Payback_Period_Years"] = (
                (first_positive_idx - 1) + abs(cumulative_cash_flow[first_positive_idx - 1]) /
                (cumulative_cash_flow[first_positive_idx] -
                 cumulative_cash_flow[first_positive_idx - 1])
            ) - construction_p + 1  # Adjust for construction period and 0-based indexing
        # Fallback if payback calculation is unusual (e.g. all positive CFs)
        else:
            metrics_results["Payback_Period_Years"] = first_positive_idx - \
                construction_p + 1
    else:
        # No payback within project lifetime
        metrics_results["Payback_Period_Years"] = np.nan

    # LCOH is now handled by calculate_lcoh_breakdown
    # ROI will be calculated in the main script after total_capex is available in annual_metrics
    metrics_results["ROI"] = np.nan
    return metrics_results


def calculate_incremental_metrics(
    optimized_cash_flows: np.ndarray,
    baseline_annual_revenue: float,
    project_lifetime: int,  # project_lifetime from config or params
    construction_period: int,  # construction_period from config or params
    discount_rt: float,  # discount_rt from config or params
    tax_rt: float,  # tax_rt from config
    annual_metrics_optimized: dict,
    capex_components_incremental: dict,  # Derived from CAPEX_COMPONENTS
    om_components_incremental: dict,  # Derived from OM_COMPONENTS
    replacement_schedule_incremental: dict,  # Derived from REPLACEMENT_SCHEDULE
    h2_subsidy_val: float,
    h2_subsidy_yrs: int,
    optimized_capacities_inc: dict,
) -> dict:
    logger.info("Calculating incremental financial metrics.")
    inc_metrics = {}
    total_project_years = len(optimized_cash_flows)
    baseline_cash_flows = np.zeros(total_project_years)
    baseline_annual_opex = annual_metrics_optimized.get("VOM_Turbine_Cost", 0)
    baseline_annual_profit_before_tax = baseline_annual_revenue - baseline_annual_opex
    logger.info(
        f"Baseline revenue: ${baseline_annual_revenue:,.2f}, OPEX: ${baseline_annual_opex:,.2f}, Profit: ${baseline_annual_profit_before_tax:,.2f}")

    for i in range(construction_period, total_project_years):
        baseline_cash_flows[i] = baseline_annual_profit_before_tax * \
            (1 - tax_rt if baseline_annual_profit_before_tax > 0 else 1)

    pure_incremental_cf = np.zeros(total_project_years)
    total_incremental_capex_sum_after_learning = 0
    initial_inc_battery_capex_energy = 0
    initial_inc_battery_capex_power = 0

    for comp_name, comp_data in capex_components_incremental.items():
        base_cost = comp_data.get("total_base_cost_for_ref_size", 0)
        ref_cap = comp_data.get("reference_total_capacity_mw", 0)
        lr = comp_data.get("learning_rate_decimal", 0)
        cap_key = comp_data.get("applies_to_component_capacity_key")
        pay_sched = comp_data.get("payment_schedule_years", {})
        actual_opt_cap_inc = optimized_capacities_inc.get(
            cap_key, ref_cap if cap_key else 0)
        adj_cost_inc = 0.0
        if cap_key and actual_opt_cap_inc == 0 and ref_cap > 0:
            adj_cost_inc = 0.0
        elif lr > 0 and ref_cap > 0 and actual_opt_cap_inc > 0 and cap_key:
            pr = 1 - lr
            b = math.log(pr) / math.log(2) if 0 < pr < 1 else 0
            adj_cost_inc = base_cost * ((actual_opt_cap_inc / ref_cap)**b)
        elif actual_opt_cap_inc > 0 and ref_cap > 0 and cap_key:
            adj_cost_inc = base_cost * (actual_opt_cap_inc / ref_cap)
        elif not cap_key:
            adj_cost_inc = base_cost

        if comp_name == "Battery_System_Energy":
            initial_inc_battery_capex_energy = adj_cost_inc
        if comp_name == "Battery_System_Power":
            initial_inc_battery_capex_power = adj_cost_inc
        total_incremental_capex_sum_after_learning += adj_cost_inc
        for constr_yr_offset, share in pay_sched.items():
            # Construction year payment indices are negative relative to operation start
            # Project year index is construction_period (e.g. 2) + offset (e.g. -2) = 0
            if 0 <= construction_period + constr_yr_offset < construction_period:
                pure_incremental_cf[construction_period +
                                    constr_yr_offset] -= (adj_cost_inc * share)

    inc_metrics["Total_Incremental_CAPEX_Learned_USD"] = total_incremental_capex_sum_after_learning
    initial_total_inc_battery_capex = initial_inc_battery_capex_energy + \
        initial_inc_battery_capex_power

    h2_rev_annual = annual_metrics_optimized.get("H2_Total_Revenue", 0)
    # Make sure this is AS_Revenue_Total from annual_metrics
    as_rev_annual = annual_metrics_optimized.get("AS_Revenue", 0)
    avg_elec_price = annual_metrics_optimized.get(
        "Avg_Electricity_Price_USD_per_MWh", 40.0)

    opp_cost_battery_annual = 0.0
    direct_cost_battery_annual = 0.0

    # Use ENABLE_BATTERY from config
    enable_battery_flag = ENABLE_BATTERY
    if enable_battery_flag and optimized_capacities_inc.get("Battery_Capacity_MWh", 0) > 0:
        battery_charge_from_npp_mwh = annual_metrics_optimized.get(
            "Annual_Battery_Charge_From_NPP_MWh", 0)
        opp_cost_battery_annual = battery_charge_from_npp_mwh * avg_elec_price
        battery_charge_from_grid_mwh = annual_metrics_optimized.get(
            "Annual_Battery_Charge_From_Grid_MWh", 0)
        direct_cost_battery_annual = battery_charge_from_grid_mwh * avg_elec_price
        logger.info(
            f"Battery charging costs: Opportunity ${opp_cost_battery_annual:,.2f}/yr, Direct ${direct_cost_battery_annual:,.2f}/yr")

    as_opportunity_cost_annual = 0.0
    total_as_deployed_mwh = 0.0
    as_deployment_keys = [  # Ensure these keys match what's in annual_metrics_optimized
        "AS_Total_Deployed_ECRS_Battery_MWh", "AS_Total_Deployed_ECRS_Electrolyzer_MWh", "AS_Total_Deployed_ECRS_Turbine_MWh",
        "AS_Total_Deployed_RegDown_Battery_MWh", "AS_Total_Deployed_RegDown_Electrolyzer_MWh", "AS_Total_Deployed_RegDown_Turbine_MWh",
        "AS_Total_Deployed_RegUp_Battery_MWh", "AS_Total_Deployed_RegUp_Electrolyzer_MWh", "AS_Total_Deployed_RegUp_Turbine_MWh",
        "AS_Total_Deployed_NSR_Battery_MWh", "AS_Total_Deployed_NSR_Electrolyzer_MWh", "AS_Total_Deployed_NSR_Turbine_MWh",
        "AS_Total_Deployed_SR_Battery_MWh", "AS_Total_Deployed_SR_Electrolyzer_MWh", "AS_Total_Deployed_SR_Turbine_MWh",
    ]
    for key in as_deployment_keys:
        if "Turbine" in key:
            total_as_deployed_mwh += annual_metrics_optimized.get(key, 0)
    as_opportunity_cost_annual = total_as_deployed_mwh * avg_elec_price
    logger.info(
        f"AS opportunity cost: Deployed by turbine {total_as_deployed_mwh:,.2f} MWh/yr, Cost ${as_opportunity_cost_annual:,.2f}/yr")

    vom_annual_inc = sum(annual_metrics_optimized.get(k, 0) for k in [
        "VOM_Electrolyzer_Cost", "VOM_Battery_Cost", "Water_Cost", "Startup_Cost", "Ramping_Cost", "H2_Storage_Cycle_Cost"
    ]) + direct_cost_battery_annual

    opp_cost_elec_annual = annual_metrics_optimized.get(
        "Annual_Electrolyzer_MWh", 0) * avg_elec_price + opp_cost_battery_annual
    total_opportunity_cost_annual = opp_cost_elec_annual + as_opportunity_cost_annual

    for op_idx in range(total_project_years - construction_period):
        proj_yr_idx = op_idx + construction_period
        op_yr_num = op_idx + 1
        cur_h2_rev = h2_rev_annual - \
            (annual_metrics_optimized.get("H2_Subsidy_Revenue", 0)
             if op_yr_num > h2_subsidy_yrs else 0)
        rev_inc = cur_h2_rev + as_rev_annual - as_opportunity_cost_annual
        # total_opportunity_cost_annual includes opp_cost_elec_annual
        costs_inc = vom_annual_inc + opp_cost_elec_annual

        fixed_om_inc_general_base = om_components_incremental.get(
            "Fixed_OM_General", {}).get("base_cost", 0)
        fixed_om_inc_general_inflation = om_components_incremental.get(
            "Fixed_OM_General", {}).get("inflation_rate", 0)
        costs_inc += fixed_om_inc_general_base * \
            ((1 + fixed_om_inc_general_inflation)**op_idx)

        if enable_battery_flag and optimized_capacities_inc.get("Battery_Capacity_MWh", 0) > 0:
            batt_fixed_om_per_mw_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}).get("base_cost_per_mw_year", 0)
            batt_fixed_om_per_mwh_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}).get("base_cost_per_mwh_year", 0)
            batt_inflation_inc = om_components_incremental.get(
                "Fixed_OM_Battery", {}).get("inflation_rate", 0)
            batt_power_inc = optimized_capacities_inc.get(
                "Battery_Power_MW", 0)
            batt_capacity_inc = optimized_capacities_inc.get(
                "Battery_Capacity_MWh", 0)
            costs_inc += (batt_power_inc * batt_fixed_om_per_mw_inc + batt_capacity_inc *
                          batt_fixed_om_per_mwh_inc) * ((1 + batt_inflation_inc)**op_idx)

        for rep_comp_name_inc, rep_data_inc in replacement_schedule_incremental.items():
            if op_yr_num in rep_data_inc.get("years", []):
                cost_val_inc = rep_data_inc.get("cost", 0)
                if rep_comp_name_inc == "Battery_Augmentation_Replacement" and rep_data_inc.get("cost_percent_initial_capex", 0) > 0:
                    cost_val_inc = initial_total_inc_battery_capex * \
                        rep_data_inc["cost_percent_initial_capex"]
                costs_inc += cost_val_inc

        profit_inc_pre_tax = rev_inc - costs_inc
        tax_inc = profit_inc_pre_tax * tax_rt if profit_inc_pre_tax > 0 else 0
        pure_incremental_cf[proj_yr_idx] += profit_inc_pre_tax - tax_inc

    inc_metrics["NPV_USD"] = npf.npv(discount_rt, pure_incremental_cf)
    try:
        inc_metrics["IRR_percent"] = npf.irr(
            pure_incremental_cf) * 100 if any(cf != 0 for cf in pure_incremental_cf) else np.nan
    except:
        inc_metrics["IRR_percent"] = np.nan

    cum_pure_inc_cf = np.cumsum(pure_incremental_cf)
    pos_idx_pure = np.where(cum_pure_inc_cf >= 0)[0]
    if pos_idx_pure.size > 0:
        first_pos = pos_idx_pure[0]
        if first_pos == 0 and pure_incremental_cf[0] >= 0:
            inc_metrics["Payback_Period_Years"] = 0
        elif first_pos > 0 and cum_pure_inc_cf[first_pos-1] < 0:
            inc_metrics["Payback_Period_Years"] = (first_pos - 1) + abs(cum_pure_inc_cf[first_pos-1]) / (
                cum_pure_inc_cf[first_pos] - cum_pure_inc_cf[first_pos-1]) - construction_period + 1
        else:
            inc_metrics["Payback_Period_Years"] = first_pos - \
                construction_period + 1
    else:
        inc_metrics["Payback_Period_Years"] = np.nan

    inc_metrics["pure_incremental_cash_flows"] = pure_incremental_cf
    inc_metrics["traditional_incremental_cash_flows"] = optimized_cash_flows - \
        baseline_cash_flows
    # This includes elec for H2 and battery from NPP
    inc_metrics["Annual_Electricity_Opportunity_Cost_USD"] = total_opportunity_cost_annual
    inc_metrics["Annual_AS_Opportunity_Cost_USD"] = as_opportunity_cost_annual
    inc_metrics["Annual_Baseline_OPEX_USD"] = baseline_annual_opex
    return inc_metrics
