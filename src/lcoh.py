import numpy as np
import logging
import math # Needed for CRF calculation

# Set up a simple logger for this function
cost_logger = logging.getLogger(__name__)
if not cost_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    cost_logger.addHandler(handler)
    cost_logger.setLevel(logging.INFO)

# --- Helper Function ---
def calculate_crf(discount_rate, lifetime_years):
    """Calculates the Capital Recovery Factor."""
    if lifetime_years <= 0:
        return 0
    if discount_rate == 0:
        # Handle edge case for zero discount rate
        # If lifetime is positive, CRF is 1/lifetime
        return 1.0 / lifetime_years if lifetime_years > 0 else 0
    # Standard CRF formula
    try:
        factor = (1 + discount_rate)**lifetime_years
        return (discount_rate * factor) / (factor - 1)
    except OverflowError:
        cost_logger.warning(f"Overflow calculating CRF with rate={discount_rate}, life={lifetime_years}. Approximating as discount_rate.")
        # For very long lifetimes, CRF approaches the discount rate
        return discount_rate
    except ValueError:
         cost_logger.error(f"Math domain error calculating CRF with rate={discount_rate}, life={lifetime_years}. Check inputs.")
         return 0 # Or raise an error

# --- Main Costing Function ---
def calculate_hydrogen_system_lcoh(
    # --- System Parameters ---
    electrolyzer_technology: str,         # e.g., "PEM", "AWE", "SOEC", "AEM"
    system_capacity_mw: float,            # DC input capacity of the electrolyzer system
    capacity_factor: float,               # Annual average utilization rate (0.0 to 1.0)
    plant_lifetime_years: int = 30,       # Economic lifetime of the overall plant

    # --- CAPEX Parameters ($/kW DC for Electrolyzer) ---
    installed_electrolyzer_capex_per_kw: float = None, # Optional: Provide specific value to override defaults
    default_pem_capex: float = 2000.0,    # $/kW (DOE Baseline Avg 2022$) [Untitled.md, Ref 1]
    default_awe_capex: float = 1800.0,    # $/kW (Estimate based on sources) [Untitled.md, Ref 27; CostAnalysis.md, Ref 16]
    default_soec_capex: float = 4000.0,   # $/kW (Estimate based on sources) [CostAnalysis.md, Ref 4, 52]
    default_aem_capex: float = 1900.0,    # $/kW (Estimate based on sources) [CostAnalysis.md, Ref 19, 27]

    # --- Compression & Storage Parameters ---
    include_compression_storage: bool = True, # Flag to include these costs
    hydrogen_outlet_pressure_bar: float = 350.0, # Required H2 pressure after compression
    required_storage_kg: float = 0.0,         # Required H2 storage capacity in kg (set > 0 to include)
    # CAPEX estimates for Compression/Storage (Highly variable, provide if known)
    compressor_capex_usd_per_kw_electrolyzer: float = None, # Optional: Estimate compressor cost relative to electrolyzer size
    # Rough default based on $1M for ~500 kg/day (~1 MW system) -> ~$1000/kW? Very rough.
    default_compressor_capex_fraction_of_electrolyzer: float = 0.3, # Alt: Estimate as fraction of Elec. CAPEX (e.g., 30%)
    storage_capex_usd_per_kg: float = 1000.0, # $/kg storage capacity (Default based on Type I tanks) [Untitled.md, Ref 59]

    # --- OPEX Parameters ---
    # Electrolyzer Electricity Efficiency
    system_efficiency_kwh_per_kg: float = None, # kWh DC / kg H2. Optional: Provide specific value
    default_pem_efficiency: float = 57.0, # kWh/kg [Untitled.md, Ref 1]
    default_awe_efficiency: float = 57.0, # kWh/kg [CostAnalysis.md, Ref 4, 18]
    default_soec_efficiency: float = 40.0,# kWh/kg (Includes heat credit) [CostAnalysis.md, Ref 18]
    default_aem_efficiency: float = 57.0, # kWh/kg (Estimate) [CostAnalysis.md, Ref 4]

    # Compressor Electricity Consumption
    # Practical range 2.2-6.4 kWh/kg depending on pressure/efficiency [Untitled.md, Ref 56]
    compressor_consumption_kwh_per_kg: float = 3.0, # kWh/kg compressed (Default estimate for moderate pressure)

    # Electricity Pricing (Idaho Falls Example)
    electricity_price_usd_per_kwh: float = 0.0435, # $/kWh (Energy Charge) [Untitled.md, Ref 9]
    demand_charge_usd_per_kw_month: float = 8.25, # $/kW/month (Demand Charge) [Untitled.md, Ref 9]
    include_demand_charge: bool = True,       # Whether to include demand charge

    # Water
    water_consumption_liter_per_kg: float = 15.0, # L/kg H2 [Untitled.md, Section 5.2]
    water_cost_usd_per_m3: float = 1.0,       # $/m³ (Placeholder - NEED ACTUAL IF RATE) [Untitled.md, Section 5.2]

    # Fixed O&M (Includes Electrolyzer + BOP + Comp/Storage routine maint.)
    fixed_om_usd_per_kw_year: float = None,   # $/kW-year (based on electrolyzer kW). Optional.
    fixed_om_percent_total_capex_per_year: float = 0.02, # % of TOTAL initial CAPEX/year [Untitled.md, Section 5.3]

    # Stack Replacement
    stack_lifetime_hours: int = None,         # Operating hours. Optional.
    default_pem_stack_life_hrs: int = 40000,  # Hours [Untitled.md, Ref 1]
    default_awe_stack_life_hrs: int = 80000,  # Hours [CostAnalysis.md, Ref 10]
    default_soec_stack_life_hrs: int = 20000, # Hours [CostAnalysis.md, Ref 10]
    default_aem_stack_life_hrs: int = 30000,  # Hours [CostAnalysis.md, Ref 10]
    stack_replacement_cost_percent_electrolyzer_capex: float = 0.30, # % of INITIAL ELECTROLYZER CAPEX (e.g., if stack=30% of initial sys cost)
                                                                      # Alt: Use DOE 11% of TOTAL installed CAPEX [Untitled.md, Ref 1] - choose one method
    # --- Financial Parameters ---
    discount_rate: float = 0.08,              # Fractional discount rate [Untitled.md, Table 6.2]

    # --- Analysis Options ---
    verbose: bool = True                      # Print detailed breakdown
):
    """
    Calculates the Levelized Cost of Hydrogen (LCOH) for a complete electrolysis system,
    optionally including compression and storage costs.

    Args:
        (Refer to previous function for standard args)
        include_compression_storage (bool): Whether to include C&S costs.
        hydrogen_outlet_pressure_bar (float): Target H2 pressure. Affects compression energy assumptions.
        required_storage_kg (float): H2 storage capacity needed (kg). If > 0, storage CAPEX is added.
        compressor_capex_usd_per_kw_electrolyzer (float, optional): Specific compressor CAPEX relative to electrolyzer kW.
        default_compressor_capex_fraction_of_electrolyzer (float): Default compressor cost as fraction of electrolyzer CAPEX.
        storage_capex_usd_per_kg (float): Specific storage CAPEX in $/kg capacity.
        compressor_consumption_kwh_per_kg (float): Compressor electricity use in kWh/kg H2.
        fixed_om_percent_total_capex_per_year (float): Fixed O&M as % of TOTAL system CAPEX/year.
        stack_replacement_cost_percent_electrolyzer_capex (float): Stack replacement cost as % of initial ELECTROLYZER CAPEX.

    Returns:
        dict: A dictionary containing LCOH and its components ($/kg H2), or None if inputs are invalid.
              Keys: 'lcoh', 'electrolyzer_capex_recovery', 'compressor_capex_recovery', 'storage_capex_recovery',
                    'electricity_electrolysis', 'electricity_compression', 'electricity_demand',
                    'water_cost', 'fixed_om', 'stack_replacement'
    """
    cost_logger.info(f"Calculating LCOH for {system_capacity_mw} MW {electrolyzer_technology} system...")

    # --- Input Validation and Parameter Selection ---
    technology = electrolyzer_technology.upper()
    if technology not in ["PEM", "AWE", "SOEC", "AEM"]:
        cost_logger.error(f"Invalid electrolyzer technology: {electrolyzer_technology}")
        return None

    system_capacity_kw = system_capacity_mw * 1000.0

    # Select Electrolyzer CAPEX
    _installed_electrolyzer_capex_per_kw = installed_electrolyzer_capex_per_kw
    if _installed_electrolyzer_capex_per_kw is None:
        if technology == "PEM":
            _installed_electrolyzer_capex_per_kw = default_pem_capex
        elif technology == "AWE":
            _installed_electrolyzer_capex_per_kw = default_awe_capex
        elif technology == "SOEC":
            _installed_electrolyzer_capex_per_kw = default_soec_capex
        elif technology == "AEM":
            _installed_electrolyzer_capex_per_kw = default_aem_capex
    total_electrolyzer_cost = _installed_electrolyzer_capex_per_kw * system_capacity_kw
    cost_logger.info(f"Using Installed Electrolyzer CAPEX: ${_installed_electrolyzer_capex_per_kw:.2f}/kW (Total: ${total_electrolyzer_cost:,.2f})")

    # Select Efficiency
    _system_efficiency_kwh_per_kg = system_efficiency_kwh_per_kg
    if _system_efficiency_kwh_per_kg is None:
        if technology == "PEM":
            _system_efficiency_kwh_per_kg = default_pem_efficiency
        elif technology == "AWE":
            _system_efficiency_kwh_per_kg = default_awe_efficiency
        elif technology == "SOEC":
            _system_efficiency_kwh_per_kg = default_soec_efficiency
        elif technology == "AEM":
            _system_efficiency_kwh_per_kg = default_aem_efficiency
    if _system_efficiency_kwh_per_kg <= 0:
        cost_logger.error("System efficiency must be positive.")
        return None
    cost_logger.info(f"Using System Efficiency: {_system_efficiency_kwh_per_kg:.2f} kWh/kg H2")

    # Select Stack Lifetime
    _stack_lifetime_hours = stack_lifetime_hours
    if _stack_lifetime_hours is None:
        if technology == "PEM":
            _stack_lifetime_hours = default_pem_stack_life_hrs
        elif technology == "AWE":
            _stack_lifetime_hours = default_awe_stack_life_hrs
        elif technology == "SOEC":
            _stack_lifetime_hours = default_soec_stack_life_hrs
        elif technology == "AEM":
            _stack_lifetime_hours = default_aem_stack_life_hrs
    if _stack_lifetime_hours <= 0:
        cost_logger.error("Stack lifetime must be positive.")
        return None
    cost_logger.info(f"Using Stack Lifetime: {_stack_lifetime_hours:,} hours")

    if not (0 < capacity_factor <= 1.0):
         cost_logger.error("Capacity factor must be between 0 (exclusive) and 1.0 (inclusive).")
         return None

    # --- Calculate Intermediate Values ---
    hours_in_year = 8760.0
    annual_operating_hours = hours_in_year * capacity_factor

    # Annual Hydrogen Production (kg/year)
    annual_h2_production_kg = (system_capacity_kw / _system_efficiency_kwh_per_kg) * annual_operating_hours
    if annual_h2_production_kg <= 0:
        cost_logger.error("Calculated annual hydrogen production is zero or negative.")
        return None
    cost_logger.info(f"Estimated Annual H2 Production: {annual_h2_production_kg:,.2f} kg/year")
    daily_h2_production_kg = annual_h2_production_kg / 365.0

    # --- Calculate CAPEX Components ---
    # Electrolyzer CAPEX
    annualized_electrolyzer_capex = 0
    crf = calculate_crf(discount_rate, plant_lifetime_years)
    if crf > 0:
         annualized_electrolyzer_capex = total_electrolyzer_cost * crf

    # Compressor CAPEX
    total_compressor_cost = 0
    annualized_compressor_capex = 0
    if include_compression_storage:
        if compressor_capex_usd_per_kw_electrolyzer is not None:
            total_compressor_cost = compressor_capex_usd_per_kw_electrolyzer * system_capacity_kw
        else:
            # Estimate based on fraction of electrolyzer cost
            total_compressor_cost = total_electrolyzer_cost * default_compressor_capex_fraction_of_electrolyzer
            cost_logger.warning(f"Compressor CAPEX not specified, estimating as {default_compressor_capex_fraction_of_electrolyzer*100:.0f}% of electrolyzer CAPEX.")
        if crf > 0:
             annualized_compressor_capex = total_compressor_cost * crf
        cost_logger.info(f"Estimated Compressor CAPEX: ${total_compressor_cost:,.2f}")

    # Storage CAPEX
    total_storage_cost = 0
    annualized_storage_capex = 0
    if include_compression_storage and required_storage_kg > 0:
        total_storage_cost = required_storage_kg * storage_capex_usd_per_kg
        if crf > 0:
            annualized_storage_capex = total_storage_cost * crf
        cost_logger.info(f"Estimated Storage CAPEX ({required_storage_kg} kg): ${total_storage_cost:,.2f}")

    # Total Initial CAPEX (for Fixed O&M calculation)
    total_initial_system_capex = total_electrolyzer_cost + total_compressor_cost + total_storage_cost
    cost_logger.info(f"Total Initial System CAPEX Estimate: ${total_initial_system_capex:,.2f}")
    annualized_total_capex = annualized_electrolyzer_capex + annualized_compressor_capex + annualized_storage_capex
    cost_logger.info(f"Total Annualized CAPEX Recovery: ${annualized_total_capex:,.2f}/year")


    # --- Calculate Annual OPEX Components ---
    # Electricity Cost - Electrolysis Energy
    annual_kwh_electrolysis = annual_h2_production_kg * _system_efficiency_kwh_per_kg
    annual_electricity_cost_electrolysis = annual_kwh_electrolysis * electricity_price_usd_per_kwh

    # Electricity Cost - Compression Energy
    annual_kwh_compression = 0.0
    annual_electricity_cost_compression = 0.0
    if include_compression_storage:
        annual_kwh_compression = annual_h2_production_kg * compressor_consumption_kwh_per_kg
        annual_electricity_cost_compression = annual_kwh_compression * electricity_price_usd_per_kwh
        cost_logger.info(f"Using Compressor Efficiency: {compressor_consumption_kwh_per_kg:.2f} kWh/kg H2")
        cost_logger.info(f"Annual Compressor Electricity Consumption: {annual_kwh_compression:,.2f} kWh")
        cost_logger.info(f"Annual Compressor Electricity Cost: ${annual_electricity_cost_compression:,.2f}/year")

    # Electricity Cost - Demand Charge (Simplified - applied to total peak kW)
    # Assume peak electrolyzer + peak compressor power demand occurs simultaneously
    # Estimate compressor peak kW: (Max kg/hr * kWh/kg) / efficiency_factor -> Hard to estimate max kg/hr without profile
    # Simplification: Base demand charge only on electrolyzer peak capacity for now
    # TODO: A better approach would estimate peak compressor power based on max throughput.
    annual_electricity_cost_demand = 0.0
    if include_demand_charge and capacity_factor > 0:
        # Using electrolyzer capacity as proxy for peak demand
        peak_demand_kw = system_capacity_kw
        # Add rough estimate for compressor peak power if possible, otherwise ignore for simplicity
        # peak_compressor_kw = (daily_h2_production_kg / 24) * compressor_consumption_kwh_per_kg # Very rough avg hourly power
        # peak_demand_kw += peak_compressor_kw
        monthly_demand_charge = peak_demand_kw * demand_charge_usd_per_kw_month
        annual_electricity_cost_demand = monthly_demand_charge * 12
        cost_logger.info(f"Annual Electricity Cost (Demand Charge based on {peak_demand_kw:.0f} kW peak): ${annual_electricity_cost_demand:,.2f}/year")

    total_annual_electricity_cost = (annual_electricity_cost_electrolysis +
                                     annual_electricity_cost_compression +
                                     annual_electricity_cost_demand)
    cost_logger.info(f"Total Annual Electricity Cost: ${total_annual_electricity_cost:,.2f}/year")


    # Water Cost
    annual_water_consumption_liter = annual_h2_production_kg * water_consumption_liter_per_kg
    annual_water_consumption_m3 = annual_water_consumption_liter / 1000.0
    annual_water_cost = annual_water_consumption_m3 * water_cost_usd_per_m3
    cost_logger.info(f"Annual Water Cost: ${annual_water_cost:,.2f}/year")

    # Fixed O&M Cost (Covers routine maint for all systems)
    if fixed_om_usd_per_kw_year is None:
        # Base percentage on TOTAL initial system CAPEX
        annual_fixed_om_cost = total_initial_system_capex * fixed_om_percent_total_capex_per_year
        cost_logger.info(f"Using Fixed O&M: {fixed_om_percent_total_capex_per_year*100:.1f}% of Total CAPEX/year")
    else:
        # Base fixed $/kW rate only on ELECTROLYZER capacity for consistency
        annual_fixed_om_cost = system_capacity_kw * fixed_om_usd_per_kw_year
        cost_logger.info(f"Using Fixed O&M: ${fixed_om_usd_per_kw_year:.2f}/kW-year (based on electrolyzer capacity)")
    cost_logger.info(f"Annual Fixed O&M Cost (Excl. Stack): ${annual_fixed_om_cost:,.2f}/year")

    # Stack Replacement Cost (Annualized)
    effective_stack_lifetime_years = _stack_lifetime_hours / annual_operating_hours if annual_operating_hours > 0 else float('inf')
    # Replacement cost based on ELECTROLYZER cost component
    single_replacement_cost = total_electrolyzer_cost * stack_replacement_cost_percent_electrolyzer_capex
    cost_logger.info(f"Single Stack Replacement Cost Estimate: ${single_replacement_cost:,.2f} (based on {stack_replacement_cost_percent_electrolyzer_capex*100:.0f}% of Electrolyzer CAPEX)")

    annualized_stack_replacement_cost = 0.0
    if effective_stack_lifetime_years > 0 and effective_stack_lifetime_years < plant_lifetime_years :
        # Simple annualization (ignores discounting)
        annualized_stack_replacement_cost = single_replacement_cost / effective_stack_lifetime_years
    cost_logger.info(f"Effective Stack Lifetime: {effective_stack_lifetime_years:.2f} years")
    cost_logger.info(f"Annualized Stack Replacement Cost: ${annualized_stack_replacement_cost:,.2f}/year")

    # Total Annual OPEX
    total_annual_opex = (total_annual_electricity_cost +
                         annual_water_cost +
                         annual_fixed_om_cost +
                         annualized_stack_replacement_cost)
    cost_logger.info(f"Total Annual OPEX: ${total_annual_opex:,.2f}/year")

    # --- Calculate LCOH ($/kg H2) ---
    total_annual_cost = annualized_total_capex + total_annual_opex
    lcoh_usd_per_kg = total_annual_cost / annual_h2_production_kg if annual_h2_production_kg > 0 else float('inf')
    cost_logger.info(f"Total Annualized Cost: ${total_annual_cost:,.2f}/year")
    cost_logger.info(f"Calculated LCOH: ${lcoh_usd_per_kg:.3f}/kg H2")

    # --- Prepare Output ---
    lcoh_components = {
        "lcoh": lcoh_usd_per_kg,
        "electrolyzer_capex_recovery": annualized_electrolyzer_capex / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "compressor_capex_recovery": annualized_compressor_capex / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "storage_capex_recovery": annualized_storage_capex / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "electricity_electrolysis": annual_electricity_cost_electrolysis / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "electricity_compression": annual_electricity_cost_compression / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "electricity_demand": annual_electricity_cost_demand / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "water_cost": annual_water_cost / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "fixed_om": annual_fixed_om_cost / annual_h2_production_kg if annual_h2_production_kg > 0 else 0,
        "stack_replacement": annualized_stack_replacement_cost / annual_h2_production_kg if annual_h2_production_kg > 0 else 0
    }

    if verbose:
        print("\n--- LCOH Breakdown ($/kg H2) ---")
        total_check = 0
        for component, value in lcoh_components.items():
            if component != 'lcoh':
                 percentage = (value / lcoh_usd_per_kg) * 100 if lcoh_usd_per_kg > 0 else 0
                 print(f"  {component:<30}: ${value:.3f} ({percentage:.1f}%)")
                 total_check += value
        print(f"  {'Calculated Component Sum':<30}: ${total_check:.3f}") # Sanity check
        print(f"  {'TOTAL LCOH':<30}: ${lcoh_usd_per_kg:.3f}")
        print("---------------------------------")

    return lcoh_components

# --- Example Usage ---
if __name__ == "__main__":
    print("Example LCOH Calculation for 100 MW PEM System in Idaho Falls (Including Compression & Storage):")

    # Define storage requirement for this example
    storage_capacity_kg = 5000 # Example: Store ~half a day's production at high CF

    lcoh_results_pem_full = calculate_hydrogen_system_lcoh(
        electrolyzer_technology="PEM",
        system_capacity_mw=100.0,
        capacity_factor=0.90, # High CF assumed
        plant_lifetime_years=30,
        # --- CAPEX ---
        installed_electrolyzer_capex_per_kw=2000.0, # Default PEM
        include_compression_storage=True,
        required_storage_kg=storage_capacity_kg,
        # Using default estimates for Compressor (30% of Elec CAPEX) & Storage ($1000/kg) CAPEX
        storage_capex_usd_per_kg=1000.0,
        # --- OPEX ---
        system_efficiency_kwh_per_kg=57.0, # Default PEM
        compressor_consumption_kwh_per_kg=3.0, # Default assumption
        electricity_price_usd_per_kwh=0.0435, # IF Rate
        demand_charge_usd_per_kw_month=8.25,  # IF Rate
        include_demand_charge=True,
        water_cost_usd_per_m3=1.0, # Placeholder
        fixed_om_percent_total_capex_per_year=0.02, # % of TOTAL CAPEX
        stack_lifetime_hours=40000, # Default PEM
        stack_replacement_cost_percent_electrolyzer_capex=0.30, # Example: Stack is 30% of initial Elec cost
        # --- Financial ---
        discount_rate=0.08,
        verbose=True
    )

    if lcoh_results_pem_full:
        print(f"\nOverall LCOH (PEM Full System Example): ${lcoh_results_pem_full['lcoh']:.2f}/kg H2")

    print("\nExample LCOH for same system WITHOUT Compression/Storage Costs:")
    lcoh_results_pem_elec_only = calculate_hydrogen_system_lcoh(
        electrolyzer_technology="PEM",
        system_capacity_mw=100.0,
        capacity_factor=0.90,
        plant_lifetime_years=30,
        installed_electrolyzer_capex_per_kw=2000.0,
        include_compression_storage=False, # Set flag to False
        required_storage_kg=0,
        system_efficiency_kwh_per_kg=57.0,
        # Compressor energy use is implicitly zero if not included
        electricity_price_usd_per_kwh=0.0435,
        demand_charge_usd_per_kw_month=8.25,
        include_demand_charge=True,
        water_cost_usd_per_m3=1.0,
        fixed_om_percent_total_capex_per_year=0.02, # Note: This % now applies only to Electrolyzer CAPEX
        stack_lifetime_hours=40000,
        stack_replacement_cost_percent_electrolyzer_capex=0.30,
        discount_rate=0.08,
        verbose=True
    )
    if lcoh_results_pem_elec_only:
        print(f"\nOverall LCOH (PEM Electrolyzer Only Example): ${lcoh_results_pem_elec_only['lcoh']:.2f}/kg H2")