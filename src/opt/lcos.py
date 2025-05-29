import math


def calculate_lcos(
    # --- Parameters without default values ---
    # Initial system capital cost ($ per kWh of energy capacity)
    capex_per_kwh,
    capacity_mwh,  # Total energy capacity of the BESS (MWh)
    power_mw,  # Rated power capacity of the BESS (MW)
    # Expected operational lifetime of the system (years)
    lifetime_years,
    cycles_per_year,  # Equivalent full charge/discharge cycles per year
    dod_percent,  # Depth of Discharge per cycle (%)
    rte_percent,  # Round-Trip Efficiency (AC-AC, %)
    # Fixed O&M cost ($ per kW of power capacity per year)
    fom_rate_per_kw_yr,
    # Average cost of electricity used for charging ($/MWh)
    avg_charging_price_per_mwh,
    # Annual discount rate for present value calculations (%)
    discount_rate_percent,
    # --- Parameters with default values ---
    # Investment Tax Credit percentage (e.g., 30 for 30% ITC)
    apply_itc_percent=0,
    # Other Variable O&M costs ($ per MWh discharged, excluding charging)
    other_vom_per_mwh_discharged=0,
):
    """
    Calculates the Levelized Cost of Storage (LCOS) for a Battery Energy Storage System (BESS).

    Args:
        capex_per_kwh (float): Initial capital cost in $/kWh.
        capacity_mwh (float): System energy capacity in MWh.
        power_mw (float): System power rating in MW.
        lifetime_years (int): Operational lifetime in years.
        cycles_per_year (int): Equivalent full cycles per year.
        dod_percent (float): Depth of Discharge (0-100).
        rte_percent (float): Round-Trip Efficiency (0-100).
        fom_rate_per_kw_yr (float): Fixed O&M cost in $/kW-year. Assumed to include augmentation.
        avg_charging_price_per_mwh (float): Average electricity cost for charging in $/MWh.
        discount_rate_percent (float): Annual discount rate (0-100).
        apply_itc_percent (float): Investment Tax Credit percentage (0-100). Defaults to 0.
        other_vom_per_mwh_discharged (float): Other Variable O&M cost in $/MWh discharged. Defaults to 0.


    Returns:
        float: The calculated LCOS in $/MWh discharged, or None if inputs are invalid.

    Notes:
        - This calculation assumes the FOM covers augmentation costs required to maintain
          the system's ability to deliver its rated energy throughput over its lifetime,
          simplifying the degradation modeling based on the NREL ATB approach mentioned
          in the provided document.
        - Assumes constant annual cycling, costs, and discount rate over the lifetime.
        - LCOS represents the average revenue needed per MWh discharged to break even over the project life.
    """
    # --- Input Validation ---
    if not all(
        [
            capex_per_kwh >= 0,
            capacity_mwh > 0,
            power_mw > 0,
            lifetime_years > 0,
            cycles_per_year >= 0,
            0 < dod_percent <= 100,
            0 < rte_percent <= 100,
            fom_rate_per_kw_yr >= 0,
            avg_charging_price_per_mwh >= 0,
            discount_rate_percent >= 0,
            0 <= apply_itc_percent <= 100,
            other_vom_per_mwh_discharged >= 0,
        ]
    ):
        print("Error: Invalid input parameters.")
        return None

    # --- Derived Calculations & Initializations ---
    total_initial_capex = capex_per_kwh * capacity_mwh * 1000  # Total CAPEX in $
    effective_capex = total_initial_capex * (
        1 - apply_itc_percent / 100
    )  # CAPEX after ITC

    discount_rate = discount_rate_percent / 100

    total_pv_costs = effective_capex  # Start with year 0 cost
    total_pv_energy_discharged = 0

    # --- Annual Calculations (Years 1 to Lifetime) ---
    # Fixed Annual O&M Cost ($/year)
    fom_cost_annual = fom_rate_per_kw_yr * power_mw * 1000

    # Annual Energy Discharged (MWh/year) - Assuming FOM maintains capacity for this throughput
    # Energy discharged per cycle = Usable Capacity * DoD
    energy_per_cycle_mwh = capacity_mwh * (dod_percent / 100)
    annual_energy_discharged_mwh = energy_per_cycle_mwh * cycles_per_year

    # Annual Energy Charged (MWh/year) - Accounting for RTE losses
    annual_energy_charged_mwh = (
        annual_energy_discharged_mwh / (rte_percent / 100) if rte_percent > 0 else 0
    )

    for year in range(1, lifetime_years + 1):
        # Calculate annual costs for the current year
        charging_cost_annual = annual_energy_charged_mwh * avg_charging_price_per_mwh
        other_vom_cost_annual = (
            annual_energy_discharged_mwh * other_vom_per_mwh_discharged
        )
        total_annual_op_cost = (
            fom_cost_annual + charging_cost_annual + other_vom_cost_annual
        )

        # Calculate Present Value (PV) of costs and energy for the current year
        pv_factor = (1 + discount_rate) ** year
        pv_op_cost_year = total_annual_op_cost / pv_factor
        pv_energy_discharged_year = annual_energy_discharged_mwh / pv_factor

        # Add to cumulative totals
        total_pv_costs += pv_op_cost_year
        total_pv_energy_discharged += pv_energy_discharged_year

    # --- Final LCOS Calculation ---
    if total_pv_energy_discharged <= 0:
        print(
            "Error: Total discounted energy discharged is zero or negative. Cannot calculate LCOS."
        )
        return None

    lcos = total_pv_costs / total_pv_energy_discharged
    return lcos


# --- Example Usage ---
if __name__ == "__main__":
    # Example based on a hypothetical Utility-Scale BESS in the US (drawing from document data)
    # Using mid-range US CAPEX, NREL FOM approach, typical performance, and plausible financial assumptions.

    # System Specs
    example_power_mw = 100  # MW (e.g., Lazard example size)
    example_duration_hr = 4  # Hours (Common duration)
    example_capacity_mwh = example_power_mw * example_duration_hr  # MWh

    # Cost & Financial Inputs (Representative values based on document)
    # $/kWh (Avg US Turnkey Cost 2024, BNEF via doc)
    example_capex_per_kwh = 236
    example_itc_percent = 30  # % (Standard US ITC rate mentioned in doc)
    example_lifetime_years = 15  # Years (NREL ATB assumption)
    # Fixed O&M: NREL uses 2.5% of CAPEX ($/kW) per year.
    # For $236/kWh & 4hr duration -> $944/kW CAPEX. 2.5% of $944/kW = $23.6/kW-yr. Let's use 25 for simplicity.
    example_fom_per_kw_yr = 25
    # $/MWh (Hypothetical average wholesale charging cost)
    example_avg_charging_price = 50
    # % (Hypothetical, common range for project finance)
    example_discount_rate = 8

    # Operational Inputs
    # Equivalent Full Cycles (Assumption for wholesale arbitrage/capacity)
    example_cycles_per_year = 300
    example_dod_percent = 90  # % (Common operational limit)
    example_rte_percent = 85  # % (Typical AC-AC RTE)
    example_other_vom = 1  # $/MWh (Small value for other variable costs)

    print(f"--- Calculating LCOS for Example BESS ---")
    print(f"Parameters:")
    print(
        f"  Power: {example_power_mw} MW, Capacity: {example_capacity_mwh} MWh ({example_duration_hr} hr)"
    )
    print(f"  CAPEX: ${example_capex_per_kwh}/kWh")
    print(f"  ITC: {example_itc_percent}%")
    print(f"  Lifetime: {example_lifetime_years} years")
    print(f"  Cycles/Year: {example_cycles_per_year}")
    print(f"  DoD: {example_dod_percent}%")
    print(f"  RTE: {example_rte_percent}%")
    print(f"  Fixed O&M: ${example_fom_per_kw_yr}/kW-year")
    print(f"  Other Var. O&M: ${example_other_vom}/MWh discharged")
    print(f"  Charging Cost: ${example_avg_charging_price}/MWh")
    print(f"  Discount Rate: {example_discount_rate}%")
    print("-" * 30)

    calculated_lcos = calculate_lcos(
        # Non-default args first
        capex_per_kwh=example_capex_per_kwh,
        capacity_mwh=example_capacity_mwh,
        power_mw=example_power_mw,
        lifetime_years=example_lifetime_years,
        cycles_per_year=example_cycles_per_year,
        dod_percent=example_dod_percent,
        rte_percent=example_rte_percent,
        fom_rate_per_kw_yr=example_fom_per_kw_yr,
        avg_charging_price_per_mwh=example_avg_charging_price,
        discount_rate_percent=example_discount_rate,
        # Default args last
        apply_itc_percent=example_itc_percent,
        other_vom_per_mwh_discharged=example_other_vom,
    )

    if calculated_lcos is not None:
        print(f"Calculated LCOS: ${calculated_lcos:.2f} / MWh discharged")

        # Compare with Lazard range mentioned in doc (Post-ITC low end: $124/MWh)
        # Note: Lazard uses complex modeling; this simplified calculation provides an estimate.
        # Differences can arise from discount rate, specific O&M/augmentation assumptions,
        # degradation modeling, financing structure, etc.
        print(
            "\nNote: Compare this estimate to ranges like Lazard's ($124-$296/MWh post-ITC for utility-scale)."
        )
        print(
            "Differences are expected due to varying assumptions and model complexity."
        )
    else:
        print("LCOS calculation failed due to invalid inputs or zero energy output.")
