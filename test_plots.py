#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import pandas as pd
from tea import *
import sys
import os
sys.path.append('tea')


def test_plot_generation():
    """Test the updated LCOH benchmarking plot generation"""

    # Use existing results for Beaver Valley
    results_dir = Path('TEA_results/cs1_tea/Beaver Valley_1_PJM_11')
    plot_dir = results_dir / 'Plots_PJM'

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Read the existing hourly results
    hourly_file = Path('output/cs1/Beaver Valley_1_PJM_11_hourly_results.csv')

    if not hourly_file.exists():
        print(f"Hourly results file not found: {hourly_file}")
        return False

    print("Loading hourly results...")
    df = pd.read_csv(hourly_file)

    # Load system parameters
    print("Loading system parameters...")
    tea_sys_params = load_tea_sys_params('PJM', Path('input'))

    # Calculate annual metrics
    print("Calculating annual metrics...")
    annual_metrics = calculate_annual_metrics(df, tea_sys_params)

    if not annual_metrics:
        print("Failed to calculate annual metrics")
        return False

    if 'lcoh_breakdown_analysis' not in annual_metrics:
        print("No LCOH breakdown analysis found in annual metrics")
        return False

    print("Found LCOH breakdown analysis, regenerating plots...")

    # Create minimal financial metrics dict for plotting
    financial_metrics = {
        'NPV_USD': 1485013953.05,
        'IRR_percent': 52.10,
        'LCOH_USD_per_kg': 3.935,
        'Payback_Period_Years': 1.49
    }

    # Create dummy cash flows
    cash_flows = np.array([0] * 11)

    try:
        # Call plot_results to regenerate the LCOH plots
        plot_results(annual_metrics, financial_metrics,
                     cash_flows, plot_dir, 2)
        print("Plots regenerated successfully!")

        # Check if the new plot was created
        new_plot_file = plot_dir / "lcoh_benchmarking_analysis.png"
        if new_plot_file.exists():
            print(f"New LCOH benchmarking plot created: {new_plot_file}")
            return True
        else:
            print("LCOH benchmarking plot was not created")
            return False

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_plot_generation()
    if success:
        print("\n✅ Plot generation test completed successfully!")
    else:
        print("\n❌ Plot generation test failed!")
        sys.exit(1)
