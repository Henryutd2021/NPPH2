import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("../input/hourly_data")
HOURS_IN_YEAR = 8760 
np.random.seed(42)  # Consistent randomness

# --- ISO Service Definitions (MUST BE CONSISTENT WITH model.py) ---
ISO_SERVICE_MAP = {
    'SPP': ['RegU', 'RegD', 'Spin', 'Sup', 'RamU', 'RamD', 'UncU'],
    'CAISO': ['RegU', 'RegD', 'Spin', 'NSpin', 'RMU', 'RMD'],
    'ERCOT': ['RegU', 'RegD', 'Spin', 'NSpin', 'ECRS'],
    'PJM': ['RegUp', 'RegDown', 'Syn', 'Rse', 'TMR'], # Syn=Spin, Rse=Secondary(30min), TMR=Tertiary(30min)
    'NYISO': ['RegUp', 'RegDown', 'Spin10', 'NSpin10', 'Res30'],
    'ISONE': ['Spin10', 'NSpin10', 'OR30'],
    'MISO': ['RegUp', 'RegDown', 'Spin', 'Sup', 'STR', 'RamU', 'RamD']
}

# Define Regulation Up/Down pairs for consistent processing
ISO_REGULATION_PAIRS = {
    'SPP': ('RegU', 'RegD'),
    'CAISO': ('RegU', 'RegD'),
    'ERCOT': ('RegU', 'RegD'),
    'PJM': ('RegUp', 'RegDown'),
    'NYISO': ('RegUp', 'RegDown'),
    'MISO': ('RegUp', 'RegDown')
}


# --- Data from PDFs for realistic generation ---

ISO_MILEAGE_MAP = {
    'PJM': {'RegUp': 5.4, 'RegDown': 5.4},
    'CAISO': {'RegU': 12.5, 'RegD': 12.5},
    'MISO': {'RegUp': 7.0, 'RegDown': 7.0},
    'ERCOT': {'RegU': 7.5, 'RegD': 7.5},
    'NYISO': {'RegUp': 13.0, 'RegDown': 13.0},
    'ISONE': {},
    'SPP': {'RegU': 0.2, 'RegD': 0.2}
}

WINNING_RATE_CATEGORIES = {
    'REGULATION': [0.70, 1.0],
    'SPIN': [0.20, 0.50],
    'NONSPIN': [0.10, 0.30],
    'SUPPLEMENTAL': [0.10, 0.30],
    'RESERVE_30MIN': [0.10, 0.40],
    'ECRS': [0.30, 0.70],
    'RAMPING_UNCERTAINTY': [0.40, 0.80]
}
SERVICE_KEY_TO_WINNING_RATE_CATEGORY = {
    'RegU': 'REGULATION', 'RegD': 'REGULATION', 'RegUp': 'REGULATION', 'RegDown': 'REGULATION',
    'Spin': 'SPIN', 'Syn': 'SPIN', 'Spin10': 'SPIN',
    'NSpin': 'NONSPIN', 'NSpin10': 'NONSPIN',
    'Sup': 'SUPPLEMENTAL',
    'Res30': 'RESERVE_30MIN', 'OR30': 'RESERVE_30MIN', 'TMR': 'RESERVE_30MIN', 'Rse': 'RESERVE_30MIN',
    'ECRS': 'ECRS',
    'RamU': 'RAMPING_UNCERTAINTY', 'RamD': 'RAMPING_UNCERTAINTY',
    'RMU': 'RAMPING_UNCERTAINTY', 'RMD': 'RAMPING_UNCERTAINTY',
    'UncU': 'RAMPING_UNCERTAINTY', 'STR': 'RAMPING_UNCERTAINTY'
}

DEPLOYMENT_FACTOR_CATEGORIES = {
    'SPIN': {'prob_range': [0.001, 0.005], 'mag_range': [0.2, 0.8]},
    'NONSPIN': {'prob_range': [0.0001, 0.001], 'mag_range': [0.2, 0.7]},
    'SUPPLEMENTAL': {'prob_range': [0.0001, 0.001], 'mag_range': [0.2, 0.7]},
    'RESERVE_30MIN': {'prob_range': [0.00001, 0.0001], 'mag_range': [0.1, 0.6]},
    'ECRS': {'prob_range': [0.01, 0.05], 'mag_range': [0.1, 0.8]},
    'RAMPING_UNCERTAINTY': {'prob_range': [0.05, 0.15], 'mag_range': [0.1, 0.8]}
}
SERVICE_KEY_TO_DEPLOYMENT_CATEGORY = {
    'Spin': 'SPIN', 'Syn': 'SPIN', 'Spin10': 'SPIN',
    'NSpin': 'NONSPIN', 'NSpin10': 'NONSPIN',
    'Sup': 'SUPPLEMENTAL',
    'Res30': 'RESERVE_30MIN', 'OR30': 'RESERVE_30MIN', 'TMR': 'RESERVE_30MIN', 'Rse': 'RESERVE_30MIN',
    'ECRS': 'ECRS',
    'RamU': 'RAMPING_UNCERTAINTY', 'RamD': 'RAMPING_UNCERTAINTY',
    'RMU': 'RAMPING_UNCERTAINTY', 'RMD': 'RAMPING_UNCERTAINTY',
    'UncU': 'RAMPING_UNCERTAINTY', 'STR': 'RAMPING_UNCERTAINTY'
}

def generate_smoothed_series(num_hours, loc, scale, smoothing_window, clip_min, clip_max,
                             daily_cycles=0, daily_amplitude=0,
                             weekly_cycles=0, weekly_amplitude=0):
    """Helper to generate a smoothed, clipped time series with optional seasonality."""
    series = np.random.normal(loc=loc, scale=scale, size=num_hours)
    if daily_cycles > 0 and daily_amplitude > 0:
        series += daily_amplitude * np.sin(np.linspace(0, 2 * np.pi * daily_cycles * (num_hours / (24*daily_cycles)), num_hours))
    if weekly_cycles > 0 and weekly_amplitude > 0:
        series += weekly_amplitude * np.sin(np.linspace(0, 2 * np.pi * weekly_cycles * (num_hours / (24*7*weekly_cycles)), num_hours))

    if smoothing_window > 1:
        series = np.convolve(series, np.ones(smoothing_window)/smoothing_window, mode='same')
    return np.clip(series, clip_min, clip_max)

# --- 1. Generate Hourly Deployment Factor Data (DeploymentFactor_hourly.csv) ---
def generate_deploy_factor_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates the hourly deployment factor CSV file."""
    print(f"Generating hourly deployment factors for {iso}: {output_path}")
    df_deploy = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])
    generated_factors_count = 0
    processed_reg_services = set()

    if iso in ISO_REGULATION_PAIRS:
        up_key, down_key = ISO_REGULATION_PAIRS[iso]
        if up_key in iso_services and down_key in iso_services:
            print(f"  Generating patterned regulation deployment factors for {up_key} and {down_key} in {iso}")

            # 1. Generate P_D_hourly with patterns
            P_D_hourly = generate_smoothed_series(num_hours, loc=0.175, scale=0.05, smoothing_window=12,
                                                  clip_min=0.05, clip_max=0.30,
                                                  daily_cycles=3, daily_amplitude=0.03, # Multiple short cycles per day
                                                  weekly_cycles=1, weekly_amplitude=0.02)

            # 2. Generate hourly bias with patterns, targeting mean 0.44
            # Noise component for bias, smoothed, zero mean
            bias_noise = generate_smoothed_series(num_hours, loc=0, scale=0.02, smoothing_window=24*3, # Smoothed over 3 days
                                                  clip_min=-0.05, clip_max=0.05, # Bias can fluctuate by +/- 0.05
                                                  daily_cycles=1, daily_amplitude=0.01)
            hourly_target_bias = 0.44 + bias_noise

            # 3. Calculate initial P_U_hourly
            P_U_hourly = P_D_hourly + hourly_target_bias

            # 4. Ensure P_U and P_D are valid and their sum is <= max_activity_sum
            # This step simultaneously clips P_U and P_D and adjusts them if their sum is too high,
            # while trying to maintain the hourly_target_bias.
            max_activity_sum = 0.98 # e.g. P_U + P_D <= 0.98, allowing P_0 >= 0.02
            P_U_final = np.zeros(num_hours)
            P_D_final = np.zeros(num_hours)

            for h in range(num_hours):
                p_u_h = P_U_hourly[h]
                p_d_h = P_D_hourly[h]
                current_bias_h = hourly_target_bias[h] # This is the P_U - P_D we want for this hour

                # If sum P_U + P_D exceeds max_activity_sum, adjust while preserving bias B_h
                # P_U_new = (max_activity_sum + B_h) / 2
                # P_D_new = (max_activity_sum - B_h) / 2
                if (p_u_h + p_d_h) > max_activity_sum:
                    p_u_h = (max_activity_sum + current_bias_h) / 2
                    p_d_h = (max_activity_sum - current_bias_h) / 2
                
                # Clip to individual bounds after sum adjustment
                p_u_h = np.clip(p_u_h, 0.01, max_activity_sum - 0.01) # Ensure P_D can be at least 0.01
                p_d_h = np.clip(p_d_h, 0.01, max_activity_sum - 0.01) # Ensure P_U can be at least 0.01

                # Final check: ensure p_u_h + p_d_h <= max_activity_sum after clipping
                if (p_u_h + p_d_h) > max_activity_sum:
                     # if still too high, proportionally scale down (this might slightly alter bias for this hour)
                    scale_factor = max_activity_sum / (p_u_h + p_d_h)
                    p_u_h *= scale_factor
                    p_d_h *= scale_factor
                
                P_U_final[h] = p_u_h
                P_D_final[h] = p_d_h

            # 5. Adjust overall series P_U_final and P_D_final to ensure mean(P_U - P_D) is close to 0.44
            current_mean_bias = np.mean(P_U_final - P_D_final)
            correction_to_mean_bias = 0.44 - current_mean_bias
            
            # Apply correction by shifting P_U and P_D
            # P_U_adj = P_U_final + correction_to_mean_bias / 2
            # P_D_adj = P_D_final - correction_to_mean_bias / 2
            # Simpler: adjust one side or adjust the original bias generation.
            # For now, let's adjust P_U and P_D and re-clip.
            P_U_adj = P_U_final + correction_to_mean_bias / 2
            P_D_adj = P_D_final - correction_to_mean_bias / 2

            # Final clipping after overall mean bias adjustment
            P_U_adj = np.clip(P_U_adj, 0.01, 0.97) # Max for P_U slightly less than max_activity_sum
            P_D_adj = np.clip(P_D_adj, 0.01, 0.97) # Max for P_D slightly less than max_activity_sum

            # Ensure sum constraint is still met as best as possible after final adjustment
            for h in range(num_hours):
                if (P_U_adj[h] + P_D_adj[h]) > max_activity_sum:
                    # If sum is violated, prioritize maintaining the achieved mean bias.
                    # Could reduce both proportionally, or cap one. For simplicity, cap sum.
                    # This can be complex to do perfectly without iteration.
                    # A simpler method is to ensure the initial patterns + bias don't create extreme sums.
                    # The (max_activity_sum + B_h)/2 logic helps a lot.
                    # If still over, we might slightly compromise the hourly sum to preserve bias.
                    # Or, simply clip one of them if sum is too high:
                    if P_U_adj[h] > (max_activity_sum - P_D_adj[h]):
                        P_U_adj[h] = max_activity_sum - P_D_adj[h]

            df_deploy[f'deploy_factor_{up_key}_{iso}'] = np.round(np.clip(P_U_adj,0.01,0.97), 4)
            df_deploy[f'deploy_factor_{down_key}_{iso}'] = np.round(np.clip(P_D_adj,0.01,0.97), 4)
            
            # print(f"  {iso} Reg final mean bias: {np.mean(df_deploy[f'deploy_factor_{up_key}_{iso}'] - df_deploy[f'deploy_factor_{down_key}_{iso}']):.4f}")

            processed_reg_services.add(up_key)
            processed_reg_services.add(down_key)
            generated_factors_count += 2

    # Handle other (contingency/ramping) reserve services
    for service_key in iso_services:
        if service_key in processed_reg_services:
            continue

        factor_col_name = f'deploy_factor_{service_key}_{iso}'
        category_key = SERVICE_KEY_TO_DEPLOYMENT_CATEGORY.get(service_key)

        if category_key and category_key in DEPLOYMENT_FACTOR_CATEGORIES:
            cat_data = DEPLOYMENT_FACTOR_CATEGORIES[category_key]
            prob_min, prob_max = cat_data['prob_range']
            mag_min, mag_max = cat_data['mag_range']
            
            hourly_event_probs = np.random.uniform(prob_min, prob_max, num_hours)
            deploy_mask = np.random.rand(num_hours) < hourly_event_probs
            deploy_magnitudes = np.random.uniform(mag_min, mag_max, num_hours)
            deploy_values = deploy_magnitudes * deploy_mask
            df_deploy[factor_col_name] = np.round(deploy_values, 4)
            generated_factors_count += 1
        else:
            print(f"Warning: Deployment category not found for non-regulation service {service_key} in {iso}. Using very low default.")
            deploy_prob = 0.0001 
            deploy_mask = np.random.rand(num_hours) < deploy_prob
            deploy_magnitude = np.random.uniform(0.01, 0.1, num_hours) 
            deploy_values = deploy_magnitude * deploy_mask
            df_deploy[factor_col_name] = np.round(deploy_values, 4)
            generated_factors_count += 1
            
    if generated_factors_count > 0:
        df_deploy.index.name = 'Timestamp'
        df_deploy.to_csv(output_path)
        print(f"Hourly deployment factors for {iso} saved to {output_path}")
    else:
        print(f"No deployment factors generated for {iso}. Skipping file save for {output_path.name}")

# --- 2. Generate Hourly Mileage/Performance Factor Data (MileageMultiplier_hourly.csv) ---
def generate_mileage_multiplier_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates hourly mileage and performance factors for regulation services,
       using realistic mileage from PDF data and ensuring consistency for pairs."""
    print(f"Generating hourly mileage/performance factors for {iso}: {output_path}")
    df_mileage_perf = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])
    generated_factors_count = 0
    processed_reg_services = set()
    iso_specific_mileage_map = ISO_MILEAGE_MAP.get(iso, {})

    # Handle paired regulation services first for consistent mileage
    if iso in ISO_REGULATION_PAIRS:
        up_key, down_key = ISO_REGULATION_PAIRS[iso]
        if up_key in iso_services and down_key in iso_services:
            print(f"  Generating consistent mileage for {up_key} and {down_key} in {iso}")
            base_mileage_up = iso_specific_mileage_map.get(up_key)
            # base_mileage_down = iso_specific_mileage_map.get(down_key) # Should be same based on current map

            if base_mileage_up is not None: # Assume consistency, use up_key's base
                noise_scale = 0.1 * base_mileage_up if base_mileage_up > 0.5 else 0.05 * base_mileage_up
                # Generate one common random noise series for the variation
                common_noise = np.random.normal(0, noise_scale, num_hours)
                mileage_values_up = np.maximum(0.01, base_mileage_up + common_noise)
                # Apply the same derived values to the down service
                mileage_values_down = mileage_values_up # Ensure identical mileage after noise

                perf_noise = np.random.normal(0, 0.05, num_hours) # Common performance noise
                perf_values_common = np.clip(0.95 + perf_noise, 0.7, 1.0)

                df_mileage_perf[f'mileage_factor_{up_key}_{iso}'] = np.round(mileage_values_up, 4)
                df_mileage_perf[f'performance_factor_{up_key}_{iso}'] = np.round(perf_values_common, 4)
                df_mileage_perf[f'mileage_factor_{down_key}_{iso}'] = np.round(mileage_values_down, 4)
                df_mileage_perf[f'performance_factor_{down_key}_{iso}'] = np.round(perf_values_common, 4)
                
                processed_reg_services.add(up_key)
                processed_reg_services.add(down_key)
                generated_factors_count +=2 # For two services
            else:
                print(f"Warning: Base mileage not found for {up_key} in {iso} for paired generation.")


    # Handle any unpaired regulation services (should not happen with current map for paired ISOs)
    for service_key in iso_services:
        if service_key in processed_reg_services:
            continue

        is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                "RegUp" in service_key or "RegDown" in service_key
        if is_regulation_service:
            mileage_col_name = f'mileage_factor_{service_key}_{iso}'
            perf_col_name = f'performance_factor_{service_key}_{iso}'
            base_mileage = iso_specific_mileage_map.get(service_key)

            if base_mileage is None:
                print(f"Warning: Mileage not defined for unpaired reg service {service_key} in {iso}. Using default 1.0-2.5.")
                mileage_values = np.maximum(0.5, 1.5 + np.random.normal(0, 0.3, num_hours))
            else:
                noise_scale = 0.1 * base_mileage if base_mileage > 0.5 else 0.05 * base_mileage
                mileage_values = np.maximum(0.01, base_mileage + np.random.normal(0, noise_scale, num_hours))
            
            df_mileage_perf[mileage_col_name] = np.round(mileage_values, 4)
            perf_values = np.clip(0.95 + np.random.normal(0, 0.05, num_hours), 0.7, 1.0) 
            df_mileage_perf[perf_col_name] = np.round(perf_values, 4)
            generated_factors_count +=1

    if generated_factors_count > 0:
        df_mileage_perf.index.name = 'Timestamp'
        df_mileage_perf.to_csv(output_path)
        print(f"Hourly mileage/performance factors for {iso} saved to {output_path}")
    else:
         print(f"No regulation mileage/performance factors generated for {iso}. Skipping file save for {output_path.name}")

# --- 3. Generate Hourly Winning Rate Data (WinningRate_hourly.csv) ---
def generate_winning_rate_hourly(output_path: Path, iso: str, time_index: pd.DatetimeIndex, num_hours=HOURS_IN_YEAR):
    """Generates the hourly AS winning rate CSV file for all services,
       using realistic rates from PDF data and ensuring consistency for paired regulation services."""
    print(f"Generating hourly AS winning rates for {iso}: {output_path}")
    df_winrate = pd.DataFrame(index=time_index)
    iso_services = ISO_SERVICE_MAP.get(iso, [])
    processed_reg_services = set()

    if iso in ISO_REGULATION_PAIRS:
        up_key, down_key = ISO_REGULATION_PAIRS[iso]
        if up_key in iso_services and down_key in iso_services:
            # print(f"  Generating consistent winning rates for {up_key} and {down_key} in {iso}")
            category_key = SERVICE_KEY_TO_WINNING_RATE_CATEGORY.get(up_key)
            if category_key and category_key in WINNING_RATE_CATEGORIES:
                min_rate, max_rate = WINNING_RATE_CATEGORIES[category_key]
                common_win_rates = np.random.uniform(min_rate, max_rate, num_hours)
                final_rates = np.round(np.clip(common_win_rates, 0.01, 1.0), 4)
                
                df_winrate[f'winning_rate_{up_key}_{iso}'] = final_rates
                df_winrate[f'winning_rate_{down_key}_{iso}'] = final_rates
                processed_reg_services.add(up_key)
                processed_reg_services.add(down_key)
            else:
                print(f"Error: Winning rate category not found for regulation key {up_key} in {iso}.")

    for service_key in iso_services:
        if service_key in processed_reg_services:
            continue
        rate_col_name = f'winning_rate_{service_key}_{iso}'
        category_key = SERVICE_KEY_TO_WINNING_RATE_CATEGORY.get(service_key)

        if category_key and category_key in WINNING_RATE_CATEGORIES:
            min_rate, max_rate = WINNING_RATE_CATEGORIES[category_key]
            win_rates = np.random.uniform(min_rate, max_rate, num_hours)
        else:
            is_regulation_service = "RegU" in service_key or "RegD" in service_key or \
                                    "RegUp" in service_key or "RegDown" in service_key
            if is_regulation_service and SERVICE_KEY_TO_WINNING_RATE_CATEGORY.get(service_key) == 'REGULATION':
                 min_rate, max_rate = WINNING_RATE_CATEGORIES['REGULATION']
                 win_rates = np.random.uniform(min_rate, max_rate, num_hours)
            else:
                print(f"Warning: Winning rate category not defined for service {service_key} in {iso}. Using default 0.5-0.8.")
                win_rates = np.random.uniform(0.5, 0.8, num_hours) 
        
        df_winrate[rate_col_name] = np.round(np.clip(win_rates, 0.01, 1.0), 4) 

    if not df_winrate.empty:
        df_winrate.index.name = 'Timestamp'
        df_winrate.to_csv(output_path)
        print(f"Hourly AS winning rates for {iso} saved to {output_path}")
    else:
        print(f"No winning rates generated for {iso}. Skipping file save for {output_path.name}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    full_2024_index = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='h')
    time_index_hourly = full_2024_index[~((full_2024_index.month == 2) & (full_2024_index.day == 29))]

    if len(time_index_hourly) != HOURS_IN_YEAR:
        print(f"Warning: Filtered 2024 index had {len(time_index_hourly)} hours. Forcing 8760 hours from 2024-01-01.")
        time_index_hourly = pd.date_range(start='2024-01-01', periods=HOURS_IN_YEAR, freq='h')
        
    print(f"Generated time index for {len(time_index_hourly)} hours, starting {time_index_hourly[0]}, ending {time_index_hourly[-1]}")

    for target_iso in ISO_SERVICE_MAP.keys():
        print(f"\n--- Generating hourly files for {target_iso} ---")
        iso_output_dir = BASE_OUTPUT_DIR / target_iso
        iso_output_dir.mkdir(parents=True, exist_ok=True)

        deploy_factor_path = iso_output_dir / "DeploymentFactor_hourly.csv"
        generate_deploy_factor_hourly(deploy_factor_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

        mileage_perf_path = iso_output_dir / "MileageMultiplier_hourly.csv" 
        generate_mileage_multiplier_hourly(mileage_perf_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

        winrate_path = iso_output_dir / "WinningRate_hourly.csv"
        generate_winning_rate_hourly(winrate_path, target_iso, time_index_hourly, num_hours=HOURS_IN_YEAR)

    print("\n--- Data generation complete for specified ancillary service factors ---")