# Sensitivity Analysis Summary

## Configuration Overview

This sensitivity analysis framework tests the impact of different ramp rate constraints on system optimization performance. The analysis covers 8 different scenarios with systematic variation of turbine and electrolyzer ramp rates.

## Script Configuration

### Test Matrix

| Component      | Ramp Rate Values (% per minute) | Scripts |
|----------------|---------------------------------|---------|
| **Turbine**    | 0.5, 1.0, 1.5, 2.0              | 4       |
| **Electrolyzer** | 0.5, 1.0, 1.5, 2.0            | 4       |
| **Total**      | -                               | **8**   |

### Script Details

#### Turbine Ramp Rate Scripts

1. **`sa_turbine_ramp_0.5.py`**
    - Ramp-Up/Down Rate: 0.5%/min
    - Parameters: `Turbine_RampUp_Rate_Percent_per_Min`, `Turbine_RampDown_Rate_Percent_per_Min`
2. **`sa_turbine_ramp_1.0.py`**
    - Ramp-Up/Down Rate: 1.0%/min
3. **`sa_turbine_ramp_1.5.py`**
    - Ramp-Up/Down Rate: 1.5%/min
4. **`sa_turbine_ramp_2.0.py`**
    - Ramp-Up/Down Rate: 2.0%/min

#### Electrolyzer Ramp Rate Scripts

1. **`sa_electrolyzer_ramp_0.5.py`**
    - Ramp-Up/Down Rate: 0.5%/min
    - Parameters: `Electrolyzer_RampUp_Rate_Percent_per_Min_HTE`, `Electrolyzer_RampDown_Rate_Percent_per_Min_HTE`
2. **`sa_electrolyzer_ramp_1.0.py`**
    - Ramp-Up/Down Rate: 1.0%/min
3. **`sa_electrolyzer_ramp_1.5.py`**
    - Ramp-Up/Down Rate: 1.5%/min
4. **`sa_electrolyzer_ramp_2.0.py`**
    - Ramp-Up/Down Rate: 2.0%/min

## Execution Strategy

### Sequential Processing

- **Execution Mode**: Sequential (one script at a time).
- **Execution Order**: Turbine scripts (0.5 → 2.0), then Electrolyzer scripts (0.5 → 2.0).
- **Resource Management**: A single-threaded approach prevents solver conflicts and resource contention.
- **Monitoring**: Real-time progress tracking with detailed logging.

### Batch Runner Features

- **Script**: `executables/sensitivity/run_all_sensitivity_scripts.sh`
- **Automation**: Runs all 8 scripts without user intervention.
- **Progress Tracking**: Shows current script progress and overall completion status.
- **Error Handling**: Continues execution even if individual scripts fail.
- **Logging**: Comprehensive logging for debugging and analysis.
- **Summary Report**: Provides a final report with success/failure counts and execution times.

## Expected Outcomes

### Research Questions

1. **How do different ramp rate constraints affect system flexibility?**
    - Lower ramp rates (e.g., 0.5%/min) may reduce system responsiveness.
    - Higher ramp rates (e.g., 2.0%/min) may increase operational flexibility.
2. **What is the trade-off between ramp rate restrictions and system performance?**
    - Tighter constraints may increase costs due to reduced operational flexibility.
    - Looser constraints may improve economic performance but require more capable equipment.
3. **How do the impacts of turbine vs. electrolyzer ramp rates compare?**
    - The analysis allows for a direct comparison across symmetric ramp rate values.

### Key Performance Metrics

- **Economic Performance**: Total system costs, revenue optimization.
- **Operational Flexibility**: Response to demand and price variations.
- **Technical Constraints**: Feasibility under different ramp rate limitations.
- **System Efficiency**: Overall energy conversion and storage efficiency.

## Analysis Framework

- **Symmetric Testing**: Using the same ramp rate values for both components enables direct comparison.
- **Incremental Steps**: 0.5%/min increments provide granular sensitivity analysis.
- **Realistic Range**: The 0.5-2.0%/min range covers typical industrial ramp rate capabilities.

## Technical Implementation

- **Modular Design**: Each script focuses on a single parameter variation.
- **Common Framework**: A consistent methodology is used across all scripts.
- **Resource Requirements**: Each script requires 1-4 hours of execution time and 100-500MB of storage. The total runtime is 8-32 hours.

## Usage Guidelines

1. **Preparation**: Verify input data and system resources.
2. **Execution**: Run the batch script: `./executables/sensitivity/run_all_sensitivity_scripts.sh`
3. **Monitoring**: Track progress through log files.
4. **Analysis**: Compare results across different ramp rate scenarios.
5. **Reporting**: Generate insights and recommendations.
