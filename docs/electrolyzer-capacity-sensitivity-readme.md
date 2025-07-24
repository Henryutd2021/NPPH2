# Electrolyzer Capacity Sensitivity Analysis

This document explains how to run a sensitivity analysis on the electrolyzer's maximum power capacity to test how different capacity limits affect the optimization results.

## Overview

This sensitivity analysis includes four different electrolyzer capacity settings:

- **100MW**: Electrolyzer power limit set to 100 MW
- **150MW**: Electrolyzer power limit set to 150 MW
- **250MW**: Electrolyzer power limit set to 250 MW
- **300MW**: Electrolyzer power limit set to 300 MW

## File Structure

The relevant scripts are now located in the `executables/sensitivity/` directory:

```
executables/sensitivity/
├── sa_electrolyzer_capacity_100.py
├── sa_electrolyzer_capacity_150.py
├── sa_electrolyzer_capacity_250.py
├── sa_electrolyzer_capacity_300.py
├── run_electrolyzer_capacity_sensitivity.py
├── run_electrolyzer_capacity_sensitivity.sh
└── ../../docs/electrolyzer-capacity-sensitivity-readme.md (This document)
```

## How to Run

### Method 1: Use the Python Batch Script (Recommended)

```bash
python3 executables/sensitivity/run_electrolyzer_capacity_sensitivity.py
```

**Features:**

- Supports parallel execution of two scripts at a time for efficiency.
- Provides detailed execution logs and progress indicators.
- Includes automatic error handling and retry mechanisms.
- Generates a structured results report.

### Method 2: Use the Bash Batch Script

```bash
./executables/sensitivity/run_electrolyzer_capacity_sensitivity.sh
```

**Features:**

- A lightweight implementation using Bash.
- Also supports parallel execution.
- Suitable for use in a pure Linux environment.

### Method 3: Run Scripts Individually

```bash
python3 executables/sensitivity/sa_electrolyzer_capacity_100.py
python3 executables/sensitivity/sa_electrolyzer_capacity_150.py
python3 executables/sensitivity/sa_electrolyzer_capacity_250.py
python3 executables/sensitivity/sa_electrolyzer_capacity_300.py
```

## Execution Plan

The batch scripts execute the analysis in the following order:

**Batch 1 (Parallel Execution):**

- 100MW and 150MW sensitivity analyses run concurrently.

**Wait for 10 seconds**

**Batch 2 (Parallel Execution):**

- 250MW and 300MW sensitivity analyses run concurrently.

## Output Results

### Result Directory Structure

```
output/sensitivity_analysis/
├── electrolyzer_capacity_100/
│   ├── electrolyzer_capacity_100.log
│   ├── electrolyzer_capacity_100_results.csv
│   └── electrolyzer_capacity_100_summary.json
├── electrolyzer_capacity_150/
│   ├── ...
├── electrolyzer_capacity_250/
│   ├── ...
├── electrolyzer_capacity_300/
│   ├── ...
└── electrolyzer_capacity/
    └── batch_logs/
        ├── electrolyzer_capacity_batch.log
        ├── electrolyzer_100MW.out
        ├── electrolyzer_150MW.out
        ├── electrolyzer_250MW.out
        └── electrolyzer_300MW.out
```

### Result File Descriptions

- **`.log` file**: Detailed execution log with debugging information.
- **`_results.csv` file**: Hourly optimization result data.
- **`_summary.json` file**: Summary of key metrics (total revenue, capacity factor, etc.).
- **`batch_logs/`**: Consolidated logs from the batch execution.

## Key Parameters

Each script modifies the following system parameter:

| Script                                 | Parameter Modified                   | Value   | Description                    |
| -------------------------------------- | ------------------------------------ | ------- | ------------------------------ |
| `sa_electrolyzer_capacity_100.py`      | `pElectrolyzer_max_upper_bound_MW`   | 100.0   | Electrolyzer power limit 100MW |
| `sa_electrolyzer_capacity_150.py`      | `pElectrolyzer_max_upper_bound_MW`   | 150.0   | Electrolyzer power limit 150MW |
| `sa_electrolyzer_capacity_250.py`      | `pElectrolyzer_max_upper_bound_MW`   | 250.0   | Electrolyzer power limit 250MW |
| `sa_electrolyzer_capacity_300.py`      | `pElectrolyzer_max_upper_bound_MW`   | 300.0   | Electrolyzer power limit 300MW |

## Log File Naming

Based on user preference, all log files use a fixed reactor name instead of a timestamp:

- **Main Log**: `electrolyzer_capacity_[capacity].log`
- **Batch Log**: `electrolyzer_capacity_batch.log`

Re-running the analysis will overwrite previous log files, keeping only the latest execution records.

## Requirements

### System Requirements

- Python 3.7+
- Required Python packages: `pyomo`, `pandas`, `logging`, etc.
- An optimization solver (Gurobi recommended, or CBC/GLPK as alternatives).

### Data Requirements

- **Input Data Directory**: `input/hourly_data/`
- **System Parameter File**: `input/hourly_data/sys_data_advanced.csv`

### Estimated Execution Time

- **Single Script**: 15-60 minutes (depending on data size and solver).
- **Total Time**: 30-120 minutes (with parallel execution).

## Troubleshooting

### Common Issues

1. **Solver Not Available**:
    - Check the Gurobi license.
    - Confirm that CBC or GLPK is installed.
2. **Input Data Missing**:
    - Ensure the `input/hourly_data/` directory exists.
    - Check the `sys_data_advanced.csv` file.
3. **Insufficient Memory**:
    - Monitor system memory usage.
    - Consider running scripts serially instead of in parallel.
4. **Permission Issues**:
    - Ensure you have write permissions for the output directory.
    - Add execute permissions to the bash script: `chmod +x executables/sensitivity/run_electrolyzer_capacity_sensitivity.sh`

### Checking Logs

If a script fails, check the corresponding log file:

- **Individual Script Log**: `output/sensitivity_analysis/electrolyzer_capacity_[capacity]/electrolyzer_capacity_[capacity].log`
- **Batch Log**: `output/sensitivity_analysis/electrolyzer_capacity/batch_logs/`

## Analysis Recommendations

After completing the sensitivity analysis, it is recommended to compare the following key metrics:

1. **Economic Indicators**:
    - Total Net Revenue
    - Hydrogen Revenue
    - Electricity Costs
2. **Technical Indicators**:
    - Electrolyzer Capacity Factor
    - Hydrogen Production
    - System Efficiency
3. **Operational Characteristics**:
    - Start/Stop Frequency
    - Ramping Behavior
    - Load Distribution

By comparing these metrics across different capacity settings, you can identify the optimal electrolyzer capacity configuration.
