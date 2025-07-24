# Sensitivity Analysis Scripts

This document describes the sensitivity analysis scripts for testing different ramp rate configurations in the optimization framework. These scripts are located in the `executables/sensitivity/` directory.

## Overview

The sensitivity analysis tests the impact of different ramp rate values on system performance by systematically varying the ramp-up and ramp-down rates for turbines and electrolyzers.

## Available Scripts

### Turbine Ramp Rate Scripts (4 scripts)

- `sa_turbine_ramp_0.5.py` - Turbine ramp rate: 0.5%/min (up & down)
- `sa_turbine_ramp_1.0.py` - Turbine ramp rate: 1.0%/min (up & down)
- `sa_turbine_ramp_1.5.py` - Turbine ramp rate: 1.5%/min (up & down)
- `sa_turbine_ramp_2.0.py` - Turbine ramp rate: 2.0%/min (up & down)

### Electrolyzer Ramp Rate Scripts (4 scripts)

- `sa_electrolyzer_ramp_0.5.py` - Electrolyzer HTE ramp rate: 0.5%/min (up & down)
- `sa_electrolyzer_ramp_1.0.py` - Electrolyzer HTE ramp rate: 1.0%/min (up & down)
- `sa_electrolyzer_ramp_1.5.py` - Electrolyzer HTE ramp rate: 1.5%/min (up & down)
- `sa_electrolyzer_ramp_2.0.py` - Electrolyzer HTE ramp rate: 2.0%/min (up & down)

### Batch Runner Script

- `run_all_sensitivity_scripts.sh` - Runs all 8 scripts sequentially with monitoring.

## Usage

### Running Individual Scripts

Each script can be run independently from the project root:

```bash
python3 executables/sensitivity/sa_turbine_ramp_0.5.py
python3 executables/sensitivity/sa_electrolyzer_ramp_1.0.py
# ... etc.
```

### Running All Scripts Sequentially

To run all 8 scripts in sequence, execute the batch runner from the project root:

```bash
./executables/sensitivity/run_all_sensitivity_scripts.sh
```

This will:

- Run scripts one by one (not in parallel).
- Show progress for each script.
- Log all output to timestamped files.
- Provide a summary at the end.
- Report any failed scripts.

## Parameters Modified

### Turbine Scripts

- `Turbine_RampUp_Rate_Percent_per_Min`: Set to the script-specific value (0.5, 1.0, 1.5, or 2.0).
- `Turbine_RampDown_Rate_Percent_per_Min`: Set to the same value as the ramp-up rate.

### Electrolyzer Scripts

- `Electrolyzer_RampUp_Rate_Percent_per_Min_HTE`: Set to the script-specific value.
- `Electrolyzer_RampDown_Rate_Percent_per_Min_HTE`: Set to the same value as the ramp-up rate.

## Output Structure

Each script creates its own output directory in `output/sensitivity_analysis/`:

```
output/sensitivity_analysis/
├── turbine_ramp_0.5/
│   ├── turbine_ramp_0.5_results.csv
│   ├── turbine_ramp_0.5_summary.json
│   └── turbine_ramp_0.5_YYYYMMDD_HHMMSS.log
├── ... (other turbine and electrolyzer directories)
└── logs/
    ├── turbine_0.5_YYYYMMDD_HHMMSS.out
    └── ... (other log files)
```

## Result Files

- **CSV Results (`*_results.csv`)**: Detailed time-series data for all variables, including power generation, consumption, storage levels, and dispatch decisions.
- **JSON Summary (`*_summary.json`)**: High-level summary metrics, including total costs, revenues, KPIs, and solver status.
- **Log Files (`*.log` and `*.out`)**: Detailed execution logs, including parameter modifications, solver output, and runtime information.

## Script Features

- **Error Handling**: Automatic fallback to alternative solvers (CBC, GLPK) if Gurobi is unavailable.
- **Resource Management**: Automatic cleanup of temporary directories and timestamped outputs to prevent overwrites.
- **Monitoring**: Real-time progress reporting and execution time tracking.

## Sequential vs. Parallel Execution

The batch runner executes scripts **sequentially** to reduce system resource usage, prevent solver conflicts, ensure stable execution, and simplify debugging.

## Troubleshooting

- **Import Errors**: Ensure all source modules are correctly pathed.
- **Solver Issues**: Check your Gurobi license or ensure alternative solvers are installed.
- **Memory Issues**: Monitor system resources; the scripts include automatic cleanup.
- **Permission Issues**: Ensure scripts have execute permissions (`chmod +x`).

For debugging, check individual script logs in the `output/sensitivity_analysis/logs/` directory and run scripts individually to isolate issues.

## Performance Notes

- Each script can take from 30 minutes to several hours to run.
- The complete sequential execution of all 8 scripts may take 4-24 hours.
- Monitor disk space, as each script can generate 100-500MB of output.
- Consider running the analysis during off-peak hours due to its computational intensity.
