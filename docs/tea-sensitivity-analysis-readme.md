# TEA CS1 Sensitivity Analysis System

This system is used to perform sensitivity analysis on the TEA (Techno-Economic Analysis) framework, focusing on the `total_fixed_costs_per_mw_year` parameter.

## System Overview

- **Parameter**: `total_fixed_costs_per_mw_year` (fixed costs for the nuclear plant)
- **Test Values**: 170,000, 200,000, 260,000, 290,000, 320,000 USD/MW/year
- **Baseline Value**: 230,000 USD/MW/year (default)
- **Feature**: All plotting functions are disabled to generate only text reports.

## System Components

### 1. Core Files

- `executables/tea/tea_cs1_sensitivity.py` - Main sensitivity analysis program
- `executables/tea/run_tea_sensitivity_parallel.sh` - Parallel execution script
- `executables/tea/analyze_sensitivity_results.py` - Results analysis script
- `src/tea/config.py` - Configuration file (modified to support parameter overrides)
- `src/tea/tea_engine.py` - TEA engine (modified to support disabling plots)

### 2. Directory Structure

```
output/
├── tea/
│   └── cs1_sensitivity/
│       ├── fixed_costs_170000/
│       ├── ... (other parameter value directories)
│       └── sensitivity_analysis_summary.txt
├── logs/
│   └── cs1/
│       └── sensitivity/
└── tea/
    └── cs1_sensitivity_analysis/
        ├── raw_sensitivity_results.csv
        ├── summary_tables/
        ├── comparison_reports/
        └── sensitivity_analysis_summary.json
```

## Usage

### 1. Run the Sensitivity Analysis

#### Method 1: Parallel Execution (Recommended)

```bash
# Run analysis for all parameter values in parallel
./executables/tea/run_tea_sensitivity_parallel.sh

# Options
./executables/tea/run_tea_sensitivity_parallel.sh -j 3        # Max 3 parallel jobs
./executables/tea/run_tea_sensitivity_parallel.sh --verbose   # Verbose output
./executables/tea/run_tea_sensitivity_parallel.sh --dry-run   # Show execution plan
```

#### Method 2: Run for a Specific Parameter Value

```bash
python3 executables/tea/tea_cs1_sensitivity.py --parameter-value 170000
python3 executables/tea/tea_cs1_sensitivity.py --parameter-value 200000
# ... and so on for other values
```

### 2. Analyze the Results

```bash
# Analyze the sensitivity analysis results
python3 executables/tea/analyze_sensitivity_results.py

# Verbose output
python3 executables/tea/analyze_sensitivity_results.py --verbose
```

## Output Files

### 1. Individual Parameter Output

Results for each parameter value are saved in `output/tea/cs1_sensitivity/fixed_costs_[value]/`:

```
fixed_costs_170000/
└── PlantName_GeneratorID_ISO_RemainingYears/
    ├── ISO_TEA_Summary_Report.txt
    ├── ISO_Cash_Flow_Analysis.txt
    └── ISO_Financial_Metrics.txt
```

### 2. Summary Analysis Output

Analysis results are saved in `output/tea/cs1_sensitivity_analysis/`:

- `raw_sensitivity_results.csv` - Raw results data
- `summary_tables/` - Summary tables (e.g., `parameter_sensitivity_summary.csv`)
- `comparison_reports/` - Comparison reports (e.g., `sensitivity_analysis_comparison.txt`)
- `sensitivity_analysis_summary.json` - Summary in JSON format

## Key Features

- **Disabled Plotting**: All plotting is disabled for faster execution and focus on numerical results.
- **Parameterized Configuration**: Supports dynamic modification of the `total_fixed_costs_per_mw_year` parameter.
- **Organized Results**: Results are saved separately for each parameter value.
- **Parallel Processing**: Supports parallel analysis of multiple parameter values.

## Monitoring and Logs

- **Execution Logs**: `output/logs/cs1/sensitivity/`
- **Parallel Execution Logs**: `output/logs/cs1/sensitivity/parallel_execution_[value].log`
- **Individual Analysis Logs**: `output/logs/cs1/sensitivity/tea_sensitivity_[value]_[timestamp].log`

## Troubleshooting

- **Input File Not Found**: Check that `output/opt/cs1/*_hourly_results.csv` files exist.
- **Parallel Task Failure**: Check system resources (`htop`) and consider reducing the number of parallel jobs.
- **Invalid Parameter Value**: Run `python3 executables/tea/tea_cs1_sensitivity.py --help` to see the list of valid values.
- **To Rerun**: Clean previous results with `rm -rf output/tea/cs1_sensitivity/` before running again.

## Performance Optimization

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ free space recommended
- **Parallel Jobs**: Adjust the number of parallel jobs based on your system's core count (`-j [cores]`).

## Interpreting Results

### Key Metrics

- **NPV (Net Present Value)**: USD
- **IRR (Internal Rate of Return)**: Percentage
- **LCOH (Levelized Cost of Hydrogen)**: USD/kg
- **Payback Period**: Years

### Sensitivity Insights

- The percentage change in NPV for a 10% change in the parameter.
- Sensitivity differences across various ISO regions.
- Identification of best and worst-performing reactors.

This analysis helps in identifying optimal cost parameters, assessing risks, and supporting investment decisions.
