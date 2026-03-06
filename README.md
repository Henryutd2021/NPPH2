# NPPH2

NPPH2 is a Python-based modeling and analysis workflow for flexible nuclear-hydrogen systems. The project combines hourly operational optimization, techno-economic analysis (TEA), life cycle assessment (LCA), sensitivity studies, and figure generation for integrated nuclear plants that can sell power to the grid, produce hydrogen, and participate in ancillary service markets.

The repository has evolved into a multi-stage pipeline:

1. Run optimization to generate hourly dispatch and revenue results.
2. Feed those results into TEA to evaluate retrofit and greenfield business cases.
3. Use TEA and optimization outputs as inputs to LCA.
4. Aggregate and visualize outputs through dedicated scripts and the plotting notebook.

## What The Project Supports

- Multi-ISO electricity market analysis for `CAISO`, `ERCOT`, `ISONE`, `MISO`, `NYISO`, `PJM`, and `SPP`
- Hybrid system configurations with nuclear generation, electrolyzers, hydrogen storage, and batteries
- Hourly dispatch optimization with ancillary service participation
- TEA metrics including `NPV`, `IRR`, `payback`, `LCOH`, and `LCOE`
- Existing-plant retrofit and greenfield nuclear-hydrogen business cases
- Reactor-by-reactor batch studies using plant metadata from `input/hourly_data/NPPs info.csv`
- LCA workflows built on top of completed optimization and TEA runs
- Price sensitivity workflows and downstream result aggregation
- Notebook-driven figure generation for reports and manuscripts

## Current Workflow Map

| Stage | Primary entry point | Purpose | Main outputs |
| --- | --- | --- | --- |
| Optimization, single ISO | `executables/opt/opt_main.py` | Run the standardized Pyomo model for one ISO | `output/opt/Results_Standardized/` and logs |
| Optimization, batch reactor study | `executables/opt/opt_cs1.py` | Run plant-specific optimization for reactors listed in `NPPs info.csv` | `output/opt/cs1/` |
| TEA, ISO-level | `executables/tea/tea_main.py` | Run TEA on standardized optimization outputs | `output/tea/iso/` |
| TEA, batch reactor study | `executables/tea/tea_cs1.py` | Run detailed TEA for each reactor in `output/opt/cs1/` | `output/tea/cs1/` |
| LCA | `executables/lca/run_lca.py` | Discover reactors with both OPT and TEA results and run LCA | `output/lca/reactor_reports/` |
| Price sensitivity | `executables/sensitivity/sa_price_sensitivity.py` | Run optimization + TEA across hourly electricity price scenarios | `output/opt/sa_price/`, `output/tea/sa_price/` |
| Price sensitivity aggregation | `executables/sensitivity/extract_price_sensitivity_results.py` | Parse TEA/OPT outputs into an Excel workbook | `output/tea/sa_price/*.xlsx` |
| Plotting | `plotting/plotting.ipynb` | Generate publication-style figures from results | `output/figs/` |

## Repository Structure

```text
NPPH2/
├── src/
│   ├── opt/              # Core optimization model, constraints, I/O, result extraction
│   ├── tea/              # TEA engine, calculations, incentives, reporting, visualization
│   ├── lca/              # LCA models, integration, reporting, analysis logic
│   └── logger_utils/     # Shared logging and progress utilities
├── executables/
│   ├── opt/              # Runnable optimization entry points
│   ├── tea/              # Runnable TEA entry points
│   ├── lca/              # Runnable LCA entry points
│   ├── sensitivity/      # Sensitivity pipelines and post-processing scripts
│   └── path_setup.py     # Shared import-path bootstrapper
├── input/
│   └── hourly_data/      # ISO hourly prices, ancillary-service data, system inputs, NPP metadata
├── output/
│   ├── opt/              # Optimization outputs
│   ├── tea/              # TEA reports and summaries
│   ├── lca/              # LCA reports
│   ├── figs/             # Final figures generated from notebook/scripts
│   └── logs/             # Runtime logs
├── plotting/
│   └── plotting.ipynb    # Main notebook for figures
├── tools/                # Parsing and downstream analysis helpers
├── docs/                 # Project notes and workflow-specific docs
└── tests/                # Test suite for optimization and result-processing logic
```

## Installation

Create a Python environment and install the project dependencies from `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key libraries used in the current stack include `pyomo`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `gridstatus`, and `gurobipy`.

## Input Data

The project expects input files under `input/hourly_data/`, including:

- `sys_data_advanced.csv` for system-level technical and economic parameters
- `NPPs info.csv` for plant-level metadata used by batch case-study workflows
- ISO-specific hourly electricity price files such as `Price_hourly.csv`
- ISO-specific ancillary service price and market files such as `Price_ANS_hourly.csv`, `WinningRate_hourly.csv`, `DeploymentFactor_hourly.csv`, and `MileageMultiplier_hourly.csv`

## Configuration

The two most important configuration files are:

- `src/opt/config.py` for optimization settings and feature flags
- `src/tea/config.py` for TEA assumptions, cost parameters, tax policies, and output defaults

Examples of optimization flags currently exposed in `src/opt/config.py` include:

```python
TARGET_ISO = "ERCOT"
HOURS_IN_YEAR = 8760

ENABLE_NUCLEAR_GENERATOR = True
ENABLE_ELECTROLYZER = True
ENABLE_LOW_TEMP_ELECTROLYZER = False
ENABLE_BATTERY = True
ENABLE_H2_STORAGE = True

ENABLE_NONLINEAR_TURBINE_EFF = True
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING = True
ENABLE_STARTUP_SHUTDOWN = True
ENABLE_OPTIMAL_H2_STORAGE_SIZING = True
SIMULATE_AS_DISPATCH_EXECUTION = True
```

On the TEA side, `src/tea/config.py` contains project lifetime assumptions, case classification, nuclear cost parameters, replacement schedules, tax incentive policies, and sensitivity-analysis overrides.

## Quick Start

Run commands from the repository root.

### 1. Single-ISO Optimization

Use this when you want to solve the standardized optimization model for one ISO.

```bash
python executables/opt/opt_main.py --iso ERCOT --solver gurobi --hours 8760
```

Useful CLI options:

- `--iso` target ISO
- `--solver` Pyomo solver name
- `--hours` simulation horizon
- `--debug-infeasibility` write an LP file for IIS debugging when supported by the solver

### 2. ISO-Level TEA

This workflow consumes a standardized optimization CSV and produces TEA outputs for one ISO.

```bash
python executables/tea/tea_main.py --iso ERCOT
```

Optional overrides include:

- `--input-file` to point TEA at a specific hourly results CSV
- `--output-dir` to redirect TEA outputs
- `--project-lifetime`, `--construction-years`, `--discount-rate`, `--tax-rate`
- `--enable-battery` or `--disable-battery`
- `--enable-greenfield` or `--disable-greenfield`

### 3. Batch Reactor Optimization For Case Study Work

This workflow reads `input/hourly_data/NPPs info.csv`, adjusts plant-specific parameters and remaining life, and runs optimization reactor by reactor.

```bash
python executables/opt/opt_cs1.py
```

Outputs are written to:

- `output/opt/cs1/`

Each reactor produces an hourly CSV such as:

- `<Plant>_<Generator>_<ISO>_<RemainingYears>_hourly_results.csv`

### 4. Batch Reactor TEA

This workflow processes all reactor-level optimization outputs from `output/opt/cs1/` and generates text-only TEA reports.

```bash
python executables/tea/tea_cs1.py
```

Outputs are written to:

- `output/tea/cs1/`
- `output/logs/cs1/`

For each reactor, the main outputs include:

- `<ISO>_TEA_Summary_Report.txt`
- `<ISO>_Comprehensive_TEA_Summary.txt`

### 5. LCA

The LCA workflow is downstream of optimization and TEA. It automatically discovers reactors that have both result types available.

```bash
python executables/lca/run_lca.py --mc 1000
```

Useful options:

- `--mc` Monte Carlo iterations
- `--tea-dir` custom TEA results directory
- `--opt-dir` custom optimization results directory
- `--output-dir` custom LCA output directory
- `--verbose` enable detailed logging

Outputs are written to:

- `output/lca/reactor_reports/`

### 6. Price Sensitivity Pipeline

The current integrated sensitivity workflow is the hourly electricity price sensitivity pipeline. For each scenario, it runs optimization first and then TEA for every eligible plant.

```bash
python executables/sensitivity/sa_price_sensitivity.py
```

Current price scenarios defined in the script are:

- `price_m10pct`
- `price_m5pct`
- `price_p5pct`
- `price_p10pct`
- `price_p20pct`
- `price_p30pct`

Outputs are organized by scenario under:

- `output/opt/sa_price/`
- `output/tea/sa_price/`
- `output/logs/sa_price/`

To aggregate these outputs into an Excel workbook:

```bash
python executables/sensitivity/extract_price_sensitivity_results.py
```

### 7. Plotting

The main plotting workflow lives in `plotting/plotting.ipynb`. It reads processed outputs and generates figures under `output/figs/`.

Recent figure outputs in the repository indicate the notebook is currently used for:

- case-comparison figures
- NPV and LCOE comparisons
- ancillary service revenue plots
- country-level nuclear cost and capacity figures
- price sensitivity figures and heatmaps

## Important Notes On Workflow Dependencies

- `tea_main.py` typically expects optimization output to already exist.
- `tea_cs1.py` expects reactor-level hourly result CSVs under `output/opt/cs1/`.
- `run_lca.py` is not a first-step workflow; it depends on both TEA and optimization outputs.
- `sa_price_sensitivity.py` is currently the most up-to-date integrated sensitivity pipeline in the repository.
- `executables/sensitivity/sa.py` still exists, but it appears to reflect an older workflow and should be treated as a legacy script unless you are intentionally working with it.

## Testing

The repository includes tests under `tests/`, covering parts of the optimization stack such as:

- configuration behavior
- model construction
- constraints
- result processing
- revenue and regulation balance logic

To run the test suite:

```bash
pytest
```

Or use the helper script:

```bash
python tests/run_tests.py
```

## Supporting Utilities

Additional downstream analysis utilities are available in `tools/`, including:

- TEA summary parsing
- LCA extraction
- sensitivity analysis support
- ancillary service analysis and visualization

The `docs/` directory also contains workflow-specific notes, including sensitivity-analysis summaries and project reorganization notes.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.

## Contact

- Honglin Li, Ph.D.
- <honglin.li@utdallas.edu>
- UT-Dallas, INL
