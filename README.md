# NPPH2: Optimizing Nuclear Plant Flexibility and Hydrogen Production for Grid Ancillary Services

## Overview

This project models and optimizes the economic operation of integrated nuclear power and hydrogen production systems. It employs a Pyomo-based optimization framework to maximize profits by optimizing the allocation of electricity between grid sales and hydrogen production, while considering ancillary services participation.

The primary focus is on enhancing nuclear power plant flexibility through the integration of electrolyzers and/or battery storage systems, enabling participation in grid ancillary services markets. The project performs detailed techno-economic feasibility analyses and life cycle assessments of these hybrid systems and explores how new technologies and strategic investments can increase nuclear power plant revenue streams.

## Key Features

- **Multi-ISO Support**: Compatible with CAISO, ERCOT, ISONE, MISO, NYISO, PJM, and SPP markets
- **Flexible System Configuration**: Models various system configurations including:
  - Nuclear power generation with variable efficiency
  - Different electrolyzer technologies (PEM, AWE, SOEC, AEM)
  - Battery energy storage systems
  - Hydrogen storage
- **Economic Analysis**:
  - Levelized Cost of Hydrogen (LCOH) calculations
  - Levelized Cost of Storage (LCOS) calculations
  - Techno-economic assessment (TEA)
  - Revenue optimization from energy and ancillary services markets
  - Investment analysis and payback period calculations
- **Advanced Operational Features**:
  - Electrolyzer degradation tracking
  - Startup/shutdown scheduling
  - Ancillary services dispatch simulation
  - Ramping constraints
  - Realistic power balance
- **Ancillary Services Participation**:
  - Regulation Up/Down services
  - Spinning reserves
  - Non-spinning reserves
  - Responsive reserves (ECRS)
  - Ramping products

## Optimization and Techno-Economic Analysis (TEA)

The project includes comprehensive models for optimization and techno-economic analysis:

- **Optimization Framework**: Utilizes Pyomo to optimize the allocation of electricity between grid sales and hydrogen production, maximizing profits while considering ancillary services.
- **Techno-Economic Assessment (TEA)**: Evaluates the economic feasibility of different system configurations, including:
  - Levelized Cost of Hydrogen (LCOH) and Levelized Cost of Storage (LCOS)
  - Net Present Value (NPV), Internal Rate of Return (IRR), and payback period calculations
  - Revenue optimization from energy and ancillary services markets
  - Investment analysis and strategic recommendations

## Life Cycle Assessment (LCA)

The project includes a comprehensive Life Cycle Assessment (LCA) framework to evaluate the environmental impact of nuclear-hydrogen systems.

- **Before/After Retrofit Analysis**: Compares the carbon footprint of the nuclear plant before and after the integration of hydrogen production and/or battery storage.
- **Comprehensive Scope**: The analysis covers the entire lifecycle, including:
  - Nuclear fuel cycle (mining, milling, enrichment, fabrication)
  - Plant construction, operation, and decommissioning
  - Electrolyzer manufacturing
  - Grid electricity displacement effects
- **Uncertainty Analysis**: Employs Monte Carlo simulations to account for uncertainties in LCA parameters.
- **Detailed Reporting**: Generates detailed reports for each plant, quantifying carbon intensity (gCO₂-eq/kWh) and total emissions.

## Business Case Analysis

The project evaluates multiple business cases:

1. **Baseline**: Traditional nuclear power plant with grid electricity sales only
2. **Hydrogen Production**: Nuclear + Electrolyzer systems with various technologies
3. **Energy Storage**: Nuclear + Battery storage systems for time-shifting and grid services
4. **Hybrid Systems**: Nuclear + Electrolyzer + Battery combinations for maximum flexibility
5. **Revenue Stacking**: Optimized participation in multiple markets (energy, H₂, ancillary services)

## Requirements

```text
pyomo>=6.9.2
numpy>=2.2.5
pandas>=2.2.3
matplotlib>=3.10.1
seaborn>=0.13.2
plotly>=6.0.1
gurobipy>=12.0.1
gridstatus>=0.30.1
```

For the complete list of dependencies, see `requirements.txt`.

## Project Structure

- `src/`: Core model implementation, including optimization, TEA, and LCA frameworks.
- `executables/`: Main execution scripts organized by function.
  - `opt/`: Optimization-related scripts (`opt_main.py`, `opt_cs1.py`).
  - `tea/`: TEA analysis scripts (`tea_main.py`, `tea_cs1.py`, etc.).
  - `lca/`: LCA analysis scripts (`run_lca.py`).
  - `sensitivity/`: Sensitivity analysis scripts (`sa.py`, etc.).
- `tools/`: Utility and analysis scripts for parsing, analysis, and data extraction.
- `plotting/`: Summary of results data used for plotting and plotting Jupyter notebook.
- `docs/`: Project documentation and guides.
- `input/`: Input data files, including ISO-specific hourly market data.
- `output/`: All output results, organized by category (optimization, TEA, LCA, sensitivity, logs).
- `tests/`: Test cases for the project.

## Usage

1. Configure the system in `src/opt/config.py` and `src/tea/config.py`.
2. Prepare input data files in the `input/` directory.
3. Run the desired analysis from the project root directory. For example:

    - **Run the main optimization model:**

        ```bash
        python executables/opt/opt_main.py
        ```

    - **Run the Techno-Economic Analysis (TEA) for Case Study 1:**

        ```bash
        python executables/tea/tea_cs1.py
        ```

        This performs a detailed financial assessment, including NPV, IRR, payback period, levelized costs (LCOH, LCOS), and cash flow projections.

    - **Run the Life Cycle Assessment (LCA):**

        ```bash
        python executables/lca/run_lca.py
        ```

        This evaluates the environmental impact and carbon footprint.

    - **Run a sensitivity analysis:**

        ```bash
        python executables/sensitivity/sa.py
        ```

        This evaluates system performance across variations in parameters, market scenarios, and technologies.

4. Use scripts in `tools/` for post-processing and `src/opt/result_processing.py` for automated result analysis.

## Configuration Options

Key configuration parameters in `src/opt/config.py`:

```python
# ISO selection
TARGET_ISO: str = "ERCOT"  # Options: CAISO, ERCOT, ISONE, MISO, NYISO, PJM, SPP

# Component selection
ENABLE_NUCLEAR_GENERATOR: bool = True
ENABLE_ELECTROLYZER: bool = True
ENABLE_LOW_TEMP_ELECTROLYZER: bool = False
ENABLE_BATTERY: bool = True
ENABLE_H2_STORAGE: bool = True

# Advanced features
ENABLE_H2_CAP_FACTOR: bool = False
ENABLE_NONLINEAR_TURBINE_EFF: bool = True
ENABLE_ELECTROLYZER_DEGRADATION_TRACKING: bool = True
ENABLE_STARTUP_SHUTDOWN: bool = True
SIMULATE_AS_DISPATCH_EXECUTION: bool = True
```

TEA-specific configuration in `src/tea/config.py`:

```python
# TEA Parameters
PROJECT_LIFETIME_YEARS = 30
DISCOUNT_RATE = 0.08
CONSTRUCTION_YEARS = 2
TAX_RATE = 0.21

# Output directories
BASE_OUTPUT_DIR_DEFAULT = "output/tea"
LOG_DIR = "output/logs"
```

## Hydrogen Cost Analysis

The project includes comprehensive models for hydrogen cost analysis:

- Capital expenditure (CAPEX) modeling
- Operational expenditure (OPEX) modeling
- Stack replacement scheduling
- Water consumption costs
- Compression and storage costs
- Capital recovery factor calculations

## Sensitivity Analysis

The model enables detailed sensitivity analysis for key parameters:

- Electrolyzer capital costs
- Battery capital costs
- Hydrogen market prices
- Electricity market prices
- Grid ancillary service price dynamics
- Technology performance parameters
- Operational constraints

## Key Results

- Quantification of economic benefits from flexible nuclear operation
- Optimal sizing for electrolyzers and battery storage systems
- Break-even points for different technology combinations
- Revenue enhancement potential from ancillary services participation
- Comparative analysis of different ISO markets
- Investment prioritization recommendations

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries, please contact:

- Name: Honglin Li
- Email: <honglin.li@utdallas.edu>
- Organization: UT-Dallas, INL
