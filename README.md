# NPPH2: Optimizing Nuclear Plant Flexibility and Hydrogen Production for Grid Ancillary Services

## Overview

This project models and optimizes the economic operation of integrated nuclear power and hydrogen production systems. It employs a Pyomo-based optimization framework to maximize profits by optimizing the allocation of electricity between grid sales and hydrogen production, while considering ancillary services participation.

The primary focus is on enhancing nuclear power plant flexibility through the integration of electrolyzers and/or battery storage systems, enabling participation in grid ancillary services markets. The project performs detailed techno-economic feasibility analyses of these hybrid systems and explores how new technologies and strategic investments can increase nuclear power plant revenue streams.

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

- `src/`: Core model implementation
  - `model.py`: Main optimization model
  - `constraints.py`: Constraint definitions
  - `revenue_cost.py`: Revenue and cost calculations
  - `lcoh.py`: Levelized Cost of Hydrogen calculations
  - `lcos.py`: Levelized Cost of Storage calculations
  - `config.py`: Configuration parameters
  - `data_io.py`: Data input/output utilities
  - `utils.py`: Common utility functions
  - `result_processing.py`: Tools for analyzing optimization results
- `input/`: Input data files
- `output/`: Output results
- `tests/`: Test cases
- `TEA_results/`: Techno-economic analysis results
- `sensitivity_analysis_results/`: Results from sensitivity analyses
- `runs/`: Simulation runs with different parameter configurations
- `temp_sensitivity_runs/`: Temporary sensitivity analysis results
- `data/`: Data utilities and sources
  - `data_gen.py`: Generates synthetic datasets required by the optimization framework
  - `data_ana.ipynb`: Notebook to clean and analyze real-world data
  - `Raw/`: Raw downloaded data files
  - `ISOs/`: ISO-specific data subsets

## Usage

1. Configure the system in `src/config.py`
2. Prepare input data files in the `input/` directory
3. Run the optimization model with:

   ```bash
   python runs/main.py
   ```

4. Use `src/result_processing.py` for automated result processing and analysis

5. The techno-economic analysis (TEA) mode performs detailed financial assessment including:

    ```bash
    python runs/tea.py
    ```

    - Net Present Value (NPV) calculations
    - Internal Rate of Return (IRR) analysis
    - Payback period estimation
    - Levelized cost metrics (LCOH, LCOS)
    - Annual cash flow projections

6. The sensitivity analysis (SA) mode evaluates system performance across:

    ```bash
    python runs/sa.py
    ```

    - Parameter variations (capital costs, efficiencies, etc.)
    - Market scenarios (different price trajectories)
    - Technology combinations
    - Operating strategies
    - Policy scenarios (subsidies, carbon pricing)

## Configuration Options

Key configuration parameters in `config.py`:

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
