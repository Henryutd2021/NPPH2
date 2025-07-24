# NPPH2 Project Reorganization Guide

## Overview

The project files have been successfully reorganized into a clearer, more modular directory structure. All file reference paths have been updated to ensure that functionality remains intact.

## New Directory Structure

### 📁 executables/ - Executable Scripts Directory

Main execution scripts organized by functional module:

- **executables/opt/** - Optimization-related scripts
  - `opt_main.py` - Main optimization script
  - `opt_cs1.py` - Case Study 1 optimization script

- **executables/tea/** - TEA (Techno-Economic Analysis) scripts
  - `tea_main.py` - Main TEA analysis script
  - `tea_cs1.py` - Case Study 1 TEA analysis
  - `tea_cs1_sensitivity.py` - TEA sensitivity analysis
  - `analyze_sensitivity_results.py` - Sensitivity results analysis
  - `run_tea_sensitivity_parallel.sh` - Parallel TEA sensitivity analysis script

- **executables/lca/** - LCA (Life Cycle Assessment) analysis scripts
  - `run_lca.py` - LCA analysis script

- **executables/sensitivity/** - General sensitivity analysis scripts
  - `sa.py` - Main sensitivity analysis script
  - `sa_*.py` - Various parameter sensitivity analysis scripts
  - `run_*.py` - Sensitivity analysis runner scripts
  - `run_*.sh` - Batch sensitivity analysis scripts

### 🔧 tools/ - Utilities and Analysis Scripts Directory

Helper scripts for analysis and data processing:

- **tools/analyzers/** - Analysis tools
  - `sensitivity_analyzer.py` - Sensitivity analyzer
  - `ancillary_analyzer.py` - Ancillary services analyzer
  - `breakdown_analysis.py` - Breakdown analysis
  - `tea_data_summary.py` - TEA data summary
  - `ancillary_visualizer.py` - Ancillary services visualizer

- **tools/parsers/** - Parsing tools
  - `tea_parser.py` - TEA results parser
  - `tea_summary_parser.py` - TEA summary parser
  - `run_tea_summary_parser.py` - TEA summary parser runner

- **tools/extractors/** - Data extraction tools
  - `lca_extractor.py` - LCA results extractor

### 📊 analysis/ - Analysis Files and Results Directory

Analysis-related files and data:

- **analysis/notebooks/** - Jupyter Notebooks
  - `plotting.ipynb` - Plotting analysis notebook

- **analysis/reports/** - Analysis reports
  - `plotting data.xlsx` - Plotting data
  - `sa.xlsx` - Sensitivity analysis data
  - `lcoh_*.txt` - LCOH analysis report files

### 📚 docs/ - Documentation Directory

Project documentation and guides:

- **docs/**
  - `document.md` - Project documentation
  - `README_*.md` - README documents for various modules
  - `reorganization-guide.md` - Reorganization guide (this document)

## Usage

### Running Scripts

Due to the change in file locations, you should now run scripts from the project root directory:

```bash
# Optimization Analysis
python executables/opt/opt_main.py --iso PJM

# TEA Analysis
python executables/tea/tea_main.py

# LCA Analysis
python executables/lca/run_lca.py

# Sensitivity Analysis
python executables/sensitivity/sa.py
```

Alternatively, you can run scripts directly from their respective directories:

```bash
# Navigate to the directory and run
cd executables/tea
python tea_main.py

cd ../opt
python opt_main.py

cd ../lca
python run_lca.py
```

### Using Tools

```bash
# Analysis Tools
python tools/analyzers/sensitivity_analyzer.py

# Parsing Tools
python tools/parsers/tea_parser.py

# Extraction Tools
python tools/extractors/lca_extractor.py
```

## Path Update Summary

All path references within the project files have been automatically updated:

1.  **Import Paths** - All `src` module import paths have been set correctly.
2.  **Output Paths** - All output is still saved to the original `output/` directory.
3.  **Input Paths** - All input file paths have been updated to the correct relative paths.
4.  **Log Paths** - Log file paths have been updated.

## Validation Status

✅ **All main script import functionalities have been verified:**

- TEA module imports ✅
- OPT module imports ✅  
- Analysis tool imports ✅

## Advantages

1.  **Clearer Organizational Structure** - Organized by function, making files easier to find and maintain.
2.  **Modular Design** - Execution scripts are separate from tools.
3.  **Centralized Documentation** - All documents are unified in the `docs` directory.
4.  **Categorized Analysis** - Notebooks and reports are stored separately.
5.  **Backward Compatibility** - Output and input paths remain unchanged.

## Important Notes

- It is recommended to run all scripts from the project root directory or from within the script's own directory.
- Output files are still saved in their original `output/` directory structure.
- If you add new scripts, please place them in the appropriate subdirectory based on their function. 