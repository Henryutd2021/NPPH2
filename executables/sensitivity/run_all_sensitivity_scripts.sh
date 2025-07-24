#!/bin/bash

# Sensitivity Analysis Batch Runner
# This script helps run all 8 sensitivity analysis scripts sequentially

echo "=== Sensitivity Analysis Batch Runner ==="
echo "This script will run 8 sensitivity analysis scripts sequentially:"
echo "  - 4 Turbine ramp rate tests: 0.5, 1.0, 1.5, 2.0"
echo "  - 4 Electrolyzer ramp rate tests: 0.5, 1.0, 1.5, 2.0"
echo ""

# Check if user wants to proceed
read -p "Do you want to start all 8 scripts sequentially? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user."
    exit 0
fi

# Create output directory for logs
mkdir -p ../output/sensitivity_analysis/logs

# Get current timestamp for log filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting all 8 sensitivity analysis scripts sequentially..."
echo "Logs will be saved to ../output/sensitivity_analysis/logs/"
echo ""

# Define scripts to run in order
SCRIPTS=(
    "sa_turbine_ramp_0.5.py"
    "sa_turbine_ramp_1.0.py" 
    "sa_turbine_ramp_1.5.py"
    "sa_turbine_ramp_2.0.py"
    "sa_electrolyzer_ramp_0.5.py"
    "sa_electrolyzer_ramp_1.0.py"
    "sa_electrolyzer_ramp_1.5.py"
    "sa_electrolyzer_ramp_2.0.py"
)

SCRIPT_NAMES=(
    "Turbine 0.5%/min"
    "Turbine 1.0%/min"
    "Turbine 1.5%/min"
    "Turbine 2.0%/min"
    "Electrolyzer 0.5%/min"
    "Electrolyzer 1.0%/min"
    "Electrolyzer 1.5%/min"
    "Electrolyzer 2.0%/min"
)

LOG_NAMES=(
    "turbine_0.5"
    "turbine_1.0"
    "turbine_1.5"
    "turbine_2.0"
    "electrolyzer_0.5"
    "electrolyzer_1.0"
    "electrolyzer_1.5"
    "electrolyzer_2.0"
)

TOTAL_SCRIPTS=${#SCRIPTS[@]}
COMPLETED_COUNT=0

# Function to run a single script
run_script() {
    local script_file=$1
    local script_name=$2
    local log_name=$3
    local current_num=$4
    local total_num=$5
    
    echo "[$current_num/$total_num] Running: $script_name"
    echo "  Script: $script_file"
    echo "  Log: ../output/sensitivity_analysis/logs/${log_name}_${TIMESTAMP}.out"
    echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run the script and capture both stdout and stderr
    if python3 $script_file > ../output/sensitivity_analysis/logs/${log_name}_${TIMESTAMP}.out 2>&1; then
        echo "  ✓ Completed successfully at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        return 0
    else
        echo "  ✗ Failed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Check log file for error details: ../output/sensitivity_analysis/logs/${log_name}_${TIMESTAMP}.out"
        echo ""
        return 1
    fi
}

# Run all scripts sequentially
FAILED_SCRIPTS=()
START_TIME=$(date '+%s')

for i in "${!SCRIPTS[@]}"; do
    current_script=${SCRIPTS[$i]}
    current_name=${SCRIPT_NAMES[$i]}
    current_log=${LOG_NAMES[$i]}
    current_num=$((i + 1))
    
    if run_script "$current_script" "$current_name" "$current_log" "$current_num" "$TOTAL_SCRIPTS"; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    else
        FAILED_SCRIPTS+=("$current_name")
    fi
done

END_TIME=$(date '+%s')
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=== Sensitivity Analysis Batch Job Summary ==="
echo "Total scripts: $TOTAL_SCRIPTS"
echo "Completed successfully: $COMPLETED_COUNT"
echo "Failed: $((TOTAL_SCRIPTS - COMPLETED_COUNT))"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
    echo "✓ All scripts completed successfully!"
    echo ""
    echo "Results can be found in:"
    echo "  ../output/sensitivity_analysis/turbine_ramp_0.5/"
    echo "  ../output/sensitivity_analysis/turbine_ramp_1.0/"
    echo "  ../output/sensitivity_analysis/turbine_ramp_1.5/"
    echo "  ../output/sensitivity_analysis/turbine_ramp_2.0/"
    echo "  ../output/sensitivity_analysis/electrolyzer_ramp_0.5/"
    echo "  ../output/sensitivity_analysis/electrolyzer_ramp_1.0/"
    echo "  ../output/sensitivity_analysis/electrolyzer_ramp_1.5/"
    echo "  ../output/sensitivity_analysis/electrolyzer_ramp_2.0/"
else
    echo "✗ The following scripts failed:"
    for failed_script in "${FAILED_SCRIPTS[@]}"; do
        echo "  - $failed_script"
    done
fi

echo ""
echo "Check log files in ../output/sensitivity_analysis/logs/ for detailed output." 