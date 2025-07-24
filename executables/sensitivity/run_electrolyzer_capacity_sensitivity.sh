#!/bin/bash

# Electrolyzer Capacity Sensitivity Analysis Batch Runner (Bash Version)
# This script runs 4 electrolyzer capacity sensitivity analysis scripts in parallel (2 at a time)
# Testing capacities: 100MW, 150MW, 250MW, 300MW

echo "=== Electrolyzer Capacity Sensitivity Analysis Batch Runner ==="
echo "This script will run 4 sensitivity analysis scripts:"
echo "  - Electrolyzer capacity: 100MW, 150MW, 250MW, 300MW"  
echo "  - Execution: 2 scripts in parallel, then next 2 scripts"
echo "  - Results will be saved in separate directories"
echo ""

# Check if user wants to proceed
read -p "Do you want to start the electrolyzer capacity sensitivity analysis? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user."
    exit 0
fi

# Create output directories
mkdir -p ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs

# Get current timestamp for log filenames (if needed for individual script logs)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting electrolyzer capacity sensitivity analysis..."
echo "Logs will be saved to ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs/"
echo ""

# Define scripts to run
SCRIPTS=(
    "sa_electrolyzer_capacity_100.py"
    "sa_electrolyzer_capacity_150.py" 
    "sa_electrolyzer_capacity_250.py"
    "sa_electrolyzer_capacity_300.py"
)

SCRIPT_NAMES=(
    "Electrolyzer 100MW"
    "Electrolyzer 150MW"
    "Electrolyzer 250MW"
    "Electrolyzer 300MW"
)

LOG_NAMES=(
    "electrolyzer_100MW"
    "electrolyzer_150MW"
    "electrolyzer_250MW"
    "electrolyzer_300MW"
)

TOTAL_SCRIPTS=${#SCRIPTS[@]}
COMPLETED_COUNT=0
FAILED_SCRIPTS=()

# Function to run a single script
run_script() {
    local script_file=$1
    local script_name=$2
    local log_name=$3
    
    echo "  Starting: $script_name"
    echo "    Script: $script_file"
    echo "    Log: ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs/${log_name}.out"
    echo "    Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Run the script and capture both stdout and stderr
    if python3 $script_file > ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs/${log_name}.out 2>&1; then
        echo "    ✓ Completed successfully at: $(date '+%Y-%m-%d %H:%M:%S')"
        return 0
    else
        echo "    ✗ Failed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "    Check log file for error details: ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs/${log_name}.out"
        return 1
    fi
}

# Function to run scripts in parallel
run_batch_parallel() {
    local batch_name=$1
    shift
    local scripts_to_run=("$@")
    
    echo "[$batch_name] Running ${#scripts_to_run[@]} scripts in parallel:"
    
    # Array to store background process PIDs
    local pids=()
    local script_indices=()
    
    # Start all scripts in background
    for i in "${scripts_to_run[@]}"; do
        local script_file=${SCRIPTS[$i]}
        local script_name=${SCRIPT_NAMES[$i]}
        local log_name=${LOG_NAMES[$i]}
        
        # Run script in background and capture PID
        (run_script "$script_file" "$script_name" "$log_name") &
        local pid=$!
        pids+=($pid)
        script_indices+=($i)
        
        echo "  Launched: $script_name (PID: $pid)"
    done
    
    echo "  Waiting for all scripts in this batch to complete..."
    
    # Wait for all background processes and collect results
    local batch_success=0
    local batch_failed=0
    
    for j in "${!pids[@]}"; do
        local pid=${pids[$j]}
        local script_idx=${script_indices[$j]}
        local script_name=${SCRIPT_NAMES[$script_idx]}
        
        if wait $pid; then
            echo "  ✓ $script_name finished successfully"
            ((batch_success++))
            ((COMPLETED_COUNT++))
        else
            echo "  ✗ $script_name failed"
            ((batch_failed++))
            FAILED_SCRIPTS+=("$script_name")
        fi
    done
    
    echo "[$batch_name] Batch completed - Success: $batch_success, Failed: $batch_failed"
    echo ""
}

# Record overall start time
START_TIME=$(date '+%s')

# Run scripts in batches (2 parallel at a time)
echo "=== Starting Batch 1: 100MW and 150MW ==="
run_batch_parallel "Batch 1" 0 1

echo "Waiting 10 seconds before starting next batch..."
sleep 10
echo ""

echo "=== Starting Batch 2: 250MW and 300MW ==="  
run_batch_parallel "Batch 2" 2 3

# Calculate total execution time
END_TIME=$(date '+%s')
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "=== Electrolyzer Capacity Sensitivity Analysis Summary ==="
echo "Total scripts: $TOTAL_SCRIPTS"
echo "Completed successfully: $COMPLETED_COUNT"
echo "Failed: $((TOTAL_SCRIPTS - COMPLETED_COUNT))"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
    echo "✓ All scripts completed successfully!"
    echo ""
    echo "Results can be found in:"
    echo "  ../output/sensitivity_analysis/electrolyzer_capacity_100/"
    echo "  ../output/sensitivity_analysis/electrolyzer_capacity_150/"
    echo "  ../output/sensitivity_analysis/electrolyzer_capacity_250/"
    echo "  ../output/sensitivity_analysis/electrolyzer_capacity_300/"
else
    echo "✗ The following scripts failed:"
    for failed_script in "${FAILED_SCRIPTS[@]}"; do
        echo "  - $failed_script"
    done
fi

echo ""
echo "Batch execution logs saved in: ../output/sensitivity_analysis/electrolyzer_capacity/batch_logs/"
echo ""

# Exit with appropriate code
if [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
    echo "=== All sensitivity analyses completed successfully! ==="
    exit 0
else
    echo "=== Some sensitivity analyses failed. Check logs for details. ==="
    exit 1
fi 