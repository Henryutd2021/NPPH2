#!/bin/bash
# TEA CS1 Sensitivity Analysis Parallel Execution Script
# This script runs TEA sensitivity analysis for all parameter values in parallel

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/tea_cs1_sensitivity.py"
OUTPUT_DIR="$PROJECT_ROOT/output/tea/cs1_sensitivity"
LOG_DIR="$PROJECT_ROOT/output/logs/cs1/sensitivity"

# Sensitivity analysis parameters
PARAMETER_VALUES=(170000 200000 260000 290000 320000)
MAX_PARALLEL_JOBS=5  # Maximum number of parallel jobs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Python script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to create output directories
create_directories() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    for param_value in "${PARAMETER_VALUES[@]}"; do
        mkdir -p "$OUTPUT_DIR/fixed_costs_$param_value"
        mkdir -p "$LOG_DIR"
    done
}

# Function to run sensitivity analysis for a single parameter value
run_single_sensitivity() {
    local param_value=$1
    local log_file="$LOG_DIR/parallel_execution_${param_value}.log"
    
    print_info "Starting sensitivity analysis for parameter value: $param_value"
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Run Python script with parameter value
    if python3 "$PYTHON_SCRIPT" --parameter-value "$param_value" > "$log_file" 2>&1; then
        print_success "Completed sensitivity analysis for parameter value: $param_value"
        echo "SUCCESS:$param_value" >> "$LOG_DIR/parallel_results.tmp"
    else
        print_error "Failed sensitivity analysis for parameter value: $param_value"
        echo "FAILED:$param_value" >> "$LOG_DIR/parallel_results.tmp"
    fi
}

# Function to monitor parallel jobs
monitor_jobs() {
    local total_jobs=$1
    local completed=0
    local failed=0
    
    print_info "Monitoring $total_jobs parallel jobs..."
    
    # Wait for all background jobs to complete
    while [ $completed -lt $total_jobs ]; do
        sleep 5
        
        # Count completed jobs
        if [ -f "$LOG_DIR/parallel_results.tmp" ]; then
            local new_completed=$(wc -l < "$LOG_DIR/parallel_results.tmp")
            if [ $new_completed -gt $completed ]; then
                completed=$new_completed
                print_info "Progress: $completed/$total_jobs jobs completed"
            fi
        fi
        
        # Check if any jobs are still running
        if [ $(jobs -r | wc -l) -eq 0 ] && [ $completed -lt $total_jobs ]; then
            print_warning "No jobs running but not all completed. Checking results..."
            break
        fi
    done
    
    # Wait for all background jobs to finish
    wait
    
    # Count successes and failures
    if [ -f "$LOG_DIR/parallel_results.tmp" ]; then
        local success_count=$(grep -c "SUCCESS:" "$LOG_DIR/parallel_results.tmp" || true)
        local failed_count=$(grep -c "FAILED:" "$LOG_DIR/parallel_results.tmp" || true)
        
        print_info "Final results: $success_count successful, $failed_count failed"
        return $failed_count
    else
        print_error "No results file found"
        return $total_jobs
    fi
}

# Function to generate summary report
generate_summary_report() {
    local summary_file="$OUTPUT_DIR/sensitivity_analysis_summary.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    print_info "Generating summary report: $summary_file"
    
    cat > "$summary_file" << EOF
TEA CS1 Sensitivity Analysis Summary Report
Generated: $timestamp

Parameter: total_fixed_costs_per_mw_year
Parameter Values Tested: ${PARAMETER_VALUES[*]}

Results by Parameter Value:
EOF
    
    if [ -f "$LOG_DIR/parallel_results.tmp" ]; then
        # Process results
        for param_value in "${PARAMETER_VALUES[@]}"; do
            if grep -q "SUCCESS:$param_value" "$LOG_DIR/parallel_results.tmp"; then
                echo "  $param_value: SUCCESS" >> "$summary_file"
                
                # Count output files if directory exists
                local output_dir="$OUTPUT_DIR/fixed_costs_$param_value"
                if [ -d "$output_dir" ]; then
                    local file_count=$(find "$output_dir" -name "*.txt" | wc -l)
                    echo "    - Generated $file_count report files" >> "$summary_file"
                fi
            else
                echo "  $param_value: FAILED" >> "$summary_file"
            fi
        done
        
        # Add summary statistics
        local total_jobs=${#PARAMETER_VALUES[@]}
        local success_count=$(grep -c "SUCCESS:" "$LOG_DIR/parallel_results.tmp" || true)
        local failed_count=$(grep -c "FAILED:" "$LOG_DIR/parallel_results.tmp" || true)
        
        cat >> "$summary_file" << EOF

Summary Statistics:
  Total Parameter Values: $total_jobs
  Successful Analyses: $success_count
  Failed Analyses: $failed_count
  Success Rate: $(echo "scale=2; $success_count * 100 / $total_jobs" | bc -l)%

Output Directories:
EOF
        
        # List output directories
        for param_value in "${PARAMETER_VALUES[@]}"; do
            if [ -d "$OUTPUT_DIR/fixed_costs_$param_value" ]; then
                echo "  fixed_costs_$param_value: $OUTPUT_DIR/fixed_costs_$param_value" >> "$summary_file"
            fi
        done
        
        cat >> "$summary_file" << EOF

Log Files:
  Main execution log: $LOG_DIR/parallel_execution_*.log
  Individual parameter logs: $LOG_DIR/tea_sensitivity_*.log

Notes:
  - All plotting functionality was disabled for sensitivity analysis
  - Results are stored in parameter-specific subdirectories
  - Each parameter analysis was run independently in parallel
EOF
    fi
    
    print_success "Summary report generated: $summary_file"
}

# Function to cleanup temporary files
cleanup() {
    print_info "Cleaning up temporary files..."
    rm -f "$LOG_DIR/parallel_results.tmp"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -j, --jobs N        Maximum number of parallel jobs (default: $MAX_PARALLEL_JOBS)
    -v, --verbose       Enable verbose output
    --dry-run           Show what would be executed without running

Examples:
    $0                  # Run with default settings
    $0 -j 3             # Run with maximum 3 parallel jobs
    $0 --verbose        # Run with verbose output
    $0 --dry-run        # Show execution plan without running

Parameter Values:
    ${PARAMETER_VALUES[*]}
EOF
}

# Parse command line arguments
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -j|--jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Starting TEA CS1 Sensitivity Analysis (Parallel Execution)"
    print_info "Project root: $PROJECT_ROOT"
    print_info "Python script: $PYTHON_SCRIPT"
    print_info "Output directory: $OUTPUT_DIR"
    print_info "Log directory: $LOG_DIR"
    print_info "Maximum parallel jobs: $MAX_PARALLEL_JOBS"
    print_info "Parameter values: ${PARAMETER_VALUES[*]}"
    
    # Check prerequisites
    check_python_script
    
    # Create directories
    create_directories
    
    # Clean up any previous temporary files
    cleanup
    
    if [ "$DRY_RUN" = true ]; then
        print_info "DRY RUN - Would execute the following:"
        for param_value in "${PARAMETER_VALUES[@]}"; do
            echo "  python3 $PYTHON_SCRIPT --parameter-value $param_value"
        done
        exit 0
    fi
    
    # Record start time
    local start_time=$(date +%s)
    
    # Launch parallel jobs
    print_info "Launching ${#PARAMETER_VALUES[@]} parallel sensitivity analyses..."
    
    local job_count=0
    for param_value in "${PARAMETER_VALUES[@]}"; do
        # Wait if we've reached the maximum number of parallel jobs
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_JOBS ]; do
            sleep 1
        done
        
        # Launch job in background
        run_single_sensitivity "$param_value" &
        job_count=$((job_count + 1))
        
        if [ "$VERBOSE" = true ]; then
            print_info "Launched job $job_count for parameter value: $param_value"
        fi
    done
    
    # Monitor job completion
    local failed_jobs=$(monitor_jobs ${#PARAMETER_VALUES[@]})
    
    # Calculate execution time
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    
    # Generate summary report
    generate_summary_report
    
    # Cleanup
    cleanup
    
    # Final status
    print_info "Execution completed in ${execution_time} seconds"
    
    if [ $failed_jobs -eq 0 ]; then
        print_success "All sensitivity analyses completed successfully!"
        print_success "Results are available in: $OUTPUT_DIR"
        print_success "Summary report: $OUTPUT_DIR/sensitivity_analysis_summary.txt"
        exit 0
    else
        print_error "$failed_jobs out of ${#PARAMETER_VALUES[@]} analyses failed"
        print_error "Check individual log files in: $LOG_DIR"
        exit 1
    fi
}

# Set up signal handling for cleanup
trap cleanup EXIT

# Run main function
main "$@" 