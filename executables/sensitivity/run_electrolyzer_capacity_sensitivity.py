#!/usr/bin/env python3
"""
Electrolyzer Capacity Sensitivity Analysis Batch Runner
This script runs 4 electrolyzer capacity sensitivity analysis scripts in parallel (2 at a time)
Testing capacities: 100MW, 150MW, 250MW, 300MW
"""

import os
import sys
import subprocess
import threading
import queue
import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class SensitivityRunner:
    """Manages parallel execution of sensitivity analysis scripts"""
    
    def __init__(self):
        self.output_dir = "../output/sensitivity_analysis/electrolyzer_capacity"
        self.log_dir = os.path.join(self.output_dir, "batch_logs")
        self.setup_logging()
        
        # Define scripts and their configurations
        self.scripts = [
            {
                'name': 'Electrolyzer 100MW',
                'script': 'sa_electrolyzer_capacity_100.py',
                'log_name': 'electrolyzer_100MW',
                'description': 'Electrolyzer capacity upper bound = 100MW'
            },
            {
                'name': 'Electrolyzer 150MW',
                'script': 'sa_electrolyzer_capacity_150.py',
                'log_name': 'electrolyzer_150MW',
                'description': 'Electrolyzer capacity upper bound = 150MW'
            },
            {
                'name': 'Electrolyzer 250MW',
                'script': 'sa_electrolyzer_capacity_250.py',
                'log_name': 'electrolyzer_250MW',
                'description': 'Electrolyzer capacity upper bound = 250MW'
            },
            {
                'name': 'Electrolyzer 300MW',
                'script': 'sa_electrolyzer_capacity_300.py',
                'log_name': 'electrolyzer_300MW',
                'description': 'Electrolyzer capacity upper bound = 300MW'
            }
        ]
        
        # Group scripts in pairs for parallel execution
        self.batches = [
            [self.scripts[0], self.scripts[1]],  # 100MW and 150MW
            [self.scripts[2], self.scripts[3]]   # 250MW and 300MW
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Use fixed log filename as per user preference
        log_filename = os.path.join(self.log_dir, "electrolyzer_capacity_batch.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Electrolyzer Capacity Sensitivity Analysis Batch Started ===")
        
    def run_single_script(self, script_config):
        """Run a single sensitivity analysis script"""
        script_name = script_config['name']
        script_file = script_config['script']
        log_name = script_config['log_name']
        description = script_config['description']
        
        self.logger.info(f"Starting {script_name}: {description}")
        start_time = time.time()
        
        try:
            # Prepare log file for this script
            script_log_file = os.path.join(self.log_dir, f"{log_name}.out")
            
            # Run the script
            with open(script_log_file, 'w') as log_file:
                process = subprocess.run(
                    [sys.executable, script_file],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    text=True
                )
                
            end_time = time.time()
            duration = end_time - start_time
            
            if process.returncode == 0:
                self.logger.info(f"✓ {script_name} completed successfully in {duration:.1f}s")
                return {
                    'script': script_name,
                    'success': True,
                    'duration': duration,
                    'log_file': script_log_file
                }
            else:
                self.logger.error(f"✗ {script_name} failed after {duration:.1f}s (exit code: {process.returncode})")
                return {
                    'script': script_name,
                    'success': False,
                    'duration': duration,
                    'log_file': script_log_file,
                    'exit_code': process.returncode
                }
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.error(f"✗ {script_name} failed with exception after {duration:.1f}s: {e}")
            return {
                'script': script_name,
                'success': False,
                'duration': duration,
                'error': str(e)
            }
            
    def run_batch_parallel(self, batch, batch_num):
        """Run a batch of scripts in parallel"""
        self.logger.info(f"Starting Batch {batch_num} with {len(batch)} scripts in parallel:")
        for script in batch:
            self.logger.info(f"  - {script['name']}")
            
        batch_start_time = time.time()
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # Submit all scripts in the batch
            future_to_script = {
                executor.submit(self.run_single_script, script): script 
                for script in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_script):
                script = future_to_script[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Exception in {script['name']}: {e}")
                    results.append({
                        'script': script['name'],
                        'success': False,
                        'error': str(e)
                    })
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        # Log batch summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        self.logger.info(f"Batch {batch_num} completed in {batch_duration:.1f}s - Success: {successful}, Failed: {failed}")
        
        return results
        
    def run_all_batches(self):
        """Run all batches sequentially (each batch runs its scripts in parallel)"""
        self.logger.info(f"Starting electrolyzer capacity sensitivity analysis")
        self.logger.info(f"Total scripts: {len(self.scripts)}")
        self.logger.info(f"Execution plan: {len(self.batches)} batches, 2 scripts per batch (parallel)")
        self.logger.info("Results will be saved in separate directories for each capacity")
        
        overall_start_time = time.time()
        all_results = []
        
        for batch_num, batch in enumerate(self.batches, 1):
            self.logger.info(f"\n{'='*60}")
            batch_results = self.run_batch_parallel(batch, batch_num)
            all_results.extend(batch_results)
            
            # Wait a bit between batches to avoid resource conflicts
            if batch_num < len(self.batches):
                self.logger.info("Waiting 10 seconds before starting next batch...")
                time.sleep(10)
        
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        return all_results, overall_duration
        
    def print_summary(self, results, total_duration):
        """Print final summary of all runs"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("=== ELECTROLYZER CAPACITY SENSITIVITY ANALYSIS SUMMARY ===")
        
        successful_runs = [r for r in results if r['success']]
        failed_runs = [r for r in results if not r['success']]
        
        self.logger.info(f"Total scripts: {len(results)}")
        self.logger.info(f"Successful: {len(successful_runs)}")
        self.logger.info(f"Failed: {len(failed_runs)}")
        self.logger.info(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        
        if successful_runs:
            self.logger.info("\n✓ Successful runs:")
            for result in successful_runs:
                self.logger.info(f"  - {result['script']}: {result['duration']:.1f}s")
                
        if failed_runs:
            self.logger.info("\n✗ Failed runs:")
            for result in failed_runs:
                duration = result.get('duration', 0)
                error_info = result.get('error', f"Exit code: {result.get('exit_code', 'unknown')}")
                self.logger.info(f"  - {result['script']}: {duration:.1f}s - {error_info}")
                
        self.logger.info("\nResults locations:")
        self.logger.info("  - ../output/sensitivity_analysis/electrolyzer_capacity_100/")
        self.logger.info("  - ../output/sensitivity_analysis/electrolyzer_capacity_150/")
        self.logger.info("  - ../output/sensitivity_analysis/electrolyzer_capacity_250/")
        self.logger.info("  - ../output/sensitivity_analysis/electrolyzer_capacity_300/")
        
        self.logger.info(f"\nBatch execution logs saved in: {self.log_dir}")
        
        return len(failed_runs) == 0


def main():
    """Main execution function"""
    try:
        # Confirm execution with user
        print("=== Electrolyzer Capacity Sensitivity Analysis Batch Runner ===")
        print("This script will run 4 sensitivity analysis scripts:")
        print("  - Electrolyzer capacity: 100MW, 150MW, 250MW, 300MW")
        print("  - Execution: 2 scripts in parallel, then next 2 scripts")
        print("  - Results will be saved in separate directories")
        print()
        
        # Create runner and execute
        runner = SensitivityRunner()
        
        # Run all batches
        results, total_duration = runner.run_all_batches()
        
        # Print summary
        success = runner.print_summary(results, total_duration)
        
        if success:
            runner.logger.info("=== All sensitivity analyses completed successfully! ===")
            return 0
        else:
            runner.logger.error("=== Some sensitivity analyses failed. Check logs for details. ===")
            return 1
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 