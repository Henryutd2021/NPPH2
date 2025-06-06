"""
Progress indicators for long-running processes in the optimization framework.
"""

import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

# Import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SolverProgressIndicator:
    """
    A progress indicator for optimization solving process that shows real progress
    based on MIP gap information from the solver output.
    """

    def __init__(self, description="Solving optimization model", target_gap=0.0005):
        self.description = description
        self.target_gap = target_gap  # Target MIP gap (e.g., 0.0005 = 0.05%)
        self.running = False
        self.thread = None
        self.start_time = None
        self.current_gap = None
        self.log_file = None
        self.completion_message = None

    def _parse_gurobi_line(self, line):
        """Parse Gurobi output line to extract gap information."""
        try:
            line = line.strip()
            if not line:
                return False

            # Pattern 1: Heuristic solution lines: "H  150     0                    2.234056e+08 2.234056e+08  0.00%     0s"
            if line.startswith('H') and '%' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if '%' in part:
                        gap_str = part.replace('%', '')
                        try:
                            self.current_gap = float(gap_str) / 100.0
                            return True
                        except ValueError:
                            pass

            # Pattern 2: Regular iteration lines with gap: "   150     0 2.234056e+08 2.234056e+08  0.00%     0s"
            elif line[0].isdigit() and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part and part != '%':
                        gap_str = part.replace('%', '')
                        try:
                            self.current_gap = float(gap_str) / 100.0
                            return True
                        except ValueError:
                            pass

            # Pattern 3: Final gap information: "Best objective 1.234567e+08, best bound 1.234567e+08, gap 0.00%"
            elif 'gap' in line.lower() and '%' in line:
                # Look for "gap X.XX%" pattern
                match = re.search(r'gap\s+(\d+\.?\d*)%', line, re.IGNORECASE)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True

            # Pattern 4: Presolve or other status lines that might contain gap
            elif 'gap:' in line.lower() and '%' in line:
                match = re.search(r'gap:\s*(\d+\.?\d*)%', line, re.IGNORECASE)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True

        except (ValueError, IndexError, AttributeError):
            pass
        return False

    def _parse_cplex_line(self, line):
        """Parse CPLEX output line to extract gap information."""
        try:
            if 'gap' in line.lower() and '%' in line:
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    self.current_gap = float(match.group(1)) / 100.0
                    return True
        except (ValueError, IndexError):
            pass
        return False

    def _monitor_solver_output(self, solver_name):
        """Monitor solver output file for gap information."""
        # Wait for log file to be created (up to 30 seconds)
        wait_time = 0
        max_wait = 30
        while not os.path.exists(self.log_file) and wait_time < max_wait and self.running:
            time.sleep(0.5)
            wait_time += 0.5

        if not os.path.exists(self.log_file):
            return

        try:
            # Start reading from the end of the file to avoid old content
            last_position = 0
            # Get initial file size to start reading from current end
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    f.seek(0, 2)  # Go to end
                    last_position = f.tell()

            gap_found_count = 0
            while self.running:
                try:
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        if new_content:
                            # Process each line in the new content
                            lines = new_content.split('\n')
                            # Exclude last potentially incomplete line
                            for line in lines[:-1]:
                                if line.strip():
                                    if solver_name.lower() == 'gurobi':
                                        if self._parse_gurobi_line(line):
                                            gap_found_count += 1
                                    elif solver_name.lower() == 'cplex':
                                        if self._parse_cplex_line(line):
                                            gap_found_count += 1
                            last_position = f.tell()
                        else:
                            time.sleep(0.2)  # Wait for more content
                except IOError:
                    # File might be locked by solver, wait and retry
                    time.sleep(0.5)
        except Exception as e:
            pass  # Silently handle errors in gap monitoring

    def _calculate_progress(self):
        """Calculate progress based on current gap and target gap."""
        if self.current_gap is None:
            return 0.0

        if self.current_gap <= self.target_gap:
            return 100.0

        # Logarithmic progress calculation
        initial_gap = 1.0

        if self.current_gap >= initial_gap:
            return 0.0

        log_current = math.log10(max(self.current_gap, 1e-6))
        log_target = math.log10(max(self.target_gap, 1e-6))
        log_initial = math.log10(initial_gap)

        progress = (log_initial - log_current) / \
            (log_initial - log_target) * 100
        return min(max(progress, 0.0), 100.0)

    def _animate(self):
        """Internal method to run the animation in a separate thread."""
        if TQDM_AVAILABLE:
            with tqdm(desc=self.description, total=100, unit="%",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| Gap: {postfix} | {elapsed}") as pbar:
                while self.running:
                    progress = self._calculate_progress()
                    gap_text = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                    pbar.set_postfix_str(gap_text)
                    pbar.n = progress
                    pbar.refresh()
                    time.sleep(0.5)

                # Set final state when animation stops
                final_progress = self._calculate_progress()
                final_gap = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                pbar.n = final_progress
                pbar.set_postfix_str(final_gap)
                pbar.refresh()

                # Store completion message for later display
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    self.completion_message = f"{self.description} completed in {elapsed:.1f}s (Final gap: {final_gap})"
        else:
            spinners = ['|', '/', '-', '\\']
            i = 0
            while self.running:
                elapsed = time.time() - self.start_time if self.start_time else 0
                progress = self._calculate_progress()
                gap_text = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                print(f"\r{self.description}... {spinners[i % len(spinners)]} Progress: {progress:.1f}% | Gap: {gap_text} | ({elapsed:.1f}s)",
                      end="", flush=True)
                i += 1
                time.sleep(0.5)
            print()

    def start(self, solver_name="gurobi", log_file=None):
        """Start the progress indicator."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.log_file = log_file
            self.current_gap = None  # Reset gap for new run

            if log_file and solver_name:
                monitor_thread = threading.Thread(
                    target=self._monitor_solver_output,
                    args=(solver_name,),
                    daemon=True,
                    name=f"GapMonitor-{solver_name}-{int(time.time())}"
                )
                monitor_thread.start()

            self.thread = threading.Thread(
                target=self._animate,
                daemon=True,
                name=f"ProgressAnim-{int(time.time())}"
            )
            self.thread.start()

    def stop(self):
        """Stop the progress indicator."""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)

            # Show completion message after progress bar is properly closed
            if TQDM_AVAILABLE and hasattr(self, 'completion_message') and self.completion_message:
                print(self.completion_message)
            elif not TQDM_AVAILABLE and self.start_time:
                elapsed = time.time() - self.start_time
                final_gap = f"{self.current_gap*100:.3f}%" if self.current_gap is not None else "N/A"
                print(
                    f"\r{self.description} completed in {elapsed:.1f}s (Final gap: {final_gap})" + " " * 20)


class TEAProgressIndicator:
    """Enhanced progress indicator with logging integration for TEA analysis"""

    def __init__(self, description="Running TEA analysis", logger=None):
        self.description = description
        self.logger = logger
        self.running = False
        self.thread = None
        self.start_time = None

    def _animate(self):
        if TQDM_AVAILABLE:
            with tqdm(desc=self.description, unit=" run", bar_format="{desc}: {elapsed} | {rate_fmt}") as pbar:
                while self.running:
                    pbar.update(0)
                    time.sleep(0.1)
        else:
            spinners = ['|', '/', '-', '\\']
            i = 0
            while self.running:
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(
                    f"\r{self.description}... {spinners[i % len(spinners)]} ({elapsed:.1f}s)", end="", flush=True)
                i += 1
                time.sleep(0.25)
            if not TQDM_AVAILABLE:
                print()

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()
            if self.logger:
                self.logger.log_phase_start(self.description)

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=0.5)
            duration = time.time() - self.start_time if self.start_time else 0
            if self.logger:
                self.logger.log_phase_complete(self.description, duration)
            elif not TQDM_AVAILABLE:
                print(
                    f"\r{self.description} completed in {duration:.1f}s." + " " * 20)
