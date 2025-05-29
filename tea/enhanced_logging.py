"""
Enhanced Logging System for CS1 TEA Analysis
Provides reactor-specific logging with data issue tracking and standardized naming
"""

import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


@dataclass
class DataIssueRecord:
    """Record for tracking data-related issues during analysis"""
    timestamp: str
    component: str
    parameter: str
    issue_type: str  # 'missing', 'invalid', 'fallback'
    description: str
    fallback_value: Optional[str] = None
    impact: str = "low"  # low, medium, high, critical
    phase: Optional[str] = None


class EnhancedReactorLogger:
    """Enhanced logger for reactor-specific analysis with data issue tracking"""

    def __init__(self, reactor_name: str, unit_id: str, iso_region: str, base_log_dir: Path):
        self.reactor_name = self._standardize_reactor_name(reactor_name)
        self.unit_id = unit_id
        self.iso_region = iso_region
        self.base_log_dir = Path(base_log_dir)
        self.session_start = datetime.now()

        # Create log directory if needed
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Generate standardized filenames
        self.log_prefix = f"{self.reactor_name}_Unit{self.unit_id}_{self.iso_region}"

        # Setup log files
        self.main_log_file = self.base_log_dir / f"{self.log_prefix}_main.log"
        self.data_issues_file = self.base_log_dir / \
            f"{self.log_prefix}_data_issues.log"
        self.session_file = self.base_log_dir / \
            f"{self.log_prefix}_session.json"

        # Initialize data tracking
        self.data_issues: List[DataIssueRecord] = []
        self.calculation_results: Dict[str, Any] = {}
        self.phases: List[Dict[str, Any]] = []
        self.current_phase: Optional[str] = None
        self.lock = threading.Lock()

        # Setup main logger
        self.logger = self._setup_main_logger()

        self.info(
            f"🚀 Enhanced logging initialized for {self.reactor_name} Unit {self.unit_id} ({self.iso_region})")

    def _standardize_reactor_name(self, reactor_name: str) -> str:
        """Standardize reactor names for consistent file naming"""
        standardization_map = {
            "Calvert Cliffs Nuclear Power Plant": "CalvertCliffs",
            "Davis Besse": "DavisBesse",
            "Beaver Valley Power Station": "BeaverValley",
            "Arkansas Nuclear One": "ArkansasNuclearOne",
            "Wolf Creek Generating Station": "WolfCreek",
            "Summer": "Summer",
            "Vogtle Electric Generating Plant": "Vogtle",
            # Add more mappings as needed
        }

        # Direct mapping if available
        if reactor_name in standardization_map:
            return standardization_map[reactor_name]

        # Fallback: remove common words and spaces
        cleaned = reactor_name.replace(
            "Nuclear Power Plant", "").replace("Power Station", "")
        cleaned = cleaned.replace("Generating Station", "").replace(
            "Electric Generating Plant", "")
        cleaned = "".join(word.capitalize() for word in cleaned.split())

        return cleaned

    def _setup_main_logger(self) -> logging.Logger:
        """Setup the main logger for this reactor"""
        logger_name = f"reactor_{self.reactor_name}_{self.unit_id}_{self.iso_region}"
        logger = logging.getLogger(logger_name)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Set level
        logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler(
            self.main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_missing_data(self, component: str, parameter: str,
                         fallback_value: Optional[str] = None, impact: str = "medium"):
        """Log missing data issue"""
        with self.lock:
            issue = DataIssueRecord(
                timestamp=datetime.now().isoformat(),
                component=component,
                parameter=parameter,
                issue_type="missing",
                description=f"Missing data for {parameter} in {component}",
                fallback_value=fallback_value,
                impact=impact,
                phase=self.current_phase
            )
            self.data_issues.append(issue)

            # Log to data issues file
            self._log_data_issue(issue)

            # Log to main logger
            self.warning(f"⚠️  Missing data: {component}.{parameter}" +
                         (f" (using fallback: {fallback_value})" if fallback_value else ""))

    def log_invalid_data(self, component: str, parameter: str, invalid_value: str,
                         corrected_value: Optional[str] = None, impact: str = "medium"):
        """Log invalid data issue"""
        with self.lock:
            issue = DataIssueRecord(
                timestamp=datetime.now().isoformat(),
                component=component,
                parameter=parameter,
                issue_type="invalid",
                description=f"Invalid data '{invalid_value}' for {parameter} in {component}",
                fallback_value=corrected_value,
                impact=impact,
                phase=self.current_phase
            )
            self.data_issues.append(issue)

            self._log_data_issue(issue)
            self.warning(f"⚠️  Invalid data: {component}.{parameter} = '{invalid_value}'" +
                         (f" (corrected to: {corrected_value})" if corrected_value else ""))

    def log_fallback_usage(self, component: str, parameter: str, fallback_value: str,
                           reason: str, impact: str = "low"):
        """Log fallback data usage"""
        with self.lock:
            issue = DataIssueRecord(
                timestamp=datetime.now().isoformat(),
                component=component,
                parameter=parameter,
                issue_type="fallback",
                description=f"Using fallback value for {parameter}: {reason}",
                fallback_value=fallback_value,
                impact=impact,
                phase=self.current_phase
            )
            self.data_issues.append(issue)

            self._log_data_issue(issue)
            self.info(
                f"ℹ️  Fallback usage: {component}.{parameter} = {fallback_value} ({reason})")

    def _log_data_issue(self, issue: DataIssueRecord):
        """Write data issue to the dedicated data issues log file"""
        try:
            with open(self.data_issues_file, 'a', encoding='utf-8') as f:
                f.write(f"{issue.timestamp} | {issue.issue_type.upper()} | "
                        f"{issue.component}.{issue.parameter} | {issue.description}")
                if issue.fallback_value:
                    f.write(f" | Fallback: {issue.fallback_value}")
                f.write(f" | Impact: {issue.impact}")
                if issue.phase:
                    f.write(f" | Phase: {issue.phase}")
                f.write("\n")
        except Exception as e:
            self.error(f"Failed to write data issue to log: {e}")

    def log_calculation_result(self, parameter: str, value: Any, unit: str = ""):
        """Log calculation result"""
        with self.lock:
            self.calculation_results[parameter] = {
                "value": value,
                "unit": unit,
                "timestamp": datetime.now().isoformat(),
                "phase": self.current_phase
            }

        self.info(f"📊 {parameter}: {value} {unit}")

    def log_phase_start(self, phase_name: str, description: str = ""):
        """Log the start of an analysis phase"""
        with self.lock:
            self.current_phase = phase_name
            phase_record = {
                "name": phase_name,
                "description": description,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration": None,
                "status": "running"
            }
            self.phases.append(phase_record)

        self.info(f"🚀 Phase started: {phase_name}" +
                  (f" - {description}" if description else ""))

    def log_phase_complete(self, phase_name: str, duration: Optional[float] = None,
                           results_summary: str = ""):
        """Log the completion of an analysis phase"""
        with self.lock:
            # Find the most recent phase with this name
            for phase in reversed(self.phases):
                if phase["name"] == phase_name and phase["status"] == "running":
                    phase["end_time"] = datetime.now().isoformat()
                    phase["status"] = "completed"
                    if duration is not None:
                        phase["duration"] = duration
                    if results_summary:
                        phase["results_summary"] = results_summary
                    break

        self.current_phase = None
        duration_str = f" ({duration:.1f}s)" if duration else ""
        summary_str = f" - {results_summary}" if results_summary else ""
        self.info(
            f"✅ Phase completed: {phase_name}{duration_str}{summary_str}")

    def create_session_summary(self) -> Dict[str, Any]:
        """Create a comprehensive session summary"""
        session_end = datetime.now()
        session_duration = (session_end - self.session_start).total_seconds()

        # Categorize data issues by type and impact
        issues_by_type = {}
        issues_by_impact = {}

        for issue in self.data_issues:
            issues_by_type.setdefault(
                issue.issue_type, []).append(asdict(issue))
            issues_by_impact.setdefault(issue.impact, []).append(asdict(issue))

        summary = {
            "reactor_info": {
                "name": self.reactor_name,
                "unit_id": self.unit_id,
                "iso_region": self.iso_region,
                "standardized_name": self.log_prefix
            },
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "end_time": session_end.isoformat(),
                "duration_seconds": session_duration,
                "log_files": {
                    "main_log": str(self.main_log_file),
                    "data_issues": str(self.data_issues_file),
                    "session_summary": str(self.session_file)
                }
            },
            "analysis_phases": self.phases,
            "calculation_results": self.calculation_results,
            "data_issues": {
                "total_count": len(self.data_issues),
                "by_type": issues_by_type,
                "by_impact": issues_by_impact,
                "summary": {
                    "missing_data": len([i for i in self.data_issues if i.issue_type == "missing"]),
                    "invalid_data": len([i for i in self.data_issues if i.issue_type == "invalid"]),
                    "fallback_usage": len([i for i in self.data_issues if i.issue_type == "fallback"]),
                    "critical_issues": len([i for i in self.data_issues if i.impact == "critical"]),
                    "high_impact": len([i for i in self.data_issues if i.impact == "high"])
                }
            }
        }

        return summary

    def save_session_summary(self):
        """Save session summary to JSON file"""
        try:
            summary = self.create_session_summary()
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.info(f"💾 Session summary saved to {self.session_file}")
        except Exception as e:
            self.error(f"Failed to save session summary: {e}")


@contextmanager
def ReactorLogSession(reactor_name: str, unit_id: str, iso_region: str,
                      base_log_dir: Path = None):
    """Context manager for reactor logging sessions"""
    if base_log_dir is None:
        base_log_dir = Path(__file__).parent.parent.parent / "logs" / "cs1"

    logger = create_reactor_logger(
        reactor_name, unit_id, iso_region, base_log_dir)

    try:
        yield logger
    finally:
        logger.save_session_summary()
        logger.info(f"🏁 Logging session completed for {logger.reactor_name}")


def create_reactor_logger(reactor_name: str, unit_id: str, iso_region: str,
                          base_log_dir: Path = None) -> EnhancedReactorLogger:
    """Factory function to create reactor logger"""
    if base_log_dir is None:
        base_log_dir = Path(__file__).parent.parent.parent / "logs" / "cs1"

    return EnhancedReactorLogger(reactor_name, unit_id, iso_region, base_log_dir)
