"""
Enhanced logging system for TEA analysis with comprehensive data tracking
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass, asdict


@dataclass
class DataIssueRecord:
    """Record for tracking data issues and fallback usage"""
    timestamp: str
    issue_type: str  # 'missing_data', 'invalid_data', 'fallback_used'
    affected_component: str  # e.g., 'battery', 'electrolyzer', 'h2_storage'
    description: str
    fallback_action: Optional[str] = None
    original_value: Optional[Any] = None
    fallback_value: Optional[Any] = None
    impact_severity: str = 'medium'  # 'low', 'medium', 'high', 'critical'


@dataclass
class ReactorSessionData:
    """Complete logging session for a reactor analysis"""
    reactor_name: str
    generator_id: str
    iso_region: str
    remaining_years: str
    session_start: str
    session_end: Optional[str] = None
    status: str = 'running'  # 'running', 'completed', 'failed'
    data_issues: List[DataIssueRecord] = None

    def __post_init__(self):
        if self.data_issues is None:
            self.data_issues = []


class EnhancedReactorLogger:
    """Enhanced logger with reactor-specific tracking and data issue monitoring"""

    def __init__(self, reactor_name: str, generator_id: str, iso_region: str,
                 remaining_years: str, log_base_dir: Path = None):
        """Initialize enhanced reactor logger"""
        self.reactor_name = reactor_name
        self.generator_id = generator_id
        self.iso_region = iso_region
        self.remaining_years = remaining_years

        # Standardized naming convention
        self.reactor_id = self._standardize_reactor_name(
            reactor_name, generator_id)

        # Base directory setup
        if log_base_dir is None:
            log_base_dir = Path(__file__).parent.parent.parent / 'logs' / 'cs1'
        self.log_dir = Path(log_base_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log files with standardized naming
        self.main_log_file = self.log_dir / \
            f"{self.reactor_id}_{iso_region}_main.log"
        self.data_log_file = self.log_dir / \
            f"{self.reactor_id}_{iso_region}_data_issues.log"
        self.summary_file = self.log_dir / \
            f"{self.reactor_id}_{iso_region}_session.json"

        # Initialize loggers
        self._setup_loggers()

        # Session tracking
        self.session = ReactorSessionData(
            reactor_name=reactor_name,
            generator_id=generator_id,
            iso_region=iso_region,
            remaining_years=remaining_years,
            session_start=datetime.now().isoformat(),
            data_issues=[]
        )

        # Thread safety
        self._lock = threading.Lock()

        # Start session
        self._start_session()

    def _standardize_reactor_name(self, reactor_name: str, generator_id: str) -> str:
        """Standardize reactor naming convention"""
        # Remove common suffixes and normalize
        name_clean = reactor_name.replace(" Nuclear Power Plant", "")
        name_clean = name_clean.replace(" Generating Station", "")
        name_clean = name_clean.replace(" Generation Station", "")
        name_clean = name_clean.replace("PSEG ", "")
        name_clean = name_clean.replace("TalenEnergy ", "")

        # Remove special characters and spaces
        name_clean = "".join(
            c for c in name_clean if c.isalnum() or c in ['-', '_'])
        name_clean = name_clean.replace(' ', '_')

        return f"{name_clean}_Unit{generator_id}"

    def _setup_loggers(self):
        """Setup main and data issue loggers"""
        # Main logger
        self.main_logger = logging.getLogger(f"reactor_{self.reactor_id}")
        self.main_logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.main_logger.handlers.clear()

        # Main log handler
        main_handler = logging.FileHandler(self.main_log_file)
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)

        # Console handler for critical messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.main_logger.addHandler(console_handler)

        # Data issues logger
        self.data_logger = logging.getLogger(f"data_issues_{self.reactor_id}")
        self.data_logger.setLevel(logging.INFO)
        self.data_logger.handlers.clear()

        data_handler = logging.FileHandler(self.data_log_file)
        data_formatter = logging.Formatter(
            '%(asctime)s - DATA_ISSUE - %(message)s'
        )
        data_handler.setFormatter(data_formatter)
        self.data_logger.addHandler(data_handler)

    def _start_session(self):
        """Log session start"""
        self.main_logger.info("="*80)
        self.main_logger.info(
            f"🚀 Starting TEA analysis session for {self.reactor_name}")
        self.main_logger.info(f"📊 Reactor ID: {self.reactor_id}")
        self.main_logger.info(f"🔌 Generator: {self.generator_id}")
        self.main_logger.info(f"🌍 ISO Region: {self.iso_region}")
        self.main_logger.info(f"⏰ Remaining Years: {self.remaining_years}")
        self.main_logger.info(f"📁 Log Directory: {self.log_dir}")
        self.main_logger.info("="*80)

    def log_data_issue(self, issue_type: str, affected_component: str,
                       description: str, fallback_action: str = None,
                       original_value: Any = None, fallback_value: Any = None,
                       impact_severity: str = 'medium'):
        """Log data issue with detailed tracking"""
        with self._lock:
            issue = DataIssueRecord(
                timestamp=datetime.now().isoformat(),
                issue_type=issue_type,
                affected_component=affected_component,
                description=description,
                fallback_action=fallback_action,
                original_value=original_value,
                fallback_value=fallback_value,
                impact_severity=impact_severity
            )

            # Add to session
            self.session.data_issues.append(issue)

            # Log to data issues file
            issue_msg = (
                f"[{issue_type.upper()}] {affected_component}: {description}")
            if fallback_action:
                issue_msg += f" | Fallback: {fallback_action}"
            if original_value is not None and fallback_value is not None:
                issue_msg += f" | Changed from {original_value} to {fallback_value}"
            issue_msg += f" | Severity: {impact_severity}"

            self.data_logger.warning(issue_msg)

            # Also log to main logger based on severity
            if impact_severity == 'critical':
                self.main_logger.error(f"🚨 CRITICAL DATA ISSUE: {issue_msg}")
            elif impact_severity == 'high':
                self.main_logger.warning(
                    f"⚠️ HIGH IMPACT DATA ISSUE: {issue_msg}")
            else:
                self.main_logger.info(f"ℹ️ Data Issue: {issue_msg}")

    def log_missing_data(self, component: str, parameter: str, fallback_value: Any = None,
                         impact: str = 'medium'):
        """Convenience method for logging missing data"""
        fallback_action = f"Using fallback value: {fallback_value}" if fallback_value else "No fallback available"
        self.log_data_issue(
            issue_type='missing_data',
            affected_component=component,
            description=f"Missing parameter: {parameter}",
            fallback_action=fallback_action,
            fallback_value=fallback_value,
            impact_severity=impact
        )

    def log_invalid_data(self, component: str, parameter: str, invalid_value: Any,
                         corrected_value: Any = None, impact: str = 'medium'):
        """Convenience method for logging invalid data"""
        corrected_action = f"Corrected to: {corrected_value}" if corrected_value else "Value removed"
        self.log_data_issue(
            issue_type='invalid_data',
            affected_component=component,
            description=f"Invalid value for {parameter}",
            fallback_action=corrected_action,
            original_value=invalid_value,
            fallback_value=corrected_value,
            impact_severity=impact
        )

    def log_fallback_usage(self, component: str, reason: str, fallback_description: str,
                           impact: str = 'low'):
        """Convenience method for logging fallback usage"""
        self.log_data_issue(
            issue_type='fallback_used',
            affected_component=component,
            description=reason,
            fallback_action=fallback_description,
            impact_severity=impact
        )

    def log_phase_start(self, phase_name: str, details: str = ""):
        """Log the start of an analysis phase"""
        msg = f"🔄 Starting phase: {phase_name}"
        if details:
            msg += f" - {details}"
        self.main_logger.info(msg)

    def log_phase_complete(self, phase_name: str, duration: float = None,
                           results_summary: str = ""):
        """Log the completion of an analysis phase"""
        msg = f"✅ Completed phase: {phase_name}"
        if duration:
            msg += f" (Duration: {duration:.2f}s)"
        if results_summary:
            msg += f" - {results_summary}"
        self.main_logger.info(msg)

    def log_calculation_result(self, metric_name: str, value: Any, unit: str = "",
                               details: str = ""):
        """Log calculation results"""
        msg = f"📈 {metric_name}: {value}"
        if unit:
            msg += f" {unit}"
        if details:
            msg += f" ({details})"
        self.main_logger.info(msg)

    def end_session(self, status: str = 'completed', error_msg: str = None):
        """End the logging session and generate summary"""
        with self._lock:
            self.session.session_end = datetime.now().isoformat()
            self.session.status = status

            # Log session end
            if status == 'completed':
                self.main_logger.info("🎉 TEA analysis completed successfully")
            elif status == 'failed':
                self.main_logger.error(f"❌ TEA analysis failed: {error_msg}")

            # Generate data issues summary
            self._generate_data_issues_summary()

            # Save session summary
            self._save_session_summary()

            self.main_logger.info("="*80)

    def _generate_data_issues_summary(self):
        """Generate summary of data issues"""
        if not self.session.data_issues:
            self.main_logger.info(
                "✅ No data issues encountered during analysis")
            return

        # Count issues by type and severity
        issue_counts = {}
        severity_counts = {}

        for issue in self.session.data_issues:
            issue_counts[issue.issue_type] = issue_counts.get(
                issue.issue_type, 0) + 1
            severity_counts[issue.impact_severity] = severity_counts.get(
                issue.impact_severity, 0) + 1

        self.main_logger.info("📊 DATA ISSUES SUMMARY:")
        self.main_logger.info(
            f"   Total Issues: {len(self.session.data_issues)}")

        for issue_type, count in issue_counts.items():
            self.main_logger.info(f"   {issue_type}: {count}")

        self.main_logger.info("📊 SEVERITY BREAKDOWN:")
        for severity, count in severity_counts.items():
            self.main_logger.info(f"   {severity}: {count}")

        # Log most critical issues
        critical_issues = [
            i for i in self.session.data_issues if i.impact_severity == 'critical']
        if critical_issues:
            self.main_logger.warning(
                f"🚨 {len(critical_issues)} CRITICAL ISSUES FOUND:")
            for issue in critical_issues:
                self.main_logger.warning(
                    f"   - {issue.affected_component}: {issue.description}")

    def _save_session_summary(self):
        """Save session summary to JSON file"""
        try:
            session_dict = asdict(self.session)
            with open(self.summary_file, 'w') as f:
                json.dump(session_dict, f, indent=2, default=str)
            self.main_logger.info(
                f"📄 Session summary saved: {self.summary_file}")
        except Exception as e:
            self.main_logger.error(f"Failed to save session summary: {e}")

    # Delegate common logging methods to main logger
    def debug(self, msg): self.main_logger.debug(msg)
    def info(self, msg): self.main_logger.info(msg)
    def warning(self, msg): self.main_logger.warning(msg)
    def error(self, msg): self.main_logger.error(msg)
    def critical(self, msg): self.main_logger.critical(msg)


def create_reactor_logger(reactor_name: str, generator_id: str, iso_region: str,
                          remaining_years: str) -> EnhancedReactorLogger:
    """Factory function to create standardized reactor logger"""
    return EnhancedReactorLogger(reactor_name, generator_id, iso_region, remaining_years)


# Context manager for automatic session management
class ReactorLogSession:
    """Context manager for reactor logging session"""

    def __init__(self, reactor_name: str, generator_id: str, iso_region: str, remaining_years: str):
        self.logger = create_reactor_logger(
            reactor_name, generator_id, iso_region, remaining_years)

    def __enter__(self):
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.end_session(status='failed', error_msg=str(exc_val))
        else:
            self.logger.end_session(status='completed')
