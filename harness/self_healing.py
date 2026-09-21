#!/usr/bin/env python3
"""
harness/self_healing.py -- Backward compatibility re-export of harness.reflection.self_healing
"""

from .reflection.self_healing import (
    ErrorDiagnosis,
    MultiStepRepairTracker,
    RepairAttempt,
    SelfHealingEngine,
)

# Legacy alias
RepairRecord = MultiStepRepairTracker

__all__ = [
    "SelfHealingEngine",
    "ErrorDiagnosis",
    "RepairAttempt",
    "MultiStepRepairTracker",
    "RepairRecord",
]
