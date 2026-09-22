"""QwerySmith v3.0 - reusable text-to-SQL evaluation and fine-tuning harness."""

__version__ = "3.0.0"

from .exceptions import (
    HarnessError,
    UnsafeSQLError,
    QueryTimeoutError,
    ConfigError,
    ValidationError,
    ProfileError,
    StageNotImplementedError,
    HoldoutLeakError,
)

__all__ = [
    "__version__",
    "HarnessError",
    "UnsafeSQLError",
    "QueryTimeoutError",
    "ConfigError",
    "ValidationError",
    "ProfileError",
    "StageNotImplementedError",
    "HoldoutLeakError",
]