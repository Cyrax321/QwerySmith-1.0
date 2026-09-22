from __future__ import annotations


class HarnessError(Exception):
    pass


class ConfigError(HarnessError):
    pass


class UnsafeSQLError(HarnessError):
    pass


class QueryTimeoutError(HarnessError):
    pass


class ProfileError(HarnessError):
    pass


class ValidationError(HarnessError):
    pass


class HoldoutLeakError(ValidationError):
    pass


class StageNotImplementedError(HarnessError):
    def __init__(self, stage: str, milestone: str):
        super().__init__(f"stage '{stage}' is not implemented yet (planned for {milestone})")