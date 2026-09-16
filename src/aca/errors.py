"""Exception hierarchy for ACA.

Kept deliberately small. Each error names a distinct failure boundary so callers can
distinguish, for example, a malformed worker result from a duplicate-ingress condition.
"""

from __future__ import annotations


class AcaError(Exception):
    """Base class for all ACA errors."""


class ConfigError(AcaError):
    """Raised when configuration is missing, malformed, or out of bounds."""


class DurationParseError(ConfigError):
    """Raised when a human-readable duration string cannot be parsed."""


class ValidationError(AcaError):
    """Raised when an untrusted proposal or payload fails validation.

    Untrusted data (LLM worker output, client payloads) that fails validation raises this
    rather than being silently coerced. The reducer treats it as "no authorized change".
    """


class PersistenceError(AcaError):
    """Raised on a durable-storage integrity or access failure."""


class LockError(AcaError):
    """Raised when the single-instance advisory lock cannot be acquired."""


class ServiceAlreadyRunningError(LockError):
    """Raised when another agent service already owns this data directory."""


class IpcError(AcaError):
    """Raised on an IPC transport or protocol failure."""
