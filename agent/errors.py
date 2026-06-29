"""Exception hierarchy for the agent.

Recoverable errors are caught inside a phase, logged, and the pipeline continues.
Fatal errors (subclasses of AgentFatalError) abort the run with a remediation message.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all agent errors."""


class AgentFatalError(AgentError):
    """Unrecoverable error. The run must stop and the user must intervene.

    Carries a human-readable remediation message describing how to fix the
    condition (e.g. accept rules, free disk space, set credentials).
    """

    def __init__(self, message: str, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


class BootstrapError(AgentFatalError):
    """Phase 0 validation failed (e.g. ambiguous resume state, bad URL)."""


class StateVersionMismatchError(AgentFatalError):
    """Resumed state.json has a state_version incompatible with the current schema.

    Never silently migrate — instruct the user to --force-restart.
    """


class DeferredOpError(AgentError):
    """A deferred in-fold FE operation (target/group encoding) was invoked
    outside the CV fold context. These operations must never be fit on the
    full training set."""


class SubmissionError(AgentError):
    """Submission generation or upload failed validation."""
