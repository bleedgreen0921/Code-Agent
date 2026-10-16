"""Fatal execution controls, never recoverable tool errors."""


class ExecutionFailure(RuntimeError):
    code = "EXECUTION_ENVIRONMENT_LOST"


class EnvironmentUnavailable(ExecutionFailure):
    code = "EXECUTION_ENVIRONMENT_UNAVAILABLE"


class EnvironmentLost(ExecutionFailure):
    pass


class CommandTimeout(ExecutionFailure):
    code = "COMMAND_TIMEOUT"


class CleanupFailed(ExecutionFailure):
    code = "RESOURCE_CLEANUP_FAILED"

    def __init__(self, message, *, reason=None):
        self.reason = reason
        super().__init__(message)


def safe_cleanup_error(exc):
    if isinstance(exc, CleanupFailed):
        return {
            "create_unknown": "Resource creation outcome unknown; release unconfirmed.",
            "identity_mismatch": "Resource identity mismatch; release unconfirmed.",
        }.get(exc.reason, "Resource release unconfirmed.")
    return "Resource release unconfirmed."
