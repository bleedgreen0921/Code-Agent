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
