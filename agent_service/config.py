"""Configuration for the core framework; no business providers or embeddings."""

import math
import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Settings:
    queue_timeout: float = 60
    run_timeout: float = 300
    model_timeout: float = 120
    max_output_tokens: int = 4096
    max_active_runs: int = 2
    max_queued_runs: int = 8
    max_unfinished_runs: int = 10
    summary_trigger_tokens: int = 20000
    summary_keep_tokens: int = 6000
    summary_trim_tokens: int = 16000
    lease_seconds: float = 30
    heartbeat_seconds: float = 2
    poll_seconds: float = 0.2

    def __post_init__(self):
        for field in fields(self):
            if (
                not math.isfinite(getattr(self, field.name))
                or getattr(self, field.name) <= 0
            ):
                raise ValueError(f"{field.name} must be positive")
        if self.summary_keep_tokens >= self.summary_trigger_tokens:
            raise ValueError("Summary retention must be smaller than its trigger")
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("Lease must exceed two heartbeat intervals")

    @classmethod
    def from_env(cls):
        default = cls()
        return cls(
            **{
                f.name: f.type(
                    os.environ.get(
                        "AGENT_MODEL_TIMEOUT_SECONDS"
                        if f.name == "model_timeout"
                        else "AGENT_" + f.name.upper(),
                        getattr(default, f.name),
                    )
                )
                for f in fields(cls)
            }
        )
