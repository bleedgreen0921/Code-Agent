from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    CONFIGURATION_CHANGED = "CONFIGURATION_CHANGED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    CONVERSATION_BUSY = "CONVERSATION_BUSY"
    QUEUE_CAPACITY_EXCEEDED = "QUEUE_CAPACITY_EXCEEDED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TurnCreate(StrictModel):
    task: str = Field(min_length=1)


class BindingFields(StrictModel):
    profile: str = Field(default="chat", min_length=1)
    workspace_id: str | None = Field(default=None, min_length=1)


class RunCreate(TurnCreate, BindingFields):
    pass


class ConversationCreate(BindingFields):
    title: str = Field(default="", max_length=200)


RunStatus = Literal[
    "queued", "running", "cancelling", "completed", "failed", "cancelled"
]
