"""Candidate-only repair inputs. Acceptance authority is never model supplied."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from contracts.code import CodeTaskCreate, WorkspaceTarget
from contracts.v1 import StrictModel


class CodeRepairCreate(CodeTaskCreate):
    stage: Literal["repair"] = "repair"
    delivery_mode: Literal["candidate_only"]
    target: WorkspaceTarget = Field(default_factory=WorkspaceTarget)
    acceptance_suite_id: str | None = Field(default=None, min_length=1)


class RepairSourceReference(StrictModel):
    evidence_id: UUID
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class RepairGoal(StrictModel):
    goal: str = Field(min_length=1)
    status: Literal["resolved", "unconfirmed", "unfinished"]
    explanation: str = Field(min_length=1)


class RepairDraftV1(StrictModel):
    version: Literal[1] = 1
    summary: str = Field(min_length=1)
    cause: str = Field(min_length=1)
    changes: list[str] = Field(min_length=1)
    baseline_confirmation: str = Field(min_length=1)
    goals: list[RepairGoal] = Field(min_length=1)
    unfinished: list[str]
    sources: list[RepairSourceReference]
    check_run_ids: list[UUID]
    test_changes: list[str]
    limitations: list[str]


class RepairReportV1(StrictModel):
    version: Literal[1] = 1
    delivery_mode: Literal["candidate_only"] = "candidate_only"
    delivery_status: Literal["not_applied"] = "not_applied"
    report: RepairDraftV1
    authority: dict
