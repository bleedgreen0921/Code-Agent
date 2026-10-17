"""Versioned public code-analysis inputs. Authority fields are never model supplied."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator

from contracts.v1 import StrictModel


class WorkspaceTarget(StrictModel):
    kind: Literal["workspace"] = "workspace"


class CommitTarget(StrictModel):
    kind: Literal["commit"]
    ref: str = Field(min_length=1, max_length=1024)

    @field_validator("ref")
    @classmethod
    def local_ref(cls, value):
        import re

        if (
            value.startswith("-")
            or ".." in value
            or not re.fullmatch(r"[^\x00\s:~^{}]+(?:[~^][0-9]*)*", value)
        ):
            raise ValueError("Expected a single local commit reference")
        return value


class CodeTaskCreate(StrictModel):
    stage: Literal["analysis"] = "analysis"
    task: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    target: Annotated[WorkspaceTarget | CommitTarget, Field(discriminator="kind")] = (
        Field(default_factory=WorkspaceTarget)
    )
    context_task_ids: list[UUID] = Field(default_factory=list, max_length=8)
    check_suite_id: str | None = Field(default=None, min_length=1)

    @field_validator("context_task_ids")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Duplicate context tasks")
        return value


class SourceReference(StrictModel):
    evidence_id: UUID
    path: str
    source: Literal["workspace", "index", "commit"] = "workspace"
    commit: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ImplementationLocation(StrictModel):
    symbol: str
    responsibility: str
    source: SourceReference


class CallRelationship(StrictModel):
    caller: str
    callee: str
    explanation: str
    evidence: list[SourceReference] = Field(min_length=1)


class BehaviorConclusion(StrictModel):
    conclusion: str
    conditions: str
    classification: Literal["fact", "inference", "unconfirmed"]
    basis: Literal["source", "execution", "context"]
    evidence: list[SourceReference] = []
    check_run_ids: list[UUID] = []


class AnalysisScope(StrictModel):
    question: str
    covered: list[str]
    missing: list[str]


class AnalysisReportV1(StrictModel):
    version: Literal[1] = 1
    scope: AnalysisScope
    implementation_locations: list[ImplementationLocation]
    call_relationships: list[CallRelationship]
    behavior_conclusions: list[BehaviorConclusion]
    execution_paths: list[str]
    exception_paths: list[str]
    unconfirmed: list[str]
