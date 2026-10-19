"""Fixed-scope Review contracts, separate from the analysis report."""

from typing import Annotated, Literal
from uuid import UUID
from pydantic import Field, field_validator
from contracts.v1 import StrictModel
from contracts.code import CodeTaskCreate, CommitTarget


class ReviewWorkspaceTarget(StrictModel):
    kind: Literal["workspace"] = "workspace"
    mode: Literal["all", "staged", "unstaged"] = "all"


class RangeTarget(StrictModel):
    kind: Literal["range"]
    base: str = Field(min_length=1, max_length=1024)
    head: str = Field(min_length=1, max_length=1024)
    comparison: Literal["direct", "merge_base"] = "direct"
    _refs = field_validator("base", "head")(CommitTarget.local_ref.__func__)


class CodeReviewCreate(CodeTaskCreate):
    stage: Literal["review"] = "review"
    target: Annotated[
        ReviewWorkspaceTarget | CommitTarget | RangeTarget, Field(discriminator="kind")
    ] = Field(default_factory=ReviewWorkspaceTarget)


class ReviewSourceReference(StrictModel):
    evidence_id: UUID
    side: Literal["base", "head"]
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ReviewLocation(StrictModel):
    change_id: str
    side: Literal["base", "head"]
    path: str
    line: int | None = Field(default=None, ge=1)


class ReviewFinding(StrictModel):
    title: str = Field(min_length=1)
    priority: Literal["P0", "P1", "P2", "P3"]
    location: ReviewLocation
    trigger: str = Field(min_length=1)
    cause: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    introduction_basis: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    diff_evidence_ids: list[UUID] = Field(min_length=1)
    evidence: list[ReviewSourceReference] = Field(default_factory=list)
    reproduction_status: Literal[
        "not_run", "observed", "not_observed", "inconclusive"
    ] = "not_run"
    reproduction_ids: list[UUID] = Field(default_factory=list)
    # A suite failure is never a finding-specific reproduction.
    check_case_refs: list[str] = Field(default_factory=list)
    recommendation: str = Field(min_length=1)


class ReviewDeclaration(StrictModel):
    change_id: str
    status: Literal["reviewed", "partial", "excluded", "unreviewed"]
    reason: str = Field(min_length=1)
    evidence: list[ReviewSourceReference] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)


class ReviewReportV1(StrictModel):
    version: Literal[1] = 1
    summary: str = Field(min_length=1)
    findings: list[ReviewFinding]
    changes: list[ReviewDeclaration]
    unconfirmed: list[str]
    missing_context: list[str]
