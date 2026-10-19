"""Trusted startup registration, independent of the legacy fixed verifier."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.code import AnalysisReportV1
from contracts.review import ReviewReportV1

VERSION = "code-analysis-v1"
PROFILE = "code-analysis-isolated"
REVIEW_PROFILE = "code-review-isolated"
CODE_PROFILES = (PROFILE, REVIEW_PROFILE, "code-repair-isolated")
REVIEW_TOOLS = (
    "glob_search",
    "grep_search",
    "read_file",
    "git_log",
    "write_todos",
    "read_skill",
    "run_project_check",
    "review_diff",
    "review_read_file",
    "read_review_execution",
    "run_review_reproduction",
)
READ_TOOLS = (
    "glob_search",
    "grep_search",
    "read_file",
    "git_status",
    "git_log",
    "git_diff",
    "git_read_file",
)
TOOLS = (*READ_TOOLS, "write_todos", "read_skill", "run_project_check")


def relative(value, *, dot=False):
    if dot and value == ".":
        return value
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or value.startswith(("/", "-", ":"))
        or any(x in ("", ".", "..", ".git") for x in value.split("/"))
    ):
        raise ValueError("Expected a literal relative source path")
    return value


class CodeAnalysisSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = False
    storage_root: str | None = None
    profile: str = PROFILE
    check_suites: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    workspace_suites: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    default_suites: dict[str, str] = Field(default_factory=dict)
    model_calls: int = Field(default=16, ge=1, le=16)
    tool_calls: int = Field(default=32, ge=1, le=32)
    check_calls: int = Field(default=12, ge=1, le=12)
    run_seconds: int = Field(default=300, ge=31, le=300)
    snapshot_seconds: int = Field(default=60, ge=1, le=60)
    check_seconds: int = Field(default=60, ge=1, le=60)
    reserve_seconds: int = Field(default=30, ge=30)
    snapshot_bytes: int = Field(default=1024**3, ge=1, le=1024**3)
    file_count: int = Field(default=50000, ge=1, le=50000)
    file_bytes: int = Field(default=10 * 1024**2, ge=1, le=10 * 1024**2)
    log_bytes: int = Field(default=1024**2, ge=1, le=1024**2)
    result_bytes: int = Field(default=1024**2, ge=1, le=1024**2)
    report_bytes: int = Field(default=65536, ge=1, le=65536)
    retention_seconds: int = Field(default=7 * 86400, ge=1)

    @field_validator("profile")
    @classmethod
    def reserved_profile(cls, value):
        if value != PROFILE:
            raise ValueError("The isolated analysis profile ID is reserved")
        return value


class CodeReviewSettings(CodeAnalysisSettings):
    profile: str = REVIEW_PROFILE
    model_calls: int = Field(default=24, ge=1, le=24)
    tool_calls: int = Field(default=48, ge=1, le=48)
    run_seconds: int = Field(default=600, ge=31, le=600)
    reproduction_calls: int = Field(default=4, ge=0, le=4)
    diff_bytes: int = Field(default=16 * 1024**2, ge=1, le=16 * 1024**2)
    patch_bytes: int = Field(default=2 * 1024**2, ge=1, le=2 * 1024**2)

    @field_validator("profile")
    @classmethod
    def reserved_profile(cls, value):
        if value != REVIEW_PROFILE:
            raise ValueError("The isolated review profile ID is reserved")
        return value


@dataclass(frozen=True)
class ProjectCheckDefinition:
    id: str
    version: str
    adapter_id: str
    workspaces: tuple[str, ...]
    description: str = ""
    cwd: str = "."
    paths: tuple[str, ...] = ("tests",)
    cases: tuple[str, ...] = ()
    pattern: str = "test*.py"
    timeout: int = 60
    expected_cases: tuple[str, ...] = ()

    def manifest(self):
        return asdict(self)

    def validate(self):
        import re

        for value in (self.id, self.version, self.adapter_id):
            if not isinstance(value, str) or not re.fullmatch(
                r"[a-zA-Z0-9_.-]{1,128}", value
            ):
                raise ValueError("Invalid check identifier/version")
        if (
            not isinstance(self.description, str)
            or len(self.description.encode()) > 4096
        ):
            raise ValueError("Check description must be bounded text")
        relative(self.cwd, dot=True)
        for path in self.paths:
            # pytest node selectors are registered, but never arbitrary CLI flags.
            relative(path.split("::")[0], dot=True)
        for values in (self.paths, self.cases, self.workspaces, self.expected_cases):
            if (
                type(values) is not tuple
                or len(values) > 1000
                or len(set(values)) != len(values)
            ):
                raise ValueError("Expected unique immutable check selections")
            if any(
                type(v) is not str or not v or "\x00" in v or len(v) > 1024
                for v in values
            ):
                raise ValueError("Invalid check selection")
        if (
            not self.workspaces
            or type(self.timeout) is not int
            or not 1 <= self.timeout <= 60
        ):
            raise ValueError("Invalid check workspaces/timeout")
        if "/" in self.pattern or "\\" in self.pattern or len(self.pattern) > 128:
            raise ValueError("Invalid unittest discovery pattern")
        if self.cases and any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", c) for c in self.cases
        ):
            raise ValueError("Expected unittest qualified test names")


class CheckAdapter:
    """Trusted adapters construct a fixed runner request, never a shell command."""

    id = ""
    version = "1"

    def validate(self, definition):
        definition.validate()

    def request(self, definition):
        return {"adapter": self.id, "definition": definition.manifest()}

    def parse(self, raw, *, exit_code, **limits):
        from agent_service.project_checks import classify_result

        return classify_result(raw, exit_code=exit_code, **limits)


class UnittestAdapter(CheckAdapter):
    id = "python-unittest-v1"

    def validate(self, definition):
        super().validate(definition)
        if not definition.cases and len(definition.paths) != 1:
            raise ValueError("unittest discovery requires exactly one path")
        for path in definition.paths:
            relative(path, dot=True)


class PytestAdapter(CheckAdapter):
    id = "python-pytest-v1"

    def validate(self, definition):
        super().validate(definition)
        if definition.cases or not definition.paths:
            raise ValueError("pytest requires registered paths/node selectors")


def assets():
    root = Path(__file__).resolve().parents[1] / "execution"
    return {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in (
            "code_snapshot.py",
            "code_review.py",
            "review_runner.py",
            "project_runner.py",
            "git_tools.py",
            "helper.py",
            "entrypoint.py",
            "repair_helper.py",
            "acceptance_runner.py",
            "acceptance_support.py",
            "acceptance_app.py",
            *(
                p.relative_to(root).as_posix()
                for p in sorted((root / "acceptance").rglob("*.py"))
            ),
        )
    }


def asset_digest():
    return hashlib.sha256(json.dumps(assets(), sort_keys=True).encode()).hexdigest()


class CodeRegistry:
    def __init__(self, settings, workspaces, execution, definitions=(), adapters=()):
        self.settings = settings
        all_adapters = (UnittestAdapter(), PytestAdapter(), *adapters)
        self.adapters = MappingProxyType({a.id: a for a in all_adapters})
        self.checks = MappingProxyType({c.id: c for c in definitions})
        if len(self.adapters) != len(all_adapters) or len(self.checks) != len(
            definitions
        ):
            raise ValueError("Duplicate project check or adapter")
        for check in definitions:
            if (
                type(check) is not ProjectCheckDefinition
                or check.adapter_id not in self.adapters
            ):
                raise ValueError("Unknown project check adapter")
            self.adapters[check.adapter_id].validate(check)
            if set(check.workspaces) - workspaces.keys():
                raise ValueError("Unknown check workspace")
        for suite, ids in settings.check_suites.items():
            if not suite or len(ids) != len(set(ids)) or set(ids) - self.checks.keys():
                raise ValueError("Unknown or duplicate checks in suite")
        for workspace, suites in settings.workspace_suites.items():
            if (
                workspace not in workspaces
                or len(suites) != len(set(suites))
                or set(suites) - settings.check_suites.keys()
            ):
                raise ValueError("Unknown workspace/check suite")
            for suite in suites:
                if any(
                    workspace not in self.checks[c].workspaces
                    for c in settings.check_suites[suite]
                ):
                    raise ValueError("Check not authorized for workspace")
        for workspace, suite in settings.default_suites.items():
            if suite not in settings.workspace_suites.get(workspace, ()):
                raise ValueError("Default suite is not authorized")
        if settings.enabled:
            if not settings.storage_root or not execution.image:
                raise ValueError(
                    "Code analysis requires storage_root and immutable execution image"
                )
            root = Path(settings.storage_root)
            if (
                not root.is_absolute()
                or not root.is_dir()
                or str(root.resolve()) != str(root)
                or "," in str(root)
            ):
                raise ValueError(
                    "Managed storage must be an existing canonical absolute directory"
                )
            if not os.access(root, os.W_OK | os.X_OK):
                raise ValueError("Managed storage must be writable and searchable")
            for workspace in workspaces.values():
                if root.is_relative_to(workspace) or Path(workspace).is_relative_to(
                    root
                ):
                    raise ValueError("Managed storage and workspaces must not overlap")
            if settings.reserve_seconds >= settings.run_seconds:
                raise ValueError("Run budget must exceed cleanup/publication reserve")
        self.manifest = json.loads(
            json.dumps(
                {
                    "version": VERSION,
                    "report_schema": (
                        ReviewReportV1
                        if settings.profile == REVIEW_PROFILE
                        else AnalysisReportV1
                    ).model_json_schema(),
                    "settings": settings.model_dump(),
                    "checks": [
                        c.manifest() for c in sorted(definitions, key=lambda c: c.id)
                    ],
                    "adapters": [
                        {"id": a.id, "version": a.version}
                        for a in sorted(all_adapters, key=lambda a: a.id)
                    ],
                    "assets": assets(),
                    "image": execution.image,
                }
            )
        )

    def select(self, workspace_id, suite):
        suite = (
            suite
            if suite is not None
            else self.settings.default_suites.get(workspace_id)
        )
        if suite is not None and suite not in self.settings.workspace_suites.get(
            workspace_id, ()
        ):
            raise ValueError("Unknown or unauthorized check suite")
        return suite, self.settings.check_suites.get(suite, ())
