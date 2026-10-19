"""Startup-only acceptance registration, separate from project development checks."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import Field, field_validator

from agent_service.code_registry import CodeAnalysisSettings, CodeRegistry, relative
from contracts.repair import RepairDraftV1

REPAIR_PROFILE = "code-repair-isolated"
REPAIR_TOOLS = (
    "glob_search",
    "grep_search",
    "read_file",
    "git_log",
    "edit_file",
    "shell",
    "candidate_diff",
    "run_project_check",
    "read_repair_execution",
    "read_repair_context",
    "write_todos",
    "read_skill",
)


class CodeRepairSettings(CodeAnalysisSettings):
    profile: str = REPAIR_PROFILE
    preserve_modes: Literal[True] = True
    model_calls: int = Field(default=32, ge=1, le=32)
    tool_calls: int = Field(default=64, ge=1, le=64)
    run_seconds: int = Field(default=900, ge=61, le=900)
    reserve_seconds: int = Field(default=60, ge=60)
    gate_rounds: int = Field(default=4, ge=1, le=4)
    round_seconds: int = Field(default=120, ge=1, le=120)
    storage_bytes: int = Field(default=4 * 1024**3, ge=1, le=4 * 1024**3)
    acceptance_suites: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    default_acceptance_suites: dict[str, str] = Field(default_factory=dict)

    @field_validator("profile")
    @classmethod
    def reserved_profile(cls, value):
        if value != REPAIR_PROFILE:
            raise ValueError("The isolated repair profile ID is reserved")
        return value


def acceptance_assets(directory):
    root = Path(__file__).resolve().parents[1] / "execution/acceptance" / directory
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Acceptance asset directory missing")
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Acceptance symlink forbidden")
        if p.is_file() and "__pycache__" not in p.parts:
            files[p.relative_to(root).as_posix()] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
    if not files:
        raise ValueError("Acceptance assets must not be empty")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AcceptanceSuiteDefinition:
    id: str
    version: str
    workspaces: tuple[str, ...]
    image: str
    asset_directory: str
    asset_digest: str
    required_cases: tuple[str, ...]
    application_module: str
    import_path: str = "."
    runner: str = "python-acceptance-rpc-v1"
    timeout: int = 60
    environment: tuple[tuple[str, str], ...] = ()

    def manifest(self):
        return asdict(self)

    def validate(self, workspaces, execution):
        for value in (self.id, self.version):
            if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", value):
                raise ValueError("Invalid acceptance identifier")
        relative(self.asset_directory)
        relative(self.import_path, dot=True)
        if (
            self.runner != "python-acceptance-rpc-v1"
            or self.image != execution.image
            or not self.image
            or not self.workspaces
            or type(self.workspaces) is not tuple
            or set(self.workspaces) - workspaces.keys()
            or type(self.required_cases) is not tuple
            or not self.required_cases
            or len(self.required_cases) > 1000
            or len(set(self.required_cases)) != len(self.required_cases)
            or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", self.application_module
            )
            or type(self.timeout) is not int
            or not 1 <= self.timeout <= 60
        ):
            raise ValueError("Invalid acceptance authority")
        if any(
            not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+", case)
            for case in self.required_cases
        ):
            raise ValueError("Invalid required acceptance cases")
        if self.asset_digest != acceptance_assets(self.asset_directory):
            raise ValueError("Acceptance asset digest mismatch")
        if type(self.environment) is not tuple or self.environment:
            raise ValueError("This runner uses its fixed empty application environment")


class RepairRegistry(CodeRegistry):
    def __init__(
        self,
        settings,
        workspaces,
        execution,
        definitions=(),
        adapters=(),
        acceptance=(),
    ):
        super().__init__(settings, workspaces, execution, definitions, adapters)
        self.acceptance = MappingProxyType({s.id: s for s in acceptance})
        if len(self.acceptance) != len(acceptance):
            raise ValueError("Duplicate acceptance suite")
        for suite in acceptance:
            if type(suite) is not AcceptanceSuiteDefinition:
                raise ValueError("Trusted acceptance definition required")
            suite.validate(workspaces, execution)
        for workspace, suites in settings.acceptance_suites.items():
            if (
                workspace not in workspaces
                or len(suites) != len(set(suites))
                or any(
                    s not in self.acceptance
                    or workspace not in self.acceptance[s].workspaces
                    for s in suites
                )
            ):
                raise ValueError("Unauthorized acceptance suite")
        for workspace, suite in settings.default_acceptance_suites.items():
            self.select_acceptance(workspace, suite)
        self.manifest.update(
            report_schema=RepairDraftV1.model_json_schema(),
            acceptance=[s.manifest() for s in sorted(acceptance, key=lambda s: s.id)],
            candidate_policy="repair-candidate-v1",
        )
        self.manifest = json.loads(json.dumps(self.manifest))

    def select_acceptance(self, workspace, suite):
        suite = (
            suite
            if suite is not None
            else self.settings.default_acceptance_suites.get(workspace)
        )
        if suite is None or suite not in self.settings.acceptance_suites.get(
            workspace, ()
        ):
            raise ValueError("An authorized independent acceptance suite is required")
        return self.acceptance[suite]

    def validate_budget(self, checks, *, acceptance=None, run_seconds=None):
        # Baseline acceptance + one full gate must fit even before a model is called.
        if self.settings.check_calls < len(checks) + 2:
            raise ValueError(
                "Budget cannot fit baseline and one required candidate gate"
            )
        if acceptance is not None:
            baseline_seconds = min(acceptance.timeout, self.settings.check_seconds)
            gate_seconds = min(
                self.settings.round_seconds,
                baseline_seconds
                + sum(
                    min(self.checks[c].timeout, self.settings.check_seconds)
                    for c in checks
                ),
            )
            available = (
                min(self.settings.run_seconds, run_seconds or self.settings.run_seconds)
                - self.settings.reserve_seconds
            )
            if baseline_seconds + gate_seconds > available:
                raise ValueError(
                    "Time budget cannot fit baseline and one required candidate gate"
                )


def repair_skill():
    from agent_service.skills import Skill

    body = (
        (Path(__file__).resolve().parents[1] / "repair_skills/isolated-repair/SKILL.md")
        .read_text()
        .split("---", 2)[-1]
        .strip()
    )
    return Skill(
        "isolated-repair",
        "Repair a versioned candidate with independent acceptance and same-graph feedback.",
        body,
        hashlib.sha256(body.encode()).hexdigest(),
        "{}",
    )
