"""Trusted completion verification. This interface is deliberately not a Python sandbox."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
    field_validator,
)

from agent_service.execution_errors import ExecutionFailure
from agent_service.failure import is_control, control_leaf
from agent_service.output import limit_output
from agent_service.reliability import remaining_execution

if TYPE_CHECKING:
    from agent_service.checks import CheckExecutionResult

log = logging.getLogger(__name__)
MAX_REPORT_BYTES = 32768


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class VerificationCheck(Strict):
    status: Literal["passed", "failed", "inconclusive", "not_checked"]
    explanation: str = ""

    @model_validator(mode="after")
    def bounded(self):
        if "\x00" in self.explanation or len(self.explanation.encode()) > 512:
            raise ValueError("Check explanation exceeds 512 UTF-8 bytes")
        return self


class VerificationResult(Strict):
    status: Literal["passed", "failed", "inconclusive", "not_checked"]
    summary: str = ""
    checks: tuple[VerificationCheck, ...] = Field(default=(), max_length=16)
    reason: str | None = Field(
        default=None, max_length=128, pattern=r"^[a-z][a-z0-9_]*$"
    )

    @model_validator(mode="after")
    def bounded(self):
        if "\x00" in self.summary or len(self.summary.encode()) > 1024:
            raise ValueError("Summary exceeds 1 KiB UTF-8")
        if (
            len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode())
            > MAX_REPORT_BYTES
        ):
            raise ValueError("Verification result exceeds report capacity")
        return self


class CheckpointReference(Strict):
    thread_id: str = Field(min_length=1, max_length=128)
    checkpoint_ns: Literal[""] = ""
    checkpoint_id: str = Field(min_length=1, max_length=128)

    @classmethod
    def from_config(cls, config):
        values = config["configurable"]
        return cls(
            thread_id=values["thread_id"],
            checkpoint_ns=values.get("checkpoint_ns", ""),
            checkpoint_id=values["checkpoint_id"],
        )


class FileObservation(Strict):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=0)
    truncated: bool
    consistent: bool = False

    @model_validator(mode="after")
    def bounded(self):
        if len(self.path.encode()) > 512 or normalize_path(self.path) != self.path:
            raise ValueError("Invalid coverage path")
        return self


class VerificationReport(VerificationResult):
    version: Literal[1] = 1
    mode: Literal["report", "required"]
    final_candidate: bool
    source_run_id: str
    checkpoint: CheckpointReference
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    binding_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    extension_version: str
    started_at: str
    finished_at: str
    files: tuple[FileObservation, ...] = Field(default=(), max_length=64)

    @field_validator("version", mode="before")
    @classmethod
    def strict_version(cls, value):
        if type(value) is not int:
            raise ValueError("Report version must be an integer")
        return value

    @model_validator(mode="after")
    def report_contract(self):
        # JSONB renders spaces; use that representation for the DB byte bound too.
        if type(self.version) is not int:
            raise ValueError("Report version must be an integer")
        if len({f.path for f in self.files}) > 32:
            raise ValueError("Too many covered files")
        if (
            len(json.dumps(self.model_dump(mode="json"), ensure_ascii=False).encode())
            > MAX_REPORT_BYTES
        ):
            raise ValueError("Report exceeds 32 KiB")
        if self.checkpoint.thread_id != self.source_run_id:
            raise ValueError("Foreign checkpoint")
        if not self.final_candidate and (
            self.status != "not_checked" or self.reason != "stage_report"
        ):
            raise ValueError("Stage reports cannot assert verification")
        if self.status in ("passed", "failed") and any(
            not f.consistent for f in self.files
        ):
            raise ValueError("Unconfirmed coverage")
        start, end = (
            datetime.fromisoformat(self.started_at),
            datetime.fromisoformat(self.finished_at),
        )
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid verification time interval")
        return self


class VerificationContractError(ExecutionFailure):
    code = "VERIFICATION_CONTRACT_ERROR"


class AccessBudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class VerificationFile:
    content: str
    sha256: str
    start_line: int
    end_line: int
    truncated: bool


@dataclass(frozen=True)
class VerificationContext:
    run_id: str
    task_id: str | None
    profile: str
    workspace_id: str | None
    workspace_identity: str | None
    binding_fingerprint: str
    extension_version: str
    candidate_answer: str
    checkpoint: CheckpointReference
    _remaining: Callable = field(repr=False, compare=False)
    _read: Callable = field(repr=False, compare=False)
    _run_check: Callable | None = field(default=None, repr=False, compare=False)

    def remaining_execution(self):
        return self._remaining()

    async def read_file(self, path, *, start_line=1, max_lines=200):
        return await self._read(path, start_line=start_line, max_lines=max_lines)

    async def run_check(self, check_id: str) -> "CheckExecutionResult":
        if self._run_check is None:
            raise VerificationContractError("No controlled checks are registered")
        return await self._run_check(check_id)


def normalize_path(path):
    if type(path) is not str or "\x00" in path:
        raise VerificationContractError("Invalid verification path")
    if path.startswith("/workspace/"):
        path = path[len("/workspace/") :]
    path = path.lstrip("/")
    parts = path.split("/")
    if any(p in ("", "..") for p in parts):
        raise VerificationContractError("Invalid verification path")
    normalized = "/".join(p for p in parts if p != ".")
    if not normalized:
        raise VerificationContractError("Regular file path required")
    return normalized


class FileAccess:
    def __init__(self, row, registry, context, scope):
        self.row, self.registry, self.context, self.scope = (
            row,
            registry,
            context,
            scope,
        )
        self.observations = []
        self.hashes = {}
        self.problem = None
        self.violation = None
        self.failure = None
        self.calls = 0
        self.open = True
        self.lock = asyncio.Lock()

    async def _read(self, path, start_line, max_lines):
        from agent_service.workspace_tools import ReadFileInput
        from agent_service.policy import PermissionDenied

        remaining_execution(self.row)
        if not self.open:
            raise VerificationContractError("Verification access has closed")
        profile = self.registry.resolve(self.row["submitted_binding"])
        if self.scope is None or "read_file" not in profile.tools:
            raise VerificationContractError("Profile does not permit file reading")
        if type(start_line) is not int or type(max_lines) is not int:
            raise VerificationContractError("Invalid verification read arguments")
        args = ReadFileInput(path=path, start_line=start_line, max_lines=max_lines)
        validator = self.registry.validators.get("read_file")
        try:
            if validator and validator(args, self.context) is False:
                raise VerificationContractError("Verification read denied")
        except PermissionDenied:
            raise VerificationContractError("Verification read denied") from None
        async with self.scope.lock:
            await self.scope.healthy()
            result = await self.scope.file_call("read_file", args.model_dump())
        return result

    async def read(self, path, *, start_line=1, max_lines=200):
        # Serialize supported access including recording; no detached read may add
        # coverage after the result has frozen and the access token is revoked.
        async with self.lock:
            try:
                normalized = normalize_path(path)
                if len(normalized.encode()) > 512:
                    self.problem = "access_budget_exhausted"
                    raise AccessBudgetExceeded()
                projected = FileObservation(
                    path=normalized,
                    sha256="0" * 64,
                    start_line=1,
                    end_line=200,
                    truncated=False,
                )
                coverage_bytes = len(
                    json.dumps(
                        [f.model_dump() for f in (*self.observations, projected)],
                        ensure_ascii=False,
                    ).encode()
                )
                if (
                    self.calls >= 64
                    or (normalized not in self.hashes and len(self.hashes) >= 32)
                    or coverage_bytes > 20000
                ):
                    self.problem = "access_budget_exhausted"
                    raise AccessBudgetExceeded()
                self.calls += 1
                try:
                    result = await self._read(normalized, start_line, max_lines)
                except OSError:
                    remaining_execution(self.row)
                    self.problem = "file_unavailable"
                    raise AccessBudgetExceeded() from None
                if "error" in result:
                    self.problem = "file_unavailable"
                    raise AccessBudgetExceeded()
                if not self.open:
                    raise VerificationContractError("Verification access has closed")
                digest = result["sha256"]
                if normalized in self.hashes and self.hashes[normalized] != digest:
                    self.problem = "workspace_changed"
                self.hashes[normalized] = digest
                content, truncated = limit_output(
                    result["content"],
                    self.scope.settings.output_bytes,
                    self.scope.settings.output_lines,
                )
                # End line reflects the actual returned lines, including byte limits.
                count = min(max_lines, max(0, result["total_lines"] - start_line + 1))
                if truncated:
                    count = min(count, max(0, len(content.splitlines()) - 1))
                observation = FileObservation(
                    path=normalized,
                    sha256=digest,
                    start_line=start_line,
                    end_line=start_line + count - 1,
                    truncated=result["truncated"] or truncated,
                )
                self.observations.append(observation)
                return VerificationFile(
                    content,
                    digest,
                    start_line,
                    observation.end_line,
                    observation.truncated,
                )
            except (VerificationContractError, ValidationError) as exc:
                self.violation = VerificationContractError(
                    "Verification access contract violated"
                )
                raise self.violation from exc
            except BaseException as exc:
                if is_control(exc) and not isinstance(exc, asyncio.CancelledError):
                    self.failure = exc
                raise

    async def recheck(self):
        for path, digest in self.hashes.items():
            try:
                result = await self._read(path, 1, 1)
            except Exception as exc:
                remaining_execution(self.row)
                if is_control(exc) and not isinstance(exc, TimeoutError):
                    raise
                self.problem = "file_recheck_failed"
                continue
            consistent = "error" not in result and result.get("sha256") == digest
            if not consistent:
                self.problem = "workspace_changed"
            self.observations = [
                f.model_copy(update={"consistent": consistent and f.sha256 == digest})
                if f.path == path
                else f
                for f in self.observations
            ]


def message_source_eligible(row):
    mode = row.get("submitted_binding", {}).get("verification", {}).get("mode")
    if row["status"] != "completed":
        return False
    if mode != "required":
        return True
    try:
        report = validate_publication(
            row.get("verification"), row, row.get("answer"), row.get("final_checkpoint")
        )
        return report["status"] == "passed" and report["final_candidate"]
    except (ValueError, KeyError, TypeError):
        return False


def validate_publication(report, row, answer, checkpoint):
    config = row["submitted_binding"].get("verification")
    if report is None:
        if config:
            raise ValueError("Enabled verification requires a report for completion")
        return None
    parsed = VerificationReport.model_validate_json(json.dumps(report))
    if (
        not config
        or parsed.mode != config["mode"]
        or parsed.source_run_id != str(row["id"])
        or parsed.binding_fingerprint != row["submitted_binding"]["fingerprint"]
        or parsed.extension_version != config["extension_version"]
        or answer is None
        or parsed.candidate_sha256 != hashlib.sha256(answer.encode()).hexdigest()
        or parsed.checkpoint != CheckpointReference.from_config(checkpoint)
    ):
        raise ValueError("Verification report publication binding mismatch")
    return parsed.model_dump(mode="json")


def publication_matches(current, result):
    expected = (
        "cancelled"
        if result.get("cancelled")
        else "failed"
        if result.get("error_code")
        else "completed"
    )
    return (
        current["status"] == expected
        and current.get("verification") == result.get("verification")
        and current.get("answer")
        == (result.get("answer") if expected == "completed" else None)
        and current.get("final_checkpoint")
        == (result.get("checkpoint") if expected == "completed" else None)
        and current.get("error_code")
        == (result.get("error_code") if expected == "failed" else None)
        and current.get("failure_info")
        == (result.get("failure_info") if expected == "failed" else None)
    )


async def verify(row, registry, store, context, scope, answer, checkpoint):
    extension = registry.extensions.get(row["profile"])
    if extension is None or extension.verification_mode is None:
        return None, None
    reference = CheckpointReference.from_config(checkpoint)
    started = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    log.info(
        "verification_started",
        extra={"run_id": row["id"], "verification_mode": extension.verification_mode},
    )
    final_candidate = await store.verification_candidate(
        row["id"], row.get("_delivered", ())
    )
    from agent_service.checks import CheckAccess

    access = FileAccess(row, registry, context, scope)
    checks = CheckAccess(access, extension)
    error = None
    if not final_candidate:
        result = VerificationResult(status="not_checked", reason="stage_report")
    elif extension.verifier is None:
        result = VerificationResult(
            status="not_checked", reason="verifier_not_registered"
        )
    else:
        verifier_context = VerificationContext(
            str(row["id"]),
            str(row["task_id"]) if row.get("task_id") else None,
            row["profile"],
            context.workspace_id,
            context.workspace_identity,
            context.binding_fingerprint,
            extension.version,
            answer,
            reference,
            lambda: remaining_execution(row),
            access.read,
            checks.run,
        )
        timer = asyncio.timeout(
            min(extension.verification_timeout, remaining_execution(row))
        )
        result = VerificationResult(status="inconclusive")
        try:
            async with timer:
                value = await extension.verifier(verifier_context, answer, reference)
                if access.failure:
                    raise access.failure
                if type(value) is not VerificationResult:
                    raise VerificationContractError(
                        "Verifier must return VerificationResult"
                    )
                if set(value.__dict__) - set(VerificationResult.model_fields):
                    raise VerificationContractError(
                        "Verifier returned controller metadata"
                    )
                # Revalidate even if trusted code used model_construct/model_copy.
                try:
                    result = VerificationResult.model_validate_json(
                        value.model_dump_json()
                    )
                except (ValueError, TypeError, RecursionError) as exc:
                    raise VerificationContractError(
                        "Invalid verifier result serialization"
                    ) from exc
                result = checks.enforce(result)
                async with access.lock:
                    await access.recheck()
                if access.violation:
                    raise access.violation
                if access.problem:
                    result = VerificationResult(
                        status="inconclusive",
                        summary=result.summary,
                        checks=result.checks,
                        reason=access.problem,
                    )
            # Trusted code can accidentally swallow cancellation or block the loop.
            # Neither behavior extends the controller's verification allowance.
            if timer.expired() or asyncio.get_running_loop().time() >= timer.when():
                result = VerificationResult(
                    status="inconclusive",
                    reason="timeout",
                    summary=result.summary,
                    checks=result.checks,
                )
        except BaseException as exc:
            remaining_execution(row)
            if isinstance(exc, TimeoutError) and timer.expired():
                result = VerificationResult(
                    status="inconclusive",
                    reason="timeout",
                    summary=result.summary,
                    checks=result.checks,
                )
            elif isinstance(exc, AccessBudgetExceeded):
                result = VerificationResult(
                    status="inconclusive",
                    reason=access.problem or "access_budget_exhausted",
                )
            elif isinstance(exc, (VerificationContractError, ValidationError)):
                result = VerificationResult(
                    status="inconclusive", reason="contract_error"
                )
                error = "VERIFICATION_CONTRACT_ERROR"
            elif isinstance(exc, TimeoutError) and not isinstance(
                exc, BaseExceptionGroup
            ):
                # A verifier's own TimeoutError is an unexpected ordinary exception.
                result = VerificationResult(
                    status="inconclusive", reason="execution_failed"
                )
                error = "VERIFICATION_EXECUTION_FAILED"
            elif is_control(exc) or not isinstance(exc, Exception):
                if isinstance(control_leaf(exc), asyncio.CancelledError):
                    raise control_leaf(exc) from None
                raise
            else:
                result = VerificationResult(
                    status="inconclusive", reason="execution_failed"
                )
                error = "VERIFICATION_EXECUTION_FAILED"
            if error:
                log.error(
                    "verification_failed",
                    exc_info=True,
                    extra={"run_id": row["id"], "error_code": error},
                )
        finally:
            checks.close()
    remaining_execution(row)
    try:
        report = VerificationReport(
            **result.model_dump(),
            mode=extension.verification_mode,
            final_candidate=final_candidate,
            source_run_id=str(row["id"]),
            checkpoint=reference,
            candidate_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            binding_fingerprint=row["submitted_binding"]["fingerprint"],
            extension_version=extension.version,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            files=tuple(access.observations),
        )
    except ValidationError:
        # Controller coverage/serialization capacity is not a verifier result error.
        # Freeze all retained coverage, but remove optional explanations to fit.
        report = VerificationReport(
            status="inconclusive",
            reason="report_budget_exhausted",
            mode=extension.verification_mode,
            final_candidate=final_candidate,
            source_run_id=str(row["id"]),
            checkpoint=reference,
            candidate_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            binding_fingerprint=row["submitted_binding"]["fingerprint"],
            extension_version=extension.version,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            files=tuple(access.observations),
        )
    if extension.verification_mode == "required" and final_candidate and not error:
        error = (
            None
            if report.status == "passed"
            else "VERIFICATION_FAILED"
            if report.status == "failed"
            else "VERIFICATION_INCONCLUSIVE"
        )
    log.info(
        "verification_finished",
        extra={
            "run_id": row["id"],
            "verification_mode": report.mode,
            "status": report.status,
            "duration_seconds": time.monotonic() - began,
            "error_code": error,
        },
    )
    return report.model_dump(mode="json"), error
