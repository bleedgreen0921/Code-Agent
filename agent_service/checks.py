"""Trusted check declarations and controller-only dispatch; never an Agent tool."""
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import shlex
import time

from pydantic import Field, model_validator

from agent_service.verification import (
    Strict, VerificationContractError, VerificationResult, VerificationCheck,
)
from agent_service.failure import is_control
from agent_service.execution_errors import EnvironmentLost, EnvironmentUnavailable

log = logging.getLogger(__name__)
ASSET_ROOT = Path(__file__).resolve().parents[1] / 'execution/checks'
ASSET_NAMES = ('runner.py', 'acceptance.py', 'candidate.py')
CAPABILITY = 'python-repair-v1'
ARGV = ('/usr/local/bin/python', '-I', '/opt/agent/checks/runner.py')
IDS = tuple('first_n.' + n for n in ('zero', 'normal', 'empty', 'over_length', 'negative', 'no_mutation'))


def asset_hashes():
    return {name: hashlib.sha256((ASSET_ROOT / name).read_bytes()).hexdigest() for name in ASSET_NAMES}


def asset_fingerprint(assets):
    return hashlib.sha256(json.dumps(dict(assets), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class CheckDefinition(Strict):
    id: str = Field(pattern=r'^[a-z][a-z0-9-]{0,63}$')
    version: str = Field(min_length=1, max_length=64)
    argv: tuple[str, ...]
    cwd: str
    workspaces: tuple[str, ...] = Field(min_length=1)
    test_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    source: str
    assets: tuple[tuple[str, str], ...]
    capability: str

    def manifest(self):
        return self.model_dump(mode='json')

    def validate_registration(self):
        CheckDefinition.model_validate(self.model_dump())
        # First implementation intentionally supports this fixed offline gate only.
        if (self.argv != ARGV or self.cwd != '/workspace' or self.source != 'sample.py'
                or self.id != 'python-first-n' or self.version != '1'
                or self.test_ids != IDS or self.workspaces != ('project',)
                or any(not w or '\x00' in w for w in self.workspaces)
                or self.capability != CAPABILITY or dict(self.assets) != asset_hashes()
                or len(self.assets) != len(ASSET_NAMES)):
            raise ValueError('Invalid fixed check definition or assets')


def first_n_check():
    return CheckDefinition(id='python-first-n', version='1', argv=ARGV,
                           cwd='/workspace', workspaces=('project',), test_ids=IDS,
                           source='sample.py', assets=tuple(asset_hashes().items()), capability=CAPABILITY)


def validate_image_checks(image, definitions):
    for definition in definitions:
        labels = image.get('Config', {}).get('Labels') or {}
        if (labels.get('io.agent-framework.checks') != definition['capability']
                or labels.get('io.agent-framework.checks-assets') != asset_fingerprint(definition['assets'])):
            raise EnvironmentUnavailable('Execution image lacks required fixed check capability/assets')


class Receipt(Strict):
    protocol: int = Field(ge=1, le=1)
    check_id: str
    version: str
    assets: dict[str, str]
    source_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    status: str = Field(pattern=r'^(passed|failed|inconclusive)$')
    reason: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    executed: int = Field(ge=0, le=32)
    failures: int = Field(ge=0, le=32)
    errors: int = Field(ge=0, le=32)
    skipped: int = Field(ge=0, le=32)
    unexpected_successes: int = Field(ge=0, le=32)
    test_ids: list[str] = Field(max_length=32)
    diagnostic: str = Field(max_length=2048)


class CheckExecutionResult(Strict):
    check_id: str = Field(min_length=1, max_length=64)
    status: str = Field(pattern=r'^(passed|failed|inconclusive)$')
    reason: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    exit_code: int | None = None
    executed: int = Field(default=0, ge=0, le=32)
    failures: int = Field(default=0, ge=0, le=32)
    errors: int = Field(default=0, ge=0, le=32)
    skipped: int = Field(default=0, ge=0, le=32)
    diagnostic: str = Field(default='', max_length=2048)

    @model_validator(mode='after')
    def bounded(self):
        if '\x00' in self.diagnostic or len(self.diagnostic.encode()) > 2048:
            raise ValueError('Check diagnostic exceeds 2 KiB')
        return self


def parse_receipt(definition, digest, code, raw):
    from pydantic import ValidationError
    try:
        # Duplicate JSON keys are rejected, including nested receipt objects.
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('Duplicate receipt key')
                result[key] = value
            return result
        value = json.loads(raw, object_pairs_hook=unique)
        receipt = Receipt.model_validate(value)
        if (type(code) is not int or receipt.check_id != definition.id or receipt.version != definition.version
                or receipt.assets != dict(definition.assets)
                or receipt.executed != len(receipt.test_ids)
                or len(set(receipt.test_ids)) != len(receipt.test_ids)
                or receipt.failures + receipt.errors + receipt.skipped + receipt.unexpected_successes > receipt.executed
                or code != {'passed': 0, 'failed': 1, 'inconclusive': 2}[receipt.status]
                or receipt.status == 'passed' and (receipt.failures or receipt.errors or receipt.unexpected_successes)
                or receipt.status == 'failed' and not (receipt.failures or receipt.errors or receipt.unexpected_successes)):
            raise ValueError('Receipt association/count/exit mismatch')
    except (ValueError, TypeError, ValidationError) as exc:
        raise VerificationContractError('Invalid fixed check receipt') from exc
    status, reason = receipt.status, receipt.reason
    if receipt.source_sha256 != digest:
        status, reason = 'inconclusive', 'source_mismatch'
    elif receipt.status != 'inconclusive' and (not receipt.executed or receipt.skipped
          or set(receipt.test_ids) != set(definition.test_ids)):
        status, reason = 'inconclusive', 'required_tests_missing'
    diagnostic = receipt.diagnostic.replace('\x00', '')
    diagnostic = diagnostic.encode()[:2048].decode('utf-8', 'ignore')
    return CheckExecutionResult(check_id=definition.id, status=status, reason=reason,
                                exit_code=code, executed=receipt.executed, failures=receipt.failures,
                                errors=receipt.errors, skipped=receipt.skipped, diagnostic=diagnostic)


async def fixed_verifier(context, answer, checkpoint):
    result = await context.run_check('python-first-n')
    return VerificationResult(status=result.status, reason=result.reason,
                              summary=f'{result.check_id}: {result.executed} executed, {result.failures} failures, {result.errors} errors, {result.skipped} skipped; sample.py only.',
                              checks=(VerificationCheck(status=result.status, explanation=f'{result.check_id}: {result.reason}'),))


class CheckAccess:
    def __init__(self, access, extension):
        self.access, self.extension = access, extension
        self.results = []
        self.tasks = set()

    async def run(self, check_id):
        task = asyncio.current_task()
        self.tasks.add(task)
        scope = self.access.scope
        if scope:
            scope.native_tasks.add(task)
        try:
            return await self._run(check_id)
        finally:
            self.tasks.discard(task)
            if scope:
                scope.native_tasks.discard(task)

    def close(self):
        self.access.open = False
        for task in tuple(self.tasks):
            task.cancel()

    async def _run(self, check_id):
        began = time.monotonic()
        access = self.access
        try:
            if not access.open:
                raise VerificationContractError('Verification access has closed')
            if type(check_id) is not str:
                raise VerificationContractError('Only a registered check ID is accepted')
            definitions = {d.id: d for d in self.extension.checks}
            if check_id not in definitions:
                raise VerificationContractError('Check is not registered for this Profile')
            definition = definitions[check_id]
            profile = access.registry.resolve(access.row['submitted_binding'])
            if (access.context.workspace_id not in definition.workspaces or access.scope is None
                    or not {'shell', 'read_file'} <= set(profile.tools)):
                raise VerificationContractError('Check workspace or permissions denied')
            from agent_service.policy import PermissionDenied
            validator = access.registry.validators.get('shell')
            args = access.registry.tools['shell'].tool_call_schema.model_validate({'command': shlex.join(definition.argv)})
            try:
                if validator and validator(args, access.context) is False:
                    raise VerificationContractError('Check Shell execution denied')
            except PermissionDenied:
                raise VerificationContractError('Check Shell execution denied') from None
            observed = await access.read(definition.source)
            scope = access.scope
            async with scope.lock:
                if not access.open:
                    raise VerificationContractError('Verification access has closed')
                await scope.healthy()
                if scope.native_tasks - self.tasks or scope.client.processes:
                    result = CheckExecutionResult(check_id=check_id, status='inconclusive', reason='resources_unsettled')
                else:
                    try:
                        code, raw, _ = await scope.client.call(
                            'exec', '-i', '--workdir', definition.cwd, scope.container_id, *definition.argv,
                            stdin=json.dumps({'sha256': observed.sha256}).encode(),
                            timeout=min(scope.remaining(), scope.settings.command_timeout), maximum=32769, check=False)
                    except (OSError, TimeoutError):
                        # The enclosing verification timer still owns the total budget.
                        result = CheckExecutionResult(check_id=check_id, status='inconclusive', reason='check_unavailable')
                    else:
                        await scope.healthy()
                        if code == 137:
                            raise EnvironmentLost('Fixed check process was killed')
                        if code in (126, 127) or not raw or len(raw) >= 32769:
                            result = CheckExecutionResult(check_id=check_id, status='inconclusive', reason='check_unavailable', exit_code=code)
                        else:
                            result = parse_receipt(definition, observed.sha256, code, raw)
            self.results.append(result)
            log.info('verification_check_finished', extra={'run_id': access.row['id'], 'check_id': check_id,
                     'duration_seconds': time.monotonic() - began, 'status': result.status,
                     'executed': result.executed, 'failures': result.failures, 'errors': result.errors, 'skipped': result.skipped})
            return result
        except VerificationContractError as exc:
            access.violation = exc
            raise
        except BaseException as exc:
            if is_control(exc) and not isinstance(exc, asyncio.CancelledError):
                access.failure = exc
            raise

    def enforce(self, result):
        # A verifier cannot assert passed while omitting a mandatory check, even
        # if it accidentally discards a failed/inconclusive result.
        if not self.extension.checks:
            return result
        missing = {d.id for d in self.extension.checks} - {r.check_id for r in self.results}
        bad = next((r for r in self.results if r.status != 'passed'), None)
        if missing or bad:
            return VerificationResult(status=bad.status if bad else 'inconclusive',
                                      reason=bad.reason if bad else 'required_check_not_executed',
                                      summary=result.summary,
                                      checks=tuple(VerificationCheck(status=r.status, explanation=f'{r.check_id}: {r.reason}') for r in self.results[-16:]))
        return result
