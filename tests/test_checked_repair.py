"""Trusted fixed Python gate: registration, real execution, boundaries and publication."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path

import httpx
import psycopg

from agent_service.app import app
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.checks import CheckAccess, first_n_check, fixed_verifier, parse_receipt
from agent_service.execution import DockerClient, ExecutionScope, check_environment
from agent_service.execution_errors import EnvironmentUnavailable
from agent_service.policy import PermissionDenied
from agent_service.profiles import Extension, Registry, RuntimeFile, Profile, Workspace
from agent_service.verification import VerificationContractError, VerificationResult
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_code_profiles import ROOT
from tests.test_core import execute
from tests.test_profiles_policy import call

GOOD = 'def first_n(items, n):\n    if n < 0:\n        raise ValueError("negative")\n    return items[:n]\n'
BUG = 'def first_n(items, n):\n    return items[:n + 1]\n'


def load_checked(tmp_path, monkeypatch, *, extension=None, validators=None):
    root = tmp_path / 'workspace'
    root.mkdir(exist_ok=True)
    (root / 'sample.py').write_text(GOOD)
    image = os.environ.get('AGENT_TEST_EXECUTION_IMAGE', 'sha256:' + '0' * 64)
    text = (ROOT / 'config/runtime.code.checked.example.toml').read_text()
    for old, new in {'/ABSOLUTE/PATH/TO/project':str(root), '/ABSOLUTE/PATH/TO/agent/project_skills':str(ROOT / 'project_skills'),
                     'sha256:REPLACE_WITH_64_HEX_DIGITS': image}.items():
        assert text.count(old) == 1
        text = text.replace(old, new)
    path = tmp_path / 'checked.toml'
    path.write_text(text)
    monkeypatch.setenv('AGENT_RUNTIME_CONFIG', str(path))
    trusted = {'validators':validators} if validators else {}
    if extension:
        trusted['extensions'] = {'code-fix-checked':extension}
    return Registry.from_env(**trusted), root


def install(r, registry):
    r.registry = r.store.registry = registry


def receipt(**updates):
    definition = first_n_check()
    return {'protocol':1, 'check_id':definition.id, 'version':definition.version,
            'assets':dict(definition.assets), 'source_sha256':'a' * 64,
            'status':'passed', 'reason':'tests_passed', 'executed':6,
            'failures':0, 'errors':0, 'skipped':0, 'unexpected_successes':0,
            'test_ids':list(definition.test_ids), 'diagnostic':'', **updates}


def test_checked_registration_binding_and_ordinary_compatibility(tmp_path, monkeypatch):
    checked, root = load_checked(tmp_path, monkeypatch)
    binding = checked.bind('code-fix-checked', 'project')
    assert binding == Registry.from_env().bind('code-fix-checked', 'project')
    assert binding['verification']['mode'] == 'required'
    assert binding['checks'] == [first_n_check().manifest()]
    assert 'run_check' not in checked.tools
    assert checked.profiles['code-fix-checked'].tools == checked.profiles['code-fix'].tools
    assert checked.profiles['code-fix-checked'].skills == ('code-fix',)
    assert not checked.extensions['code-fix-checked'].no_progress_enabled
    config = RuntimeFile(version=1, execution=checked.execution, workspaces=[Workspace(id='project', root=str(root))],
                         profiles=[p for name,p in checked.profiles.items() if name.startswith('code-') and name != 'code-fix-checked'],
                         skills={'directories':(str(ROOT / 'project_skills'),)})
    ordinary = Registry(config)
    for name in ordinary.profiles:
        assert ordinary.bind(name, 'project') == checked.bind(name, 'project')
        assert 'checks' not in ordinary.bind(name, 'project')
    assert replace(checked.extensions['code-fix-checked'], no_progress_enabled=True).no_progress_enabled
    with pytest.raises(ValueError):
        checked.bind('code-fix-checked', 'other')


@pytest.mark.parametrize('kind', ['extension', 'definitions', 'shell', 'read_file', 'workspace', 'invalid_definition', 'image'])
def test_checked_registration_fails_closed(tmp_path, kind):
    from agent_service.execution_config import ExecutionSettings

    definition = first_n_check()
    tools = ('shell', 'read_file')
    if kind in tools:
        tools = tuple(t for t in tools if t != kind)
    extension = Extension(verification_mode='required', verifier=fixed_verifier,
                          checks=() if kind == 'definitions' else (definition,))
    if kind == 'invalid_definition':
        with pytest.raises(ValueError):
            replace(extension, checks=(definition.model_copy(update={'argv':('echo', 'passed')}),))
        return
    with pytest.raises(ValueError):
        Registry(RuntimeFile(version=1, execution=ExecutionSettings(image=None if kind == 'image' else 'sha256:' + '0' * 64), profiles=[Profile(id='code-fix-checked', tools=tools, requires_workspace=True)],
                             workspaces=[] if kind == 'workspace' else [Workspace(id='project', root=str(tmp_path))]),
                 extensions={} if kind == 'extension' else {'code-fix-checked':extension})


@pytest.mark.parametrize('updates', [
    {'protocol':True}, {'executed':True}, {'failures':False}, {'skipped':'0'},
    {'executed':0}, {'executed':7}, {'failures':7}, {'failures':1},
    {'check_id':'foreign'}, {'version':'2'}, {'assets':{}}, {'extra':'forged'},
    {'test_ids':['first_n.zero'] * 6}, {'status':'failed'}, {'unexpected_successes':1},
    {'source_sha256':'bad'}, {'diagnostic':'x' * 2049},
])
def test_invalid_receipt_contract(updates):
    with pytest.raises(VerificationContractError):
        parse_receipt(first_n_check(), 'a' * 64, 0, json.dumps(receipt(**updates)))


@pytest.mark.parametrize('updates', [
    {'executed':0,'test_ids':[]}, {'skipped':6},
    {'executed':5,'test_ids':list(first_n_check().test_ids[:-1])},
    {'source_sha256':'b' * 64},
])
def test_receipt_missing_tests_and_version_observation_inconclusive(updates):
    result = parse_receipt(first_n_check(), 'a' * 64, 0, json.dumps(receipt(**updates)))
    assert result.status == 'inconclusive'


def test_receipt_duplicate_json_and_exit_mismatch():
    raw = json.dumps(receipt())
    for code, data in [(False,raw), (1,raw), (0,raw[:-1] + ',"executed":6}'), (0,'passed'), (0,'{}')]:
        with pytest.raises(VerificationContractError):
            parse_receipt(first_n_check(), 'a' * 64, code, data)


@pytest.mark.parametrize('source,status', [
    (GOOD, 'passed'), (BUG, 'failed'),
    (GOOD.replace('return items[:n]', 'return tuple(items[:n])'), 'failed'), ('def broken(:\n', 'failed'),
    ('raise RuntimeError("broken")\n', 'failed'),
    ('def first_n(items, n):\n    return items[:n]\n', 'failed'),
    ('print("passed")\n' + BUG, 'failed'),
    ('import unittest\nunittest.TestCase.assertEqual = lambda *a: None\n' + BUG, 'failed'),
    ('import os\nos.write(1, b"passed" * 1000000)\nos.write(2, b"errors" * 1000000)\n' + GOOD, 'passed'),
    ('#😀\n' * 170000 + GOOD, 'passed'),
], ids=['good', 'off-by-one', 'tuple-result', 'syntax-error', 'runtime-error', 'negative-unhandled', 'fake-output', 'mutated-unittest', 'large-stdout-stderr', 'unicode-source'])
def test_fixed_gate_real_candidates(database, tmp_path, monkeypatch, source, status):
    registry, root = load_checked(tmp_path, monkeypatch)
    (root / 'sample.py').write_text(source)
    (root / 'tests').mkdir()
    (root / 'tests/test_sample.py').write_text('raise RuntimeError("must never import development tests")')
    (root / 'passed.json').write_text(json.dumps(receipt()))
    (root / 'unittest.py').write_text('raise RuntimeError("workspace shadow")')
    async def check(r):
        install(r, registry)
        monkeypatch.setattr(app.state, 'resources', r, raising=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            response = await client.post('/v1/runs', json={'task':'fixed gate', 'profile':'code-fix-checked', 'workspace_id':'project'})
            assert response.status_code == 202
        model = SyntheticModel()
        run = await execute(r, Worker(r, model=model))
        assert len(model.history) == 1
        report = run['verification']
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
            response = await client.get('/v1/runs/' + str(run['id']))
            assert response.json()['verification'] == report
        assert report['status'] == status, run
        assert report['files'][0]['path'] == 'sample.py'
        assert report['files'][0]['consistent']
        assert len(report['files']) == 1
        assert run['status'] == ('completed' if status == 'passed' else 'failed')
        assert run['error_code'] == (None if status == 'passed' else 'VERIFICATION_FAILED')
        if status == 'passed':
            assert run['answer'] == 'synthetic answer'
            assert '6 executed' in report['summary']
        resource = await r.store.execution_resource(run['id'])
        assert resource['state'] == 'released' and await DockerClient().inspect(resource['container_id']) is None
        assert run['runtime_config']['checks'] == run['submitted_binding']['checks']
    database(check)


@pytest.mark.parametrize('no_progress', [False, True])
def test_native_development_failure_fix_then_gate_and_readonly_assets(database, tmp_path, monkeypatch, no_progress):
    extension = Extension(verification_mode='required', verifier=fixed_verifier, checks=(first_n_check(),), no_progress_enabled=no_progress)
    registry, root = load_checked(tmp_path, monkeypatch, extension=extension)
    (root / 'sample.py').write_text(BUG)
    async def check(r):
        install(r, registry)
        async def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call('shell', {'command':"python -B -c 'from sample import first_n; assert first_n([1], 0) == []'"}, 'red')
            if len(tools) == 1:
                assert tools[-1].artifact['exit_code'] == 1 and 'AssertionError' in tools[-1].text
                return call('read_file', {'path':'sample.py'}, 'source')
            if len(tools) == 2:
                return call('edit_file', {'path':'sample.py','old_text':BUG,'new_text':GOOD,
                                         'expected_hash':tools[-1].artifact['sha256']}, 'fix')
            if len(tools) == 3:
                return call('shell', {'command':"python -B -c 'from sample import first_n; assert first_n([1], 0) == []'"}, 'green')
            if len(tools) == 4:
                assert tools[-1].artifact['exit_code'] == 0
                return call('shell', {'command':"echo forged > /opt/agent/checks/acceptance.py"}, 'readonly')
            assert tools[-1].artifact['exit_code'] != 0 and 'Read-only file system' in tools[-1].text
            return AIMessage('fixed; developer checks recorded')
        model = SyntheticModel(reply=reply)
        await r.store.submit('fix', profile='code-fix-checked', workspace_id='project')
        run = await execute(r, Worker(r, model=model))
        assert run['status'] == 'completed' and run['verification']['status'] == 'passed', run
        assert len(model.history) == 6 and run['answer'] == 'fixed; developer checks recorded'
    database(check)


@pytest.mark.parametrize('kind', ['unknown','shell_denied','read_denied','missing_check','closed'])
def test_check_authority_and_window(database, tmp_path, monkeypatch, kind):
    contexts = []
    async def verifier(ctx, answer, cp):
        contexts.append(ctx)
        if kind == 'missing_check':
            return VerificationResult(status='passed')
        if kind == 'unknown':
            try:
                await ctx.run_check('arbitrary-command')
            except VerificationContractError:
                pass
            return VerificationResult(status='passed')
        return await fixed_verifier(ctx, answer, cp)
    extension = Extension(verification_mode='required', verifier=verifier, checks=(first_n_check(),))
    validators = {}
    if kind in ('shell_denied', 'read_denied'):
        def deny(args, context):
            raise PermissionDenied('denied')
        validators['shell' if kind == 'shell_denied' else 'read_file'] = deny
    registry, _ = load_checked(tmp_path, monkeypatch, extension=extension, validators=validators)
    async def check(r):
        install(r, registry)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        run = await execute(r)
        expected = 'completed' if kind == 'closed' else 'failed'
        assert run['status'] == expected, run
        if kind != 'closed':
            assert run['error_code'] == ('VERIFICATION_INCONCLUSIVE' if kind == 'missing_check' else 'VERIFICATION_CONTRACT_ERROR')
        with pytest.raises(VerificationContractError, match='closed'):
            await contexts[0].run_check('python-first-n')
    database(check)


@pytest.mark.parametrize('kind', ['missing','changes','load_mismatch','recheck','timeout','descendants'])
def test_file_and_process_bounds(database, tmp_path, monkeypatch, kind):
    registry, root = load_checked(tmp_path, monkeypatch)
    if kind == 'missing':
        (root / 'sample.py').unlink()
    if kind == 'changes':
        (root / 'sample.py').write_text("from pathlib import Path\nPath('/workspace/sample.py').write_text('# changed')\n" + GOOD)
    if kind == 'timeout':
        (root / 'sample.py').write_text('while True:\n    pass\n')
    if kind == 'descendants':
        (root / 'sample.py').write_text("import subprocess\nsubprocess.Popen(['python', '-I', '-c', 'import time; time.sleep(60)'], start_new_session=True)\n" + GOOD)
    original = ExecutionScope.file_call
    reads = []
    async def observe(self, name, args):
        result = await original(self, name, args)
        if name == 'read_file':
            reads.append(args)
            if kind == 'load_mismatch' and len(reads) == 1:
                (root / 'sample.py').write_text('# changed\n' + GOOD)
            if kind == 'recheck' and len(reads) == 2:
                raise OSError('recheck unavailable')
        return result
    monkeypatch.setattr(ExecutionScope, 'file_call', observe)
    async def check(r):
        install(r, registry)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        run = await execute(r)
        assert run['verification']['status'] == ('passed' if kind == 'descendants' else 'inconclusive'), run
        assert run['status'] == ('completed' if kind == 'descendants' else 'failed')
        assert (await r.store.execution_resource(run['id']))['state'] == 'released'
    database(check)


def test_checked_commit_ack_loss_does_not_replay(database, tmp_path, monkeypatch):
    registry, _ = load_checked(tmp_path, monkeypatch)
    check_calls = []
    original_check = CheckAccess.run
    async def counted(self, check_id):
        check_calls.append(check_id)
        return await original_check(self, check_id)
    monkeypatch.setattr(CheckAccess, 'run', counted)
    async def check(r):
        install(r, registry)
        original_finish = r.store.finish
        finish_calls = []
        async def finish(*args, **kwargs):
            result = await original_finish(*args, **kwargs)
            finish_calls.append(result)
            if len(finish_calls) == 1:
                raise psycopg.OperationalError('ack lost')
            return result
        monkeypatch.setattr(r.store, 'finish', finish)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        model = SyntheticModel()
        run = await execute(r, Worker(r, model=model))
        assert run['status'] == 'completed', run
        assert len(model.history) == 1 and check_calls == ['python-first-n']
        assert len(finish_calls) == 1
    database(check)


def test_check_capability_required_only_for_checked_profiles(database, tmp_path, monkeypatch):
    registry, _ = load_checked(tmp_path, monkeypatch)
    original = DockerClient.call
    async def legacy(self, *args, **kwargs):
        code, raw, err = await original(self, *args, **kwargs)
        if args[:2] == ('image', 'inspect'):
            images = json.loads(raw)
            labels = images[0]['Config']['Labels']
            labels.pop('io.agent-framework.checks', None)
            labels.pop('io.agent-framework.checks-assets', None)
            raw = json.dumps(images).encode()
        return code, raw, err
    monkeypatch.setattr(DockerClient, 'call', legacy)
    async def check(r):
        await check_environment(registry.execution)
        with pytest.raises(EnvironmentUnavailable, match='check capability'):
            await check_environment(registry.execution, checks=registry.check_definitions())
        install(r, registry)
        model = SyntheticModel()
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        run = await execute(r, Worker(r, model=model))
        assert run['status'] == 'failed' and not model.history
        assert run['error_code'] == 'EXECUTION_ENVIRONMENT_UNAVAILABLE'
    database(check)


def test_stage_three_binding_snapshot(tmp_path, monkeypatch):
    registry, _ = load_checked(tmp_path, monkeypatch)
    # Snapshot was captured after the phase-three Skill update, before check
    # support. Image values are held constant for this compatibility assertion.
    config_path = Path(os.environ['AGENT_RUNTIME_CONFIG'])
    text = config_path.read_text().replace(registry.execution.image, 'sha256:' + '0' * 64)
    config_path.write_text(text)
    current = Registry.from_env()
    baseline = json.loads((ROOT / 'tests/fixtures/code_phase3_bindings.json').read_text())
    for name, expected in baseline.items():
        actual = current.bind(name, 'project')
        assert {k:v for k,v in actual.items() if k not in ('workspace_identity','fingerprint')} == expected
        from agent_service.profiles import fingerprint
        assert actual['fingerprint'] == fingerprint({**expected, 'workspace_identity':actual['workspace_identity']})


@pytest.mark.parametrize('kind', ['stage', 'cleanup', 'cancel', 'lease'])
def test_checked_publication_controls_and_stage(database, tmp_path, monkeypatch, kind):
    registry, _ = load_checked(tmp_path, monkeypatch)
    calls = []
    original_check = CheckAccess.run
    async def counted(self, check_id):
        calls.append(check_id)
        return await original_check(self, check_id)
    monkeypatch.setattr(CheckAccess, 'run', counted)
    async def check(r):
        install(r, registry)
        if kind == 'stage':
            async def stage(*args, **kwargs):
                return False
            monkeypatch.setattr(r.store, 'verification_candidate', stage)
        original = ExecutionScope.close
        async def close(scope, *args):
            result = await original(scope, *args)
            if kind == 'cleanup':
                return False
            if kind == 'cancel':
                await r.store.cancel(scope.row['id'])
            if kind == 'lease':
                async with r.store.pool.connection() as conn:
                    await conn.execute("UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s", (scope.row['id'],))
            return result
        monkeypatch.setattr(ExecutionScope, 'close', close)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        model = SyntheticModel()
        run = await execute(r, Worker(r, model=model))
        assert len(model.history) == 1
        if kind == 'stage':
            assert not calls and run['verification']['reason'] == 'stage_report'
            assert run['verification']['status'] == 'not_checked'
        else:
            assert calls == ['python-first-n'] and run['status'] in ('failed','cancelled')
            assert run['answer'] is None
    database(check)


@pytest.mark.parametrize('kind', ['shared_timer', 'container_lost', 'pending_check'])
def test_checked_verification_window_controls(database, tmp_path, monkeypatch, kind):
    events = []
    async def verifier(ctx, answer, cp):
        if kind == 'container_lost':
            raise EnvironmentUnavailable('container gone')
        if kind == 'pending_check':
            task = asyncio.create_task(ctx.run_check('python-first-n'))
            events.append(task)
            await asyncio.sleep(0)
            return VerificationResult(status='passed')
        result = await fixed_verifier(ctx, answer, cp)
        await asyncio.sleep(1)
        return result
    extension = Extension(verification_mode='required', verifier=verifier, checks=(first_n_check(),),
                          verification_timeout=0.7 if kind == 'shared_timer' else 10.0)
    registry, _ = load_checked(tmp_path, monkeypatch, extension=extension)
    async def check(r):
        install(r, registry)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        run = await execute(r)
        assert run['status'] == 'failed', run
        if kind == 'shared_timer':
            assert run['verification']['status'] == 'inconclusive' and run['verification']['reason'] == 'timeout'
        if kind == 'pending_check':
            assert run['verification']['status'] == 'inconclusive'
            assert all(task.done() for task in events)
        assert (await r.store.execution_resource(run['id']))['state'] == 'released'
    database(check)


def test_existing_but_disallowed_workspace(tmp_path, monkeypatch):
    registry, _ = load_checked(tmp_path, monkeypatch)
    path = Path(os.environ['AGENT_RUNTIME_CONFIG'])
    path.write_text(path.read_text() + '\n[[workspaces]]\nid = "other"\nroot = ' + json.dumps(str(tmp_path)) + '\n')
    registry = Registry.from_env()
    assert registry.bind('code-fix', 'other')
    with pytest.raises(ValueError, match='not allowed'):
        registry.bind('code-fix-checked', 'other')


def test_delivered_config_role_preflight(tmp_path, monkeypatch):
    from db.config_check import main
    import sys

    registry, _ = load_checked(tmp_path, monkeypatch)
    seen = []
    factory = Registry.from_env
    def registered(cls, **trusted):
        value = factory(**trusted)
        seen.append(value.bind('code-fix-checked', 'project'))
        return value
    monkeypatch.setattr(Registry, 'from_env', classmethod(registered))
    async def forbidden(*args, **kwargs):
        raise AssertionError('API config must not access Docker')
    monkeypatch.setattr(sys, 'argv', ['config_check', '--role', 'agent-api'])
    with monkeypatch.context() as guard:
        guard.setattr(DockerClient, 'call', forbidden)
        main()
    monkeypatch.setattr(sys, 'argv', ['config_check', '--role', 'agent-worker'])
    main()
    assert seen == [registry.bind('code-fix-checked', 'project')] * 2


def test_readonly_profile_cannot_borrow_registered_check(database, tmp_path, monkeypatch):
    registry, _ = load_checked(tmp_path, monkeypatch)
    registry = Registry.from_env(extensions={'code-analyze':Extension(verification_mode='required', verifier=fixed_verifier)})
    async def check(r):
        install(r, registry)
        await r.store.submit('test', profile='code-analyze', workspace_id='project')
        run = await execute(r)
        assert run['status'] == 'failed' and run['error_code'] == 'VERIFICATION_CONTRACT_ERROR'
        assert not run['verification']['files']
    database(check)


@pytest.mark.parametrize('kind', ['container_lost', 'run_deadline'])
def test_check_run_controls_propagate(database, tmp_path, monkeypatch, kind):
    from datetime import datetime, timedelta, timezone
    registry, root = load_checked(tmp_path, monkeypatch)
    if kind == 'run_deadline':
        (root / 'sample.py').write_text('while True:\n    pass\n')
    original = DockerClient.call
    async def lost(self, *args, **kwargs):
        if kind == 'container_lost' and '/opt/agent/checks/runner.py' in args:
            await original(self, 'rm', '--force', args[4])
        return await original(self, *args, **kwargs)
    monkeypatch.setattr(DockerClient, 'call', lost)
    async def check(r):
        install(r, registry)
        await r.store.submit('test', profile='code-fix-checked', workspace_id='project')
        model = SyntheticModel()
        worker = Worker(r, model=model)
        row = await r.store.claim(worker.id)
        if kind == 'run_deadline':
            row['deadline_at'] = datetime.now(timezone.utc) + timedelta(seconds=registry.execution.cleanup_timeout + registry.reliability.publication_reserve + 0.8)
        await worker.execute(row)
        run = await r.store.get(row['id'])
        assert run['status'] == 'failed' and run['answer'] is None
        assert run['error_code'] == ('EXECUTION_ENVIRONMENT_LOST' if kind == 'container_lost' else 'RUN_TIMEOUT'), run
        assert len(model.history) <= 1
        assert (await r.store.execution_resource(row['id']))['state'] == 'released'
    database(check)
