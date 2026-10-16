"""Calibrate three extracted AMR tasks or run a local-model repair pilot."""

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / '.data' / 'amr-pilot'
CASES = json.loads((HERE / 'cases.json').read_text())

REFERENCES = {
    'normalize_zero': '''import numpy as np
def l2_normalize(x, axis=-1):
    norms = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))
    return np.divide(x, norms, out=np.zeros_like(x, dtype=float), where=norms != 0)
''',
    'confusion_missing': '''import numpy as np
def calculate_confusion_matrix(Y, Y_hat, classes):
    counts = np.zeros((len(classes), len(classes)))
    if len(Y):
        np.add.at(counts, (np.argmax(Y, axis=1), np.argmax(Y_hat, axis=1)), 1)
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)
    right = np.trace(counts)
    return normalized, right, counts.sum() - right
''',
    'split_variable': '''import pickle
import numpy as np
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    mods = sorted({key[0] for key in data})
    snrs = sorted({key[1] for key in data})
    rng = np.random.RandomState(2016)
    arrays, labels, indices, offset = [], [], [[], [], []], 0
    for mod in mods:
        for snr in snrs:
            x = data[mod, snr]
            size = len(x)
            arrays.append(x)
            labels.extend([(mod, snr)] * size)
            selected = rng.permutation(np.arange(offset, offset + size))
            train, val = int(size * .6), int(size * .2)
            indices[0].extend(selected[:train])
            indices[1].extend(selected[train:train + val])
            indices[2].extend(selected[train + val:])
            offset += size
    x = np.vstack(arrays)
    partitions = []
    for selected in indices:
        rng.shuffle(selected)
        y = np.eye(len(mods))[[mods.index(labels[i][0]) for i in selected]]
        partitions.append((x[selected], y))
    return ((mods, snrs, labels), *partitions, tuple(indices))
''',
}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def grade(workspace, case, image):
    name = 'amr-pilot-' + uuid4().hex[:12]
    command = [
        'docker', 'run', '--rm', '--name', name, '--network', 'none',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--user', f'{os.getuid()}:{os.getgid()}', '--memory', '512m', '--cpus', '1',
        '--pids-limit', '64', '--tmpfs', '/tmp:rw,size=64m,mode=1777',
        '--mount', f'type=bind,src={workspace.resolve()},dst=/workspace,readonly',
        '--mount', f'type=bind,src={HERE / "evaluator.py"},dst=/checks/evaluator.py,readonly',
        image, case,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {'passed': False, 'infrastructure_error': result.stderr[-2000:], 'exit_code': result.returncode}
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {'passed': False, 'evaluation_error': 'No valid evaluator JSON', 'output': result.stdout[-2000:]}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'evaluation_error': 'Evaluation exceeded 30 seconds'}
    finally:
        subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=10)


def verify_fixtures():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    for case in provenance['cases']:
        path = HERE / 'fixtures' / case['id'] / 'candidate.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['fixture_sha256'], path


def calibrate(image):
    records = []
    for case in CASES:
        case_id = case['id']
        original = grade(HERE / 'fixtures' / case_id, case_id, image)
        folder = DATA / 'references' / case_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'candidate.py').write_text(REFERENCES[case_id])
        reference = grade(folder, case_id, image)
        valid = (
            len(original.get('checks', [])) == 2
            and original['checks'][0]['passed']
            and not original['checks'][1]['passed']
            and reference.get('passed') is True
        )
        records.append({'case': case_id, 'valid': valid, 'original': original, 'reference': reference})
        print(case_id, 'valid=' + str(valid), flush=True)
    write_json(DATA / 'calibration.json', records)
    if not all(row['valid'] for row in records):
        raise SystemExit('Calibration failed; inspect .data/amr-pilot/calibration.json')


async def episode(case, repeat, args, output):
    import httpx
    from datetime import datetime, timedelta, timezone
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from agent_service.config import Settings
    from agent_service.runtime import RunContext, build_agent, run_config

    folder = output / f'{case["id"]}-{repeat}'
    workspace = folder / 'workspace'
    workspace.mkdir(parents=True)
    candidate = workspace / 'candidate.py'
    shutil.copyfile(HERE / 'fixtures' / case['id'] / 'candidate.py', candidate)
    initial = candidate.read_text()

    @tool
    def read_file() -> str:
        """Read the complete candidate.py file in the current task workspace."""
        return candidate.read_text()

    @tool
    def edit_file(new_text: str) -> str:
        """Replace candidate.py with complete new Python source, including imports."""
        if len(new_text.encode()) > 16384:
            return 'Rejected: source exceeds 16 KiB.'
        candidate.write_text(new_text)
        return 'candidate.py updated. Use run_checks to verify.'

    @tool
    async def run_checks() -> str:
        """Run frozen regression and edge-case checks in an isolated Docker container."""
        return json.dumps(await asyncio.to_thread(grade, workspace, case['id'], args.image), ensure_ascii=False)

    settings = Settings(
        max_output_tokens=1024, model_timeout=45,
        summary_trigger_tokens=5000, summary_keep_tokens=1500, summary_trim_tokens=4000,
    )
    usage = UsageMetadataCallbackHandler()
    start = time.monotonic()
    record = {'case': case['id'], 'repeat': repeat, 'model': args.model, 'task': case['task']}
    async with httpx.AsyncClient() as async_http:
        with httpx.Client() as sync_http:
            model = ChatOpenAI(
                model=args.model, base_url=args.base_url, api_key='local',
                temperature=0, max_tokens=settings.max_output_tokens,
                timeout=settings.model_timeout, max_retries=0,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}},
                http_async_client=async_http, http_client=sync_http,
            )
            graph = build_agent(
                model, tools=[read_file, edit_file, run_checks], settings=settings,
                checkpointer=InMemorySaver(),
                system_prompt='You repair Python code in candidate.py. Use the tools to read, edit, and verify the file. Preserve required behavior. A final claim is not proof; use run_checks. Do not return a patch only in chat.',
                model_call_limit=8, tool_call_limit=8,
            )
            identity = str(uuid4())
            config = run_config(identity)
            config['callbacks'] = [usage]
            try:
                state = await asyncio.wait_for(graph.ainvoke(
                    {'messages': [{'role': 'user', 'content': case['task']}]}, config,
                    context=RunContext(
                        run_id=identity, profile='amr-pilot', workspace_root=str(workspace),
                        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=90),
                    ),
                ), timeout=90)
                record['messages'] = [m.model_dump(mode='json') for m in state['messages']]
                record['answer'] = state['messages'][-1].text
            except Exception as exc:
                record['execution_error'] = f'{type(exc).__name__}: {exc}'
                snapshot = await graph.aget_state(config)
                record['messages'] = [m.model_dump(mode='json') for m in snapshot.values.get('messages', [])]
    record['elapsed_seconds'] = round(time.monotonic() - start, 3)
    record['reported_usage'] = usage.usage_metadata
    record['verification'] = await asyncio.to_thread(grade, workspace, case['id'], args.image)
    record['passed'] = record['verification'].get('passed') is True and not record.get('execution_error')
    (folder / 'changes.diff').write_text(''.join(difflib.unified_diff(initial.splitlines(True), candidate.read_text().splitlines(True), fromfile='original/candidate.py', tofile='modified/candidate.py')))
    write_json(folder / 'result.json', record)
    print(case['id'], repeat, 'passed=' + str(record['passed']), 'seconds=' + str(record['elapsed_seconds']), flush=True)
    return record


async def pilot(args):
    output = DATA / ('run-' + uuid4().hex[:12])
    output.mkdir(parents=True)
    calibration = json.loads((DATA / 'calibration.json').read_text())
    assert all(row['valid'] for row in calibration), 'Calibrate the tasks first'
    records = []
    selected = [case for case in CASES if args.case is None or args.case == case['id']]
    for case in selected:
        for repeat in range(1, args.repeats + 1):
            records.append(await episode(case, repeat, args, output))
    summary = {
        'model': args.model, 'base_url': args.base_url, 'image': args.image,
        'evaluator_sha256': hashlib.sha256((HERE / 'evaluator.py').read_bytes()).hexdigest(),
        'tasks_sha256': hashlib.sha256((HERE / 'cases.json').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'temperature': 0, 'enable_thinking': False, 'max_output_tokens': 1024,
        'model_call_limit': 8, 'tool_call_limit': 8, 'episode_timeout_seconds': 90,
        'scope': 'project build_agent with three trusted pilot tools; no API/Worker/PostgreSQL execution',
        'episodes': len(records), 'passed': sum(row['passed'] for row in records),
        'cases': {case['id']: {'passed': sum(row['passed'] for row in records if row['case'] == case['id']), 'runs': args.repeats} for case in selected},
    }
    write_json(output / 'summary.json', summary)
    print('Results:', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['calibrate', 'pilot'])
    parser.add_argument('--image', default=None)
    parser.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    parser.add_argument('--model', default='qwen3.5-0.8b')
    parser.add_argument('--repeats', type=int, default=1, choices=range(1, 4))
    parser.add_argument('--case', choices=[case['id'] for case in CASES])
    args = parser.parse_args()
    args.image = args.image or (DATA / 'image-id.txt').read_text().strip()
    verify_fixtures()
    if args.mode == 'calibrate':
        calibrate(args.image)
    else:
        asyncio.run(pilot(args))


if __name__ == '__main__':
    main()
