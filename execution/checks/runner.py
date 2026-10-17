"""Read-only fixed gate supervisor. No workspace tests/modules/bytecode are loaded."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
ASSETS = ('runner.py', 'acceptance.py', 'candidate.py')
CHECK_ID = 'python-first-n'
VERSION = '1'


def assets():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ASSETS}


def descendants():
    found = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) == os.getpid():
                found.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def idle():
    shells = 0
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        pid = int(path.parent.name)
        if pid in (1, os.getpid()):
            continue
        try:
            command = path.read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if command == [b'/bin/bash', b'--noprofile', b'--norc', b''] and shells == 0:
            shells += 1
        else:
            return False
    return True


def converge():
    # As a subreaper, detached grandchildren are adopted here too. Repeatedly
    # kill/reap until none remain; never assert success with live descendants.
    until = time.monotonic() + 0.5
    while time.monotonic() < until:
        children = descendants()
        if not children:
            return True
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(0.005)
    return not descendants()


def invoke(source, items, n):
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            [sys.executable, '-I', str(ROOT / 'candidate.py'), str(write_fd)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            pass_fds=(write_fd,), start_new_session=True,
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    response = os.fdopen(read_fd, 'rb', buffering=0)
    data = json.dumps({'source': source, 'items': items, 'n': n}, ensure_ascii=False).encode()
    # Nonblocking stdin and both output pipes prevent candidate output from
    # deadlocking the supervisor or allocating unbounded communicate() buffers.
    selector = selectors.DefaultSelector()
    buffers = {kind: bytearray() for kind in ('stdout', 'stderr', 'response')}
    offset = 0
    for stream, kind, events in ((proc.stdin, 'stdin', selectors.EVENT_WRITE),
                                 (proc.stdout, 'stdout', selectors.EVENT_READ),
                                 (proc.stderr, 'stderr', selectors.EVENT_READ),
                                 (response, 'response', selectors.EVENT_READ)):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, events, kind)
    until = time.monotonic() + 1.0
    try:
        while selector.get_map():
            if proc.poll() is not None and not converge():
                raise RuntimeError('candidate descendants did not converge')
            if time.monotonic() >= until:
                raise TimeoutError('candidate timeout')
            for key, _ in selector.select(0.02):
                if key.data == 'stdin':
                    try:
                        offset += os.write(key.fd, data[offset:offset + 4096])
                    except BrokenPipeError:
                        offset = len(data)
                    if offset == len(data):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        target = buffers[key.data]
                        capacity = 8193 if key.data == 'response' else 2048
                        target.extend(chunk[:max(0, capacity - len(target))])
        proc.wait(timeout=max(0.001, until - time.monotonic()))
        if proc.returncode != 0 or len(buffers['response']) > 8192:
            raise RuntimeError('candidate response unavailable or excessive')
        value = json.loads(buffers['response'])
        if (type(value) is not dict or set(value) != {'value', 'value_type', 'items', 'error'}
                or type(value['items']) is not list
                or value['value_type'] is not None and type(value['value_type']) is not str
                or value['error'] is not None and type(value['error']) is not str):
            raise RuntimeError('invalid candidate response')
        value['diagnostic'] = bytes(buffers['stdout'] + buffers['stderr'])[:2048].decode('utf-8', 'replace')
        return value
    finally:
        selector.close()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr, response):
            stream.close()
        if not converge():
            raise RuntimeError('candidate descendants did not converge')


class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.timed_out = False

    def addError(self, test, error):
        self.timed_out |= issubclass(error[0], (TimeoutError, subprocess.TimeoutExpired))
        super().addError(test, error)

    def startTest(self, test):
        super().startTest(test)
        self.ids.append(test.id())


def main():
    # Linux execution images only. A failed setup cannot yield a passing receipt.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('subreaper unavailable')
    request = json.loads(sys.stdin.buffer.read(4096))
    if set(request) != {'sha256'} or type(request['sha256']) is not str:
        raise ValueError('invalid request')
    receipt = {'protocol': 1, 'check_id': CHECK_ID, 'version': VERSION,
               'assets': assets(), 'source_sha256': request['sha256'],
               'status': 'inconclusive', 'reason': 'source_unavailable',
               'executed': 0, 'failures': 0, 'errors': 0, 'skipped': 0,
               'unexpected_successes': 0, 'test_ids': [], 'diagnostic': ''}
    exit_code = 2
    try:
        if not idle():
            receipt['reason'] = 'resources_unsettled'
            print(json.dumps(receipt), flush=True)
            return 2
        fd = os.open('/workspace/sample.py', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as source_file:
            import stat
            if not stat.S_ISREG(os.fstat(source_file.fileno()).st_mode):
                raise OSError('regular source required')
            raw = source_file.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise OSError('source too large')
        digest = hashlib.sha256(raw).hexdigest()
        receipt['source_sha256'] = digest
        if digest != request['sha256']:
            receipt['reason'] = 'source_mismatch'
        else:
            source = raw.decode('utf-8')
            acceptance = {}
            exec(compile((ROOT / 'acceptance.py').read_bytes(), str(ROOT / 'acceptance.py'), 'exec'), acceptance)
            result = Result()
            diagnostics = ''
            def candidate(items, n):
                nonlocal diagnostics
                value = invoke(source, items, n)
                diagnostics += value['diagnostic'][:max(0, 1024 - len(diagnostics))]
                return value
            acceptance['suite'](candidate).run(result)
            status = 'inconclusive' if result.timed_out else 'passed' if result.wasSuccessful() else 'failed'
            receipt.update(status=status, reason='candidate_timeout' if result.timed_out else 'tests_' + status,
                           executed=result.testsRun, failures=len(result.failures),
                           errors=len(result.errors), skipped=len(result.skipped),
                           unexpected_successes=len(result.unexpectedSuccesses),
                           test_ids=result.ids,
                           diagnostic=(diagnostics + '\n'.join(text for _, text in result.failures + result.errors))[:2048])
            exit_code = {'passed': 0, 'failed': 1, 'inconclusive': 2}[status]
    except (OSError, UnicodeError) as exc:
        receipt['diagnostic'] = type(exc).__name__
    finally:
        if not converge() or not idle():
            receipt.update(status='inconclusive', reason='processes_unsettled')
            exit_code = 2
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
