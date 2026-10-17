"""Disposable candidate interpreter; only its parent owns the fixed assertions."""
import json
import os
import sys


def main():
    # The result pipe is distinct from project stdout/stderr, including os.write.
    protocol = os.fdopen(int(sys.argv[1]), 'w', encoding='utf-8')
    request = json.loads(sys.stdin.buffer.read(3 * 1024 * 1024))
    items = request['items']
    namespace = {'__name__': 'repair_candidate', '__file__': '/workspace/sample.py'}
    value, error, value_type = None, None, None
    try:
        exec(compile(request['source'], '/workspace/sample.py', 'exec'), namespace)
        value = namespace['first_n'](items, request['n'])
        value_type = type(value).__name__
    except BaseException as exc:
        error = type(exc).__name__
    print(json.dumps({'value': value, 'value_type': value_type, 'items': items,
                      'error': error}, allow_nan=False), file=protocol, flush=True)
    protocol.close()


if __name__ == '__main__':
    main()
