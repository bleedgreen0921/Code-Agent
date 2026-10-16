"""Frozen behavioral checks; candidate code runs only inside the test container."""

import contextlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def normalize_check(module, edge):
    fn = module.l2_normalize
    if edge:
        x = np.array([[0., 0., 0.], [3., 4., 0.], [-2., 0., 0.]])
        for axis in (-1, 0):
            original = x.copy()
            actual = fn(x, axis=axis)
            norms = np.sqrt(np.sum(original ** 2, axis=axis, keepdims=True))
            expected = np.divide(original, norms, out=np.zeros_like(original), where=norms != 0)
            assert np.isfinite(actual).all(), "Zero vectors must produce finite values"
            np.testing.assert_allclose(actual, expected)
            np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(fn(np.zeros((2, 4))), np.zeros((2, 4)))
    else:
        x = np.array([[3., 4.], [-5., 12.]])
        np.testing.assert_allclose(fn(x), [[.6, .8], [-5 / 13, 12 / 13]])


def confusion_check(module, edge):
    pairs = [([], [], 3), ([0, 0, 2], [0, 1, 2], 4)] if edge else [([0, 0, 1, 2], [0, 1, 1, 0], 3)]
    if edge:
        rng = np.random.default_rng(948)
        pairs.extend((rng.integers(0, 3, n).tolist(), rng.integers(0, 5, n).tolist(), 5) for n in (1, 7, 19))
    for truth, prediction, count in pairs:
        eye = np.eye(count)
        y = eye[np.array(truth, dtype=int)]
        yhat = eye[np.array(prediction, dtype=int)]
        cm, right, wrong = module.calculate_confusion_matrix(y, yhat, list(range(count)))
        expected = np.zeros((count, count))
        for i in range(count):
            support = sum(label == i for label in truth)
            if support:
                for j in range(count):
                    expected[i, j] = sum(a == i and b == j for a, b in zip(truth, prediction)) / support
        assert np.shape(cm) == (count, count)
        assert np.isfinite(cm).all(), "Absent classes must have finite, zero rows"
        np.testing.assert_allclose(cm, expected)
        correct = sum(a == b for a, b in zip(truth, prediction))
        assert right == correct and wrong == len(truth) - correct


def split_check(module, edge):
    sizes_list = [(5, 7, 10, 11), (19, 21, 8, 6)] if edge else [(1000,) * 4]
    for sizes in sizes_list:
        data, labels, ranges, offset = {}, [], [], 0
        for key, size in zip([('A', -2), ('A', 2), ('B', -2), ('B', 2)], sizes):
            x = np.zeros((size, 2, 8), dtype=np.float32)
            x[:, 0, 0] = np.arange(offset, offset + size)
            data[key] = x
            labels.extend([key] * size)
            ranges.append(set(range(offset, offset + size)))
            offset += size
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.pkl'
            path.write_bytes(pickle.dumps(data))
            output = module.load_data(str(path))
            repeated = module.load_data(str(path))
        assert len(output) == 5, "Preserve metadata, three partitions, and index tuple"
        (mods, snrs, lbl), *partitions, indices = output
        assert mods == ['A', 'B'] and snrs == [-2, 2] and lbl == labels
        assert len(partitions) == len(indices) == 3
        sets = [set(map(int, values)) for values in indices]
        assert set.union(*sets) == set(range(offset)), "Every sample must be assigned"
        assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i))
        for i, ((x, y), selected) in enumerate(zip(partitions, indices)):
            assert len(selected) == len(sets[i]), "Duplicate indices"
            assert len(x) == len(y) == len(selected)
            np.testing.assert_array_equal(x[:, 0, 0], selected)
            expected_y = np.eye(2)[[mods.index(lbl[j][0]) for j in selected]]
            np.testing.assert_array_equal(y, expected_y)
            for group, size in zip(ranges, sizes):
                train, val = int(size * .6), int(size * .2)
                assert len(sets[i] & group) == (train, val, size - train - val)[i]
            np.testing.assert_array_equal(selected, repeated[-1][i])


CHECKS = {'normalize_zero': normalize_check, 'confusion_missing': confusion_check, 'split_variable': split_check}


def main():
    case = sys.argv[1]
    checks = []
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            spec = importlib.util.spec_from_file_location('candidate', '/workspace/candidate.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except Exception as exc:
        print(json.dumps({'passed': False, 'checks': [], 'error': f'{type(exc).__name__}: {exc}'}))
        return
    for edge in (False, True):
        result = {'name': 'edge' if edge else 'regression'}
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured), np.errstate(divide='raise', invalid='raise'):
                CHECKS[case](module, edge)
            result['passed'] = True
        except Exception as exc:
            result.update(passed=False, error=f'{type(exc).__name__}: {exc}', detail=traceback.format_exc(limit=2)[-1800:])
        checks.append(result)
    print(json.dumps({'passed': all(c['passed'] for c in checks), 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
