# Extracted without function-body changes from AMR-Benchmark.
# Source: RML2018/DAE/main.py:24-26
import numpy as np

def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)
