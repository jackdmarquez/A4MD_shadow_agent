from __future__ import annotations
import numpy as np

try:
    import scipy.linalg as lin_alg
except Exception:
    lin_alg = None


def compute_lev(points: np.ndarray, atom_index_groups) -> float:
    if lin_alg is None:
        raise RuntimeError("scipy is required to compute LEV")

    nsegs = len(atom_index_groups)
    X = np.zeros((nsegs, 3), dtype=float)

    for i in range(nsegs):
        g = atom_index_groups[i]
        if isinstance(g, (list, tuple, np.ndarray)) and not np.isscalar(g):
            X[i] = np.mean(points[g], axis=0) * 10.0
        else:
            X[i] = points[int(g)] * 10.0

    if len(X) == 0:
        return float("nan")

    M = np.zeros((len(X), len(X)), dtype=float)
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d2 = (X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2 + (X[i][2] - X[j][2]) ** 2
            M[i, j] = d2
            M[j, i] = d2

    return float(lin_alg.eigvalsh(M)[-1])
