"""Paired randomization (sign-flip) test for comparing two systems query by query."""
import numpy as np


def paired_randomization_test(a, b, n_permutations: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for "mean(a - b) differs from 0".

    a[i] and b[i] are the two systems' scores on the SAME query i. If the systems were
    really interchangeable, each per-query difference would be equally likely to have
    either sign. So we flip signs at random many times and ask how often a mean
    difference at least as large as the observed one appears by chance.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
    permuted = np.abs((signs * d).mean(axis=1))
    extreme = int(np.sum(permuted >= observed - 1e-12))
    return (extreme + 1) / (n_permutations + 1)
