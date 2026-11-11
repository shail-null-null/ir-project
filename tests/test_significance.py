import numpy as np
import pytest

from src.significance import paired_randomization_test


def test_identical_systems_give_p_of_one():
    a = [0.2, 0.5, 0.9, 0.1]
    assert paired_randomization_test(a, a) == pytest.approx(1.0)


def test_clear_difference_gives_tiny_p():
    a, b = [1.0] * 50, [0.0] * 50
    assert paired_randomization_test(a, b) < 0.001


def test_matches_scipy_permutation_test():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    b = rng.uniform(0, 1, 40)
    a = b + rng.normal(0.15, 0.5, 40)
    ours = paired_randomization_test(a, b, n_permutations=20000)
    ref = scipy_stats.permutation_test(
        (a, b), lambda x, y: np.mean(x - y), permutation_type="samples",
        n_resamples=20000, alternative="two-sided", random_state=0).pvalue
    assert ours == pytest.approx(ref, abs=0.02)
