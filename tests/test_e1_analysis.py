import numpy as np

from scripts.analyze_e1 import boot_diff, holm, kappa


def test_holm_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj[2] == 0.06 and adj[1] == 0.06
    assert holm([0.6, 0.9]) == [1.0, 1.0]


def test_boot_diff_clear_effect_and_null():
    rng = np.random.default_rng(0)
    m = np.column_stack([np.ones(20), np.zeros(20)])
    r = boot_diff(m, 0, 1, rng)
    assert r["diff"] == 1.0 and r["p"] == 0.0 and r["ci95"] == [1.0, 1.0]
    m0 = np.column_stack([np.r_[np.ones(10), np.zeros(10)], np.r_[np.zeros(10), np.ones(10)]])
    r0 = boot_diff(m0, 0, 1, rng)
    assert abs(r0["diff"]) < 1e-12 and r0["p"] > 0.5


def test_kappa():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-12
