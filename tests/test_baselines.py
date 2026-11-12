"""The CPU baselines used in the benchmarks compute exactly what the package computes."""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import stica

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import baselines as B  # noqa: E402

RNG = np.random.default_rng(0)


def _rule_array_case(T=30, W=23):
    return (RNG.random(W) < 0.4).astype(np.int8), RNG.integers(0, 3, (T, W)).astype(np.int8), \
        np.asarray(stica.rule_tables([4, 146, 30]))


def _stica_case(n, T=6, G=9, shared=False):
    ic = RNG.integers(0, n, (G, G)).astype(np.int8)
    shape = (T, 1, 1, stica.num_rules(n)) if shared else (T, G, G, stica.num_rules(n))
    return ic, RNG.integers(0, n, shape).astype(np.int8)


def test_rule_array_numpy():
    ic, ra, tables = _rule_array_case()
    np.testing.assert_array_equal(B.rule_array_numpy(ic, ra, tables), stica.simulate_rule_array(ic, ra, tables))


def test_rule_array_numba():
    pytest.importorskip("numba")
    ic, ra, tables = _rule_array_case()
    np.testing.assert_array_equal(B.rule_array_numba(ic, ra, tables), stica.simulate_rule_array(ic, ra, tables))


def test_rule_array_cellpylib():
    pytest.importorskip("cellpylib")
    ic, ra, tables = _rule_array_case()
    np.testing.assert_array_equal(B.rule_array_cellpylib(ic, ra, tables), stica.simulate_rule_array(ic, ra, tables))


@pytest.mark.parametrize("n, shared", [(2, False), (3, False), (4, True)])
def test_stica_numpy(n, shared):
    ic, srt = _stica_case(n, shared=shared)
    np.testing.assert_array_equal(B.stica_numpy(ic, srt, n), stica.simulate(ic, srt, n, num_steps=srt.shape[0]))


@pytest.mark.parametrize("n, shared", [(2, False), (3, False), (5, True)])
def test_stica_numba(n, shared):
    pytest.importorskip("numba")
    ic, srt = _stica_case(n, shared=shared)
    np.testing.assert_array_equal(B.stica_numba(ic, srt, n), stica.simulate(ic, srt, n, num_steps=srt.shape[0]))


@pytest.mark.parametrize("n", [2, 3])
def test_stica_cellpylib(n):
    pytest.importorskip("cellpylib")
    ic, srt = _stica_case(n, T=4, G=7)
    np.testing.assert_array_equal(B.stica_cellpylib(ic, srt, n), stica.simulate(ic, srt, n))


def test_lifetime_evolution_numba():
    pytest.importorskip("numba")
    tables = np.asarray(stica.rule_tables([4, 146]))
    ic, ra0 = np.asarray(stica.single_cell(15)), np.zeros((20, 15), np.int8)
    loss = stica.LifetimeLoss(12)
    for seed in range(5):
        ra, best, used = B.evolve_lifetime_numba(ic, ra0, tables, 12, 3000, seed)
        assert best == float(loss(stica.simulate_rule_array(ic, ra, tables)))
        if best == 0:
            assert 0 < used <= 3000
    # the early-exit lifetime agrees with the package's lifetime loss on random arrays
    lifetime_loss = B._numba()["lifetime_loss"]
    for _ in range(50):
        ra = (RNG.random((20, 15)) < RNG.random()).astype(np.int8)
        assert lifetime_loss(ic, ra, tables, 12.0) == float(loss(stica.simulate_rule_array(ic, ra, tables)))


def test_parallel_and_batched_variants():
    pytest.importorskip("numba")
    ic, ra, tables = _rule_array_case()
    ref = np.asarray(stica.simulate_rule_array(ic, ra, tables))
    np.testing.assert_array_equal(B.rule_array_numba_par(ic, ra, tables), ref)
    ics = np.stack([ic, 1 - ic, np.roll(ic, 3)])
    ras = np.stack([ra, ra[::-1], 2 - ra])
    expected = np.stack([np.asarray(stica.simulate_rule_array(i, r, tables)) for i, r in zip(ics, ras)])
    np.testing.assert_array_equal(B.rule_array_numba_batch_par(ics, ras, tables), expected)
    np.testing.assert_array_equal(B.rule_array_numpy_batch(ics, ras, tables), expected)
    ic2, srt = _stica_case(3)
    np.testing.assert_array_equal(B.stica_numba_par(ic2, srt, 3), stica.simulate(ic2, srt, 3))
    t = np.asarray(stica.rule_tables([4, 146]))
    best, used = B.evolve_lifetime_numba_par(np.asarray(stica.single_cell(15)), np.zeros((20, 15), np.int8), t, 12,
                                             3000, np.arange(4))
    assert best.shape == used.shape == (4,) and np.all((best == 0) <= (used <= 3000))
