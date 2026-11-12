import json
from pathlib import Path

import jax
import numpy as np
import pytest

from stica import (LifetimeLoss, evolve_packed_batch, evolve_rule_array_batch, lifetime, pack_cells, pack_rule_array,
                   simulate_packed, simulate_rule_array, single_cell, unpack_cells, unpack_rule_array)

DATA = json.loads((Path(__file__).parent / "data" / "wolfram_rule_arrays.json").read_text())
RNG = np.random.default_rng(0)


@pytest.mark.parametrize("width", [1, 7, 31, 32, 33, 64, 100])
def test_pack_roundtrip(width):
    cells = RNG.integers(0, 2, (3, width)).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(unpack_cells(pack_cells(cells), width)), cells)
    ra = RNG.integers(0, 5, (4, width)).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(unpack_rule_array(pack_rule_array(ra, 5), width)), ra)


def _random_rules(count, radius):
    inputs = 2 if radius == 0.5 else 2 * radius + 1
    return [int(r) for r in RNG.integers(0, 2 ** (2 ** inputs), count)]


@pytest.mark.parametrize("width", [7, 31, 32, 33, 64, 95, 130])
@pytest.mark.parametrize("boundary", ["periodic", "fixed"])
def test_matches_unpacked_radius_1(width, boundary):
    for K in (2, 3, 5):
        rules = [4, 146] if K == 2 else _random_rules(K, 1)
        ra = RNG.integers(0, K, (40, width))
        ic = (RNG.random(width) < 0.5).astype(np.int8)
        np.testing.assert_array_equal(
            np.asarray(simulate_packed(ic, ra, rules, boundary=boundary)),
            np.asarray(simulate_rule_array(ic, ra, rules, boundary=boundary)))


@pytest.mark.parametrize("radius, neighborhood", [(0.5, "alternating"), (0.5, "left_center"),
                                                  (0.5, "center_right"), (2, "alternating"), (0, "alternating")])
def test_matches_unpacked_other_neighbourhoods(radius, neighborhood):
    for width in (21, 33, 70):
        rules = _random_rules(3, radius)
        ra = RNG.integers(0, 3, (25, width))
        ic = (RNG.random(width) < 0.5).astype(np.int8)
        for boundary in ("periodic", "fixed"):
            kw = dict(radius=radius, neighborhood=neighborhood, boundary=boundary)
            np.testing.assert_array_equal(np.asarray(simulate_packed(ic, ra, rules, **kw)),
                                          np.asarray(simulate_rule_array(ic, ra, rules, **kw)))


@pytest.mark.parametrize("shape", [(12, 1), (1, 19), (1, 1)])
def test_shared_rule_arrays(shape):
    ra = RNG.integers(0, 2, shape)
    ic = (RNG.random(19) < 0.5).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(simulate_packed(ic, ra, [4, 146], num_steps=12)),
                                  np.asarray(simulate_rule_array(ic, ra, [4, 146], num_steps=12)))


def test_wolfram_published_solutions():
    for panel in DATA["figures"]["lifetime50_solutions"]["panels"]:
        ra = np.array([[int(c) for c in r] for r in panel["rule_array"]], np.int8)
        states = np.array([[int(c) for c in r] for r in panel["states"]], np.int8)
        np.testing.assert_array_equal(np.asarray(simulate_packed(states[0], ra, [4, 146])), states)
        words = simulate_packed(states[0], ra, [4, 146], packed_output=True)
        assert int(lifetime(words)) == 50                     # lifetime works on packed words as they are


def _check_lifetime(ra, width, target, rules=(4, 146), radius=1):
    traj = simulate_rule_array(single_cell(width), np.asarray(ra), list(rules), radius=radius)
    return float(LifetimeLoss(target)(traj))


GPU = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the fused kernel needs a GPU")
BACKENDS = ["xla", pytest.param("pallas", marks=GPU)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("radius, rules", [(1, (4, 146)), (0.5, (8, 6))])
def test_evolution_without_restart(radius, rules, backend):
    H, W, target = 24, 33, 12
    res = evolve_packed_batch(jax.random.PRNGKey(0), 16, single_cell(W), np.zeros((H, W), np.int8), list(rules),
                              objective=LifetimeLoss(target), iterations=3000, radius=radius, stop_at=0.0,
                              record_trace=True, backend=backend)
    assert np.mean(np.asarray(res.solved_at) > 0) >= 0.75
    for i in range(16):
        assert float(res.best[i]) == _check_lifetime(res.ra[i], W, target, rules, radius)
        s = int(res.solved_at[i])
        if s > 0:
            trace = np.asarray(res.best_trace[i])
            assert trace[s - 1] == 0 and np.all(trace[s - 1:] == 0) and not np.asarray(res.accepted[i])[s:].any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_evolution_with_restart(backend):
    H, W, target = 20, 31, 9
    res = evolve_packed_batch(jax.random.PRNGKey(1), 8, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                              objective=LifetimeLoss(target), iterations=4000, stop_at=0.0, restart=True,
                              record_trace=True, backend=backend)
    solves = np.asarray(res.solves)
    assert solves.sum() > 8                                    # several solutions per lane
    trace = np.asarray(res.best_trace)
    np.testing.assert_array_equal((trace == 0).sum(1), solves)  # every solution appears once in the trace
    for i in np.flatnonzero(solves):
        assert _check_lifetime(res.last_solution[i], W, target) == 0.0
        hits = np.flatnonzero(trace[i] == 0) + 1                  # solution lengths = gaps between solutions
        assert int(res.solve_steps[i]) == hits[-1] and int(res.solved_at[i]) == hits[0]


def test_same_distribution_as_unpacked():
    """Packed and unpacked evolution are the same algorithm with different random streams."""
    H, W, target = 30, 31, 20
    kw = dict(objective=LifetimeLoss(target), iterations=3000, stop_at=0.0)
    a = evolve_packed_batch(jax.random.PRNGKey(2), 512, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                            backend="xla", **kw)
    b = evolve_rule_array_batch(jax.random.PRNGKey(3), 512, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                                **kw)
    sa, sb = np.asarray(a.solved_at), np.asarray(b.solved_at)
    assert abs((sa > 0).mean() - (sb > 0).mean()) < 0.06
    ma, mb = np.median(sa[sa > 0]), np.median(sb[sb > 0])
    assert abs(ma - mb) < 0.2 * mb


def test_unpacked_restart_counts():
    H, W, target = 20, 15, 9
    res = evolve_rule_array_batch(jax.random.PRNGKey(4), 4, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                                  objective=LifetimeLoss(target), iterations=3000, stop_at=0.0, restart=True)
    trace = np.asarray(res.best_trace)
    np.testing.assert_array_equal((trace == 0).sum(1), np.asarray(res.solves))
    assert np.asarray(res.solves).sum() > 4


@GPU
@pytest.mark.parametrize("width, steps, radius, neighborhood, boundary, rules", [
    (31, 64, 1, "alternating", "periodic", (4, 146)), (33, 37, 1, "alternating", "fixed", (30, 110)),
    (311, 200, 1, "alternating", "periodic", (4, 146)), (21, 40, 0.5, "alternating", "periodic", (8, 6)),
    (70, 25, 0.5, "left_center", "fixed", (7, 1)), (64, 25, 0.5, "center_right", "periodic", (14, 9))])
def test_fused_kernel_matches_unpacked(width, steps, radius, neighborhood, boundary, rules):
    import jax.numpy as jnp
    from stica.packed import fused_rollout
    B = 256
    ics = (RNG.random((B, width)) < 0.5).astype(np.int8)
    ras = RNG.integers(0, 2, (B, steps, width)).astype(np.int8)
    kw = dict(radius=radius, neighborhood=neighborhood, boundary=boundary)
    out = fused_rollout(pack_cells(ics).T, jnp.moveaxis(pack_cells(ras), 0, -1), rules=rules, width=width, **kw)
    got = np.asarray(unpack_cells(jnp.moveaxis(out, -1, 0), width))
    for b in range(0, B, 17):
        np.testing.assert_array_equal(got[b], np.asarray(simulate_rule_array(ics[b], ras[b], list(rules), **kw)))


@GPU
def test_fused_same_distribution_as_xla():
    H, W, target = 30, 31, 20
    kw = dict(objective=LifetimeLoss(target), iterations=3000, stop_at=0.0)
    a = evolve_packed_batch(jax.random.PRNGKey(5), 512, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                            backend="pallas", **kw)
    b = evolve_packed_batch(jax.random.PRNGKey(6), 512, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                            backend="xla", **kw)
    sa, sb = np.asarray(a.solved_at), np.asarray(b.solved_at)
    assert abs((sa > 0).mean() - (sb > 0).mean()) < 0.06
    assert abs(np.median(sa[sa > 0]) - np.median(sb[sb > 0])) < 0.2 * np.median(sb[sb > 0])
