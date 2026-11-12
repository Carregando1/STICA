import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stica import (LifetimeLoss, evolve_rule_array, evolve_rule_array_batch, lifetime, rule_table, rule_tables,
                   simulate_rule_array, simulate_table_array, single_cell, steps_to_target)
from reference import rule_array_reference, wolfram_lifetime_reference, wolfram_rule_array_reference

DATA = json.loads((Path(__file__).parent / "data" / "wolfram_rule_arrays.json").read_text())
AND, XOR = 8, 6   # the k = 2, radius-1/2 rules Wolfram calls And and Xor


def _bits(rows):
    return np.array([[int(c) for c in r] for r in rows], np.int8)


def _random_case(seed, T, W, rules, p=0.5):
    rng = np.random.default_rng(seed)
    ra = rng.integers(0, len(rules), (T, W))
    ic = (rng.random(W) < p).astype(np.int8)
    return ic, ra


# ------------------------------------------------------------------------------------------------- rule numbering

def test_rule_table_digits():
    assert rule_table(30).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]          # 30 = 0b00011110
    assert rule_table(146).tolist() == [0, 1, 0, 0, 1, 0, 0, 1]
    assert rule_table(AND, radius=0.5).tolist() == [0, 0, 0, 1]
    assert rule_table(XOR, radius=0.5).tolist() == [0, 1, 1, 0]
    assert rule_table(777, k=3).tolist()[:7] == [0, 1, 2, 1, 0, 0, 1]   # 777 = 1001210 in base 3, reversed
    with pytest.raises(ValueError):
        rule_table(256)


def test_rule_90_is_pascals_triangle_mod_2():
    T, W = 40, 101
    traj = np.asarray(simulate_rule_array(single_cell(W), np.zeros((1, 1), np.int8), [90], num_steps=T))
    c = (W - 1) // 2
    for t in range(T + 1):
        for x in range(-t, t + 1):
            m = (t + x) // 2
            expected = int((t + x) % 2 == 0 and (m & t) == m)     # Lucas: C(t, m) is odd iff m & t == m
            assert traj[t, c + x] == expected


def test_rule_30_centre_column():
    # OEIS A051023: the centre column of rule 30 grown from a single cell
    a051023 = [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    W = 2 * len(a051023) + 1
    traj = np.asarray(simulate_rule_array(single_cell(W), np.zeros((1, 1), np.int8), [30],
                                          num_steps=len(a051023) - 1))
    assert traj[:, (W - 1) // 2].tolist() == a051023


# ----------------------------------------------------------------------------------- against Wolfram's own code

@pytest.mark.parametrize("seed", range(4))
def test_matches_wolfram_code_rules_4_146(seed):
    ic, ra = _random_case(seed, 20, 17, (4, 146))
    ours = np.asarray(simulate_rule_array(ic, ra, [4, 146]))
    ref = wolfram_rule_array_reference(ic, np.array([4, 146])[ra])
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("neighborhood, wl_name", [("alternating", "Alternating"), ("left_center", "LeftCenter"),
                                                    ("center_right", "RightCenter")])
def test_matches_wolfram_code_half_radius(neighborhood, wl_name):
    rules = (AND, XOR, 7, 1)   # And, Xor, Nand, Nor
    ic, ra = _random_case(7, 25, 21, rules)
    ours = np.asarray(simulate_rule_array(ic, ra, list(rules), radius=0.5, neighborhood=neighborhood))
    ref = wolfram_rule_array_reference(ic, np.array(rules)[ra], radius=0.5, neighborhood=wl_name)
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("k, radius, boundary", [(3, 1, "periodic"), (3, 1, "fixed"), (2, 2, "fixed"),
                                                 (4, 1, "periodic"), (2, 0, "periodic")])
def test_matches_generic_reference(k, radius, boundary):
    rng = np.random.default_rng(k * 10 + radius)
    size = k ** (2 * radius + 1)
    numbers = [int("".join(map(str, rng.integers(0, k, size))), k) for _ in range(3)]
    ra = rng.integers(0, 3, (15, 13))
    ic = rng.integers(0, k, 13)
    ours = np.asarray(simulate_rule_array(ic, ra, numbers, k=k, radius=radius, boundary=boundary))
    ref = rule_array_reference(ic, np.array(numbers, dtype=object)[ra], k, radius, boundary)
    np.testing.assert_array_equal(ours, ref)


def test_table_array_matches_rule_array():
    ic, ra = _random_case(3, 12, 19, (4, 146, 30))
    tables = np.asarray(rule_tables([4, 146, 30]))[ra]                 # (T, W, 8): every cell owns a full table
    np.testing.assert_array_equal(np.asarray(simulate_table_array(ic, tables)),
                                  np.asarray(simulate_rule_array(ic, ra, [4, 146, 30])))


@pytest.mark.parametrize("shape", [(12, 1), (1, 19), (1, 1)])
def test_shared_rule_arrays(shape):
    rng = np.random.default_rng(5)
    ra = rng.integers(0, 3, shape)
    ic = (rng.random(19) < 0.5).astype(np.int8)
    full = np.broadcast_to(ra, (12, 19))
    np.testing.assert_array_equal(np.asarray(simulate_rule_array(ic, ra, [4, 146, 30], num_steps=12)),
                                  np.asarray(simulate_rule_array(ic, full, [4, 146, 30])))


def test_steps_past_the_end_are_refused():
    with pytest.raises(ValueError):
        simulate_rule_array(single_cell(9), np.zeros((5, 9), np.int8), [4, 146], num_steps=6)
    with pytest.raises(ValueError):
        simulate_rule_array(single_cell(9), np.zeros((1, 1), np.int8), [4, 146])       # shared in time: say how long


# ---------------------------------------------------------------------- Wolfram's published rule arrays (figures)

@pytest.mark.parametrize("figure", ["lifetime50_solutions", "examples", "simple_solutions"])
def test_reproduces_wolfram_figures_cell_for_cell(figure):
    """Rule arrays and states read off the essay's figures (reproduce/extract_wolfram_figures.py)."""
    fig = DATA["figures"][figure]
    for panel in fig["panels"]:
        ra, states = _bits(panel["rule_array"]), _bits(panel["states"])
        assert states[0].tolist() == np.asarray(single_cell(fig["width"])).tolist()
        traj = np.asarray(simulate_rule_array(states[0], ra, DATA["rules"]))
        np.testing.assert_array_equal(traj, states)


def test_published_lifetime_50_solutions():
    fig = DATA["figures"]["lifetime50_solutions"]
    assert len(fig["panels"]) == 12
    loss = LifetimeLoss(50)
    for panel in fig["panels"]:
        traj = simulate_rule_array(single_cell(31), _bits(panel["rule_array"]), [4, 146])
        assert int(lifetime(traj)) == 50 and float(loss(traj)) == 0.0


def test_simple_solutions_have_lifetime_2i_plus_1():
    for i, panel in enumerate(DATA["figures"]["simple_solutions"]["panels"], start=1):
        ra = _bits(panel["rule_array"])
        assert np.argwhere(ra).tolist() == [[2 * i - 1, 6]]               # ReplacePart[..., {2 i, 7} -> 2]
        assert int(lifetime(simulate_rule_array(single_cell(15), ra, [4, 146]))) == 2 * i + 1


# ------------------------------------------------------------------------------------------------------ lifetime

def test_lifetime_definition():
    ra = np.zeros((6, 9), np.int8)
    assert int(lifetime(simulate_rule_array(np.zeros(9, np.int8), ra, [4, 146]))) == 0      # blank start
    assert int(lifetime(simulate_rule_array(single_cell(9), ra, [4, 146]))) == 7            # survives: T + 1
    assert float(LifetimeLoss(3)(simulate_rule_array(single_cell(9), ra, [4, 146]))) == np.inf
    ra[3, 3] = 1                                                                             # rule 146 at row 3
    traj = np.asarray(simulate_rule_array(single_cell(9), ra, [4, 146]))
    assert int(lifetime(traj)) == wolfram_lifetime_reference(traj.tolist()) == 5
    assert float(LifetimeLoss(3)(traj)) == 2.0


# ----------------------------------------------------------------------------------------------------- evolution

def test_evolution_reaches_a_lifetime_target():
    H, W, target = 20, 15, 12
    ic, ra0 = single_cell(W), np.zeros((H, W), np.int8)
    res = evolve_rule_array_batch(jax.random.PRNGKey(0), 8, ic, ra0, [4, 146], objective=LifetimeLoss(target),
                                  iterations=3000, stop_at=0.0)
    assert float(res.initial[0]) == np.inf
    solved = np.asarray(res.solved_at) > 0
    assert solved.mean() >= 0.75
    for i in np.flatnonzero(solved):
        ra = np.asarray(res.ra[i])
        ref = wolfram_rule_array_reference(np.asarray(ic), np.array([4, 146])[ra])
        assert wolfram_lifetime_reference(ref.tolist()) == target
        trace = np.asarray(res.best_trace[i])
        s = int(res.solved_at[i])
        assert trace[s - 1] == 0 and (s == 1 or trace[s - 2] > 0) and np.all(trace[s - 1:] == 0)
        assert not np.asarray(res.accepted[i])[s:].any()                    # frozen once solved


def test_evolution_mutations_and_determinism():
    ic, ra0 = single_cell(15), np.zeros((10, 15), np.int8)
    kw = dict(objective=LifetimeLoss(6), iterations=40, mutations=3, record_mutations=True)
    a = evolve_rule_array(jax.random.PRNGKey(1), ic, ra0, [4, 146, 30], **kw)
    b = evolve_rule_array(jax.random.PRNGKey(1), ic, ra0, [4, 146, 30], **kw)
    np.testing.assert_array_equal(np.asarray(a.ra), np.asarray(b.ra))
    m = a.mutations
    assert m.srt_pos.shape == (40, 3, 2)
    assert all(len({tuple(p) for p in step}) == 3 for step in np.asarray(m.srt_pos))
    assert np.all(np.asarray(m.srt_old) != np.asarray(m.srt_new))
    assert np.asarray(m.srt_new).max() <= 2
    # replaying the accepted mutations rebuilds the final array
    ra = np.array(ra0)
    for acc, pos, new in zip(np.asarray(a.accepted), np.asarray(m.srt_pos), np.asarray(m.srt_new)):
        if acc:
            ra[tuple(pos.T)] = new
    np.testing.assert_array_equal(ra, np.asarray(a.ra))


def test_batch_matches_single_runs():
    ic, ra0 = single_cell(11), np.zeros((12, 11), np.int8)
    kw = dict(objective=LifetimeLoss(7), iterations=60, stop_at=0.0)
    batch = evolve_rule_array_batch(jax.random.PRNGKey(2), 3, ic, ra0, [4, 146], **kw)
    for i, key in enumerate(jax.random.split(jax.random.PRNGKey(2), 3)):
        single = evolve_rule_array(key, ic, ra0, [4, 146], **kw)
        np.testing.assert_array_equal(np.asarray(single.ra), np.asarray(batch.ra[i]))
        assert int(single.solved_at) == int(batch.solved_at[i])


def test_steps_to_target():
    trace = jnp.array([[5.0, 3, 3, 0, 0], [4.0, 4, 4, 4, 4]])
    assert steps_to_target(jnp.array([9.0, 9.0]), trace, 0.0).tolist() == [4, -1]
    assert int(steps_to_target(jnp.array(0.0), trace[1], 0.0)) == 0


def test_and_xor_lifetime_setup_runs():
    """Wolfram's second lifetime example: And/Xor (radius 1/2, alternating), 40 x 21, target 30."""
    res = evolve_rule_array(jax.random.PRNGKey(3), single_cell(21), np.zeros((40, 21), np.int8), [AND, XOR],
                            objective=LifetimeLoss(30), iterations=200, radius=0.5, stop_at=0.0)
    ra = np.asarray(res.ra)
    ref = wolfram_rule_array_reference(np.asarray(single_cell(21)), np.array([AND, XOR])[ra], radius=0.5)
    life = wolfram_lifetime_reference(ref.tolist())
    assert float(res.best) == (np.inf if life == 41 else abs(life - 30))
