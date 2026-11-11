import numpy as np
import jax.numpy as jnp
import pytest

from stica import neighbor_counts, neighbor_tuples, num_rules, reorder_two_state_rules, rule_index, simulate
from stica.rules import validate_srt
from reference import moore_neighbours, simulate_reference_ascending_2state


def test_rule_counts():
    assert [num_rules(n) for n in (2, 3, 4)] == [18, 135, 660]


def test_documented_order_n3():
    tuples = neighbor_tuples(3)
    assert len(tuples) == 45
    assert tuples[:6] == [(8, 0), (7, 1), (7, 0), (6, 2), (6, 1), (6, 0)]
    assert tuples[12] == (4, 2)          # the worked example in the original NSTICA docstring
    assert tuples[-1] == (0, 0)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8])
def test_rule_index_exhaustive(n):
    """Every (state, neighbour tuple) maps to its documented index, including all-non-zero neighbourhoods."""
    tuples = neighbor_tuples(n)
    L = len(tuples)
    counts = jnp.asarray(np.array(tuples, np.int32))[None]
    for s in range(n):
        got = np.asarray(rule_index(jnp.full((1, L), s, jnp.int8), counts, n))[0]
        np.testing.assert_array_equal(got, s * L + np.arange(L))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_neighbor_counts_brute_force(n):
    cells = np.random.default_rng(n).integers(0, n, (7, 9))
    got = np.asarray(neighbor_counts(jnp.asarray(cells, jnp.int8), n))
    for r in range(7):
        for c in range(9):
            nb = moore_neighbours(cells, r, c)
            assert got[r, c].tolist() == [nb.count(k) for k in range(1, n)]


def test_reorder_two_state_rules():
    rng = np.random.default_rng(0)
    srt_asc = rng.integers(0, 2, (4, 8, 8, 18))
    np.testing.assert_array_equal(reorder_two_state_rules(reorder_two_state_rules(srt_asc)), srt_asc)
    ic = rng.choice(2, (8, 8), p=[0.3, 0.7])
    got = simulate(jnp.asarray(ic, jnp.int8), jnp.asarray(reorder_two_state_rules(srt_asc), jnp.int8), 2)
    np.testing.assert_array_equal(np.asarray(got), simulate_reference_ascending_2state(ic, srt_asc))


@pytest.mark.parametrize("srt, n, grid, msg", [
    (np.zeros((2, 4, 4)), 2, None, "4 axes"),
    (np.zeros((2, 4, 4, 17)), 2, None, "18 rules"),
    (np.zeros((2, 3, 4, 18)), 2, (4, 4), "axis 1"),
    (np.full((2, 4, 4, 18), 2), 2, None, r"\[0, 1\]"),
    (np.zeros((2, 4, 4, 18)), 1, None, "number of states"),
])
def test_validate_srt_errors(srt, n, grid, msg):
    with pytest.raises(ValueError, match=msg):
        validate_srt(srt, n, grid)
