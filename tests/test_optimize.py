import numpy as np
import jax
import jax.numpy as jnp
import pytest

from stica import init_state, neg_surface, run_ga, run_ga_batch, sample_distinct, simulate, srt_shape

KEY = jax.random.PRNGKey(0)


@pytest.mark.parametrize("shape, k", [((3, 4, 5, 6), 300), ((7,), 6), ((4, 4), 0), ((2, 3), 6)])
def test_sample_distinct(shape, k):
    c = np.asarray(sample_distinct(KEY, shape, k))
    assert c.shape == (k, len(shape))
    assert len({tuple(x) for x in c}) == k
    assert np.all((c >= 0) & (c < np.array(shape)))


def test_sample_distinct_marginals_uniform():
    draws = jax.jit(jax.vmap(lambda k: sample_distinct(k, (20,), 5)))(jax.random.split(KEY, 2000))
    draws = np.asarray(draws)[..., 0]
    assert all(len(set(d)) == 5 for d in draws)
    hits = np.bincount(draws.ravel(), minlength=20)
    assert hits.min() > 0.85 * hits.mean() and hits.max() < 1.15 * hits.mean()


def test_sample_distinct_too_many():
    with pytest.raises(ValueError):
        sample_distinct(KEY, (3, 3), 10)


@pytest.fixture(scope="module")
def start():
    return init_state(jax.random.PRNGKey(1), 8, 3)


def test_init_state(start):
    ic, srt = start
    assert ic.shape == (8, 8) and ic.dtype == jnp.int8 and set(np.unique(ic)) <= {0, 1, 2}
    assert srt.shape == (7, 8, 8, 135) and not np.asarray(srt).any()


def test_single_iteration_changes_exactly_the_requested_entries(start):
    ic0, srt0 = start
    for seed in range(6):
        res = run_ga(jax.random.PRNGKey(seed), ic0, srt0, n=3, iterations=1, ic_count=5, srt_count=40)
        changes = (int((res.ic != ic0).sum()), int((res.srt != srt0).sum()))
        assert changes == ((5, 40) if bool(res.accepted[0]) else (0, 0))


def test_ga_trace_and_best(start):
    ic0, srt0 = start
    res = run_ga(jax.random.PRNGKey(2), ic0, srt0, n=3, iterations=150, ic_count=5, srt_count=40)
    trace = np.asarray(res.best_trace)
    assert np.all(np.diff(trace) <= 0)
    assert trace[-1] == float(res.best)
    assert float(neg_surface(simulate(res.ic, res.srt, 3))) == float(res.best)
    assert res.accepted.sum() > 0
    initial = float(neg_surface(simulate(ic0, srt0, 3)))
    assert float(res.best) < initial


def test_ga_is_deterministic(start):
    ic0, srt0 = start
    a = run_ga(jax.random.PRNGKey(3), ic0, srt0, n=3, iterations=20, ic_count=5, srt_count=40)
    b = run_ga(jax.random.PRNGKey(3), ic0, srt0, n=3, iterations=20, ic_count=5, srt_count=40)
    np.testing.assert_array_equal(np.asarray(a.srt), np.asarray(b.srt))
    np.testing.assert_array_equal(np.asarray(a.best_trace), np.asarray(b.best_trace))


def test_frozen_ic_and_mutation_record(start):
    ic0, srt0 = start
    res = run_ga(jax.random.PRNGKey(4), ic0, srt0, n=3, iterations=10, ic_count=0, srt_count=12,
                 record_mutations=True)
    np.testing.assert_array_equal(np.asarray(res.ic), np.asarray(ic0))   # the legacy code crashed here
    m = res.mutations
    assert m.ic_pos.shape == (10, 0, 2) and m.srt_pos.shape == (10, 12, 4)
    assert np.all(np.asarray(m.srt_old) != np.asarray(m.srt_new))


def test_batch_matches_single_runs():
    batch = run_ga_batch(jax.random.PRNGKey(5), 3, grid_size=7, n=2, iterations=15, ic_count=4, srt_count=30)
    assert batch.srt.shape == (3, 6, 7, 7, 18) and batch.best.shape == (3,)
    k_init, k_run = jax.random.split(jax.random.PRNGKey(5))
    for i, (ki, kr) in enumerate(zip(jax.random.split(k_init, 3), jax.random.split(k_run, 3))):
        ic, srt = init_state(ki, 7, 2)
        single = run_ga(kr, ic, srt, n=2, iterations=15, ic_count=4, srt_count=30)
        assert float(single.best) == float(batch.best[i])
        np.testing.assert_array_equal(np.asarray(single.srt), np.asarray(batch.srt[i]))


@pytest.mark.parametrize("inhomogeneity", ["spacetime", "time", "space", "none"])
def test_rule_sharing_levels(inhomogeneity):
    G, n = 7, 2
    res = run_ga_batch(jax.random.PRNGKey(6), 2, grid_size=G, n=n, iterations=10, ic_count=3,
                       srt_count=10, inhomogeneity=inhomogeneity)
    assert res.srt.shape[1:] == srt_shape(G, n, inhomogeneity)
    for i in range(2):
        traj = simulate(res.ic[i], res.srt[i], n, num_steps=G - 1)
        assert traj.shape == (G, G, G)
        assert float(neg_surface(traj)) == float(res.best[i])


def _flat_objective(grid):
    return jnp.float32(0.0)


def test_acceptance_rule(start):
    """Ties are accepted by default (Wolfram's adaptive evolution) and rejected with accept_ties=False (legacy)."""
    ic0, srt0 = start
    kw = dict(n=3, iterations=12, ic_count=2, srt_count=5, objective=_flat_objective)
    ties = run_ga(jax.random.PRNGKey(7), ic0, srt0, **kw)
    strict = run_ga(jax.random.PRNGKey(7), ic0, srt0, accept_ties=False, **kw)
    assert bool(ties.accepted.all()) and not bool(strict.accepted.any())
    np.testing.assert_array_equal(np.asarray(strict.srt), np.asarray(srt0))
    res = run_ga(jax.random.PRNGKey(8), ic0, srt0, n=3, iterations=60, ic_count=5, srt_count=40, accept_ties=False)
    trace = np.r_[float(neg_surface(simulate(ic0, srt0, 3))), np.asarray(res.best_trace)]
    assert np.all(np.diff(trace)[np.asarray(res.accepted)] < 0)


def test_count_validation(start):
    ic0, srt0 = start
    with pytest.raises(ValueError, match="ic_count"):
        run_ga(KEY, ic0, srt0, n=3, iterations=1, ic_count=65, srt_count=1)
