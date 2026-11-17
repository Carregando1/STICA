import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stica import evolve_packed_batch, neighborhood_codes, rule_tables, simulate_rule_array, single_cell
from stica.packed import num_words, pack_cells, unpack_cells
from stica.rulearray import lifetime
from stica.stochastic import (Dynamics, bernoulli_words, evolve_stochastic_batch, quantize, simulate_stochastic,
                              stochastic_rollout, threshold_planes)

RNG = np.random.default_rng(7)
GPU = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the fused kernel needs a GPU")
BACKENDS = ["xla", pytest.param("pallas", marks=GPU)]


def _bits_of(words, width):
    return np.asarray(unpack_cells(words, width))


@pytest.mark.parametrize("q", [0, 1, 37, 128, 200, 255, 256])
def test_bernoulli_rate_shared(q):
    words = bernoulli_words(jax.random.PRNGKey(q), q, 8, (4000, 4))
    frac = _bits_of(words, 128).mean()
    p = q / 256
    assert abs(frac - p) <= 5 * np.sqrt(p * (1 - p) / (4000 * 128)) + 1e-12


def test_bernoulli_rate_per_cell():
    W, n = 70, 6000
    q = RNG.integers(0, 257, (3, W))
    planes = threshold_planes(q, 8)[:, :, None, :]                                   # (9, 3, 1, nw) over samples
    words = bernoulli_words(jax.random.PRNGKey(1), planes, 8, (3, n, num_words(W)))
    freq = _bits_of(words, W).mean(1)                                                # (3, W)
    p = q / 256
    assert np.all(np.abs(freq - p) <= 5 * np.sqrt(p * (1 - p) / n) + 1e-12)
    assert np.all(freq[q == 0] == 0) and np.all(freq[q == 256] == 1)


def _reference(ic, rule_mask, update, noise, rules, radius, neighborhood, boundary):
    """NumPy: new = rules[mask](code) where updated, else old; then flipped where noise."""
    tables = np.asarray(rule_tables(list(rules), 2, radius))
    cells, traj = np.asarray(ic, np.int8), [np.asarray(ic, np.int8)]
    for t in range(rule_mask.shape[0]):
        codes = np.asarray(neighborhood_codes(jnp.asarray(cells), t, radius=radius, neighborhood=neighborhood,
                                              boundary=boundary))
        new = tables[rule_mask[t], codes]
        new = np.where(update[t] == 1, new, cells) ^ noise[t]
        cells = new.astype(np.int8)
        traj.append(cells)
    return np.stack(traj)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("width, steps, radius, neighborhood, boundary, rules", [
    (31, 40, 1, "alternating", "periodic", (4, 146)), (45, 30, 1, "alternating", "fixed", (30, 110)),
    (70, 25, 1, "alternating", "periodic", (90, 150)), (21, 40, 0.5, "alternating", "periodic", (8, 6)),
    (33, 20, 0.5, "left_center", "fixed", (7, 1))])
def test_rollout_matches_reference(backend, width, steps, radius, neighborhood, boundary, rules):
    for trial in range(4):
        ic = (RNG.random(width) < 0.4).astype(np.int8)
        m = (RNG.random((steps, width)) < 0.5).astype(np.int8)
        u = (RNG.random((steps, width)) < [1.0, 0.7, 0.3, 0.9][trial]).astype(np.int8)
        n = (RNG.random((steps, width)) < [0.0, 0.05, 0.0, 0.2][trial]).astype(np.int8)
        got = np.asarray(stochastic_rollout(ic, m, rules, update_mask=u, noise_mask=n, radius=radius,
                                            neighborhood=neighborhood, boundary=boundary, backend=backend))
        np.testing.assert_array_equal(got, _reference(ic, m, u, n, rules, radius, neighborhood, boundary))


@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic_limit(backend):
    H, W = 40, 33
    ra = (RNG.random((H, W)) < 0.3).astype(np.int8)
    ic = single_cell(W)
    want = np.asarray(simulate_rule_array(ic, ra, [4, 146]))
    for kw in (dict(ra=ra), dict(prob=ra.astype(float))):
        out = np.asarray(simulate_stochastic(jax.random.PRNGKey(0), ic, [4, 146], num_samples=5, backend=backend, **kw))
        for s in out:
            np.testing.assert_array_equal(s, want)


@pytest.mark.parametrize("backend", BACKENDS)
def test_async_and_noise_limits(backend):
    W = 64
    ic = (RNG.random(W) < 0.5).astype(np.int8)
    ra = (RNG.random((10, W)) < 0.5).astype(np.int8)
    frozen = np.asarray(simulate_stochastic(jax.random.PRNGKey(1), ic, [30, 110], ra=ra, alpha=0.0, num_samples=3,
                                            backend=backend))
    assert np.all(frozen == ic)
    # from a blank lattice under rules that keep 000 -> 0, one step leaves exactly the noise
    eps = 2 ** -5
    one = np.asarray(simulate_stochastic(jax.random.PRNGKey(2), np.zeros(W, np.int8), [4, 146],
                                         ra=np.zeros((1, W), np.int8), epsilon=eps, num_samples=4000,
                                         backend=backend))[:, 1]
    assert abs(one.mean() - eps) <= 5 * np.sqrt(eps * (1 - eps) / one.size)
    # probabilistic rules 0 (all -> 0) and 255 (all -> 1): the new state is the sampled rule
    p = 0.25
    out = np.asarray(simulate_stochastic(jax.random.PRNGKey(3), np.zeros(W, np.int8), [0, 255],
                                         prob=np.full((1, W), p), num_samples=4000, backend=backend))[:, 1]
    assert abs(out.mean() - p) <= 5 * np.sqrt(p * (1 - p) / out.size)


@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic_evolution_is_wolframs(backend):
    """genome='deterministic', one sample, alpha = 1, epsilon = 0: the same algorithm as evolve_packed_batch."""
    H, W, target, runs = 30, 31, 20, 512
    a = evolve_stochastic_batch(jax.random.PRNGKey(4), runs, single_cell(W), [4, 146], target=target,
                                iterations=3000, num_steps=H, samples=1, genome="deterministic", backend=backend)
    b = evolve_packed_batch(jax.random.PRNGKey(5), runs, single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                            iterations=3000, stop_at=0.0, objective=__import__("stica").LifetimeLoss(target),
                            backend="xla")
    sa, sb = np.asarray(a.reliable_at), np.asarray(b.solved_at)
    assert abs((sa > 0).mean() - (sb > 0).mean()) < 0.06
    assert abs(np.median(sa[sa > 0]) - np.median(sb[sb > 0])) < 0.2 * np.median(sb[sb > 0])
    g = np.asarray(a.genome)
    for i in np.flatnonzero(sa > 0)[:20]:                                            # solved runs stay solved
        traj = np.asarray(simulate_rule_array(single_cell(W), g[i], [4, 146]))
        assert int(lifetime(traj)) == target


@pytest.mark.parametrize("backend", BACKENDS)
def test_stochastic_evolution(backend):
    """A probabilistic genome restricted to p in {0, 1} makes exactly the deterministic runs; stochastic settings still
    reach solutions that succeed on every sample of an iteration; a fresh evaluation agrees with the reported score."""
    H, W, target = 20, 31, 9
    run = lambda **kw: evolve_stochastic_batch(jax.random.PRNGKey(6), 64, single_cell(W), [4, 146], target=target,
                                               iterations=3000, num_steps=H, backend=backend, **kw)
    det = run(genome="deterministic", samples=1)
    zero_bits = run(genome="probabilistic", rule_bits=0, samples=1)
    np.testing.assert_array_equal(np.asarray(det.reliable_at), np.asarray(zero_bits.reliable_at))
    np.testing.assert_array_equal(np.asarray(det.genome), np.asarray(zero_bits.genome))
    for kw, floor in ((dict(genome="probabilistic", rule_bits=1, samples=16), 0.3),
                      (dict(genome="deterministic", epsilon=2 ** -12, samples=16), 0.9),
                      (dict(genome="deterministic", alpha=1 - 2 ** -8, samples=16), 0.9)):
        res = run(**kw)
        assert (np.asarray(res.reliable_at) > 0).mean() > floor
        g, score = np.asarray(res.genome), np.asarray(res.score)
        for i in range(4):                                                           # fresh samples: mean loss agrees
            if kw["genome"] == "probabilistic":
                sims = simulate_stochastic(jax.random.PRNGKey(100 + i), single_cell(W), [4, 146],
                                           prob=g[i] / 2 ** kw["rule_bits"], num_samples=512, backend=backend)
            else:
                sims = simulate_stochastic(jax.random.PRNGKey(100 + i), single_cell(W), [4, 146], ra=g[i],
                                           alpha=kw.get("alpha", 1.0), epsilon=kw.get("epsilon", 0.0),
                                           num_samples=512, backend=backend)
            life = np.array([int(lifetime(s)) for s in np.asarray(sims)])
            fresh = np.where(life == H + 1, H + 1, np.abs(life - target)).mean()
            assert abs(fresh - score[i]) < 3.0 + 0.5 * fresh                         # 16-sample estimate vs 512


def test_quantize_and_dynamics():
    assert list(quantize([0, 0.5, 1, 1e-9, 0.99999], 4)) == [0, 8, 16, 0, 16]
    d = Dynamics(alpha=1.0, epsilon=0.0)
    assert d.synchronous and d.noiseless
    assert Dynamics(alpha=0.5).q_alpha == 2 ** 15


@pytest.mark.parametrize("backend", BACKENDS)
def test_success_probability(backend):
    """Wolfram's twelve published lifetime-50 arrays: certain without noise, as probabilistic genomes at p in {0, 1}
    too; less than certain under noise, rule flips and asynchrony."""
    import json
    from pathlib import Path
    from stica.stochastic import success_probability
    data = json.loads((Path(__file__).parent / "data" / "wolfram_rule_arrays.json").read_text())
    ras = np.array([[[int(c) for c in r] for r in p["rule_array"]]
                    for p in data["figures"]["lifetime50_solutions"]["panels"]], np.int8)
    kw = dict(target=50, samples=64, backend=backend)
    ic = single_cell(31)
    clean = success_probability(jax.random.PRNGKey(0), ic, [4, 146], ras, **kw)
    assert np.all(np.asarray(clean.success) == 1.0) and np.all(np.asarray(clean.mean_loss) == 0.0)
    prob = success_probability(jax.random.PRNGKey(0), ic, [4, 146], ras * 16, genome="probabilistic", rule_bits=4, **kw)
    assert np.all(np.asarray(prob.success) == 1.0)
    for cond in (dict(epsilon=2 ** -6), dict(flip=2 ** -4), dict(alpha=1 - 2 ** -4)):
        noisy = success_probability(jax.random.PRNGKey(1), ic, [4, 146], ras, **kw, **cond)
        assert np.asarray(noisy.success).mean() < 1.0
