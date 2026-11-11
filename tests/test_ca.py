import numpy as np
import jax.numpy as jnp
import pytest

from stica import STICA, initial_state, num_rules, simulate
from reference import simulate_reference

rng = np.random.default_rng(1)


def random_case(n, G=11, T=5, dense=False, shape=None):
    p = np.r_[0.05, np.full(n - 1, 0.95 / (n - 1))] if dense else None
    ic = rng.choice(n, (G, G), p=p)
    srt = rng.integers(0, n, shape or (T, G, G, num_rules(n)))
    return ic, srt


@pytest.mark.parametrize("n", [2, 3, 4, 6])
@pytest.mark.parametrize("dense", [False, True])
def test_simulate_matches_reference(n, dense):
    ic, srt = random_case(n, dense=dense)
    got = simulate(jnp.asarray(ic, jnp.int8), jnp.asarray(srt, jnp.int8), n)
    np.testing.assert_array_equal(np.asarray(got), simulate_reference(ic, srt, n))


@pytest.mark.parametrize("n", [2, 3])
def test_cax_model_matches_simulate(n):
    ic, srt = random_case(n, dense=True)
    ca = STICA(srt, num_states=n)
    np.testing.assert_array_equal(np.asarray(ca.run(ic)), np.asarray(simulate(jnp.asarray(ic), jnp.asarray(srt), n)))


def test_step_by_step_time_channel_and_resume():
    n, T = 3, 6
    ic, srt = random_case(n, T=T)
    ref = simulate_reference(ic, srt, n)
    ca = STICA(srt, num_states=n)
    state = initial_state(ic)
    for t in range(T):
        state, cells = ca.step(state)
        np.testing.assert_array_equal(np.asarray(cells), ref[t + 1])
        assert np.all(np.asarray(state[..., 1]) == t + 1)
    half, _ = ca(initial_state(ic), num_steps=2)
    resumed, _ = ca(half, num_steps=T - 2)
    np.testing.assert_array_equal(np.asarray(resumed[..., 0]), ref[-1])
    np.testing.assert_array_equal(np.asarray(ca.run(np.asarray(half[..., 0]), t0=2))[-1], ref[-1])


def test_set_srt_replaces_rules():
    n = 2
    ic, srt_a = random_case(n)
    _, srt_b = random_case(n)
    ca = STICA(srt_a, num_states=n)
    ca.run(ic)
    ca.set_srt(srt_b)
    np.testing.assert_array_equal(np.asarray(ca.run(ic)), simulate_reference(ic, srt_b, n))


@pytest.mark.parametrize("shape", ["time", "space", "none"])
def test_shared_rule_tensors(shape):
    n, G, T = 3, 9, 5
    R = num_rules(n)
    full = {"time": (T, 1, 1, R), "space": (1, G, G, R), "none": (1, 1, 1, R)}[shape]
    ic, srt = random_case(n, G=G, T=T, shape=full)
    ref = simulate_reference(ic, srt, n, num_steps=T)
    got = simulate(jnp.asarray(ic, jnp.int8), jnp.asarray(srt, jnp.int8), n, num_steps=T)
    np.testing.assert_array_equal(np.asarray(got), ref)
    np.testing.assert_array_equal(np.asarray(STICA(srt, num_states=n).run(ic, num_steps=T)), ref)


def test_game_of_life_as_homogeneous_stica():
    """A (1,1,1,18) SRT is a homogeneous CA; Conway's rule must move a glider by (1, 1) every 4 steps."""
    life = np.zeros(18, int)
    life[8 - 3] = 1                    # dead cell, 3 live neighbours -> born   (index 9*s + 8 - count)
    life[9 + 8 - 2] = life[9 + 8 - 3] = 1  # live cell, 2 or 3 live neighbours -> survives
    ic = np.zeros((10, 10), int)
    ic[1, 2] = ic[2, 3] = ic[3, 1] = ic[3, 2] = ic[3, 3] = 1
    traj = np.asarray(STICA(life.reshape(1, 1, 1, 18), num_states=2).run(ic, num_steps=8))
    np.testing.assert_array_equal(traj[4], np.roll(ic, (1, 1), axis=(0, 1)))
    np.testing.assert_array_equal(traj[8], np.roll(ic, (2, 2), axis=(0, 1)))


def test_cells_freeze_after_last_rule_step():
    n, T = 2, 3
    ic, srt = random_case(n, T=T)
    ca = STICA(srt, num_states=n)
    final, cells = ca(initial_state(ic), num_steps=T + 2)
    np.testing.assert_array_equal(np.asarray(cells[T - 1]), np.asarray(cells[T + 1]))
    assert int(final[0, 0, 1]) == T + 2
    with pytest.raises(ValueError, match="cannot run"):
        ca.run(ic, num_steps=T + 1)


def test_time_shared_rules_need_num_steps():
    ic, srt = random_case(2, shape=(1, 11, 11, 18))
    with pytest.raises(ValueError, match="num_steps is required"):
        STICA(srt, num_states=2).run(ic)


def test_render():
    ic, srt = random_case(4)
    ca = STICA(srt, num_states=4)
    img = ca.render(initial_state(ic))
    assert img.shape == ic.shape + (3,) and img.dtype == jnp.uint8
    assert np.all(np.asarray(img)[ic == 0] == 255)
