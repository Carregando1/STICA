#%pip install -U "jax[cuda12]"
#%pip install -U "cax"
import jax
import jax.numpy as jnp
from flax import nnx
from .stica2s import STICA2S

def transition(ic, srt, steps):
	"""
	transition.py: Takes in an IC and SRT of a strictly 2-state STICA and outputs the state after {steps} steps.
	ic: a 2D square array with only 0s and 1s.
	srt: a 4D array with dims (time, width, height, 18), the 18 derived from 2 cell states * 9 neighbor states.
	steps: The number of steps the 2-state STICA is run. Must be less than time = srt.shape[0].
	"""

	seed = 0
	num_steps = steps
	time = 0
	rngs = nnx.Rngs(seed)

	ca = STICA2S(time=time, rngs=rngs)
	state_init = ic.reshape(ic.shape[0], -1, 1).astype(float)
	srt = srt.astype(float)
	ca.update.update_srt(srt=srt)
	assert num_steps <= srt.shape[0], f"Requested time {num_steps} is out of bounds, max time allowed is {srt.shape[0]}"
	state_final, states = ca(state=state_init, num_steps=num_steps)
	states = jnp.concatenate([(state_init.astype(int)).reshape(1, ic.shape[0], -1), (states.astype(int)).reshape(states.shape[0], ic.shape[0], -1)])
	return states