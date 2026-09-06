#%pip install -U "jax[cuda12]"
#%pip install -U "cax"
import jax
import jax.numpy as jnp
from flax import nnx
from .nstica import NSTICA

def transition(ic, srt, steps, states, generated):
	"""
	transition.py: Takes in an IC and SRT and outputs the state after {steps} steps of simulation of an NSTICA.
	Args:
	ic: a 2D square array with only 0s and 1s.
	srt: a 4D array with dims (time, width, height, rules), with rules equal to states * ((7 + states)C8), where xCy = x!/(y! * (x-y)!)
	steps: The number of steps the NSTICA is run. Must be less than srt.shape[0].
	states: The number of states of the NSTICA.
	generated: Result of the NSTICA progression, if already computed.
	"""
	if (generated is None):
		seed = 0
		num_steps = steps
		time = 0
		rngs = nnx.Rngs(seed)

		ca = NSTICA(time=time, rngs=rngs, states=states)
		state_init = ic.reshape(ic.shape[0], -1, 1).astype(float)
		srt = srt.astype(float)
		ca.update.update_srt(srt=srt)
		assert num_steps <= srt.shape[0], f"Requested time {num_steps} is out of bounds, max time {srt.shape[0]}"
		state_final, states = ca(state=state_init, num_steps=num_steps)
		states = jnp.concatenate([(state_init.astype(int)).reshape(1, ic.shape[0], -1), (states.astype(int)).reshape(states.shape[0], ic.shape[0], -1)])
		return states
	return generated