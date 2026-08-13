"""Model for 2-state Spacetime-Inhomogeneous Cellular Automata."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core.perceive import ConvPerceive, identity_kernel, neighbors_kernel

from cax.core.update.update import Update
from cax.types import Input, Perception, Rule, State

from cax.core.ca import CA, metrics_fn
from cax.utils import clip_and_uint8

import jax.numpy as jnp
from flax import nnx

import math
import jax
jax.config.update("jax_enable_x64", True)

"""
--- SRT Syntax Information ---
Numerical indices are calculated using 0 as the first index.
The SRT indexing routine within NSTICA is as follows:
- For a single cell *C*, let the state of *C* be C_state. For all states in [1, 2, ..., max_state], 
the number of Moore neighbors of *C* in each state can be expressed in a tuple, C_tuple, of the form 
(a_1, a_2, ..., a_n). Note that a_1 + a_2 + ... + a_n 
must be equal to or less than 8 in order to satisfy the Moore neighborhood condition; 
the difference between 8 and this sum is equal to the number of neighbors in the state 0.
- We order all possible tuples (a_1, a_2, ..., a_n) satisfying a_1 + a_2 + ... + a_n ≤ 8 as follows:
  * The tuples are first sorted in descending value of a_1, so the tuple (8, 0, ..., 0) is first in the list
  (index 0);
  * Tuple groups with the same value of a_1 are then sorted in descending value of a_2;
  * Tuple groups with the same values of both a_1 and a_2 are then sorted in descending value of a_3, and so on.
  * Let the resulting ordering of tuples be fin_order; let the length of this list be order_len.
  * In this ordering, we define the function Ind(C_tuple) = index_f such that the index of C_tuple within fin_order is equal to index_f.
- Each SRT raw rule (C_state, a_1, a_2, ..., a_n) -> final_state is added to the SRT array as follows:
  * The index of the rule within the array is (C_state * order_len + Ind((a_1, a_2, ..., a_n)));
  * The value of the index is final_state.
- The tuple C_tuple is then matched to its corresponding index in the SRT to determine its final state.
For example, the ordering of the tuples with (n = max_state = 2, number of states = 3) is:
[(8,0), (7,1), (7,0), (6,2), (6,1), (6,0), (5,3), (5,2), (5,1), (5,0), (4,4), (4,3), (4,2), (4,1), (4,0), (3,5),
(3,4), (3,3), (3,2), (3,1), (3,0), (2,6), (2,5), (2,4), (2,3), (2,2), (2,1), (2,0), (1,7), (1,6), (1,5), (1,4),
(1,3), (1,2), (1,1), (1,0), (0,8), (0,7), (0,6), (0,5), (0,4), (0,3), (0,2), (0,1), (0,0)], with length 45.
A sample SRT ruleset with number of states = 3 (guaranteed to be of length 3 * 45 = 135) could start with:
[1,2,0,1,0,0,2,1,0,2,2,2,1,2,0,1,2,0, ...]
The final state of a sample cell with C_state = 0 and C_tuple = (4,2) would be determined by the
value of the SRT at index (0 * 45) + 12 (12 is the index of the tuple (4,2)). This value is 1.
Thus, the final state of this cell would be 1.
"""

class NSTICAPerceive(ConvPerceive):
	"""N-State STICA Perceive class."""

	def __init__(self, rngs: nnx.Rngs, *, states: int = 2, padding: str = "SAME"):
		"""Initialize NSTICAPerceive."""
		channel_size = 1
		super().__init__(
			channel_size=channel_size,
			perception_size=2 * channel_size,
			rngs=rngs,
			kernel_size=(3, 3),
			padding=padding,
			feature_group_count=channel_size,
		)

		kernel = jnp.concatenate([identity_kernel(2), neighbors_kernel(2)], axis=-1)
		kernel = jnp.expand_dims(kernel, axis=-2)
		self.conv.kernel = nnx.Param(kernel)

		self.states = states

	@nnx.jit
	def __call__(self, state: State) -> Perception:
		"""Apply perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		convolution = jnp.array(self.conv(10**state), dtype=jnp.int64)
		alivestate = (jnp.array([jnp.round(jnp.log10(convolution[:,:,0]))]),jnp.array([convolution[:,:,1]%(10**1)]))
		for i in range(1,self.states):
			alivestate += (jnp.array([convolution[:,:,1]%(10**(i+1))-convolution[:,:,1]%(10**i)])/(10**i),)
		return jnp.concatenate(alivestate, dtype=jnp.int32)

class NSTICAUpdate(Update):
	"""N-State STICA Update class."""

	def __init__(self, time, states, rngs: nnx.Rngs):
		"""Initialize NSTICAUpdate."""
		self.srt = jnp.zeros((1, 1, 1, 18))
		self.time = jnp.array([time])
		self.states = states

	@nnx.jit
	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the STICA rules based on SRT input.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		self_state = perception[0]
		indexcontributions = ()
		for i in range(2,perception.shape[0]):
			indexcontributions += (jnp.round(jnp.array([jax.scipy.special.factorial(8-i+self.states-jnp.sum(perception[2:i+1], axis=0))/jax.scipy.special.factorial(self.states+1-i)/jax.scipy.special.factorial(7-jnp.sum(perception[2:i+1], axis=0))])),)
		position = jnp.zeros(self_state.shape).astype(jnp.int32)
		position = jnp.concatenate((jnp.array([jnp.indices(self_state.shape)[0]]), jnp.array([jnp.indices(self_state.shape)[1]])), axis=0)
		
		indices = (self_state * math.comb(7+self.states, 8)) + jnp.sum(jnp.concatenate(indexcontributions,axis=0)[0:self.states-1], axis=0).astype(jnp.int32)
		state = self.srt[self.time[0], position[0], position[1], indices]
		self.time += jnp.array([1])
		return state.reshape(state.shape[0], -1, 1)

	@nnx.jit
	def update_srt(self, srt) -> None:
		"""Update the SRT of a NSTICA from an array input.

		Args:
			srt: An NDArray with 4 dimensions and last dimension of the form n * (7+n) C 8, where n is an integer and x C y = x! / (y! * (x-y)!), ! denoting factorial.

		"""
		assert len(srt.shape) == 4, f"Expected 4 dimensions in SRT, received {len(srt.shape)}"
		assert (self.states * math.comb(7+self.states, 8) == srt.shape[3]), f"Expected {self.states * math.comb(7+self.states, 8)} indices in SRT ruleset, found {srt.shape[3]}"
		self.srt = srt

class NSTICA(CA):
	"""Generalized (N-State) Spacetime-Inhomogeneous Cellular Automata."""

	def __init__(self, time, rngs: nnx.Rngs, *, states: int = 2, metrics_fn: Callable = metrics_fn):
		"""Initialize N-state STICA."""
		assert (states >= 2), f"Number of SRT states must be at least 2 (received value: {states})"
		perceive = NSTICAPerceive(rngs=rngs, states=states)
		update = NSTICAUpdate(time=time, rngs=rngs, states=states)
		super().__init__(perceive, update, metrics_fn=metrics_fn)

	@nnx.jit
	def render(self, state: State) -> Array:
		"""Render state to RGB.

		Args:
			state: An array with two spatial/time dimensions.

		Returns:
			The rendered RGB image in uint8 format.

		"""
		rgb = jnp.repeat(state, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)