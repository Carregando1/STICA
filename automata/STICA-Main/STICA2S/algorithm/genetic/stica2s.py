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

class STICA2SPerceive(ConvPerceive):
	"""2-State STICA Perceive class."""

	def __init__(self, rngs: nnx.Rngs, *, padding: str = "SAME"):
		"""Initialize STICA2SPerceive."""
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

class STICA2SUpdate(Update):
	"""2-State STICA Update class."""

	def __init__(self, time, rngs: nnx.Rngs):
		"""Initialize STICA2SUpdate."""
		self.srt = jnp.zeros((1, 1, 1, 18))
		self.time = jnp.array([time])

	def __call__(self, state: State, perception: Perception, input: Input | None = None) -> State:
		"""Apply the 2-state STICA rules based on SRT input.

		Args:
			state: Current state of the cellular automaton.
			perception: Perceived state, including cell state and neighbor count.
			input: Input to the cellular automaton (unused in this implementation).

		Returns:
			Updated state of the cellular automaton.

		"""
		self_alive = perception[..., 0:1]
		num_alive_neighbors = perception[..., 1:2].astype(jnp.int32)
		position = jnp.zeros(self_alive.shape[:-1]).astype(jnp.int32)
		position = jnp.concatenate((jnp.indices(self_alive.shape[:-1])[0].reshape(self_alive.shape[:-1]+(1,)), jnp.indices(self_alive.shape[:-1])[1].reshape(self_alive.shape[:-1]+(1,))), axis=2)
		state = self.srt[self.time[0], position[:,:,0], position[:,:,1], (9*self_alive.reshape(self_alive.shape[0], -1)[position[:,:,0], position[:,:,1]]+num_alive_neighbors.reshape(num_alive_neighbors.shape[0], -1)[position[:,:,0], position[:,:,1]]).astype(jnp.int32)]
		self.time += jnp.array([1])
		return state.reshape(state.shape[0], -1, 1)

	@nnx.jit
	def update_srt(self, srt) -> None:
		"""Update the SRT of a 2-state STICA from an array input.

		Args:
			srt: An NDArray with 4 dimensions and last dimension of size 18.

		"""
		assert len(srt.shape) == 4, f"Expected 4 dimensions in SRT, received {len(srt.shape)}"
		assert srt.shape[3] == 18, f"Expected 18 rules in SRT, received {srt.shape[3]}"
		self.srt = srt

class STICA2S(CA):
	"""2-state Spacetime-Inhomogeneous Cellular Automata."""

	def __init__(self, time, rngs: nnx.Rngs, *, metrics_fn: Callable = metrics_fn):
		"""Initialize 2-state STICA."""
		perceive = STICA2SPerceive(rngs=rngs)
		update = STICA2SUpdate(time=time, rngs=rngs)
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