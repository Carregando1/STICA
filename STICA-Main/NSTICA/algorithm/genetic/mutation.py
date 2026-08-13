"""Mutation: Generalized to NSTICA and adapted for the removal of state.py."""

from typing import List

import numpy as np
from numpy._typing import NDArray

import math
import jax
import jax.numpy as jnp

"""MutationSet is obsolete, replaced with a two-membered tuple."""

class Mutator:
    def mutate(self, state: tuple[NDArray, NDArray, int, NDArray | None]) -> tuple[tuple[NDArray, NDArray, int, NDArray | None], tuple[List, List]]:
        """
        This function should return a new mutation that we could try.
        """
        raise NotImplementedError

    def init_state(self) -> tuple[NDArray, NDArray, int, NDArray | None]:
        """
        This function should return some initial configuration that our genetic
        algorithm can work with.
        """
        raise NotImplementedError

class ArbitraryRulesetMutator(Mutator):
    """
    Starts off with a random initial condition (IC) with an empty ruleset (SRT),
    and then arbitrarily mutates the initial condition and/or ruleset.
    """

    def __init__(
        self,
        grid_size: int = 32,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
        rule_mutate_p: float = 1 / 3,
        strict: bool = False,
        num_strict: bool = True,
        ic_ct: int = 20,
        srt_ct: int = 240,
        states: int = 2,
        ic_enable: bool = True,
        srt_enable: bool = True,
    ):
        """
        Initializes the arbitrary ruleset mutator.

        :param List[] rules: the set of rules to apply to the grid.

        :param int grid_size: the size of the grid that our mutator will work on.

        :param List[float] state_init_p: the probability that we should choose a particular state when initializing (its length MUST be equal to the value of the states parameter, see below).

        :param float mutate_p: the probability that we mutate any cell, which include the cells representing the initial configuration space and the cells for each ruleset.

        :param float rule_mutate_p: the probability that we mutate a given sub-rule, given that we have selected the rule for modification.

        :param bool strict: if true, SRT cells will be chosen then SRT rules within the chosen cells will be mutated, instead of choosing arbitrary SRT rules for mutation.

        :param bool num_strict: if true, the number of IC and SRT mutations are respectively held constant; if false, the number of IC or SRT mutations may vary probabilistically.

        :param int ic_ct: the number of IC mutations, applicable iff num_strict is true.

        :param int srt_ct: the number of SRT mutations, applicable iff num_strict is true.

        :param int states: the number of states of the NSTICA; must be equal to the length of the parameter state_init_p.

        :param bool ic_enable: if true, the IC will be mutated; if false, the IC will be unchanged.

        :param bool srt_enable: if true, the SRT will be mutated; if false, the SRT will be unchanged.
        """
        # initialize grid size
        self.grid_size = grid_size

        # initialize states
        self.states = states

        # initialize strict mode: 
        # True = cells in the SRT are selected for mutation then the individual rulesets of those cells are mutated;
        # False = individual rulesets are selected directly for mutation regardless of cells
        self.strict = strict

        # initialize num_strict mode: 
        # True = number of IC/SRT mutations must be constant;
        # False = number of IC/SRT mutations may vary according to probability
        self.num_strict = num_strict

        # initialize target number of IC/SRT mutations
        self.ic_ct = ic_ct
        self.srt_ct = srt_ct

        # initialize ICmutation and SRTmutation permissions
        self.ic_enable = ic_enable
        self.srt_enable = srt_enable

        assert (
            state_init_p is None or len(state_init_p) == states
        ), "The length of the state probs array does not equal the number of valid states!"
        self.state_init_p = state_init_p

        self.mutate_p = mutate_p

        self.rule_mutate_p = rule_mutate_p

    def init_state(self) -> tuple[NDArray, NDArray, int, NDArray | None]:
        # initialize the initial based on state_init_p and the rules as a zeros array.
        rules = np.zeros((self.grid_size - 1, self.grid_size, self.grid_size, self.states * math.comb(7+self.states, 8)))
        initial = np.random.choice(
            range(self.states),
            size=(self.grid_size, self.grid_size),
            p=self.state_init_p,
        )
        return (initial, rules, self.states, None)

    def mutate(self, state: tuple[NDArray, NDArray, int, NDArray | None]) -> tuple[tuple[NDArray, NDArray, int, NDArray | None], tuple[List, List]]:
        """
        Mutates an existing state to a new state stochastically.
        """

        ic_mutations = []
        srt_mutations = []
        
        if (self.num_strict):
            #Sets up exactly {self.ic_ct} random mutations for the IC
            assert state[0].shape[0] * state[0].shape[1] >= self.ic_ct, f"Number of IC mutations must be less than or equal to number of IC cells\nAvailable IC cells: {state[0].shape[0] * state[0].shape[1]}, requested # of mutations: {self.ic_ct}"
            ic_rands = np.unique(np.random.randint(0, state[0].shape[0] * state[0].shape[1], self.ic_ct))
            while (ic_rands.shape[0] != self.ic_ct):
                ic_rands = np.unique(np.concatenate((ic_rands, (np.random.randint(0, state[0].shape[0] * state[0].shape[1], self.ic_ct - ic_rands.shape[0])))))
            ic_rands = np.concatenate((np.expand_dims(ic_rands%(state[0].shape[0]), -1), np.expand_dims(np.floor(ic_rands/(state[0].shape[0])), -1), np.expand_dims(np.random.randint(1, self.states, self.ic_ct), -1)), axis=1)
            ic_shift = np.zeros_like(state[0])
            ic_shift[(np.transpose(ic_rands)[0].astype(int), np.transpose(ic_rands)[1].astype(int))] = ic_rands[:, 2]
            
            #Sets up exactly {self.srt_ct} random mutations for the SRT
            assert state[1].shape[0] * state[1].shape[1] * state[1].shape[2] * state[1].shape[3] >= self.srt_ct, f"Number of SRT mutations must be less than or equal to number of SRT cells\nAvailable SRT cells: {state[1].shape[0] * state[1].shape[1] * state[1].shape[2] * state[1].shape[3]}, requested # of mutations: {self.srt_ct}"
            srt_rands = np.unique(np.random.randint(0, state[1].shape[0] * state[1].shape[1] * state[1].shape[2] * state[1].shape[3], self.srt_ct))
            while (srt_rands.shape[0] != self.srt_ct):
                srt_rands = np.unique(np.concatenate((srt_rands, (np.random.randint(0, state[1].shape[0] * state[1].shape[1] * state[1].shape[2] * state[1].shape[3], self.srt_ct - srt_rands.shape[0])))))
            srt_rands = np.concatenate((np.expand_dims(srt_rands%(state[1].shape[0]), -1), np.expand_dims(np.floor(srt_rands/(state[1].shape[0]))%state[1].shape[1], -1), np.expand_dims(np.floor(srt_rands/(state[1].shape[0] * state[1].shape[1]))%state[1].shape[2], -1), np.expand_dims(np.floor(srt_rands/(state[1].shape[0]*state[1].shape[1]*state[1].shape[2])), -1), np.expand_dims(np.random.randint(1, self.states, self.srt_ct), -1)), axis=1)
            srt_shift = np.zeros_like(state[1])
            srt_shift[(np.transpose(srt_rands)[0].astype(int), np.transpose(srt_rands)[1].astype(int), np.transpose(srt_rands)[2].astype(int), np.transpose(srt_rands)[3].astype(int))] = srt_rands[:, 4]
        else:
            ic_shift = ((np.random.rand(state[0].shape[0],state[0].shape[1]) < self.mutate_p)*np.random.randint(1, self.states, size=state[0].shape))
            if (self.strict):
                #Chooses SRT cells to mutate with probability mutate_p,
                #then chooses individual rules within the chosen SRT cells to mutate with probability rule_mutate_p
                srt_shift = (np.random.rand(state[1].shape[0], state[1].shape[1], state[1].shape[2], 1) < self.mutate_p) * (np.random.rand(state[1].shape[0], state[1].shape[1], state[1].shape[2], state[1].shape[3]) < self.rule_mutate_p) * np.random.randint(1, self.states, size=state[1].shape)
            else:
                #Selects arbitrary SRT cells with probability rule_mutate_p
                srt_shift = (np.random.rand(state[1].shape[0],state[1].shape[1],state[1].shape[2],state[1].shape[3]) < self.rule_mutate_p)*np.random.randint(1, self.states, size=state[1].shape)
            
        #Use of ICMutation and SRTMutation objects are obsolete.

        #Mutate initials.
        if (self.ic_enable):
            new_ic = state[0] + ic_shift
            new_ic = new_ic%self.states
            a = np.nonzero(new_ic != state[0])
            ic_mutations = np.concatenate((np.transpose(a), state[0][a].reshape(-1,1), new_ic[a].reshape(-1,1)), axis=1).tolist()

        #Mutate SRT.
        if (self.srt_enable):
            new_srt = state[1] + srt_shift
            new_srt = new_srt%self.states
            a = np.nonzero(new_srt != state[1])
            srt_mutations = np.concatenate((np.transpose(a), state[1][a].reshape(-1,1), new_srt[a].reshape(-1,1)), axis=1).tolist()

        #Return new NSTICA state as well as the set of mutations applied.
        return ((new_ic, new_srt, state[2], state[3]), (ic_mutations, srt_mutations))
