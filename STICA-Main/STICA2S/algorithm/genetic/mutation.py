#Mutation: Degeneralized to strictly 2 state STICA.

from typing import List

import numpy as np
from numpy._typing import NDArray

from algorithm.genetic.state import CurrentState

"""Mutation classes except for MutationSet no longer needed; replaced by list objects."""

class MutationSet:
    """
    A class storing a vector of ICMutation objects and SRTMutation objects, that is, a set of mutations being applied at once.
    """
    def __init__(self):
        self.ic_mutations = []
        self.srt_mutations = []

class Mutator:
    def mutate(self, state: CurrentState) -> tuple[CurrentState, MutationSet]:
        """
        This function should return a new mutation that we could try.
        """
        raise NotImplementedError

    def init_state(self) -> CurrentState:
        """
        This function should return some initial configuration that our genetic
        algorithm can work with.
        """
        raise NotImplementedError


class RulesetMutator(Mutator):
    """
    Ruleset Mutator:
    Starts off with a finite set of rules to start with (e.g. toggle between
    conway's game of life and seed)
    """

    def __init__(
        self,
        rules: List[NDArray],
        grid_size: int = 32,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
        rule_mutate_p: float = 1 / 3,
        strict: bool = False,
        num_strict: bool = True,
        ic_ct: int = 20,
        srt_ct: int = 240,
        ic_enable: bool = True,
        srt_enable: bool = True,
    ):
        """
        Initializes the ruleset mutator.

        :param List[] rules: the set of rules to apply to the grid. Ruleset object is no longer defined so the type within the List should be NDArray.

        :param int grid_size: the size of the grid that our mutator will work on.

        :param List[float] state_init_p: the probability that we should choose a particular
        state when initializing (it MUST have the same length as the number of
        states in all the elements).

        :param float mutate_p: the probability that we mutate any cell, which
        include the cells representing the initial configuration space and the
        cells for each ruleset.

        :param float rule_mutate_p: the probability that we mutate a given sub-rule, given that we have selected the rule for modification.

        :param bool strict: if true, SRT cells will be chosen then SRT rules within the chosen cells will be mutated, instead of choosing arbitrary SRT rules for mutation.

        The following 3 params only work for ArbitraryRulesetMutator:

        :param bool num_strict: if true, the number of IC and SRT mutations are respectively held constant; if false, the number of IC or SRT mutations may vary probabilistically.

        :param int ic_ct: the number of IC mutations, applicable iff num_strict is true.

        :param int srt_ct: the number of SRT mutations, applicable iff num_strict is true.

        These two params work for both mutator types:

        :param bool ic_enable: if true, the IC will be mutated; if false, the IC will be unchanged.

        :param bool srt_enable: if true, the SRT will be mutated; if false, the SRT will be unchanged.
        """
        # initialize grid size
        self.grid_size = grid_size

        # initialize rules array
        self.rules: List[NDArray[np.int32]] = []
        for rule in rules:
            self.rules.append(rule)

        # States parameter removed with degeneralization.

        # initialize strict mode: 
        # True = cells in the SRT are selected for mutation then the individual rulesets of those cells are mutated;
        # False = individual rulesets are selected directly for mutation regardless of cells
        self.strict = strict

        assert (
            state_init_p is None or len(state_init_p) == 2
        ), "The length of the state probs array does not equal the number of valid states!"
        self.state_init_p = state_init_p

        self.mutate_p = mutate_p

        self.rule_mutate_p = rule_mutate_p

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

    def init_state(self) -> CurrentState:
        ruleindices = np.array(
            np.random.choice(
                range(len(self.rules)),
                size=(self.grid_size - 1, self.grid_size, self.grid_size),
            )
        )
        # initialize the rules based on the rule indices above.
        rules = (np.array(self.rules)[ruleindices.tolist()])
        initial = np.random.choice(
            [0,1],
            size=(self.grid_size, self.grid_size),
            p=self.state_init_p,
        )
        return CurrentState(rules=rules, initial=initial, ruleindices=ruleindices)

    def mutate(self, state: CurrentState) -> tuple[CurrentState, MutationSet]:
        """
        Mutates an existing state to a new state stochastically.
        """
        assert not (
            state.ruleindices is None
        ), "You cannot supply a state with no initialized ruleindices"

        new_state = CurrentState(
            initial=state.initial.copy(),
            rules=state.rules.copy(),
            ruleindices=state.ruleindices,
        )
        assert not (new_state.ruleindices is None), "Impossible"

        mutations = MutationSet()
        
        if len(self.rules) <= 1:
            return (new_state, mutations)

        #Mutate initials like Arbitrary
        if (self.ic_enable):
            new_state.initial = state.initial ^ (np.random.rand(state.initial.shape[0],state.initial.shape[1]) < self.mutate_p)
            a = np.nonzero(new_state.initial != state.initial)
            mutations.ic_mutations = np.concatenate((np.transpose(a), state.initial[a].reshape(-1,1), new_state.initial[a].reshape(-1,1)), axis=1).tolist()
        
        #Mutate ruleindices then declare rules array
        #Strict mode means SRT mut locs must equal IC mut locs; not valid with nonarb mutator
        
        if (self.srt_enable):
            new_state.ruleindices = state.ruleindices + (np.random.rand(state.ruleindices.shape[0],state.ruleindices.shape[1],state.ruleindices.shape[2]) < self.mutate_p) * np.random.randint(1, len(self.rules))
            new_state.ruleindices = new_state.ruleindices%len(self.rules)
            new_state.rules = np.array(self.rules)[new_state.ruleindices.tolist()]
            a = np.nonzero(new_state.rules != state.rules)
            mutations.srt_mutations = np.concatenate((np.transpose(a), state.rules[a].reshape(-1,1), new_state.rules[a].reshape(-1,1)), axis=1).tolist()

        return (new_state, mutations)


class ArbitraryRulesetMutator(RulesetMutator):
    """
    Starts off with an initial condition consisting of some finite set of rules,
    and then arbitrarily mutates rules.
    """

    def __init__(
        self,
        rules: List[NDArray],
        grid_size: int = 32,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
        rule_mutate_p: float = 1 / 3,
        strict: bool = False,
        num_strict: bool = True,
        ic_ct: int = 20,
        srt_ct: int = 240,
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

        :param bool ic_enable: if true, the IC will be mutated; if false, the IC will be unchanged.

        :param bool srt_enable: if true, the SRT will be mutated; if false, the SRT will be unchanged.
        """
        super().__init__(
            rules, grid_size=grid_size, state_init_p=state_init_p, mutate_p=mutate_p, strict=strict, num_strict = num_strict, ic_ct = ic_ct, srt_ct = srt_ct, ic_enable = ic_enable, srt_enable = srt_enable
        )
        self.rule_mutate_p = rule_mutate_p

    def mutate(self, state: CurrentState) -> tuple[CurrentState, MutationSet]:
        """
        Mutates an existing state to a new state stochastically.
        """
        new_state = CurrentState(
            initial=state.initial.copy(),
            rules=state.rules.copy(),
        )

        # this logic is shared with RulesetMutator
        mutations = MutationSet()
        
        """Use of ICMutation and SRTMutation objects are now obsolete to remove all for loops."""

        if (self.num_strict):
            #Sets up exactly {self.ic_ct} random mutations for the IC
            assert state.initial.shape[0] * state.initial.shape[1] >= self.ic_ct, f"Number of IC mutations must be less than or equal to number of IC cells\nAvailable IC cells: {state.initial.shape[0] * state.initial.shape[1]}, requested # of mutations: {self.ic_ct}"
            ic_rands = np.unique(np.random.randint(0, state.initial.shape[0] * state.initial.shape[1], self.ic_ct))
            while (ic_rands.shape[0] != self.ic_ct):
                ic_rands = np.unique(np.concatenate((ic_rands, (np.random.randint(0, state.initial.shape[0] * state.initial.shape[1], self.ic_ct - ic_rands.shape[0])))))
            ic_rands = np.concatenate((np.expand_dims(ic_rands%(state.initial.shape[0]), -1), np.expand_dims(np.floor(ic_rands/(state.initial.shape[0])), -1)), axis=1)
            ic_shift = np.zeros_like(state.initial)
            ic_shift[(np.transpose(ic_rands)[0].astype(int), np.transpose(ic_rands)[1].astype(int))] = 1
            
            #Sets up exactly {self.srt_ct} random mutations for the SRT
            assert state.rules.shape[0] * state.rules.shape[1] * state.rules.shape[2] * state.rules.shape[3] >= self.srt_ct, f"Number of SRT mutations must be less than or equal to number of SRT cells\nAvailable SRT cells: {state.rules.shape[0] * state.rules.shape[1] * state.rules.shape[2] * state.rules.shape[3]}, requested # of mutations: {self.srt_ct}"
            srt_rands = np.unique(np.random.randint(0, state.rules.shape[0] * state.rules.shape[1] * state.rules.shape[2] * state.rules.shape[3], self.srt_ct))
            while (srt_rands.shape[0] != self.srt_ct):
                srt_rands = np.unique(np.concatenate((srt_rands, (np.random.randint(0, state.rules.shape[0] * state.rules.shape[1] * state.rules.shape[2] * state.rules.shape[3], self.srt_ct - srt_rands.shape[0])))))
            srt_rands = np.concatenate((np.expand_dims(srt_rands%(state.rules.shape[0]), -1), np.expand_dims(np.floor(srt_rands/(state.rules.shape[0]))%state.rules.shape[1], -1), np.expand_dims(np.floor(srt_rands/(state.rules.shape[0] * state.rules.shape[1]))%state.rules.shape[2], -1), np.expand_dims(np.floor(srt_rands/(state.rules.shape[0]*state.rules.shape[1]*state.rules.shape[2])), -1)), axis=1)
            srt_shift = np.zeros_like(state.rules)
            srt_shift[(np.transpose(srt_rands)[0].astype(int), np.transpose(srt_rands)[1].astype(int), np.transpose(srt_rands)[2].astype(int), np.transpose(srt_rands)[3].astype(int))] = 1
        else:
            ic_shift = (np.random.rand(state.initial.shape[0],state.initial.shape[1]) < self.mutate_p)
            if (self.strict):
                #Chooses SRT cells to mutate with probability mutate_p,
                #then chooses individual rules within the chosen SRT cells to mutate with probability rule_mutate_p
                srt_shift = (np.random.rand(state.rules.shape[0], state.rules.shape[1], state.rules.shape[2], 1) < self.mutate_p) * (np.random.rand(state.rules.shape[0], state.rules.shape[1], state.rules.shape[2], state.rules.shape[3]) < self.rule_mutate_p)
            else:
                #Selects arbitrary SRT cells with probability rule_mutate_p
                srt_shift = (np.random.rand(state.rules.shape[0],state.rules.shape[1],state.rules.shape[2],state.rules.shape[3]) < self.rule_mutate_p)
            
        #Mutate initials.
        if (self.ic_enable):
            new_state.initial = state.initial ^ ic_shift
            a = np.nonzero(new_state.initial != state.initial)
            mutations.ic_mutations = np.concatenate((np.transpose(a), state.initial[a].reshape(-1,1), new_state.initial[a].reshape(-1,1)), axis=1).tolist()
        #Mutate SRT.
        if (self.srt_enable):
            new_state.rules = state.rules ^ srt_shift
            a = np.nonzero(new_state.rules != state.rules)
            mutations.srt_mutations = np.concatenate((np.transpose(a), state.rules[a].reshape(-1,1), new_state.rules[a].reshape(-1,1)), axis=1).tolist()
        return (new_state, mutations)
