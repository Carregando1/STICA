from typing import List

import numpy as np
from numpy._typing import NDArray

from EditedCode.genetic.state import CurrentState
from EditedCode.ruleset import RuleSet

class Mutation:
    """
    A generic class that stores the position of the cell being mutated, its old state, and its new state.
    """
    def __init__(self, cell: tuple, old, new):
        self.cell_pos = cell
        self.old_state = old
        self.new_state = new

class ICMutation(Mutation):
    """
    A class storing the position of a cell in the IC tensor that's being mutated, as well as its initial state and its new state.
    """
    def __init__(self, cell: tuple[int, int], old: int, new: int):
        super().__init__(cell, old, new)

class SRTMutation(Mutation):
    """
    A class storing the position of a cell in the SRT tensor that's being mutated, as well as its initial state and its new state.
    """
    def __init__(self, cell: tuple[int, int, int], old: NDArray[np.int32], new: NDArray[np.int32]):
        super().__init__(cell, old, new)


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
        rules: List[RuleSet],
        grid_size: int = 32,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
    ):
        """
        Initializes the ruleset mutator.

        :param List[Ruleset] rules: the set of rules to apply to the grid.

        :param int grid_size: the size of the grid that our mutator will work on.

        :param List[float] state_init_p: the probability that we should choose a particular
        state when initializing (it MUST have the same length as the number of
        states in all the elements).

        :param float mutate_p: the probability that we mutate any cell, which
        include the cells representing the initial configuration space and the
        cells for each ruleset.
        """
        # initialize grid size
        self.grid_size = grid_size

        # initialize rules array
        self.rules: List[NDArray[np.int32]] = []
        for rule in rules:
            self.rules.append(rule.nparray())

        # initialize states
        states = set()
        for rule in rules:
            states.add(rule.states)
        assert (
            len(states) == 1
        ), "Rulesets do not all require the same number of states!"
        self.states = states.pop()

        assert (
            state_init_p is None or len(state_init_p) == self.states
        ), "The length of the state probs array does not equal the number of valid states!"
        self.state_init_p = state_init_p

        self.mutate_p = mutate_p

    def init_state(self) -> CurrentState:
        ruleindices = np.array(
            np.random.choice(
                range(len(self.rules)),
                size=(self.grid_size - 1, self.grid_size, self.grid_size),
            )
        )
        # initialize the rules based on the rule indices above.
        rules = np.array(
            [
                [
                    [
                        self.rules[ruleindices[i][j][k]]
                        for k in range(ruleindices.shape[2])
                    ]
                    for j in range(ruleindices.shape[1])
                ]
                for i in range(ruleindices.shape[0])
            ]
        )
        initial = np.random.choice(
            range(self.states),
            size=(self.grid_size, self.grid_size),
            p=self.state_init_p,
        )
        return CurrentState(rules=rules, initial=initial, ruleindices=ruleindices)

    def should_mutate(self):
        """
        Method for whether we should mutate a cell.
        """
        return np.random.random() < self.mutate_p

    def mutate_initial(self, state: CurrentState, new_state: CurrentState) -> List[ICMutation]:
        """
        Mutates the initial condition based on the conditions of the mutator.
        """
        ic_mutations = []
        for i in range(state.initial.shape[0]):
            for j in range(state.initial.shape[1]):
                if self.should_mutate():
                    ic_mutations.append(self.mutate_initial_pos((i,j), state, new_state))
        
        return ic_mutations

    def mutate_initial_pos(self, pos: List[int], state: CurrentState, new_state: CurrentState) -> ICMutation:
        """
        Mutates a specific cell of the IC tensor and returns an ICMutation object.
        
        :param List[int] pos: the position of the cell whose initial state should be mutated, 2 numbers
        
        :param CurrentState state: the initial state of the automaton

        :param CurrentState new_state: the new state of the automaton, after mutating the specific cell

        :returns ICMutation: the ICMutation object containing the information of the mutation of the IC tensor
        
        """
        i, j = pos[0], pos[1]
        options = [
            x for x in range(self.states) if x != state.initial[i][j]
        ]
        new_state.initial[i][j] = np.random.choice(options)
        return ICMutation((i,j),state.initial[i][j], new_state.initial[i][j])

    def mutate_pos(self, pos: List[int], state: CurrentState, new_state: CurrentState) -> SRTMutation:
        """
        Mutates a specific cell of the SRT tensor and returns an SRTMutation object.
        
        :param List[int] pos: the position of the cell (space & time) whose ruleset should be mutated, 3 numbers
        
        :param CurrentState state: the initial state of the automaton

        :param CurrentState new_state: the new state of the automaton, after mutating the specific cell

        :returns SRTMutation: the SRTMutation object containing the information of the mutation of the IC tensor
        
        """
        i, j, k = pos[0], pos[1], pos[2]
        indexoptions = [
            x
            for x in range(len(self.rules))
            if x != state.ruleindices[i][j][k]
        ]
        indexchoice = np.random.choice(indexoptions)
        new_state.ruleindices[i][j][k] = indexchoice
        new_state.rules[i][j][k] = self.rules[indexchoice]
        return SRTMutation((i,j,k),state.rules[i][j][k],new_state.rules[i][j][k])

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
        
        mutations.ic_mutations.extend(self.mutate_initial(state, new_state))

        if len(self.rules) <= 1:
            return (new_state, mutations)

        # we are working with at least two rules, so we will always have a choice.
        for i in range(state.rules.shape[0]):
            for j in range(state.rules.shape[1]):
                for k in range(state.rules.shape[2]):
                    if self.should_mutate():
                        mutations.srt_mutations.append(self.mutate_pos((i,j,k), state, new_state))

        return (new_state, mutations)


class ArbitraryRulesetMutator(RulesetMutator):
    """
    Starts off with an initial condition consisting of some finite set of rules,
    and then arbitrarily scrambles rules.
    """

    def __init__(
        self,
        rules: List[RuleSet],
        grid_size: int = 32,
        state_init_p: List[float] | None = None,
        mutate_p: float = 1 / (32**2),
        rule_mutate_p: float = 1 / 3,
    ):
        """
        Initializes the arbitrary ruleset mutator.

        :param List[Ruleset] rules: the set of rules to apply to the grid.

        :param int grid_size: the size of the grid that our mutator will work on.

        :param List[float] state_init_p: the probability that we should choose a particular state when initializing (it MUST have the same length as the number of states in all the elements).

        :param float mutate_p: the probability that we mutate any cell, which include the cells representing the initial configuration space and the cells for each ruleset.

        :param float rule_mutate_p: the probability that we mutate a given sub-rule, given that we have selected the rule for modification.

        """
        super().__init__(
            rules, grid_size=grid_size, state_init_p=state_init_p, mutate_p=mutate_p
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
        
        ic_mutations = self.mutate_initial(state, new_state)
        mutations.ic_mutations.extend(ic_mutations)

        for i in range(state.rules.shape[0]):
            for j in range(state.rules.shape[1]):
                for k in range(state.rules.shape[2]):
                    if not self.should_mutate():
                        continue
                    # mutate the rules corresponding to that cell.
                    for ruleidx in range(state.rules.shape[3]):
                        if np.random.random() > self.rule_mutate_p:
                            continue
                        options = [
                            x
                            for x in range(self.states)
                            if x != state.rules[i][j][k][ruleidx][self.states + 1]
                        ]
                        choice = np.random.choice(options)
                        new_state.rules[i][j][k][ruleidx][self.states + 1] = choice
                    mutations.srt_mutations.append(SRTMutation((i,j,k),state.rules[i][j][k],new_state.rules[i][j][k]))

        return (new_state, mutations)
