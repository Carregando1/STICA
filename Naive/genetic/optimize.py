from typing import Callable, List

from numpy import int32
from numpy._typing import NDArray

from EditedCode.genetic.mutation import Mutator, MutationSet
from EditedCode.genetic.state import CurrentState

class Optimizer:
    """
    An Optimizer class which puts all the components of our genetic algorithm together.
    """

    def __init__(self, mutator: Mutator, objective: Callable[[NDArray[int32]], float]) -> None:
        """
        :param Mutator mutator - the mutator to use
        :param objective - the objective function (should just expect one 3d lattice) that we will aim to **minimize**.
        """
        self.mutator = mutator
        self.objective = objective

        self.state = mutator.init_state()
        self.objvalue = objective(self.state.generate())

    def step(self) -> "tuple[bool, CurrentState, CurrentState, MutationSet]":
        """
        Causes the optimizer to take a step.

        Returns a tuple containing:
        1. Whether the new candidate was accepted.
        2. the new candidate
        3. the old state.
        """
        old_state = self.state
        new_state, mutations = self.mutator.mutate(self.state)
        new_value = self.objective(new_state.generate())

        if new_value >= self.objvalue:
            return (False, new_state, old_state, mutations)

        self.state = new_state
        self.objvalue = new_value

        return (True, new_state, old_state, mutations)
    
    def step_muts(self, ic_muts: List[List[int]], srt_muts: List[List[int]]) -> "tuple[bool, CurrentState]":
        """
        Applies a set of IC and SRT mutations, given by their cell positions, and returns whether the mutation was accepted and the new state of the automaton.

        :param List[List[int]] ic_muts: an array of IC mutation cell positions

        :param List[List[int]] srt_muts: an array of SRT mutation cell positions

        :returns tuple[bool, CurrentState]: whether the mutation set was accepted and the new state of the automaton

        """
        if len(ic_muts) == 0 and len(srt_muts) == 0:
            return (False, self.state)

        for ic_mut in ic_muts:
            new_state = CurrentState(
                initial=self.state.initial.copy(),
                rules=self.state.rules.copy(),
                ruleindices=self.state.ruleindices,
            )
            self.mutator.mutate_initial_pos(ic_mut, self.state, new_state)
            self.state = new_state
        
        for srt_mut in srt_muts:
            new_state = CurrentState(
                initial=self.state.initial.copy(),
                rules=self.state.rules.copy(),
                ruleindices=self.state.ruleindices,
            )
            self.mutator.mutate_pos(srt_mut, self.state, new_state)
            self.state = new_state
        
        new_value = self.objective(new_state.generate())

        if new_value >= self.objvalue:
            return (False, new_state)

        self.state = new_state
        self.objvalue = new_value

        return (True, new_state)
