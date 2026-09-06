#Optimize: Attempt 2 version.

from typing import Callable, List

import numpy as np
from numpy import int32
from numpy._typing import NDArray

from .mutation import Mutator
from .transition import transition
import time

class Optimizer:
    """
    An Optimizer class which puts all the components of our genetic algorithm together.
    """

    def __init__(self, mutator: Mutator, objective: Callable[[NDArray[int32]], float]) -> None:
        """
        :param Mutator mutator - the mutator to use; as of now, the only option is ArbitraryRulesetMutator
        :param objective - the objective function (should just expect one 3d lattice) that we will aim to **minimize**.
        """
        self.mutator = mutator
        self.objective = objective

        self.state = mutator.init_state()
        self.objvalue = objective(transition(self.state[0], self.state[1], self.state[1].shape[0], self.state[2], self.state[3]))

    def step(self) -> "tuple[bool, tuple[NDArray[int32], NDArray[int32], int, NDArray[int32] | None], tuple[NDArray[int32], NDArray[int32], int, NDArray[int32] | None], tuple[List,List]]":
        """
        Causes the optimizer to take a step.

        Returns a tuple containing:
        1. Whether the new candidate was accepted;
        2. the new candidate;
        3. the old state;
        4. the mutations applied.
        """
        #init = time.time()

        old_state = self.state
        new_state, mutations = self.mutator.mutate(self.state)
        #print(f"Total mutate time: {time.time() - init}")
        #init = time.time()
        test = transition(new_state[0], new_state[1], new_state[1].shape[0], new_state[2], new_state[3])
        new_value = self.objective(test)
        #print(f"Generate time: {time.time() - init}")
        #init = time.time()
        if new_value >= self.objvalue:
            return (False, new_state, old_state, mutations)
        self.state = new_state
        self.objvalue = new_value
        #print(f"Final time: {time.time() - init}")
        return (True, new_state, old_state, mutations)
    
    def step_muts(self, ic_muts: List[List[int]], srt_muts: List[List[int]]) -> "tuple[bool, tuple[NDArray[int32], NDArray[int32], int, NDArray[int32] | None]]":
        """
        Applies a set of IC and SRT mutations, given by their cell positions, and returns whether the mutation was accepted and the new state of the automaton.

        :param List[List[int]] ic_muts: an array of IC mutation cell positions

        :param List[List[int]] srt_muts: an array of SRT mutation cell positions

        :returns tuple[bool, CurrentState]: whether the mutation set was accepted and the new state of the automaton

        Currently only for 2 states due to degeneralization. Currently untested due to lack of implementation in the non-AI genetic algorithm portion.

        TODO: This section needs to be updated for the NSTICA construct.
        """
        
        if len(ic_muts) == 0 and len(srt_muts) == 0:
            return (False, self.state)

        new_ic = self.state[0]
        new_srt = self.state[1]
        
        ic_mut_addition = np.zeros(new_ic.shape)
        srt_mut_addition = np.zeros(new_srt.shape)

        ic_mut_addition[tuple(ic_muts[:, 0:2])] = ic_muts[:, 3]
        srt_mut_addition[tuple(srt_muts[:, 0:3])] = ic_muts[:, 4]

        new_ic += ic_mut_addition
        new_ic %= self.states
        new_srt += srt_mut_addition
        new_srt %= self.states

        new_value = self.objective(transition(new_ic, new_srt, new_srt.shape[0], self.state[2], self.state[3]))

        if new_value >= self.objvalue:
            return (False, (new_ic, new_srt, self.state[2], self.state[3]))

        self.state = (new_ic, new_srt, self.state[2], self.state[3])
        self.objvalue = new_value

        return (True, (new_ic, new_srt, self.state[2], self.state[3]))
