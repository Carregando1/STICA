from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from EditedCode import transition


@dataclass
class CurrentState:
    """
    Dataclass that implements the current state of the genetic algorithm.

    the shape of 'initial' is (width, length), with all entries being integers
    between 0 and states-1.

    the shape of 'rules' is (height-1, width, length, rule_size, states+2), where the
    states+2 comes from [...frequencies of each state, initial state, final
    state].

    the shape of 'ruleindices' is either none or the same as 'rules'
    """

    initial: NDArray[np.int32]
    rules: NDArray[np.int32]
    ruleindices: NDArray[np.int32] | None = None
    _generated: NDArray[np.int32] | None = None

    def states(self):
        """
        Just returns the number of possible GOL states associated with this state.
        """
        return self.rules.shape[4] - 2

    # generate results get cached.
    def generate(self) -> NDArray[np.int32]:
        """
        Returns the effect of applying the spacetime-inhomogeneous set of rules.
        """
        if self._generated is not None:
            return self._generated
        grid = np.zeros(
            shape=(self.rules.shape[0] + 1, self.rules.shape[1], self.rules.shape[2]),
            dtype=np.int32,
        )
        grid[0] = self.initial
        for i in range(self.rules.shape[0]):
            grid[i + 1] = transition(grid[i], self.rules[i], num_states=2)

        self._generated = grid
        return grid
