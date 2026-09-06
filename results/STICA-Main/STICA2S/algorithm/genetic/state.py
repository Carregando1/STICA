#State: Now merged with transition.

from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray

from .transition import transition

@dataclass
class CurrentState:
    """
    Dataclass that implements the current state of the genetic algorithm.

    Args:
    initial: the IC of the 2-state STICA. Shape of 'initial' is (width, length), with all entries being integers
    between 0 and states-1.

    rules: The SRT of the 2-state STICA. Shape of 'rules' is (height-1, width, length, 18), where the
    18 is (neighbor-states)*(self-states).

    the shape of 'ruleindices' is either none or the same as 'rules'
    """

    initial: NDArray[np.int32]
    rules: NDArray[np.int32]
    ruleindices: NDArray[np.int32] | None = None
    _generated: NDArray[np.int32] | None = None

    # generate results get cached.
    def generate(self) -> NDArray[np.int32]:
        """
        Returns the effect of applying the spacetime-inhomogeneous set of rules.
        """
        if self._generated is not None:
            return self._generated
        grid = transition(self.initial, self.rules, self.rules.shape[0])
        self._generated = grid
        return grid
