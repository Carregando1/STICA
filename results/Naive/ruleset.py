"""
ruleset.py

Implements functions for constructing rulesets
"""

from typing import Dict, Tuple
from math import comb
import numpy as np
from numpy._typing import NDArray


class RuleSet:
    """
    Class that implements a general ruleset.
    """

    def __init__(self, states=2) -> None:
        self.states: int = states
        self.rules: Dict[Tuple[int, ...], int] = {}

    def add(self, neighbors: Tuple[int, ...], initial: int, final: int, safe=True):
        """
        Add a rule

        neighbors: a tuple of the number of neighbors in each state. Moore neighborhood only.

        initial: the initial state of the center cell for which this rulset is applicable.

        final: the final state of the center cell given all conditions above are satisfied.
        """
        #error detection
        if safe:
            if len(neighbors) != self.states:
                raise ValueError(
                    f"Invalid neighbor state provided: expected {self.states}, got {len(neighbors)}"
                )
            if sum(neighbors) != 8:
                raise ValueError("Expected exactly 8 neighbors.")
            if not (0 <= initial < self.states) or not (0 <= final < self.states):
                raise ValueError(
                    f"Initial and Final State need to be integers between 0 and {self.states - 1} inclusive"
                )
        #Adds rule as a key; example syntax: (7,1,1) = 1
        self.rules[neighbors + (initial,)] = final

    def __getitem__(self, state: NDArray[np.int32]) -> int:
        return self.rules[tuple(state)]
    
    def nparray(self) -> NDArray[np.int32]:
        """
        Turn the ruleset into a corresponding numpy array

        Example syntax: [[8,0,1,0][8,0,0,0][7,1,1,0]...]]
        """
        return np.array(
            [np.array(state + (final,)) for state, final in self.rules.items()]
        )

    #Strict mode: The array input can not be missing any rules. Default: false.

    @classmethod
    def fromarray(cls, arr: NDArray[np.int32], safe=True, strict=False):
        """
        Turn a numpy array into a ruleset
        """
        if safe:
            assert len(arr.shape) == 2, "You did not supply a two-dimensional array"
            assert (
                arr.shape[1] > 2
            ), "You need to supply at least one state"
        num_states = arr.shape[1] - 2
        #math checks: no duplicates, no missing rules
        if safe and strict:
            assert (arr.shape[0] == num_states * comb(7+num_states, num_states-1)
            ), "You need exactly "+str(num_states * comb(7+num_states, num_states-1))+" rules in the array, found "+str(arr.shape[0])
        res = RuleSet(num_states)
        #checks for duplicate tuples by converting to dictionary then back to list then checking length change
        testlist = [str(rule[:-1]) for rule in arr]
        testdict = list(dict.fromkeys(testlist))
        if safe:
            assert (arr.shape[0] == len(testdict)), "Rules can not be duplicates or conflicting"
        for rule in arr:
            if safe:
                assert (sum(rule[:-2]) == 8), "Sum of neighbors should always be 8"
            res.add(tuple(rule[:-2]), rule[-2], rule[-1], safe=safe)
        return res

def conway() -> RuleSet:
    """
    Returns the ruleset corresponding to Conway's Game of Life.
    """
    res = RuleSet(2)
    alive = lambda x: (8 - x, x)
    for i in range(9):
        res.add(alive(i), 1, 0)
        res.add(alive(i), 0, 0)
    res.add(alive(2), 1, 1)
    res.add(alive(3), 1, 1)
    res.add(alive(3), 0, 1)
    return res


def seeds() -> RuleSet:
    """
    Returns the ruleset corresponding to the seeds cellular automaton.
    """
    res = RuleSet(2)
    alive = lambda x: (8 - x, x)
    for i in range(9):
        res.add(alive(i), 1, 0)
        res.add(alive(i), 0, 0)
    res.add(alive(2), 0, 1)
    return res

def test1() -> RuleSet:
    """
    Returns a 3-state test ruleset.
    """
    res = RuleSet.fromarray(np.array([
    [6,2,0,0,1],
    [5,3,0,0,1],
    [6,1,1,0,1],
    [6,2,0,1,2],
    ]))
    return res
