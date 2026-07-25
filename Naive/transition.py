"""
transition.py:

perform transitions for states and the corresponding rules
"""

import numpy as np
from numpy import int32
from numpy._typing import NDArray
from EditedCode.neighbors import neighbors

#intentional misspelling to circumvent variable conflict

def _transition(neybors: NDArray[int32], rules: NDArray[int32]):
    #-1 is a placeholder so that the x-dimension of neybors is equal to the x-dimension of rules
    #This program compares the modified neybors matrix with across every row in rules, until it finds the row which contains all the correct elements, which then it returns the corresponding final value.
    indices = np.nonzero(
        np.sum(rules == np.append(neybors, -1), axis=1) == neybors.shape[0]
    )[0]
    if indices.size > 0:
        return rules[indices[0]][-1]
    return 0


def  transition(states: NDArray[int32], rules: NDArray[int32], num_states = None):
    """
    performs a transition given computed states and a set of rules.

    the shape of states is (width, height), with all entries being integers between 0 and states-1.

    the shape of rules is (width, height, rule_size, states+2), where the states+2 comes from
    [...frequencies of each state, initial state, final state].
    """
    computed_neighbors = neighbors(states, num_states)
    #reshape width*height to one dimension for ease of computation
    flattened_neighbors = computed_neighbors.reshape(-1, computed_neighbors.shape[2])
    flattened_rules = rules.reshape(-1, rules.shape[2], rules.shape[3])
    result = np.zeros(flattened_rules.shape[0], dtype=int32)
    #replace every cell with its post-transition state
    for i in range(flattened_rules.shape[0]):
        result[i] = _transition(flattened_neighbors[i], flattened_rules[i])
    #reshape result to width x height
    result = result.reshape(states.shape[0], states.shape[1])
    return result