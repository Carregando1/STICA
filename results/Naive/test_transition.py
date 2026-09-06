import numpy as np

from EditedCode import conway, seeds, test1, transition

cn = conway().nparray()
sd = seeds().nparray()
t1 = test1().nparray()

def rules(dim):
    return np.array([[cn for i in range(dim)] for _ in range(dim)])

def seedrules(dim):
    return np.array([[sd for i in range(dim)] for _ in range(dim)])

def t1rules(dim):
    return np.array([[t1 for i in range(dim)] for _ in range(dim)])

def sicatest():
    return np.array([[cn for i in range(3)]+[sd for j in range(3)] for _ in range(6)])

def test_transition_1():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            rules(5),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
    )


def test_transition_2():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            rules(6),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )


def test_transition_3():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            rules(6),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

def test_transition_4():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [0, 1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            rules(6),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

#test case for other automata

def test_transition_5():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            seedrules(6),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

def test_transition_6():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            t1rules(6),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0],
                [0, 0, 2, 2, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )

def test_transition_7():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 2, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            t1rules(6),
        )
        == np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 2, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
            ]
        )
    )

#test case for SICA
def test_transition_8():
    assert np.all(
        transition(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            sicatest(),
        )
        == np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )
