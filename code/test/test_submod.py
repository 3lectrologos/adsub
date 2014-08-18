import unittest
import submod


def f_mon(s):
    s = set(s)
    if s == set(): return 0
    if s == set([1]): return 1.05
    if s == set([2]): return 1
    if s == set([3]): return 1
    if s == set([1, 2]): return 1.5
    if s == set([1, 3]): return 2.05
    if s == set([2, 3]): return 2
    if s == set([1, 2, 3]): return 2.5

def f_nonmon1(s):
    s = set(s)
    return f_mon(s) - 0.4*len(s)

def f_nonmon2(s):
    s = set(s)
    return f_mon(s) - 0.6*len(s)

class TestRandomGreedy(unittest.TestCase):
    def test_simple(self):
        (sol, fsol) = submod.random_greedy(f_mon, [1, 2, 3], 1)
        self.assertEqual(sol, [1])
        self.assertAlmostEqual(fsol, f_mon([1]))

    def test_monotone(self):
        (sol, fsol) = submod.random_greedy(f_mon, [1, 2, 3], 3)
        self.assertEqual(set(sol), set([1, 2, 3]))
        self.assertAlmostEqual(fsol, 2.5)

    def test_nonmonotone1(self):
        (sol, fsol) = submod.random_greedy(f_nonmon1, [1, 2, 3], 3)
        self.assertEqual(set(sol), set([1, 2, 3]))
        self.assertAlmostEqual(fsol, 1.3)

    def test_nonmonotone2(self):
        (sol, fsol) = submod.random_greedy(f_nonmon2, [1, 2, 3], 3)
        self.assertEqual(len(sol), 2)

    def test_nonmonotone_card(self):
        (sol, fsol) = submod.random_greedy(f_nonmon1, [1, 2, 3], 2)
        self.assertEqual(len(sol), 2)
