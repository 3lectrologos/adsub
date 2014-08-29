import unittest2
import networkx as nx
import bitarray as ba
import influence


class TestInfluence(unittest2.TestCase):
    def setUp(self):
        self.g = influence.test_graph()

    def test_conditional_instance_p_0(self):
        elive = set([(3, 4), (4, 3)])
        edead = set([(0, 3), (3, 0), (4, 1), (1, 4), (4, 2), (2, 4)])
        rem = set(self.g.edges()) - elive - edead
        h = influence.copy_without_edges(self.g, edead)
        h = influence.random_instance(h, 0, rem)
        self.assertEqual(set(h.edges()), elive)

    def test_conditional_instance_p_1(self):
        elive = set([(3, 4), (4, 3)])
        edead = set([(0, 3), (3, 0), (4, 1), (1, 4), (4, 2), (2, 4)])
        rem = set(self.g.edges()) - elive - edead
        h = influence.copy_without_edges(self.g, edead)
        h = influence.random_instance(h, 1, rem)
        self.assertEqual(set(self.g.edges()) - set(h.edges()), edead)

    def test_conditional_instance_random(self):
        elive = set([(0, 1), (1, 4), (2, 0), (4, 2), (1, 0), (4, 1)])
        edead = set([(0, 3), (0, 2), (4, 3), (2, 4)])
        rem = set(self.g.edges()) - elive - edead
        h = influence.copy_without_edges(self.g, edead)
        h = influence.random_instance(h, 0.5, rem)
        self.assertGreaterEqual(set(h.edges()), elive)
        self.assertEqual(set(h.edges()) & edead, set())

    def test_live_dead_empty(self):
        h = self.g.copy()
        elive, edead = influence.get_live_dead(self.g, h, set())
        self.assertEqual(elive, set())
        self.assertEqual(edead, set())

    def test_live_dead_one_1(self):
        h = self.g.copy()
        elive, edead = influence.get_live_dead(self.g, h, set([0]))
        self.assertEqual(elive, set([(0, 1), (0, 2), (0, 3)]))
        self.assertEqual(edead, set())

    def test_live_dead_one_2(self):
        h = influence.copy_without_edges(self.g, [(0, 2)])
        elive, edead = influence.get_live_dead(self.g, h, set([0]))
        self.assertEqual(elive, set([(0, 1), (0, 3)]))
        self.assertEqual(edead, set([(0, 2)]))

    def test_live_dead_multi(self):
        h = influence.copy_without_edges(self.g, [(0, 2), (1, 4), (3, 4)])
        elive, edead = influence.get_live_dead(self.g, h, set([0, 1, 3]))
        self.assertEqual(elive, set([(0, 1), (1, 0), (0, 3), (3, 0)]))
        self.assertEqual(edead, set([(0, 2), (1, 4), (3, 4)]))

    def test_independent_cascade_empty(self):
        h = nx.create_empty_copy(self.g)
        active = influence.ic(h, [0])
        self.assertEqual(active, set([0]))

    def test_independent_cascade_1(self):
        h = influence.copy_without_edges(self.g, [(0, 2), (1, 4), (3, 4)])
        active = influence.ic(h, [0])
        self.assertEqual(active, set([0, 1, 3]))

    def test_independent_cascade_1_half(self):
        active = influence.ic(self.g, [0])
        self.assertEqual(active, set([0, 1, 2, 3, 4]))

    def test_independent_cascade_2_half(self):
        active = influence.ic(self.g, [0, 4])
        self.assertEqual(active, set([0, 1, 2, 3, 4]))

    def test_independent_cascade_2_full(self):
        active = influence.ic(self.g, [0, 6])
        self.assertEqual(active, set([0, 1, 2, 3, 4, 5, 6]))

    def test_independent_cascade_full_full(self):
        active = influence.ic(self.g, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(active, set([0, 1, 2, 3, 4, 5, 6]))

    def test_cascade_sim_1_iter_full(self):
        csim = influence.ic_sim(self.g, 1, 1, rem=self.g.edges())
        correct = {
            0: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            1: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            2: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            3: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            4: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True}
            }
        for v in self.g.nodes_iter():
            for u in self.g.nodes_iter():
                self.assertEqual(csim[v, u][0], correct[v][u])

    def test_cascade_sim_2_iter_full(self):
        csim = influence.ic_sim(self.g, 1, 2, rem=self.g.edges())
        correct = {
            0: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            1: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            2: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            3: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            4: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True}
            }
        for i in range(2):
            for v in self.g.nodes_iter():
                for u in self.g.nodes_iter():
                    self.assertEqual(csim[v, u][i], correct[v][u])

    def test_cascade_sim_2_iter_empty(self):
        csim = influence.ic_sim(self.g, 0, 2, rem=self.g.edges())
        correct = {
            0: {0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False},
            1: {0: False, 1: True, 2: False, 3: False, 4: False, 5: False, 6: False},
            2: {0: False, 1: False, 2: True, 3: False, 4: False, 5: False, 6: False},
            3: {0: False, 1: False, 2: False, 3: True, 4: False, 5: False, 6: False},
            4: {0: False, 1: False, 2: False, 3: False, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: False},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: True}
            }
        for i in range(2):
            for v in self.g.nodes_iter():
                for u in self.g.nodes_iter():
                    self.assertEqual(csim[v, u][i], correct[v][u])

    def test_cascade_sim_2_iter_contrained(self):
        h = influence.copy_without_edges(self.g,
                                         [(0, 2), (0, 3), (2, 4), (4, 3)])
        csim = influence.ic_sim(h, 1, 2, rem=h.edges())
        correct = {
            0: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            1: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            2: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            3: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            4: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True}
            }
        for i in range(2):
            for v in self.g.nodes_iter():
                for u in self.g.nodes_iter():
                    self.assertEqual(csim[v, u][i], correct[v][u])

    def test_conditional_pipeline(self):
        h = influence.copy_without_edges(self.g,
                                         [(0, 2), (0, 3), (2, 4), (4, 3)])
        active = influence.ic(h, [2])
        elive, edead = influence.get_live_dead(self.g, h, active)
        h.remove_edges_from(edead)
        rem = set(h.edges()) - elive - edead
        csim = influence.ic_sim(h, 0.5, 10, rem=rem)
        correct = {
            0: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            1: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            2: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            3: {0: False, 1: False, 2: False, 3: True, 4: False, 5: False, 6: False},
            4: {0: True, 1: True, 2: True, 3: False, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False}
            }
        correct[0] = ba.bitarray('1110100')
        correct[1] = ba.bitarray('1110100')
        correct[2] = ba.bitarray('1110100')
        correct[3] = ba.bitarray('0001000')
        correct[4] = ba.bitarray('1110100')
        correct[5] = ba.bitarray('0000000')
        correct[6] = ba.bitarray('0000000')
        for i in range(2):
            for v in self.g.nodes_iter():
                for u in self.g.nodes_iter():
                    self.assertEqual(csim[v, u][i] and correct[v][u],
                                     correct[v][u])
