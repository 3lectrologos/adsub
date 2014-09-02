import unittest2
import igraph as ig
import bitarray as ba
import influence


class TestInfluence(unittest2.TestCase):
    def setUp(self):
        self.g = influence.test_graph()
        self.pruned = ig.Graph(directed=True)
        self.pruned.add_vertices(7)
        for v in self.pruned.vs:
            v['i'] = v.index
        self.pruned.add_edge(0, 1)
        self.pruned.add_edge(1, 4)
        self.pruned.add_edge(4, 2)
        self.pruned.add_edge(3, 0)
        self.pruned.add_edge(3, 4)
        self.pruned.add_edge(6, 5)

    def test_random_instance_p_0(self):
        h = influence.random_instance(self.g, 0)
        self.assertEqual(self.g.ecount(), 0)

    def test_random_instance_p_1(self):
        nedges = self.g.ecount()
        h = influence.random_instance(self.g, 1)
        self.assertEqual(self.g.ecount(), nedges)

    def test_random_instance_copy(self):
        nedges = self.g.ecount()
        h = influence.random_instance(self.g, 0, copy=True)
        self.assertEqual(self.g.ecount(), nedges)
        self.assertEqual(h.ecount(), 0)

    def test_random_instance_ret(self):
        orig = set(self.g.get_edgelist())
        (_, removed) = influence.random_instance(self.g, 0.5, ret=True)
        self.assertNotEqual(set(self.g.get_edgelist()), orig)
        self.assertEqual(set(self.g.get_edgelist()) | set(removed), set(orig))

    def test_independent_cascade_empty(self):
        h = ig.Graph(directed=True)
        h.add_vertices(10)
        active = influence.ic(h, [])
        self.assertEqual(active, set())

    def test_independent_cascade_one(self):
        h = ig.Graph(directed=True)
        h.add_vertices(10)
        active = influence.ic(h, [5])
        self.assertEqual(active, set([5]))

    def test_independent_cascade_1(self):
        self.g.delete_edges([(0, 2), (1, 4), (3, 4)])
        active = influence.ic(self.g, [0])
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
        csim = influence.ic_sim(self.g, 1, 1)
        correct = {
            0: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            1: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            2: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            3: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            4: {0: True, 1: True, 2: True, 3: True, 4: True, 5: False, 6: False},
            5: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True},
            6: {0: False, 1: False, 2: False, 3: False, 4: False, 5: True, 6: True}
            }
        for v in self.g.vs:
            for u in self.g.vs:
                self.assertEqual(csim[v.index, u.index][0],
                                 correct[v.index][u.index])

    def test_cascade_sim_2_iter_full(self):
        csim = influence.ic_sim(self.g, 1, 2)
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
            for v in self.g.vs:
                for u in self.g.vs:
                    self.assertEqual(csim[v.index, u.index][i],
                                     correct[v.index][u.index])

    def test_cascade_sim_2_iter_empty(self):
        csim = influence.ic_sim(self.g, 0, 2)
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
            for v in self.g.vs:
                for u in self.g.vs:
                    self.assertEqual(csim[v.index, u.index][i],
                                     correct[v.index][u.index])

    def test_cascade_sim_2_iter_contrained(self):
        self.g.delete_edges([(0, 2), (0, 3), (2, 4), (4, 3)])
        csim = influence.ic_sim(self.g, 1, 2)
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
            for v in self.g.vs:
                for u in self.g.vs:
                    self.assertEqual(csim[v.index, u.index, i],
                                     correct[v.index][u.index])

    def test_conditional_pipeline_new_p_0(self):
        self.pruned.delete_edges([(1, 4)])
        active = influence.ic(self.pruned, [0])
        influence.delete_active(self.pruned, active)
        csim = influence.ic_sim_cond(self.pruned, 0, 1, active=active)
        all_correct = {0: 2, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3}
        correct = {v: all_correct[v] for v in all_correct if v not in active}
        self.assertEqual(csim, correct)

    def test_conditional_pipeline_new_p_1(self):
        self.pruned.delete_edges([(1, 4)])
        active = influence.ic(self.pruned, [0])
        influence.delete_active(self.pruned, active)
        csim = influence.ic_sim_cond(self.pruned, 1, 1, active=active)
        all_correct = {0: 2, 1: 2, 2: 3, 3: 5, 4: 4, 5: 3, 6: 4}
        correct = {v: all_correct[v] for v in all_correct if v not in active}
        self.assertEqual(csim, correct)
