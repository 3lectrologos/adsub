import unittest2
import graph_tool.all as gt
import bitarray as ba
import influence
import util


class TestInfluence(unittest2.TestCase):
    def setUp(self):
        self.g = influence.test_graph()

    def view_without_edges(self, g, elist):
        pm = self.g.new_edge_property('bool')
        for e in g.edges():
            if util.e(e) in elist:
                pm[e] = False
            else:
                pm[e] = True
        return gt.GraphView(self.g, efilt=pm)

    def test_conditional_instance_p_0(self):
        nlive = [(3, 4), (4, 3)]
        ndead = [(0, 3), (3, 0), (4, 1), (1, 4), (4, 2), (2, 4)]
        h = influence.random_instance(self.g, 0, prz=(nlive, ndead))
        self.assertEqual(set(h.edges()), set(h.edge(*e) for e in nlive))

    def test_conditional_instance_p_1(self):
        nlive = [(3, 4), (4, 3)]
        ndead = [(0, 3), (3, 0), (4, 1), (1, 4), (4, 2), (2, 4)]
        nrest = [(0, 1), (1, 0), (0, 2), (2, 0), (5, 6), (6, 5)]
        h = influence.random_instance(self.g, 1, prz=(nlive, ndead))
        eall = set(h.edge(*e) for e in nlive) | set(h.edge(*e) for e in nrest)
        self.assertEqual(set(h.edges()), eall)

    def test_conditional_instance_random(self):
        nlive = [(0, 1), (1, 4), (2, 0), (4, 2), (1, 0), (4, 1)]
        ndead = [(0, 3), (0, 2), (4, 3), (2, 4)]
        h = influence.random_instance(self.g, 0.5, prz=(nlive, ndead))
        edges = set([util.e(e) for e in h.edges()])
        self.assertGreaterEqual(edges, set(nlive))
        self.assertEqual(edges & set(ndead), set())

    def test_live_dead_empty(self):
        h = gt.GraphView(self.g)
        elive, edead = influence.get_live_dead(self.g, h, set())
        self.assertEqual(elive, set())
        self.assertEqual(edead, set())

    def test_live_dead_one_1(self):
        h = gt.GraphView(self.g)
        elive, edead = influence.get_live_dead(self.g, h, set([0]))
        self.assertEqual(elive, set([(0, 1), (0, 2), (0, 3)]))
        self.assertEqual(edead, set())

    def test_live_dead_one_2(self):
        h = self.view_without_edges(self.g, [(0, 2)])
        elive, edead = influence.get_live_dead(self.g, h, set([0]))
        self.assertEqual(elive, set([(0, 1), (0, 3)]))
        self.assertEqual(edead, set([(0, 2)]))

    def test_live_dead_multi(self):
        h = self.view_without_edges(self.g, [(0, 2), (1, 4), (3, 4)])
        elive, edead = influence.get_live_dead(self.g, h, set([0, 1, 3]))
        self.assertEqual(elive, set([(0, 1), (1, 0), (0, 3), (3, 0)]))
        self.assertEqual(edead, set([(0, 2), (1, 4), (3, 4)]))

    def test_independent_cascade_empty(self):
        h = gt.GraphView(self.g)
        h.clear_edges()
        active = influence.independent_cascade(h, [0])
        self.assertEqual(active, set([0]))

    def test_independent_cascade_1(self):
        h = self.view_without_edges(self.g, [(0, 2), (1, 4), (3, 4)])
        active = influence.independent_cascade(h, [0])
        self.assertEqual(active, set([0, 1, 3]))

    def test_independent_cascade_1_half(self):
        active = influence.independent_cascade(self.g, [0])
        self.assertEqual(active, set([0, 1, 2, 3, 4]))

    def test_independent_cascade_2_half(self):
        active = influence.independent_cascade(self.g, [0, 4])
        self.assertEqual(active, set([0, 1, 2, 3, 4]))

    def test_independent_cascade_2_full(self):
        active = influence.independent_cascade(self.g, [0, 6])
        self.assertEqual(active, set([0, 1, 2, 3, 4, 5, 6]))

    def test_independent_cascade_full_full(self):
        active = influence.independent_cascade(self.g, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(active, set([0, 1, 2, 3, 4, 5, 6]))

    def test_cascade_sim_1_iter_full(self):
        csim = influence.cascade_sim(self.g, 1, 1)
        correct = {}
        correct[0] = ba.bitarray('1111100')
        correct[1] = ba.bitarray('1111100')
        correct[2] = ba.bitarray('1111100')
        correct[3] = ba.bitarray('1111100')
        correct[4] = ba.bitarray('1111100')
        correct[5] = ba.bitarray('0000011')
        correct[6] = ba.bitarray('0000011')
        self.assertEqual(csim, [correct])

    def test_cascade_sim_2_iter_full(self):
        csim = influence.cascade_sim(self.g, 1, 2)
        correct = {}
        correct[0] = ba.bitarray('1111100')
        correct[1] = ba.bitarray('1111100')
        correct[2] = ba.bitarray('1111100')
        correct[3] = ba.bitarray('1111100')
        correct[4] = ba.bitarray('1111100')
        correct[5] = ba.bitarray('0000011')
        correct[6] = ba.bitarray('0000011')
        self.assertEqual(csim, [correct, correct])

    def test_cascade_sim_2_iter_empty(self):
        csim = influence.cascade_sim(self.g, 0, 2)
        correct = {}
        correct[0] = ba.bitarray('1000000')
        correct[1] = ba.bitarray('0100000')
        correct[2] = ba.bitarray('0010000')
        correct[3] = ba.bitarray('0001000')
        correct[4] = ba.bitarray('0000100')
        correct[5] = ba.bitarray('0000010')
        correct[6] = ba.bitarray('0000001')
        self.assertEqual(csim, [correct, correct])

    def test_cascade_sim_2_iter_contrained(self):
        h = self.view_without_edges(self.g, [(0, 2), (0, 3), (2, 4), (4, 3)])
        csim = influence.cascade_sim(h, 1, 2)
        correct = {}
        correct[0] = ba.bitarray('1110100')
        correct[1] = ba.bitarray('1110100')
        correct[2] = ba.bitarray('1110100')
        correct[3] = ba.bitarray('1111100')
        correct[4] = ba.bitarray('1110100')
        correct[5] = ba.bitarray('0000011')
        correct[6] = ba.bitarray('0000011')
        self.assertEqual(csim, [correct, correct])

    def test_conditional_pipeline(self):
        h = self.view_without_edges(self.g, [(0, 2), (0, 3), (2, 4), (4, 3)])
        active = influence.independent_cascade(h, [2])
        elive, edead = influence.get_live_dead(self.g, h, active)
        csim = influence.cascade_sim(h, 0.5, 10, prz=(elive, edead))
        correct = {}
        correct[0] = ba.bitarray('1110100')
        correct[1] = ba.bitarray('1110100')
        correct[2] = ba.bitarray('1110100')
        correct[3] = ba.bitarray('0001000')
        correct[4] = ba.bitarray('1110100')
        correct[5] = ba.bitarray('0000000')
        correct[6] = ba.bitarray('0000000')
        for cs in csim:
            for i in range(7):
                self.assertEqual(cs[i] & correct[i], correct[i])
