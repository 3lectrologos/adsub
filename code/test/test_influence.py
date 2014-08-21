import unittest
import networkx as nx
import influence


def test_graph():
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(1, 4)
    g.add_edge(5, 2)
    g.add_edge(5, 3)
    g.add_edge(5, 4)
    g.add_edge(6, 7)
    return g.to_directed()

class TestGraphInstanceGeneration(unittest.TestCase):
    def test_conditional_instance_p0(self):
        g = test_graph()
        elive = set([(4, 5), (5, 4)])
        edead = set([(1, 4), (4, 1), (5, 2), (2, 5), (5, 3), (3, 5)])
        h = influence.random_instance(g, 0, prz=(elive, edead))
        self.assertEqual(set(h.edges()), elive)

    def test_conditional_instance_p1(self):
        g = test_graph()
        elive = set([(4, 5), (5, 4)])
        edead = set([(1, 4), (4, 1), (5, 2), (2, 5), (5, 3), (3, 5)])
        h = influence.random_instance(g, 1, prz=(elive, edead))
        self.assertEqual(set(g.edges()) - set(h.edges()), edead)
