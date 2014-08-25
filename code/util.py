import graph_tool.all as gt


def e(e):
    g = e.get_graph()
    return (g.vertex_index[e.source()], g.vertex_index[e.target()])
