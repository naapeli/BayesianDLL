import networkx as nx


def plot_model(model):
    try:
        pos = nx.planar_layout(model.graph)
    except nx.NetworkXException:
        pos = nx.random_layout(model.graph)
    nx.draw(model.graph, pos, with_labels=True)
