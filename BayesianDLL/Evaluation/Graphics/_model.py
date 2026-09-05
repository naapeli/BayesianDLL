"""Model graph visualization using probabilistic-programming conventions."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
import networkx as nx


_NODE_STYLE = {
    "random": {"facecolor": "#FFFFFF", "edgecolor": "#333333"},
    "observed": {"facecolor": "#D3D3D3", "edgecolor": "#333333"},
    "deterministic": {"facecolor": "#FFFFFF", "edgecolor": "#333333"},
    "data": {"facecolor": "#EEEEEE", "edgecolor": "#333333"},
}


def _visible_graph(model, include_data):
    nodes = [
        node
        for node, attributes in model.graph.nodes(data=True)
        if include_data or attributes.get("type") != "data"
    ]
    return model.graph.subgraph(nodes).copy()


def _layered_layout(graph):
    """Return a stable, top-to-bottom layout for a directed model graph."""
    if not graph:
        return {}

    try:
        order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        # A model graph should be a DAG, but still draw one that was manually
        # modified so it can be diagnosed.
        order = list(graph.nodes)

    insertion_order = {node: index for index, node in enumerate(graph.nodes)}
    depth = {}
    for node in order:
        parent_depths = [depth[parent] for parent in graph.predecessors(node) if parent in depth]
        depth[node] = max(parent_depths, default=-1) + 1

    layers = defaultdict(list)
    for node in order:
        layers[depth[node]].append(node)

    positions = {}
    for layer_index in sorted(layers):
        nodes = layers[layer_index]

        def sort_key(node):
            parent_x = [positions[parent][0] for parent in graph.predecessors(node) if parent in positions]
            return (sum(parent_x) / len(parent_x) if parent_x else 0.0, insertion_order[node])

        nodes.sort(key=sort_key)
        spacing = 2.8
        start = -spacing * (len(nodes) - 1) / 2
        for index, node in enumerate(nodes):
            positions[node] = (start + index * spacing, -layer_index * 2.15)

    return positions


def _distribution_name(model, node, node_type):
    if node_type == "random":
        parameter = model.params.get(node)
    elif node_type == "observed":
        parameter = model.observed_params.get(node)
    else:
        return None
    distribution = getattr(parameter, "distribution", None)
    return type(distribution).__name__ if distribution is not None else None


def _data_shape(model, node):
    data = model.data.get(node)
    if data is None:
        return None
    shape = tuple(data.value.shape)
    return "scalar" if not shape else " × ".join(str(size) for size in shape)


def _plate_members(model, visible_nodes):
    members = defaultdict(list)
    sizes = {}
    collections = (model.params, model.observed_params, model.deterministic_params)
    for collection in collections:
        for node, parameter in collection.items():
            if node not in visible_nodes:
                continue
            for plate in getattr(parameter, "plates", ()):
                members[plate.name].append(node)
                try:
                    sizes[plate.name] = plate.size
                except (IndexError, ValueError):
                    sizes[plate.name] = "?"
    return members, sizes


def _draw_plates(ax, model, positions, node_sizes):
    members, sizes = _plate_members(model, positions)
    # Larger groups are drawn first so nested/smaller plates remain legible.
    groups = sorted(members.items(), key=lambda item: len(set(item[1])), reverse=True)
    for level, (name, plate_nodes) in enumerate(groups):
        plate_nodes = list(dict.fromkeys(plate_nodes))
        if not plate_nodes:
            continue
        left = min(positions[node][0] - node_sizes[node][0] / 2 for node in plate_nodes)
        right = max(positions[node][0] + node_sizes[node][0] / 2 for node in plate_nodes)
        bottom = min(positions[node][1] - node_sizes[node][1] / 2 for node in plate_nodes)
        top = max(positions[node][1] + node_sizes[node][1] / 2 for node in plate_nodes)
        padding = 0.34 + 0.10 * level
        rectangle = FancyBboxPatch(
            (left - padding, bottom - padding),
            right - left + 2 * padding,
            top - bottom + 2 * padding,
            boxstyle="round,pad=0.02,rounding_size=0.13",
            facecolor="none",
            edgecolor="#666666",
            linewidth=1.0,
            linestyle="solid",
            zorder=0,
        )
        ax.add_patch(rectangle)
        ax.text(
            right + padding - 0.08,
            bottom - padding + 0.09,
            f"{name}  [{sizes.get(name, '?')}]",
            ha="right",
            va="bottom",
            color="#333333",
            fontsize=9,
            fontfamily="serif",
            zorder=1,
        )


def _node_subtitle(model, node, node_type):
    subtitle = _distribution_name(model, node, node_type)
    if node_type == "deterministic":
        return "Deterministic"
    if node_type == "data":
        shape = _data_shape(model, node)
        return f"Data · {shape}" if shape else "Data"
    return subtitle


def plot_model(
    model,
    *,
    ax=None,
    include_data=True,
    show_distributions=True,
    show_plates=True,
    legend=False,
):
    """Plot a probabilistic model's dependency graph.

    Ellipses are stochastic variables, shaded ellipses are observations,
    square boxes are deterministic values, and rounded grey boxes are named
    data inputs. Set ``include_data=False`` for a compact generative view.

    Parameters
    ----------
    model : Model
        The model to visualize.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.
    include_data : bool, default True
        Whether named ``Data`` inputs are included.
    show_distributions : bool, default True
        Add distribution names and data shapes below node names.
    show_plates : bool, default True
        Draw boxes around nodes created in a ``plate`` context.
    legend : bool, default False
        Show a compact key for node types.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the graph.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")

    graph = _visible_graph(model, include_data)
    positions = _layered_layout(graph)
    if not positions:
        ax.text(
            0.5, 0.5, "Empty model", transform=ax.transAxes,
            ha="center", va="center", color="#555555", fontfamily="serif",
        )
        ax.set_axis_off()
        return ax

    node_sizes = {}
    patches = {}
    for node, (x, y) in positions.items():
        node_type = graph.nodes[node].get("type", "deterministic")
        subtitle = _node_subtitle(model, node, node_type)
        longest_line = max(len(str(node)), len(subtitle or ""))
        width = max(1.42, min(3.1, 0.105 * longest_line + 0.85))
        height = 1.30 if show_distributions and subtitle else 0.78
        node_sizes[node] = (width, height)
        style = _NODE_STYLE.get(node_type, _NODE_STYLE["deterministic"])

        if node_type in {"random", "observed"}:
            patch = Ellipse((x, y), width, height, linewidth=1.5, zorder=3, **style)
        else:
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle=(
                    "round,pad=0.02,rounding_size=0.13"
                    if node_type == "data" else "square,pad=0.02"
                ),
                linewidth=1.4,
                linestyle="solid",
                zorder=3,
                **style,
            )
        patches[node] = patch

    if show_plates:
        _draw_plates(ax, model, positions, node_sizes)

    for source, target in graph.edges:
        ax.add_patch(FancyArrowPatch(
            positions[source], positions[target],
            patchA=patches[source], patchB=patches[target],
            arrowstyle="-|>", mutation_scale=12, linewidth=1.15,
            color="#555555", connectionstyle="arc3,rad=0.0",
            shrinkA=3, shrinkB=3, zorder=2,
        ))

    for node, patch in patches.items():
        ax.add_patch(patch)
        x, y = positions[node]
        node_type = graph.nodes[node].get("type", "deterministic")
        subtitle = _node_subtitle(model, node, node_type)
        has_subtitle = show_distributions and bool(subtitle)
        label = f"{node}\n~\n{subtitle}" if has_subtitle else str(node)
        ax.text(
            x, y, label,
            ha="center", va="center", fontsize=10, fontweight="normal",
            fontfamily="serif", linespacing=0.85, color="#222222", zorder=4,
        )

    left = min(positions[node][0] - node_sizes[node][0] / 2 for node in positions)
    right = max(positions[node][0] + node_sizes[node][0] / 2 for node in positions)
    ys = [position[1] for position in positions.values()]
    ax.set_xlim(left - 0.75, right + 0.75)
    ax.set_ylim(min(ys) - 1.15, max(ys) + 0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if legend:
        node_types = {graph.nodes[node].get("type") for node in graph}
        labels = {
            "random": "Latent",
            "observed": "Observed",
            "deterministic": "Deterministic",
            "data": "Data",
        }
        handles = [
            Line2D(
                [], [], marker="o" if node_type in {"random", "observed"} else "s",
                linestyle="None", markersize=8,
                markerfacecolor=_NODE_STYLE[node_type]["facecolor"],
                markeredgecolor=_NODE_STYLE[node_type]["edgecolor"],
                label=labels[node_type],
            )
            for node_type in labels if node_type in node_types
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
            ncols=max(1, len(handles)), frameon=False, fontsize=8.5,
            handletextpad=0.35, columnspacing=1.1,
        )

    return ax
