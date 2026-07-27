"""Render a computation graph produced by the micrograd `value` engine.

The engine is loaded directly from micrograd.ipynb (first code cell), so this
script always visualises the same class the notebook defines — nothing is
duplicated or edited. Output: assets/computation_graph.png

Run:  python render_graph.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_engine():
    nb = json.load(open(os.path.join(HERE, "micrograd.ipynb"), encoding="utf-8"))
    src = "".join(nb["cells"][0]["source"])  # the `value` class lives in cell 0
    ns = {}
    exec(src, ns)
    return ns["value"]


def trace(root):
    nodes, edges = set(), set()  # walk _prev to collect nodes and edges
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def depth_of(nodes, edges):
    children = {n: [] for n in nodes}
    for a, b in edges:
        children[b].append(a)
    memo = {}
    def d(n):
        if n not in memo:
            memo[n] = 0 if not children[n] else 1 + max(d(c) for c in children[n])
        return memo[n]
    return {n: d(n) for n in nodes}


def render(root, names=None, path=None):
    names = names or {}
    nodes, edges = trace(root)
    depth = depth_of(nodes, edges)

    # layered layout: x by depth, y spread within each layer
    layers = {}
    for n in nodes:
        layers.setdefault(depth[n], []).append(n)
    pos = {}
    for dx, layer in layers.items():
        layer = sorted(layer, key=lambda n: (n._op, n.data))
        for i, n in enumerate(layer):
            y = i - (len(layer) - 1) / 2.0
            pos[n] = (dx * 3.0, y * 1.6)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for a, b in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x0 + 1.05, y0), (x1 - 1.05, y1),
                     arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#5566aa"))
    for n, (x, y) in pos.items():
        label = names.get(id(n), n._op if n._op != "leaf node bruh" else "input")
        ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.55), 2.1, 1.1,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=1.2, edgecolor="#1f2a44", facecolor="#eef2f7"))
        ax.text(x, y + 0.26, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1f2a44")
        ax.text(x, y - 0.18, f"data {n.data:.2f}\ngrad {n.grad:.2f}",
                ha="center", va="center", fontsize=8.5, color="#333333")

    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(min(ys) - 1.4, max(ys) + 1.4)
    ax.axis("off")
    ax.set_title("Computation graph built and differentiated by the value engine",
                 fontsize=12, color="#1f2a44")
    plt.tight_layout()
    out = path or os.path.join(HERE, "assets", "computation_graph.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    value = load_engine()
    # Karpathy's first example: L = (a*b + c) * f
    a, b, c, f = value(2.0), value(-3.0), value(10.0), value(-2.0)
    e = a * b
    d = e + c
    L = d * f
    L.backward()
    names = {id(a): "a", id(b): "b", id(c): "c", id(f): "f",
             id(e): "a*b", id(d): "a*b+c", id(L): "L"}
    render(L, names)
