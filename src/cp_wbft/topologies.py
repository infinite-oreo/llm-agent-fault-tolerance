"""
[INPUT]:  依赖 Python 标准库 random / collections.abc
[OUTPUT]: 对外提供 Adjacency 类型别名，build_topology / sorted_neighbors 两个公开函数
[POS]:    cp_wbft 的图构造层，被 consensus 和 simulation 消费，六种拓扑完整封装于此
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

import random
from collections.abc import Iterable

Adjacency = dict[int, set[int]]

# ---- 拓扑名称 → 构造函数 ------------------------------------------------
# 需要 seed 参数的拓扑单独列出，避免派发时传参歧义
_SEEDED = {"random", "random-graph"}
_BUILDERS: dict[str, object] = {}  # 填充见模块尾部


def build_topology(name: str, n: int, *, seed: int = 7) -> Adjacency:
    """Build one of the paper's representative MAS communication topologies."""

    if n < 2:
        raise ValueError("n must be at least 2")

    normalized = name.lower().replace("_", "-")
    builder = _BUILDERS.get(normalized)
    if builder is None:
        raise ValueError(f"unknown topology: {name}")

    if normalized in _SEEDED:
        return builder(n, seed=seed)  # type: ignore[call-arg]
    return builder(n)  # type: ignore[call-arg]


def sorted_neighbors(graph: Adjacency, node: int) -> list[int]:
    return sorted(graph.get(node, set()))


def _empty(n: int) -> Adjacency:
    return {i: set() for i in range(n)}


def _connect(graph: Adjacency, edges: Iterable[tuple[int, int]]) -> Adjacency:
    for a, b in edges:
        if a == b:
            continue
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _complete(n: int) -> Adjacency:
    return _connect(_empty(n), ((i, j) for i in range(n) for j in range(i + 1, n)))


def _chain(n: int) -> Adjacency:
    return _connect(_empty(n), ((i, i + 1) for i in range(n - 1)))


def _star(n: int) -> Adjacency:
    return _connect(_empty(n), ((0, i) for i in range(1, n)))


def _tree(n: int) -> Adjacency:
    edges: list[tuple[int, int]] = []
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            edges.append((i, left))
        if right < n:
            edges.append((i, right))
    return _connect(_empty(n), edges)


def _random_connected(n: int, *, seed: int) -> Adjacency:
    rng = random.Random(seed)
    graph = _chain(n)
    possible = [(i, j) for i in range(n) for j in range(i + 1, n) if j != i + 1]
    target_extra_edges = max(1, n // 2)
    for a, b in rng.sample(possible, k=min(target_extra_edges, len(possible))):
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _layered(n: int) -> Adjacency:
    graph = _empty(n)
    layers: list[list[int]] = []
    remaining = list(range(n))
    width = 1
    while remaining:
        layers.append(remaining[:width])
        remaining = remaining[width:]
        width += 1

    for left, right in zip(layers, layers[1:]):
        _connect(graph, ((a, b) for a in left for b in right))
    return graph


# ---- 拓扑注册表（模块级，避免循环引用） ------------------------------------
_BUILDERS.update({
    "complete":       _complete,
    "complete-graph": _complete,
    "chain":          _chain,
    "star":           _star,
    "tree":           _tree,
    "random":         _random_connected,
    "random-graph":   _random_connected,
    "layered":        _layered,
    "layered-graph":  _layered,
})
