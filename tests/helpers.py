"""Builders for small, hand-checkable argument maps and syncIALO-shaped debates."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import networkx as nx

from belcy_eval.maps import ArgMap, Claim, Edge, RawDebate
from belcy_eval.relevance import select_pairs


def make_map(
    edges: list[tuple[str, str, int]],
    *,
    claims: list[str] | None = None,
    map_id: str = "test",
    max_path_length: int = 3,
    max_pairs: int = 0,
    mece_roots: bool = True,
) -> ArgMap:
    """Build a small argument map from ``(source, target, polarity)`` triples.

    Pairs are selected as for a drawn map, every pair by default.
    """
    ids = list(claims) if claims is not None else []
    for src, tgt, _ in edges:
        for node in (src, tgt):
            if node not in ids:
                ids.append(node)
    argmap = ArgMap(
        map_id=map_id,
        topic="a test topic",
        claims=tuple(Claim(id=i, label=i.upper(), text=f"claim {i}") for i in ids),
        edges=tuple(Edge(source=s, target=t, polarity=p) for s, t, p in edges),
        mece_roots=mece_roots,
    )
    pairs = select_pairs(argmap, max_path_length=max_path_length, max_pairs=max_pairs)
    return replace(argmap, pairs=pairs)


def synthetic_debate(
    *,
    branching: int = 3,
    depth: int = 3,
    seed: int = 0,
    stem: str = "debate-test-0001",
) -> RawDebate:
    """A syncIALO-shaped debate: rooted tree, edges child to parent, PRO/CON."""
    rng = random.Random(seed)
    graph = nx.DiGraph()
    root = "n0"
    graph.add_node(root, claim="The root claim.", label="Root")
    frontier = [root]
    counter = 1
    for _ in range(depth):
        nxt: list[str] = []
        for parent in frontier:
            for _ in range(branching):
                node = f"n{counter}"
                counter += 1
                graph.add_node(
                    node,
                    claim=f"Claim number {counter}.",
                    label=f"Label {counter}",
                    premises=[f"premise {counter}a", f"premise {counter}b"],
                )
                graph.add_edge(
                    node,
                    parent,
                    valence=rng.choice(["PRO", "CON"]),
                    target_idx=rng.randint(0, 1),
                )
                nxt.append(node)
        frontier = nxt
    return RawDebate(
        corpus_id="synthetic_corpus-test",
        split="test",
        path=Path(f"node_link_data-{stem}.json"),
        idx=0,
        graph=graph,
    )


def dag_debate(*, stem: str = "debate-test-dag", topic: str | None = None) -> RawDebate:
    """A small syncIALO-shaped debate with two rival roots and a shared reason.

    ``s`` bears on both ``a`` (under root ``r1``) and ``b`` (under ``r2``), so
    the graph is a DAG but not a tree.
    """
    graph = nx.DiGraph()
    if topic is not None:
        graph.graph["topic"] = topic
    for node in ("r1", "r2", "a", "b", "c", "s"):
        graph.add_node(node, claim=f"Claim {node}.", label=node.upper())
    for src, tgt, valence in [
        ("a", "r1", "PRO"),
        ("b", "r2", "PRO"),
        ("c", "r2", "CON"),
        ("s", "a", "PRO"),
        ("s", "b", "CON"),
    ]:
        graph.add_edge(src, tgt, valence=valence, target_idx=0)
    return RawDebate(
        corpus_id="synthetic_corpus-test",
        split="test",
        path=Path(f"node_link_data-{stem}.json"),
        idx=0,
        graph=graph,
    )


#: A Codeforces-style statement, braces and LaTeX included, as syncIALO stores it.
CODING_PROBLEM = r"""You are given two integers $x$ and $y$. Output the minimum of $x$ and $y$, followed by the maximum of $x$ and $y$.

Input
The first line contains a single integer $t$ ($1 \leq t \leq 100$) -- the number of test cases. The only line of each test case contains two space-separated integers $x$ and $y$ ($0 \leq x, y \leq 9$).

Output
For each test case, output $\min\{x, y\}$ followed by $\max\{x, y\}$.

Tags: ['implementation', 'sortings']
Time limit: 1 seconds
Memory limit: 256 megabytes"""

#: The design decision a coding debate is about.
CODING_DESIGN_DECISION = (
    "Whether to use built-in min and max operations versus implementing the comparison logic "
    "manually with conditionals, trading off brevity against explicitness."
)

_CODING_CLAIMS = {
    "r1": "When solving this coding problem, we should implement the comparison with the "
    "built-in min and max functions.",
    "r2": "When solving this coding problem, we should implement the comparison manually "
    "with an if-else on x and y.",
    "a": "Built-ins state the intent in one line: print(min(x, y), max(x, y)).",
    "b": "An explicit branch makes the ordering logic visible to a reader.",
    "c": r"With $0 \leq x, y \leq 9$ a hand-written branch has no edge cases worth {guarding}.",
    "s": "Fewer lines of code leave fewer places for a swapped-output bug.",
}


def coding_debate(
    *,
    stem: str = "debate-test-coding",
    problem: str | None = CODING_PROBLEM,
    design_decision: str | None = CODING_DESIGN_DECISION,
) -> RawDebate:
    """A syncIALO-shaped coding debate: :func:`dag_debate`'s shape, rival solution approaches.

    Like the deepcoder corpus, the graph states the problem and the design
    decision as graph attributes, and no topic.
    """
    graph = nx.DiGraph()
    if problem is not None:
        graph.graph["problem"] = problem
    if design_decision is not None:
        graph.graph["design_decision"] = design_decision
    for node, claim in _CODING_CLAIMS.items():
        graph.add_node(node, claim=claim, label=node.upper())
    for src, tgt, valence in [
        ("a", "r1", "PRO"),
        ("b", "r2", "PRO"),
        ("c", "r2", "CON"),
        ("s", "a", "PRO"),
        ("s", "b", "CON"),
    ]:
        graph.add_edge(src, tgt, valence=valence, target_idx=0)
    return RawDebate(
        corpus_id="deepcoder_codeforces-test",
        split="test",
        path=Path(f"node_link_data-{stem}.json"),
        idx=0,
        graph=graph,
    )
