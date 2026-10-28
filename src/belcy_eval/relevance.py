"""Exact probabilistic relevance between claims, as the map constrains it.

An argument map is read as a propositional theory ``F`` over its claims:

* every claim ``z`` is a propositional variable;
* every edge ``e = (x, y, sigma)`` gets a fresh variable ``k_e`` -- the hidden
  background assumption under which the reason goes through -- and contributes
  ``x & k_e -> y`` for support, ``x & k_e -> not-y`` for attack;
* if the map has two or more roots (claims with no outgoing edge) and reads
  them as rival positions (``ArgMap.mece_roots``), exactly one of them is
  true. A single root, or roots read as independent, are left unconstrained.

The measure is uniform over the models of ``F`` on all of its variables, the
``k_e`` included. Marginalising ``k_e`` out, an assignment to the claims that
violates an edge counts half as much as one that satisfies it, so an edge is a
defeasible constraint rather than a strict one: a reason and a counter-reason
can both be true.

Model counts are exact, from Ganak via ``pyganak``. For an ordered pair
``(x, y)``::

    Pr(y|x) - Pr(y|not-x) > 0   iff   #F * #(F & x & y) - #(F & x) * #(F & y) > 0

so the sign of relevance is decided in integer arithmetic, with no division and
no rounding. Pairs where ``x`` is certain either way, or where the sign is
exactly zero, assert nothing and are dropped.

Pairs are drawn from claims joined by a directed path: ``x`` is a reason, at
some remove, bearing on ``y``. Their distance is the shortest such path. No
assumption about the graph's shape is made -- in a DAG with several paths
between two claims, the paths' contributions are weighed against each other by
the count itself.

With rival roots, a reason for one root bears on every other root through the
exclusivity constraint. Path search then also steps from each root to every
other, the step counting as one edge, so a reason at distance ``d`` from its
own root is paired with each rival at distance ``d + 1``. The step exists only
for path search: it is not an edge of the map and adds no clause to ``F``.
Pairs of two roots are left out.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import TYPE_CHECKING, Any

import networkx as nx
from pyganak import Counter

if TYPE_CHECKING:
    from belcy_eval.maps import ArgMap


def pair_key(source: str, target: str) -> str:
    """Stable string key for an ordered claim pair, for JSON-friendly metadata."""
    return f"{source}|{target}"


@dataclass(frozen=True)
class Pair:
    """An ordered claim pair ``(x, y)`` and the relevance the map asserts between them.

    ``source`` is the supposition ``x``, ``target`` the query ``y``.
    ``distance`` is the shortest directed path from ``x`` to ``y``, ``sign``
    is ``+1`` if ``x`` confirms ``y`` and ``-1`` if it disconfirms it, and
    ``pr_pos``, ``pr_neg`` are ``Pr(y|x)`` and ``Pr(y|not-x)`` under the map's
    measure, rounded to floats for reporting only.
    """

    source: str
    target: str
    distance: int
    sign: int
    pr_pos: float
    pr_neg: float

    @property
    def key(self) -> str:
        return pair_key(self.source, self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "distance": self.distance,
            "sign": self.sign,
            "pr_pos": self.pr_pos,
            "pr_neg": self.pr_neg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pair:
        return cls(
            source=data["source"],
            target=data["target"],
            distance=int(data["distance"]),
            sign=int(data["sign"]),
            pr_pos=float(data["pr_pos"]),
            pr_neg=float(data["pr_neg"]),
        )


# ---------------------------------------------------------------------------
# the formula and its model counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cnf:
    """The map's theory ``F`` in DIMACS-style CNF.

    Claims are variables ``1..n`` in claim-id order; the edge variables
    ``k_e`` follow as ``n+1..n+|E|`` in edge order.
    """

    var_of: dict[str, int]
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]


def cnf(argmap: ArgMap) -> Cnf:
    """Encode an argument map as the CNF theory ``F``."""
    ids = sorted(c.id for c in argmap.claims)
    var_of = {claim_id: i for i, claim_id in enumerate(ids, start=1)}

    clauses: list[tuple[int, ...]] = []
    for j, edge in enumerate(argmap.edges, start=len(ids) + 1):
        x, y = var_of[edge.source], var_of[edge.target]
        clauses.append((-x, -j, y if edge.polarity > 0 else -y))

    roots = [var_of[r] for r in argmap.roots]
    if argmap.mece_roots and len(roots) >= 2:
        clauses.append(tuple(roots))
        clauses.extend((-a, -b) for a, b in combinations(roots, 2))

    return Cnf(var_of=var_of, n_vars=len(ids) + len(argmap.edges), clauses=tuple(clauses))


def model_count(formula: Cnf, assumptions: Sequence[int] = ()) -> int:
    """Exact number of models of ``F`` plus the unit clauses ``assumptions``.

    Every variable is declared up front, so a variable no clause mentions
    still doubles the count.
    """
    counter = Counter()
    counter.new_vars(formula.n_vars)
    for clause in formula.clauses:
        counter.add_clause(list(clause))
    for literal in assumptions:
        counter.add_clause([literal])
    return counter.count()


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------


def candidate_pairs(argmap: ArgMap, max_path_length: int) -> list[tuple[str, str, int]]:
    """``(x, y, d)`` for every ``x`` with a directed path to ``y`` of length ``d <= l``.

    With rival roots, paths may step from one root to another (see the module
    docstring); pairs of two roots are excluded.
    """
    if max_path_length < 1:
        raise ValueError("max_path_length must be at least 1")
    graph = argmap.graph
    roots = set(argmap.roots)
    if argmap.mece_roots and len(roots) >= 2:
        graph.add_edges_from(permutations(sorted(roots), 2))
    found: list[tuple[str, str, int]] = []
    for source in sorted(graph.nodes):
        lengths = nx.single_source_shortest_path_length(graph, source, cutoff=max_path_length)
        found.extend(
            (source, target, d)
            for target, d in lengths.items()
            if d >= 1 and not (source in roots and target in roots)
        )
    return sorted(found)


def exact_pairs(argmap: ArgMap, max_path_length: int) -> list[Pair]:
    """Every candidate pair whose relevance under the map's measure is nonzero."""
    candidates = candidate_pairs(argmap, max_path_length)
    if not candidates:
        return []

    formula = cnf(argmap)
    var_of = formula.var_of
    total = model_count(formula)
    if total == 0:
        raise ValueError("the map's theory has no models")

    claims = {x for x, _, _ in candidates} | {y for _, y, _ in candidates}
    counts = {z: model_count(formula, [var_of[z]]) for z in sorted(claims)}

    pairs: list[Pair] = []
    for x, y, d in candidates:
        n_x, n_y = counts[x], counts[y]
        if n_x == 0 or n_x == total:
            continue
        n_xy = model_count(formula, [var_of[x], var_of[y]])
        diff = total * n_xy - n_x * n_y
        if diff == 0:
            continue
        pairs.append(
            Pair(
                source=x,
                target=y,
                distance=d,
                sign=1 if diff > 0 else -1,
                pr_pos=n_xy / n_x,
                pr_neg=(n_y - n_xy) / (total - n_x),
            )
        )
    return pairs


def sample_pairs(pairs: Sequence[Pair], max_pairs: int, rng: random.Random) -> tuple[Pair, ...]:
    """Draw up to ``max_pairs`` pairs, balanced across distances.

    Each distance stratum is shuffled, then strata are drawn from in turn,
    nearest first, until the cap is hit or every stratum is spent. Raising the
    cap only ever adds pairs. ``max_pairs <= 0`` keeps them all.
    """
    ordered = sorted(pairs, key=lambda p: (p.distance, p.source, p.target))
    if max_pairs <= 0 or len(ordered) <= max_pairs:
        return tuple(ordered)

    strata: dict[int, list[Pair]] = {}
    for pair in ordered:
        strata.setdefault(pair.distance, []).append(pair)
    for distance in sorted(strata):
        rng.shuffle(strata[distance])

    chosen: list[Pair] = []
    depth = 0
    while len(chosen) < max_pairs:
        for distance in sorted(strata):
            if depth < len(strata[distance]) and len(chosen) < max_pairs:
                chosen.append(strata[distance][depth])
        depth += 1
    return tuple(sorted(chosen, key=lambda p: (p.distance, p.source, p.target)))


def exact_marginals(argmap: ArgMap) -> dict[str, float]:
    """``Pr(z)`` for every claim under the map's measure, for reference profiles."""
    formula = cnf(argmap)
    total = model_count(formula)
    return {z: model_count(formula, [v]) / total for z, v in formula.var_of.items()}


def select_pairs(argmap: ArgMap, *, max_path_length: int, max_pairs: int) -> tuple[Pair, ...]:
    """The pairs a map is elicited and scored on, reproducibly per map."""
    rng = random.Random(f"pairs:{argmap.map_id}")
    return sample_pairs(exact_pairs(argmap, max_path_length), max_pairs, rng)
