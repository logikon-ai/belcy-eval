from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pytest
from helpers import CODING_PROBLEM, coding_debate, dag_debate, synthetic_debate

from belcy_eval.maps import CON, PRO, ArgMap, RawDebate, normalize_label, to_argmap


def draw(debate, *, seed=0, min_claims=6, max_claims=10, **kwargs) -> ArgMap:
    return to_argmap(
        debate,
        rng=random.Random(seed),
        min_claims=min_claims,
        max_claims=max_claims,
        **kwargs,
    )


def reaches_a_root(argmap) -> bool:
    graph = argmap.graph
    roots = set(argmap.roots)
    return all(c.id in roots or roots & nx.descendants(graph, c.id) for c in argmap.claims)


def test_subgraph_is_connected_and_rooted(debate):
    argmap = draw(debate)

    assert argmap.roots == ("n0",)
    assert 6 <= argmap.n_claims <= 10
    assert reaches_a_root(argmap)


def test_edges_stay_inside_the_map(debate):
    argmap = draw(debate)
    ids = {c.id for c in argmap.claims}
    for edge in argmap.edges:
        assert edge.source in ids and edge.target in ids


def test_a_dag_keeps_every_edge_inside_the_map_and_every_root():
    argmap = to_argmap(dag_debate(), rng=random.Random(0), min_claims=6, max_claims=6)
    assert argmap.roots == ("r1", "r2")
    assert {(e.source, e.target) for e in argmap.edges} == {
        ("a", "r1"),
        ("b", "r2"),
        ("c", "r2"),
        ("s", "a"),
        ("s", "b"),
    }
    assert reaches_a_root(argmap)
    assert argmap.depth == 2


def test_rival_roots_are_tested_against_each_other_s_reasons():
    argmap = to_argmap(dag_debate(), rng=random.Random(0), min_claims=6, max_claims=6)
    assert argmap.mece_roots
    rival = {p.key: p.distance for p in argmap.pairs if p.key in ("a|r2", "c|r1")}
    assert rival == {"a|r2": 2, "c|r1": 2}
    assert not any(p.source in argmap.roots for p in argmap.pairs)


def test_independent_roots_are_tested_on_directed_paths_only():
    argmap = to_argmap(
        dag_debate(), rng=random.Random(0), min_claims=6, max_claims=6, mece_roots=False
    )
    assert not argmap.mece_roots
    graph = argmap.graph
    assert argmap.pairs
    assert all(nx.has_path(graph, p.source, p.target) for p in argmap.pairs)


def test_drawn_maps_carry_pairs_at_their_shortest_distance(debate):
    argmap = draw(debate, max_claims=12)
    assert argmap.pairs
    graph = argmap.graph
    for pair in argmap.pairs:
        assert pair.sign in (1, -1)
        assert 1 <= pair.distance <= 3
        assert pair.distance == nx.shortest_path_length(graph, pair.source, pair.target)


def test_pair_selection_follows_the_path_length_and_cap(debate):
    short = to_argmap(
        debate,
        rng=random.Random(0),
        min_claims=10,
        max_claims=10,
        max_path_length=1,
        max_pairs=0,
    )
    assert {p.distance for p in short.pairs} == {1}
    capped = to_argmap(
        debate,
        rng=random.Random(0),
        min_claims=10,
        max_claims=10,
        max_path_length=3,
        max_pairs=4,
    )
    assert capped.n_pairs == 4
    assert capped.map_id == short.map_id  # the pairs do not change which map it is


def test_a_map_asserting_no_relevance_is_rejected():
    # x supports both of two exclusive roots, which leaves each exactly as likely
    graph = nx.DiGraph()
    for node in ("r1", "r2", "x"):
        graph.add_node(node, claim=f"Claim {node}.", label=node)
    graph.add_edge("x", "r1", valence="PRO")
    graph.add_edge("x", "r2", valence="PRO")
    debate = RawDebate(corpus_id="c", split="test", path=Path("d.json"), idx=0, graph=graph)
    with pytest.raises(ValueError, match="no relevant pair"):
        to_argmap(debate, rng=random.Random(0), min_claims=3, max_claims=3)


def test_valence_maps_to_polarity(debate):
    argmap = draw(debate)
    graph = debate.graph
    assert {e.polarity for e in argmap.edges} <= {PRO, CON}
    for edge in argmap.edges:
        valence = graph.edges[edge.source, edge.target]["valence"]
        assert edge.polarity == (PRO if valence == "PRO" else CON)


def test_claims_carry_text_including_the_root(debate):
    argmap = draw(debate)
    assert all(c.text for c in argmap.claims)
    assert argmap.text(argmap.roots[0]) == "The root claim."
    assert argmap.topic == "The root claim."


def test_sampling_is_deterministic_given_a_seed(debate):
    a, b = draw(debate, seed=7), draw(debate, seed=7)
    assert a.map_id == b.map_id
    assert [c.id for c in a.claims] == [c.id for c in b.claims]


def test_different_debates_get_different_sizes():
    """The ported bug: a shared seed gave every map in a run the same size."""
    sizes = set()
    for i in range(12):
        deb = synthetic_debate(seed=i, stem=f"debate-test-{i:04d}")
        rng = random.Random(f"42:test:{i}:{deb.stem}")
        sizes.add(to_argmap(deb, rng=rng, min_claims=6, max_claims=14).n_claims)
    assert len(sizes) > 1


def test_target_idx_is_preserved_but_polarity_drives_the_reading(debate):
    argmap = draw(debate)
    assert any(e.target_idx is not None for e in argmap.edges)


def test_shape_properties(simple_map):
    assert simple_map.n_claims == 3
    assert simple_map.n_edges == 2
    assert simple_map.depth == 1
    assert simple_map.roots == ("y",)
    assert simple_map.n_pairs == 2
    assert simple_map.pairs_by_distance() == {1: 2}


def test_round_trip_through_dict(simple_map):
    assert simple_map.pairs
    assert ArgMap.from_dict(simple_map.to_dict()) == simple_map


def test_the_roots_reading_round_trips_and_defaults_to_mece(simple_map):
    independent = replace(simple_map, mece_roots=False)
    assert ArgMap.from_dict(independent.to_dict()) == independent
    logged_before = simple_map.to_dict()
    del logged_before["mece_roots"]
    assert ArgMap.from_dict(logged_before).mece_roots


def test_a_coding_debate_s_problem_and_design_decision_are_read_stripped():
    argmap = draw(
        coding_debate(problem="  Solve $a_{1}$.\n", design_decision="\tMin or max?  "),
        min_claims=6,
        max_claims=6,
    )
    assert argmap.problem == "Solve $a_{1}$."
    assert argmap.design_decision == "Min or max?"
    assert ArgMap.from_dict(argmap.to_dict()) == argmap


def test_a_societal_debate_has_no_problem():
    argmap = draw(dag_debate(), min_claims=6, max_claims=6)
    assert argmap.problem == argmap.design_decision == ""


def test_the_problem_does_not_change_the_map_id():
    with_problem = draw(coding_debate(), min_claims=6, max_claims=6)
    without = draw(coding_debate(problem=None, design_decision=None), min_claims=6, max_claims=6)
    assert with_problem.problem == CODING_PROBLEM
    assert with_problem.map_id == without.map_id
    assert with_problem.pairs == without.pairs


def test_maps_logged_before_the_problem_was_read_load_without_one():
    logged_before = draw(coding_debate(), min_claims=6, max_claims=6).to_dict()
    del logged_before["problem"], logged_before["design_decision"]
    argmap = ArgMap.from_dict(logged_before)
    assert argmap.problem == argmap.design_decision == ""


def test_map_rejects_dangling_edges():
    from belcy_eval.maps import Claim, Edge

    with pytest.raises(ValueError, match="leaves the map"):
        ArgMap(
            map_id="m",
            topic="t",
            claims=(Claim(id="y", label="Y", text="y"),),
            edges=(Edge(source="ghost", target="y", polarity=PRO),),
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Misuse and Error", "Misuse_and_Error"),
        ("  Profit vs. People  ", "Profit_vs_People"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_debate_below_min_claims_is_rejected():
    small = synthetic_debate(branching=2, depth=1)  # 3 claims
    with pytest.raises(ValueError, match="below min_claims"):
        to_argmap(small, rng=random.Random(0), min_claims=4, max_claims=8)


def test_min_claims_is_validated(debate):
    with pytest.raises(ValueError, match="min_claims"):
        to_argmap(debate, rng=random.Random(0), min_claims=1, max_claims=5)
    with pytest.raises(ValueError, match="max_claims"):
        to_argmap(debate, rng=random.Random(0), min_claims=8, max_claims=5)


def test_map_rejects_pairs_that_leave_it(simple_map):
    ghost = replace(simple_map.pairs[0], source="ghost")
    with pytest.raises(ValueError, match="pair ghost->y leaves the map"):
        replace(simple_map, pairs=(ghost,))
