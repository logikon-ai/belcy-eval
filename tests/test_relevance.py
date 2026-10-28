from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction

import pytest
from helpers import make_map

from belcy_eval.relevance import (
    Pair,
    candidate_pairs,
    cnf,
    exact_marginals,
    exact_pairs,
    model_count,
    sample_pairs,
    select_pairs,
)


def conditionals(argmap, x, y):
    """``Pr(y|x)`` and ``Pr(y|not-x)`` as exact fractions, straight off the counts."""
    formula = cnf(argmap)
    var = formula.var_of
    total = model_count(formula)
    n_x = model_count(formula, [var[x]])
    n_y = model_count(formula, [var[y]])
    n_xy = model_count(formula, [var[x], var[y]])
    return Fraction(n_xy, n_x), Fraction(n_y - n_xy, total - n_x)


def sign_of(argmap, x, y):
    """The selected pair's sign, or ``None`` if the pair was dropped."""
    return {p.key: p.sign for p in argmap.pairs}.get(f"{x}|{y}")


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------


def test_edges_become_defeasible_conditionals():
    argmap = make_map([("x", "y", 1), ("z", "y", -1)])
    formula = cnf(argmap)
    x, y, z = (formula.var_of[c] for c in ("x", "y", "z"))
    assert formula.var_of == {"x": 1, "y": 2, "z": 3}
    assert formula.n_vars == 5  # three claims, one k_e per edge
    assert formula.clauses == ((-x, -4, y), (-z, -5, -y))


def test_a_single_root_is_left_unconstrained():
    formula = cnf(make_map([("x", "y", 1)]))
    assert formula.clauses == ((-1, -3, 2),)


def test_several_roots_are_mutually_exclusive_and_jointly_exhaustive():
    argmap = make_map([("x", "r1", 1)], claims=["r2", "r3"])
    formula = cnf(argmap)
    r1, r2, r3 = (formula.var_of[r] for r in ("r1", "r2", "r3"))
    assert argmap.roots == ("r1", "r2", "r3")
    assert set(formula.clauses[1:]) == {
        (r1, r2, r3),
        (-r1, -r2),
        (-r1, -r3),
        (-r2, -r3),
    }


def test_roots_read_as_independent_are_left_unconstrained():
    argmap = make_map([("x", "r1", 1)], claims=["r2", "r3"], mece_roots=False)
    formula = cnf(argmap)
    x, r1 = formula.var_of["x"], formula.var_of["r1"]
    assert formula.clauses == ((-x, -5, r1),)


def test_counts_are_over_every_variable_the_edge_variables_included():
    formula = cnf(make_map([("x", "y", 1)]))
    # of the 8 assignments to x, y, k, only x & k & not-y is excluded
    assert model_count(formula) == 7
    assert model_count(formula, [formula.var_of["x"]]) == 3


def test_a_variable_no_clause_mentions_still_doubles_the_count():
    formula = cnf(make_map([("x", "y", 1)]))
    wider = replace(formula, n_vars=formula.n_vars + 1)
    assert model_count(wider) == 2 * model_count(formula)


# ---------------------------------------------------------------------------
# exact relevance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edges", "claims", "x", "y", "pr_pos", "pr_neg", "sign"),
    [
        # a direct reason, either way
        ([("x", "y", 1)], None, "x", "y", Fraction(2, 3), Fraction(1, 2), 1),
        ([("x", "y", -1)], None, "x", "y", Fraction(1, 3), Fraction(1, 2), -1),
        # chains: the sign is the product of the edge signs
        ([("x", "y", 1), ("z", "x", 1)], None, "z", "y", Fraction(3, 5), Fraction(4, 7), 1),
        ([("x", "y", -1), ("z", "x", -1)], None, "z", "y", Fraction(5, 11), Fraction(3, 7), 1),
        ([("x", "y", 1), ("z", "x", -1)], None, "z", "y", Fraction(6, 11), Fraction(4, 7), -1),
        # rival roots
        ([("x", "a", 1), ("a", "r1", 1)], ["r2"], "x", "r1", Fraction(3, 5), Fraction(4, 7), 1),
        # a reason for one root bears on its rival through the exclusion
        ([("x", "r1", 1)], ["r2"], "x", "r2", Fraction(1, 3), Fraction(1, 2), -1),
        ([("x", "r1", -1)], ["r2"], "x", "r2", Fraction(2, 3), Fraction(1, 2), 1),
        # and a rival's reasons shift the measure on a root's own reasons
        ([("x", "r1", 1), ("y", "r2", 1)], None, "x", "r1", Fraction(3, 5), Fraction(3, 7), 1),
        ([("x", "r1", 1), ("y", "r2", 1)], None, "x", "r2", Fraction(2, 5), Fraction(4, 7), -1),
        # two paths of opposite sign: the count weighs them, a path sum would give 0
        (
            [("x", "a", 1), ("a", "y", 1), ("x", "b", -1), ("b", "y", 1)],
            None,
            "x",
            "y",
            Fraction(9, 14),
            Fraction(16, 25),
            1,
        ),
        # a direct attack outweighs two supporting chains
        (
            [("x", "y", -1), ("x", "a", 1), ("a", "y", 1), ("x", "b", 1), ("b", "y", 1)],
            None,
            "x",
            "y",
            Fraction(9, 17),
            Fraction(16, 25),
            -1,
        ),
    ],
)
def test_exact_relevance_on_hand_checked_maps(edges, claims, x, y, pr_pos, pr_neg, sign):
    argmap = make_map(edges, claims=claims)
    assert conditionals(argmap, x, y) == (pr_pos, pr_neg)
    assert sign_of(argmap, x, y) == sign
    pair = next(p for p in argmap.pairs if p.key == f"{x}|{y}")
    assert pair.pr_pos == pytest.approx(float(pr_pos))
    assert pair.pr_neg == pytest.approx(float(pr_neg))


@pytest.mark.parametrize(
    ("edges", "x", "y"),
    [
        # supporting both of two exclusive roots leaves either as likely as before
        ([("x", "r1", 1), ("x", "r2", 1)], "x", "r1"),
        # two paths that cancel exactly
        ([("x", "a", -1), ("a", "y", -1), ("x", "b", -1), ("b", "y", 1)], "x", "y"),
    ],
)
def test_exactly_irrelevant_pairs_are_dropped(edges, x, y):
    argmap = make_map(edges)
    pr_pos, pr_neg = conditionals(argmap, x, y)
    assert pr_pos == pr_neg
    assert sign_of(argmap, x, y) is None


def test_independent_roots_leave_each_tree_to_itself():
    argmap = make_map([("x", "r1", 1), ("y", "r2", 1)], mece_roots=False)
    # as for a lone direct reason: the other tree only doubles the counts
    assert conditionals(argmap, "x", "r1") == (Fraction(2, 3), Fraction(1, 2))
    assert [p.key for p in argmap.pairs] == ["x|r1", "y|r2"]


def test_the_counts_decide_a_sign_too_close_for_floats_to_be_trusted():
    argmap = make_map([("x", "a", 1), ("a", "y", 1), ("x", "b", -1), ("b", "y", 1)])
    pr_pos, pr_neg = conditionals(argmap, "x", "y")
    assert pr_pos - pr_neg == Fraction(1, 350)
    assert sign_of(argmap, "x", "y") == 1


def test_exact_marginals_come_from_the_same_measure():
    argmap = make_map([("x", "y", 1)])
    # of the 7 models, x holds in 3 and y in 4 (x and k free)
    assert exact_marginals(argmap) == pytest.approx({"x": 3 / 7, "y": 4 / 7})


# ---------------------------------------------------------------------------
# candidate pairs
# ---------------------------------------------------------------------------


def test_candidates_follow_directed_paths_at_their_shortest_distance():
    argmap = make_map([("a", "b", 1), ("b", "c", 1), ("a", "c", -1), ("d", "a", 1)])
    assert candidate_pairs(argmap, 3) == [
        ("a", "b", 1),
        ("a", "c", 1),
        ("b", "c", 1),
        ("d", "a", 1),
        ("d", "b", 2),
        ("d", "c", 2),
    ]
    assert candidate_pairs(argmap, 1) == [
        ("a", "b", 1),
        ("a", "c", 1),
        ("b", "c", 1),
        ("d", "a", 1),
    ]


def test_candidates_never_run_against_the_arrows():
    argmap = make_map([("x", "y", 1)])
    assert candidate_pairs(argmap, 3) == [("x", "y", 1)]


def test_candidates_step_between_rival_roots_but_never_pair_two_roots():
    argmap = make_map([("z", "x", 1), ("x", "r1", 1)], claims=["r2", "r3"])
    assert candidate_pairs(argmap, 3) == [
        ("x", "r1", 1),
        ("x", "r2", 2),
        ("x", "r3", 2),
        ("z", "r1", 2),
        ("z", "r2", 3),
        ("z", "r3", 3),
        ("z", "x", 1),
    ]
    assert ("z", "r2", 3) not in candidate_pairs(argmap, 2)


def test_candidates_stay_on_directed_paths_when_roots_are_independent():
    argmap = make_map([("z", "x", 1), ("x", "r1", 1)], claims=["r2"], mece_roots=False)
    assert candidate_pairs(argmap, 3) == [("x", "r1", 1), ("z", "r1", 2), ("z", "x", 1)]


@pytest.mark.parametrize("mece_roots", [True, False])
def test_a_single_root_is_unaffected_by_the_switch(mece_roots):
    argmap = make_map([("x", "y", -1), ("z", "x", -1)], mece_roots=mece_roots)
    assert [(p.key, p.distance, p.sign) for p in argmap.pairs] == [
        ("x|y", 1, -1),
        ("z|x", 1, -1),
        ("z|y", 2, 1),
    ]


def test_path_length_must_be_positive():
    with pytest.raises(ValueError, match="max_path_length"):
        candidate_pairs(make_map([("x", "y", 1)]), 0)


def test_exact_pairs_carry_distance_and_sign():
    argmap = make_map([("x", "y", -1), ("z", "x", -1)])
    assert [(p.key, p.distance, p.sign) for p in exact_pairs(argmap, 3)] == [
        ("x|y", 1, -1),
        ("z|x", 1, -1),
        ("z|y", 2, 1),
    ]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def fake_pairs(counts: dict[int, int]) -> list[Pair]:
    return [
        Pair(source=f"s{d}_{i}", target="t", distance=d, sign=1, pr_pos=0.6, pr_neg=0.4)
        for d, n in counts.items()
        for i in range(n)
    ]


def test_sampling_is_deterministic_given_the_rng():
    pairs = fake_pairs({1: 10, 2: 10, 3: 10})
    a = sample_pairs(pairs, 9, random.Random("seed"))
    b = sample_pairs(pairs, 9, random.Random("seed"))
    assert a == b


def test_sampling_balances_distances_and_respects_the_cap():
    pairs = fake_pairs({1: 20, 2: 6, 3: 2})
    chosen = sample_pairs(pairs, 12, random.Random(0))
    by_distance = {d: sum(1 for p in chosen if p.distance == d) for d in (1, 2, 3)}
    # distance 3 runs out after two, and its quota passes to the others
    assert by_distance == {1: 5, 2: 5, 3: 2}
    assert len(chosen) == 12


def test_raising_the_cap_only_adds_pairs():
    pairs = fake_pairs({1: 15, 2: 9, 3: 4})
    small = set(sample_pairs(pairs, 8, random.Random("s")))
    large = set(sample_pairs(pairs, 16, random.Random("s")))
    assert small <= large


@pytest.mark.parametrize("cap", [0, -1, 100])
def test_an_absent_or_loose_cap_keeps_every_pair(cap):
    pairs = fake_pairs({1: 4, 2: 3})
    assert set(sample_pairs(pairs, cap, random.Random(0))) == set(pairs)


def test_selection_is_reproducible_per_map():
    argmap = make_map([(f"x{i}", "y", 1) for i in range(6)] + [("z", "x0", -1)], max_pairs=0)
    a = select_pairs(argmap, max_path_length=2, max_pairs=4)
    b = select_pairs(argmap, max_path_length=2, max_pairs=4)
    assert a == b
    assert len(a) == 4


def test_pairs_round_trip_through_dict():
    pair = Pair(source="x", target="y", distance=2, sign=-1, pr_pos=0.25, pr_neg=0.5)
    assert Pair.from_dict(pair.to_dict()) == pair
