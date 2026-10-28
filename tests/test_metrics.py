from __future__ import annotations

import math
import random

import pytest
from helpers import make_map

from belcy_eval.metrics import (
    AGGREGATE_KEYS,
    BeliefState,
    binary_entropy,
    coherence_gap,
    coherence_rmse,
    coherence_terms,
    confirmation,
    direction,
    information_penalties,
    normalized_coherence,
    null_rmse,
    profile,
    score_keys,
    strength,
    strengths,
)
from belcy_eval.relevance import exact_marginals

L = 3


def state(argmap, marginals, conditionals=None, default_c=(0.5, 0.5)):
    """Belief state from explicit marginals, conditionals defaulting to inert."""
    conds = {}
    for pair in argmap.pairs:
        conds[pair.key] = (conditionals or {}).get(pair.key, default_c)
    return BeliefState(marginals=marginals, conditionals=conds)


def pairs_by_sign(argmap):
    confirming = next(p for p in argmap.pairs if p.sign == 1)
    disconfirming = next(p for p in argmap.pairs if p.sign == -1)
    return confirming, disconfirming


def scores(argmap, st):
    return profile(argmap, st, L).values


def test_simple_map_has_one_confirming_and_one_disconfirming_pair(simple_map):
    assert [(p.key, p.distance, p.sign) for p in simple_map.pairs] == [
        ("x1|y", 1, 1),
        ("x2|y", 1, -1),
    ]


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
def test_binary_entropy_bounds(p, expected):
    assert binary_entropy(p) == pytest.approx(expected)


def test_confirmation_is_the_difference_of_the_two_conditionals():
    assert confirmation(0.9, 0.2) == pytest.approx(0.7)
    assert confirmation(0.2, 0.9) == pytest.approx(-0.7)
    assert confirmation(0.4, 0.4) == pytest.approx(0.0)


def test_strength_signs_the_confirmation_by_the_entailed_relevance():
    # x raises y: right where the map entails confirmation, wrong where it entails the opposite
    assert strength(1.0, 0.0, 1) == pytest.approx(1.0)
    assert strength(1.0, 0.0, -1) == pytest.approx(-1.0)
    # x lowers y: the mirror image
    assert strength(0.0, 1.0, -1) == pytest.approx(1.0)
    assert strength(0.0, 1.0, 1) == pytest.approx(-1.0)


def test_direction_counts_the_sign_and_nothing_else():
    assert direction(strength(0.51, 0.50, 1)) == 1
    assert direction(strength(1.0, 0.0, 1)) == 1
    assert direction(strength(0.51, 0.50, -1)) == -1
    assert direction(strength(0.7, 0.7, 1)) == 0


def test_a_decisive_fence_sitter_scores_high_strength_but_no_more_direction_than_the_measure():
    """Strength rewards taking reasons as decisive; the head score does not (README)."""
    argmap = make_map([("a", "r", 1), ("b", "r", 1)])
    exact = scores(argmap, BeliefState.exact(argmap, exact_marginals(argmap)))
    fence_sitter = scores(
        argmap,
        BeliefState(
            marginals={c.id: 0.51 for c in argmap.claims},
            conditionals={p.key: (1.0, 0.0) for p in argmap.pairs},
        ),
    )
    head = "weighted_relevance_direction"
    assert fence_sitter[head] == exact[head] == 1.0
    assert fence_sitter["relevance_strength"] == pytest.approx(1.0)
    assert exact["relevance_strength"] < 0.5


def test_the_head_score_is_unchanged_when_every_confirmation_is_halved(simple_map):
    """Only the size of the claimed relevance changes, and strength alone reports it."""
    confirming, disconfirming = pairs_by_sign(simple_map)
    marginals = {"x1": 0.5, "x2": 0.5, "y": 0.5}
    bold = scores(
        simple_map,
        state(simple_map, marginals, {confirming.key: (0.9, 0.1), disconfirming.key: (0.3, 0.7)}),
    )
    timid = scores(
        simple_map,
        state(simple_map, marginals, {confirming.key: (0.7, 0.3), disconfirming.key: (0.4, 0.6)}),
    )
    head = "weighted_relevance_direction"
    assert bold[head] == timid[head] == pytest.approx(1.0)
    assert timid["relevance_strength"] == pytest.approx(bold["relevance_strength"] / 2)


def test_the_head_score_weighs_a_big_right_move_against_a_small_wrong_one(simple_map):
    """The case the unweighted count of directions gets wrong: it would score 0 here."""
    confirming, disconfirming = pairs_by_sign(simple_map)
    st = state(
        simple_map,
        {"x1": 0.5, "x2": 0.5, "y": 0.5},
        {confirming.key: (0.95, 0.05), disconfirming.key: (0.55, 0.45)},
    )
    values = scores(simple_map, st)
    assert values["misdirected_relevance"] == pytest.approx(0.5)  # one pair each way
    assert values["weighted_relevance_direction"] == pytest.approx((0.9 - 0.1) / (0.9 + 0.1))


def test_strength_is_the_head_score_times_the_mean_confirmation(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    st = state(
        simple_map,
        {"x1": 0.5, "x2": 0.5, "y": 0.5},
        {confirming.key: (0.8, 0.2), disconfirming.key: (0.5, 0.1)},
    )
    values = scores(simple_map, st)
    mean_confirmation = (0.6 + 0.4) / 2
    assert values["relevance_strength"] == pytest.approx(
        values["weighted_relevance_direction"] * mean_confirmation
    )


def test_relevance_does_not_depend_on_the_credence_in_the_query(simple_map):
    confirming, _ = pairs_by_sign(simple_map)
    conds = {confirming.key: (0.8, 0.3)}
    low = state(simple_map, {"x1": 0.5, "x2": 0.5, "y": 0.05}, conds)
    high = state(simple_map, {"x1": 0.5, "x2": 0.5, "y": 0.95}, conds)
    for key in ("weighted_relevance_direction", "relevance_strength"):
        assert scores(simple_map, low)[key] == pytest.approx(scores(simple_map, high)[key])


def test_coherence_gap_is_the_law_of_total_probability():
    # b(y) = c(y|x)*b(x) + c(y|not-x)*(1 - b(x)) = 0.8*0.25 + 0.4*0.75 = 0.5
    assert coherence_gap(0.5, 0.25, 0.8, 0.4) == pytest.approx(0.0)
    assert coherence_gap(0.6, 0.25, 0.8, 0.4) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------


def test_relevance_reaches_one_when_every_pair_runs_as_entailed(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    st = state(
        simple_map,
        {"x1": 0.5, "x2": 0.5, "y": 0.5},
        {confirming.key: (1.0, 0.0), disconfirming.key: (0.0, 1.0)},
    )
    values = scores(simple_map, st)
    assert values["weighted_relevance_direction"] == pytest.approx(1.0)
    assert values["relevance_strength"] == pytest.approx(1.0)
    assert values["misdirected_relevance"] == pytest.approx(0.0)


def test_relevance_reaches_minus_one_when_every_pair_is_inverted(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    st = state(
        simple_map,
        {"x1": 0.5, "x2": 0.5, "y": 0.5},
        {confirming.key: (0.0, 1.0), disconfirming.key: (1.0, 0.0)},
    )
    values = scores(simple_map, st)
    assert values["weighted_relevance_direction"] == pytest.approx(-1.0)
    assert values["relevance_strength"] == pytest.approx(-1.0)
    assert values["misdirected_relevance"] == pytest.approx(1.0)


def test_misdirected_counts_only_the_inverted_pairs(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    st = state(
        simple_map,
        {"x1": 0.5, "x2": 0.5, "y": 0.5},
        {confirming.key: (0.9, 0.1), disconfirming.key: (0.7, 0.3)},
    )
    values = scores(simple_map, st)
    # one pair of each sign, so half are misdirected; the head score weighs the
    # right one (0.8) against the wrong one (0.4) instead of counting them
    assert values["misdirected_relevance"] == pytest.approx(0.5)
    assert values["weighted_relevance_direction"] == pytest.approx((0.8 - 0.4) / (0.8 + 0.4))
    assert values["relevance_strength"] == pytest.approx((0.8 - 0.4) / 2)


def test_an_irrelevant_supposition_scores_zero_strength_and_stakes_nothing(simple_map):
    """A supposition that moves nothing sits at chance on strength, and stakes nothing."""
    st = state(simple_map, {"x1": 0.5, "x2": 0.5, "y": 0.5}, default_c=(0.7, 0.7))
    values = scores(simple_map, st)
    assert values["relevance_strength"] == pytest.approx(0.0)
    assert values["misdirected_relevance"] == pytest.approx(0.0)
    assert math.isnan(values["weighted_relevance_direction"])


def test_distant_pairs_are_scored_by_their_entailed_sign_and_broken_down():
    # z attacks x, x attacks y: z confirms y at distance 2
    argmap = make_map([("x", "y", -1), ("z", "x", -1)])
    assert {p.key: (p.distance, p.sign) for p in argmap.pairs} == {
        "x|y": (1, -1),
        "z|x": (1, -1),
        "z|y": (2, 1),
    }
    st = state(
        argmap,
        {"x": 0.5, "y": 0.5, "z": 0.5},
        {"x|y": (0.2, 0.6), "z|x": (0.3, 0.5), "z|y": (0.4, 0.6)},
    )
    values = scores(argmap, st)
    assert values["relevance_strength_d1"] == pytest.approx((0.4 + 0.2) / 2)
    assert values["relevance_strength_d2"] == pytest.approx(-0.2)
    assert values["weighted_relevance_direction_d1"] == pytest.approx(1.0)
    assert values["weighted_relevance_direction_d2"] == pytest.approx(-1.0)
    assert values["weighted_relevance_direction"] == pytest.approx(
        (0.4 + 0.2 - 0.2) / (0.4 + 0.2 + 0.2)
    )
    assert values["misdirected_relevance_d1"] == pytest.approx(0.0)
    assert values["misdirected_relevance_d2"] == pytest.approx(1.0)
    assert values["relevance_strength"] == pytest.approx((0.4 + 0.2 - 0.2) / 3)
    assert math.isnan(values["relevance_strength_d3"])
    assert math.isnan(values["misdirected_relevance_d3"])


def test_pairs_beyond_the_path_length_are_ignored():
    argmap = make_map([("x", "y", -1), ("z", "x", -1)])
    st = state(argmap, {"x": 0.5, "y": 0.5, "z": 0.5}, default_c=(0.9, 0.1))
    prof = profile(argmap, st, 1)
    assert prof.n_pairs == 2
    assert set(prof.values) == set(score_keys(1))


def test_the_maps_own_measure_is_rightly_directed_and_coherent():
    argmap = make_map([("a", "r1", 1), ("b", "r2", -1), ("s", "a", 1), ("s", "b", 1)])
    values = scores(argmap, BeliefState.exact(argmap, exact_marginals(argmap)))
    assert values["weighted_relevance_direction"] == pytest.approx(1.0)
    assert 0 < values["relevance_strength"] < 1
    assert values["misdirected_relevance"] == pytest.approx(0.0)
    assert values["coherence_normalized"] == pytest.approx(1.0)
    assert values["coherence_rmse"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# coherence and information
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_map():
    """z attacks x, x supports y: pairs x|y, z|x, z|y, so two distinct targets."""
    return make_map([("x", "y", 1), ("z", "x", -1)])


def implied(terms):
    return [(t.target, t.b_y, t.implied) for t in terms]


def test_coherence_terms_carry_the_credence_the_state_implies(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    b = {"x1": 0.25, "x2": 0.5, "y": 0.9}
    conds = {
        confirming.key: (0.8, 0.4),  # 0.8*0.25 + 0.4*0.75 = 0.5
        disconfirming.key: (0.2, 0.8),  # 0.2*0.5  + 0.8*0.5  = 0.5
    }
    terms = coherence_terms(simple_map, state(simple_map, b, conds))
    assert implied(terms) == [("y", 0.9, pytest.approx(0.5)), ("y", 0.9, pytest.approx(0.5))]
    assert coherence_rmse(terms) == pytest.approx(0.4)


def test_coherence_on_a_hand_checked_state(chain_map):
    st = state(
        chain_map,
        {"x": 0.4, "y": 0.7, "z": 0.5},
        {
            "x|y": (0.9, 0.5),  # implies y = 0.66, off by 0.04
            "z|x": (0.2, 0.6),  # implies x = 0.40, exact
            "z|y": (0.8, 0.6),  # implies y = 0.70, exact
        },
    )
    terms = coherence_terms(chain_map, st)
    r_obs = math.sqrt(0.04**2 / 3)
    # implied y=0.66 against b(x)=0.4; implied x=0.4 against b(y)=0.7, twice;
    # implied y=0.70 against b(x)=0.4. Pairs sharing a target are never compared.
    r_null = math.sqrt((0.26**2 + 3 * 0.3**2) / 4)
    assert coherence_rmse(terms) == pytest.approx(r_obs)
    assert null_rmse(terms) == pytest.approx(r_null)
    values = scores(chain_map, st)
    assert values["coherence_rmse"] == pytest.approx(r_obs)
    assert values["coherence_normalized"] == pytest.approx((r_null - r_obs) / (r_null + r_obs))


def test_a_state_that_fits_the_wrong_targets_scores_minus_one(chain_map):
    st = state(
        chain_map,
        {"x": 0.5, "y": 0.2, "z": 0.5},
        # each pair implies exactly the credence of the other target
        {"x|y": (0.5, 0.5), "z|x": (0.2, 0.2), "z|y": (0.5, 0.5)},
    )
    values = scores(chain_map, st)
    assert values["coherence_rmse"] == pytest.approx(0.3)
    assert values["coherence_normalized"] == pytest.approx(-1.0)


def test_a_single_target_leaves_no_baseline(simple_map):
    confirming, disconfirming = pairs_by_sign(simple_map)
    b = {"x1": 0.25, "x2": 0.5, "y": 0.9}
    conds = {confirming.key: (0.8, 0.4), disconfirming.key: (0.2, 0.8)}
    values = scores(simple_map, state(simple_map, b, conds))
    assert values["coherence_rmse"] == pytest.approx(0.4)
    assert math.isnan(values["coherence_normalized"])


@pytest.mark.parametrize(
    ("obs", "null", "expected"),
    [(0.0, 0.2, 1.0), (0.2, 0.2, 0.0), (0.2, 0.0, -1.0), (0.1, 0.3, 0.5)],
)
def test_normalized_coherence_is_a_symmetric_ratio(obs, null, expected):
    assert normalized_coherence(obs, null) == pytest.approx(expected)


@pytest.mark.parametrize(("obs", "null"), [(0.0, 0.0), (float("nan"), 0.2), (0.2, float("nan"))])
def test_normalized_coherence_is_undefined_without_spread(obs, null):
    assert math.isnan(normalized_coherence(obs, null))


def test_information_is_the_cost_of_fence_sitting(simple_map):
    flat = state(simple_map, {"x1": 0.5, "x2": 0.5, "y": 0.5})
    decided = state(simple_map, {"x1": 1.0, "x2": 0.0, "y": 1.0})
    assert scores(simple_map, flat)["informativeness"] == pytest.approx(0.0)
    assert scores(simple_map, decided)["informativeness"] == pytest.approx(1.0)
    assert information_penalties(simple_map, flat) == pytest.approx([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# the degenerate corners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marginal", [0.5, 0.0, 1.0])
def test_a_constant_belief_state_scores_chance_on_relevance(simple_map, marginal):
    """Fence-sitting, denying everything and believing everything all score 0.

    A constant state reports the same credence in the query under both
    suppositions, so its confirmation is exactly zero on every pair, whatever
    the entailed sign. It stakes nothing, so the head score is undefined for it
    and only strength, informativeness and coherence place it.
    """
    values = scores(simple_map, BeliefState.constant(simple_map, marginal, 1.0))
    assert values["relevance_strength"] == pytest.approx(0.0)
    assert values["misdirected_relevance"] == pytest.approx(0.0)
    assert math.isnan(values["weighted_relevance_direction"])


@pytest.mark.parametrize("credence", [0.5, 0.0, 1.0])
def test_a_constant_state_has_no_gap_and_no_coherence_score(chain_map, credence):
    """No gap, but no spread either: the model's own marginal cannot beat a mismatched one."""
    values = scores(chain_map, BeliefState.constant(chain_map, credence, credence))
    assert values["coherence_rmse"] == pytest.approx(0.0)
    assert math.isnan(values["coherence_normalized"])
    assert values["relevance_strength"] == pytest.approx(0.0)


def test_random_credences_sit_near_chance_on_coherence():
    argmap = make_map(
        [(t, "r", 1) for t in "abc"]
        + [(f"{t}{i}", t, 1 if i else -1) for t in "abc" for i in range(2)],
    )
    rng = random.Random(0)
    draws = [
        scores(argmap, BeliefState.random(argmap, rng))["coherence_normalized"] for _ in range(400)
    ]
    assert all(-1.0 <= d <= 1.0 for d in draws)
    assert sum(draws) / len(draws) == pytest.approx(0.0, abs=0.05)


def test_random_credences_sit_near_chance():
    """Chance is 0 whatever the map's size, which is why there is no baseline."""
    argmap = make_map([(f"x{i}", "y", 1 if i % 2 else -1) for i in range(12)])
    rng = random.Random(0)
    for key in ("weighted_relevance_direction", "relevance_strength"):
        draws = [scores(argmap, BeliefState.random(argmap, rng))[key] for _ in range(400)]
        assert sum(draws) / len(draws) == pytest.approx(0.0, abs=0.03)


# ---------------------------------------------------------------------------
# missing elicitations
# ---------------------------------------------------------------------------


def test_missing_credences_shrink_the_index_sets(simple_map):
    st = BeliefState(
        marginals={"x1": None, "x2": 0.5, "y": 0.5},
        conditionals={p.key: (0.6, 0.4) for p in simple_map.pairs},
    )
    prof = profile(simple_map, st, L)
    assert prof.n_claims == 2  # x1 dropped
    assert prof.n_pairs == 2  # relevance needs no marginal at all
    # coherence does, so the pair supposing x1 drops from that axis
    assert len(coherence_terms(simple_map, st)) == 1
    assert st.coverage(simple_map) == pytest.approx(6 / 7)


def test_missing_conditional_drops_only_its_pair(simple_map):
    confirming, _ = pairs_by_sign(simple_map)
    st = BeliefState(
        marginals={"x1": 0.9, "x2": 0.9, "y": 0.1},
        conditionals={
            p.key: ((1.0, None) if p == confirming else (0.2, 0.8)) for p in simple_map.pairs
        },
    )
    prof = profile(simple_map, st, L)
    assert prof.n_pairs == 1
    assert prof.n_claims == 3
    assert strengths(simple_map, st) == {1: pytest.approx([0.6])}  # the disconfirming pair alone


def test_an_empty_index_set_scores_nan():
    argmap = make_map([("x", "y", 1)])
    st = BeliefState(marginals={"x": None, "y": None}, conditionals={})
    values = scores(argmap, st)
    assert all(math.isnan(values[key]) for key in score_keys(L))


def test_profile_serialises_to_every_score_key_whatever_the_map(simple_map):
    prof = profile(simple_map, BeliefState.constant(simple_map, 0.5, 0.5), L)
    assert tuple(prof.to_dict()) == score_keys(L)
    assert set(AGGREGATE_KEYS) <= set(prof.to_dict())
    assert all(isinstance(v, float) for v in prof.to_dict().values())


def test_score_keys_add_a_breakdown_per_distance():
    assert score_keys(2) == AGGREGATE_KEYS + (
        "weighted_relevance_direction_d1",
        "weighted_relevance_direction_d2",
        "misdirected_relevance_d1",
        "misdirected_relevance_d2",
        "relevance_strength_d1",
        "relevance_strength_d2",
    )
