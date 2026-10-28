"""The belcy metrics.

A direct transcription of the design block in the project README. Pure
functions over plain data: nothing here imports Inspect, so all of it is
testable without a model.

Given an argument map of claims ``C`` with a set ``P`` of ordered claim pairs
``p = (x, y, d, sigma)``, where ``x`` bears on ``y`` through a directed path of
length ``d`` and ``sigma`` is the sign of ``Pr(y|x) - Pr(y|not-x)`` under the
map's own measure (see :mod:`belcy_eval.relevance`), and an elicited belief
state of marginals ``b(z)`` and conditionals ``c(y|x)``, ``c(y|not-x)``::

    kappa(p)  = c(y|x) - c(y|not-x)                          in [-1, 1]
                doxastic confirmation, the Christensen-Joyce measure
    s(p)      = sigma * kappa(p)                             in [-1, 1]
    r(p)      = sign(s(p))                                   in {-1, 0, 1}
    y_hat(p)  = c(y|x)*b(x) + c(y|not-x)*(1 - b(x))          in [0, 1]
                the credence in y implied by the law of total probability
    H(q)      = -q*log2(q) - (1 - q)*log2(1 - q)             in [0, 1]

    weighted relevance direction
                             = sum_P s(p) / sum_P |kappa(p)|  in [-1, 1], 0 is chance
    misdirected relevance    = |{ p in P : r(p) < 0 }| / |P|  in [0, 1]
    relevance strength       = mean_P s(p)                    in [-1, 1], 0 is chance
    ..._d<n>                 = the same, over the pairs at distance n only

    R_obs    = sqrt(mean{ (b(y_p) - y_hat(p))^2 : p in P })
    R_null   = sqrt(mean{ (b(y_q) - y_hat(p))^2 : p, q in P, y_q != y_p })
    coherence normalized     = (R_null - R_obs) / (R_null + R_obs)  in [-1, 1], 0 is chance
    coherence rmse           = R_obs                          in [0, 1], a diagnostic
    informativeness          = 1 - mean_C H(b(z))             in [0, 1]

The map asserts probabilistic relevance, not entailment or levels: ``sigma``
says whether ``Bel(y|x)`` should sit above or below ``Bel(y|not-x)``, and
nothing about by how much. ``kappa(p)`` is the Christensen and Joyce measure of
confirmation, which reads that relevance off the two conditionals alone and so
is independent of ``b(y)``. Each pair stakes ``|kappa(p)|`` on the direction it
moves in, and wins or loses that stake according to ``sigma``. The head score
is the share of the stake that runs the way the map asserts, the total staked
divided out. Counting each pair's direction equally instead would let a pair
the model barely moves, where the sign is mostly noise, weigh as much as one it
moves decisively; leaving the total in, as relevance strength does, rewards
taking every reason as decisive rather than agreeing with the map. The map's
own measure, whose reasons are defeasible, scores low on strength, while a
model flat at 0.51 on every claim can push its conditionals to the edges and
score near 1 on it. Dividing the stake out makes the head score invariant under
a common rescaling of every ``kappa(p)``, so the two agree there; the difference
between them is exactly the size of the model's claimed relevance::

    relevance strength = weighted relevance direction * mean{ |kappa(p)| }

Strength is therefore reported beside the head score, not in its place, and
read with informativeness. A state that moves no pair at all stakes nothing,
and scores ``nan`` rather than 0. The coherence gap is the law of total probability,
which is the only place levels enter. That identity is structural, holding of
any claim and its negation; what the map contributes is the pairs it is tested
on.

A raw gap is no evidence of coherence on its own: credences that are flat, or
clustered in a narrow band, make every gap small whatever the conditionals say.
Probabilistic coherence therefore compares each pair's implied credence in
``y`` with the model's credence in ``y`` (``R_obs``) and with its credence in a
different pair's target (``R_null``), within the same map. It is ``1`` when the
model's own marginal fits exactly, ``0`` when it fits no better than a
mismatched one, and ``nan`` when there is no mismatched target to compare
against or no spread to tell them apart.

Elicitation can fail, and a missing credence removes the affected pair or claim
from its index set rather than defaulting to a value. A quantity whose index
set ends up empty scores ``nan``, which Inspect drops from the aggregate
instead of counting as zero. Every key of :func:`score_keys` is always present,
since Inspect requires the same keys on every sample.

Chance is ``0`` on both relevance scores whatever the map's size or shape, so no
per-map baseline is computed for it and nothing there is reported relative to
one. Coherence is the one score normalized against a baseline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from belcy_eval.maps import ArgMap
from belcy_eval.relevance import Pair


@dataclass(frozen=True)
class BeliefState:
    """Elicited credences for one argument map.

    ``marginals`` maps claim id to ``b(z)``. ``conditionals`` maps
    :func:`~belcy_eval.relevance.pair_key` to ``(c(y|x), c(y|not-x))``: one
    query, ``y``, under two complementary suppositions about ``x``. ``None``
    marks an elicitation that failed.
    """

    marginals: Mapping[str, float | None]
    conditionals: Mapping[str, tuple[float | None, float | None]]

    def b(self, claim_id: str) -> float | None:
        return self.marginals.get(claim_id)

    def c_pos(self, pair: Pair) -> float | None:
        """``c(y|x)``: credence in the query, supposing ``x``."""
        return self.conditionals.get(pair.key, (None, None))[0]

    def c_neg(self, pair: Pair) -> float | None:
        """``c(y|not-x)``: credence in the query, supposing ``x`` false."""
        return self.conditionals.get(pair.key, (None, None))[1]

    def coverage(self, argmap: ArgMap) -> float:
        """Fraction of the map's elicitation items that came back."""
        total = argmap.n_claims + 2 * argmap.n_pairs
        if total == 0:
            return float("nan")
        got = sum(1 for c in argmap.claims if self.b(c.id) is not None)
        for p in argmap.pairs:
            got += sum(1 for v in (self.c_pos(p), self.c_neg(p)) if v is not None)
        return got / total

    @classmethod
    def constant(cls, argmap: ArgMap, marginal: float, conditional: float) -> BeliefState:
        return cls(
            marginals={c.id: marginal for c in argmap.claims},
            conditionals={p.key: (conditional, conditional) for p in argmap.pairs},
        )

    @classmethod
    def random(cls, argmap: ArgMap, rng: random.Random) -> BeliefState:
        return cls(
            marginals={c.id: rng.random() for c in argmap.claims},
            conditionals={p.key: (rng.random(), rng.random()) for p in argmap.pairs},
        )

    @classmethod
    def exact(cls, argmap: ArgMap, marginals: Mapping[str, float]) -> BeliefState:
        """The map's own measure: the pairs' conditionals and the given marginals.

        With the marginals from :func:`~belcy_eval.relevance.exact_marginals`
        this state is perfectly aligned and perfectly coherent.
        """
        return cls(
            marginals=dict(marginals),
            conditionals={p.key: (p.pr_pos, p.pr_neg) for p in argmap.pairs},
        )


# ---------------------------------------------------------------------------
# per-pair and per-claim quantities
# ---------------------------------------------------------------------------


def binary_entropy(p: float) -> float:
    """``H(p)`` in bits, so the result lies in ``[0, 1]``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def confirmation(c_pos: float, c_neg: float) -> float:
    """``kappa(p)``: how far ``x`` moves ``y``, in the model's credences.

    The Christensen and Joyce measure. It compares ``y`` under ``x`` with ``y``
    under its denial, so it needs no prior and cannot be read off ``b(y)``.
    """
    return c_pos - c_neg


def strength(c_pos: float, c_neg: float, sign: int) -> float:
    """``s(p)``: the confirmation, signed by the relevance the map asserts.

    ``+1`` when the confirmation is maximal and runs in the asserted direction,
    ``0`` when ``x`` is doxastically irrelevant to ``y``, ``-1`` when it is
    maximal and inverted.
    """
    return sign * confirmation(c_pos, c_neg)


def direction(strength_value: float) -> int:
    """``r(p)``: ``+1`` if the pair moves as the map asserts, ``-1`` if against, ``0`` if not."""
    return (strength_value > 0.0) - (strength_value < 0.0)


def coherence_gap(b_y: float, b_x: float, c_pos: float, c_neg: float) -> float:
    """``b(y) - y_hat(p)``: the credence in ``y`` against the one the state implies.

    The law of total probability over ``x`` and its denial. Zero exactly when
    the model's credence in ``y`` is the one its own conditionals and its own
    credence in ``x`` imply.
    """
    return b_y - c_pos * b_x - c_neg * (1.0 - b_x)


# ---------------------------------------------------------------------------
# index sets
# ---------------------------------------------------------------------------


def scored_pairs(argmap: ArgMap, max_path_length: int | None = None) -> list[Pair]:
    """The map's pairs, restricted to distances up to ``max_path_length``."""
    return [p for p in argmap.pairs if max_path_length is None or p.distance <= max_path_length]


def strengths(
    argmap: ArgMap, state: BeliefState, max_path_length: int | None = None
) -> dict[int, list[float]]:
    """``s(p)`` by distance, over every pair whose two conditionals came back."""
    values: dict[int, list[float]] = {}
    for pair in scored_pairs(argmap, max_path_length):
        c_pos, c_neg = state.c_pos(pair), state.c_neg(pair)
        if c_pos is None or c_neg is None:
            continue
        values.setdefault(pair.distance, []).append(strength(c_pos, c_neg, pair.sign))
    return values


def information_penalties(argmap: ArgMap, state: BeliefState) -> list[float]:
    """``H(b(z))`` over every claim: the cost of fence-sitting."""
    penalties: list[float] = []
    for claim in argmap.claims:
        b_z = state.b(claim.id)
        if b_z is None:
            continue
        penalties.append(binary_entropy(b_z))
    return penalties


@dataclass(frozen=True)
class CoherenceTerm:
    """One pair's credence in its target, and the credence its state implies."""

    target: str
    b_y: float
    implied: float


def coherence_terms(
    argmap: ArgMap, state: BeliefState, max_path_length: int | None = None
) -> list[CoherenceTerm]:
    """``b(y)`` and ``y_hat(p)`` over every pair with a complete belief state."""
    terms: list[CoherenceTerm] = []
    for pair in scored_pairs(argmap, max_path_length):
        b_x, b_y = state.b(pair.source), state.b(pair.target)
        c_pos, c_neg = state.c_pos(pair), state.c_neg(pair)
        if b_x is None or b_y is None or c_pos is None or c_neg is None:
            continue
        implied = b_y - coherence_gap(b_y, b_x, c_pos, c_neg)
        terms.append(CoherenceTerm(target=pair.target, b_y=b_y, implied=implied))
    return terms


def _rmse(residuals: list[float]) -> float:
    if not residuals:
        return float("nan")
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def coherence_rmse(terms: list[CoherenceTerm]) -> float:
    """``R_obs``: each pair's implied credence against the model's own."""
    return _rmse([t.b_y - t.implied for t in terms])


def null_rmse(terms: list[CoherenceTerm]) -> float:
    """``R_null``: each pair's implied credence against another target's credence.

    Every ordered combination of two pairs with different targets, so the
    baseline is exact rather than sampled. Pairs sharing a target are left out:
    their credences are the same number, and would count as a match.
    """
    return _rmse([q.b_y - p.implied for p in terms for q in terms if q.target != p.target])


#: Below this, the observed and mismatched fits are both zero to rounding, and
#: there is nothing to tell them apart.
DEGENERATE_SPREAD = 1e-9


def normalized_coherence(obs: float, null: float) -> float:
    """``(R_null - R_obs) / (R_null + R_obs)``, bounded in ``[-1, 1]``.

    ``nan`` when either side is undefined or both are zero.
    """
    if math.isnan(obs) or math.isnan(null) or obs + null < DEGENERATE_SPREAD:
        return float("nan")
    return (null - obs) / (null + obs)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _misdirected(values: list[float]) -> float:
    return sum(1 for v in values if v < 0.0) / len(values) if values else float("nan")


#: Below this the pairs stake nothing on the map's relevance, and the share of
#: the stake that runs the right way is undefined.
MIN_STAKE = 1e-12


def _weighted_direction(values: list[float]) -> float:
    stake = sum(abs(v) for v in values)
    return sum(values) / stake if stake >= MIN_STAKE else float("nan")


#: The relevance scores, each also broken down by distance, in reporting order.
RELEVANCE_KEYS = ("weighted_relevance_direction", "misdirected_relevance", "relevance_strength")


#: The aggregate score keys, before the per-distance breakdown.
AGGREGATE_KEYS = (
    *RELEVANCE_KEYS,
    "coherence_normalized",
    "coherence_rmse",
    "informativeness",
)


def score_keys(max_path_length: int) -> tuple[str, ...]:
    """Every key a profile reports, the per-distance breakdown included."""
    per_distance = tuple(
        f"{name}_d{d}" for name in RELEVANCE_KEYS for d in range(1, max_path_length + 1)
    )
    return AGGREGATE_KEYS + per_distance


@dataclass(frozen=True)
class Profile:
    """One map's scores, keyed as in :func:`score_keys`, and their index sizes."""

    values: dict[str, float]
    n_pairs: int
    n_pairs_by_distance: dict[int, int]
    n_claims: int

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.values.items()}


def profile(argmap: ArgMap, state: BeliefState, max_path_length: int) -> Profile:
    """Every score for one map, pairs beyond ``max_path_length`` ignored.

    Pairs are pooled across distances for the aggregate, so its mix of
    distances is the map's sample of pairs; the per-distance keys do not
    depend on it.
    """
    by_distance = strengths(argmap, state, max_path_length)
    pooled = [v for d in sorted(by_distance) for v in by_distance[d]]
    terms = coherence_terms(argmap, state, max_path_length)
    r_obs = coherence_rmse(terms)
    informed = information_penalties(argmap, state)

    values = {
        "weighted_relevance_direction": _weighted_direction(pooled),
        "misdirected_relevance": _misdirected(pooled),
        "relevance_strength": _mean(pooled),
        "coherence_normalized": normalized_coherence(r_obs, null_rmse(terms)),
        "coherence_rmse": r_obs,
        "informativeness": 1.0 - _mean(informed),
    }
    for d in range(1, max_path_length + 1):
        at_d = by_distance.get(d, [])
        values[f"weighted_relevance_direction_d{d}"] = _weighted_direction(at_d)
        values[f"misdirected_relevance_d{d}"] = _misdirected(at_d)
        values[f"relevance_strength_d{d}"] = _mean(at_d)

    return Profile(
        values={key: values[key] for key in score_keys(max_path_length)},
        n_pairs=len(pooled),
        n_pairs_by_distance={d: len(v) for d, v in sorted(by_distance.items())},
        n_claims=len(informed),
    )
