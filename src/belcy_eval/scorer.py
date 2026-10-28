"""Scoring a belief state against the map it was elicited on.

Three axes, never traded off against each other and never combined into a
number. ``weighted_relevance_direction`` asks whether the model's own credences
move in the directions of probabilistic relevance the map entails between
claims up to ``max_path_length`` apart, each pair weighted by how far it moves
and the total divided out, so that a pair the model barely moves counts for
little and overstating every reason buys nothing. ``misdirected_relevance``
reports the share of pairs that move the wrong way, counted rather than
weighted, which a mean over a large map would dilute. ``relevance_strength``
keeps the total in, the head score times the mean distance moved; the map sets
no size, so it is secondary. All three
are also broken down by distance (``_d1``, ``_d2``, ...), and every one of
those keys is present on every sample, ``nan`` where a map has no pair at that
distance, because Inspect requires the same keys throughout.

``coherence_normalized`` catches conditionals and marginals that cannot
belong to one distribution, by the law of total probability, tested at the
pairs the map says are argumentatively connected. It is normalized within the
map: the model's credence in each pair's target is set against the credence
its conditionals imply, and so is the credence in a different target, so flat
or clustered credences score ``0`` rather than ``1``. ``coherence_rmse`` is the
raw gap behind it, reported as a diagnostic. ``informativeness`` catches a
model that buys relevance strength by suspending judgement on every claim,
since the confirmation measure reads the two conditionals alone and is blind to
levels.

Chance on ``weighted_relevance_direction`` and ``relevance_strength`` is 0 whatever the map's
size or shape, so no per-map baseline is drawn for it and nothing there is reported relative
to one. A state that moves no pair at all stakes nothing on the map's relevance and scores
``nan`` on the head score, where strength scores 0.

The score value holds the three axes, the tail and the strength of the first, their breakdown,
the coherence diagnostic, ``elicitation_coverage``, and how the elicitation itself went
(``failure_rate``, ``overflow_rate``, ``unconstrained``), and nothing else: every
key in it is aggregated by
the ``"*"`` metric glob, so a quantity that is a property of the map rather
than a measurement of the model would be reported as though it were a score.
The diagnostics are in the value rather than the metadata so that a run degraded by a
truncated context window or a rejected constraint is visible in the reported table.

A quantity whose index set is empty on a map scores ``nan`` there. Inspect drops those
per key before it calls a metric, and reports the survivors as that key's
``scored_samples``, so an aggregate is a mean over the maps that defined the quantity
and no guard is needed here.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from belcy_eval.datasets import map_from_metadata
from belcy_eval.metrics import AGGREGATE_KEYS, BeliefState, profile

logger = logging.getLogger(__name__)


def belief_state_from_metadata(metadata: dict[str, Any]) -> BeliefState:
    """Rebuild the elicited belief state from the solver's metadata."""
    conditionals = {
        key: (value.get("pos"), value.get("neg"))
        for key, value in (metadata.get("conditionals") or {}).items()
    }
    return BeliefState(
        marginals=metadata.get("credences") or {},
        conditionals=conditionals,
    )


@scorer(metrics={"*": [mean(), stderr()]})
def belcy_scorer(max_path_length: int = 3):
    """Score one map's elicited belief state on the three axes.

    Pairs further apart than ``max_path_length`` are ignored, and it fixes the
    per-distance keys, so it must match the task's.
    """

    async def score(state: TaskState, target: Target) -> Score:
        argmap = map_from_metadata(state.metadata)
        belief = belief_state_from_metadata(state.metadata)

        prof = profile(argmap, belief, max_path_length)
        value = prof.to_dict()
        value["elicitation_coverage"] = belief.coverage(argmap)

        # Elicitation diagnostics, so a degraded run says so in the reported
        # table rather than only in the log's sample metadata. They are rates
        # rather than counts, since the call budget is a property of the map.
        n_calls = state.metadata.get("n_calls") or 0
        value["failure_rate"] = (
            (state.metadata.get("n_failures", 0) / n_calls) if n_calls else float("nan")
        )
        value["overflow_rate"] = (
            (state.metadata.get("n_overflows", 0) / n_calls) if n_calls else float("nan")
        )
        value["unconstrained"] = 0.0 if state.metadata.get("constrained", True) else 1.0

        defined = [key for key in AGGREGATE_KEYS if not math.isnan(value[key])]
        explanation = ", ".join(f"{key} {value[key]:.3f}" for key in defined)
        if len(defined) < len(AGGREGATE_KEYS):
            undefined = sorted(set(AGGREGATE_KEYS) - set(defined))
            explanation += f" (undefined: {', '.join(undefined)})"
            logger.debug("map %s has empty index sets for %s", argmap.map_id, undefined)

        return Score(
            value=value,
            explanation=explanation,
            metadata={
                "map_id": argmap.map_id,
                "index_sizes": {
                    "claims": prof.n_claims,
                    "pairs": prof.n_pairs,
                    "pairs_by_distance": prof.n_pairs_by_distance,
                },
                "credences": dict(belief.marginals),
                "conditionals": state.metadata.get("conditionals"),
                "n_calls": state.metadata.get("n_calls"),
                "n_failures": state.metadata.get("n_failures"),
                "n_overflows": state.metadata.get("n_overflows"),
                "constrained": state.metadata.get("constrained"),
            },
        )

    return score
