"""Print a markdown report of belcy eval results.

Usage::

    uv run python -m belcy_eval.scripts.print_evals --log-dir logs/run-1
"""

from __future__ import annotations

import argparse

from belcy_eval.scripts.evals_summary import (
    distance_table,
    eval_config,
    format_config,
    load_evals,
    metrics_table,
    model_names,
    model_slice,
)

PREAMBLE = """\
Three axes, never traded off against each other.

**Weighted relevance direction** measures whether the model's own credences move
in the directions of probabilistic relevance the map entails: each pair stakes
how far its conditionals move, and wins or loses that stake according to the
direction the map says, the total staked divided out. A pair the model barely
moves therefore counts for little, and moving every pair further buys nothing.
The map is read as a
propositional theory (each reason defeasibly implies or excludes its target;
rival root positions exclude each other, unless `mece_roots` is off), and the
relevance of claim `x` to claim `y` is computed exactly under the uniform
measure over its models, for pairs joined by a directed path of up to
`max_path_length` edges, where a step between rival roots counts as one. It runs from
`-1` to `+1`, and `0` is chance whatever the size or shape of the map, so the
raw figures are comparable across a heterogeneous corpus with no baseline drawn.

The per-distance table is the primary view: the aggregate pools a capped sample
of pairs, so its mix of distances is a sampling choice. Distance 1 is the
direct reasons.

**Misdirected relevance** is the one relevance column where lower is better. It
is the share of pairs whose confirmation runs against the entailed relevance,
counted rather than weighted, which a mean over a large map dilutes. It is the
tail of the score above, not a fourth axis.

**Relevance strength** leaves the stake in: it is the weighted direction times
how far the conditionals move on average. The map sets no size, so strength
rewards taking reasons as decisive, not agreeing with the map: the map's own
measure scores low on it, and a model flat at 0.51 can score near 1. Read it
with informativeness, never on its own.

A run made before the weighted score existed shows a blank column for it, and
keeps its own `relevance direction` (the unweighted count) in the table by
distance.

**Probabilistic coherence** falls when the marginals and the conditionals
cannot belong to one distribution, by the law of total probability, tested at
the pairs the map says are argumentatively connected. Reported as
`coherence_normalized`, it compares, within each
map, how well the model's credence in a pair's target fits the credence its
conditionals imply (RMSE) with how well the credence in a different target
fits it: `(null − obs) / (null + obs)`, from `-1` to `+1`, with `0` as chance.
Flat or clustered credences therefore score `0`, not `1`, and maps with no
spread at all score `nan`. `coherence_rmse` is the raw gap behind it, a
diagnostic in the units of credence where lower is better.

**Informativeness** falls when the model sits on the fence, which matters
because the relevance scores read the conditionals alone and are blind to levels.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_evals(args.log_dir)

    print("# belcy-eval results\n")
    print(PREAMBLE)

    if df.empty:
        print("\nNo evals found.")
        return

    for model in model_names(df):
        rows = model_slice(df, model)
        print(f"\n## {model}\n")

        configs = {format_config(eval_config(row)) for _, row in rows.iterrows()}
        for config in sorted(c for c in configs if c):
            print(f"- {config}")
        if configs:
            print()

        table = metrics_table(rows)
        print(table.to_markdown(index=False, floatfmt=f".{args.decimals}f"))

        by_distance = distance_table(rows)
        if not by_distance.empty:
            print("\nBy distance:\n")
            print(by_distance.to_markdown(index=False, floatfmt=f".{args.decimals}f"))


if __name__ == "__main__":
    main()
