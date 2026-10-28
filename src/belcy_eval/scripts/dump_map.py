"""Print sampled maps and the prompts they generate, without calling a model.

The prompt wording is the measurement instrument, so read it before spending
GPU time on a run::

    uv run python -m belcy_eval.scripts.dump_map --corpora synthetic_corpus-001:eval --limit 2
    uv run python -m belcy_eval.scripts.dump_map --show-map --prompts 4
    uv run python -m belcy_eval.scripts.dump_map --personas schwartz
"""

from __future__ import annotations

import argparse
import random

from belcy_eval.datasets import (
    DEFAULT_CORPORA,
    map_from_metadata,
    persona_from_metadata,
    syncialo_dataset,
)
from belcy_eval.elicit import items_for, render_map, system_prompt
from belcy_eval.metrics import BeliefState, profile, score_keys
from belcy_eval.personas import GENERATORS
from belcy_eval.relevance import exact_marginals

#: Degenerate strategies: one credence everywhere, and conditionals that agree
#: with it, so each corner has a coherence rmse of 0, but with no spread to
#: beat a mismatched baseline its coherence normalized is nan. All three
#: score exactly 0 on relevance strength, and nan on the weighted direction,
#: having staked nothing, which is why they are printed on every map. The map's
#: own measure is printed beside them as the ceiling of the weighted direction
#: and of coherence (both 1); its strength is low, since the map's reasons are
#: defeasible.
CORNERS: dict[str, float] = {"flat_half": 0.5, "all_deny": 0.0, "all_accept": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora",
        default=DEFAULT_CORPORA,
        help="corpus mix as comma-joined corpus_id:split[:weight] entries",
    )
    parser.add_argument("--limit", type=int, default=1, help="maps to print")
    parser.add_argument("--min-claims", type=int, default=12)
    parser.add_argument("--max-claims", type=int, default=20)
    parser.add_argument("--max-path-length", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=30, help="0 keeps every pair")
    parser.add_argument(
        "--no-mece-roots",
        dest="mece_roots",
        action="store_false",
        help="read several roots as independent, and test directed-path pairs only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-map", action="store_true", help="render map-visible prompts")
    parser.add_argument("--personas", choices=GENERATORS, default="none", help="persona generator")
    parser.add_argument("--prompts", type=int, default=2, help="example prompts per map")
    parser.add_argument("--local-dir", default=None, help="local syncIALO snapshot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = syncialo_dataset(
        corpora=args.corpora,
        n_maps=args.limit,
        min_claims=args.min_claims,
        max_claims=args.max_claims,
        max_path_length=args.max_path_length,
        max_pairs=args.max_pairs,
        mece_roots=args.mece_roots,
        seed=args.seed,
        personas=args.personas,
        local_dir=args.local_dir,
    )

    for sample in dataset:
        argmap = map_from_metadata(sample.metadata or {})
        source = argmap.source
        print(f"\n{'=' * 78}")
        print(
            f"map {argmap.map_id}  from {source.get('corpus_id')}:{source.get('split')}"
            f" {source.get('debate')}"
        )
        print(
            f"{argmap.n_claims} claims, {argmap.n_edges} edges, {len(argmap.roots)} root(s)"
            f"{' (rival)' if argmap.mece_roots and len(argmap.roots) > 1 else ''}, "
            f"depth {argmap.depth}"
        )
        print(f"topic: {argmap.topic}")
        print(f"{'=' * 78}\n")
        print(render_map(argmap))

        by_distance = ", ".join(f"d{d}: {n}" for d, n in argmap.pairs_by_distance().items())
        print(f"\n{argmap.n_pairs} pairs ({by_distance})\n")
        print(f"{'x':<28} {'y':<28} {'d':>2} {'sign':>4} {'Pr(y|x)':>8} {'Pr(y|~x)':>8}")
        for pair in argmap.pairs:
            print(
                f"{pair.source[:28]:<28} {pair.target[:28]:<28} {pair.distance:>2} "
                f"{pair.sign:>+4d} {pair.pr_pos:>8.3f} {pair.pr_neg:>8.3f}"
            )

        items = items_for(argmap, show_map=args.show_map)
        marginals = [i for i in items if i.kind == "marginal"]
        conditionals = [i for i in items if i.kind == "conditional"]
        print(
            f"\n{len(items)} items: {len(marginals)} marginals, {len(conditionals)} conditionals\n"
        )

        print(f"{'-' * 78}\n[system]\n{'-' * 78}")
        print(system_prompt(persona_from_metadata(sample.metadata or {})))
        print()

        for item in marginals[: args.prompts] + conditionals[: args.prompts]:
            tag = item.kind if not item.direction else f"{item.kind}/{item.direction}"
            print(f"{'-' * 78}\n[{tag}] {item.key}\n{'-' * 78}")
            print(item.prompt(swapped=False))
            print()

        # What the scores would look like without a model in the loop.
        length = args.max_path_length
        rng = random.Random(argmap.map_id)
        states = {"exact": BeliefState.exact(argmap, exact_marginals(argmap))}
        states["random"] = BeliefState.random(argmap, rng)
        for name, credence in CORNERS.items():
            states[name] = BeliefState.constant(argmap, credence, credence)
        rows = {name: profile(argmap, state, length).to_dict() for name, state in states.items()}

        keys = score_keys(length)
        width = 32 + 12 * len(rows)
        print(f"{'-' * width}\nreference profiles for this map\n{'-' * width}")
        print(f"{'score':<32}" + "".join(f"{name:>12}" for name in rows))
        for key in keys:
            print(f"{key:<32}" + "".join(f"{rows[name][key]:>12.3f}" for name in rows))


if __name__ == "__main__":
    main()
