"""Write the map manifest of a corpus split, and check eval logs against it.

The manifest is the unit of the paired design: every run on a corpus must draw
exactly the maps it lists. ``export`` builds the dataset as the task does and
writes the sample ids, in dataset order, plus per-map shape statistics::

    uv run python -m belcy_eval.scripts.export_manifest export \\
        --corpus synthetic_corpus-001:train --n-maps 1000

``check`` compares the sample ids of an Inspect log with a manifest::

    uv run python -m belcy_eval.scripts.export_manifest check \\
        --manifest synthetic_corpus-001.ids.txt logs/run.eval
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log_sample_summaries

from belcy_eval.datasets import map_from_metadata, parse_corpora, syncialo_dataset

DEFAULT_OUT_DIR = Path(".")

STATS_COLUMNS = (
    "id",
    "debate",
    "n_claims",
    "n_edges",
    "n_roots",
    "depth",
    "n_pairs",
    "pairs_d1",
    "pairs_d2",
    "pairs_d3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="write the manifest of one corpus split")
    export.add_argument("--corpus", required=True, help="one corpus_id:split, never a mix")
    export.add_argument("--n-maps", type=int, default=1000)
    export.add_argument("--min-claims", type=int, default=12)
    export.add_argument("--max-claims", type=int, default=20)
    export.add_argument("--max-path-length", type=int, default=3)
    export.add_argument("--max-pairs", type=int, default=30, help="0 keeps every pair")
    export.add_argument("--no-mece-roots", dest="mece_roots", action="store_false")
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--local-dir", default=None, help="local syncIALO snapshot")
    export.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    check = commands.add_parser("check", help="compare eval logs with a manifest")
    check.add_argument("--manifest", type=Path, required=True, help="an .ids.txt file")
    check.add_argument("logs", nargs="+", help="Inspect .eval logs")
    return parser.parse_args()


def export(args: argparse.Namespace) -> None:
    sources = parse_corpora(args.corpus)
    if len(sources) != 1:
        raise SystemExit("export takes one corpus split, not a mix")
    corpus_id = sources[0].corpus_id

    dataset = syncialo_dataset(
        corpora=args.corpus,
        n_maps=args.n_maps,
        min_claims=args.min_claims,
        max_claims=args.max_claims,
        max_path_length=args.max_path_length,
        max_pairs=args.max_pairs,
        mece_roots=args.mece_roots,
        seed=args.seed,
        local_dir=args.local_dir,
    )

    rows: list[dict[str, object]] = []
    for sample in dataset:
        argmap = map_from_metadata(sample.metadata or {})
        by_distance = argmap.pairs_by_distance()
        row: dict[str, object] = {
            "id": sample.id,
            "debate": argmap.source.get("debate"),
            "n_claims": argmap.n_claims,
            "n_edges": argmap.n_edges,
            "n_roots": len(argmap.roots),
            "depth": argmap.depth,
            "n_pairs": argmap.n_pairs,
        }
        row.update({f"pairs_d{d}": by_distance.get(d, 0) for d in (1, 2, 3)})
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.out_dir / f"{corpus_id}.ids.txt"
    ids_path.write_text("".join(f"{row['id']}\n" for row in rows))
    stats_path = args.out_dir / f"{corpus_id}.stats.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} maps -> {ids_path}, {stats_path}")


def check_log(log: str, manifest: list[str]) -> bool:
    """Report whether a log's sample ids are exactly the manifest's."""
    ids = [str(s.id) for s in read_eval_log_sample_summaries(log)]
    expected = set(manifest)
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    ok = not missing and not extra
    status = "ok" if ok else "MISMATCH"
    print(f"{status}: {log} ({len(set(ids))} ids, manifest {len(expected)})")
    for label, found in (("missing", missing), ("extra", extra)):
        if found:
            print(f"  {len(found)} {label}:")
            for sample_id in found:
                print(f"    {sample_id}")
    return ok


def check(args: argparse.Namespace) -> None:
    manifest = [line for line in args.manifest.read_text().splitlines() if line]
    results = [check_log(log, manifest) for log in args.logs]
    if not all(results):
        sys.exit(1)


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
