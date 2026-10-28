"""Shared helpers for turning belcy eval logs into tables.

The column contract lives here so the reporting scripts agree on it. Inspect
names a dict-valued score's columns ``score_<key>_<metric>``, and qualifies
them with the scorer name when the key would otherwise be ambiguous, so both
spellings are tried.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from inspect_ai.analysis import evals_df

from belcy_eval.metrics import AGGREGATE_KEYS, RELEVANCE_KEYS

SCORER = "belcy_scorer"
TASK_COLUMN = "task_name"
MODEL_COLUMN = "model"


#: Score keys renamed since, by their current name. Logs written before the
#: rename carry the old one, which measured the same quantity.
LEGACY_KEYS = {"relevance_strength": "relevance_alignment"}

#: Score keys no longer reported, which older logs still carry. They are kept
#: in the per-distance table so that a run made before they were retired reads
#: as it did; no current run writes them.
RETIRED_KEYS = ("relevance_direction",)


#: Metric suffixes a column may end in, by the metric asked for. Inspect names
#: the column after the metric's registry name, which is namespaced by the
#: package that defines the metric, so the runs made while the scorer briefly
#: registered nan-safe metrics of its own carry ``belcy_eval/nan_mean`` where
#: every other run carries ``mean``. Inspect drops a key's ``nan`` samples
#: before it calls either one, so the two spellings report the same quantity.
METRIC_SUFFIXES = {
    "mean": ("mean", "belcy_eval/nan_mean"),
    "stderr": ("stderr", "belcy_eval/nan_stderr"),
}


def score_columns(key: str, metric: str = "mean") -> tuple[str, ...]:
    """The column names one score key may appear under, in order of preference."""
    names = [key]
    for current, old in LEGACY_KEYS.items():
        if key == current or key.startswith(f"{current}_d"):
            names.append(old + key[len(current) :])
    return tuple(
        column
        for name in names
        for suffix in METRIC_SUFFIXES.get(metric, (metric,))
        for column in (f"score_{name}_{suffix}", f"score_{SCORER}_{name}_{suffix}")
    )


def score_value(row: pd.Series[Any], key: str, metric: str = "mean") -> Any:
    """One score off an eval row, or ``None`` if the run never reported it."""
    for column in score_columns(key, metric):
        if column in row.index:
            return row[column]
    return None


#: The three axes, the tail and the strength of the first, how much of the map
#: was elicited, and how the elicitation itself went. The last three are
#: diagnostics: a run with a non-zero failure or overflow rate, or one where
#: structured outputs were rejected, is not comparable with a clean one.
REPORTED_KEYS = AGGREGATE_KEYS + (
    "elicitation_coverage",
    "failure_rate",
    "overflow_rate",
    "unconstrained",
)

#: The per-distance breakdown's column names, for either spelling.
DISTANCE_COLUMN = re.compile(
    rf"^score_(?:{SCORER}_)?"
    rf"({'|'.join((*RELEVANCE_KEYS, *LEGACY_KEYS.values(), *RETIRED_KEYS))})"
    rf"_d(\d+)_(?:belcy_eval/nan_mean|mean)$"
)

#: Task arguments that change what the scores mean. Scores are only comparable
#: across runs that share them.
CONFIG_ARGS = (
    "show_map",
    "personas",
    "corpora",
    "n_maps",
    "min_claims",
    "max_claims",
    "max_path_length",
    "max_pairs_per_map",
    "mece_roots",
    "seed",
    "debias_order",
    "system_suffix",
)

#: Column naming an eval's terminal state; only ``success`` carries results.
STATUS_COLUMN = "status"


def load_evals(log_dir: str) -> pd.DataFrame:
    """Load every eval under ``log_dir``.

    Directory paths are walked recursively, so pointing this at the parent of
    the per-``show_map`` subdirectories picks up both context modes.

    Only evals that finished are kept. An interrupted run leaves its log behind
    with status ``started``; ``eval_set`` resumes from it and carries its
    completed samples into the retry, so it holds no results of its own and
    would otherwise print as a row of blanks next to the one that succeeded.
    """
    df = evals_df(log_dir)
    if STATUS_COLUMN in df.columns:
        df = df[df[STATUS_COLUMN] == "success"]
    return df


def model_names(df: pd.DataFrame) -> list[str]:
    return sorted(df[MODEL_COLUMN].dropna().unique().tolist())


def model_slice(df: pd.DataFrame, model: str) -> pd.DataFrame:
    return df[df[MODEL_COLUMN] == model]


def eval_config(row: pd.Series[Any]) -> dict[str, Any]:
    """The configuration one eval row was measured under.

    Belcy scores depend on how credences were elicited, so the configuration
    accompanies the metric values wherever they are reported.
    """
    config: dict[str, Any] = {}
    for arg in CONFIG_ARGS:
        value = row.get(f"task_arg_{arg}")
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        if value is not None and not pd.isna(value):
            config[arg] = value

    generate_config = row.get("model_generate_config")
    if generate_config is not None and not pd.isna(generate_config):
        temperature = json.loads(generate_config).get("temperature")
        if temperature is not None:
            config["run_temperature"] = temperature

    samples = row.get("total_samples")
    if samples is not None and not pd.isna(samples):
        config["maps_scored"] = int(samples)

    return config


def format_config(config: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in config.items())


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task: the three axes, the tail of the first, coverage and diagnostics."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        entry: dict[str, Any] = {"task": row.get(TASK_COLUMN)}
        for key in REPORTED_KEYS:
            entry[key.replace("_", " ")] = score_value(row, key)
        rows.append(entry)

    table = pd.DataFrame(rows)
    return table.sort_values("task").reset_index(drop=True) if not table.empty else table


def distance_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task and distance: the relevance scores broken down by distance.

    Distances are read off whichever per-distance columns the logs carry, so
    runs with different ``max_path_length`` report what each measured. Logs
    from before the breakdown existed yield an empty table. A retired score a
    log still carries is kept, in a column of its own after the current ones.
    """
    found: dict[int, dict[str, str]] = {}
    for column in df.columns:
        match = DISTANCE_COLUMN.match(str(column))
        if match:
            key = {old: new for new, old in LEGACY_KEYS.items()}.get(match.group(1), match.group(1))
            found.setdefault(int(match.group(2)), {}).setdefault(key, column)

    keys = RELEVANCE_KEYS + tuple(
        key for key in RETIRED_KEYS if any(key in columns for columns in found.values())
    )
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for distance in sorted(found):
            entry: dict[str, Any] = {"task": row.get(TASK_COLUMN), "distance": distance}
            for key in keys:
                column = found[distance].get(key)
                entry[key.replace("_", " ")] = row[column] if column is not None else None
            rows.append(entry)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["task", "distance"]).reset_index(drop=True)
