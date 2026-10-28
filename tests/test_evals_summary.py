"""Reading score columns off eval rows, including logs written before a rename."""

from __future__ import annotations

import pandas as pd

from belcy_eval.scripts import evals_summary
from belcy_eval.scripts.evals_summary import distance_table, metrics_table, score_value


def test_old_logs_report_their_alignment_as_strength_and_no_head_score():
    df = pd.DataFrame(
        [
            {
                "task_name": "belcy_blind",
                "score_belcy_scorer_relevance_alignment_mean": 0.1,
                "score_belcy_scorer_misdirected_relevance_mean": 0.3,
                "score_belcy_scorer_relevance_alignment_d1_mean": 0.2,
                "score_belcy_scorer_misdirected_relevance_d1_mean": 0.25,
            }
        ]
    )
    row = df.iloc[0]
    assert score_value(row, "relevance_strength") == 0.1
    assert score_value(row, "relevance_strength_d1") == 0.2
    assert score_value(row, "weighted_relevance_direction") is None

    table = metrics_table(df)
    assert table.loc[0, "relevance strength"] == 0.1
    assert table.loc[0, "weighted relevance direction"] is None

    by_distance = distance_table(df)
    assert by_distance.loc[0, "relevance strength"] == 0.2
    assert by_distance.loc[0, "misdirected relevance"] == 0.25
    assert by_distance.loc[0, "weighted relevance direction"] is None


def test_a_retired_score_a_log_carries_is_still_shown_by_distance():
    """Logs written while relevance direction was the head score keep reading as they did."""
    df = pd.DataFrame(
        [
            {
                "task_name": "belcy_blind",
                "score_belcy_scorer_relevance_direction_mean": 0.6,
                "score_belcy_scorer_relevance_direction_d1_mean": 0.7,
                "score_belcy_scorer_relevance_strength_d1_mean": 0.2,
            }
        ]
    )
    by_distance = distance_table(df)
    assert by_distance.loc[0, "distance"] == 1
    assert by_distance.loc[0, "relevance direction"] == 0.7
    assert by_distance.loc[0, "relevance strength"] == 0.2
    assert by_distance.loc[0, "weighted relevance direction"] is None


def test_new_logs_are_read_under_their_own_names():
    row = pd.Series(
        {
            "score_weighted_relevance_direction_mean": 0.6,
            "score_relevance_strength_mean": 0.1,
            "score_relevance_alignment_mean": 0.9,
        }
    )
    assert score_value(row, "weighted_relevance_direction") == 0.6
    assert score_value(row, "relevance_strength") == 0.1


def test_the_interim_nan_metric_spelling_is_still_read():
    """Runs made while the scorer registered its own metrics name the columns after them."""
    df = pd.DataFrame(
        [
            {
                "task_name": "belcy_blind",
                "score_relevance_strength_belcy_eval/nan_mean": 0.4,
                "score_misdirected_relevance_d1_belcy_eval/nan_mean": 0.25,
            }
        ]
    )
    row = df.iloc[0]
    assert score_value(row, "relevance_strength") == 0.4

    assert metrics_table(df).loc[0, "relevance strength"] == 0.4
    assert distance_table(df).loc[0, "misdirected relevance"] == 0.25


def test_an_unfinished_log_is_left_out(monkeypatch):
    """An interrupted attempt carries no results; eval_set resumes from it."""
    frame = pd.DataFrame(
        [
            {"task_name": "belcy_map", "status": "success", "score_relevance_strength_mean": 0.4},
            {"task_name": "belcy_map", "status": "started", "score_relevance_strength_mean": None},
        ]
    )
    monkeypatch.setattr(evals_summary, "evals_df", lambda log_dir: frame)

    kept = evals_summary.load_evals("logs/whatever")
    assert len(kept) == 1
    assert kept.iloc[0]["status"] == "success"


def test_a_frame_without_a_status_column_is_left_alone(monkeypatch):
    """A frame with no status to filter on comes back whole."""
    frame = pd.DataFrame([{"task_name": "belcy_map", "score_relevance_strength_mean": 0.4}])
    monkeypatch.setattr(evals_summary, "evals_df", lambda log_dir: frame)

    assert len(evals_summary.load_evals("logs/whatever")) == 1
