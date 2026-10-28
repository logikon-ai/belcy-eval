from __future__ import annotations

import pytest
from helpers import make_map, synthetic_debate

from belcy_eval.datasets import DATASET_CACHE_ENV
from belcy_eval.maps import ArgMap, RawDebate


@pytest.fixture(autouse=True)
def _isolate_map_cache(monkeypatch, tmp_path):
    """Keep drawn maps out of the real cache, which drawing falls back to."""
    monkeypatch.setenv(DATASET_CACHE_ENV, str(tmp_path / "map-cache"))


@pytest.fixture
def simple_map() -> ArgMap:
    """One conclusion, one supporting and one attacking reason."""
    return make_map([("x1", "y", 1), ("x2", "y", -1)])


@pytest.fixture
def debate() -> RawDebate:
    return synthetic_debate()
