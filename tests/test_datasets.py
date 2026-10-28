from __future__ import annotations

import json
import random
from collections import Counter

import networkx as nx
import pytest
from helpers import coding_debate, synthetic_debate

from belcy_eval import datasets
from belcy_eval.datasets import (
    DATASET_CACHE_ENV,
    DEFAULT_CORPORA,
    MapSource,
    allocate_maps,
    map_from_metadata,
    parse_corpora,
    persona_from_metadata,
    read_id_list,
    sample_from_map,
    syncialo_dataset,
)
from belcy_eval.elicit import PROMPT_VERSION
from belcy_eval.maps import to_argmap


def test_parse_string_and_inspect_list_agree():
    spec = "corpus-a:test:0.7,corpus-b:eval:0.3"
    expected = [MapSource("corpus-a", "test", 0.7), MapSource("corpus-b", "eval", 0.3)]
    assert parse_corpora(spec) == expected
    # `-T corpora=...` arrives split on commas
    assert parse_corpora(["corpus-a:test:0.7", "corpus-b:eval:0.3"]) == expected


@pytest.mark.parametrize("spec", [None, "", []])
def test_parse_empty_spec_selects_the_default(spec):
    assert parse_corpora(spec) == parse_corpora(DEFAULT_CORPORA)


def test_parse_weight_defaults_to_one_and_whitespace_is_stripped():
    assert parse_corpora(" corpus-a : eval , corpus-a:test: 2 ") == [
        MapSource("corpus-a", "eval", 1.0),
        MapSource("corpus-a", "test", 2.0),
    ]


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("corpus-a", "corpus_id:split"),
        ("corpus-a:eval:1:2", "corpus_id:split"),
        (":eval", "no corpus_id"),
        ("corpus-a:dev", "split must be"),
        ("corpus-a:eval:heavy", "not a number"),
        ("corpus-a:eval:0", "positive"),
        ("corpus-a:eval:-1", "positive"),
        ("corpus-a:eval,corpus-a:eval:2", "twice"),
    ],
)
def test_parse_rejects_malformed_specs(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_corpora(spec)


@pytest.mark.parametrize(
    ("n_maps", "spec", "expected"),
    [
        (50, "a:eval:0.7,b:eval:0.3", [35, 15]),
        (10, "a:eval,b:eval,c:eval", [4, 3, 3]),
        (7, "a:eval:1,b:eval:1", [4, 3]),
        (1, "a:eval:0.4,b:eval:0.6", [0, 1]),
    ],
)
def test_allocation_is_exact_and_proportional(n_maps, spec, expected):
    sources = parse_corpora(spec)
    counts = {s.key: n for s, n in allocate_maps(n_maps, sources)}
    assert [counts.get(s.key, 0) for s in sources] == expected
    assert sum(counts.values()) == n_maps


def test_allocation_drops_sources_without_maps():
    allocation = allocate_maps(1, parse_corpora("a:eval:0.4,b:eval:0.6"))
    assert [s.key for s, _ in allocation] == ["b:eval"]


def write_corpus(root, corpus_id, split, n_debates):
    folder = root / "data" / corpus_id / split
    folder.mkdir(parents=True)
    for i in range(n_debates):
        stem = f"debate-{split}-{i:04d}"  # identical stems across corpora
        graph = synthetic_debate(seed=sum(map(ord, corpus_id)) + i, stem=stem).graph
        data = nx.node_link_data(graph, edges="links")
        (folder / f"node_link_data-{stem}.json").write_text(json.dumps(data))


def write_debate(root, corpus_id, split, stem, *, root_claim=None, **shape):
    folder = root / "data" / corpus_id / split
    folder.mkdir(parents=True, exist_ok=True)
    graph = synthetic_debate(stem=stem, **shape).graph
    if root_claim is not None:
        graph.nodes["n0"]["claim"] = root_claim
    data = nx.node_link_data(graph, edges="links")
    (folder / f"node_link_data-{stem}.json").write_text(json.dumps(data))


def draw(tmp_path, n_maps=6, **kwargs):
    dataset = syncialo_dataset(
        corpora="corpus-a:eval",
        n_maps=n_maps,
        min_claims=4,
        max_claims=8,
        local_dir=tmp_path,
        **kwargs,
    )
    return list(dataset)


def test_skipped_maps_are_replaced(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 3)
    write_debate(tmp_path, "corpus-a", "eval", "debate-broken", root_claim="")

    samples = draw(tmp_path)

    assert len(samples) == 6
    debates = {map_from_metadata(s.metadata).source["debate"] for s in samples}
    assert "debate-broken" not in debates
    assert len({s.id for s in samples}) == len(samples)


def test_split_of_broken_debates_raises(tmp_path):
    for i in range(2):
        write_debate(tmp_path, "corpus-a", "eval", f"debate-broken-{i}", root_claim="")

    with pytest.raises(ValueError, match="consecutive"):
        draw(tmp_path)


def test_small_debates_are_skipped(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 3)
    write_debate(tmp_path, "corpus-a", "eval", "debate-small", branching=2, depth=1)

    samples = draw(tmp_path)

    assert len(samples) == 6
    maps = [map_from_metadata(s.metadata) for s in samples]
    assert "debate-small" not in {m.source["debate"] for m in maps}
    assert all(m.n_claims >= 4 for m in maps)


def test_split_of_small_debates_raises(tmp_path):
    write_debate(tmp_path, "corpus-a", "eval", "debate-small", branching=2, depth=1)

    with pytest.raises(ValueError, match="at least"):
        draw(tmp_path)


def test_dataset_mixes_corpora_by_weight(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 3)
    write_corpus(tmp_path, "corpus-b", "eval", 3)

    dataset = syncialo_dataset(
        corpora="corpus-a:eval:3,corpus-b:eval:1",
        n_maps=8,
        min_claims=4,
        max_claims=8,
        local_dir=tmp_path,
    )
    samples = list(dataset)
    corpora = [s.metadata["corpus_id"] for s in samples]

    assert Counter(corpora) == {"corpus-a": 6, "corpus-b": 2}
    assert len({s.id for s in samples}) == len(samples)
    # shuffled, so a --limit prefix is not drawn from one corpus only
    assert corpora != sorted(corpora) and corpora != sorted(corpora, reverse=True)


def test_samples_carry_their_pairs(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 2)

    for sample in draw(tmp_path, n_maps=2):
        argmap = map_from_metadata(sample.metadata)
        assert argmap.pairs
        assert sample.metadata["n_pairs"] == argmap.n_pairs
        assert sample.metadata["n_roots"] == 1
        assert sum(sample.metadata["pairs_by_distance"].values()) == argmap.n_pairs
        assert sample.metadata["prompt_version"] == PROMPT_VERSION
        assert sample.metadata["has_problem"] is False


def test_a_coding_sample_records_that_its_prompts_show_the_problem():
    argmap = to_argmap(coding_debate(), rng=random.Random(0), min_claims=6, max_claims=6)
    sample = sample_from_map(argmap)
    assert sample.metadata["has_problem"] is True
    assert sample.metadata["prompt_version"] == PROMPT_VERSION
    assert map_from_metadata(sample.metadata).problem == argmap.problem


def test_dataset_is_reproducible(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 2)
    write_corpus(tmp_path, "corpus-a", "test", 2)

    def ids():
        dataset = syncialo_dataset(
            corpora="corpus-a:eval,corpus-a:test",
            n_maps=6,
            min_claims=4,
            max_claims=8,
            max_pairs=5,
            local_dir=tmp_path,
        )
        return [(s.id, s.metadata["map"]["pairs"]) for s in dataset]

    assert ids() == ids()


def test_excluded_maps_are_dropped_without_replacement(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 4)

    def draw(exclude_ids=None):
        dataset = syncialo_dataset(
            corpora="corpus-a:eval",
            n_maps=4,
            min_claims=4,
            max_claims=8,
            max_pairs=5,
            exclude_ids=exclude_ids,
            local_dir=tmp_path,
        )
        return [(s.id, s.metadata["map"]["pairs"]) for s in dataset]

    full = draw()
    dropped = {full[1][0], "syncialo::no::such::map"}
    assert draw(dropped) == [entry for entry in full if entry[0] not in dropped]


def test_id_lists_skip_comments_and_blank_lines(tmp_path):
    path = tmp_path / "exclude.txt"
    path.write_text("# header\n\nsyncialo::a::eval::d::m1\n  syncialo::a::eval::d::m2  \n")
    assert read_id_list(path) == ["syncialo::a::eval::d::m1", "syncialo::a::eval::d::m2"]


def cached_draw(corpus_dir, cache_dir, **overrides):
    params = {
        "corpora": "corpus-a:eval",
        "n_maps": 4,
        "min_claims": 4,
        "max_claims": 8,
        "max_pairs": 5,
        "personas": "schwartz",
        "local_dir": corpus_dir,
        "cache_dir": cache_dir,
    }
    return [(s.id, s.metadata) for s in syncialo_dataset(**{**params, **overrides})]


@pytest.fixture
def count_draws(monkeypatch):
    """Count the calls that actually draw maps, i.e. cache misses."""
    calls = []
    original = datasets.iter_argmaps

    def counting(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(datasets, "iter_argmaps", counting)
    return calls


def test_a_cache_hit_returns_the_same_samples_without_drawing(tmp_path, count_draws):
    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    cache = tmp_path / "cache"

    # a cache dir of its own, so this leg is a guaranteed miss: a fresh draw
    fresh = cached_draw(tmp_path / "corpus", tmp_path / "elsewhere")
    first = cached_draw(tmp_path / "corpus", cache)
    second = cached_draw(tmp_path / "corpus", cache)

    assert first == second == fresh
    assert len(count_draws) == 2  # the fresh draw and the first into `cache`
    assert len(list(cache.glob("*.json.gz"))) == 1


def test_the_cache_is_shared_by_persona_and_keyed_by_map_parameters(tmp_path, count_draws):
    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    cache = tmp_path / "cache"

    cached_draw(tmp_path / "corpus", cache)
    cached_draw(tmp_path / "corpus", cache, personas="none")
    assert len(count_draws) == 1

    cached_draw(tmp_path / "corpus", cache, max_pairs=3)
    cached_draw(tmp_path / "corpus", cache, seed=7)
    assert len(count_draws) == 3
    assert len(list(cache.glob("*.json.gz"))) == 3


def test_changed_corpus_files_miss_the_cache(tmp_path, count_draws):
    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    cache = tmp_path / "cache"
    cached_draw(tmp_path / "corpus", cache)

    debate = min((tmp_path / "corpus").rglob("*.json"))
    debate.write_text(debate.read_text() + " ")
    cached_draw(tmp_path / "corpus", cache)

    assert len(count_draws) == 2


def test_an_unreadable_cache_file_is_redrawn(tmp_path, count_draws):
    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    cache = tmp_path / "cache"
    expected = cached_draw(tmp_path / "corpus", cache)
    (entry,) = cache.glob("*.json.gz")
    entry.write_bytes(b"not gzip")

    assert cached_draw(tmp_path / "corpus", cache) == expected
    assert len(count_draws) == 2


def test_the_cache_directory_defaults_to_the_environment(tmp_path, monkeypatch, count_draws):
    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    monkeypatch.setenv(DATASET_CACHE_ENV, str(tmp_path / "cache"))

    cached_draw(tmp_path / "corpus", None)
    cached_draw(tmp_path / "corpus", None)

    assert len(count_draws) == 1
    assert len(list((tmp_path / "cache").glob("*.json.gz"))) == 1


def test_no_persona_by_default(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 2)

    for sample in draw(tmp_path, n_maps=2):
        assert sample.metadata["persona"] is None
        assert persona_from_metadata(sample.metadata) is None


def test_personas_are_drawn_per_map_and_reproducibly(tmp_path):
    write_corpus(tmp_path, "corpus-a", "eval", 4)

    first = draw(tmp_path, n_maps=4, personas="schwartz")
    again = draw(tmp_path, n_maps=4, personas="schwartz")
    assert [s.metadata["persona"] for s in first] == [s.metadata["persona"] for s in again]

    for sample in first:
        persona = persona_from_metadata(sample.metadata)
        assert persona is not None
        assert sample.metadata["persona"]["generator"] == "schwartz"
        assert sample.metadata["persona"]["top_distance"] == persona.top_distance

    # the persona is the only thing the generator changes
    blind = draw(tmp_path, n_maps=4)
    assert [s.id for s in blind] == [s.id for s in first]


def test_unknown_persona_generator_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="personas must be one of"):
        draw(tmp_path, personas="socratic")


def test_the_corpus_revision_reaches_the_hub_download(tmp_path, monkeypatch):
    """``corpus_revision`` pins every Hub download: the draw and the cache key's."""
    from huggingface_hub import HfApi

    from belcy_eval.belcy import belcy
    from belcy_eval.maps import SYNCIALO_DIR_ENV

    write_corpus(tmp_path / "corpus", "corpus-a", "eval", 4)
    downloads = []

    def snapshot_download(self, **kwargs):
        downloads.append(kwargs)
        return str(tmp_path / "corpus")

    monkeypatch.delenv(SYNCIALO_DIR_ENV, raising=False)
    monkeypatch.setattr(HfApi, "snapshot_download", snapshot_download)
    monkeypatch.setenv(DATASET_CACHE_ENV, str(tmp_path / "cache"))

    task = belcy(corpora="corpus-a:eval", n_maps=2, min_claims=4, corpus_revision="abc123")

    assert len(task.dataset) == 2
    assert len(downloads) == 2  # the cache key's fingerprint, then the draw
    assert {d["revision"] for d in downloads} == {"abc123"}
