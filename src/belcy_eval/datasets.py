"""Argument maps as Inspect samples.

One sample is one map. The map itself travels in sample metadata; the solver
reads it from there and elicits against it, and the scorer reads it back to
compute the profile. ``Sample.input`` carries only the topic, so the log
remains readable. A persona, when the task assigns one, travels the same way:
drawn once per map and read back by the solver.

Maps may be drawn from a weighted mix of corpus splits. A mix is written as
``corpus_id:split[:weight]`` entries joined by commas, e.g.
``synthetic_corpus-001:test:0.7,synthetic_corpus-002:eval:0.3``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import constants as hf_constants
from inspect_ai.dataset import MemoryDataset, Sample

from belcy_eval import maps, relevance
from belcy_eval.elicit import PROMPT_VERSION, shows_problem
from belcy_eval.maps import HF_REPO_ID, ArgMap, iter_argmaps, split_fingerprint
from belcy_eval.personas import Persona, persona_for, validate_generator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSource:
    """Hub coordinates of a map corpus.

    Kept in a registry rather than inline in the loader so that reporting can
    reference the very same identifiers.
    """

    path: str
    pretty_name: str
    splits: tuple[str, ...]


CORPUS_SOURCES: dict[str, CorpusSource] = {
    "syncialo": CorpusSource(
        path=HF_REPO_ID,
        pretty_name="syncIALO",
        splits=("train", "test", "eval"),
    ),
}

#: The mix used when none is given.
DEFAULT_CORPORA = "synthetic_corpus-001:eval"

#: A directory to cache built maps in. Drawing a split's maps takes minutes
#: (mostly pair selection), and every run of the same maps would pay it again.
DATASET_CACHE_ENV = "BELCY_DATASET_CACHE"

#: Where they go when the environment names no directory. Beside the Hub's own
#: cache, which is where the corpus they are drawn from already lives, so one
#: ``HF_HOME`` moves both.
DEFAULT_DATASET_CACHE = Path(hf_constants.HF_HOME) / "belcy-datasets"

#: Bumped when the layout of a cache file changes.
CACHE_FORMAT = 1


@dataclass(frozen=True)
class MapSource:
    """One corpus split in a mix, with its relative weight."""

    corpus_id: str
    split: str
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f"{self.corpus_id}:{self.split}"


def parse_corpora(spec: str | Sequence[str] | None) -> list[MapSource]:
    """Parse a corpus mix into its sources.

    Accepts the comma-joined string as well as the list Inspect makes of it
    (``-T`` values are split on commas). ``None`` or an empty spec selects
    :data:`DEFAULT_CORPORA`.
    """

    entries = [spec] if isinstance(spec, str) else list(spec or [])
    entries = [part.strip() for entry in entries for part in str(entry).split(",")]
    entries = [entry for entry in entries if entry]
    if not entries:
        return parse_corpora(DEFAULT_CORPORA)

    splits = CORPUS_SOURCES["syncialo"].splits
    sources: list[MapSource] = []
    for entry in entries:
        fields = [f.strip() for f in entry.split(":")]
        if len(fields) not in (2, 3):
            raise ValueError(f"corpus entry {entry!r} is not corpus_id:split[:weight]")
        corpus_id, split = fields[0], fields[1]
        if not corpus_id:
            raise ValueError(f"corpus entry {entry!r} has no corpus_id")
        if split not in splits:
            raise ValueError(f"corpus entry {entry!r}: split must be one of {splits}")
        weight = 1.0
        if len(fields) == 3:
            try:
                weight = float(fields[2])
            except ValueError:
                raise ValueError(f"corpus entry {entry!r}: weight is not a number") from None
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"corpus entry {entry!r}: weight must be positive")
        sources.append(MapSource(corpus_id=corpus_id, split=split, weight=weight))

    keys = [s.key for s in sources]
    if len(set(keys)) != len(keys):
        raise ValueError(f"corpus mix lists a corpus split twice: {keys}")
    return sources


def allocate_maps(n_maps: int, sources: Sequence[MapSource]) -> list[tuple[MapSource, int]]:
    """Split ``n_maps`` across sources in proportion to their weights.

    Largest-remainder apportionment: the counts sum to ``n_maps`` exactly, and
    ties go to the source listed first. Sources left with no maps are dropped.
    """

    total = sum(s.weight for s in sources)
    quotas = [n_maps * s.weight / total for s in sources]
    counts = [math.floor(q) for q in quotas]
    by_remainder = sorted(range(len(sources)), key=lambda i: (counts[i] - quotas[i], i))
    for i in by_remainder[: n_maps - sum(counts)]:
        counts[i] += 1

    allocation: list[tuple[MapSource, int]] = []
    for source, count in zip(sources, counts):
        if count == 0:
            logger.warning("corpus %s gets no maps out of %d; skipping", source.key, n_maps)
            continue
        allocation.append((source, count))
    return allocation


def sample_from_map(argmap: ArgMap, *, personas: str = "none", seed: int = 42) -> Sample:
    """Wrap one argument map as an Inspect sample.

    With a persona generator, the persona is drawn from ``seed`` and the sample
    id, so it does not depend on anything else the task is run with.
    """
    validate_generator(personas)
    source = argmap.source
    sample_id = "::".join(
        [
            "syncialo",
            str(source.get("corpus_id")),
            str(source.get("split")),
            str(source.get("debate")),
            argmap.map_id,
        ]
    )
    persona: dict[str, Any] | None = None
    if personas != "none":
        drawn = persona_for(sample_id, seed)
        persona = {"generator": personas, **drawn.to_dict(), "top_distance": drawn.top_distance}
    metadata: dict[str, Any] = {
        "map": argmap.to_dict(),
        "topic": argmap.topic,
        "n_claims": argmap.n_claims,
        "n_edges": argmap.n_edges,
        "n_roots": len(argmap.roots),
        "mece_roots": argmap.mece_roots,
        "n_pairs": argmap.n_pairs,
        "pairs_by_distance": {str(d): n for d, n in argmap.pairs_by_distance().items()},
        "depth": argmap.depth,
        "corpus_id": source.get("corpus_id"),
        "split": source.get("split"),
        "debate": source.get("debate"),
        "persona": persona,
        # Which prompts the map is asked with, so that runs pair only on maps
        # asked alike (see elicit.PROMPT_VERSION).
        "prompt_version": PROMPT_VERSION,
        "has_problem": shows_problem(argmap),
    }
    return Sample(
        input=f"Topic under discussion: {argmap.topic}",
        id=sample_id,
        metadata=metadata,
    )


def _code_fingerprint() -> str:
    """A hash of the code that draws maps: any change to it invalidates the cache."""
    digest = hashlib.sha256(f"format={CACHE_FORMAT}".encode())
    for module in (maps, relevance):
        digest.update(Path(str(module.__file__)).read_bytes())
    return digest.hexdigest()


def draw_maps(
    source: MapSource,
    count: int,
    *,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    **params: Any,
) -> list[ArgMap]:
    """Draw ``count`` maps from one corpus split, through the cache if there is one.

    ``params`` are :func:`~belcy_eval.maps.iter_argmaps`'s drawing parameters. The
    cache key covers them, the split's files and the drawing code, so a hit
    returns exactly the maps a fresh draw would. Persona and context settings
    are not part of a map, so every run over the same maps shares one entry.
    The corpus ``revision`` is not in the key either: the files it selects are,
    so revisions with the same files share an entry too.
    """

    def draw() -> list[ArgMap]:
        return list(
            iter_argmaps(
                corpus_id=source.corpus_id,
                split=source.split,
                n_maps=count,
                local_dir=local_dir,
                revision=revision,
                **params,
            )
        )

    if cache_dir is None:
        return draw()

    key_data = {
        "corpus_id": source.corpus_id,
        "split": source.split,
        "n_maps": count,
        **params,
        "data": split_fingerprint(
            corpus_id=source.corpus_id,
            split=source.split,
            local_dir=local_dir,
            revision=revision,
        ),
        "code": _code_fingerprint(),
    }
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"{source.corpus_id}__{source.split}__{key}.json.gz"

    if path.is_file():
        try:
            cached = json.loads(gzip.decompress(path.read_bytes()))
            argmaps = [ArgMap.from_dict(d) for d in cached["maps"]]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("unreadable map cache %s (%s); drawing afresh", path, exc)
        else:
            logger.info("loaded %d maps for %s from %s", len(argmaps), source.key, path)
            return argmaps

    argmaps = draw()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": key_data, "maps": [m.to_dict() for m in argmaps]}
    # Written aside and moved into place, so a reader never sees half a file.
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    partial.write_bytes(gzip.compress(json.dumps(payload).encode()))
    os.replace(partial, path)
    logger.info("cached %d maps for %s in %s", len(argmaps), source.key, path)
    return argmaps


def syncialo_dataset(
    *,
    corpora: str | Sequence[str] | None = None,
    n_maps: int = 50,
    min_claims: int = 12,
    max_claims: int = 20,
    max_path_length: int = 3,
    max_pairs: int = 30,
    mece_roots: bool = True,
    seed: int = 42,
    personas: str = "none",
    exclude_ids: Collection[str] | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> MemoryDataset:
    """Draw ``n_maps`` argument maps from a weighted corpus mix, one sample each.

    Samples are shuffled across sources, so a ``--limit`` prefix stays mixed.
    ``personas`` names the generator that assigns each map a persona.
    ``exclude_ids`` drops those samples after the draw, without drawing
    replacements, so every other map stays exactly as it would be without the
    exclusion. ``revision`` pins the syncIALO corpus to a Hub commit, see
    :func:`~belcy_eval.maps.split_files`. ``cache_dir`` (default: the :data:`DATASET_CACHE_ENV`
    variable, else :data:`DEFAULT_DATASET_CACHE`) caches the drawn maps, see :func:`draw_maps`.
    """
    validate_generator(personas)
    if cache_dir is None:
        cache_dir = os.environ.get(DATASET_CACHE_ENV) or DEFAULT_DATASET_CACHE
    samples: list[Sample] = []
    for source, count in allocate_maps(n_maps, parse_corpora(corpora)):
        samples.extend(
            sample_from_map(argmap, personas=personas, seed=seed)
            for argmap in draw_maps(
                source,
                count,
                cache_dir=cache_dir,
                local_dir=local_dir,
                revision=revision,
                min_claims=min_claims,
                max_claims=max_claims,
                max_path_length=max_path_length,
                max_pairs=max_pairs,
                mece_roots=mece_roots,
                seed=seed,
            )
        )
    random.Random(f"{seed}:mix").shuffle(samples)
    if exclude_ids:
        excluded = set(exclude_ids)
        kept = [sample for sample in samples if sample.id not in excluded]
        unmatched = len(excluded) - (len(samples) - len(kept))
        logger.info("excluded %d of %d maps", len(samples) - len(kept), len(samples))
        if unmatched:
            logger.warning("%d excluded ids match no drawn map", unmatched)
        samples = kept
    return MemoryDataset(samples=samples, name="syncialo")


def read_id_list(path: str | Path) -> list[str]:
    """Read sample ids from a file, one per line; blank lines and ``#`` comments are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def map_from_metadata(metadata: dict[str, Any]) -> ArgMap:
    """Recover the argument map a solver or scorer was handed."""
    return ArgMap.from_dict(metadata["map"])


def persona_from_metadata(metadata: dict[str, Any]) -> Persona | None:
    """Recover the persona a map is elicited under, if it has one."""
    persona = metadata.get("persona")
    return Persona.from_dict(persona) if persona else None
