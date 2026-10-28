"""Argument maps: the fixed instrument the eval measures a belief state against.

Maps are drawn from the syncIALO corpus (``DebateLabKIT/syncialo-raw``), whose
debates are directed acyclic graphs of up to several hundred nodes stored as
NetworkX node-link JSON. Most have a single root; some have several, which are
read as rival positions -- mutually exclusive and jointly exhaustive -- unless
:attr:`ArgMap.mece_roots` is off, in which case they are independent claims.
Whole debates are far too large to elicit against, so each eval sample is a
random subgraph grown from the roots.

Each map carries the claim pairs it is elicited and scored on, with the
relevance the map asserts between them (see :mod:`belcy_eval.relevance`). They
are selected when the map is drawn, so they travel with it into the log.

The loading and subtree-sampling logic is ported from
``reason_scales_env.preprocessing.syncialo`` (that package is not published, so
it cannot be imported). Two defects of the original are fixed here:

* the target subtree size is drawn from a per-map RNG, so maps in one run
  actually differ in size rather than all landing on the same draw;
* the root claim's text is carried through under the same key as every other
  claim, so it cannot silently arrive empty downstream.

The premise-level structure of syncIALO arguments is deliberately discarded.
Each node contributes exactly one claim (its ``claim`` text), and each edge is
read as a claim-to-claim support or attack, which is the reading the metrics
assume. An edge's ``target_idx`` -- which premise of the parent argument the
reason bears on -- is preserved in :attr:`Edge.target_idx` so the premise-level
variant stays open, but nothing in the eval reads it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import networkx as nx

from belcy_eval.relevance import Pair, select_pairs

logger = logging.getLogger(__name__)

HF_REPO_ID = "DebateLabKIT/syncialo-raw"

#: Environment variable pointing at a local syncIALO snapshot directory. When
#: set, the corpus is read from disk instead of downloaded from the Hub.
SYNCIALO_DIR_ENV = "BELCY_SYNCIALO_DIR"

#: syncIALO edge valences, mapped to the polarity sigma of the spec.
PRO = 1
CON = -1
_VALENCE_MAP = {"pro": PRO, "con": CON}

#: Debates with fewer nodes than this are unusable as maps.
_MIN_DEBATE_NODES = 3


@dataclass(frozen=True)
class Claim:
    """A single claim: one node of an argument map."""

    id: str
    label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(id=data["id"], label=data["label"], text=data["text"])


@dataclass(frozen=True)
class Edge:
    """A reason relation ``e = (x, y, sigma)``.

    ``source`` is the reason ``x``, ``target`` the conclusion ``y``, and
    ``polarity`` is ``+1`` for support and ``-1`` for attack.
    """

    source: str
    target: str
    polarity: int
    target_idx: int | None = None

    @property
    def supports(self) -> bool:
        return self.polarity == PRO

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "polarity": self.polarity,
            "target_idx": self.target_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(
            source=data["source"],
            target=data["target"],
            polarity=int(data["polarity"]),
            target_idx=data.get("target_idx"),
        )


@dataclass(frozen=True)
class ArgMap:
    """An argument map: claims, the reason relations between them, and the pairs to test.

    ``mece_roots`` reads several roots as rival positions, exactly one of which
    is true. It shapes the map's measure, the pairs it is tested on, and the
    prompts, so it travels with the map rather than beside it.

    ``problem`` and ``design_decision`` say what a coding debate is about (see
    :data:`PROBLEM_ATTRIBUTES`), and are empty for a debate that states neither.
    The prompts show them; the map id ignores them.
    """

    map_id: str
    topic: str
    claims: tuple[Claim, ...]
    edges: tuple[Edge, ...]
    pairs: tuple[Pair, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)
    mece_roots: bool = True
    #: The programming problem a coding debate is about, verbatim.
    problem: str = ""
    #: The design decision a coding debate is about, in one sentence.
    design_decision: str = ""

    def __post_init__(self) -> None:
        ids = {c.id for c in self.claims}
        if len(ids) != len(self.claims):
            raise ValueError("claim ids are not unique")
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                raise ValueError(f"edge {e.source}->{e.target} leaves the map")
        if self.claims and not self.roots:
            raise ValueError("the map has no root (every claim has an outgoing edge)")
        for p in self.pairs:
            if p.source not in ids or p.target not in ids:
                raise ValueError(f"pair {p.source}->{p.target} leaves the map")

    # -- lookups -------------------------------------------------------

    @property
    def claim_by_id(self) -> dict[str, Claim]:
        return {c.id: c for c in self.claims}

    def text(self, claim_id: str) -> str:
        return self.claim_by_id[claim_id].text

    def incoming(self, claim_id: str) -> list[Edge]:
        """Edges pointing at ``claim_id``, i.e. the reasons bearing on it."""
        return [e for e in self.edges if e.target == claim_id]

    def children(self, claim_id: str) -> list[Edge]:
        """Alias of :meth:`incoming`, for traversal from the roots."""
        return self.incoming(claim_id)

    @property
    def roots(self) -> tuple[str, ...]:
        """Claims with no outgoing edge, in id order."""
        sources = {e.source for e in self.edges}
        return tuple(sorted(c.id for c in self.claims if c.id not in sources))

    @property
    def graph(self) -> nx.DiGraph:
        """The map as a directed graph, edges running from reason to conclusion."""
        graph = nx.DiGraph()
        graph.add_nodes_from(c.id for c in self.claims)
        graph.add_edges_from((e.source, e.target) for e in self.edges)
        return graph

    # -- shape ---------------------------------------------------------

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def depth(self) -> int:
        """Longest directed path, counted in edges."""
        return nx.dag_longest_path_length(self.graph) if self.claims else 0

    def pairs_by_distance(self) -> dict[int, int]:
        """Number of pairs at each distance."""
        counts: dict[int, int] = {}
        for p in self.pairs:
            counts[p.distance] = counts.get(p.distance, 0) + 1
        return dict(sorted(counts.items()))

    # -- serialization -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "map_id": self.map_id,
            "topic": self.topic,
            "claims": [c.to_dict() for c in self.claims],
            "edges": [e.to_dict() for e in self.edges],
            "pairs": [p.to_dict() for p in self.pairs],
            "source": self.source,
            "mece_roots": self.mece_roots,
            "problem": self.problem,
            "design_decision": self.design_decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArgMap:
        return cls(
            map_id=data["map_id"],
            topic=data["topic"],
            claims=tuple(Claim.from_dict(c) for c in data["claims"]),
            edges=tuple(Edge.from_dict(e) for e in data["edges"]),
            pairs=tuple(Pair.from_dict(p) for p in data.get("pairs", ())),
            source=data.get("source", {}),
            # Maps logged before the switch existed were all read this way.
            mece_roots=bool(data.get("mece_roots", True)),
            # Maps logged before these were read carry neither.
            problem=data.get("problem", ""),
            design_decision=data.get("design_decision", ""),
        )


@dataclass
class RawDebate:
    """A full syncIALO debate, before subgraph sampling."""

    corpus_id: str
    split: str
    path: Path
    idx: int
    graph: nx.DiGraph

    @property
    def stem(self) -> str:
        """Debate identifier, e.g. ``debate-eval-0042``."""
        name = self.path.stem
        return name.removeprefix("node_link_data-")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def syncialo_root(local_dir: str | Path | None = None) -> Path | None:
    """Resolve a local syncIALO snapshot directory, if one is configured."""

    candidate = local_dir if local_dir is not None else os.environ.get(SYNCIALO_DIR_ENV)
    if candidate is None:
        return None
    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"syncIALO directory does not exist: {path}")
    return path


def _load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text())
    edge_key = "links" if "links" in data else "edges"
    return nx.node_link_graph(data, edges=edge_key)


def split_files(
    *,
    corpus_id: str,
    split: str,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> tuple[Path, list[Path]]:
    """The debate files of one corpus split, sorted, and the directory they are under.

    Reads from a local snapshot when :data:`SYNCIALO_DIR_ENV` is set or
    ``local_dir`` is passed, otherwise downloads the split from the Hub, at
    ``revision`` (a commit sha, branch or tag; unset means the latest). A local
    snapshot is read as it is, whatever ``revision`` says.
    """

    pattern = f"data/{corpus_id}/{split}/**/*.json"
    root = syncialo_root(local_dir)

    if root is None:
        from huggingface_hub import HfApi

        root = Path(
            HfApi().snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                revision=revision,
                allow_patterns=pattern,
            )
        )

    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no syncIALO debates for corpus_id={corpus_id!r}, split={split!r} under {root}"
        )
    return root, files


def split_fingerprint(
    *,
    corpus_id: str,
    split: str,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> str:
    """A hash that changes whenever the files of a corpus split do.

    Hub snapshots link each file to a blob named by its content hash, so a
    changed debate changes the fingerprint. A local snapshot has no such name;
    there, file names and sizes stand in for the content.
    """

    root, files = split_files(
        corpus_id=corpus_id, split=split, local_dir=local_dir, revision=revision
    )
    digest = hashlib.sha256()
    for path in files:
        entry = f"{path.relative_to(root)}\0{path.resolve().name}\0{path.stat().st_size}\n"
        digest.update(entry.encode())
    return digest.hexdigest()


def load_debates(
    *,
    corpus_id: str,
    split: str,
    local_dir: str | Path | None = None,
    revision: str | None = None,
    limit: int | None = None,
) -> list[RawDebate]:
    """Load raw syncIALO debates for one corpus split (see :func:`split_files`)."""

    _, files = split_files(corpus_id=corpus_id, split=split, local_dir=local_dir, revision=revision)
    debates: list[RawDebate] = []
    for idx, path in enumerate(files):
        if limit is not None and len(debates) >= limit:
            break
        try:
            graph = _load_graph(path)
        except Exception as exc:  # noqa: BLE001 - a bad file must not abort the run
            logger.warning("could not load %s: %s", path.name, exc)
            continue
        if graph.number_of_nodes() < _MIN_DEBATE_NODES:
            logger.warning("skipping %s: only %d node(s)", path.name, graph.number_of_nodes())
            continue
        debates.append(RawDebate(corpus_id=corpus_id, split=split, path=path, idx=idx, graph=graph))

    return debates


# ---------------------------------------------------------------------------
# subgraph sampling
# ---------------------------------------------------------------------------


def grow_subtree(
    graph: nx.DiGraph,
    roots: Sequence[str],
    target_n: int,
    rng: random.Random,
) -> set[str]:
    """Grow a connected subgraph of ``target_n`` nodes outwards from the roots.

    syncIALO edges point from reason to conclusion, so growth proceeds along
    predecessors: from a claim to the reasons bearing on it.
    """

    subtree: set[str] = set(roots)
    frontier: set[str] = set()
    for r in roots:
        frontier.update(p for p in graph.predecessors(r) if p not in subtree)

    while len(subtree) < target_n and frontier:
        chosen = rng.choice(sorted(frontier))
        subtree.add(chosen)
        frontier.discard(chosen)
        frontier.update(p for p in graph.predecessors(chosen) if p not in subtree)

    return subtree


def reachable_claims(graph: nx.DiGraph) -> int:
    """Count the claims :func:`grow_subtree` can reach: the roots and their ancestors."""

    reachable: set[str] = set()
    for root in (n for n in graph.nodes if graph.out_degree(n) == 0):
        reachable.add(root)
        reachable.update(nx.ancestors(graph, root))
    return len(reachable)


def normalize_label(label: str) -> str:
    """Normalize a syncIALO label into a short identifier-like string."""

    if not label:
        return ""
    normalized = unicodedata.normalize("NFKC", label)
    kept = "".join(ch if (ch.isalnum() or ch in " -") else " " for ch in normalized)
    collapsed = " ".join(kept.split()).strip()
    if not collapsed:
        return ""
    return re.sub(r"_+", "_", collapsed.replace(" ", "_")).strip("_")


#: Graph attributes a debate may state its topic under, in order of preference.
TOPIC_ATTRIBUTES = ("topic", "question")

#: Graph attributes a coding debate states its programming problem and its
#: design decision under, in that order. Societal debates have neither.
PROBLEM_ATTRIBUTES = ("problem", "design_decision")


def map_topic(graph: nx.DiGraph, root_texts: Sequence[str], *, mece_roots: bool = True) -> str:
    """The topic line of a map's prompts.

    The debate's own topic if it states one. Otherwise the root claim, when
    there is only one; with several roots, naming any one of them would present
    one position as the topic, so the roots are all listed -- as alternatives
    if ``mece_roots``, else as independent positions.
    """
    for attr in TOPIC_ATTRIBUTES:
        value = str(graph.graph.get(attr) or "").strip()
        if value:
            return value
    if len(root_texts) == 1:
        return root_texts[0]
    listed = "; ".join(f"({i}) {text}" for i, text in enumerate(root_texts, start=1))
    if mece_roots:
        return f"which of these rival positions is correct: {listed}"
    return f"whether each of these positions is correct: {listed}"


def to_argmap(
    debate: RawDebate,
    *,
    rng: random.Random,
    min_claims: int,
    max_claims: int,
    max_path_length: int = 3,
    max_pairs: int = 30,
    mece_roots: bool = True,
) -> ArgMap:
    """Draw one argument map from a debate, with the pairs it is tested on.

    The target size is drawn from ``rng``, so distinct debates in one run get
    distinct sizes. ``mece_roots`` is set on the map (see :class:`ArgMap`).
    Raises :class:`ValueError` if the map asserts no relevance between any pair
    within ``max_path_length``.
    """

    if min_claims < 2:
        raise ValueError("min_claims must be at least 2")
    if max_claims < min_claims:
        raise ValueError("max_claims must not be below min_claims")

    graph = debate.graph
    roots = sorted(n for n in graph.nodes if graph.out_degree(n) == 0)
    if not roots:
        raise ValueError(f"{debate.stem}: no root node (every node has an outgoing edge)")

    target_n = rng.randint(min_claims, max_claims)
    subtree = grow_subtree(graph, roots, target_n, rng)
    if len(subtree) < min_claims:
        raise ValueError(
            f"{debate.stem}: only {len(subtree)} claims reachable, below min_claims={min_claims}"
        )

    claims: list[Claim] = []
    for node_id in sorted(subtree):
        attrs = graph.nodes[node_id]
        text = (attrs.get("claim") or "").strip()
        if not text:
            raise ValueError(f"{debate.stem}: node {node_id} has no claim text")
        claims.append(Claim(id=node_id, label=normalize_label(attrs.get("label", "")), text=text))

    edges: list[Edge] = []
    for u, v, attrs in graph.edges(data=True):
        if u not in subtree or v not in subtree:
            continue
        polarity = _VALENCE_MAP.get(str(attrs.get("valence", "")).lower())
        if polarity is None:
            logger.warning(
                "%s: unknown valence %r on edge %s->%s; skipping",
                debate.stem,
                attrs.get("valence"),
                u,
                v,
            )
            continue
        edges.append(
            Edge(source=u, target=v, polarity=polarity, target_idx=attrs.get("target_idx"))
        )
    edges.sort(key=lambda e: (e.target, e.source))

    topic = map_topic(
        graph, [graph.nodes[r]["claim"].strip() for r in roots], mece_roots=mece_roots
    )

    payload = json.dumps(
        {
            "claims": [c.to_dict() for c in claims],
            "edges": [e.to_dict() for e in edges],
        },
        sort_keys=True,
    )
    map_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    problem, design_decision = (
        str(graph.graph.get(attr) or "").strip() for attr in PROBLEM_ATTRIBUTES
    )

    argmap = ArgMap(
        map_id=map_id,
        topic=topic,
        claims=tuple(claims),
        edges=tuple(edges),
        source={
            "corpus_id": debate.corpus_id,
            "split": debate.split,
            "debate": debate.stem,
        },
        mece_roots=mece_roots,
        problem=problem,
        design_decision=design_decision,
    )
    pairs = select_pairs(argmap, max_path_length=max_path_length, max_pairs=max_pairs)
    if not pairs:
        raise ValueError(f"{debate.stem}: no relevant pair within distance {max_path_length}")
    return replace(argmap, pairs=pairs)


def iter_argmaps(
    *,
    corpus_id: str,
    split: str,
    n_maps: int,
    min_claims: int = 12,
    max_claims: int = 20,
    max_path_length: int = 3,
    max_pairs: int = 30,
    mece_roots: bool = True,
    seed: int = 42,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Iterator[ArgMap]:
    """Yield exactly ``n_maps`` argument maps, one per debate, cycling if necessary.

    Debates with fewer than ``min_claims`` reachable claims are dropped up
    front. A map that cannot be drawn is replaced by one from the next debate;
    if a full cycle over the debates yields nothing, the split is unusable and
    this raises :class:`ValueError`.

    Each map gets its own RNG seeded from ``seed``, its corpus split and its
    attempt position, so the whole dataset is reproducible, maps drawn from the
    same debate on a second pass differ from the first, and same-named debates
    in different corpora are drawn independently.
    """

    debates = load_debates(corpus_id=corpus_id, split=split, local_dir=local_dir, revision=revision)
    if not debates:
        raise ValueError(f"no debates loaded for {corpus_id}:{split}")

    usable: list[RawDebate] = []
    for debate in debates:
        n_reachable = reachable_claims(debate.graph)
        if n_reachable < min_claims:
            logger.warning(
                "skipping %s: only %d claims reachable, below min_claims=%d",
                debate.stem,
                n_reachable,
                min_claims,
            )
            continue
        usable.append(debate)
    if not usable:
        raise ValueError(f"no debates in {corpus_id}:{split} with at least {min_claims} claims")

    produced = failures = attempt = 0
    while produced < n_maps:
        debate = usable[attempt % len(usable)]
        rng = random.Random(f"{seed}:{corpus_id}:{split}:{attempt}:{debate.stem}")
        attempt += 1
        try:
            argmap = to_argmap(
                debate,
                rng=rng,
                min_claims=min_claims,
                max_claims=max_claims,
                max_path_length=max_path_length,
                max_pairs=max_pairs,
                mece_roots=mece_roots,
            )
        except ValueError as exc:
            logger.warning("skipping %s: %s", debate.stem, exc)
            failures += 1
            if failures >= len(usable):
                raise ValueError(
                    f"{corpus_id}:{split}: no map could be drawn in {failures} consecutive "
                    f"attempts ({produced} of {n_maps} drawn)"
                ) from exc
            continue
        failures = 0
        produced += 1
        yield argmap
