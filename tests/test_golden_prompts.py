"""The exact prompts the model is shown, pinned.

A prompt change changes what every result measures, so it must be deliberate.
Each case renders a few items of one map, under both option orderings, and
compares them with a file under ``tests/golden``. To accept a deliberate
change, rerun with ``BELCY_UPDATE_GOLDEN=1`` and review the diff of those files.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
from helpers import coding_debate, synthetic_debate

from belcy_eval.elicit import Item, conditional_item, marginal_item
from belcy_eval.maps import ArgMap, to_argmap

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE_ENV = "BELCY_UPDATE_GOLDEN"


def societal_map() -> ArgMap:
    """One root, no graph attributes: the shape of the societal corpus."""
    debate = synthetic_debate(branching=2, depth=2)
    return to_argmap(debate, rng=random.Random(0), min_claims=7, max_claims=7)


def coding_map() -> ArgMap:
    """Rival solution approaches as roots, the problem as graph attributes."""
    return to_argmap(coding_debate(), rng=random.Random(0), min_claims=6, max_claims=6)


MAPS = {"societal": societal_map, "coding": coding_map}


def golden_items(argmap: ArgMap, show_map: bool) -> list[Item]:
    """The first and last claim's marginals, both conditionals of the first and last pair."""
    claims = (argmap.claims[0].id, argmap.claims[-1].id)
    items = [marginal_item(argmap, c, show_map=show_map) for c in claims]
    for pair in (argmap.pairs[0], argmap.pairs[-1]):
        for direction in ("pos", "neg"):
            items.append(conditional_item(argmap, pair, direction=direction, show_map=show_map))
    return items


def render_case(argmap: ArgMap, show_map: bool) -> str:
    sections = []
    for item in golden_items(argmap, show_map):
        for swapped in (False, True):
            head = f"### {item.kind} {item.key} {item.direction or '-'} swapped={swapped}"
            sections.append(f"{head}\n{item.prompt(swapped)}\n")
    return "\n".join(sections)


@pytest.mark.parametrize("show_map", [False, True], ids=["blind", "map"])
@pytest.mark.parametrize("name", list(MAPS))
def test_prompts_match_the_golden_files(name, show_map):
    path = GOLDEN_DIR / f"{name}_{'map' if show_map else 'blind'}.txt"
    rendered = render_case(MAPS[name](), show_map)
    if os.environ.get(UPDATE_ENV):
        path.parent.mkdir(exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    assert rendered == path.read_text(encoding="utf-8")


def test_the_golden_coding_map_has_rival_roots():
    argmap = coding_map()
    assert len(argmap.roots) == 2 and argmap.mece_roots
    assert argmap.topic.startswith("which of these rival positions is correct")
