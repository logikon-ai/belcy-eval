"""Personas an elicitation can be run under.

A persona is a value profile put in front of the elicitation instruction, so a
model's belief state can be read with and without one. It is assembled from the
tables in ``personas.yaml`` rather than written, which keeps the vocabulary a
persona can bring into a prompt fixed and small:

* two **endorsed** wants, the first weighing more heavily, and two **rejected**
  ones, all four from Schwartz's circle of basic values;
* one level on each **disposition** scale (time and risk).

Each half, endorsed or rejected, is *coherent*: its two wants sit at most
:data:`COHERENCE` places apart on the circle, so it is an outlook somebody
could hold. The two halves are disjoint, and their leading wants sit between
two and five places apart, from near neighbours to full opposition.

One persona is drawn per map, from the task seed and the sample id, so a map
gets the same persona whatever the elicitation context, and every item of the
map is asked under it: coherence compares a map's marginals with its
conditionals, which is only meaningful if they come from one believer.
"""

from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

#: The persona generators a task can be run with. ``"none"`` elicits under the
#: bare instruction.
GENERATORS = ("none", "schwartz")

#: How many places apart the two wants of one half may sit on the circle. At 2
#: a half is a want and one of its four nearest neighbours; a facing pair is
#: never a half.
COHERENCE = 2

#: How many places apart the two halves' leading wants may sit.
TOP_DISTANCES = (2, 3, 4, 5)


def validate_generator(name: str) -> str:
    if name not in GENERATORS:
        raise ValueError(f"personas must be one of {GENERATORS}, got {name!r}")
    return name


@functools.cache
def load_pool() -> dict[str, Any]:
    """The wants, the circle and the disposition scales."""
    pool = yaml.safe_load(files("belcy_eval").joinpath("personas.yaml").read_text("utf-8"))
    if sorted(pool["circle"]) != sorted(pool["wants"]):
        raise ValueError("personas.yaml: the circle must list every want exactly once")
    if len(pool["circle"]) < 2 * max(TOP_DISTANCES):
        raise ValueError("personas.yaml: the circle is too small for the top distances")
    return pool


def distance(circle: tuple[str, ...], a: str, b: str) -> int:
    """How many places apart two wants sit, going the short way round."""
    gap = abs(circle.index(a) - circle.index(b))
    return min(gap, len(circle) - gap)


Half = tuple[str, str]


@functools.cache
def draws(circle: tuple[str, ...], top: str, gap: int) -> tuple[tuple[Half, Half], ...]:
    """Every (endorsed, rejected): two disjoint coherent halves, leads ``gap`` apart.

    A half is ordered, its leading want first. The endorsed half leads with
    ``top``. Enumerated rather than drawn by rejection, so a persona costs a
    fixed amount of randomness whatever the pool.
    """
    n = len(circle)
    at = circle.index(top)
    offsets = [d for d in range(-COHERENCE, COHERENCE + 1) if d]
    out = []
    for lead_b in sorted({(at + s * gap) % n for s in (-1, 1)}):
        for second_a in [(at + d) % n for d in offsets]:
            for second_b in [(lead_b + d) % n for d in offsets]:
                if len({at, second_a, lead_b, second_b}) == 4:
                    out.append(
                        (
                            (circle[at], circle[second_a]),
                            (circle[lead_b], circle[second_b]),
                        )
                    )
    return tuple(out)


@dataclass(frozen=True)
class Persona:
    endorsed: Half
    rejected: Half
    #: ``(scale, level)`` for every disposition scale, in pool order.
    dispositions: tuple[tuple[str, str], ...]

    @property
    def top_distance(self) -> int:
        circle = tuple(load_pool()["circle"])
        return distance(circle, self.endorsed[0], self.rejected[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "endorsed": list(self.endorsed),
            "rejected": list(self.rejected),
            "dispositions": dict(self.dispositions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        endorsed, rejected = data["endorsed"], data["rejected"]
        return cls(
            endorsed=(endorsed[0], endorsed[1]),
            rejected=(rejected[0], rejected[1]),
            dispositions=tuple(data["dispositions"].items()),
        )


def draw_persona(rng: random.Random) -> Persona:
    """Draw one persona: leading want, top distance, halves, then one level per scale.

    Always the same number of draws, so personas stay aligned across runs.
    """
    pool = load_pool()
    circle = tuple(pool["circle"])
    top = circle[rng.randrange(len(circle))]
    gap = TOP_DISTANCES[rng.randrange(len(TOP_DISTANCES))]
    options = draws(circle, top, gap)
    endorsed, rejected = options[rng.randrange(len(options))]
    dispositions = tuple(
        (scale, spec["levels"][rng.randrange(len(spec["levels"]))])
        for scale, spec in pool["dispositions"].items()
    )
    return Persona(endorsed=endorsed, rejected=rejected, dispositions=dispositions)


def persona_for(sample_id: str, seed: int) -> Persona:
    """The persona a map is elicited under, fixed by the seed and the sample id."""
    return draw_persona(random.Random(f"{seed}:persona:{sample_id}"))


def render(persona: Persona) -> str:
    """The persona as it opens the system prompt."""
    pool = load_pool()
    wants = pool["wants"]
    frames = {scale: spec["frame"] for scale, spec in pool["dispositions"].items()}

    def bullets(lines: list[str]) -> str:
        return "\n".join(f"- {line}" for line in lines)

    return "\n\n".join(
        [
            (
                "These are yours. Not positions you consult and set aside — they are how you"
                " see what matters and what does not:"
            ),
            bullets([wants[w] for w in persona.endorsed]),
            "Where these two pull against each other, the earlier weighs more heavily.",
            "And these are not yours. You reject them, and would not have them:",
            bullets([wants[w] for w in persona.rejected]),
            "This is also true of you:",
            bullets([frames[s].format(level=level) for s, level in persona.dispositions]),
        ]
    )
