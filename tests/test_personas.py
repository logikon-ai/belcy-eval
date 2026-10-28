from __future__ import annotations

import random

import pytest

from belcy_eval.personas import (
    COHERENCE,
    TOP_DISTANCES,
    Persona,
    distance,
    draw_persona,
    load_pool,
    persona_for,
    render,
    validate_generator,
)


@pytest.fixture
def pool():
    return load_pool()


@pytest.fixture
def circle(pool):
    return tuple(pool["circle"])


def test_pool_is_consistent(pool):
    assert sorted(pool["circle"]) == sorted(pool["wants"])
    for spec in pool["dispositions"].values():
        assert "{level}" in spec["frame"]
        assert all(len(level.split()) == 1 for level in spec["levels"])


def test_distance_goes_the_short_way_round(circle):
    assert distance(circle, circle[0], circle[1]) == 1
    assert distance(circle, circle[0], circle[-1]) == 1
    assert distance(circle, circle[0], circle[5]) == 5


def test_draws_are_coherent_disjoint_halves(circle):
    rng = random.Random(0)
    for _ in range(500):
        persona = draw_persona(rng)
        wants = persona.endorsed + persona.rejected
        assert len(set(wants)) == 4
        for half in (persona.endorsed, persona.rejected):
            assert distance(circle, *half) <= COHERENCE
        assert persona.top_distance in TOP_DISTANCES


def test_draws_reach_the_whole_circle_and_every_distance(circle):
    rng = random.Random(1)
    personas = [draw_persona(rng) for _ in range(1000)]
    assert {p.endorsed[0] for p in personas} == set(circle)
    assert {p.top_distance for p in personas} == set(TOP_DISTANCES)


def test_persona_is_fixed_by_seed_and_sample(pool):
    assert persona_for("map-1", 42) == persona_for("map-1", 42)
    drawn = {persona_for(f"map-{i}", 42) for i in range(50)}
    assert len(drawn) > 1
    assert {persona_for(f"map-{i}", 7) for i in range(50)} != drawn


def test_persona_round_trips_through_a_dict():
    persona = persona_for("map-1", 42)
    assert Persona.from_dict(persona.to_dict()) == persona
    # metadata carries extra fields beside the persona itself
    assert Persona.from_dict({**persona.to_dict(), "generator": "schwartz"}) == persona


def test_render_holds_every_sentence_and_no_name(pool):
    persona = persona_for("map-1", 42)
    text = render(persona)
    for want in persona.endorsed + persona.rejected:
        assert f"- {pool['wants'][want]}" in text
    for scale, level in persona.dispositions:
        assert pool["dispositions"][scale]["frame"].format(level=level) in text
    # endorsed before rejected, leading want first
    positions = [text.index(pool["wants"][w]) for w in persona.endorsed + persona.rejected]
    assert positions == sorted(positions)
    assert "You are" not in text


def test_unknown_generator_is_rejected():
    assert validate_generator("none") == "none"
    with pytest.raises(ValueError, match="personas must be one of"):
        validate_generator("socratic")
