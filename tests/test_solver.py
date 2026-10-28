"""The solver and scorer, exercised against a scripted model."""

from __future__ import annotations

import math
import re

import httpx
import pytest
from helpers import make_map
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelName,
    ModelOutput,
)
from inspect_ai.model._model_output import Logprob, Logprobs, TopLogprob
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from openai import BadRequestError

from belcy_eval.datasets import sample_from_map
from belcy_eval.elicit import SYSTEM_PROMPT, yes_label
from belcy_eval.metrics import score_keys
from belcy_eval.personas import Persona, render
from belcy_eval.scorer import belcy_scorer, belief_state_from_metadata
from belcy_eval.solver import belcy_solver


def output_with(p_yes: float, swapped: bool) -> ModelOutput:
    """A model output whose option logprobs encode ``p_yes``."""
    yes, no = yes_label(swapped), "B" if not swapped else "A"
    top = [
        TopLogprob(token=yes, logprob=math.log(max(p_yes, 1e-12)), bytes=None),
        TopLogprob(token=no, logprob=math.log(max(1.0 - p_yes, 1e-12)), bytes=None),
    ]
    logprobs = Logprobs(
        content=[Logprob(token=top[0].token, logprob=top[0].logprob, top_logprobs=top, bytes=None)]
    )
    return ModelOutput(
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(content=top[0].token), logprobs=logprobs
            )
        ]
    )


def task_state(argmap, personas: str = "none") -> TaskState:
    sample = sample_from_map(argmap, personas=personas)
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=sample.id,
        epoch=1,
        input=sample.input,
        messages=[],
        metadata=dict(sample.metadata or {}),
    )


MARGINAL_RE = re.compile(r"Consider this claim:\n  (?P<claim>[^\n]+)\n")
CONDITIONAL_RE = re.compile(
    r"the following is (?P<negated>not )?the case:\n  (?P<given>[^\n]+)\n"
    r".*?is the following true\?\n  (?P<query>[^\n]+)\n",
    re.DOTALL,
)


def read_prompt(prompt: str) -> tuple[str, ...]:
    """Recover what a prompt asks about: ``(claim,)`` or ``(direction, given, query)``.

    Doubles as a check that the templates stay machine-readable, the negated
    supposition included.
    """
    conditional = CONDITIONAL_RE.search(prompt)
    if conditional:
        direction = "neg" if conditional.group("negated") else "pos"
        return direction, conditional.group("given"), conditional.group("query")
    marginal = MARGINAL_RE.search(prompt)
    assert marginal, f"unparseable prompt:\n{prompt}"
    return (marginal.group("claim"),)


def scripted_generate(
    marginals: dict[str, float],
    conditionals: dict[tuple[str, str, str], float] | None = None,
    *,
    bias: float = 0.0,
):
    """A model that answers from a script, optionally biased towards option A.

    ``marginals`` maps claim text to the credence to report. ``conditionals``
    maps ``(direction, supposition, query)`` triples, where direction is
    ``"pos"`` or ``"neg"``; unlisted triples fall back to the query's marginal,
    i.e. a supposition that changes nothing.
    """
    conditionals = conditionals or {}
    calls: list[str] = []
    systems: list[str] = []

    async def generate(state: TaskState, **kwargs):
        prompt = state.messages[-1].text
        calls.append(prompt)
        systems.append(state.messages[0].text)

        asked = read_prompt(prompt)
        if len(asked) == 1:
            p_yes = marginals[asked[0]]
        else:
            p_yes = conditionals.get(asked, marginals[asked[2]])

        swapped = bool(re.search(r"A: No\b", prompt))
        # A position bias pushes probability onto whatever option A is.
        p_a = (p_yes if not swapped else 1.0 - p_yes) + bias
        p_a = min(max(p_a, 1e-6), 1.0 - 1e-6)
        p_for_yes = p_a if not swapped else 1.0 - p_a

        state.output = output_with(p_for_yes, swapped)
        return state

    generate.calls = calls  # type: ignore[attr-defined]
    generate.systems = systems  # type: ignore[attr-defined]
    return generate


@pytest.fixture
def argmap():
    return make_map([("x1", "y", 1), ("x2", "y", -1)])


# --- the solver ------------------------------------------------------------


async def test_solver_elicits_every_quantity(argmap):
    state = task_state(argmap)
    answers = {"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75}

    await belcy_solver(show_map=False, debias_order=False)(state, scripted_generate(answers))

    assert state.metadata["credences"] == pytest.approx({"x1": 0.9, "x2": 0.2, "y": 0.75})
    assert set(state.metadata["conditionals"]) == {p.key for p in argmap.pairs}
    for value in state.metadata["conditionals"].values():
        assert set(value) == {"pos", "neg"}
    # 3 marginals + 2 pairs * 2 suppositions
    assert state.metadata["n_calls"] == 7
    assert state.metadata["n_failures"] == 0
    assert state.metadata["constrained"] is True


async def test_each_item_gets_its_own_context(argmap):
    """No prompt may carry an earlier answer: every call starts from scratch."""
    state = task_state(argmap)
    generate = scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75})

    await belcy_solver(show_map=False, debias_order=False)(state, generate)

    for prompt in generate.calls:  # type: ignore[attr-defined]
        assert prompt.count("Is the claim true?") + prompt.count("Under that supposition") == 1


async def test_the_frame_reaches_the_conditional_prompts(argmap):
    state = task_state(argmap)
    generate = scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75})

    await belcy_solver(show_map=False, frame="learn", debias_order=False)(state, generate)

    prompts = generate.calls  # type: ignore[attr-defined]
    assert sum("You have just learned" in p for p in prompts) == 2 * len(argmap.pairs)
    assert not any("for the sake of argument" in p for p in prompts)


@pytest.mark.parametrize("personas", ["none", "schwartz"])
async def test_every_item_of_a_map_shares_one_system_prompt(argmap, personas):
    state = task_state(argmap, personas)
    generate = scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75})

    await belcy_solver(debias_order=False)(state, generate)

    systems = set(generate.systems)  # type: ignore[attr-defined]
    assert len(systems) == 1
    (system,) = systems
    assert system.endswith(SYSTEM_PROMPT)
    persona = state.metadata["persona"]
    if persona is None:
        assert system == SYSTEM_PROMPT
    else:
        assert system.startswith(render(Persona.from_dict(persona)))


@pytest.mark.parametrize("personas", ["none", "schwartz"])
async def test_a_system_suffix_is_appended_to_every_item(argmap, personas):
    """A model whose only switch is its system prompt gets it on every call."""
    state = task_state(argmap, personas)
    generate = scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75})

    await belcy_solver(debias_order=False, system_suffix="detailed thinking off")(state, generate)

    systems = set(generate.systems)  # type: ignore[attr-defined]
    assert len(systems) == 1
    (system,) = systems
    assert system.endswith("\n\ndetailed thinking off")
    assert SYSTEM_PROMPT in system


async def test_no_system_suffix_leaves_the_prompt_byte_identical(argmap):
    """The default must not move: every run made so far was made without a suffix."""
    plain, none_given = (
        scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75}) for _ in range(2)
    )

    await belcy_solver(debias_order=False)(task_state(argmap), plain)
    await belcy_solver(debias_order=False, system_suffix=None)(task_state(argmap), none_given)

    assert set(plain.systems) == {SYSTEM_PROMPT}  # type: ignore[attr-defined]
    assert set(none_given.systems) == {SYSTEM_PROMPT}  # type: ignore[attr-defined]


async def test_debiasing_cancels_an_option_position_bias(argmap):
    answers = {"claim x1": 0.5, "claim x2": 0.5, "claim y": 0.5}

    biased = task_state(argmap)
    await belcy_solver(debias_order=False)(biased, scripted_generate(answers, bias=0.2))
    assert biased.metadata["credences"]["y"] == pytest.approx(0.7)

    debiased = task_state(argmap)
    await belcy_solver(debias_order=True)(debiased, scripted_generate(answers, bias=0.2))
    assert debiased.metadata["credences"]["y"] == pytest.approx(0.5)
    assert debiased.metadata["n_calls"] == 14


async def test_solver_records_failures_without_raising(argmap):
    async def broken_generate(state: TaskState, **kwargs):
        state.output = ModelOutput(
            choices=[ChatCompletionChoice(message=ChatMessageAssistant(content="A"), logprobs=None)]
        )
        return state

    state = task_state(argmap)
    await belcy_solver(debias_order=False)(state, broken_generate)

    assert state.metadata["n_failures"] == 7
    assert all(v is None for v in state.metadata["credences"].values())


def bad_request(message: str) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "http://vllm/v1/chat/completions"))
    return BadRequestError(message, response=response, body={"message": message})


async def test_a_context_overflow_output_is_counted_apart(argmap):
    """Inspect turns vLLM's "maximum context length" 400 into a model_length output."""

    async def overflowing_generate(state: TaskState, **kwargs):
        state.output = ModelOutput.from_content(
            "model",
            content="This model's maximum context length is 4096 tokens.",
            stop_reason="model_length",
        )
        return state

    state = task_state(argmap)
    await belcy_solver(debias_order=False)(state, overflowing_generate)

    assert state.metadata["n_failures"] == 7
    assert state.metadata["n_overflows"] == 7
    assert state.metadata["constrained"] is True


async def test_a_context_overflow_error_keeps_the_constraint(argmap):
    """The engine's wording reaches the solver as a 400; it must not read as a grammar rejection."""
    answers = {"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75}
    scripted = scripted_generate(answers)

    async def generate(state: TaskState, **kwargs):
        if "claim x1" in state.messages[-1].text:
            raise bad_request(
                "The decoder prompt (length 5000) is longer than the maximum model length of 4096."
            )
        return await scripted(state, **kwargs)

    state = task_state(argmap)
    await belcy_solver(debias_order=False)(state, generate)

    assert state.metadata["constrained"] is True
    assert state.metadata["n_overflows"] == state.metadata["n_failures"] > 0
    assert state.metadata["credences"]["x1"] is None
    assert state.metadata["credences"]["y"] == pytest.approx(0.75)


async def test_a_rejected_grammar_falls_back_to_unconstrained_decoding(argmap):
    answers = {"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75}
    scripted = scripted_generate(answers)

    async def generate(state: TaskState, **kwargs):
        if "structured_outputs" in (kwargs.get("extra_body") or {}):
            raise bad_request("structured_outputs is not supported for this model")
        return await scripted(state, **kwargs)

    state = task_state(argmap)
    await belcy_solver(debias_order=False)(state, generate)

    assert state.metadata["constrained"] is False
    assert state.metadata["n_overflows"] == 0
    assert state.metadata["credences"] == pytest.approx({"x1": 0.9, "x2": 0.2, "y": 0.75})


# --- solver to scorer ------------------------------------------------------


async def test_scorer_reads_the_solvers_metadata(argmap):
    state = task_state(argmap)
    marginals = {"claim x1": 0.95, "claim x2": 0.05, "claim y": 0.2}
    conditionals = {
        # the supporting reason raises y, the attacking one lowers it
        ("pos", "claim x1", "claim y"): 0.95,
        ("neg", "claim x1", "claim y"): 0.15,
        ("pos", "claim x2", "claim y"): 0.10,
        ("neg", "claim x2", "claim y"): 0.60,
    }
    await belcy_solver(debias_order=False)(state, scripted_generate(marginals, conditionals))

    belief = belief_state_from_metadata(state.metadata)
    assert belief.b("x1") == pytest.approx(0.95)

    score = await belcy_scorer()(state, Target(""))

    assert score.value["elicitation_coverage"] == pytest.approx(1.0)
    assert score.value["weighted_relevance_direction"] == pytest.approx(1.0)
    assert score.value["relevance_strength"] == pytest.approx((0.8 + 0.5) / 2)
    assert score.value["relevance_strength_d1"] == pytest.approx((0.8 + 0.5) / 2)
    assert score.value["misdirected_relevance"] == pytest.approx(0.0)
    assert math.isnan(score.value["weighted_relevance_direction_d2"])
    assert math.isnan(score.value["relevance_strength_d2"])
    assert score.metadata["index_sizes"] == {"claims": 3, "pairs": 2, "pairs_by_distance": {1: 2}}
    # every key here is aggregated by the metric glob, so nothing rides along,
    # and every sample carries the same keys whatever its map
    assert list(score.value) == list(score_keys(3)) + [
        "elicitation_coverage",
        "failure_rate",
        "overflow_rate",
        "unconstrained",
    ]


async def test_scorer_survives_a_total_elicitation_failure(argmap):
    state = task_state(argmap)
    state.metadata["credences"] = {c.id: None for c in argmap.claims}
    state.metadata["conditionals"] = {p.key: {"pos": None, "neg": None} for p in argmap.pairs}

    score = await belcy_scorer()(state, Target(""))

    # nan keeps the map out of the aggregate rather than scoring it zero
    assert all(math.isnan(score.value[key]) for key in score_keys(3))
    assert score.value["elicitation_coverage"] == pytest.approx(0.0)
    assert "undefined" in score.explanation


# --- elicitation diagnostics ------------------------------------------------


async def test_the_scorer_reports_how_the_elicitation_went(argmap):
    """A run degraded by overflows and a rejected grammar says so in the score value."""

    async def generate(state: TaskState, **kwargs):
        if "structured_outputs" in (kwargs.get("extra_body") or {}):
            raise bad_request("structured_outputs is not supported for this model")
        state.output = ModelOutput.from_content(
            "model",
            content="This model's maximum context length is 4096 tokens.",
            stop_reason="model_length",
        )
        return state

    state = task_state(argmap)
    await belcy_solver(debias_order=False)(state, generate)
    score = await belcy_scorer()(state, Target(""))

    assert state.metadata["constrained"] is False
    assert score.value["failure_rate"] == pytest.approx(1.0)
    assert score.value["overflow_rate"] == pytest.approx(1.0)
    assert score.value["unconstrained"] == 1.0


async def test_a_clean_elicitation_reports_no_diagnostics(argmap):
    state = task_state(argmap)
    await belcy_solver(debias_order=False)(
        state, scripted_generate({"claim x1": 0.9, "claim x2": 0.2, "claim y": 0.75})
    )
    score = await belcy_scorer()(state, Target(""))

    assert score.value["failure_rate"] == 0.0
    assert score.value["overflow_rate"] == 0.0
    assert score.value["unconstrained"] == 0.0


async def test_the_diagnostics_do_not_divide_by_a_missing_call_count(argmap):
    """A hand-built state, or a log written before the solver counted calls."""
    state = task_state(argmap)
    state.metadata["credences"] = {c.id: 0.5 for c in argmap.claims}
    state.metadata["conditionals"] = {p.key: {"pos": 0.6, "neg": 0.4} for p in argmap.pairs}

    score = await belcy_scorer()(state, Target(""))

    assert math.isnan(score.value["failure_rate"])
    assert math.isnan(score.value["overflow_rate"])
    assert score.value["unconstrained"] == 0.0
