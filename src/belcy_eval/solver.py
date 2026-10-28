"""The elicitation solver.

Runs every elicitation for one map and writes the resulting belief state into
sample metadata for the scorer.

Metadata contract, filled by this solver and read by ``scorer.belcy_scorer``:

* ``metadata["credences"]``: ``dict[claim_id, float | None]`` -- the marginals
  ``b(z)``, one per claim, ``None`` where elicitation failed.
* ``metadata["conditionals"]``: ``dict[pair_key, {"pos": float | None,
  "neg": float | None}]`` -- ``c(y|x)`` and ``c(y|not-x)`` for each of the
  map's pairs: one query, ``y``, under ``x`` affirmed and ``x`` denied.
* ``metadata["n_calls"]``: total generate calls made.
* ``metadata["n_failures"]``: calls that yielded no usable distribution.
* ``metadata["n_overflows"]``: the part of those failures whose prompt did not
  fit the served context window. They are counted apart because they say
  something about the map and the serving setup, not about the model.
* ``metadata["constrained"]``: whether structured outputs were accepted, so a
  silent fallback to unconstrained decoding is visible in the log.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import NotRequired, TypedDict, Unpack

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import collect
from openai import BadRequestError, UnprocessableEntityError

from belcy_eval.datasets import map_from_metadata, persona_from_metadata
from belcy_eval.elicit import (
    DEFAULT_FRAME,
    LABELS,
    Item,
    combine,
    credence_from_choice,
    expand,
    items_for,
    system_prompt,
)

logger = logging.getLogger(__name__)

#: Logprobs to request at the option position. The option letter may arrive
#: under several tokenizations, so a handful of alternatives are needed; vLLM
#: caps this at ``--max-logprobs`` (20 by default).
TOP_LOGPROBS = 8

#: The answer is a single letter. Four tokens leaves room for a stray space or
#: a wrapper the constraint did not catch.
MAX_TOKENS = 4


#: Fragments of the messages a prompt too long for the served context window
#: comes back with. vLLM words this differently depending on where the check
#: trips ("maximum context length" in the API server, "maximum model length" in
#: the engine); Inspect converts the first into a ``model_length`` output
#: itself, the second reaches the solver as a 400.
CONTEXT_OVERFLOW_MARKERS = ("maximum context length", "maximum model length", "max_model_len")


def is_context_overflow(exc: BadRequestError | UnprocessableEntityError) -> bool:
    """Whether a rejected call failed on prompt length rather than on its request body."""
    body = exc.body.get("message") if isinstance(exc.body, dict) else None
    message = f"{exc} {body or ''}".lower()
    return any(marker in message for marker in CONTEXT_OVERFLOW_MARKERS)


class SolverArguments(TypedDict):
    show_map: bool
    frame: NotRequired[str]
    debias_order: bool
    max_concurrent: int
    system_suffix: NotRequired[str | None]


@solver
def belcy_solver(**kwargs: Unpack[SolverArguments]) -> Solver:
    """Elicit the full belief state for one argument map.

    Every item is asked in its own context, built from a fresh copy of the
    state, so nothing an earlier answer said can anchor a later one.
    """

    show_map = kwargs.get("show_map", False)
    frame = kwargs.get("frame", DEFAULT_FRAME)
    debias_order = kwargs.get("debias_order", True)
    max_concurrent = kwargs.get("max_concurrent", 16)
    system_suffix = kwargs.get("system_suffix")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        argmap = map_from_metadata(state.metadata)
        items = items_for(argmap, show_map=show_map, frame=frame)
        # One believer per map: every item is asked under the same persona.
        system = system_prompt(persona_from_metadata(state.metadata), system_suffix)
        limiter = asyncio.Semaphore(max_concurrent)

        # Flags shared across the fan-out: whether structured outputs were
        # accepted, how many calls came back unusable, and how many of those
        # did not fit the context window.
        constrained = True
        failures = 0
        overflows = 0

        async def ask(item: Item, swapped: bool) -> tuple[Item, float | None]:
            nonlocal constrained, failures, overflows

            probe = deepcopy(state)
            probe.messages = [
                ChatMessageSystem(content=system),
                ChatMessageUser(content=item.prompt(swapped)),
            ]

            async with limiter:
                use_constraint = constrained
                try:
                    result = await generate(
                        probe,
                        logprobs=True,
                        top_logprobs=TOP_LOGPROBS,
                        max_tokens=MAX_TOKENS,
                        # Logprobs are reported after temperature scaling, so
                        # anything but 1.0 would report a sharpened or
                        # flattened distribution rather than the model's own
                        # credence. This must not inherit the run temperature.
                        temperature=1.0,
                        extra_body={
                            "structured_outputs": {"choice": list(LABELS)},
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                        if use_constraint
                        else {"chat_template_kwargs": {"enable_thinking": False}},
                    )
                except (BadRequestError, UnprocessableEntityError) as exc:
                    # Only a rejected request lands here: transient failures
                    # have already exhausted Inspect's retry ladder, so
                    # retrying those would just double the load. A prompt too
                    # long for the context window fails the same way with or
                    # without the constraint, so it must not switch it off.
                    if is_context_overflow(exc):
                        logger.warning("prompt exceeds the served context window: %s", exc)
                        failures += 1
                        overflows += 1
                        return item, None
                    if not use_constraint:
                        logger.warning("elicitation call rejected: %s", exc)
                        failures += 1
                        return item, None
                    logger.warning(
                        "structured outputs rejected, falling back to unconstrained "
                        "decoding for the rest of this sample: %s",
                        exc,
                    )
                    constrained = False
                    use_constraint = False
                    result = await generate(
                        probe,
                        logprobs=True,
                        top_logprobs=TOP_LOGPROBS,
                        max_tokens=MAX_TOKENS,
                        temperature=1.0,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )

            if result.output.stop_reason == "model_length":
                failures += 1
                overflows += 1
                return item, None

            choices = result.output.choices
            if not choices:
                failures += 1
                return item, None

            credence = credence_from_choice(choices[0], swapped=swapped, constrained=use_constraint)
            if credence is None:
                failures += 1
            return item, credence

        pairs = list(expand(items, debias_order))
        results = await collect(*(ask(item, swapped) for item, swapped in pairs))

        # Average the orderings of each item back together.
        gathered: dict[tuple[str, str, str], list[float | None]] = {}
        for item, credence in results:
            gathered.setdefault((item.kind, item.key, item.direction), []).append(credence)

        credences: dict[str, float | None] = {}
        conditionals: dict[str, dict[str, float | None]] = {}
        for (kind, key, direction), estimates in gathered.items():
            value = combine(estimates)
            if kind == "marginal":
                credences[key] = value
            else:
                conditionals.setdefault(key, {})[direction] = value

        state.metadata["credences"] = credences
        state.metadata["conditionals"] = conditionals
        state.metadata["n_calls"] = len(pairs)
        state.metadata["n_failures"] = failures
        state.metadata["n_overflows"] = overflows
        state.metadata["constrained"] = constrained

        return state

    return solve
