"""Eliciting credences from a model.

Every quantity the metrics need is elicited as a forced binary choice whose
answer is read off the logprobs of the two option tokens. That gives a
continuous credence from a single call, with no number for the model to
verbalize and no parsing failure mode beyond the logprobs being absent.

Two hazards the design has to control:

* **Anchoring.** Each item is asked in its own context, containing one
  question and nothing else. The two conditionals of a pair never share a
  context. They are the same query under complementary suppositions, so
  presenting both invites the model to answer the second by negating the first,
  which would void the confirmation measure as a measurement; and presenting
  the marginal beside them invites enforcing the law of total probability by
  arithmetic, which would void probabilistic coherence.
* **Option-order bias.** The order of Yes and No shifts the logprob on its
  own, and that shift would surface as incoherence. Each item is therefore
  asked twice with the options swapped, and the two estimates averaged.

Conditionals are elicited for claim pairs ``(x, y)`` joined by a directed path,
not only for direct edges, and, with rival roots, for a reason under one root
paired with each rival root (see :mod:`belcy_eval.relevance`). Both conditionals of a pair query ``y``. They differ
in the supposition: ``x`` affirmed, then ``x`` denied. The relevance the map
asserts between them is never shown, so the model is never told which answer
the map expects. Credence in the arrow itself is never elicited: read as a material conditional it is inflated
by every case where the premise is improbable, and Lewis' triviality results
rule out any proposition whose credence tracks the conditional probability in
general.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from belcy_eval.maps import ArgMap
from belcy_eval.personas import Persona, render
from belcy_eval.relevance import Pair

logger = logging.getLogger(__name__)

LABELS = ("A", "B")

SYSTEM_PROMPT = (
    "You report your own degrees of belief. Answer every question with a single "
    "letter and nothing else."
)


def system_prompt(persona: Persona | None = None, suffix: str | None = None) -> str:
    """The instruction every item is asked under, behind the persona if there is one.

    ``suffix`` appends a line the served model reads as a control string, for a
    model whose only switch is its system prompt. It is a property of the model,
    not of the task, so a comparison must give it to every model it compares.
    """
    prompt = SYSTEM_PROMPT if persona is None else f"{render(persona)}\n\n{SYSTEM_PROMPT}"
    return f"{prompt}\n\n{suffix}" if suffix else prompt


# Each item is a context and a question, which a chat prompt joins with a blank
# line (see ``Item.body``).

TOPIC_CONTEXT = "Topic under discussion: {topic}"

MARGINAL_QUESTION = """\
Consider this claim:
  {claim}

Is the claim true?"""

CONDITIONAL_CONTEXT = """\
Topic under discussion: {topic}

Suppose, for the sake of argument, that the following {polarity}:
  {supposition}"""

CONDITIONAL_QUESTION = """\
Under that supposition, is the following true?
  {query}"""

#: Wordings of the conditional question, for testing how much the credences
#: depend on it. They differ only in the sentences that introduce the
#: supposition and the query; the polarity clause and the claims are the same.
#: ``suppose`` is the default. ``learn`` frames the supposition as news, which
#: invites updating rather than supposing. Each frame is a context and a
#: question template, as for every item.
CONDITIONAL_FRAMES = {
    "suppose": (CONDITIONAL_CONTEXT, CONDITIONAL_QUESTION),
    "assume": (
        """\
Topic under discussion: {topic}

Assume that the following {polarity}:
  {supposition}""",
        """\
Given this assumption, is the following true?
  {query}""",
    ),
    "learn": (
        """\
Topic under discussion: {topic}

You have just learned that the following {polarity}:
  {supposition}""",
        """\
In light of this, is the following true?
  {query}""",
    ),
}

DEFAULT_FRAME = "suppose"


def validate_frame(frame: str) -> None:
    if frame not in CONDITIONAL_FRAMES:
        raise ValueError(f"unknown frame {frame!r}; one of {sorted(CONDITIONAL_FRAMES)}")


#: The one clause that separates the two conditionals of a pair.
SUPPOSITION_POLARITY = {"pos": "is the case", "neg": "is not the case"}

MAP_HEADER = """\
The debate on this topic has been mapped out as follows. Each indented entry is \
a reason bearing on the entry above it: `<+` for a reason in favour, `<-` for a \
reason against.{roots_note}

{rendered}

"""

#: How several unindented entries are to be read, keyed by ``ArgMap.mece_roots``.
ROOTS_NOTE = {
    True: (
        " Where there are several unindented entries, they are rival positions, "
        "of which exactly one is correct."
    ),
    False: (
        " Where there are several unindented entries, they are separate positions, "
        "each of which may be correct or not independently of the others."
    ),
}

#: What a coding debate is about, opening every context of its map. Each part
#: is shown only if the map has it, so maps with neither are asked as before.
PROBLEM_CONTEXT = """\
The debate concerns the following programming problem.

Problem:
{problem}

"""

DESIGN_DECISION_CONTEXT = """\
Design decision: {design_decision}

"""

#: Bumped whenever the prompts change, and recorded with every sample, so runs
#: asked differently are not paired by mistake. Logs without it are version 1.
#: Version 2 opens coding maps with :data:`PROBLEM_CONTEXT`; on maps with
#: neither a problem nor a design decision its prompts are those of version 1.
PROMPT_VERSION = 2

OPTION_BLOCK = """
A: {a}
B: {b}"""


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_map(argmap: ArgMap, *, indent: str = "    ") -> str:
    """Render a map as indented argdown-style trees, one per root.

    A claim bearing on several others is rendered in full under each of them.
    """

    lines: list[str] = []

    def walk(claim_id: str, depth: int, marker: str, seen: frozenset[str]) -> None:
        prefix = indent * depth
        text = argmap.text(claim_id)
        lines.append(f"{prefix}{marker}{text}" if marker else f"{prefix}{text}")
        if claim_id in seen:
            lines.append(f"{prefix}{indent}(see above)")
            return
        seen = seen | {claim_id}
        for edge in sorted(argmap.incoming(claim_id), key=lambda e: (-e.polarity, e.source)):
            walk(edge.source, depth + 1, "<+ " if edge.supports else "<- ", seen)

    for i, root in enumerate(argmap.roots):
        if i:
            lines.append("")
        walk(root, 0, "", frozenset())
    return "\n".join(lines)


def _options(swapped: bool) -> str:
    a, b = ("No", "Yes") if swapped else ("Yes", "No")
    return OPTION_BLOCK.format(a=a, b=b)


def yes_label(swapped: bool) -> str:
    """Which option letter means Yes under this ordering."""
    return "B" if swapped else "A"


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

ItemKind = Literal["marginal", "conditional"]


@dataclass(frozen=True)
class Item:
    """One quantity to elicit, and the text of the question that elicits it.

    ``context`` and ``question`` are already formatted, so they hold claim text
    verbatim and must never be formatted again: braces in the text (LaTeX,
    regexes, code) would be read as replacement fields.
    """

    kind: ItemKind
    key: str
    #: What the question is asked against: the map if shown, the topic, and
    #: for conditionals the supposition.
    context: str
    #: What is asked: the claim or the query, up to the options.
    question: str
    #: For conditionals: ``"pos"`` is ``c(y|x)``, ``"neg"`` is ``c(y|not-x)``.
    direction: str = ""

    @property
    def body(self) -> str:
        """The chat question up to the options."""
        return f"{self.context}\n\n{self.question}"

    def prompt(self, swapped: bool) -> str:
        return self.body + _options(swapped)


def shows_problem(argmap: ArgMap) -> bool:
    """Whether the map's prompts open with a problem or a design decision."""
    return bool(argmap.problem or argmap.design_decision)


def _problem(argmap: ArgMap) -> str:
    block = ""
    if argmap.problem:
        block += PROBLEM_CONTEXT.format(problem=argmap.problem)
    if argmap.design_decision:
        block += DESIGN_DECISION_CONTEXT.format(design_decision=argmap.design_decision)
    return block


def _prefix(argmap: ArgMap, show_map: bool) -> str:
    """What comes before the topic line: the problem if any, then the map if shown."""
    if not show_map:
        return _problem(argmap)
    return _problem(argmap) + MAP_HEADER.format(
        roots_note=ROOTS_NOTE[argmap.mece_roots],
        rendered=render_map(argmap),
    )


def marginal_item(argmap: ArgMap, claim_id: str, *, show_map: bool) -> Item:
    return Item(
        kind="marginal",
        key=claim_id,
        context=_prefix(argmap, show_map) + TOPIC_CONTEXT.format(topic=argmap.topic),
        question=MARGINAL_QUESTION.format(claim=argmap.text(claim_id)),
    )


def conditional_item(
    argmap: ArgMap, pair: Pair, *, direction: str, show_map: bool, frame: str = DEFAULT_FRAME
) -> Item:
    """Query ``y``, supposing ``x``: affirmed on ``pos``, denied on ``neg``.

    The pair's relevance sign is never shown. Only the supposition is ever negated,
    and it is negated by a clause of the frame rather than by rewriting the
    claim, so the claim text reaches the model verbatim either way. ``frame``
    picks the wording (:data:`CONDITIONAL_FRAMES`).
    """
    if direction not in SUPPOSITION_POLARITY:
        raise ValueError(f"unknown conditional direction: {direction!r}")
    validate_frame(frame)

    context_template, question_template = CONDITIONAL_FRAMES[frame]
    context = _prefix(argmap, show_map) + context_template.format(
        topic=argmap.topic,
        polarity=SUPPOSITION_POLARITY[direction],
        supposition=argmap.text(pair.source),
    )
    return Item(
        kind="conditional",
        key=pair.key,
        context=context,
        question=question_template.format(query=argmap.text(pair.target)),
        direction=direction,
    )


def items_for(argmap: ArgMap, *, show_map: bool, frame: str = DEFAULT_FRAME) -> list[Item]:
    """Every elicitation item for a map: one per claim, two per pair."""
    items = [marginal_item(argmap, c.id, show_map=show_map) for c in argmap.claims]
    for pair in argmap.pairs:
        for direction in ("pos", "neg"):
            items.append(
                conditional_item(argmap, pair, direction=direction, show_map=show_map, frame=frame)
            )
    return items


def orderings(debias_order: bool) -> tuple[bool, ...]:
    """Option orderings to ask each item under."""
    return (False, True) if debias_order else (False,)


def expand(items: Sequence[Item], debias_order: bool) -> Iterator[tuple[Item, bool]]:
    for item in items:
        for swapped in orderings(debias_order):
            yield item, swapped


# ---------------------------------------------------------------------------
# reading a credence off the logprobs
# ---------------------------------------------------------------------------


def normalize_token(token: str) -> str:
    return token.strip().strip("\"'*`").upper()


def label_probs(
    candidates: Sequence[tuple[str, float]],
    labels: Sequence[str] = LABELS,
) -> dict[str, float] | None:
    """Normalize token logprobs into a distribution over the option labels.

    ``candidates`` are ``(token, logprob)`` pairs from one token position. A
    label may appear under several tokenizations (``"A"``, ``" A"``); the best
    logprob wins. Returns ``None`` unless every label is present, since a
    distribution over one option carries no information about the other.
    """
    best: dict[str, float] = {}
    for token, logprob in candidates:
        label = normalize_token(token)
        if label in labels and logprob > best.get(label, -math.inf):
            best[label] = logprob

    if len(best) < len(labels):
        return None

    probs = {label: math.exp(lp) for label, lp in best.items()}
    total = sum(probs.values())
    if total <= 0.0:
        return None
    return {label: p / total for label, p in probs.items()}


def candidates_from_choice(choice: Any, *, position: int = 0) -> list[tuple[str, float]]:
    """Pull ``(token, logprob)`` pairs at one position out of a model choice.

    With the completion constrained to a single option letter, the position of
    interest is the first content token.
    """
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content or position >= len(content):
        return []

    entry = content[position]
    pairs = [(entry.token, entry.logprob)]
    for alt in entry.top_logprobs or []:
        pairs.append((alt.token, alt.logprob))
    return pairs


def find_label_position(choice: Any, labels: Sequence[str] = LABELS) -> int | None:
    """First token position whose candidates include a label.

    Only needed on the unconstrained fallback path, where the model may emit a
    preamble before the letter.
    """
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs else None
    if not content:
        return None
    for i, entry in enumerate(content):
        if normalize_token(entry.token) in labels:
            return i
    return None


def credence_from_choice(choice: Any, *, swapped: bool, constrained: bool) -> float | None:
    """``P(the claim is true)`` from one model choice, or ``None`` on failure."""
    position = 0 if constrained else find_label_position(choice)
    if position is None:
        logger.warning("no option label found among the generated tokens")
        return None

    probs = label_probs(candidates_from_choice(choice, position=position))
    if probs is None:
        logger.warning("could not normalize a distribution over the option labels")
        return None
    return probs[yes_label(swapped)]


def combine(estimates: Sequence[float | None]) -> float | None:
    """Average the estimates that came back, ignoring failures."""
    got = [e for e in estimates if e is not None]
    if not got:
        return None
    return sum(got) / len(got)
