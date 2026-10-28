from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

import pytest
from helpers import CODING_DESIGN_DECISION, CODING_PROBLEM, coding_debate, dag_debate, make_map

from belcy_eval.elicit import (
    CONDITIONAL_FRAMES,
    DEFAULT_FRAME,
    SYSTEM_PROMPT,
    combine,
    conditional_item,
    credence_from_choice,
    expand,
    find_label_position,
    items_for,
    label_probs,
    marginal_item,
    render_map,
    shows_problem,
    system_prompt,
    yes_label,
)
from belcy_eval.maps import to_argmap
from belcy_eval.personas import persona_for, render
from belcy_eval.relevance import pair_key

# --- fakes shaped like inspect's logprob objects ---------------------------


@dataclass
class FakeTop:
    token: str
    logprob: float


@dataclass
class FakeLogprob:
    token: str
    logprob: float
    top_logprobs: list[FakeTop] = field(default_factory=list)


@dataclass
class FakeLogprobs:
    content: list[FakeLogprob]


@dataclass
class FakeChoice:
    logprobs: FakeLogprobs | None


def choice_over(p_a: float, *, prefix: list[str] | None = None, tokens=("A", "B")) -> FakeChoice:
    """A choice whose option position splits ``p_a`` / ``1 - p_a`` over the labels."""
    a_tok, b_tok = tokens
    entries = [FakeLogprob(token=t, logprob=-1.0, top_logprobs=[]) for t in (prefix or [])]
    entries.append(
        FakeLogprob(
            token=a_tok,
            logprob=math.log(p_a),
            top_logprobs=[
                FakeTop(token=a_tok, logprob=math.log(p_a)),
                FakeTop(token=b_tok, logprob=math.log(1.0 - p_a)),
            ],
        )
    )
    return FakeChoice(logprobs=FakeLogprobs(content=entries))


# --- the distribution ------------------------------------------------------


def test_label_probs_normalizes():
    probs = label_probs([("A", math.log(0.3)), ("B", math.log(0.1))])
    assert probs == pytest.approx({"A": 0.75, "B": 0.25})


def test_label_probs_tolerates_tokenization_variants():
    probs = label_probs([(" A", math.log(0.5)), ('"B"', math.log(0.5))])
    assert probs == pytest.approx({"A": 0.5, "B": 0.5})


def test_label_probs_keeps_the_best_logprob_per_label():
    probs = label_probs([("A", math.log(0.1)), (" A", math.log(0.9)), ("B", math.log(0.1))])
    assert probs["A"] == pytest.approx(0.9)


def test_label_probs_needs_both_labels():
    assert label_probs([("A", math.log(0.99))]) is None
    assert label_probs([("Yes", math.log(0.5)), ("No", math.log(0.5))]) is None


# --- credence from a choice ------------------------------------------------


def test_credence_reads_the_yes_option():
    assert credence_from_choice(choice_over(0.8), swapped=False, constrained=True) == pytest.approx(
        0.8
    )


def test_swapped_ordering_flips_which_letter_means_yes():
    assert yes_label(False) == "A" and yes_label(True) == "B"
    # the same distribution, asked with the options swapped, means the opposite
    assert credence_from_choice(choice_over(0.8), swapped=True, constrained=True) == pytest.approx(
        0.2
    )


def test_unconstrained_path_skips_a_preamble():
    choice = choice_over(0.7, prefix=["Answer", ":", " "])
    assert find_label_position(choice) == 3
    assert credence_from_choice(choice, swapped=False, constrained=False) == pytest.approx(0.7)


def test_missing_logprobs_yield_none():
    assert credence_from_choice(FakeChoice(logprobs=None), swapped=False, constrained=True) is None
    empty = FakeChoice(logprobs=FakeLogprobs(content=[]))
    assert credence_from_choice(empty, swapped=False, constrained=True) is None


def test_no_label_among_the_tokens_yields_none():
    choice = FakeChoice(
        logprobs=FakeLogprobs(content=[FakeLogprob(token="Yes", logprob=-0.1, top_logprobs=[])])
    )
    assert credence_from_choice(choice, swapped=False, constrained=False) is None


# --- averaging the orderings ----------------------------------------------


def test_combine_averages_and_ignores_failures():
    assert combine([0.8, 0.6]) == pytest.approx(0.7)
    assert combine([0.8, None]) == pytest.approx(0.8)
    assert combine([None, None]) is None


def test_order_debiasing_cancels_a_position_bias():
    """A model that leans on A by a fixed amount lands on the truth after averaging."""
    bias = 0.1
    asked_plain = credence_from_choice(choice_over(0.5 + bias), swapped=False, constrained=True)
    asked_swapped = credence_from_choice(choice_over(0.5 + bias), swapped=True, constrained=True)
    assert combine([asked_plain, asked_swapped]) == pytest.approx(0.5)


# --- items and prompts -----------------------------------------------------


def test_one_marginal_per_claim_and_two_conditionals_per_pair(simple_map):
    items = items_for(simple_map, show_map=False)
    assert sum(1 for i in items if i.kind == "marginal") == 3
    assert sum(1 for i in items if i.kind == "conditional") == 4
    assert {i.direction for i in items if i.kind == "conditional"} == {"pos", "neg"}


def test_expand_doubles_items_when_debiasing(simple_map):
    items = items_for(simple_map, show_map=False)
    assert len(list(expand(items, True))) == 2 * len(items)
    assert len(list(expand(items, False))) == len(items)


def test_marginal_prompt_states_the_claim_and_the_options(simple_map):
    prompt = marginal_item(simple_map, "y", show_map=False).prompt(swapped=False)
    assert "claim y" in prompt
    assert prompt.rstrip().endswith("A: Yes\nB: No")


def test_swapped_prompt_lists_no_first(simple_map):
    prompt = marginal_item(simple_map, "y", show_map=False).prompt(swapped=True)
    assert prompt.rstrip().endswith("A: No\nB: Yes")


def test_both_conditionals_suppose_x_and_query_y(simple_map):
    pair = next(p for p in simple_map.pairs if p.sign == 1)
    pos = conditional_item(simple_map, pair, direction="pos", show_map=False).prompt(False)
    neg = conditional_item(simple_map, pair, direction="neg", show_map=False).prompt(False)

    for prompt in (pos, neg):
        assert prompt.index("claim x1") < prompt.index("claim y")


def test_a_distant_pair_supposes_the_remote_reason_and_queries_its_conclusion():
    argmap = make_map([("x", "y", 1), ("z", "x", -1)])
    pair = next(p for p in argmap.pairs if p.distance == 2)
    assert (pair.source, pair.target) == ("z", "y")
    prompt = conditional_item(argmap, pair, direction="pos", show_map=False).prompt(False)
    assert prompt.index("claim z") < prompt.index("claim y")
    assert "claim x" not in prompt
    assert len(items_for(argmap, show_map=False)) == 3 + 2 * argmap.n_pairs


def test_the_two_conditionals_differ_only_in_the_supposition_clause(simple_map):
    pair = simple_map.pairs[0]
    pos = conditional_item(simple_map, pair, direction="pos", show_map=False).prompt(False)
    neg = conditional_item(simple_map, pair, direction="neg", show_map=False).prompt(False)

    assert "that the following is the case:" in pos
    assert "that the following is not the case:" in neg
    assert pos == neg.replace("is not the case", "is the case")


def test_the_negated_supposition_leaves_the_claim_text_verbatim(simple_map):
    """Only the frame is negated, never the claim, so no negation scope is invented."""
    pair = simple_map.pairs[0]
    neg = conditional_item(simple_map, pair, direction="neg", show_map=False).prompt(False)
    assert f"  {simple_map.text(pair.source)}\n" in neg
    assert f"  {simple_map.text(pair.target)}\n" in neg


def test_conditional_never_reveals_the_sign(simple_map):
    for pair in simple_map.pairs:
        for direction in ("pos", "neg"):
            prompt = conditional_item(simple_map, pair, direction=direction, show_map=False).prompt(
                False
            )
            assert "support" not in prompt.lower()
            assert "attack" not in prompt.lower()
            assert "against" not in prompt.lower()


def test_conditional_key_matches_the_metrics_key(simple_map):
    pair = simple_map.pairs[0]
    item = conditional_item(simple_map, pair, direction="pos", show_map=False)
    assert item.key == pair_key(pair.source, pair.target)


@pytest.mark.parametrize("show_map", [False, True])
def test_braces_in_the_text_reach_the_prompt_verbatim(show_map):
    """Claims about code and maths carry braces that must not be read as format fields."""
    texts = {
        "y": "a_{1} + a_{2} is maximal",
        "x": r"the pattern \d{1,} matches, and {P} holds",
        "z": "digits {5-9} occur, as does {options}",
    }
    base = make_map([("x", "y", 1), ("z", "y", -1)])
    argmap = replace(
        base,
        topic="Codeforces {0}",
        claims=tuple(replace(c, text=texts[c.id]) for c in base.claims),
    )

    for item in items_for(argmap, show_map=show_map):
        for swapped in (False, True):
            prompt = item.prompt(swapped)
            assert "Codeforces {0}" in prompt
            assert prompt.rstrip().endswith("A: No\nB: Yes" if swapped else "A: Yes\nB: No")

    pair = next(p for p in argmap.pairs if p.source == "z")
    prompt = conditional_item(argmap, pair, direction="neg", show_map=show_map).prompt(False)
    assert f"  {texts['z']}\n" in prompt
    assert f"  {texts['y']}\n" in prompt
    assert prompt.count("A: Yes") == 1


def test_unknown_direction_is_rejected(simple_map):
    with pytest.raises(ValueError, match="direction"):
        conditional_item(simple_map, simple_map.pairs[0], direction="sideways", show_map=False)


# --- map rendering ---------------------------------------------------------


def test_render_map_marks_support_and_attack(simple_map):
    rendered = render_map(simple_map)
    lines = rendered.splitlines()
    assert lines[0] == "claim y"
    assert any(line.strip().startswith("<+ ") for line in lines[1:])
    assert any(line.strip().startswith("<- ") for line in lines[1:])


def test_render_map_nests_by_depth():
    argmap = make_map([("x", "y", 1), ("z", "x", 1)])
    lines = render_map(argmap).splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines]
    assert indents == [0, 4, 8]


def test_show_map_puts_the_tree_in_the_prompt(simple_map):
    prompt = marginal_item(simple_map, "y", show_map=True).prompt(False)
    assert render_map(simple_map) in prompt
    assert "<+" in prompt


def test_a_suffix_goes_last_so_the_instruction_still_reads_as_one_block():
    persona = persona_for("m", 0)
    assert (
        system_prompt(None, "detailed thinking off") == f"{SYSTEM_PROMPT}\n\ndetailed thinking off"
    )
    assert system_prompt(persona, "detailed thinking off") == (
        f"{render(persona)}\n\n{SYSTEM_PROMPT}\n\ndetailed thinking off"
    )
    # an empty suffix is no suffix: a model file that sets it blank changes nothing
    assert system_prompt(None, "") == SYSTEM_PROMPT
    assert system_prompt(None, None) == SYSTEM_PROMPT


def test_system_prompt_puts_the_persona_before_the_instruction():
    assert system_prompt(None) == SYSTEM_PROMPT
    persona = persona_for("map-1", 42)
    assert system_prompt(persona) == f"{render(persona)}\n\n{SYSTEM_PROMPT}"


def test_blind_prompt_mentions_only_its_own_claim(simple_map):
    prompt = marginal_item(simple_map, "x1", show_map=False).prompt(False)
    assert "claim x2" not in prompt


def test_render_map_lists_rival_roots_as_separate_trees():
    argmap = make_map([("a", "r1", 1), ("b", "r2", -1), ("s", "a", 1), ("s", "b", 1)])
    blocks = render_map(argmap).split("\n\n")
    assert [block.splitlines()[0] for block in blocks] == ["claim r1", "claim r2"]
    # a reason bearing on two claims appears under each
    assert all("claim s" in block for block in blocks)


def test_multi_root_topic_lists_the_rivals_instead_of_picking_one():
    argmap = to_argmap(dag_debate(), rng=random.Random(0), min_claims=6, max_claims=6)
    assert argmap.roots == ("r1", "r2")
    assert argmap.topic.startswith("which of these rival positions is correct")
    assert "(1) Claim r1." in argmap.topic and "(2) Claim r2." in argmap.topic


def test_independent_roots_are_listed_without_calling_them_rivals():
    argmap = to_argmap(
        dag_debate(), rng=random.Random(0), min_claims=6, max_claims=6, mece_roots=False
    )
    assert argmap.topic.startswith("whether each of these positions is correct")
    assert "rival" not in argmap.topic
    assert "(1) Claim r1." in argmap.topic and "(2) Claim r2." in argmap.topic


@pytest.mark.parametrize(
    ("mece_roots", "note"),
    [(True, "of which exactly one is correct"), (False, "independently of the others")],
)
def test_the_map_header_reads_the_roots_as_the_map_does(mece_roots, note):
    argmap = make_map([("a", "r1", 1), ("b", "r2", -1)], mece_roots=mece_roots)
    prompt = marginal_item(argmap, "a", show_map=True).prompt(False)
    assert note in prompt
    assert ("exactly one" in prompt) == mece_roots


def test_a_stated_topic_is_used_whatever_the_roots():
    debate = dag_debate(topic="Which algorithm?")
    argmap = to_argmap(debate, rng=random.Random(0), min_claims=6, max_claims=6)
    assert argmap.topic == "Which algorithm?"


def test_the_default_frame_is_the_supposition_wording(simple_map):
    pair = simple_map.pairs[0]
    default = conditional_item(simple_map, pair, direction="pos", show_map=False)
    explicit = conditional_item(simple_map, pair, direction="pos", show_map=False, frame="suppose")
    assert DEFAULT_FRAME == "suppose"
    assert default == explicit
    assert "Suppose, for the sake of argument" in default.body


@pytest.mark.parametrize("frame", sorted(CONDITIONAL_FRAMES))
@pytest.mark.parametrize("show_map", [False, True])
def test_every_frame_keeps_the_conditional_invariants(simple_map, frame, show_map):
    for pair in simple_map.pairs:
        pos, neg = (
            conditional_item(simple_map, pair, direction=d, show_map=show_map, frame=frame).prompt(
                False
            )
            for d in ("pos", "neg")
        )
        assert pos == neg.replace("is not the case", "is the case")
        assert pos != neg
        for prompt in (pos, neg):
            assert f"  {simple_map.text(pair.source)}\n" in prompt
            assert prompt.rstrip().endswith(f"  {simple_map.text(pair.target)}\nA: Yes\nB: No")
            assert not {"support", "attack", "against"} & set(prompt.lower().split())


def test_frames_change_the_conditionals_only(simple_map):
    by_frame = {f: items_for(simple_map, show_map=False, frame=f) for f in CONDITIONAL_FRAMES}
    bodies = {f: [i.body for i in items] for f, items in by_frame.items()}
    marginals = {
        f: [i.body for i in items if i.kind == "marginal"] for f, items in by_frame.items()
    }
    assert len({tuple(m) for m in marginals.values()}) == 1
    assert len({tuple(b) for b in bodies.values()}) == len(CONDITIONAL_FRAMES)


def test_unknown_frame_is_rejected(simple_map):
    with pytest.raises(ValueError, match="frame"):
        conditional_item(
            simple_map, simple_map.pairs[0], direction="pos", show_map=False, frame="wonder"
        )


# --- context and question --------------------------------------------------


def test_a_chat_prompt_is_the_context_and_the_question(simple_map):
    for item in items_for(simple_map, show_map=True):
        assert item.body == f"{item.context}\n\n{item.question}"
        assert item.prompt(False) == item.body + "\nA: Yes\nB: No"


# --- the problem a coding debate is about ----------------------------------


def coding_map(**kwargs):
    return to_argmap(coding_debate(**kwargs), rng=random.Random(0), min_claims=6, max_claims=6)


@pytest.mark.parametrize("show_map", [False, True])
def test_the_problem_opens_every_context_once_and_verbatim(show_map):
    argmap = coding_map()
    assert shows_problem(argmap)
    for item in items_for(argmap, show_map=show_map):
        prompt = item.prompt(False)
        assert item.context.startswith("The debate concerns the following programming problem.")
        assert prompt.count(CODING_PROBLEM) == 1
        assert prompt.count(CODING_DESIGN_DECISION) == 1
        assert r"$\min\{x, y\}$" in prompt and "{guarding}" in render_map(argmap)
        end_of_problem = prompt.index(CODING_DESIGN_DECISION)
        assert end_of_problem < prompt.index("Topic under discussion:")
        if show_map:
            assert end_of_problem < prompt.index(render_map(argmap))
        assert CODING_PROBLEM not in item.question


def test_only_the_parts_a_debate_states_are_shown():
    only_problem = marginal_item(coding_map(design_decision=None), "a", show_map=False)
    assert CODING_PROBLEM in only_problem.context
    assert "Design decision:" not in only_problem.context

    only_decision = marginal_item(coding_map(problem=None), "a", show_map=False)
    assert only_decision.context.startswith(f"Design decision: {CODING_DESIGN_DECISION}\n\n")
    assert "programming problem" not in only_decision.context

    blank = coding_map(problem="  \n", design_decision="")
    assert not shows_problem(blank)
    assert marginal_item(blank, "a", show_map=False).context.startswith("Topic under discussion:")


def test_a_map_without_a_problem_is_asked_as_before(simple_map):
    assert not shows_problem(simple_map)
    for item in items_for(simple_map, show_map=False):
        assert item.context.startswith("Topic under discussion:")
