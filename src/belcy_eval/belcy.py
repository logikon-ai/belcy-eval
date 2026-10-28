"""The belcy task.

Usage, from the project root::

    export VLLM_API_KEY=... VLLM_BASE_URL=https://your-endpoint/v1

    uv run inspect eval src/belcy_eval/belcy.py \
      --model vllm/Qwen/Qwen3.5-0.8B \
      --max-connections 32 \
      -T n_maps=50

    # the same maps, with the argument map visible in every prompt
    uv run inspect eval src/belcy_eval/belcy.py -T show_map=true ...

One task, parameterised. ``show_map`` decides whether credences are elicited
from a claim in isolation or with the whole map in context; the gap between the
two is itself a result, since it says how much of the relevance scores depend on the
model being shown the structure. The two land in the log under distinct task
names.

``personas`` puts a value profile in front of the elicitation instruction, one
per map (see :mod:`belcy_eval.personas`). Runs with a persona generator carry
its name in the task name; runs without keep the plain one.

``frame`` rewords the conditional question, to test how much the credences
depend on its wording. Runs with a frame other than the default carry it in the
task name.
"""

from __future__ import annotations

from inspect_ai import Task, task

from belcy_eval.elicit import DEFAULT_FRAME, PROMPT_VERSION, validate_frame
from belcy_eval.personas import validate_generator
from belcy_eval.scorer import belcy_scorer
from belcy_eval.solver import belcy_solver


@task
def belcy(
    show_map: bool = False,
    personas: str = "none",
    corpora: str | list[str] | None = None,
    n_maps: int = 50,
    min_claims: int = 12,
    max_claims: int = 20,
    max_path_length: int = 3,
    max_pairs_per_map: int = 30,
    mece_roots: bool = True,
    seed: int = 42,
    debias_order: bool = True,
    max_concurrent: int = 16,
    exclude: str | None = None,
    corpus_revision: str | None = None,
    frame: str = DEFAULT_FRAME,
    system_suffix: str | None = None,
) -> Task:
    """Measure how far a model's credences track the relevance an argument map entails.

    Args:
        show_map: Render the whole argument map in every elicitation prompt.
        personas: Persona generator. ``"none"`` elicits under the bare
            instruction; ``"schwartz"`` gives each map a persona built from
            Schwartz's basic values and two dispositions, the same for every
            item of the map.
        corpora: syncIALO corpus splits to draw maps from, as comma-joined
            ``corpus_id:split[:weight]`` entries, e.g.
            ``synthetic_corpus-001:test:0.7,synthetic_corpus-002:eval:0.3``.
            Weights are relative and default to 1. Unset means
            ``synthetic_corpus-001:eval``.
        n_maps: Number of maps, one sample each, split across ``corpora`` by
            weight.
        min_claims: Smallest map to draw.
        max_claims: Largest map to draw.
        max_path_length: Test claim pairs joined by a directed path of at most
            this many edges; ``1`` tests direct reasons only.
        max_pairs_per_map: Cap on the pairs elicited per map, drawn evenly
            across distances. ``0`` or less means every pair.
        mece_roots: Read a map's several roots as rival positions, exactly one
            of which is true, and also test each reason under one root against
            every rival root. ``false`` treats the roots as independent claims
            and tests directed-path pairs only. Maps with one root are
            unaffected.
        seed: Seeds map sampling, so a run is reproducible.
        debias_order: Ask each item under both option orderings and average,
            which cancels the Yes/No position bias at twice the calls.
        max_concurrent: Cap on in-flight elicitation calls per sample.
        exclude: Path to a file of sample ids, one per line, to drop from the
            drawn maps; ``#`` starts a comment. No replacements are drawn, so
            the remaining maps are the same as without it. Use it to keep maps
            out that some model in a comparison cannot fit in its context.
        corpus_revision: The syncIALO Hub commit to draw maps from (a sha,
            branch or tag). Unset means the latest, which changes when the
            dataset does; pin it for runs that must be comparable later.
            Ignored when ``BELCY_SYNCIALO_DIR`` points at a local snapshot.
        frame: Wording of the conditional question: ``"suppose"`` ("Suppose,
            for the sake of argument, ..."), ``"assume"`` or ``"learn"`` ("You
            have just learned ..."). Marginals are asked the same way under
            every frame.
        system_suffix: A line appended to the elicitation instruction, for a
            model whose only switch for some serving behaviour is its system
            prompt. It changes the prompt, so it is a property of the model
            rather than of the task: every model in a comparison must be given
            the same one, and a run that used it is not comparable to one that
            did not.
    """

    # Imported here so `inspect eval-set` can enumerate tasks without touching
    # the corpus.
    from belcy_eval.datasets import read_id_list, syncialo_dataset

    if max_path_length < 1:
        raise ValueError("max_path_length must be at least 1")
    validate_generator(personas)
    validate_frame(frame)

    dataset = syncialo_dataset(
        corpora=corpora,
        n_maps=n_maps,
        min_claims=min_claims,
        max_claims=max_claims,
        max_path_length=max_path_length,
        max_pairs=max_pairs_per_map,
        mece_roots=mece_roots,
        seed=seed,
        personas=personas,
        exclude_ids=read_id_list(exclude) if exclude else None,
        revision=corpus_revision,
    )

    return Task(
        dataset=dataset,
        solver=belcy_solver(
            show_map=show_map,
            frame=frame,
            debias_order=debias_order,
            max_concurrent=max_concurrent,
            system_suffix=system_suffix,
        ),
        scorer=belcy_scorer(max_path_length=max_path_length),
        name=f"belcy_{'map' if show_map else 'blind'}"
        + ("" if personas == "none" else f"_{personas}")
        + ("" if frame == DEFAULT_FRAME else f"_{frame}"),
        metadata={"prompt_version": PROMPT_VERSION},
    )
