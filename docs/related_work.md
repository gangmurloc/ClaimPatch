# Related work and positioning

DECAP combines known ingredients—claim decomposition, evidence attribution,
dependency graphs, and structured patches—but studies a narrower maintenance
problem: **given an already generated answer and an evidence update, which
direct and downstream claims must change, and which dependent claims should be
preserved?** The contribution should not be described as the first claim graph,
the first factual revision system, or a solution to stale knowledge in LLMs.

## Post-hoc factual revision

[RARR](https://aclanthology.org/2023.acl-long.910/) retrieves attribution for a
generated output and revises unsupported content while preserving the original
text where possible. It is DECAP's closest established revision neighbor.
DECAP differs in the maintained artifact and evaluation target: it processes an
explicit evidence delta over a versioned claim graph, propagates impact through
typed dependencies, applies executable patch operations, and measures both
residual staleness and collateral edits.

## Attribution, factuality, and claim–evidence interfaces

[AIS](https://aclanthology.org/2023.cl-4.2/),
[ALCE](https://aclanthology.org/2023.emnlp-main.398/), and
[FActScore](https://aclanthology.org/2023.emnlp-main.741/) establish the value
of statement-level attribution, citation-grounded generation, and atomic-fact
evaluation. [Provenance](https://aclanthology.org/2024.emnlp-industry.97/) and
[Claimify](https://aclanthology.org/2025.acl-long.348/) provide adjacent
fact-checking and claim-extraction capabilities. These methods mostly evaluate
or produce an answer under a fixed evidence set. DECAP instead studies answer
maintenance after that set changes.

[PaperTrail](https://arxiv.org/abs/2602.21045) is a particularly close
structural neighbor: it decomposes scholarly answers and sources into claims
and evidence for human inspection. PaperTrail is primarily a provenance
interface; DECAP is an automated update mechanism with dependency propagation,
typed patch execution, and update-specific preservation metrics.

## Model and memory editing

[MEND](https://arxiv.org/abs/2110.11309),
[ROME](https://arxiv.org/abs/2202.05262), and
[MEMIT](https://arxiv.org/abs/2210.07229) modify model behavior or internal
associations. DECAP leaves model parameters unchanged and updates a concrete
answer artifact together with its claim, evidence, and dependency records.

Recent stale-memory work such as
[STALE](https://arxiv.org/abs/2605.06527) and
[Supersede](https://arxiv.org/abs/2606.27472) asks whether LLM agents recognize
superseded information and update memory correctly. DECAP is complementary: it
focuses on dependency-complete, minimal revision of one generated answer rather
than general agent memory.

## Structured editing and DECAP's distinction

Executable structured edits and transactional validation are established ideas
in program repair and data systems. DECAP applies them to evidence-grounded
natural-language claim graphs. Its distinctive combination is:

1. an explicit evidence-update event rather than a generic unsupported-output
   signal;
2. typed direct and downstream claim dependencies;
3. semantic revalidation inside the dependency closure;
4. executable answer-level patch operations with preconditions; and
5. joint measurement of dependency coverage, residual stale claims, preserved
   claims, citation bindings, and collateral edits.

This positioning remains provisional because the current evidence is synthetic.
A natural-text pilot is the most important next step for testing whether the
same maintenance problem and trade-off persist outside the constructed graph
benchmark.
