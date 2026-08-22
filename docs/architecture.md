# Architecture

ClaimPatch stores an answer as a versioned set of claims, evidence bindings, and
typed dependency hyperedges. An evidence update is processed in five stages.

1. **Impact detection** identifies claims directly affected by the update.
2. **Dependency propagation** computes downstream candidates over the claim
   graph.
3. **Semantic revalidation** distinguishes downstream claims that truly need
   revision from claims whose conclusion remains invariant.
4. **Patch construction** emits typed operations with explicit preconditions,
   preservation targets, and dependency updates.
5. **Transactional execution** applies the patch to a copy and commits only if
   version, status, evidence, and graph checks pass.

The executor supports `REPLACE`, `DELETE`, `INSERT`, `SPLIT`, `MERGE`,
`REBIND`, and `INVALIDATE`. Failed preconditions or graph validation raise a
`PatchExecutionError`; the input answer remains unchanged.

The P0 path is deterministic. The P1 path uses the same schemas and executor
but obtains dependency, impact, and patch payloads through versioned prompts.
Deterministic schema repair addresses formatting and executability errors. It
must not be interpreted as a semantic oracle.

