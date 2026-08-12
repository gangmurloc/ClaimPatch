# Evaluation

The synthetic benchmark contains four update families: threshold-hold,
threshold-cross, multi-parent metric revision, and citation-only refresh.
Sequential experiments apply multiple evidence updates to the same answer.

Key metrics:

- **Dependency-complete success (DCS):** all required claim changes are made.
- **Patch precision/recall:** agreement between touched claims and gold
  must-change claims.
- **Collateral edit rate:** preserved claims that were changed unnecessarily.
- **Residual stale rate:** required changes that remain unapplied.
- **Preservation precision/recall:** accuracy of explicit preserve decisions.
- **Citation orphan rate:** active claims pointing to unavailable evidence.
- **Transaction success:** patch execution and post-application validation.

Paired bootstrap intervals are computed over aligned update steps. Aggregate
metrics must be read together: descendant-all can achieve perfect DCS while
incurring unnecessary edits, and attribute-only editing can avoid collateral
changes while leaving downstream claims stale.

