# Research status

ClaimPatch is a research prototype, not a production updater.

## Supported by the included evidence

- Dependency-aware structured editing outperforms an unstructured selective
  editor on DCS in the included 300-step synthetic Qwen diagnostic.
- The transactional executor preserves input state when a patch fails its
  preconditions or graph checks.
- Metadata ablation shows that invariance-under-change is harder than detecting
  that a value changed.
- A fixed-prompt Llama diagnostic reproduces the threshold-hold over-edit
  failure mode.

## Not established

- Robust performance on naturally occurring document revisions.
- Reliable claim decomposition or graph extraction from unrestricted prose.
- Generalization across domains, languages, or broad model families.
- Token-level or monetary cost advantages in a production serving stack.

The clean held-out seed-2028 result is included for transparency, but perfect
scores on a small synthetic split should not be read as proof of external
validity. The larger sequential diagnostic is the headline artifact because it
exercises repeated updates and exposes both residual staleness and collateral
editing.

