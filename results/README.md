# Included results

This directory contains compact aggregate artifacts only.

- `p1_qwen_sequential_100x3/`: main 300-step synthetic Qwen diagnostic,
  including metrics and paired bootstrap output.
- `diagnostics/qwen_heldout_seed2028.md`: fresh 100-instance held-out check.
- `diagnostics/metadata_ablation_seed2028.md`: structured-input dependence.
- `diagnostics/llama31_borderline.md`: second-model failure-mode diagnostic.

Raw prompts, generations, updated answers, model weights, and exploratory logs
are omitted to keep the release small and auditable. Results are reported as
synthetic diagnostics rather than external benchmark claims.

