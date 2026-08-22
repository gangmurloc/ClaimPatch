# ClaimPatch seed2028 metadata ablation report

This report compares the original held-out seed2028 run against soft and hard input-only metadata ablations. Gold labels, evaluator code, synthetic seed, model, prompt text, and schema-repair setting are intended to remain unchanged; only the impact-classification prompt input differs.

## Aggregate metrics

| condition | dependency_complete_success | collateral_edit_rate | residual_stale_rate | patch_precision | patch_recall | schema_valid | transaction_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| soft | 1.000 | 0.060 | 0.000 | 0.955 | 1.000 | 1.000 | 1.000 |
| hard | 0.970 | 0.237 | 0.007 | 0.880 | 0.993 | 1.000 | 1.000 |

## Repair intervention summary

| condition | n_patches | repair_count | repair_rate | failure_count | failure_types | repair_log_counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 100 | 23 | 0.230 | 0 | {} | {"operation_aligned_to_impact": 11, "precondition_added": 26} |
| soft | 100 | 24 | 0.240 | 0 | {} | {"operation_aligned_to_impact": 7, "precondition_added": 32} |
| hard | 100 | 24 | 0.240 | 0 | {} | {"operation_aligned_to_impact": 11, "precondition_added": 44} |

## Case-type metrics


### metric_revision_threshold_cross

| condition | n | DCS | collateral | residual_stale | patch_precision | patch_recall | schema_valid | transaction_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| soft | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard | 25.000 | 0.960 | 0.160 | 0.010 | 0.936 | 0.990 | 1.000 | 1.000 |

### metric_revision_threshold_hold

| condition | n | DCS | collateral | residual_stale | patch_precision | patch_recall | schema_valid | transaction_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| soft | 25.000 | 1.000 | 0.240 | 0.000 | 0.820 | 1.000 | 1.000 | 1.000 |
| hard | 25.000 | 0.960 | 0.320 | 0.010 | 0.790 | 0.990 | 1.000 | 1.000 |

### metric_revision_b_multi_parent

| condition | n | DCS | collateral | residual_stale | patch_precision | patch_recall | schema_valid | transaction_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| soft | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard | 25.000 | 0.960 | 0.460 | 0.010 | 0.816 | 0.990 | 1.000 | 1.000 |

### citation_refresh

| condition | n | DCS | collateral | residual_stale | patch_precision | patch_recall | schema_valid | transaction_success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| soft | 25.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard | 25.000 | 1.000 | 0.008 | 0.000 | 0.980 | 1.000 | 1.000 | 1.000 |

## Pre-registered interpretation hooks

- If soft and hard both keep DCS high with low collateral, metadata was not carrying the result.
- If soft degrades, explicit category labels were carrying the result.
- If soft holds but hard degrades, ClaimPatch partly depends on structured adapter metadata.
- Treat `metric_revision_threshold_cross` under hard ablation as the strongest shortcut test.
