# ClaimPatch P1 Local Qwen Structured Held-Out 100 Seed 2028

- instances: 100
- sequential steps per instance: 1
- total evaluated update steps: 100
- elapsed seconds: 6184.22
- failures: 0

## Main metrics

| system | DCS | patch_precision | patch_recall | collateral_edit_rate | residual_stale_rate | preserve_precision | token_reduction_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1_prompted_local | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.664 |
| unstructured_selective_edit | 0.600 | 0.922 | 0.888 | 0.072 | 0.113 | 0.867 | 0.663 |
| attribute_no_graph | 0.250 | 1.000 | 0.655 | 0.000 | 0.345 | 0.670 | 0.863 |
| descendant_all | 1.000 | 0.755 | 1.000 | 0.260 | 0.000 | 1.000 | 0.521 |
| full_regeneration | 1.000 | 0.512 | 1.000 | 1.000 | 0.000 | 0.000 | -0.001 |

## Hard-case breakdown

| hard_case | system | DCS | patch_precision | collateral_edit_rate | residual_stale_rate |
|---|---|---:|---:|---:|---:|
| citation_refresh | attribute_no_graph | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_b_multi_parent | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.500 |
| metric_revision_threshold_cross | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.500 |
| metric_revision_threshold_hold | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.380 |
| citation_refresh | descendant_all | 1.000 | 0.200 | 0.800 | 0.000 |
| metric_revision_b_multi_parent | descendant_all | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_cross | descendant_all | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_hold | descendant_all | 1.000 | 0.820 | 0.240 | 0.000 |
| citation_refresh | full_regeneration | 1.000 | 0.167 | 1.000 | 0.000 |
| metric_revision_b_multi_parent | full_regeneration | 1.000 | 0.667 | 1.000 | 0.000 |
| metric_revision_threshold_cross | full_regeneration | 1.000 | 0.667 | 1.000 | 0.000 |
| metric_revision_threshold_hold | full_regeneration | 1.000 | 0.547 | 1.000 | 0.000 |
| citation_refresh | p1_prompted_local | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_b_multi_parent | p1_prompted_local | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_cross | p1_prompted_local | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_hold | p1_prompted_local | 1.000 | 1.000 | 0.000 | 0.000 |
| citation_refresh | unstructured_selective_edit | 1.000 | 0.920 | 0.048 | 0.000 |
| metric_revision_b_multi_parent | unstructured_selective_edit | 0.960 | 1.000 | 0.000 | 0.010 |
| metric_revision_threshold_cross | unstructured_selective_edit | 0.320 | 1.000 | 0.000 | 0.170 |
| metric_revision_threshold_hold | unstructured_selective_edit | 0.120 | 0.770 | 0.240 | 0.270 |

## Paired bootstrap

- dependency_complete_success delta, p1_prompted_local-unstructured_selective_edit: 0.400 [0.310, 0.490] (paired n=100, primary n=100, baseline n=100)
- dependency_complete_success delta, p1_prompted_local-attribute_no_graph: 0.750 [0.660, 0.830] (paired n=100, primary n=100, baseline n=100)
- dependency_complete_success delta, p1_prompted_local-descendant_all: 0.000 [0.000, 0.000] (paired n=100, primary n=100, baseline n=100)
- collateral_edit_rate delta, p1_prompted_local-unstructured_selective_edit: -0.072 [-0.100, -0.044] (paired n=100, primary n=100, baseline n=100)
- collateral_edit_rate delta, p1_prompted_local-attribute_no_graph: 0.000 [0.000, 0.000] (paired n=100, primary n=100, baseline n=100)
- collateral_edit_rate delta, p1_prompted_local-descendant_all: -0.260 [-0.330, -0.196] (paired n=100, primary n=100, baseline n=100)
- residual_stale_rate delta, p1_prompted_local-unstructured_selective_edit: -0.113 [-0.138, -0.087] (paired n=100, primary n=100, baseline n=100)
- residual_stale_rate delta, p1_prompted_local-attribute_no_graph: -0.345 [-0.385, -0.302] (paired n=100, primary n=100, baseline n=100)
- residual_stale_rate delta, p1_prompted_local-descendant_all: 0.000 [0.000, 0.000] (paired n=100, primary n=100, baseline n=100)
- patch_precision delta, p1_prompted_local-unstructured_selective_edit: 0.078 [0.047, 0.110] (paired n=100, primary n=100, baseline n=100)
- patch_precision delta, p1_prompted_local-attribute_no_graph: 0.000 [0.000, 0.000] (paired n=100, primary n=100, baseline n=100)
- patch_precision delta, p1_prompted_local-descendant_all: 0.245 [0.181, 0.314] (paired n=100, primary n=100, baseline n=100)
- patch_recall delta, p1_prompted_local-unstructured_selective_edit: 0.112 [0.087, 0.138] (paired n=100, primary n=100, baseline n=100)
- patch_recall delta, p1_prompted_local-attribute_no_graph: 0.345 [0.302, 0.385] (paired n=100, primary n=100, baseline n=100)
- patch_recall delta, p1_prompted_local-descendant_all: 0.000 [0.000, 0.000] (paired n=100, primary n=100, baseline n=100)

## Interpretation

Second fresh held-out synthetic evaluation after seed 2027. Do not modify prompts, repair rules, or model settings before running this seed if treating it as independent held-out evidence.
