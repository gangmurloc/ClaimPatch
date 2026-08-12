# DECAP P1 Local Qwen Structured Full 100x3

- instances: 100
- sequential steps per instance: 3
- total evaluated update steps: 300
- elapsed seconds: 18024.98
- failures: 0

## Main metrics

| system | DCS | patch_precision | patch_recall | collateral_edit_rate | residual_stale_rate | preserve_precision | token_reduction_vs_full |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1_prompted_local | 0.820 | 0.835 | 0.926 | 0.183 | 0.074 | 0.919 | 0.613 |
| unstructured_selective_edit | 0.500 | 0.780 | 0.807 | 0.223 | 0.193 | 0.797 | 0.627 |
| attribute_no_graph | 0.250 | 1.000 | 0.662 | 0.000 | 0.338 | 0.680 | 0.863 |
| descendant_all | 1.000 | 0.745 | 1.000 | 0.273 | 0.000 | 1.000 | 0.521 |
| full_regeneration | 1.000 | 0.505 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Hard-case breakdown

| hard_case | system | DCS | patch_precision | collateral_edit_rate | residual_stale_rate |
|---|---|---:|---:|---:|---:|
| citation_refresh | attribute_no_graph | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_b_multi_parent | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.500 |
| metric_revision_threshold_cross | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.500 |
| metric_revision_threshold_hold | attribute_no_graph | 0.000 | 1.000 | 0.000 | 0.353 |
| citation_refresh | descendant_all | 1.000 | 0.200 | 0.800 | 0.000 |
| metric_revision_b_multi_parent | descendant_all | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_cross | descendant_all | 1.000 | 1.000 | 0.000 | 0.000 |
| metric_revision_threshold_hold | descendant_all | 1.000 | 0.780 | 0.293 | 0.000 |
| citation_refresh | full_regeneration | 1.000 | 0.167 | 1.000 | 0.000 |
| metric_revision_b_multi_parent | full_regeneration | 1.000 | 0.667 | 1.000 | 0.000 |
| metric_revision_threshold_cross | full_regeneration | 1.000 | 0.667 | 1.000 | 0.000 |
| metric_revision_threshold_hold | full_regeneration | 1.000 | 0.520 | 1.000 | 0.000 |
| citation_refresh | p1_prompted_local | 0.907 | 0.589 | 0.291 | 0.093 |
| metric_revision_b_multi_parent | p1_prompted_local | 0.600 | 0.836 | 0.333 | 0.133 |
| metric_revision_threshold_cross | p1_prompted_local | 0.920 | 1.000 | 0.000 | 0.020 |
| metric_revision_threshold_hold | p1_prompted_local | 0.853 | 0.916 | 0.107 | 0.048 |
| citation_refresh | unstructured_selective_edit | 0.787 | 0.559 | 0.264 | 0.213 |
| metric_revision_b_multi_parent | unstructured_selective_edit | 0.387 | 0.808 | 0.333 | 0.237 |
| metric_revision_threshold_cross | unstructured_selective_edit | 0.213 | 1.000 | 0.000 | 0.197 |
| metric_revision_threshold_hold | unstructured_selective_edit | 0.613 | 0.753 | 0.293 | 0.127 |

## Paired bootstrap

- dependency_complete_success delta, p1_prompted_local-unstructured_selective_edit: 0.320 [0.257, 0.380] (paired n=300, primary n=300, baseline n=300)
- dependency_complete_success delta, p1_prompted_local-attribute_no_graph: 0.570 [0.510, 0.630] (paired n=300, primary n=300, baseline n=300)
- dependency_complete_success delta, p1_prompted_local-descendant_all: -0.180 [-0.227, -0.140] (paired n=300, primary n=300, baseline n=300)
- collateral_edit_rate delta, p1_prompted_local-unstructured_selective_edit: -0.040 [-0.062, -0.019] (paired n=300, primary n=300, baseline n=300)
- collateral_edit_rate delta, p1_prompted_local-attribute_no_graph: 0.183 [0.155, 0.210] (paired n=300, primary n=300, baseline n=300)
- collateral_edit_rate delta, p1_prompted_local-descendant_all: -0.091 [-0.132, -0.049] (paired n=300, primary n=300, baseline n=300)
- residual_stale_rate delta, p1_prompted_local-unstructured_selective_edit: -0.120 [-0.151, -0.087] (paired n=300, primary n=300, baseline n=300)
- residual_stale_rate delta, p1_prompted_local-attribute_no_graph: -0.265 [-0.295, -0.231] (paired n=300, primary n=300, baseline n=300)
- residual_stale_rate delta, p1_prompted_local-descendant_all: 0.074 [0.054, 0.096] (paired n=300, primary n=300, baseline n=300)
- patch_precision delta, p1_prompted_local-unstructured_selective_edit: 0.055 [0.026, 0.083] (paired n=300, primary n=300, baseline n=300)
- patch_precision delta, p1_prompted_local-attribute_no_graph: -0.165 [-0.195, -0.137] (paired n=300, primary n=300, baseline n=300)
- patch_precision delta, p1_prompted_local-descendant_all: 0.090 [0.056, 0.124] (paired n=300, primary n=300, baseline n=300)
- patch_recall delta, p1_prompted_local-unstructured_selective_edit: 0.120 [0.087, 0.151] (paired n=300, primary n=300, baseline n=300)
- patch_recall delta, p1_prompted_local-attribute_no_graph: 0.265 [0.231, 0.295] (paired n=300, primary n=300, baseline n=300)
- patch_recall delta, p1_prompted_local-descendant_all: -0.074 [-0.096, -0.054] (paired n=300, primary n=300, baseline n=300)

## Interpretation

Large sequential local-transformers diagnostic. Run only after 20/100 smoke and held-out checks pass.
