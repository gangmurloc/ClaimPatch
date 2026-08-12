# P1 Second-Model Llama 3.1 Borderline Report

Date: 2026-07-25 KST

## Run

- Config:
  `configs/experiments/p1_prompted_local_llama31_borderline100_seed3031_metadata_hard.yaml`
- Script:
  `scripts/run_p1_local_llama31_borderline100_seed3031_metadata_hard.sh`
- Output:
  `outputs/p1_prompted_local_llama31_borderline100_seed3031_metadata_hard/summary.md`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Dataset: borderline profile, seed3031, 100 examples
- Metadata ablation: hard
- Failures: 0
- Elapsed: 6609.45 seconds (~1h 50m)

## Main metrics

| system | DCS | patch_precision | patch_recall | collateral_edit_rate | residual_stale_rate |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B P1 hard-metadata | 0.920 | 0.775 | 0.975 | 0.388 | 0.025 |

## Hard-case breakdown

| hard_case | DCS | patch_precision | collateral_edit_rate | residual_stale_rate |
|---|---:|---:|---:|---:|
| threshold_cross | 0.960 | 0.882 | 0.290 | 0.010 |
| threshold_hold | 0.880 | 0.669 | 0.487 | 0.040 |

## Pre-registered diagnostic

The preregistered target was the `c_comparison` cross/hold asymmetry:

- if the failure mode generalizes, threshold-hold `c_comparison` over-edit should
  remain high, especially relative to threshold-cross misses;
- if Llama is simply incapable under the prompt, both threshold-cross and
  threshold-hold should collapse.

Observed `c_comparison` transitions:

| hard_case | gold -> predicted | count |
|---|---|---:|
| threshold_cross | `MUST_CHANGE -> MUST_CHANGE` | 50/50 |
| threshold_hold | `STILL_VALID -> MUST_CHANGE` | 50/50 |

This is a strong replication under the pre-registered rule. It is not a
model-incapacity/inconclusive case because threshold-cross `c_comparison` is
perfect and threshold-cross DCS remains high at 0.960. The model can detect that
the comparative claim must change when the threshold boundary is crossed, but it
systematically over-edits the same comparative claim when the numeric parent
changes without changing the qualitative comparison.

## Comparison to Qwen2.5-7B

| model | threshold_cross `c_comparison` miss | threshold_hold `c_comparison` over-edit |
|---|---:|---:|
| Qwen2.5-7B-Instruct | 4/50 | 47/50 |
| Llama-3.1-8B-Instruct | 0/50 | 50/50 |

The qualitative result is therefore not Qwen-specific. The second model makes
the asymmetry sharper: it always updates the comparative claim under cross, and
also always updates it under hold.

## Caveats

- This does not make the full DECAP result multi-model. The complete held-out,
  sequential, repair, and metadata suite remains primarily Qwen2.5-7B.
- Llama's aggregate collateral is slightly worse than Qwen's on the hard-metadata
  borderline setting. The second-model result supports the failure-mode insight,
  not a broad claim that DECAP performs equally across model families.
- The run uses the same synthetic borderline domain. It does not solve the
  real-world adapter/external-validity limitation.

## Manuscript claim enabled

Use:

> The invariance-under-change failure is not confined to Qwen2.5-7B. In a
> fixed-prompt second-model diagnostic with Llama-3.1-8B, threshold-cross
> comparative claims were handled in 50/50 cases, while threshold-hold
> comparative claims were over-edited in 50/50 cases.

Avoid:

> DECAP has been fully validated across model families.
