# Reported Qwen run environment

This document separates facts captured with the archived experiment from a
best-effort reconstruction of the local software environment. It does not fill
missing provenance with guessed values.

## Captured by the experiment configuration

| Field | Value |
|---|---|
| Model identifier | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision / commit | **Not recorded** |
| Precision | `bf16` |
| Device mapping | `auto` |
| Quantization | disabled |
| Decoding | deterministic (`do_sample: false`, temperature `0.0`) |
| Maximum new tokens | `1536` |
| Synthetic seed | `13` |
| Instances × sequential updates | `100 × 3` |
| Bootstrap resamples | `1,000` |

The archived `environment.json` records Python 3.8.10 and the Linux platform,
but its model field incorrectly contains the P0 placeholder
`none-rule-based-p0`, and its Git commit is `unavailable`. Those two fields are
therefore not treated as valid model or source-revision provenance.

## Best available local environment reconstruction

The following versions were still installed in the original workspace when the
public artifact was prepared. They are useful compatibility evidence but are
not guaranteed to be an immutable capture of the exact run environment.

| Component | Version |
|---|---|
| Python | 3.8.10 |
| PyTorch | 2.4.0 + CUDA 12.1 build |
| Transformers | 4.46.3 |
| Sentence Transformers | 3.2.1 |
| Pydantic | 2.10.6 |
| NetworkX | 3.1 |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| GPU | NVIDIA RTX A5000, 24 GiB |
| NVIDIA driver | 535.230.02 |

The public package targets Python 3.10+ and its CI tests Python 3.10 and 3.11.
That support target should not be confused with the archived experiment's
Python version.
