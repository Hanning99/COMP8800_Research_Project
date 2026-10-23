# One Solve, Every Step: Accelerating Actor–Critic Learning

COMP8800 research artefact — Hanning Deng, Australian National University, 2026.

This repository contains the Policy Differential Revaluation (PDR) implementations, recorded configurations, evaluation commands and results for **four benchmark reproduction packages**. The complete checkpoint/data packages are hosted separately:

- Code: https://github.com/Hanning99/COMP8800_Research_Project
- Checkpoints: https://huggingface.co/datasets/denghanning/COMP8800_Research_Project

Both repositories remain **private** for author/supervisor review. The dataset revision is pinned in `configs/release.json` after upload. The publishing script prints an immutable GitHub commit link to submit to Wattle. An unset revision means checkpoint publication is pending.

## Scope

| Benchmark | Retained comparison | Compressed package |
|---|---|---:|
| KeyCorridor | 640 runs; four tasks, eight methods, 20 seeds per setting | 6.56 GB |
| MiniGridLoCA | 480 runs; AWAC/SAC/QAC branches, eight methods, 20 paired seeds | 8.05 GB |
| DeepSea | 520 runs; N=10/20/30/40, eight methods, 20/19/13/13 eligible seeds | 3.51 GB |
| SecureCourier | 50 paired seeds; Standard TD and PDR, all retained score/checkpoint grids through cycle 2,048 | 10.63 GB |

Together the packages preserve 71,343 file paths and approximately **28.75 GB** of deduplicated compressed content. They include saved starts, final/intermediate states actually retained by each release, reference tables, optimizers/RNG states, replay inputs, configs, raw curves/scores and native scripts. Unreported training histories and checkpoint collections for other analysis/ablation studies are outside this release.

The report includes additional analysis. This artefact's checkpoint-availability claim is restricted to these four packages; it does not claim that every analysis in the report has a downloadable checkpoint or that all training was repeated.

## Install

Use Python 3.10. Original training used PyTorch 2.5.1+cu121. CPU suffices for checkpoint evaluation; training continuations require the release-specific CUDA environment and, where applicable, C++17 kernels.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-publish.txt
```

Per-benchmark dependency/build details are in `src/releases/<benchmark>/README.md`.

## Download the pinned checkpoints

Authenticate for the private dataset using `HF_TOKEN` or your Hugging Face login. Obtain access to both private repositories before evaluation.

```bash
python scripts/download_checkpoints.py --out checkpoints/local
```

This refuses an unspecified revision. Before publication, an already prepared local archive may be used with `--archive /path/to/archive`.

## Evaluate saved checkpoints

```bash
bash scripts/evaluate.sh --benchmark DeepSea --archive checkpoints/local
bash scripts/evaluate.sh --benchmark MiniGridLoCA --archive checkpoints/local
bash scripts/evaluate.sh --benchmark KeyCorridor --archive checkpoints/local
bash scripts/evaluate.sh --benchmark SecureCourier --archive checkpoints/local
```

These commands restore normal files into a new workspace and load saved actors. The first three compare exact graph metrics against archived curves; SecureCourier calls the original checkpoint rescoring script. They do not train. To reuse a restored release, pass `--data-root /path/to/restored/AAMAS-Experiment/Effectiveness/<benchmark>`.

For a quick KeyCorridor check, add `--representative` (one seed per task/method). A quick SecureCourier check can append `-- --events confirmatory_000 confirmatory_025 confirmatory_049 --cycles 0 1024 2048 --workers 2`. Subset checks are reported as subsets.

## Rebuild results and replay training

```bash
python scripts/run.py results --benchmark DeepSea --archive checkpoints/local
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce.sh --benchmark DeepSea \
  --archive checkpoints/local -- --size 10 --method PDR --updates 8192
```

The results command reconstructs figures/statistics from recorded data. The replay command resumes the supplied initial state, using original schedules and recipes, and checks recorded metrics. These are **training continuations from supplied checkpoints**, not from-scratch pretraining. Native commands and method-specific flags for all four benchmarks are documented in their release READMEs. Outputs always use a new directory.

## Validation evidence

Fresh packaging checks include all 520 DeepSea and 480 MiniGridLoCA final actors, 32 KeyCorridor representatives spanning four tasks/eight methods, and 10,560 input/path/hash checks over all 1,640 run records in those three comparisons. Reports and their exact scopes are in `evaluation/results/`. Original verification reports are labelled historical; their `PASS` does not imply retraining during this delivery. Whole-archive byte checks are reported separately from model evaluation.

## Structure

```text
README.md
requirements.txt
configs/                 # Exact run recipes, source hashes and pinned revisions
src/releases/            # Four benchmark implementations and native scripts
scripts/
  evaluate.sh
  reproduce.sh
  checkpoint_archive.py
  download_checkpoints.py
  publish_tonight.py
evaluation/results/      # Recorded summaries/figures and scoped validation evidence
checkpoints/README.md    # Dataset access, storage format and integrity
docs/                    # Report, method map, reproducibility and handover
```

See [reproducibility](docs/reproducibility.md), [implementation map](docs/method_map.md), and [submission](docs/submission.md). Original source notices and experimental qualifications remain with each release.
