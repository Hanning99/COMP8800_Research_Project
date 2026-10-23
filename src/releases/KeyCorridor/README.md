# KeyCorridor neural-reference reproducibility release

This is the **neural main experiment only**: S3R2, S3R3, S4R3 and S5R3; PDR and TD, PER, DisCor, 8× TD, IQL, CQL, PS + actor; 20 paired seeds per task/method, **640 complete 512-update curves**. The frozen source is recorded in `PACKAGE_SCOPE.json`. No exact-reference run or checkpoint is included.

## What is included

- `RUNS.json`: authoritative mapping from every plotted curve to its selected configuration, initial/early/final checkpoint, original source and SHA256. All paths used for reproduction are package-relative.
- `runs/`: every file and every checkpoint retained in the 640 selected run directories, not reduced actor-only snapshots. These include target networks, optimizer and RNG states, and algorithm-specific replay/planning states.
- `starts/`: shared starting checkpoints and original initialization checks, including common/native starts where present.
- `inputs/`: closed transition graphs, full-state features, fixed action masks, batch schedules, initial actor checkpoints, neural-reference `ANCHOR.pt` and the unchanged cached `NEURAL_QOLD.npy` used by training. Neural-reference **pretraining is not required or included**. Keeping the original cached critic outputs avoids silently changing the archived continuation through rematerialization rounding.
- `src/`, `vendor/`: archived numerical learner, evaluator and network implementations. Only two machine-specific Python search paths in `common.py` were removed; see `provenance/PORTABILITY.patch`. Updates and objectives are unchanged.
- `provenance/original_code/`: original source snapshots, training workers, configuration grids and selection metadata. These historical workers retain original paths for audit; use `scripts/reproduce.py` for portable execution.
- `provenance/frozen_analysis/`: neural rows of the source analysis at packaging time. Some historical main tables predate the most recent CQL selection; use regenerated `results/SUMMARY.csv` and `results/CONFIGURATIONS.csv` as the canonical current tables.
- `results/`: regenerated per-seed CSV, curve statistics, summaries, paired comparisons, and early/full four-panel neural figures. Plot styling is confined to `scripts/aggregate.py`.
- `validation/`: integrity, CSV/statistics reconstruction, and actual replay checks performed for this delivery. A saved checkpoint and archived verification claim alone are not a claim that every training run has been rerun.

The release stores independent copies, with optional hardlinks **only between files within the release**. It has no external symlink dependencies. Original experiments are untouched. Unreported tuning-candidate checkpoints are outside scope; selection records and grids are retained. Some historical methods saved only initial/final checkpoints, whereas others also saved update 128; the package includes all that actually existed and does not invent missing intermediate checkpoints.

## Environment

The validated training stack is Python 3.10, PyTorch 2.5.1+cu121 and the versions in `requirements.txt`; CUDA-capable NVIDIA GPU required for training/replay. Original implementation uses CUDA graphs and cannot run unmodified on CPU. CPU is sufficient for CSV aggregation and checkpoint-content verification. Use one visible GPU per process, preferably a 24 GB card for S5R3. Install PyTorch from its cu121 distribution, then the requirements. A C++17 compiler is needed for PS:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/build.py
```

All commands below work after moving the whole release to a different location. Output directories must be new; scripts refuse to overwrite existing runs. Load PyTorch checkpoints only from trusted sources (`weights_only=False` is needed for optimizer/RNG state).

## Reproduce the plotted data and figures (CPU)

```sh
sha256sum -c SHA256SUMS
python scripts/verify.py --out validation/local_integrity.json
python scripts/aggregate.py --out reproduced_results
```

The aggregation command reads each local raw curve, recomputes success means and all pointwise bootstrap intervals, checks them against the frozen original graph data to absolute tolerance `1e-12`, and only then draws the figures. Early = trapezoidal mean success over updates 0–128; Full = mean over 0–512; Final = success at update 512. Confidence bands are percentile seed bootstrap intervals, 10,000 draws, RNG seed 20260925. Training-axis unit is actor updates. No smoothing or seed filtering is performed.

## Re-run a selected continuation from its complete initial state (GPU)

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --task S3R2 --method PDR --updates 512 --out reruns/s3r2_pdr
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --task S3R2 --method CQL --updates 512 --out reruns/s3r2_cql
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --task S5R3 --method PS --updates 16 --out reruns/s5r3_ps_smoke
```

Choose any listed seed with `--seed INTEGER`; by default the first archived seed is used. Methods are `TD PER DISCOR EXTRA_TD IQL CQL PS PDR`. To reproduce every plotted training trajectory, run the selected configuration for every entry of `RUNS.json`, each in a fresh process/output directory. The runner checks initial models, optimizers and RNG against the archived common start, verifies batch schedules, and compares evaluated metrics against the archived CSV at every recorded checkpoint. Default tolerance is `1e-5`, acknowledging numerical differences across GPU/software stacks; discrepancies fail rather than being hidden.

To check the neural reference directly against its checkpoint, or regenerate its full cache into a new directory:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reference.py --task S3R2 --out reference_check
CUDA_VISIBLE_DEVICES=0 python scripts/reference.py --task S3R2 --materialize --out regenerated_reference
```

The default checks every state at five time layers. `--materialize` evaluates all time layers; the archived cache is never overwritten. Checkpoint/cache agreement does not imply accurate policy evaluation.

```sh
# Evaluate a complete final checkpoint without training:
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --task S4R3 --method PDR --mode final --out reruns/s4r3_final
# Resume from update 128 where an EARLY checkpoint exists:
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --task S3R2 --method PDR --mode resume --updates 512 --out reruns/s3r2_resume
```

## Protocol and interpretation

These are matched **fixed-replay continuation** experiments, not from-scratch online benchmark leaderboards. Learners use full-state features and the archived task-semantic action mask; all required successor support is supplied. Actor/critic networks are defined in `vendor/run_replication.py`. PDR uses AWAC with temperature 0.3, cap 20 and fully detached weights; critic learning is one-step TD. Baselines retain their native objectives. Batch size 64, discount 0.97, and target-update coefficient 0.005 are unchanged. PDR actor/critic learning rates are 0.0003. CQL can have separately tuned rates: the exact settings are in `results/CONFIGURATIONS.csv` and each run record.

The currently reported PDR coefficients are S3R2 `(0,6)` (change-only), S3R3 `(16,32)`, S4R3 `(1.25,3)`, S5R3 `(2,32)`. S3R2/S4R3 and all current CQL configurations were selected using full-horizon development performance; other selections retain their recorded early-window protocol. The same cohort informed tuning and reporting, so intervals describe seed variability, not selection-corrected or held-out evidence. Early and full figures use the **same selected trajectory** for each seed. This release reproduces the supplied selection, not a new hyperparameter search or a claim of global optimality.

S5R3 uses the supplied learned-reference checkpoint at step 166400. Its original normalized-Q-MAE acceptance threshold was not met; the checkpoint was subsequently accepted as a known approximation without exact-value repair. Original diagnostics/acceptance metadata are preserved in `provenance/reference_diagnostics/`. Numerical replay agreement is not evidence that this reference equals the true policy value.
