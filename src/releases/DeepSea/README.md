# DeepSea exact-reference reproducibility release

This release contains the **520 plotted continuations** in the DeepSea N=10/20/30/40 main comparison: PDR-SAC and TD, PER, DisCor, 8× TD, IQL, CQL, PS + actor. The paired seed counts are **20 / 19 / 13 / 13**. Every continuation runs for 8,192 actor updates. Original experiments are unchanged; this directory is independent of their absolute paths.

## Exact reference versus live critic

`references/N*/seed_*/EXACT_REFERENCE.pt` supplies the fixed **exact policy-value table** for each initial policy, including `policy`, `soft_qold`, the non-entropy `qold`, entropy coefficient, and source checksums. It is a tabular checkpoint, **not a neural state_dict**. The original `REFERENCE.npz` is also retained. PDR uses the exact soft reference (entropy coefficient 0.001) and computes the current-policy soft value change on the closed transition graph. The replay script loads the supplied table after checking it against direct policy evaluation. There is no reference-fitting stage to rerun.

The ongoing PDR/CQL critics remain **neural twin critics**, trained with soft TD plus the recorded CQL penalty. Their complete models, targets, optimizers and RNG states are supplied, as are those of every baseline. PDR uses the SAC actor objective with `Q_actor = Q_old + 32 Delta_Q`; its detached value input is recomputed at every actor step. Archived cell names ending in `exact_AWAC` describe the inherited initial branch, **not** the PDR actor objective. TD/PER/DisCor/8×TD/PS use the archived AWAC objective; IQL retains its native actor update.

## Contents

- `RUNS.json`: authoritative package-relative mapping for every plotted seed, method, recipe, checkpoint and raw curve; original paths are provenance only.
- `runs/`: all files and all checkpoints actually saved in each selected run directory, including `INITIAL.pt`, `FINAL.pt`, raw curves, saved policies, diagnostics and configs. The original runs retained no separate intermediate checkpoint after completing; this release does not invent one.
- `starts/`: 65 original common initialization states, including model/optimizer/RNG states.
- `inputs/deepsea/worlds/`: original `COMMON_START.pt`, closed graph, frozen replay schedule and both replay tables. Common starts contain actor, neural critic, targets, optimizers, initial policy and exact non-entropy value table. **Anchor training is not required** to reproduce continuations; saved starts are provided instead.
- `references/`: 65 exact-reference checkpoints described above.
- `src/`, `vendor/`, `scripts/`: numerical code, bsuite-based environment/evaluator, C++ solver sources, portable reproduction and plotting entry points. Compile local kernels with the original flags before training.
- `provenance/`: original code/protocols, frozen analysis, input/source checksums, initialization-eligibility records, and a portability diff. N40's existing device-selection wrapper is reused for all sizes; its GPU numerical updates are unchanged. Machine-specific Python search paths are removed.
- `results/`: regenerated CSVs and the early/full figures in PDF/PNG/SVG. The combined PDF has two pages. Editing the plotting function in `scripts/aggregate.py` changes presentation without rerunning training.
- `validation/`, `DELIVERY_VERIFICATION.json`, `SHA256SUMS`: actual checks performed for this delivery. See these reports for the precise rerun scope, rather than assuming every trajectory was retrained.

Identical files may be hardlinked **within this release only**; there are no external symlinks or hardlinks to source archives. Unreported tuning-candidate checkpoints and anchor-training histories are outside scope; their selection protocols are retained. Do not use the historical workers under `provenance/` as portable entry points.

## Environment

Validated with Python 3.10 and PyTorch 2.5.1+cu121 on NVIDIA RTX 3090. CPU suffices for aggregation and integrity/reference/environment verification; the replay entry point requires exactly one visible CUDA GPU. Verify the delivered checksums **before** rebuilding (which rewrites the local build log), then install dependencies in a fresh virtual environment:

```sh
sha256sum -c SHA256SUMS
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python scripts/build.py
```

A C++17 compiler is required. Original `kernels.cpp` and `planning.cpp` build with `-O3 -std=c++17 -ffp-contract=off -shared -fPIC`. bsuite 0.3.6 supplies the original DeepSea environment. Checkpoints require `weights_only=False` because they include optimizer/RNG state; load only trusted checkpoints.

## Regenerate the plotted statistics and figures

Run from the release root; output directories must be new:

```sh
python scripts/verify.py --out validation/local_integrity.json
python scripts/audit_policies.py --out validation/local_final_actors.json
python scripts/aggregate.py --out reproduced_results
```

The actor audit reconstructs all 520 final actors from checkpoint on CPU and compares their policies and returns with the archived GPU results (tolerance `1e-5` for CPU/GPU rounding). Aggregation reconstructs every mean, pointwise bootstrap interval and early/final summary from raw curves, and fails if it differs from the archived data by more than `1e-12`. The figure metric is **expected episode return**, not success rate. Early is trapezoidal AUC over updates 0–2,048 divided by 2,048; Final is return at update 8,192. The release additionally reports full-window mean return. Bands use 10,000 paired seed-bootstrap draws with RNG seed 20260925; seed ordering is preserved. No smoothing or outcome-based seed filtering is applied.

## Replay from the supplied checkpoints

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --size 10 --method PDR --updates 8192 --out reruns/n10_pdr
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --size 40 --method CQL --updates 8192 --out reruns/n40_cql
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --size 20 --method PER --updates 16 --out reruns/n20_per_smoke
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --size 30 --method PDR --mode final --out reruns/n30_final
```

Methods: `TD PER DISCOR EXTRA_TD IQL CQL PS PDR` (`OURS` is an alias for PDR). Choose a listed seed with `--seed`; default is the first archived seed for that setting. To retrain all plotted trajectories, iterate over every record in `RUNS.json`. A run restores complete method state, checks shared initialization bitwise and the replay schedule hash, then compares metrics **and policy tables** against the archive at each recorded evaluation point. Default cross-stack numerical tolerance is `1e-5`; mismatches fail visibly. Outputs never overwrite original checkpoints.

Resume a checkpoint generated by the portable runner:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce.py --size 20 --method PER --resume reruns/n20_per_smoke/REPRODUCED.pt --updates 32 --out reruns/n20_per_resume
```

All operational paths are package-relative, so the complete directory can be moved. Historical paths in original configs/checkpoints are audit metadata and are not followed by the portable runner.

## Protocol and interpretation

These are **fixed-replay continuations from supplied initial checkpoints**, not from-scratch online exploration comparisons. Each batch contains 16 historical and 48 support-completion transitions; methods share starting actor/critic states and fixed actor-batch schedules. Full-state inputs and closed transition support are supplied. The environment is deterministic bsuite DeepSea with randomized action mappings, gamma=1 and the archived movement cost. PDR/CQL use actor/critic learning rates 0.0003, batch size 64 and target coefficient 0.005. The CQL penalty is 0.01 for N10, 0.1 otherwise. Other methods' gains and recipes are preserved per run, not silently standardized.

For N10–30, configurations were selected on earlier development data; the plotted continuations use the eligible predeclared validation cohort (20/19/13 out of 20 attempts each). Eight initialization failures were not replaced. N40 transfers the N30 configuration and pools **all 13 eligible starts from 100 prespecified attempts across two GPU batches**; 87 unavailable starts are retained in the eligibility records. It does not pool the earlier CPU pilot. Thus the continuation results are conditional on meeting the initialization protocol, not an unconditional result over all attempted starts. See the frozen protocols and `provenance/ALL_N40_ATTEMPTS.csv`.

Archived per-run training timings came from shared GPUs and are not a matched equal-time comparison. This release does not turn them into computational-cost evidence.
