# SecureCourier: PDR versus Standard TD

This release supports the SecureCourier comparison in paper Section 5.1
(Effectiveness / EQ1): shortcut choice at the junction and expected discounted
return from the depot, over learner cycles 0–1,024. It contains all 50 paired
seeds, including all retained checkpoint evaluations through cycle 2,048.

The release is self-contained. Historical absolute paths in `legacy/` and
`provenance/` are retained as provenance only. Use the entrypoints in `scripts/`;
they resolve runtime dependencies inside this directory. No external research
directory, environment download, API key or network connection is needed after
installing Python dependencies. Nothing here publishes data to an external site.

## Install

Use Python 3.10 (original version: 3.10.13):

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The recorded runs used PyTorch 2.5.1+cu121. CSV export, plotting and checkpoint
evaluation work on CPU. GPU retraining needs a compatible PyTorch/CUDA setup.
Checkpoint and vocabulary files use PyTorch/pickle serialization; load the
hash-verified files bundled here, not arbitrary downloaded replacements.

## 1. Reproduce the figure and reported statistics

Run all commands from this directory:

```bash
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/export_csv.py
.venv/bin/python scripts/plot.py
```

The CSV export reads the original per-checkpoint score records, validates their
16-instance averages, and writes `data/per_seed.csv`. It then **reads that CSV**
to calculate the means, confidence intervals, normalized AUCs and paired effects.
All plotted means and intervals are checked against the retained thesis summaries
to absolute tolerance 1e-12. No seed selection, smoothing or new learning occurs.

`scripts/plot.py` reads only `data/curve_summary.csv`. It does not load JSON,
checkpoints or research source trees. Change the plot style there, or use the CSV
with another plotting tool. Outputs are vector PDF, editable SVG, and 300-dpi PNG:

- `figures/securecourier_effectiveness.pdf`
- `figures/securecourier_effectiveness.svg`
- `figures/securecourier_effectiveness.png`

The left panel is the shortcut-choice probability at the canonical junction at
episode step 8. The right panel is the exact stochastic-policy expected discounted
return from the depot. Both use the complete 0,16,…,1024 evaluation grid.

## 2. Recompute results directly from checkpoints

```bash
# All 6,500 seed/arm/checkpoint observations used for the early figure.
.venv/bin/python scripts/rescore.py --workers 4
# All 12,900 observations on the regular grid through cycle 2,048.
.venv/bin/python scripts/rescore.py --end 2048 --workers 4 --output outputs/rescored_full.csv
# Original unfactorized scoring path, independently checking sample checkpoints.
.venv/bin/python scripts/rescore.py --native --events confirmatory_000 confirmatory_025 confirmatory_049 --cycles 0 1024 2048 --workers 2 --output outputs/native.csv
```

Every loaded checkpoint is hashed against its original score record. The default
evaluator uses the archived actor's algebraic factorization (the observation-only
score cancels in softmax) and exact finite-horizon dynamic programming. `--native`
uses the original unfactorized actor and original task evaluator. Both compare
the new probabilities and returns with the archived observations, failing above
absolute error 2e-6. Small differences reflect float32 evaluation across devices
and algebraically equivalent forward passes. See `validation/` for actual errors.
The freshly calculated values and individual errors are saved to CSV.

## 3. Re-run the learning continuation

```bash
.venv/bin/python scripts/check_method.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --event confirmatory_000 --arm STANDARD_TD --cycles 2048 --device cuda:0 --output outputs/rerun_000_td
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --event confirmatory_000 --arm FRESH_PDR --cycles 2048 --device cuda:0 --output outputs/rerun_000_pdr
.venv/bin/python scripts/rescore.py --run outputs/rerun_000_td --output outputs/rerun_000_td.csv
.venv/bin/python scripts/rescore.py --run outputs/rerun_000_pdr --output outputs/rerun_000_pdr.csv
```

Repeat for the 50 event IDs in `provenance/cohort.json`. CPU execution is supported,
but does not reproduce the original CUDA training trajectory: the CPU 16-cycle
checks differed from the archived parameters. Use CUDA for training reproduction.
The CUDA 16-cycle checks for both arms matched all actor, critic and target
parameters exactly. `--cycles 16` provides a short engineering check; compare it with the
archive using `scripts/check_training.py --run outputs/your_run --output outputs/check.json`.
The portable launcher uses the unchanged archived collection, replay sampling,
TD critic, PDR solve, detached actor weights, optimizer and target-update functions.
It verifies input hashes and refuses to overwrite an existing run. Scores are
computed separately after training. New runs are explicitly marked as reproductions.

This reproduces **continuation from the reported starting checkpoints**, not
capability acquisition from scratch. The before-practice reference, post-practice
initial models, optimizer/RNG states, vocabulary, offline records and acquired
experience are all included. Full checkpoints and collected continuation streams
are retained for both original arms. CPU/GPU differences can compound during
stochastic retraining; exact original curves are reproducible from the original
checkpoints and recorded evaluations. We do not claim that full-cohort fresh
training was repeated during packaging; see `validation/VALIDATION.md`.

## Experimental settings

- 50 paired seeds; 16 training task/text variants averaged within each seed.
- Both arms start at the same post-door-training checkpoint, with matched actor,
  critic, target, optimizers, replay and RNG. Collection uses the same assisted
  protocol; trajectories and later replay contents can diverge after policies change.
- Each cycle collects one assisted episode and takes one critic and one actor step.
  Actor/critic batches contain 6 offline and 58 online records.
- PDR uses the frozen **learned pre-practice** reference and a simulator-enumerated
  closed graph. It revalues all records in the actor batch, not only selected
  historical shortcut records. This graph is supplied transition support, not a
  learned model obtained solely from the actor's sampled minibatches.
- Full revaluation: `Q_ref + D`, hence `A_ref + Delta_A` (alpha=beta=1).
  No separate coefficient selection is used in this comparison.
- Ordinary TD critic learning in both arms. PDR weights are refreshed before each
  actor update in cycles 1–1,024. From cycle 1,025, both actors use live-critic weights.
- Discount 0.97; episode horizon 40; AWAC temperature 0.3; weight cap 20;
  Adam learning rate 3e-4; gradient clip 5; target Polyak coefficient 0.005.
- Evaluation every 16 cycles; PDR also retains checkpoints at 1 and 1,025.
- Early mean is trapezoidal AUC over 0–1,024 divided by 1,024, not endpoint return.
- Equal seed weights; paired percentile bootstrap with 20,000 draws and seed
  20260907. Curve bands are pointwise 95% intervals, not simultaneous bands.

Expected checks: junction probability at cycle 1,024 is about 33.4% for PDR and
13.5% for TD (shared initial probability 18.9%). Early mean returns are
0.47196783036098255 and 0.46232640396378544. The paired early-return gain is
0.009641426397196971, with 95% CI [0.008877257122839943, 0.010491403968784145];
all 50 paired differences are positive.

## Retained qualifications

The early-return result does not imply higher final greedy task success. At cycle
2,048, the archived full-task success difference is -0.00375 (PDR minus TD), with
two-sided 95% CI [-0.0125, 0.0025]. The prespecified noninferiority criterion was
not met; this interval does not establish a significant decrease either. Gate and
safe-route success remain 100%. Full score records and the original verdict are
retained in `provenance/locked_results/` and `records/scored/`.

Event 000 had a disclosed one-update engineering exposure before the protocol lock,
with a mismatched pilot reference. The official run restarted from its original
matched initial checkpoint. All 50 events remain in the primary cohort; the
49-event exclusion is retained as sensitivity, not a replacement cohort. See
`provenance/SC_D_PRELOCK_DEVIATION_001.md`. Do not describe this as a completely
unexposed 50-event preregistration.

## File map

| Path | Contents |
| --- | --- |
| `legacy/sc_d/src/` | Byte-identical historical PDR, training and scoring implementations |
| `legacy/parent/src/` | Byte-identical parent code, networks, task environment and evaluators |
| `inputs/events/` | All 50 reference/initial checkpoints and starting replay |
| `inputs/references/`, `inputs/support_graph.json` | Frozen learned reference arrays and closed graph |
| `checkpoints/` | Both arms' original intermediate and final full checkpoints |
| `records/scored/` | Original per-checkpoint scores, including per-instance measurements |
| `records/runs/` | Original run configurations, manifests, training logs and collected streams |
| `data/` | Per-seed CSV, plot CSV, AUC CSV, reported statistics and definitions |
| `scripts/` | Portable verification, export, plot, rescoring and continuation commands |
| `provenance/checkpoints.csv` | Every score-to-checkpoint link and checkpoint SHA-256 |
| `provenance/source_files.json` | Source paths, sizes and hashes for every copied artifact |
| `provenance/release_files.json` | Integrity manifest for the assembled release |
| `validation/` | Executed checks and their limitations |

The primary figure intentionally contains only PDR and TD. Historical aggregate
files may also mention the selected-record exact-weight intervention used in the
Introduction. Its separate training/checkpoint collection is outside this release's
PDR–TD scope. Original source notices are preserved; no third-party license terms
are changed by packaging.
