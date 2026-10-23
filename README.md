<div align="center">

# One Solve, Every Step

**Accelerating Actor–Critic Learning**

[Report](docs/report.pdf) &nbsp;·&nbsp; [Checkpoints](https://huggingface.co/datasets/denghanning/COMP8800_Research_Project) &nbsp;·&nbsp; [Quick start](#quick-start)

</div>

<br>

When the critic learns something new, the actor often takes many updates to act on it. **Policy Differential Revaluation (PDR)** targets that delay: it recomputes the value signal for the actor from the current policy, a fixed reference policy and a transition graph, while the critic keeps training as usual.

This repository contains the code and run configurations for the four benchmarks in the report.

| Benchmark | Setting |
|:--|:--|
| KeyCorridor | S3R2 · S3R3 · S4R3 · S5R3 |
| MiniGridLoCA | Reward change from AWAC, SAC and QAC starts |
| DeepSea | N = 10 · 20 · 30 · 40 |
| SecureCourier | Gate skill and route choice |

## Quick start

**1. Install** (Python 3.10)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-publish.txt
```

**2. Download checkpoints** (~28.75 GB)

```bash
python scripts/download_checkpoints.py --out checkpoints/local
```

**3. Run**

```bash
bash scripts/evaluate.sh  --benchmark DeepSea --archive checkpoints/local   # evaluate saved actors (CPU)
python scripts/run.py results --benchmark DeepSea --archive checkpoints/local   # rebuild results
bash scripts/reproduce.sh --benchmark DeepSea --archive checkpoints/local \
  -- --size 10 --method PDR --updates 8192                                  # continue training (GPU)
```

`--benchmark` takes `KeyCorridor`, `MiniGridLoCA`, `DeepSea` or `SecureCourier`.

## Layout

```
src/releases/   benchmark environments and methods
configs/        recorded run configurations
scripts/        download, evaluation and training entry points
docs/           report
```
