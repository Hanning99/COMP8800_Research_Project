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

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-publish.txt

python run.py evaluate DeepSea                                   # evaluate saved actors (CPU)
python run.py results  DeepSea                                   # rebuild results
python run.py train    DeepSea -- --size 10 --method PDR         # continue training (GPU)
```

The benchmark can be `DeepSea`, `KeyCorridor`, `MiniGridLoCA` or `SecureCourier`. Its checkpoints are downloaded from Hugging Face on first use (3.5–10.6 GB each) and kept in `data/`. Add `--quick` to evaluate one seed per setting.

## Layout

```
run.py      single entry point
src/        benchmark environments and methods
configs/    recorded run configurations
scripts/    lower-level download, restore and evaluation tools
docs/       report
```
