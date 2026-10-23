# One Solve, Every Step: Accelerating Actor–Critic Learning

Code and configurations for the experiments in *One Solve, Every Step*. Checkpoints and raw outcomes are in the companion [Hugging Face dataset](https://huggingface.co/datasets/denghanning/COMP8800_Research_Project).

Policy Differential Revaluation (PDR) recomputes the value signal used by the actor from the current policy, a fixed reference policy and a transition graph; the critic is trained as usual.

## Benchmarks

| Benchmark | Setting |
|:--|:--|
| KeyCorridor | S3R2, S3R3, S4R3, S5R3 |
| MiniGridLoCA | Reward change; AWAC, SAC, QAC starts |
| DeepSea | N = 10, 20, 30, 40 |
| SecureCourier | Gate skill and route choice |

## Reproduction

Requires Python 3.10. Evaluation runs on CPU; the archive is about 28.75 GB compressed.

```bash
git clone https://github.com/Hanning99/COMP8800_Research_Project.git
cd COMP8800_Research_Project
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-publish.txt

python scripts/download_checkpoints.py --out checkpoints/local
bash scripts/evaluate.sh --benchmark DeepSea --archive checkpoints/local
```

`--benchmark` accepts `KeyCorridor`, `MiniGridLoCA`, `DeepSea` or `SecureCourier`. Restoring packages, rebuilding results and continuing training are described in [docs/reproducibility.md](docs/reproducibility.md).

## Layout

| Path | Contents |
|:--|:--|
| `src/releases/` | Benchmark and method implementations |
| `configs/` | Run configurations; `release.json` pins the dataset revision |
| `scripts/` | Download, restore, evaluation and training entry points |
| `evaluation/results/` | Verification and evaluation records |
