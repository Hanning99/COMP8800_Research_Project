# One Solve, Every Step: Accelerating Actor–Critic Learning

Code for the experiments in *One Solve, Every Step* ([report](docs/report.pdf)). Checkpoints and raw outcomes are in the [Hugging Face dataset](https://huggingface.co/datasets/denghanning/COMP8800_Research_Project).

| Benchmark | Setting |
|:--|:--|
| KeyCorridor | S3R2, S3R3, S4R3, S5R3 |
| MiniGridLoCA | Reward change; AWAC, SAC, QAC starts |
| DeepSea | N = 10, 20, 30, 40 |
| SecureCourier | Gate skill and route choice |

## Setup

Python 3.10. The checkpoint archive is about 28.75 GB compressed.

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-publish.txt
python scripts/download_checkpoints.py --out checkpoints/local
```

## Usage

`--benchmark` is one of `KeyCorridor`, `MiniGridLoCA`, `DeepSea`, `SecureCourier`.

```bash
# Evaluate saved actors (CPU)
bash scripts/evaluate.sh --benchmark DeepSea --archive checkpoints/local

# Rebuild results from recorded outcomes
python scripts/run.py results --benchmark DeepSea --archive checkpoints/local

# Restore a benchmark package
python scripts/checkpoint_archive.py restore --archive-root checkpoints/local \
  --release DeepSea --output work/DeepSea

# Continue training from a saved checkpoint (GPU)
bash scripts/reproduce.sh --benchmark DeepSea --archive checkpoints/local \
  -- --size 10 --method PDR --updates 8192
```

## Layout

| Path | Contents |
|:--|:--|
| `src/releases/` | Benchmark and method implementations |
| `configs/` | Run configurations; `release.json` pins the dataset revision |
| `scripts/` | Download, restore, evaluation and training entry points |
