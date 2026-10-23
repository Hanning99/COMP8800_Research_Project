# Checkpoints

Checkpoints are hosted at https://huggingface.co/datasets/denghanning/COMP8800_Research_Project (revision pinned in `configs/release.json`).

```bash
python scripts/download_checkpoints.py --out checkpoints/local
python scripts/checkpoint_archive.py verify --archive-root checkpoints/local --release DeepSea
```
