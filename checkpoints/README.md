# Checkpoints

Private dataset: https://huggingface.co/datasets/denghanning/COMP8800_Research_Project

Scope: complete KeyCorridor, MiniGridLoCA, DeepSea and SecureCourier release packages, approximately 28.75 GB compressed. Analysis/ablation checkpoint histories outside these four packages are not uploaded.

The pinned dataset commit is written to `configs/release.json` after successful publication. Authenticate with dataset access, then run:

```bash
python scripts/download_checkpoints.py --out checkpoints/local
python scripts/checkpoint_archive.py verify --archive-root checkpoints/local --release DeepSea
```

`manifest.sqlite` maps each retained file to a SHA-256 payload and its location in `shards/*.pack`. The loader restores ordinary files, checks decompressed digests/sizes and refuses to overwrite existing output directories. It never follows server-specific original paths to fetch missing data.

A benchmark's initial/final/intermediate checkpoint availability follows what that release actually saved; no unsaved state is invented. Exact tables are labelled as tables. Saved starts, reference/replay inputs and native scripts accompany the models. Grant supervisors access to both private repositories.
