# Release scope for author review

The initial upload is private. Public visibility is left to the author after review.

## GitHub

- Four benchmark numerical implementations under `src/releases/`, preserving original source notices.
- Actual per-run configurations, hashes and reproduction entry points.
- Environment requirements and build instructions.
- Raw-result summaries/plots, scoped evaluation reports and the report PDF.
- Dataset download/restoration code and an immutable checkpoint revision after upload.

## Hugging Face

Complete KeyCorridor, MiniGridLoCA, DeepSea and SecureCourier packages, approximately 28.75 GB compressed. These include actually saved model states, matched starting/reference/replay inputs, raw curves/scores and native code/configuration/provenance retained by those releases.

## Excluded

- Checkpoint collections for other report analysis/ablation studies.
- Unreported pretraining histories, duplicated historical archives and session/credential files.
- User access tokens, virtual environments and machine caches.

The availability statement is limited to the four benchmark packages. It does not assert that all report analyses have released checkpoints or that all training was rerun. Evaluation and training-continuation commands are distinct.
