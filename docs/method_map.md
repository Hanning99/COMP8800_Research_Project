# Implementation map

| Component | Code |
|---|---|
| DeepSea PDR-SAC and reference/change weighting | `src/releases/DeepSea/src/contrast.py`, `learner.py`, `kernels.cpp` |
| KeyCorridor learned-reference PDR-AWAC | `src/releases/KeyCorridor/src/emphasis.py`, `learner.py`, `key_cuda.py` |
| LoCA reward-change/entropy source and solve | `src/releases/MiniGridLoCA/src/learner.py`, `hypotheses.py`, `vendor/minigrid_loca_core.py` |
| SecureCourier actor/critic, PDR and scorer | `src/releases/SecureCourier/legacy/sc_d/src/`, `legacy/parent/src/`, `scripts/` |

Numerical sources are preserved, with source digests in `configs/PRIMARY_SOURCE_FILES.json`. The new top-level scripts orchestrate archive restoration and call the retained model classes/evaluators. DeepSea's inherited cell labels ending in `exact_AWAC` do not change the current PDR-SAC objective. Exact-reference tables and neural reference models remain distinct.
