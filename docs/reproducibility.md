# Reproducibility

This release supplies four complete benchmark packages: KeyCorridor, MiniGridLoCA, DeepSea and SecureCourier. Their saved checkpoints, inputs, raw records, configs and scripts are included. Other report analyses/ablations and unreported historical training checkpoints are outside the checkpoint upload scope.

Evaluation reproduction loads archived actors and compares their exact metrics with recorded observations. DeepSea's primary acceptance metric is expected return; count-valued auxiliary diagnostics are recorded separately. KeyCorridor/LoCA compare their shared exact metrics. SecureCourier uses its native scorer and 2e-6 tolerance. See each report for actual cases and errors.

Results reconstruction regenerates statistics and figures from recorded raw curves or scores. Training reproduction is a continuation from a supplied initial checkpoint with recorded optimizer/RNG/replay state. It is not a claim to regenerate all source pretraining. Original experiment outcomes, seed eligibility, actor objectives and reference types remain unchanged.

Fresh checks cover all 520 DeepSea final actors, all 480 LoCA final actors and 32 representative KeyCorridor actors; all 1,640 primary run records pass 10,560 path/hash checks. Pre-existing validation reports are labelled historical. Original checkpoints were not silently replaced with new training outcomes.

The archive stores each distinct SHA-256 payload once and validates decompressed bytes. Metadata reconstructs normal files and internal duplicates without linking to original server research folders. The final shard-verification report states its precise coverage. Some historical native files retain absolute paths as provenance; portable benchmark entry points resolve inputs within their package.

The linked report's abstract names the four benchmark release scope using future tense until publication/access have actually been verified. The abstract revision leaves the body and 134-page count unchanged; `docs/report-revision.json` retains both report digests.
