# Phase 1 corrections and verified main run

Fahad completed the fixed Kaggle experiment on 8 November 2026. The original training checkpoints were recovered from a saved backup; all 1,726 protocol image hashes and all four checkpoint hashes were rechecked before final evaluation. The uploaded results subsequently passed 165 consistency checks. Read [VERIFIED_RESULTS.md](VERIFIED_RESULTS.md) for the complete findings and limits of that review.

## Fixed data foundation

Original input: 1,800 images, 0 unreadable, 402 duplicate copies removed, 1,398 retained at 224x224. Verified caption links identify 499 COCO parents, forming 498 connected source groups because one duplicate connects two families. Development has 1,165 training images/415 groups and 233 validation images/83 groups. The canonical workflow reads the checksum-verified fixed manifest instead of regenerating a shuffled split across library versions.

The recorded source-label audit has 100 publisher-reference decisions, zero disagreements and a one-sided 95% Clopper–Pearson upper bound of 2.9513%. It is an automated source correspondence audit, not a claim of human visual review or universal real-world authenticity.

The new test has 328 images/68 source groups: 68 Real and 260 DFGAN Fake. Development Fake images are GALIP. No historical source-group, exact-pixel or pHash<=4 overlap was found. This test cohort has now been evaluated and is closed to further model development. The committed acquisition report is the immutable pre-evaluation snapshot; the completed evaluation lives in `completed_run/`.

## Verified final measurements

| Strategy | Primary accuracy | 95% group-bootstrap CI | Macro-F1 | Real recall | Fake recall |
|---|---:|---:|---:|---:|---:|
| Scratch unweighted | 87.20% | 83.90–90.45% | 76.72% | 48.53% | 97.31% |
| Scratch weighted | 86.59% | 82.45–90.52% | 78.65% | 61.76% | 93.08% |
| Frozen comparison | 88.72% | 83.53–92.95% | 85.02% | 94.12% | 87.31% |
| Full, selected on validation | 68.60% | 62.54–74.56% | 65.72% | 95.59% | 61.54% |

Full was chosen by the lowest validation loss before testing. Its 100/260 missed DFGAN fakes expose a generalization gap. The stronger frozen test comparison does not retrospectively replace the validation choice. Both full-unbalanced and balanced-secondary results, per-class metrics, confidence intervals and ECE are saved in `completed_run/`.

Controlled blur on validation reduces full accuracy by 16.31 percentage points. Natural low-sharpness DFGAN test images instead have fewer errors (10/60 versus 90/200; Holm-adjusted p approximately 0.002398). These are different observations; they do not establish blur as the cause of the DFGAN misses. Image-level Fisher/Holm values are exploratory under shared-parent dependence; an additional group-bootstrap review supports the observed sharpness association. Brightness, contrast and sharpness were measured, source/scene/class slices were counted, and the weighted scratch comparison's improved Real recall is reported alongside its lower overall accuracy.

## Independent member execution

Another member executes [the independent reproduction notebook](../../notebooks/phase1_independent_reproduction.ipynb) in a fresh Kaggle GPU runtime. Attach the corrections code and original image dataset, enter the actual operator's name and run the three code cells in order. The complete experiment runs in a fresh Python process after installing the pinned requirements. It creates a fresh run ID and trains every strategy; it does not copy the main trained checkpoints.

Return `Phase1_Independent_Results_Only.zip` and retain `Phase1_Independent_Models.zip`. Compare the actual member run with the main run using `scripts/compare_reproduction.py`, with the member name exactly matching the recorded operator. Independent member execution and its side-by-side comparison remain pending. The new notebook's JSON and code have been checked; execution is a human action still required.

The separate correction report retains the original uploaded template and now includes the actual primary metrics, CI, measured error patterns and an accurate checklist. Its independent comparison is explicitly pending. Insert the member's actual output when available, then finalize published source/model references. This code/data correction does not guarantee a revised mark.

## Installation and evidence

Use `requirements.txt`, which includes the minimal pinned locks and matching CUDA 12.8 Torch/TorchVision/TorchAudio builds. Use `scripts/preflight_phase1.py` to display the recorded status. The completed run's full installed-package inventory is provenance, not the installation specification.

For a quick CPU check of the recorded result tables, install `requirements-data.txt` and run `python scripts/verify_main_results.py`. This recomputes metrics and group intervals from saved predictions and checks the recorded protocol/configuration/source hashes; it does not evaluate trained weights or replace the independent run.

`evidence/` retains the original cleaning, grouping, audit and acquisition records. `historical/` retains the original exploratory predictions and histories. `completed_run/` contains the actual corrected run's predictions, metrics, histories, conditions and verification record. Raw images, publisher parquet shards, credentials and trained weight bytes are excluded from Git. Retain the complete model ZIP separately and follow the underlying data/model licences.
