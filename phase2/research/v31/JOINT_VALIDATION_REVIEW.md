# Real-Abshar joint-validation review and V3.1 calibration

Reviewed 19 November 2026. The uploaded V3 joint-validation run is internally consistent but failed its declared macro-F1 preservation gate. A separately recorded adaptive validation refinement selected raw cutoff **4.25** for the supplemental model. The original failed V3 run has not been rewritten.

## What the uploaded V3 run showed

**Data:** CSI validation only, 126 images (105 edited, 21 pristine), 21 source families. **Inputs:** Swin 224×224; supplemental 384×384; detector/segmenter 640×640, all FP32. **Baseline:** Swin fake score ≥0.5; **V3:** Swin ≥0.99 OR supplemental raw logit ≥4.0. Both retain detector confidence 0.35, NMS IoU 0.45, maximum 50 detections, mask pixel threshold 0.5 and minimum edited-area fraction 0.005. These are full-system referral metrics, not classifier-only or pixel segmentation metrics.

| CSI validation measure | Original service | Uploaded V3 (4.0) | V3.1 (4.25), cached recomputation |
|---|---:|---:|---:|
| Accuracy | 91.27% (115/126) | 90.48% (114/126) | 91.27% (115/126) |
| Macro-F1 | 86.07% | 84.56% | 86.07% |
| Edited images referred | 96/105 | 96/105 | 96/105 |
| Pristine images wrongly referred | 2/21 | 3/21 | 2/21 |

Exactly one uploaded V3 decision changed from the baseline: pristine `csi_0109_original` (source 109). Its supplemental raw logit was **4.215185165405273**. Original Swin score was about 1.46e-7 and neither local flag was active. At cutoff 4.0 it was unnecessarily referred. At 4.25 it is not. V3.1's cached decisions match the original baseline on every CSI validation image; it adds no measured CSI recall benefit.

The uploaded V3 gates passed the permitted extra-pristine-referral allowance, edited recall and runtime, but failed macro-F1 preservation. Warm baseline HTTP p95 was **907.31 ms**; warm V3 client p95 was **1246.74 ms**, below the declared 1610.97 ms limit. These timings belong to the actual V3 4.0 run. V3.1 uses the same graphs, but its latency has **not been remeasured** and no new live acceptance claim is made.

## What changed during development

This stage changes only the supplemental raw cutoff. It does not retrain a model, change M4/M5, apply an image-specific exception or use a dataset-dependent routing rule.

After the uploaded run, a documented coarse validation grid {4, 4.5, 5, 5.5, 6} found no cutoff satisfying all proposed retention constraints. A second, explicitly adaptive refinement checked {4, 4.25, 4.5} in the identified bracket. Constraints and ranking were unchanged. Cutoff 4.25 was feasible and had the best worst-domain balanced accuracy. The newly inspected CSI scores and already-used global validation scores both influenced selection. This is normal development selection and **not independent confirmation**. The proposed retention constraints are engineering choices, not additional course requirements.

**Global data:** original validation 233 COCO/GALIP images and additional development 72 images, 36 labelled real / 36 generated, grouped into 14 conservative scene/series families. **Inputs:** original Swin 224×224 and supplemental 384×384; baseline Swin cutoff 0.5, V3/V3.1 Swin cutoff 0.99 and supplemental cutoffs 4.0/4.25 respectively. These tables measure global generation flags only, excluding local box/mask flags.

| Global development measure | Original Swin | V3 (4.0) | V3.1 (4.25), cached recomputation |
|---|---:|---:|---:|
| Original-validation accuracy | 99.57% | 98.71% | 98.71% |
| Original-validation macro-F1 | 99.53% | 98.61% | 98.61% |
| Additional-development accuracy | 47.22% | 87.50% | 88.89% |
| Additional-development macro-F1 | 36.13% | 87.48% | 88.85% |
| Additional generated images flagged | 2/36 | 30/36 | 30/36 |
| Additional real images wrongly flagged | 4/36 | 3/36 | 2/36 |

For comparison, cutoff 4.5 also restores the CSI result and reaches 87.5% additional-development accuracy, but flags only 29/36 generated images. This is why accuracy alone was insufficient for operating-point selection.

All original V3 checksums matched the supplied archive. Metrics were independently recomputed from per-image records and all V3 row decisions matched their rule. Global cached scores were checked against the frozen candidate reference and baseline records. No original test pixels or masks were accessed. The evidence/calibration folder includes the original upload, both specifications, all candidate outcomes, cached global scores and a dependency-free recomputation script.

## Decision and next action

Keep **automatic authenticity decisions at NO-GO**. V3.1 is a validation-selected research comparison: its 88.89% additional-development accuracy and 88.85% macro-F1 still miss the 90% target. The 72 images have repeatedly influenced development; external pretraining overlap cannot be excluded. Neither domain supplies a new untouched performance claim. The publisher also reports failed robustness checks.

No additional GPU training is needed to use this policy. The next concrete step is a Windows local smoke test through the unchanged original service using the new V3.1 client. Follow README.md. This verifies the user's actual installation and saves latest_v31_comparison.json; it is not an accuracy evaluation. Original service files, published V1 commit, W&B records and grader submission remain separately identifiable.

The supplemental checkpoint is an externally trained Thermostatic frontier fine-tune of Community Forensics, not a model trained by the team. Team work includes preprocessing verification, development evaluation, validation-only policy selection, ONNX export and system integration. See MODEL_CARD.md for exact revisions, attribution and limitations.
