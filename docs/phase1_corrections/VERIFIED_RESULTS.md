# Phase 1 verified results and remaining work

Review date: 8 November 2026. Main operator: Fahad Falah H. AlQahtani. Source: the uploaded `Phase1_Results_Only.zip` (49 entries, 505,322 bytes).

## Verification outcome

**165 consistency checks passed.** Accuracy, macro-F1, class recalls, prediction frequencies, per-class precision/F1, calibration ECE and all eight group-bootstrap confidence intervals were independently recomputed from the saved predictions. The prediction/metrics checksums, fixed protocol, configuration, source-code fingerprints, validation checkpoint choices, condition tests and error counts agree.

The selection record was locked at 08:04:26 UTC, before final testing began at 10:34:23 UTC on 8 November 2026. The selected model is `full`, chosen by minimum unweighted validation cross-entropy (0.016354). Its validation accuracy is 232/233, or 99.57%.

## Primary results: all 328 test images

| Strategy | Accuracy | 95% group-bootstrap CI | Macro-F1 | Real recall | Fake recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Scratch unweighted | 87.20% | 83.90–90.45% | 76.72% | 48.53% | 97.31% | 0.1319 |
| Scratch weighted | 86.59% | 82.45–90.52% | 78.65% | 61.76% | 93.08% | 0.1679 |
| Frozen comparison | 88.72% | 83.53–92.95% | 85.02% | 94.12% | 87.31% | 0.0774 |
| Full — validation-selected | 68.60% | 62.54–74.56% | 65.72% | 95.59% | 61.54% | 0.2548 |

These are the full, naturally imbalanced results: 68 Real and 260 Fake images in 68 source groups. The balanced secondary results use 136 images selected from the same saved prediction vectors. The selected full model has 80.15% accuracy and 79.66% macro-F1 on that secondary subset; these do not replace the primary figures.

## What the results establish

The full model has 65 true Real decisions, 3 Real images incorrectly called Fake, 160 true Fake decisions and 100 Fake images incorrectly called Real. The frozen model has 64, 4, 227 and 33 respectively. The full model's strong validation score did not transfer to the new DFGAN cohort. This is a measured generalization gap; the specific mechanism has not been established.

Keep `full` as the validation-selected model for this completed experiment. Report `frozen` as the stronger test comparison. Selecting a new winner using these test results would be a new, test-informed decision; future model development needs separate validation and a fresh final cohort.

Because 260/328 test images are Fake, an always-Fake rule obtains 79.27% accuracy but zero Real recall. Its macro-F1 is 44.22%. This shows why accuracy must be interpreted alongside macro-F1 and both class recalls. The selected full model's accuracy is below that majority baseline, while its macro-F1 and Real recall are higher.

The controlled weighted scratch comparison improves Real recall from 48.53% (33/68) to 61.76% (42/68), and macro-F1 from 76.72% to 78.65%. Accuracy decreases from 87.20% to 86.59%. Describe this as an observed class trade-off, rather than improvement in every metric. Both controls used the same seed, architecture, data, batch size and selection procedure.

## Scene and condition evidence

Full-model errors are 72/262 indoors and 31/66 outdoors. Within the Fake class they are 69/208 indoors (33.17%) and 31/52 outdoors (59.62%). Real-image errors are 3/54 indoors and 0/14 outdoors. The small outdoor Real sample limits interpretation of its zero errors.

Controlled Gaussian blur on validation reduces full-model accuracy from 99.57% to 83.26%, a 16.31 percentage-point drop. Darkening, brightening and the defined synthetic-haze overlay reduce it by 2.15, 0.86 and 1.29 points respectively. These interventions measure sensitivity on validation; they do not prove the cause of a natural test mistake.

**The natural sharpness association points in a different direction.** Among DFGAN test images, the low-sharpness slice has 10/60 errors (16.67%), versus 90/200 (45.00%) elsewhere. Its Fisher test survives the 36-test Holm correction (adjusted p approximately 0.002398). The overall low-sharpness association also survives (adjusted p approximately 0.036736). No other natural-condition test survives that correction. Thus, do not attribute the 100 DFGAN misses to blur. Natural image sharpness and a controlled blur perturbation on a different population are separate observations. Low contrast is not a confirmed haze annotation, and no human visual cause review was claimed.


The Fisher/Holm values above are exploratory image-level tests: multiple images share a COCO parent, and Holm correction alone does not model that dependence. An additional post-result source-group bootstrap check (50,000 draws, seed 42) gives a low-sharpness-minus-other error-rate difference of −28.33 percentage points within DFGAN Fake images, with an approximate 95% interval of -40.84 to -14.55 points. At the conservative 99.861% interval level corresponding to 36 candidate comparisons, the interval is -47.82 to -4.33 points. This supports the observed direction after accounting for parent grouping; it remains an observational, exploratory result. The method and numbers are saved in `completed_run/verification/group_aware_sharpness_review.json`.

A future development plan can test broader generator coverage, generator-separated validation, backbone-freezing/partial-tuning comparisons and controlled blur augmentation. These are hypotheses for a new development cycle; this completed test cohort is closed to tuning. Choose and calibrate future models on development data and evaluate a newly reserved final cohort.

## Data and protocol checks

- Original cleaning: 1,800 images; 0 unreadable; 402 duplicate copies removed; 1,398 retained at 224x224.
- Verified development groups: 499 original COCO parents form 498 connected components; train 1,165 images/415 groups; validation 233 images/83 groups.
- Test: 328 images/68 new groups; training/validation Fake images are GALIP; test Fake images are DFGAN. No connected group, COCO parent, decoded pixel hash or historical pHash<=4 candidate crosses into test in the saved metadata.
- The canonical development assignments are loaded from the immutable manifest; they are not regenerated with version-dependent shuffled splitting.
- The earlier source audit contains 100 recorded publisher-reference decisions, zero disagreements and a one-sided 95% Clopper–Pearson upper bound of 2.9513%. Its automated provenance method should be named accurately.
- Accuracy intervals resample 68 whole source groups with 2,000 seeded bootstrap draws. They describe this fixed cohort/model evaluation, not variability across retraining seeds.

## Reproduction and submission status

The main corrected experiment, final evaluation and exports are complete. **Another member's independent run has not been supplied.** Use `Phase1_Independent_Reproduction.ipynb` in that member's fresh Kaggle GPU runtime, then compare the actual output against this archive. A recovery of Fahad's saved weights, a copied CSV or a second run under Fahad's name is not the requested another-member reproduction.

The correction report now retains the original uploaded template and includes the verified primary results, confidence interval, selection rule, measured error findings, complete source audit, actual-result figures and an accurate checklist. Its PDF has compiled and passed visual review. The independent comparison remains explicitly pending. After the actual member run, insert that comparison; after delivery, insert the published commit and grader model-access details. No revised mark or full score is guaranteed by this review.

The source bundle also includes `scripts/verify_main_results.py`, checked on the actual main records. It recomputes all eight primary/secondary result rows without images, weights or a GPU and checks their recorded hashes, source/configuration/protocol, validation checkpoint choices and group intervals. This is a check of saved predictions, not new inference or independent training.

Use the minimal pinned project requirements to install the experiment. `requirements-lock-executed.txt` is an inventory of the Kaggle training environment, including unrelated preinstalled tools; it is not a clean standalone installation specification.

## Limits of this review

The small results archive intentionally excludes trained weight bytes and raw image pixels. This review verifies recorded checkpoint hashes and internal consistency; direct weight/image integrity was checked in Fahad's Kaggle recovery run. Retain `Phase1_Corrected_Models.zip` and the original backup. The archive contains training-derived condition thresholds and test features, but not the complete raw training feature table, so those thresholds were not independently recalculated from pixels here. Software logs and timestamps are useful records, not external authentication of human independence. The study does not establish geographic performance on Saudi events or an operational deployment claim.

## Traceability

Main run ID: `347c0f0188f04f8c873e2eac415e5723`.

Protocol SHA256: `0544ea4ee4940c56874855021b0f0a5873f121ddeaf26c4880894374529eeb8b`.

Selected checkpoint SHA256: `a2fd5d5ec37c1bcf2a9fc6243115306aadc987fa3404d6945a5fd97961d5504e`.

Uploaded results archive SHA256: `070bfa62493904cbf6f5f753004456bac3cbd2ef459c651ef6a72f007e4de709`.
