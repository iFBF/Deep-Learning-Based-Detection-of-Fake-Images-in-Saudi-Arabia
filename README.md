# Real-Absher — real / AI-generated image detection

The corrected Phase 1 main experiment is complete. On the source-disjoint 328-image DFGAN test, the validation-selected full model scores **68.60% accuracy (95% group-bootstrap CI 62.54–74.56%)** and **65.72% macro-F1**. The frozen comparison scores 88.72% accuracy and 85.02% macro-F1. Model selection remains based on validation cross-entropy; these test results document a generator-generalization limitation.

See the [verified results and limitations](docs/phase1_corrections/VERIFIED_RESULTS.md) and [corrections/data guide](docs/phase1_corrections/README.md). The results archive passed 165 consistency checks. Another member's independent execution is still pending.

## Check the recorded results without a GPU

```bash
python -m pip install -r requirements-data.txt
python scripts/verify_main_results.py
```

This recomputes all eight primary/secondary metric rows from the saved predictions and checks their hashes, source/configuration records, validation checkpoint choices and group confidence intervals. It does not load weights or rerun inference. Use the full notebook below for fresh training.

## Reproduce the fixed experiment

Use the [clean independent-member notebook](notebooks/phase1_independent_reproduction.ipynb) in a fresh Kaggle T4 GPU runtime with Internet enabled. Attach this code as a dataset and the original prepared DANI `data-dfd` images or `data.zip`. Make private inputs available to the actual member or grader. Enter the actual operator's name, install the pinned requirements, then run the experiment cell. The job trains all four strategies anew in a fresh Python process and exports real models and results.

```bash
python -m pip install -r requirements.txt
python scripts/preflight_phase1.py
```

Use the minimal pinned project files for installation. The completed run's `requirements-lock-executed.txt` records the whole Kaggle environment, including unrelated packages; it is an environment inventory rather than a standalone resolver input.

Keep the model ZIP and executed notebook. Return the smaller results ZIP for comparison with [the main run](docs/phase1_corrections/completed_run/). Historical notebook outputs and old result CSVs remain exploratory and are preserved for traceability. The earlier 97.56% result is not the corrected test score.

- [Dataset card](docummentation/dataset_card.md)
- [Main-run verification record](docs/phase1_corrections/completed_run/verification/results_verification.json)
- [Original main-run notebook](notebooks/phase1_corrections_kaggle.ipynb)

Team: Fahad Falah H. AlQahtani, Anoud Mohammd A. Alenzei, Abdulaziz Salim AlQahtani.

## Phase 2: localization and integrated service

The fixed Phase2 test is complete on114 images from19 source families. The selected detector reaches57.49% AP50/36.71% AP50–95; the selected segmenter reaches94.20% foreground IoU/97.01% pooled foreground Dice. The integrated human-review policy reaches87.72% accuracy and79.57% macro-F1. These are CSI local-edit measurements and do not replace the original Phase1 criteria or establish Saudi operational accuracy.

See the [Phase2 source and reproduction map](phase2/README.md), [verified results and conditions](phase2/VERIFIED_RESULTS.md), [final model card](phase2/MODEL_CARD.md), and [working local service instructions](phase2/RUN_LOCAL.md). Public source includes aggregate results; dataset images/annotations, private per-image caches, weights and student personal details are excluded. The preliminary verdict remains NO-GO for automatic authenticity decisions.
