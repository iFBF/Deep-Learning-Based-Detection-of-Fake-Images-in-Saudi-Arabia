# Real-Abshar Phase 2 — M4, M5 and M8

The selected detector, segmenter and integrated CPU service have completed the fixed Phase 2 test. The test is closed. See [verified measurements](VERIFIED_RESULTS.md), the [final model card](MODEL_CARD.md), and [working local service instructions](RUN_LOCAL.md).

The title describes a research goal. These results do not establish Saudi operational accuracy. The supported use is a human-review research demonstration; the preliminary verdict remains NO-GO for automatic authenticity decisions.

## Source map

| Folder | Purpose |
|---|---|
| `preparation/` | Actual mask QC, pairing, source grouping and frozen preparation code |
| `m4/` | Actual executed YOLO11n comparison, validation selection and checkpoint backup sources |
| `m5/` | Actual executed FCN/U-Net comparison, losses, mask metrics and checkpoint backup sources |
| `export/` | Actual ONNX export/parity sources and selected-model metadata |
| `service/` | Unchanged frozen HTTP service source, model manifest, processor config, policy and aggregate benchmark evidence |
| `evaluation/` | Actual fixed final evaluator/ledger/cache/report source |
| `notebooks/` | Clean code-only preparation and M4/M5 reproduction launchers |
| `results/` | Aggregate validation/test metrics and completion addendum |

The preparation notebook's input path is a placeholder: set it to the visible private CSI gold input. Training notebooks require the exact reviewed prepared-data archive and enabled `WANDB_API_KEY` secret. M4 uses Ultralytics8.3.221/W&B0.28.1; M5 uses the recorded Torch2.11/torchvision0.26 GPU pair. Retain the actual Kaggle torch/CUDA build. These notebooks train separate experiments; their new weights do not replace the frozen test models.

The export/benchmark/final evaluator additionally require private release assets and, for the pre-test validation gate, the original native validation prediction reference. Those private assets remain in the delivered project kits and confirmed W&B artifacts. Publishing evaluator source does not make private dataset annotations or cache records public. The completed test can be audited/replotted from its private cache; do not start a second model-selection/test experiment.

## Verify the public aggregate summary

```bash
python phase2/verify_public_summary.py
```

This checks committed source hashes and metric arithmetic using the aggregate counts. The original 2,005-check independent review additionally used private per-image counts, detections and trusted source assignment; those records are deliberately absent here. Public arithmetic verification alone is not a new inference run or an independent reproduction of the training.

W&B project: https://wandb.ai/f1k80-kaust/Real-Abshar-Phase2

Private inputs, pixel/box annotations, per-image predictions, failure-gallery photographs, model binaries, API keys and personal student details are excluded from this publication. Grant actual graders authorized private access to required weights/data. The earlier Phase1 code and its recorded results remain unchanged; the independent teammate Phase1 rerun is pending.
