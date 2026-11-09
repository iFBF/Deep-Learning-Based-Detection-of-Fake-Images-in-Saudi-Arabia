# Real-Abshar — final fixed-evaluation model card

The reserved Phase2 test evaluation is complete under the frozen service/protocol.

## Intended use

Assist a human reviewer by displaying a global generated-image model score, detected editing
regions, a pixel editing mask and an edited-area estimate. The final service output is a
review recommendation, not an authenticity certificate or an accusation. The project title
does not establish performance on Saudi operational images.

## Components and provenance

- Global classifier: corrected Phase1 full-fine-tuned Swin, Real/Fake order,224×224,
  native checkpoint SHA`a2fd5d5ec37c1bcf2a9fc6243115306aadc987fa3404d6945a5fd97961d5504e`.
- Detector: one-class edited-region YOLO11n, selected standard-augmentation run,640×640,
  native checkpoint SHA`66b440225fa64a30ef035706ddde8d01489476a1c3331f2d289655b5be9afb40`.
- Segmenter: U-Net with ResNet18 encoder, selected epoch25, binary background/edited region,
  640×640, native checkpoint SHA`301cae566f84ec9ffa7753373320e90bb9a0af70d64e85a8d3d89b2a92481b2c`.

Exact ONNX hashes, file sizes, class orders, preprocessing, thresholds, source/configuration
hashes and processor files are in the frozen release manifest/lock. All ONNX graphs are FP32,
fixed batch1, opset17; NMS and native mask restoration remain in code. The selected graph
hashes are unchanged from the passing M8 export review.

Repository: https://github.com/iFBF/Deep-Learning-Based-Detection-of-Fake-Images-in-Saudi-Arabia
Corrected Phase1 code commit:`26a12b0a69641990be2ef4b2794a19e420f87e20`.
Phase2 source and this model card are versioned together. Record the actual repository commit containing this file in the report; obtain it with `git rev-parse HEAD` after publication.
W&B project: https://wandb.ai/f1k80-kaust/Real-Abshar-Phase2

## Phase2 data and annotations

CSI-IMD gold prepared data, reviewed complete-aligned-family subset:132 source families,
792 images; train552/92 families, validation126/21, reserved test114/19. Each retained family
has one pristine image and five edited versions. Rejected annotations remain rejected:
335 dimension-mismatched pairs and one grey-valued mask among the1000 candidate edits;
the complete-family restriction drops further partial-family records without reassigning splits.
Targets are native binary editing/inpainting regions and their union boxes, not semantic
object silhouettes or semantic-relevance masks. No source/group/RGB hash/exact pHash crosses
the recorded splits; this does not prove the absence of every hidden near duplicate.

## Validation evidence and limits

All following Phase2 task results are **validation126, region-model input640×640**.
Detector confidence0.35/NMS0.45/matchIoU0.50/max50: precision87.30%, recall52.38%, F165.48%.
Low-floor AP uses confidence0.001 at NMS0.45: mAP50=60.62%, mAP50–95=44.00%.

Segmenter threshold0.5/no postprocessing/native-size scoring: foreground IoU93.02%,
foreground Dice96.39%, two-class mIoU96.03%. Mean Dice over105 edited images is84.07%.
Small-mask subset6 images has pooled foreground Dice42.03%;7/21 pristine images have at least
one false mask pixel. These small related-image samples limit precision and generalization.

M8 export parity passed all13 predeclared checks on validation126: global labels agree100%,
selected detector boxes agree100%, native-mask agreement differs at2 of36,679,680 pixels.
This is export consistency, not evidence that the predictions are correct.

At the unchanged global threshold0.5, the Phase1 classifier flags **0/105** edited CSI validation
images. Its original global Real/Fake task differs from local edit localization. A global Real
label is not evidence that local edits are absent. Every branch runs independently.

The validation-selected image review policy is stored in `decision_policy.json`; it combines
global/detector flags with a minimum native mask-area fraction. Selection maximizes balanced
accuracy on validation because the split contains105 edited and21 pristine images. Report
macro-F1 and sensitivity/specificity alongside accuracy/positive F1. The review-area rule does
not remove mask pixels or retune the segmentation threshold. Validation selection is not a
held-out performance estimate, and family-level correlations reduce effective sample size.

Original Phase1 acceptance criteria remain accuracy≥90% and macro-F1≥90%. The corrected main
Phase1 test reached approximately68.60% accuracy and65.72% macro-F1, so those criteria were not met.
The teammate independent rerun is still pending. Better segmentation overlap does not replace
the original classifier criteria. Preliminary stance: **NO-GO for automatic authenticity
decisions**; a constrained human-review demonstration remains the supported research use.

## Service and benchmark

Local FastAPI HTTP service, ONNX Runtime CPU, FP32, batch1, one process/concurrency1.
The actual benchmark uses10 validation warmups and126 timed native-PNG requests, measures
per-stage and direct HTTP p50/p95/p99, serial active-request throughput and sampled server RSS.
See `evidence/http_benchmark.json` for hardware, versions, timer boundaries and actual values.
No TensorRT, FP16/INT8, production load testing or public hosting is claimed.

A generated12-frame validation-image slideshow is used only as a functional video replay.
Its resize/codec change pixels; no video accuracy or threshold tuning is performed. Its
playback FPS is separate from the measured processing FPS.

## Out-of-scope uses and failures

Do not use this prototype alone for enforcement, identity decisions, accusations, or deciding
whether media is authentic. It can miss small edits, confuse broad editing masks with visible
objects, create false mask regions on pristine images and fail outside the training domains.
Confidence scores are uncalibrated. No geographic, demographic or Saudi operational fairness
or accuracy claim is supported. A missing box, empty mask or no-model-flag decision is not proof.


## Final Phase2 test addendum

### Cached test measurements

Release `real-abshar-m8-fbaa1c95818d`; protocol `bcd076d95a56df90797e0c2c6946da430378159d3fb8ba45fd687b764b3672e7`.

Test114 images/19 source families; classifier224×224, detector/segmenter640×640; FP32,batch1. Classifier0.5; detector confidence>0.35,NMS0.45,max50,operating matchIoU0.50; segmenter pixel>=0.5,no morphology; review area>=0.005.

| Measure | Result |
|---|---:|
|Detector AP50 / AP50–95, input coordinates, confidence floor0.001|57.49% / 36.71%|
|Detector P / R / F1 at frozen operating point|72.73% / 50.53% / 59.63%|
|Foreground IoU / Dice|94.20% / 97.01%|
|Two-class mIoU / mean Dice|96.72% / 98.32%|
|Mean Dice over edited images|86.18%|
|Mean absolute native area-fraction error|0.003673|
|Image referral accuracy / macro-F1|87.72% / 79.57%|
|Image referral sensitivity / specificity|90.53% / 73.68%|

Primary AP is scored before native clipping. Secondary served-native AP is retained in JSON for transparency; the earlier M4 validator/training comparison remains separately labelled. No test confidence threshold was chosen. All figures and confidence intervals derive from cached predictions; plotting does not repeat model inference.

Original Phase1 accuracy/macro-F1 criteria were not met on its closed test. These Phase2 targets do not replace them. Preliminary decision: NO-GO for automatic authenticity decisions; human-review prototype only. No Saudi operational accuracy or natural-video accuracy claim is supported. The teammate independent Phase1 rerun remains pending.
