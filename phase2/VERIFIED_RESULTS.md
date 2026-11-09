# Verified final measurements

The uploaded final review passed 2,005 read-only numerical/provenance checks. No model was loaded and no inference was repeated during that audit. AP was recalculated with the pinned official NumPy metric functions from saved detections/GT metadata; mask metrics and family bootstrap intervals were recalculated from saved per-image confusion counts. The review ZIP does not contain the input images, GT mask files, raw-output numeric cache or model binaries. The saved W&B receipt records a confirmed 114-image completion; the remote W&B server was not independently queried during the audit.

**Conditions:** held-out test114,19 source families (95 edited,19 pristine); classifier224×224; detector/segmenter640×640; CPU ONNX Runtime FP32,batch1. Classifier fake≥0.5; detector confidence>0.35,NMS0.45,max50,operating matchIoU≥0.50; segmenter pixel≥0.5,no morphology,native mask scoring; review-area≥0.005. AP uses floor0.001 and matchingIoU0.50–0.95.

| Measure | Final test |
|---|---:|
| M4 AP50 / AP50–95 (primary input coordinates) | 57.49% / 36.71% |
| M4 AP50–95 (secondary served native coordinates) | 36.89% |
| M4 precision / recall / F1 | 72.73% / 50.53% / 59.63% |
| M4 TP / FP / FN | 48 / 18 / 47 |
| M4 FP taxonomy (duplicate / localization / background) | 0 / 3 / 15 |
| M4 small / medium / large recall | 0/5 / 2/26 / 46/64 |
| M5 foreground IoU / foreground Dice | 94.20% / 97.01% |
| M5 background IoU / Dice | 99.25% / 99.62% |
| M5 two-class mIoU / mean Dice | 96.72% / 98.32% |
| M5 mean Dice across the95 edited images | 86.18% |
| M5 small / medium / large pooled foreground Dice | 59.91% / 85.07% / 97.55% (n6/32/57) |
| M5 pristine images with any false mask pixel | 4/19 |
| Mean absolute native area-fraction error | 0.003673 (0.3673 percentage points) |
| Median relative area error among edited images | 4.19% |
| Integrated image referral TP / FP / FN / TN | 86 / 5 / 9 / 14 |
| Integrated image referral accuracy / macro-F1 | 87.72% / 79.57% |
| Integrated sensitivity / specificity | 90.53% / 73.68% |
| Integrated balanced accuracy | 82.11% |

Do not confuse positive-class referral F1 (92.47%) with macro-F1 (79.57%). Pooled pixel Dice is weighted by region size; the edited-image mean and size slices expose weaker small-edit behavior. The global Phase1 classifier alone flags5/95 CSI local edits and1/19 pristine images at0.5; this proxy differs from its original generated-image task. All branches run independently.

The final inference host was AMD EPYC7B12,4 logical CPUs. The separately measured actual HTTP benchmark remains the earlier **validation126, Intel Xeon@2.20GHz,4 logical CPUs**, CPUFP32batch1/concurrency1,10 warmups: HTTP p50=922.95ms,p95=963.71ms,p99=982.33ms,active serial throughput1.084images/s,peak sampled server RSS1413.76MiB. C224/D,S640 and thresholds as above. Video replay was a12-frame validation slideshow,0.953processingFPS; playback3FPS and detector-only speed are different quantities. Do not relabel those measurements as a test benchmark or as natural-video accuracy.

Primary AP is scored in640 letterbox input coordinates before clipping; secondary native AP uses served clipped boxes. This convention was frozen before the test. No thresholds or model choices changed on the test.

The original Phase1 criteria remain accuracy≥90% and macro-F1≥90%; its corrected primary test reached68.60% and65.72%, respectively. Phase2 image referral is a different target and does not replace those criteria. Preliminary verdict: **NO-GO for automatic authenticity decisions; human-review research demonstration only**. No Saudi operational-performance claim is supported. The final presentation must defend the verdict, failure gallery and model card.

Release: `real-abshar-m8-fbaa1c95818d`. Final protocol: `bcd076d95a56df90797e0c2c6946da430378159d3fb8ba45fd687b764b3672e7`.
