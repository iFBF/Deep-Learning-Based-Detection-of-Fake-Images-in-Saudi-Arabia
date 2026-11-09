# M8 export review and service handoff

Reviewed upload: `Real-Abshar_Phase2_M8_Export_Review.zip`,98,234 bytes,36 members.
The ZIP is readable. Independently checked protocol/source/asset/cache hashes, exact validation
UID order, and equality of the M8 and M5 validation manifests. Recomputed the entire parity
summary from the126 saved per-image records: exact match. Verified bytes/hashes for34 metadata
files against the backup inventory. The three actual ONNX graph files are not in the small
review ZIP; their recorded hashes/sizes agree among receipts, manifest and backup inventory.
The service notebook will verify the actual graph bytes again before loading them.

## Export result

**All13 predeclared export checks passed.** Conditions: validation126/21 source families,
classifier224×224, detector/segmenter640×640, batch1, FP32, CPUExecutionProvider;
classifier threshold0.5, detector confidence0.35/NMS0.45/max50, segmenter pixel threshold0.5,
no morphology, native-coordinate scoring. The global label and localization targets have
different meanings; agreement below measures export consistency, not predictive correctness.

| Check | Measured |
|---|---:|
| Classifier maximum logit absolute error |0.0000810623|
| Classifier maximum fake-score absolute error |0.000000321877|
| Classifier label agreement |126/126|
| Detector raw box-component maximum error at640 |0.00479126 pixels|
| Detector maximum score error |0.00000315905|
| Selected detector box agreement |100%|
| Custom adapter versus official Ultralytics agreement |100%|
| Detector operating TP/FP/FN |55/8/50, unchanged|
| Segmentation maximum logit absolute error |0.000514597|
| Native-mask differing pixels |2/36,679,680|
| Native-versus-ONNX pooled foreground IoU/Dice difference |0/0|

The CPU reference differs from the earlier GPU M5 aggregate by one foreground true-positive
pixel, within the declared tolerance. ONNX foreground IoU0.9302332903 and Dice0.9638558147.
This is numerically consistent with the earlier selected M5 result, not a new model selection.

Export graphs: classifier351,810,074 bytes, detector10,564,809 bytes, segmenter49,846,361 bytes.
FP32, fixed batch1, opset17, no embedded detector NMS. Exact SHA-256 values are in
`proof/export_manifest.json` and the generated service manifest.

Recorded runtime: Kaggle CPU, Intel Xeon@2.20GHz,4 logical CPUs, Python3.13.18,
Torch2.11.0+cpu/torchvision0.26.0+cpu, Transformers5.16.1, ONNX1.19.1, ONNX Runtime1.31.1,
two intra-operation and one inter-operation thread. These export comparisons are not
latency benchmarks. The next notebook measures the real serving process/path separately.

## Preservation

The supplied SDK-confirmed receipt records37 curated files and this completed artifact:
`f1k80-kaust/Real-Abshar-Phase2/real-abshar-m8-export-b90893924c38:v0`.
This review did not independently authenticate to W&B or download its graph bytes.
The next notebook can restore those exact entries and checks their hashes before use.
Keep the private ONNX Release ZIP as an additional copy.

## Validation-only review-rule development

At the unchanged global threshold0.5, the classifier flags0/105 edited and0/21 pristine
CSI validation images. It must not gate localization. Each branch runs for every image.

The service kit evaluates the recorded area grid0,0.00001,0.000025,0.00005,0.0001,0.00025,
0.0005,0.001,0.0025,0.005,0.01,0.025,0.05. The candidate rule is global flag OR any selected
detector box OR a sufficiently large native predicted mask. Selection maximizes validation
balanced accuracy; ties prefer sensitivity, macro-F1, then a higher area threshold.
This is validation development, not preregistered inference or final-test evidence.

Selected minimum mask-area fraction **0.005=0.5% of image area**. The pixel probability
threshold remains0.5 and mask pixels are not removed. At the selected combined review rule:

| Validation image-referral quantity | Value |
|---|---:|
| TP/FP/FN/TN |96/2/9/19|
| Edited-image sensitivity |91.43%|
| Pristine-image specificity |90.48%|
| Balanced accuracy |90.95%|
| Accuracy |91.27%|
| Edited-referral F1 |94.58%|
| Two-class macro-F1 |86.07%|

These image-level referral metrics differ from detection IoU matching and pixel mask overlap.
Do not present the94.58% positive F1 as macro-F1. The criterion was selected on these same
21 validation families, so it is not an unbiased held-out performance estimate. The original
Phase1 acceptance criteria remain unchanged and were not met by its corrected main test.

## Remaining requirement boundary

The export step is complete. The service kit supplies the next implementation, but actual
HTTP validation, measured latency/memory, video replay and new service backup still need the
Kaggle execution. The Phase2 test remains unopened. After service review, freeze the final
test runner, evaluate once, finish the model card/failure gallery and compile the supplied
Overleaf template. Team size3: delete Section5 and the associated open-world rows.
Phase2 source publication/actual commit and the teammate Phase1 independent rerun remain pending.
