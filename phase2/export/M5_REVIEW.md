# M5 verified training and validation review

Reviewed input: `Real-Abshar_Phase2_M5_Results_Review.zip`, 2,105,297 bytes, 48 members.
The archive is readable. Its seven source hashes match its run protocol; the cohort UID
checksum and source assignments match M4. Independently recomputing all 38 model/threshold
combinations from the per-image counts reproduces the selected architecture and threshold.
The aggregate metrics, area errors and size slices also reproduce the submitted values.
The actual checkpoint tensors are not present in the review ZIP; their recorded hashes
and the SDK-confirmed backup receipt are used to retrieve and check them in the next notebook.

## Cohort and targets

The retained dataset is `csi-complete-aligned-families-v1`: 132 complete source families,
792 images. Train: 552 images / 92 families; validation: 126 images / 21 families;
reserved test: 114 images / 19 families. Validation contains 105 edited and 21 pristine images.
No source/group/RGB hash/exact pHash crosses splits in the recorded manifests.
This does not rule out every possible hidden near-duplicate relationship.

The targets are the reviewed native 0/1 editing/inpainting masks, not the semantic-relevance
masks. They can describe broad edited regions rather than the visible object silhouette.
The original 336 rejected edit/mask pairs remain rejected. The complete-family restriction
remains unchanged. No positive training/validation mask disappeared under the recorded
nearest-neighbor target resize. GPU smoke checks passed for shapes, finite gradients,
padding exclusion and an all-pristine batch.

## Models, development and selected settings

Both models use an ImageNet-initialized ResNet18 encoder. Their before-task-fine-tuning
baselines have random task decoders; they are not pretrained manipulation segmenters.
Training: AdamW, learning rate0.0003, weight decay0.0001, batch4, seed42, FP32,
maximum25 epochs, patience6, cosine schedule, paired deterministic horizontal flips,
and 0.5 valid-pixel BCE plus0.5 positive-image soft Dice. Padding contributes no loss.
Each checkpoint is chosen by validation foreground IoU at pixel threshold0.5;
the declared validation threshold grid0.05–0.95 is then applied to the selected checkpoints.

All results in this section use **validation126, input640×640, native-coordinate scoring,
no morphology or connected-component removal**. RGB uses centered black letterboxing,
PIL bilinear resizing and ImageNet normalization. Masks use nearest-neighbor training resize.
At scoring, crop padding from probabilities, resize probabilities to native dimensions,
then threshold and compare against the untouched native 0/1 mask.

| Model | Selected epoch / completed epochs | Tuned pixel threshold | Foreground IoU | Foreground Dice | Two-class mIoU |
|---|---:|---:|---:|---:|---:|
| A: FCN ResNet18 |11 /17|0.30|92.13%|95.90%|95.52%|
| B: U-Net ResNet18 |25 /25|0.50|93.02%|96.39%|96.03%|

At the common fixed threshold0.5, A has foreground IoU91.94% and Dice95.80%; B has
IoU93.02% and Dice96.39%. B wins both the common-threshold and declared tuned-IoU comparisons.
Before task fine-tuning at threshold0.5, A has foreground IoU0.00% / Dice0.00%, and B
IoU6.49% / Dice12.20%. These are same-split development comparisons, not test results.
Only one training seed was run per architecture; no significance or multi-seed stability claim is made.

Selected model: `B_UNet_ResNet18`, native checkpoint SHA-256:
`301cae566f84ec9ffa7753373320e90bb9a0af70d64e85a8d3d89b2a92481b2c`.
Keep its pixel threshold0.5 and no-postprocessing setting fixed for M8.

## Selected model metrics and limitations

The following table uses **validation126, input640×640, pixel threshold≥0.5,
no postprocessing, native-size ground truth**.

| Quantity | Value |
|---|---:|
| Foreground TP / FP / FN / background TN |4,165,928 /181,022 /131,418 /32,201,312|
| Foreground IoU / Dice |93.02% /96.39%|
| Background IoU / Dice |99.04% /99.52%|
| Two-class mIoU / mean Dice |96.03% /97.95%|
| Foreground pixel precision / recall |95.84% /96.94%|
| Mean Dice across the105 edited images |84.07%|
| Pristine images with any false foreground |7 /21 (33.33%)|
| Mean false foreground fraction on pristine images |0.214% of image area|
| Mean absolute edited-area fraction error, all126 images |0.005434 (0.5434 percentage points)|
| Median relative edited-area error,105 edited images |5.05%|

Pooled foreground metrics weight large edited regions strongly. The mean over edited images
is lower, and seven edited validation images have zero foreground Dice. Small regions remain
a clear failure mode. Size bins below use **foreground mask area after scaling to640**:
small<32², medium32²–96², large≥96²; these differ from M4's union-box area bins.

| Edited-region size group | Validation images | Pooled foreground IoU | Pooled foreground Dice |
|---|---:|---:|---:|
| Small |6|26.61%|42.03%|
| Medium |30|63.54%|77.71%|
| Large |69|94.35%|97.09%|

These are small, related-image samples from21 source families. They do not establish
Saudi-domain deployment performance. The strongest numerical result is not a basis for
calling every image authentic or fake.

## Integration consequence

Run segmentation on every accepted input independently of the detector. M4's validation
recall is55/105 edited regions at confidence0.35, NMS0.45, matching IoU0.50 and input640×640;
detector gating would suppress segmentation for50 edited validation images.

Do not turn "any predicted foreground pixel" into the service's final fake-image label:
7/21 pristine validation images have at least one such pixel. M8 Part2 should choose an
image-level review rule using validation only, while keeping the selected pixel threshold0.5.
Return classifier score, boxes and mask/area separately so their different label scopes remain clear.

## Backup and remaining work

The submitted receipt records SDK-confirmed stage `M5_completed`,48 curated files,
artifact `f1k80-kaust/Real-Abshar-Phase2/real-abshar-m5-backup-6ea92ac6178c:v15`.
Selected entry: `runs/B_UNet_ResNet18/weights/best.pt`.
This review did not authenticate to W&B or independently download that remote artifact.
The export notebook retrieves it and checks the exact selected weight hash before loading it.

Keep `Real-Abshar_Phase2_M5_Models.zip` and saved Kaggle outputs privately.
The Phase2 test split remains unevaluated. M8 service/benchmarks, the single final test,
Phase2 code publication, model card and Overleaf report remain to be completed.
The Phase1 main correction/publication is complete; the teammate independent rerun is still pending.
