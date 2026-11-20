# Real-Abshar V3.1 model card

## Intended use and status

Local research screening for human review. This is a validation-selected comparison client for the team's original M8 service. Automatic authenticity decisions remain NO-GO. Additional-development accuracy is 88.89% and macro-F1 88.85%, below the 90% target. No new independent test or live V3.1 end-to-end timing run has been performed.

The original V3 4.0 joint run failed macro-F1 preservation; that negative result is retained. V3.1's 4.25 threshold was selected using cached outputs from the same inspected validation/development data. Its cached CSI decisions match the baseline exactly. This does not establish new CSI gains or generalization.

## Model credit

Supplemental model: `Thermostatic/community-forensics-frontier-detector-2026-08`, revision `16db135220b318d811b207db576d90368980b595`; original safetensors SHA-256 `2f445aecc310caee9b55d3ae0aeebc404a32d72e5836e30523ce0a9a1720cd47`. This is an independently published fine-tune of OwensLab Community Forensics, not an official OwensLab checkpoint. The Real-Abshar team did not train the supplemental weights. Publisher claims about its newer-generator training mix were not independently audited here.

Base research: Park and Owens, Community Forensics, CVPR 2025. Official upstream checkpoint revision `OwensLab/commfor-model-384@6076002bf0d9dd37537f965ee2f06f826c333b61`. Publisher/upstream model cards identify MIT licensing; the upstream license is included. Dataset rights remain separate. No raw photos are included.

Original specialist: frozen team Swin in `real-abshar-m8-fbaa1c95818d`, ONNX SHA-256 `32cb5ce78c63aeb0061b5dd659f33b85c6a38918993670596b777aa79701abd3`. Original M4 detector and M5 segmenter remain in the unchanged private service. Their graphs are not duplicated here.

## Inputs, outputs and selected policy

Swin uses original 224×224 preprocessing. Supplemental FP32 ViT-S/16 uses RGB, PIL bilinear resize of shorter edge to 440, center crop 384×384, division by 255 and ImageNet normalization. It returns one raw logit. Supplemental ONNX SHA-256 is `5282d49db75058c9d37b4ba05337bf631e2dbe6b8d893e8561e3c52b492f1c6e`.

Global flag: **Swin fake score ≥0.99 OR supplemental raw logit ≥4.25**. Full research referral adds unchanged local edited-region-box OR edited-area flags. M4/M5 inputs are 640×640; detector confidence 0.35, NMS IoU 0.45, maximum 50 boxes; mask pixel threshold 0.5 and minimum edited-area fraction 0.005. All use FP32.

The publisher's operating point is raw logit 1.359375, with sigmoid(raw−0.7403357915937764) and probability threshold 0.65. Local threshold 4.25 is an operating-point selection, not a newly fitted probability calibration. No calibrated ensemble probability is claimed. A referral is model evidence for review; no flag is not proof of authenticity. Whole-image generation evidence does not establish the location or fraction of an edit.

Selection history: initial 20 two-model policies selected V3 at .99/4; subsequent CSI review prompted coarse raw grid {4,4.5,5,5.5,6}, then adaptive bracket refinement {4,4.25,4.5}. Both stages and their negative results are included. All selection used development/validation outputs, with original closed tests left closed.

## Development evidence

Split, resolutions and thresholds are stated above. Original global validation: 233 COCO/GALIP images, V3.1 accuracy 98.71%, macro-F1 98.61%, confusion matrix [[83,0],[3,147]] (rows real/generated; columns unflagged/flagged). Additional development: 72 supplied images (36/36, 14 conservative groups), V3.1 accuracy 88.89%, macro-F1 88.85%, confusion matrix [[34,2],[6,30]]. These are cached recomputations at the new threshold; global flags exclude local models.

Full-system CSI validation: 126 images, 21 source families, 105 edited and 21 pristine. V3.1 cached accuracy 91.27%, macro-F1 86.07%, confusion matrix [[19,2],[9,96]] (rows pristine/edited; columns unflagged/referred). Every decision equals the baseline. Uploaded live V3 4.0 macro-F1 was 84.56%, below baseline 86.07%. Prior warm p95: baseline HTTP 907.31 ms, V3 client 1246.74 ms on Kaggle CPU with two ONNX threads. No V3.1 latency measurement is claimed.

JOINT_VALIDATION_REVIEW.md gives the complete comparison and validation selection limits. ONNX/native agreement at the new operating point and fixture checks are recorded in evidence/QA_SUMMARY.json. Those establish implementation consistency, not external accuracy.

## Limitations

The 72 supplied images were already inspected and repeatedly used in selection. Prior exposure is uncertain; they are not an untouched test. Labels reflect supplied provenance, not independent authentication. External training overlap with public COCO/GALIP cannot be excluded. A small set of scene/series groups is insufficient for universal, Saudi operational or deployment performance claims.

The publisher marks its robustness validation invalid because of failures on small synthetic composites, severe low resolution and missing external attack coverage. These limits carry into this candidate. Cropping, recompression, new generators and domain shifts can change scores. Local baseline boxes/masks can also be false positives. A genuinely new, predeclared holdout would be needed for a new final-performance claim.

## Sources

- https://huggingface.co/Thermostatic/community-forensics-frontier-detector-2026-08
- https://huggingface.co/OwensLab/commfor-model-384
- https://github.com/JeongsooP/Community-Forensics
- https://jespark.net/projects/2024/community_forensics/
