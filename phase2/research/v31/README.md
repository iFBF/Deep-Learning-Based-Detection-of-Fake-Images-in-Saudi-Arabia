# Real-Abshar V3.1 research extension

This source snapshot accompanies the frozen original Phase 2 detector/segmenter/service. It adds a supplemental whole-image generation model and a browser interface that places the combined review recommendation above component scores.

The team trained the original Swin, YOLO11n and segmentation models. The supplemental Thermostatic frontier checkpoint was trained externally. Team work on this extension comprises model/preprocessing review, development evaluation, operating-point selection, FP32 ONNX export/parity checking and integration. See MODEL_CARD.md and THIRD_PARTY_NOTICE.md for exact provenance and limitations.

## Evidence and scope

| Global comparison | Original validation 233: accuracy | Additional development 72: accuracy |
|---|---:|---:|
| Original Swin | 99.57% | 47.22% |
| Rejected V2 frozen-CLIP linear head | 93.13% | 41.67% |
| V3 policy | 98.71% | 87.50% |
| V3.1 policy, cached recomputation | 98.71% | 88.89% |

Conditions: original COCO/GALIP validation and reused 36-real/36-generated additional development images (14 conservative scene/series groups). Swin and V2 CLIP inputs224×224; supplemental384×384; all FP32. Baseline/V2 thresholds.5; V3/V3.1 Swin.99 plus supplemental raw4.0/4.25. This table excludes local box/mask flags. It is not a new independent test.

The actual V3 CSI joint validation failed macro-F1 preservation: 84.56% versus86.07% baseline. After an explicitly adaptive coarse/refined validation search, cutoff4.25 restores all baseline CSI validation decisions. That is cached policy recomputation, not a new live V3.1 cohort run. The failed original gate is retained. `development_results.json` contains the aggregate records and all refinement candidates; private per-image caches remain in the research archive.

CSI conditions:126 validation images/21families,105edited/21pristine; C224/supplemental384/D,S640, FP32. Original Swin.5; V3/V3.1 Swin.99 and raw4/4.25. Original local flags use box confidence.35,NMS.45,max50,mask pixel.5,minimum mask-area.005. V3.1 cached referral CM[[19,2],[9,96]]; rowspristine/edited,columnsunflagged/referred.

## Run or verify

- `python phase2/research/v31/verify_public_v31.py` checks the source snapshot and aggregate arithmetic without images, weights or inference.
- Follow RUN_LOCAL.md to use the private, checksum-verified V3.1 candidate and original M8 service. Runtime source is published under `runtime/`; the original candidate contains the model and private evidence needed for its strict release verification.
- The portal listens on loopback8001 and calls the original service on8000. It uses the exact frozen V3.1 client from the supplied private candidate. No original service model or threshold is overwritten.
- `log_development_evidence.py` records cached historical development evidence to the existing W&B project and backs up the image-free research release. It performs no training or new evaluation. Only a successful returned receipt establishes upload completion.

`windows_demo_summary.json` records one user-provided Windows result from the known squirrel/conductor demonstration. It is a smoke test, not an accuracy estimate or p95 benchmark. The full-resolution mask was checked against its foreground count. Browser QA itself used labelled synthetic fixtures.

## Remaining limits

The additional development accuracy and macro-F1 remain below90%. The72images repeatedly influenced selection; external training overlap with public sources cannot be excluded. Publisher robustness failures and original small-edit weaknesses remain. Scores are screening evidence, not calibrated authenticity probabilities. Automatic authenticity decisions remain NO-GO. The original closed Phase1/Phase2 tests remain closed.

This directory contains reviewed source and aggregate results. Raw photos, masks, individual prediction caches, model binaries, private report/student IDs and credentials are supplied only through authorized private artifacts. The report source includes an optional frozen future-holdout plan; no new untouched cohort has been collected or evaluated.
