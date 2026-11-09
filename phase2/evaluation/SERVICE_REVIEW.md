# M8 service review and frozen final-test handoff

Reviewed input:`Real-Abshar_Phase2_M8_Service_Review.zip`,2,961,275 bytes,56 members.
ZIP integrity passed. Checked the frozen manifest and all locked source/configuration hashes;
verified49 metadata files against the SDK-confirmed backup inventory. The three graph bytes
are not included in this small ZIP; the new notebook verifies them before loading.

Independently recomputed all reported latency means/percentiles/throughput from126 CSV rows,
the segmentation metrics from per-image confusion counts, referral metrics and the complete
validation area-threshold selection. All agree with the submitted results within float precision.
Viewed both generated figures; their splits, model input sizes and thresholds are stated.

## Frozen release

Release:`real-abshar-m8-fbaa1c95818d`.
Manifest SHA-256:`b06365983742f70d42f7ae6ff02eb805a7776f6ea02eabc816b5b4386f83133f`.
Release-lock SHA-256:`2e96338b8b2b5a9fc7f63968250c1907205e25e60377bbe0eae35d2b138f34ad`.
Confirmed model/service artifact:
`f1k80-kaust/Real-Abshar-Phase2/real-abshar-m8-service-fbaa1c95818d:v0`,52 curated files.
The supplied receipt records confirmation; this review did not authenticate to or independently
download the remote artifact. The final notebook checks actual graph bytes when restoring.

All13 HTTP contract cases passed: PNG/JPEG/WebP, EXIF orientation, decoded-content format
checking, empty/corrupt input, missing field, unsupported/animated format and byte/pixel/size limits.
All126 validation requests returned200. Compared with the reviewed export cache, classifier
scores differ0, selected detection agreement is100%, foreground-pixel-count differences are0,
and all image-review decisions agree. The native segmentation aggregate also matches.

Model settings remain classifier224×224/threshold0.5; detector640×640/strict confidence>0.35,
NMS0.45,max50; segmenter640×640/pixel threshold>=0.5/no morphology/native scoring;
image-review area threshold0.005=0.5% of native image area. Every branch runs independently.

## Actual validation HTTP benchmark

Kaggle Linux CPU, Intel Xeon@2.20GHz,4 logical CPUs; ONNX Runtime1.31.1 CPUExecutionProvider;
FP32,batch1,concurrency1,one serving process,two intra-operation and one inter-operation thread.
Ten validation warmups then126 timed native-PNG requests from21 source families.
Client timer includes multipart upload,loopback transport,server processing,response transfer
and JSON parsing. It excludes source-file reads,GT evaluation,rendering and model startup.

| Measurement | Value |
|---|---:|
| HTTP p50 |922.95ms|
| HTTP p95 |963.71ms|
| HTTP p99 |982.33ms|
| Mean HTTP request |922.67ms|
| Serial active-request throughput |1.084 images/s|
| Direct pipeline p50 /p95 |919.41 /959.78ms|
| Peak sampled server RSS |1,413.76MiB|
| Server process readiness, including imports/boot |25.76s|

RSS sampled every0.05s; short peaks may be missed. It includes model/runtime/preprocessing
memory and excludes the benchmark client. These are CPU measurements, with exploratory
tail percentiles from126 related validation inputs. No GPU/FP16/TensorRT speedup, production
load performance or public hosting is claimed. The temporary server was stopped after the run.
Portable local launch instructions are in the retained Service Release ZIP.

The actual12-frame,640×640 MJPG validation-image replay processed at0.953FPS including decode,
HTTP/all three models,overlay and video writing. Detector ONNX-forward-only rate9.169FPS
excludes other stages and is not whole-system video FPS. Playback3FPS is a separate setting.
This slideshow is a functional video test; it is not natural-motion accuracy evidence.

## Validation task performance

Validation126,21 families,C224/D,S640,thresholds as above:
image-referral TP96,FP2,FN9,TN19; accuracy91.27%,balanced accuracy90.95%,macro-F186.07%,
edited-referral F194.58%. Do not substitute positive-class F1 for macro-F1.
Foreground IoU93.02%,foreground Dice96.39%,two-class mIoU96.03%; mean Dice over105 edited
images84.07%;7/21 pristine images retain some false mask pixels. The area review rule does
not remove those pixels. The global classifier flags0/105 edited CSI validation images at0.5.
The original global task and the local editing target remain distinct.

## Final-test plan

The model/processor/source/threshold release is frozen. The new evaluator adds only read-only
raw-output capture and metric/report code. It first validates on the126 validation images and
the original M4 cached native predictions. Only a passing gate permits the reserved114-image,
19-family test. All test metrics/figures come from one saved forward per branch per image.

Detector AP uses a declared low confidence floor0.001 with NMS0.45/max50. Primary AP matches
boxes in the640 letterbox coordinate system before native clipping; secondary served-native
clipped AP is retained for transparency. These conventions can differ at strict IoUs because
of clipping/rounding. The earlier M4 training comparison used the official dataset validator.
Native AP validation compares the new outputs against the original M4 native prediction cache
using the same pinned official matching/AP functions. Operating P/R/F1 use the actual frozen
service outputs at0.35. The AP function's internally optimized P/R/F1 are never reported.

The original Phase1 acceptance criteria remain unchanged. Its corrected main test did not
meet accuracy90%/macro-F190%; it stays closed. No final Saudi operational accuracy claim is
supported. Preliminary use decision remains NO-GO for automatic authenticity decisions;
human-review demonstration only. The teammate independent Phase1 rerun remains pending.

Final test metrics,model card/failure gallery,Phase2 publication and the supplied Overleaf
report still need completion. Team3 deletes Section5 and open-world rows. Keep the retained
Service Release and local run instructions working through the final presentation.
