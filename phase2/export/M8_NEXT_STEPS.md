# Phase2 completion path after the export review

This is a three-person team. Delete Section5 (open-world extension) and the associated
open-world rows from the supplied Overleaf template. Keep the other sections and replace
every grey prompt. Do not copy the example report's numerical results.

## What this notebook completes if its runtime checks pass

- Retrieves the checksum-pinned Phase1 full Swin, M4 A YOLO11n and M5 B U-Net models.
- Exports three fixed-batch1, FP32, opset17 ONNX graphs without embedded detector NMS.
- Preserves the Phase1 saved processor, architecture and Real/Fake class order.
- Checks graph contracts and native/ONNX parity on all126 validation inputs.
- Checks the custom detector preprocessing/NMS adapter against official Ultralytics predictions.
- Records model/source hashes, settings, packages, runtime and resumable validation predictions.
- Attempts a confirmed W&B model-artifact backup and produces private review/release ZIPs.

Passing export parity permits service assembly. It does not establish deployment acceptance.
These comparison runs are not latency benchmarks: they execute both native and ONNX branches,
perform reference checks and collect metrics.

## Next after reviewing actual exported graphs

1. Build the HTTP service around these exact ONNX graphs. Define accepted image formats,
   upload/pixel limits, EXIF/RGB handling, error responses, health endpoint and prediction JSON.
   Return the global Real/Fake score, class0 region boxes, native-size mask and edited-area fraction.
   Run every model for every accepted image. Keep thresholds0.5 /0.35+NMS0.45 /0.5 fixed.
2. Select an explicit image-level review/abstention rule on validation. A minimum mask-area
   rule may suppress tiny false masks, but its effect on small edits must be reported.
   Report sensitivity/specificity or balanced accuracy alongside F1 because validation has105
   edited versus21 pristine images. The image-level classifier and region editing targets have
   different meanings; a global score is not a calibrated probability of forensic authenticity.
3. Measure real preprocessing, per-model inference, postprocessing, complete pipeline and HTTP
   request latency. Record host/CPU, providers, threads, precision, batch1, input sizes,
   warmup/timed requests, concurrency, p50/p95/p99, throughput and process memory.
   Measure end-to-end time directly rather than summing percentile values.
   FP16/TensorRT are optional; a working CPU ONNX service with measured conditions is sufficient
   for a concrete backend. Compare only the backends actually executed.
4. Provide image inference and a video/stream demonstration. A validation-image replay may
   serve as a clearly labelled functional stream demonstration; it is not a natural-video
   accuracy study. Report the actual measured FPS and conditions for the implemented path.
5. Freeze a final release manifest, model hashes, decision rule and test protocol. Then perform
   the reserved Phase2 test evaluation once. Save per-image predictions so AP, fixed-operating
   detector metrics, per-class semantic metrics, area error and final failure examples come
   from the same frozen run. Test contains114 images /19 source families. Do not tune after
   seeing that test. The already closed Phase1 test stays closed.
6. Finish the model card, failure gallery and go/no-go justification. Preserve the original
   Phase1 acceptance criteria (accuracy≥90%, macro-F1≥90%); its corrected main test did not meet
   those criteria. Do not silently replace them with segmentation overlap scores. Any new
   Phase2 criterion must be explicitly identified and justified.
7. Publish only code and permitted aggregate evidence to the repository. Keep dataset images,
   masks, checkpoint/release ZIPs, per-image manifests/predictions and secrets private.
   Record the actual Phase2 publication commit, not the older Phase1 commit as if it contained
   the new implementation. A teammate's independent rerun remains pending until its evidence arrives.
8. Compile the supplied Overleaf template. Section1 needs repository URL+actual commit,
   W&B project, working service URL or tested run instructions, and model-card link.
   Include actual team members/contributions, obstacles, checklist and self-assessment.
   Every figure caption must state its split, input resolution and relevant thresholds.
   Submit one `Real-Abshar_Phase2.pdf` and one `Real-Abshar_Phase2_source.zip` per team.

Keep the service or tested run instructions working through the final presentation.
The final go/no-go decision, failure gallery and model card are assessed there.

## Current boundary

No HTTP-service implementation, HTTP benchmark, final test result or final deployment decision
is claimed by this export kit. The release manifest deliberately leaves those fields incomplete.
The provided Overleaf template is already available; do not replace it with a new report format.
