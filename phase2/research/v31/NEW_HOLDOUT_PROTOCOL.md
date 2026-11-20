# Optional untouched holdout — protocol only, not executed

No unused labelled evaluation cohort has been supplied for V3.1. There are no new independent-test results in this release. The original Phase1 and Phase2 tests remain closed. This future experiment is extra evidence for a new V3.1 performance claim, not an additional mandatory course component.

Before collecting or scoring new images, freeze the canonical V3.1 manifest/release, the original baseline release, all preprocessing and the two global cutoffs (Swin0.99 / supplemental raw4.25), along with the unchanged local flags. Record both manifests and the exact source commit. Do not use the results to choose thresholds, models, exclusions or a different main claim.

Proposed cohort:200 images,100 photographically real and100 generated, at least20 independent source/scene groups per class. Include at least two generator sources and diverse real-camera sources with documented academic-use rights. The count and composition are engineering design choices, not instructor requirements. Define source quotas and eligibility before predictions. Record id, relative path, provenance label, source/generator, scene/series group, collection date and rights information. Labels need provenance review before scoring; appearance alone cannot establish authenticity.

Exclude every previously inspected image, alternate crop/recompression of it, scene/series sibling, original project train/validation/test image, the72additional development images and the known squirrel demo. Check original cached source-group/hash/perceptual-hash ledgers plus all available CSI metadata without reopening old test pixels. A project-unused cohort does not establish that an external pretrained model never saw its source; disclose remaining uncertainty.

Audit readability and provenance, finalize the manifest, then checksum-lock it before inference. Run both frozen policies once on exactly those images. Retain per-image results atomically so interruption resumes completed records without repeating them. Do not replace failed examples after seeing predictions. Report global generation classification separately from whole-system referral; do not pool these with old CSI metrics.

Report confusion matrices, accuracy, macro-F1, per-class recall and group-bootstrap uncertainty for both global and referral decisions. Report source/size slices only if predeclared with enough examples; avoid claiming mask or box accuracy without localization ground truth. Measure warmed service latency separately under stated hardware and timer boundaries; a single demo request is not p95. Keep negative outcomes and preserve the original acceptance targets.

The unused images, labels and a new locked execution are still required to perform this experiment. No claim of completion, new generalization or improved final-test accuracy is made here.
