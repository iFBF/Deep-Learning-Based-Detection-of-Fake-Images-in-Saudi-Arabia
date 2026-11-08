# Dataset card — corrected Phase 1 protocol

## Sources and scope

Prepared indoor/outdoor subset of [Renyang/DANI](https://huggingface.co/datasets/Renyang/DANI), originally distributed as [Kaggle anoudmohammd/data-dfd](https://www.kaggle.com/datasets/anoudmohammd/data-dfd). Real=0 maps from publisher reference=True; Fake=1 maps from False. Original fake images are GALIP; the fresh test uses DFGAN. No Saudi-event deployment result is established.

## Cleaning and verified source grouping

Original 1,800 images: unreadable 0; duplicate copies removed 402; retained 1,398. There are 499 original COCO parent IDs. One identical fake connects two parent families, producing 498 connected source groups. No pHash pair at Hamming distance <=4 remained after deduplication. Captions were resolved from original DANI filenames and checked against the publisher's caption-to-image dictionary. A DANI row index is never treated as a COCO image ID.

Development splits: train 1,165 images / 415 groups (416 Real, 749 Fake); validation 233 images / 83 groups (83 Real, 150 Fake). All historical images are treated as development data. The previously reported exploratory test is not an unseen holdout.

The new indoor/outdoor cohort contains 328 images in 68 source groups (68 Real, 260 DFGAN Fake), after removing two duplicate copies. It shares no parent groups, decoded pixel hashes or pHash<=4 candidates with the historical data. Fahad evaluated this fixed cohort in Kaggle on 8 November 2026. It is now closed to tuning; the completed evaluation is versioned under `docs/phase1_corrections/completed_run/`. DFGAN is held out from training. Full unbalanced evaluation is primary; a fixed balanced secondary subset uses the same final predictions.

## Recorded audit

A fixed seed-42 random sample of 100 retained original images was independently compared with original DANI parquet image bytes and reference fields. All 100 decoded images matched exactly; 0 label disagreements were found. `label_audit.csv` contains the per-image decision, reviewer/method, timestamp, publisher evidence and notes. The method is automated publisher-record checking; no human visual review is claimed for these new decisions. The one-sided 95% Clopper-Pearson upper bound is 2.9513% for source-label disagreement. This does not prove real-world authenticity of every publisher-labelled photograph.

## Processing, limitations and privacy

Model input is consistently 224x224 RGB. Physical batch eight, training-only flip and brightness/contrast jitter, and saved Swin processor normalization are implemented in the clean workflow. Class weights are computed from training counts only. The completed unweighted/weighted scratch ablation uses identical settings and initialization: Real recall increases from 48.53% to 61.76% and macro-F1 from 76.72% to 78.65%, while overall accuracy decreases from 87.20% to 86.59%.

The publisher scene field is inherited, not a separately audited literal scene description. Source-generator shortcuts, unknown pretrained-data overlap, limited generator coverage and geographic bias remain. A held-out DFGAN test does not establish performance on every generator or Saudi security events.

DANI lists CC BY-NC 4.0; the prepared export's underlying rights and individual consent are not independently certified. Full datasets, downloaded parquet files and credentials are excluded from Git. Only required metadata and aggregate/prediction evidence are committed. The main GPU run and final metrics are complete. Another member's independent training/reproduction remains outstanding. The validation-selected full model scores 68.60% primary accuracy (95% group-bootstrap CI 62.54–74.56%); the frozen comparison scores 88.72%. These expose a generator-generalization limitation, not a geographic deployment result.

See [the corrections guide](../docs/phase1_corrections/README.md) and its versioned evidence for full records.
