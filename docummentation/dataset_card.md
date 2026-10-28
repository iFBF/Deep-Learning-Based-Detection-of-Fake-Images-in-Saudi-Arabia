# Dataset Card
### 1. Dataset Overview
Dataset Name: Indoor/Outdoor Real/Fake Subset of Renyang/DANI  
Source: https://huggingface.co/datasets/Renyang/DANI  
Purpose: Academic binary image classification to distinguish real photographs from AI-generated images.
Image Format: RGB PNG
Labels:
 1. Real = 0
 2. Fake = 1

### 2. Data Source and Verification
The dataset was prepared from the Renyang/DANI dataset available on Hugging Face.
We reviewed the dataset documentation and original download code to verify the source metadata and label mapping.

The reference field determines the binary labels:
1. True = Real (0)
2. False = Fake (1)

The source_model and source_index fields preserve metadata provided by DANI, while relative_path identifies each saved image.

Unique source indices do not necessarily guarantee independence between related images. This limitation is documented.

### 3. Data Cleaning and Preprocessing
The data preparation pipeline includes:
1. EXIF orientation correction
2. RGB conversion
3. PNG export
4. Removal of exact duplicate images
5. Perceptual similarity checks
6. Grouping of related images
7. Group-aware train, validation, and test splitting

Original images: **1,800**
Removed exact duplicates: **402**
Retained images: **1,398**


### 4. Manual Label Audit

| Check | Result |
|---|---|
| Audit Sample Size | 100 |
| Images Reviewed | 100 |
| Samples Pending Review | 0 |
| Reviewer | Group Members |
| Review Method | Manual Visual Inspection |
| Manual Label Audit Status | Complete — Confirmed by Group Members |
| Review Scope | Visual Review; Independent Provenance Verified |
| Reported Label Disagreements | 0 |

All 100 sampled images were manually reviewed by the group members, with no reported label disagreements.

This review supports the recorded label audit but does not independently establish generation provenance.


## 5. Dataset Class Distribution

| Split | Real (0) | Fake (1) | Total |
|---|---:|---:|---:|
| Train | 339 | 592 | 931 |
| Validation | 78 | 156 | 234 |
| Test | 82 | 151 | 233 |
| **Total** | **499** | **899** | **1,398** |
The complete cleaned dataset exhibits class imbalance, with more fake images than real images.

### 6. Experimental Dataset Distribution
The experiments used balanced Real/Fake subsets for validation and testing.

| Split | Real (0) | Fake (1) | Total |
|---|---:|---:|---:|
| Train | 339 | 592 | 931 |
| Validation | 78 | 78 | 156 |
| Test | 82 | 82 | 164 |
| **Total** | **499** | **752** | **1,251** |

The full training split was retained, while validation and test subsets contained equal numbers of real and fake images.

### 7. Data Splitting and Reproducibility
Random Seed: 42

Related images were grouped before splitting to reduce potential data leakage.

The same fixed experimental splits were used across model comparisons.

Data augmentation was restricted to the training stage.

Although grouping and duplicate checks were performed, complete independence between all shared-source images has not been established.

### 8. License and Privacy
8. License and Privacy

Dataset-level license: CC BY-NC 4.0, as documented for DANI.

The rights of underlying source images and individual consent have not been independently verified.

The dataset is intended for academic research. Redistribution and other uses require appropriate consideration of licensing and privacy limitations.

### 9. Intended Use and Limitations
9. Intended Use and Limitations

Intended use: Academic classification of real versus AI-generated indoor and outdoor images.

Limitations:
1. Class imbalance in the cleaned dataset
2. Potential source-specific shortcuts
3. Incomplete verification of shared-source independence
4. Unknown overlap with pretrained models' training data
5. Exploratory evaluation due to previous development use of local data
6. No established validation for real-world Saudi security events

The dataset should not be treated as sufficient evidence for operational security decisions.

### 10. Documentation and Verification Status


| Check | Status |
|---|---|
| Source Metadata and Label Mapping | Complete — Documented |
| Automatic Grouping Checks | Complete |
| Source IDs, Paths, and Duplicate Checks | Complete, with documented limitations |
| Manual Label Review | Complete — 100/100 Reviewed |
| Dataset Card Documentation | Complete |
| License and Privacy | Documented with limitations |
| Random Seed | 42 |
| Pretrained Label Aggregation | Reviewed During Model Loading |

Note: The saved notebook also contains an earlier audit-gate message requesting recalculation or checking of the review file. The subsequently supplied manual-audit summary reports completion; the two records are preserved as distinct pieces of evidence.
