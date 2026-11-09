"""Validation-only selection of an explicit image referral rule; model thresholds stay fixed."""
import json
from pathlib import Path

from cohort_data import dump, sha_file, write_csv

AREA_GRID = [0.0, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]


def binary_metrics(truth, predictions):
    pairs = list(zip(truth, predictions, strict=True))
    tp = sum(bool(t) and bool(p) for t, p in pairs)
    fp = sum(not bool(t) and bool(p) for t, p in pairs)
    fn = sum(bool(t) and not bool(p) for t, p in pairs)
    tn = sum(not bool(t) and not bool(p) for t, p in pairs)
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "images": len(pairs),
            "sensitivity": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "edited_referral_f1": positive_f1,
            "pristine_no_flag_f1": negative_f1, "macro_f1": (positive_f1 + negative_f1) / 2,
            "accuracy": (tp + tn) / len(pairs) if pairs else None}


def decide(fake_score, boxes, area_fraction, settings):
    area_threshold = float(settings["minimum_edited_area_fraction"])
    area_flag = area_fraction > 0 if area_threshold == 0 else area_fraction >= area_threshold
    flags = {"global_generation_score": fake_score >= settings["classifier_fake_threshold"],
             "edited_region_box": bool(boxes), "edited_area": bool(area_flag)}
    reasons = [key for key, enabled in flags.items() if enabled]
    return {"status": "review_recommended" if reasons else "no_model_flag", "review_recommended": bool(reasons),
            "reasons": reasons, "flags": flags, "minimum_edited_area_fraction": area_threshold,
            "note": "Model evidence for human review. No model flag does not establish image authenticity."}


def fit_policy(cache_path, output):
    records = [json.loads(line) for line in Path(cache_path).read_text().splitlines()]
    if len(records) != 126 or {r["split"] for r in records} != {"val"} or len({r["source_id"] for r in records}) != 21:
        raise ValueError("Referral selection requires exactly the reviewed 126-image,21-family validation cache.")
    truth = [r["kind"] == "edited" for r in records]
    classifier = [r["classifier_onnx_fake_score"] >= 0.5 for r in records]
    detector = [bool(r["detector_onnx"]) for r in records]
    grid = []
    for threshold in AREA_GRID:
        masks = [r["segmenter_onnx_area_fraction"] > 0 if threshold == 0 else r["segmenter_onnx_area_fraction"] >= threshold
                 for r in records]
        combined = [c or d or m for c, d, m in zip(classifier, detector, masks, strict=True)]
        grid.append({"minimum_edited_area_fraction": threshold, **binary_metrics(truth, combined)})
    chosen = max(grid, key=lambda r: (r["balanced_accuracy"], r["sensitivity"], r["macro_f1"],
                                     r["minimum_edited_area_fraction"]))
    policy = {"rule": "classifier_flag OR any_selected_detector_box OR edited_area_flag",
              "classifier_fake_threshold": 0.5, "detector_confidence": 0.35, "detector_nms_iou": 0.45,
              "detector_max_detections": 50, "segmenter_pixel_threshold": 0.5,
              "segmenter_postprocessing": "none", "minimum_edited_area_fraction": chosen["minimum_edited_area_fraction"],
              "area_comparator": ">0 if threshold=0; otherwise >= threshold", "area_units": "fraction of native image pixels",
              "selection_split": "val", "validation_cache_sha256": sha_file(cache_path), "area_grid": AREA_GRID,
              "selection_rule": "maximum balanced accuracy; ties: higher sensitivity, higher macro-F1, higher area threshold",
              "reason": "105 edited versus21 pristine images; an always-review rule has misleadingly high positive-class F1.",
              "validation_metrics": chosen, "models_or_pixel_thresholds_reselected": False,
              "development_status": "Chosen during validation development; not a preregistered statistical analysis.",
              "original_phase1_acceptance_criteria_unchanged": True, "test_evaluated": False}
    selected_area = policy["minimum_edited_area_fraction"]
    mask_only = [r["segmenter_onnx_area_fraction"] > 0 if selected_area == 0 else r["segmenter_onnx_area_fraction"] >= selected_area for r in records]
    any_mask = [r["segmenter_onnx_area_fraction"] > 0 for r in records]
    variants = [("always_review", [True] * len(records)), ("classifier_only_at_0.5", classifier),
                ("any_detector_box_at_0.35_NMS0.45", detector), ("any_mask_pixel_at_0.5", any_mask),
                ("mask_area_only_at_selected_area", mask_only),
                ("selected_three_branch_OR", [c or d or m for c, d, m in zip(classifier, detector, mask_only, strict=True)])]
    ablations = [{"variant": name, **binary_metrics(truth, values)} for name, values in variants]
    dump(Path(output) / "decision_policy.json", policy)
    write_csv(Path(output) / "evidence" / "val_referral_area_sweep.csv", grid)
    write_csv(Path(output) / "evidence" / "val_referral_ablations.csv", ablations)
    return policy
