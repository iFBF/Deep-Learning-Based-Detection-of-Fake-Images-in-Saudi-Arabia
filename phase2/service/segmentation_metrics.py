"""Native-pixel binary segmentation metrics with explicit empty-mask handling."""
import math
import numpy as np

THRESHOLDS = [round(i / 20, 2) for i in range(1, 20)]


def counts(truth, probability, threshold):
    truth = np.asarray(truth)
    probability = np.asarray(probability)
    if truth.shape != probability.shape or truth.ndim != 2:
        raise ValueError("Native truth/probability map dimensions must match.")
    if not set(np.unique(truth)).issubset({0, 1}) or not np.isfinite(probability).all():
        raise ValueError("Expected binary truth and finite probabilities.")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Probabilities are outside [0,1].")
    gt, pred = truth.astype(bool), probability >= threshold
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(gt.size - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def from_counts(tp, fp, fn, tn):
    foreground_union = tp + fp + fn
    background_union = tn + fp + fn
    iou = tp / foreground_union if foreground_union else None
    background_iou = tn / background_union if background_union else None
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    background_dice = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else None
    return {"foreground_iou": iou, "foreground_dice": dice,
            "background_iou": background_iou, "mean_iou_2_classes": (iou + background_iou) / 2
            if iou is not None and background_iou is not None else None,
            "background_dice": background_dice,
            "mean_dice_2_classes": (dice + background_dice) / 2 if dice is not None and background_dice is not None else None,
            "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
            "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else None}


def summarize(records):
    total = {key: sum(int(r[key]) for r in records) for key in ["tp", "fp", "fn", "tn"]}
    positive = [r for r in records if r["kind"] == "edited"]
    negative = [r for r in records if r["kind"] == "pristine"]
    result = {**total, **from_counts(**total), "images": len(records),
              "edited_images": len(positive), "pristine_images": len(negative)}
    for key, metric in [("positive_image_mean_dice", "foreground_dice"), ("positive_image_mean_iou", "foreground_iou")]:
        values = [from_counts(**{k: int(r[k]) for k in total})[metric] for r in positive]
        result[key] = float(np.mean(values)) if values else None
    result["pristine_images_with_any_false_foreground"] = sum(int(r["fp"]) > 0 for r in negative)
    result["pristine_any_pixel_false_alarm_rate"] = (result["pristine_images_with_any_false_foreground"] / len(negative)
                                                      if negative else None)
    result["pristine_mean_false_foreground_fraction"] = (float(np.mean([int(r["fp"]) / (int(r["fp"]) + int(r["tn"]))
                                                                        for r in negative])) if negative else None)
    area_errors = [abs(int(r["fp"]) - int(r["fn"])) / sum(int(r[k]) for k in total) for r in records]
    positive_area_errors = [abs(int(r["fp"]) - int(r["fn"])) / (int(r["tp"]) + int(r["fn"])) for r in positive]
    result["mean_absolute_area_fraction_error"] = float(np.mean(area_errors)) if area_errors else None
    result["edited_median_relative_area_error"] = float(np.median(positive_area_errors)) if positive_area_errors else None
    return result


def choose_threshold(grid):
    return max(grid, key=lambda r: (r["foreground_iou"], r["foreground_dice"], r["pixel_precision"], r["threshold"]))


def size_slices(records):
    result = {}
    for name, low, high in [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf"))]:
        group = [r for r in records if r["kind"] == "edited" and low <= float(r["mask_area_at_input"]) < high]
        result[name] = summarize(group) if group else {"images": 0}
    return result
