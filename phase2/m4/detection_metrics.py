"""Operating-point metrics for one edited-region class and at most one GT box per image."""
from collections import Counter
import json
import math


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0


def operating_metrics(rows, predictions, confidence, matching_iou=0.5, input_size=640):
    totals = Counter()
    per_image = []
    sizes = {name: Counter(instances=0, matched=0) for name in ["small", "medium", "large"]}
    for row in rows:
        gt = json.loads(row["box_xyxy_exclusive"])
        boxes = sorted((p for p in predictions.get(row["uid"], []) if float(p["score"]) >= confidence),
                       key=lambda p: (-float(p["score"]), p["box"]))
        local = Counter(tp=0, fp=0, fn=0, duplicate=0, localisation=0, background=0)
        matched = False
        for prediction in boxes:
            box, score = prediction["box"], float(prediction["score"])
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or not math.isfinite(score):
                raise ValueError("Non-finite or malformed prediction.")
            if int(prediction.get("class", 0)) != 0:
                raise ValueError("This evaluator expects the single edited_region class.")
            iou = box_iou(box, gt) if gt is not None else 0.0
            if gt is not None and iou >= matching_iou and not matched:
                local["tp"] += 1
                matched = True
            else:
                local["fp"] += 1
                local["duplicate" if gt is not None and iou >= matching_iou else
                      "localisation" if gt is not None and iou >= 0.1 else "background"] += 1
        if gt is not None:
            local["fn"] = int(not matched)
            scale = input_size / max(int(row["width"]), int(row["height"]))
            area = (gt[2] - gt[0]) * (gt[3] - gt[1]) * scale * scale
            bin_name = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
            sizes[bin_name]["instances"] += 1
            sizes[bin_name]["matched"] += int(matched)
        totals.update(local)
        per_image.append({"uid": row["uid"], "source_id": row["source_id"], "kind": row["kind"],
                          "manipulation": row.get("manipulation", ""), **local})
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
              "confidence": confidence, "matching_iou": matching_iou,
              "duplicate_fp": totals["duplicate"], "localisation_fp": totals["localisation"],
              "background_fp": totals["background"], "images": len(rows),
              "recall_by_size": {name: {**value, "recall": value["matched"] / value["instances"]
                                 if value["instances"] else None} for name, value in sizes.items()}}
    return result, per_image


def threshold_sweep(rows, caches, input_size=640):
    values = []
    thresholds = [0.01, 0.025] + [round(i * 0.05, 3) for i in range(1, 20)]
    for nms_iou, predictions in sorted(caches.items()):
        for confidence in thresholds:
            metrics, _ = operating_metrics(rows, predictions, confidence, input_size=input_size)
            values.append({key: metrics[key] for key in ["tp", "fp", "fn", "precision", "recall", "f1", "confidence"]}
                          | {"nms_iou": nms_iou, "split": "val", "input_size": input_size})
    selected = max(values, key=lambda r: (r["f1"], r["recall"], r["precision"], r["confidence"], -r["nms_iou"]))
    return selected, values
