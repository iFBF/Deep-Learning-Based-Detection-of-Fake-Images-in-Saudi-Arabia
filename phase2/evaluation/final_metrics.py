"""Fixed metrics from one saved forward per image; official Ultralytics AP, no test threshold sweep."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image


def ap_correct(predictions, gt_box):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    prediction = torch.as_tensor(predictions, dtype=torch.float32).reshape(-1, 6)
    if gt_box is None or len(prediction) == 0:
        return np.zeros((len(prediction), 10), dtype=bool)
    target = torch.tensor([gt_box], dtype=torch.float32)
    evaluator = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    return BaseValidator.match_predictions(evaluator, prediction[:, 5], torch.zeros(1),
                box_iou(target, prediction[:, :4]), use_scipy=False).cpu().numpy()


def official_low_floor(raw, row, manifest):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from segmentation_data import geometry
    settings = manifest["thresholds"]
    size = int(manifest["preprocessing"]["detector"]["size"])
    prediction = non_max_suppression(torch.from_numpy(np.asarray(raw, np.float32)).clone(),
        conf_thres=0.001, iou_thres=float(settings["detector_nms_iou"]), nc=1,
        max_det=int(settings["detector_max_detections"]))[0]
    input_boxes = prediction.cpu().numpy().copy()
    native = prediction.clone()
    width, height = int(row["width"]), int(row["height"])
    if len(native):
        native[:, :4] = scale_boxes((size, size), native[:, :4], (height, width))
    native_boxes = native.cpu().numpy().copy()
    gt = json.loads(row["box_xyxy_exclusive"])
    if gt is not None:
        g, gain = geometry(width, height, size), size / max(width, height)
        input_gt = [gt[0] * gain + g["left"], gt[1] * gain + g["top"],
                    gt[2] * gain + g["left"], gt[3] * gain + g["top"]]
    else:
        input_gt = None
    return {"ap_input_predictions": input_boxes, "ap_native_predictions": native_boxes,
            "ap_input_correct": ap_correct(input_boxes, input_gt), "ap_native_correct": ap_correct(native_boxes, gt)}


def summarize_ap(entries, coordinate="input"):
    from ultralytics.utils.metrics import ap_per_class
    arrays = [e["ap_" + coordinate + "_predictions"] for e in entries]
    correct = np.concatenate([e["ap_" + coordinate + "_correct"] for e in entries], axis=0)
    boxes = np.concatenate(arrays, axis=0)
    target = np.zeros(sum(e["metadata"]["kind"] == "edited" for e in entries), dtype=np.int64)
    if len(target) == 0:
        raise ValueError("Expected positive edited-region ground truth for AP.")
    result = ap_per_class(correct, boxes[:, 4], boxes[:, 5], target, plot=False, names={0: "edited_region"})
    ap, classes = result[5], result[6]
    if list(classes) != [0] or ap.shape != (1, 10):
        raise ValueError("Unexpected one-class AP result.")
    return {"class": "edited_region", "instances": len(target), "AP50": float(ap[0, 0]),
            "AP50_95": float(ap[0].mean()), "AP_by_matching_iou": [float(x) for x in ap[0]],
            "matching_ious": [round(0.5 + i * 0.05, 2) for i in range(10)],
            "confidence_floor": 0.001, "nms_iou": 0.45, "max_det": 50, "input_size": [640, 640],
            "coordinate_convention": "640 letterbox coordinates, before native clipping" if coordinate == "input" else "served native coordinates, clipped to image",
            "metric_implementation": "Ultralytics8.3.221 BaseValidator.match_predictions + ap_per_class",
            "internal_best_F1_threshold_PR_return_values_used": False}


def reference_native_ap(rows, cache_path):
    predictions = json.loads(Path(cache_path).read_text())["predictions"]
    entries = []
    for row in rows:
        boxes = np.asarray([p["box"] + [p["score"], p["class"]] for p in predictions[row["uid"]]], np.float32).reshape(-1, 6)
        entries.append({"metadata": {"kind": row["kind"]}, "ap_native_predictions": boxes,
                        "ap_native_correct": ap_correct(boxes, json.loads(row["box_xyxy_exclusive"]))})
    return summarize_ap(entries, "native")


def aggregate(rows, entries, manifest):
    from detection_metrics import operating_metrics
    from service_decision import binary_metrics
    from segmentation_metrics import summarize, size_slices
    settings = manifest["thresholds"]
    metadata = [e["metadata"] for e in entries]
    predictions = {m["uid"]: m["result"]["detections"] for m in metadata}
    operating, detection_rows = operating_metrics(rows, predictions, float(settings["detector_confidence"]),
                                                  matching_iou=0.5, input_size=640)
    segments = [{"uid": m["uid"], "source_id": m["source_id"], "kind": m["kind"],
                 "mask_area_at_input": m["mask_area_at_input"], **m["segmentation_counts"]} for m in metadata]
    truth = [m["kind"] == "edited" for m in metadata]
    referral = [m["result"]["decision"]["review_recommended"] for m in metadata]
    global_flags = [m["result"]["classifier"]["fake_score"] >= settings["classifier_fake_threshold"] for m in metadata]
    return {"split": rows[0]["split"], "images": len(rows), "source_families": len({r["source_id"] for r in rows}),
            "release_id": manifest["release_id"], "thresholds": settings,
            "input_resolutions": {"classifier": [224, 224], "detector": [640, 640], "segmenter": [640, 640]},
            "detector_AP_primary_input_coordinates": summarize_ap(entries, "input"),
            "detector_AP_secondary_served_native_coordinates": summarize_ap(entries, "native"),
            "detector_operating_metrics": operating, "detector_per_image_errors": detection_rows,
            "segmentation": summarize(segments), "segmentation_by_mask_size_at640": size_slices(segments),
            "image_review_referral": binary_metrics(truth, referral),
            "global_classifier_on_CSI_edited_proxy": {**binary_metrics(truth, global_flags),
                 "scope": "CSI local-edit proxy, not the original Phase1 generated-image test"},
            "reported_P_R_F1_use_frozen_operating_point": True, "test_threshold_sweep_performed": False,
            "AP_clipping_note": "Primary AP matches low-floor detections and GT in640 letterbox coordinates before clipping. "
                                "Secondary AP scores native clipped boxes; clipping/rounding can change strict-IoU AP. "
                                "Both derive from the same saved detector forward.",
            "original_Phase1_test_reopened": False}


def family_bootstrap(entries, repetitions=1000, seed=42):
    from service_decision import binary_metrics
    from segmentation_metrics import from_counts
    families = {}
    for entry in entries:
        m = entry["metadata"]
        families.setdefault(m["source_id"], []).append(m)
    groups = list(families.values())
    generator = np.random.default_rng(seed)
    values = {k: [] for k in ["referral_accuracy", "referral_macro_f1", "referral_balanced_accuracy", "foreground_iou", "foreground_dice"]}
    for _ in range(repetitions):
        sample = [m for i in generator.integers(0, len(groups), len(groups)) for m in groups[int(i)]]
        referral = binary_metrics([m["kind"] == "edited" for m in sample], [m["result"]["decision"]["review_recommended"] for m in sample])
        counts = {k: sum(m["segmentation_counts"][k] for m in sample) for k in ["tp", "fp", "fn", "tn"]}
        segmentation = from_counts(**counts)
        for key, value in [("referral_accuracy", referral["accuracy"]), ("referral_macro_f1", referral["macro_f1"]),
                           ("referral_balanced_accuracy", referral["balanced_accuracy"]), ("foreground_iou", segmentation["foreground_iou"]),
                           ("foreground_dice", segmentation["foreground_dice"])]:
            values[key].append(value)
    return {"method": "source-family percentile bootstrap, resample whole source families", "source_families": len(groups),
            "repetitions": repetitions, "seed": seed, "conditional_on_this_dataset_and_frozen_models": True,
            "intervals95": {k: {"lower": float(np.percentile(v, 2.5)), "upper": float(np.percentile(v, 97.5))} for k, v in values.items()},
            "scope": "Sampling uncertainty among these source families; no domain-shift/generalization guarantee; no retraining or new forwards"}
