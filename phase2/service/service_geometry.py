"""Manifest-driven versions of the already parity-checked preprocessing and one-class NMS."""
import numpy as np
from PIL import Image
from segmentation_data import geometry, restore_probability
from export_metrics import iou, fake_score, sigmoid


def detector_input(image, spec):
    import cv2
    size = int(spec["size"])
    g = geometry(image.width, image.height, size)
    resized = cv2.resize(np.asarray(image), (g["width"], g["height"]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), int(spec["padding_value"]), dtype=np.uint8)
    canvas[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / float(spec["scale_divisor"]))


def segmenter_input(image, spec):
    size = int(spec["size"])
    g = geometry(image.width, image.height, size)
    canvas = np.full((size, size, 3), int(spec["padding_value"]), dtype=np.uint8)
    canvas[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = np.asarray(
        image.resize((g["width"], g["height"]), Image.Resampling.BILINEAR))
    mean = np.asarray(spec["mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(spec["std"], np.float32).reshape(3, 1, 1)
    value = (canvas.astype(np.float32).transpose(2, 0, 1) / float(spec["scale_divisor"]) - mean) / std
    return np.ascontiguousarray(value[None])


def detector_boxes(raw, width, height, spec, thresholds):
    value = np.asarray(raw, np.float32)
    if value.ndim != 3 or value.shape[0:2] != (1, 5) or not np.isfinite(value).all():
        raise ValueError("Expected finite one-class detector output [1,5,anchors].")
    value = value[0]
    boxes = []
    for index in np.flatnonzero(value[4] > float(thresholds["detector_confidence"])):
        cx, cy, w, h, score = map(float, value[:, index])
        boxes.append({"box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "score": score, "class": 0})
    boxes.sort(key=lambda r: -r["score"])
    kept = []
    for box in boxes:
        if all(iou(box["box"], previous["box"]) <= float(thresholds["detector_nms_iou"]) for previous in kept):
            kept.append(box)
            if len(kept) == int(thresholds["detector_max_detections"]):
                break
    size = int(spec["size"])
    gain = size / max(width, height)
    pad_x, pad_y = round((size - width * gain) / 2 - 0.1), round((size - height * gain) / 2 - 0.1)
    for box in kept:
        x0, y0, x1, y1 = box["box"]
        box["box"] = [float(np.clip((x0 - pad_x) / gain, 0, width)), float(np.clip((y0 - pad_y) / gain, 0, height)),
                      float(np.clip((x1 - pad_x) / gain, 0, width)), float(np.clip((y1 - pad_y) / gain, 0, height))]
    return kept
