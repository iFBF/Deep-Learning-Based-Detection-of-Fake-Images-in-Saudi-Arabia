"""Export parity postprocessing and decision matching, independent of a Torch model."""
import numpy as np
from PIL import Image

from segmentation_data import geometry


def detector_input(image):
    import cv2
    image = image.convert("RGB")
    g = geometry(image.width, image.height, 640)
    pixels = cv2.resize(np.asarray(image), (g["width"], g["height"]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = pixels
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1 - x0) * max(0., y1 - y0)
    aa = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    bb = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / (aa + bb - intersection) if aa + bb - intersection else 0.


def detections(raw, width, height, confidence=0.35, nms_iou=0.45, max_det=50):
    value = np.asarray(raw, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[0] != 5 or not np.isfinite(value).all():
        raise ValueError("Expected one-class YOLO output [1,5,anchors].")
    boxes = []
    # Ultralytics 8.3.221 uses a strict confidence comparison.
    for index in np.flatnonzero(value[4] > confidence):
        cx, cy, w, h, score = map(float, value[:, index])
        boxes.append({"box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], "score": score, "class": 0})
    boxes.sort(key=lambda r: -r["score"])
    kept = []
    for prediction in boxes:
        if all(iou(prediction["box"], existing["box"]) <= nms_iou for existing in kept):
            kept.append(prediction)
            if len(kept) == max_det:
                break
    gain = 640 / max(width, height)
    # Same native-coordinate padding convention as Ultralytics ops.scale_boxes.
    pad_x, pad_y = round((640 - width * gain) / 2 - 0.1), round((640 - height * gain) / 2 - 0.1)
    for prediction in kept:
        x0, y0, x1, y1 = prediction["box"]
        prediction["box"] = [float(np.clip((x0 - pad_x) / gain, 0, width)),
                             float(np.clip((y0 - pad_y) / gain, 0, height)),
                             float(np.clip((x1 - pad_x) / gain, 0, width)),
                             float(np.clip((y1 - pad_y) / gain, 0, height))]
    return kept


def match_detections(reference, candidate, minimum_iou=0.99, score_tolerance=0.001):
    possibilities = []
    for i, a in enumerate(reference):
        for j, b in enumerate(candidate):
            if a.get("class", 0) != b.get("class", 0) or abs(a["score"] - b["score"]) > score_tolerance:
                continue
            overlap = iou(a["box"], b["box"])
            # Clipping can leave a padding-only prediction with zero area. IoU is undefined there;
            # accept only effectively identical native coordinates, not an arbitrary empty box.
            degenerate = lambda box: box[2] <= box[0] or box[3] <= box[1]
            if degenerate(a["box"]) and degenerate(b["box"]) and np.allclose(a["box"], b["box"], atol=0.001, rtol=0):
                overlap = 1.0
            possibilities.append((overlap, i, j))
    possibilities.sort(reverse=True)
    used_reference, used_candidate = set(), set()
    for overlap, i, j in possibilities:
        if overlap >= minimum_iou and i not in used_reference and j not in used_candidate:
            used_reference.add(i)
            used_candidate.add(j)
    return len(used_reference), max(len(reference), len(candidate))


def fake_score(logits):
    value = np.asarray(logits, dtype=np.float64).reshape(-1)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("Expected two finite classifier logits ordered Real, Fake.")
    exp = np.exp(value - value.max())
    return float(exp[1] / exp.sum())


def sigmoid(value):
    value = np.clip(np.asarray(value, np.float32), -80, 80)
    return 1.0 / (1.0 + np.exp(-value))
