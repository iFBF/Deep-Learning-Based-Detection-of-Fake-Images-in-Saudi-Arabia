"""Atomic numeric prediction records and read-only tracing of the frozen inference path."""
from io import BytesIO
from pathlib import Path
import base64
import json
import os

import numpy as np
from PIL import Image


def safe_uid(uid):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
        raise ValueError("Unsafe sample UID.")
    return uid


def record_path(directory, uid):
    return Path(directory) / (safe_uid(uid) + ".npz")


def write_record(directory, metadata, arrays):
    path = record_path(directory, metadata["uid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Do not overwrite an existing evaluated prediction: " + metadata["uid"])
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata_utf8=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), np.uint8), **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def read_record(path, row, protocol_sha):
    with np.load(path, allow_pickle=False) as opened:
        metadata = json.loads(opened["metadata_utf8"].tobytes().decode())
        values = {k: opened[k].copy() for k in opened.files if k != "metadata_utf8"}
    if metadata["uid"] != row["uid"] or metadata["protocol_sha256"] != protocol_sha or metadata["split"] != row["split"]:
        raise ValueError("Cached prediction belongs to another sample/protocol/split.")
    for key in ["source_id", "rgb_sha256", "mask_sha256", "kind"]:
        if str(metadata[key]) != str(row[key]):
            raise ValueError("Cached prediction/data identity changed: " + row["uid"] + ":" + key)
    if metadata["forward_calls"] != {"classifier": 1, "detector": 1, "segmenter": 1}:
        raise ValueError("Expected exactly one model forward per branch for this saved sample.")
    shape = (int(row["height"]), int(row["width"]))
    expected_pixels = shape[0] * shape[1]
    mask = np.unpackbits(values["mask_packed"], count=expected_pixels).reshape(shape).astype(np.uint8)
    if int(mask.sum()) != metadata["result"]["mask"]["foreground_pixels"]:
        raise ValueError("Cached mask differs from its saved prediction.")
    if values["detector_raw"].shape != (1, 5, 8400) or not np.isfinite(values["detector_raw"]).all():
        raise ValueError("Cached raw detector contract is invalid.")
    for coordinate in ["input", "native"]:
        predictions = values["ap_" + coordinate + "_predictions"]
        correct = values["ap_" + coordinate + "_correct"]
        if predictions.ndim != 2 or predictions.shape[1] != 6 or correct.shape != (len(predictions), 10):
            raise ValueError("Invalid cached AP arrays.")
    return {"metadata": metadata, "mask": mask, **values}


def evaluate_image(runtime, root, row, protocol_sha, directory):
    from cohort_data import safe_path, verify_row
    from segmentation_data import geometry
    from segmentation_metrics import counts
    from final_metrics import official_low_floor
    path = record_path(directory, row["uid"])
    if path.is_file():
        return read_record(path, row, protocol_sha), False
    verify_row(root, row)
    captured, forward_calls = {}, {name: 0 for name in ["classifier", "detector", "segmenter"]}
    original = runtime._forward
    def observer(name, value):
        output = original(name, value)
        forward_calls[name] += 1
        if name == "detector":
            captured["detector_raw"] = output.copy()
        return output
    runtime._forward = observer
    try:
        result = runtime.predict(safe_path(root, row["export_image"]).read_bytes())
    finally:
        runtime._forward = original
    if forward_calls != {"classifier": 1, "detector": 1, "segmenter": 1}:
        raise ValueError("The frozen pipeline did not execute each branch exactly once.")
    with Image.open(BytesIO(base64.b64decode(result["mask"]["png_base64"], validate=True))) as opened:
        mask = np.asarray(opened, np.uint8)
    with Image.open(safe_path(root, row["export_mask"])) as opened:
        truth = np.asarray(opened, np.uint8)
    if mask.shape != truth.shape or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("Native prediction mask is invalid.")
    g = geometry(int(row["width"]), int(row["height"]), 640)
    area_at_input = int(np.asarray(Image.fromarray(truth).resize((g["width"], g["height"]), Image.Resampling.NEAREST)).sum())
    compact = {**result, "mask": {k: v for k, v in result["mask"].items() if k != "png_base64"}}
    metadata = {"protocol_sha256": protocol_sha, "uid": row["uid"], "source_id": row["source_id"], "kind": row["kind"],
                "split": row["split"], "rgb_sha256": row["rgb_sha256"], "mask_sha256": row["mask_sha256"],
                "width": int(row["width"]), "height": int(row["height"]), "mask_area_at_input": area_at_input,
                "ground_truth_box_xyxy": json.loads(row["box_xyxy_exclusive"]),
                "segmentation_counts": counts(truth, mask.astype(np.float32), 0.5),
                "result": compact, "forward_calls": forward_calls}
    metadata["execution_environment"] = getattr(runtime, "evaluation_environment", {})
    arrays = {"detector_raw": captured["detector_raw"], "mask_packed": np.packbits(mask.reshape(-1)),
              **official_low_floor(captured["detector_raw"], row, runtime.manifest)}
    write_record(directory, metadata, arrays)
    return read_record(path, row, protocol_sha), True
