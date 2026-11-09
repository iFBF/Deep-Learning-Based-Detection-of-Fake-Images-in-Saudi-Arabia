"""Exact image/mask pairing, letterbox geometry and train/validation-only exports."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
from PIL import Image

from cohort_data import (COHORT_NAME, COHORT_UID_SHA256, EXPECTED, SOURCE_MANIFEST_SHA256,
                         SOURCE_ASSIGNMENT_SHA256, dump, read_preparation, safe_path,
                         verify_row, write_csv)

INPUT_SIZE = 640
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def geometry(width, height, size=INPUT_SIZE):
    scale = size / max(width, height)
    w = min(size, max(1, int(round(width * scale))))
    h = min(size, max(1, int(round(height * scale))))
    return {"width": w, "height": h, "left": (size - w) // 2, "top": (size - h) // 2, "size": size}


def transform(image, mask, flip=False, size=INPUT_SIZE):
    image = image.convert("RGB")
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (image.height, image.width) or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("Expected an aligned native 0/1 mask.")
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = np.fliplr(mask).copy()
    g = geometry(image.width, image.height, size)
    rgb = np.zeros((size, size, 3), np.uint8)
    target = np.zeros((size, size), np.uint8)
    valid = np.zeros((size, size), np.uint8)
    sy, sx = slice(g["top"], g["top"] + g["height"]), slice(g["left"], g["left"] + g["width"])
    rgb[sy, sx] = np.asarray(image.resize((g["width"], g["height"]), Image.Resampling.BILINEAR))
    target[sy, sx] = np.asarray(Image.fromarray(mask).resize((g["width"], g["height"]), Image.Resampling.NEAREST))
    valid[sy, sx] = 1
    normalized = (rgb.astype(np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD
    return normalized.copy(), target[None].astype(np.float32), valid[None].astype(np.float32)


def restore_probability(probability, width, height, size=INPUT_SIZE):
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != (size, size) or not np.isfinite(probability).all():
        raise ValueError("Expected finite square probability map.")
    g = geometry(width, height, size)
    cropped = probability[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]]
    restored = np.asarray(Image.fromarray(cropped).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.clip(restored, 0.0, 1.0)


def flip_for(uid, epoch, seed=42):
    digest = hashlib.sha256(f"{seed}:{epoch}:{uid}".encode()).digest()
    return bool(digest[0] & 1)


def prepare_view(prepared_root, work_root):
    root, work = Path(prepared_root), Path(work_root)
    rows, omitted = read_preparation(root)
    cohort = work / "cohort"
    lock = {"cohort": COHORT_NAME, "cohort_uid_sha256": COHORT_UID_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_assignment_sha256": SOURCE_ASSIGNMENT_SHA256, "split_counts": EXPECTED,
            "source_assignments_same_as_M4": True}
    lock_path = cohort / "cohort_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Different cohort already locked in this M5 output folder.")
    dump(lock_path, lock)
    write_csv(cohort / "manifest.csv", rows)
    write_csv(cohort / "valid_records_omitted_by_cohort_rule.csv", omitted)
    for split in ["train", "val", "test"]:
        write_csv(cohort / f"{split}_manifest.csv", [r for r in rows if r["split"] == split])
    fit = [r for r in rows if r["split"] in {"train", "val"}]
    rasterization = []
    for index, row in enumerate(fit, 1):
        verify_row(root, row)
        for key in ["export_image", "export_mask"]:
            source, target = safe_path(root, row[key]), safe_path(work / "data_view", row[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with Image.open(safe_path(root, row["export_mask"])) as opened:
            mask = np.asarray(opened, dtype=np.uint8)
        g = geometry(int(row["width"]), int(row["height"]))
        resized = np.asarray(Image.fromarray(mask).resize((g["width"], g["height"]), Image.Resampling.NEAREST))
        rasterization.append({"uid": row["uid"], "split": row["split"], "kind": row["kind"],
                              "native_foreground_pixels": int(mask.sum()), "resized_foreground_pixels": int(resized.sum()),
                              "foreground_lost_in_nearest_resize": bool(mask.any() and not resized.any())})
        if index % 100 == 0 or index == len(fit):
            print(f"Verified/copied train+val masks/images: {index}/{len(fit)}", flush=True)
    write_csv(cohort / "mask_rasterization_audit.csv", rasterization)
    dump(cohort / "preflight.json", {"verified_train_val_records": len(fit), "split_counts": EXPECTED,
         "network_input": [INPUT_SIZE, INPUT_SIZE], "test_pixels_decoded_for_M5": False,
         "test_inference_performed": False,
         "positive_masks_lost_in_nearest_resize": sum(r["foreground_lost_in_nearest_resize"] for r in rasterization),
         "scoring": "restore probabilities to native dimensions; compare with untouched native 0/1 masks"})
    return rows
