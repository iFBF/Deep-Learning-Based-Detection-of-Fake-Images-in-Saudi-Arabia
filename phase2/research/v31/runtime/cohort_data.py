"""Load the reviewed preparation and preserve a complete-family Phase 2 cohort."""
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image

SOURCE_MANIFEST_SHA256 = "8e964394838aba82aa85d8e7b93551a4018f0c1d7ce1e96ade17b625aeee14b8"
SOURCE_ASSIGNMENT_SHA256 = "bcef2a4e57004f1e66645a9ea8a1d14275c3d8f0c7bbe45625760513623ab7cd"
COHORT_UID_SHA256 = "c24eb30ac2269840eb4f92e5ada2870493e7624c211a96cb8791d80ce8f40476"
COHORT_NAME = "csi-complete-aligned-families-v1"
EXPECTED = {"train": {"images": 552, "edited": 460, "pristine": 92, "sources": 92},
            "val": {"images": 126, "edited": 105, "pristine": 21, "sources": 21},
            "test": {"images": 114, "edited": 95, "pristine": 19, "sources": 19}}


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_path(root, relative):
    value = str(relative)
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe dataset-relative path: " + value)
    candidate = (Path(root) / value).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError("Dataset path escapes its root.")
    return candidate


def filter_complete_families(rows):
    families = defaultdict(list)
    for row in rows:
        families[row["source_id"]].append(row)
    complete = set()
    for source, values in families.items():
        if (len(values) == 6 and {int(r["variant"]) for r in values} == set(range(6))
                and sum(r["kind"] == "pristine" for r in values) == 1
                and sum(r["kind"] == "edited" for r in values) == 5):
            complete.add(source)
    kept = [row for row in rows if row["source_id"] in complete]
    omitted = [{"uid": row["uid"], "source_id": row["source_id"], "split": row["split"],
                "reason": "incomplete_source_family_after_annotation_QC"}
               for row in rows if row["source_id"] not in complete]
    return kept, omitted


def split_counts(rows):
    result = {}
    for split in ["train", "val", "test"]:
        part = [r for r in rows if r["split"] == split]
        result[split] = {"images": len(part), "edited": sum(r["kind"] == "edited" for r in part),
                         "pristine": sum(r["kind"] == "pristine" for r in part),
                         "sources": len({r["source_id"] for r in part})}
    return result


def read_preparation(root):
    root = Path(root)
    manifest = root / "manifest.csv"
    if not manifest.is_file() or sha_file(manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("This is not the exact 864-record prepared dataset reviewed in this chat.")
    candidates = [root / "protocol_lock.json", root.parent / "protocol_lock.json"]
    locks = [p for p in candidates if p.is_file()]
    if not locks or not any(json.loads(p.read_text()).get("assignment_sha256") == SOURCE_ASSIGNMENT_SHA256 for p in locks):
        raise ValueError("The source preparation split lock does not match the reviewed version.")
    rows = list(csv.DictReader(manifest.open()))
    kept, omitted = filter_complete_families(rows)
    digest = hashlib.sha256(json.dumps([r["uid"] for r in kept], separators=(",", ":")).encode()).hexdigest()
    if digest != COHORT_UID_SHA256 or split_counts(kept) != EXPECTED:
        raise ValueError("The complete-family cohort differs from the pre-training review decision.")
    for key in ["source_id", "group_id", "rgb_sha256", "phash"]:
        values = defaultdict(set)
        for row in kept:
            values[row[key]].add(row["split"])
        if any(len(splits) > 1 for splits in values.values()):
            raise ValueError("Cross-split relationship found: " + key)
    return kept, omitted


def locate_prepared(input_root, working_root, explicit=None):
    if explicit:
        root = Path(explicit)
        read_preparation(root)
        return root
    candidates = []
    roots = [Path(input_root), Path("/kaggle/working/real_abshar_phase2_preparation_v1")]
    for parent in roots:
        if parent.exists():
            for manifest in parent.rglob("manifest.csv"):
                root = manifest.parent
                if (root / "images").is_dir() and (root / "labels").is_dir() and (root / "segmentation_masks").is_dir():
                    try:
                        read_preparation(root)
                        candidates.append(root)
                    except ValueError:
                        pass
    candidates = sorted(set(p.resolve() for p in candidates))
    if candidates:
        if len(candidates) > 1:
            raise ValueError("Multiple reviewed prepared folders are attached. Set PREPARED_ROOT explicitly: " + str(candidates))
        return candidates[0]
    archives = sorted(p for p in Path(input_root).rglob("*.zip")
                      if p.name.startswith("Real-Abshar_Phase2_Prepared_Data"))
    matching = []
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            if "manifest.csv" in archive.namelist() and hashlib.sha256(archive.read("manifest.csv")).hexdigest() == SOURCE_MANIFEST_SHA256:
                matching.append(path)
    if len(matching) == 1:
        destination = Path(working_root) / "unpacked_preparation"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(matching[0]) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Dataset archives must not contain symbolic links.")
                target = safe_path(destination, item.filename)
                if target.suffix.lower() not in {".png", ".txt", ".csv", ".json", ".yaml"}:
                    raise ValueError("Unexpected file in the prepared-data archive: " + item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        read_preparation(destination)
        return destination
    raise FileNotFoundError(
        "Attach Real-Abshar_Phase2_Prepared_Data.zip as a PRIVATE Kaggle input, or attach the saved "
        "preparation notebook outputs. The small review ZIP and the original CSI download are not the prepared dataset.")


def verify_row(root, row):
    with Image.open(safe_path(root, row["export_image"])) as opened:
        image = opened.convert("RGB")
        image.load()
    size = (int(row["width"]), int(row["height"]))
    if image.size != size or hashlib.sha256(str(size).encode() + image.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise ValueError("Image integrity check failed: " + row["uid"])
    with Image.open(safe_path(root, row["export_mask"])) as opened:
        mask = np.asarray(opened, dtype=np.uint8)
    if mask.shape != (size[1], size[0]) or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("Invalid 0/1 segmentation mask: " + row["uid"])
    if hashlib.sha256(str(size).encode() + mask.tobytes()).hexdigest() != row["mask_sha256"]:
        raise ValueError("Mask integrity check failed: " + row["uid"])
    line = safe_path(root, row["export_detection_label"]).read_text().strip()
    expected_box = json.loads(row["box_xyxy_exclusive"])
    if row["kind"] == "pristine":
        if line or mask.any() or expected_box is not None:
            raise ValueError("Pristine image has a positive target: " + row["uid"])
    else:
        yy, xx = np.where(mask > 0)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        if box != expected_box:
            raise ValueError("Mask and box disagree: " + row["uid"])
        values = list(map(float, line.split()))
        x0, y0, x1, y1 = box
        expected = [(x0 + x1) / (2 * size[0]), (y0 + y1) / (2 * size[1]),
                    (x1 - x0) / size[0], (y1 - y0) / size[1]]
        if len(values) != 5 or values[0] != 0 or not np.allclose(values[1:], expected, atol=1e-8):
            raise ValueError("Incorrect YOLO label: " + row["uid"])


def build_training_view(prepared_root, work_root):
    prepared_root, work_root = Path(prepared_root), Path(work_root)
    rows, omitted = read_preparation(prepared_root)
    cohort_dir = work_root / "cohort"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    lock = {"cohort": COHORT_NAME, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_assignment_sha256": SOURCE_ASSIGNMENT_SHA256, "cohort_uid_sha256": COHORT_UID_SHA256,
            "rule": "retain only one pristine plus all five QC-valid edits per source; preserve existing split",
            "split_counts": EXPECTED, "selected_before_any_model_training": True}
    lock_path = cohort_dir / "cohort_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("A different cohort is already locked in this M4 work folder.")
    dump(lock_path, lock)
    write_csv(cohort_dir / "manifest.csv", rows)
    write_csv(cohort_dir / "valid_records_omitted_by_cohort_rule.csv", omitted)
    for split in ["train", "val", "test"]:
        write_csv(cohort_dir / f"{split}_manifest.csv", [r for r in rows if r["split"] == split])
    view = work_root / "data_view"
    fit_rows = [r for r in rows if r["split"] in {"train", "val"}]
    for index, row in enumerate(fit_rows, 1):
        verify_row(prepared_root, row)
        for name in ["export_image", "export_detection_label"]:
            target = safe_path(view, row[name])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(prepared_root, row[name]), target)
        if index % 100 == 0 or index == len(fit_rows):
            print(f"Verified/copied train+val {index}/{len(fit_rows)}")
    data_yaml = view / "data.yaml"
    data_yaml.write_text("path: " + json.dumps(str(view.resolve()))
                         + "\ntrain: images/train\nval: images/val\nnames:\n  0: edited_region\n")
    dump(cohort_dir / "preflight.json", {"train_val_records_verified": len(fit_rows),
         "test_image_or_mask_pixels_read": False, "test_images_copied_to_fit_view": False,
         "original_preparation_records": 864, "cohort_records": len(rows),
         "valid_records_omitted_by_complete_family_rule": len(omitted), "split_counts": EXPECTED})
    return rows, data_yaml
