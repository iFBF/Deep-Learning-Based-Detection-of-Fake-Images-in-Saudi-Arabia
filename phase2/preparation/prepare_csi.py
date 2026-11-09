"""Prepare CSI-IMD localization data without model training or test inference."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import re
import sys
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy.fft import dctn

PHASE1_COMMIT = "26a12b0a69641990be2ef4b2794a19e420f87e20"
REFERENCE_SHA256 = "0544ea4ee4940c56874855021b0f0a5873f121ddeaf26c4880894374529eeb8b"
REFERENCE_URL = (
    "https://raw.githubusercontent.com/iFBF/"
    "Deep-Learning-Based-Detection-of-Fake-Images-in-Saudi-Arabia/"
    + PHASE1_COMMIT + "/docs/phase1_corrections/completed_run/experiment_manifest.csv"
)
SOURCE_DIR = "pristine_images_used_for_manipulation"
EDIT_DIR = "manipulated_images"
MASK_DIR = "pixel_based_truth_mask"
SEMANTIC_DIR = "semantic_relavent_pixel_truth_mask"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULTS = {
    "version": "csi-localization-v1", "seed": 42,
    "expected_sources": 200, "variants_per_source": 5,
    "phash_distance": 4, "train_percent": 70, "val_percent": 15, "test_percent": 15,
    "primary_mask_foreground": 255, "box_policy": "one union box of all foreground pixels per edited image",
}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_root(explicit, input_root="/kaggle/input"):
    def valid(path):
        return (path / "image_info.csv").is_file() and all((path / name).is_dir()
                for name in [SOURCE_DIR, EDIT_DIR, MASK_DIR])
    if explicit is not None and Path(explicit).exists():
        path = Path(explicit)
        if not valid(path):
            raise ValueError("DATA_ROOT exists but lacks image_info.csv or the observed CSI image/mask folders.")
        return path
    candidates = [p.parent for p in Path(input_root).rglob("image_info.csv") if valid(p.parent)]
    if len(candidates) != 1:
        print("CSI roots found:", [str(p) for p in candidates])
        raise FileNotFoundError("Set DATA_ROOT to the attached, unpacked CSI Gold folder shown in Kaggle Input.")
    return candidates[0]


def load_reference(path, destination):
    if path is None:
        request = urllib.request.Request(REFERENCE_URL, headers={"User-Agent": "Real-Abshar-Phase2-Data"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(2 * 1024 * 1024)
        except Exception as error:
            raise RuntimeError(
                "Could not obtain the pinned Phase 1 reference. Enable Kaggle Internet, or attach "
                "the corrected experiment_manifest.csv and set PHASE1_REFERENCE_PATH. "
                "Do not skip the overlap check. Underlying error: " + type(error).__name__
            ) from error
    else:
        payload = Path(path).read_bytes()
    if sha_bytes(payload) != REFERENCE_SHA256:
        raise ValueError("The Phase 1 reference checksum is wrong. Use the corrected manifest from commit " + PHASE1_COMMIT)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(rows) != 1726 or not {"relative_path", "sha256", "phash"}.issubset(rows[0]):
        raise ValueError("Unexpected corrected Phase 1 reference schema.")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_bytes(payload)
    return rows


def numeric_id(value):
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError("Expected a numeric CSI source ID, got " + repr(text))
    return str(int(text))


def indexed_images(folder, edited=False):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if edited:
            matched = re.fullmatch(r"(\d+)_(\d+)", path.stem)
            if not matched:
                raise ValueError("Unexpected edited-image or mask name: " + path.name)
            key = (numeric_id(matched.group(1)), int(matched.group(2)))
        else:
            key = numeric_id(path.stem)
        if key in result:
            raise ValueError("Multiple input files map to the same image ID: " + repr(key))
        result[key] = path
    return result


def rgb_image(path):
    with Image.open(path) as opened:
        orientation = int(opened.getexif().get(274, 1))
        if orientation not in [0, 1]:
            raise ValueError("Nonstandard EXIF orientation requires an explicit image/mask alignment review.")
        image = opened.convert("RGB")
        image.load()
    return image


def rgb_fingerprint(image):
    return sha_bytes(str(image.size).encode() + image.tobytes())


def phash_image(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()
    # Match phase1_corrections.fingerprint exactly: first DCT bit is the low bit;
    # the DC bit is retained. The historical prototype used a different encoding.
    bits = low > np.median(low[1:])
    return sum(int(bit) << index for index, bit in enumerate(bits))


def image_features(image):
    normalized = [rgb_fingerprint(image.resize((224, 224), mode)) for mode in
                  [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]]
    return {"width": image.width, "height": image.height, "rgb_sha256": rgb_fingerprint(image),
            "phash": f"{phash_image(image):016x}", "normalized224_hashes": normalized}


def binary_mask(path, size):
    with Image.open(path) as opened:
        opened.load()
        if opened.size != size:
            raise ValueError(f"Mask/image size mismatch: mask={opened.size}, image={size}")
        values = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(values[:, :, 0], values[:, :, 1]) or not np.array_equal(values[:, :, 1], values[:, :, 2]):
        raise ValueError("Primary pixel mask has unequal RGB channels; automatic label conversion is unsafe.")
    gray = values[:, :, 0]
    unique = set(np.unique(gray).tolist())
    if not unique.issubset({0, 255}):
        raise ValueError("Primary mask values differ from {0,255}: " + repr(sorted(unique)[:20]))
    mask = (gray == 255).astype(np.uint8)
    if not mask.any():
        raise ValueError("Empty primary mask on an image labelled manipulated.")
    return mask


def box_from_mask(mask):
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def yolo_line(box, width, height):
    if box is None:
        return ""
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Invalid box bounds.")
    values = [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]
    if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
        raise ValueError("Invalid normalized box.")
    return "0 " + " ".join(f"{value:.10f}" for value in values) + "\n"


class Groups:
    def __init__(self, identifiers):
        self.parent = {value: value for value in identifiers}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, a, b):
        aa, bb = self.find(a), self.find(b)
        if aa != bb:
            lower, upper = sorted([aa, bb], key=int)
            self.parent[upper] = lower


def allocate_groups(group_members, metadata, seed=42):
    strata = defaultdict(list)
    for group, members in group_members.items():
        origin = "|".join(sorted({metadata[s]["Original Dataset"].strip() for s in members}))
        strata[origin].append(group)
    allocations = {}
    for stratum, values in sorted(strata.items()):
        ordered = sorted(values, key=lambda g: (sha_bytes(f"{seed}:{g}".encode()), g))
        count = len(ordered)
        held = max(1, (count * 15 + 50) // 100) if count >= 3 else 0
        if count - 2 * held < 1:
            raise ValueError("Too few independent components for a split.")
        train_end = count - 2 * held
        for index, group in enumerate(ordered):
            allocations[group] = "train" if index < train_end else "val" if index < count - held else "test"
    if set(allocations.values()) != {"train", "val", "test"}:
        raise ValueError("Cannot form all three splits from the available independent source components.")
    return allocations


def audit_inputs(root, cfg):
    metadata_bytes = (root / "image_info.csv").read_bytes()
    source_rows = list(csv.DictReader(io.StringIO(metadata_bytes.decode("utf-8-sig"))))
    required = {"ID", "Original Natural Image Name", "Original Dataset"}
    required.update(f"manipulation_{i}" for i in range(1, cfg["variants_per_source"] + 1))
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError("image_info.csv differs from the inspected CSI schema.")
    metadata = {}
    for row in source_rows:
        source_id = numeric_id(row["ID"])
        if source_id in metadata:
            raise ValueError("Duplicate source ID in image_info.csv: " + source_id)
        metadata[source_id] = row
    if cfg.get("expected_sources") is not None and len(metadata) != cfg["expected_sources"]:
        raise ValueError(f"Expected {cfg['expected_sources']} CSV source rows; got {len(metadata)}.")
    originals = indexed_images(root / SOURCE_DIR)
    edits = indexed_images(root / EDIT_DIR, edited=True)
    masks = indexed_images(root / MASK_DIR, edited=True)
    semantic = indexed_images(root / SEMANTIC_DIR, edited=True) if (root / SEMANTIC_DIR).exists() else {}
    records, issues, fingerprint_rows = [], [], []
    source_hashes = {}
    def issue(source_id, variant, reason, stage="pairing", error=None):
        issues.append({"source_id": source_id, "variant": variant, "stage": stage, "reason": reason,
                       "error_type": type(error).__name__ if error else "",
                       "unreadable_file": isinstance(error, (OSError, SyntaxError)) if error else False})
    for source_id, row in sorted(metadata.items(), key=lambda pair: int(pair[0])):
        if source_id not in originals:
            issue(source_id, "all", "Missing pristine source image; all variants excluded.")
            continue
        try:
            original = rgb_image(originals[source_id])
            original_features = image_features(original)
        except Exception as error:
            issue(source_id, "all", "Unreadable/invalid pristine source: " + str(error), "pristine_image", error)
            continue
        source_hashes[source_id] = original_features["rgb_sha256"]
        base = {"source_id": source_id, "original_name": row["Original Natural Image Name"].strip(),
                "original_dataset": row["Original Dataset"].strip()}
        original_record = {**base, **original_features, "uid": f"csi_{int(source_id):04d}_original",
                           "kind": "pristine", "variant": 0, "image_path": str(originals[source_id].relative_to(root)),
                           "raw_mask_path": "", "object_class": "background", "manipulation": "none",
                           "semantic_impact": "none", "mask_pixels": 0, "mask_fraction": 0.0, "box": None,
                           "mask_sha256": sha_bytes(str(original.size).encode() + bytes(original.width * original.height))}
        records.append(original_record)
        fingerprint_rows.append({"path": original_record["image_path"], "sha256": original_features["rgb_sha256"]})
        for variant in range(1, cfg["variants_per_source"] + 1):
            key = (source_id, variant)
            if key not in edits or key not in masks:
                issue(source_id, variant, "Missing edited image or primary mask.")
                continue
            try:
                image = rgb_image(edits[key])
                features = image_features(image)
            except Exception as error:
                issue(source_id, variant, "Invalid edited image: " + str(error), "edited_image", error)
                continue
            try:
                mask = binary_mask(masks[key], image.size)
                mask_sha = sha_bytes(str(image.size).encode() + mask.tobytes())
            except Exception as error:
                issue(source_id, variant, "Invalid primary mask: " + str(error), "primary_mask", error)
                continue
            record = {**base, **features, "uid": f"csi_{int(source_id):04d}_edit_{variant}",
                      "kind": "edited", "variant": variant, "image_path": str(edits[key].relative_to(root)),
                      "raw_mask_path": str(masks[key].relative_to(root)),
                      "object_class": row.get(f"manipulation_{variant}_obj_class", "").strip(),
                      "manipulation": row[f"manipulation_{variant}"].strip(),
                      "semantic_impact": row.get(f"manipulation_{variant}_semantic_impact_category", "").strip(),
                      "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
                      "box": box_from_mask(mask), "mask_sha256": mask_sha}
            records.append(record)
            fingerprint_rows.extend([{"path": record["image_path"], "sha256": features["rgb_sha256"]},
                                     {"path": record["raw_mask_path"], "sha256": mask_sha}])
    expected_pairs = {(s, i) for s in metadata for i in range(1, cfg["variants_per_source"] + 1)}
    for kind, mapping in [("edited image", edits), ("primary mask", masks)]:
        for source_id, variant in sorted(set(mapping) - expected_pairs):
            issue(source_id, variant, "Orphan " + kind + " has no expected metadata pair.")
    for source_id in sorted(set(originals) - set(metadata), key=int):
        issue(source_id, "all", "Pristine image has no metadata row.")
    return metadata, records, issues, {
        "metadata_sources": len(metadata), "pristine_files": len(originals), "edited_files": len(edits),
        "primary_masks": len(masks), "semantic_masks_not_used_as_targets": len(semantic),
        "metadata_sha256": sha_bytes(metadata_bytes),
        "input_fingerprint": canonical_hash({"metadata": sha_bytes(metadata_bytes), "records": fingerprint_rows}),
    }


def establish_groups(records, metadata, reference, cfg):
    graph = Groups(metadata)
    edges = []
    identities = {}
    for source_id, row in metadata.items():
        name = (row["Original Dataset"].strip().lower(), row["Original Natural Image Name"].strip().lower())
        if name[1] and name in identities:
            graph.join(source_id, identities[name])
            edges.append({"a": source_id, "b": identities[name], "reason": "same_publisher_source_identity", "distance": ""})
        elif name[1]:
            identities[name] = source_id
    exact = defaultdict(list)
    for record in records:
        exact[record["rgb_sha256"]].append(record)
    excluded = {}
    for values in exact.values():
        for record in values[1:]:
            graph.join(values[0]["source_id"], record["source_id"])
            if values[0]["source_id"] != record["source_id"]:
                edges.append({"a": values[0]["source_id"], "b": record["source_id"],
                              "reason": "same_decoded_pixels", "distance": 0,
                              "a_image": values[0]["uid"], "b_image": record["uid"]})
        if len({record["mask_sha256"] for record in values}) > 1:
            for record in values:
                excluded[record["uid"]] = "identical_input_pixels_have_conflicting_masks"
        else:
            for record in sorted(values, key=lambda r: r["uid"])[1:]:
                excluded[record["uid"]] = "exact_duplicate_image_and_mask"
    hashes = [(record, int(record["phash"], 16)) for record in records]
    for i, (a, value_a) in enumerate(hashes):
        for b, value_b in hashes[i + 1:]:
            if a["source_id"] == b["source_id"]:
                continue
            distance = (value_a ^ value_b).bit_count()
            if distance <= cfg["phash_distance"]:
                graph.join(a["source_id"], b["source_id"])
                edges.append({"a": a["source_id"], "b": b["source_id"], "reason": "phash_candidate",
                              "distance": distance, "a_image": a["uid"], "b_image": b["uid"]})
    group_members = defaultdict(list)
    for source_id in metadata:
        group_members[graph.find(source_id)].append(source_id)
    id_to_group = {}
    for members in group_members.values():
        group = "csi_group_" + canonical_hash(sorted(members, key=int))[:12]
        for source_id in members:
            id_to_group[source_id] = group
    phase1_hashes = {row["sha256"]: row for row in reference}
    phase1_phashes = [(row, int(row["phash"], 16)) for row in reference]
    overlaps = []
    affected_groups = set()
    for record, value in hashes:
        exact_match = next((phase1_hashes[h] for h in [record["rgb_sha256"], *record["normalized224_hashes"]]
                            if h in phase1_hashes), None)
        nearest, minimum = None, 65
        if exact_match:
            nearest, minimum = exact_match, 0
            reason = "exact_native_or_resized224_pixel_hash"
        else:
            for old, old_value in phase1_phashes:
                distance = (value ^ old_value).bit_count()
                if distance < minimum:
                    minimum, nearest = distance, old
            reason = "phash_candidate"
        if nearest is not None and minimum <= cfg["phash_distance"]:
            group = id_to_group[record["source_id"]]
            affected_groups.add(group)
            overlaps.append({"uid": record["uid"], "source_id": record["source_id"], "group_id": group,
                             "phase1_path": nearest["relative_path"], "reason": reason, "distance": minimum})
    for record in records:
        record["group_id"] = id_to_group[record["source_id"]]
        if record["group_id"] in affected_groups:
            excluded[record["uid"]] = "component_has_possible_Phase1_overlap"
    kept = [record for record in records if record["uid"] not in excluded]
    eligible = defaultdict(set)
    for record in kept:
        eligible[record["group_id"]].add(record["source_id"])
    return kept, excluded, edges, overlaps, {group: sorted(members, key=int) for group, members in eligible.items()}


def manifest_rows(records):
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in ["box", "normalized224_hashes"]}
        row["box_xyxy_exclusive"] = json.dumps(record["box"])
        rows.append(row)
    return rows


def create_gallery(records, source_root, destination, limit=12):
    training = [record for record in records if record["split"] == "train"]
    edited = [record for record in training if record["kind"] == "edited"]
    selected = {}
    if edited:
        for record in [min(edited, key=lambda r: r["mask_fraction"]), max(edited, key=lambda r: r["mask_fraction"])]:
            selected[record["uid"]] = record
    by_type = defaultdict(list)
    for record in edited:
        by_type[record["manipulation"]].append(record)
    for name in sorted(by_type, key=lambda n: sha_bytes(n.encode())):
        record = min(by_type[name], key=lambda r: sha_bytes(r["uid"].encode()))
        if len(selected) < limit:
            selected[record["uid"]] = record
    for record in sorted(training, key=lambda r: sha_bytes(r["uid"].encode())):
        if record["kind"] == "pristine" and len(selected) < limit + 2:
            selected[record["uid"]] = record
    selected = list(selected.values())
    output = []
    destination.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    panel_w, panel_h, header = 290, 220, 48
    for start in range(0, len(selected), 6):
        batch = selected[start:start + 6]
        canvas = Image.new("RGB", (panel_w * 4, header + len(batch) * (panel_h + 38)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "TRAIN ONLY | native resolution per row | ground truth: white=255 | model confidence/NMS: N/A", fill="black", font=font)
        for column, title in enumerate(["Pristine source", "Edited image (or real negative)", "Pixel GT (displayed as 0/255)", "GT overlay + union box"]):
            draw.text((column * panel_w + 8, 25), title, fill="black", font=font)
        for row_number, record in enumerate(batch):
            image = rgb_image(source_root / record["image_path"])
            original = next(p for p in (source_root / SOURCE_DIR).iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES and numeric_id(p.stem) == record["source_id"])
            pristine = rgb_image(original)
            mask = (binary_mask(source_root / record["raw_mask_path"], image.size)
                    if record["kind"] == "edited" else np.zeros((image.height, image.width), np.uint8))
            overlay = np.asarray(image).copy()
            overlay[mask > 0] = (.55 * overlay[mask > 0] + .45 * np.array([255, 0, 0])).astype(np.uint8)
            overlay = Image.fromarray(overlay)
            if record["box"]:
                box = record["box"]
                ImageDraw.Draw(overlay).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=(0, 255, 0), width=2)
            panels = [pristine, image, Image.fromarray(mask * 255).convert("RGB"), overlay]
            top = header + row_number * (panel_h + 38)
            for column, panel in enumerate(panels):
                method = Image.Resampling.NEAREST if column == 2 else Image.Resampling.LANCZOS
                panel = ImageOps.contain(panel, (panel_w - 10, panel_h), method)
                canvas.paste(panel, (column * panel_w + (panel_w - panel.width) // 2, top))
            caption = (f"{record['uid']} | source={pristine.width}x{pristine.height}; edit={image.width}x{image.height}"
                       f" | {record['manipulation']} | fg={record['mask_fraction']:.3f}")
            draw.text((8, top + panel_h + 3), caption, fill="black", font=font)
        path = destination / f"train_mask_review_{start // 6 + 1:02d}.png"
        canvas.save(path)
        output.append(path)
    dump(destination / "visual_review_record.json", {"split": "train", "images": [r["uid"] for r in selected],
         "visual_review_complete": False, "reviewer": None,
         "questions": ["Does white/red foreground correspond to the edited area?", "Does the green box enclose that foreground?",
                       "Are pristine examples free of positive regions?"],
         "note": "Generated overlays are not evidence that a human has reviewed them."})
    write_csv(destination / "visual_review_checklist.csv", [
        {"uid": r["uid"], "kind": r["kind"], "split": "train", "mask_alignment": "",
         "box_correct": "", "negative_target_correct": "", "reviewer": "", "notes": ""} for r in selected
    ], ["uid", "kind", "split", "mask_alignment", "box_correct", "negative_target_correct", "reviewer", "notes"])
    return output


def prepare(root, work_root, reference_path=None, overrides=None):
    cfg = {**DEFAULTS, **(overrides or {})}
    if (cfg["train_percent"], cfg["val_percent"], cfg["test_percent"]) != (70, 15, 15):
        raise ValueError("This protocol implements 70/15/15 by source component.")
    if cfg["primary_mask_foreground"] != 255:
        raise ValueError("This schema version encodes raw foreground as 255; change the audited parser for another convention.")
    root, work_root = Path(root), Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    reference = load_reference(reference_path, work_root / "audit" / "phase1_reference.csv")
    metadata, records, issues, input_report = audit_inputs(root, cfg)
    if not records:
        raise ValueError("No valid image/mask records remain after input checks.")
    print("Valid input records before duplicate/overlap exclusions:", len(records))
    kept, excluded, edges, overlaps, components = establish_groups(records, metadata, reference, cfg)
    if not kept:
        raise ValueError("No records remain after Phase 1 overlap and label-conflict exclusions.")
    allocations = allocate_groups(components, metadata, cfg["seed"])
    for record in kept:
        record["split"] = allocations[record["group_id"]]
        record["export_image"] = f"images/{record['split']}/{record['uid']}.png"
        record["export_mask"] = f"segmentation_masks/{record['split']}/{record['uid']}.png"
        record["export_detection_label"] = f"labels/{record['split']}/{record['uid']}.txt"
    kept.sort(key=lambda record: (int(record["source_id"]), record["variant"]))
    assignments = [{"uid": r["uid"], "source_id": r["source_id"], "group_id": r["group_id"], "split": r["split"]} for r in kept]
    lock = {"configuration": cfg, "input_fingerprint": input_report["input_fingerprint"],
            "phase1_reference_sha256": REFERENCE_SHA256, "assignment_sha256": canonical_hash(assignments)}
    environment = {"python": sys.version, "packages": {name: importlib.metadata.version(name)
                   for name in ["Pillow", "numpy", "scipy"]}}
    lock_path = work_root / "protocol_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("This output folder has a different frozen protocol. Use a new WORK_ROOT; do not silently re-split it.")
    # Check separation on the complete kept inventory, not only one representative per source.
    for key in ["source_id", "group_id", "rgb_sha256", "phash"]:
        seen = defaultdict(set)
        for record in kept:
            seen[record[key]].add(record["split"])
        if any(len(values) > 1 for values in seen.values()):
            raise ValueError("Split separation check failed: " + key)
    for split in ["train", "val", "test"]:
        if {r["kind"] for r in kept if r["split"] == split} != {"pristine", "edited"}:
            raise ValueError(split + " does not contain edited images and pristine negatives.")
    dump(lock_path, lock)
    dump(work_root / "preparation_environment.json", environment)
    audit = work_root / "audit"
    write_csv(audit / "invalid_or_unpaired_records.csv", issues,
              ["source_id", "variant", "stage", "reason", "error_type", "unreadable_file"])
    write_csv(audit / "excluded_records.csv", [{"uid": uid, "reason": reason} for uid, reason in sorted(excluded.items())], ["uid", "reason"])
    write_csv(audit / "source_group_edges.csv", edges, ["a", "b", "reason", "distance", "a_image", "b_image"])
    write_csv(audit / "phase1_overlap_candidates.csv", overlaps, ["uid", "source_id", "group_id", "phase1_path", "reason", "distance"])
    write_csv(audit / "group_assignments.csv", [{"group_id": group, "source_ids": ";".join(members), "split": allocations[group]}
              for group, members in sorted(components.items())])
    prepared = work_root / "prepared"
    for index, record in enumerate(kept, 1):
        image = rgb_image(root / record["image_path"])
        mask = (binary_mask(root / record["raw_mask_path"], image.size) if record["kind"] == "edited"
                else np.zeros((image.height, image.width), np.uint8))
        if rgb_fingerprint(image) != record["rgb_sha256"]:
            raise ValueError("Source image changed during preparation.")
        for name in ["export_image", "export_mask", "export_detection_label"]:
            (prepared / record[name]).parent.mkdir(parents=True, exist_ok=True)
        image.save(prepared / record["export_image"], compress_level=1)
        Image.fromarray(mask).save(prepared / record["export_mask"])
        (prepared / record["export_detection_label"]).write_text(yolo_line(record["box"], image.width, image.height))
        # PNG conversion must preserve decoded RGB pixels and exact 0/1 segmentation labels.
        with Image.open(prepared / record["export_image"]) as check:
            if rgb_fingerprint(check.convert("RGB")) != record["rgb_sha256"]:
                raise ValueError("RGB pixels changed in export.")
        with Image.open(prepared / record["export_mask"]) as check:
            if not np.array_equal(np.asarray(check), mask):
                raise ValueError("Segmentation mask changed in export.")
        if index % 100 == 0 or index == len(kept):
            print(f"Exported and checked {index}/{len(kept)} records")
    manifest = manifest_rows(kept)
    write_csv(prepared / "manifest.csv", manifest)
    for split in ["train", "val", "test"]:
        write_csv(prepared / f"{split}_manifest.csv", [r for r in manifest if r["split"] == split])
    (prepared / "data.yaml").write_text("# Paths are relative to this YAML; the training notebook will set the runtime root explicitly.\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: edited_region\n")
    dump(prepared / "dataset_spec.json", {
        "detection_class_order": {"0": "edited_region"},
        "segmentation_class_order": {"0": "background", "1": "edited_region"},
        "segmentation_values": [0, 1], "segmentation_ignore_value": None,
        "mask_loading_note": "Masks already contain 0/1; do not divide them by 255.",
        "primary_raw_mask_folder": MASK_DIR, "semantic_masks_used_as_targets": False,
        "box_policy": cfg["box_policy"], "coordinate_convention": "xyxy with right/bottom exclusive; YOLO normalized xywh",
        "preprocessing": "native size, lossless RGB PNG; model-specific resizing belongs to training/inference",
        "format_limitation": "PNG standardization removes filename/extension differences, not pre-existing JPEG artifacts.",
        "label_scope": "local edits; labels do not establish whole-image AI-generation or Saudi-event performance",
    })
    dump(prepared / "preparation_environment.json", environment)
    galleries = create_gallery(kept, root, work_root / "review")
    split_counts = {}
    for split in ["train", "val", "test"]:
        subset = [r for r in kept if r["split"] == split]
        split_counts[split] = {"images": len(subset), "edited": sum(r["kind"] == "edited" for r in subset),
            "pristine": sum(r["kind"] == "pristine" for r in subset), "boxes": sum(r["box"] is not None for r in subset),
            "sources": len({r["source_id"] for r in subset}), "components": len({r["group_id"] for r in subset})}
    summary = {
        "status": "PHASE2_DATA_PREPARED_VISUAL_REVIEW_PENDING", "configuration": cfg,
        "input": input_report, "valid_before_exclusions": len(records), "exported_records": len(kept),
        "invalid_or_unpaired_records": len(issues), "excluded_records": dict(Counter(excluded.values())),
        "unreadable_files": sum(bool(issue["unreadable_file"]) for issue in issues),
        "expected_input_records": len(metadata) * (cfg["variants_per_source"] + 1),
        "quarantined_expected_records": len(metadata) * (cfg["variants_per_source"] + 1) - len(records),
        "phase1_overlap_candidates": len(overlaps), "group_edges": len(edges), "split_counts": split_counts,
        "primary_mask_encoding_checks_passed": True, "all_exported_RGB_and_mask_pixels_verified": True,
        "source_group_split_separation_passed": True,
        "grouping_scope": "publisher original identity plus decoded hashes and conservative pHash<=4 links; hidden scene relationships may remain",
        "phase1_overlap_check_scope": "recorded corrected Phase 1 hashes, three 224px resize hashes and pHash<=4 candidates; not a universal independence proof",
        "phash_encoding": "corrected Phase 1: DCT bits in little-endian order, DC bit retained",
        "visual_review_complete": False, "white_foreground_interpretation_visually_confirmed": False,
        "training_executed": False, "test_inference_executed": False,
        "protocol_assignment_sha256": lock["assignment_sha256"],
        "prepared_manifest_sha256": sha_bytes((prepared / "manifest.csv").read_bytes()),
        "primary_raw_mask_folder": MASK_DIR, "semantic_masks_used_as_targets": False,
        "preparation_environment": environment,
        "next_step": "Inspect the training-only review gallery and exclusions, then proceed to model training with the frozen split.",
    }
    dump(work_root / "preparation_summary.json", summary)
    dump(prepared / "preparation_summary.json", summary)
    review_zip = work_root.parent / "Real-Abshar_Phase2_Preparation_Review.zip"
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(work_root / "preparation_summary.json", "preparation_summary.json")
        archive.write(lock_path, "protocol_lock.json")
        archive.write(work_root / "preparation_environment.json", "preparation_environment.json")
        for path in sorted(audit.iterdir()):
            if path.is_file() and path.name != "phase1_reference.csv":
                archive.write(path, "audit/" + path.name)
        for path in sorted((work_root / "review").iterdir()):
            archive.write(path, "review/" + path.name)
        archive.write(prepared / "manifest.csv", "manifest.csv")
    data_zip = work_root.parent / "Real-Abshar_Phase2_Prepared_Data.zip"
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(prepared.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(prepared).as_posix())
        archive.write(lock_path, "protocol_lock.json")
    print("PHASE2_DATA_PREPARED_VISUAL_REVIEW_PENDING")
    print(json.dumps(split_counts, indent=2))
    print("Send the smaller review ZIP:", review_zip)
    print("Retain the prepared data ZIP privately:", data_zip)
    return summary
