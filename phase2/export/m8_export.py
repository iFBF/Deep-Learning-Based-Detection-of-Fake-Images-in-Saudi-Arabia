"""M8 Part 1: checksum-pinned FP32 exports and paired validation parity. No test inference."""
from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import traceback
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageOps

from cohort_data import (COHORT_NAME, COHORT_UID_SHA256, EXPECTED, SOURCE_ASSIGNMENT_SHA256,
                         SOURCE_MANIFEST_SHA256, dump, locate_prepared, read_preparation,
                         safe_path, sha_file, verify_row, write_csv)
from export_metrics import detector_input, detections, fake_score, iou, match_detections, sigmoid
from model_assets import MODEL_SPECS, acquire
from segmentation_data import restore_probability, transform
from segmentation_metrics import counts, summarize

P1_COMMIT = "26a12b0a69641990be2ef4b2794a19e420f87e20"
PINNED = {"transformers": "5.16.1", "ultralytics": "8.3.221", "onnx": "1.19.1",
          "onnxruntime": "1.31.1", "wandb": "0.28.1"}
TOLERANCES = {"classifier_max_logit_abs_error": 0.001,
              "classifier_max_fake_score_abs_error": 0.0001,
              "classifier_label_agreement_minimum": 0.99,
              "detector_max_box_component_abs_error_640px": 0.1,
              "detector_max_score_abs_error": 0.0002,
              "detector_detection_agreement_minimum": 0.99,
              "detector_match_iou": 0.99, "detector_match_score_tolerance": 0.001,
              "detector_reference_f1_abs_difference_maximum": 0.005,
              "segmenter_max_logit_abs_error": 0.002,
              "segmenter_native_pixel_agreement_minimum": 0.99,
              "segmenter_foreground_metric_abs_difference_maximum": 0.005,
              "segmenter_recorded_val_iou_abs_difference_maximum": 0.005}
SOURCES = ["cohort_data.py", "segmentation_data.py", "segmentation_models.py", "segmentation_metrics.py",
           "export_metrics.py", "model_assets.py", "m8_backup.py", "m8_export.py"]
ASSETS = ["classifier_config.json", "classifier_processor.json", "phase1_selection_locked.json",
          "verified_m5_selection.json", "verified_cloud_backup_receipt.json",
          "m4_m4_selection.json", "m4_cloud_backup_receipt.json", "M5_REVIEW.md", "M8_NEXT_STEPS.md"]


def runtime_environment():
    import torch
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "transformers", "ultralytics", "onnx", "onnxruntime",
                 "numpy", "pillow", "opencv-python", "wandb"]}
    for name, required in PINNED.items():
        if versions[name] != required:
            raise RuntimeError(f"Install the notebook dependencies first: {name}=={required}; found {versions[name]}")
    if not versions["torch"].startswith("2.11.") or not versions["torchvision"].startswith("0.26."):
        raise RuntimeError("Use the Kaggle image with Torch 2.11.x / torchvision 0.26.x used by your successful runs. "
                           "Do not overwrite Torch to work around a model-loading error; send the review/error first.")
    cpu = "unknown"
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(42)
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_model": cpu,
            "logical_cpus": os.cpu_count(), "packages": versions, "device": "cpu", "precision": "float32",
            "torch_intra_threads": torch.get_num_threads(), "torch_inter_threads": torch.get_num_interop_threads(),
            "ort_intra_threads": 2, "ort_inter_threads": 1, "ort_execution_mode": "sequential",
            "ort_provider": "CPUExecutionProvider", "batch": 1,
            "is_latency_benchmark": False, "started_at_utc": datetime.now(timezone.utc).isoformat()}


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def check_graph(path, expected_shape):
    import onnx
    onnx.checker.check_model(str(path))
    sess = session(path)
    inputs, outputs = sess.get_inputs(), sess.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != expected_shape or inputs[0].type != "tensor(float)" or len(outputs) != 1:
        raise ValueError("Unexpected exported ONNX contract: " + str(path))
    if outputs[0].type != "tensor(float)":
        raise ValueError("Expected an FP32 ONNX output.")
    return {"input_name": inputs[0].name, "input_shape": inputs[0].shape, "input_type": inputs[0].type,
            "output_name": outputs[0].name, "output_shape": outputs[0].shape, "output_type": outputs[0].type,
            "providers": sess.get_providers()}


def math_attention():
    # Keep the SDPA implementation selected by Transformers; choose the portable mathematical kernel.
    # Both the native reference and traced export use this context; the kernel choice is recorded.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.MATH])


def load_native(paths, assets, release):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    from ultralytics import YOLO
    from segmentation_models import build_model

    classifier_dir = release / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assets / "classifier_config.json", classifier_dir / "config.json")
    shutil.copyfile(assets / "classifier_processor.json", classifier_dir / "preprocessor_config.json")
    config = AutoConfig.from_pretrained(str(classifier_dir), local_files_only=True)
    if config.model_type != "swin" or config.id2label != {0: "Real", 1: "Fake"} or config.image_size != 224:
        raise ValueError("Classifier architecture/class order differs from corrected Phase 1.")
    model = AutoModelForImageClassification.from_config(config).float().eval()
    saved = torch.load(paths["classifier"], map_location="cpu", weights_only=True)
    if saved.get("mode") != "full":
        raise ValueError("Expected the selected full-fine-tuning classifier checkpoint.")
    model.load_state_dict(saved["state_dict"], strict=True)
    del saved
    processor = AutoImageProcessor.from_pretrained(str(classifier_dir), backend="torchvision", local_files_only=True)

    class Classifier(nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, images):
            return self.classifier(pixel_values=images).logits

    classifier = Classifier(model).eval()
    saved = torch.load(paths["segmenter"], map_location="cpu", weights_only=True)
    if saved["architecture"] != "B_UNet_ResNet18":
        raise ValueError("Wrong selected M5 architecture.")
    segmenter = build_model("B_UNet_ResNet18", pretrained=None).float().eval()
    segmenter.load_state_dict(saved["model_state"], strict=True)
    del saved
    # YOLO's loader reads its own pickled model class. The exact checkpoint hash was verified BEFORE this call.
    detector = YOLO(str(paths["detector"]), task="detect")
    if detector.names != {0: "edited_region"}:
        raise ValueError("Wrong detector class order.")
    detector.model.to("cpu").float().eval().fuse(verbose=False)
    metadata = {"classifier_class": type(model).__name__, "classifier_attention_implementation": config._attn_implementation,
                "classifier_sdpa_kernel_for_export_and_reference": "MATH",
                "classifier_processor_class": type(processor).__name__,
                "classifier_processor_backend": "torchvision", "segmenter": "B_UNet_ResNet18",
                "detector": "YOLO11n", "all_native_models_eval_mode": True}
    return classifier, processor, detector, segmenter, metadata


def export_one(name, model, shape, release, protocol_sha, detector=None):
    import torch
    output = release / "models" / (name + ".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(".export.json")
    identity = {"name": name, "native_checkpoint_sha256": MODEL_SPECS[name]["sha256"],
                "export_protocol_sha256": protocol_sha, "input_shape": shape,
                "opset": 17, "precision": "FP32", "dynamic_shapes": False, "nms_embedded": False}
    if output.exists() or receipt_path.exists():
        if not (output.is_file() and receipt_path.is_file()):
            raise RuntimeError("Interrupted export found for " + name + ". Remove only its incomplete .onnx and .export.json files, then rerun.")
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(k) != v for k, v in identity.items()) or sha_file(output) != receipt["onnx_sha256"]:
            raise RuntimeError("Export identity changed. Use a new WORK_ROOT; do not overwrite a reviewed candidate.")
        check_graph(output, shape)
        print("Reusing verified ONNX graph:", name, flush=True)
        return receipt
    print("Exporting ONNX on CPU:", name, flush=True)
    if detector is not None:
        produced = Path(detector.export(format="onnx", imgsz=640, batch=1, device="cpu", half=False,
                                         dynamic=False, simplify=False, nms=False, opset=17))
        shutil.copyfile(produced, output)
    else:
        dummy = torch.zeros(shape, dtype=torch.float32)
        with torch.inference_mode(), math_attention():
            torch.onnx.export(model, (dummy,), str(output), input_names=["images"], output_names=["logits"],
                              opset_version=17, dynamo=False, external_data=False,
                              do_constant_folding=True, training=torch.onnx.TrainingMode.EVAL)
    contract = check_graph(output, shape)
    receipt = {**identity, **contract, "onnx_sha256": sha_file(output), "bytes": output.stat().st_size}
    dump(receipt_path, receipt)
    print("Checked ONNX graph:", name, receipt["onnx_sha256"], flush=True)
    gc.collect()
    return receipt


def _ort(sess, value):
    result = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(value, dtype=np.float32)})[0]
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite ONNX output")
    return result


def _torch_output(model, value):
    import torch
    with torch.inference_mode(), math_attention():
        output = model(torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)))
    if isinstance(output, (tuple, list)):
        output = output[0]
    result = output.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite native model output")
    return result


def _official_predictions(detector, image):
    result = detector.predict(source=image, device="cpu", imgsz=640, rect=False, half=False,
                               conf=0.35, iou=0.45, max_det=50, batch=1, save=False, verbose=False)[0]
    return [{"box": row[:4].tolist(), "score": float(row[4]), "class": int(row[5])}
            for row in result.boxes.data.detach().cpu().numpy()]


def _detector_counts(record):
    predictions, box = record["detector_official"], record["ground_truth_box"]
    tp = int(box is not None and any(iou(p["box"], box) >= 0.5 for p in predictions))
    return {"tp": tp, "fp": len(predictions) - tp, "fn": int(box is not None) - tp}


def parity_record(row, root, classifier, processor, detector, segmenter, sessions):
    with Image.open(safe_path(root, row["export_image"])) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
    with Image.open(safe_path(root, row["export_mask"])) as opened:
        truth = np.asarray(opened, dtype=np.uint8)
    # The original Phase 1 processor takes a PIL RGB image through its torchvision backend.
    classifier_input = processor(images=image, return_tensors="pt")["pixel_values"].float().cpu().numpy()
    if classifier_input.shape != (1, 3, 224, 224):
        raise ValueError("Classifier preprocessing changed shape.")
    native_c, exported_c = _torch_output(classifier, classifier_input), _ort(sessions["classifier"], classifier_input)
    native_score, exported_score = fake_score(native_c), fake_score(exported_c)

    dinput = detector_input(image)
    native_d, exported_d = _torch_output(detector.model, dinput), _ort(sessions["detector"], dinput)
    if native_d.shape != exported_d.shape or native_d.shape != (1, 5, 8400):
        raise ValueError("Detector output contract differs from one-class YOLO11n at 640.")
    native_boxes = detections(native_d, image.width, image.height)
    exported_boxes = detections(exported_d, image.width, image.height)
    official_boxes = _official_predictions(detector, image)
    detector_match, detector_denominator = match_detections(native_boxes, exported_boxes)
    adapter_match, adapter_denominator = match_detections(official_boxes, native_boxes)

    sinput, _, _ = transform(image, truth)
    sinput = sinput[None]
    native_s, exported_s = _torch_output(segmenter, sinput), _ort(sessions["segmenter"], sinput)
    if native_s.shape != (1, 1, 640, 640) or exported_s.shape != native_s.shape:
        raise ValueError("Unexpected segmentation output shape.")
    native_probability = restore_probability(sigmoid(native_s[0, 0]), image.width, image.height)
    exported_probability = restore_probability(sigmoid(exported_s[0, 0]), image.width, image.height)
    native_mask, exported_mask = native_probability >= 0.5, exported_probability >= 0.5
    return {"uid": row["uid"], "source_id": row["source_id"], "kind": row["kind"], "split": "val",
            "width": image.width, "height": image.height, "ground_truth_box": json.loads(row["box_xyxy_exclusive"]),
            "classifier_native_fake_score": native_score, "classifier_onnx_fake_score": exported_score,
            "classifier_logit_abs_error": float(np.max(np.abs(native_c - exported_c))),
            "classifier_score_abs_error": abs(native_score - exported_score),
            "classifier_label_agrees": bool((native_score >= 0.5) == (exported_score >= 0.5)),
            "detector_box_component_abs_error_640px": float(np.max(np.abs(native_d[:, :4] - exported_d[:, :4]))),
            "detector_score_abs_error": float(np.max(np.abs(native_d[:, 4] - exported_d[:, 4]))),
            "detector_onnx_matched": detector_match, "detector_onnx_match_denominator": detector_denominator,
            "detector_adapter_matched": adapter_match, "detector_adapter_match_denominator": adapter_denominator,
            "detector_native": native_boxes, "detector_onnx": exported_boxes, "detector_official": official_boxes,
            "segmenter_logit_abs_error": float(np.max(np.abs(native_s - exported_s))),
            "segmenter_native_counts": counts(truth, native_probability, 0.5),
            "segmenter_onnx_counts": counts(truth, exported_probability, 0.5),
            "segmenter_pixels_agree": int(np.count_nonzero(native_mask == exported_mask)),
            "image_pixels": int(truth.size), "segmenter_native_foreground_pixels": int(native_mask.sum()),
            "segmenter_onnx_foreground_pixels": int(exported_mask.sum()),
            "segmenter_native_area_fraction": float(native_mask.mean()),
            "segmenter_onnx_area_fraction": float(exported_mask.mean())}


def parity_summary(records, selection, m4_selection):
    max_error = lambda key: max(float(r[key]) for r in records)
    ratio = lambda a, b: sum(r[a] for r in records) / sum(r[b] for r in records) if sum(r[b] for r in records) else 1.0
    native_seg = summarize([{"kind": r["kind"], **r["segmenter_native_counts"]} for r in records])
    onnx_seg = summarize([{"kind": r["kind"], **r["segmenter_onnx_counts"]} for r in records])
    detector_counts = {key: sum(_detector_counts(r)[key] for r in records) for key in ["tp", "fp", "fn"]}
    tp, fp, fn = [detector_counts[k] for k in ["tp", "fp", "fn"]]
    detector_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    m4_recorded_f1 = m4_selection["validation_operating_metrics"]["f1"]
    measured = {"classifier_max_logit_abs_error": max_error("classifier_logit_abs_error"),
                "classifier_max_fake_score_abs_error": max_error("classifier_score_abs_error"),
                "classifier_label_agreement": float(np.mean([r["classifier_label_agrees"] for r in records])),
                "detector_max_box_component_abs_error_640px": max_error("detector_box_component_abs_error_640px"),
                "detector_max_score_abs_error": max_error("detector_score_abs_error"),
                "detector_detection_agreement": ratio("detector_onnx_matched", "detector_onnx_match_denominator"),
                "detector_official_adapter_agreement": ratio("detector_adapter_matched", "detector_adapter_match_denominator"),
                "detector_reference_operating_counts": detector_counts,
                "detector_reference_operating_f1": detector_f1,
                "detector_reference_f1_abs_difference": abs(detector_f1 - m4_recorded_f1),
                "segmenter_max_logit_abs_error": max_error("segmenter_logit_abs_error"),
                "segmenter_native_pixel_agreement": ratio("segmenter_pixels_agree", "image_pixels"),
                "segmenter_foreground_iou_abs_difference": abs(native_seg["foreground_iou"] - onnx_seg["foreground_iou"]),
                "segmenter_foreground_dice_abs_difference": abs(native_seg["foreground_dice"] - onnx_seg["foreground_dice"]),
                "segmenter_recorded_val_iou_abs_difference": abs(native_seg["foreground_iou"] - selection["validation_metrics"]["foreground_iou"])}
    t = TOLERANCES
    checks = {"classifier_logits": measured["classifier_max_logit_abs_error"] <= t["classifier_max_logit_abs_error"],
              "classifier_scores": measured["classifier_max_fake_score_abs_error"] <= t["classifier_max_fake_score_abs_error"],
              "classifier_labels": measured["classifier_label_agreement"] >= t["classifier_label_agreement_minimum"],
              "detector_raw_boxes": measured["detector_max_box_component_abs_error_640px"] <= t["detector_max_box_component_abs_error_640px"],
              "detector_raw_scores": measured["detector_max_score_abs_error"] <= t["detector_max_score_abs_error"],
              "detector_selected_boxes": measured["detector_detection_agreement"] >= t["detector_detection_agreement_minimum"],
              "detector_official_adapter": measured["detector_official_adapter_agreement"] >= t["detector_detection_agreement_minimum"],
              "detector_recorded_operating_f1": measured["detector_reference_f1_abs_difference"] <= t["detector_reference_f1_abs_difference_maximum"],
              "segmenter_logits": measured["segmenter_max_logit_abs_error"] <= t["segmenter_max_logit_abs_error"],
              "segmenter_native_pixels": measured["segmenter_native_pixel_agreement"] >= t["segmenter_native_pixel_agreement_minimum"],
              "segmenter_iou": measured["segmenter_foreground_iou_abs_difference"] <= t["segmenter_foreground_metric_abs_difference_maximum"],
              "segmenter_dice": measured["segmenter_foreground_dice_abs_difference"] <= t["segmenter_foreground_metric_abs_difference_maximum"],
              "segmenter_recorded_validation": measured["segmenter_recorded_val_iou_abs_difference"] <= t["segmenter_recorded_val_iou_abs_difference_maximum"]}
    return {"split": "val", "images": len(records), "source_families": len({r["source_id"] for r in records}),
            "classifier_input_resolution": [224, 224], "classifier_fake_threshold": 0.5,
            "region_input_resolution": [640, 640], "detector_confidence_threshold": 0.35,
            "detector_confidence_comparator": "> (Ultralytics convention)",
            "detector_nms_iou": 0.45, "detector_matching_iou_for_reference_metrics": 0.5,
            "detector_max_detections": 50, "segmenter_pixel_threshold": 0.5, "segmenter_postprocessing": "none",
            "precision": "FP32", "batch": 1, "tolerances_predeclared": t, "measured": measured,
            "degenerate_box_parity_rule": "Only identical clipped zero-area boxes match, coordinate tolerance0.001 native pixels.",
            "checks": checks, "all_checks_passed": all(checks.values()),
            "native_segmenter_validation": native_seg, "onnx_segmenter_validation": onnx_seg,
            "scope": "Export parity on validation. This is not a final-test evaluation or deployment acceptance decision.",
            "test_evaluation_performed": False}


def _zip_tree(destination, root, include):
    destination = Path(destination)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and include(path):
                z.write(path, path.relative_to(root).as_posix())
    return destination


def package_outputs(work, release):
    from m8_backup import release_allowlist
    allowed = release_allowlist()
    review = work.parent / "Real-Abshar_Phase2_M8_Export_Review.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(release.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.relative_to(release).as_posix() in allowed and path.suffix != ".onnx":
                z.write(path, path.relative_to(release).as_posix())
        for name in ["run_status.json", "cloud_backup_receipt.json", "error.txt"]:
            path = work / name
            if path.is_file():
                z.write(path, name)
    release_zip = None
    if all((release / "models" / (name + ".onnx")).is_file() for name in MODEL_SPECS):
        receipt = work / "cloud_backup_receipt.json"
        if receipt.is_file():
            shutil.copyfile(receipt, release / "cloud_backup_receipt.json")
        release_zip = _zip_tree(work.parent / "Real-Abshar_Phase2_M8_ONNX_Release.zip", release,
                                lambda p: not p.is_symlink() and p.relative_to(release).as_posix() in allowed)
    return {"review_zip": str(review), "release_zip": str(release_zip) if release_zip else None}


def run(input_root, work_root, assets_root, prepared_root=None, classifier_path=None):
    work, assets = Path(work_root), Path(assets_root)
    work.mkdir(parents=True, exist_ok=True)
    release = work / "onnx_release"
    release.mkdir(parents=True, exist_ok=True)
    status_path = work / "run_status.json"
    run_id = json.loads(status_path.read_text())["run_id"] if status_path.exists() else uuid.uuid4().hex[:12]
    status = {"run_id": run_id, "stage": "M8_export", "status": "started", "test_evaluation_performed": False}
    dump(status_path, status)
    manifest_path = release / "manifest.json"
    if manifest_path.is_file():
        stale_manifest = json.loads(manifest_path.read_text())
        stale_manifest["ready_for_service_assembly"] = False
        stale_manifest["last_export_run_status"] = "in_progress"
        dump(manifest_path, stale_manifest)
    try:
        runtime = runtime_environment()
        dump(release / "runtime.json", runtime)
        (release / "requirements-lock-executed.txt").write_text("\n".join(
            f"{name}=={version}" for name, version in sorted(runtime["packages"].items())) + "\n")
        for name in SOURCES + ASSETS:
            destination = release / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets / name, destination)
        root = locate_prepared(input_root, work, prepared_root)
        rows, omitted = read_preparation(root)
        validation = [r for r in rows if r["split"] == "val"]
        for index, row in enumerate(validation, 1):
            verify_row(root, row)
            if index % 25 == 0 or index == len(validation):
                print(f"Verified validation inputs: {index}/{len(validation)}", flush=True)
        write_csv(release / "evidence" / "val_manifest.csv", validation)
        dump(release / "evidence" / "cohort_preflight.json", {
            "cohort": COHORT_NAME, "cohort_uid_sha256": COHORT_UID_SHA256, "split_counts": EXPECTED,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_assignment_sha256": SOURCE_ASSIGNMENT_SHA256,
            "validation_records_verified": len(validation), "test_image_or_mask_pixels_decoded": False,
            "test_inference_performed": False, "valid_records_omitted_by_original_cohort_rule": len(omitted)})
        paths = acquire(input_root, work, classifier_path)
        classifier, processor, detector, segmenter, native_metadata = load_native(paths, assets, release)
        dump(release / "native_model_load.json", native_metadata)
        protocol = {"native_models": MODEL_SPECS, "phase1_commit": P1_COMMIT,
                    "cohort_uid_sha256": COHORT_UID_SHA256, "validation_manifest_sha256": sha_file(release / "evidence" / "val_manifest.csv"),
                    "input_shapes": {"classifier": [1, 3, 224, 224], "detector": [1, 3, 640, 640], "segmenter": [1, 3, 640, 640]},
                    "batch": 1, "precision": "FP32", "opset": 17, "dynamic_shapes": False,
                    "nms_embedded": False, "test_evaluation_performed": False,
                    "predeclared_export_tolerances": TOLERANCES, "native_loading": native_metadata,
                    "packages": runtime["packages"],
                    "cpu_settings": {k: runtime[k] for k in ["cpu_model", "torch_intra_threads", "torch_inter_threads",
                                                            "ort_intra_threads", "ort_inter_threads", "ort_execution_mode"]},
                    "code_sha256": {n: sha_file(assets / n) for n in SOURCES},
                    "asset_sha256": {n: sha_file(assets / n) for n in ASSETS}}
        protocol_path = release / "export_protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("The export protocol changed. Use a new WORK_ROOT.")
        dump(protocol_path, protocol)
        protocol_sha = sha_file(protocol_path)
        receipts = {}
        for name, model in [("classifier", classifier), ("detector", detector.model), ("segmenter", segmenter)]:
            receipts[name] = export_one(name, model, protocol["input_shapes"][name], release, protocol_sha,
                                       detector=detector if name == "detector" else None)
        sessions = {name: session(release / "models" / (name + ".onnx")) for name in MODEL_SPECS}
        cache = release / "evidence" / "val_export_parity_predictions.jsonl"
        cache_lock = cache.with_suffix(".lock.json")
        cache_identity = {"export_protocol_sha256": protocol_sha,
                          "onnx_sha256": {n: r["onnx_sha256"] for n, r in receipts.items()}}
        if cache.exists() or cache_lock.exists():
            if not (cache.exists() and cache_lock.is_file()) or json.loads(cache_lock.read_text()) != cache_identity:
                raise RuntimeError("Cached validation parity belongs to a different export. Use a new WORK_ROOT.")
            finished_digest = cache.with_suffix(".sha256")
            if finished_digest.is_file() and sha_file(cache) != finished_digest.read_text().strip():
                raise ValueError("Completed validation parity cache changed; do not use the modified predictions.")
            records = []
            # A partial final line from a stopped session is trimmed; complete rows are retained.
            lines = cache.read_text().splitlines()
            for line_index, line in enumerate(lines):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if line_index != len(lines) - 1:
                        raise
                    break
            cache.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        else:
            records = []
            cache.touch()
            dump(cache_lock, cache_identity)
        if [r["uid"] for r in records] != [r["uid"] for r in validation[:len(records)]]:
            raise ValueError("Cached validation UID order differs from the locked cohort.")
        for index, row in enumerate(validation[len(records):], len(records) + 1):
            result = parity_record(row, root, classifier, processor, detector, segmenter, sessions)
            records.append(result)
            with cache.open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
            if index % 5 == 0 or index == len(validation):
                print(f"Compared native/ONNX on validation only: {index}/{len(validation)}", flush=True)
        selection = json.loads((assets / "verified_m5_selection.json").read_text())
        m4_selection = json.loads((assets / "m4_m4_selection.json").read_text())
        if len(records) != EXPECTED["val"]["images"] or {r["split"] for r in records} != {"val"}:
            raise ValueError("Incomplete validation parity predictions.")
        cache.with_suffix(".sha256").write_text(sha_file(cache) + "\n")
        summary = parity_summary(records, selection, m4_selection)
        dump(release / "evidence" / "export_parity_summary.json", summary)
        write_csv(release / "evidence" / "val_export_parity_summary.csv", [{k: v for k, v in r.items()
                  if not isinstance(v, (list, dict))} for r in records])
        manifest = {"release_stage": "M8_part1_export_candidate", "run_id": run_id, "cohort": COHORT_NAME,
                    "export_protocol_sha256": protocol_sha, "phase1_code_commit": P1_COMMIT,
                    "phase2_publication_commit": None, "test_evaluation_performed": False,
                    "export_parity_passed": summary["all_checks_passed"],
                    "ready_for_service_assembly": summary["all_checks_passed"],
                    "models": receipts, "native_model_sha256": {n: s["sha256"] for n, s in MODEL_SPECS.items()},
                    "class_order": {"classifier": ["Real", "Fake"], "detector": ["edited_region"],
                                    "segmenter": ["background", "edited_region"]},
                    "preprocessing": {"classifier": "EXIF transpose; RGB; saved ViTImageProcessor with torchvision backend; "
                                      "bicubic resize to 224x224; /255; ImageNet mean/std; NCHW float32",
                                      "detector": "RGB; OpenCV linear resize; centered 640x640 letterbox fill114; /255; NCHW float32",
                                      "segmenter": "RGB; PIL bilinear centered 640x640 letterbox fill0; /255; ImageNet mean/std; NCHW float32"},
                    "postprocessing": {"classifier": "softmax(logits)[Fake]; frozen threshold0.5; score is uncalibrated",
                                       "detector": "strict confidence>0.35; class0 NMS IoU0.45; max50; native xyxy clipped to image dimensions",
                                       "segmenter": "sigmoid; crop letterbox; PIL float bilinear native-size restoration; threshold>=0.5; no morphology"},
                    "branch_execution": "Run all three branches for every accepted input; segmentation is independent of detections.",
                    "system_review_rule": None, "http_service_built": False, "latency_benchmark_completed": False,
                    "final_go_no_go_decision": None,
                    "scope": "Research prototype. CSI aligned complete-family cohort; no Saudi deployment-performance claim."}
        dump(release / "manifest.json", manifest)
        failed = [name for name, passed in summary["checks"].items() if not passed]
        status.update(status="export_parity_passed" if not failed else "export_parity_failed", failed_checks=failed)
        print("Export parity:", status["status"], flush=True)
        if failed:
            print("Return the small review ZIP. Keep thresholds unchanged until the differences are reviewed:", failed, flush=True)
        else:
            print("All validation parity checks passed. Next: HTTP service and measured latency; test remains unopened.", flush=True)
        from m8_backup import backup_release
        receipt = backup_release(release, work, status["status"], run_id)
        status["cloud_backup_confirmed"] = receipt["confirmed"]
    except Exception as error:
        message = traceback.format_exc()
        for key in [os.environ.get("WANDB_API_KEY"), os.environ.get("HF_TOKEN")]:
            if key:
                message = message.replace(key, "[REDACTED]")
        (work / "error.txt").write_text(message)
        status.update(status="blocked", error_type=type(error).__name__,
                      error="See error.txt in the review ZIP. No test evaluation was performed.")
        print(message, flush=True)
    dump(status_path, status)
    packaged = package_outputs(work, release)
    print("STATUS:", status["status"], flush=True)
    for label, value in packaged.items():
        if value:
            print(label + ":", value, flush=True)
    return {**status, **packaged}
