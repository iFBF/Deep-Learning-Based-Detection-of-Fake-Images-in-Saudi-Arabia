"""Kaggle M4 experiment: train two detectors and select on validation only."""
from pathlib import Path
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from cohort_data import (COHORT_NAME, COHORT_UID_SHA256, EXPECTED, build_training_view,
                         dump, locate_prepared, safe_path, sha_file, write_csv)
from detection_metrics import operating_metrics, threshold_sweep
from wandb_backup import BACKUP_EVERY_EPOCHS, upload_backup

ULTRALYTICS_VERSION = "8.3.221"
WANDB_VERSION = "0.28.1"
PRETRAINED_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
PRETRAINED_BYTES = 5613764
IMGSZ = 640
LOW_CONFIDENCE = 0.001
COMPARISON_NMS = 0.7
MAX_DET = 50
COMMON = dict(epochs=30, patience=8, imgsz=IMGSZ, batch=8, workers=2, device=0,
              optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
              nbs=64, seed=42, deterministic=True, amp=False, cache=False, rect=False,
              cos_lr=False, val=True, save=True, save_period=1, plots=False,
              conf=LOW_CONFIDENCE, iou=COMPARISON_NMS, max_det=MAX_DET,
              degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
              mixup=0.0, cutmix=0.0, copy_paste=0.0, fliplr=0.5)
EXPERIMENTS = {
    "A_standard_augmentation": dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                                    mosaic=1.0, close_mosaic=10, translate=0.1, scale=0.5),
    "B_reduced_augmentation": dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                                   mosaic=0.0, close_mosaic=0, translate=0.05, scale=0.1),
}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def environment():
    import torch
    names = ["ultralytics", "wandb", "torch", "torchvision", "numpy", "Pillow",
             "scipy", "opencv-python", "albumentations", "matplotlib"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions["ultralytics"] != ULTRALYTICS_VERSION or versions["wandb"] != WANDB_VERSION:
        raise RuntimeError("Run the pinned dependency cell in a fresh Kaggle session before importing this module.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator, then restart and run all cells.")
    return {"python": platform.python_version(), "packages": versions,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "gpu_device": 0, "test_inference_performed": False}


def initial_checkpoint(work_root):
    destination = work_root / "pretrained" / "yolo11n.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".download")
        with urllib.request.urlopen(PRETRAINED_URL, timeout=60) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != PRETRAINED_BYTES:
            temporary.unlink()
            raise ValueError("The official pretrained asset size changed; review the download before training.")
        temporary.replace(destination)
    if destination.stat().st_size != PRETRAINED_BYTES:
        raise ValueError("The local pretrained checkpoint has an unexpected size.")
    return destination


def lock_protocol(work_root, pretrained, env):
    sources = {p.name: sha_file(p) for p in Path(__file__).parent.glob("*.py")}
    protocol = {"cohort": COHORT_NAME, "cohort_uid_sha256": COHORT_UID_SHA256,
                "splits": EXPECTED, "architecture": "YOLO11n, one class: edited_region",
                "pretrained_url": PRETRAINED_URL, "pretrained_sha256": sha_file(pretrained),
                "common_train_settings": COMMON, "experiments": EXPERIMENTS,
                "comparison": {"split": "val", "input_size": [IMGSZ, IMGSZ],
                               "confidence_floor": LOW_CONFIDENCE, "nms_iou": COMPARISON_NMS,
                               "criterion": "highest validation mAP50-95; tie resolves to run A"},
                "operating_point": {"split": "val", "matching_iou": 0.5,
                                    "nms_iou_grid": [0.45, 0.6, 0.75],
                                    "confidence_grid": [0.01, 0.025] + [round(i * 0.05, 3) for i in range(1, 20)],
                                    "criterion": "max F1; tie: recall, precision, higher confidence, lower NMS IoU"},
                "source_sha256": sources,
                "run_purpose": "replacement_after_unsaved_session_outputs",
                "checkpoint_backup": {"backend": "W&B Artifacts", "every_epochs": BACKUP_EVERY_EPOCHS,
                                      "also": ["preflight", "epoch_1", "stopping_epoch", "run_complete", "M4_complete"],
                                      "dataset_images_or_masks_uploaded": False},
                "package_versions": env["packages"], "python": env["python"],
                "test_evaluation_authorized_in_this_notebook": False}
    path = work_root / "m4_protocol_lock.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("This work folder has a different locked protocol. Restore its original files/environment or use a new folder.")
    dump(path, protocol)
    dump(work_root / "environment.json", env)
    return protocol


def start_wandb(work_root, project, entity, name, configuration):
    import wandb
    online = bool(os.environ.get("WANDB_API_KEY"))
    if not online:
        raise RuntimeError("Recovery training requires an online WANDB_API_KEY secret so checkpoints can be backed up.")
    ids = work_root / "wandb_ids"
    ids.mkdir(parents=True, exist_ok=True)
    path = ids / (name + ".txt")
    if not path.exists():
        path.write_text(wandb.util.generate_id())
    kwargs = dict(project=project, name=name, id=path.read_text().strip(),
                  config=configuration, mode="online" if online else "offline",
                  dir=str(work_root), save_code=False,
                  tags=["phase2", "M4", "replacement-after-session-output-loss"])
    if entity:
        kwargs["entity"] = entity
    if online:
        kwargs["resume"] = "allow"
    # Manual logging prevents duplicate runs from Ultralytics' optional integration.
    run = wandb.init(**kwargs)
    dump(work_root / "wandb_ids" / (name + ".json"),
         {"mode": "online" if online else "offline", "project": project,
          "entity": entity, "id": run.id, "url": run.url if online else None})
    return run


def free_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def train_experiment(work_root, data_yaml, pretrained, name, augmentation, project, entity, protocol):
    from ultralytics import YOLO
    directory = work_root / "runs" / name
    best = directory / "weights" / "best.pt"
    last = directory / "weights" / "last.pt"
    marker = directory / "training_complete.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if best.is_file() and previous["best_sha256"] == sha_file(best):
            print("Reusing completed training:", name)
            return best
        raise ValueError("A completed run's best checkpoint changed: " + name)
    if directory.exists() and not last.exists() and (directory / "results.csv").exists():
        raise RuntimeError("Partial run has no last.pt to resume. Retain its outputs and use a fresh WORK_ROOT.")
    run = start_wandb(work_root, project, entity, name,
                     {"cohort": COHORT_NAME, "split_counts": EXPECTED, "settings": COMMON | augmentation,
                      "pretrained_sha256": protocol["pretrained_sha256"], "test_used": False})
    model = YOLO(str(last if last.is_file() else pretrained))

    def log_epoch(trainer):
        values = {"epoch": int(trainer.epoch) + 1}
        for key, value in getattr(trainer, "metrics", {}).items():
            numeric = number(value)
            if numeric is not None:
                values[key] = numeric
        run.log(values)

    model.add_callback("on_fit_epoch_end", log_epoch)

    def back_up_checkpoint(trainer):
        from wandb_backup import should_backup
        epoch = int(trainer.epoch) + 1
        if should_backup(epoch, bool(getattr(trainer, "stop", False))):
            upload_backup(run, work_root, f"{name}_epoch_{epoch}")

    model.add_callback("on_model_save", back_up_checkpoint)
    started = time.monotonic()
    try:
        upload_backup(run, work_root, name + "_preflight")
        if last.is_file():
            print("Resuming saved checkpoint:", name)
            try:
                model.train(resume=True)
            except AssertionError as error:
                # A session may end after Ultralytics strips the final optimizer but before our marker is saved.
                if "nothing to resume" not in str(error).lower() or not best.is_file() or not (directory / "results.csv").is_file():
                    raise
                print("Checkpoint already completed training; retaining its best.pt.")
        else:
            model.train(data=str(data_yaml), project=str(work_root / "runs"), name=name,
                        exist_ok=True, **COMMON, **augmentation)
        if not best.is_file():
            raise FileNotFoundError("Training did not produce best.pt for " + name)
        completed_epochs = len(list(csv.DictReader((directory / "results.csv").open())))
        metadata = {"run": name, "best_sha256": sha_file(best), "completed_epochs": completed_epochs,
                    "checkpoint_selection": "Ultralytics validation fitness within this run",
                    "session_training_seconds": time.monotonic() - started, "test_used": False}
        dump(marker, metadata)
        run.summary.update(metadata)
        upload_backup(run, work_root, name + "_complete")
    finally:
        run.finish()
        del model
        free_gpu()
    return best


def validate_checkpoint(checkpoint, data_yaml, work_root, name, nms_iou):
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    metrics = model.val(data=str(data_yaml), split="val", imgsz=IMGSZ, batch=8, workers=2,
                        device=0, half=False, rect=False, conf=LOW_CONFIDENCE,
                        iou=nms_iou, max_det=MAX_DET, plots=False, save_json=False,
                        project=str(work_root / "validation"), name=name, exist_ok=True, verbose=False)
    result = {"split": "val", "input_width": IMGSZ, "input_height": IMGSZ,
              "preprocessing": "aspect-preserving letterbox to 640x640; native-coordinate scoring",
              "confidence_floor": LOW_CONFIDENCE, "nms_iou": nms_iou, "max_det": MAX_DET,
              "mAP50_95": float(metrics.box.map), "mAP50": float(metrics.box.map50),
              "checkpoint_sha256": sha_file(checkpoint)}
    dump(work_root / "metrics" / (name + ".json"), result)
    if name == "selected_detector_val_AP":
        curve = metrics.box.curves_results[0]
        recall, precision = np.asarray(curve[0]), np.asarray(curve[1])
        if precision.size:
            precision = precision.reshape(-1, len(recall))[0]
            write_csv(work_root / "metrics" / "val_selected_PR.csv",
                      [{"recall": float(x), "precision": float(y)} for x, y in zip(recall, precision, strict=True)])
    del model, metrics
    free_gpu()
    return result


def validation_predictions(checkpoint, rows, work_root, nms_iou):
    from ultralytics import YOLO
    path = work_root / "validation_predictions" / f"val_nms_{nms_iou:.2f}.json"
    signature = {"split": "val", "input_size": [IMGSZ, IMGSZ], "confidence_floor": LOW_CONFIDENCE,
                 "nms_iou": nms_iou, "max_det": MAX_DET, "checkpoint_sha256": sha_file(checkpoint),
                 "cohort_uid_sha256": COHORT_UID_SHA256}
    if path.exists():
        cache = json.loads(path.read_text())
        if cache["settings"] != signature or set(cache["predictions"]) != {r["uid"] for r in rows}:
            raise ValueError("The saved validation predictions belong to different settings.")
        return cache["predictions"]
    model = YOLO(str(checkpoint))
    paths = [str(safe_path(work_root / "data_view", r["export_image"])) for r in rows]
    predictions = {}
    for row, result in zip(rows, model.predict(source=paths, stream=True, batch=8, device=0,
                                             imgsz=IMGSZ, rect=False, half=False,
                                             conf=LOW_CONFIDENCE, iou=nms_iou, max_det=MAX_DET,
                                             save=False, verbose=False), strict=True):
        if Path(result.path).stem != row["uid"]:
            raise ValueError("Prediction image order changed; refusing to mispair predictions.")
        if tuple(result.orig_shape) != (int(row["height"]), int(row["width"])):
            raise ValueError("Prediction dimensions differ from the manifest.")
        predictions[row["uid"]] = [{"box": [float(v) for v in box], "score": float(score), "class": int(cls)}
                                   for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                                               result.boxes.conf.cpu().tolist(),
                                                               result.boxes.cls.cpu().tolist(), strict=True)]
    dump(path, {"settings": signature, "predictions": predictions})
    del model
    free_gpu()
    return predictions


def make_figures(work_root, rows, comparison, grid, selection, predictions, per_image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figures = work_root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison))
    ax.bar(x - 0.17, [r["mAP50_95"] for r in comparison], width=0.34, label="mAP50–95")
    ax.bar(x + 0.17, [r["mAP50"] for r in comparison], width=0.34, label="mAP50")
    ax.set_xticks(x, [r["run"].replace("_", " ") for r in comparison], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average precision")
    ax.legend()
    ax.set_title("Validation • letterbox 640×640 • confidence floor 0.001\nNMS IoU 0.70 • AP matching IoU 0.50:0.95 • max detections 50", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "val_detector_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for nms in [0.45, 0.6, 0.75]:
        part = [r for r in grid if r["nms_iou"] == nms]
        ax.plot([r["confidence"] for r in part], [r["f1"] for r in part], marker=".", label=f"NMS IoU {nms:.2f}")
    ax.scatter([selection["confidence"]], [selection["f1"]], marker="*", s=150, color="black", zorder=5, label="Selected")
    ax.set(xlabel="Confidence threshold (sweep 0.01, 0.025, 0.05…0.95)", ylabel="Detection F1", ylim=(0, 1))
    ax.set_title("Validation • letterbox 640×640 • matching IoU ≥ 0.50\nConfidence and NMS thresholds selected jointly; max detections 50", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "val_threshold_sweep.png", dpi=160)
    plt.close(fig)

    pr_path = work_root / "metrics" / "val_selected_PR.csv"
    if pr_path.exists():
        pr = list(csv.DictReader(pr_path.open()))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([float(r["recall"]) for r in pr], [float(r["precision"]) for r in pr])
        ax.set(xlabel="Recall", ylabel="Interpolated precision", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f'Validation PR at matching IoU 0.50 • letterbox 640×640\n'
                     f'Confidence floor 0.001 • NMS IoU {selection["nms_iou"]:.2f} • max detections 50', fontsize=10)
        fig.tight_layout()
        fig.savefig(figures / "val_selected_PR.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for item in comparison:
        with (work_root / "runs" / item["run"] / "results.csv").open() as stream:
            history = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
        epochs = [float(r["epoch"]) for r in history]
        short_name = item["run"][0]
        axes[0].plot(epochs, [float(r["train/box_loss"]) for r in history], label=f'{short_name} train')
        axes[0].plot(epochs, [float(r["val/box_loss"]) for r in history], linestyle="--", label=f'{short_name} val')
        axes[1].plot(epochs, [float(r["metrics/mAP50-95(B)"]) for r in history], label=short_name)
    axes[0].set(xlabel="Epoch", ylabel="Recorded box regression loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation mAP50–95", ylim=(0, 1))
    for ax in axes:
        ax.legend()
    fig.suptitle('Train/validation learning curves • letterbox 640×640\n'
                 'Validation confidence floor 0.001 • NMS IoU 0.70 • AP matching IoU 0.50:0.95', fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "train_val_learning_curves.png", dpi=160)
    plt.close(fig)

    # Deterministic validation examples; this is a development gallery, not the final test gallery.
    candidates = []
    for rule in [lambda r: r["fn"] > 0, lambda r: r["fp"] > 0,
                 lambda r: r["tp"] > 0, lambda r: r["kind"] == "pristine" and r["fp"] == 0]:
        subset = sorted((r for r in per_image if rule(r)), key=lambda r: (-r["fn"], -r["fp"], r["uid"]))
        candidates.extend([r["uid"] for r in subset[:2]])
    uids = list(dict.fromkeys(candidates))
    uids += [r["uid"] for r in per_image if r["uid"] not in uids]
    uids = uids[:8]
    by_uid = {r["uid"]: r for r in rows}
    count_by_uid = {r["uid"]: r for r in per_image}
    fig, axes = plt.subplots(4, 2, figsize=(12, 17))
    for ax, uid in zip(axes.flat, uids, strict=True):
        row = by_uid[uid]
        with Image.open(safe_path(work_root / "data_view", row["export_image"])) as opened:
            ax.imshow(opened.convert("RGB"))
        gt = json.loads(row["box_xyxy_exclusive"])
        if gt is not None:
            x0, y0, x1, y1 = gt
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00e070", linewidth=2))
        for prediction in predictions[uid]:
            if prediction["score"] < selection["confidence"]:
                continue
            x0, y0, x1, y1 = prediction["box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff9e24", linewidth=1.7))
            ax.text(x0, y0, f'{prediction["score"]:.2f}', color="#ff9e24", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
        counts = count_by_uid[uid]
        ax.set_title(f'{uid} • TP {counts["tp"]}, FP {counts["fp"]}, FN {counts["fn"]}', fontsize=9)
        ax.axis("off")
    fig.suptitle(f'Validation development gallery • letterbox {IMGSZ}×{IMGSZ}\n'
                 f'Confidence ≥ {selection["confidence"]:.3f} • NMS IoU {selection["nms_iou"]:.2f} • matching IoU ≥ 0.50\n'
                 'Green: ground-truth union region; orange: predicted region; coordinates shown on native images', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "val_development_gallery.png", dpi=130)
    plt.close(fig)


def bundle_outputs(work_root, selected_checkpoint, selection):
    output = work_root.parent
    review_path = output / "Real-Abshar_Phase2_M4_Results_Review.zip"
    models_path = output / "Real-Abshar_Phase2_M4_Models.zip"
    with zipfile.ZipFile(review_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ["cohort", "metrics", "figures", "validation_predictions", "wandb_ids", "source"]:
            for path in sorted((work_root / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(work_root).as_posix())
        for filename in ["m4_protocol_lock.json", "environment.json", "m4_selection.json", "cloud_backup_receipt.json"]:
            archive.write(work_root / filename, filename)
        for name in EXPERIMENTS:
            for filename in ["args.yaml", "results.csv", "training_complete.json"]:
                path = work_root / "runs" / name / filename
                if path.exists():
                    archive.write(path, f"runs/{name}/{filename}")
    with zipfile.ZipFile(models_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(work_root / "pretrained" / "yolo11n.pt", "pretrained/yolo11n.pt")
        for name in EXPERIMENTS:
            archive.write(work_root / "runs" / name / "weights" / "best.pt", f"runs/{name}/best.pt")
        archive.write(selected_checkpoint, "selected/detector_best.pt")
        for filename in ["m4_protocol_lock.json", "environment.json", "m4_selection.json", "cloud_backup_receipt.json"]:
            archive.write(work_root / filename, filename)
        for path in sorted((work_root / "source").glob("*.py")):
            archive.write(path, "source/" + path.name)
    dump(work_root / "output_archives.json", {"review": {"file": review_path.name, "sha256": sha_file(review_path)},
         "models": {"file": models_path.name, "sha256": sha_file(models_path)},
         "note": "Keep both archives; send the small Results_Review archive for review. Test inference was not run."})
    return review_path, models_path


def run(input_root="/kaggle/input", work_root="/kaggle/working/real_abshar_phase2_m4_recovery_v1",
        prepared_root=None, wandb_project="Real-Abshar-Phase2", wandb_entity=None):
    import ultralytics
    # Manual W&B logging plus curated model/metadata artifacts; dataset images/masks stay in Kaggle.
    ultralytics.settings.update({"wandb": False})
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    env = environment()
    root = locate_prepared(input_root, work_root, prepared_root)
    print("Prepared input:", root)
    rows, data_yaml = build_training_view(root, work_root)
    pretrained = initial_checkpoint(work_root)
    protocol = lock_protocol(work_root, pretrained, env)
    checkpoints, comparisons = {}, []
    for name, augmentation in EXPERIMENTS.items():
        checkpoint = train_experiment(work_root, data_yaml, pretrained, name, augmentation,
                                      wandb_project, wandb_entity, protocol)
        checkpoints[name] = checkpoint
        metrics = validate_checkpoint(checkpoint, data_yaml, work_root, name + "_common_val", COMPARISON_NMS)
        comparisons.append({"run": name, **metrics})
    write_csv(work_root / "metrics" / "val_run_comparison.csv", comparisons)
    winner = max(comparisons, key=lambda r: r["mAP50_95"])
    checkpoint = checkpoints[winner["run"]]
    print("Validation-selected detector:", winner["run"])
    validation = [r for r in rows if r["split"] == "val"]
    caches = {nms: validation_predictions(checkpoint, validation, work_root, nms) for nms in [0.45, 0.6, 0.75]}
    selection, grid = threshold_sweep(validation, caches, IMGSZ)
    write_csv(work_root / "metrics" / "val_threshold_sweep.csv", grid)
    selected_predictions = caches[selection["nms_iou"]]
    operating, per_image = operating_metrics(validation, selected_predictions, selection["confidence"], input_size=IMGSZ)
    write_csv(work_root / "metrics" / "val_operating_per_image.csv", per_image)
    selected_ap = validate_checkpoint(checkpoint, data_yaml, work_root, "selected_detector_val_AP", selection["nms_iou"])
    final = {"selected_run": winner["run"], "checkpoint_sha256": sha_file(checkpoint),
             "cohort_uid_sha256": COHORT_UID_SHA256, "selection_split": "val",
             "input_size": [IMGSZ, IMGSZ], "confidence": selection["confidence"],
             "nms_iou": selection["nms_iou"], "max_det": MAX_DET,
             "matching_iou_for_operating_metrics": 0.5,
             "validation_operating_metrics": operating, "validation_AP": selected_ap,
             "test_evaluation_performed": False,
             "limitations": ["132 complete aligned source families; not the full CSI-IMD benchmark",
                             "one union region per edited image; not semantic object instances",
                             "one seed per augmentation condition; no multi-seed uncertainty claim",
                             "M5 segmentation and M8 integration still required"]}
    dump(work_root / "m4_selection.json", final)
    make_figures(work_root, validation, comparisons, grid, selection, selected_predictions, per_image)
    run_record = start_wandb(work_root, wandb_project, wandb_entity, "M4_validation_selection",
                             {"cohort": COHORT_NAME, "selected_run": winner["run"], "test_used": False})
    try:
        run_record.log({"val/mAP50_95": selected_ap["mAP50_95"], "val/mAP50": selected_ap["mAP50"],
                        "val/precision": operating["precision"], "val/recall": operating["recall"],
                        "val/f1": operating["f1"], "confidence": selection["confidence"], "nms_iou": selection["nms_iou"]})
        run_record.summary.update({"selected_run": winner["run"], "test_evaluated": False})
        upload_backup(run_record, work_root, "M4_completed")
    finally:
        run_record.finish()
    review, models = bundle_outputs(work_root, checkpoint, final)
    print(json.dumps({"selected_run": winner["run"], "validation_mAP50_95": selected_ap["mAP50_95"],
                      "validation_F1": operating["f1"], "confidence": selection["confidence"],
                      "nms_iou": selection["nms_iou"], "test_evaluated": False}, indent=2))
    print("SEND HERE:", review)
    print("KEEP FOR SEGMENTATION/INTEGRATION:", models)
    return final, review, models
