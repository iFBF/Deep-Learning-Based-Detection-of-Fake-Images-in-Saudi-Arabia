"""Back up model checkpoints and experiment metadata, without dataset images/masks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import tempfile
import uuid

from cohort_data import dump, safe_path, sha_file

BACKUP_EVERY_EPOCHS = 5
BACKUP_WAIT_SECONDS = 300


def backup_files(work_root):
    root = Path(work_root)
    selected = set()
    for name in ["m4_protocol_lock.json", "environment.json", "m4_selection.json", "backup_session_id.txt"]:
        path = root / name
        if path.is_file():
            selected.add(path)
    for folder, suffixes in {"cohort": {".csv", ".json"}, "metrics": {".csv", ".json"},
                             "validation_predictions": {".json"}, "wandb_ids": {".txt", ".json"},
                             "source": {".py"}}.items():
        for path in (root / folder).glob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                selected.add(path)
    pretrained = root / "pretrained" / "yolo11n.pt"
    if pretrained.is_file():
        selected.add(pretrained)
    for run in ["A_standard_augmentation", "B_reduced_augmentation"]:
        for relative in ["args.yaml", "results.csv", "training_complete.json", "weights/best.pt", "weights/last.pt"]:
            path = root / "runs" / run / relative
            if path.is_file():
                selected.add(path)
    # Scientific plots are regenerated from backed-up numbers. No photographs enter the artifact.
    for path in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Backup file escapes its work folder: " + str(path))
    return sorted(selected)


def should_backup(epoch, stopping=False):
    return epoch == 1 or epoch % BACKUP_EVERY_EPOCHS == 0 or stopping


def upload_backup(run, work_root, stage):
    import wandb
    if not os.environ.get("WANDB_API_KEY") or getattr(run.settings, "mode", None) != "online":
        raise RuntimeError("Online W&B is required for recovery backups. Enable the WANDB_API_KEY secret and Internet before training.")
    root = Path(work_root)
    session = root / "backup_session_id.txt"
    if not session.exists():
        session.write_text(uuid.uuid4().hex[:12])
    paths = backup_files(root)
    if not paths:
        raise RuntimeError("No experiment files exist to back up.")
    manifest = {"schema": "real-abshar-m4-backup-v1", "stage": stage,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_images_or_masks_included": False,
                "files": {p.relative_to(root).as_posix(): {"sha256": sha_file(p), "bytes": p.stat().st_size}
                          for p in paths}}
    manifest_path = root / "cloud_backup_manifest.json"
    dump(manifest_path, manifest)
    artifact = wandb.Artifact("real-abshar-m4-backup-" + session.read_text().strip(), type="model",
                              metadata={"stage": stage, "dataset_images_included": False,
                                        "file_count": len(paths), "test_used": False})
    for path in paths:
        artifact.add_file(str(path), name=path.relative_to(root).as_posix(), policy="mutable")
    artifact.add_file(str(manifest_path), name="cloud_backup_manifest.json", policy="mutable")
    logged = run.log_artifact(artifact, aliases=["latest"])
    logged.wait(timeout=BACKUP_WAIT_SECONDS)
    # Publish a receipt only after the SDK confirms the upload completed.
    receipt = {"stage": stage, "artifact": logged.qualified_name, "url": logged.url,
               "confirmed_utc": datetime.now(timezone.utc).isoformat(),
               "file_count": len(paths), "dataset_images_or_masks_included": False}
    dump(root / "cloud_backup_receipt.json", receipt)
    run.summary["latest_model_backup"] = receipt["artifact"]
    print("BACKUP_CONFIRMED:", stage, receipt["artifact"], flush=True)
    print("Backup link:", receipt["url"], flush=True)
    return receipt


def restore_backup(reference, destination):
    """Restore an own-project artifact into its original work path; reject corrupt/conflicting files."""
    import wandb
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m4_restore_", dir=destination.parent) as directory:
        downloaded = Path(wandb.Api().artifact(reference).download(root=directory))
        manifest = json.loads((downloaded / "cloud_backup_manifest.json").read_text())
        if manifest.get("schema") != "real-abshar-m4-backup-v1" or manifest.get("dataset_images_or_masks_included") is not False:
            raise ValueError("Unexpected model-backup schema.")
        for name, expected in manifest["files"].items():
            source, target = safe_path(downloaded, name), safe_path(destination, name)
            if not source.is_file() or source.stat().st_size != expected["bytes"] or sha_file(source) != expected["sha256"]:
                raise ValueError("Corrupt backup file: " + name)
            if target.exists() and (not target.is_file() or sha_file(target) != expected["sha256"]):
                raise ValueError("Restore would replace a different local file: " + str(target))
        for name in manifest["files"]:
            source, target = safe_path(downloaded, name), safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(downloaded / "cloud_backup_manifest.json", destination / "cloud_backup_manifest.json")
    print("Restored model/metadata backup:", reference)
    print("Keep the original WORK_ROOT path and attach the prepared dataset before resuming.")
    return destination
