"""Back up the model-only ONNX candidate; never traverse Kaggle inputs or native checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import os

from cohort_data import dump, sha_file


def release_allowlist():
    from m8_export import SOURCES, ASSETS
    allowed = {"runtime.json", "requirements-lock-executed.txt", "native_model_load.json",
               "export_protocol.json", "manifest.json", "cloud_backup_receipt.json",
               "classifier/config.json", "classifier/preprocessor_config.json",
               "evidence/val_manifest.csv", "evidence/cohort_preflight.json",
               "evidence/val_export_parity_predictions.jsonl", "evidence/val_export_parity_predictions.lock.json",
               "evidence/val_export_parity_predictions.sha256", "evidence/export_parity_summary.json",
               "evidence/val_export_parity_summary.csv"}
    allowed.update("source/" + name for name in SOURCES + ASSETS)
    for name in ["classifier", "detector", "segmenter"]:
        allowed.update({"models/" + name + ".onnx", "models/" + name + ".export.json"})
    return allowed


def backup_release(release, work, status, run_id):
    import wandb
    release, work = Path(release), Path(work)
    receipt = {"stage": "M8_export", "run_id": run_id, "status": status,
               "confirmed": False, "test_evaluation_performed": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    run = None
    try:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is unavailable; download both ZIPs immediately.")
        if not all((release / "models" / (name + ".onnx")).is_file()
                   for name in ["classifier", "detector", "segmenter"]):
            raise RuntimeError("All three ONNX graphs are required for a complete export backup.")
        # release is constructed by our script. Additional files are rejected, not silently uploaded.
        allowed = release_allowlist()
        files = sorted(p for p in release.rglob("*") if p.is_file() and p.name != "cloud_backup_receipt.json")
        for p in files:
            if p.is_symlink() or p.relative_to(release).as_posix() not in allowed:
                raise ValueError("Unexpected file in model-only release: " + p.name)
            if any(part in {"data", "images", "masks", "native_checkpoints", "wandb", "__pycache__"}
                   for part in p.relative_to(release).parts):
                raise ValueError("Excluded directory in export release.")
        run = wandb.init(entity="f1k80-kaust", project="Real-Abshar-Phase2",
                         name="M8_export_" + run_id, job_type="onnx_export_validation",
                         config={"status": status, "test_evaluated": False, "precision": "FP32",
                                 "batch": 1, "classifier_size": 224, "region_model_size": 640},
                         dir=str(work), reinit="finish_previous",
                         settings=wandb.Settings(disable_git=True, save_code=False))
        artifact = wandb.Artifact("real-abshar-m8-export-" + run_id, type="model",
                                  metadata={"status": status, "test_evaluated": False})
        inventory = []
        for p in files:
            relative = p.relative_to(release).as_posix()
            artifact.add_file(str(p), name=relative, policy="mutable")
            inventory.append({"path": relative, "bytes": p.stat().st_size, "sha256": sha_file(p)})
        logged = run.log_artifact(artifact, aliases=["export_candidate"])
        logged.wait(timeout=600)
        receipt.update(confirmed=True, artifact_qualified_name=logged.qualified_name,
                       artifact_url=logged.url, file_count=len(inventory), files=inventory)
        run.summary.update({"export_status": status, "backup_confirmed": True})
    except Exception as error:
        message = str(error)
        key = os.environ.get("WANDB_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        receipt["error"] = type(error).__name__ + ": " + message[:1000]
        print("Export cloud backup is NOT confirmed. Download and save both ZIPs.", flush=True)
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        dump(work / "cloud_backup_receipt.json", receipt)
    return receipt
