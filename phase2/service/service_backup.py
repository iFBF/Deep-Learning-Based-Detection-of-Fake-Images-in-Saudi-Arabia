"""Confirm a curated model/service backup; exclude all images, masks, video and user uploads."""
from datetime import datetime, timezone
from pathlib import Path
import os

from cohort_data import dump, sha_file


def backup_release(release, work, run_id, evaluation, latency):
    import wandb
    from m8_service_run import release_allowlist
    release, work = Path(release), Path(work)
    receipt = {"stage": "M8_service", "run_id": run_id, "confirmed": False, "test_evaluated": False,
               "images_masks_videos_included": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    run = None
    try:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY unavailable; download the private service release ZIP.")
        allowed = release_allowlist()
        files = sorted(p for p in release.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(release).parts)
        for path in files:
            if path.is_symlink() or path.relative_to(release).as_posix() not in allowed:
                raise ValueError("Unexpected file in curated service release: " + path.name)
        run = wandb.init(entity="f1k80-kaust", project="Real-Abshar-Phase2", name="M8_service_" + run_id,
                         job_type="http_service_validation_benchmark", dir=str(work), reinit="finish_previous",
                         settings=wandb.Settings(disable_git=True, save_code=False),
                         config={"split": "val", "test_evaluated": False, "provider": "CPUExecutionProvider", "precision": "FP32", "batch": 1})
        http_stage = next(r for r in latency["stages"] if r["stage"] == "http_client_ms")
        run.summary.update({"val/referral_balanced_accuracy": evaluation["referral_metrics"]["balanced_accuracy"],
                            "val/referral_macro_f1": evaluation["referral_metrics"]["macro_f1"],
                            "http/p50_ms": http_stage["p50_ms"], "http/p95_ms": http_stage["p95_ms"], "http/p99_ms": http_stage["p99_ms"],
                            "http/peak_sampled_process_RSS_MiB": latency["memory"]["peak_sampled_process_rss_MiB"]})
        artifact = wandb.Artifact("real-abshar-m8-service-" + run_id, type="model",
                                  metadata={"service_validation_passed": True, "test_evaluated": False})
        inventory = []
        for path in files:
            relative = path.relative_to(release).as_posix()
            artifact.add_file(str(path), name=relative, policy="mutable")
            inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha_file(path)})
        logged = run.log_artifact(artifact, aliases=["service_candidate"])
        logged.wait(timeout=600)
        receipt.update(confirmed=True, artifact_qualified_name=logged.qualified_name, artifact_url=logged.url,
                       file_count=len(inventory), files=inventory)
    except Exception as error:
        message = str(error)
        key = os.environ.get("WANDB_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        receipt["error"] = type(error).__name__ + ": " + message[:1000]
        print("Service cloud backup NOT confirmed. Download the service release ZIP immediately.", flush=True)
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        dump(work / "cloud_backup_receipt.json", receipt)
    return receipt
