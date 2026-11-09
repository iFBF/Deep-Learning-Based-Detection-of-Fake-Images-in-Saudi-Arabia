"""Notebook orchestration. New worker processes use the exact frozen service source."""
from pathlib import Path
import json
import os
import subprocess
import sys
import zipfile

from final_assets import dump, sha, restore_service, restore_attached_cache


def launch(stage, release, prepared, work, assets):
    command = [sys.executable, str(assets / "final_worker.py"), "--stage", stage,
               "--release", str(release), "--prepared", str(prepared), "--work", str(work)]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    with (work / (stage + "_worker.log")).open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            for value in [os.environ.get("WANDB_API_KEY"), os.environ.get("HF_TOKEN")]:
                if value:
                    line = line.replace(value, "[REDACTED]")
            log.write(line);log.flush()
            print(line, end="", flush=True)
        return process.wait()


def state(work, status):
    ledger = json.loads((work / "test_ledger.json").read_text()) if (work / "test_ledger.json").is_file() else None
    result = {"status": status, "test_opened": ledger is not None,
              "saved_test_predictions": len(list((work / "test_cache").glob("*.npz"))),
              "test_completed": (work / "reports" / "phase2_test_metrics.json").is_file(),
              "thresholds_changed_on_test": False}
    dump(work / "run_status.json", result)
    return result


def package(work, assets):
    work = Path(work)
    review = work.parent / "Real-Abshar_Phase2_Final_Test_Review.zip"
    metadata_names = ["protocol.json", "validation_gate.json", "test_ledger.json", "evaluation_environment.json", "run_status.json",
                      "cloud_backup_receipt.json", "test_started.json", "test_execution_summary.json", "worker_failure.json", "error.txt",
                      "validation_worker.log", "test_worker.log"]
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in metadata_names:
            if (work / name).is_file():
                z.write(work / name, name)
        for directory in ["reports", "figures"]:
            for path in sorted((work / directory).glob("*")):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".png"}:
                    z.write(path, path.relative_to(work).as_posix())
        for name in ["final_assets.py", "final_controller.py", "final_worker.py", "final_records.py", "final_metrics.py", "final_ledger.py", "final_report.py", "detection_metrics.py", "SERVICE_REVIEW.md"]:
            if (assets / name).is_file():
                z.write(assets / name, "evaluator_source/" + name)
        for name in ["manifest.json", "release_lock.json", "evidence/service_validation_gate.json", "evidence/http_benchmark.json", "evidence/runtime.json"]:
            source = work / "frozen_service" / name
            if source.is_file():
                z.write(source, "frozen_service/" + name)
    cache = work.parent / "Real-Abshar_Phase2_Final_Test_Cache.zip"
    with zipfile.ZipFile(cache, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name in metadata_names:
            if (work / name).is_file():
                z.write(work / name, name)
        for directory in ["validation_cache", "test_cache", "reports"]:
            for path in sorted((work / directory).glob("*")):
                if path.is_file() and path.suffix in {".npz", ".json", ".jsonl", ".csv", ".md"}:
                    z.write(path, path.relative_to(work).as_posix())
    return {"review_zip": str(review), "prediction_cache_zip": str(cache)}


def prepare(input_root, work_root, assets_root, prepared_root=None):
    work, assets = Path(work_root).resolve(), Path(assets_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    try:
        release = restore_service(input_root, assets, work)
        sys.path.insert(0, str(release))
        from cohort_data import locate_prepared, read_preparation
        prepared = locate_prepared(input_root, work, prepared_root)
        rows, _ = read_preparation(prepared)  # source/split metadata only
        restore_attached_cache(input_root, work)
        dump(work / "local_paths.json", {"release": str(release), "prepared": str(prepared)})
        if launch("validation", release, prepared, work, assets) != 0:
            result = state(work, "blocked_before_test")
        else:
            result = state(work, "validation_gate_passed")
    except Exception as error:
        message = type(error).__name__ + ": " + str(error)
        if os.environ.get("WANDB_API_KEY"):
            message = message.replace(os.environ["WANDB_API_KEY"], "[REDACTED]")
        (work / "error.txt").write_text(message)
        print(message, flush=True)
        result = state(work, "blocked_before_test")
    return {**result, **package(work, assets)}


def run_reserved_test(work_root, assets_root):
    work, assets = Path(work_root).resolve(), Path(assets_root).resolve()
    gate = json.loads((work / "validation_gate.json").read_text()) if (work / "validation_gate.json").is_file() else {}
    current = json.loads((work / "run_status.json").read_text()) if (work / "run_status.json").is_file() else {}
    if not gate.get("all_checks_passed") or current.get("status") not in {"validation_gate_passed", "final_test_complete"}:
        print("BLOCKED: validation gate has not passed. Do not change test settings; send the review ZIP.", flush=True)
        return {**state(work, "blocked_before_test"), **package(work, assets)}
    paths = json.loads((work / "local_paths.json").read_text())
    code = launch("test", Path(paths["release"]), Path(paths["prepared"]), work, assets)
    result = state(work, "final_test_complete" if code == 0 else "blocked_resume_same_frozen_job")
    outputs = package(work, assets)
    print("STATUS:", result["status"], flush=True)
    for name, path in outputs.items():
        print(name + ":", path, flush=True)
    return {**result, **outputs}
