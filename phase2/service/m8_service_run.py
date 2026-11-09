"""Assemble and freeze the selected service, then run real validation HTTP checks/benchmarks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from cohort_data import dump, locate_prepared, read_preparation, safe_path, sha_file, verify_row
from service_assets import restore_models, EXPORT_ARTIFACT
from service_decision import fit_policy

RUNTIME_FILES = ["app.py", "serve.py", "service_runtime.py", "service_decision.py", "service_geometry.py", "service_assets.py",
                 "service_benchmark.py", "service_demo.py", "m8_service_run.py", "service_backup.py", "cohort_data.py",
                 "segmentation_data.py", "segmentation_metrics.py", "export_metrics.py", "index.html",
                 "RUN_LOCAL.md", "MODEL_CARD.md", "requirements_serve.txt", "requirements_benchmark.txt", "M8_EXPORT_REVIEW.md"]
PROOF_FILES = ["export_manifest.json", "export_protocol.json", "export_parity_summary.json", "export_backup_receipt.json",
               "classifier_config.json", "classifier_processor.json", "val_export_predictions.jsonl", "val_export_predictions.sha256",
               "val_manifest.csv", "m4_selection.json", "m5_selection.json"]
EVIDENCE_FILES = ["runtime.json", "decision_policy.json", "val_referral_area_sweep.csv", "val_referral_ablations.csv",
                  "cohort_preflight.json", "http_contract_checks.json", "http_benchmark.json", "service_validation_gate.json",
                  "val_http_latency_and_parity.csv", "latency_stage_percentiles.csv", "val_http_predictions.jsonl",
                  "example_http_request_response.json", "validation_video_replay.json", "server_lifecycle.json"]


def release_allowlist():
    allowed = set(RUNTIME_FILES + ["manifest.json", "release_lock.json", "decision_policy.json",
                                  "classifier/config.json", "classifier/preprocessor_config.json"])
    allowed.update("models/" + n + ".onnx" for n in ["classifier", "detector", "segmenter"])
    allowed.update("provenance/" + n for n in PROOF_FILES)
    allowed.update("evidence/" + n for n in EVIDENCE_FILES)
    return allowed


def runtime_environment():
    packages = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "transformers", "onnxruntime", "numpy",
                "pillow", "opencv-python", "fastapi", "starlette", "pydantic", "uvicorn", "python-multipart", "httpx", "psutil", "wandb", "matplotlib"]}
    pinned = {"transformers": "5.16.1", "onnxruntime": "1.31.1", "fastapi": "0.141.1", "uvicorn": "0.54.0",
              "python-multipart": "0.0.33", "httpx": "0.28.1", "psutil": "7.2.2", "wandb": "0.28.1",
              "numpy": "2.1.3", "pillow": "13.0.0", "opencv-python": "4.14.0.94"}
    pinned.update(starlette="1.6.0", pydantic="2.13.4")
    for name, version in pinned.items():
        if packages[name] != version:
            raise RuntimeError(f"Run the dependency cell first: {name}=={version}; found{packages[name]}")
    if not packages["torch"].startswith("2.11.") or not packages["torchvision"].startswith("0.26."):
        raise RuntimeError("Use the successful Kaggle CPU image with Torch2.11.x / torchvision0.26.x.")
    cpu = "unknown"
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), "unknown")
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_model": cpu,
            "logical_cpus": os.cpu_count(), "packages": packages, "device": "CPU", "precision": "FP32",
            "created_at_utc": datetime.now(timezone.utc).isoformat()}


def make_manifest(release, export, policy, run_id):
    manifest = {"release_id": "real-abshar-m8-" + run_id, "stage": "service_candidate_before_reserved_test",
                "models": {name: {"file": "models/" + name + ".onnx", "sha256": spec["onnx_sha256"],
                                  "bytes": spec["bytes"], "input_shape": spec["input_shape"], "output_shape": spec["output_shape"],
                                  "native_checkpoint_sha256": spec["native_checkpoint_sha256"]} for name, spec in export["models"].items()},
                "class_names": export["class_order"], "phase1_code_commit": export["phase1_code_commit"],
                "phase2_publication_commit": None, "source_export_artifact": EXPORT_ARTIFACT,
                "batch": 1, "precision": "FP32", "cohort": export["cohort"],
                "runtime": {"provider": "CPUExecutionProvider", "transformers": "5.16.1", "onnxruntime": "1.31.1",
                            "torch_intra_threads": 2, "torch_inter_threads": 1, "ort_intra_threads": 2, "ort_inter_threads": 1,
                            "ort_execution_mode": "sequential", "workers": 1},
                "input_contract": {"formats": ["JPEG", "PNG", "WEBP"], "max_file_bytes": 10 * 1024 * 1024,
                                   "max_pixels": 12000000, "min_side": 16, "color": "EXIF-transposed RGB", "animated_images": "reject"},
                "preprocessing": {"classifier": {"processor_directory": "classifier", "backend": "torchvision", "size": 224},
                                  "detector": {"size": 640, "padding_value": 114, "interpolation": "OpenCV linear", "scale_divisor": 255.0},
                                  "segmenter": {"size": 640, "padding_value": 0, "interpolation": "PIL bilinear", "scale_divisor": 255.0,
                                                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                "thresholds": {key: policy[key] for key in ["classifier_fake_threshold", "detector_confidence", "detector_nms_iou",
                    "detector_max_detections", "segmenter_pixel_threshold", "minimum_edited_area_fraction"]},
                "detector_confidence_comparator": ">", "segmenter_pixel_comparator": ">=", "segmenter_postprocessing": "none",
                "mask_restoration": "Crop letterbox; PIL float bilinear to native dimensions; threshold after restoration",
                "decision_rule": policy["rule"], "policy_sha256": sha_file(release / "decision_policy.json"),
                "branch_execution": "Always execute classifier, detector and segmenter; no classifier/detector gating",
                "retention": "No user-image/prediction retention by the service; uploaded temporary files are closed after each request",
                "test_evaluated": False, "final_test_must_require": "evidence/service_validation_gate.json all_checks_passed=true",
                "scope": "Research prototype for human review; no proof of authenticity, no Saudi deployment-performance claim"}
    dump(release / "manifest.json", manifest)
    return manifest


def freeze_release(release):
    relatives = RUNTIME_FILES + ["classifier/config.json", "classifier/preprocessor_config.json", "decision_policy.json"]
    lock = {"manifest_sha256": sha_file(release / "manifest.json"),
            "source_and_config_sha256": {n: sha_file(release / n) for n in relatives}, "test_evaluated": False}
    dump(release / "release_lock.json", lock)
    return lock


def start_server(release, work):
    import httpx
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    env = os.environ.copy()
    for name in ["WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    log = (work / "server.log").open("w")
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, str(release / "serve.py"), "--release", str(release), "--port", str(port)],
                               cwd=release, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            while time.perf_counter() - started < 180:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup. See server.log in the review ZIP.")
                try:
                    response = client.get(url + "/health")
                    if response.status_code == 200 and response.json().get("status") == "ready":
                        return process, log, url, time.perf_counter() - started
                except (httpx.HTTPError, ValueError):
                    pass
                time.sleep(0.25)
        raise TimeoutError("Service did not become ready within180 seconds.")
    except Exception:
        process.terminate()
        process.wait(timeout=15)
        log.close()
        raise


def stop_server(process, log):
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    if log is not None:
        log.close()


def package_outputs(work, release):
    allowed = release_allowlist()
    review = work.parent / "Real-Abshar_Phase2_M8_Service_Review.zip"
    with zipfile.ZipFile(review, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(release.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.relative_to(release).as_posix() in allowed and path.suffix != ".onnx":
                z.write(path, path.relative_to(release).as_posix())
        for name in ["run_status.json", "cloud_backup_receipt.json", "error.txt", "server.log"]:
            path = work / name
            if path.is_file():
                z.write(path, name)
        demo = work / "private_demo"
        for name in ["validation_replay_annotated.avi", "validation_replay_preview.gif", "val_http_stage_latency.png", "val_actual_http_request_response.png"]:
            path = demo / name
            if path.is_file() and not path.is_symlink():
                z.write(path, "private_demo/" + path.name)
    bundle = None
    if all((release / "models" / (n + ".onnx")).is_file() for n in ["classifier", "detector", "segmenter"]):
        bundle = work.parent / "Real-Abshar_Phase2_M8_Service_Release.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in sorted(release.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.relative_to(release).as_posix() in allowed:
                    z.write(path, path.relative_to(release).as_posix())
            for name in ["run_status.json", "cloud_backup_receipt.json"]:
                if (work / name).is_file():
                    z.write(work / name, name)
    return {"review_zip": str(review), "service_release_zip": str(bundle) if bundle else None}


def run(input_root, work_root, assets_root, prepared_root=None):
    work, assets = Path(work_root).resolve(), Path(assets_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    release = work / "service_release"
    release.mkdir(parents=True, exist_ok=True)
    status_path = work / "run_status.json"
    previous = json.loads(status_path.read_text()) if status_path.is_file() else None
    run_id = previous["run_id"] if previous else uuid.uuid4().hex[:12]
    status = {"run_id": run_id, "stage": "M8_service", "status": "started", "test_evaluated": False}
    dump(status_path, status)
    if (work / "error.txt").is_file():
        (work / "error.txt").unlink()
    dump(work / "cloud_backup_receipt.json", {"stage": "M8_service", "run_id": run_id, "confirmed": False,
         "status": "not_attempted_this_execution", "test_evaluated": False})
    dump(release / "evidence" / "service_validation_gate.json", {"all_checks_passed": False, "status": "not_run", "test_evaluated": False})
    process, log = None, None
    try:
        runtime = runtime_environment()
        for name in RUNTIME_FILES:
            shutil.copyfile(assets / name, release / name)
        for name in PROOF_FILES:
            target = release / "provenance" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets / "proof" / name, target)
        dump(release / "evidence" / "runtime.json", runtime)
        cache = release / "provenance" / "val_export_predictions.jsonl"
        if sha_file(cache) != (release / "provenance" / "val_export_predictions.sha256").read_text().strip():
            raise ValueError("The reviewed validation export cache changed.")
        export = json.loads((release / "provenance" / "export_manifest.json").read_text())
        if not export["export_parity_passed"] or export["test_evaluation_performed"]:
            raise ValueError("Expected the reviewed validation-only passing export candidate.")
        policy = fit_policy(cache, release)
        print("Validation-selected review area fraction:", policy["minimum_edited_area_fraction"], flush=True)
        root = locate_prepared(input_root, work, prepared_root)
        rows, _ = read_preparation(root)
        validation = [r for r in rows if r["split"] == "val"]
        for index, row in enumerate(validation, 1):
            verify_row(root, row)
            if index % 25 == 0 or index == len(validation):
                print(f"Verified validation images/masks: {index}/{len(validation)}", flush=True)
        dump(release / "evidence" / "cohort_preflight.json", {"images_verified": 126, "split": "val", "source_families": 21,
             "source_cohort_same_as_M4_M5_export": True, "test_pixels_decoded": False, "test_inference_performed": False})
        restore_models(input_root, release, export)
        (release / "classifier").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets / "proof" / "classifier_config.json", release / "classifier" / "config.json")
        shutil.copyfile(assets / "proof" / "classifier_processor.json", release / "classifier" / "preprocessor_config.json")
        make_manifest(release, export, policy, run_id)
        freeze_release(release)
        dump(release / "evidence" / "service_validation_gate.json", {"all_checks_passed": False, "status": "not_run", "test_evaluated": False})
        from service_benchmark import benchmark
        from service_demo import draw_figures, video_replay
        import httpx
        process, log, url, ready_seconds = start_server(release, work)
        print("Actual HTTP service ready on loopback. Running checks and benchmark.", flush=True)
        reference = [json.loads(line) for line in cache.read_text().splitlines()]
        with httpx.Client(timeout=180, trust_env=False) as client:
            latency, validation_result, example = benchmark(client, url, process.pid, root, validation, reference, release, work / "private_demo")
            if not validation_result["all_checks_passed"]:
                raise RuntimeError("HTTP service validation parity failed; keep thresholds unchanged and return the review ZIP.")
            video_replay(client, url, root, validation, release, work / "private_demo")
            validation_result["video_replay_completed"] = True
            dump(release / "evidence" / "service_validation_gate.json", validation_result)
        stop_server(process, log)
        process, log = None, None
        dump(release / "evidence" / "server_lifecycle.json", {"actual_endpoint": url, "startup_to_ready_wall_seconds": ready_seconds,
             "server_stopped_after_notebook_benchmark": True, "public_hosting_claimed": False,
             "local_relaunch_instructions": "RUN_LOCAL.md", "test_evaluated": False})
        draw_figures(root, release, work / "private_demo", latency, example)
        status.update(status="service_validation_passed", cloud_backup_confirmed=False)
        print("Actual HTTP contract, validation parity and video replay completed.", flush=True)
        from service_backup import backup_release
        receipt = backup_release(release, work, run_id, validation_result, latency)
        status["cloud_backup_confirmed"] = receipt["confirmed"]
    except Exception as error:
        message = traceback.format_exc()
        for key in [os.environ.get("WANDB_API_KEY"), os.environ.get("HF_TOKEN")]:
            if key:
                message = message.replace(key, "[REDACTED]")
        (work / "error.txt").write_text(message)
        gate_path = release / "evidence" / "service_validation_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        gate.update(all_checks_passed=False, status="blocked", blocking_error_type=type(error).__name__, test_evaluated=False)
        dump(gate_path, gate)
        status.update(status="blocked", error_type=type(error).__name__)
        print(message, flush=True)
    finally:
        stop_server(process, log)
    dump(status_path, status)
    outputs = package_outputs(work, release)
    print("STATUS:", status["status"], flush=True)
    for key, value in outputs.items():
        if value:
            print(key + ":", value, flush=True)
    return {**status, **outputs}
