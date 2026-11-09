"""Fresh-process validation and the single frozen test job. Test predictions are never overwritten."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import platform
import sys
import traceback

from final_assets import dump, sha, verify_frozen, EXPECTED_MANIFEST_SHA, EXPECTED_LOCK_SHA

CORE_FILES = ["final_worker.py", "final_records.py", "final_metrics.py", "detection_metrics.py"]


def environment():
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "transformers", "onnxruntime",
                "numpy", "pillow", "opencv-python", "ultralytics", "wandb", "matplotlib"]}
    pinned = {"transformers": "5.16.1", "onnxruntime": "1.31.1", "numpy": "2.1.3", "pillow": "13.0.0",
              "opencv-python": "4.14.0.94", "ultralytics": "8.3.221", "wandb": "0.28.1", "matplotlib": "3.10.0"}
    for name, required in pinned.items():
        if versions[name] != required:
            raise RuntimeError(f"Install the notebook requirements first: {name}=={required}; found{versions[name]}")
    if not versions["torch"].startswith("2.11.") or not versions["torchvision"].startswith("0.26."):
        raise RuntimeError("Use the successful Kaggle CPU Torch2.11.x/torchvision0.26.x image.")
    cpu = "unknown"
    if Path("/proc/cpuinfo").is_file():
        cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), "unknown")
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_model": cpu,
            "logical_cpus": os.cpu_count(), "packages": versions, "device": "CPUExecutionProvider", "precision": "FP32", "batch": 1}


def protocol(release, rows, code_root, env):
    from cohort_data import COHORT_UID_SHA256, SOURCE_MANIFEST_SHA256, SOURCE_ASSIGNMENT_SHA256
    import hashlib
    manifest = json.loads((release / "manifest.json").read_text())
    test_uids = [r["uid"] for r in rows if r["split"] == "test"]
    return {"service_manifest_sha256": EXPECTED_MANIFEST_SHA, "service_release_lock_sha256": EXPECTED_LOCK_SHA,
            "release_id": manifest["release_id"], "cohort_uid_sha256": COHORT_UID_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "source_assignment_sha256": SOURCE_ASSIGNMENT_SHA256,
            "test_uid_order_sha256": hashlib.sha256(json.dumps(test_uids, separators=(",", ":")).encode()).hexdigest(),
            "test_images": 114, "test_source_families": 19, "thresholds": manifest["thresholds"],
            "model_sha256": {n: s["sha256"] for n, s in manifest["models"].items()},
            "evaluator_core_sha256": {n: sha(code_root / n) for n in CORE_FILES},
            "numerical_runtime_versions": env["packages"],
            "AP": {"confidence_floor": 0.001, "nms_iou": 0.45, "max_det": 50,
                   "matching_iou_levels": [round(0.5 + i * 0.05, 2) for i in range(10)],
                   "primary": "640 letterbox coordinates before native clipping; GT gain=size/max(width,height), actual centered integer padding",
                   "secondary": "served native coordinates after the frozen Ultralytics-style scale/clip convention",
                   "implementation": "Ultralytics8.3.221 non_max_suppression, BaseValidator.match_predictions and ap_per_class",
                   "best_F1_threshold_returned_by_AP_function_used": False},
            "fixed_operating_matching_iou": 0.5, "detector_size_bins": "native union-box area times640/max(width,height) squared;32^2,96^2",
            "segmentation_size_bins": "GT foreground area after nearest resize into640 letterbox;32^2,96^2",
            "pixel_metric_space": "untouched native0/1 ground truth; thresholded frozen native-size prediction",
            "bootstrap": {"unit": "source family", "repetitions": 1000, "seed": 42},
            "inference_rule": "one saved forward per classifier/detector/segmenter per image; atomic records reused on resume",
            "test_tuning": False, "original_Phase1_test_reopened": False}


def ap_smoke(manifest):
    import numpy as np
    from final_metrics import ap_correct, official_low_floor, summarize_ap
    good = np.array([[10, 10, 20, 20, 0.9, 0]], np.float32)
    assert ap_correct(good, [10, 10, 20, 20]).all()
    bad = good.copy();bad[:, 5] = 1
    assert not ap_correct(bad, [10, 10, 20, 20]).any()
    assert not ap_correct(good, None).any()
    raw = np.zeros((1, 5, 8400), np.float32)
    raw[0, :, 0] = [120, 230, 40, 40, 0.9]
    row = {"width": "640", "height": "320", "box_xyxy_exclusive": "[100,50,140,90]"}
    arrays = official_low_floor(raw, row, manifest)
    assert arrays["ap_input_correct"].all() and arrays["ap_native_correct"].all()
    result = summarize_ap([{"metadata": {"kind": "edited"}, **arrays}])
    assert result["AP50"] >= 0.99 and result["AP50_95"] >= 0.99
    return {"perfect_box_all10IoUs": True, "wrong_class_and_empty_GT": True,
            "input_native_coordinate_fixture": True, "official_AP_perfect_fixture_ge0.99": True}


def validate(release, root, rows, work, protocol_sha, env, code_root):
    from service_runtime import Runtime
    from final_records import evaluate_image
    from final_metrics import aggregate, reference_native_ap
    from export_metrics import match_detections
    from final_ledger import restore_remote
    manifest = verify_frozen(release)
    restored = restore_remote(work, protocol_sha)
    if restored is not None:
        gate = json.loads((work / "validation_gate.json").read_text())
        if gate["protocol_sha256"] != protocol_sha or not gate["all_checks_passed"]:
            raise ValueError("The restored final-test ledger has no valid pre-test evaluation gate.")
        print("Pre-test validation gate restored; no new model selection or test inference.", flush=True)
        return gate
    if (work / "validation_gate.json").is_file():
        gate = json.loads((work / "validation_gate.json").read_text())
        if gate.get("protocol_sha256") == protocol_sha and gate.get("all_checks_passed"):
            print("Reusing the completed pre-test validation gate.", flush=True)
            return gate
    selected = [r for r in rows if r["split"] == "val"]
    runtime = Runtime(release)
    runtime.evaluation_environment = env
    smoke = ap_smoke(manifest)
    entries = []
    for index, row in enumerate(selected, 1):
        entry, _ = evaluate_image(runtime, root, row, protocol_sha, work / "validation_cache")
        entries.append(entry)
        if index % 10 == 0 or index == len(selected):
            print(f"Pre-test evaluator validation only: {index}/{len(selected)}", flush=True)
    results = aggregate(selected, entries, manifest)
    reference = {r["uid"]: r for r in [json.loads(l) for l in (release / "evidence" / "val_http_predictions.jsonl").read_text().splitlines()]}
    maximum_score_error, maximum_area_error, matches, denominators, same_decisions = 0.0, 0, 0, 0, True
    for entry in entries:
        m = entry["metadata"];r = m["result"];expected = reference[m["uid"]]
        maximum_score_error = max(maximum_score_error, abs(r["classifier"]["fake_score"] - expected["classifier_fake_score"]))
        maximum_area_error = max(maximum_area_error, abs(r["mask"]["foreground_pixels"] - expected["foreground_pixels"]))
        a, b = match_detections(expected["detections"], r["detections"])
        matches += a;denominators += b
        same_decisions &= r["decision"]["status"] == expected["decision"]["status"]
    ap_reference = reference_native_ap(selected, code_root / "m4_native_val_predictions.json")
    deployed_ap = results["detector_AP_secondary_served_native_coordinates"]
    old = json.loads((release / "evidence" / "service_validation_gate.json").read_text())
    checks = {"same126_validation_records": len(entries) == 126,
              "classifier_score_error_le1e_minus5": maximum_score_error <= 0.00001,
              "native_mask_area_error_le2pixels": maximum_area_error <= 2,
              "selected_detector_box_agreement_ge0.99": matches / denominators >= 0.99 if denominators else True,
              "same_image_review_decisions": same_decisions,
              "foreground_iou_error_le0.0005": abs(results["segmentation"]["foreground_iou"] - old["segmentation_metrics"]["foreground_iou"]) <= 0.0005,
              "same_fixed_operating_TP_FP_FN": [results["detector_operating_metrics"][k] for k in ["tp", "fp", "fn"]] == [55, 8, 50],
              "native_clipped_AP50_difference_le0.005": abs(deployed_ap["AP50"] - ap_reference["AP50"]) <= 0.005,
              "native_clipped_AP50_95_difference_le0.005": abs(deployed_ap["AP50_95"] - ap_reference["AP50_95"]) <= 0.005,
              "official_metric_smoke_checks": all(smoke.values())}
    gate = {"protocol_sha256": protocol_sha, "split": "val", "images": 126, "all_checks_passed": all(checks.values()),
            "checks": checks, "maximum_classifier_score_error": maximum_score_error, "maximum_mask_area_pixel_error": maximum_area_error,
            "native_coordinate_AP_reference_from_original_M4_prediction_cache": ap_reference,
            "new_validation_metrics": results, "official_AP_smoke_checks": smoke,
            "AP_note": "Earlier M4 training/selection AP used the official dataset validator. The final metric geometry is explicit in protocol.json. "
                       "Native clipped AP is compared against the original M4 native prediction cache using the same pinned official metric function.",
            "test_pixels_decoded": False, "test_inference_performed": False}
    dump(work / "validation_gate.json", gate)
    dump(work / "reports" / "validation_evaluator_metrics.json", results)
    if not gate["all_checks_passed"]:
        raise RuntimeError("Pre-test evaluator gate failed. The test remains closed; return the review ZIP without changing thresholds.")
    print("VALIDATION_EVALUATOR_GATE: PASSED. Reserved test is still unopened.", flush=True)
    return gate


def test_once(release, root, rows, work, protocol_sha, env):
    from final_ledger import restore_remote, assert_no_other_active_run, LedgerWriter
    from final_records import evaluate_image, read_record, record_path
    from final_metrics import aggregate, family_bootstrap
    from final_report import build_reports
    manifest = verify_frozen(release)
    gate = json.loads((work / "validation_gate.json").read_text())
    if gate.get("protocol_sha256") != protocol_sha or not gate.get("all_checks_passed"):
        raise RuntimeError("Run and pass validation Cell4 before opening the reserved test.")
    previous = restore_remote(work, protocol_sha)
    selected = [r for r in rows if r["split"] == "test"]
    if len(selected) != 114 or len({r["source_id"] for r in selected}) != 19:
        raise ValueError("Reserved test cohort changed.")
    expected_files = {r["uid"] + ".npz" for r in selected}
    if {p.name for p in (work / "test_cache").glob("*.npz")} - expected_files:
        raise ValueError("Unexpected sample in the reserved test cache.")
    cached = {r["uid"]: read_record(record_path(work / "test_cache", r["uid"]), r, protocol_sha)
              for r in selected if record_path(work / "test_cache", r["uid"]).is_file()}
    writer, success, newly_saved = None, False, 0
    try:
        if len(cached) < 114:
            assert_no_other_active_run(previous)
            writer = LedgerWriter(work, protocol_sha)
            # Confirm the frozen cohort/protocol ledger BEFORE decoding any new test pixels.
            writer.commit("test_started_or_resuming", len(cached))
            dump(work / "test_started.json", {"protocol_sha256": protocol_sha, "cached_images_before_resume": len(cached),
                  "model_and_threshold_changes_permitted": False})
            from service_runtime import Runtime
            runtime = Runtime(release)
            runtime.evaluation_environment = env
            families = {}
            for row in selected:
                families.setdefault(row["source_id"], []).append(row)
            for family in families.values():
                for row in family:
                    if row["uid"] not in cached:
                        entry, _ = evaluate_image(runtime, root, row, protocol_sha, work / "test_cache")
                        cached[row["uid"]] = entry
                        newly_saved += 1
                        print(f"Reserved frozen test saved: {len(cached)}/114", flush=True)
                writer.commit("predictions_complete" if len(cached) == 114 else "test_in_progress", len(cached))
        else:
            print("All114 saved test predictions restored. No test model forwards will be repeated.", flush=True)
        entries = [cached[r["uid"]] for r in selected]
        results = aggregate(selected, entries, manifest)
        results["protocol_sha256"] = protocol_sha
        results["source_family_bootstrap"] = family_bootstrap(entries)
        results["test_evaluation_completed"] = True
        results["saved_one_forward_per_branch_per_test_image"] = True
        dump(work / "reports" / "phase2_test_metrics.json", results)
        build_reports(root, selected, entries, results, work, release)
        success = True
        print("FINAL_TEST_COMPLETE:114 saved predictions. Settings remain frozen.", flush=True)
        return results
    finally:
        dump(work / "test_execution_summary.json", {"protocol_sha256": protocol_sha,
             "newly_saved_test_predictions_this_execution": newly_saved, "total_saved_test_predictions": len(cached),
             "inference_settings_changed": False, "metrics_and_gallery_completed": success})
        if writer is not None:
            writer.finish(success)


def main(args):
    work, release, root = Path(args.work).resolve(), Path(args.release).resolve(), Path(args.prepared).resolve()
    sys.path.insert(0, str(release))
    from cohort_data import read_preparation
    rows, _ = read_preparation(root)  # metadata only; no test pixels opened here
    env = environment()
    spec = protocol(release, rows, Path(__file__).resolve().parent, env)
    protocol_path = work / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != spec:
        raise RuntimeError("The frozen evaluation protocol changed. Do not reopen the test; return the review ZIP.")
    dump(protocol_path, spec)
    protocol_sha = sha(protocol_path)
    if not (work / "evaluation_environment.json").exists():
        dump(work / "evaluation_environment.json", env)
    if args.stage == "validation":
        validate(release, root, rows, work, protocol_sha, env, Path(__file__).resolve().parent)
    else:
        test_once(release, root, rows, work, protocol_sha, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--work", required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception:
        message = traceback.format_exc()
        for value in [os.environ.get("WANDB_API_KEY"), os.environ.get("HF_TOKEN")]:
            if value:
                message = message.replace(value, "[REDACTED]")
        Path(arguments.work).mkdir(parents=True, exist_ok=True)
        (Path(arguments.work) / "error.txt").write_text(message)
        dump(Path(arguments.work) / "worker_failure.json", {"stage": arguments.stage, "status": "blocked"})
        print(message, flush=True)
        sys.exit(1)
