"""Actual loopback HTTP benchmark and validation check for the deployed one-process path."""
from io import BytesIO
from pathlib import Path
import base64
import json
import threading
import time

import numpy as np
from PIL import Image

from cohort_data import dump, safe_path, write_csv
from export_metrics import match_detections
from segmentation_metrics import counts, summarize
from service_decision import binary_metrics, decide


def mask_from_response(result):
    with Image.open(BytesIO(base64.b64decode(result["mask"]["png_base64"], validate=True))) as opened:
        mask = np.asarray(opened, np.uint8)
    if mask.shape != (result["image"]["height"], result["image"]["width"]) or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("HTTP response mask is not an aligned native-size0/1 PNG.")
    if int(mask.sum()) != result["mask"]["foreground_pixels"]:
        raise ValueError("HTTP mask foreground count disagrees with its metadata.")
    return mask


def post_image(client, url, payload):
    response = client.post(url + "/predict", files={"file": ("input.png", payload, "image/png")})
    if response.status_code != 200:
        raise RuntimeError("HTTP prediction failed: " + str(response.status_code) + " " + response.text[:300])
    return response.json(), response


def contract_checks(client, url, valid_payload):
    checks = []
    def record(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        if not passed:
            raise RuntimeError("HTTP contract check failed: " + name)
    health = client.get(url + "/health")
    record("health_startup_models_and_smoke", health.status_code == 200 and health.json()["startup_self_test_passed"])
    record("browser_form_route", client.get(url + "/").status_code == 200)
    record("missing_file_422", client.post(url + "/predict").status_code == 422)
    for name, payload, expected in [("empty_upload_400", b"", 400), ("corrupt_image_400", b"not-an-image", 400),
                                    ("oversized_file_413", b"0" * (10 * 1024 * 1024 + 1), 413)]:
        response = client.post(url + "/predict", files={"file": ("anything.png", payload, "image/png")})
        record(name, response.status_code == expected)
    for name, fmt, size, expected in [("unsupported_BMP_415", "BMP", (32, 32), 415),
                                      ("too_small_422", "PNG", (1, 16), 422)]:
        buffer = BytesIO()
        Image.new("RGB", size).save(buffer, format=fmt)
        record(name, client.post(url + "/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")}).status_code == expected)
    buffer = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (16, 32), (80, 120, 40)).save(buffer, format="JPEG", exif=exif)
    rotated, _ = post_image(client, url, buffer.getvalue())
    record("JPEG_EXIF_orientation_and_native_mask", (rotated["image"]["width"], rotated["image"]["height"]) == (32, 16)
           and mask_from_response(rotated).shape == (16, 32))
    # The decoded format, rather than a filename or declared MIME type, is authoritative.
    response = client.post(url + "/predict", files={"file": ("untrusted-name.txt", valid_payload, "text/plain")})
    record("format_checked_from_decoded_content", response.status_code == 200)
    buffer = BytesIO()
    Image.new("RGB", (32, 32), (60, 100, 140)).save(buffer, format="WEBP", lossless=True)
    webp, _ = post_image(client, url, buffer.getvalue())
    record("WebP_success", webp["image"]["decoded_format"] == "WEBP")
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (32, 32), "blue")], duration=100, loop=0)
    record("animated_PNG_415", client.post(url + "/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")}).status_code == 415)
    buffer = BytesIO()
    Image.new("RGB", (4000, 3001), "black").save(buffer, format="PNG")
    record("pixel_limit_413", client.post(url + "/predict", files={"file": ("image.png", buffer.getvalue(), "image/png")}).status_code == 413)
    return checks, health.json()


class MemorySampler:
    def __init__(self, pid):
        import psutil
        self.process = psutil.Process(pid)
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.values.append(self.process.memory_info().rss)
            self.stop.wait(0.05)

    def __enter__(self):
        self.values.append(self.process.memory_info().rss)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.values.append(self.process.memory_info().rss)


def summarize_latency(records, health, memory):
    fields = ["http_client_ms", "http_handler_ms", "decode_ms", "preprocess_ms", "classifier_inference_ms",
              "detector_inference_ms", "segmenter_inference_ms", "postprocess_and_decision_ms", "mask_encoding_ms",
              "pipeline_total_ms", "queue_wait_ms"]
    stages = []
    for name in fields:
        values = np.asarray([r[name] for r in records], np.float64)
        stages.append({"stage": name, "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                       "p99_ms": float(np.percentile(values, 99)), "mean_ms": float(values.mean())})
    return {"backend": "ONNX Runtime / CPUExecutionProvider through FastAPI HTTP", "precision": "FP32", "batch": 1,
            "concurrency": 1, "server_processes": 1, "timed_requests": len(records), "warmup_requests": 10,
            "additional_startup_smoke_requests": 1, "split": "val", "source_families": 21,
            "classifier_input": [224, 224], "region_model_input": [640, 640], "native_uploads": "reviewed lossless PNGs at native dimensions",
            "stages": stages, "serial_active_request_throughput_images_per_second": len(records) * 1000 / sum(r["http_client_ms"] for r in records),
            "memory": {"method": "psutil RSS of the separate serving process sampled every0.05s during timed HTTP calls",
                       "loaded_process_rss_before_timing_MiB": memory.values[0] / 2**20,
                       "peak_sampled_process_rss_MiB": max(memory.values) / 2**20, "samples": len(memory.values),
                       "includes": "runtime, loaded model sessions, preprocessing and inference allocations; excludes benchmark client process",
                       "limitation": "Sampled RSS can miss short peaks; this is not GPU-memory measurement."},
            "client_timer_includes": "multipart upload, loopback HTTP, server processing, response body transfer and client JSON parsing",
            "client_timer_excludes": "source-file disk reads, validation GT/mask decoding, plotting, UI rendering and model startup",
            "pipeline_timer_includes": "decode, all preprocessing, three sequential model forwards, postprocessing, decision and PNG/base64 mask encoding",
            "pipeline_timer_excludes": "request upload/parsing, queue wait, final JSON serialization and network response transfer",
            "throughput_definition": "N divided by summed timed client request durations, concurrency1; no network/load scalability claim",
            "percentile_method": "numpy.percentile default linear method;126 correlated validation images; exploratory tail estimates, not a load-test guarantee",
            "health": health, "test_evaluated": False}


def benchmark(client, url, server_pid, root, validation, reference_records, release, private_demo):
    release, private_demo = Path(release), Path(private_demo)
    private_demo.mkdir(parents=True, exist_ok=True)
    reference = {r["uid"]: r for r in reference_records}
    cases = [(r, safe_path(root, r["export_image"]).read_bytes()) for r in validation]
    checks, health = contract_checks(client, url, cases[0][1])
    for index in range(10):
        post_image(client, url, cases[index % len(cases)][1])
    order = np.random.default_rng(42).permutation(len(cases))
    records, predictions, metrics = [], [], []
    first_response = None
    settings = json.loads((release / "manifest.json").read_text())["thresholds"]
    with MemorySampler(server_pid) as memory:
        for number, index in enumerate(order, 1):
            row, payload = cases[int(index)]
            tick = time.perf_counter()
            result, response = post_image(client, url, payload)
            duration_ms = (time.perf_counter() - tick) * 1000
            mask = mask_from_response(result)
            if mask.shape != (int(row["height"]), int(row["width"])):
                raise ValueError("HTTP output dimensions differ from the reviewed input: " + row["uid"])
            with Image.open(safe_path(root, row["export_mask"])) as opened:
                truth = np.asarray(opened, np.uint8)
            native_counts = counts(truth, mask.astype(np.float32), 0.5)
            metrics.append({"uid": row["uid"], "kind": row["kind"], **native_counts})
            expected = reference[row["uid"]]
            matched, denominator = match_detections(expected["detector_onnx"], result["detections"])
            expected_decision = decide(expected["classifier_onnx_fake_score"], expected["detector_onnx"],
                                       expected["segmenter_onnx_area_fraction"], settings)
            record = {"uid": row["uid"], "source_id": row["source_id"], "kind": row["kind"], "split": "val",
                      "http_client_ms": duration_ms, "http_handler_ms": float(response.headers["X-Handler-Ms"]),
                      **{key + "_ms": float(value) for key, value in result["timing_ms"].items()},
                      "classifier_fake_score_abs_error": abs(result["classifier"]["fake_score"] - expected["classifier_onnx_fake_score"]),
                      "detector_matched": matched, "detector_match_denominator": denominator,
                      "mask_foreground_pixels_abs_error": abs(result["mask"]["foreground_pixels"] - expected["segmenter_onnx_foreground_pixels"]),
                      "decision_agrees": result["decision"]["status"] == expected_decision["status"],
                      "http_status": response.status_code}
            records.append(record)
            compact = {"uid": row["uid"], "source_id": row["source_id"], "split": "val", "kind": row["kind"],
                       "classifier_fake_score": result["classifier"]["fake_score"], "detections": result["detections"],
                       "foreground_pixels": result["mask"]["foreground_pixels"], "edited_area_fraction": result["mask"]["edited_area_fraction"],
                       "decision": result["decision"], "segmentation_counts": native_counts}
            predictions.append(compact)
            if first_response is None:
                first_response = {"uid": row["uid"], "response": result, "url": url + "/predict", "http_client_ms": duration_ms,
                                  "image_relative_path": row["export_image"]}
            if number % 10 == 0 or number == len(cases):
                print(f"Timed and verified real HTTP requests: {number}/{len(cases)}", flush=True)
    latency = summarize_latency(records, health, memory)
    segmentation = summarize(metrics)
    expected_seg = summarize([{"kind": r["kind"], **r["segmenter_onnx_counts"]} for r in reference_records])
    denominator = sum(r["detector_match_denominator"] for r in records)
    agreement = sum(r["detector_matched"] for r in records) / denominator if denominator else 1.0
    parity_checks = {"all126_http_requests_200": len(records) == 126 and all(r["http_status"] == 200 for r in records),
                     "classifier_score_max_error_le_1e_minus5": max(r["classifier_fake_score_abs_error"] for r in records) <= 0.00001,
                     "detector_box_agreement_ge_0.99": agreement >= 0.99,
                     "native_mask_area_max_difference_le_2pixels": max(r["mask_foreground_pixels_abs_error"] for r in records) <= 2,
                     "foreground_iou_difference_le_0.0005": abs(segmentation["foreground_iou"] - expected_seg["foreground_iou"]) <= 0.0005,
                     "same_review_decisions_as_selected_cache": all(r["decision_agrees"] for r in records),
                     "HTTP_contract_checks_passed": all(c["passed"] for c in checks)}
    evaluation = {"split": "val", "images": 126, "source_families": 21, "input_resolutions": {"classifier": [224, 224], "region_models": [640, 640]},
                  "thresholds": settings, "checks": parity_checks, "all_checks_passed": all(parity_checks.values()),
                  "detector_detection_agreement": agreement, "maximum_classifier_score_abs_error": max(r["classifier_fake_score_abs_error"] for r in records),
                  "maximum_foreground_pixel_count_difference": max(r["mask_foreground_pixels_abs_error"] for r in records),
                  "segmentation_metrics": segmentation,
                  "referral_metrics": binary_metrics([r["kind"] == "edited" for r in predictions],
                                                       [r["decision"]["review_recommended"] for r in predictions]),
                  "test_evaluated": False}
    evidence = release / "evidence"
    dump(evidence / "http_contract_checks.json", checks)
    dump(evidence / "http_benchmark.json", latency)
    dump(evidence / "service_validation_gate.json", evaluation)
    write_csv(evidence / "val_http_latency_and_parity.csv", records)
    write_csv(evidence / "latency_stage_percentiles.csv", latency["stages"])
    (evidence / "val_http_predictions.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in predictions))
    compact_example = {**first_response, "response": {**first_response["response"],
                       "mask": {**first_response["response"]["mask"], "png_base64": "[omitted: native0/1 PNG returned in actual HTTP response]"}}}
    dump(evidence / "example_http_request_response.json", compact_example)
    return latency, evaluation, first_response
