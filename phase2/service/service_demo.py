"""Private validation-derived request figure and a clearly labelled functional video replay."""
from io import BytesIO
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from cohort_data import dump, safe_path
from segmentation_data import geometry
from service_benchmark import mask_from_response, post_image


def overlay(image, result):
    import cv2
    pixels = np.asarray(image.convert("RGB")).copy()
    mask = mask_from_response(result).astype(bool)
    pixels[mask] = (0.6 * pixels[mask] + 0.4 * np.array([235, 55, 55])).astype(np.uint8)
    for prediction in result["detections"]:
        x0, y0, x1, y1 = [int(round(v)) for v in prediction["box"]]
        cv2.rectangle(pixels, (x0, y0), (x1, y1), (20, 190, 110), 2)
    return pixels


def video_replay(client, url, root, validation, release, private_demo):
    import cv2
    destination = Path(private_demo)
    destination.mkdir(parents=True, exist_ok=True)
    selected = validation[:12]
    source = destination / "validation_replay_input.avi"
    output = destination / "validation_replay_annotated.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 3.0, (640, 640))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG VideoWriter is unavailable; send the review ZIP before substituting another video path.")
    for row in selected:
        with Image.open(safe_path(root, row["export_image"])) as opened:
            image = opened.convert("RGB")
        g = geometry(image.width, image.height, 640)
        frame = np.zeros((640, 640, 3), np.uint8)
        frame[g["top"]:g["top"] + g["height"], g["left"]:g["left"] + g["width"]] = np.asarray(
            image.resize((g["width"], g["height"]), Image.Resampling.BILINEAR))
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    writer.release()
    capture = cv2.VideoCapture(str(source))
    rendered = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), 3.0, (640, 640))
    if not capture.isOpened() or not rendered.isOpened():
        raise RuntimeError("Replay video could not be opened for decode/render.")
    frames, detector_ms, previews = 0, [], []
    tick = time.perf_counter()
    try:
        while True:
            valid, bgr = capture.read()
            if not valid:
                break
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            rgb.save(buffer, format="PNG")
            result, _ = post_image(client, url, buffer.getvalue())
            decorated = overlay(rgb, result)
            label = "VALIDATION REPLAY / " + result["decision"]["status"]
            cv2.rectangle(decorated, (0, 0), (640, 37), (15, 30, 25), -1)
            cv2.putText(decorated, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 250, 245), 1, cv2.LINE_AA)
            cv2.rectangle(decorated, (0, 615), (640, 640), (15, 30, 25), -1)
            cv2.putText(decorated, "CPU FP32 / C224 D,S640 / C.5 D.35 NMS.45 S.5 / review area.005",
                        (10, 632), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (245, 250, 245), 1, cv2.LINE_AA)
            rendered.write(np.ascontiguousarray(decorated[:, :, ::-1]))
            previews.append(Image.fromarray(decorated).resize((480, 480)))
            detector_ms.append(float(result["timing_ms"]["detector_inference"]))
            frames += 1
    finally:
        capture.release()
        rendered.release()
    elapsed = time.perf_counter() - tick
    if frames != len(selected):
        raise RuntimeError("Video replay frame count differs from its generated source.")
    previews[0].save(destination / "validation_replay_preview.gif", save_all=True, append_images=previews[1:],
                     duration=333, loop=0, optimize=True)
    summary = {"scope": "Functional slideshow replay of12 validation images, not a natural-motion video accuracy benchmark",
               "source_uids": [r["uid"] for r in selected], "split": "val-derived", "frames": frames,
               "source_codec": "MJPG AVI", "source_resolution": [640, 640], "source_nominal_playback_fps": 3.0,
               "model_inputs": {"classifier": [224, 224], "detector_and_segmenter": [640, 640]},
               "thresholds": json.loads((Path(release) / "manifest.json").read_text())["thresholds"],
               "actual_decode_HTTP_all_models_overlay_encode_seconds": elapsed,
               "actual_complete_replay_processing_fps": frames / elapsed,
               "detector_forward_only_fps": frames * 1000 / sum(detector_ms),
               "detector_forward_only_fps_scope": "ONNX forward timing inside each HTTP response; excludes other models/pre/post/transport",
               "codec_and_resize_change_pixels": True, "accuracy_evaluation_performed_on_replay": False,
               "threshold_tuning_performed_on_replay": False, "test_evaluated": False}
    dump(Path(release) / "evidence" / "validation_video_replay.json", summary)
    return summary


def draw_figures(root, release, private_demo, latency, example):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(private_demo)
    thresholds = json.loads((Path(release) / "manifest.json").read_text())["thresholds"]
    condition = ("Validation126 / CPU FP32 batch1 / C224; D,S640\n"
                 f"C tau{thresholds['classifier_fake_threshold']}; D conf{thresholds['detector_confidence']} NMS{thresholds['detector_nms_iou']}; "
                 f"S pixel{thresholds['segmenter_pixel_threshold']}; review area{thresholds['minimum_edited_area_fraction']}")
    table = latency["stages"]
    names = ["preprocess_ms", "classifier_inference_ms", "detector_inference_ms", "segmenter_inference_ms",
             "postprocess_and_decision_ms", "mask_encoding_ms", "pipeline_total_ms", "http_client_ms"]
    indexed = {r["stage"]: r for r in table}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(names))
    ax.barh(positions - 0.18, [indexed[n]["p50_ms"] for n in names], height=0.35, color="#22775d", label="p50")
    ax.barh(positions + 0.18, [indexed[n]["p95_ms"] for n in names], height=0.35, color="#d1a542", label="p95")
    ax.set_yticks(positions, [n.removesuffix("_ms").replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Measured milliseconds per image")
    ax.set_title(condition, fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "val_http_stage_latency.png", dpi=160)
    plt.close(fig)

    with Image.open(safe_path(root, example["image_relative_path"])) as opened:
        image = opened.convert("RGB")
    result = example["response"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(overlay(image, result))
    axes[0].set_title(example["uid"] + " / actual HTTP output", fontsize=10)
    axes[0].axis("off")
    compact = {"HTTP": "200 OK", "release_id": result["release_id"], "global_label": result["classifier"]["label"],
               "global_fake_score": round(result["classifier"]["fake_score"], 8), "boxes": len(result["detections"]),
               "edited_area_fraction": round(result["mask"]["edited_area_fraction"], 6),
               "review": result["decision"]["status"], "native_mask": [image.height, image.width]}
    axes[1].text(0.02, 0.95, "POST " + example["url"] + "\n\n" + json.dumps(compact, indent=2),
                 va="top", family="monospace", fontsize=9, transform=axes[1].transAxes, wrap=True)
    axes[1].axis("off")
    fig.suptitle(condition.replace("Validation126", "Validation example"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output / "val_actual_http_request_response.png", dpi=160)
    plt.close(fig)
