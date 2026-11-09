"""The portable, manifest-driven ONNX CPU inference path used by the HTTP service."""
from io import BytesIO
from pathlib import Path
import base64
import json
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from cohort_data import sha_file
from segmentation_data import restore_probability
from service_geometry import detector_input, segmenter_input, detector_boxes, fake_score, sigmoid
from service_decision import decide


class InputError(ValueError):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code, self.status_code = code, status_code


def decode_image(data, limits):
    if not data:
        raise InputError("empty_upload", "Upload a nonempty JPEG, PNG or WebP image.")
    if len(data) > int(limits["max_file_bytes"]):
        raise InputError("file_too_large", "The uploaded file exceeds the file-size limit.", 413)
    try:
        with Image.open(BytesIO(data)) as opened:
            file_format = opened.format
            if file_format not in limits["formats"]:
                raise InputError("unsupported_format", "Supported formats: JPEG, PNG and WebP.", 415)
            if getattr(opened, "n_frames", 1) != 1:
                raise InputError("animated_image", "Upload one still image; the demonstration processes video frame by frame.", 415)
            if opened.width * opened.height > int(limits["max_pixels"]):
                raise InputError("too_many_pixels", "The decoded image exceeds the pixel limit.", 413)
            if min(opened.size) < int(limits["min_side"]):
                raise InputError("image_too_small", "Image width and height must each meet the minimum size.", 422)
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image, file_format
    except InputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InputError("too_many_pixels", "The decoded image exceeds the pixel limit.", 413) from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InputError("invalid_image", "The file could not be decoded as a valid image.") from None


class Runtime:
    def __init__(self, release):
        import torch
        import onnxruntime as ort
        from transformers import AutoImageProcessor
        from importlib.metadata import version
        started = time.perf_counter()
        self.release = Path(release).resolve()
        self.manifest = json.loads((self.release / "manifest.json").read_text())
        lock = json.loads((self.release / "release_lock.json").read_text())
        if sha_file(self.release / "manifest.json") != lock["manifest_sha256"]:
            raise ValueError("The service manifest changed after the release was frozen.")
        for relative, digest in lock["source_and_config_sha256"].items():
            path = (self.release / relative).resolve()
            if not path.is_relative_to(self.release) or sha_file(path) != digest:
                raise ValueError("Release source/configuration changed: " + relative)
        runtime = self.manifest["runtime"]
        if version("transformers") != runtime["transformers"] or version("onnxruntime") != runtime["onnxruntime"]:
            raise RuntimeError("Install the pinned serving requirements; Transformers/ONNX Runtime version differs.")
        if not version("torch").startswith("2.11.") or not version("torchvision").startswith("0.26."):
            raise RuntimeError("Classifier preprocessing requires the recorded Torch2.11.x / torchvision0.26.x pair.")
        torch.set_num_threads(int(runtime["torch_intra_threads"]))
        try:
            torch.set_num_interop_threads(int(runtime["torch_inter_threads"]))
        except RuntimeError:
            pass
        cspec = self.manifest["preprocessing"]["classifier"]
        self.processor = AutoImageProcessor.from_pretrained(str(self.release / cspec["processor_directory"]),
                                                           backend=cspec["backend"], local_files_only=True)
        self.sessions, self.model_hashes = {}, {}
        for name, spec in self.manifest["models"].items():
            path = self.release / spec["file"]
            if sha_file(path) != spec["sha256"]:
                raise ValueError("ONNX graph integrity failed: " + name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(runtime["ort_intra_threads"])
            options.inter_op_num_threads = int(runtime["ort_inter_threads"])
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(str(path), sess_options=options, providers=[runtime["provider"]])
            inp, out = sess.get_inputs(), sess.get_outputs()
            if len(inp) != 1 or inp[0].shape != spec["input_shape"] or inp[0].type != "tensor(float)":
                raise ValueError("ONNX input contract differs: " + name)
            if len(out) != 1 or out[0].shape != spec["output_shape"] or out[0].type != "tensor(float)":
                raise ValueError("ONNX output contract differs: " + name)
            self.sessions[name], self.model_hashes[name] = sess, spec["sha256"]
        self.lock = threading.Lock()
        self.startup_seconds = None
        # A synthetic RGB smoke input checks the real deployed path without opening any dataset split.
        sample = BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(sample, format="PNG")
        result = self.predict(sample.getvalue())
        if result["image"]["width"] != 64 or result["image"]["height"] != 64 or result["mask"]["values"] != [0, 1]:
            raise RuntimeError("Startup smoke input returned an invalid contract.")
        self.startup_seconds = time.perf_counter() - started

    def info(self):
        return {"status": "ready", "release_id": self.manifest["release_id"], "provider": self.manifest["runtime"]["provider"],
                "model_sha256": self.model_hashes, "startup_self_test_passed": True,
                "startup_seconds": self.startup_seconds, "batch": 1, "precision": "FP32",
                "startup_seconds_scope": "model/config/session/processor checks plus smoke; excludes Python imports and server boot",
                "class_names": self.manifest["class_names"], "thresholds": self.manifest["thresholds"],
                "input_contract": self.manifest["input_contract"], "one_process_one_inference_at_a_time": True}

    def _forward(self, name, value):
        sess = self.sessions[name]
        result = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(value, dtype=np.float32)})[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite model output: " + name)
        return result

    def predict(self, data):
        waiting = time.perf_counter()
        with self.lock:
            started = time.perf_counter()
            times = {"queue_wait": (started - waiting) * 1000}
            tick = started
            image, file_format = decode_image(data, self.manifest["input_contract"])
            times["decode"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            cinput = self.processor(images=image, return_tensors="pt")["pixel_values"].float().cpu().numpy()
            cshape = self.manifest["models"]["classifier"]["input_shape"]
            if list(cinput.shape) != cshape:
                raise ValueError("Classifier processor input contract changed.")
            dspec = self.manifest["preprocessing"]["detector"]
            sspec = self.manifest["preprocessing"]["segmenter"]
            dinput, sinput = detector_input(image, dspec), segmenter_input(image, sspec)
            times["preprocess"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            coutput = self._forward("classifier", cinput)
            times["classifier_inference"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            doutput = self._forward("detector", dinput)
            times["detector_inference"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            soutput = self._forward("segmenter", sinput)
            times["segmenter_inference"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            settings = self.manifest["thresholds"]
            score = fake_score(coutput)
            boxes = detector_boxes(doutput, image.width, image.height, dspec, settings)
            probability = restore_probability(sigmoid(soutput[0, 0]), image.width, image.height, size=int(sspec["size"]))
            mask = (probability >= float(settings["segmenter_pixel_threshold"])).astype(np.uint8)
            foreground = int(mask.sum())
            area = foreground / mask.size
            decision = decide(score, boxes, area, settings)
            times["postprocess_and_decision"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            buffer = BytesIO()
            Image.fromarray(mask).save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            times["mask_encoding"] = (time.perf_counter() - tick) * 1000
            label_index = int(score >= float(settings["classifier_fake_threshold"]))
            output = {"schema_version": "1.0", "request_id": uuid.uuid4().hex[:12], "release_id": self.manifest["release_id"],
                      "image": {"width": image.width, "height": image.height, "decoded_format": file_format,
                                "color_order": "RGB", "exif_orientation_applied": True},
                      "classifier": {"label": self.manifest["class_names"]["classifier"][label_index], "fake_score": score,
                                     "fake_threshold": settings["classifier_fake_threshold"], "calibrated": False,
                                     "scope": "Phase1 global Real/Fake classifier; its label is not proof of no local edits."},
                      "detections": [{**box, "label": self.manifest["class_names"]["detector"][box["class"]]} for box in boxes],
                      "box_coordinates": "native oriented image pixels; xyxy; right/bottom coordinates are exclusive",
                      "mask": {"encoding": "base64_png", "png_base64": encoded, "values": [0, 1], "foreground_value": 1,
                               "width": image.width, "height": image.height, "pixel_threshold": settings["segmenter_pixel_threshold"],
                               "foreground_pixels": foreground, "edited_area_fraction": area, "postprocessing": "none"},
                      "decision": decision, "timing_ms": times}
            times["pipeline_total"] = (time.perf_counter() - started) * 1000
            return output
