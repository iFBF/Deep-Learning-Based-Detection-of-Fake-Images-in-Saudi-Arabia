"""Restore the three exact reviewed ONNX graphs from a private input or the completed own-project backup."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import zipfile

from cohort_data import sha_file

EXPORT_ARTIFACT = "f1k80-kaust/Real-Abshar-Phase2/real-abshar-m8-export-b90893924c38:v0"


def restore_models(input_root, release, verified_export_manifest):
    root, release = Path(input_root), Path(release)
    models = verified_export_manifest["models"]
    destination = release / "models"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / (name + ".onnx") for name in models}
    remaining = set(models)
    for name, path in paths.items():
        if path.is_file():
            if sha_file(path) != models[name]["onnx_sha256"]:
                raise ValueError("Cached ONNX graph changed: " + name)
            remaining.remove(name)
    for path in sorted(root.rglob("*.onnx")):
        if not remaining:
            break
        digest = sha_file(path)
        for name in list(remaining):
            if digest == models[name]["onnx_sha256"]:
                shutil.copyfile(path, paths[name])
                remaining.remove(name)
    if remaining:
        archives = sorted(p for p in root.rglob("*.zip") if "M8_ONNX_Release" in p.name)
        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                for name in list(remaining):
                    entries = [n for n in z.namelist() if n == "models/" + name + ".onnx" or n.endswith("/models/" + name + ".onnx")]
                    for entry in entries:
                        digest = hashlib.sha256()
                        with z.open(entry) as stream:
                            while block := stream.read(1024 * 1024):
                                digest.update(block)
                        if digest.hexdigest() == models[name]["onnx_sha256"]:
                            with z.open(entry) as source, paths[name].open("wb") as output:
                                shutil.copyfileobj(source, output)
                            remaining.remove(name)
                            break
    if remaining:
        import wandb
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Attach your private M8 ONNX Release ZIP or enable WANDB_API_KEY to restore its confirmed backup.")
        artifact = wandb.Api().artifact(EXPORT_ARTIFACT)
        for name in sorted(remaining):
            print("Restoring reviewed ONNX graph:", name, flush=True)
            downloaded = Path(artifact.get_entry("models/" + name + ".onnx").download(root=str(release.parent / "artifact_downloads")))
            if sha_file(downloaded) != models[name]["onnx_sha256"]:
                raise ValueError("Restored ONNX hash mismatch: " + name)
            shutil.copyfile(downloaded, paths[name])
    for name, path in paths.items():
        if sha_file(path) != models[name]["onnx_sha256"] or path.stat().st_size != models[name]["bytes"]:
            raise ValueError("Model integrity mismatch: " + name)
        print("Verified selected ONNX graph:", name, flush=True)
    return paths
