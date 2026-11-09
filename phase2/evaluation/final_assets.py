"""Restore the reviewed frozen service without editing its manifest, code or thresholds."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import zipfile

SERVICE_ARTIFACT = "f1k80-kaust/Real-Abshar-Phase2/real-abshar-m8-service-fbaa1c95818d:v0"
EXPECTED_MANIFEST_SHA = "b06365983742f70d42f7ae6ff02eb805a7776f6ea02eabc816b5b4386f83133f"
EXPECTED_LOCK_SHA = "2e96338b8b2b5a9fc7f63968250c1907205e25e60377bbe0eae35d2b138f34ad"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_target(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError("Unsafe cache-relative path.")
    candidate = (Path(root) / relative).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError("Cache path escapes its destination.")
    return candidate


def verify_frozen(release, graphs=True):
    release = Path(release)
    if sha(release / "manifest.json") != EXPECTED_MANIFEST_SHA or sha(release / "release_lock.json") != EXPECTED_LOCK_SHA:
        raise ValueError("Use the exact reviewed M8 service release; its manifest/lock changed.")
    manifest = json.loads((release / "manifest.json").read_text())
    lock = json.loads((release / "release_lock.json").read_text())
    for relative, digest in lock["source_and_config_sha256"].items():
        if sha(safe_target(release, relative)) != digest:
            raise ValueError("Frozen service source/configuration changed: " + relative)
    gate = json.loads((release / "evidence" / "service_validation_gate.json").read_text())
    if not gate["all_checks_passed"] or not gate["video_replay_completed"] or gate["test_evaluated"]:
        raise ValueError("The reviewed M8 service gate is not the passing validation-only release.")
    if graphs:
        for name, spec in manifest["models"].items():
            path = safe_target(release, spec["file"])
            if sha(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
                raise ValueError("Frozen ONNX graph integrity failed: " + name)
    return manifest


def restore_service(input_root, assets, work):
    root, assets, work = Path(input_root), Path(assets), Path(work)
    release = work / "frozen_service"
    release.mkdir(parents=True, exist_ok=True)
    for source in sorted((assets / "frozen_service").rglob("*")):
        if source.is_file():
            target = release / source.relative_to(assets / "frozen_service")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != sha(source):
                raise ValueError("An existing frozen service metadata/source file changed; do not overwrite it.")
            if not target.exists():
                shutil.copyfile(source, target)
    manifest = verify_frozen(release, graphs=False)
    remaining = set(manifest["models"])
    for name, spec in manifest["models"].items():
        path = release / spec["file"]
        if path.is_file():
            if sha(path) != spec["sha256"]:
                raise ValueError("Cached model changed: " + name)
            remaining.remove(name)
    for source in sorted(root.rglob("*.onnx")):
        if not remaining:
            break
        digest = sha(source)
        for name in list(remaining):
            spec = manifest["models"][name]
            if digest == spec["sha256"]:
                target = release / spec["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                remaining.remove(name)
    for archive in sorted(root.rglob("*.zip")):
        if not remaining:
            break
        if not any(term in archive.name for term in ["M8_Service_Release", "M8_ONNX_Release"]):
            continue
        with zipfile.ZipFile(archive) as z:
            for name in list(remaining):
                spec = manifest["models"][name]
                for entry in [n for n in z.namelist() if n == spec["file"] or n.endswith("/" + spec["file"])]:
                    digest = hashlib.sha256()
                    with z.open(entry) as stream:
                        while block := stream.read(1024 * 1024):
                            digest.update(block)
                    if digest.hexdigest() == spec["sha256"]:
                        target = release / spec["file"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(entry) as stream, target.open("wb") as output:
                            shutil.copyfileobj(stream, output)
                        remaining.remove(name)
                        break
    if remaining:
        import wandb
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Attach the private Service Release ZIP or enable WANDB_API_KEY for its verified backup.")
        artifact = wandb.Api().artifact(SERVICE_ARTIFACT)
        for name in sorted(remaining):
            spec = manifest["models"][name]
            print("Restoring frozen graph:", name, flush=True)
            source = Path(artifact.get_entry(spec["file"]).download(root=str(work / "model_downloads")))
            if sha(source) != spec["sha256"]:
                raise ValueError("Restored graph hash mismatch: " + name)
            target = release / spec["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    verify_frozen(release)
    return release


def restore_attached_cache(input_root, work):
    # Kaggle may expose a private uploaded ZIP as an extracted directory.
    for marker in sorted(Path(input_root).rglob("protocol.json")):
        folder = marker.parent
        if not (folder / "test_cache").is_dir():
            continue
        try:
            spec = json.loads(marker.read_text())
        except (ValueError, OSError):
            continue
        if spec.get("service_manifest_sha256") != EXPECTED_MANIFEST_SHA:
            continue
        sources = [folder / n for n in ["protocol.json", "validation_gate.json", "test_ledger.json", "evaluation_environment.json"]]
        sources += sorted((folder / "test_cache").glob("*.npz"))
        for source in sources:
            if not source.is_file():
                continue
            target = safe_target(work, source.relative_to(folder).as_posix())
            if target.exists() and sha(target) != sha(source):
                raise ValueError("Extracted evaluation cache conflicts with an existing record: " + source.name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for archive in sorted(Path(input_root).rglob("*Final_Test_Cache*.zip")):
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                name = item.filename
                allowed = name in {"protocol.json", "validation_gate.json", "test_ledger.json", "evaluation_environment.json"} or name.startswith("test_cache/") and name.endswith(".npz")
                if item.is_dir() or not allowed:
                    continue
                target = safe_target(work, name)
                data = z.read(item)
                if target.exists():
                    if sha(target) != hashlib.sha256(data).hexdigest():
                        raise ValueError("Attached evaluation cache conflicts with an existing record: " + name)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
