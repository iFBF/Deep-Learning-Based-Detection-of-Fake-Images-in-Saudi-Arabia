"""Locate only the checksum-pinned classifier; obtain selected M4/M5 weights from confirmed own-project backups."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import zipfile

from cohort_data import dump, sha_file

MODEL_SPECS = {
    "classifier": {"sha256": "a2fd5d5ec37c1bcf2a9fc6243115306aadc987fa3404d6945a5fd97961d5504e",
                   "selected": "Phase 1 full Swin, minimum validation cross-entropy"},
    "detector": {"sha256": "66b440225fa64a30ef035706ddde8d01489476a1c3331f2d289655b5be9afb40",
                 "selected": "A_standard_augmentation",
                 "artifact": "f1k80-kaust/Real-Abshar-Phase2/real-abshar-m4-backup-8c9cbe426ba1:v18",
                 "entry": "runs/A_standard_augmentation/weights/best.pt"},
    "segmenter": {"sha256": "301cae566f84ec9ffa7753373320e90bb9a0af70d64e85a8d3d89b2a92481b2c",
                  "selected": "B_UNet_ResNet18",
                  "artifact": "f1k80-kaust/Real-Abshar-Phase2/real-abshar-m5-backup-6ea92ac6178c:v15",
                  "entry": "runs/B_UNet_ResNet18/weights/best.pt"},
}


def acquire(input_root, work, classifier_path=None):
    root, work = Path(input_root), Path(work)
    destination = work / "native_checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / (name + ".pt") for name in MODEL_SPECS}
    remaining = {name for name, path in paths.items() if not path.exists()}
    for name, path in paths.items():
        if path.exists() and sha_file(path) != MODEL_SPECS[name]["sha256"]:
            raise ValueError("Existing native checkpoint has the wrong hash: " + name)
    if classifier_path:
        path = Path(classifier_path)
        if not path.is_file() or sha_file(path) != MODEL_SPECS["classifier"]["sha256"]:
            raise ValueError("CLASSIFIER_PATH is not the corrected Phase 1 full/best.pt checkpoint.")
        shutil.copyfile(path, paths["classifier"])
        remaining.discard("classifier")
    candidates = sorted(root.rglob("*.pt"), key=lambda p: ("full" not in p.parts, str(p)))
    for candidate in candidates:
        if not remaining:
            break
        digest = sha_file(candidate)
        for name in list(remaining):
            if digest == MODEL_SPECS[name]["sha256"]:
                shutil.copyfile(candidate, paths[name])
                remaining.remove(name)
    if "classifier" in remaining:
        # Some Kaggle inputs keep the model ZIP intact. Read members without trusting their filesystem paths.
        archives = sorted(p for p in root.rglob("*.zip") if "phase1" in p.name.lower() and "model" in p.name.lower())
        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                members = sorted((i for i in z.infolist() if not i.is_dir() and i.filename.endswith(".pt")),
                                 key=lambda i: ("full" not in PurePosixPath(i.filename).parts, i.filename))
                for member in members:
                    digest = hashlib.sha256()
                    with z.open(member) as stream:
                        while block := stream.read(1024 * 1024):
                            digest.update(block)
                    if digest.hexdigest() == MODEL_SPECS["classifier"]["sha256"]:
                        with z.open(member) as source, paths["classifier"].open("wb") as output:
                            shutil.copyfileobj(source, output)
                        remaining.remove("classifier")
                        break
            if "classifier" not in remaining:
                break
    if "classifier" in remaining:
        raise FileNotFoundError("Attach your retained Phase1_Corrected_Models.zip as a PRIVATE Kaggle input, "
                                "or set CLASSIFIER_PATH to the corrected full/best.pt. No retraining is needed; "
                                "the required checkpoint SHA-256 is " + MODEL_SPECS["classifier"]["sha256"])
    if remaining:
        import wandb
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Enable the WANDB_API_KEY secret to download your selected M4/M5 checkpoints.")
        api = wandb.Api()
        for name in sorted(remaining):
            spec = MODEL_SPECS[name]
            print("Downloading selected checkpoint from confirmed W&B backup:", name, flush=True)
            artifact = api.artifact(spec["artifact"])
            path = Path(artifact.get_entry(spec["entry"]).download(root=str(work / "artifact_downloads" / name)))
            if sha_file(path) != spec["sha256"]:
                raise ValueError("Downloaded checkpoint hash mismatch: " + name)
            shutil.copyfile(path, paths[name])
    for name, path in paths.items():
        if sha_file(path) != MODEL_SPECS[name]["sha256"]:
            raise ValueError("Checkpoint integrity mismatch: " + name)
        print("Verified selected checkpoint:", name, MODEL_SPECS[name]["sha256"], flush=True)
    dump(work / "native_checkpoints.json", {name: {"file": str(path), **MODEL_SPECS[name]} for name, path in paths.items()})
    return paths
