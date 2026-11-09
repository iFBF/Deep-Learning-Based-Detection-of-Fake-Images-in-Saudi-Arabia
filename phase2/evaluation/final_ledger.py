"""A cohort-wide W&B evaluation ledger blocks silent new tests and preserves resumable records."""
from pathlib import Path
from datetime import datetime, timezone
import os

from final_assets import dump, sha, safe_target

ENTITY, PROJECT = "f1k80-kaust", "Real-Abshar-Phase2"
ARTIFACT_NAME = "real-abshar-phase2-final-c24eb30ac226"
ARTIFACT_PATH = ENTITY + "/" + PROJECT + "/" + ARTIFACT_NAME


def find_remote():
    import wandb
    try:
        return wandb.Api().artifact(ARTIFACT_PATH + ":latest", type="evaluation")
    except wandb.errors.CommError as error:
        message = str(error).lower()
        # Only the explicit SDK missing-artifact response counts as absent. Network/auth/project
        # failures must stop the job so a temporary read failure cannot reopen a finished test.
        if ARTIFACT_NAME in message and "not found" in message and ("artifact membership " in message or "artifact '" in message or 'artifact "' in message):
            return None
        raise


def allowed_ledger_file(name):
    return name in {"protocol.json", "validation_gate.json", "evaluation_environment.json"} or (
        name.startswith("test_cache/") and name.endswith(".npz") and "/" not in name[len("test_cache/"):])


def restore_remote(work, protocol_sha):
    import json
    work = Path(work)
    artifact = find_remote()
    if artifact is None:
        return None
    location = Path(artifact.get_entry("test_ledger.json").download(root=str(work / "remote_ledger")))
    ledger = json.loads(location.read_text())
    if ledger["protocol_sha256"] != protocol_sha:
        raise RuntimeError("This cohort already has a final-test ledger under a different frozen protocol. "
                           "Do not reopen the test or change thresholds; return the review ZIP.")
    for entry in ledger["files"]:
        name = entry["path"]
        if not allowed_ledger_file(name):
            raise ValueError("Unexpected file in the final-test ledger.")
        target = safe_target(work, name)
        if target.is_file() and sha(target) == entry["sha256"]:
            continue
        downloaded = Path(artifact.get_entry(name).download(root=str(work / "remote_ledger")))
        if sha(downloaded) != entry["sha256"] or downloaded.stat().st_size != entry["bytes"]:
            raise ValueError("Remote evaluation cache integrity mismatch: " + name)
        if target.exists() and (name.startswith("test_cache/") or name == "protocol.json"):
            # A local cache may include later samples, but a shared UID must be byte-identical.
            raise ValueError("Local and remote evaluation records conflict: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(downloaded.read_bytes())
    dump(work / "test_ledger.json", ledger)
    dump(work / "cloud_backup_receipt.json", {"confirmed": True, "artifact_qualified_name": artifact.qualified_name,
         "artifact_url": artifact.url, "state": ledger["state"], "completed_images": ledger["completed_images"],
         "protocol_sha256": protocol_sha, "restored_existing_backup": True})
    print("Restored final evaluation ledger:", ledger["state"], "saved images:", ledger["completed_images"], flush=True)
    return ledger


def assert_no_other_active_run(ledger):
    if not ledger or ledger["completed_images"] == 114:
        return
    import wandb
    owner = ledger.get("owner_run_id")
    if owner:
        state = wandb.Api().run(ENTITY + "/" + PROJECT + "/" + owner).state
        if state not in {"finished", "failed", "crashed", "killed", "stopped"}:
            raise RuntimeError("A final evaluation run is still active. Let that notebook finish; do not run a second session in parallel.")


class LedgerWriter:
    def __init__(self, work, protocol_sha):
        import wandb
        self.work, self.protocol_sha = Path(work), protocol_sha
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Enable WANDB_API_KEY before opening the reserved test.")
        self.run = wandb.init(entity=ENTITY, project=PROJECT, name="Phase2_final_fixed_" + protocol_sha[:10],
            job_type="single_frozen_test_evaluation", dir=str(self.work), reinit="finish_previous",
            settings=wandb.Settings(disable_git=True, save_code=False), config={"protocol_sha256": protocol_sha, "test_threshold_tuning": False})
        self.last = None

    def commit(self, state, completed):
        import wandb
        paths = [self.work / n for n in ["protocol.json", "validation_gate.json", "evaluation_environment.json"]]
        paths += sorted((self.work / "test_cache").glob("*.npz"))
        inventory = [{"path": p.relative_to(self.work).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]
        ledger = {"protocol_sha256": self.protocol_sha, "state": state, "completed_images": completed,
                  "expected_images": 114, "owner_run_id": self.run.id, "files": inventory,
                  "updated_at_utc": datetime.now(timezone.utc).isoformat(), "contains_input_images": False,
                  "contains_ground_truth_masks": False, "contains_predicted_masks_packed_in_numeric_records": True,
                  "contains_raw_detector_outputs": True, "thresholds_changed_on_test": False}
        dump(self.work / "test_ledger.json", ledger)
        artifact = wandb.Artifact(ARTIFACT_NAME, type="evaluation", metadata={k: ledger[k] for k in
             ["protocol_sha256", "state", "completed_images", "thresholds_changed_on_test"]})
        for path in paths + [self.work / "test_ledger.json"]:
            artifact.add_file(str(path), name=path.relative_to(self.work).as_posix(), policy="mutable")
        logged = self.run.log_artifact(artifact, aliases=["latest"])
        logged.wait(timeout=300)
        self.last = {"confirmed": True, "artifact_qualified_name": logged.qualified_name, "artifact_url": logged.url,
                     "state": state, "completed_images": completed, "protocol_sha256": self.protocol_sha}
        dump(self.work / "cloud_backup_receipt.json", self.last)
        print("Confirmed final-evaluation backup:", completed, "/114", state, flush=True)
        return self.last

    def finish(self, success):
        self.run.finish(exit_code=0 if success else 1)
