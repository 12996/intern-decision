"""Validate immutable receipts for the one explicitly authorized E04 retry."""

import json
import math
from pathlib import Path

from quick_validation import digest


EXPECTED = {
    "training_budget.json": "6f49e4d70b806eef4b56a94b248321df6203af2b188dd8814d4b56069e4663ba",
    "initial_restore.json": "3e61aa01f163ba43ada32c7f39eab3611f9d4d0b90d88c73295909162bce2501",
    "metrics-step-0.json": "19f8dae3b2a862c747bced73be9dca646d8ebe755b69d67310924dd5e4151a88",
}
TRAIN_SHA256 = "1b577306c8e15f497648d54e96f64823db4df87cf6bfbf401fae7bbb24ecd531"
OBSERVE_SHA256 = "1c343f6b0ce9363b1592f3647aad7794006ffcb01b860b710cd1c670b4478a66"
MIGRATED_ATTEMPT_SHA256 = "c2be000351ba5114efbc459f60724411e4f39015c9b5894a48fe8864901261db"
REBASE_ATTEMPT_SHA256 = "e28e61ba0c6779ca65d792ebcb5a242f066bbd53b70ac9f43b32660c9f5fa3be"
SWAPFIX_ATTEMPT_SHA256 = "dc0af1b90189a2aebd62458ad336853a60fb9a3819707734504dbe997fb71e22"


def validate_previous_attempt(root, attempt, prior):
    root, attempt = Path(root).resolve(), Path(attempt).resolve()
    preceding = None
    if attempt == root / "outputs/goal-20261008-retry-v1":
        expected = MIGRATED_ATTEMPT_SHA256
    elif attempt == root / "outputs/goal-20261008-rebase-v1":
        preceding = validate_previous_attempt(root, root / "outputs/goal-20261008-retry-v1", prior)
        prior = preceding["prior_gpu_hours"]
        expected = REBASE_ATTEMPT_SHA256
    elif attempt == root / "outputs/goal-20261008-swapfix-v1":
        preceding = validate_previous_attempt(root, root / "outputs/goal-20261008-rebase-v1", prior)
        prior = preceding["prior_gpu_hours"]
        expected = SWAPFIX_ATTEMPT_SHA256
    else:
        raise ValueError("Only explicitly approved terminal attempts may be charged")
    path = attempt / "training_budget.json"
    if path.is_symlink() or digest(path) != expected:
        raise ValueError("Migrated attempt ledger changed")
    ledger = json.loads(path.read_text())
    charge = ledger["gpu_hours_charged"]
    if (ledger["result"] != "fail" or ledger["owned_remaining"]
            or not math.isfinite(charge) or charge < 0
            or abs(ledger["prior_gpu_hours"] - prior) > 1e-9
            or abs(ledger["total_gpu_hours_charged"] - prior - charge) > 1e-9):
        raise ValueError("Previous attempt is live or accounting is inconsistent")
    return {"prior_gpu_hours": prior + charge, "path": str(path),
            "sha256": expected, "charge_gpu_hours": charge, "preceding_attempt": preceding}


def validate_retry_runtime():
    import scaleup_observe
    path = Path(scaleup_observe.__file__).resolve()
    if digest(path) != OBSERVE_SHA256:
        raise ValueError("Wrong scaleup_observe runtime imported; do not shadow the executed repo version")
    return {"path": str(path), "sha256": OBSERVE_SHA256}


def validate_retry_source(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if source != root / "outputs/goal-20261003-quick-v1":
        raise ValueError("Only the approved failed E04 source may be reused")
    for name, expected in EXPECTED.items():
        if digest(source / name) != expected:
            raise ValueError(f"Historical retry receipt changed: {name}")
    ledger = json.loads((source / "training_budget.json").read_text())
    restore = json.loads((source / "initial_restore.json").read_text())
    if ledger["result"] != "fail" or ledger["owned_remaining"] or not restore["pass"] or restore["step"] != 0:
        raise ValueError("Retry requires terminal old job and passed step0 restore")
    provenance = json.loads((source / "balanced_train.provenance.json").read_text())
    if provenance["output_sha256"] != TRAIN_SHA256 or digest(source / "balanced_train.jsonl") != TRAIN_SHA256:
        raise ValueError("Historical balanced training input changed")
    checkpoint = Path(json.loads((source / "last.json").read_text())["path"])
    approved = Path(ledger["work_dir"]) / "20261003144427/checkpoints/ckpt-step-0"
    if checkpoint.is_symlink() or checkpoint.resolve() != approved or not checkpoint.is_dir():
        raise ValueError("Unexpected retry checkpoint")
    if json.loads((checkpoint / "train_state.json").read_text())["cur_step"] != 0:
        raise ValueError("Only the verified initial optimizer state may be reused")
    for role in ["positive", "random", "hard"]:
        parity = json.loads((source / f"cache-parity-{role}.json").read_text())
        if not parity["pass"] or parity["raw_error"] > 0.001 or parity["margin_error"] > 0.002:
            raise ValueError("Historical visual cache parity is invalid")
    return {"source": source, "checkpoint": checkpoint, "restore": restore,
            "prior_gpu_hours": ledger["total_gpu_hours_charged"], "ledger": ledger}
