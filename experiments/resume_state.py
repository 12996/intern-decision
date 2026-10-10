"""Verify and restore the approved post-update DCP without restarting its schedule."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint._nested_dict import flatten_state_dict

from fp32_hf_export import tensor_sha256
from quick_validation import digest, metrics


SOURCE_NAME = "goal-20261008-swapfix-v1"
RUN_NAME = "charades-id2b-e04-balanced-20261008-192643"
RECEIPTS = {
    "last.json": "20492f5f8c9bbecf88443a4df9758a7858883ec780162e094883db819acd7b15",
    "metrics-step-0.json": "35dd76b574bcb397f5e7975e7644e53d2ad6a1ea6600d07c3c0ee2a161d959de",
    "scores-step-0.jsonl": "89382d881b11aad23a21f68737a1b37b6dd21e7685d7701ff512250f44aba0b4",
}
STEPS_SHA256 = "f987296f34f5a39371416210cc529e7d89cc165132e6e77c7304177766857640"


def resume_source(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if source != root / "outputs" / SOURCE_NAME:
        raise ValueError("Only the approved terminal step25 source may be resumed")
    from retry_inputs import SWAPFIX_ATTEMPT_SHA256
    if digest(source / "training_budget.json") != SWAPFIX_ATTEMPT_SHA256:
        raise ValueError("Resume terminal ledger changed")
    ledger = json.loads((source / "training_budget.json").read_text())
    if ledger["result"] != "fail" or ledger["owned_remaining"]:
        raise ValueError("Cannot resume a live/nonterminal attempt")
    for name, expected in RECEIPTS.items():
        if digest(source / name) != expected:
            raise ValueError(f"Resume receipt changed: {name}")
    checkpoint = root / "runs" / RUN_NAME / "20261008192654/checkpoints/ckpt-step-25"
    last = json.loads((source / "last.json").read_text())
    if Path(last["path"]).resolve() != checkpoint or last["step"] != 25:
        raise ValueError("Unexpected resume checkpoint")
    from quick_train_hooks import verify_dcp
    if verify_dcp(checkpoint, 25) != last:
        raise ValueError("Resume checkpoint metadata/size changed")
    steps = root / "runs" / RUN_NAME / "scaleup_steps.jsonl"
    if digest(steps) != STEPS_SHA256:
        raise ValueError("Historical sample-order evidence changed")
    rows = [json.loads(line) for line in steps.read_text().splitlines()]
    expected_samples = {str(row["step"]): [s["id"] for s in row["samples"]]
                        for row in rows if 26 <= row["step"] <= 30}
    if set(expected_samples) != {str(i) for i in range(26, 31)}:
        raise ValueError("Missing first five continuation sample IDs")
    return {"source": source, "checkpoint": checkpoint, "step": 25,
            "expected_next_samples": expected_samples, "ledger": ledger}


def fingerprint(value):
    if torch.is_tensor(value):
        local = value.to_local() if hasattr(value, "to_local") else value
        return {"tensor_sha256": tensor_sha256(local), "shape": list(local.shape), "dtype": str(local.dtype)}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported checkpoint fingerprint value: {type(value)}")


def make_reference(checkpoint, step=25, total_step=100):
    """Read all model/optimizer values using the official DCP CPU loader."""
    checkpoint = Path(checkpoint)
    reader = dcp.FileSystemReader(checkpoint / "weights")
    metadata = reader.read_metadata().state_dict_metadata
    total = sum(v.size.numel() * torch.empty((), dtype=v.properties.dtype).element_size()
                for v in metadata.values() if isinstance(v, TensorStorageMetadata))
    if total > 24 * 1024 ** 3:
        raise ValueError("Resume reference exceeds the 24GiB tensor allocation bound")
    state = {k: torch.empty(v.size, dtype=v.properties.dtype) if isinstance(v, TensorStorageMetadata) else None
             for k, v in metadata.items()}
    dcp.load(state, storage_reader=reader)
    steps = [v for k, v in state.items() if k.startswith("optimizer.state.") and k.endswith(".step")]
    if not steps or any(float(v) != step for v in steps):
        raise ValueError(f"Expected nonzero optimizer step{step} for every parameter")
    if not any(torch.count_nonzero(v).item() for k, v in state.items() if k.endswith(".exp_avg")):
        raise ValueError("Resume checkpoint has no learned optimizer moments")
    result = {"checkpoint": str(checkpoint), "step": step, "total_step": total_step, "tensor_allocation_bytes": total,
              "state": {k: fingerprint(v) for k, v in state.items()},
              "small_file_sha256": {name: digest(checkpoint / name) for name in
                  ["dataloader", "lr_scheduler", "rng-rank0.pt", "train_state.json", "trainer_config.json", "weights/.metadata"]}}
    return result


def check_loaded_state(trainer, reference):
    if trainer.cur_step != reference["step"] or trainer.total_step != reference.get("total_step", 100):
        raise ValueError("Resume step/horizon changed")
    current, _ = flatten_state_dict(trainer._engine._get_dcp_state_dict(cpu_offload=False))
    if set(current) != set(reference["state"]):
        raise ValueError("Restored model/optimizer key coverage changed")
    for key, expected in reference["state"].items():
        if fingerprint(current[key]) != expected:
            raise ValueError(f"Restored DCP value differs: {key}")
    if any(float(group["lr"]) <= 0 for group in trainer._engine.optimizer.param_groups):
        raise ValueError("Restored optimizer incorrectly restarted warmup")
    return {"all_model_optimizer_entries_exact": True, "entries": len(current)}


def restore_resume(trainer, resume, reference):
    from xtuner.v1.train.trainer import LoadCheckpointConfig
    from scaleup_observe import _rng_state
    from quick_train_tests import assert_tree_equal
    from swap_restore_fix import rebind_swap_param_groups
    checkpoint = resume["checkpoint"]
    if reference["checkpoint"] != str(checkpoint) or reference["step"] != resume.get("step", 25):
        raise ValueError("Wrong CPU resume reference")
    for name, expected in reference["small_file_sha256"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint changed after CPU preflight: {name}")
    original = trainer._load_checkpoint_cfg
    try:
        trainer._load_checkpoint_cfg = LoadCheckpointConfig(checkpoint_path=checkpoint,
            load_optimizer_states=True, load_optimizer_args=True, load_dataset=True, load_scheduler=True)
        trainer._load_checkpoint()
        binding = rebind_swap_param_groups(trainer._engine.optimizer)
        if not binding["applicable"]:
            raise ValueError("Expected restored SwapAdamW")
    finally:
        trainer._load_checkpoint_cfg = original
    for filename, actual in [("lr_scheduler", trainer._lr_scheduler.state_dict()),
                             ("dataloader", trainer._dataloader.get_state_dict()), ("rng-rank0.pt", _rng_state())]:
        assert_tree_equal(torch.load(checkpoint / filename, map_location="cpu", weights_only=False), actual)
    state = json.loads((checkpoint / "train_state.json").read_text())
    if trainer._cur_epoch != state["cur_epoch"] or trainer._init_total_tokens != state["total_consumed_tokens"]:
        raise ValueError("Restored epoch/token position differs")
    proof = check_loaded_state(trainer, reference)
    return {"pass": True, "step": reference["step"], "total_step": trainer.total_step, "checkpoint": str(checkpoint),
            "scheduler_data_rng_exact": True, "optimizer_group_binding": binding, **proof}


def reuse_baseline(resume, manifest, output):
    source, output = resume["source"], Path(output)
    result = json.loads((source / "metrics-step-0.json").read_text())
    scores = [json.loads(line) for line in (source / "scores-step-0.jsonl").read_text().splitlines()]
    recomputed = metrics(manifest, scores)
    for key in ["macro_mrr5", "positive_negative_win_rate", "rows", "pair_count", "query_count"]:
        if recomputed[key] != result[key]:
            raise ValueError("Same-host baseline evidence mismatch")
    for name in ["metrics-step-0.json", "scores-step-0.jsonl"]:
        target = output / name
        if target.exists():
            raise ValueError("Refuse overwrite of continuation baseline")
        target.write_bytes((source / name).read_bytes())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("Resume reference generation is CPU only")
    source = resume_source(args.root, args.source)
    reference = make_reference(source["checkpoint"])
    target = args.output / "resume_reference.json"
    with target.open("x") as handle:
        json.dump(reference, handle, indent=2)
    print(json.dumps({"pass": True, "step": reference["step"], "entries": len(reference["state"]),
                      "reference_sha256": digest(target), "tensor_allocation_bytes": reference["tensor_allocation_bytes"]}))
