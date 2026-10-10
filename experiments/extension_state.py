"""Explicit E06 source checks and scheduler fork after exact step100 restoration."""

import json
import math
from pathlib import Path

import torch

from quick_validation import digest, metrics


SOURCE = "goal-20261008-resume25-v1"
TRAIN_SHA = "1b577306c8e15f497648d54e96f64823db4df87cf6bfbf401fae7bbb24ecd531"
MANIFEST_SHA = "c226b985145344d7415c8e46f4e601f31fbf6147f669d144d53caa05c61ef355"


def source_info(root):
    root = Path(root).resolve()
    source = root / "outputs" / SOURCE
    ledger = json.loads((source / "training_budget.json").read_text())
    if ledger["result"] != "pass" or ledger["owned_remaining"]:
        raise ValueError("E04 must be terminal with no owned processes")
    last = json.loads((source / "last.json").read_text())
    checkpoint = Path(last["path"]).resolve()
    expected = root / "runs/charades-id2b-e04-balanced-20261008-213420/20261008213430/checkpoints/ckpt-step-100"
    from quick_train_hooks import verify_dcp
    if checkpoint != expected or verify_dcp(checkpoint, 100) != last:
        raise ValueError("Unexpected step100 recovery point")
    if digest(root / "outputs/goal-20261003-quick-v1/balanced_train.jsonl") != TRAIN_SHA:
        raise ValueError("Training input drift")
    return {"source": source, "checkpoint": checkpoint, "step": 100, "ledger": ledger}


def reuse_proxy(root, out, manifest):
    source = Path(root) / "outputs" / SOURCE
    out = Path(out)
    checked = {}
    for step in [0, 50, 100]:
        result = json.loads((source / f"metrics-step-{step}.json").read_text())
        scores = [json.loads(line) for line in (source / f"scores-step-{step}.jsonl").read_text().splitlines()]
        computed = metrics(manifest, scores)
        for key in ["macro_mrr5", "positive_negative_win_rate", "pair_count", "query_count", "rows"]:
            if computed[key] != result[key]:
                raise ValueError(f"Historical proxy mismatch at {step}: {key}")
        for name in [f"metrics-step-{step}.json", f"scores-step-{step}.jsonl"]:
            with (out / name).open("xb") as handle:
                handle.write((source / name).read_bytes())
        checked[step] = result
    best = json.loads((source / "proxy_best.json").read_text())
    if best["step"] != 50 or best["macro_mrr5"] != checked[50]["macro_mrr5"]:
        raise ValueError("Historical best changed")
    return checked[0], checked[100], best


def lr_multiplier(update):
    if update < 5:
        return .2 + .8 * update / 5
    return .2 + .8 * (1 + math.cos(math.pi * min(update - 5, 395) / 395)) / 2


def fork_schedule(trainer):
    if trainer.cur_step != 100 or trainer.total_step != 500:
        raise ValueError("Schedule fork requires restored step100 and total500")
    optimizer = trainer._engine.optimizer
    old = trainer._lr_scheduler.state_dict()
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] = 1e-6
    trainer._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    from swap_restore_fix import assert_swap_param_groups
    assert_swap_param_groups(optimizer)
    return {"kind": "schedule_fork_not_original_schedule_continuation", "source_step": 100,
            "total_step": 500, "peak_lr": 1e-6, "min_lr": 2e-7, "warmup_updates": 5,
            "cosine_updates": 395, "old_state": old, "new_state": trainer._lr_scheduler.state_dict()}
