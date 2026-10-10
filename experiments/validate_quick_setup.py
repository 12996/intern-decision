"""Read-only/CPU configuration and execution-host launch gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-source", type=Path)
    parser.add_argument("--fresh-destination-baseline", action="store_true")
    parser.add_argument("--previous-attempt", type=Path)
    parser.add_argument("--resume-source", type=Path)
    args = parser.parse_args()
    root, out = args.root, args.output
    repo, code = root / "Intern-Decision", out / "code"
    retry = None
    if args.retry_source:
        from retry_inputs import validate_retry_source
        retry = validate_retry_source(root, args.retry_source)
    input_dir = retry["source"] if retry else out
    if args.fresh_destination_baseline != (args.previous_attempt is not None):
        raise ValueError("Rebaseline requires previous-attempt accounting")
    previous_attempt = None
    if args.previous_attempt:
        if retry is None:
            raise ValueError("Rebaseline requires verified old inputs")
        from retry_inputs import validate_previous_attempt
        previous_attempt = validate_previous_attempt(root, args.previous_attempt, retry["prior_gpu_hours"])
    resume_check = None
    if args.resume_source:
        from resume_state import resume_source
        resumed = resume_source(root, args.resume_source)
        if not previous_attempt or previous_attempt["prior_gpu_hours"] != resumed["ledger"]["total_gpu_hours_charged"]:
            raise ValueError("Resume accounting mismatch")
        reference = json.loads((out / "resume_reference.json").read_text())
        if reference["checkpoint"] != str(resumed["checkpoint"]) or reference["step"] != 25:
            raise ValueError("Resume reference mismatch")
        resume_check = {"step": 25, "total_step": 100, "checkpoint": str(resumed["checkpoint"]),
                        "reference_entries": len(reference["state"]),
                        "reference_sha256": hashlib.sha256((out / "resume_reference.json").read_bytes()).hexdigest()}
    os.environ["QUICK_BASELINE_POLICY"] = "fresh_destination" if args.fresh_destination_baseline else "historical_replay"
    os.environ.update(MODEL_PATH=str(root / "models/Intern-Decision-2B"),
                      DATA_PATH=str(input_dir / "balanced_train.jsonl"), WORK_DIR=str(out / "unlaunched-config-fixture"),
                      INTERN_DECISION_REPO=str(repo), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
                      TOTAL_STEP="100", GLOBAL_BATCH_SIZE="1", SWAP_OPTIMIZER="1",
                      SAMPLE_MAX_LENGTH="2066", PACK_MAX_LENGTH="2066", HF_INTERVAL="0", DCP_INTERVAL="25",
                      CHECKPOINT_INTERVAL="0", AUTO_RESUME="0", LR="2e-6", WEIGHT_DECAY="0.01",
                      WARMUP_RATIO="0.03", LR_MIN="2e-7", LOSS_MODE="chunk", LOSS_REDUCTION="token")
    cfg = runpy.run_path(str(code / "quick_train_config.py"))["trainer"]
    assert cfg.total_step == 100 and cfg.total_epoch is None and cfg.seed == 42
    assert cfg.checkpoint_interval == 25 and cfg.checkpoint_maxkeep is None and cfg.hf_interval is None
    assert not cfg.auto_resume and cfg.resume_cfg is None
    assert cfg.optim_cfg.swap_optimizer and cfg.global_batch_size == 1
    assert cfg.model_cfg.freeze_vision and cfg.model_cfg.freeze_projector and not cfg.model_cfg.freeze_language
    assert cfg.loss_cfg.mode == "chunk" and cfg.loss_cfg.loss_reduction == "token"
    assert cfg.optim_cfg.lr == 2e-6 and cfg.optim_cfg.weight_decay == 0.01
    assert cfg.lr_cfg.warmup_ratio == 0.03 and cfg.lr_cfg.lr_min == 2e-7
    from quick_validation import digest, metrics
    provenance = json.loads((input_dir / "balanced_train.provenance.json").read_text())
    if digest(input_dir / "balanced_train.jsonl") != provenance["output_sha256"]:
        raise ValueError("Balanced dataset hash mismatch")
    reuse_check = None
    if retry:
        from retry_inputs import validate_retry_runtime
        runtime = validate_retry_runtime()
        if provenance["output_sha256"] != "1b577306c8e15f497648d54e96f64823db4df87cf6bfbf401fae7bbb24ecd531":
            raise ValueError("Only the approved balanced dataset may be reused")
        for name in ["decision_scorer.py", "quick_train_config.py"]:
            if digest(code / name) != retry["ledger"]["code_sha256"][name]:
                raise ValueError(f"Baseline implementation drift: {name}")
        manifest_path = code / "quick_validation_pairs_20261003.json"
        if digest(manifest_path) != "c226b985145344d7415c8e46f4e601f31fbf6147f669d144d53caa05c61ef355":
            raise ValueError("Frozen proxy manifest drift")
        manifest = json.loads(manifest_path.read_text())
        scores = [json.loads(line) for line in (input_dir / "scores-step-0.jsonl").read_text().splitlines()]
        replay = metrics(manifest, scores)
        original = json.loads((input_dir / "metrics-step-0.json").read_text())
        for key in ["macro_mrr5", "positive_negative_win_rate", "rows", "pair_count", "query_count"]:
            if replay[key] != original[key]:
                raise ValueError(f"Historical baseline scores/metrics mismatch: {key}")
        from quick_train_hooks import verify_dcp
        reuse_check = {"input_check": "pass", "checkpoint": verify_dcp(retry["checkpoint"], 0),
                       "baseline_macro_mrr5": replay["macro_mrr5"],
                       "cache_pt_files": len(list((input_dir / "visual-cache").glob("*.pt"))),
                       "runtime_update_probe_sha256": digest(code / "train_update_probe.py"),
                       "scaleup_observe_runtime": runtime}
    free = shutil.disk_usage(root).free
    gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    from run_quick_training import required_free, choose_gpu
    needed = required_free(args)
    try:
        free_mib, gpu = choose_gpu()
        gpu_gate = {"result": "pass", "candidate": gpu, "free_mib": free_mib}
    except RuntimeError as error:
        gpu_gate = {"result": "blocked", "reason": str(error)}
    report = {"config_check": "pass", "at": time.time(), "gpu_snapshot": gpus.strip().splitlines(),
              "disk_free_bytes": free, "required_free_bytes": needed,
              "disk_gate": "pass" if free >= needed else "blocked", "gpu_gate": gpu_gate,
              "save_policy": "Best unchanged; space-gated full DCP keeps previous durable last when skipped",
              "gpu2_forbidden": True, "snapshot_is_not_a_reservation": True,
              "new_gpu_work_started": False,
              "gpu_hours_prior": previous_attempt["prior_gpu_hours"] if previous_attempt else retry["prior_gpu_hours"] if retry else 0.5258348898092906,
              "baseline_policy": os.environ["QUICK_BASELINE_POLICY"],
              "previous_attempt_accounting": previous_attempt,
              "resume_check": resume_check,
              "retry_source": str(input_dir) if retry else None, "reuse_check": reuse_check,
              "config_sha256": hashlib.sha256((code / "quick_train_config.py").read_bytes()).hexdigest()}
    (out / "setup_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
