"""Bounded evaluation only: exact model loads, fixed pairs, lossless cache."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from confirmation_protocol import digest, replay


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def model_digest(model, parts):
    import torch
    result = {}
    for name in parts:
        h = hashlib.sha256()
        for key, p in getattr(model, name).named_parameters():
            p = p.to_local() if hasattr(p, "to_local") else p
            h.update(key.encode())
            h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        result[name] = h.hexdigest()
    return result


def load_final_model_only(model, checkpoint):
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    from xtuner.v1.patch import InterleavedShardLoadPlanner
    from fp32_hf_export import tensor_sha256
    if json.loads((checkpoint / "train_state.json").read_text())["cur_step"] != 100:
        raise ValueError("Not final step100")
    state = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True, ignore_frozen_params=True))
    metadata = dcp.FileSystemReader(checkpoint / "weights").read_metadata().state_dict_metadata
    if not state or any(f"model.{key}" not in metadata for key in state):
        raise ValueError("DCP missing requested trainable model keys")
    if sum(t.numel() * t.element_size() for t in state.values()) > 10 * 1024 ** 3:
        raise ValueError("Model-only CPU restore exceeds 10GiB allocation bound")
    dcp.load({"model": state}, checkpoint_id=checkpoint / "weights", planner=InterleavedShardLoadPlanner())
    expected = {key: tensor_sha256(value) for key, value in state.items()}
    set_model_state_dict(model, state, options=StateDictOptions(cpu_offload=True, strict=False))
    actual = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True, ignore_frozen_params=True))
    if actual.keys() != expected.keys() or any(tensor_sha256(v) != expected[k] for k, v in actual.items()):
        raise ValueError("Model-only DCP runtime value mismatch")
    model.eval()
    return {"pass": True, "step": 100, "model_entries_bitwise_verified": len(expected),
            "optimizer_loaded": False, "weights_exported": False, "inference_only_not_training_resume": True,
            "dcp_metadata_sha256": digest(checkpoint / "weights/.metadata")}


def main():
    import torch
    import torch.distributed as dist
    from decision_scorer import DecisionScorer, append_record
    from frozen_visual_cache import FrozenVisualCache
    root, out = Path(os.environ["DECISION_ROOT"]), Path(os.environ["CONFIRM_OUT"])
    deadline = float(os.environ["CONFIRM_DEADLINE"])
    name = os.environ["CONFIRM_MODEL"]
    selected_first = os.environ.get("CONFIRM_SELECTED_FIRST") == "1"
    released_completion = os.environ.get("CONFIRM_RELEASED_COMPLETION") == "1"
    if released_completion and (name != "released" or selected_first):
        raise ValueError("Released completion cannot change the model or select a checkpoint")
    if selected_first and name != "step50":
        raise ValueError("Selected-first mode only permits the fixed step50 model")
    started = time.time()
    manifest_path = out / "confirmation_manifest.json"
    if digest(manifest_path) != os.environ["CONFIRM_MANIFEST_SHA256"]:
        raise ValueError("Frozen confirmation manifest changed")
    manifest = json.loads(manifest_path.read_text())
    legacy = root / "outputs/goal-20261003-quick-v1"
    history = root / "outputs/goal-20261008-resume25-v1"
    original = root / "models/Intern-Decision-2B"
    best = json.loads((history / "proxy_best.json").read_text())
    if best["step"] != 50:
        raise ValueError("Selected50 artifact changed")
    checkpoint = Path(best["weights"]) if name == "step50" else original
    scorer = DecisionScorer(checkpoint)
    load_receipt = {"pass": True, "mode": "strict_hf", "checkpoint": str(checkpoint)}
    if name == "step100":
        checkpoint = Path(json.loads((history / "last.json").read_text())["path"])
        load_receipt = load_final_model_only(scorer.model, checkpoint)
    frozen = model_digest(scorer.model, ["vision_tower", "multi_modal_projector"])
    old_restore = json.loads((legacy / "initial_restore.json").read_text())
    expected_frozen = {key: old_restore["weights"][key] for key in frozen}
    if frozen != expected_frozen:
        raise ValueError("Frozen visual parameters differ; cache forbidden")
    old_provenance = {"frozen_parameters": frozen, "model_revision": "8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
        "manifest": "c226b985145344d7415c8e46f4e601f31fbf6147f669d144d53caa05c61ef355",
        "scorer_code": digest(Path(__file__).with_name("decision_scorer.py"))}
    old_cache = FrozenVisualCache(legacy / "visual-cache", old_provenance, read_only=True)
    new_provenance = dict(old_provenance, manifest=digest(manifest_path))
    cache_directory = root / "outputs/goal-20261009-test-step50-v1/visual-cache" if released_completion else out / "visual-cache"
    new_cache = FrozenVisualCache(cache_directory, new_provenance,
                                 read_only=released_completion or (name != "released" and not selected_first), max_bytes=6 * 1024 ** 3)
    shared_cache = None
    if selected_first or released_completion:
        shared = root / "outputs/goal-20261008-confirm-v2"
        from selected_eval_protocol import MANIFEST_SHA
        if digest(shared / "confirmation_manifest.json") != MANIFEST_SHA or digest(manifest_path) != MANIFEST_SHA:
            raise ValueError("Shared cache confirmation protocol mismatch")
        shared_cache = FrozenVisualCache(shared / "visual-cache", new_provenance, read_only=True,
                                         max_bytes=6 * 1024 ** 3)
    def choose_cache(identity):
        key = hashlib.sha256(json.dumps(old_cache.identity(identity), sort_keys=True).encode()).hexdigest()
        if (old_cache.directory / f"{key}.pt").is_file():
            return old_cache
        if shared_cache is not None:
            key = hashlib.sha256(json.dumps(shared_cache.identity(identity), sort_keys=True).encode()).hexdigest()
            if (shared_cache.directory / f"{key}.pt").is_file():
                return shared_cache
        return new_cache
    def guard():
        if time.time() >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
            raise TimeoutError("Confirmation time/disk bound reached")
    proxy = json.loads((history / "code/quick_validation_pairs_20261003.json").read_text())
    historical = {r["pair_id"]: r for r in
        (json.loads(line) for line in (history / f"scores-step-{ {'released': 0, 'step50': 50, 'step100': 100}[name]}.jsonl").read_text().splitlines())}
    sentinel = []
    for kind in ("positive", "hard", "random"):
        guard()
        pair = next(p for p in proxy["pairs"] if p["pair_type"] == kind)
        prepared = scorer.prepare(pair)
        reference = historical[pair["id"]]
        if prepared[2] != reference["input"]:
            raise ValueError("Sentinel input drift")
        result = scorer.forward(prepared, visual_cache=old_cache)
        raw_error = max(abs(result["raw_logits"][k] - reference["raw_logits"][k]) for k in ("no", "yes"))
        margin_error = abs(result["score"] - reference["score"])
        passed = raw_error <= .001 and margin_error <= .002 and result["visual_cache_hit"]
        sentinel.append({"pair_id": pair["id"], "raw_error": raw_error, "margin_error": margin_error, "pass": passed})
        if not passed:
            raise ValueError("Prospective same-host sentinel tolerance failed")
        del prepared
    write_json(out / f"load-and-parity-{name}.json", {"pass": True, "model": name, "load": load_receipt,
        "frozen_parameters": frozen, "sentinel": sentinel, "elapsed_s": time.time() - started})
    if time.time() - started > 1800:
        raise TimeoutError("Model load/sentinel gate exceeds30min")
    sources = json.loads((out / "frame_sources.json").read_text())["frame_sha256"]
    pairs = [{"id": f"{r['cap_id']}#candidate#{v}", "cap_id": r["cap_id"], "query": r["query"], "video_id": v}
             for r in manifest["rows"] if r["positive_rank"] <= 20 for v in r["top20"]]
    score_path = out / f"scores-{name}.jsonl"
    if score_path.exists():
        raise FileExistsError("No automatic scoring retry or overwrite")
    prior_inputs = {}
    seed = []
    if released_completion:
        from released_completion_protocol import verify_seed
        seed_path = out / "scores-released-seed.jsonl"
        if digest(seed_path) != os.environ["CONFIRM_SEED_SHA256"]:
            raise ValueError("Released seed changed after preflight")
        seed = [json.loads(line) for line in seed_path.read_text().splitlines()]
        selected = [json.loads(line) for line in (out / "scores-step50.jsonl").read_text().splitlines()]
        _, prior_inputs = verify_seed(manifest, seed, selected)
        if len(seed) != 369:
            raise ValueError("Only the preserved369pair seed is authorized")
        checks = []
        for row in (seed[0], seed[len(seed)//2], seed[-1]):
            guard()
            pair = next(p for p in pairs if p["id"] == row["pair_id"])
            prepared = scorer.prepare(pair)
            if prepared[2] != row["input"]:
                raise ValueError("Reused released input drift")
            current = scorer.forward(prepared, visual_cache=choose_cache(prepared[2]))
            raw_error = max(abs(current["raw_logits"][k] - row["raw_logits"][k]) for k in ("yes", "no"))
            margin_error = abs(current["score"] - row["score"])
            if not current["visual_cache_hit"] or raw_error > .001 or margin_error > .002:
                raise ValueError("Reused released scoring parity failed")
            checks.append({"pair_id": row["pair_id"], "raw_error": raw_error, "margin_error": margin_error})
        write_json(out / "seed-replay-parity.json", {"pass": True, "checks": checks, "reused_pairs": len(seed)})
        with score_path.open("xb") as handle:
            handle.write(seed_path.read_bytes())
    elif selected_first:
        from selected_eval_protocol import partial_inputs
        prior_inputs = partial_inputs(manifest, [json.loads(line) for line in
            (root / "outputs/goal-20261008-confirm-v2/scores-released.jsonl").read_text().splitlines()])
    elif name != "released":
        prior_inputs = {r["pair_id"]: r["input"] for r in
            (json.loads(line) for line in (out / "scores-released.jsonl").read_text().splitlines())}
    scores, new_miss_times, checked_new = list(seed), [], False
    reused_ids = {row["pair_id"] for row in seed}
    for i, pair in enumerate(pairs):
        guard()
        if pair["id"] in reused_ids:
            continue
        prepared = scorer.prepare(pair)
        identity = prepared[2]
        if identity["frame_sha256"] != [sources[f"{pair['video_id']}/frame_{j:02d}.png"] for j in range(8)]:
            raise ValueError("Frame source changed")
        if (selected_first and pair["id"] in prior_inputs and prior_inputs[pair["id"]] != identity
                or not selected_first and prior_inputs and prior_inputs[pair["id"]] != identity):
            raise ValueError("Models saw different scoring inputs")
        cache = choose_cache(identity)
        result = scorer.forward(prepared, visual_cache=cache)
        if not result["visual_cache_hit"]:
            new_miss_times.append(result["timing"]["forward_sync_s"])
            if not checked_new:
                again = scorer.prepare(pair)
                if again[2] != identity:
                    raise ValueError("New cache parity input changed")
                cached = scorer.forward(again, visual_cache=cache)
                raw_error = max(abs(result["raw_logits"][k] - cached["raw_logits"][k]) for k in ("yes", "no"))
                margin_error = abs(result["score"] - cached["score"])
                write_json(out / "new-cache-parity.json", {"pass": raw_error <= .001 and margin_error <= .002,
                    "raw_error": raw_error, "margin_error": margin_error, "pair_id": pair["id"]})
                if raw_error > .001 or margin_error > .002 or not cached["visual_cache_hit"]:
                    raise ValueError("New visual cache parity failed")
                checked_new = True
                del again
        result.update(model=name, pair_id=pair["id"], cap_id=pair["cap_id"], video_id=pair["video_id"],
                      confirmation_manifest_sha256=digest(manifest_path))
        append_record(score_path, result)
        scores.append(result)
        write_json(out / "progress.json", {"model": name, "worker_pid": os.getpid(), "pairs_done": i+1,
            "pairs_total": len(pairs), "at": time.time(), "new_visual_forwards": len(new_miss_times),
            "mean_new_visual_forward_s": sum(new_miss_times) / len(new_miss_times) if new_miss_times else None})
        if i == 0 or (i+1) % 10 == 0:
            print(json.dumps({"event": "progress", "model": name, "pairs": i+1, "total": len(pairs),
                "new_visual_forwards": len(new_miss_times), "at": time.time()}), flush=True)
        if len(new_miss_times) == 5:
            estimated_remaining = (390 - len(new_miss_times)) * sum(new_miss_times) / len(new_miss_times) + 1800
            if name == "released" and estimated_remaining > deadline - time.time():
                raise TimeoutError("Measured throughput cannot complete locked evaluation within10GPUh")
        del prepared
    report = replay(manifest, scores)
    report.update(model=name, manifest_sha256=digest(manifest_path), source_checkpoint=str(checkpoint),
                  scoring_scope="metric-exact label-aware miss shortcut; not full deployed latency",
                  selected_first=selected_first, partial_released_input_checks=len(prior_inputs) if selected_first else None,
                  released_completion=released_completion, reused_pairs=len(seed), new_pairs=len(pairs)-len(seed))
    write_json(out / f"metrics-{name}.json", report)
    print(json.dumps({"event": "model_complete", "model": name, "metrics": report["metrics"]}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
