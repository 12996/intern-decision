"""Fixed proxy validation, best weights and verified rolling full resume state."""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import time

import torch

from frozen_visual_cache import FrozenVisualCache
from quick_validation import metrics, digest
from train_update_probe import LanguageUpdateProbe
from baseline_policy import baseline_policy, requires_historical_replay
from swap_restore_fix import rebind_swap_param_groups, assert_swap_param_groups
from fp32_hf_export import save_hf_fp32
from scaleup_observe import _rng_state, _restore_rng, _optimizer_probe, _data_probe, _scheduler_probe


GIB = 1024 ** 3


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def event(trainer, name, **payload):
    record = {"event": name, "step": trainer.cur_step, "epoch": time.time(), **payload}
    line = "QUICK_EVENT " + json.dumps(record)
    print(line, flush=True)
    for name in ["quick_events.jsonl", "log.txt"]:
        with (Path(trainer.work_dir) / name).open("a") as handle:
            handle.write(line.removeprefix("QUICK_EVENT ") + "\n" if name.endswith("jsonl") else line + "\n")
            handle.flush()


def check_disk(path, reserve=0):
    free = shutil.disk_usage(path).free
    if free < 50 * GIB + reserve:
        raise RuntimeError(f"Disk gate: free={free}, need={50 * GIB + reserve}")
    return free


def model_digest(model, parts):
    result = {}
    for name in parts:
        h = hashlib.sha256()
        for key, p in getattr(model, name).named_parameters():
            p = p.to_local() if hasattr(p, "to_local") else p
            h.update(key.encode())
            h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        result[name] = h.hexdigest()
    return result


@contextmanager
def validation_state(model):
    rng = _rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        _restore_rng(rng)


def safe_remove(path, parent, prefix):
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.resolve().parent != parent or not path.name.startswith(prefix):
        raise ValueError("Refuse removal outside this run's checkpoint namespace")
    shutil.rmtree(path)


def verify_dcp(path, step):
    path = Path(path)
    for filename in ["dataloader", "lr_scheduler", "rng-rank0.pt", "train_state.json", "trainer_config.json"]:
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: {filename}")
    if json.loads((path / "train_state.json").read_text())["cur_step"] != step:
        raise ValueError("Checkpoint step mismatch")
    from torch.distributed.checkpoint import FileSystemReader
    metadata = FileSystemReader(path / "weights").read_metadata()
    keys = list(metadata.state_dict_metadata)
    if not any(k.startswith("model.") for k in keys) or not any(k.startswith("optimizer.") for k in keys):
        raise ValueError("DCP does not contain model AND optimizer")
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if size > 23 * GIB:
        raise ValueError("Actual DCP exceeds the frozen23GiB rolling-save allowance")
    return {"path": str(path), "step": step, "metadata_entries": len(keys), "bytes": size}


def maybe_save_with_space(trainer, original, is_snapshot=False):
    interval = trainer._snapshot_interval if is_snapshot else trainer._checkpoint_interval
    if interval is None:
        return False
    due = trainer.cur_step == trainer.total_step if interval == -1 else (
        trainer.cur_step % interval == 0 or (not is_snapshot and trainer.cur_step == trainer.total_step))
    if not due:
        return False
    free = shutil.disk_usage(trainer.work_dir).free
    required = 74 * GIB  # 50GiB reserve + 23GiB new DCP + 1GiB write margin.
    if free < required:
        if trainer.cur_step == 0:
            raise RuntimeError("Initial restore-test checkpoint cannot be skipped")
        record = {"step": trainer.cur_step, "free_bytes": free, "required_bytes": required,
                  "previous_last_kept": True, "final_resume_may_precede_final_weights": trainer.cur_step == trainer.total_step}
        atomic_json(Path(os.environ["QUICK_OUT"]) / f"last-save-skipped-{trainer.cur_step}.json", record)
        event(trainer, "last_save_skipped_for_space", **record)
        return False
    return original(trainer, is_snapshot=is_snapshot)


def maybe_save_then_validate(trainer, original, is_snapshot=False):
    # The official loop calls _maybe_save_hf before _maybe_save. Keep validation
    # here instead, so an export exception cannot prevent the periodic DCP save.
    if getattr(trainer, "_async_checkpoint", False):
        raise ValueError("Best export requires synchronous durable checkpoint saves")
    saved = maybe_save_with_space(trainer, original, is_snapshot=is_snapshot)
    validation_due = trainer.cur_step in [50, 100] and getattr(trainer, "_quick_validated_step", None) != trainer.cur_step
    if not is_snapshot and validation_due:
        event(trainer, "validation_after_resume_save", full_resume_saved=bool(saved),
              previous_last_kept_if_space_skip=not bool(saved))
        result = evaluate(trainer, trainer.cur_step)
        select_best(trainer, result)
        trainer._quick_validated_step = trainer.cur_step
        if trainer.cur_step == 50:
            trainer._quick_below_start = result["macro_mrr5"] < trainer._quick_baseline["macro_mrr5"]
        elif (getattr(trainer, "_quick_below_start", False)
              and result["macro_mrr5"] < trainer._quick_baseline["macro_mrr5"]):
            event(trainer, "two_validations_below_start_do_not_extend")
    return saved


def paired_change(baseline, current):
    old = {r["positive_video"]: r["rr"] for r in baseline["rows"]}
    deltas = [r["rr"] - old[r["positive_video"]] for r in current["rows"]]
    rng = random.Random(42)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    return {"mean_rr_delta": sum(deltas) / len(deltas), "paired_positive_video_bootstrap95": [means[49], means[1949]],
            "improved_queries": sum(d > 0 for d in deltas), "worsened_queries": sum(d < 0 for d in deltas),
            "unchanged_queries": sum(d == 0 for d in deltas), "single_seed_exploratory": True}


def evaluate(trainer, step):
    out = Path(os.environ["QUICK_OUT"])
    model = trainer._engine.model
    if model_digest(model, ["vision_tower", "multi_modal_projector"]) != trainer._quick_frozen_hashes:
        raise ValueError("Frozen visual weights changed")
    results, times = [], []
    event(trainer, "validation_start", validation_step=step)
    with validation_state(model):
        for i, pair in enumerate(trainer._quick_manifest["pairs"]):
            check_disk(out)
            if time.time() >= float(os.environ["QUICK_GLOBAL_DEADLINE"]):
                raise TimeoutError("TotalGPUbudget reached during validation")
            prepared = trainer._quick_scorer.prepare(pair)
            if prepared[2]["frame_sha256"] != pair["frame_sha256"]:
                raise ValueError("Frozen frame bytes changed")
            result = trainer._quick_scorer.forward(prepared, visual_cache=trainer._quick_cache)
            if step == 0 and requires_historical_replay() and hasattr(trainer, "_quick_reused_scores"):
                reference = trainer._quick_reused_scores[pair["id"]]
                raw_error = max(abs(result["raw_logits"][k] - reference["raw_logits"][k]) for k in ["no", "yes"])
                if not result["visual_cache_hit"] or raw_error > 0.001 or abs(result["score"] - reference["score"]) > 0.002:
                    raise ValueError("Reused baseline cached replay is not equivalent")
            result.update(pair_id=pair["id"], cap_id=pair["cap_id"], video_id=pair["video_id"], step=step,
                          scoring_full_vocab_ce_fp32_diagnostic=result["decision_full_vocab_logsumexp_fp32"] - result["raw_logits"][pair["label"]])
            with (out / f"scores-step-{step}.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
                handle.flush()
            results.append(result)
            times.append(result["timing"]["forward_sync_s"])
            if step == 0 and not trainer._quick_cache.validated:
                # Three roles are checked against a fresh complete8-frame forward.
                kind = pair["pair_type"]
                if kind not in trainer._quick_cache_roles:
                    repeated = trainer._quick_scorer.forward(prepared, visual_cache=trainer._quick_cache)
                    raw_error = max(abs(repeated["raw_logits"][k] - result["raw_logits"][k]) for k in ["no", "yes"])
                    margin_error = abs(repeated["score"] - result["score"])
                    passed = repeated["visual_cache_hit"] and raw_error <= 0.001 and margin_error <= 0.002
                    atomic_json(out / f"cache-parity-{kind}.json", {"pass": passed, "pair": pair["id"],
                                "raw_error": raw_error, "margin_error": margin_error, "cache_key": result["visual_cache_key"],
                                "uncached_s": times[-1], "cached_s": repeated["timing"]["forward_sync_s"]})
                    if not passed:
                        raise ValueError("Cache parity failed; training gate remains closed")
                    trainer._quick_cache_roles.add(kind)
                    trainer._quick_cache.validated = len(trainer._quick_cache_roles) == 3
                    event(trainer, "cache_parity", kind=kind, pass_check=passed, raw_error=raw_error, margin_error=margin_error)
            event(trainer, "validation_pair", validation_step=step, complete=i + 1, total=160,
                  cache_hit=result["visual_cache_hit"], forward_s=times[-1], pair_id=pair["id"])
            del prepared
    if not trainer._quick_cache.validated:
        raise ValueError("Missing three-role cache gate")
    result = metrics(trainer._quick_manifest, results)
    result.update(step=step, manifest_sha256=os.environ["QUICK_MANIFEST_SHA256"],
                  mean_forward_s=sum(times) / len(times))
    if step != 0:
        result["paired_change_from_start"] = paired_change(trainer._quick_baseline, result)
    atomic_json(out / f"metrics-step-{step}.json", result)
    event(trainer, "validation_complete", macro_mrr5=result["macro_mrr5"], mean_forward_s=result["mean_forward_s"],
          positive_negative_win_rate=result["positive_negative_win_rate"])
    return result


def select_best(trainer, result):
    out = Path(os.environ["QUICK_OUT"])
    old = getattr(trainer, "_quick_best", None)
    if old is not None and result["macro_mrr5"] <= old["macro_mrr5"]:
        return
    step = result["step"]
    if step == 0:
        path = Path(os.environ["MODEL_PATH"])
    else:
        check_disk(out, 9 * GIB)
        path = Path(trainer.exp_dir) / f"proxy-best-step-{step}"
        # Preserve selected FP32 master parameters. A BF16 export can round away
        # small updates and no longer represent the model we just validated.
        export_receipt = save_hf_fp32(trainer._engine, path)
        trainer.tokenizer.save_pretrained(path)
        # Export uses the same pinned architecture/processor; fill absent metadata.
        for source in Path(os.environ["MODEL_PATH"]).glob("*.json"):
            if not (path / source.name).exists() and not source.name.startswith("model.safetensors"):
                shutil.copy2(source, path / source.name)
        for source in Path(os.environ["MODEL_PATH"]).glob("*.jinja"):
            if not (path / source.name).exists():
                shutil.copy2(source, path / source.name)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        for filename in set(index["weight_map"].values()):
            if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
                raise ValueError("Incomplete best HF export")
        if sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) > 9 * GIB:
            raise ValueError("HF export exceeds9GiB disk allowance")
        event(trainer, "best_fp32_export_verified", **export_receipt)
    record = {"step": step, "macro_mrr5": result["macro_mrr5"], "weights": str(path),
              "manifest_sha256": os.environ["QUICK_MANIFEST_SHA256"], "proxy_only_not_confirmation_best": True}
    atomic_json(out / "proxy_best.json", record)
    trainer._quick_best = record
    if old is not None and old["step"] != 0 and Path(old["weights"]).resolve().parent == Path(trainer.exp_dir).resolve():
        safe_remove(old["weights"], trainer.exp_dir, "proxy-best-step-")
    event(trainer, "proxy_best_selected", **record)


def install():
    from xtuner.v1.train.trainer import Trainer, LoadCheckpointConfig
    if getattr(Trainer.fit, "_quick_wrapped", False):
        return
    origin_fit = Trainer.fit
    origin_finalize, origin_log = Trainer._finalize_dcp_save, Trainer._log_step
    origin_save = Trainer._maybe_save

    def finalize(self, *args, **kwargs):
        path = Path(kwargs["checkpoint_path"])
        check_disk(self.work_dir)
        result = origin_finalize(self, *args, **kwargs)
        record = verify_dcp(path, self.cur_step)
        atomic_json(Path(os.environ["QUICK_OUT"]) / "last.json", record)
        # Delete only prior checkpoints created by this new experiment, after RNG
        # and DCP metadata have passed; never invoke the legacy historical pruner.
        listed = self.meta.latest_exp.checkpoint_list
        for previous in list(listed[:-1]):
            safe_remove(previous, self.checkpoint_dir, "ckpt-step-")
            listed.remove(previous)
        atomic_json(Path(self.work_dir) / self._META_PATH, json.loads(self.meta.model_dump_json()))
        event(self, "last_saved_verified", **record)
        return result

    def fit(self, *args, **kwargs):
        from decision_scorer import DecisionScorer
        if self.world_size != 1 or self.cur_step != 0:
            raise ValueError("This isolated pilot requires fresh single-rank training")
        out = Path(os.environ["QUICK_OUT"])
        manifest_path = Path(os.environ["QUICK_MANIFEST"])
        if digest(manifest_path) != os.environ["QUICK_MANIFEST_SHA256"]:
            raise ValueError("Quick manifest hash mismatch")
        self._quick_manifest = json.loads(manifest_path.read_text())
        retry = None
        if os.environ.get("QUICK_RETRY_SOURCE"):
            from retry_inputs import validate_retry_source, validate_retry_runtime
            retry = validate_retry_source(os.environ["DECISION_ROOT"], os.environ["QUICK_RETRY_SOURCE"])
            validate_retry_runtime()
        resume = None
        if os.environ.get("QUICK_RESUME_SOURCE"):
            from resume_state import resume_source, restore_resume, reuse_baseline
            resume = resume_source(os.environ["DECISION_ROOT"], os.environ["QUICK_RESUME_SOURCE"])
            if not retry or requires_historical_replay():
                raise ValueError("Continuation requires verified cache and same-host step0 comparison")
            reference_path = out / "resume_reference.json"
            if digest(reference_path) != os.environ["QUICK_RESUME_REFERENCE_SHA256"]:
                raise ValueError("CPU resume reference changed")
            self._quick_resume_reference = json.loads(reference_path.read_text())
            self._quick_expected_samples = resume["expected_next_samples"]
        self._quick_frozen_hashes = model_digest(self._engine.model, ["vision_tower", "multi_modal_projector"])
        provenance = {"frozen_parameters": self._quick_frozen_hashes,
                      "model_revision": "8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
                      "manifest": os.environ["QUICK_MANIFEST_SHA256"],
                      "scorer_code": digest(Path(__file__).with_name("decision_scorer.py"))}
        cache_dir = retry["source"] / "visual-cache" if retry else out / "visual-cache"
        self._quick_cache = FrozenVisualCache(cache_dir, provenance, read_only=retry is not None)
        self._quick_cache_roles = set()
        if retry:
            if digest(Path(__file__).with_name("decision_scorer.py")) != retry["ledger"]["code_sha256"]["decision_scorer.py"]:
                raise ValueError("Scorer changed; old baseline cannot be reused")
            rows = [json.loads(line) for line in (retry["source"] / "scores-step-0.jsonl").read_text().splitlines()]
            self._quick_reused_scores = {row["pair_id"]: row for row in rows}
            if len(rows) != 160 or set(self._quick_reused_scores) != {p["id"] for p in self._quick_manifest["pairs"]}:
                raise ValueError("Reused baseline has missing/duplicate pairs")
            self._quick_cache_roles = {"positive", "random", "hard"}
            self._quick_cache.validated = True
        self._quick_scorer = DecisionScorer(os.environ["MODEL_PATH"], existing_model=self._engine.model)
        # Real official save+load before the data iterator starts. The same
        # initialized optimizer, scheduler, data position and RNG must survive.
        before_weights = model_digest(self._engine.model, ["language_model", "vision_tower", "multi_modal_projector"])
        if retry and before_weights != retry["restore"]["weights"]:
            raise ValueError("Released model weights drifted; baseline reuse forbidden")
        if resume:
            check_disk(out)
            restore = restore_resume(self, resume, self._quick_resume_reference)
            path = resume["checkpoint"]
            after_weights = model_digest(self._engine.model, list(before_weights))
            if any(after_weights[name] != self._quick_frozen_hashes[name] for name in self._quick_frozen_hashes):
                raise ValueError("Resume changed frozen visual weights")
            if after_weights["language_model"] == before_weights["language_model"]:
                raise ValueError("Step25 incorrectly restored released language weights")
            restore.update(weights=after_weights, work_dir=str(self.work_dir),
                           reference_sha256=os.environ["QUICK_RESUME_REFERENCE_SHA256"])
            atomic_json(out / "initial_restore.json", restore)
            atomic_json(out / "optimizer_group_restore.json", restore["optimizer_group_binding"])
            atomic_json(out / "last.json", verify_dcp(path, 25))
            event(self, "post_update_resume_exact", **restore)
        else:
            before_rng = pickle.dumps(_rng_state())
            before_scheduler = self._lr_scheduler.state_dict()
            before_data = self._dataloader.get_state_dict()
            before_optimizer = _optimizer_probe(self)
            if before_optimizer.get("step") != 0 or before_optimizer.get("device") != "cpu":
                raise ValueError("Fresh SwapAdamW must have CPU moments and zero optimizer step")
            if retry:
                check_disk(out)
                path = retry["checkpoint"]
                before_rng = pickle.dumps(torch.load(path / "rng-rank0.pt", map_location="cpu", weights_only=False))
            else:
                check_disk(out, 25 * GIB)
                self._maybe_save(is_snapshot=False)
                path = self._get_checkpoint_path(epoch=self._cur_epoch, step=0)
            saved_cfg = self._load_checkpoint_cfg
            try:
                self._load_checkpoint_cfg = LoadCheckpointConfig(checkpoint_path=path)
                self._load_checkpoint()
                binding = rebind_swap_param_groups(self._engine.optimizer)
                if not binding["applicable"]:
                    raise ValueError("Expected SwapAdamW in this fixed pilot")
                atomic_json(out / "optimizer_group_restore.json", binding)
                event(self, "optimizer_groups_rebound", **binding)
            finally:
                self._load_checkpoint_cfg = saved_cfg
            after_weights = model_digest(self._engine.model, list(before_weights))
            from quick_train_tests import assert_tree_equal
            assert_tree_equal(pickle.loads(before_rng), _rng_state())
            assert_tree_equal(before_scheduler, self._lr_scheduler.state_dict())
            assert_tree_equal(before_data, self._dataloader.get_state_dict())
            if before_weights != after_weights or self.cur_step != 0:
                raise ValueError("Initial save/restore changed model/step")
            if _optimizer_probe(self) != before_optimizer:
                raise ValueError("Initial save/restore changed optimizer probe")
            atomic_json(out / "initial_restore.json", {"pass": True, "weights": after_weights,
                        "scheduler": _scheduler_probe(self._lr_scheduler), "data": _data_probe(self),
                        "optimizer": _optimizer_probe(self), "step": self.cur_step,
                        "checkpoint": str(path), "reused_existing_checkpoint": retry is not None,
                        "limitation": "Initialized optimizer before actual updates; nonzero moments covered by CPU DCP test, not a2Bpost-update restore claim"})
            event(self, "initial_save_restore_pass")
            if retry:
                atomic_json(out / "last.json", verify_dcp(path, 0))
        self._quick_start_step = self.cur_step
        self._quick_smoke_step = self.cur_step + 5
        self._quick_update_probe = LanguageUpdateProbe(self._engine.model.language_model,
            lambda: model_digest(self._engine.model, ["language_model"])["language_model"], after_weights["language_model"])
        self._quick_baseline = reuse_baseline(resume, self._quick_manifest, out) if resume else evaluate(self, 0)
        if retry and requires_historical_replay() and self._quick_baseline["macro_mrr5"] != json.loads((retry["source"] / "metrics-step-0.json").read_text())["macro_mrr5"]:
            raise ValueError("Reused released baseline ranking changed")
        atomic_json(out / "baseline_policy.json", {
            "policy": baseline_policy(), "comparison_baseline": str(out / "metrics-step-0.json"),
            "reused_same_host_step0": bool(resume), "resume_step": self._quick_start_step,
            "historical_score_equivalence_required": requires_historical_replay(),
            "historical_cross_host_improvement_claim_allowed": requires_historical_replay(),
            "same_host_cached_proxy_only": True, "single_seed": 42})
        select_best(self, self._quick_baseline)
        start = time.time()
        atomic_json(out / "training_started.json", {"epoch": start, "deadline_epoch": start + 4 * 3600,
                    "trainer_pid": os.getpid(), "work_dir": str(self.work_dir), "exp_dir": str(self.exp_dir),
                    "resume_step": self._quick_start_step, "remaining_updates": 100 - self._quick_start_step})
        self._quick_train_start = start
        event(self, "optimizer_phase_start", total_step=100, resume_step=self._quick_start_step,
              remaining_updates=100 - self._quick_start_step, max_seconds=14400)
        result = origin_fit(self, *args, **kwargs)
        # Certify that the exported artifact, not just the in-memory model,
        # retains the selected score. This runs after the full last is safe.
        selected = self._quick_best
        self._engine.model.from_hf(selected["weights"], strict=True)
        pair = self._quick_manifest["pairs"][0]
        reference = json.loads((out / f"scores-step-{selected['step']}.jsonl").read_text().splitlines()[0])
        with validation_state(self._engine.model):
            if model_digest(self._engine.model, ["vision_tower", "multi_modal_projector"]) != self._quick_frozen_hashes:
                raise ValueError("Export reload changed frozen weights")
            replay = self._quick_scorer.forward(self._quick_scorer.prepare(pair), visual_cache=self._quick_cache)
        raw_error = max(abs(replay["raw_logits"][k] - reference["raw_logits"][k]) for k in ["no", "yes"])
        margin_error = abs(replay["score"] - reference["score"])
        passed = raw_error <= 0.001 and margin_error <= 0.002
        atomic_json(out / "best_export_reload_parity.json", {"pass": passed, "best": selected,
                    "raw_error": raw_error, "margin_error": margin_error, "sentinel_pair": pair["id"],
                    "scope": "One same-input score after2BHF reload; not a full extra evaluation"})
        if not passed:
            raise ValueError("Selected HF artifact reload parity failed")
        event(self, "best_export_reload_parity_pass", raw_error=raw_error, margin_error=margin_error)
        return result

    def log_step(self, *args, **kwargs):
        result = origin_log(self, *args, **kwargs)
        binding = assert_swap_param_groups(self._engine.optimizer)
        event(self, "optimizer_group_reference_check", **binding)
        path = Path(self.work_dir) / "scaleup_steps.jsonl"
        row = json.loads(path.read_text().splitlines()[-1])
        if not row["loss_finite"] or not math.isfinite(row["grad_norm"]):
            raise ValueError("Nonfinite training step")
        if row["vision_max_abs_delta"] != 0 or row["projector_max_abs_delta"] != 0:
            raise ValueError("Frozen parameter sample changed")
        expected = getattr(self, "_quick_expected_samples", {}).get(str(self.cur_step))
        if expected is not None and [s["id"] for s in row["samples"]] != expected:
            raise ValueError("Resumed data order differs from original continuation")
        updates = self.cur_step - self._quick_start_step
        proof = self._quick_update_probe.check(updates, verify_full=self.cur_step == self._quick_smoke_step)
        proof.update(step=self.cur_step, updates_since_restore=updates)
        event(self, "language_update_probe", **proof)
        if updates >= 3 and not proof["update_detected"]:
            raise ValueError("Language parameters unchanged: full-state hash verified")
        if self.cur_step == self._quick_smoke_step:
            if proof["after_sha256"] == proof["before_sha256"]:
                raise ValueError("Five-step whole-language hash did not change")
            frozen = model_digest(self._engine.model, ["vision_tower", "multi_modal_projector"])
            if frozen != self._quick_frozen_hashes:
                raise ValueError("Frozen visual parameters changed in the five-step smoke")
            if os.environ.get("QUICK_SMOKE_DEADLINE") and time.time() >= float(os.environ["QUICK_SMOKE_DEADLINE"]):
                raise TimeoutError("Five-step smoke exceeded30minutes; no continuation authorized")
            receipt = {"pass": True, "step": self.cur_step, "resume_step": self._quick_start_step,
                       "updates_since_restore": updates, "total_step": 100, "at": time.time(), "update_proof": proof,
                       "work_dir": str(self.work_dir), "trainer_pid": os.getpid(),
                       "frozen_visual_unchanged": True, "finite_loss_and_gradient": True,
                       "first_five_steps_are_part_of_the100_update_pilot": True}
            atomic_json(Path(os.environ["QUICK_OUT"]) / "smoke_pass.json", receipt)
            event(self, "five_step_smoke_pass")
        check_disk(self.work_dir)
        return result

    def maybe_hf(self):
        # Only time control here; validation/export follow the durable DCP save.
        if time.time() - self._quick_train_start >= 4 * 3600 - 120:
            event(self, "time_limit_stop_after_current_step")
            self._total_step = self.cur_step
        return None

    def maybe_save(self, is_snapshot=False):
        return maybe_save_then_validate(self, origin_save, is_snapshot=is_snapshot)

    fit._quick_wrapped = True
    Trainer.fit = fit
    Trainer._finalize_dcp_save = finalize
    Trainer._log_step = log_step
    Trainer._maybe_save_hf = maybe_hf
    Trainer._maybe_save = maybe_save
