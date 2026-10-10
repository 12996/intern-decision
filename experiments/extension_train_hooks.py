"""Restore first, then train; historical confirmation is not a startup gate."""

import json
import math
import os
from pathlib import Path
import time

from decision_scorer import DecisionScorer
from extension_state import source_info, reuse_proxy, fork_schedule, MANIFEST_SHA
from frozen_visual_cache import FrozenVisualCache
from quick_train_hooks import (atomic_json, event, check_disk, model_digest, evaluate,
    select_best, maybe_save_with_space, verify_dcp, safe_remove, validation_state)
from quick_validation import digest
from resume_state import restore_resume
from swap_restore_fix import assert_swap_param_groups
from train_update_probe import LanguageUpdateProbe


def install():
    from xtuner.v1.train.trainer import Trainer
    if getattr(Trainer.fit, "_extension_wrapped", False):
        return
    original_fit, original_log = Trainer.fit, Trainer._log_step
    original_save, original_finalize = Trainer._maybe_save, Trainer._finalize_dcp_save

    def fit(self, *args, **kwargs):
        if self.world_size != 1 or self.cur_step != 0:
            raise ValueError("E06 requires an isolated single-rank run")
        out = Path(os.environ["QUICK_OUT"])
        root = Path(os.environ["DECISION_ROOT"])
        source = source_info(root)
        manifest_path = Path(os.environ["QUICK_MANIFEST"])
        if digest(manifest_path) != MANIFEST_SHA:
            raise ValueError("Proxy manifest drift")
        self._quick_manifest = json.loads(manifest_path.read_text())
        reference_path = out / "resume_reference.json"
        if digest(reference_path) != os.environ["QUICK_RESUME_REFERENCE_SHA256"]:
            raise ValueError("CPU reference drift")
        reference = json.loads(reference_path.read_text())
        frozen_before = model_digest(self._engine.model, ["vision_tower", "multi_modal_projector"])
        proof = restore_resume(self, source, reference)
        weights = model_digest(self._engine.model, ["language_model", "vision_tower", "multi_modal_projector"])
        if any(weights[k] != v for k, v in frozen_before.items()):
            raise ValueError("Restore changed frozen parameters")
        proof.update(weights=weights, work_dir=str(self.work_dir))
        atomic_json(out / "initial_restore.json", proof)
        atomic_json(out / "last.json", verify_dcp(source["checkpoint"], 100))
        event(self, "post_update_resume_exact", **proof)
        schedule = fork_schedule(self)
        atomic_json(out / "schedule_fork.json", schedule)
        event(self, "schedule_fork", **schedule)
        self._quick_frozen_hashes = frozen_before
        provenance = {"frozen_parameters": frozen_before,
            "model_revision": "8797836c65fc91a2435b1fb6850b5f0aabd75cc3",
            "manifest": MANIFEST_SHA, "scorer_code": digest(Path(__file__).with_name("decision_scorer.py"))}
        self._quick_cache = FrozenVisualCache(root / "outputs/goal-20261003-quick-v1/visual-cache", provenance, read_only=True)
        self._quick_cache.validated = True
        self._quick_scorer = DecisionScorer(os.environ["MODEL_PATH"], existing_model=self._engine.model)
        self._quick_baseline, starting_proxy, self._quick_best = reuse_proxy(root, out, self._quick_manifest)
        # A single cached sentinel checks this GPU/runtime without re-scoring160pairs.
        pair = self._quick_manifest["pairs"][0]
        old = json.loads((out / "scores-step-100.jsonl").read_text().splitlines()[0])
        with validation_state(self._engine.model):
            replay = self._quick_scorer.forward(self._quick_scorer.prepare(pair), visual_cache=self._quick_cache)
        raw_error = max(abs(replay["raw_logits"][k] - old["raw_logits"][k]) for k in ["yes", "no"])
        if not replay["visual_cache_hit"] or raw_error > .001 or abs(replay["score"] - old["score"]) > .002:
            raise ValueError("Step100 cached sentinel parity failed")
        atomic_json(out / "start_proxy_parity.json", {"pass": True, "raw_error": raw_error,
            "margin_error": abs(replay["score"] - old["score"]), "reused_proxy": starting_proxy["macro_mrr5"]})
        atomic_json(out / "proxy_best.json", self._quick_best)
        self._quick_update_probe = LanguageUpdateProbe(self._engine.model.language_model,
            lambda: model_digest(self._engine.model, ["language_model"])["language_model"], weights["language_model"])
        self._quick_start_step = 100
        self._quick_train_start = time.time()
        atomic_json(out / "training_started.json", {"epoch": self._quick_train_start,
            "trainer_pid": os.getpid(), "work_dir": str(self.work_dir), "exp_dir": str(self.exp_dir),
            "resume_step": 100, "remaining_updates": 400, "deadline_epoch": float(os.environ["QUICK_GLOBAL_DEADLINE"])})
        event(self, "optimizer_phase_start", total_step=500, resume_step=100, remaining_updates=400)
        result = original_fit(self, *args, **kwargs)
        selected = self._quick_best
        self._engine.model.from_hf(selected["weights"], strict=True)
        reference_score = json.loads((out / f"scores-step-{selected['step']}.jsonl").read_text().splitlines()[0])
        with validation_state(self._engine.model):
            replay = self._quick_scorer.forward(self._quick_scorer.prepare(pair), visual_cache=self._quick_cache)
        raw_error = max(abs(replay["raw_logits"][k] - reference_score["raw_logits"][k]) for k in ["yes", "no"])
        margin_error = abs(replay["score"] - reference_score["score"])
        passed = raw_error <= .001 and margin_error <= .002
        atomic_json(out / "best_export_reload_parity.json", {"pass": passed, "best": selected,
            "raw_error": raw_error, "margin_error": margin_error})
        if not passed:
            raise ValueError("Best HF reload parity failed")
        event(self, "best_export_reload_parity_pass", raw_error=raw_error, margin_error=margin_error)
        return result

    def log_step(self, *args, **kwargs):
        result = original_log(self, *args, **kwargs)
        assert_swap_param_groups(self._engine.optimizer)
        row = json.loads((Path(self.work_dir) / "scaleup_steps.jsonl").read_text().splitlines()[-1])
        if not row["loss_finite"] or not math.isfinite(row["grad_norm"]):
            raise ValueError("Nonfinite training step")
        if row["vision_max_abs_delta"] != 0 or row["projector_max_abs_delta"] != 0:
            raise ValueError("Frozen parameter sample changed")
        updates = self.cur_step - 100
        proof = self._quick_update_probe.check(updates, verify_full=updates == 5)
        proof.update(step=self.cur_step, updates_since_restore=updates)
        event(self, "language_update_probe", **proof)
        if updates >= 3 and not proof["update_detected"]:
            raise ValueError("Language parameters unchanged")
        if updates == 5:
            if time.time() >= float(os.environ["QUICK_SMOKE_DEADLINE"]):
                raise TimeoutError("Five-update smoke exceeded30minutes")
            if model_digest(self._engine.model, ["vision_tower", "multi_modal_projector"]) != self._quick_frozen_hashes:
                raise ValueError("Frozen full hash changed")
            atomic_json(Path(os.environ["QUICK_OUT"]) / "smoke_pass.json", {"pass": True,
                "step": 105, "resume_step": 100, "total_step": 500, "updates_since_restore": 5,
                "finite_loss_and_gradient": True, "frozen_visual_unchanged": True,
                "work_dir": str(self.work_dir), "at": time.time(), "update_proof": proof})
            event(self, "five_step_smoke_pass")
        check_disk(self.work_dir)
        return result

    def finalize(self, *args, **kwargs):
        result = original_finalize(self, *args, **kwargs)
        record = verify_dcp(Path(kwargs["checkpoint_path"]), self.cur_step)
        atomic_json(Path(os.environ["QUICK_OUT"]) / "last.json", record)
        listed = self.meta.latest_exp.checkpoint_list
        for previous in list(listed[:-1]):
            safe_remove(previous, self.checkpoint_dir, "ckpt-step-")
            listed.remove(previous)
        atomic_json(Path(self.work_dir) / self._META_PATH, json.loads(self.meta.model_dump_json()))
        event(self, "last_saved_verified", **record)
        return result

    def maybe_hf(self):
        if time.time() >= float(os.environ["QUICK_GLOBAL_DEADLINE"]) - 240:
            self._total_step = self.cur_step
            event(self, "time_limit_finish_current_step")

    def maybe_save(self, is_snapshot=False):
        saved = maybe_save_with_space(self, original_save, is_snapshot=is_snapshot)
        due = self.cur_step > 100 and (self.cur_step % 50 == 0 or self.cur_step == self.total_step)
        if not is_snapshot and due and getattr(self, "_quick_validated_step", None) != self.cur_step:
            event(self, "validation_after_resume_save", full_resume_saved=bool(saved))
            select_best(self, evaluate(self, self.cur_step))
            self._quick_validated_step = self.cur_step
        return saved

    fit._extension_wrapped = True
    Trainer.fit, Trainer._log_step = fit, log_step
    Trainer._finalize_dcp_save = finalize
    Trainer._maybe_save_hf, Trainer._maybe_save = maybe_hf, maybe_save
