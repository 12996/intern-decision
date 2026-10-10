"""Per-step training record for the 1000-step Charades run.

This does not change the loss, optimizer, or sample order. It records the
batch metadata popped before the model forward, then logs it after the
optimizer step so the ids match the update that just happened.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from pathlib import Path

import numpy
import torch

_INSTALLED = False


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
    }


def _restore_rng(payload: dict) -> None:
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])
    random.setstate(payload["python"])
    numpy.random.set_state(payload["numpy"])


def _as_float(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
        if value.numel() != 1:
            value = value.mean()
        return float(value)
    return float(value)


def _model_of(trainer):
    engine = getattr(trainer, "_engine", None)
    return None if engine is None else getattr(engine, "model", None)


def _freeze_flags(model) -> dict:
    report = {}
    for name in ("language_model", "vision_tower", "multi_modal_projector"):
        module = getattr(model, name, None)
        trainable = frozen = grads = 0
        if module is not None:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    trainable += 1
                    if parameter.grad is not None:
                        grads += 1
                else:
                    frozen += 1
                    if parameter.grad is not None:
                        grads += 1
        report[name] = {"trainable": trainable, "frozen": frozen, "grads_present": grads}
    return report


def _slice(module):
    if module is None:
        return None
    for parameter in module.parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        if local.numel() < 1024:
            continue
        return local.detach().flatten()[:1024].float().cpu().clone()
    return None


def _capture(model):
    return {
        "language": _slice(getattr(model, "language_model", None)),
        "vision": _slice(getattr(model, "vision_tower", None)),
        "projector": _slice(getattr(model, "multi_modal_projector", None)),
    }


def _delta(before, after):
    if before is None or after is None:
        return None
    return float((after - before).abs().max())


def _scheduler_probe(scheduler) -> dict | None:
    if scheduler is None:
        return None
    children = []
    for child in list(getattr(scheduler, "_schedulers", []) or []):
        children.append(
            {
                "cls": type(child).__name__,
                "last_epoch": getattr(child, "last_epoch", None),
                "total_iters": getattr(child, "total_iters", None),
                "T_max": getattr(child, "T_max", None),
            }
        )
    milestones = getattr(scheduler, "_milestones", None)
    return {
        "cls": type(scheduler).__name__,
        "last_epoch": getattr(scheduler, "last_epoch", None),
        "milestones": None if milestones is None else [int(item) for item in milestones],
        "children": children,
    }


def _optimizer_probe(trainer) -> dict | None:
    engine = getattr(trainer, "_engine", None)
    optimizer = None if engine is None else getattr(engine, "optimizer", None)
    if optimizer is None:
        return None
    for group in optimizer.param_groups:
        for parameter in group.get("params", []):
            state = optimizer.state.get(parameter)
            if not state or "exp_avg" not in state:
                continue
            moment = state["exp_avg"]
            local = moment.to_local() if hasattr(moment, "to_local") else moment
            sample = local.detach().float().flatten()[:1024]
            return {
                "device": str(moment.device),
                "dtype": str(moment.dtype),
                "step": _as_float(state.get("step")),
                "exp_avg_sample_sum": float(sample.sum()),
                "numel": int(local.numel()),
            }
    return {"empty": True}


def _offload_swap_moments(trainer) -> int:
    engine = getattr(trainer, "_engine", None)
    optimizer = None if engine is None else getattr(engine, "optimizer", None)
    cpu_map = None if optimizer is None else getattr(optimizer, "_param_to_cpu_states_map", None)
    if optimizer is None or not isinstance(cpu_map, dict):
        return 0
    moved = 0
    for param, cpu_state in cpu_map.items():
        loaded = optimizer.state.get(param)
        if not isinstance(loaded, dict):
            continue
        for key, value in list(loaded.items()):
            if not torch.is_tensor(value):
                continue
            local = value.to_local() if hasattr(value, "to_local") else value
            if local.device.type != "cpu":
                local = local.detach().to(device="cpu", non_blocking=True).pin_memory()
                moved += int(local.numel())
            if isinstance(cpu_state, dict):
                cpu_state[key] = local
            loaded[key] = local
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def _data_probe(trainer) -> dict | None:
    loader = getattr(trainer, "_dataloader", None)
    if loader is None or not hasattr(loader, "get_state_dict"):
        return None
    state = loader.get_state_dict()
    if not isinstance(state, dict):
        return {"type": type(state).__name__}
    integers = {str(key): int(value) for key, value in state.items() if isinstance(value, int)}
    return {"keys": [str(key) for key in list(state)[:20]], "ints": integers}


def _history_source(trainer):
    raw = os.environ.get("SCALEUP_HISTORY_JSONL", "").strip()
    max_raw = os.environ.get("SCALEUP_HISTORY_MAX_STEP", "").strip()
    max_step = int(max_raw) if max_raw else 0
    if raw:
        return Path(raw), max_step
    return Path(trainer.work_dir) / "scaleup_steps.jsonl", max_step


def _iter_history(path: Path, max_step: int):
    """Yield (row, status). Status is ok, future, or malformed.

    A resume reads the original JSONL only up to the saved step. Later rows
    and the truncated crash line stay in that file and are not copied forward.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                yield None, "malformed"
                continue
            step = item.get("step")
            if max_step and isinstance(step, int) and step > max_step:
                yield None, "future"
                continue
            yield item, "ok"


def _event(trainer, name: str, payload: dict) -> None:
    record = {"event": name, "step": int(getattr(trainer, "cur_step", -1)), **payload}
    print("SCALEUP_EVENT " + json.dumps(record, ensure_ascii=False), flush=True)
    path = Path(trainer.work_dir) / "scaleup_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from xtuner.v1.train.trainer import Trainer

    if getattr(Trainer._log_step, "_scaleup_wrapped", False):
        _INSTALLED = True
        return
    origin_prepare = Trainer._prepare_model_input
    origin_log = Trainer._log_step
    origin_fit = Trainer.fit
    origin_finalize = Trainer._finalize_dcp_save
    origin_load = Trainer._load_checkpoint
    origin_hf = Trainer._maybe_save_hf

    def _prepare_model_input(self, data_batch):
        metas = []
        for data in data_batch:
            if isinstance(data, dict):
                metas.extend(data.pop("scaleup", []) or [])
        self._scaleup_batch = metas
        return origin_prepare(self, data_batch)

    def fit(self, *args, **kwargs):
        seen = set()
        queries = set()
        groups = {}
        losses = deque(maxlen=50)
        source, max_step = _history_source(self)
        restored = 0
        last_step = None
        skipped_future = 0
        malformed = 0
        if source is not None and source.is_file() and not source.is_symlink():
            for item, status in _iter_history(source, max_step):
                if status == "malformed":
                    malformed += 1
                    continue
                if status == "future":
                    skipped_future += 1
                    continue
                restored += 1
                last_step = item.get("step")
                for sample in item.get("samples") or []:
                    if not isinstance(sample, dict):
                        continue
                    if sample.get("id"):
                        seen.add(sample["id"])
                    if sample.get("query_id"):
                        queries.add(sample["query_id"])
                for kind, stats in (item.get("groups_cumulative") or {}).items():
                    bucket = groups.setdefault(kind, {"n": 0, "loss_sum": 0.0})
                    bucket["n"] = stats.get("n", bucket["n"])
                    bucket["loss_sum"] = stats.get("loss_sum", bucket["loss_sum"])
                loss = item.get("loss")
                if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
                    losses.append(float(loss))
        self._scaleup_seen = seen
        self._scaleup_queries = queries
        self._scaleup_groups = groups
        self._scaleup_losses = losses
        self._scaleup_recent = []
        if getattr(self, "rank", 0) == 0:
            print(
                "SCALEUP_HISTORY_RESTORED "
                + json.dumps(
                    {
                        "source": None if source is None else str(source),
                        "max_step": max_step,
                        "rows": restored,
                        "last_step": last_step,
                        "skipped_future": skipped_future,
                        "malformed": malformed,
                        "loss_window": len(losses),
                        "unique_pairs": len(seen),
                        "unique_queries": len(queries),
                        "groups": {kind: bucket["n"] for kind, bucket in groups.items()},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        model = _model_of(self)
        self._scaleup_before = _capture(model) if model is not None else None
        flags = _freeze_flags(model) if model is not None else {}
        print("SCALEUP_FREEZE " + json.dumps(flags), flush=True)
        if getattr(self, "rank", 0) == 0:
            (Path(self.work_dir) / "trainer.pid").write_text(str(os.getpid()), encoding="utf-8")
        return origin_fit(self, *args, **kwargs)

    def _log_step(
        self,
        train_step_info,
        training_metrics,
        grad_norm,
        data_time,
        step_time,
        internal_metrics=None,
    ):
        loss = None
        if isinstance(train_step_info, dict):
            # Official logging pops total_loss, so read it before the wrapped call.
            loss = train_step_info.get("total_loss")
            logs = train_step_info.get("logs_info")
            if loss is None and isinstance(logs, dict) and logs:
                loss = next(iter(logs.values()))
        result = origin_log(
            self,
            train_step_info,
            training_metrics,
            grad_norm,
            data_time,
            step_time,
            internal_metrics,
        )
        loss = _as_float(loss)
        samples = list(getattr(self, "_scaleup_batch", []) or [])
        if not samples:
            raise RuntimeError("scaleup metadata missing from the consumed batch")
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            kind = sample.get("pair_type")
            expected = {"positive": "yes", "random": "no", "hard": "no"}.get(kind)
            if expected is not None and sample.get("label") != expected:
                raise RuntimeError(f"pair label mismatch for {sample.get('id')}: {kind} {sample.get('label')}")
            if sample.get("images") not in (None, 8):
                raise RuntimeError(f"image count {sample.get('images')} for {sample.get('id')}")
        ids = [sample.get("id") for sample in samples if isinstance(sample, dict)]
        recent = getattr(self, "_scaleup_recent", [])
        recent.append(tuple(ids))
        self._scaleup_recent = recent[-3:]
        if len(self._scaleup_recent) == 3 and len(set(self._scaleup_recent)) == 1 and ids:
            raise RuntimeError(f"first-batch loop: sample ids repeated {ids}")
        seen = getattr(self, "_scaleup_seen", set())
        queries = getattr(self, "_scaleup_queries", set())
        groups = getattr(self, "_scaleup_groups", {})
        losses = getattr(self, "_scaleup_losses", deque(maxlen=50))
        if loss is not None and math.isfinite(loss):
            losses.append(loss)
        self._scaleup_losses = losses
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            if sample.get("id"):
                seen.add(sample["id"])
            if sample.get("query_id"):
                queries.add(sample["query_id"])
            kind = sample.get("pair_type") or "other"
            bucket = groups.setdefault(kind, {"n": 0, "loss_sum": 0.0})
            bucket["n"] += 1
            if loss is not None and math.isfinite(loss):
                bucket["loss_sum"] += loss
        model = _model_of(self)
        current = _capture(model) if model is not None else None
        before = getattr(self, "_scaleup_before", None) or {}
        flags = _freeze_flags(model) if model is not None and self.cur_step % 100 == 0 else None
        img_tokens = text_tokens = seqlen_tokens = None
        if isinstance(training_metrics, dict):
            img_tokens = training_metrics.get("local_step_consumed_img_tokens")
            text_tokens = training_metrics.get("local_step_consumed_tokens")
            seqlen_tokens = training_metrics.get("local_step_seqlen_tokens")
        lr = self._lr_scheduler.get_last_lr()[0]
        record = {
            "step": int(self.cur_step),
            "total_step": int(self.total_step),
            "aligned": "after_optimizer",
            "samples": samples,
            "loss": loss,
            "loss_finite": loss is not None and math.isfinite(loss),
            "lr": lr,
            "grad_norm": _as_float(grad_norm),
            "data_time": _as_float(data_time),
            "step_time": _as_float(step_time),
            "text_tokens": _as_float(text_tokens),
            "img_tokens": _as_float(img_tokens),
            "seqlen_tokens": _as_float(seqlen_tokens),
            "rolling_mean_50": (sum(losses) / len(losses)) if losses else None,
            "unique_pairs": len(seen),
            "unique_queries": len(queries),
            "groups_cumulative": groups,
            "language_max_abs_delta": _delta(before.get("language"), None if current is None else current.get("language")),
            "vision_max_abs_delta": _delta(before.get("vision"), None if current is None else current.get("vision")),
            "projector_max_abs_delta": _delta(before.get("projector"), None if current is None else current.get("projector")),
            "freeze_flags": flags,
            "scheduler": _scheduler_probe(getattr(self, "_lr_scheduler", None)),
            "optimizer": _optimizer_probe(self),
            "data_state": _data_probe(self),
        }
        line = "SCALEUP_STEP " + json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        work = Path(self.work_dir)
        with (work / "scaleup_steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        with (work / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        return result

    def _finalize_dcp_save(self, *args, **kwargs):
        result = origin_finalize(self, *args, **kwargs)
        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path is None and args:
            checkpoint_path = args[0]
        if checkpoint_path is not None:
            torch.save(_rng_state(), Path(checkpoint_path) / f"rng-rank{self.rank}.pt")
            if self.rank == 0:
                print(f"SCALEUP_RNG_SAVED {checkpoint_path}", flush=True)
                _event(
                    self,
                    "dcp_saved",
                    {
                        "checkpoint": str(checkpoint_path),
                        "total_step": int(self.total_step),
                        "scheduler": _scheduler_probe(getattr(self, "_lr_scheduler", None)),
                        "optimizer": _optimizer_probe(self),
                        "data_state": _data_probe(self),
                    },
                )
        return result

    def _load_checkpoint(self):
        result = origin_load(self)
        # DCP reloads SwapAdamW moments onto CUDA. The optimizer step reads the
        # CPU map and copies one parameter to the GPU at a time. Rebind the
        # loaded values onto that CPU map before the next forward.
        moved = _offload_swap_moments(self)
        if self.rank == 0 and moved:
            print(f"SCALEUP_SWAP_OFFLOAD numel={moved}", flush=True)
        load_cfg = self._load_checkpoint_cfg
        resume_from = getattr(load_cfg, "checkpoint_path", None)
        if resume_from:
            path = Path(resume_from) / f"rng-rank{self.rank}.pt"
            if path.is_file():
                _restore_rng(torch.load(path, map_location="cpu", weights_only=False))
                if self.rank == 0:
                    print(f"SCALEUP_RNG_RESTORED {path}", flush=True)
            elif self.rank == 0:
                print(f"SCALEUP_RNG_ABSENT {path}", flush=True)
        if self.rank == 0:
            _event(
                self,
                "resumed",
                {
                    "checkpoint": None if resume_from is None else str(resume_from),
                    "total_step": int(self.total_step),
                    "scheduler": _scheduler_probe(getattr(self, "_lr_scheduler", None)),
                    "optimizer": _optimizer_probe(self),
                    "data_state": _data_probe(self),
                },
            )
        return result

    def _maybe_save_hf(self):
        # Storage policy only: replay would otherwise write a second hf-800.
        raw = os.environ.get("SKIP_HF_STEPS", "")
        skip = set()
        for part in raw.replace(",", " ").split():
            if part.strip():
                skip.add(int(part))
        if int(getattr(self, "cur_step", -1)) in skip:
            if getattr(self, "rank", 0) == 0:
                print(
                    f"SKIP_HF_SAVE step={int(self.cur_step)} storage_policy=replay_redundant",
                    flush=True,
                )
                _event(self, "skip_hf_save", {"storage_policy": "replay_redundant"})
            return None
        return origin_hf(self)

    _prepare_model_input._scaleup_wrapped = True
    _log_step._scaleup_wrapped = True
    Trainer._prepare_model_input = _prepare_model_input
    Trainer._log_step = _log_step
    Trainer.fit = fit
    Trainer._finalize_dcp_save = _finalize_dcp_save
    Trainer._load_checkpoint = _load_checkpoint
    Trainer._maybe_save_hf = _maybe_save_hf
    _INSTALLED = True
