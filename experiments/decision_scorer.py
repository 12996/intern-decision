"""Official XTuner decision logits, with an equivalent one-position output head.

No training patch: reference mode retains masked full-vocabulary CE; ranking
mode has no targets/loss and selects only the shifted decision hidden state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(os.environ.get("DECISION_ROOT", "/home/zhaopu/intern-decision"))
REPO = ROOT / "Intern-Decision"
os.environ.setdefault("INTERN_DECISION_REPO", str(REPO))
os.environ.setdefault("MEDIA_ROOT", str(ROOT / "frames/id_pilot_frames_8"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import torch
from src.inputs.masked import DECISION_TOKEN, MaskedQwen3VLTokenizeFunction, masked_qwen3_vl_collator
from src.inputs.schema import _options, compile_row
from transformers import AutoTokenizer
from xtuner.v1.loss import CELossConfig

from diagnose_loss_contrast import load_model


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def local_tensor(tensor):
    if hasattr(tensor, "to_local"):
        local = tensor.to_local()
        return local if local.shape == tensor.shape else tensor.full_tensor()
    return tensor


def restored_results(previous, checks, model):
    latest_checks = {(r["model"], r["pair_id"], r["mode"], r.get("projection")): r["pass"]
                     for r in checks if "mode" in r}
    return {(r["pair_id"], r["mode"]): r for r in previous if r["model"] == model
            and (r["mode"] == "reference_ce" or
                 (r.get("projection") == "reference_post_fsdp_fp32" and
                  latest_checks.get((r["model"], r["pair_id"], r["mode"], r.get("projection"))) is True))}


def segmented_vision_eager(q, k, v, cu_seqlens_q, softmax_scale, causal=False, dropout_p=0.0, **kwargs):
    """Skip cross-image entries which the reference always masks to -infinity.

    Keep eager BF16 matmul and FP32 softmax, not a new fused attention backend.
    This inference-only optimization is enabled only in the explicit benchmark.
    """
    if causal or dropout_p != 0 or q.shape[1] != k.shape[1]:
        raise ValueError("Segmented visual optimization requires equal-head, noncausal, dropout-free attention")
    boundaries = cu_seqlens_q.tolist()
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        logits = torch.matmul(q[:, :, start:end], k[:, :, start:end].transpose(2, 3)) * softmax_scale
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        pieces.append(torch.matmul(probabilities, v[:, :, start:end]))
    return {"raw_output": torch.cat(pieces, dim=2).transpose(1, 2).contiguous(), "attn_logits": None}


class DecisionScorer:
    def __init__(self, checkpoint, existing_model=None):
        self.checkpoint = str(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.tokenize = MaskedQwen3VLTokenizeFunction(
            self.tokenizer, processor_path=str(checkpoint), media_root=os.environ["MEDIA_ROOT"],
            anno_name="goal-score.jsonl", max_length=16384, max_pixels=16777216,
            chat_template="qwen3.5-vl",
        )
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        self.model = existing_model if existing_model is not None else load_model(str(checkpoint))
        self.head = self.model.language_model.lm_head
        self.freeze = {
            name: {"parameters": sum(p.numel() for p in getattr(self.model, name).parameters()),
                   "trainable": sum(p.numel() for p in getattr(self.model, name).parameters() if p.requires_grad)}
            for name in ["vision_tower", "multi_modal_projector", "language_model"]
        }
        assert self.freeze["vision_tower"]["trainable"] == 0
        assert self.freeze["multi_modal_projector"]["trainable"] == 0

    def prepare(self, pair, reference=False):
        started = time.perf_counter()
        row = {
            "state": {"query": pair["query"]},
            "images": [f"{pair['video_id']}/frame_{i:02d}.png" for i in range(8)],
            "questions": {"noul": {"type": "noul", "instructions": "Do these ordered frames show the activity described by the query?"}},
        }
        if reference:
            row["targets"] = {"noul": {"label": pair["label"]}}
        compiled = compile_row(row, include_targets=reference)
        symbol_ids = {}
        symbols = {}
        for symbol, (value, _) in zip(compiled.symbols["noul"], _options(row["questions"]["noul"])):
            encoded = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("Answer symbol is not single-token")
            symbol_ids[value] = encoded[0]
            symbols[value] = symbol
        item = self.tokenize(row, include_targets=reference)
        positions = [i - 1 for i, token in enumerate(item["input_ids"]) if token == self.marker_id]
        if len(positions) != 1 or positions[0] < 0:
            raise ValueError("Expected one shifted decision position")
        decoded = self.tokenizer.decode(item["input_ids"]).replace(pair["query"], "")
        if pair["video_id"] in decoded or pair["id"] in decoded or "frame_" in decoded:
            raise ValueError("Identifier/path leaked into model text")
        position = positions[0]
        assert item["input_ids"][position + 1] == self.marker_id
        if reference:
            assert sum(label != -100 for label in item["labels"]) == 1
        else:
            assert all(label == -100 for label in item["labels"])
        batch = masked_qwen3_vl_collator([[item]], item["num_tokens"], self.tokenizer.pad_token_id, pack_to_max_length=False)[0]
        ctx = batch["seq_ctx"]
        if ctx.pixel_values is None or ctx.pixel_values.numel() == 0 or len(ctx.image_grid_thw) != 8:
            raise ValueError("Missing eight real image grids/pixels")
        identity = {
            "tokens_sha256": tensor_hash(ctx.input_ids), "positions_sha256": tensor_hash(ctx.position_ids),
            "pixels_sha256": tensor_hash(ctx.pixel_values), "image_grid_thw": ctx.image_grid_thw.tolist(),
            "num_tokens": item["num_tokens"], "decision_position": position,
            "frame_sha256": [hashlib.sha256((Path(os.environ["MEDIA_ROOT"]) / name).read_bytes()).hexdigest() for name in row["images"]],
            "symbols": symbols, "symbol_ids": symbol_ids, "targets_in_scoring_input": False,
        }
        prepare_s = time.perf_counter() - started
        transfer_start = time.perf_counter()
        ctx.to("cuda")
        ctx.pixel_values = ctx.pixel_values.cuda()
        ctx.image_grid_thw = ctx.image_grid_thw.cuda()
        torch.cuda.synchronize()
        transfer_s = time.perf_counter() - transfer_start
        # Qwen3VLTextDense's loss_ctx=None directly F.linear's a still-sharded
        # head. The non-None container calls the FSDP-aware LMHead with no loss.
        loss_ctx = {"lm": None}
        if reference:
            labels = batch["shifted_labels"]
            assert int(labels[0, position]) == symbol_ids[pair["label"]]
            cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="token")
            lm_loss = cfg.build({"shifted_labels": labels})
            lm_loss = type(lm_loss).build_batches([lm_loss], cu_seq_lens_list=[ctx.cu_seq_lens_k])[0]
            loss_ctx = {"lm": lm_loss}
        return ctx, loss_ctx, identity, {"read_processor_hash_s": prepare_s, "transfer_sync_s": transfer_s}

    def forward(self, prepared, reference=False, segmented=False, detail_profile=False,
                visual_cache=None, force_uncached=False):
        hooks, old_attention, original_methods = [], [], []
        try:
            with torch.no_grad():
                binding = (visual_cache.bind(self.model, prepared[2], force_uncached)
                           if visual_cache is not None else nullcontext())
                with binding:
                    return self._forward(prepared, reference, segmented, detail_profile,
                                         hooks, old_attention, original_methods, visual_cache)
        finally:
            for hook in hooks:
                hook.remove()
            for module, implementation in old_attention:
                module.attn_impl_func = implementation
            for name, method in original_methods:
                setattr(self.model.vision_tower, name, method)

    def _forward(self, prepared, reference, segmented, detail_profile,
                 hooks, old_attention, original_methods, visual_cache=None):
        ctx, loss_ctx, identity, timing = prepared
        position = identity["decision_position"]
        events, captured, recorded = {}, {}, set()
        detailed_events = {}
        if segmented:
            for module in self.model.vision_tower.modules():
                if module.__class__.__name__ == "Qwen3VLVisionAttention":
                    old_attention.append((module, module.attn_impl_func))
                    module.attn_impl_func = segmented_vision_eager
        for name, module in [("vision", self.model.vision_tower), ("projector", self.model.multi_modal_projector),
                             ("language_including_head", self.model.language_model), ("ce_head", self.head)]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            events[name] = (start, end)
            def begin_event(_m, _a, e=start, key=name):
                recorded.add(key)
                e.record()
            hooks.append(module.register_forward_pre_hook(begin_event))
            hooks.append(module.register_forward_hook(lambda _m, _a, _o, e=end: e.record()))
        if detail_profile:
            for name in ["fast_pos_embed_interpolate", "rot_pos_emb"]:
                method = getattr(self.model.vision_tower, name)
                original_methods.append((name, method))
                def timed_method(*args, _name=name, _method=method, **kwargs):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    detailed_events.setdefault(_name, []).append((start, end))
                    start.record()
                    result = _method(*args, **kwargs)
                    end.record()
                    return result
                setattr(self.model.vision_tower, name, timed_method)
            for name, module in self.model.vision_tower.named_modules():
                category = "patch_embed" if name == "patch_embed" else {
                    "Qwen3VLVisionAttention": "vision_attention", "Qwen3VLVisionMLP": "vision_mlp",
                    "LayerNorm": "vision_norm",
                }.get(module.__class__.__name__)
                if not category:
                    continue
                stack = []
                def begin(_module, _args, _category=category, _stack=stack):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    detailed_events.setdefault(_category, []).append((start, end))
                    _stack.append(end)
                    start.record()
                def finish(_module, _args, _output, _stack=stack):
                    _stack.pop().record()
                hooks.append(module.register_forward_pre_hook(begin))
                hooks.append(module.register_forward_hook(finish))
        def capture(module, args, output):
            hidden = local_tensor(args[0])
            index = position if reference else 0
            vector = hidden[0, index] if hidden.ndim == 3 else hidden[index]
            # The historical hook runs after FSDP restores FP32 master weights.
            # Native BF16 inference rounds these logits; preserve reference math.
            weight = local_tensor(module.weight)
            if weight.dtype != torch.float32:
                raise ValueError(f"Reference post-FSDP head is not FP32: {weight.dtype}")
            bias = local_tensor(module.bias) if getattr(module, "bias", None) is not None else None
            captured["row"] = torch.nn.functional.linear(vector.unsqueeze(0).to(weight.dtype), weight, bias)[0].detach().float()
            captured["hidden_dtype"] = str(vector.dtype)
            captured["weight_dtype"] = str(weight.dtype)
        hooks.append(self.head.register_forward_hook(capture))
        if not reference:
            def select_position(_module, _args, output):
                return output[:, position:position + 1, :]
            hooks.append(self.model.language_model.norm.register_forward_hook(select_position))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(seq_ctx=ctx, loss_ctx=loss_ctx)
        torch.cuda.synchronize()
        timing["forward_sync_s"] = time.perf_counter() - started
        row = captured["row"]
        logits = {label: float(row[index].cpu()) for label, index in identity["symbol_ids"].items()}
        timing["cuda_ms"] = {name: start.elapsed_time(end) if name in recorded else 0.0
                             for name, (start, end) in events.items()}
        cache_hit = visual_cache is not None and visual_cache.last_hit
        if detailed_events:
            timing["vision_detail_cuda_ms"] = {name: sum(a.elapsed_time(b) for a, b in pairs) for name, pairs in detailed_events.items()}
        timing["language_without_head_cuda_ms"] = timing["cuda_ms"]["language_including_head"] - timing["cuda_ms"]["ce_head"]
        if any(not math.isfinite(value) for value in logits.values()):
            raise ValueError("Nonfinite logits")
        if not cache_hit and (timing["cuda_ms"]["vision"] <= 0 or timing["cuda_ms"]["projector"] <= 0):
            raise ValueError("Vision/projector did not execute")
        score = logits["yes"] - logits["no"]
        result = {
            "mode": "reference_ce" if reference else ("segmented_decision_position" if segmented else "decision_position"), "raw_logits": logits, "score": score,
            "projection": "reference_post_fsdp_fp32", "head_weight_dtype": captured["weight_dtype"], "decision_hidden_dtype": captured["hidden_dtype"],
            "p_yes_candidate_softmax_not_calibrated": float(torch.softmax(torch.tensor([logits["no"], logits["yes"]]), 0)[1]),
            "input": identity, "timing": timing, "freeze": self.freeze,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "visual_cache_hit": cache_hit,
            "visual_cache_key": visual_cache.last_key if visual_cache is not None else None,
            "decision_full_vocab_logsumexp_fp32": float(torch.logsumexp(row, dim=0).cpu()),
        }
        if reference:
            result["reference_full_vocabulary_ce"] = float(output.loss.detach().float().cpu())
        return result


def append_record(path, payload):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def benchmark():
    out = Path(os.environ["GOAL_OUT"])
    ledger = out / "forward_attempts.jsonl"
    deadline = float(os.environ["GOAL_DEADLINE"])
    if time.time() >= deadline:
        raise TimeoutError("Budget exhausted before model load")
    scorer = DecisionScorer(os.environ["CHECKPOINT"])
    frozen = json.loads((out / "frozen_pairs.json").read_text())
    pairs = frozen["pairs"][:3]
    if os.environ["MODEL_NAME"] == "hf800":
        pairs = pairs[:1]
    protocol = json.loads((out / "evaluation_protocol.json").read_text())
    score_path = out / "benchmark_scores.jsonl"
    previous = [json.loads(line) for line in score_path.read_text().splitlines()] if score_path.exists() else []
    checks_path = out / "equivalence.jsonl"
    checks = [json.loads(line) for line in checks_path.read_text().splitlines()] if checks_path.exists() else []
    results = restored_results(previous, checks, os.environ["MODEL_NAME"])
    phases = ["reference_ce", "decision_position", "segmented_decision_position"] if os.environ["MODEL_NAME"] == "released" else ["segmented_decision_position"]
    for mode in phases:
        for pair in pairs:
            reference = mode == "reference_ce"
            segmented = mode == "segmented_decision_position"
            key = pair["id"], mode
            if key in results:
                continue
            attempts = len(ledger.read_text().splitlines()) if ledger.exists() else 0
            if time.time() >= deadline or attempts >= 12:
                raise TimeoutError("First-check forward/time budget exhausted")
            prepared = scorer.prepare(pair, reference=reference)
            existing_reference = results.get((pair["id"], "reference_ce"))
            if existing_reference and existing_reference["input"] != prepared[2]:
                raise ValueError("Reference/ranking inputs differ")
            append_record(ledger, {"model": os.environ["MODEL_NAME"], "pair": pair["id"], "reference": reference, "at": time.time()})
            result = scorer.forward(prepared, reference=reference, segmented=segmented)
            result.update(model=os.environ["MODEL_NAME"], pair_id=pair["id"], video_id=pair["video_id"])
            append_record(score_path, result)
            print(json.dumps({"pair": pair["id"], "mode": result["mode"], "score": result["score"], "timing": result["timing"]}), flush=True)
            results[key] = result
            del prepared
            if not reference:
                ref = results.get((pair["id"], "reference_ce"))
                if ref is not None:
                    logit_error = max(abs(ref["raw_logits"][k] - result["raw_logits"][k]) for k in ["no", "yes"])
                    score_error = abs(ref["score"] - result["score"])
                    passed = logit_error <= protocol["logit_abs_tolerance"] and score_error <= protocol["score_abs_tolerance"]
                else:
                    historical = [json.loads(line) for line in (out / "historical_margin_scores.jsonl").read_text().splitlines()]
                    ref = next(r for r in historical if r["model"] == os.environ["MODEL_NAME"] and r["id"] == pair["id"])
                    logit_error = None
                    score_error = abs(ref["yes_no_margin_logit_yes_minus_no"] - result["score"])
                    passed = score_error <= protocol["score_abs_tolerance"] and ref["num_tokens"] == result["input"]["num_tokens"]
                append_record(out / "equivalence.jsonl", {"model": os.environ["MODEL_NAME"], "pair_id": pair["id"], "mode": mode,
                                                        "projection": result["projection"], "logit_abs_error": logit_error, "score_abs_error": score_error, "pass": passed,
                                                        "reference_scope": "current raw logits" if logit_error is not None else "historical margin only"})
                if not passed:
                    raise ValueError("Prospective logit tolerance failed; do not expand it")
    import torch.distributed as dist
    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark()
