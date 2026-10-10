"""Compare the released model and the step-20 checkpoint on one fixed set.

The set is a prefix of the immutable pilot manifest, plus held-out validation
positives and other already-extracted validation videos as negative candidates.
Full-vocabulary CE is the training loss. yes/no margin and renormalized
probabilities use only the two decision symbols and are labeled separately.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

REPO = Path(os.environ["INTERN_DECISION_REPO"])
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from src.inputs.masked import MaskedQwen3VLTokenizeFunction, masked_qwen3_vl_collator
from src.inputs.schema import _options, compile_row
from src.model.size_config import decision_config_from_hf
from xtuner.v1.config import FSDPConfig
from xtuner.v1.loss import CELossConfig

INSTRUCTIONS = "Do these ordered frames show the activity described by the query?"
FRAMES = Path(os.environ["MEDIA_ROOT"])


def _visible_text(content) -> str:
    """Text the chat template can turn into tokens. Image URLs stay loader paths."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    chunks = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def row_of(example_id: str, query: str, video_id: str, label: str, pair_type: str, control: str) -> dict:
    images = [f"{video_id}/frame_{index:02d}.png" for index in range(8)]
    missing = [name for name in images if not (FRAMES / name).is_file()]
    if missing:
        raise SystemExit(f"missing frame {missing[0]}")
    return {
        "id": example_id,
        "pair_type": pair_type,
        "control": control,
        "video_id": video_id,
        "state": {"query": query},
        "images": images,
        "questions": {"noul": {"type": "noul", "instructions": INSTRUCTIONS}},
        "targets": {"noul": {"label": label}},
    }


def fixed_rows() -> list[dict]:
    payload = json.loads(Path(os.environ["PILOT_MANIFEST"]).read_text(encoding="utf-8"))
    rows = []
    for example in payload["train"][:4]:
        cap_id = example["cap_id"]
        query = example["query"]
        rows.append(row_of(f"{cap_id}#pos", query, example["positive_video"], "yes", "positive", "matched"))
        rows.append(row_of(f"{cap_id}#rand", query, example["random_negative_video"], "no", "random", "matched"))
        rows.append(row_of(f"{cap_id}#hard", query, example["hard_negative_video"], "no", "hard", "matched"))
    val = payload["val"][:4]
    for index, example in enumerate(val):
        rows.append(
            row_of(
                f"{example['cap_id']}#valpos",
                example["query"],
                example["positive_video"],
                "yes",
                "val_positive",
                "matched",
            )
        )
        negative = val[(index + 1) % len(val)]
        if negative["positive_video"] == example["positive_video"]:
            raise SystemExit("val negative candidate collided with the positive")
        rows.append(
            row_of(
                f"{example['cap_id']}#valneg",
                example["query"],
                negative["positive_video"],
                "no",
                "val_negative",
                "matched",
            )
        )
    for example in payload["train"][:2]:
        rows.append(
            row_of(
                f"{example['cap_id']}#mismatch",
                example["query"],
                example["hard_negative_video"],
                "yes",
                "positive_query_mismatched_frames",
                "mismatched",
            )
        )
    return rows


_FRACTION_SET = False


def load_model(checkpoint: str):
    global _FRACTION_SET
    fraction = float(os.environ.get("MEM_FRACTION", "0.85"))
    if not _FRACTION_SET:
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        _FRACTION_SET = True
    model_cfg = decision_config_from_hf(
        checkpoint,
        only_llm_forward=False,
        freeze_vision=True,
        freeze_projector=True,
        freeze_language=False,
        compile_cfg=False,
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    with torch.device("meta"):
        model = model_cfg.build()
    model = model.fully_shard(
        FSDPConfig(
            recompute_ratio=1.0,
            vision_recompute_ratio=0.0,
            torch_compile=False,
            checkpoint_preserve_rng_state=False,
            cpu_offload=False,
        )
    )
    model.from_hf(checkpoint, strict=True)
    model.eval()
    return model


def score_rows(model, tokenizer, rows: list[dict], checkpoint: str) -> list[dict]:
    tokenize = MaskedQwen3VLTokenizeFunction(
        tokenizer,
        processor_path=checkpoint,
        media_root=str(FRAMES),
        anno_name="loss-contrast.jsonl",
        max_length=16384,
        max_pixels=int(os.environ.get("MAX_PIXELS", "16777216")),
        chat_template="qwen3.5-vl",
    )
    head = None
    for name, module in model.named_modules():
        if name.endswith("lm_head"):
            head = module
    if head is None:
        raise SystemExit("lm_head not found")
    records = []
    loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="token")
    for row in rows:
        compiled = compile_row(row, include_targets=True)
        options = _options(row["questions"]["noul"])
        symbols = compiled.symbols["noul"]
        if [value for value, _description in options] != ["no", "yes"]:
            raise SystemExit(f"noul option order changed: {options}")
        symbol_ids = {
            value: tokenizer.encode(symbol, add_special_tokens=False)[0]
            for symbol, (value, _description) in zip(symbols, options)
        }
        user_content = compiled.messages[1]["content"]
        assistant = compiled.messages[2]["content"]
        label = row["targets"]["noul"]["label"]
        # "no" is a substring of the field name "noul", so match the quoted answer only.
        if "<decision>" not in assistant or f'"{label}"' in assistant:
            raise SystemExit(f"label text leaked into the assistant skeleton for {row['id']}")
        visible = _visible_text(compiled.messages[0]["content"]) + "\n" + _visible_text(user_content) + "\n" + assistant
        visible = visible.replace(row["state"]["query"], "")
        for key in ("id", "pair_type", "video_id"):
            if row[key] and row[key] in visible:
                raise SystemExit(f"{key} leaked into the model text for {row['id']}")
        if isinstance(user_content, list):
            urls = [
                str(part.get("image_url", {}).get("url", ""))
                for part in user_content
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
            if len(urls) != 8 or any(row["video_id"] not in url for url in urls):
                raise SystemExit(f"image paths do not address {row['video_id']} for {row['id']}")
        item = tokenize(row)
        decoded = tokenizer.decode([int(token) for token in item["input_ids"]], skip_special_tokens=False)
        decoded = decoded.replace(row["state"]["query"], "")
        if row["id"] in decoded or row["video_id"] in decoded:
            raise SystemExit(f"identifier survived tokenization for {row['id']}")
        labels = item["labels"]
        supervised = [index for index, label in enumerate(labels) if label != -100]
        if len(supervised) != 1:
            raise SystemExit(f"{row['id']} supervised targets {supervised}")
        if int(item["input_ids"][supervised[0]]) == int(labels[supervised[0]]):
            raise SystemExit(f"{row['id']} writes the answer token into the model input")
        pack_length = int(item["num_tokens"])
        batch = masked_qwen3_vl_collator([[item]], pack_length, tokenizer.pad_token_id, pack_to_max_length=False)[0]
        shifted = batch["shifted_labels"]
        position = supervised[0] - 1
        if int(shifted[0, position]) != int(labels[supervised[0]]):
            raise SystemExit(f"label shift mismatch for {row['id']}")
        captured = {}

        def hook(module, inputs, output, box=captured, index=position):
            # Chunked CE returns logits=None. The head weight is gathered for this
            # forward, so one decision position is enough for the full vocabulary.
            hidden = inputs[0]
            if hidden.ndim == 3:
                vector = hidden[0, index]
            elif hidden.ndim == 2:
                vector = hidden[index]
            elif hidden.ndim == 1:
                vector = hidden
            else:
                raise RuntimeError(f"unexpected hidden shape {tuple(hidden.shape)}")
            def materialize(tensor):
                if hasattr(tensor, "to_local") and hasattr(tensor, "full_tensor"):
                    local = tensor.to_local()
                    if tuple(local.shape) != tuple(tensor.shape):
                        local = tensor.full_tensor()
                    return local
                return tensor

            weight = materialize(module.weight)
            bias = materialize(module.bias) if getattr(module, "bias", None) is not None else None
            vector = materialize(vector).to(dtype=weight.dtype)
            row = torch.nn.functional.linear(vector.unsqueeze(0), weight, bias).squeeze(0)
            box["row"] = row.detach().float().cpu()
            box["shape"] = tuple(row.shape)
            box["returned"] = type(output).__name__

        handle = head.register_forward_hook(hook)
        seq_ctx = batch["seq_ctx"]
        seq_ctx.to("cuda")
        seq_ctx.pixel_values = seq_ctx.pixel_values.cuda()
        seq_ctx.image_grid_thw = seq_ctx.image_grid_thw.cuda()
        loss_ctx = loss_cfg.build({"shifted_labels": shifted})
        prepared = type(loss_ctx).build_batches([loss_ctx], cu_seq_lens_list=[seq_ctx.cu_seq_lens_k])
        loss_ctx = prepared[0]
        with torch.no_grad():
            output = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        handle.remove()
        loss = float(output.loss.detach().float().cpu())
        if not math.isfinite(loss):
            raise SystemExit(f"non-finite CE for {row['id']}")
        logit_row = captured.get("row")
        if logit_row is None or logit_row.ndim != 1 or int(logit_row.numel()) < 1000:
            raise SystemExit(f"lm_head hook missed logits for {row['id']}: shape={captured.get('shape')}")
        target_id = int(labels[supervised[0]])
        full_log_prob = torch.log_softmax(logit_row, dim=-1)
        yes_logit = float(logit_row[symbol_ids["yes"]])
        no_logit = float(logit_row[symbol_ids["no"]])
        pair = torch.tensor([no_logit, yes_logit])
        renorm = torch.softmax(pair, dim=0)
        pred = "yes" if yes_logit >= no_logit else "no"
        records.append(
            {
                "id": row["id"],
                "pair_type": row["pair_type"],
                "control": row["control"],
                "video_id": row["video_id"],
                "target": row["targets"]["noul"]["label"],
                "target_symbol_id": target_id,
                "full_vocabulary_ce": loss,
                "full_vocabulary_target_logprob": float(full_log_prob[target_id]),
                "full_vocabulary_target_prob": float(full_log_prob[target_id].exp()),
                "full_vocabulary_argmax_id": int(torch.argmax(logit_row)),
                "yes_no_margin_logit_yes_minus_no": yes_logit - no_logit,
                "yes_no_renormalized_p_yes": float(renorm[1]),
                "yes_no_renormalized_p_no": float(renorm[0]),
                "predicted_label_yes_no": pred,
                "positive_prediction": pred == "yes",
                "image_count": 8,
                "num_tokens": pack_length,
                "label_in_model_input": False,
                "supervised_targets": 1,
            }
        )
        del output, seq_ctx, logit_row
        torch.cuda.empty_cache()
        print(json.dumps({"scored": row["id"], "ce": loss, "pred": pred}), flush=True)
    return records


def summarize(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        bucket = groups.setdefault(record["pair_type"], {"n": 0, "positive_predictions": 0, "ce_sum": 0.0})
        bucket["n"] += 1
        bucket["positive_predictions"] += int(record["positive_prediction"])
        bucket["ce_sum"] += record["full_vocabulary_ce"]
    for bucket in groups.values():
        bucket["positive_prediction_rate"] = bucket["positive_predictions"] / bucket["n"]
        bucket["mean_full_vocabulary_ce"] = bucket["ce_sum"] / bucket["n"]
    return groups


def main() -> None:
    rows = fixed_rows()
    tokenizer_source = os.environ["TOKENIZER_CHECKPOINT"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    reports = []
    for name, checkpoint in json.loads(os.environ["CHECKPOINTS"]).items():
        model = load_model(checkpoint)
        records = score_rows(model, tokenizer, rows, checkpoint)
        reports.append({"model": name, "checkpoint": checkpoint, "reload_ok": True, "rows": records, "groups": summarize(records)})
        del model
        torch.cuda.empty_cache()
    matched = {}
    mismatched = {}
    for record in reports[0]["rows"]:
        target = mismatched if record["control"] == "mismatched" else matched
        target[record["id"].rsplit("#", 1)[0]] = record
    image_flow_bug = False
    comparisons = []
    for key, other in mismatched.items():
        base = next(row for row in reports[0]["rows"] if row["id"] == f"{key}#pos")
        delta = abs(base["yes_no_margin_logit_yes_minus_no"] - other["yes_no_margin_logit_yes_minus_no"])
        same = delta < 1e-5 and base["full_vocabulary_ce"] == other["full_vocabulary_ce"]
        image_flow_bug = image_flow_bug or same
        comparisons.append({"query": key, "margin_abs_delta": delta, "identical_logits": same})
    payload = {
        "rows_per_model": len(rows),
        "probability_definitions": {
            "full_vocabulary_ce": "official token-reduced CE over the whole vocabulary at the decision position",
            "yes_no_renormalized": "softmax over only the no and yes decision-symbol logits",
        },
        "image_flow_bug": image_flow_bug,
        "mismatch_control": comparisons,
        "models": reports,
    }
    Path(os.environ["OUTPUT"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"image_flow_bug": image_flow_bug, "models": [item["model"] for item in reports]}), flush=True)
    if image_flow_bug:
        raise SystemExit("image flow bug: mismatched frames did not change the decision logits")


if __name__ == "__main__":
    main()
