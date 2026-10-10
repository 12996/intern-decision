"""Bounded route-A architecture smoke for the real Intern-Decision model.

This is not training or retrieval evaluation. It checks that the visual path is
query-independent, that a text-only query forward returns a usable hidden
state, and that a new projection head has a real gradient path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from diagnose_loss_contrast import load_model
from src.inputs.masked import (
    MaskedOpenaiTokenizeFunction,
    MaskedQwen3VLTokenizeFunction,
    masked_qwen3_vl_collator,
    masked_sft_llm_collator,
)


def _local(tensor):
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def _row(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.loads(stream.readline())


def main() -> None:
    root = Path(os.environ["DECISION_ROOT"])
    checkpoint = os.environ["CHECKPOINT"]
    media_root = os.environ["MEDIA_ROOT"]
    out = Path(os.environ["OUTPUT"])
    out.parent.mkdir(parents=True, exist_ok=True)

    row = _row(Path(os.environ["ROW_PATH"]))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = load_model(checkpoint)
    model.eval()

    image_fn = MaskedQwen3VLTokenizeFunction(
        tokenizer,
        processor_path=checkpoint,
        media_root=media_root,
        anno_name="route-a-smoke.jsonl",
        max_length=16384,
        max_pixels=16777216,
        chat_template="qwen3.5-vl",
    )
    image_item = image_fn(row, include_targets=False)
    image_batch = masked_qwen3_vl_collator(
        [[image_item]], image_item["num_tokens"], tokenizer.pad_token_id, pack_to_max_length=False
    )[0]
    image_ctx = image_batch["seq_ctx"]
    image_ctx.to("cuda")
    image_ctx.pixel_values = image_ctx.pixel_values.cuda()
    image_ctx.image_grid_thw = image_ctx.image_grid_thw.cuda()

    with torch.no_grad():
        visual, deepstack = model.get_visual_features(
            image_ctx.pixel_values, image_ctx.image_grid_thw, image_ctx.sequence_parallel_mesh
        )
    visual_local = _local(visual)
    visual_shape = list(visual_local.shape)
    grid = image_ctx.image_grid_thw.detach().cpu().tolist()
    deepstack_shapes = [list(_local(value).shape) for value in (deepstack or [])]

    # A temporary projection verifies a non-detached training path without
    # changing or saving the base model.
    pixels_for_grad = image_ctx.pixel_values.detach().requires_grad_(True)
    visual_for_grad, _ = model.get_visual_features(
        pixels_for_grad, image_ctx.image_grid_thw, image_ctx.sequence_parallel_mesh
    )
    projection = torch.nn.Linear(visual_for_grad.shape[-1], 128, bias=False).cuda()
    projection.zero_grad(set_to_none=True)
    projection(visual_for_grad.float()).square().mean().backward()
    pixel_grad_norm = float(pixels_for_grad.grad.detach().float().norm().cpu())
    projection_grad_norm = float(projection.weight.grad.detach().float().norm().cpu())

    text_row = {
        "state": {"query": row["state"]["query"]},
        "questions": row["questions"],
    }
    text_fn = MaskedOpenaiTokenizeFunction(
        tokenizer, chat_template="qwen3.5-vl", max_length=16384
    )
    text_item = text_fn(text_row, include_targets=False)
    text_batch = masked_sft_llm_collator(
        [[text_item]], text_item["num_tokens"], tokenizer.pad_token_id, pack_to_max_length=False
    )[0]
    text_ctx = text_batch["seq_ctx"]
    text_ctx.to("cuda")
    query_capture = {}

    def capture(_module, _args, output):
        value = _local(output)
        query_capture["shape"] = list(value.shape)
        query_capture["last_norm"] = float(value[0, -1].float().norm().detach().cpu())

    hook = model.language_model.norm.register_forward_hook(capture)
    with torch.no_grad():
        model(seq_ctx=text_ctx, loss_ctx={"lm": None})
    hook.remove()

    payload = {
        "status": "pass",
        "checkpoint": checkpoint,
        "row_id": row.get("id"),
        "query": row["state"]["query"],
        "image_grid_thw": grid,
        "visual_shape": visual_shape,
        "deepstack_shapes": deepstack_shapes,
        "visual_tokens_per_grid": [int(t * h * w // 4) for t, h, w in grid],
        "query_only_hidden_shape": query_capture.get("shape"),
        "query_only_last_hidden_norm": query_capture.get("last_norm"),
        "projection_grad_norm": projection_grad_norm,
        "pixel_input_grad_norm": pixel_grad_norm,
        "vision_parameters_trainable": sum(p.numel() for p in model.vision_tower.parameters() if p.requires_grad),
        "projector_parameters_trainable": sum(p.numel() for p in model.multi_modal_projector.parameters() if p.requires_grad),
        "interpretation": {
            "visual_cache_candidate": True,
            "decision_score_cacheable_without_query": False,
            "next": "add segment pooling and query/video projection head after this smoke",
        },
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
