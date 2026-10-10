"""Lossless, synchronous FP32 export through XTuner's supported save policy."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch.distributed as dist


def export_parts(model):
    names = ("language_model", "vision_tower", "multi_modal_projector")
    if all(hasattr(model, name) for name in names):
        return [getattr(model, name) for name in names]
    return [model]


@contextmanager
def fp32_save_policy(model):
    # Set this only during synchronous I/O: the same field also affects FSDP
    # construction, so it must NOT be set in the training configuration.
    configs, seen = [], set()
    for part in export_parts(model):
        cfg = part.config.hf_save_cfg
        if not hasattr(cfg, "fp32_keys_pattern"):
            raise ValueError("XTuner has no FP32-key save policy")
        if id(cfg) not in seen:
            configs.append((cfg, cfg.fp32_keys_pattern))
            seen.add(id(cfg))
    try:
        for cfg, _ in configs:
            cfg.fp32_keys_pattern = [".*"]
        yield
    finally:
        for cfg, original in configs:
            cfg.fp32_keys_pattern = original


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_fp32_export(model, path):
    """Compare every indexed tensor against the current official HF mapping."""
    path = Path(path).resolve()
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("Empty best HF export")
    files = {}
    for filename in set(weight_map.values()):
        shard = path / filename
        if shard.is_symlink() or shard.resolve().parent != path or not shard.is_file():
            raise ValueError("Missing or unsafe HF shard")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {k for k, v in weight_map.items() if v == filename}:
                raise ValueError("HF shard/index key mismatch")
        files[filename] = shard

    checked, fp32_parameters = {}, set()
    for part in export_parts(model):
        specs = part._load_spec_params()
        buffers = {part._clean_param_name(name) for name, _ in part.named_buffers()}
        parameter_keys = {
            key for tensor, spec in specs if spec.name not in buffers and tensor.is_floating_point()
            for key in spec.global_hf_keys
        }
        for names, tensors in part._get_hf_param(specs, dtype=torch.bfloat16, distributed_save=True):
            for name, expected in zip(names, tensors, strict=True):
                if name in checked:
                    continue  # Same first-key policy as the official tied-weight writer.
                if name not in weight_map:
                    raise ValueError(f"Missing HF tensor: {name}")
                if name in parameter_keys:
                    if expected.dtype != torch.float32:
                        raise ValueError(f"FP32 save policy did not apply: {name}")
                    fp32_parameters.add(name)
                with safe_open(files[weight_map[name]], framework="pt", device="cpu") as handle:
                    actual = handle.get_tensor(name)
                expected = expected.detach().cpu()
                if actual.dtype != expected.dtype or actual.shape != expected.shape:
                    raise ValueError(f"HF tensor dtype/shape mismatch: {name}")
                if actual.is_floating_point() and not torch.isfinite(actual).all():
                    raise ValueError(f"Nonfinite HF tensor: {name}")
                expected_hash = tensor_sha256(expected)
                if tensor_sha256(actual) != expected_hash:
                    raise ValueError(f"Lossy HF export: {name}")
                checked[name] = expected_hash
    if set(checked) != set(weight_map) or not fp32_parameters:
        raise ValueError("Incomplete FP32 HF export coverage")
    return {"pass": True, "checked_tensors": len(checked), "fp32_parameter_tensors": len(fp32_parameters),
            "all_tensor_values_bitwise_equal": True,
            "hf_tensor_hashes_sha256": hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest(),
            "api_save_dtype": "bfloat16", "floating_parameter_save_dtype": "float32",
            "persistent_buffer_dtype": "runtime_dtype"}


def update_dtype_metadata(path):
    config_path = Path(path) / "config.json"
    config = json.loads(config_path.read_text())
    for item in [config, config.get("text_config"), config.get("vision_config")]:
        if isinstance(item, dict):
            # Support both Transformers spellings without an auto-load BF16 cast.
            item["torch_dtype"] = "float32"
            item["dtype"] = "float32"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def save_hf_fp32(engine, path):
    if dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError("This verified FP32 adapter is single-rank only")
    path = Path(path)
    if path.exists():
        raise ValueError("Refuse overwrite of existing best export")
    with torch.no_grad(), fp32_save_policy(engine.model):
        # BF16 selects the supported non-FP8 writer; fp32_keys_pattern overrides
        # the actual parameter dtype BEFORE any runtime-to-checkpoint conversion.
        engine.save_hf(str(path), save_dtype=torch.bfloat16)
        receipt = verify_fp32_export(engine.model, path)
    update_dtype_metadata(path)
    (path / "fp32_export_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
