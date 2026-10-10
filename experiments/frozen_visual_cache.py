"""Lossless validation-only cache at the unchanged frozen visual boundary."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch


def tensor_digest(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tree_map(fn, value):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    if value is None:
        return None
    raise TypeError(f"Unexpected visual output: {type(value)}")


class FrozenVisualCache:
    def __init__(self, directory, provenance, read_only=False, max_bytes=int(2.25 * 1024 ** 3)):
        self.directory = Path(directory)
        self.read_only = read_only
        if max_bytes <= 0:
            raise ValueError("Visual cache byte limit must be positive")
        self.max_bytes = max_bytes
        if read_only:
            if not self.directory.is_dir():
                raise FileNotFoundError("Required existing visual cache is absent")
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        self.last_hit = False
        self.last_key = None
        self.validated = False

    def identity(self, prepared_identity):
        return {"provenance": self.provenance,
                **{k: prepared_identity[k] for k in ["pixels_sha256", "image_grid_thw", "frame_sha256"]}}

    @contextmanager
    def bind(self, model, identity, force_uncached=False):
        if model.training or torch.is_grad_enabled():
            raise ValueError("Visual cache must be eval/no-grad only")
        for name in ["vision_tower", "multi_modal_projector"]:
            if any(p.requires_grad for p in getattr(model, name).parameters()):
                raise ValueError("Cache requires frozen visual weights")
        record_identity = self.identity(identity)
        key = hashlib.sha256(json.dumps(record_identity, sort_keys=True).encode()).hexdigest()
        path = self.directory / f"{key}.pt"
        self.last_hit, self.last_key = False, key
        original = model.get_visual_features
        had_attribute = "get_visual_features" in model.__dict__

        def cached(pixels, grid, mesh=None):
            if mesh is not None and mesh.size() != 1:
                raise ValueError("Cache supports single-rank validation only")
            if tensor_digest(pixels) != identity["pixels_sha256"] or grid.tolist() != identity["image_grid_thw"]:
                raise ValueError("Cache binding/input mismatch")
            if path.exists() and not force_uncached:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["identity"] != record_identity:
                    raise ValueError("Cache provenance mismatch")
                def verify(tensor):
                    expected = next(hashes)
                    if tensor_digest(tensor) != expected:
                        raise ValueError("Cached tensor checksum mismatch")
                    return tensor.to(pixels.device)
                hashes = iter(payload["tensor_sha256"])
                output = tree_map(verify, payload["output"])
                if next(hashes, None) is not None:
                    raise ValueError("Cache checksum count mismatch")
                self.last_hit = True
                return output
            if self.read_only:
                raise FileNotFoundError("Read-only visual cache miss; no uncached retry forward allowed")
            output = original(pixels, grid, mesh)
            copied = tree_map(lambda t: t.detach().cpu().clone(), output)
            hashes = []
            tree_map(lambda t: hashes.append(tensor_digest(t)) or t, copied)
            if not path.exists():
                temporary = path.with_suffix(".tmp")
                torch.save({"identity": record_identity, "output": copied, "tensor_sha256": hashes}, temporary)
                os.replace(temporary, path)
                if sum(p.stat().st_size for p in self.directory.glob("*.pt")) > self.max_bytes:
                    raise ValueError("Validation-only cache exceeds configured byte allowance")
            return output

        model.get_visual_features = cached
        try:
            yield self
        finally:
            if had_attribute:
                model.get_visual_features = original
            else:
                del model.get_visual_features
