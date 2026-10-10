"""CPU-only size accounting for the existing, owned official checkpoint."""

import json
import math
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader


ROOT = Path("/home/zhaopu/intern-decision")
PATH = ROOT / "runs/charades-id2b-e03-20260929-224918/20260929225053/checkpoints/ckpt-step-750"
OUT = ROOT / "outputs/cleanup-20261003/checkpoint_storage.json"


def main():
    metadata = FileSystemReader(PATH / "weights").read_metadata()
    groups = {}
    for index, stored in metadata.storage_data.items():
        name = index.fqn
        if name.startswith("model.vision_tower."):
            group = "model_frozen_vision"
        elif name.startswith("model.multi_modal_projector."):
            group = "model_frozen_projector"
        elif name.startswith("model."):
            group = "model_language"
        elif "vision_tower" in name or "multi_modal_projector" in name:
            group = "optimizer_frozen"
        else:
            group = "optimizer_or_other"
        result = groups.setdefault(group, {"serialized_bytes": 0, "tensor_elements": 0, "entries": 0, "dtypes": {}})
        result["serialized_bytes"] += stored.length
        result["entries"] += 1
        tensor = metadata.state_dict_metadata[name]
        if hasattr(tensor, "size"):
            result["tensor_elements"] += math.prod(tensor.size)
            dtype = str(tensor.properties.dtype)
            result["dtypes"][dtype] = result["dtypes"].get(dtype, 0) + 1
    report = {"checkpoint": str(PATH), "groups": groups,
              "model_prefixes": sorted({".".join(k.split(".")[:3]) for k in metadata.state_dict_metadata if k.startswith("model.")}),
              "total_file_bytes": sum(p.stat().st_size for p in PATH.rglob("*") if p.is_file())}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
