"""Hash actual protected weights and active external model sources, read-only."""

import argparse
import json
import subprocess
from pathlib import Path

from audit_goal_inputs import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    released = root / "models/Intern-Decision-2B"
    hf800 = root / "runs/charades-id2b-e03-20260929-224918/20260929225053/hf-800"
    models = {}
    for name, checkpoint in [("released", released), ("hf800", hf800)]:
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        files = sorted(set(index["weight_map"].values()))
        hashes = {filename: digest(checkpoint / filename) for filename in files}
        models[name] = {"path": str(checkpoint), "weight_sha256": hashes,
                        "bytes": sum((checkpoint / filename).stat().st_size for filename in files),
                        "interface_sha256": {filename: digest(checkpoint / filename) for filename in
                                             ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja", "config.json"]}}
        if name == "released":
            for filename, actual in hashes.items():
                metadata = (checkpoint / ".cache/huggingface/download" / (filename + ".metadata")).read_text().splitlines()
                assert metadata[0] == "8797836c65fc91a2435b1fb6850b5f0aabd75cc3"
                assert actual == metadata[1], f"Released SHA256 mismatch: {filename}"
    for filename in ["model-vision-0001-save_rank0.safetensors", "model-projector-0001-save_rank0.safetensors"]:
        assert models["released"]["weight_sha256"][filename] == models["hf800"]["weight_sha256"][filename]
    assert models["released"]["interface_sha256"] == models["hf800"]["interface_sha256"]
    xtuner = root / "Intern-Decision/third_party/xtuner"
    source_names = ["xtuner/v1/ops/attn_imp.py", "xtuner/v1/model/dense/qwen3vl_text.py",
                    "xtuner/v1/model/compose/qwen3_vl/modeling_vision.py", "xtuner/v1/module/lm_head/lm_head.py"]
    payload = {"result": "pass", "models": models, "frozen_visual_projector_bytes_identical": True,
               "active_xtuner_sources": {name: digest(xtuner / name) for name in source_names},
               "xtuner_dirty_stat": subprocess.check_output(["git", "-C", str(xtuner), "diff", "--stat"], text=True)}
    if args.output.exists():
        raise FileExistsError("Refuse overwrite")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"result": "pass", "frozen_visual_projector_bytes_identical": True,
                      "model_bytes": {name: model["bytes"] for name, model in models.items()}}))


if __name__ == "__main__":
    main()
