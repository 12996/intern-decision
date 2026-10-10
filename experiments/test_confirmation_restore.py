"""Actual Torch/vendor model-only DCP load; no GPU or optimizer allocation."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp

from confirmation_worker import load_final_model_only


class RestoreTests(unittest.TestCase):
    def test_model_only_restores_nonzero_updates_and_preserves_frozen(self):
        model = torch.nn.ModuleDict({"language_model": torch.nn.Linear(3, 2), "vision_tower": torch.nn.Linear(3, 2)})
        model["vision_tower"].requires_grad_(False)
        expected = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("language_model")}
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("vision_tower")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "train_state.json").write_text(json.dumps({"cur_step": 100}))
            dcp.save({"model": expected, "optimizer": {"unloaded_extra_tensor": torch.ones(2)}}, checkpoint_id=path / "weights")
            with torch.no_grad():
                for p in model["language_model"].parameters():
                    p.add_(1)
            result = load_final_model_only(model, path)
            self.assertEqual(result["model_entries_bitwise_verified"], 2)
            self.assertFalse(result["optimizer_loaded"])
            for key, tensor in {**expected, **frozen}.items():
                self.assertTrue(torch.equal(model.state_dict()[key], tensor))

    def test_missing_model_keys_rejected(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "train_state.json").write_text(json.dumps({"cur_step": 100}))
            dcp.save({"model": {"weight": model.weight}}, checkpoint_id=path / "weights")
            with self.assertRaises(ValueError):
                load_final_model_only(model, path)


if __name__ == "__main__":
    unittest.main()
