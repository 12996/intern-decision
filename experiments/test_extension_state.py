"""Actual vendor CPU restoration and explicit non-reset optimizer schedule fork."""

import copy
import unittest

import torch

from extension_state import fork_schedule, lr_multiplier
from resume_state import restore_resume
from test_resume_state import NonzeroRestoreTests


class ExtensionRestoreTests(unittest.TestCase):
    STEP = 100
    setUp = NonzeroRestoreTests.setUp

    def test_step100_exact_restore_then_schedule_and_next_optimizer_update(self):
        self.trainer._total_step = 500
        self.reference["total_step"] = 500
        proof = restore_resume(self.trainer, {"checkpoint": self.checkpoint, "step": 100}, self.reference)
        self.assertTrue(proof["scheduler_data_rng_exact"])
        self.assertTrue(proof["all_model_optimizer_entries_exact"])
        optimizer = self.trainer._engine.optimizer
        moments = {id(p): state["exp_avg"].clone() for p, state in optimizer.state.items()}
        data = copy.deepcopy(self.trainer._dataloader.get_state_dict())
        rng = torch.get_rng_state().clone()
        result = fork_schedule(self.trainer)
        self.assertEqual(result["source_step"], 100)
        self.assertEqual(self.trainer.cur_step, 100)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-7)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(data, self.trainer._dataloader.get_state_dict())
        for p, state in optimizer.state.items():
            self.assertTrue(torch.equal(moments[id(p)], state["exp_avg"]))
            self.assertEqual(float(state["step"]), 100)
        p = next(self.trainer._engine.model.language_model.parameters())
        old = p.clone()
        optimizer.zero_grad()
        self.trainer._engine.model.language_model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        self.trainer._lr_scheduler.step()
        self.assertFalse(torch.equal(old, p))
        self.assertEqual(float(optimizer.state[p]["step"]), 101)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3.6e-7)

    def test_schedule_endpoints_and_wrong_restore_rejection(self):
        self.assertEqual(lr_multiplier(0), .2)
        self.assertEqual(lr_multiplier(5), 1.)
        self.assertEqual(lr_multiplier(400), .2)
        self.assertTrue(all(.2 <= lr_multiplier(i) <= 1 for i in range(401)))
        with self.assertRaisesRegex(ValueError, "restored step100"):
            fork_schedule(self.trainer)


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("CPU tests must not expose GPUs")
    unittest.main(verbosity=2)
