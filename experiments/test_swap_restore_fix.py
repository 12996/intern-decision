"""No-GPU tests of restored optimizer reference repair."""

from types import SimpleNamespace
import unittest

from swap_restore_fix import rebind_swap_param_groups, assert_swap_param_groups


class SwapRestoreTests(unittest.TestCase):
    def test_rebind_tracks_future_scheduler_updates(self):
        parameter = object()
        old = {"params": [parameter], "lr": 0.0}
        new = {"params": [parameter], "lr": 0.0}
        optimizer = SimpleNamespace(param_groups=[new], _param_to_group_map={parameter: old})
        result = rebind_swap_param_groups(optimizer)
        self.assertEqual(result["stale_groups_rebound"], 1)
        new["lr"] = 2e-6
        self.assertEqual(optimizer._param_to_group_map[parameter]["lr"], 2e-6)
        self.assertIs(optimizer._param_to_group_map[parameter], new)
        self.assertEqual(assert_swap_param_groups(optimizer)["learning_rates"], [2e-6])

    def test_later_stale_references_are_rejected(self):
        parameter = object()
        optimizer = SimpleNamespace(param_groups=[{"params": [parameter], "lr": 0.0}],
                                    _param_to_group_map={parameter: {"params": [parameter], "lr": 0.0}})
        with self.assertRaisesRegex(ValueError, "no longer track"):
            assert_swap_param_groups(optimizer)

    def test_non_swap_optimizer_is_unchanged(self):
        self.assertEqual(rebind_swap_param_groups(SimpleNamespace()), {"applicable": False})

    def test_changed_parameter_identity_is_rejected(self):
        optimizer = SimpleNamespace(param_groups=[{"params": [object()], "lr": 0.0}],
                                    _param_to_group_map={object(): {"lr": 0.0}})
        with self.assertRaisesRegex(ValueError, "identity changed"):
            rebind_swap_param_groups(optimizer)


if __name__ == "__main__":
    unittest.main()
