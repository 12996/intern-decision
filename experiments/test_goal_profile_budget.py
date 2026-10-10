"""CPU-only budget invariants for the explicitly authorized second diagnostic."""

import unittest
from run_goal_profile import check_resume, new_budget


class BudgetTests(unittest.TestCase):
    def test_limits_independent_round_without_erasing_prior_charge(self):
        prior = {"gpu_hours_charged": 0.27566717982292177}
        budget = new_budget(prior, "frozen", 100, 123)
        self.assertEqual(budget["deadline_epoch"], 1000)
        self.assertEqual(budget["max_forward_attempts"], 6)
        self.assertEqual(budget["prior_gpu_hours"], prior["gpu_hours_charged"])
        self.assertEqual(budget["gpu_hours_charged"], 0)
        self.assertEqual(prior, {"gpu_hours_charged": 0.27566717982292177})

    def test_original_total_can_shorten_but_not_extend_round(self):
        budget = new_budget({"gpu_hours_charged": 23.9}, "frozen", 100, 123)
        self.assertAlmostEqual(budget["deadline_epoch"], 460)
        with self.assertRaises(TimeoutError):
            new_budget({"gpu_hours_charged": 24}, "frozen", 100, 123)

    def test_resume_cannot_reset_deadline_or_attempts(self):
        budget = new_budget({"gpu_hours_charged": 0.25}, "frozen", 100, 123)
        check_resume(budget, 999, 5)
        for now, attempts in [(1000, 0), (1010, 1), (999, 6)]:
            with self.assertRaises(TimeoutError):
                check_resume(budget, now, attempts)
        self.assertEqual(budget["deadline_epoch"], 1000)


if __name__ == "__main__":
    unittest.main()
