import unittest

from config import CurriculumConfig, TrainingConfig
from scheduler import CurriculumScheduler


class SchedulerTests(unittest.TestCase):
    def _phase4_config(self, ratio=0.2):
        return TrainingConfig(
            curriculum=CurriculumConfig(
                phase1_end=0, phase2_end=0, phase3_end=0,
                phase4_rgb_sampling_ratio=ratio,
            ),
            total_iters=20, device="cpu",
        )

    def test_phase4_default_budget_has_four_hundred_rgb_steps(self):
        scheduler = CurriculumScheduler(TrainingConfig(device="cpu"))
        steps = [scheduler.get_next_step(step) for step in range(8000, 10000)]
        self.assertEqual(steps.count("p4_rgb_supervised"), 400)
        self.assertEqual(steps.count("p4_ir_focus"), 1600)
        self.assertEqual(steps[:5], ["p4_rgb_supervised"] + ["p4_ir_focus"] * 4)

    def test_phase4_sampling_extremes_select_only_the_requested_source(self):
        for ratio, expected in ((0.0, "p4_ir_focus"), (1.0, "p4_rgb_supervised")):
            with self.subTest(ratio=ratio):
                scheduler = CurriculumScheduler(self._phase4_config(ratio))
                self.assertEqual(
                    [scheduler.get_next_step(step) for step in range(20)],
                    [expected] * 20,
                )

    def test_phase4_counter_resumes_inside_a_five_step_cycle(self):
        config = self._phase4_config()
        original = CurriculumScheduler(config)
        for step in range(3):
            original.get_next_step(step)
        resumed = CurriculumScheduler(config)
        resumed.load_state_dict(original.state_dict())
        self.assertEqual(
            [resumed.get_next_step(step) for step in range(3, 10)],
            ["p4_ir_focus", "p4_ir_focus", "p4_rgb_supervised"] + ["p4_ir_focus"] * 4,
        )

    def test_phase4_starts_a_new_cycle_for_legacy_scheduler_state(self):
        scheduler = CurriculumScheduler(self._phase4_config())
        scheduler.load_state_dict({"PHASE2_TRANSITION": 7, "PHASE3_ADAPTATION": 3})
        self.assertEqual(scheduler.get_next_step(0), "p4_rgb_supervised")

    def test_phase3_ratio_is_rgb_route_ratio(self):
        config = TrainingConfig(
            curriculum=CurriculumConfig(
                phase1_end=0,
                phase2_end=0,
                phase3_end=10,
                phase3_rgb_sampling_ratio=0.3,
            ),
            total_iters=10,
            device="cpu",
        )
        scheduler = CurriculumScheduler(config)
        steps = [scheduler.get_next_step(step) for step in range(10)]
        self.assertEqual(steps.count("p3_rgb_flow"), 3)
        self.assertEqual(steps.count("p3_ir_flow"), 7)

    def test_sampling_extremes_are_deterministic(self):
        self.assertEqual(CurriculumScheduler._ratio_to_period(0.0), 1)
        self.assertEqual(CurriculumScheduler._ratio_to_period(1.0), 1)

        config = TrainingConfig(
            curriculum=CurriculumConfig(
                phase1_end=0,
                phase2_end=0,
                phase3_end=4,
                phase3_rgb_sampling_ratio=0.0,
            ),
            total_iters=4,
            device="cpu",
        )
        scheduler = CurriculumScheduler(config)
        self.assertEqual(
            [scheduler.get_next_step(i) for i in range(4)],
            ["p3_ir_flow"] * 4,
        )

    def test_scheduler_counters_can_resume(self):
        config = TrainingConfig(
            curriculum=CurriculumConfig(
                phase1_end=0,
                phase2_end=0,
                phase3_end=10,
                phase3_rgb_sampling_ratio=0.3,
            ),
            total_iters=10,
            device="cpu",
        )
        first = CurriculumScheduler(config)
        for step in range(4):
            first.get_next_step(step)
        resumed = CurriculumScheduler(config)
        resumed.load_state_dict(first.state_dict())
        self.assertEqual(
            resumed.get_next_step(4), first.get_next_step(4)
        )


if __name__ == "__main__":
    unittest.main()
