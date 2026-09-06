import unittest

from config import CurriculumConfig, TrainingConfig
from scheduler import CurriculumScheduler


class SchedulerTests(unittest.TestCase):
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
