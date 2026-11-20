"""Behavior tests for spreading domains and continuing a saved schedule."""

import copy
from collections import Counter
import unittest

import config as config_module
import scheduler as scheduler_module


class DomainRatioSamplerTests(unittest.TestCase):
    def test_seven_to_three_is_spread_instead_of_grouped(self):
        sampler = scheduler_module.DomainRatioSampler({"rgb": 7, "ir": 3})
        self.assertEqual(
            [sampler.next_domain() for _ in range(10)],
            ["rgb", "ir", "rgb", "rgb", "rgb", "ir", "rgb", "rgb", "ir", "rgb"],
        )

    def test_equal_deficits_follow_domain_order(self):
        sampler = scheduler_module.DomainRatioSampler({"rgb": 1, "saga": 1, "gan": 1})
        self.assertEqual(
            [sampler.next_domain() for _ in range(6)],
            ["rgb", "saga", "gan", "rgb", "saga", "gan"],
        )

    def test_scaled_fractional_weights_produce_the_same_sequence(self):
        integer = scheduler_module.DomainRatioSampler({"rgb": 7, "ir": 3})
        fractional = scheduler_module.DomainRatioSampler({"rgb": 0.7, "ir": 0.3})
        self.assertEqual(
            [integer.next_domain() for _ in range(1000)],
            [fractional.next_domain() for _ in range(1000)],
        )

    def test_invalid_weights_are_rejected(self):
        for ratios in ({}, {"rgb": 0}, {"rgb": -1, "ir": 2},
                       {"rgb": float("nan")}, {"rgb": float("inf")},
                       {"rgb": "7"}, {"": 1}):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                scheduler_module.DomainRatioSampler(ratios)

    def test_state_is_a_snapshot_and_changed_ratios_are_rejected(self):
        sampler = scheduler_module.DomainRatioSampler({"rgb": 7, "ir": 3})
        sampler.next_domain()
        saved = copy.deepcopy(sampler.state_dict())
        snapshot = sampler.state_dict()
        sampler.next_domain()
        self.assertEqual(snapshot, saved)
        changed = scheduler_module.DomainRatioSampler({"rgb": 3, "ir": 7})
        with self.assertRaisesRegex(ValueError, "ratio|schedule"):
            changed.load_state_dict(saved)


class BridgeDomainSchedulerTests(unittest.TestCase):
    def build_scheduler(self, total_iters=100, **kwargs):
        config = config_module.BridgeScheduleConfig(**kwargs)
        return scheduler_module.BridgeDomainScheduler(config, total_iters)

    def test_five_stages_mix_all_four_domains_at_the_requested_ratios(self):
        schedule = self.build_scheduler()
        counts = []
        for stage in range(5):
            domains = [schedule.get_next_domain(it) for it in range(stage * 20, (stage + 1) * 20)]
            counts.append(tuple(Counter(domains)[d] for d in ("rgb", "saga", "ir_gan", "ir_real")))
        self.assertEqual(counts, [(20, 0, 0, 0), (10, 6, 4, 0), (6, 6, 6, 2),
                                  (4, 4, 8, 4), (2, 2, 4, 12)])

    def test_phase_boundary_resets_allocation_even_with_identical_ratios(self):
        ratios = ((1, 0, 0, 0),) + ((7, 3, 0, 0),) * 4
        schedule = self.build_scheduler(total_iters=15, phase_ratios=ratios)
        domains = [schedule.get_next_domain(it) for it in range(15)]
        self.assertEqual(domains[3:6], ["rgb", "saga", "rgb"])
        self.assertEqual(domains[6:9], ["rgb", "saga", "rgb"])

    def test_short_runs_skip_empty_stages_without_going_out_of_range(self):
        schedule = self.build_scheduler(total_iters=2)
        self.assertEqual(schedule.get_next_domain(0), "rgb")
        self.assertEqual(schedule.phase_for_iter(1), 2)
        self.assertEqual(schedule.get_next_domain(1), "rgb")
        with self.assertRaises(ValueError):
            schedule.get_next_domain(2)

    def test_explicit_stage_durations_control_boundaries(self):
        schedule = self.build_scheduler(
            total_iters=100, phase_fractions=(0.1, 0.2, 0.5, 0.9, 1.0)
        )
        self.assertEqual(
            [schedule.phase_for_iter(it) for it in (0, 9, 10, 19, 20, 49, 50, 89, 90, 99)],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        )

    def test_resume_continues_the_domain_sequence_at_every_stage_boundary(self):
        full = self.build_scheduler()
        expected = [full.get_next_domain(it) for it in range(100)]
        for cut in (0, 1, 19, 20, 21, 39, 40, 59, 60, 79, 80, 99, 100):
            with self.subTest(cut=cut):
                first = self.build_scheduler()
                for it in range(cut):
                    first.get_next_domain(it)
                resumed = self.build_scheduler()
                resumed.load_state_dict(first.state_dict())
                self.assertEqual(
                    [resumed.get_next_domain(it) for it in range(cut, 100)], expected[cut:]
                )

    def test_invalid_stage_configuration_is_rejected(self):
        cases = [
            {"phase_ratios": ((1, 0, 0, 0),) * 4},
            {"phase_ratios": ((1, 0, 0),) * 5},
            {"phase_ratios": ((0, 1, 0, 0),) * 5},
            {"phase_ratios": ((1, 0, 0, 0),) + ((0, 0, 0, 0),) * 4},
            {"phase_ratios": ((1, 0, 0, 0),) + ((-1, 1, 1, 1),) * 4},
            {"phase_ratios": ((1, 0, 0, 0),) + ((float("nan"), 1, 1, 1),) * 4},
            {"phase_fractions": (0.2, 0.4, 0.4, 0.8, 1.0)},
            {"phase_fractions": (0.0, 0.4, 0.6, 0.8, 1.0)},
            {"phase_fractions": (0.2, 0.4, 0.6, 0.8, 0.9)},
            {"phase_fractions": (0.2, 0.4, float("inf"), 0.8, 1.0)},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build_scheduler(**kwargs)

    def test_invalid_iteration_or_nonsequential_call_is_rejected(self):
        for total in (0, -1, 2.5, True):
            with self.subTest(total=total), self.assertRaises(ValueError):
                self.build_scheduler(total_iters=total)
        schedule = self.build_scheduler()
        for it in (-1, 1, 1.5, True):
            with self.subTest(it=it), self.assertRaises(ValueError):
                schedule.get_next_domain(it)
        self.assertEqual(schedule.get_next_domain(0), "rgb")

    def test_restore_rejects_changed_schedule_without_changing_live_progress(self):
        first = self.build_scheduler()
        for it in range(23):
            first.get_next_domain(it)
        changed = self.build_scheduler(phase_fractions=(0.1, 0.3, 0.6, 0.8, 1.0))
        before = changed.state_dict()
        with self.assertRaisesRegex(ValueError, "schedule"):
            changed.load_state_dict(first.state_dict())
        self.assertEqual(changed.state_dict(), before)

    def test_restore_rejects_inconsistent_counts_without_changing_live_progress(self):
        schedule = self.build_scheduler()
        for it in range(23):
            schedule.get_next_domain(it)
        saved = copy.deepcopy(schedule.state_dict())
        saved["next_step"] = 24
        fresh = self.build_scheduler()
        before = fresh.state_dict()
        with self.assertRaises(ValueError):
            fresh.load_state_dict(saved)
        self.assertEqual(fresh.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
