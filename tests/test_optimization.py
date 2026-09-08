import unittest

import torch
from torch import nn

from config import (
    CurriculumConfig,
    DataLoaderConfig,
    FCOSModelConfig,
    OptimizerConfig,
    TrainLoaderConfig,
    TrainingConfig,
)
from optimization import build_lr_scheduler, build_optimizer


class OptimizationTests(unittest.TestCase):
    def _config(self, **optimizer_kwargs):
        return TrainingConfig(
            model=FCOSModelConfig(weights=None, pretrained_backbone=False),
            optimizer=OptimizerConfig(**optimizer_kwargs),
            curriculum=CurriculumConfig(
                phase1_end=10, phase2_end=20, phase3_end=30
            ),
            loader=DataLoaderConfig(train=TrainLoaderConfig(batch_size=4)),
            total_iters=40,
            device="cpu",
        )

    def test_default_lr_scales_from_d3t_reference_batch(self):
        config = self._config()

        self.assertEqual(config.optimizer.base_lr, 0.004)
        self.assertEqual(config.optimizer.warmup_iters, 10)

    def test_optimizer_contains_every_trainable_parameter_once(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        model[1].bias.requires_grad_(False)
        config = self._config(base_lr=0.002)

        optimizer = build_optimizer(model, config)

        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        expected = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(optimized), len(expected))
        self.assertEqual({id(p) for p in optimized}, {id(p) for p in expected})
        self.assertEqual(optimizer.defaults["lr"], 0.002)

    def test_linear_warmup_and_milestone_are_iteration_based(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        config = self._config(
            base_lr=1.0,
            warmup_iters=2,
            warmup_factor=0.1,
            milestones=(3,),
            gamma=0.1,
        )
        optimizer = build_optimizer(nn.ParameterList([parameter]), config)
        scheduler = build_lr_scheduler(optimizer, config)

        observed = [optimizer.param_groups[0]["lr"]]
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            observed.append(optimizer.param_groups[0]["lr"])

        self.assertEqual(observed, [0.1, 0.55, 1.0, 0.1])

    def test_scheduler_state_roundtrip_restores_iteration(self):
        config = self._config(
            base_lr=1.0,
            warmup_iters=4,
            warmup_factor=0.1,
            milestones=(3,),
            gamma=0.1,
        )
        first_parameter = nn.Parameter(torch.tensor(1.0))
        first_optimizer = build_optimizer(
            nn.ParameterList([first_parameter]), config
        )
        first_scheduler = build_lr_scheduler(first_optimizer, config)
        for _ in range(2):
            first_optimizer.step()
            first_scheduler.step()

        different_config = self._config(
            base_lr=1.0,
            warmup_iters=1,
            warmup_factor=1.0,
            milestones=(100,),
            gamma=0.9,
        )
        second_parameter = nn.Parameter(torch.tensor(1.0))
        second_optimizer = build_optimizer(
            nn.ParameterList([second_parameter]), different_config
        )
        second_scheduler = build_lr_scheduler(second_optimizer, different_config)
        second_optimizer.load_state_dict(first_optimizer.state_dict())
        second_scheduler.load_state_dict(first_scheduler.state_dict())

        self.assertEqual(second_scheduler.last_epoch, first_scheduler.last_epoch)
        self.assertEqual(
            second_optimizer.param_groups[0]["lr"],
            first_optimizer.param_groups[0]["lr"],
        )
        first_optimizer.step()
        first_scheduler.step()
        second_optimizer.step()
        second_scheduler.step()
        self.assertEqual(
            second_optimizer.param_groups[0]["lr"],
            first_optimizer.param_groups[0]["lr"],
        )


if __name__ == "__main__":
    unittest.main()
