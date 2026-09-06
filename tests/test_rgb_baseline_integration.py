import unittest

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from config import (
    AugConfig,
    CurriculumConfig,
    DataLoaderConfig,
    EMAConfig,
    EvalLoaderConfig,
    FCOSModelConfig,
    TrainLoaderConfig,
    TrainingConfig,
)
from trainer import CurriculumDomainAdaptationTrainer


class RGBDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (
            torch.full((3, 16, 16), 0.5 + index * 0.01),
            {'boxes': torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
             'labels': torch.tensor([0]), 'stem': f'rgb-{index}'},
            f'rgb-{index}',
        )


class IRDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(3, 16, 16), f'ir-{index}'


class CountingOptimizer(SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class CountingWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.raw_calls = 0
        self.supervised_calls = 0
        self.distill_calls = 0

    def raw(self, images, targets=None, sample_ids=None):
        self.raw_calls += 1
        return {'images': images, 'targets': targets, 'sample_ids': sample_ids}

    def supervised_from_output(self, output, targets):
        self.supervised_calls += 1
        return {'loss_cls': self.weight.square()}

    def distill_from_outputs(self, student_output, teacher_output, settings):
        self.distill_calls += 1
        raise AssertionError('RGB baseline must not distill')


def collate_rgb(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch], tuple(item[2] for item in batch)


def collate_ir(batch):
    return torch.stack([item[0] for item in batch]), tuple(item[1] for item in batch)


class RGBBaselineIntegrationTests(unittest.TestCase):
    def test_rgb_baseline_forwards_student_once_and_steps_once(self):
        config = TrainingConfig(
            model=FCOSModelConfig(weights=None, pretrained_backbone=False),
            ema=EMAConfig(start_steps=100),
            curriculum=CurriculumConfig(phase1_end=10, phase2_end=20, phase3_end=30),
            loader=DataLoaderConfig(
                train=TrainLoaderConfig(batch_size=2, num_workers=0),
                eval=EvalLoaderConfig(batch_size=2, num_workers=0),
            ),
            aug=AugConfig(hflip_prob=0.0, blur_prob=0.0,
                          brightness_prob=0.0, contrast_prob=0.0),
            workflow='rgb_baseline',
            total_iters=1,
            device='cpu',
        )
        student = CountingWrapper()
        rgb_teacher = CountingWrapper()
        ir_teacher = CountingWrapper()
        optimizer = CountingOptimizer(student.parameters())
        loader = DataLoader(RGBDataset(), batch_size=2, collate_fn=collate_rgb)
        ir_loader = DataLoader(IRDataset(), batch_size=2, collate_fn=collate_ir)
        trainer = CurriculumDomainAdaptationTrainer(
            student=student,
            rgb_teacher=rgb_teacher,
            ir_teacher=ir_teacher,
            optimizer=optimizer,
            config=config,
            rgb_loader=loader,
            ir_loader=ir_loader,
            val_loader=None,
        )
        logs = trainer.train_one_iteration()
        self.assertEqual(student.raw_calls, 1)
        self.assertEqual(student.supervised_calls, 1)
        self.assertEqual(student.distill_calls, 0)
        self.assertEqual(rgb_teacher.raw_calls, 0)
        self.assertEqual(ir_teacher.raw_calls, 0)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(trainer.global_step, 1)
        self.assertAlmostEqual(logs["total_loss"], 2.0, places=6)
        self.assertTrue(torch.isfinite(torch.tensor(logs['total_loss'])))


if __name__ == '__main__':
    unittest.main()
