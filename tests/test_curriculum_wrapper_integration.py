import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import (
    AugConfig,
    CurriculumConfig,
    DataLoaderConfig,
    DistillConfig,
    EMAConfig,
    EvalLoaderConfig,
    FCOSModelConfig,
    LossConfig,
    TrainLoaderConfig,
    TrainingConfig,
    TeacherSchedule,
)
from models.d3t_adapter import (
    AdapterOutput,
    CriterionResult,
    D3TCriterion,
    DetectorAdapter,
    DistillationPair,
    Predictions,
    SupervisedBatch,
)
from models.d3t_wrapper import D3TWrapper
from scheduler import CurriculumScheduler
from trainer import CurriculumDomainAdaptationTrainer
from ema import copy_student_to_teacher, ema_update


class TinyAdapter(DetectorAdapter):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, images, targets=None, sample_ids=None):
        self.forward_calls += 1
        predictions = tuple(
            Predictions(
                self.weight.expand(2, 3),
                self.weight.expand(2, 4),
                self.weight.expand(2),
            )
            for _ in images
        )
        return AdapterOutput(predictions, sample_ids=sample_ids)

    def prepare_supervised(self, output, targets):
        prepared = tuple(
            {
                "class_targets": torch.zeros_like(prediction.class_logits),
                "box_targets": torch.zeros_like(prediction.boxes),
                "quality_targets": torch.zeros_like(prediction.quality_logits),
                "foreground": torch.zeros_like(
                    prediction.quality_logits, dtype=torch.bool
                ),
            }
            for prediction in output.predictions
        )
        return SupervisedBatch(output.predictions, prepared)

    def prepare_distillation(self, student, teacher):
        if student.sample_ids != teacher.sample_ids:
            raise ValueError("sample IDs do not correspond")
        return DistillationPair(student.predictions, teacher.predictions)

    def postprocess(self, output):
        return []


class TinyCriterion(D3TCriterion):
    def __init__(self):
        super().__init__()
        self.distill_calls = 0

    def supervised(self, batch):
        return {
            "loss_cls": sum(
                prediction.class_logits.square().mean()
                for prediction in batch.predictions
            )
        }

    def distillation(self, pair, settings):
        self.distill_calls += 1
        return CriterionResult(
            losses={
                "loss_kd_cls": sum(
                    (student.class_logits - teacher.class_logits)
                    .square()
                    .mean()
                    for student, teacher in zip(pair.student, pair.teacher)
                )
            },
            metrics={"kd_selected_count": torch.tensor(1.0)},
        )


class RGBDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (
            torch.full((3, 16, 16), 0.5 + index * 0.01),
            {
                "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
                "labels": torch.tensor([0]),
                "stem": f"rgb-{index}",
            },
            f"rgb-{index}",
        )


class IRDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.full((3, 16, 16), 0.2), f"ir-{index}"


def collate_rgb(batch):
    return (
        torch.stack([item[0] for item in batch]),
        [item[1] for item in batch],
        tuple(item[2] for item in batch),
    )


def collate_ir(batch):
    return torch.stack([item[0] for item in batch]), tuple(item[1] for item in batch)


def build_config(**kwargs):
    curriculum = kwargs.pop(
        "curriculum",
        CurriculumConfig(phase1_end=0, phase2_end=10, phase3_end=20),
    )
    return TrainingConfig(
        model=FCOSModelConfig(weights=None, pretrained_backbone=False),
        distill=DistillConfig(
            rgb_teacher=TeacherSchedule(
                phase1=(1.0, 0.0), phase2=(1.0, 0.0),
                phase3=(1.0, 0.0), phase4=(1.0, 0.0),
            ),
            ir_teacher=TeacherSchedule(
                phase1=(1.0, 0.0), phase2=(1.0, 0.0),
                phase3=(1.0, 0.0), phase4=(1.0, 0.0),
            ),
        ),
        ema=kwargs.pop("ema", EMAConfig(alpha=0.0, start_steps=2)),
        curriculum=curriculum,
        loss=LossConfig(
            p1_sup_weight=1.0, p1_distill_weight=0.0,
            p2_sup_weight=1.0, p2_distill_weight=1.0,
            p3_sup_weight=1.0, p3_distill_weight=1.0,
            p4_sup_weight=0.0, p4_distill_weight=1.0,
        ),
        loader=DataLoaderConfig(
            train=TrainLoaderConfig(batch_size=1, num_workers=0),
            eval=EvalLoaderConfig(batch_size=1, num_workers=0),
        ),
        aug=AugConfig(
            hflip_prob=0.0,
            blur_prob=0.0,
            brightness_prob=0.0,
            contrast_prob=0.0,
        ),
        total_iters=kwargs.pop("total_iters", 4),
        device="cpu",
        output_dir=kwargs.pop("output_dir", "outputs"),
        teacher_mode=kwargs.pop("teacher_mode", "two_teacher"),
    )


def build_trainer(config):
    student = D3TWrapper(TinyAdapter(), TinyCriterion())
    rgb_teacher = D3TWrapper(TinyAdapter(), TinyCriterion())
    ir_teacher = D3TWrapper(TinyAdapter(), TinyCriterion())
    rgb_loader = DataLoader(RGBDataset(), batch_size=1, collate_fn=collate_rgb)
    ir_loader = DataLoader(IRDataset(), batch_size=1, collate_fn=collate_ir)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
    )
    return trainer, student, rgb_teacher, ir_teacher


class CurriculumIntegrationTests(unittest.TestCase):
    def test_ema_copies_integral_buffers_and_interpolates_float_buffers(self):
        student = nn.BatchNorm1d(2)
        teacher = nn.BatchNorm1d(2)
        with torch.no_grad():
            student.running_mean.fill_(4.0)
            student.running_var.fill_(9.0)
            student.num_batches_tracked.fill_(7)
            teacher.running_mean.zero_()
            teacher.running_var.fill_(1.0)
            teacher.num_batches_tracked.zero_()
        copy_student_to_teacher(teacher, student)
        self.assertTrue(torch.equal(teacher.running_mean, student.running_mean))
        self.assertTrue(torch.equal(teacher.num_batches_tracked, student.num_batches_tracked))
        with torch.no_grad():
            student.running_mean.fill_(8.0)
            student.running_var.fill_(5.0)
            student.num_batches_tracked.fill_(11)
        ema_update(teacher, student, alpha=0.5)
        self.assertTrue(torch.allclose(teacher.running_mean, torch.full((2,), 6.0)))
        self.assertTrue(torch.allclose(teacher.running_var, torch.full((2,), 7.0)))
        self.assertTrue(torch.equal(teacher.num_batches_tracked, student.num_batches_tracked))

    def test_ema_start_kd_and_resume_lifecycle(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = build_config(output_dir=output_dir)
            trainer, student, rgb_teacher, ir_teacher = build_trainer(config)

            trainer.train_one_iteration()
            trainer.train_one_iteration()
            self.assertFalse(trainer.ema_initialized)
            self.assertEqual(rgb_teacher.adapter.forward_calls, 0)
            self.assertEqual(ir_teacher.adapter.forward_calls, 0)
            self.assertEqual(student.criterion.distill_calls, 0)

            student_before_ema_start = student.adapter.weight.item()
            trainer.train_one_iteration()
            self.assertTrue(trainer.ema_initialized)
            self.assertEqual(rgb_teacher.adapter.forward_calls, 1)
            self.assertEqual(ir_teacher.adapter.forward_calls, 0)
            self.assertEqual(student.criterion.distill_calls, 1)
            self.assertAlmostEqual(
                ir_teacher.adapter.weight.item(), student_before_ema_start
            )
            self.assertNotEqual(
                rgb_teacher.adapter.weight.item(),
                ir_teacher.adapter.weight.item(),
            )

            trainer.save_checkpoint("resume.pth")
            with mock.patch("trainer.copy_student_to_teacher") as hard_copy:
                trainer.load_checkpoint(
                    f"{output_dir}/resume.pth"
                )
                trainer._ensure_ema_initialized()
                hard_copy.assert_not_called()

    def test_trainer_can_run_with_only_the_selected_teacher(self):
        config = build_config(
            teacher_mode="rgb", ema=EMAConfig(alpha=0.0, start_steps=0)
        )
        trainer, _, rgb_teacher, _ = build_trainer(config)
        trainer.ir_teacher = None
        logs = trainer.train_one_iteration()
        self.assertTrue(trainer.ema_initialized)
        self.assertEqual(rgb_teacher.adapter.forward_calls, 1)
        self.assertIn("loss_kd_cls_rgb", logs)

    def test_p2_ir_route_keeps_rgb_ground_truth(self):
        config = build_config()
        trainer, _, _, _ = build_trainer(config)
        route = config.mid_routing.p2_ir_flow
        data = trainer._prepare_data_by_route("p2_ir_flow", route)
        self.assertIsNotNone(data["targets"])
        self.assertEqual(data["sample_ids"], ("rgb-0",))
        self.assertEqual(data["targets"][0]["stem"], "rgb-0")

    def test_ir_route_uses_one_sample_for_both_views(self):
        config = build_config()
        trainer, _, _, _ = build_trainer(config)
        route = config.mid_routing.p3_ir_flow
        data = trainer._prepare_data_by_route("p3_ir_flow", route)
        self.assertEqual(data["sample_ids"], ("ir-0",))
        self.assertTrue(torch.equal(data["student_images"], data["teacher_images"]))
        self.assertIsNone(data["targets"])

    def test_positionwise_wrapper_rejects_different_sample_ids(self):
        student = D3TWrapper(TinyAdapter(), TinyCriterion())
        teacher = D3TWrapper(TinyAdapter(), TinyCriterion()).eval()
        student_output = student.raw(
            [torch.zeros(3, 8, 8)], sample_ids=("sample-a",)
        )
        teacher_output = teacher.raw(
            [torch.zeros(3, 8, 8)], sample_ids=("sample-b",)
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            student.distill_from_outputs(student_output, teacher_output)

    def test_every_scheduler_route_is_exactly_named(self):
        config = build_config(
            curriculum=CurriculumConfig(
                phase1_end=1,
                phase2_end=3,
                phase3_end=5,
                phase2_rgb_sampling_ratio=0.5,
                phase3_rgb_sampling_ratio=0.5,
            ),
            total_iters=6,
        )
        scheduler = CurriculumScheduler(config)
        self.assertEqual(scheduler.get_next_step(0), "p1_rgb_supervised")
        self.assertIn(scheduler.get_next_step(1), {"p2_rgb_flow", "p2_ir_flow"})
        self.assertIn(scheduler.get_next_step(2), {"p2_rgb_flow", "p2_ir_flow"})
        self.assertIn(scheduler.get_next_step(3), {"p3_rgb_flow", "p3_ir_flow"})
        self.assertIn(scheduler.get_next_step(4), {"p3_rgb_flow", "p3_ir_flow"})
        self.assertEqual(scheduler.get_next_step(5), "p4_ir_focus")


if __name__ == "__main__":
    unittest.main()
