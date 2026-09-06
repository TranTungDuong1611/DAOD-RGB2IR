"""Training orchestration for RGB baseline and dual-teacher D3T routes."""

from collections import defaultdict
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import Phase, StepRouting, TrainingConfig
from ema import copy_student_to_teacher, ema_update
from models.d3t_adapter import CriterionResult, DistillationSettings
from models.d3t_wrapper import D3TWrapper
from scheduler import CurriculumScheduler, DomainStep
from data.augmentations import StudentAugmentor

logger = logging.getLogger(__name__)


class CurriculumDomainAdaptationTrainer:
    """Own routing, optimization, EMA, checkpointing, and evaluation hooks."""

    def __init__(
        self,
        student: nn.Module,
        rgb_teacher: Optional[nn.Module],
        ir_teacher: Optional[nn.Module],
        optimizer: Optimizer,
        config: TrainingConfig,
        rgb_loader: DataLoader,
        ir_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        distill_adapter: Any = None,
        phase_evaluator: Any = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.val_loader = val_loader
        self.phase_evaluator = phase_evaluator
        # Kept as a non-operational compatibility argument so old callers fail
        # by behavior rather than by import; the new trainer never uses it.
        self.distill_adapter = None
        self.augmentor = StudentAugmentor(config)

        self._setup_models()
        self._rgb_iter = iter(self._infinite(rgb_loader))
        self._ir_iter = None if ir_loader is None else iter(self._infinite(ir_loader))
        self.scheduler = CurriculumScheduler(config)
        self.global_step = 0
        self.best_map = 0.0
        self.loss_history = defaultdict(list)
        self.ema_initialized = False

    def _setup_models(self) -> None:
        self.student.to(self.device)
        for teacher in (self.rgb_teacher, self.ir_teacher):
            if teacher is None:
                continue
            teacher.to(self.device)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _infinite(loader: Iterable):
        while True:
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("training DataLoader is empty")

    def _move_images(self, images):
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        return tuple(image.to(self.device) for image in images)

    def _move_targets(self, targets):
        moved = []
        for target in targets:
            moved.append({
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            })
        return moved

    @staticmethod
    def _ids_from_targets(targets) -> tuple[str, ...]:
        ids = tuple(target.get("stem") for target in targets)
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("labeled batches must carry one non-empty sample stem per image")
        return ids

    def _unpack_rgb_batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("RGB loader must return (images, targets[, sample_ids])")
        images, targets = batch[0], list(batch[1])
        ids = tuple(batch[2]) if len(batch) == 3 else self._ids_from_targets(targets)
        if len(ids) != len(targets):
            raise ValueError("RGB sample_ids must match target count")
        return self._move_images(images), self._move_targets(targets), ids

    def _unpack_ir_batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) not in (1, 2, 3):
            raise ValueError("IR loader must return (images[, sample_ids])")
        images = self._move_images(batch[0])
        if len(batch) >= 2:
            ids = tuple(batch[1])
        else:
            raise ValueError("unlabeled IR batches must carry sample_ids")
        image_count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        if len(ids) != image_count or any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("IR sample_ids must contain one non-empty ID per image")
        return images, ids

    def _get_saga_images(self, images, targets, alpha: float):
        if targets is None:
            return images.clone()
        from saga import SoftSAGA
        boxes_list = [target["boxes"] for target in targets]
        return SoftSAGA().apply_to_batch(images, boxes_list, alpha)

    def _alpha_map(self):
        ss_cfg = self.config.soft_saga
        return {
            "rgb": 1.0,
            "weak": ss_cfg.alpha_near_rgb,
            "mid": ss_cfg.alpha_intermediate,
            "high": ss_cfg.alpha_near_ir,
            "ir": 0.0,
        }

    def _prepare_data_by_route(self, step_name: str, route: StepRouting):
        """Create aligned student/teacher views from exactly one source batch."""

        alpha_map = self._alpha_map()
        rgb_source_steps = {
            "p1_rgb_supervised",
            "p2_rgb_flow",
            "p2_ir_flow",
            "p3_rgb_flow",
        }
        if step_name in rgb_source_steps:
            images, targets, sample_ids = self._unpack_rgb_batch(next(self._rgb_iter))
            geometric_images, geometric_targets, _ = self.augmentor.apply_weak_aug(
                images, targets
            )
            student_images = self._get_saga_images(
                geometric_images,
                geometric_targets,
                alpha_map[route.student_saga_level],
            )
            teacher_images = self._get_saga_images(
                geometric_images,
                geometric_targets,
                alpha_map[route.teacher_saga_level],
            )
            student_images = self.augmentor.apply_photometric_aug(student_images)
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": geometric_targets if route.use_gt else None,
                "sample_ids": sample_ids,
            }

        if step_name in {"p3_ir_flow", "p4_ir_focus"}:
            if self._ir_iter is None:
                raise ValueError("IR route requested without an IR DataLoader")
            images, sample_ids = self._unpack_ir_batch(next(self._ir_iter))
            geometric_images, _, _ = self.augmentor.apply_weak_aug(images)
            teacher_images = geometric_images.clone()
            student_images = self.augmentor.apply_photometric_aug(geometric_images)
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": None,
                "sample_ids": sample_ids,
            }
        raise ValueError(f"Unsupported curriculum step: {step_name}")

    def _ensure_ema_initialized(self) -> None:
        if self.ema_initialized:
            return
        if self.global_step < self.config.ema.start_steps:
            return
        active_teachers = {
            "rgb": self.rgb_teacher,
            "ir": self.ir_teacher,
        }
        if self.config.teacher_mode == "two_teacher":
            active_names = ("rgb", "ir")
        else:
            active_names = (self.config.teacher_mode,)
        for name in active_names:
            teacher = active_teachers[name]
            if teacher is None:
                raise ValueError(f"EMA start requires the '{name}' teacher")
            copy_student_to_teacher(teacher, self.student)
        self.ema_initialized = True

    def _enabled_teacher_names(self, route: StepRouting) -> tuple[str, ...]:
        if self.config.teacher_mode == "two_teacher":
            enabled = {"rgb", "ir"}
        else:
            enabled = {self.config.teacher_mode}
        return tuple(name for name in route.teacher_names if name in enabled)

    def _teacher_for_name(self, name: str) -> nn.Module:
        teacher = {"rgb": self.rgb_teacher, "ir": self.ir_teacher}.get(name)
        if teacher is None:
            raise ValueError(f"teacher '{name}' is not configured")
        teacher.eval()
        return teacher

    def _distillation_settings(self, teacher_name: str, phase: Phase):
        schedule = getattr(self.config.distill, f"{teacher_name}_teacher")
        top_ratio, min_hm = schedule.get_params(phase)
        return DistillationSettings(
            top_ratio=top_ratio,
            min_hm=min_hm,
            hm_alpha=self.config.distill.hm_alpha,
            hm_beta=self.config.distill.hm_beta,
            uncertainty_alpha=self.config.distill.un_regular_alpha,
        )

    @staticmethod
    def _component_weight(name: str, config) -> float:
        if "quality" in name:
            return config.weight_quality
        if "box" in name or "delta" in name:
            return config.weight_deltas
        if "cls" in name or "logit" in name:
            return config.weight_logits
        return 1.0

    @staticmethod
    def _as_result(value) -> CriterionResult:
        if isinstance(value, CriterionResult):
            return value
        if isinstance(value, dict):
            return CriterionResult(value, {})
        raise TypeError("wrapper distillation must return CriterionResult")

    @staticmethod
    def _add_loss(total, value):
        return value if total is None else total + value

    def train_one_iteration(self) -> Dict[str, float]:
        self.student.train()
        if self.config.workflow == "rgb_baseline":
            step_name: DomainStep = "p1_rgb_supervised"
        else:
            step_name = self.scheduler.get_next_step(self.global_step)
        phase = self.config.get_phase(self.global_step)
        route = self.config.mid_routing.get_routing(step_name)
        if self.config.workflow == "rgb_baseline":
            route = self.config.mid_routing.p1_rgb_supervised

        data = self._prepare_data_by_route(step_name, route)
        targets = data["targets"]
        sample_ids = data["sample_ids"]
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.workflow != "rgb_baseline":
            self._ensure_ema_initialized()

        student_output = self.student.raw(
            data["student_images"],
            targets,
            sample_ids=sample_ids,
        )
        total_loss = None
        logs: Dict[str, float] = {}
        sup_weight, distill_weight = self.config.loss.get_phase_weights(phase)
        if self.config.workflow == "rgb_baseline":
            sup_weight, distill_weight = 1.0, 0.0

        if targets is not None and route.use_gt and sup_weight > 0:
            supervised_losses = self.student.supervised_from_output(
                student_output, targets
            )
            for name, value in supervised_losses.items():
                weighted = value * sup_weight * self._component_weight(
                    name, self.config.loss
                )
                total_loss = self._add_loss(total_loss, weighted)
                logs[f"sup_{name}"] = float(weighted.detach().item())

        if (
            self.config.workflow != "rgb_baseline"
            and self.ema_initialized
            and distill_weight > 0
        ):
            enabled_teacher_names = self._enabled_teacher_names(route)
            for teacher_name in enabled_teacher_names:
                teacher = self._teacher_for_name(teacher_name)
                with torch.no_grad():
                    teacher_output = teacher.raw(
                        data["teacher_images"],
                        sample_ids=sample_ids,
                    )
                result = self._as_result(
                    self.student.distill_from_outputs(
                        student_output,
                        teacher_output,
                        self._distillation_settings(teacher_name, phase),
                    )
                )
                suffix = f"_{teacher_name}"
                for name, value in result.losses.items():
                    weighted = value * distill_weight * self._component_weight(
                        name, self.config.loss
                    )
                    total_loss = self._add_loss(total_loss, weighted)
                    logs[f"{name}{suffix}"] = float(weighted.detach().item())
                for name, value in result.metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        logs[f"{name}{suffix}"] = float(value.detach().item())

        if total_loss is None:
            total_loss = torch.zeros((), device=self.device)
        did_step = False
        if total_loss.requires_grad and bool(torch.isfinite(total_loss).item()):
            total_loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), self.config.grad_clip
                )
            self.optimizer.step()
            did_step = True

        if (
            did_step
            and self.ema_initialized
            and route.ema_target in self._enabled_teacher_names(route)
        ):
            ema_update(
                self._teacher_for_name(route.ema_target),
                self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step,
            )

        logs["total_loss"] = float(total_loss.detach().item())
        logs["step_type"] = step_name
        logs["phase"] = phase.name
        logs["global_step"] = self.global_step
        logs["ema_initialized"] = float(self.ema_initialized)
        self.loss_history[step_name].append(logs["total_loss"])
        self.global_step += 1
        return logs

    def train(self) -> None:
        logger.info(
            "Starting training: total_iters=%s starting_from_step=%s workflow=%s",
            self.config.max_iter,
            self.global_step,
            self.config.workflow,
        )
        for _ in range(self.global_step, self.config.max_iter):
            started = time.time()
            logs = self.train_one_iteration()
            if self.global_step % self.config.log_interval == 0:
                logs["iter_time"] = time.time() - started
                logs["lr"] = self.optimizer.param_groups[0]["lr"]
                self._log(logs)
            if self.phase_evaluator is not None:
                phase = self.config.get_phase(self.global_step)
                results = self.phase_evaluator.step(
                    self.student, self.global_step, phase
                )
                if results and "mAP@0.5" in results:
                    self.best_map = max(self.best_map, results["mAP@0.5"])
            if self.global_step > 0 and self.global_step % 5000 == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step:06d}.pth")

    def save_checkpoint(self, filename: str) -> None:
        from models.fcos_factory import build_checkpoint_metadata

        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)
        payload = {
            "metadata": build_checkpoint_metadata(self.config),
            "global_step": self.global_step,
            "best_map": self.best_map,
            "ema_initialized": self.ema_initialized,
            "student": self.student.state_dict(),
            "rgb_teacher": None if self.rgb_teacher is None else self.rgb_teacher.state_dict(),
            "ir_teacher": None if self.ir_teacher is None else self.ir_teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(payload, path)
        logger.info("Checkpoint saved -> %s", path)

    def load_checkpoint(self, path: str) -> None:
        from models.fcos_factory import validate_checkpoint_metadata

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get("metadata")
        if metadata is None:
            raise ValueError("Checkpoint has no D3T metadata; legacy checkpoints are unsupported")
        validate_checkpoint_metadata(metadata, self.config)
        self.student.load_state_dict(checkpoint["student"])
        if self.rgb_teacher is not None and checkpoint.get("rgb_teacher") is not None:
            self.rgb_teacher.load_state_dict(checkpoint["rgb_teacher"])
        if self.ir_teacher is not None and checkpoint.get("ir_teacher") is not None:
            self.ir_teacher.load_state_dict(checkpoint["ir_teacher"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_map = float(checkpoint.get("best_map", 0.0))
        self.ema_initialized = bool(checkpoint.get("ema_initialized", False))
        logger.info("Resumed from step %s", self.global_step)

    def _log(self, log: Dict[str, Any]) -> None:
        components = []
        for key, value in sorted(log.items()):
            if isinstance(value, float) and ("loss" in key or "kd_" in key):
                components.append(f"{key}={value:.4f}")
        message = (
            f"[{int(log.get('global_step', self.global_step)):06d}] "
            f"Phase: {log.get('phase', 'N/A'):<22} | "
            f"Step: {log.get('step_type', 'N/A'):<18} | "
            f"Loss: {log.get('total_loss', 0.0):.4f}"
        )
        if components:
            message += " | " + " | ".join(components)
        logger.info(message)
