"""Training orchestration for RGB baseline and dual-teacher D3T routes."""

from collections import defaultdict
import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import Phase, StepRouting, TrainingConfig
from models.d3t_adapter import DistillationSettings
from models.d3t_wrapper import D3TWrapper
from scheduler import CurriculumScheduler, DomainStep
from data.augmentations import StudentAugmentor
from training import teachers
from training.data_flow import RouteDataFlow
from training.loss_flow import LossFlow, component_weight
from training.observability import log_iteration

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
        phase_evaluator: Any = None,
        lr_scheduler: Any = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.phase_evaluator = phase_evaluator
        self.lr_scheduler = lr_scheduler
        self.augmentor = StudentAugmentor(config)

        self._setup_models()
        self.data_flow = RouteDataFlow(
            rgb_loader, ir_loader, device=self.device, augmentor=self.augmentor
        )
        self.scheduler = CurriculumScheduler(config)
        self.global_step = 0
        self.best_map = 0.0
        self.loss_history = defaultdict(list)
        self.ema_initialized = False

    def _teacher_models(self):
        # Keep live references on the trainer to preserve model/checkpoint ownership.
        return {"rgb": self.rgb_teacher, "ir": self.ir_teacher}

    def _setup_models(self) -> None:
        teachers.prepare_models(self.student, self._teacher_models(), device=self.device)

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
        """Resolve configured values before delegating data execution."""
        source_domain = self.scheduler.get_source_domain(step_name)
        alpha_map = self._alpha_map()
        return self.data_flow.prepare(
            source_domain=source_domain,
            use_gt=route.use_gt,
            student_alpha=(
                alpha_map[route.student_saga_level] if source_domain == "rgb" else None
            ),
            teacher_alpha=(
                alpha_map[route.teacher_saga_level] if source_domain == "rgb" else None
            ),
        )

    def _ensure_ema_initialized(self) -> None:
        if self.ema_initialized:
            return
        if self.global_step < self.config.ema.start_steps:
            return
        if self.config.teacher_mode == "two_teacher":
            active_names = ("rgb", "ir")
        else:
            active_names = (self.config.teacher_mode,)
        teachers.initialize_teachers(
            self.student, self._teacher_models(), names=active_names
        )
        self.ema_initialized = True

    def _enabled_teacher_names(self, route: StepRouting) -> tuple[str, ...]:
        if self.config.teacher_mode == "two_teacher":
            enabled = {"rgb", "ir"}
        else:
            enabled = {self.config.teacher_mode}
        return tuple(name for name in route.teacher_names if name in enabled)

    def _teacher_for_name(self, name: str) -> nn.Module:
        return teachers.get_teacher(self._teacher_models(), name)

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
        """Keep the existing helper while passing explicit weights to the executor."""
        return component_weight(
            name, uhl=config.weight_uhl, quality=config.weight_quality,
            deltas=config.weight_deltas, logits=config.weight_logits,
        )

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
        iteration_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.zero_grad(set_to_none=True)

        if self.config.workflow != "rgb_baseline":
            self._ensure_ema_initialized()

        select_resize = getattr(
            self.student, "select_training_resize_short_edge", None
        )
        resize_short_edge = None if select_resize is None else select_resize()
        student_raw_kwargs = {"sample_ids": sample_ids}
        if resize_short_edge is not None:
            student_raw_kwargs["resize_short_edge"] = resize_short_edge
        student_output = self.student.raw(
            data["student_images"], targets, **student_raw_kwargs
        )
        # Resolve configuration here; the loss executor only receives values.
        losses = LossFlow(
            uhl=self.config.loss.weight_uhl,
            quality=self.config.loss.weight_quality,
            deltas=self.config.loss.weight_deltas,
            logits=self.config.loss.weight_logits,
        )
        sup_weight, distill_weight = self.config.loss.get_phase_weights(phase)
        if self.config.workflow == "rgb_baseline":
            sup_weight, distill_weight = 1.0, 0.0

        if targets is not None and route.use_gt and sup_weight > 0:
            losses.supervised(
                self.student, student_output, targets, phase_weight=sup_weight
            )

        if (
            self.config.workflow != "rgb_baseline"
            and self.ema_initialized
            and distill_weight > 0
        ):
            enabled_teacher_names = self._enabled_teacher_names(route)
            for teacher_name in enabled_teacher_names:
                teacher = self._teacher_for_name(teacher_name)
                teacher_output = teachers.forward_teacher(
                    teacher,
                    data["teacher_images"],
                    sample_ids=sample_ids,
                    resize_short_edge=resize_short_edge,
                )
                losses.distill(
                    self.student,
                    student_output,
                    teacher_output,
                    self._distillation_settings(teacher_name, phase),
                    teacher_name=teacher_name,
                    phase_weight=distill_weight,
                )

        total_loss = losses.total_loss
        logs = losses.logs

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
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            did_step = True

        if (
            did_step
            and self.ema_initialized
            and route.ema_target in self._enabled_teacher_names(route)
        ):
            teachers.update_teacher(
                self._teacher_for_name(route.ema_target),
                self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step,
            )

        logs["total_loss"] = float(total_loss.detach().item())
        logs["step_type"] = step_name
        logs["phase"] = phase.name
        logs["ema_initialized"] = float(self.ema_initialized)
        logs["lr"] = iteration_lr
        self.loss_history[step_name].append(logs["total_loss"])
        self.global_step += 1
        logs["global_step"] = self.global_step
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
            "lr_scheduler": (
                None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
            ),
            "phase_evaluator": (
                self.phase_evaluator.state_dict()
                if self.phase_evaluator is not None
                and hasattr(self.phase_evaluator, "state_dict")
                else None
            ),
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
        configured_teachers = {
            "rgb_teacher": self.rgb_teacher,
            "ir_teacher": self.ir_teacher,
        }
        missing_teacher_states = [
            name
            for name, teacher in configured_teachers.items()
            if teacher is not None and checkpoint.get(name) is None
        ]
        if missing_teacher_states:
            missing = ", ".join(missing_teacher_states)
            raise ValueError(
                f"Checkpoint is missing state for configured teacher(s): {missing}"
            )
        self.student.load_state_dict(checkpoint["student"])
        for name, teacher in configured_teachers.items():
            if teacher is not None:
                teacher.load_state_dict(checkpoint[name])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.lr_scheduler is not None:
            lr_scheduler_state = checkpoint.get("lr_scheduler")
            if lr_scheduler_state is None:
                raise ValueError("Checkpoint is missing LR scheduler state")
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
        evaluator_state = checkpoint.get("phase_evaluator")
        if (
            evaluator_state is not None
            and self.phase_evaluator is not None
            and hasattr(self.phase_evaluator, "load_state_dict")
        ):
            self.phase_evaluator.load_state_dict(evaluator_state)
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_map = float(checkpoint.get("best_map", 0.0))
        self.ema_initialized = bool(checkpoint.get("ema_initialized", False))
        logger.info("Resumed from step %s", self.global_step)

    def _log(self, log: Dict[str, Any]) -> None:
        # Presentation stays separate from iteration execution and log cadence.
        log_iteration(log, self.global_step, logger=logger)
