"""Training orchestration for RGB baseline and dual-teacher D3T routes."""

from collections import defaultdict
import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from config import TrainingConfig
from models import fcos_models
from models.detector_wrapper import DetectorWrapper as DetectorWrapper
from training import checkpointing
from training.data_flow import RouteDataFlow
from training.loss_flow import LossFlow
from training.observability import log_iteration
from training.step_policy import resolve_step

logger = logging.getLogger(__name__)


class CurriculumDomainAdaptationTrainer:
    """Execute step plans and own training progress, updates and persistence."""

    def __init__(
        self,
        student: nn.Module,
        rgb_teacher: Optional[nn.Module],
        ir_teacher: Optional[nn.Module],
        optimizer: Optimizer,
        config: TrainingConfig,
        data_flow: RouteDataFlow,
        scheduler: Any,
        evaluator: Any = None,
        lr_scheduler: Any = None,
    ) -> None:
        """Receive prepared models and an optimizer bound after device placement."""
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.evaluator = evaluator
        self.lr_scheduler = lr_scheduler
        self.data_flow = data_flow
        self.scheduler = scheduler
        self.global_step = 0
        # A negative sentinel preserves the first-evaluation behavior, including
        # a first measured mAP of zero. Best metrics belong only to this trainer.
        self.best_map = -1.0
        self.best_rgb_map = -1.0
        self.loss_history = defaultdict(list)
        self.ema_initialized = False
    def _teacher_models(self):
        # References stay on the trainer so checkpoint keys and ownership are unchanged.
        return {"rgb": self.rgb_teacher, "ir": self.ir_teacher}

    def _teacher_for_name(self, name: str) -> nn.Module:
        return fcos_models.get_teacher(self._teacher_models(), name)

    def train_one_iteration(self) -> Dict[str, Any]:
        self.student.train()
        plan = resolve_step(
            self.config, self.scheduler, self.global_step,
            ema_initialized=self.ema_initialized,
        )

        # Data preparation: obtain one source batch with aligned student/teacher
        # views, optional GT, and explicit sample identity.
        data = self.data_flow.prepare(
            source_domain=plan.source_domain,
            appearance_domain=plan.appearance_domain,
            use_gt=plan.use_gt,
            student_alpha=plan.student_alpha,
            teacher_alpha=plan.teacher_alpha,
        )
        targets = data["targets"]
        sample_ids = data["sample_ids"]
        iteration_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.zero_grad(set_to_none=True)

        # Establish teacher references before any forward that can consume them.
        if plan.initialize_teacher_names:
            fcos_models.initialize_teachers(
                self.student, self._teacher_models(), names=plan.initialize_teacher_names,
            )
            self.ema_initialized = True

        # Student forward: choose resize once and reuse this raw output for
        # every applicable loss, avoiding a second student forward for KD.
        select_resize = getattr(
            self.student, "select_training_resize_short_edge", None
        )
        resize_short_edge = None if select_resize is None else select_resize()
        student_raw_kwargs = {
            "sample_ids": sample_ids,
        }
        if resize_short_edge is not None:
            student_raw_kwargs["resize_short_edge"] = resize_short_edge
        student_output = self.student.raw(
            data["student_images"], targets, **student_raw_kwargs
        )
        # Each iteration gets a fresh accumulator; the plan supplies phase weights.
        losses = LossFlow(
            uhl=self.config.loss.weight_uhl,
            quality=self.config.loss.weight_quality,
            deltas=self.config.loss.weight_deltas,
            logits=self.config.loss.weight_logits,
        )
        # Supervised contribution: use transformed GT only when this route and
        # the effective phase weight request it.
        if targets is not None and plan.use_gt and plan.supervised_weight > 0:
            losses.supervised(
                self.student, student_output, targets, phase_weight=plan.supervised_weight,
            )

        # Distillation contribution: use only enabled, initialized references.
        # Each teacher consumes the paired view with the student's resize.
        for teacher_name, settings in plan.distillation:
            teacher = self._teacher_for_name(teacher_name)
            teacher_output = fcos_models.forward_teacher(
                teacher,
                data["teacher_images"],
                sample_ids=sample_ids,
                resize_short_edge=resize_short_edge,
            )
            losses.distill(
                self.student,
                student_output,
                teacher_output,
                settings,
                teacher_name=teacher_name,
                phase_weight=plan.distill_weight,
            )

        # Preserve the existing optimizer eligibility rule, including a scalar
        # fallback when this route produced no differentiable contributions.
        total_loss = losses.total_loss
        logs = losses.logs

        if total_loss is None:
            total_loss = torch.zeros((), device=self.device)
        did_step = False
        # Student update: backward, clipping, optimizer, and LR scheduling stay
        # together and keep their original order within a successful step.
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

        # Teacher update: follow an eligible student update with EMA only for
        # the teacher targeted by this route and enabled by the experiment.
        if did_step and plan.ema_target is not None:
            fcos_models.ema_update(
                self._teacher_for_name(plan.ema_target),
                self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step,
            )

        # Observation and progress: collect detached values after updates, then
        # advance the global step once and return the existing metric payload.
        logs["total_loss"] = float(total_loss.detach().item())
        logs["step_type"] = plan.step_name
        logs["phase"] = plan.phase.name
        logs["ema_initialized"] = float(self.ema_initialized)
        logs["lr"] = iteration_lr
        logs.update(plan.log_metadata(did_step=did_step, ema_initialized=self.ema_initialized))
        self.loss_history[plan.step_name].append(logs["total_loss"])
        self.global_step += 1
        logs["global_step"] = self.global_step
        if self.evaluator is not None:
            records = self.evaluator.audit(
                self.global_step, plan.phase, ema_initialized=self.ema_initialized,
            )
            if records:
                logs["pseudo_label_evaluations"] = records
        return logs

    def evaluate(self, *, current_phase=None, trigger_reason: str = "manual") -> Optional[Dict[str, Any]]:
        """Complete evaluation, update best metrics, then persist an IR improvement."""
        if self.evaluator is None:
            return None
        if current_phase is None:
            current_phase = self.config.get_phase(self.global_step)
        if trigger_reason == "scheduled":
            results = self.evaluator.step(self.student, self.global_step, current_phase)
        else:
            results = self.evaluator.evaluate(
                self.student, self.global_step, current_phase, trigger_reason=trigger_reason,
            )
        if results is None:
            return None

        ir_map = results.get("mAP@0.5")
        is_best_ir = ir_map is not None and ir_map > self.best_map
        if is_best_ir:
            self.best_map = ir_map
            results["is_best_ir"] = True
        rgb_map = results.get("rgb_mAP@0.5")
        if rgb_map is not None and rgb_map > self.best_rgb_map:
            self.best_rgb_map = rgb_map
            results["is_best_rgb"] = True
        if is_best_ir:
            self.save_checkpoint("best_model.pth")
            logger.info("New best IR mAP@0.5 saved: %.4f", self.best_map)
        return results

    def train(self) -> None:
        logger.info(
            "Starting training: total_iters=%s starting_from_step=%s workflow=%s",
            self.config.max_iter,
            self.global_step,
            self.config.workflow,
        )
        for _ in range(self.global_step, self.config.max_iter):
            # Execute one iteration before applying the existing log, validation,
            # and checkpoint cadence to the resulting global step.
            started = time.time()
            logs = self.train_one_iteration()
            if self.global_step % self.config.log_interval == 0:
                logs["iter_time"] = time.time() - started
                self._log(logs)
            # Initial/manual and scheduled validation share the same best/save
            # ordering. Domain metrics and history finish before checkpointing.
            self.evaluate(trigger_reason="scheduled")
            # Preserve the current periodic snapshot cadence alongside the
            # separately triggered best-model and final-model saves.
            if self.global_step > 0 and self.global_step % 5000 == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step:06d}.pth")

        if self.evaluator is not None:
            self.evaluator.audit(
                self.global_step, self.config.get_phase(max(0, self.global_step - 1)),
                ema_initialized=self.ema_initialized, final=True,
            )

    def save_checkpoint(self, filename: str) -> None:
        checkpointing.save_checkpoint(self, filename)

    def load_checkpoint(self, path: str) -> None:
        checkpointing.load_checkpoint(self, path)

    def _log(self, log: Dict[str, Any]) -> None:
        log_iteration(log, self.global_step, logger=logger)
