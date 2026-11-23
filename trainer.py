"""Training orchestration for RGB baseline and dual-teacher D3T routes."""

from collections import defaultdict
import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import TrainingConfig
from models import fcos_factory
from models.detector_wrapper import DetectorWrapper as DetectorWrapper
from scheduler import BridgeDomainScheduler, CurriculumScheduler, should_evaluate_pseudo_labels
from data.augmentations import StudentAugmentor
from training import checkpointing
from training.data_flow import RouteDataFlow
from training.loss_flow import LossFlow, component_weight
from training.observability import log_iteration, log_pseudo_label_evaluation
from training.step_policy import configured_teacher_names, distillation_settings, resolve_step

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
        rgb_loader: DataLoader,
        ir_loader: Optional[DataLoader] = None,
        phase_evaluator: Any = None,
        lr_scheduler: Any = None,
        gan_translator: Optional[nn.Module] = None,
        pseudo_label_evaluator: Any = None,
    ) -> None:
        """Receive prepared models and an optimizer bound after device placement."""
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.phase_evaluator = phase_evaluator
        self.lr_scheduler = lr_scheduler
        self.augmentor = StudentAugmentor(config)
        self.gan_translator = gan_translator
        if config.workflow == "gan_bridge":
            if gan_translator is None or not getattr(gan_translator, "identity", None):
                raise ValueError("gan_bridge requires a frozen GAN translator with checkpoint identity")
            if rgb_teacher is None or ir_teacher is None:
                raise ValueError("gan_bridge requires both RGB and IR teachers")

        self.data_flow = RouteDataFlow(
            rgb_loader, ir_loader, device=self.device, augmentor=self.augmentor,
            gan_translator=gan_translator,
        )
        # Own the progress and lifecycle state that must survive checkpoint
        # save/resume; execution modules do not maintain competing copies.
        self.scheduler = self._new_scheduler()
        self.global_step = 0
        self.best_map = 0.0
        self.loss_history = defaultdict(list)
        self.ema_initialized = False
        # Bootstrap supplies isolated IR validation loaders. Audit progress stays
        # separate from student validation history and best-checkpoint selection.
        self.last_pseudo_label_eval_step = 0
        self.last_full_pseudo_label_eval_step = 0
        self.pseudo_label_evaluator = pseudo_label_evaluator
        if config.pseudo_label_eval.enabled and config.workflow != "rgb_baseline" and pseudo_label_evaluator is None:
            raise ValueError("enabled pseudo-label evaluation requires an IR validation evaluator")

    def _new_scheduler(self):
        if self.config.workflow == "gan_bridge":
            return BridgeDomainScheduler(self.config.bridge_schedule, self.config.max_iter)
        return CurriculumScheduler(self.config)

    def _teacher_models(self):
        # References stay on the trainer so checkpoint keys and ownership are unchanged.
        return {"rgb": self.rgb_teacher, "ir": self.ir_teacher}

    def _teacher_for_name(self, name: str) -> nn.Module:
        return fcos_factory.get_teacher(self._teacher_models(), name)

    @staticmethod
    def _component_weight(name: str, config) -> float:
        """Keep the existing helper while passing explicit weights to the executor."""
        return component_weight(
            name, uhl=config.weight_uhl, quality=config.weight_quality,
            deltas=config.weight_deltas, logits=config.weight_logits,
        )

    def train_one_iteration(self) -> Dict[str, Any]:
        if self.config.workflow != "gan_bridge":
            return self._train_one_iteration()
        schedule_state = self.scheduler.state_dict()
        starting_step = self.global_step
        try:
            return self._train_one_iteration()
        except BaseException:
            # Roll back routing only when the iteration did not complete.
            # A later audit failure must retain the already committed update.
            if self.global_step == starting_step:
                self.scheduler.load_state_dict(schedule_state)
            raise

    def _train_one_iteration(self) -> Dict[str, Any]:
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
            fcos_factory.initialize_teachers(
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
            teacher_output = fcos_factory.forward_teacher(
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
            fcos_factory.ema_update(
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
        # Observe the current teachers after the iteration/EMA update using a
        # separate fixed dataset, never the train batch or supervised targets.
        if (
            self.pseudo_label_evaluator is not None and self.ema_initialized
            and should_evaluate_pseudo_labels(
                self.config.pseudo_label_eval, self.global_step - 1,
                last_eval_step=self.last_pseudo_label_eval_step,
            )
        ):
            logs["pseudo_label_evaluations"] = self._evaluate_pseudo_labels(plan.phase)
        return logs

    def _evaluate_pseudo_labels(self, phase, *, full_dataset=False):
        """Execute isolated IR evaluation with the shared teacher/settings policy."""
        names = configured_teacher_names(self.config)
        records = []
        for name in names:
            settings = distillation_settings(self.config, name, phase)
            result = self.pseudo_label_evaluator.evaluate(
                self._teacher_for_name(name), settings,
                iou_threshold=self.config.pseudo_label_eval.iou_threshold,
                resize_short_edge=self.config.model.min_sizes[-1], full_dataset=full_dataset,
            )
            records.append({
                **result, "global_step": self.global_step, "phase": phase.name,
                "source_domain": "ir", "teacher": name,
                "trigger": "final" if full_dataset else "periodic",
                "top_ratio": settings.top_ratio, "min_hm": settings.min_hm,
                "hm_alpha": settings.hm_alpha, "hm_beta": settings.hm_beta,
                "resize_short_edge": self.config.model.min_sizes[-1],
                "iou_threshold": self.config.pseudo_label_eval.iou_threshold,
            })
        for record in records:
            log_pseudo_label_evaluation(record, output_dir=self.config.output_dir, logger=logger)
        if full_dataset:
            self.last_full_pseudo_label_eval_step = self.global_step
        else:
            self.last_pseudo_label_eval_step = self.global_step
        return records

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
            # Delegate evaluation triggers and best-model callbacks to the
            # evaluator, then retain the best score in trainer checkpoint state.
            if self.phase_evaluator is not None:
                phase = self.config.get_phase(self.global_step)
                results = self.phase_evaluator.step(
                    self.student, self.global_step, phase
                )
                if results and "mAP@0.5" in results:
                    self.best_map = max(self.best_map, results["mAP@0.5"])
            # Preserve the current periodic snapshot cadence alongside the
            # separately triggered best-model and final-model saves.
            if self.global_step > 0 and self.global_step % 5000 == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step:06d}.pth")

        # A normal completed run also measures every IR validation image once.
        # Persisting this flag makes resuming a completed final model idempotent.
        if (self.pseudo_label_evaluator is not None and self.ema_initialized
                and self.global_step >= self.config.max_iter
                and self.last_full_pseudo_label_eval_step != self.global_step):
            self._evaluate_pseudo_labels(
                self.config.get_phase(max(0, self.global_step - 1)), full_dataset=True,
            )

    def save_checkpoint(self, filename: str) -> None:
        checkpointing.save_checkpoint(self, filename)

    def load_checkpoint(self, path: str) -> None:
        checkpointing.load_checkpoint(self, path)

    def _log(self, log: Dict[str, Any]) -> None:
        log_iteration(log, self.global_step, logger=logger)
