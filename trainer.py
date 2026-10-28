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
from models.d3t_wrapper import D3TWrapper as D3TWrapper
from scheduler import CurriculumScheduler, DomainStep
from data.augmentations import StudentAugmentor
from training import teachers
from training.data_flow import RouteDataFlow
from training.loss_flow import LossFlow, component_weight
from training.observability import log_iteration, log_pseudo_label_evaluation

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
        pseudo_label_evaluator: Any = None,
    ) -> None:
        # Retain the live dependencies and effective configuration on the
        # orchestrator so execution modules receive resolved inputs per step.
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.phase_evaluator = phase_evaluator
        self.lr_scheduler = lr_scheduler
        self.augmentor = StudentAugmentor(config)

        # Prepare model state before constructing lazy data cursors. Their
        # creation order is kept stable to preserve the existing RNG behavior.
        self._setup_models()
        self.data_flow = RouteDataFlow(
            rgb_loader, ir_loader, device=self.device, augmentor=self.augmentor
        )
        # Own the progress and lifecycle state that must survive checkpoint
        # save/resume; execution modules do not maintain competing copies.
        self.scheduler = CurriculumScheduler(config)
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

    def _teacher_models(self):
        # References stay on the trainer so checkpoint keys and ownership are unchanged.
        return {"rgb": self.rgb_teacher, "ir": self.ir_teacher}

    def _setup_models(self) -> None:
        teachers.prepare_models(self.student, self._teacher_models(), device=self.device)

    def _alpha_map(self):
        # Resolve appearance-level names from config before invoking data flow;
        # these levels describe SAGA strength, not the physical input domain.
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
        # Decide whether initialization is due from persisted state and config.
        # A resumed, initialized run must continue without another hard copy.
        if self.ema_initialized:
            return
        if self.global_step < self.config.ema.start_steps:
            return
        # Select the enabled references here, then ask the teacher executor to
        # copy weights; schedule decisions remain on the orchestration side.
        if self.config.teacher_mode == "two_teacher":
            active_names = ("rgb", "ir")
        else:
            active_names = (self.config.teacher_mode,)
        teachers.initialize_teachers(
            self.student, self._teacher_models(), names=active_names
        )
        self.ema_initialized = True

    def _enabled_teacher_names(self, route: StepRouting) -> tuple[str, ...]:
        # Intersect this route's teacher requests with the experiment's enabled
        # teacher mode while preserving the configured route order.
        if self.config.teacher_mode == "two_teacher":
            enabled = {"rgb", "ir"}
        else:
            enabled = {self.config.teacher_mode}
        return tuple(name for name in route.teacher_names if name in enabled)

    def _teacher_for_name(self, name: str) -> nn.Module:
        return teachers.get_teacher(self._teacher_models(), name)

    def _distillation_settings(self, teacher_name: str, phase: Phase):
        # Resolve teacher-specific selection thresholds and shared KD settings
        # for this phase before passing them to the loss executor.
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

    def train_one_iteration(self) -> Dict[str, Any]:
        # Route selection: the baseline uses its fixed supervised route, while
        # curriculum training asks the scheduler for the current step's route.
        self.student.train()
        if self.config.workflow == "rgb_baseline":
            step_name: DomainStep = "p1_rgb_supervised"
        else:
            step_name = self.scheduler.get_next_step(self.global_step)
        phase = self.config.get_phase(self.global_step)
        route = self.config.mid_routing.get_routing(step_name)
        if self.config.workflow == "rgb_baseline":
            route = self.config.mid_routing.p1_rgb_supervised

        # Data preparation: obtain one source batch with aligned student/teacher
        # views, optional GT, and explicit sample identity.
        data = self._prepare_data_by_route(step_name, route)
        targets = data["targets"]
        sample_ids = data["sample_ids"]
        iteration_lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.zero_grad(set_to_none=True)

        # Establish teacher references before any forward that can consume them.
        if self.config.workflow != "rgb_baseline":
            self._ensure_ema_initialized()

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
        # Loss setup: resolve phase/component weights in the orchestrator and
        # give the executor a fresh accumulator for this iteration only.
        losses = LossFlow(
            uhl=self.config.loss.weight_uhl,
            quality=self.config.loss.weight_quality,
            deltas=self.config.loss.weight_deltas,
            logits=self.config.loss.weight_logits,
        )
        sup_weight, distill_weight = self.config.loss.get_phase_weights(phase)
        if self.config.workflow == "rgb_baseline":
            sup_weight, distill_weight = 1.0, 0.0

        # Supervised contribution: use transformed GT only when this route and
        # the effective phase weight request it.
        if targets is not None and route.use_gt and sup_weight > 0:
            losses.supervised(
                self.student, student_output, targets, phase_weight=sup_weight
            )

        # Distillation contribution: use only enabled, initialized references.
        # Each teacher consumes the paired view with the student's resize.
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
                settings = self._distillation_settings(teacher_name, phase)
                losses.distill(
                    self.student,
                    student_output,
                    teacher_output,
                    settings,
                    teacher_name=teacher_name,
                    phase_weight=distill_weight,
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

        # Observation and progress: collect detached values after updates, then
        # advance the global step once and return the existing metric payload.
        logs["total_loss"] = float(total_loss.detach().item())
        logs["step_type"] = step_name
        logs["phase"] = phase.name
        logs["ema_initialized"] = float(self.ema_initialized)
        logs["lr"] = iteration_lr
        self.loss_history[step_name].append(logs["total_loss"])
        self.global_step += 1
        logs["global_step"] = self.global_step
        # Observe the current teachers after the iteration/EMA update using a
        # separate fixed dataset, never the train batch or supervised targets.
        if (
            self.pseudo_label_evaluator is not None and self.ema_initialized
            and self.scheduler.should_evaluate_pseudo_labels(
                self.global_step - 1, last_eval_step=self.last_pseudo_label_eval_step
            )
        ):
            logs["pseudo_label_evaluations"] = self._evaluate_pseudo_labels(phase)
        return logs

    def _evaluate_pseudo_labels(self, phase, *, full_dataset=False):
        """Resolve teacher/settings policy and delegate isolated IR evaluation."""
        names = ("rgb", "ir") if self.config.teacher_mode == "two_teacher" else (self.config.teacher_mode,)
        records = []
        for name in names:
            settings = self._distillation_settings(name, phase)
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
        from models.fcos_factory import build_checkpoint_metadata

        # Snapshot model, optimization, routing, and evaluation progress together
        # using the existing payload keys consumed by resume and evaluation.
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)
        payload = {
            "metadata": build_checkpoint_metadata(self.config),
            "global_step": self.global_step,
            "best_map": self.best_map,
            "ema_initialized": self.ema_initialized,
            "last_pseudo_label_eval_step": self.last_pseudo_label_eval_step,
            "last_full_pseudo_label_eval_step": self.last_full_pseudo_label_eval_step,
            "pseudo_label_evaluator": (None if self.pseudo_label_evaluator is None
                                       else self.pseudo_label_evaluator.state_dict()),
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

        # Validate experiment compatibility and required teacher states before
        # restoring model tensors, keeping the current rejection behavior.
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
        # Restore learned state first, followed by optimization and routing
        # state, so the next iteration continues with the saved update schedule.
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
        # Restore evaluation history and lifecycle flags, including the EMA
        # initialization flag that prevents an unintended teacher reset.
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
        # Older baseline checkpoints do not carry observer progress. An audit
        # is eligible after teacher initialization if a global tick has passed.
        self.last_pseudo_label_eval_step = int(checkpoint.get("last_pseudo_label_eval_step", 0))
        self.last_full_pseudo_label_eval_step = int(checkpoint.get("last_full_pseudo_label_eval_step", 0))
        audit_state = checkpoint.get("pseudo_label_evaluator")
        if self.pseudo_label_evaluator is not None and audit_state is not None:
            self.pseudo_label_evaluator.load_state_dict(audit_state)
        logger.info("Resumed from step %s", self.global_step)

    def _log(self, log: Dict[str, Any]) -> None:
        log_iteration(log, self.global_step, logger=logger)
