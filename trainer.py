"""
CurriculumDomainAdaptationTrainer

Orchestrates the full RGB → MID(SAGA) → IR curriculum training loop.

Architecture:
  - 1 student   (trained by gradient descent, has optimizer)
  - 2 teachers:
      rgb_teacher  (EMA of student, updated in RGB step)
      ir_teacher   (EMA of student, updated in IR  step)
  Teachers are ALWAYS in eval mode and require no grad.

Training flow per iteration:
  scheduler.get_next_step(global_step) → "rgb" | "mid" | "ir"
  dispatch to train_rgb_step / train_mid_step / train_ir_step
"""

import copy
import logging
import random
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from batch_types import IRBatch, MidBatch, RGBBatch
from config import TrainingConfig
from ema import ema_update
from losses import compute_ir_loss, compute_mid_loss, compute_rgb_loss
from saga import SemanticAwareGrayAugmentation, SoftSAGA
from scheduler import CurriculumScheduler, DomainStep, Phase

try:
    import torchvision.transforms.functional as TF
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# Optional — imported only when provided at runtime
try:
    from adaptive_threshold import AdaptiveThresholdScheduler
    from evaluator import PhaseEvaluator
except ImportError:
    AdaptiveThresholdScheduler = None  # type: ignore
    PhaseEvaluator = None              # type: ignore

logger = logging.getLogger(__name__)


class CurriculumDomainAdaptationTrainer:
    """
    Curriculum Domain Adaptation Trainer.

    Args:
        student      : model being trained (must follow detector train/eval API)
        rgb_teacher  : EMA teacher for RGB domain
        ir_teacher   : EMA teacher for IR  domain
        optimizer    : optimizer attached to student parameters ONLY
        config       : full TrainingConfig
        rgb_loader   : DataLoader yielding (images, targets) — labeled RGB
        ir_loader    : DataLoader yielding images (or (images,)) — unlabeled IR
    """

    def __init__(
        self,
        student: nn.Module,
        rgb_teacher: nn.Module,
        ir_teacher: nn.Module,
        optimizer: Optimizer,
        config: TrainingConfig,
        rgb_loader: DataLoader,
        ir_loader: DataLoader,
        threshold_scheduler: Optional["AdaptiveThresholdScheduler"] = None,
        phase_evaluator: Optional["PhaseEvaluator"] = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        # Optional extensions
        self.threshold_scheduler = threshold_scheduler
        self.phase_evaluator     = phase_evaluator

        self._setup_models()

        self.saga      = SemanticAwareGrayAugmentation(apply_prob=config.saga.apply_prob)
        self.soft_saga = SoftSAGA()
        self.scheduler = CurriculumScheduler(config.curriculum)

        # Infinite data iterators — never exhaust
        self._rgb_iter: Iterator = self._infinite(rgb_loader)
        self._ir_iter: Iterator  = self._infinite(ir_loader)

        self.global_step: int = 0
        self._last_phase: Optional[Phase] = None
        self._phase_step_count: int = 0   # steps taken in current phase (for early logging)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Move models to device; freeze & eval teachers."""
        for model in (self.student, self.rgb_teacher, self.ir_teacher):
            model.to(self.device)

        # Teachers must NOT be trained — freeze params and keep in eval mode
        for teacher in (self.rgb_teacher, self.ir_teacher):
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()

    # ------------------------------------------------------------------
    # Phase transition handler
    # ------------------------------------------------------------------

    def _on_phase_transition(self, from_phase: Optional[Phase], to_phase: Phase) -> None:
        """
        Called exactly once when the curriculum phase changes.

        Phase 1 → Phase 2  (RGB warmup → RGB+MID):
          Neither teacher was updated during Phase 1 (EMA skipped in warmup phase).
          Copy student → rgb_teacher AND student → ir_teacher so both teachers
          start Phase 2 from the exact pretrained weights.

        Phase 2 → Phase 3  (RGB+MID → MID+IR):
          ir_teacher has been partially updated (if mid_update_ir_teacher=True)
          or still needs a sync. Copy student → ir_teacher for a fresh start.
        """
        from ema import copy_student_to_teacher

        if from_phase is None:
            return  # initial call, no transition

        if from_phase == Phase.PHASE1_RGB_WARMUP and to_phase == Phase.PHASE2_RGB_MID:
            logger.info(
                f"[Phase transition] PHASE1→PHASE2: "
                f"copying pretrained student → rgb_teacher AND ir_teacher "
                f"(both teachers initialised from exact pretrained weights)"
            )
            copy_student_to_teacher(self.rgb_teacher, self.student)
            copy_student_to_teacher(self.ir_teacher,  self.student)

        elif from_phase == Phase.PHASE2_RGB_MID and to_phase == Phase.PHASE3_MID_IR:
            logger.info(
                f"[Phase transition] PHASE2→PHASE3: "
                f"copying student → ir_teacher for IR domain warm start"
            )
            copy_student_to_teacher(self.ir_teacher, self.student)

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def _get_threshold(self, phase: Phase, teacher: str = "both") -> float:
        """
        Return current confidence threshold for pseudo-label filtering.

        If threshold_scheduler is provided, use curriculum-based values.
        Otherwise fall back to config.pseudo_label_conf_thresh (fixed).

        Args:
            phase   : current curriculum phase
            teacher : "rgb" | "ir" | "both"
        """
        if self.threshold_scheduler is None:
            return self.config.pseudo_label_conf_thresh
        if teacher == "rgb":
            return self.threshold_scheduler.rgb_teacher(phase)
        if teacher == "ir":
            return self.threshold_scheduler.ir_teacher(phase)
        return self.threshold_scheduler.both(phase)

    # ------------------------------------------------------------------
    # Infinite data utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infinite(loader: DataLoader) -> Iterator:
        """Wrap a DataLoader in an infinite iterator."""
        while True:
            yield from loader

    def _next_rgb(self) -> RGBBatch:
        images, targets = next(self._rgb_iter)
        images = images.to(self.device)
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        return RGBBatch(images=images, targets=targets)

    def _next_ir(self) -> IRBatch:
        raw = next(self._ir_iter)
        # Support loaders that yield (images,) tuples or bare images
        images = raw[0] if isinstance(raw, (list, tuple)) else raw
        return IRBatch(images=images.to(self.device))

    def _next_mid(self, mid_level: DomainStep = "mid_near_rgb") -> MidBatch:
        """Pull an RGB batch and apply SoftSAGA with the alpha for mid_level."""
        rgb = self._next_rgb()
        boxes_list = [t["boxes"] for t in rgb.targets]

        alpha_map = {
            "mid_near_rgb":    self.config.soft_saga.alpha_near_rgb,
            "mid_intermediate": self.config.soft_saga.alpha_intermediate,
            "mid_near_ir":     self.config.soft_saga.alpha_near_ir,
        }
        alpha = alpha_map.get(mid_level, self.config.soft_saga.alpha_near_rgb)
        mid_images = self.soft_saga.apply_to_batch(rgb.images, boxes_list, alpha)
        return MidBatch(
            images=mid_images,
            targets=rgb.targets,
            source_images=rgb.images,
        )

    def _weak_aug(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ):
        """
        Weak aug: stochastic horizontal flip only.

        Returns:
            aug_images  : [B, 3, H, W]
            aug_targets : boxes updated if flipped
            did_flip    : bool — whether flip was applied (shared with student)
        """
        aug = self.config.aug
        W = images.shape[-1]
        if random.random() < aug.hflip_prob:
            images = torch.flip(images, dims=[-1])
            if targets is not None:
                new_targets = []
                for t in targets:
                    boxes = t["boxes"]
                    if boxes.numel() > 0:
                        flipped = boxes.clone()
                        flipped[:, 0] = W - boxes[:, 2]
                        flipped[:, 2] = W - boxes[:, 0]
                        new_targets.append({**t, "boxes": flipped})
                    else:
                        new_targets.append(t)
                targets = new_targets
            return images, targets, True
        return images, targets, False

    def _photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Photometric aug: blur + brightness/contrast. No geometric change."""
        if not _HAS_TF:
            return images
        aug = self.config.aug
        if random.random() < aug.blur_prob:
            sigma = random.uniform(0.1, aug.blur_sigma_max)
            images = TF.gaussian_blur(images, kernel_size=[3, 3], sigma=sigma)
        if random.random() < aug.brightness_prob:
            factor = 1.0 + random.uniform(-aug.brightness_mag, aug.brightness_mag)
            images = torch.clamp(images * factor, 0.0, 1.0)
        if random.random() < aug.contrast_prob:
            mean = images.mean(dim=[-1, -2], keepdim=True)
            factor = 1.0 + random.uniform(-aug.contrast_mag, aug.contrast_mag)
            images = torch.clamp((images - mean) * factor + mean, 0.0, 1.0)
        return images

    def _student_aug(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ):
        """
        Strong aug: hflip (stochastic) + photometric (blur/brightness/contrast).

        Returns:
            aug_images  : [B, 3, H, W]
            aug_targets : boxes updated if flipped
        """
        images, targets, _ = self._weak_aug(images, targets)
        images = self._photometric_aug(images)
        return images, targets

    # ------------------------------------------------------------------
    # Gradient clip + optimizer step
    # ------------------------------------------------------------------

    def _clip_and_step(self) -> float:
        """Clip gradients then step optimizer. Returns grad norm."""
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.grad_clip
            ).item()
        else:
            grad_norm = 0.0
        self.optimizer.step()
        return grad_norm

    # ------------------------------------------------------------------
    # Domain steps
    # ------------------------------------------------------------------

    def train_rgb_step(self, phase: Phase = Phase.PHASE1_RGB_WARMUP) -> Dict:
        """
        RGB step — supervised learning on labeled source data.

        Loss:   L_gt  +  optional L_pseudo (rgb_teacher on RGB)
        Update: EMA(rgb_teacher ← student)
        """
        self.student.train()
        batch = self._next_rgb()

        self.optimizer.zero_grad()

        teacher_for_pseudo = (
            self.rgb_teacher
            if self.config.loss.rgb_pseudo_weight > 0.0
            else None
        )
        # Phase 1 warmup: weak aug (hflip only) — stable supervised pretrain
        # Phase 2+: strong aug (hflip + photometric)
        if phase == Phase.PHASE1_RGB_WARMUP:
            student_images, student_targets, _ = self._weak_aug(batch.images, batch.targets)
        else:
            student_images, student_targets = self._student_aug(batch.images, batch.targets)

        loss, log = compute_rgb_loss(
            student=self.student,
            images=student_images,
            gt_targets=student_targets,
            rgb_teacher=teacher_for_pseudo,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="rgb"),
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm

        # EMA update rgb_teacher — but NOT during Phase 1 (pretrain).
        # Phase 1 is pure supervised pretrain; teachers are initialized from
        # student via init_teachers_from_student() at the Phase 1→2 transition.
        if self.config.teacher_update.rgb_update_rgb_teacher and phase != Phase.PHASE1_RGB_WARMUP:
            ema_update(
                teacher=self.rgb_teacher,
                student=self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step if self.config.ema.use_warmup else None,
            )

        log["domain"] = "rgb"
        return log

    def train_mid_step(
        self,
        phase: Phase = Phase.PHASE2_RGB_MID,
        mid_level: DomainStep = "mid_near_rgb",
    ) -> Dict:
        """
        MID step — SoftSAGA images, teacher routing per mid_level.

        mid_near_rgb   : rgb_teacher infers + EMA(rgb_teacher)
        mid_intermediate: both teachers infer + EMA(ir_teacher, lighter alpha)
        mid_near_ir    : ir_teacher infers  + EMA(ir_teacher)
        """
        self.student.train()
        batch = self._next_mid(mid_level)

        self.optimizer.zero_grad()

        # Routing config for this MID level
        rt = self.config.mid_routing
        level_cfg = {
            "mid_near_rgb": (
                rt.near_rgb_teacher_source,
                rt.near_rgb_ema_target,
                self.config.ema.alpha,
                rt.near_rgb_rgb_weight,
                rt.near_rgb_ir_weight,
            ),
            "mid_intermediate": (
                rt.intermediate_teacher_source,
                rt.intermediate_ema_target,
                rt.intermediate_ema_alpha,
                rt.intermediate_rgb_weight,
                rt.intermediate_ir_weight,
            ),
            "mid_near_ir": (
                rt.near_ir_teacher_source,
                rt.near_ir_ema_target,
                self.config.ema.alpha,
                rt.near_ir_rgb_weight,
                rt.near_ir_ir_weight,
            ),
        }
        teacher_source, ema_target, ema_alpha, rgb_w, ir_w = level_cfg[mid_level]

        gt_for_mid = (
            batch.targets
            if self.config.loss.mid_gt_weight > 0.0
            else None
        )

        # Teacher: weak aug (hflip only) — stable pseudo-label generation
        # Student: same flipped base + photometric — spatial structure stays aligned
        weak_images, _, did_flip = self._weak_aug(batch.images, None)
        strong_images = self._photometric_aug(weak_images.clone())

        loss, log = compute_mid_loss(
            student=self.student,
            mid_images=strong_images,       # student sees strong aug
            teacher_images=weak_images,     # teacher sees weak (original MID)
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            gt_targets=gt_for_mid,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher=teacher_source),
            teacher_source=teacher_source,
            rgb_weight_override=rgb_w,
            ir_weight_override=ir_w,
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm
        log["mid_level"] = mid_level

        # EMA update — only the routed teacher, NEVER both
        if ema_target != "none":
            ema_kwargs = dict(
                alpha=ema_alpha,
                global_step=self.global_step if self.config.ema.use_warmup else None,
            )
            if ema_target == "rgb":
                ema_update(teacher=self.rgb_teacher, student=self.student, **ema_kwargs)
            elif ema_target == "ir":
                ema_update(teacher=self.ir_teacher,  student=self.student, **ema_kwargs)

        log["domain"] = mid_level
        return log

    def train_ir_step(self, phase: Phase = Phase.PHASE3_MID_IR) -> Dict:
        """
        IR step — unsupervised learning on unlabeled target data.

        Loss:   L_rgb_teacher(IR)  +  L_ir_teacher(IR)
        Update: EMA(ir_teacher ← student)
        """
        self.student.train()
        batch = self._next_ir()

        self.optimizer.zero_grad()

        loss, log = compute_ir_loss(
            student=self.student,
            ir_images=batch.images,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="ir"),
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm

        if self.config.teacher_update.ir_update_ir_teacher:
            ema_update(
                teacher=self.ir_teacher,
                student=self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step if self.config.ema.use_warmup else None,
            )

        log["domain"] = "ir"
        return log

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def train_one_iteration(self) -> Dict:
        """
        Execute one training iteration.

        1. Ask the scheduler which domain to train (rgb/mid/ir).
        2. Dispatch to the corresponding step (uses adaptive threshold if set).
        3. Trigger PhaseEvaluator if set.
        4. Log if needed.
        5. Increment global_step.
        """
        step:  DomainStep = self.scheduler.get_next_step(self.global_step)
        phase: Phase      = self.scheduler.get_phase(self.global_step)

        # Detect phase transition and sync teachers if needed
        if phase != self._last_phase:
            self._on_phase_transition(from_phase=self._last_phase, to_phase=phase)
            self._phase_step_count = 0
        self._last_phase = phase

        if step == "rgb":
            log = self.train_rgb_step(phase=phase)
        elif step in ("mid_near_rgb", "mid_intermediate", "mid_near_ir"):
            log = self.train_mid_step(phase=phase, mid_level=step)
        elif step == "ir":
            log = self.train_ir_step(phase=phase)
        else:
            raise ValueError(f"Unknown domain step: {step!r}")

        log["phase"]       = phase.name
        log["global_step"] = self.global_step

        # Adaptive threshold logging
        if self.threshold_scheduler is not None:
            log["thresh_rgb"] = self.threshold_scheduler.rgb_teacher(phase)
            log["thresh_ir"]  = self.threshold_scheduler.ir_teacher(phase)

        # Phase evaluation trigger
        if self.phase_evaluator is not None:
            self.phase_evaluator.step(
                model=self.student,
                global_step=self.global_step,
                current_phase=phase,
            )

        # Log every log_interval steps OR for the first 10 steps of each phase
        if self.global_step % self.config.log_interval == 0 or self._phase_step_count < 10:
            self._log(log)

        self._phase_step_count += 1
        self.global_step += 1
        return log

    def train_one_epoch(self, steps_per_epoch: int) -> List[Dict]:
        """Run `steps_per_epoch` iterations and return all log dicts."""
        return [self.train_one_iteration() for _ in range(steps_per_epoch)]

    def train(self, total_iterations: int) -> None:
        """Full training loop."""
        logger.info(
            f"Starting Curriculum DA training  total_iters={total_iterations}"
            f"  scheduler={self.scheduler}"
        )
        for _ in range(total_iterations):
            self.train_one_iteration()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "global_step":   self.global_step,
                "student":       self.student.state_dict(),
                "rgb_teacher":   self.rgb_teacher.state_dict(),
                "ir_teacher":    self.ir_teacher.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}  (step {self.global_step})")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.student.load_state_dict(ckpt["student"])
        self.rgb_teacher.load_state_dict(ckpt["rgb_teacher"])
        self.ir_teacher.load_state_dict(ckpt["ir_teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Checkpoint loaded ← {path}  (step {self.global_step})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """
        Run student inference on val_loader.
        Returns raw predictions — plug in your own mAP metric here.
        """
        self.student.eval()
        all_predictions = []
        for batch in val_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)
            preds = self.student(images)
            all_predictions.extend(preds)
        self.student.train()
        return {"num_samples": len(all_predictions)}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, log: Dict) -> None:
        step   = log.get("global_step", self.global_step)
        phase  = log.get("phase",  "?")
        domain = log.get("domain", "?")
        losses = "  ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(log.items())
            if isinstance(v, float)
        )
        logger.info(f"[{step:07d}]  phase={phase:<22s}  domain={domain}  {losses}")
