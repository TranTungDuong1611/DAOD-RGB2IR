"""
CurriculumDomainAdaptationTrainer

Orchestrates the full RGB → MID(SAGA) → IR curriculum training loop.

Architecture:
  - 1 student   (trained by gradient descent, has optimizer)
  - 2 teachers:
      rgb_teacher  (EMA of student, updated in RGB and MID steps)
      ir_teacher   (EMA of student, updated in MID and IR  steps)
  Teachers are ALWAYS in eval mode and require no grad.

Training flow per iteration:
  scheduler.get_next_step(global_step) → "rgb" | "mid" | "ir"
  dispatch to train_rgb_step / train_mid_step / train_ir_step

Phase 2 (MID): SAGA applied 100%, both teachers infer and receive EMA updates.
"""

import logging
import os
import random
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from batch_types import IRBatch, MidBatch, RGBBatch
from config import TrainingConfig
from ema import ema_update
from losses import compute_ir_loss, compute_mid_loss, compute_rgb_loss
from saga import SemanticAwareGrayAugmentation
from scheduler import CurriculumScheduler, DomainStep, Phase

if TYPE_CHECKING:
    from adaptive_threshold import AdaptiveThresholdScheduler
    from evaluator import PhaseEvaluator

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

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
        phase1_best_path: Optional[str] = None,
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
        # phase_best_paths: maps phase name → path of best checkpoint for that phase
        self.phase_best_paths: Dict[str, str] = {}
        if phase1_best_path:
            self.phase_best_paths["PHASE1_RGB_WARMUP"] = phase1_best_path

        self._setup_models()

        # SAGA always applied at 100% (apply_prob=1.0) in MID phase
        self.saga      = SemanticAwareGrayAugmentation(apply_prob=config.saga.apply_prob)
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

        Phase 1 → Phase 2  (RGB warmup → MID):
          Copy student → rgb_teacher AND ir_teacher from best Phase 1 checkpoint
          (or current student if no checkpoint), so both teachers start Phase 2
          from the exact pretrained RGB weights.
        """
        if from_phase is None:
            return  # initial call, no transition

        if from_phase == Phase.PHASE1_RGB_WARMUP and to_phase == Phase.PHASE2_RGB_MID:
            self._init_teachers_from_checkpoint(
                "PHASE1_RGB_WARMUP", teachers=["rgb", "ir"],
                fallback_msg="PHASE1→PHASE2: no best checkpoint, using current student",
            )

    def _init_teachers_from_checkpoint(
        self,
        phase_key: str,
        teachers: list,
        fallback_msg: str,
    ) -> None:
        """Load student weights from the best checkpoint of a phase into teachers."""
        from ema import copy_student_to_teacher
        best_path = self.phase_best_paths.get(phase_key)
        if best_path and os.path.exists(best_path):
            logger.info(f"[Phase transition] loading {best_path} → {teachers} teacher(s)")
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            state = ckpt["student"]
            if "rgb" in teachers:
                self.rgb_teacher.load_state_dict(state)
            if "ir" in teachers:
                self.ir_teacher.load_state_dict(state)
        else:
            logger.info(f"[Phase transition] {fallback_msg}")
            if "rgb" in teachers:
                copy_student_to_teacher(self.rgb_teacher, self.student)
            if "ir" in teachers:
                copy_student_to_teacher(self.ir_teacher,  self.student)

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def _get_threshold(self, phase: Phase, teacher: str = "both"):
        """
        Return current confidence threshold for pseudo-label filtering.

        For Phase 3 ir_teacher, passes steps_into_phase to enable linear ramp-up.
        Falls back to config.pseudo_label_conf_thresh if no scheduler is set.
        """
        if self.threshold_scheduler is None:
            return self.config.pseudo_label_conf_thresh

        steps_into_phase = (
            max(0, self.global_step - self.config.curriculum.phase2_end)
            if phase == Phase.PHASE3_IR_FOCUS
            else 0
        )

        if teacher == "rgb":
            return self.threshold_scheduler.rgb_teacher(phase, steps_into_phase)
        if teacher == "ir":
            return self.threshold_scheduler.ir_teacher(phase, steps_into_phase)
        return self.threshold_scheduler.both(phase, steps_into_phase)

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

    def _next_mid(self) -> MidBatch:
        """Pull an RGB batch and apply hard SAGA at 100% (apply_prob from config)."""
        rgb = self._next_rgb()
        boxes_list = [t["boxes"] for t in rgb.targets]
        mid_images = self.saga.apply_to_batch(rgb.images, boxes_list)
        return MidBatch(
            images=mid_images,
            targets=rgb.targets,
            source_images=rgb.images,
        )

    def _geometric_aug(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]],
        aug,                            # RGBAugConfig or IRAugConfig
    ):
        """
        Geometric aug shared between teacher and student:
          1. Random horizontal flip
          2. Random scale in [multiscale_min, multiscale_max] × original size
          3. Per-image random crop (if scaled > target) or zero-pad (if scaled < target)

        Both teacher and student receive the SAME geometric transform.
        Student additionally receives photometric aug on top.

        Returns:
            aug_images  : [B, 3, target_h, target_w]
            aug_targets : boxes updated for flip + scale + crop/pad (None if input was None)
        """
        B, C, H, W = images.shape

        # 1. Horizontal flip
        if random.random() < aug.hflip_prob:
            images = torch.flip(images, dims=[-1])
            if targets is not None:
                targets = [self._flip_boxes(t, W) for t in targets]

        # 2. Random scale (same factor for entire batch)
        s = random.uniform(aug.multiscale_min, aug.multiscale_max)
        new_h = max(1, int(H * s))
        new_w = max(1, int(W * s))
        images = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        if targets is not None:
            targets = [self._scale_boxes(t, s) for t in targets]

        # 3. Per-image crop/pad to fixed size
        out_imgs, out_tgts = [], []
        for i in range(B):
            img, tgt = self._crop_or_pad(
                images[i],
                targets[i] if targets is not None else None,
                aug.multiscale_target_h,
                aug.multiscale_target_w,
            )
            out_imgs.append(img)
            out_tgts.append(tgt)

        return torch.stack(out_imgs), (out_tgts if targets is not None else None)

    # ------------------------------------------------------------------
    # Geometric aug helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flip_boxes(target: Dict, W: int) -> Dict:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return target
        flipped = boxes.clone()
        flipped[:, 0] = W - boxes[:, 2]
        flipped[:, 2] = W - boxes[:, 0]
        return {**target, "boxes": flipped}

    @staticmethod
    def _scale_boxes(target: Dict, s: float) -> Dict:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return target
        return {**target, "boxes": boxes * s}

    @staticmethod
    def _crop_or_pad(
        img: torch.Tensor,
        target: Optional[Dict],
        target_h: int,
        target_w: int,
    ):
        """
        Randomly crop to (target_h, target_w) if img is larger,
        or zero-pad if smaller. Updates boxes accordingly.
        """
        C, H, W = img.shape

        top  = random.randint(0, max(0, H - target_h))
        left = random.randint(0, max(0, W - target_w))

        img = img[:, top:top + min(H, target_h), left:left + min(W, target_w)]

        pad_b = target_h - img.shape[1]
        pad_r = target_w - img.shape[2]
        if pad_b > 0 or pad_r > 0:
            img = torch.nn.functional.pad(img, (0, pad_r, 0, pad_b), value=0.0)

        if target is not None:
            boxes = target.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                offset = torch.tensor(
                    [left, top, left, top], dtype=boxes.dtype, device=boxes.device
                )
                boxes = boxes - offset
                boxes[:, 0::2].clamp_(0, target_w)
                boxes[:, 1::2].clamp_(0, target_h)
                keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
                target = {**target, "boxes": boxes[keep], "labels": target["labels"][keep]}

        return img, target

    def _rgb_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """
        Strong photometric aug for RGB / MID (SAGA) images.
        Gaussian blur → Color jitter → Random erasing.
        """
        if not _HAS_TF:
            return images
        aug = self.config.rgb_aug

        if random.random() < aug.blur_prob:
            sigma = random.uniform(0.1, aug.blur_sigma_max)
            images = TF.gaussian_blur(images, kernel_size=[3, 3], sigma=sigma)

        if random.random() < aug.color_jitter_prob:
            jitter = T.ColorJitter(
                brightness=aug.cj_brightness,
                contrast=aug.cj_contrast,
                saturation=aug.cj_saturation,
                hue=aug.cj_hue,
            )
            images = torch.stack([jitter(img) for img in images])

        if random.random() < aug.random_erasing_prob:
            eraser = T.RandomErasing(
                p=1.0,
                scale=(aug.random_erasing_scale_min, aug.random_erasing_scale_max),
                ratio=(aug.random_erasing_ratio_min, aug.random_erasing_ratio_max),
                value=0,
            )
            images = torch.stack([eraser(img) for img in images])

        return images

    def _ir_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """
        Strong photometric aug for IR (thermal) images.
        Intensity shift → Contrast jitter → Gamma correction → Gaussian noise.
        """
        aug = self.config.ir_aug

        if random.random() < aug.intensity_shift_prob:
            shift = random.uniform(-aug.intensity_shift_mag, aug.intensity_shift_mag)
            images = torch.clamp(images + shift, 0.0, 1.0)

        if random.random() < aug.contrast_jitter_prob:
            factor = 1.0 + random.uniform(-aug.contrast_jitter_mag, aug.contrast_jitter_mag)
            mean = images.mean(dim=[-1, -2], keepdim=True)
            images = torch.clamp((images - mean) * factor + mean, 0.0, 1.0)

        if random.random() < aug.gamma_prob:
            gamma = random.uniform(aug.gamma_min, aug.gamma_max)
            images = torch.clamp(images, 1e-6, 1.0).pow(gamma)

        if random.random() < aug.gaussian_noise_prob:
            noise = torch.randn_like(images) * aug.gaussian_noise_std
            images = torch.clamp(images + noise, 0.0, 1.0)

        return images

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
            if self.config.loss.rgb_pseudo_weight > 0.0 and phase != Phase.PHASE1_RGB_WARMUP
            else None
        )

        aug = self.config.rgb_aug
        if phase == Phase.PHASE1_RGB_WARMUP:
            # Phase 1: geometric only (hflip + multiscale), no teacher
            student_images, student_targets = self._geometric_aug(
                batch.images, batch.targets, aug
            )
            teacher_images = None
        else:
            # Phase 2+: teacher sees geometric aug, student sees same geometry + RGB photometric
            teacher_images, student_targets = self._geometric_aug(
                batch.images, batch.targets, aug
            )
            student_images = self._rgb_photometric_aug(teacher_images.clone())

        loss, log = compute_rgb_loss(
            student=self.student,
            images=student_images,
            gt_targets=student_targets,
            rgb_teacher=teacher_for_pseudo,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="rgb"),
            teacher_images=teacher_images,
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm

        # EMA update rgb_teacher — but NOT during Phase 1 (pretrain).
        if self.config.teacher_update.rgb_update_rgb_teacher and phase != Phase.PHASE1_RGB_WARMUP:
            ema_update(
                teacher=self.rgb_teacher,
                student=self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step if self.config.ema.use_warmup else None,
            )

        log["domain"] = "rgb"
        return log

    def train_mid_step(self, phase: Phase = Phase.PHASE2_RGB_MID) -> Dict:
        """
        MID step — SAGA images (100%), both teachers infer, only ir_teacher EMA updated.

        Teacher: weak aug (hflip only)  → stable pseudo-label generation
        Student: strong aug (hflip + photometric) → robust feature learning

        Loss:   L_rgb_teacher(MID)  +  L_ir_teacher(MID)  +  optional L_gt
        Update: EMA(ir_teacher ← student)   [rgb_teacher NOT updated in MID step]
        """
        self.student.train()
        batch = self._next_mid()

        self.optimizer.zero_grad()

        # Geometric aug — teacher and student see same spatial transform.
        # Pass targets so boxes are updated correctly after scale + crop/pad.
        aug = self.config.rgb_aug
        weak_images, aug_targets = self._geometric_aug(batch.images, batch.targets, aug)
        gt_for_mid = aug_targets if self.config.loss.mid_gt_weight > 0.0 else None

        # Student sees same geometric base + RGB photometric aug on top
        strong_images = self._rgb_photometric_aug(weak_images.clone())

        loss, log = compute_mid_loss(
            student=self.student,
            mid_images=strong_images,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            gt_targets=gt_for_mid,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="both"),
            teacher_images=weak_images,
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm

        # MID step: only ir_teacher is updated.
        # rgb_teacher is updated exclusively in the RGB step (same phase).
        ema_update(
            teacher=self.ir_teacher,
            student=self.student,
            alpha=self.config.ema.alpha,
            global_step=self.global_step if self.config.ema.use_warmup else None,
        )

        log["domain"] = "mid"
        return log

    def train_ir_step(self, phase: Phase = Phase.PHASE3_IR_FOCUS) -> Dict:
        """
        IR step — unsupervised on unlabeled IR data, ir_teacher pseudo-labels only.

        Loss:   L_ir_teacher(IR)
        Update: EMA(ir_teacher ← student)
        """
        self.student.train()
        batch = self._next_ir()

        self.optimizer.zero_grad()

        # Teacher sees geometric aug, student sees same geometry + IR photometric aug
        weak_images, _ = self._geometric_aug(batch.images, None, self.config.ir_aug)
        strong_images = self._ir_photometric_aug(weak_images.clone())

        loss, log = compute_ir_loss(
            student=self.student,
            ir_images=strong_images,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="ir"),
            teacher_images=weak_images,
        )

        loss.backward()
        grad_norm = self._clip_and_step()
        log["grad_norm"] = grad_norm

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
        2. Dispatch to the corresponding step.
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
        elif step == "mid":
            log = self.train_mid_step(phase=phase)
        elif step == "ir":
            log = self.train_ir_step(phase=phase)
        else:
            raise ValueError(f"Unknown domain step: {step!r}")  # pragma: no cover

        log["phase"]       = phase.name
        log["global_step"] = self.global_step

        # Adaptive threshold logging (reuse _get_threshold so ramp is reflected)
        if self.threshold_scheduler is not None:
            rt = self._get_threshold(phase, teacher="rgb")
            it = self._get_threshold(phase, teacher="ir")
            log["thresh_rgb"] = min(rt.values()) if isinstance(rt, dict) else rt
            log["thresh_ir"]  = min(it.values()) if isinstance(it, dict) else it

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
