"""
BlockDomainAdaptationTrainer — D3T-style dense distillation with a
BlockDomainScheduler (rgb / mid / ir short-cycle block scheduling).

Architecture:
  - 1 student     (trained by gradient descent every iteration)
  - 2 teachers     rgb_teacher (frozen, pretrained on RGB)
                   ir_teacher  (frozen, pretrained on MID/GAN-IR)
  Teachers are NOT trained inside the curriculum loop anymore — they are
  expected to be pretrained (see example_flir.py: pretrain_teacher) and are
  only updated afterwards via EMA from the student. Student is initialized
  from rgb_teacher weights before training starts (see example_flir.py).

MID domain bridge:
  GAN translator (frozen CycleGAN generator, RGB → grayscale 3-channel),
  applied on-the-fly to RGB batches to synthesize "mid" domain images.

Training flow per iteration (see BlockDomainScheduler in scheduler.py):
  domain = block_scheduler.get_domain(global_step)  → "rgb" | "mid" | "ir"

  domain == "rgb" : batch = labeled RGB (disjoint source half).
                    t_match = rgb_teacher, t_cross = ir_teacher.
  domain == "mid" : batch = RGB → GAN-translated (still has GT, same boxes).
                    t_match = ir_teacher,  t_cross = rgb_teacher.
  domain == "ir"  : batch = unlabeled real IR (disjoint target half).
                    t_match = ir_teacher,  t_cross = rgb_teacher.  No GT loss.

Loss per iteration:
  L = gt_weight * L_gt(student, weak_images, targets)        [rgb / mid only]
    + lambda(it) * ( w_match * distill(s, t_match)
                    + w_cross * distill(s, t_cross) )

  Dense distillation (D3T-style, see losses.py):
    student forward on STRONG-aug images (with grad),
    teacher forward on WEAK-aug images (no grad, same geometry so the dense
    per-location grid lines up between student and teacher).

EMA: only the teacher matching the current domain is EMA-updated per step
(rgb_teacher for "rgb", ir_teacher for "mid"/"ir") — mirrors the reference
D3T-block-scheduler script.

FCOS ONLY: dense distillation requires FCOSDetector.dense_forward (see
fcos_wrapper.py). Do not use this trainer with Faster-RCNN models.
"""

import logging
import random
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from batch_types import IRBatch, RGBBatch
from config import TrainingConfig
from ema import ema_update
from gan_translator import GANTranslator
from losses import distill, _sum_loss_dict
from scheduler import BlockDomainScheduler, DomainStep

if TYPE_CHECKING:
    from evaluator import PhaseEvaluator

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

logger = logging.getLogger(__name__)


def lambda_for_iter(it: int, lambda_max: float, ramp_iters: int) -> float:
    """Linear ramp 0 → lambda_max over `ramp_iters` iterations (D3T-style)."""
    if ramp_iters <= 0:
        return lambda_max
    return lambda_max * min(1.0, it / ramp_iters)


class BlockDomainAdaptationTrainer:
    """
    Block-scheduled, dense-distillation domain adaptation trainer.

    Args:
        student        : model being trained (FCOSDetector, must expose dense_forward)
        rgb_teacher    : frozen teacher pretrained on RGB (EMA-updated on "rgb" steps)
        ir_teacher     : frozen teacher pretrained on MID/GAN-IR (EMA-updated on "mid"/"ir")
        optimizer      : optimizer attached to student parameters ONLY
        config         : full TrainingConfig
        rgb_loader     : DataLoader yielding (images, targets) — labeled RGB (source half)
        ir_loader      : DataLoader yielding images (or (images,)) — unlabeled real IR (target half)
        gan_translator : RGB → MID (frozen, no grad)
        phase_evaluator: optional PhaseEvaluator hook
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
        gan_translator: GANTranslator,
        block_scheduler: BlockDomainScheduler,
        phase_evaluator: Optional["PhaseEvaluator"] = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        self.gan_translator = gan_translator
        self.block_scheduler = block_scheduler
        self.phase_evaluator = phase_evaluator

        self._setup_models()

        self._rgb_iter: Iterator = self._infinite(rgb_loader)
        self._ir_iter: Iterator = self._infinite(ir_loader)

        self.global_step: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Move models to device. Student trains from step 0; teachers are frozen
        (pretrained externally) and only ever change via EMA."""
        for model in (self.student, self.rgb_teacher, self.ir_teacher):
            model.to(self.device)
        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = True
        for teacher in (self.rgb_teacher, self.ir_teacher):
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Infinite data utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infinite(loader: DataLoader) -> Iterator:
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
        images = raw[0] if isinstance(raw, (list, tuple)) else raw
        return IRBatch(images=images.to(self.device))

    # ------------------------------------------------------------------
    # Geometric aug (weak — shared by teacher and student)
    # ------------------------------------------------------------------

    def _geometric_aug(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]],
        aug,
    ):
        """
        Geometric aug shared between teacher and student:
          1. Random horizontal flip
          2. Random scale in [multiscale_min, multiscale_max] × original size
          3. Per-image random crop (if scaled > target) or zero-pad (if scaled < target)
        """
        B, C, H, W = images.shape

        if random.random() < aug.hflip_prob:
            images = torch.flip(images, dims=[-1])
            if targets is not None:
                targets = [self._flip_boxes(t, W) for t in targets]

        s = random.uniform(aug.multiscale_min, aug.multiscale_max)
        new_h = max(1, int(H * s))
        new_w = max(1, int(W * s))
        images = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        if targets is not None:
            targets = [self._scale_boxes(t, s) for t in targets]

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
        C, H, W = img.shape
        top = random.randint(0, max(0, H - target_h))
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

    # ------------------------------------------------------------------
    # Photometric aug (strong — student only)
    # ------------------------------------------------------------------

    def _rgb_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Strong photometric aug for RGB images (3-channel, color-aware)."""
        if not _HAS_TF or images.numel() == 0:
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
        """Strong photometric aug for IR / MID (thermal-like) images."""
        if images.numel() == 0:
            return images
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
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.grad_clip
            ).item()
        else:
            grad_norm = 0.0
        self.optimizer.step()
        return grad_norm

    # ------------------------------------------------------------------
    # Dense distillation loss (both teachers, w_match/w_cross weighted)
    # ------------------------------------------------------------------

    def _distill_loss(self, strong_images: torch.Tensor, weak_images: torch.Tensor,
                       t_match: nn.Module, t_cross: nn.Module) -> torch.Tensor:
        num_classes = self.config.distill_num_classes
        s_out = self.student.dense_forward(strong_images)
        with torch.no_grad():
            t_match_out = t_match.dense_forward(weak_images)
            t_cross_out = t_cross.dense_forward(weak_images)
        loss_match = distill(s_out, t_match_out, num_classes, self.config.distill)
        loss_cross = distill(s_out, t_cross_out, num_classes, self.config.distill)
        tw = self.config.teacher_weight
        return tw.w_match * loss_match + tw.w_cross * loss_cross

    # ------------------------------------------------------------------
    # Domain steps
    # ------------------------------------------------------------------

    def train_rgb_step(self) -> Dict:
        """domain == "rgb": labeled RGB, supervised + distillation (match=rgb_teacher)."""
        rgb = self._next_rgb()
        weak_images, targets = self._geometric_aug(rgb.images, rgb.targets, self.config.rgb_aug)
        strong_images = self._rgb_photometric_aug(weak_images.clone())

        loss_sup = _sum_loss_dict(self.student(weak_images, targets)) * self.config.gt_weight
        loss_uns = self._distill_loss(
            strong_images, weak_images, t_match=self.rgb_teacher, t_cross=self.ir_teacher
        )
        return self._finish_step(loss_sup, loss_uns, ema_teacher=self.rgb_teacher, domain="rgb")

    def train_mid_step(self) -> Dict:
        """domain == "mid": RGB → GAN-translated, still has GT, distill match=ir_teacher."""
        rgb = self._next_rgb()
        weak_rgb, targets = self._geometric_aug(rgb.images, rgb.targets, self.config.rgb_aug)
        with torch.no_grad():
            weak_images = self.gan_translator.apply_to_batch(weak_rgb)
        strong_images = self._ir_photometric_aug(weak_images.clone())

        loss_sup = _sum_loss_dict(self.student(weak_images, targets)) * self.config.gt_weight
        loss_uns = self._distill_loss(
            strong_images, weak_images, t_match=self.ir_teacher, t_cross=self.rgb_teacher
        )
        return self._finish_step(loss_sup, loss_uns, ema_teacher=self.ir_teacher, domain="mid")

    def train_ir_step(self) -> Dict:
        """domain == "ir": unlabeled real IR, unsupervised only, distill match=ir_teacher."""
        ir = self._next_ir()
        weak_images, _ = self._geometric_aug(ir.images, None, self.config.ir_aug)
        strong_images = self._ir_photometric_aug(weak_images.clone())

        loss_sup = weak_images.sum() * 0.0
        loss_uns = self._distill_loss(
            strong_images, weak_images, t_match=self.ir_teacher, t_cross=self.rgb_teacher
        )
        return self._finish_step(loss_sup, loss_uns, ema_teacher=self.ir_teacher, domain="ir")

    def _finish_step(self, loss_sup: torch.Tensor, loss_uns: torch.Tensor,
                      ema_teacher: nn.Module, domain: str) -> Dict:
        lam = lambda_for_iter(
            self.global_step, self.config.lambda_cfg.lambda_max, self.config.lambda_cfg.ramp_iters
        )
        total_loss = loss_sup + lam * loss_uns

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = self._clip_and_step()

        ema_update(
            teacher=ema_teacher, student=self.student,
            alpha=self.config.ema.alpha,
            global_step=self.global_step if self.config.ema.use_warmup else None,
        )

        return {
            "domain": domain,
            "lambda": lam,
            "loss_sup": loss_sup.item(),
            "loss_uns": loss_uns.item(),
            "total_loss": total_loss.item(),
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def train_one_iteration(self) -> Dict:
        domain: DomainStep = self.block_scheduler.get_domain(self.global_step)

        if domain == "rgb":
            log = self.train_rgb_step()
        elif domain == "mid":
            log = self.train_mid_step()
        elif domain == "ir":
            log = self.train_ir_step()
        else:
            raise ValueError(f"Unknown domain: {domain!r}")  # pragma: no cover

        phase_idx = self.block_scheduler.phase_for_iter(self.global_step)
        ratio = self.block_scheduler.ratio_for_iter(self.global_step)
        log["phase"] = f"phase{phase_idx}"
        log["ratio"] = ratio
        log["global_step"] = self.global_step

        if self.phase_evaluator is not None:
            self.phase_evaluator.step(
                model=self.student,
                global_step=self.global_step,
                phase_idx=phase_idx,
            )

        if self.global_step % self.config.log_interval == 0:
            self._log(log)

        self.global_step += 1
        return log

    def train_one_epoch(self, steps_per_epoch: int) -> List[Dict]:
        return [self.train_one_iteration() for _ in range(steps_per_epoch)]

    def train(self, total_iterations: int) -> None:
        logger.info(
            f"Starting Block Domain Adaptation training  total_iters={total_iterations}"
            f"  scheduler={self.block_scheduler}"
        )
        for _ in range(total_iterations):
            self.train_one_iteration()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            "global_step": self.global_step,
            "student": self.student.state_dict(),
            "rgb_teacher": self.rgb_teacher.state_dict(),
            "ir_teacher": self.ir_teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)
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
        step = log.get("global_step", self.global_step)
        phase = log.get("phase", "?")
        domain = log.get("domain", "?")
        ratio = log.get("ratio", "?")
        losses = "  ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(log.items())
            if isinstance(v, float)
        )
        logger.info(
            f"[{step:07d}]  phase={phase:<8s}  domain={domain:<4s}  ratio={ratio}  {losses}"
        )
