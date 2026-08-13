import logging
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from collections import defaultdict

from config import TrainingConfig, StepRouting
from scheduler import CurriculumScheduler, DomainStep, Phase
from ema import ema_update
from models.custom_fcos import FCOSDistillAdapter
from loss.orchestrator import compute_combined_loss
from data.augmentations import StudentAugmentor

logger = logging.getLogger(__name__)

class CurriculumDomainAdaptationTrainer:
    def __init__(
        self,
        student: nn.Module,
        rgb_teacher: nn.Module,
        ir_teacher: nn.Module,
        optimizer: Optimizer,
        config: TrainingConfig,
        rgb_loader: DataLoader,
        ir_loader: DataLoader,
        val_loader: DataLoader,
        distill_adapter: FCOSDistillAdapter,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)
        self.distill_adapter = distill_adapter
        self.val_loader = val_loader
        self.augmentor = StudentAugmentor(config)

        self._setup_models()

        # Infinite iterators
        self._rgb_iter = iter(self._infinite(rgb_loader))
        self._ir_iter = iter(self._infinite(ir_loader))
        
        self.scheduler = CurriculumScheduler(config)
        self.global_step = 0
        
        # Metric tracking
        self.best_map = 0.0
        self.loss_history = defaultdict(list)

    def _setup_models(self):
        """Freeze teachers and move models to device."""
        for teacher in (self.rgb_teacher, self.ir_teacher):
            teacher.to(self.device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
        self.student.to(self.device)

    @staticmethod
    def _infinite(loader):
        while True:
            yield from loader

    def _get_saga_images(self, rgb_images, targets, alpha: float):
        """Apply SAGA transformation."""
        boxes_list = [t["boxes"] for t in targets]
        from saga import SoftSAGA
        return SoftSAGA().apply_to_batch(rgb_images, boxes_list, alpha)
    
    def _prepare_data_by_route(self, step_name: str, route: StepRouting):
        """
        Use augmentation for images
        """
        ss_cfg = self.config.soft_saga
        alpha_map = {
            "rgb": 1.0, 
            "weak": ss_cfg.alpha_near_rgb, 
            "mid": ss_cfg.alpha_intermediate,
            "high": ss_cfg.alpha_near_ir, 
            "ir": 0.0
        }

        # From ir source no bale
        if "ir_flow" in step_name or "ir_focus" in step_name:
            images_ir = next(self._ir_iter).to(self.device)
            
            # Geometric augmentation
            stu_img, _, did_flip = self.augmentor.apply_weak_aug(images_ir)
            
            # In Phase 3 IR: Teacher uses SAGA-High 
            if step_name == "p3_ir_flow":
                rgb_batch = next(self._rgb_iter)
                images_rgb, targets_rgb = rgb_batch[0].to(self.device), rgb_batch[1]

                if did_flip:
                    images_rgb = torch.flip(images_rgb, dims=[-1])
                    # Flip boxes for saga
                    W = images_rgb.shape[-1]
                    for t in targets_rgb:
                        boxes = t["boxes"]
                        if boxes.numel() > 0:
                            flipped = boxes.clone()
                            flipped[:, 0] = W - boxes[:, 2]
                            flipped[:, 2] = W - boxes[:, 0]
                            t["boxes"] = flipped

                # Applying SAGA after augmentation
                tea_img = self._get_saga_images(images_rgb, targets_rgb, alpha_map["high"])
            else:
                # In Phase 4 IR
                tea_img = stu_img.clone()

            # Student use strong photometric augmentation
            stu_img = self.augmentor.apply_photometric_aug(stu_img)
            
            return stu_img, tea_img, tea_img, None

        # RGB source
        else:
            rgb_batch = next(self._rgb_iter)
            images_raw, targets_raw = rgb_batch[0].to(self.device), rgb_batch[1]
            
            # Geometric augmentation + box adjustment
            flipped_img, flipped_targets, _ = self.augmentor.apply_weak_aug(images_raw, targets_raw)
            
            # SAGA
            stu_img = self._get_saga_images(flipped_img, flipped_targets, alpha_map[route.student_saga_level])
            tea_img = self._get_saga_images(flipped_img, flipped_targets, alpha_map[route.teacher_saga_level])
            
            # Strong Aug for Student
            stu_img = self.augmentor.apply_photometric_aug(stu_img)
            
            return stu_img, tea_img, tea_img, flipped_targets

    def train_one_iteration(self) -> Dict[str, float]:
        self.student.train()
        self.optimizer.zero_grad()
        
        # 1. Ask scheduler for the step and phase
        step_name: DomainStep = self.scheduler.get_next_step(self.global_step)
        phase = self.config.get_phase(self.global_step)
        
        # 2. Get routing instructions for this step
        route = self.config.mid_routing.get_routing(step_name)
        ss_cfg = self.config.soft_saga
        
        
        
        # 4. Data Loading & Preparation
        # If it's an IR flow (P3-IR or P4), load IR images, else use RGB for SAGA
        stu_img, tea_rgb, tea_ir, targets = self._prepare_data_by_route(step_name, route)

        # 5. Compute Combined Loss
        total_loss, losses = compute_combined_loss(
            self.student, self.distill_adapter,
            student_images=stu_img,
            targets=(targets if route.use_gt else None),
            teacher_rgb_images=tea_rgb,
            teacher_ir_images=tea_ir,
            global_step=self.global_step,
            phase=phase,
            loss_cfg=self.config.loss
        )

        

        # 6. Optimization
        if total_loss > 0:
            total_loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        self.optimizer.step()

        # 7. EMA Update based on routing
        log_res = {}
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_res[k] = v.item()
            else:
                log_res[k] = v
        
        log_res["total_loss"] = total_loss.item()
        log_res["step_type"] = step_name
        log_res["phase"] = phase.name
        log_res["global_step"] = self.global_step

        self.global_step += 1
        return log_res

    def train(self) -> None:
        """
        Full curriculum training loop with integrated logging and evaluation.
        """
        total_iters = self.config.max_iter
        logger.info(
            f"Starting Curriculum DA training: total_iters={total_iters} "
            f"starting_from_step={self.global_step}"
        )
        
        for _ in range(self.global_step, total_iters):
            # 1. Execute one iteration (includes Data Prep, Forward, Backward, EMA)
            iter_start_time = time.time()
            logs = self.train_one_iteration()
            iter_time = time.time() - iter_start_time

            # 2. Periodic Logging
            if self.global_step % self.config.log_interval == 0:
                logs["iter_time"] = iter_time
                logs["lr"] = self.optimizer.param_groups[0]['lr']
                self._log(logs)

            # 4. Periodic Checkpoint (for resume)
            if self.global_step > 0 and self.global_step % 5000 == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step:06d}.pth")

        logger.info("Training complete.")

    def save_checkpoint(self, filename: str) -> None:
        """
        Saves full state of training to resume later or use for inference.
        """
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)
        path = os.path.join(self.config.output_dir, filename)
        
        torch.save(
            {
                "global_step":   self.global_step,
                "best_map":      self.best_map,
                "student":       self.student.state_dict(),
                "rgb_teacher":   self.rgb_teacher.state_dict(),
                "ir_teacher":    self.ir_teacher.state_dict(),
                "optimizer":     self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads the training state from a file.
        """
        logger.info(f"Loading checkpoint ← {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.global_step = ckpt.get("global_step", 0)
        self.best_map    = ckpt.get("best_map", 0.0)
        
        self.student.load_state_dict(ckpt["student"])
        self.rgb_teacher.load_state_dict(ckpt["rgb_teacher"])
        self.ir_teacher.load_state_dict(ckpt["ir_teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        
        logger.info(f"Resumed from step {self.global_step} (Best mAP so far: {self.best_map:.4f})")

    def _log(self, log: Dict) -> None:
        """
        Organizes and prints iteration metrics to the console/logger.
        """
        step   = log.get("global_step", self.global_step)
        phase  = log.get("phase",  "N/A")
        step_type = log.get("step_type", "N/A")
        total_loss = log.get("total_loss", 0.0)
        
        # Format individual loss components (kd_logits, sup_box, etc.)
        components = []
        for k, v in sorted(log.items()):
            if isinstance(v, float) and any(x in k for x in ["loss", "sup_", "kd_"]) and k != "total_loss":
                components.append(f"{k}={v:.4f}")
        
        comp_str = " | ".join(components)
        
        log_msg = (
            f"[{step:06d}] Phase: {phase:<18} | Step: {step_type:<18} | "
            f"Loss: {total_loss:.4f}"
        )
        if comp_str:
            log_msg += f" | {comp_str}"
            
        logger.info(log_msg)