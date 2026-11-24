"""Prepare aligned views using routing decisions supplied by the caller."""

from typing import Iterable

import torch


class RouteDataFlow:
    """Own loader cursors and view preparation, without reading training config."""

    def __init__(self, rgb_loader, ir_loader, *, device, augmentor, gan_translator=None):
        # Keep execution dependencies here; the trainer supplies the source,
        # supervision flag, and appearance settings separately for each step.
        self.device = device
        self.augmentor = augmentor
        self.gan_translator = gan_translator
        # Construct lazy cursors in the same order as the original trainer.
        self._rgb_iter = iter(self._infinite(rgb_loader))
        self._ir_iter = None if ir_loader is None else iter(self._infinite(ir_loader))

    @staticmethod
    def _infinite(loader: Iterable):
        # Present each loader as a continuous stream across curriculum phases.
        # An empty loader is rejected instead of leaving training waiting forever.
        while True:
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("training DataLoader is empty")

    def _move_targets(self, targets):
        # Move tensor annotations while retaining sample metadata used to align KD.
        moved = []
        for target in targets:
            moved.append({
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            })
        return moved

    def _unpack_rgb_batch(self, batch):
        images, targets, sample_ids = batch
        return images.to(self.device), self._move_targets(targets), sample_ids

    def _unpack_ir_batch(self, batch):
        images, sample_ids = batch
        return images.to(self.device), sample_ids

    def _get_saga_images(self, images, targets, alpha: float):
        # Apply the caller's appearance setting to annotated object regions.
        # Geometry and sample correspondence remain shared between both views.
        boxes_list = [target["boxes"] for target in targets]
        return self.augmentor.saga.apply_to_batch(images, boxes_list, alpha)

    def prepare(
        self,
        *,
        source_domain: str,
        use_gt: bool,
        student_alpha: float | None,
        teacher_alpha: float | None,
        appearance_domain: str | None = None,
    ):
        """Execute the supplied source/augmentation settings on one batch."""
        if source_domain == "rgb":
            # Labeled-source path, including RGB-backed routes named "IR flow":
            # fetch one batch and establish shared geometry for both views and GT.
            images, targets, sample_ids = self._unpack_rgb_batch(next(self._rgb_iter))
            geometric_images, geometric_targets, _ = self.augmentor.apply_weak_aug(
                images, targets
            )
            # Build different appearance views from that same geometric batch;
            # the stronger photometric augmentation belongs to the student only.
            if appearance_domain == "ir_gan":
                teacher_images = self.gan_translator(geometric_images)
                student_images = teacher_images.clone()
            elif appearance_domain == "rgb":
                teacher_images = geometric_images.clone()
                student_images = geometric_images.clone()
            else:
                student_images = self._get_saga_images(
                    geometric_images, geometric_targets, student_alpha,
                )
                teacher_images = self._get_saga_images(
                    geometric_images, geometric_targets, teacher_alpha,
                )
            student_images = self.augmentor.apply_photometric_aug(student_images)
            # Publish one prepared batch for forward/loss execution. The caller's
            # supervision flag controls whether its transformed GT is exposed.
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": geometric_targets if use_gt else None,
                "sample_ids": sample_ids,
            }

        if source_domain == "ir":
            # Real-target path: create paired views from one unlabeled IR batch.
            # No annotation-based SAGA is applied on this path.
            images, sample_ids = self._unpack_ir_batch(next(self._ir_iter))
            geometric_images, _, _ = self.augmentor.apply_weak_aug(images)
            teacher_images = geometric_images.clone()
            student_images = self.augmentor.apply_photometric_aug(geometric_images)
            # Publish aligned target views and identity without supervised GT.
            # The model keeps the baseline detector's existing forward interface.
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": None,
                "sample_ids": sample_ids,
            }
        raise ValueError(f"Unsupported physical source domain: {source_domain}")
