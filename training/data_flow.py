"""Prepare aligned views using routing decisions supplied by the caller."""

from typing import Iterable

import torch


class RouteDataFlow:
    """Own loader cursors and view preparation, without reading training config."""

    def __init__(self, rgb_loader, ir_loader, *, device, augmentor):
        # Keep execution dependencies here; the trainer supplies the source,
        # supervision flag, and appearance settings separately for each step.
        self.device = device
        self.augmentor = augmentor
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

    def _move_images(self, images):
        # Accept both supported image batch representations at the device seam.
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        return tuple(image.to(self.device) for image in images)

    def _move_targets(self, targets):
        # Move tensor annotations while retaining sample metadata used to align KD.
        moved = []
        for target in targets:
            moved.append({
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            })
        return moved

    @staticmethod
    def _ids_from_targets(targets) -> tuple[str, ...]:
        # Support labeled loaders that carry identity inside targets rather than
        # as a separate batch field; correspondence must still be explicit.
        ids = tuple(target.get("stem") for target in targets)
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("labeled batches must carry one non-empty sample stem per image")
        return ids

    def _unpack_rgb_batch(self, batch):
        # Normalize the labeled source contract before view creation so images,
        # annotations, and identity stay associated throughout augmentation.
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("RGB loader must return (images, targets[, sample_ids])")
        images, targets = batch[0], list(batch[1])
        ids = tuple(batch[2]) if len(batch) == 3 else self._ids_from_targets(targets)
        if len(ids) != len(targets):
            raise ValueError("RGB sample_ids must match target count")
        return self._move_images(images), self._move_targets(targets), ids

    def _unpack_ir_batch(self, batch):
        # Normalize the unlabeled target contract; identity is required even
        # though real IR training batches do not provide supervised targets.
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
        # Apply the caller's appearance setting to annotated object regions.
        # Geometry and sample correspondence remain shared between both views.
        if targets is None:
            return images.clone()
        from saga import SoftSAGA
        boxes_list = [target["boxes"] for target in targets]
        return SoftSAGA().apply_to_batch(images, boxes_list, alpha)

    def prepare(
        self,
        *,
        source_domain: str,
        use_gt: bool,
        student_alpha: float | None,
        teacher_alpha: float | None,
        capture_geometry: bool = False,
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
            student_images = self._get_saga_images(
                geometric_images,
                geometric_targets,
                student_alpha,
            )
            teacher_images = self._get_saga_images(
                geometric_images,
                geometric_targets,
                teacher_alpha,
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
            if self._ir_iter is None:
                raise ValueError("IR route requested without an IR DataLoader")
            images, sample_ids = self._unpack_ir_batch(next(self._ir_iter))
            geometric_images, _, did_flip = self.augmentor.apply_weak_aug(images)
            teacher_images = geometric_images.clone()
            student_images = self.augmentor.apply_photometric_aug(geometric_images)
            # Publish aligned target views and identity without supervised GT.
            # The model keeps the baseline detector's existing forward interface.
            result = {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": None,
                "sample_ids": sample_ids,
            }
            # Audit geometry is a separate side channel. IR supervised targets
            # remain absent even when annotations are available for diagnostics.
            if capture_geometry:
                result["diagnostic_hflip"] = did_flip
            return result
        raise ValueError(f"Unsupported physical source domain: {source_domain}")
