"""Prepare aligned views using routing decisions supplied by the caller."""

from typing import Iterable

import torch


class RouteDataFlow:
    """Own loader cursors and view preparation, without reading training config."""

    def __init__(self, rgb_loader, ir_loader, *, device, augmentor):
        self.device = device
        self.augmentor = augmentor
        # Construct lazy cursors in the same order as the original trainer.
        self._rgb_iter = iter(self._infinite(rgb_loader))
        self._ir_iter = None if ir_loader is None else iter(self._infinite(ir_loader))

    @staticmethod
    def _infinite(loader: Iterable):
        while True:
            yielded = False
            for batch in loader:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("training DataLoader is empty")

    def _move_images(self, images):
        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        return tuple(image.to(self.device) for image in images)

    def _move_targets(self, targets):
        moved = []
        for target in targets:
            moved.append({
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            })
        return moved

    @staticmethod
    def _ids_from_targets(targets) -> tuple[str, ...]:
        ids = tuple(target.get("stem") for target in targets)
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("labeled batches must carry one non-empty sample stem per image")
        return ids

    def _unpack_rgb_batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("RGB loader must return (images, targets[, sample_ids])")
        images, targets = batch[0], list(batch[1])
        ids = tuple(batch[2]) if len(batch) == 3 else self._ids_from_targets(targets)
        if len(ids) != len(targets):
            raise ValueError("RGB sample_ids must match target count")
        return self._move_images(images), self._move_targets(targets), ids

    def _unpack_ir_batch(self, batch):
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
    ):
        """Execute the supplied source/augmentation settings on one batch."""
        if source_domain == "rgb":
            images, targets, sample_ids = self._unpack_rgb_batch(next(self._rgb_iter))
            geometric_images, geometric_targets, _ = self.augmentor.apply_weak_aug(
                images, targets
            )
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
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": geometric_targets if use_gt else None,
                "sample_ids": sample_ids,
                # SAGA changes appearance but the physical source is RGB.
                # The IR residual neck is reserved for real thermal frames.
                "student_domain": "rgb",
                "teacher_domain": "rgb",
            }

        if source_domain == "ir":
            if self._ir_iter is None:
                raise ValueError("IR route requested without an IR DataLoader")
            images, sample_ids = self._unpack_ir_batch(next(self._ir_iter))
            geometric_images, _, _ = self.augmentor.apply_weak_aug(images)
            teacher_images = geometric_images.clone()
            student_images = self.augmentor.apply_photometric_aug(geometric_images)
            return {
                "student_images": student_images,
                "teacher_images": teacher_images,
                "targets": None,
                "sample_ids": sample_ids,
                # Both views originate from a real thermal frame. SAGA level
                # names describe routing policy, not the physical input domain.
                "student_domain": "ir",
                "teacher_domain": "ir",
            }
        raise ValueError(f"Unsupported physical source domain: {source_domain}")
