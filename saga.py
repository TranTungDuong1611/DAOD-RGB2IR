"""Blend annotated RGB object regions toward grayscale; retain the background."""

import torch


class SoftSAGA:
    """Apply deterministic RGB/grayscale interpolation inside bounding boxes."""

    _LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])

    @staticmethod
    def _object_mask(boxes, height, width, device):
        mask = torch.zeros(height, width, dtype=torch.bool, device=device)
        for box in boxes:
            x1 = max(0, int(box[0].item()))
            y1 = max(0, int(box[1].item()))
            x2 = min(width, int(box[2].item()))
            y2 = min(height, int(box[3].item()))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
        return mask

    def apply(self, image: torch.Tensor, boxes: torch.Tensor | None, alpha: float):
        if boxes is None or boxes.numel() == 0:
            return image.clone()
        _, height, width = image.shape
        weights = self._LUMA_WEIGHTS.to(device=image.device, dtype=image.dtype)
        gray = (image * weights.view(3, 1, 1)).sum(dim=0, keepdim=True)
        mask = self._object_mask(boxes, height, width, image.device)
        blended = alpha * image + (1.0 - alpha) * gray
        return torch.where(mask.unsqueeze(0), blended, image)

    def apply_to_batch(self, images, batch_boxes, alpha):
        if len(images) != len(batch_boxes):
            raise ValueError("SAGA requires one box tensor per image")
        return torch.stack([
            self.apply(image, boxes, alpha)
            for image, boxes in zip(images, batch_boxes)
        ])
