from torchvision import transforms as T
import torch
import random
import torchvision.transforms.functional as TF
from typing import List, Dict, Optional, Tuple

def default_rgb_transform() -> T.Compose:
    """ToTensor only — FCOS GeneralizedRCNNTransform handles ImageNet normalisation."""
    return T.Compose([T.ToTensor()])


def default_ir_transform() -> T.Compose:
    """IR (thermal JPEG): convert to 3-channel float [0,1]. FCOS expects 3-ch input."""
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])

class StudentAugmentor:
    """
    Handles standard augmentations for Student and Teacher models.
    Supports Batch Tensors [B, 3, H, W] and Torchvision-style targets.
    """
    def __init__(self, config):
        """
        Args:
            config: The TrainingConfig object containing 'aug' settings.
        """
        self.cfg = config.aug

    def apply_weak_aug(
        self, 
        images: torch.Tensor, 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict]], bool]:
        """
        Applies a stochastic horizontal flip to the entire batch.
        
        Returns:
            aug_images: Flipped images.
            aug_targets: Updated bounding boxes (if targets provided).
            did_flip: Boolean flag indicating if the flip was performed.
        """
        W = images.shape[-1]
        did_flip = False
        
        if random.random() < self.cfg.hflip_prob:
            did_flip = True
            images = torch.flip(images, dims=[-1])
            
            if targets is not None:
                new_targets = []
                for t in targets:
                    boxes = t["boxes"] # Expected format: [N, 4] (xyxy)
                    if boxes.numel() > 0:
                        flipped_boxes = boxes.clone()
                        # x1_new = W - x2, x2_new = W - x1
                        flipped_boxes[:, 0] = W - boxes[:, 2]
                        flipped_boxes[:, 2] = W - boxes[:, 0]
                        new_targets.append({**t, "boxes": flipped_boxes})
                    else:
                        new_targets.append(t)
                targets = new_targets
                
        return images, targets, did_flip

    def apply_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies blur, brightness, and contrast adjustments.
        This does NOT affect bounding box coordinates.
        """
        # 1. Gaussian Blur
        if random.random() < self.cfg.blur_prob:
            sigma = random.uniform(0.1, self.cfg.blur_sigma_max)
            # kernel_size must be odd
            images = TF.gaussian_blur(images, kernel_size=[3, 3], sigma=[sigma, sigma])

        # 2. Brightness
        if random.random() < self.cfg.brightness_prob:
            factor = 1.0 + random.uniform(-self.cfg.brightness_mag, self.cfg.brightness_mag)
            images = torch.clamp(images * factor, 0.0, 1.0)

        # 3. Contrast
        if random.random() < self.cfg.contrast_prob:
            mean = images.mean(dim=[-1, -2], keepdim=True)
            factor = 1.0 + random.uniform(-self.cfg.contrast_mag, self.cfg.contrast_mag)
            images = torch.clamp((images - mean) * factor + mean, 0.0, 1.0)

        return images

    def apply_strong_aug(
        self, 
        images: torch.Tensor, 
        targets: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """
        Strong Augmentation: Combined Geometric (Weak) + Photometric transformations.
        Used for the Student model.
        """
        images, targets, _ = self.apply_weak_aug(images, targets)
        images = self.apply_photometric_aug(images)
        
        return images, targets
