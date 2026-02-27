"""
Problem 4, Question 3 — Data augmentation strategies for retinal vessel segmentation.

Four strategies are implemented:
  1. geometric      — random horizontal/vertical flip + random rotation ±30°
  2. photometric    — random brightness & contrast jitter (image only)
  3. gaussian_noise — additive Gaussian noise (image only)
  4. zoom           — random crop + resize (image + mask)
  5. combined       — geometric + photometric + gaussian_noise + zoom

All augmentations are applied only during training.
Image and mask are always transformed consistently.
"""

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


# ─── Individual augmentation functions ───────────────────────────────────────

def aug_geometric(image, mask):
    """Random horizontal flip, vertical flip, and rotation ±30°."""
    # Horizontal flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask,  1)
    # Vertical flip
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 0)
        mask  = cv2.flip(mask,  0)
    # Random rotation ±30°
    angle = np.random.uniform(-30, 30)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    mask  = cv2.warpAffine(mask,  M, (w, h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT)
    return image, mask


def aug_photometric(image, mask):
    """Random brightness and contrast jitter applied to the image only."""
    image = image.astype(np.float32)
    # Brightness: additive offset ∈ [-40, 40]
    image += np.random.uniform(-40, 40)
    # Contrast: multiplicative factor ∈ [0.7, 1.3]
    mean  = image.mean()
    image = (image - mean) * np.random.uniform(0.7, 1.3) + mean
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, mask


def aug_gaussian_noise(image, mask, std_range=(5, 25)):
    """
    Additive Gaussian noise applied to the image only (mask unchanged).
    std is sampled uniformly from std_range each call.
    """
    std = np.random.uniform(*std_range)
    noise = np.random.randn(*image.shape).astype(np.float32) * std
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, mask


def aug_zoom(image, mask, scale_range=(0.7, 1.0)):
    """
    Random zoom-in: crop a random sub-region of scale s ∈ scale_range and
    resize it back to the original resolution.  Applied consistently to
    both image (bilinear) and mask (nearest-neighbour).
    """
    h, w = image.shape[:2]
    s = np.random.uniform(*scale_range)
    h_crop, w_crop = int(h * s), int(w * s)
    y0 = np.random.randint(0, h - h_crop + 1)
    x0 = np.random.randint(0, w - w_crop + 1)
    image = cv2.resize(image[y0:y0+h_crop, x0:x0+w_crop], (w, h),
                       interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask[y0:y0+h_crop, x0:x0+w_crop],  (w, h),
                       interpolation=cv2.INTER_NEAREST)
    return image, mask


def aug_combined(image, mask):
    """Apply geometric, photometric, gaussian_noise and zoom in sequence."""
    image, mask = aug_geometric(image, mask)
    image, mask = aug_photometric(image, mask)
    image, mask = aug_gaussian_noise(image, mask)
    image, mask = aug_zoom(image, mask)
    return image, mask


# ─── Augmented Dataset ────────────────────────────────────────────────────────

AUG_FN = {
    'none':          lambda img, msk: (img, msk),
    'geometric':     aug_geometric,
    'photometric':   aug_photometric,
    'gaussian_noise': aug_gaussian_noise,
    'zoom':          aug_zoom,
    'combined':      aug_combined,
}


class AugmentedDataset(Dataset):
    """
    Drop-in replacement for GetDataset that applies one of the four
    augmentation strategies during training.

    Args:
        images_path: list of image file paths
        masks_path:  list of mask file paths
        augmentation: one of 'none' | 'geometric' | 'photometric' |
                      'gaussian_noise' | 'zoom' | 'combined'
    """
    def __init__(self, images_path, masks_path, augmentation='none'):
        assert augmentation in AUG_FN, \
            f"Unknown augmentation '{augmentation}'. Choose from {list(AUG_FN)}"
        self.images_path = images_path
        self.masks_path  = masks_path
        self.n_samples   = len(images_path)
        self.aug_fn      = AUG_FN[augmentation]

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)   # H×W×3, uint8
        mask  = cv2.imread(self.masks_path[index],  cv2.IMREAD_GRAYSCALE)  # H×W, uint8

        image, mask = self.aug_fn(image, mask)

        image = (image / 255.0).astype(np.float32)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))

        mask  = (mask / 255.0).astype(np.float32)
        mask  = torch.from_numpy(np.expand_dims(mask, axis=0))

        return image, mask

    def __len__(self):
        return self.n_samples
