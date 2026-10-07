"""Training-only perturbations of normalized 46-joint landmark windows."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class LandmarkAugmentor:
    """Apply small spatial and temporal changes without inventing a missing hand."""

    def __init__(
        self,
        *,
        noise_std: float = 0.015,
        scale_range: tuple[float, float] = (0.92, 1.08),
        speed_range: tuple[float, float] = (0.85, 1.15),
        max_rotation_degrees: float = 5.0,
    ) -> None:
        if noise_std < 0 or scale_range[0] <= 0 or speed_range[0] <= 0:
            raise ValueError("augmentation scales and noise must be non-negative")
        if scale_range[0] > scale_range[1] or speed_range[0] > speed_range[1]:
            raise ValueError("augmentation ranges must be ordered")
        if max_rotation_degrees < 0:
            raise ValueError("max_rotation_degrees must be non-negative")
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.speed_range = speed_range
        self.max_rotation_degrees = max_rotation_degrees

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand(()))

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        if window.shape != (45, 138) or window.dtype != torch.float32:
            raise ValueError("augmentation expects a float32 (45, 138) window")
        # One factor for all joints preserves each frame's geometric layout.
        speed = self._uniform(*self.speed_range)
        resampled_length = max(2, round(45 / speed))
        resampled = F.interpolate(
            window.transpose(0, 1).unsqueeze(0), size=resampled_length,
            mode="linear", align_corners=True,
        )
        # Crop slowed sequences or repeat-pad faster ones around their center.
        # Resizing straight back to 45 would cancel the cadence perturbation.
        if resampled_length >= 45:
            start = (resampled_length - 45) // 2
            resampled = resampled[:, :, start:start + 45]
        else:
            padding = 45 - resampled_length
            resampled = F.pad(
                resampled, (padding // 2, padding - padding // 2),
                mode="replicate",
            )
        output = resampled.squeeze(0).transpose(0, 1).contiguous()

        # Empty hand blocks encode structural absence, never sensor noise.
        neutral_frames = [
            (output[:, block] == 0).all(dim=1)
            for block in (slice(0, 63), slice(63, 126))
        ]
        joints = output.reshape(45, 46, 3)
        angle = math.radians(self._uniform(
            -self.max_rotation_degrees, self.max_rotation_degrees
        ))
        cosine, sine = math.cos(angle), math.sin(angle)
        x = joints[..., 0].clone()
        y = joints[..., 1].clone()
        joints[..., 0] = cosine * x - sine * y
        joints[..., 1] = sine * x + cosine * y
        joints.mul_(self._uniform(*self.scale_range))
        if self.noise_std:
            joints.add_(torch.randn_like(joints) * self.noise_std)

        # Restore the normalization origin; then restore neutral hands exactly.
        shoulder_midpoint = (joints[:, 42] + joints[:, 43]) / 2.0
        joints.sub_(shoulder_midpoint.unsqueeze(1))
        output = joints.reshape(45, 138)
        for block, neutral in zip((slice(0, 63), slice(63, 126)),
                                  neutral_frames, strict=True):
            output[neutral, block] = 0.0
        return output.contiguous()


__all__ = ["LandmarkAugmentor"]
