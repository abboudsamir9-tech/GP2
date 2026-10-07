"""Shared, scriptable, non-mutating alignment of normalized landmark windows.

46-joint contract: left hand 0:63, right hand 63:126, pose 126:138.
Exact repeated tails are a padding heuristic, not original length metadata.
"""

from __future__ import annotations

import torch


ALIGNMENT_VERSION = "dominant_right_neutral_prefix_v1"


def hand_coordinate_variances(inputs: torch.Tensor) -> torch.Tensor:
    """Per-hand temporal coordinate variance, excluding neutral zero frames."""
    hands = inputs[..., :126].reshape(inputs.shape[0], inputs.shape[1], 2, 63)
    present = (hands != 0).any(dim=-1)
    weights = present.to(inputs.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (hands * weights).sum(dim=1) / count
    variance = ((hands - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
    return variance.mean(dim=-1)


def dominant_left_mask(inputs: torch.Tensor) -> torch.Tensor:
    """Only mirror a moving left hand with a structurally absent right hand.

    A visible but static second hand is conservatively kept bilateral: motion
    alone cannot distinguish a resting hand from a two-handed sign's anchor.
    """
    variances = hand_coordinate_variances(inputs)
    right_absent = (inputs[..., 63:126] == 0).all(dim=-1).all(dim=1)
    return (variances[:, 0] > 1e-10) & right_absent


def canonicalize_dominant_hand(inputs: torch.Tensor) -> torch.Tensor:
    """Mirror X and swap anatomical sides, including shoulders and elbows."""
    joints = inputs.reshape(inputs.shape[0], inputs.shape[1], 46, 3)
    # Mirroring the complete skeleton requires swapping pose pairs too, so
    # body/hand anatomical sides stay consistent and the shoulder origin holds.
    swapped = torch.cat((joints[:, :, 21:42], joints[:, :, :21],
                         joints[:, :, 43:44], joints[:, :, 42:43],
                         joints[:, :, 45:46], joints[:, :, 44:45]), dim=2)
    mirrored = torch.stack((-swapped[..., 0], swapped[..., 1], swapped[..., 2]), dim=-1)
    selected = torch.where(dominant_left_mask(inputs)[:, None, None, None], mirrored, joints)
    return selected.reshape_as(inputs).contiguous()


def repeated_tail_padding_lengths(inputs: torch.Tensor) -> torch.Tensor:
    """Detect >=4 identical suffix frames; never reinterpret a constant clip."""
    equal_to_last = (inputs == inputs[:, -1:, :]).all(dim=-1)
    suffix = torch.cumprod(equal_to_last.flip([1]).to(torch.int64), dim=1).sum(dim=1)
    usable = (suffix >= 4) & (suffix < inputs.shape[1])
    return torch.where(usable, suffix - 1, torch.zeros_like(suffix))


def fifo_neutral_context(inputs: torch.Tensor) -> torch.Tensor:
    """Replace detectable repeated tails with neutral *leading* context.

    Real motion is kept in order and at its original cadence, ending at the
    latest frame as in a causal FIFO. Missing pre-sign context has zero hands
    and the first normalized pose. This is an explicit approximation: exact
    source lengths and original pre-sign poses cannot be recovered from HDF5.
    """
    padding = repeated_tail_padding_lengths(inputs)
    timeline = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
    indices = (timeline - padding.unsqueeze(1)).clamp_min(0)
    shifted = torch.gather(inputs, 1, indices.unsqueeze(-1).expand(-1, -1, inputs.shape[2]))
    neutral = torch.cat((torch.zeros_like(inputs[:, :1, :126]), inputs[:, :1, 126:]), dim=-1)
    return torch.where((timeline < padding.unsqueeze(1)).unsqueeze(-1), neutral, shifted).contiguous()


def align_feature_window(inputs: torch.Tensor) -> torch.Tensor:
    """Idempotent transform used by dataset, FIFO, eager model and TorchScript."""
    return fifo_neutral_context(canonicalize_dominant_hand(inputs))


__all__ = ["ALIGNMENT_VERSION", "align_feature_window", "canonicalize_dominant_hand",
           "dominant_left_mask", "fifo_neutral_context", "hand_coordinate_variances",
           "repeated_tail_padding_lengths"]
