"""Validated stereo projection-matrix loading with explicit availability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    P_front: npt.NDArray[np.float64] | None = None
    P_side: npt.NDArray[np.float64] | None = None
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if (self.P_front is None) != (self.P_side is None):
            raise ValueError("both projection matrices must be supplied together")
        if self.P_front is None:
            return
        front = _validated_projection("P_front", self.P_front)
        side = _validated_projection("P_side", self.P_side)
        object.__setattr__(self, "P_front", front)
        object.__setattr__(self, "P_side", side)

    @property
    def available(self) -> bool:
        return self.P_front is not None and self.P_side is not None

    @classmethod
    def from_file(cls, path: str | Path) -> "StereoCalibration":
        calibration_path = Path(path)
        if not calibration_path.is_file():
            return cls(source_path=calibration_path)

        suffix = calibration_path.suffix.lower()
        if suffix == ".npz":
            with np.load(calibration_path, allow_pickle=False) as data:
                try:
                    front = data["P_front"]
                    side = data["P_side"]
                except KeyError as error:
                    raise ValueError("NPZ calibration requires P_front and P_side") from error
        elif suffix == ".json":
            with calibration_path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict) or "P_front" not in data or "P_side" not in data:
                raise ValueError("JSON calibration requires P_front and P_side")
            front = data["P_front"]
            side = data["P_side"]
        else:
            raise ValueError("calibration file must use .json or .npz")
        return cls(P_front=front, P_side=side, source_path=calibration_path)

    @classmethod
    def from_default(
        cls, path: str | Path = "configs/calibration_params.json"
    ) -> "StereoCalibration":
        return cls.from_file(path)


def _validated_projection(
    name: str, matrix: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    output = np.asarray(matrix, dtype=np.float64)
    if output.shape != (3, 4):
        raise ValueError(f"{name} must have shape (3, 4)")
    if not np.isfinite(output).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.linalg.matrix_rank(output) < 3:
        raise ValueError(f"{name} must have rank 3")
    output = np.array(output, dtype=np.float64, order="C", copy=True)
    output.setflags(write=False)
    return output


__all__ = ["StereoCalibration"]
