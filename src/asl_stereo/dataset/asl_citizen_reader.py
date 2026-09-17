"""Validated ASL Citizen metadata and video enumeration."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("video_id", "gloss_label", "signer_id")


class ASLCitizenReader:
    def __init__(self, metadata_csv: str | Path, raw_dir: str | Path) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.raw_dir = Path(raw_dir)
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"metadata CSV not found: {self.metadata_csv}")
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"raw video directory not found: {self.raw_dir}")

        metadata = pd.read_csv(
            self.metadata_csv,
            dtype={column: "string" for column in REQUIRED_COLUMNS},
        )
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(metadata.columns))
        if missing_columns:
            raise ValueError(f"metadata CSV is missing columns: {missing_columns}")
        metadata = metadata.loc[:, REQUIRED_COLUMNS].copy()
        for column in REQUIRED_COLUMNS:
            if metadata[column].isna().any():
                raise ValueError(f"metadata column {column!r} contains missing values")
            metadata[column] = metadata[column].str.strip()
            if metadata[column].eq("").any():
                raise ValueError(f"metadata column {column!r} contains empty values")
        self.metadata_df = metadata
        self._paths = self._validate_video_paths(metadata)

    def _validate_video_paths(self, metadata: pd.DataFrame) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        missing: list[Path] = []
        for video_id in metadata["video_id"]:
            video_id = str(video_id)
            if Path(video_id).name != video_id:
                raise ValueError(f"video_id must not contain path components: {video_id!r}")
            filename = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
            path = self.raw_dir / filename
            paths[video_id] = path
            if not path.is_file():
                missing.append(path)
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            suffix = " …" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing)} metadata video file(s) are missing: {preview}{suffix}"
            )
        return paths

    def video_path(self, video_id: str) -> Path:
        try:
            return self._paths[str(video_id)]
        except KeyError as error:
            raise KeyError(f"unknown video_id: {video_id!r}") from error

    def iter_metadata(
        self, metadata_df: pd.DataFrame | None = None
    ) -> Iterator[tuple[str, str, str, Path]]:
        frame = self.metadata_df if metadata_df is None else metadata_df
        for row in frame.loc[:, REQUIRED_COLUMNS].itertuples(index=False, name=None):
            video_id, gloss, signer_id = (str(value) for value in row)
            yield video_id, gloss, signer_id, self.video_path(video_id)

    def __iter__(self) -> Iterator[tuple[str, str, str, Path]]:
        return self.iter_metadata()

    def __len__(self) -> int:
        return len(self.metadata_df)


__all__ = ["ASLCitizenReader", "REQUIRED_COLUMNS"]
