"""Extract ASL Citizen landmark audits and signer-independent HDF5 windows."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.contracts import FEATURE_COUNT
from asl_stereo.dataset import (
    ASLCitizenReader,
    BatchFeatureExtractor,
    partition_by_signer,
    write_dataset_h5,
)

LOGGER = logging.getLogger("asl_stereo.dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract signer-independent ASL Citizen feature windows."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "metadata.csv",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "asl_dataset.h5",
    )
    parser.add_argument(
        "--audit-csv-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "csv_audits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--stride", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    reader = ASLCitizenReader(args.metadata_csv, args.raw_dir)
    partitions = partition_by_signer(reader.metadata_df, seed=args.seed)
    glosses = sorted(str(value) for value in reader.metadata_df["gloss_label"].unique())
    class_mapping = {gloss: index for index, gloss in enumerate(glosses)}
    partition_features: dict[str, np.ndarray] = {}
    partition_labels: dict[str, np.ndarray] = {}
    statistics: dict[str, dict[str, int]] = {}

    with BatchFeatureExtractor(
        args.audit_csv_dir,
        window_size=args.window_size,
        stride=args.stride,
    ) as extractor:
        for partition_name in ("train", "val", "test"):
            metadata = partitions[partition_name]
            feature_chunks: list[np.ndarray] = []
            label_chunks: list[np.ndarray] = []
            discarded = 0
            iterator = reader.iter_metadata(metadata)
            for video_id, gloss, _signer_id, video_path in tqdm(
                iterator,
                total=len(metadata),
                desc=f"Extracting {partition_name}",
                unit="video",
            ):
                result = extractor.extract_video(video_id, video_path)
                if not result.accepted:
                    discarded += 1
                    LOGGER.warning("Discarded %s: %s", video_id, result.discarded_reason)
                    continue
                feature_chunks.append(result.windows)
                label_chunks.append(
                    np.full(
                        result.windows.shape[0],
                        class_mapping[gloss],
                        dtype=np.int64,
                    )
                )

            if feature_chunks:
                features = np.ascontiguousarray(
                    np.concatenate(feature_chunks, axis=0), dtype=np.float32
                )
                labels = np.concatenate(label_chunks).astype(np.int64, copy=False)
            else:
                features = np.empty(
                    (0, args.window_size, FEATURE_COUNT), dtype=np.float32
                )
                labels = np.empty(0, dtype=np.int64)
            partition_features[partition_name] = features
            partition_labels[partition_name] = labels
            statistics[partition_name] = {
                "signers": len(getattr(partitions, f"{partition_name}_signers")),
                "sequences": len(metadata),
                "discarded": discarded,
                "windows": features.shape[0],
            }

    write_dataset_h5(
        args.output_h5,
        partition_features,
        partition_labels,
        class_mapping=class_mapping,
        signer_ids=partitions.signer_ids,
        window_size=args.window_size,
    )
    _print_statistics(statistics, len(partitions.train_signers | partitions.val_signers | partitions.test_signers))
    print(f"HDF5 artifact: {args.output_h5}")
    return 0


def _print_statistics(
    statistics: dict[str, dict[str, int]], unique_signer_count: int
) -> None:
    print(f"Unique signers: {unique_signer_count}")
    print(f"Total sequences: {sum(item['sequences'] for item in statistics.values())}")
    print("Partition  Signers  Sequences  Discarded  Windows")
    for partition in ("train", "val", "test"):
        item = statistics[partition]
        print(
            f"{partition:<9}  {item['signers']:>7}  {item['sequences']:>9}  "
            f"{item['discarded']:>9}  {item['windows']:>7}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
