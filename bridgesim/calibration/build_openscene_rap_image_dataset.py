#!/usr/bin/env python3
"""
Filter NavSim image references by checking availability in both rendered and real sensor blobs.

The script walks every log pickle under the NavSim logs root, verifies that each camera
image exists in both rendered and raw sensor blob roots, reports coverage statistics,
and writes a JSONL file where each line is {log_filename: [[frame_idx, cam_name], ...]}.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DEFAULT_SENSOR_ROOT = Path("/avl-west/nuplan/nuplan-v1.1/sensor_blobs")
DEFAULT_RENDERED_ROOT = Path("/avl-west/nuplan/nuplan-v1.1/rendered_sensor_blobs")
DEFAULT_CAM_DIRS: Sequence[str] = (
    "CAM_B0",
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
)


@dataclass
class CoverageStats:
    total: int = 0
    present: int = 0
    missing_sensor: int = 0
    missing_rendered: int = 0

    def add(self, other: "CoverageStats") -> None:
        self.total += other.total
        self.present += other.present
        self.missing_sensor += other.missing_sensor
        self.missing_rendered += other.missing_rendered

    @property
    def missing(self) -> int:
        return self.total - self.present


def iter_log_pickles(log_root: Path) -> Iterable[Path]:
    """Yield every pickle log file under the provided root."""
    return (p for p in sorted(log_root.rglob("*.pkl")) if p.is_file())


def process_log_file(
    log_path: Path,
    sensor_root: Path,
    rendered_root: Path,
    cam_names: Sequence[str],
) -> Tuple[List[Tuple[int, str]], CoverageStats]:
    """Return valid (frame_idx, cam_name) pairs for a single log and coverage stats."""
    stats = CoverageStats()
    matches: List[Tuple[int, str]] = []

    with open(log_path, "rb") as f:
        scene_dict_list = pickle.load(f)

    for frame in scene_dict_list:
        frame_idx = frame.get("frame_idx")
        cams = frame.get("cams") or {}
        for cam_name in cam_names:
            cam_entry = cams.get(cam_name)
            if not cam_entry:
                continue
            data_path = cam_entry.get("data_path")
            if not data_path:
                continue

            stats.total += 1
            sensor_exists = (sensor_root / data_path).exists()
            rendered_exists = (rendered_root / data_path).exists()

            if sensor_exists and rendered_exists:
                stats.present += 1
                matches.append((frame_idx, cam_name))
            else:
                if not sensor_exists:
                    stats.missing_sensor += 1
                if not rendered_exists:
                    stats.missing_rendered += 1

    return matches, stats


def save_matches_online(
    log_files: Iterable[Path],
    sensor_root: Path,
    rendered_root: Path,
    cam_names: Sequence[str],
    output_path: Path,
) -> CoverageStats:
    """Process logs sequentially and stream results to disk to limit memory use."""
    summary = CoverageStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(log_files, desc="Checking logs") if tqdm else log_files
    with open(output_path, "w", encoding="utf-8") as out_f:
        for log_file in iterator:
            matches, stats = process_log_file(log_file, sensor_root, rendered_root, cam_names)
            summary.add(stats)
            if tqdm:
                iterator.set_postfix(checked=summary.total, pairs=summary.present, refresh=False)
            if matches:
                serializable_matches = [[int(m[0]) if m[0] is not None else None, m[1]] for m in matches]
                json.dump({log_file.stem: serializable_matches}, out_f)
                out_f.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter NavSim images present in both rendered and real blobs.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/closed-loop-e2e/navsim_logs"),
        help="Root directory containing NavSim pickle logs (searched recursively).",
    )
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=DEFAULT_SENSOR_ROOT,
        help="Root directory of sensor_blobs.",
    )
    parser.add_argument(
        "--rendered-root",
        type=Path,
        default=DEFAULT_RENDERED_ROOT,
        help="Root directory of rendered_sensor_blobs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rap_image_matches.jsonl"),
        help="Path to JSONL output. Each line is {log_filename: [[frame_idx, cam_name], ...]}",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=list(DEFAULT_CAM_DIRS),
        help="Camera names to check.",
    )
    args = parser.parse_args()

    log_root = args.log_root
    sensor_root = args.sensor_root
    rendered_root = args.rendered_root
    output_path = args.output
    cam_names = tuple(args.cams)

    log_files = list(iter_log_pickles(log_root))
    if not log_files:
        print(f"No pickle logs found under {log_root}")
        return
    print(f"Processing {len(log_files)} logs from {log_root}...")
    summary = save_matches_online(log_files, sensor_root, rendered_root, cam_names, output_path)

    if summary.total == 0:
        print("No camera entries found in logs.")
        return

    present_rate = summary.present / summary.total
    missing_rate = summary.missing / summary.total
    print(f"Checked {len(log_files)} logs, {summary.total} camera images.")
    print(f"Present in both blobs: {summary.present} ({present_rate:.2%})")
    print(f"Missing in at least one blob: {summary.missing} ({missing_rate:.2%})")
    print(f"Missing in sensor_blobs only: {summary.missing_sensor}")
    print(f"Missing in rendered_sensor_blobs only: {summary.missing_rendered}")
    print(f"Per-log matches written to {output_path}")

if __name__ == "__main__":
    main()
