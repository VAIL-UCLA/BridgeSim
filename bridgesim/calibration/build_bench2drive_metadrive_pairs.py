#!/usr/bin/env python3
"""
Build a list of Bench2Drive/MetaDrive image pairs for matching scenarios.

The script mirrors `build_openscene_rap_image_dataset.py` by checking that every
candidate image exists on both sides before writing it to the output. Missing
pairs are skipped and recorded in a text file for inspection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Camera subdirectories created by convert_bench2drive.py
DEFAULT_CAMERA_DIRS: Sequence[str] = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
    "rgb_top_down",
)

DEFAULT_EXTENSIONS: Sequence[str] = (".jpg", ".png")


@dataclass
class PairStats:
    attempted: int = 0
    matched: int = 0
    missing_bench2drive: int = 0
    missing_metadrive: int = 0
    missing_bench2drive_scenarios: int = 0
    missing_metadrive_scenarios: int = 0

    @property
    def missing(self) -> int:
        return self.attempted - self.matched

    def add(self, other: "PairStats") -> None:
        self.attempted += other.attempted
        self.matched += other.matched
        self.missing_bench2drive += other.missing_bench2drive
        self.missing_metadrive += other.missing_metadrive
        self.missing_bench2drive_scenarios += other.missing_bench2drive_scenarios
        self.missing_metadrive_scenarios += other.missing_metadrive_scenarios


def normalize_scenario_name(name: str) -> str:
    """Strip archive suffixes so file and directory names align."""
    normalized = name
    if normalized.endswith(".tar.gz"):
        normalized = normalized[: -len(".tar.gz")]
    if normalized.endswith(".tar"):
        normalized = normalized[: -len(".tar")]
    return normalized


def discover_scenarios(root: Path) -> Dict[str, Path]:
    """Return a mapping of normalized scenario name -> directory for immediate children."""
    return {normalize_scenario_name(p.name): p for p in sorted(root.iterdir()) if p.is_dir()}


def list_frames(cam_dir: Path, extensions: Sequence[str]) -> MutableMapping[str, Path]:
    """Return stem -> full path for all frames under a camera directory."""
    frames: MutableMapping[str, Path] = {}
    for ext in extensions:
        for img_path in cam_dir.glob(f"*{ext}"):
            frames[img_path.stem] = img_path
    return frames


def collect_pairs_for_scenario(
    scenario: str,
    bench_path: Path,
    metadrive_path: Path,
    camera_dirs: Sequence[str],
    extensions: Sequence[str],
) -> Tuple[List[Mapping[str, str]], PairStats, List[str]]:
    """Gather all matching image pairs for a single scenario."""
    pairs: List[Mapping[str, str]] = []
    stats = PairStats()
    missing_notes: List[str] = []

    for cam_dir_name in camera_dirs:
        bench_cam_dir = bench_path / "camera" / cam_dir_name
        meta_cam_dir = metadrive_path / "camera" / cam_dir_name

        if not bench_cam_dir.exists() or not meta_cam_dir.exists():
            missing_notes.append(
                f"{scenario} {cam_dir_name}: missing "
                f"{'bench2drive' if not bench_cam_dir.exists() else ''}"
                f"{' ' if (not bench_cam_dir.exists() and not meta_cam_dir.exists()) else ''}"
                f"{'metadrive' if not meta_cam_dir.exists() else ''}".strip()
            )
            continue

        bench_frames = list_frames(bench_cam_dir, extensions)
        meta_frames = list_frames(meta_cam_dir, extensions)
        if not bench_frames and not meta_frames:
            continue

        for frame_id in sorted(set(bench_frames) | set(meta_frames)):
            bench_img = bench_frames.get(frame_id)
            meta_img = meta_frames.get(frame_id)
            stats.attempted += 1

            if bench_img and meta_img:
                stats.matched += 1
                pairs.append(
                    {
                        "scenario": scenario,
                        "camera": cam_dir_name,
                        "frame": frame_id,
                        "bench2drive_path": str(bench_img.resolve()),
                        "metadrive_path": str(meta_img.resolve()),
                    }
                )
            else:
                if not bench_img:
                    stats.missing_bench2drive += 1
                if not meta_img:
                    stats.missing_metadrive += 1
                missing_notes.append(
                    f"{scenario} {cam_dir_name} frame {frame_id}: "
                    f"{'bench2drive missing' if not bench_img else ''}"
                    f"{' and ' if (not bench_img and not meta_img) else ''}"
                    f"{'metadrive missing' if not meta_img else ''}"
                )

    return pairs, stats, missing_notes


def save_pairs_online(
    bench_root: Path,
    metadrive_root: Path,
    output_path: Path,
    missing_log_path: Path,
    camera_dirs: Sequence[str],
    extensions: Sequence[str],
    scenario_list: Sequence[Tuple[str, str]] | None = None,
) -> PairStats:
    """Stream pairs to disk while collecting coverage stats."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_log_path.parent.mkdir(parents=True, exist_ok=True)

    bench_scenarios = discover_scenarios(bench_root)
    metadrive_scenarios = discover_scenarios(metadrive_root)

    if scenario_list:
        base_iterable: List[Tuple[str, Path | None, Path | None, str]] = [
            (normalized, bench_scenarios.get(normalized), metadrive_scenarios.get(normalized), raw_name)
            for normalized, raw_name in scenario_list
        ]
    else:
        base_iterable = [
            (name, path, metadrive_scenarios.get(name), name) for name, path in bench_scenarios.items()
        ]

    iterator: Iterable[Tuple[str, Path | None, Path | None, str]] = base_iterable
    if tqdm:
        iterator = tqdm(base_iterable, total=len(base_iterable), desc="Scanning scenarios")

    summary = PairStats()
    missing_lines: List[str] = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for scenario, bench_path, meta_path, raw_name in iterator:
            if not bench_path:
                summary.missing_bench2drive_scenarios += 1
                missing_lines.append(f"{raw_name}: missing in Bench2Drive root {bench_root}")
                continue

            if not meta_path:
                summary.missing_metadrive_scenarios += 1
                missing_lines.append(f"{raw_name}: missing in MetaDrive root {metadrive_root}")
                continue

            pairs, stats, missing_notes = collect_pairs_for_scenario(
                scenario,
                bench_path,
                meta_path,
                camera_dirs,
                extensions,
            )
            summary.add(stats)
            missing_lines.extend(missing_notes)

            if pairs:
                json.dump({"scenario": scenario, "pairs": pairs}, out_f)
                out_f.write("\n")

    with open(missing_log_path, "w", encoding="utf-8") as miss_f:
        miss_f.write("\n".join(missing_lines))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Bench2Drive/MetaDrive image pairs for matching scenarios."
    )
    parser.add_argument(
        "--bench2drive-root",
        type=Path,
        required=True,
        help="Root directory containing Bench2Drive scenarios with camera subfolders.",
    )
    parser.add_argument(
        "--metadrive-root",
        type=Path,
        required=True,
        help="Root directory containing MetaDrive scenarios with camera subfolders.",
    )
    parser.add_argument(
        "--scenario-json",
        type=Path,
        help="Path to JSON/JSONL list of scenarios to check (e.g., bench2drive_mini_10.json keys).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench2drive_metadrive_pairs.jsonl"),
        help="Path to JSONL output where each line stores one scenario's matched pairs.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=Path("bench2drive_metadrive_missing.txt"),
        help="Path to the text file where missing pairs/scenarios are recorded.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERA_DIRS),
        help="Camera subdirectories to pair (relative to each scenario's camera folder).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Image extensions (with dot) to search for when pairing frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bench_root: Path = args.bench2drive_root
    metadrive_root: Path = args.metadrive_root
    output_path: Path = args.output
    missing_log_path: Path = args.missing_log
    camera_dirs: Sequence[str] = tuple(args.cameras)
    extensions: Sequence[str] = tuple(args.extensions)
    scenario_json: Path | None = args.scenario_json

    if not bench_root.exists():
        raise SystemExit(f"Bench2Drive root not found: {bench_root}")
    if not metadrive_root.exists():
        raise SystemExit(f"MetaDrive root not found: {metadrive_root}")

    scenario_list: List[Tuple[str, str]] | None = None
    if scenario_json:
        if not scenario_json.exists():
            raise SystemExit(f"Scenario JSON not found: {scenario_json}")
        with open(scenario_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            names = list(data.keys())
        elif isinstance(data, list):
            names = [str(item) for item in data]
        else:
            raise SystemExit("Scenario JSON must be an object mapping or a list of names.")
        scenario_list = [(normalize_scenario_name(name), name) for name in names]

    stats = save_pairs_online(
        bench_root,
        metadrive_root,
        output_path,
        missing_log_path,
        camera_dirs,
        extensions,
        scenario_list,
    )

    if stats.attempted == 0:
        print("No frames were checked. Ensure the roots contain scenario folders with camera images.")
        return

    print(f"Attempted frame matches: {stats.attempted}")
    print(f"Matched pairs: {stats.matched}")
    print(f"Missing Bench2Drive images: {stats.missing_bench2drive}")
    print(f"Missing MetaDrive images: {stats.missing_metadrive}")
    print(f"Scenarios missing in Bench2Drive root: {stats.missing_bench2drive_scenarios}")
    print(f"Scenarios missing in MetaDrive root: {stats.missing_metadrive_scenarios}")
    print(f"Pairs written to {output_path}")
    print(f"Missing entries written to {missing_log_path}")


if __name__ == "__main__":
    main()
