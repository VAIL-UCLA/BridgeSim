#!/usr/bin/env python3
"""
Build per-frame Bench2Drive/MetaDrive image bundles for UniAD.

Each output line corresponds to a single frame that has all required UniAD
cameras present in both Bench2Drive and MetaDrive, plus references to a fixed
history window (default: previous 3 frames spaced by sample_interval) so
downstream code can construct history BEVs. Frames missing any required camera
on either side are skipped and recorded in a log file.
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

# UniAD expects six surround cameras on Bench2Drive.
UNIAD_CAMERA_DIRS: Sequence[str] = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
)

DEFAULT_EXTENSIONS: Sequence[str] = (".jpg", ".png")


@dataclass
class FrameStats:
    attempted_frames: int = 0
    matched_frames: int = 0
    missing_frames_bench2drive: int = 0
    missing_frames_metadrive: int = 0
    missing_bench2drive_scenarios: int = 0
    missing_metadrive_scenarios: int = 0

    @property
    def missing_frames(self) -> int:
        return self.attempted_frames - self.matched_frames

    def add(self, other: "FrameStats") -> None:
        self.attempted_frames += other.attempted_frames
        self.matched_frames += other.matched_frames
        self.missing_frames_bench2drive += other.missing_frames_bench2drive
        self.missing_frames_metadrive += other.missing_frames_metadrive
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


def collect_frames_for_scenario(
    scenario: str,
    bench_path: Path,
    metadrive_path: Path,
    camera_dirs: Sequence[str],
    extensions: Sequence[str],
    history_length: int,
    sample_interval: int,
    allow_short_history: bool,
) -> Tuple[List[Mapping[str, object]], FrameStats, List[str]]:
    """
    Gather per-frame camera bundles for a single scenario, including links to the
    previous valid frames (default: 3, spaced by sample_interval) when available.

    Only frames that contain every required camera in both Bench2Drive and
    MetaDrive are included in the output list.
    """
    stats = FrameStats()
    missing_notes: List[str] = []
    frames_payload: List[Mapping[str, object]] = []

    bench_frames_by_cam: Dict[str, MutableMapping[str, Path]] = {}
    meta_frames_by_cam: Dict[str, MutableMapping[str, Path]] = {}

    for cam_dir_name in camera_dirs:
        bench_cam_dir = bench_path / "camera" / cam_dir_name
        meta_cam_dir = metadrive_path / "camera" / cam_dir_name

        if not bench_cam_dir.exists():
            missing_notes.append(f"{scenario} {cam_dir_name}: missing Bench2Drive camera directory {bench_cam_dir}")
        else:
            bench_frames_by_cam[cam_dir_name] = list_frames(bench_cam_dir, extensions)

        if not meta_cam_dir.exists():
            missing_notes.append(f"{scenario} {cam_dir_name}: missing MetaDrive camera directory {meta_cam_dir}")
        else:
            meta_frames_by_cam[cam_dir_name] = list_frames(meta_cam_dir, extensions)

    # If any required camera directory is absent, the scenario cannot produce valid frames.
    if len(bench_frames_by_cam) < len(camera_dirs) or len(meta_frames_by_cam) < len(camera_dirs):
        return frames_payload, stats, missing_notes

    all_frame_ids = set()
    for cam_dir_name in camera_dirs:
        all_frame_ids.update(bench_frames_by_cam[cam_dir_name].keys())
        all_frame_ids.update(meta_frames_by_cam[cam_dir_name].keys())

    history_frame_ids: List[str] = []
    history_bench_frames: List[Dict[str, str]] = []
    history_meta_frames: List[Dict[str, str]] = []
    history_lookup: Dict[int, Tuple[str, Dict[str, str], Dict[str, str]]] = {}

    for frame_id in sorted(all_frame_ids):
        stats.attempted_frames += 1
        missing_bench_cams = [cam for cam in camera_dirs if frame_id not in bench_frames_by_cam[cam]]
        missing_meta_cams = [cam for cam in camera_dirs if frame_id not in meta_frames_by_cam[cam]]

        if missing_bench_cams or missing_meta_cams:
            if missing_bench_cams:
                stats.missing_frames_bench2drive += 1
            if missing_meta_cams:
                stats.missing_frames_metadrive += 1
            missing_notes.append(
                f"{scenario} frame {frame_id}: "
                f"{'bench2drive missing ' + ','.join(missing_bench_cams) if missing_bench_cams else ''}"
                f"{'; ' if (missing_bench_cams and missing_meta_cams) else ''}"
                f"{'metadrive missing ' + ','.join(missing_meta_cams) if missing_meta_cams else ''}"
            )
            continue

        bench_paths = {cam: str(bench_frames_by_cam[cam][frame_id].resolve()) for cam in camera_dirs}
        meta_paths = {cam: str(meta_frames_by_cam[cam][frame_id].resolve()) for cam in camera_dirs}

        # Build history using fixed spacing when possible to mirror training.
        selected_history_ids: List[str] = []
        selected_history_bench: List[Dict[str, str]] = []
        selected_history_meta: List[Dict[str, str]] = []

        try:
            frame_num = int(frame_id)
        except ValueError:
            frame_num = None

        if frame_num is not None and history_length > 0:
            for offset in range(history_length, 0, -1):
                target = frame_num - offset * sample_interval
                if target in history_lookup:
                    prev_id, prev_bench, prev_meta = history_lookup[target]
                    selected_history_ids.append(prev_id)
                    selected_history_bench.append(prev_bench)
                    selected_history_meta.append(prev_meta)
                elif not allow_short_history:
                    missing_notes.append(
                        f"{scenario} frame {frame_id}: missing history frame {target} with interval {sample_interval}"
                    )
                    # Skip this frame entirely if strict history is required.
                    break
            else:
                pass  # only runs if loop not broken
            if not allow_short_history and len(selected_history_ids) < history_length:
                continue
        else:
            # Fallback to most recent valid frames when frame numbers are non-numeric.
            selected_history_ids = history_frame_ids[-history_length:] if history_length else []
            selected_history_bench = history_bench_frames[-history_length:] if history_length else []
            selected_history_meta = history_meta_frames[-history_length:] if history_length else []

        stats.matched_frames += 1
        frames_payload.append(
            {
                "scenario": scenario,
                "frame": frame_id,
                "prev_frame": selected_history_ids[-1] if selected_history_ids else None,
                "prev_frames": selected_history_ids,
                "bench2drive": bench_paths,
                "metadrive": meta_paths,
                "prev_bench2drive": selected_history_bench,
                "prev_metadrive": selected_history_meta,
            }
        )

        history_frame_ids.append(frame_id)
        history_bench_frames.append(bench_paths)
        history_meta_frames.append(meta_paths)
        if frame_num is not None:
            history_lookup[frame_num] = (frame_id, bench_paths, meta_paths)

    return frames_payload, stats, missing_notes


def save_frames_online(
    bench_root: Path,
    metadrive_root: Path,
    output_path: Path,
    missing_log_path: Path,
    camera_dirs: Sequence[str],
    extensions: Sequence[str],
    history_length: int,
    sample_interval: int,
    allow_short_history: bool,
    scenario_list: Sequence[Tuple[str, str]] | None = None,
) -> FrameStats:
    """Stream per-frame bundles to disk while collecting coverage stats."""
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

    summary = FrameStats()
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

            frames, stats, missing_notes = collect_frames_for_scenario(
                scenario,
                bench_path,
                meta_path,
                camera_dirs,
                extensions,
                history_length,
                sample_interval,
                allow_short_history,
            )
            summary.add(stats)
            missing_lines.extend(missing_notes)

            for payload in frames:
                json.dump(payload, out_f)
                out_f.write("\n")

    with open(missing_log_path, "w", encoding="utf-8") as miss_f:
        miss_f.write("\n".join(missing_lines))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-frame Bench2Drive/MetaDrive bundles with all UniAD cameras present."
    )
    parser.add_argument(
        "--bench2drive-root",
        type=Path,
        default=Path("/closed-loop-e2e/Bench2Drive/Bench2Drive-Base"),
        help="Root directory containing Bench2Drive scenarios with camera subfolders.",
    )
    parser.add_argument(
        "--metadrive-root",
        type=Path,
        default=Path("/closed-loop-e2e/Bench2Drive/converted-base"),
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
        default=Path("bench2drive_metadrive_frame_pairs.jsonl"),
        help="Path to JSONL output where each line stores one frame's matched camera bundle.",
    )
    parser.add_argument(
        "--missing-log",
        type=Path,
        default=Path("bench2drive_metadrive_frame_missing.txt"),
        help="Path to the text file where missing frames/scenarios are recorded.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(UNIAD_CAMERA_DIRS),
        help="Camera subdirectories to require (relative to each scenario's camera folder).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Image extensions (with dot) to search for when pairing frames.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=4,
        help="Number of past frames to include (default matches UniAD queue_length-1).",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Frame spacing expected between history frames (e.g., UniAD sample_interval).",
    )
    parser.add_argument(
        "--allow-short-history",
        action="store_true",
        help="Allow outputting frames even if fewer than history-length past frames are available.",
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
    history_length: int = max(0, args.history_length)
    sample_interval: int = max(1, args.sample_interval)
    allow_short_history: bool = bool(args.allow_short_history)

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

    stats = save_frames_online(
        bench_root,
        metadrive_root,
        output_path,
        missing_log_path,
        camera_dirs,
        extensions,
        history_length,
        sample_interval,
        allow_short_history,
        scenario_list,
    )

    if stats.attempted_frames == 0:
        print("No frames were checked. Ensure the roots contain scenario folders with camera images.")
        return

    print(f"Attempted frames: {stats.attempted_frames}")
    print(f"Frames with all cameras present: {stats.matched_frames}")
    print(f"Frames missing Bench2Drive cameras: {stats.missing_frames_bench2drive}")
    print(f"Frames missing MetaDrive cameras: {stats.missing_frames_metadrive}")
    print(f"Scenarios missing in Bench2Drive root: {stats.missing_bench2drive_scenarios}")
    print(f"Scenarios missing in MetaDrive root: {stats.missing_metadrive_scenarios}")
    print(f"Per-frame entries written to {output_path}")
    print(f"Missing entries written to {missing_log_path}")


if __name__ == "__main__":
    main()
