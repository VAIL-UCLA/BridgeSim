#!/usr/bin/env python3
"""
Split a RAP image match JSONL into balanced sub-datasets of varying sizes.

The input JSONL should come from build_openscene_rap_image_dataset.py where each
line is shaped as {log_filename: [[frame_idx, cam_name], ...]}.

Sub-datasets are produced with evenly interleaved sampling across scenarios and
cameras to keep distributions balanced. Outputs are JSONL files matching the
input structure.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

DEFAULT_SIZES = ("10000", "50000", "100000", "500000", "1000000", "all")
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


def load_matches(jsonl_path: Path) -> Dict[str, Dict[str, List[object]]]:
    """Load scenario -> camera -> [frame_idx, ...] from JSONL."""
    scenarios: Dict[str, Dict[str, List[object]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not entry:
                continue
            scenario, pairs = next(iter(entry.items()))
            cam_map = scenarios.setdefault(scenario, {})
            for frame_idx, cam_name in pairs:
                cam_map.setdefault(cam_name, []).append(frame_idx)
    return scenarios


def shuffle_frames(scenarios: Dict[str, Dict[str, List[object]]], seed: int) -> None:
    """Shuffle frame lists in-place for randomness with reproducibility."""
    rng = random.Random(seed)
    for cam_map in scenarios.values():
        for frames in cam_map.values():
            rng.shuffle(frames)


def iter_balanced_pairs(
    scenarios: Dict[str, Dict[str, List[object]]], camera_order: Sequence[str]
) -> Iterable[Tuple[str, object, str]]:
    """
    Yield (scenario, frame_idx, cam_name) interleaving scenarios and cameras.

    We iterate scenarios in round-robin; for each scenario we emit one frame per
    camera (if available) per pass. This keeps distribution even across both
    axes until data is exhausted.
    """
    scenario_queue: Deque[str] = deque(sorted(scenarios))
    while scenario_queue:
        scenario = scenario_queue.popleft()
        cam_map = scenarios[scenario]
        emitted = False
        for cam in camera_order:
            frames = cam_map.get(cam)
            if frames:
                frame_idx = frames.pop()
                yield scenario, frame_idx, cam
                emitted = True
        if any(cam_map.get(cam) for cam in camera_order):
            scenario_queue.append(scenario)
        elif emitted and any(cam_map.values()):
            # Scenario has frames for cams outside camera_order; keep cycling.
            scenario_queue.append(scenario)


def parse_sizes(size_args: Sequence[str], total_pairs: int) -> List[Tuple[int, str]]:
    """Normalize size arguments into numeric counts and labels."""
    targets: List[Tuple[int, str]] = []
    for raw in size_args:
        if raw.lower() == "all":
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count > 0:
            targets.append((count, raw))

    targets.sort(key=lambda x: x[0])
    targets = [(c, label) for c, label in targets if c <= total_pairs]
    targets.append((total_pairs, "all"))
    return targets


def write_jsonl(scenario_map: Dict[str, List[Tuple[object, str]]], output_path: Path) -> None:
    """Write {scenario: [[frame_idx, cam], ...]} per line JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for scenario, pairs in scenario_map.items():
            json.dump({scenario: [[frame_idx, cam] for frame_idx, cam in pairs]}, f)
            f.write("\n")


def build_subsets(
    scenarios: Dict[str, Dict[str, List[object]]],
    camera_order: Sequence[str],
    targets: List[Tuple[int, str]],
    output_dir: Path,
    prefix: str,
) -> None:
    """Create subset JSONL files for each requested target size."""
    subset_buffers: Dict[str, Dict[str, List[Tuple[object, str]]]] = {
        label: {} for _, label in targets
    }
    targets_sorted = sorted(targets, key=lambda x: x[0])

    total_needed = targets_sorted[-1][0]
    for idx, (scenario, frame_idx, cam_name) in enumerate(
        iter_balanced_pairs(scenarios, camera_order), start=1
    ):
        if idx > total_needed:
            break
        for size, label in targets_sorted:
            if idx > size:
                continue
            bucket = subset_buffers[label].setdefault(scenario, [])
            bucket.append((frame_idx, cam_name))

    for size, label in targets_sorted:
        output_path = output_dir / f"{prefix}_{label}.jsonl"
        write_jsonl(subset_buffers[label], output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split RAP image match JSONL into balanced sub-datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rap_image_matches.jsonl"),
        help="JSONL file generated by build_openscene_rap_image_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diverse_subsets"),
        help="Directory to write subset JSONL files.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="Subset sizes to generate (use 'all' to include the full set).",
    )
    parser.add_argument(
        "--cams",
        nargs="+",
        default=list(DEFAULT_CAM_DIRS),
        help="Camera order to enforce; defaults to NavSim camera order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling frames within each camera.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix; defaults to input stem.",
    )
    args = parser.parse_args()

    scenarios = load_matches(args.input)
    total_pairs = sum(len(frames) for cam_map in scenarios.values() for frames in cam_map.values())
    if total_pairs == 0:
        print(f"No pairs found in {args.input}; nothing to split.")
        return

    camera_names = tuple(args.cams) if args.cams else tuple(
        sorted({cam for cam_map in scenarios.values() for cam in cam_map})
    )
    shuffle_frames(scenarios, args.seed)
    targets = parse_sizes(args.sizes, total_pairs)
    prefix = args.prefix or args.input.stem

    print(
        f"Loaded {len(scenarios)} scenarios, {total_pairs} pairs. "
        f"Generating subsets: {[label for _, label in targets]}"
    )
    build_subsets(scenarios, camera_names, targets, args.output_dir, prefix)
    print(f"Done. Subsets written under {args.output_dir}")


if __name__ == "__main__":
    main()
