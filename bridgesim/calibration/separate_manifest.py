from __future__ import annotations

import argparse
import json
from collections import Counter
import heapq
from pathlib import Path
from typing import Iterable, List, Tuple

ManifestEntry = Tuple[str, List[Tuple[int, str]]]


def _parse_manifest_line(raw_line: str, line_no: int, manifest_path: Path) -> ManifestEntry | None:
    """
    Parse and validate a single JSONL line matching calibration/dataloader.py expectations.
    Returns (log_id, pairs) or None for blank lines.
    """
    line = raw_line.strip()
    if not line:
        return None

    payload = json.loads(line)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError(f"Line {line_no} in {manifest_path} should contain one log mapping, found {payload}")

    log_id, pairs = next(iter(payload.items()))
    if not isinstance(pairs, list):
        raise ValueError(f"Line {line_no} in {manifest_path} has non-list pairs for {log_id}")

    cleaned_pairs: List[Tuple[int, str]] = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pair on line {line_no} for {log_id}: {pair}")
        frame_idx, cam_name = pair
        if frame_idx is None:
            raise ValueError(f"Missing frame_idx for {log_id}:{cam_name} on line {line_no}")
        cleaned_pairs.append((int(frame_idx), str(cam_name)))

    return str(log_id), cleaned_pairs


def _balance_by_pairs(manifest: Iterable[ManifestEntry], num_splits: int, output_dir: Path, stem: str) -> List[Path]:
    totals = [0] * num_splits
    width = len(str(num_splits))
    outputs = [output_dir / f"{stem}_{i + 1:0{width}d}.jsonl" for i in range(num_splits)]
    handles = [path.open("w", encoding="utf-8") for path in outputs]
    try:
        entries = list(manifest)
        shard_entries: List[List[ManifestEntry]] = [[] for _ in range(num_splits)]
        heap: List[Tuple[int, int]] = [(0, i) for i in range(num_splits)]
        heapq.heapify(heap)

        for log_id, pairs in sorted(entries, key=lambda item: len(item[1]), reverse=True):
            total, shard_idx = heapq.heappop(heap)
            shard_entries[shard_idx].append((log_id, pairs))
            total += len(pairs)
            totals[shard_idx] = total
            heapq.heappush(heap, (total, shard_idx))

        for shard_idx, logs in enumerate(shard_entries):
            for log_id, pairs in logs:
                json.dump({log_id: pairs}, handles[shard_idx])
                handles[shard_idx].write("\n")
    finally:
        for handle in handles:
            handle.close()
    return outputs


def separate_manifest(manifest_path: Path, num_splits: int, output_dir: Path | None = None) -> List[Path]:
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")

    manifest_path = manifest_path.expanduser()
    output_dir = (output_dir or manifest_path.parent).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _entries() -> Iterable[ManifestEntry]:
        with open(manifest_path, "r", encoding="utf-8") as src:
            for line_no, raw in enumerate(src, start=1):
                parsed = _parse_manifest_line(raw, line_no, manifest_path)
                if parsed:
                    yield parsed

    return _balance_by_pairs(_entries(), num_splits, output_dir, manifest_path.stem)


def summarize_manifest(manifest_path: Path) -> None:
    manifest_path = manifest_path.expanduser()
    cameras: Counter[str] = Counter()
    log_counts: List[int] = []
    total_pairs = 0

    with open(manifest_path, "r", encoding="utf-8") as src:
        for line_no, raw in enumerate(src, start=1):
            parsed = _parse_manifest_line(raw, line_no, manifest_path)
            if not parsed:
                continue
            _, pairs = parsed
            pair_count = len(pairs)
            total_pairs += pair_count
            log_counts.append(pair_count)
            cameras.update(cam for _, cam in pairs)

    log_count = len(log_counts)
    mean_pairs = (total_pairs / log_count) if log_count else 0
    min_pairs = min(log_counts) if log_counts else 0
    max_pairs = max(log_counts) if log_counts else 0

    print(f"Manifest: {manifest_path}")
    print(f"Logs: {log_count}")
    print(f"Total pairs: {total_pairs}")
    print(f"Pairs per log: min={min_pairs}, max={max_pairs}, mean={mean_pairs:.2f}")
    print("Camera counts:")
    for cam, count in cameras.most_common():
        print(f"  {cam}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a RAP image manifest into balanced JSONL shards by log.")
    parser.add_argument("manifest", type=Path, help="Input JSONL manifest path.")
    parser.add_argument(
        "num_splits",
        type=int,
        nargs="?",
        help="Number of shards to create (required unless --dry-run is used).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the shards (defaults to the manifest's directory).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Summarize manifest stats without writing shards.")
    args = parser.parse_args()

    if args.dry_run:
        summarize_manifest(args.manifest)
        return

    if args.num_splits is None:
        parser.error("num_splits is required unless --dry-run is used.")

    outputs = separate_manifest(args.manifest, args.num_splits, args.output_dir)
    print("Wrote:", *(str(p) for p in outputs), sep="\n")


if __name__ == "__main__":
    main()
