#!/usr/bin/env python3
"""
Build a JSONL index of TransFuser BEV feature pairs from cached H5 files.

This script scans the BEV cache directory structure created by
precompute_transfuser_bev_cache.py and creates a JSONL file mapping
tokens to their H5 file locations across all three domains:
  - navsim (camera + real LiDAR)
  - navsim_no_lidar (camera + dummy LiDAR)
  - metadrive (camera + dummy LiDAR)

Only tokens present in ALL three domains are included in the output.
A summary of missing pairs is printed to help identify cache gaps.

Cache structure expected:
    cache_dir/
    ├── navsim/
    │   └── <log_name>.h5  # HDF5 with token datasets, shape (512, 8, 8)
    ├── navsim_no_lidar/
    │   └── <log_name>.h5
    └── metadrive/
        └── <log_name>.h5

Output JSONL format (one entry per line):
    {"token": "abc123", "log_name": "log_001", "navsim_h5": "/path/navsim/log_001.h5", ...}
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

try:
    import h5py
except ImportError:
    h5py = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DOMAINS = ("navsim", "navsim_no_lidar", "metadrive")


@dataclass
class ScanStats:
    """Statistics for the BEV cache scan."""

    total_tokens: int = 0
    complete_pairs: int = 0
    missing_navsim: int = 0
    missing_navsim_no_lidar: int = 0
    missing_metadrive: int = 0
    logs_per_domain: Dict[str, int] = field(default_factory=dict)
    tokens_per_domain: Dict[str, int] = field(default_factory=dict)

    @property
    def incomplete_pairs(self) -> int:
        return self.total_tokens - self.complete_pairs


def scan_domain(cache_dir: Path, domain: str) -> Dict[str, Set[str]]:
    """
    Scan all H5 files in a domain directory.

    Args:
        cache_dir: Root cache directory
        domain: Domain name (navsim, navsim_no_lidar, metadrive)

    Returns:
        Dictionary mapping log_name -> set of tokens
    """
    if h5py is None:
        raise ImportError("h5py is required. Install with: pip install h5py")

    result: Dict[str, Set[str]] = {}
    domain_dir = cache_dir / domain

    if not domain_dir.exists():
        return result

    h5_files = list(domain_dir.glob("*.h5"))
    iterator = tqdm(h5_files, desc=f"Scanning {domain}") if tqdm else h5_files

    for h5_file in iterator:
        log_name = h5_file.stem
        try:
            with h5py.File(h5_file, "r") as f:
                result[log_name] = set(f.keys())
        except Exception as e:
            print(f"Warning: Failed to read {h5_file}: {e}")
            result[log_name] = set()

    return result


def build_token_index(
    domain_caches: Dict[str, Dict[str, Set[str]]]
) -> tuple[Dict[str, str], Set[str]]:
    """
    Build a mapping from token to log_name and collect all unique tokens.

    Args:
        domain_caches: Dict mapping domain -> {log_name -> set(tokens)}

    Returns:
        Tuple of (token_to_log mapping, set of all tokens)
    """
    token_to_log: Dict[str, str] = {}
    all_tokens: Set[str] = set()

    for domain, log_tokens in domain_caches.items():
        for log_name, tokens in log_tokens.items():
            all_tokens.update(tokens)
            for token in tokens:
                # Use first encountered log_name for this token
                if token not in token_to_log:
                    token_to_log[token] = log_name

    return token_to_log, all_tokens


def find_complete_pairs(
    all_tokens: Set[str],
    token_to_log: Dict[str, str],
    domain_caches: Dict[str, Dict[str, Set[str]]],
) -> tuple[List[str], ScanStats]:
    """
    Find tokens that exist in all three domains.

    Args:
        all_tokens: Set of all unique tokens
        token_to_log: Mapping from token to log_name
        domain_caches: Dict mapping domain -> {log_name -> set(tokens)}

    Returns:
        Tuple of (list of complete tokens, scan statistics)
    """
    stats = ScanStats()
    stats.total_tokens = len(all_tokens)

    # Compute per-domain stats
    for domain in DOMAINS:
        domain_logs = domain_caches.get(domain, {})
        stats.logs_per_domain[domain] = len(domain_logs)
        stats.tokens_per_domain[domain] = sum(len(t) for t in domain_logs.values())

    complete_tokens: List[str] = []

    for token in all_tokens:
        log_name = token_to_log.get(token)
        if not log_name:
            continue

        in_navsim = token in domain_caches.get("navsim", {}).get(log_name, set())
        in_navsim_no_lidar = token in domain_caches.get("navsim_no_lidar", {}).get(
            log_name, set()
        )
        in_metadrive = token in domain_caches.get("metadrive", {}).get(log_name, set())

        if in_navsim and in_navsim_no_lidar and in_metadrive:
            complete_tokens.append(token)
        else:
            if not in_navsim:
                stats.missing_navsim += 1
            if not in_navsim_no_lidar:
                stats.missing_navsim_no_lidar += 1
            if not in_metadrive:
                stats.missing_metadrive += 1

    stats.complete_pairs = len(complete_tokens)
    return complete_tokens, stats


def analyze_missing_by_log(
    all_tokens: Set[str],
    token_to_log: Dict[str, str],
    domain_caches: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, int]]:
    """
    Analyze which logs have the most missing tokens per domain.

    Returns:
        Dict mapping domain -> {log_name -> count of missing tokens}
    """
    missing_by_log: Dict[str, Dict[str, int]] = {d: defaultdict(int) for d in DOMAINS}

    for token in all_tokens:
        log_name = token_to_log.get(token)
        if not log_name:
            continue

        for domain in DOMAINS:
            if token not in domain_caches.get(domain, {}).get(log_name, set()):
                missing_by_log[domain][log_name] += 1

    return missing_by_log


def write_jsonl(
    output_path: Path,
    complete_tokens: List[str],
    token_to_log: Dict[str, str],
    cache_dir: Path,
) -> None:
    """
    Write the JSONL output file.

    Args:
        output_path: Path to output JSONL file
        complete_tokens: List of tokens to write
        token_to_log: Mapping from token to log_name
        cache_dir: Root cache directory
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for token in sorted(complete_tokens):
            log_name = token_to_log[token]
            entry = {
                "token": token,
                "log_name": log_name,
                "navsim_h5": str(cache_dir / "navsim" / f"{log_name}.h5"),
                "navsim_no_lidar_h5": str(
                    cache_dir / "navsim_no_lidar" / f"{log_name}.h5"
                ),
                "metadrive_h5": str(cache_dir / "metadrive" / f"{log_name}.h5"),
            }
            f.write(json.dumps(entry) + "\n")


def print_summary(
    stats: ScanStats,
    missing_by_log: Dict[str, Dict[str, int]],
    top_n: int = 5,
) -> None:
    """Print a summary of the scan results."""
    print("\n" + "=" * 50)
    print("BEV Cache Scan Summary")
    print("=" * 50)

    print(f"\nTotal unique tokens found: {stats.total_tokens:,}")
    print(f"Complete pairs (all 3 domains): {stats.complete_pairs:,}")
    print(f"Incomplete pairs: {stats.incomplete_pairs:,}")

    print("\nPer-domain statistics:")
    for domain in DOMAINS:
        logs = stats.logs_per_domain.get(domain, 0)
        tokens = stats.tokens_per_domain.get(domain, 0)
        print(f"  {domain}: {logs} logs, {tokens:,} tokens")

    print("\nMissing breakdown (tokens not in domain):")
    print(f"  - Missing from navsim: {stats.missing_navsim:,}")
    print(f"  - Missing from navsim_no_lidar: {stats.missing_navsim_no_lidar:,}")
    print(f"  - Missing from metadrive: {stats.missing_metadrive:,}")

    # Show top logs with missing tokens for each domain
    for domain in DOMAINS:
        domain_missing = missing_by_log.get(domain, {})
        if not domain_missing:
            continue

        sorted_logs = sorted(domain_missing.items(), key=lambda x: x[1], reverse=True)
        if sorted_logs and sorted_logs[0][1] > 0:
            print(f"\nTop {top_n} logs with most missing {domain} tokens:")
            for log_name, count in sorted_logs[:top_n]:
                if count > 0:
                    print(f"  - {log_name}: {count:,} missing")


def write_missing_report(
    report_path: Path,
    stats: ScanStats,
    missing_by_log: Dict[str, Dict[str, int]],
) -> None:
    """Write detailed missing report to file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BEV Cache Missing Pairs Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total unique tokens: {stats.total_tokens}\n")
        f.write(f"Complete pairs: {stats.complete_pairs}\n")
        f.write(f"Incomplete pairs: {stats.incomplete_pairs}\n\n")

        f.write("Missing by domain:\n")
        f.write(f"  navsim: {stats.missing_navsim}\n")
        f.write(f"  navsim_no_lidar: {stats.missing_navsim_no_lidar}\n")
        f.write(f"  metadrive: {stats.missing_metadrive}\n\n")

        for domain in DOMAINS:
            domain_missing = missing_by_log.get(domain, {})
            sorted_logs = sorted(
                domain_missing.items(), key=lambda x: x[1], reverse=True
            )
            if sorted_logs:
                f.write(f"\nLogs with missing {domain} tokens:\n")
                for log_name, count in sorted_logs:
                    if count > 0:
                        f.write(f"  {log_name}: {count}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JSONL index of TransFuser BEV feature pairs from cached H5 files."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Root cache directory containing navsim/, navsim_no_lidar/, metadrive/ subdirs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("transfuser_bev_pairs.jsonl"),
        help="Output JSONL file path (default: transfuser_bev_pairs.jsonl).",
    )
    parser.add_argument(
        "--report-missing",
        type=Path,
        default=None,
        help="Optional: Write detailed missing report to this file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir: Path = args.cache_dir
    output_path: Path = args.output
    report_path: Path | None = args.report_missing

    if not cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    # Check that at least one domain directory exists
    existing_domains = [d for d in DOMAINS if (cache_dir / d).exists()]
    if not existing_domains:
        raise SystemExit(
            f"No domain directories found in {cache_dir}. "
            f"Expected: {', '.join(DOMAINS)}"
        )

    print(f"Scanning cache directory: {cache_dir}")
    print(f"Found domain directories: {', '.join(existing_domains)}")

    # Step 1: Scan all domains
    domain_caches: Dict[str, Dict[str, Set[str]]] = {}
    for domain in DOMAINS:
        domain_caches[domain] = scan_domain(cache_dir, domain)

    # Step 2: Build token index
    token_to_log, all_tokens = build_token_index(domain_caches)
    print(f"\nFound {len(all_tokens):,} unique tokens across all domains")

    # Step 3: Find complete pairs
    complete_tokens, stats = find_complete_pairs(
        all_tokens, token_to_log, domain_caches
    )

    # Step 4: Analyze missing by log
    missing_by_log = analyze_missing_by_log(all_tokens, token_to_log, domain_caches)

    # Step 5: Write output
    write_jsonl(output_path, complete_tokens, token_to_log, cache_dir)
    print(f"\nWrote {len(complete_tokens):,} pairs to {output_path}")

    # Step 6: Print summary
    print_summary(stats, missing_by_log)

    # Step 7: Optional detailed report
    if report_path:
        write_missing_report(report_path, stats, missing_by_log)
        print(f"\nDetailed missing report written to {report_path}")


if __name__ == "__main__":
    main()
