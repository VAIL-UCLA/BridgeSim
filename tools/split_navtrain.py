#!/usr/bin/env python3
"""
Script to split navtrain.yaml into n smaller yaml files by distributing log_names.
"""

import argparse
import yaml
from pathlib import Path


def split_yaml(input_file: str, n: int, output_dir: str = None, include_tokens: bool = True):
    """
    Split a navtrain yaml file into n smaller files.

    Args:
        input_file: Path to the input yaml file
        n: Number of output files to create
        output_dir: Directory for output files (defaults to same directory as input)
        include_tokens: Whether to include tokens in output files
    """
    input_path = Path(input_file)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read the yaml file
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    # Extract components
    log_names = data.get('log_names', [])
    tokens = data.get('tokens', [])

    # Get header fields (everything except log_names and tokens)
    header_fields = {k: v for k, v in data.items() if k not in ['log_names', 'tokens']}

    print(f"Found {len(log_names)} log_names and {len(tokens)} tokens")
    print(f"Splitting into {n} files...")

    # Calculate split sizes for log_names
    total_logs = len(log_names)
    base_size = total_logs // n
    remainder = total_logs % n

    # Split log_names
    log_splits = []
    start = 0
    for i in range(n):
        # Distribute remainder among first 'remainder' splits
        size = base_size + (1 if i < remainder else 0)
        log_splits.append(log_names[start:start + size])
        start += size

    # Write output files
    base_name = input_path.stem
    for i in range(n):
        output_data = header_fields.copy()
        output_data['log_names'] = log_splits[i]
        if include_tokens and tokens:
            output_data['tokens'] = tokens  # Include all tokens in each file

        output_file = output_dir / f"{base_name}_part{i+1}.yaml"

        with open(output_file, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"Created {output_file}: {len(log_splits[i])} log_names, {len(tokens)} tokens")

    print(f"\nDone! Created {n} files in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Split navtrain.yaml into n smaller files')
    parser.add_argument('input_file', help='Path to the input yaml file')
    parser.add_argument('n', type=int, help='Number of output files to create')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input)')
    parser.add_argument('--no-tokens', action='store_true', help='Exclude tokens from output files')

    args = parser.parse_args()

    split_yaml(
        input_file=args.input_file,
        n=args.n,
        output_dir=args.output_dir,
        include_tokens=not args.no_tokens
    )


if __name__ == '__main__':
    main()
