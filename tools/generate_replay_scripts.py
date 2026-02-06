#!/usr/bin/env python3
"""
Generate bash scripts to run replay_navsim_log.py on all scenarios.

Usage:
    python tools/generate_replay_scripts.py \
        --input-dir /avl-west/navsim/trainval_md_logs \
        --output-root /path/to/output/images \
        --num-scripts 10 \
        --script-dir ./replay_scripts
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate bash scripts for batch replay of NavSim logs"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing scenario folders (e.g., /avl-west/navsim/trainval_md_logs)")
    parser.add_argument("--output-root", required=True,
                        help="Output directory for replay images")
    parser.add_argument("--num-scripts", "-n", type=int, required=True,
                        help="Number of bash scripts to split into")
    parser.add_argument("--script-dir", default="./replay_scripts",
                        help="Directory to save generated bash scripts (default: ./replay_scripts)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Optional: comma-separated camera IDs to pass to replay script")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional: max frames to pass to replay script")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Add verbose output (set -x) to generated scripts for debugging")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    script_dir = Path(args.script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    # Find all scenario folders
    print(f"Scanning {input_dir} for scenario folders...")
    scenario_folders = sorted([
        d.name for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("sd_")
    ])

    total_scenarios = len(scenario_folders)
    print(f"Found {total_scenarios} scenario folders")

    if total_scenarios == 0:
        print("No scenario folders found!")
        return

    # Calculate split
    n = args.num_scripts
    chunk_size = (total_scenarios + n - 1) // n  # Ceiling division

    # Build base command
    python_bin = "/opt/conda/envs/mdsn/bin/python"
    replay_script = "/root/BridgeSim/tools/replay_navsim_log.py"

    # Generate scripts
    for i in range(n):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_scenarios)

        if start_idx >= total_scenarios:
            break

        chunk = scenario_folders[start_idx:end_idx]
        script_name = script_dir / f"run_replay_part_{i+1}.sh"

        with open(script_name, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Replay script part {i+1}/{n}\n")
            f.write(f"# Scenarios {start_idx+1} to {end_idx} ({len(chunk)} scenarios)\n\n")
            f.write("set -e  # Exit on error\n")
            if args.verbose:
                f.write("set -x  # Print commands as they execute\n")
            f.write("\n")

            # Add progress tracking variables
            f.write(f"TOTAL={len(chunk)}\n")
            f.write("CURRENT=0\n")
            f.write(f"PART={i+1}\n\n")

            # Add progress bar function
            f.write('''# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local scenario=$3
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    printf "\\r[Part %d] [" "$PART"
    printf "%0.s#" $(seq 1 $filled 2>/dev/null) || true
    printf "%0.s-" $(seq 1 $empty 2>/dev/null) || true
    printf "] %3d%% (%d/%d) %s" "$percent" "$current" "$total" "$scenario"
}

''')

            for scenario_id in chunk:
                # Construct path: {input_dir}/{id}/{id}_0/{id}.pkl
                pkl_path = input_dir / scenario_id / f"{scenario_id}_0" / f"{scenario_id}.pkl"

                # Increment counter and show progress
                f.write(f"((CURRENT++)) || true\n")
                f.write(f'show_progress $CURRENT $TOTAL "{scenario_id}"\n')

                cmd = f"{python_bin} {replay_script} \\\n"
                cmd += f"    --scenario-path {pkl_path} \\\n"
                cmd += f"    --output-root {args.output_root}"

                if args.cameras:
                    cmd += f" \\\n    --cameras {args.cameras}"
                if args.max_frames:
                    cmd += f" \\\n    --max-frames {args.max_frames}"

                cmd += "\n\n"
                f.write(cmd)

            # Print newline at the end
            f.write('echo ""\n')
            f.write(f'echo "Part {i+1} completed!"\n')

        # Make executable
        os.chmod(script_name, 0o755)
        print(f"Generated: {script_name} ({len(chunk)} scenarios)")

    print(f"\nDone! Generated {min(n, (total_scenarios + chunk_size - 1) // chunk_size)} scripts in {script_dir}")
    print(f"Each script processes ~{chunk_size} scenarios")
    print(f"\nTo run all scripts in parallel:")
    print(f"  for script in {script_dir}/run_replay_part_*.sh; do bash $script & done")
    print(f"\nOr run individually:")
    print(f"  bash {script_dir}/run_replay_part_001.sh")


if __name__ == "__main__":
    main()
