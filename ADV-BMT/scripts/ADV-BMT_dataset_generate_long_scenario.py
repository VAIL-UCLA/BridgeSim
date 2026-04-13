import argparse
import os
from tqdm import tqdm
import copy
import pickle
import numpy as np
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import gc
import torch

def create_dataset_files(scenario_file_path, output_dir, scenario_data):
    """
    Create dataset_mapping.pkl and dataset_summary.pkl files
    to match the convert_openscene output format.
    """
    from pathlib import Path

    scenario_filename = os.path.basename(scenario_file_path)

    # dataset_mapping: {filename: parent_dir_name}
    dataset_mapping = {scenario_filename: Path(scenario_file_path).parent.name}

    # dataset_summary: metadata about the scenario
    metadata = scenario_data.get("metadata", {})
    dataset_summary = {
        scenario_filename: {
            "id": metadata.get("scenario_id", "unknown"),
            "scenario_id": metadata.get("scenario_id", "unknown"),
            "sample_rate": metadata.get("sample_rate", 0.1),
            "ts": metadata.get("timestep", []),
            "length": scenario_data.get("length", metadata.get("track_length", 0)),
            "coordinate": metadata.get("coordinate", "right-handed"),
            "sdc_id": metadata.get("sdc_id", "0"),
            "dataset": metadata.get("dataset", "scgen"),
            "map_features": list(scenario_data.get("map_features", {}).keys()),
            "number_summary": metadata.get("number_summary", {}),
            "object_summary": metadata.get("object_summary", {}),
        }
    }

    with open(os.path.join(output_dir, "dataset_mapping.pkl"), 'wb') as f:
        pickle.dump(dataset_mapping, f)

    with open(os.path.join(output_dir, "dataset_summary.pkl"), 'wb') as f:
        pickle.dump(dataset_summary, f)


def get_filenames(folder_path, prefix="sd_"):
    """
    Find scenario pickle files. Supports both flat and nested directory structures:

    Flat structure:
        folder_path/sd_xxx.pkl

    Nested structure (e.g., navtest_converted):
        folder_path/sd_xxx/sd_xxx_0/sd_xxx.pkl
    """
    all_files = []

    # First, try flat structure
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.startswith(prefix) and file.endswith('.pkl'):
            all_files.append(file_path)

    # If no files found, try nested structure
    if not all_files:
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path) and subdir.startswith(prefix):
                # Look for sd_xxx_0 subdirectory
                for inner_subdir in os.listdir(subdir_path):
                    inner_path = os.path.join(subdir_path, inner_subdir)
                    if os.path.isdir(inner_path):
                        # Look for pkl files inside
                        for file in os.listdir(inner_path):
                            if file.startswith(prefix) and file.endswith('.pkl'):
                                all_files.append(os.path.join(inner_path, file))

    return all_files


def get_track_length(SD):
    """Get the length of tracks in the scenario (based on SDC track)."""
    sdc_id = SD['metadata']['sdc_id']
    return SD['tracks'][sdc_id]['state']['position'].shape[0]


def fix_scenario_dtypes(SD, max_traffic_lights=64):
    """
    Fix data types in scenario data for compatibility with SCGEN.
    Converts 'valid' arrays from float to bool, ensures float32 dtypes,
    and adds missing metadata fields to match Waymo format.
    """
    sdc_id = SD['metadata'].get('sdc_id', 'ego')

    # Rename 'ego' to numeric ID if needed (preprocessor expects int-convertible IDs)
    if sdc_id == 'ego':
        new_sdc_id = '0'  # Use '0' as the ego vehicle ID
        # Rename in tracks - preserve order by rebuilding dict with SDC first
        if 'ego' in SD['tracks']:
            old_tracks = SD['tracks']
            ego_track = old_tracks.pop('ego')
            # Rebuild tracks dict with SDC (renamed to '0') first
            SD['tracks'] = {new_sdc_id: ego_track}
            SD['tracks'].update(old_tracks)
        # Update metadata
        SD['metadata']['sdc_id'] = new_sdc_id
        sdc_id = new_sdc_id

    for idx, (track_name, track_data) in enumerate(SD['tracks'].items()):
        state = track_data['state']

        # Convert 'valid' to boolean
        if 'valid' in state and isinstance(state['valid'], np.ndarray):
            state['valid'] = state['valid'].astype(bool)

        # Convert other arrays to float32 (matching Waymo format)
        for key in ['position', 'velocity', 'heading', 'length', 'width', 'height']:
            if key in state and isinstance(state[key], np.ndarray):
                if state[key].dtype == np.float64:
                    state[key] = state[key].astype(np.float32)

        # Ensure position has 3 columns (x, y, z) - some formats might have only 2
        if 'position' in state and isinstance(state['position'], np.ndarray):
            if state['position'].ndim == 2 and state['position'].shape[1] == 2:
                # Add z=0 column
                z_col = np.zeros((state['position'].shape[0], 1), dtype=np.float32)
                state['position'] = np.concatenate([state['position'], z_col], axis=1)

    # Limit traffic lights to max_traffic_lights to avoid preprocessor index errors
    if 'dynamic_map_states' in SD and len(SD['dynamic_map_states']) > max_traffic_lights:
        # Keep only the first max_traffic_lights
        tl_keys = list(SD['dynamic_map_states'].keys())[:max_traffic_lights]
        SD['dynamic_map_states'] = {k: SD['dynamic_map_states'][k] for k in tl_keys}

    # Ensure required metadata fields exist (matching Waymo format)
    if 'metadata' in SD:
        # objects_of_interest should include at least the SDC
        if 'objects_of_interest' not in SD['metadata'] or SD['metadata']['objects_of_interest'] is None:
            SD['metadata']['objects_of_interest'] = [sdc_id]
        elif sdc_id not in SD['metadata']['objects_of_interest']:
            SD['metadata']['objects_of_interest'].append(sdc_id)

        # tracks_to_predict should include at least the SDC
        if 'tracks_to_predict' not in SD['metadata'] or not SD['metadata']['tracks_to_predict']:
            SD['metadata']['tracks_to_predict'] = {}

        # Find SDC track index
        track_names = list(SD['tracks'].keys())
        sdc_track_index = track_names.index(sdc_id) if sdc_id in track_names else 0

        if sdc_id not in SD['metadata']['tracks_to_predict']:
            SD['metadata']['tracks_to_predict'][sdc_id] = {
                'track_index': sdc_track_index,
                'track_id': sdc_id,
                'difficulty': 0,
                'object_type': 'VEHICLE'
            }

    return SD


def pad_scenario_to_length(SD, target_length=91):
    """
    Pad scenario data to a target length by repeating the last valid state.

    Args:
        SD: Scenario data dict
        target_length: Target number of frames (default 91)

    Returns:
        Padded scenario data dict
    """
    padded_SD = copy.deepcopy(SD)
    current_length = get_track_length(padded_SD)

    if current_length >= target_length:
        return padded_SD

    pad_length = target_length - current_length

    for track_name, track_data in padded_SD['tracks'].items():
        state = track_data['state']
        for key, value in state.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 1 and value.shape[0] == current_length:
                if key == 'valid':
                    # Pad valid with False
                    padding = np.zeros((pad_length,), dtype=value.dtype)
                else:
                    # Pad other arrays with zeros
                    padding_shape = list(value.shape)
                    padding_shape[0] = pad_length
                    padding = np.zeros(padding_shape, dtype=value.dtype)
                state[key] = np.concatenate([value, padding], axis=0)

        # Update metadata track_length if present
        if 'metadata' in track_data and 'track_length' in track_data['metadata']:
            track_data['metadata']['track_length'] = target_length

    # Update scenario length in metadata
    if 'length' in padded_SD:
        padded_SD['length'] = target_length

    return padded_SD


def truncate_scenario_to_window(SD, start_frame, window_size=91):
    """
    Truncate scenario data to a specific window of frames.

    Args:
        SD: Original scenario data dict
        start_frame: Starting frame index
        window_size: Number of frames to extract (default 91)

    Returns:
        Truncated scenario data dict
    """
    truncated_SD = copy.deepcopy(SD)
    end_frame = start_frame + window_size

    # Truncate all track states
    for track_name, track_data in truncated_SD['tracks'].items():
        state = track_data['state']
        for key, value in state.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 1:
                # Check if first dimension matches track length
                if value.shape[0] >= end_frame:
                    state[key] = value[start_frame:end_frame]
                elif value.shape[0] > start_frame:
                    # Pad if needed
                    state[key] = value[start_frame:]
                    padding_shape = list(value.shape)
                    padding_shape[0] = window_size - state[key].shape[0]
                    padding = np.zeros(padding_shape, dtype=value.dtype)
                    state[key] = np.concatenate([state[key], padding], axis=0)

        # Update metadata track_length if present
        if 'metadata' in track_data and 'track_length' in track_data['metadata']:
            track_data['metadata']['track_length'] = window_size

    # Update scenario length
    if 'length' in truncated_SD:
        truncated_SD['length'] = window_size

    return truncated_SD


def extend_generated_scenario(original_SD, generated_SD, start_frame, window_size=91):
    """
    Extend the generated scenario back to full length by combining with original frames.

    The generated adversary agent will be placed starting at start_frame.
    Other agents keep their original trajectories, with the window portion replaced by generated data.

    Args:
        original_SD: Original full-length scenario data
        generated_SD: Generated scenario with 91 frames
        start_frame: Frame index where the window starts in original
        window_size: Size of the generated window (default 91)

    Returns:
        Full-length scenario with generated adversary
    """
    full_SD = copy.deepcopy(original_SD)
    original_length = get_track_length(original_SD)
    end_frame = start_frame + window_size

    # Get the new adversary agent ID
    adv_id = generated_SD['metadata'].get('new_adv_id', '99999')

    # Add the adversary agent to the full scenario
    if adv_id in generated_SD['tracks']:
        adv_track = generated_SD['tracks'][adv_id]

        # Create full-length arrays for adversary, padded with zeros/False
        full_adv_track = {
            'state': {},
            'type': adv_track['type'],
            'metadata': copy.deepcopy(adv_track.get('metadata', {}))
        }

        for key, value in adv_track['state'].items():
            if isinstance(value, np.ndarray):
                # Create full-length array
                full_shape = list(value.shape)
                full_shape[0] = original_length

                if key == 'valid':
                    full_array = np.zeros(full_shape, dtype=bool)
                else:
                    full_array = np.zeros(full_shape, dtype=value.dtype)

                # Place the generated window data at the correct position
                actual_end = min(end_frame, original_length)
                actual_window = actual_end - start_frame
                full_array[start_frame:actual_end] = value[:actual_window]

                full_adv_track['state'][key] = full_array

        # Update metadata
        full_adv_track['metadata']['track_length'] = original_length

        # Add adversary to tracks
        full_SD['tracks'][adv_id] = full_adv_track

        # Update scenario metadata
        full_SD['metadata']['new_adv_id'] = adv_id
        if 'objects_of_interest' in full_SD['metadata']:
            if adv_id not in full_SD['metadata']['objects_of_interest']:
                full_SD['metadata']['objects_of_interest'].append(adv_id)
        else:
            full_SD['metadata']['objects_of_interest'] = [adv_id]

        # Update tracks_to_predict
        if 'tracks_to_predict' not in full_SD['metadata']:
            full_SD['metadata']['tracks_to_predict'] = {}

        tracks_length = len(list(full_SD['tracks'].keys()))
        full_SD['metadata']['tracks_to_predict'][adv_id] = {
            'difficulty': 0,
            'object_type': 'VEHICLE',
            'track_id': adv_id,
            'track_index': tracks_length - 1
        }

    return full_SD


def process_single_scenario(scenario_idx, SD_path, args, generator=None):
    """
    Process a single scenario. Can be used for both sequential and parallel processing.

    Args:
        scenario_idx: Index of the scenario being processed
        SD_path: Path to the scenario pickle file
        args: Argument namespace with configuration
        generator: SCGEN_Generator instance (for sequential mode)

    Returns:
        dict with processing results
    """
    # Set seed based on scenario index for reproducibility
    if args.seed is not None:
        scenario_seed = args.seed + scenario_idx
        random.seed(scenario_seed)
        np.random.seed(scenario_seed)

    # For parallel mode, create generator in each process
    if generator is None:
        from bmt.rl_train.train.scgen_generator import SCGEN_Generator
        generator = SCGEN_Generator()

    try:
        with open(SD_path, "rb") as f:
            SD = pickle.load(f)

        # Fix data types for SCGEN compatibility
        SD = fix_scenario_dtypes(SD)

        sid = SD["id"]
        track_length = get_track_length(SD)

        results = {
            'scenario_idx': scenario_idx,
            'sid': sid,
            'track_length': track_length,
            'status': 'success',
            'modes_generated': 0,
            'output_paths': []
        }

        if track_length < args.window_size:
            # Scenario is shorter than required window, skip it
            print(f"  Skipping: scenario has {track_length} frames, need {args.window_size}")
            return {
                'scenario_idx': scenario_idx,
                'sid': sid,
                'track_length': track_length,
                'status': 'skipped',
                'reason': f'too short ({track_length} < {args.window_size})',
                'modes_generated': 0,
                'output_paths': []
            }
        elif track_length == args.window_size:
            # Exact fit
            start_frame = 0
            truncated_SD = SD
        else:
            # Scenario is longer, randomly select window
            max_start = track_length - args.window_size
            start_frame = random.randint(0, max_start)
            truncated_SD = truncate_scenario_to_window(SD, start_frame, args.window_size)

        results['start_frame'] = start_frame

        for mode_idx in range(args.num_mode):
            SCGEN_SD = generator.generate_from_raw_SD(
                scenario_data=copy.deepcopy(truncated_SD),
                track_length=args.window_size
            )

            if SCGEN_SD is None:
                results['status'] = 'partial' if mode_idx > 0 else 'failed'
                break

            if track_length > args.window_size:
                # Original was longer, extend back to full length
                SCGEN_SD = extend_generated_scenario(
                    original_SD=SD,
                    generated_SD=SCGEN_SD,
                    start_frame=start_frame,
                    window_size=args.window_size
                )

            SCGEN_SD['metadata']['generation_info'] = {
                'original_length': track_length,
                'window_start_frame': start_frame,
                'window_size': args.window_size
            }

            # Create nested directory structure matching convert_openscene format:
            # save_dir/sd_<token>/sd_<token>_0/sd_<token>.pkl
            scenario_dir = os.path.join(args.save_dir, f"sd_{sid}")
            scenario_subdir = os.path.join(scenario_dir, f"sd_{sid}_{mode_idx}")
            os.makedirs(scenario_subdir, exist_ok=True)

            output_filename = f"sd_{sid}.pkl"
            output_path = os.path.join(scenario_subdir, output_filename)

            with open(output_path, "wb") as f:
                pickle.dump(SCGEN_SD, f)

            # Generate dataset_mapping.pkl and dataset_summary.pkl
            create_dataset_files(output_path, scenario_dir, SCGEN_SD)

            results['modes_generated'] += 1
            results['output_paths'].append(output_path)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error processing {SD_path}:")
        print(error_traceback)
        return {
            'scenario_idx': scenario_idx,
            'sid': SD_path,
            'status': 'error',
            'error': str(e),
            'traceback': error_traceback
        }


def process_scenario_wrapper(args_tuple):
    """Wrapper for multiprocessing Pool.map"""
    scenario_idx, SD_path, args = args_tuple
    return process_single_scenario(scenario_idx, SD_path, args, generator=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="The data folder storing raw scenario files.")
    parser.add_argument("--num_scenario", type=int, default=None,
                        help="Number of scenarios to process. Default: all scenarios in dir")
    parser.add_argument("--save_dir", required=True, help="The place to store output .pkl files")
    parser.add_argument("--TF_mode", type=str, default="no_TF")
    parser.add_argument("--num_mode", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=91, help="Window size for generation (default 91)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Batch processing options
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting scenario index (for batch processing)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending scenario index (exclusive). Default: start_idx + num_scenario")
    parser.add_argument("--batch_id", type=int, default=None,
                        help="Batch ID for splitting work across multiple jobs")
    parser.add_argument("--num_batches", type=int, default=None,
                        help="Total number of batches (used with --batch_id)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing within this job")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers. Default: number of CPU cores")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_scenario_files = sorted(get_filenames(args.dir))
    total_scenarios = len(all_scenario_files)

    print(f"Found {total_scenarios} scenario files in {args.dir}")

    # Determine which scenarios to process
    if args.batch_id is not None and args.num_batches is not None:
        # Split scenarios across batches
        scenarios_per_batch = (total_scenarios + args.num_batches - 1) // args.num_batches
        start_idx = args.batch_id * scenarios_per_batch
        end_idx = min(start_idx + scenarios_per_batch, total_scenarios)
        print(f"Batch {args.batch_id}/{args.num_batches}: processing scenarios {start_idx} to {end_idx-1}")
    else:
        start_idx = args.start_idx
        if args.end_idx is not None:
            end_idx = min(args.end_idx, total_scenarios)
        elif args.num_scenario is not None:
            end_idx = min(start_idx + args.num_scenario, total_scenarios)
        else:
            end_idx = total_scenarios

    scenario_indices = list(range(start_idx, end_idx))
    num_to_process = len(scenario_indices)

    print(f"Processing {num_to_process} scenarios (indices {start_idx} to {end_idx-1})")
    print(f"Window size: {args.window_size}, Modes per scenario: {args.num_mode}")

    if args.parallel:
        # Parallel processing mode
        num_workers = args.num_workers or cpu_count()
        print(f"Using parallel processing with {num_workers} workers")

        # Prepare arguments for each scenario
        process_args = [
            (idx, all_scenario_files[idx], args)
            for idx in scenario_indices
        ]

        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_scenario_wrapper, process_args),
                total=num_to_process,
                desc="Processing scenarios"
            ))
    else:
        # Sequential processing mode (shares generator instance)
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)

        from bmt.rl_train.train.scgen_generator import SCGEN_Generator
        generator = SCGEN_Generator()

        results = []
        for idx in tqdm(scenario_indices, desc="Processing scenarios"):
            SD_path = all_scenario_files[idx]
            result = process_single_scenario(idx, SD_path, args, generator)
            results.append(result)

            # Print progress
            status = result['status']
            sid = result.get('sid', 'unknown')
            if status == 'success':
                print(f"  [{idx}] {sid}: {result['track_length']} frames, "
                      f"window start={result['start_frame']}, "
                      f"{result['modes_generated']} modes generated")
            elif status == 'error':
                print(f"  [{idx}] {sid}: ERROR - {result.get('error', 'unknown')}")
            else:
                print(f"  [{idx}] {sid}: {status}")

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    partial = sum(1 for r in results if r['status'] == 'partial')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Partial:    {partial}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    print(f"  Errors:     {errors}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
