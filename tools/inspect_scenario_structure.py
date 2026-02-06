#!/usr/bin/env python3
"""
Inspect MetaDrive scenario pickle file structure to find route/waypoint data.
"""
import pickle5 as pickle
import numpy as np
from pathlib import Path
from pprint import pprint

# Direct pickle inspection
pkl_path = Path("/home/abhijit/Work/BridgeSim/converted_scenarios/HardBreakRoute_Town01_Route30_Weather3/HardBreakRoute_Town01_Route30_Weather3_0/sd_bench2drive_HardBreakRoute-Town01-Route30-Weather3.pkl")

print("="*80)
print("LOADING PICKLE FILE DIRECTLY")
print("="*80)
print(f"File: {pkl_path}\n")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")

if isinstance(data, dict):
    print(f"\nTop-level keys: {list(data.keys())}")

    print("\n" + "="*80)
    print("TOP-LEVEL STRUCTURE")
    print("="*80)

    for key in data.keys():
        val = data[key]
        print(f"\n{key}:")
        print(f"  Type: {type(val)}")

        if isinstance(val, dict):
            print(f"  Dict keys: {list(val.keys())}")

            # For tracks, show structure
            if key == 'tracks':
                print(f"  Number of tracks: {len(val)}")
                # Show first track structure
                if len(val) > 0:
                    first_track_id = list(val.keys())[0]
                    first_track = val[first_track_id]
                    print(f"\n  First track ID: {first_track_id}")
                    print(f"  First track type: {type(first_track)}")
                    if isinstance(first_track, dict):
                        print(f"  First track keys: {list(first_track.keys())}")
                        for track_key in first_track.keys():
                            track_val = first_track[track_key]
                            print(f"    {track_key}: {type(track_val)}", end="")
                            if isinstance(track_val, (np.ndarray, list)):
                                arr = np.array(track_val)
                                print(f" - shape: {arr.shape}")
                                if track_key in ['state', 'position', 'heading', 'velocity']:
                                    print(f"      First 3 frames: {arr[:3]}")
                            else:
                                print(f" - value: {track_val}")

        elif isinstance(val, (np.ndarray, list)):
            arr = np.array(val)
            print(f"  Shape: {arr.shape}")
            if len(arr) > 0:
                print(f"  First element: {arr[0]}")
        else:
            print(f"  Value: {val}")

    print("\n" + "="*80)
    print("SDC_TRACK (EGO VEHICLE) DETAILED STRUCTURE")
    print("="*80)

    if 'sdc_track' in data:
        sdc = data['sdc_track']
        print(f"\nType: {type(sdc)}")

        if isinstance(sdc, dict):
            print(f"Keys: {list(sdc.keys())}")

            for key in sdc.keys():
                val = sdc[key]
                print(f"\n  {key}:")
                print(f"    Type: {type(val)}")

                if isinstance(val, (np.ndarray, list)):
                    arr = np.array(val)
                    print(f"    Shape: {arr.shape}")
                    print(f"    First 3 frames:")
                    for i in range(min(3, len(arr))):
                        print(f"      Frame {i}: {arr[i]}")
                elif isinstance(val, dict):
                    print(f"    Dict keys: {list(val.keys())}")
                    for subkey in val.keys():
                        subval = val[subkey]
                        print(f"      {subkey}: {type(subval)}", end="")
                        if isinstance(subval, (np.ndarray, list)):
                            print(f" - shape: {np.array(subval).shape}")
                        else:
                            print(f" - value: {subval}")
                else:
                    print(f"    Value: {val}")

    print("\n" + "="*80)
    print("SEARCHING FOR ROUTE/WAYPOINT/PATH DATA")
    print("="*80)

    def recursive_search(obj, path="root", keywords=None, max_depth=6, current_depth=0):
        """Recursively search for keywords in dict keys."""
        if keywords is None:
            keywords = ['route', 'waypoint', 'path', 'command', 'gt', 'ground',
                       'trajectory', 'plan', 'target', 'goal', 'reference', 'lane']

        if current_depth >= max_depth:
            return []

        results = []

        if isinstance(obj, dict):
            for key in obj.keys():
                key_lower = str(key).lower()

                # Check if any keyword matches
                matched = False
                for keyword in keywords:
                    if keyword in key_lower:
                        results.append({
                            'path': f"{path}.{key}",
                            'key': key,
                            'type': type(obj[key]).__name__
                        })

                        # Add shape/value info
                        if isinstance(obj[key], (np.ndarray, list)):
                            results[-1]['shape'] = np.array(obj[key]).shape
                            if len(obj[key]) > 0:
                                results[-1]['sample'] = obj[key][0] if isinstance(obj[key][0], (int, float, str, bool)) else str(obj[key][0])[:100]
                        elif isinstance(obj[key], dict):
                            results[-1]['dict_keys'] = list(obj[key].keys())
                        else:
                            results[-1]['value'] = obj[key]

                        matched = True
                        break

                # Recurse into nested structures
                if current_depth < max_depth - 1:
                    sub_results = recursive_search(obj[key], f"{path}.{key}", keywords, max_depth, current_depth + 1)
                    results.extend(sub_results)

        elif isinstance(obj, (list, tuple)) and len(obj) > 0 and current_depth < max_depth - 1:
            # Only check first element of lists
            if isinstance(obj[0], dict):
                sub_results = recursive_search(obj[0], f"{path}[0]", keywords, max_depth, current_depth + 1)
                results.extend(sub_results)

        return results

    search_results = recursive_search(data)

    if search_results:
        print(f"\nFound {len(search_results)} potential matches:")
        for result in search_results:
            print(f"\n  Path: {result['path']}")
            print(f"  Key: {result['key']}")
            print(f"  Type: {result['type']}")
            if 'shape' in result:
                print(f"  Shape: {result['shape']}")
            if 'sample' in result:
                print(f"  Sample: {result['sample']}")
            if 'dict_keys' in result:
                print(f"  Dict keys: {result['dict_keys']}")
            if 'value' in result:
                print(f"  Value: {result['value']}")
    else:
        print("\nNo route/waypoint related data found with keyword search")

    print("\n" + "="*80)
    print("METADATA")
    print("="*80)

    if 'metadata' in data:
        metadata = data['metadata']
        print(f"\nType: {type(metadata)}")
        if isinstance(metadata, dict):
            for key, val in metadata.items():
                print(f"  {key}: {val}")

    print("\n" + "="*80)
    print("MAP FEATURES")
    print("="*80)

    if 'map_features' in data:
        map_features = data['map_features']
        print(f"\nType: {type(map_features)}")
        if isinstance(map_features, dict):
            print(f"Keys: {list(map_features.keys())}")
            for key in map_features.keys():
                val = map_features[key]
                print(f"\n  {key}:")
                print(f"    Type: {type(val)}")
                if isinstance(val, (np.ndarray, list)):
                    arr = np.array(val)
                    print(f"    Shape: {arr.shape}")
                    if len(arr) > 0:
                        print(f"    First element shape: {arr[0].shape if isinstance(arr[0], np.ndarray) else type(arr[0])}")
                        if len(arr[0]) > 0:
                            print(f"    First polyline sample: {arr[0][:2]}")
                elif isinstance(val, dict):
                    print(f"    Dict keys: {list(val.keys())}")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
