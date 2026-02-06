#!/usr/bin/env python3
import pickle
import numpy as np
from pprint import pprint

# Load the pickle file
pkl_path = "/home/abhijit/Work/BridgeSim/converted_scenarios/AccidentTwoWays_Town12_Route1444_Weather0/AccidentTwoWays_Town12_Route1444_Weather0_0/sd_bench2drive_AccidentTwoWays-Town12-Route1444-Weather0.pkl"

print("Loading pickle file...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("\n" + "="*80)
print("TOP-LEVEL STRUCTURE")
print("="*80)
print(f"Type: {type(data)}")

if isinstance(data, dict):
    print(f"\nTop-level keys: {list(data.keys())}")

    for key in data.keys():
        print(f"\n{key}:")
        print(f"  Type: {type(data[key])}")
        if isinstance(data[key], dict):
            print(f"  Keys: {list(data[key].keys())}")
        elif isinstance(data[key], (list, np.ndarray)):
            print(f"  Length: {len(data[key])}")
            if len(data[key]) > 0:
                print(f"  First element type: {type(data[key][0])}")

print("\n" + "="*80)
print("SDC_TRACK STRUCTURE (Ego Vehicle Data)")
print("="*80)

if 'sdc_track' in data:
    sdc = data['sdc_track']
    print(f"Type: {type(sdc)}")

    if isinstance(sdc, dict):
        print(f"Keys: {list(sdc.keys())}")

        for key in sdc.keys():
            print(f"\n  {key}:")
            print(f"    Type: {type(sdc[key])}")

            if isinstance(sdc[key], (np.ndarray, list)):
                print(f"    Shape/Length: {np.array(sdc[key]).shape if isinstance(sdc[key], np.ndarray) else len(sdc[key])}")
                if len(sdc[key]) > 0:
                    print(f"    First few elements:")
                    if isinstance(sdc[key], np.ndarray):
                        print(f"      {sdc[key][:min(3, len(sdc[key]))]}")
                    else:
                        print(f"      {sdc[key][:min(3, len(sdc[key]))]}")
            elif isinstance(sdc[key], dict):
                print(f"    Dict keys: {list(sdc[key].keys())}")
            else:
                print(f"    Value: {sdc[key]}")

print("\n" + "="*80)
print("METADATA STRUCTURE")
print("="*80)

if 'metadata' in data:
    metadata = data['metadata']
    print(f"Type: {type(metadata)}")
    if isinstance(metadata, dict):
        print(f"Keys: {list(metadata.keys())}")
        for key, value in metadata.items():
            print(f"\n  {key}: {value}")

print("\n" + "="*80)
print("SEARCHING FOR ROUTE/WAYPOINT/PATH DATA")
print("="*80)

# Recursively search for route-related keys
def search_for_keywords(obj, keywords, path="root", max_depth=5, current_depth=0):
    if current_depth >= max_depth:
        return

    results = []

    if isinstance(obj, dict):
        for key in obj.keys():
            key_lower = str(key).lower()
            for keyword in keywords:
                if keyword in key_lower:
                    results.append({
                        'path': f"{path}.{key}",
                        'type': type(obj[key]).__name__,
                        'key': key
                    })
                    if isinstance(obj[key], (np.ndarray, list)):
                        results[-1]['shape'] = np.array(obj[key]).shape if isinstance(obj[key], np.ndarray) else len(obj[key])
                    break

            # Recurse
            sub_results = search_for_keywords(obj[key], keywords, f"{path}.{key}", max_depth, current_depth + 1)
            results.extend(sub_results)

    elif isinstance(obj, (list, tuple)) and len(obj) > 0 and current_depth < max_depth - 1:
        # Check first element
        sub_results = search_for_keywords(obj[0], keywords, f"{path}[0]", max_depth, current_depth + 1)
        results.extend(sub_results)

    return results

keywords = ['route', 'waypoint', 'path', 'navigation', 'command', 'gt', 'ground_truth',
            'trajectory', 'plan', 'target', 'goal', 'reference']

print("Searching for keywords:", keywords)
results = search_for_keywords(data, keywords)

if results:
    print(f"\nFound {len(results)} potential route-related fields:")
    for r in results:
        print(f"\n  Path: {r['path']}")
        print(f"  Type: {r['type']}")
        if 'shape' in r:
            print(f"  Shape: {r['shape']}")
else:
    print("\nNo direct route-related keywords found in top-level structure")

print("\n" + "="*80)
print("COMPLETE SDC_TRACK FIRST FRAME DATA")
print("="*80)

if 'sdc_track' in data:
    sdc = data['sdc_track']
    if isinstance(sdc, dict):
        for key in sdc.keys():
            if isinstance(sdc[key], (np.ndarray, list)):
                arr = np.array(sdc[key])
                if len(arr) > 0:
                    print(f"\n{key}:")
                    print(f"  Shape: {arr.shape}")
                    print(f"  Frame 0 data: {arr[0]}")
                    if len(arr) > 1:
                        print(f"  Frame 1 data: {arr[1]}")

print("\n" + "="*80)
print("ALL OTHER TOP-LEVEL DATA")
print("="*80)

for key in data.keys():
    if key not in ['sdc_track', 'metadata']:
        print(f"\n{key}:")
        print(f"  Type: {type(data[key])}")
        if isinstance(data[key], dict):
            print(f"  Keys: {list(data[key].keys())}")
            # Print first level of nested structure
            for subkey in list(data[key].keys())[:5]:  # Limit to first 5
                print(f"    {subkey}: {type(data[key][subkey])}")
                if isinstance(data[key][subkey], (np.ndarray, list)):
                    print(f"      Shape: {np.array(data[key][subkey]).shape}")
        elif isinstance(data[key], (np.ndarray, list)):
            arr = np.array(data[key])
            print(f"  Shape: {arr.shape}")
            if len(arr) > 0:
                print(f"  First element: {arr[0]}")
