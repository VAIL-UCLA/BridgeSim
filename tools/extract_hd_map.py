#!/usr/bin/env python3
"""
Extract HD map data from .npz file and save as readable formats for analysis
"""

import numpy as np
import json
import pickle
import argparse
from pathlib import Path


def extract_hd_map_structure(npz_path: str, output_dir: str = "hd_map_analysis"):
    """Extract HD map structure and save in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading HD map from: {npz_path}")
    
    # Load the npz file
    with np.load(npz_path, allow_pickle=True) as data:
        hd_map = data['arr']
        print(f"HD map shape: {hd_map.shape}")
        print(f"HD map type: {type(hd_map)}")
    
    # Extract structure information
    structure_info = {
        "total_roads": len(hd_map),
        "road_summary": [],
        "sample_data": {}
    }
    
    print(f"Processing {len(hd_map)} roads...")
    
    # Process first few roads for detailed analysis
    for i in range(min(5, len(hd_map))):
        road_id, lane_data = hd_map[i]
        print(f"Processing road {road_id}...")
        
        road_info = {
            "road_id": road_id,
            "lane_data_type": str(type(lane_data)),
        }
        
        if isinstance(lane_data, dict):
            lane_ids = list(lane_data.keys())
            road_info["num_lanes"] = len(lane_ids)
            road_info["lane_ids"] = lane_ids[:10]  # First 10 lane IDs
            
            # Analyze first lane in detail
            if lane_ids:
                first_lane_id = lane_ids[0]
                first_lane = lane_data[first_lane_id]
                road_info["first_lane"] = {
                    "lane_id": first_lane_id,
                    "data_type": str(type(first_lane)),
                }
                
                if isinstance(first_lane, list) and len(first_lane) > 0:
                    road_info["first_lane"]["num_elements"] = len(first_lane)
                    road_info["first_lane"]["element_keys"] = list(first_lane[0].keys())
                    
                    # Save sample lane element for detailed inspection
                    if i == 0:  # Save detailed sample from first road
                        structure_info["sample_data"]["road_id"] = road_id
                        structure_info["sample_data"]["lane_id"] = first_lane_id
                        structure_info["sample_data"]["first_element"] = first_lane[0]
        
        structure_info["road_summary"].append(road_info)
    
    # Quick summary of all roads
    all_road_summary = []
    for i, (road_id, lane_data) in enumerate(hd_map):
        summary = {"road_id": road_id}
        if isinstance(lane_data, dict):
            summary["num_lanes"] = len(lane_data)
            summary["lane_ids"] = list(lane_data.keys())[:3]  # First 3 only
        all_road_summary.append(summary)
        
        if i % 100 == 0:
            print(f"  Processed {i+1}/{len(hd_map)} roads...")
    
    structure_info["all_roads_summary"] = all_road_summary
    
    # Save structure info as JSON
    structure_file = output_dir / "hd_map_structure.json"
    with open(structure_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def process_dict(d):
            if isinstance(d, dict):
                return {k: process_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [process_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        clean_structure = process_dict(structure_info)
        json.dump(clean_structure, f, indent=2)
    
    print(f"Structure analysis saved to: {structure_file}")
    
    # Save raw data as pickle for Python processing
    raw_file = output_dir / "hd_map_raw.pkl"
    with open(raw_file, 'wb') as f:
        pickle.dump(hd_map, f)
    print(f"Raw data saved to: {raw_file}")
    
    # Create a summary text file
    summary_file = output_dir / "hd_map_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("HD Map Analysis Summary\n")
        f.write("=======================\n\n")
        f.write(f"Total roads: {len(hd_map)}\n\n")
        
        f.write("First 5 roads detailed analysis:\n")
        for road_info in structure_info["road_summary"]:
            f.write(f"\nRoad {road_info['road_id']}:\n")
            f.write(f"  Lane data type: {road_info['lane_data_type']}\n")
            if "num_lanes" in road_info:
                f.write(f"  Number of lanes: {road_info['num_lanes']}\n")
                f.write(f"  Lane IDs: {road_info['lane_ids']}\n")
                if "first_lane" in road_info:
                    fl = road_info["first_lane"]
                    f.write(f"  First lane ({fl['lane_id']}):\n")
                    f.write(f"    Data type: {fl['data_type']}\n")
                    if "num_elements" in fl:
                        f.write(f"    Number of elements: {fl['num_elements']}\n")
                        f.write(f"    Element keys: {fl['element_keys']}\n")
        
        f.write(f"\nSample data structure (Road {structure_info['sample_data'].get('road_id', 'N/A')}):\n")
        if "sample_data" in structure_info and "first_element" in structure_info["sample_data"]:
            sample = structure_info["sample_data"]["first_element"]
            for key, value in sample.items():
                f.write(f"  {key}: {type(value).__name__}")
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    f.write(f" (length: {len(value)}, first: {value[0] if len(value) > 0 else 'None'})")
                f.write(f"\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return structure_info


def main():
    parser = argparse.ArgumentParser(description="Extract HD map data for analysis")
    parser.add_argument("npz_file", help="Path to HD map .npz file")
    parser.add_argument("--output-dir", "-o", default="hd_map_analysis",
                       help="Output directory for extracted data")
    
    args = parser.parse_args()
    
    structure_info = extract_hd_map_structure(args.npz_file, args.output_dir)
    
    print("\nExtraction complete! Files created:")
    print(f"  - {args.output_dir}/hd_map_structure.json (structured data)")
    print(f"  - {args.output_dir}/hd_map_raw.pkl (raw data for processing)")
    print(f"  - {args.output_dir}/hd_map_summary.txt (human-readable summary)")


if __name__ == "__main__":
    main()