#!/usr/bin/env python3
import os
import json
import gzip
import tarfile
import tempfile

def extract_type_ids_from_scenario(scenario_path, use_extracted=False):
    """Extract type_id values from a single scenario (tar.gz or extracted directory)"""
    type_ids = set()
    
    if use_extracted and os.path.isdir(scenario_path):
        # Use already extracted directory
        anno_dir = os.path.join(scenario_path, "anno")
        if os.path.exists(anno_dir):
            for filename in os.listdir(anno_dir):
                if filename.endswith('.json.gz'):
                    filepath = os.path.join(anno_dir, filename)
                    try:
                        with gzip.open(filepath, 'rt') as f:
                            data = json.load(f)
                            if 'bounding_boxes' in data:
                                for bbox in data['bounding_boxes']:
                                    if 'type_id' in bbox:
                                        type_ids.add(bbox['type_id'])
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                elif filename.endswith('.json'):
                    filepath = os.path.join(anno_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            if 'bounding_boxes' in data:
                                for bbox in data['bounding_boxes']:
                                    if 'type_id' in bbox:
                                        type_ids.add(bbox['type_id'])
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
    else:
        # Extract tar.gz file temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(scenario_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
                
                # Find the scenario directory inside temp_dir
                extracted_dirs = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
                if extracted_dirs:
                    scenario_dir = os.path.join(temp_dir, extracted_dirs[0])
                    anno_dir = os.path.join(scenario_dir, "anno")
                    
                    if os.path.exists(anno_dir):
                        for filename in os.listdir(anno_dir):
                            if filename.endswith('.json.gz'):
                                filepath = os.path.join(anno_dir, filename)
                                try:
                                    with gzip.open(filepath, 'rt') as f:
                                        data = json.load(f)
                                        if 'bounding_boxes' in data:
                                            for bbox in data['bounding_boxes']:
                                                if 'type_id' in bbox:
                                                    type_ids.add(bbox['type_id'])
                                except Exception as e:
                                    print(f"Error processing {filepath}: {e}")
    
    return type_ids

def main():
    bench2drive_dir = "/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini"
    
    if not os.path.exists(bench2drive_dir):
        print(f"Directory {bench2drive_dir} not found!")
        return
    
    all_type_ids = set()
    scenario_count = 0
    type_id_to_scenarios = {}  # Track which scenarios contain each type_id
    
    # Process all files in the directory
    for filename in os.listdir(bench2drive_dir):
        filepath = os.path.join(bench2drive_dir, filename)
        
        if filename.endswith('.tar.gz'):
            print(f"Processing tar.gz: {filename}")
            type_ids = extract_type_ids_from_scenario(filepath, use_extracted=False)
            all_type_ids.update(type_ids)
            scenario_count += 1
            print(f"  Found {len(type_ids)} unique type_ids in this scenario")
            
            # Track type_ids to scenarios
            for type_id in type_ids:
                if type_id not in type_id_to_scenarios:
                    type_id_to_scenarios[type_id] = []
                type_id_to_scenarios[type_id].append(filename)
                
        elif os.path.isdir(filepath) and not filename.startswith('.'):
            # Check if it's an already extracted scenario directory
            anno_dir = os.path.join(filepath, "anno")
            if os.path.exists(anno_dir):
                print(f"Processing extracted directory: {filename}")
                type_ids = extract_type_ids_from_scenario(filepath, use_extracted=True)
                all_type_ids.update(type_ids)
                scenario_count += 1
                print(f"  Found {len(type_ids)} unique type_ids in this scenario")
                
                # Track type_ids to scenarios
                for type_id in type_ids:
                    if type_id not in type_id_to_scenarios:
                        type_id_to_scenarios[type_id] = []
                    type_id_to_scenarios[type_id].append(filename)
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {scenario_count} scenarios")
    print(f"Found {len(all_type_ids)} unique type_id values total:")
    print()
    
    # Sort the type_ids for better readability
    sorted_type_ids = sorted(all_type_ids)
    for i, type_id in enumerate(sorted_type_ids, 1):
        print(f"{i:2d}. {type_id}")
    
    # Find pedestrian scenario
    print(f"\n=== PEDESTRIAN SCENARIO ===")
    pedestrian_type_id = "walker.pedestrian.0007"
    if pedestrian_type_id in type_id_to_scenarios:
        print(f"Pedestrian '{pedestrian_type_id}' found in scenarios:")
        for scenario in type_id_to_scenarios[pedestrian_type_id]:
            print(f"  - {scenario}")
    else:
        print(f"Pedestrian '{pedestrian_type_id}' not found in any scenario!")
    
    # Find traffic light scenarios
    print(f"\n=== TRAFFIC LIGHT SCENARIOS ===")
    traffic_light_type_id = "traffic.traffic_light"
    if traffic_light_type_id in type_id_to_scenarios:
        print(f"Traffic lights '{traffic_light_type_id}' found in scenarios:")
        for scenario in type_id_to_scenarios[traffic_light_type_id]:
            print(f"  - {scenario}")
    else:
        print(f"Traffic lights '{traffic_light_type_id}' not found in any scenario!")

if __name__ == "__main__":
    main()