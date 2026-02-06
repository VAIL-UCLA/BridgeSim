#!/usr/bin/env python3
"""
Script to generate XML route file for HardBreakRoute_Town01_Route30_Weather3 scenario
from Bench2Drive-mini dataset annotations
"""

import json
import gzip
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
from pathlib import Path

def load_annotation_file(filepath):
    """Load a single annotation file (handles both .json and .json.gz)"""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)

def filter_waypoints(waypoints, min_distance=2.0, angle_threshold=10.0):
    """
    Filter waypoints to remove redundant ones based on distance and direction changes.
    
    Args:
        waypoints: List of waypoint dictionaries with 'x', 'y', 'z' keys
        min_distance: Minimum distance in meters between waypoints
        angle_threshold: Minimum angle change in degrees to keep a waypoint
    
    Returns:
        Filtered list of waypoints
    """
    if len(waypoints) <= 2:
        return waypoints
    
    filtered = [waypoints[0]]  # Always keep the first waypoint
    
    for i in range(1, len(waypoints) - 1):
        current = waypoints[i]
        last_kept = filtered[-1]
        next_wp = waypoints[i + 1]
        
        # Calculate distance from last kept waypoint
        dx = current['x'] - last_kept['x']
        dy = current['y'] - last_kept['y']
        dz = current['z'] - last_kept['z']
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # If distance is significant, keep this waypoint
        if distance >= min_distance:
            filtered.append(current)
            continue
        
        # Calculate direction change (angle between vectors)
        # Vector from last_kept to current
        v1 = [current['x'] - last_kept['x'], current['y'] - last_kept['y']]
        # Vector from current to next
        v2 = [next_wp['x'] - current['x'], next_wp['y'] - current['y']]
        
        # Calculate angle between vectors
        if math.sqrt(v1[0]**2 + v1[1]**2) > 0.1 and math.sqrt(v2[0]**2 + v2[1]**2) > 0.1:
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            angle_deg = math.degrees(math.acos(cos_angle))
            
            # If there's a significant direction change, keep this waypoint
            if angle_deg >= angle_threshold:
                filtered.append(current)
    
    # Always keep the last waypoint
    filtered.append(waypoints[-1])
    
    return filtered

def extract_waypoints_from_annotations(anno_dir, sample_interval=1):
    """
    Extract waypoints from annotation files
    sample_interval: Extract waypoint every N frames
    """
    waypoints = []
    brake_points = []

    # Get all annotation files sorted by frame number
    anno_files = sorted([f for f in os.listdir(anno_dir) if f.endswith(('.json', '.json.gz'))])

    print(f"Found {len(anno_files)} annotation files")

    for i, filename in enumerate(anno_files):
        if i % sample_interval == 0:  # Sample every N frames
            filepath = os.path.join(anno_dir, filename)
            data = load_annotation_file(filepath)

            # Extract ego vehicle position from bounding boxes
            ego_vehicle = None
            for bbox in data.get('bounding_boxes', []):
                if bbox.get('class') == 'ego_vehicle':
                    ego_vehicle = bbox
                    break
            
            if ego_vehicle and 'location' in ego_vehicle:
                x, y, z = ego_vehicle['location']
                waypoints.append({'x': x, 'y': y, 'z': z})
            else:
                # Fallback to original x, y if ego vehicle not found
                x = data.get('x', 0)
                y = data.get('y', 0)
                z = 0.0
                waypoints.append({'x': x, 'y': y, 'z': z})

            # Check for brake events (only if we found ego vehicle position)
            if ego_vehicle and 'location' in ego_vehicle:
                if data.get('should_brake', False) or data.get('brake', 0) > 0.5:
                    brake_points.append({
                        'frame': i,
                        'x': x,
                        'y': y,
                        'z': z
                    })
                    print(f"Found brake event at frame {i}: x={x:.2f}, y={y:.2f}")

    # Always include the last frame
    if len(anno_files) > 0:
        last_file = os.path.join(anno_dir, anno_files[-1])
        last_data = load_annotation_file(last_file)
        
        # Extract ego vehicle position from bounding boxes for last frame
        ego_vehicle = None
        for bbox in last_data.get('bounding_boxes', []):
            if bbox.get('class') == 'ego_vehicle':
                ego_vehicle = bbox
                break
        
        if ego_vehicle and 'location' in ego_vehicle:
            x, y, z = ego_vehicle['location']
            waypoints.append({'x': x, 'y': y, 'z': z})
        else:
            # Fallback to original x, y if ego vehicle not found
            waypoints.append({
                'x': last_data.get('x', 0),
                'y': last_data.get('y', 0),
                'z': 0.0
            })

    # Filter waypoints to remove redundant ones (car stopped or minimal movement)
    original_count = len(waypoints)
    filtered_waypoints = filter_waypoints(waypoints)
    print(f"Filtered waypoints: {original_count} -> {len(filtered_waypoints)} (removed {original_count - len(filtered_waypoints)} redundant points)")
    
    return filtered_waypoints, brake_points

def create_route_xml(waypoints, brake_points, route_id="30", town="Town01"):
    """Create XML structure for the route"""

    # Create root element
    root = ET.Element("routes")

    # Create route element
    route = ET.SubElement(root, "route", {
        "id": route_id,
        "town": town
    })

    # Add waypoints
    waypoints_elem = ET.SubElement(route, "waypoints")
    for wp in waypoints:
        ET.SubElement(waypoints_elem, "position", {
            "x": f"{wp['x']:.1f}",
            "y": f"{wp['y']:.1f}",
            "z": f"{wp['z']:.1f}"
        })

    # Add scenarios
    scenarios_elem = ET.SubElement(route, "scenarios")

    # Add HardBreak scenario using the first waypoint as trigger point
    if waypoints:
        # Use the first waypoint as trigger point
        first_waypoint = waypoints[0]
        scenario = ET.SubElement(scenarios_elem, "scenario", {
            "name": "HardBreakRoute_1",
            "type": "HardBreakRoute"
        })

        # Add trigger point at the first position
        ET.SubElement(scenario, "trigger_point", {
            "x": f"{first_waypoint['x']:.1f}",
            "y": f"{first_waypoint['y']:.1f}",
            "z": f"{first_waypoint['z']:.1f}",
            "yaw": "180"  # Assuming northbound traffic
        })

    # Add weather configuration (Weather3)
    weathers_elem = ET.SubElement(route, "weathers")

    # Weather3 from weather.xml: cloudiness=60, precipitation=60, etc.
    weather_start = ET.SubElement(weathers_elem, "weather", {
        "route_percentage": "0",
        "cloudiness": "60.0",
        "precipitation": "60.0",
        "precipitation_deposits": "60.0",
        "wetness": "0.0",
        "wind_intensity": "60.0",
        "sun_azimuth_angle": "-1.0",
        "sun_altitude_angle": "45.0",
        "fog_density": "3.0",
        "fog_distance": "0.0",
        "fog_falloff": "0.0"
    })

    weather_end = ET.SubElement(weathers_elem, "weather", {
        "route_percentage": "100",
        "cloudiness": "60.0",
        "precipitation": "60.0",
        "precipitation_deposits": "60.0",
        "wetness": "0.0",
        "wind_intensity": "60.0",
        "sun_azimuth_angle": "-1.0",
        "sun_altitude_angle": "45.0",
        "fog_density": "3.0",
        "fog_distance": "0.0",
        "fog_falloff": "0.0"
    })

    return root

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="   ")

def main():
    # Path to the HardBreakRoute scenario
    scenario_dir = "/home/abhijit/Work/BridgeSim/Bench2Drive/Bench2Drive-mini/HardBreakRoute_Town01_Route30_Weather3"
    anno_dir = os.path.join(scenario_dir, "anno")

    # Output XML file path
    output_file = "/home/abhijit/Work/BridgeSim/Bench2Drive/leaderboard/data/hardbreak_route30.xml"

    print(f"Processing annotations from: {anno_dir}")

    # Extract waypoints (using every frame since we're filtering)
    waypoints, brake_points = extract_waypoints_from_annotations(anno_dir, sample_interval=1)

    print(f"Extracted {len(waypoints)} filtered waypoints")
    print(f"Found {len(brake_points)} brake events")

    # Show route summary
    if waypoints:
        print(f"Route starts at: x={waypoints[0]['x']:.2f}, y={waypoints[0]['y']:.2f}")
        print(f"Route ends at: x={waypoints[-1]['x']:.2f}, y={waypoints[-1]['y']:.2f}")

        # Calculate approximate route length
        total_distance = 0
        for i in range(1, len(waypoints)):
            dx = waypoints[i]['x'] - waypoints[i-1]['x']
            dy = waypoints[i]['y'] - waypoints[i-1]['y']
            total_distance += math.sqrt(dx**2 + dy**2)
        print(f"Approximate route length: {total_distance:.2f} meters")

    # Create XML structure
    root = create_route_xml(waypoints, brake_points)

    # Save to file
    xml_str = prettify_xml(root)

    # Remove the XML declaration added by minidom (we'll add our own)
    lines = xml_str.split('\n')
    if lines[0].startswith('<?xml'):
        lines = lines[1:]

    # Add proper XML declaration and save
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('\n'.join(lines))

    print(f"\nXML route file generated: {output_file}")
    print("\nRoute configuration:")
    print("- Waypoints extracted from ego vehicle bounding box locations")
    print("- Filtered to remove redundant waypoints (min 2m distance or 10° direction change)")
    print("- Trigger point set to first ego vehicle position")
    print("\nTo use this route:")
    print("1. Update BASE_ROUTES in run_evaluation_multi_uniad.sh to:")
    print(f"   BASE_ROUTES=leaderboard/data/hardbreak_route30")
    print("2. Ensure IS_BENCH2DRIVE=True in the evaluation script")
    print("3. Run the evaluation with the modified script")

if __name__ == "__main__":
    main()