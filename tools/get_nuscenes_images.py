import os
import shutil
from nuscenes import NuScenes

def extract_all_sweeps():
    # 1. Initialise (Read-Only access)
    nusc = NuScenes(version='v1.0-mini', dataroot='/cogrob-avl-west/nuScenes/', verbose=True)

    dest_root = '/tmp/nuscenes'
    
    sensors_of_interest = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    print(f"Starting extraction of full sweep data to {dest_root}...")

    for scene in nusc.scene:
        scene_name = scene['name']
        print(f"Processing {scene_name}...")

        # Get the start and end tokens for this scene
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']

        # We need the timestamp of the last sample to know when to stop
        last_sample = nusc.get('sample', last_sample_token)
        end_timestamp = last_sample['timestamp']

        # Get the first sample so we can find the starting point for each sensor
        first_sample = nusc.get('sample', first_sample_token)

        for sensor_name in sensors_of_interest:
            # 1. Start at the first keyframe for this sensor
            current_sd_token = first_sample['data'][sensor_name]

            # Prepare destination folder
            dest_dir = os.path.join(dest_root, scene_name, sensor_name)
            os.makedirs(dest_dir, exist_ok=True)

            # 2. Traverse the linked list (Sweeps + Samples)
            while current_sd_token != '':
                # Fetch the sensor data record
                sd_record = nusc.get('sample_data', current_sd_token)
                
                # CHECK: Have we passed the end of the scene?
                if sd_record['timestamp'] > end_timestamp:
                    break

                # Get the full source path (handles both 'samples' and 'sweeps' folders automatically)
                src_path = nusc.get_sample_data_path(current_sd_token)
                
                # Copy file
                filename = os.path.basename(src_path)
                dest_path = os.path.join(dest_dir, filename)
                
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)

                # Move to the next frame in the stream (approx 0.08s later)
                current_sd_token = sd_record['next']

    print("Done! All sweeps organised.")

if __name__ == "__main__":
    extract_all_sweeps()