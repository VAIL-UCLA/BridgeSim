import argparse
import os
import time
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from advgen.adv_generator_cat import NewAdvGenerator
from bmt.rl_train.train.scgen_generator import SCGEN_Generator
from metadrive.scenario.utils import get_number_of_scenarios
import pickle
import copy
import numpy as np


def count_files(directory):
    """Count the number of files in a directory."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])-1


def find_files_with_prefix(directory, prefix='sd_'):
    file_names = []
    file_paths = []

    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        # Filter files that start with the given prefix and store the path and file name
        for file in files:
            if file.startswith(prefix):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
                file_names.append(file)


    return file_names, file_paths


def overwrite_new_sdc_traj(original_SD, new_ego_traj, new_ego_heading, new_ego_vel):
    import copy
    new_SD = copy.deepcopy(original_SD)

    ego_id = 0
    new_ego_mask = np.ones((91,))

    assert new_ego_traj.shape[0] == new_ego_heading.shape[0] and new_ego_heading.shape[0] == new_ego_vel.shape[0]
    traj_len = new_ego_traj.shape[0]

    if new_ego_traj.shape[0] < 91:
        padding_length = 91 - new_ego_traj.shape[0]
        padding_traj = np.zeros((padding_length, 2))  # For positions
        padding_heading = np.zeros((padding_length,))  # For heading
        padding_vel = np.zeros((padding_length, 2))  # For velocity


        new_ego_traj = np.concatenate((new_ego_traj, padding_traj), axis=0)
        new_ego_heading = np.concatenate((new_ego_heading, padding_heading), axis=0)
        new_ego_vel = np.concatenate((new_ego_vel, padding_vel), axis=0)

        new_ego_mask[traj_len:] = 0

    else:
        new_ego_traj = new_ego_traj[:91]
        new_ego_heading = new_ego_heading[:91]
        new_ego_vel = new_ego_vel[:91]

    sdc_track_name = new_SD['metadata']['sdc_id']

    original_SD['tracks'][sdc_track_name]['state']['position'] = new_ego_traj
    original_SD['tracks'][sdc_track_name]['state']['velocity'] = new_ego_vel
    original_SD['tracks'][sdc_track_name]['state']['heading'] = new_ego_heading
    original_SD['tracks'][sdc_track_name]['state']['valid'] = new_ego_mask

    # new_SD["decoder/agent_position"][:, ego_id, 2] = 0.0

    return new_SD


def overwrite_new_sdc_traj_to_SD(new_SD, new_ego_traj, new_ego_heading, new_ego_vel, track_length):
    new_ego_mask = np.ones((track_length,), dtype=bool)

    assert new_ego_traj.shape[0] == new_ego_heading.shape[0] and new_ego_heading.shape[0] == new_ego_vel.shape[0]
    traj_len = new_ego_traj.shape[0]

    if new_ego_traj.shape[0] < track_length:
        padding_length = track_length - new_ego_traj.shape[0]
        padding_traj = np.zeros((padding_length, 2))  # For positions
        padding_heading = np.zeros((padding_length,))  # For heading
        padding_vel = np.zeros((padding_length, 2))  # For velocity

        new_ego_traj = np.concatenate((new_ego_traj, padding_traj), axis=0)
        new_ego_heading = np.concatenate((new_ego_heading, padding_heading), axis=0)
        new_ego_vel = np.concatenate((new_ego_vel, padding_vel), axis=0)

        new_ego_mask[traj_len:] = 0

    else:
        new_ego_traj = new_ego_traj[:track_length]
        new_ego_heading = new_ego_heading[:track_length]
        new_ego_vel = new_ego_vel[:track_length]

    sdc_track_name = new_SD['metadata']['sdc_id']

    original_ego_init_pos = new_SD['tracks'][sdc_track_name]['state']['position'][0][..., :2]
    new_ego_init_pos = new_ego_traj[0]
    dist = np.linalg.norm(original_ego_init_pos - new_ego_init_pos)
    if dist > 1:
        print(
            f"ERROR?? Original SDC initial position {original_ego_init_pos} and new SDC initial position {new_ego_init_pos} are not the same. Please check your code.")

    new_SD['tracks'][sdc_track_name]['state']['position'][..., :2] = new_ego_traj
    new_SD['tracks'][sdc_track_name]['state']['velocity'] = new_ego_vel
    new_SD['tracks'][sdc_track_name]['state']['heading'] = new_ego_heading
    new_SD['tracks'][sdc_track_name]['state']['valid'] = new_ego_mask
    for agent_name in new_SD['tracks']:
        new_SD['tracks'][agent_name]['state']['position'][..., -1] = 0  # Reset Z axis to 0

    return new_SD

class CAT_Generator(SCGEN_Generator):
    def __init__(self):
        super().__init__()  
        print("CAT generator init")
        self.adv_generator = NewAdvGenerator() # CAT API


    def generate(self, scenario_data=None, track_length=91):
        if scenario_data is None:
            assert self.env is not None
            scenario_data = self.env.engine.data_manager.current_scenario

        sid = scenario_data["id"]
        sdc_traj = self.storage[sid].get('SDC_traj')

        if sdc_traj.shape[0] <= 10:
            print("SDC traj length is too short, please check the scenario data. Skipping editing this scenario. ")
            return None

        sdc_heading = self.storage[sid].get('SDC_heading')
        sdc_vel = self.storage[sid].get('SDC_vel')

        if isinstance(sdc_traj, list):  # first time scenario in training
            sdc_traj = np.array(sdc_traj)
            sdc_vel = np.array(sdc_vel)
            sdc_heading = np.array(sdc_heading)

        if self.no_adaptive: # FIXME
            overwritten_sd = copy.deepcopy(scenario_data)
        else:
            overwritten_sd = overwrite_new_sdc_traj_to_SD(copy.deepcopy(scenario_data), sdc_traj, sdc_heading, sdc_vel,
                                                        track_length=track_length)  # need to overwrite mask as well
            
        _,_,_,_,adv_traj = self.adv_generator.generate_on_SD(overwritten_sd)

        if adv_traj is None:
            print("adv_traj is None")
            return None

        _adv_gen_cache = { 
            'adv_agent': self.adv_generator.adv_agent,
            'adv_traj_raw': adv_traj
        }

        assert _adv_gen_cache['adv_traj_raw'].shape[0] == 91, f"Wrong output traj shape: {_adv_gen_cache['adv_traj_raw'].shape[0]}"
        adv_agent = _adv_gen_cache['adv_agent']
        adv_traj = _adv_gen_cache['adv_traj_raw']


        new_SD = copy.deepcopy(scenario_data)
        new_SD['tracks'][adv_agent]['state']['position'] = adv_traj[:,:2]
        new_SD['tracks'][adv_agent]['state']['velocity'] = adv_traj[:,2:4]
        new_SD['tracks'][adv_agent]['state']['heading'] = adv_traj[:,4]

        history_length=11
        adv_history_mask = overwritten_sd['tracks'][adv_agent]['state']['valid'][:history_length] 
        reconstructed_mask = np.ones((track_length-history_length,)).astype(bool)
        adv_mask = np.concatenate([adv_history_mask, reconstructed_mask], axis=0)
        assert adv_mask.shape[0] == track_length, f"adv_mask shape is not correct: {adv_mask.shape[0]}"
        new_SD['tracks'][adv_agent]['state']['valid'] = adv_mask
        

        return new_SD



    def generate_deprecated(self, original_SD, save_scenario=True): # TODO: add save_scenario
        new_SD = copy.deepcopy(original_SD)
        current_scenario_id = new_SD['id']

        sid = original_SD["id"]
        
        sdc_traj = self.storage[sid].get('SDC_traj')
        sdc_heading = self.storage[sid].get('SDC_heading')
        sdc_vel = self.storage[sid].get('SDC_vel')

        if isinstance(sdc_traj, list): # first time scenario in training
            sdc_traj = np.array(sdc_traj)
            sdc_vel = np.array(sdc_vel)
            sdc_heading = np.array(sdc_heading)

        new_SD = overwrite_new_sdc_traj(new_SD, sdc_traj, sdc_heading, sdc_vel) # need to overwrite mask as well

        _,_,_,_,adv_traj = self.adv_generator.generate_on_SD(new_SD)

        if adv_traj is None:
            print("adv_traj is None")
            return None

        _adv_gen_cache = { # FIXME: will be deleted!!!
            'adv_agent': self.adv_generator.adv_agent,
            'adv_traj_raw': adv_traj
        }

        assert _adv_gen_cache['adv_traj_raw'].shape[0] == 91, f"Wrong output traj shape: {_adv_gen_cache['adv_traj_raw'].shape[0]}"

        adv_agent = _adv_gen_cache['adv_agent']
        adv_traj = _adv_gen_cache['adv_traj_raw']

        new_SD['tracks'][adv_agent]['state']['position'] = adv_traj[:,:2]
        new_SD['tracks'][adv_agent]['state']['velocity'] = adv_traj[:,2:4]
        new_SD['tracks'][adv_agent]['state']['heading'] = adv_traj[:,4]
        new_SD['tracks'][adv_agent]['state']['valid'] = np.ones((91,)).astype(bool)

        new_SD['id'] = original_SD['id']
        new_SD['metadata']['id'] = original_SD['metadata']['id']
        new_SD['metadata']['scenario_id'] = original_SD['metadata']['scenario_id']
        new_SD['metadata']['dataset'] = 'waymo_CAT'
        # new_SD['metadata']['source_file'] = original_SD['metadata']['source_file']
        new_SD['metadata']['selected_adv_id'] = adv_agent

        # for debugging
        if save_scenario:
            with open(f"/home/yuxin/infgen/vis_closed_loop_RL_CAT/sd_{current_scenario_id}_CAT.pkl", 'wb') as f:
                pickle.dump(new_SD, f)

        return new_SD






    
