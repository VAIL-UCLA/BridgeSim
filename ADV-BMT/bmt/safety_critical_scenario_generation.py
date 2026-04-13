import argparse
import os
import time
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from tqdm import trange
from metadrive.scenario.utils import get_number_of_scenarios
import pickle
import pdb
import omegaconf
from bmt.utils import REPO_ROOT
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
from bmt.utils import utils
import copy
from bmt.utils.config import global_config, cfg_from_yaml_file
from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
import numpy as np
import easydict
import copy


import PIL
import hydra
import matplotlib.pyplot as plt
import omegaconf
from omegaconf import OmegaConf
import seaborn as sns

from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch
import copy
import pdb

import lightning.pytorch as pl


def _to_dict(d):
    if isinstance(d, easydict.EasyDict):
        return {k: _to_dict(v) for k, v in d.items()}
    return d


def find_files_with_prefix(directory, prefix='sd_'):
    file_names = []
    file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(prefix):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
                file_names.append(file)


    return file_names, file_paths



def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes and k not in ["decoder/track_name"]:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


def run_backward(
    model, config, backward_input_dict, tokenizer, backward_mode, not_teacher_forcing_ids=None
):
    device = backward_input_dict["decoder/agent_position"].device

    # Force to run backward prediction first to make sure the data is tokenized correctly.
    tok_data_dict, _ = tokenizer.tokenize(backward_input_dict, backward_prediction=True)
    backward_input_dict.update(tok_data_dict)

    backward_input_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
    backward_input_dict["in_backward_prediction"] = torch.tensor([1], dtype=bool).to(device)

    with torch.no_grad():
        if backward_mode == "no_Adv":
            ar_func = model.model.autoregressive_rollout
            backward_output_dict = ar_func(
                backward_input_dict,
                # num_decode_steps=None,
                sampling_method=config.SAMPLING.SAMPLING_METHOD,
                temperature=config.SAMPLING.TEMPERATURE,
                topp=config.SAMPLING.TOPP,
                backward_prediction=True
            )

        else:
            assert not_teacher_forcing_ids is not None
            ar_func = model.model.autoregressive_rollout_backward_prediction_with_replay
            backward_output_dict = ar_func(
                backward_input_dict,
                # num_decode_steps=None,
                sampling_method=config.SAMPLING.SAMPLING_METHOD,
                temperature=config.SAMPLING.TEMPERATURE,
                topp=config.SAMPLING.TOPP,
                not_teacher_forcing_ids=not_teacher_forcing_ids,
            )


    backward_output_dict = tokenizer.detokenize(
        backward_output_dict,
        detokenizing_gt=False, # TODO: check is this parameter correct?
        flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING,
        backward_prediction=True,
    )
    return backward_output_dict


def load_config(config_name):
    from bmt.utils.config import global_config, cfg_from_yaml_file
    default_config = OmegaConf.load(REPO_ROOT / f"cfgs/motion_default.yaml")
    config = OmegaConf.load(REPO_ROOT / f"cfgs/{config_name}.yaml")
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.BACKWARD_PREDICTION = True
    config.ADD_CONTOUR_RELATION = True
    omegaconf.OmegaConf.set_struct(config, True)

    if default_config is not None:
        # Merge config and default config
        if isinstance(default_config, easydict.EasyDict):
            default_config = _to_dict(default_config)
        default_config = OmegaConf.create(default_config)
        config = OmegaConf.merge(default_config, config)


    return config

def choose_nearest_adv(data_dict):
    # find nearest adv for ego's ending position
    sdc_id = data_dict["decoder/sdc_index"]
    all_ooi = data_dict["decoder/agent_id"]
    sdc_mask = data_dict["decoder/agent_valid_mask"][:91:5, sdc_id]
    sdc_pos = data_dict["decoder/agent_position"][:91:5, sdc_id, :2]

    min_dist = float('inf')
    adv_id = None
    adv_closes_step = None

    for id in all_ooi:
        if id == sdc_id:
            continue
        agent_mask = data_dict["decoder/agent_valid_mask"][:91:5, id]

        mask = sdc_mask & agent_mask
        valid_steps = np.where(mask)[0] * 5  # get the original indices where valid_step is True

        agent_pos = data_dict["decoder/agent_position"][:91:5, id, :2]
            
        distances = np.linalg.norm(sdc_pos[mask] - agent_pos[mask], axis=-1)

        if distances.size == 0:
            continue
        dist = np.min(distances)
        closest_index = np.argmin(distances)
        closest_step = valid_steps[closest_index]
        

        if dist < min_dist:
            adv_id = id
            min_dist = dist
            adv_closes_step = closest_step

    if adv_id is None:
        return None, None, None, None, None
    
    # adv_closes_step = round(adv_closes_step / 5) * 5
    if adv_closes_step < 5:
        adv_closes_step = 5

    # now get adv last valid step's information
    adv_pos = data_dict["decoder/agent_position"][adv_closes_step, adv_id, :2]
    adv_heading = data_dict["decoder/agent_heading"][adv_closes_step, adv_id]
    adv_vel = data_dict["decoder/agent_velocity"][adv_closes_step, adv_id]

    return adv_id, adv_pos, adv_heading, adv_vel, adv_closes_step


def check_sdc_adv_collision(data_dict, sdc_id, sdc_pos, sdc_heading, adv_id, adv_pos, adv_heading):

    from bmt.dataset.preprocess_action_label import cal_polygon_contour, detect_collision
    contours = []

    sdc_contour = cal_polygon_contour(
        sdc_pos[0], sdc_pos[1], sdc_heading, data_dict["decoder/agent_shape"][10, sdc_id, 1],
        data_dict["decoder/agent_shape"][10, sdc_id, 0]
    )
    adv_contour = cal_polygon_contour(
        adv_pos[0], adv_pos[1], adv_heading, data_dict["decoder/agent_shape"][10, adv_id, 1],
        data_dict["decoder/agent_shape"][10, adv_id, 0]
    )

    collision_tags = detect_collision(adv_contour, [True], sdc_contour, [True])
    collision_detected = np.array(collision_tags)

    if np.any(collision_detected):
        # print("collision")
        return True

    return False


def set_adv(data_dict):
    """
    here is the current design: from existing agents, choose the one with its lastest step having nearest distance among all
    """
    ego_id = data_dict["decoder/sdc_index"]
    ego_traj = data_dict["decoder/agent_position"][:91, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:91, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:91, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:91, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:91, ego_id]

    # adv_id, adv_pos, adv_heading, adv_vel, last_valid_step = choose_nearest_adv_last_step(data_dict)
    adv_id, adv_pos, adv_heading, adv_vel, last_valid_step = choose_nearest_adv(data_dict)

    if adv_id is None:
        return None, None

    ego_last_pos = data_dict["decoder/agent_position"][last_valid_step, ego_id, :2]
    ego_last_heading = data_dict["decoder/agent_heading"][last_valid_step, ego_id]

    # begin to search
    alphas = np.arange(0, 1.05, 0.05)
    collision_point = ego_last_pos - np.random.normal(loc=0.0, scale=2, size=ego_last_pos.shape[0])
    # print("collision point:", collision_point)
    # print("ego last point:", ego_last_pos)
    for alpha in alphas:
        cand_adv_pos = (1 - alpha) * adv_pos + alpha * ego_last_pos

        if check_sdc_adv_collision(data_dict, ego_id, ego_last_pos, ego_last_heading, adv_id, cand_adv_pos,
                                             adv_heading):
            collision_point = cand_adv_pos
            break

    adv_mask = np.zeros_like(ego_mask)
    adv_mask[:last_valid_step + 1] = 1  # just like create_new_adv()
    
    data_dict["decoder/agent_valid_mask"][:, adv_id] = adv_mask

    # ===== Position =====
    data_dict["decoder/agent_position"][
        last_valid_step,
        adv_id, :2] = collision_point  # ego_traj[last_valid_step] - np.random.normal(loc=0.0, scale=2, size=3)
    # ====================

    # ===== Heading =====
    data_dict["decoder/agent_heading"][last_valid_step,
                                       adv_id] = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    # ===================

    # ===== Velocity =====
    adv_speed = np.linalg.norm(ego_velocity[last_valid_step]) * 0.5 + np.random.normal(loc=0.0, scale=0.5)
    adv_vel = adv_speed * np.array([np.cos(adv_heading), np.sin(adv_heading)])     # Project speed in heading direction
    data_dict["decoder/agent_velocity"][last_valid_step, adv_id] = adv_vel
    # ====================

    return data_dict, adv_id


def create_new_adv(data_dict):
    """
    for now, we create a new ADV and collide with ego at the final step
    """
    ego_id = data_dict["decoder/sdc_index"]

    ego_traj = data_dict["decoder/agent_position"][:, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:, ego_id]

    last_valid_step = np.where(ego_mask)[0][-1]  # Create a new ADV at the final step.
    print("last valid step:", last_valid_step)

    adv_mask = np.zeros_like(ego_mask)
    adv_mask[:last_valid_step + 1] = True

    adv_traj = np.zeros_like(ego_traj)
    adv_heading = np.zeros_like(ego_heading)
    adv_velocity = np.zeros_like(ego_velocity)
    adv_shape = np.zeros_like(ego_shape)

    # Copy the final pos/head/vel/shape of ego
    ego_last_pos = ego_traj[last_valid_step]
    collision_point = ego_last_pos # TODO: post-process the collision point 

    # ===== Position =====
    adv_traj[last_valid_step] = ego_last_pos 
    print("Ego position: ", ego_traj[last_valid_step])
    print("Adv position: ", adv_traj[last_valid_step])
    # ====================

    # ===== Heading =====
    """
    On-going: can we use modified road heading as the new adv heading?
    """
    ego_last_heading = ego_heading[last_valid_step]

    # we can try different headings and see which one is more realistic

    adv_heading[last_valid_step] = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    # + np.random.normal(loc=0.0, scale=0.5, size=1)

    print("Ego heading: ", np.rad2deg(ego_heading[last_valid_step]))
    print("Adv heading: ", np.rad2deg(adv_heading[last_valid_step]))
    # ===================

    vel_step = 10
    # ===== Velocity =====
    adv_vel = 0.5 * (ego_velocity[vel_step] + np.random.normal(loc=0.0, scale=0.5, size=2))
    ego_vel = 0.5 * (ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=2))
    print("Ego velocity: ", ego_vel, ego_velocity[last_valid_step])
    print("Adv velocity: ", adv_velocity[last_valid_step])
    data_dict["decoder/agent_velocity"][last_valid_step, ego_id] = ego_vel
    adv_velocity[last_valid_step] = adv_vel
    # ====================

    # ===== Shape =====
    for i in range(data_dict["decoder/agent_shape"].shape[0]):
        adv_shape[i] = ego_shape[last_valid_step]
    # =================

    # Insert new agent, id is the last
    data_dict["decoder/agent_position"] = np.concatenate(
        [data_dict["decoder/agent_position"], adv_traj[:, None]], axis=1
    )
    data_dict["decoder/agent_heading"] = np.concatenate(
        [data_dict["decoder/agent_heading"], adv_heading[:, None]], axis=1
    )
    data_dict["decoder/agent_velocity"] = np.concatenate(
        [data_dict["decoder/agent_velocity"], adv_velocity[:, None]], axis=1
    )

    data_dict["decoder/agent_shape"] = np.concatenate([data_dict["decoder/agent_shape"], adv_shape[:, None]], axis=1)

    data_dict["decoder/agent_valid_mask"] = np.concatenate(
        [data_dict["decoder/agent_valid_mask"], adv_mask[:, None]], axis=1
    )

    data_dict["decoder/current_agent_shape"] = np.concatenate(
        [data_dict["decoder/current_agent_shape"], data_dict["decoder/current_agent_shape"][ego_id:ego_id + 1]], axis=0
    )

    data_dict["decoder/current_agent_position"] = np.concatenate(
        [data_dict["decoder/current_agent_position"], data_dict["decoder/current_agent_shape"][ego_id:ego_id + 1]], axis=0
    )

    data_dict["decoder/agent_type"] = np.concatenate(
        [data_dict["decoder/agent_type"], data_dict["decoder/agent_type"][ego_id:ego_id + 1]], axis=0
    )
    data_dict["decoder/agent_id"] = np.concatenate(
        [data_dict["decoder/agent_id"], [len(data_dict["decoder/agent_id"])]], axis=0
    )

    # Add ADV into OOI:
    data_dict["decoder/object_of_interest_id"] = np.concatenate(
        [data_dict["decoder/object_of_interest_id"], [len(data_dict["decoder/agent_id"]) - 1]], axis=0
    )

    # Deal with some thing for forward prediction:
    data_dict["decoder/current_agent_valid_mask"] = np.concatenate(
        [data_dict["decoder/current_agent_valid_mask"], [1]], axis=0
    )

    print("====================================")
    print(
        "The new ADV is created at the final step {}, it's ID is: {}".format(
            last_valid_step,
            len(data_dict["decoder/agent_id"]) - 1
        )
    )
    print("====================================")

    return data_dict, len(data_dict["decoder/agent_id"]) - 1





def generat_one_safety_scenario(scenarionet_data, config, model, tokenizer, num_modes=1, TF_mode="no_TF", no_ADV=False, new_adv_agent=False):
    device = model.device

    flip_heading_accordingly = True
    backward_prediction = True

    scenario_data_dict = preprocess_scenario_description_for_motionlm(
        scenario=scenarionet_data, config=config, in_evaluation=True, keep_all_data=True, tokenizer=tokenizer, backward_prediction=True
    )

    raw_data_dict = data_dict = scenario_data_dict

    adv_id = None
    if not no_ADV:
        if not new_adv_agent:
            data_dict, adv_id = set_adv(data_dict, new_adv_agent=new_adv_agent)
        else:
            data_dict, adv_id = create_new_adv(data_dict)

    from bmt.utils.utils import numpy_to_torch
    input_data = numpy_to_torch(data_dict, device=model.device)

    original_data_dict_tensor = copy.deepcopy(input_data)

    # Extend the batch dim:
    input_data = {
        k: utils.expand_for_modes(v.unsqueeze(0), num_modes=num_modes) if isinstance(v, torch.Tensor) else v
        for k, v in input_data.items()
    }
    input_data["in_backward_prediction"] = torch.tensor([True] * num_modes, dtype=bool).to(model.device)

    not_tf_ids = None
    if not no_ADV:
        all_agents = input_data["decoder/agent_id"][0]

        if TF_mode == "no_TF":
            not_tf_ids = all_agents

        elif TF_mode == "sdc_TF":
            not_tf_ids = all_agents[all_agents != 0]

        elif TF_mode == "all_TF_except_adv":
            not_tf_ids = torch.tensor([adv_id]).to(model.device)

    else:
        TF_mode = "no_Adv"

    with torch.no_grad():
        output_data = run_backward(
            model=model,
            config=config,
            backward_input_dict=input_data,
            tokenizer=tokenizer,
            backward_mode=TF_mode,
            not_teacher_forcing_ids=not_tf_ids
        )

    output_data_all_modes = {
        k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        for k, v in output_data.items()
    }

    best_mode = 0

    # ======= mode selection, not used for now ==========
    # min_Dist_to_GT = float('inf')
    # best_mode = None
    # for i in range(num_modes): # compare which mode to select
    #     output_dict_mode = _get_mode(output_data_all_modes, i, num_modes=num_modes)

    #     if no_ADV:
    #         ooi = None
    #     else:
    #         ooi = [adv_id]
    #     dist_to_GT = compute_avg_startpoint_distance_to_GT(output_dict_mode, ooi=ooi)

    #     if dist_to_GT < min_Dist_to_GT:
    #         best_mode = i
    #         min_Dist_to_GT = dist_to_GT

    # print("best mode:", best_mode)
    # ======= mode selection ==========



    output_dict_mode = _get_mode(output_data_all_modes, best_mode, num_modes=num_modes)

    return output_dict_mode, adv_id, not_tf_ids



def overwrite_to_scenario_description(output_dict_mode, original_SD, ooi=None, adv_id=None):
    # overwrite original SD with all predicted ooi trajectories included
    if not ooi:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    sdc_track_name = original_SD['metadata']['sdc_id']
    if adv_id is not None:
        adv_track_name = str(output_dict_mode['decoder/track_name'][int(adv_id)].item())

    for id in ooi:
        track_name = output_dict_mode['decoder/track_name']
        agent_track_name = str(output_dict_mode['decoder/track_name'][int(id)].item())
            
        agent_traj = output_dict_mode["decoder/agent_position"][:, id, ]
        agent_heading = output_dict_mode["decoder/agent_heading"][:, id]
        agent_vel = output_dict_mode["decoder/agent_velocity"][:, id]
        agent_traj_mask = output_dict_mode["decoder/agent_valid_mask"][:, id]

        # modify adv info
        # agent_z = original_SD['tracks'][agent_track_name]['state']['position'][10, 2]  # fill the z-axis
        # agent_traj_z = np.full((91, 1), agent_z)
        # agent_new_traj = np.concatenate([agent_traj, agent_traj_z], axis=1)
        # print("new_traj:", agent_new_traj.shape)
        original_SD['tracks'][agent_track_name]['state']['position'] = agent_traj
        original_SD['tracks'][agent_track_name]['state']['velocity'] = agent_vel
        original_SD['tracks'][agent_track_name]['state']['heading'] = agent_heading
        original_SD['tracks'][agent_track_name]['state']['valid'] = agent_traj_mask

        length = original_SD['tracks'][agent_track_name]['state']['length'][10]
        width = original_SD['tracks'][agent_track_name]['state']['width'][10]
        height = original_SD['tracks'][agent_track_name]['state']['height'][10]
        original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
        original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
        original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)

    if adv_id is not None:
        original_SD['metadata']['selected_adv_id'] = adv_track_name

    return original_SD


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    assert "decoder/agent_position" in data_dict, "Have you set EVALUATION.PREDICT_ALL_AGENTS to False?"
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert data_dict["decoder/agent_position"].ndim == 3

    expanded_mask = data_dict["decoder/agent_valid_mask"][:, :, None]
    data_dict["decoder/agent_position"] += map_center * expanded_mask

    return data_dict

def compute_avg_ADE_to_GT(output_data_current_mode, ooi=None):
    # compute the distance from gt to all reconstructed agents
    T, N, _ = output_data_current_mode["decoder/agent_position"].shape
    sum_ADE = 0
    mask = output_data_current_mode["decoder/agent_valid_mask"]

    if not ooi:
        ooi = np.arange(N)

    for i in ooi:
        agent_mask = mask[:,i]
        dist = np.linalg.norm(output_data_current_mode["decoder/agent_position"][:91,i,:2][agent_mask] - output_data_current_mode["decoder/reconstructed_position"][5:,i,:2][agent_mask], axis=-1)
        sum_ADE += np.sum(dist)

    return sum_ADE/N
    

def compute_avg_startpoint_distance_to_GT(output_data_current_mode, ooi=None):
    # compute the distance from gt to all reconstructed agents
    T, N, _ = output_data_current_mode["decoder/agent_position"].shape
    sum_ADE = 0
    mask = output_data_current_mode["decoder/agent_valid_mask"]

    if not ooi:
        ooi = np.arange(N)

    for i in ooi:

        agent_mask = mask[:11,i]
        dist = np.linalg.norm(output_data_current_mode["decoder/agent_position"][:11,i,:2][agent_mask] - output_data_current_mode["decoder/reconstructed_position"][5:16,i,:2][agent_mask], axis=-1)
        sum_ADE += np.sum(dist)

    return sum_ADE/N

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="The data folder storing raw tfrecord from Waymo dataset.")
    parser.add_argument("--num_scenario", type=int, default=1)
    parser.add_argument("--save_dir", required=True, help="The place to store .pkl file if different from given in --dir")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/last.ckpt")
    parser.add_argument("--config_name", type=str, default="1026_gpt")
    parser.add_argument("--TF_mode", type=str, default="no_TF")
    parser.add_argument("--no_ADV", action='store_true')
    parser.add_argument("--num_mode", type=int, default=1)
    parser.add_argument("--new_agent_ADV", action='store_true')
    
    """
    TF_mode: [no_TF, sdc_TF, all_TF_except_adv]
    """

    args = parser.parse_args()
    CKPT_PATH = args.ckpt_path
    CONFIG_NAME = args.config_name

    original_dataset_dir = args.dir
    save_dir = args.save_dir

    TF_mode = args.TF_mode
    no_Adv = args.no_ADV
    new_agent_adv = args.new_agent_ADV

    config = load_config(CONFIG_NAME)
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.BACKWARD_PREDICTION = True  # <<<
    config.ADD_CONTOUR_RELATION = True
    omegaconf.OmegaConf.set_struct(config, True)

    if CKPT_PATH:
        model = utils.get_model(checkpoint_path=CKPT_PATH)
    else:
        model = utils.get_model(config=config)

    model = model.to("cuda")
    device = model.device

    tokenizer = model.model.tokenizer

    num_scenario = int(args.num_scenario)
    if num_scenario == -1:
        num_scenario = get_number_of_scenarios(original_dataset_dir)
    print("number of scenarios to be generated:", num_scenario)


    file_names, file_paths = find_files_with_prefix(original_dataset_dir)

    num_modes = args.num_mode
    total_test_scenarios = num_scenario
    tested_scenario_count = 0
    success_scenarios = []


    pbar = trange(num_scenario)
    for i in pbar:
        scenario_file = file_paths[i]
        with open(scenario_file, 'rb') as f:
            scenario_data = pickle.load(f)
        f.close()

        current_scenario_id = scenario_data['id']
        print("now processing:", current_scenario_id)
        original_SD = copy.deepcopy(scenario_data)

        output_dict, adv_id, not_tf_ids = generat_one_safety_scenario(scenario_data, config, model, tokenizer, TF_mode=TF_mode, no_ADV=no_Adv, num_modes=num_modes, new_adv_agent=new_agent_adv)

        # output_dict_mode = _get_mode(output_dict, i) # TODO: mode selection

        # ============ plot for debugging ================
        from bmt.gradio_ui.plot import plot_gt, plot_pred
        plot_pred(output_dict, save_path=f"vis_SCGEN_new_agent/vis_{current_scenario_id}_pred.png", ooi=[adv_id])
        plot_gt(output_dict, save_path=f"vis_SCGEN_new_agent/vis_{current_scenario_id}_gt.png", ooi=[adv_id])

        continue # for debugging

        from bmt.dataset.scenarionet_utils import _overwrite_datadict_all_agents

        if no_Adv:
            output_dict = _overwrite_datadict_all_agents(output_dict, backward_detokenize=True) 
        else:
            output_dict = _overwrite_datadict_all_agents(output_dict, ooi=not_tf_ids.cpu().numpy(), backward_detokenize=True) 

        # ============ plot for debugging ================
        from bmt.gradio_ui.plot import plot_gt, plot_pred
        # plot_gt(output_dict, save_path=f"debug_backward_{current_scenario_id}_gt.png")


        output_dict = transform_to_global_coordinate(output_dict)
        plot_gt(output_dict, save_path=f"debug_backward_{current_scenario_id}_global.png", draw_map=False)

        if no_Adv:
            original_SD = overwrite_to_scenario_description(output_dict, original_SD)
        else:
            original_SD = overwrite_to_scenario_description(output_dict, original_SD, ooi=[adv_id], adv_id=adv_id)

        original_SD['id'] = original_SD['id'] + '_SCGen'
        if "id" in original_SD['metadata']:
            original_SD['metadata']['id'] = original_SD['metadata']['id'] + '_SCGen'
        original_SD['metadata']['scenario_id'] = original_SD['metadata']['scenario_id'] + '_SCGen'
        original_SD['metadata']['dataset'] = 'SCGen_waymo'
        original_SD['metadata']['source_file'] = original_SD['metadata']['source_file'] + '_scgen'

        success_scenarios.append(original_SD)

        tested_scenario_count += 1
        pbar.update(1)

    from metadrive.scenario.utils import save_dataset

    save_dataset(
        scenario_list=success_scenarios,
        dataset_name='SCGEN_Waymo',
        dataset_version='v0',
        dataset_dir=save_dir
    )

    return



if __name__ == '__main__':
    main()

