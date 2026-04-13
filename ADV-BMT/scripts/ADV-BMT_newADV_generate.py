"""
This script is used to test backward and forward to replay all trajectories to get the ADV's probability.
"""

import pickle
import hydra
import numpy as np
import tqdm
from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
from bmt.utils import utils
from omegaconf import OmegaConf
import torch
import copy
from bmt.utils import cal_polygon_contour
from bmt.gradio_ui.plot import plot_pred, plot_gt
import os
import time as timer
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType

REPO_ROOT = utils.REPO_ROOT

SOURCE_DIR = "/home/yuxin/infgen/1023_failed_baseline_3"
SAVE_DIR = "/home/yuxin/infgen/1023_failed_baseline_generated_scene_3"

object_type_to_int = {
    MetaDriveType.UNSET: 0,
    MetaDriveType.VEHICLE: 1,
    MetaDriveType.PEDESTRIAN: 2,
    MetaDriveType.CYCLIST: 3,
    MetaDriveType.OTHER: 4
}

object_type_to_shape = {
    MetaDriveType.PEDESTRIAN: np.array([1.2169257, 1.0367688, 1.9874878], dtype=np.float32),
    MetaDriveType.CYCLIST: np.array([1.9883213, 1.0218738, 1.752892 ], dtype=np.float32),
}

def create_new_adv(data_dict, col_heading=None, col_step=None, ADV_type=MetaDriveType.VEHICLE):
    """
    for now, we create a new ADV and collide with ego at the final step
    """

    ego_id = data_dict["decoder/sdc_index"]

    ego_traj = data_dict["decoder/agent_position"][:, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:, ego_id]

    if col_step is None:
        last_valid_step = np.where(ego_mask)[0][-1]  # Create a new ADV at the final step.
    else:
        last_valid_step = col_step
    # print("last valid step:", last_valid_step)

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
    # ====================

    # ===== Heading =====
    """
    On-going: can we use modified road heading as the new adv heading?
    """
    ego_last_heading = ego_heading[last_valid_step]

    if col_heading is None:
        adv_heading[last_valid_step] = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    else:
        adv_heading[last_valid_step] = col_heading
    # adv_heading[last_valid_step] = COL_HEADING 

    # print("Ego heading: ", np.rad2deg(ego_heading[last_valid_step]))
    # print("Adv heading: ", np.rad2deg(adv_heading[last_valid_step]))
    # ===================

    # ===== Velocity =====
    adv_speed = np.linalg.norm(ego_velocity[last_valid_step]) * 0.1 + np.random.normal(loc=0.0, scale=0.5)
    new_heading = adv_heading[last_valid_step]
    adv_velocity[last_valid_step] = adv_speed * np.array([np.cos(new_heading), np.sin(new_heading)])     # Project speed in heading direction
 
    # ego_speed = np.linalg.norm(ego_velocity[last_valid_step]) * 0.5 + np.random.normal(loc=0.0, scale=0.5)
    # ego_heading_last = ego_heading[last_valid_step]
    # ego_velocity[last_valid_step] = ego_speed * np.array([np.cos(ego_heading_last), np.sin(ego_heading_last)])
    # ====================

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

    data_dict["decoder/agent_valid_mask"] = np.concatenate(
        [data_dict["decoder/agent_valid_mask"], adv_mask[:, None]], axis=1
    )

    data_dict["decoder/current_agent_position"] = np.concatenate(
        [data_dict["decoder/current_agent_position"], data_dict["decoder/current_agent_position"][ego_id:ego_id + 1]], axis=0
    )

    if ADV_type == MetaDriveType.VEHICLE:
        data_dict["decoder/current_agent_shape"] = np.concatenate(
            [data_dict["decoder/current_agent_shape"], data_dict["decoder/current_agent_shape"][ego_id:ego_id + 1]], axis=0
        )
        # ===== Shape =====
        for i in range(data_dict["decoder/agent_shape"].shape[0]):
            adv_shape[i] = ego_shape[last_valid_step]
        # =================

    else:
        data_dict["decoder/current_agent_shape"] = np.concatenate(
            [data_dict["decoder/current_agent_shape"], object_type_to_shape[ADV_type][None, :]], axis=0
        )
        # ===== Shape =====
        for i in range(data_dict["decoder/agent_shape"].shape[0]):
            adv_shape[i] = object_type_to_shape[ADV_type]
        # =================

    data_dict["decoder/agent_shape"] = np.concatenate([data_dict["decoder/agent_shape"], adv_shape[:, None]], axis=1)


    # modify the ADV type to corresponding type
    data_dict["decoder/agent_type"] = np.concatenate(
        [data_dict["decoder/agent_type"], [object_type_to_int[ADV_type]]], axis=0
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


def _overwrite_datadict_all_agents(source_data_dict, dest_data_dict, ooi=None):
    import copy
    new_data_dict = copy.deepcopy(dest_data_dict)
    B, T, N, _ = source_data_dict["decoder/reconstructed_position"].shape

    if ooi is None:
        ooi = np.arange(N)

    for id in ooi:  # overwrite all agents
        traj = source_data_dict["decoder/reconstructed_position"][:, :91, id, ]
        traj_mask = source_data_dict["decoder/reconstructed_valid_mask"][:, :91, id]
        theta = source_data_dict['decoder/reconstructed_heading'][:, :91, id]
        vel = source_data_dict['decoder/reconstructed_velocity'][:, :91, id]

        new_data_dict["decoder/agent_position"][:, :, id, :2] = traj
        new_data_dict["decoder/agent_position"][:, :, id, 2] = 0.0
        new_data_dict["decoder/agent_valid_mask"][:, :, id] = traj_mask
        new_data_dict["decoder/agent_heading"][:, :, id] = theta
        new_data_dict["decoder/agent_velocity"][:, :, id] = vel

    return new_data_dict


def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


def transform_to_global_coordinate(data_dict, from_GT):
    new_data_dict = copy.deepcopy(data_dict)
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    assert "decoder/agent_position" in data_dict, "Have you set EVALUATION.PREDICT_ALL_AGENTS to False?"
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert data_dict["decoder/agent_position"].ndim == 3

    if from_GT:
        expanded_mask = data_dict["decoder/agent_valid_mask"][:, :, None]
        new_data_dict["decoder/agent_position"] += map_center * expanded_mask
    else:
        expanded_mask = data_dict["decoder/reconstructed_valid_mask"][:, :, None]
        new_data_dict["decoder/reconstructed_position"] += map_center[:,:,:2] * expanded_mask

    return new_data_dict


def overwrite_to_scenario_description_new_agent(output_dict_mode, original_SD, adv_id, from_GT, ADV_type='VEHICLE', ooi=None, success_mode=0):
    """
    Overwrite from GT field in data_dict to tracks in SD for OOI agents.
    Note that here we assume trajectory starts from T_start=0 to T_end=90, 
    """

    if ooi is None:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    # adv_track_name = 'new_adv_agent'
    adv_track_name = '99999'
    original_SD['tracks'][adv_track_name] = {'state': {}, 'type': ADV_type, 'metadata': {}}
    sdc_track_name = original_SD['metadata']['sdc_id']

    for id in ooi:
        if id == adv_id:
            # agent_track_name = 'new_adv_agent'
            agent_track_name = adv_track_name
        else:
            agent_track_name = str(output_dict_mode['decoder/track_name'][id].item())


        if from_GT:
            agent_traj = output_dict_mode["decoder/agent_position"][:, id, ]
            agent_heading = output_dict_mode["decoder/agent_heading"][:, id]
            agent_vel = output_dict_mode["decoder/agent_velocity"][:, id]
            agent_traj_mask = output_dict_mode["decoder/agent_valid_mask"][:, id]

        else:
            agent_traj = output_dict_mode["decoder/reconstructed_position"][:, id, ]
            agent_heading = output_dict_mode["decoder/reconstructed_heading"][:, id]
            agent_vel = output_dict_mode["decoder/reconstructed_velocity"][:, id]
            agent_traj_mask = output_dict_mode["decoder/reconstructed_valid_mask"][:, id]

        # modify adv info
        # agent_z = original_SD['tracks'][agent_track_name]['state']['position'][10, 2]  # fill the z-axis
        # agent_traj_z = np.full((91, 1), agent_z)
        # agent_new_traj = np.concatenate([agent_traj, agent_traj_z], axis=1)
        # print("new_traj:", agent_new_traj.shape)
        original_SD['tracks'][agent_track_name]['state']['position'] = agent_traj
        original_SD['tracks'][agent_track_name]['state']['velocity'] = agent_vel
        original_SD['tracks'][agent_track_name]['state']['heading'] = agent_heading
        original_SD['tracks'][agent_track_name]['state']['valid'] = agent_traj_mask

        if ADV_type=='VEHICLE':
            length = original_SD['tracks'][sdc_track_name]['state']['length'][10]
            width = original_SD['tracks'][sdc_track_name]['state']['width'][10]
            height = original_SD['tracks'][sdc_track_name]['state']['height'][10]
            original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
            original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
            original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)

        else:
            length = object_type_to_shape[ADV_type][0]
            width = object_type_to_shape[ADV_type][1]
            height = object_type_to_shape[ADV_type][2]
            original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
            original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
            original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)



    original_SD['tracks'][adv_track_name]['metadata']['dataset'] = 'waymo'
    original_SD['tracks'][adv_track_name]['metadata']['object_id'] = adv_track_name
    original_SD['tracks'][adv_track_name]['metadata']['track_length'] = 91
    original_SD['tracks'][adv_track_name]['metadata']['type'] = ADV_type
    original_SD['metadata']['new_adv_id'] = adv_track_name
    original_SD['metadata']['objects_of_interest'].append(adv_track_name)
    tracks_length = len(list(original_SD['tracks'].keys()))

    original_SD['metadata']['tracks_to_predict'][adv_track_name] = {
        'difficulty': 0,
        'object_type': ADV_type, 
        'track_id': adv_track_name,
        'track_index': tracks_length - 1
    }

    original_SD['id'] = original_SD['id'] + f"_SCGen_{success_mode}"
    if "id" in original_SD['metadata']:
        original_SD['metadata']['id'] = original_SD['metadata']['id'] + f"_SCGen_{success_mode}"
    original_SD['metadata']['scenario_id'] = original_SD['metadata']['scenario_id'] + f"_SCGen_{success_mode}"
    # original_SD['metadata']['dataset'] = 'SCGen_waymo'
    # original_SD['metadata']['source_file'] = original_SD['metadata']['source_file'] + f"_SCGen_{success_mode}"

    return original_SD

@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="0202_midgpt.yaml")
def main(config):
    OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.DATA.TEST_DATA_DIR = SOURCE_DIR
    OmegaConf.set_struct(config, True)
 
    NUM_MODE = 1 # for now

    path = "../ckpt/last.ckpt"  # change to your checkpoint path
    model = utils.get_model(checkpoint_path=path).eval()
    tokenizer = model.model.tokenizer
    device = model.device

    test_dataset = InfgenDataset(config, "test", backward_prediction=True)
    success = 0
    
    times = []
    
    for data_count, data_dict in enumerate(tqdm.tqdm(test_dataset, desc="Scenario")):
        start_time = timer.time()
        sid = data_dict["metadata/scenario_id"]
        print("current Scenario ID: ", sid)


        ego_traj = data_dict["decoder/agent_position"][:, 0, :2] # (B, 91, 2)
        ego_dist = np.linalg.norm(ego_traj[-1, :] - ego_traj[0, :])

        # vis_data_dict = {
        #     k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        #     for k, v in data_dict.items()
        # }
        # plot_gt(vis_data_dict, save_path=f"vis_scgen_diverse_training_500_TF/{sid}_GT.png") # ooi=[self.adv_index]

        mode_count = 0
        mode_success_count = 0

        while True:
            if mode_success_count >= 30:
                break

            # if mode_count >= 1:
            #     break

            mode_count += 1

            cur_data_dict = copy.deepcopy(data_dict)

            COL_STEP = 10
            COL_HEADING = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1) # np.random.uniform(-np.pi, np.pi)
            
            # step 1: backward as normal with a new agent
            cur_data_dict, adv_id = create_new_adv(cur_data_dict, col_heading=COL_HEADING, col_step=COL_STEP, ADV_type=MetaDriveType.VEHICLE)

            input_data_dict = utils.numpy_to_torch(cur_data_dict, device)
            input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)

            # Extend the batch dim:
            input_data_dict = {
                k: utils.expand_for_modes(v.unsqueeze(0), num_modes=NUM_MODE) if isinstance(v, torch.Tensor) else v
                for k, v in input_data_dict.items()
            }

            # Force to run backward prediction first to make sure the data is tokenized correctly.
            tok_data_dict, _ = tokenizer.tokenize(input_data_dict, backward_prediction=True)
            input_data_dict.update(tok_data_dict)

            input_data_dict["in_backward_prediction"] = torch.tensor([True] * NUM_MODE, dtype=bool).to(model.device)

            all_agents = cur_data_dict["decoder/agent_id"]
            all_agents_except_sdc = all_agents[all_agents != 0]
            not_teacher_forcing_ids = [adv_id] # all_agents_except_sdc # [adv_id]

            # since currently teacher forcing only supports B=1, we could only run one scenario at a time
            cur_input_data_dict = copy.deepcopy(input_data_dict)

            with torch.no_grad():
                ar_func = model.model.autoregressive_rollout_backward_prediction_with_replay
                backward_output_dict = ar_func(
                    cur_input_data_dict,
                    # num_decode_steps=None,
                    sampling_method=config.SAMPLING.SAMPLING_METHOD,
                    temperature=config.SAMPLING.TEMPERATURE,
                    topp=config.SAMPLING.TOPP,
                    not_teacher_forcing_ids=not_teacher_forcing_ids,
                )

            backward_output_dict = tokenizer.detokenize(
                backward_output_dict,
                detokenizing_gt=False,
                backward_prediction=True,
                flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING,
                teacher_forcing=True # FIXME: is this wrong????
            )

            vis_backward_output = {
                k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in backward_output_dict.items()
            }

            vis_backward_output = _get_mode(vis_backward_output, 0, num_modes=NUM_MODE)
            # collision_steps = detect_collision_steps(vis_backward_output, sdc_id=0, adv_id=adv_id)

            # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.
            curvature_threshold = 0.8
            # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.
            adv_traj = backward_output_dict["decoder/reconstructed_position"][:,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][:,:,-1]] # (T_valid, 2)
            displacements = torch.norm(torch.diff(adv_traj, dim=0), dim=1) + 1e-6

            heading = backward_output_dict["decoder/reconstructed_heading"][:,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][:,:,-1]] 
            heading_diffs = torch.abs(torch.diff(heading))
            heading_diffs = torch.minimum(heading_diffs, 2*torch.pi - heading_diffs)
            curvatures = heading_diffs / displacements

            adv_dist = torch.linalg.norm(adv_traj[-1, :] - adv_traj[0, :], dim=-1)
            adv_dist = adv_dist.mean()  # Shape (1,)

            coll_step = COL_STEP

            if adv_dist < coll_step/10 or torch.max(curvatures).item() > curvature_threshold:
                # print("failed cases:", sid)
                # plot_pred(vis_backward_output, save_path=f"vis_scgen_diverse_backward/{sid}_SCGEN_col_{coll_step}_failed_{mode_count}_backward_out.png")
                # print("saving to...", f"{sid}_SCGEN_failed_{mode_count}_backward_out.png")
                continue

            else:
                # step 2: trim the trajectory that as soon as there is collision, stops.
                # plot_pred(vis_backward_output, save_path=f"{SAVE_DIR}/sd_{sid}_SCGEN_backward_Coll_step_{COL_STEP}_heading_{COL_HEADING}_{mode_success_count}.png") # ooi=[self.adv_index]
                pass

            # mode_success_count += 1 # just for testing
            # continue 

            # step 2: try to overwrite original SD and save into local
            global_output_dict_mode = transform_to_global_coordinate(data_dict=vis_backward_output, from_GT=False)
            original_SD = data_dict["original_SD"]
            new_original_SD = overwrite_to_scenario_description_new_agent(output_dict_mode=global_output_dict_mode, original_SD=copy.deepcopy(original_SD), adv_id=adv_id, from_GT=False, ADV_type='VEHICLE', ooi=[adv_id], success_mode=mode_success_count)


            import pickle 
            with open(f"{SAVE_DIR}/sd_{sid}_SCGEN_backward_Coll_step_{COL_STEP}_heading_{COL_HEADING}_{mode_success_count}.pkl", "wb") as f:
                pickle.dump(new_original_SD, f)
            f.close()

            success += 1
            mode_success_count += 1


    return
    


if __name__ == '__main__':
    main()