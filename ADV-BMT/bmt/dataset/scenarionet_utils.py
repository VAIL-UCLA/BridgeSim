import numpy as np
import torch
from bmt.utils import wrap_to_pi, rotate
import copy


def overwrite_gt_to_pred_field(data_dict):
    new_data_dict = copy.deepcopy(data_dict)
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert T == 91

    new_data_dict["decoder/reconstructed_position"] = np.zeros((91, N, 2)).astype(np.float32)
    new_data_dict["decoder/reconstructed_valid_mask"] = np.zeros((
        T,
        N,
    )).astype(bool)
    new_data_dict["decoder/reconstructed_heading"] = np.zeros((
        T,
        N,
    )).astype(np.float32)
    new_data_dict["decoder/reconstructed_velocity"] = np.zeros((T, N, 2)).astype(np.float32)

    for id in range(N):  # overwrite all agents
        traj = new_data_dict["decoder/agent_position"][:91, id, :2].astype(np.float32)
        traj_mask = new_data_dict["decoder/agent_valid_mask"][:91, id].astype(bool)
        theta = new_data_dict['decoder/agent_heading'][:91, id].astype(np.float32)
        vel = new_data_dict['decoder/agent_velocity'][:91, id].astype(np.float32)

        new_data_dict["decoder/reconstructed_position"][:91, id, :2] = traj
        # new_data_dict["decoder/reconstructed_position"][:91, id, 2] = 0.0
        new_data_dict["decoder/reconstructed_valid_mask"][:91, id] = traj_mask
        new_data_dict["decoder/reconstructed_heading"][:91, id] = theta
        new_data_dict["decoder/reconstructed_velocity"][:91, id] = vel

    return new_data_dict


def overwrite_to_scenario_description(output_dict_mode, original_SD, ooi=None, adv_id=None, add_offset=False):
    # overwrite original SD with all predicted ooi trajectories included
    if not ooi:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    sdc_track_name = original_SD['metadata']['sdc_id']
    if adv_id is not None:
        adv_track_name = str(output_dict_mode['decoder/track_name'][int(adv_id)].item())

    offset = original_SD['tracks'][sdc_track_name]['state']['position'][0,:2]

    for id in ooi:
        track_name = output_dict_mode['decoder/track_name']
        agent_track_name = str(output_dict_mode['decoder/track_name'][int(id)].item())
            
        agent_traj = output_dict_mode["decoder/agent_position"][:, id, ]
        agent_heading = output_dict_mode["decoder/agent_heading"][:, id]
        agent_vel = output_dict_mode["decoder/agent_velocity"][:, id]
        agent_traj_mask = output_dict_mode["decoder/agent_valid_mask"][:, id]

        if add_offset:
            agent_traj = agent_traj[:, :2] + offset

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


def overwrite_to_scenario_description_new_agent(output_dict_mode, original_SD, ooi=None):
    # overwrite original SD with all predicted ooi trajectories included
    ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    adv_track_name = 'new_adv_agent'
    original_SD['tracks'][adv_track_name] = {'state': {}, 'type': 'VEHICLE', 'metadata': {}}
    sdc_track_name = original_SD['metadata']['sdc_id']

    for id in ooi:
        if id == ooi[-1]:
            agent_track_name = 'new_adv_agent'
        else:
            agent_track_name = str(output_dict_mode['decoder/track_name'][id].item())

        # begin to overwrite original scenario_data
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

        length = original_SD['tracks'][sdc_track_name]['state']['length'][10]
        width = original_SD['tracks'][sdc_track_name]['state']['width'][10]
        height = original_SD['tracks'][sdc_track_name]['state']['height'][10]
        original_SD['tracks'][agent_track_name]['state']['length'] = np.full((91, ), length)
        original_SD['tracks'][agent_track_name]['state']['width'] = np.full((91, ), width)
        original_SD['tracks'][agent_track_name]['state']['height'] = np.full((91, ), height)

    original_SD['tracks'][adv_track_name]['metadata']['dataset'] = 'waymo'
    original_SD['tracks'][adv_track_name]['metadata']['object_id'] = 'new_adv_agent'
    original_SD['tracks'][adv_track_name]['metadata']['track_length'] = 91
    original_SD['tracks'][adv_track_name]['metadata']['type'] = 'VEHICLE'
    original_SD['metadata']['new_adv_id'] = 'new_adv_agent'
    original_SD['metadata']['objects_of_interest'].append('new_adv_agent')
    tracks_length = len(list(original_SD['tracks'].keys()))
    original_SD['metadata']['tracks_to_predict']['new_adv_agent'] = {
        'difficulty': 0,
        'object_type': 'VEHICLE',
        'track_id': 'new_adv_agent',
        'track_index': tracks_length - 1
    }

    return original_SD


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    assert "decoder/agent_position" in data_dict, "Have you set EVALUATION.PREDICT_ALL_AGENTS to False?"
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert data_dict["decoder/agent_position"].ndim == 3

    expanded_mask = data_dict["decoder/agent_valid_mask"][:, :, None]
    data_dict["decoder/agent_position"] += map_center * expanded_mask

    return data_dict


def _overwrite_datadict_all_agents(source_data_dict, dest_data_dict, ooi=None):
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


def merge_preds_along_mode_dim(pred_dicts):
    for k in ["decoder/reconstructed_position", "decoder/reconstructed_valid_mask", "decoder/reconstructed_heading", "decoder/reconstructed_velocity"]:
        pred_dicts[0][k] = torch.cat([pred_dicts[i][k] for i in range(len(pred_dicts))], dim=0)
    return pred_dicts[0]