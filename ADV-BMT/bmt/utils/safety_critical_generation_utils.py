import PIL
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle, Rectangle

from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch
import copy
import pdb
import pathlib

def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret

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
    # print("Ego position: ", ego_traj[last_valid_step])
    # print("Adv position: ", adv_traj[last_valid_step])
    # ====================

    # ===== Heading =====
    """
    On-going: can we use modified road heading as the new adv heading?
    """
    ego_last_heading = ego_heading[last_valid_step]

    # we can try different headings and see which one is more realistic

    adv_heading[last_valid_step] = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    # + np.random.normal(loc=0.0, scale=0.5, size=1)

    # print("Ego heading: ", np.rad2deg(ego_heading[last_valid_step]))
    # print("Adv heading: ", np.rad2deg(adv_heading[last_valid_step]))
    # ===================

    vel_step = 10
    # ===== Velocity =====
    adv_vel = 0.5 * (ego_velocity[vel_step] + np.random.normal(loc=0.0, scale=0.5, size=2))
    ego_vel = 0.5 * (ego_velocity[last_valid_step] + np.random.normal(loc=0.0, scale=0.5, size=2))
    # print("Ego velocity: ", ego_vel, ego_velocity[last_valid_step])
    # print("Adv velocity: ", adv_velocity[last_valid_step])
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

def _overwrite_datadict_all_agents(source_data_dict, dest_data_dict, ooi=None, T_end=91):
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
