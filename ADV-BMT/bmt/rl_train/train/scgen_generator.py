from omegaconf import OmegaConf
import hydra
from bmt.utils import utils
import omegaconf
from bmt.utils import REPO_ROOT
import easydict
import copy
from bmt.safety_critical_scenario_generation import set_adv
from collections import deque
import numpy as np
from bmt.utils.utils import numpy_to_torch
import torch
import subprocess
import pickle
import os
import copy
from bmt.dataset.scenarionet_utils import overwrite_to_scenario_description
from bmt.gradio_ui.plot import plot_pred, plot_gt

def overwrite_to_scenario_description_new_agent(output_dict_mode, original_SD, adv_id, from_GT, ooi=None):
    """
    Overwrite from GT field in data_dict to tracks in SD for OOI agents.
    Note that here we assume trajectory starts from T_start=0 to T_end=90, 
    """

    if ooi is None:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    # adv_track_name = 'new_adv_agent'
    adv_track_name = '99999'
    original_SD['tracks'][adv_track_name] = {'state': {}, 'type': 'VEHICLE', 'metadata': {}}
    sdc_track_name = original_SD['metadata']['sdc_id']

    for id in ooi:
        if id == adv_id:
            # agent_track_name = 'new_adv_agent'
            agent_track_name = adv_track_name
        else:
            agent_track_name = output_dict_mode['decoder/track_name'][id]
            assert isinstance(output_dict_mode['decoder/track_name'][id], str)

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
        original_SD['tracks'][agent_track_name]['state']['position'] = agent_traj # FIXME: add z-axis????
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
    original_SD['tracks'][adv_track_name]['metadata']['object_id'] = adv_track_name
    original_SD['tracks'][adv_track_name]['metadata']['track_length'] = 91
    original_SD['tracks'][adv_track_name]['metadata']['type'] = 'VEHICLE'
    original_SD['metadata']['new_adv_id'] = adv_track_name
    original_SD['metadata']['objects_of_interest'].append(adv_track_name)
    tracks_length = len(list(original_SD['tracks'].keys()))

    original_SD['metadata']['tracks_to_predict'][adv_track_name] = {
        'difficulty': 0,
        'object_type': 'VEHICLE', 
        'track_id': adv_track_name,
        'track_index': tracks_length - 1
    }

    # original_SD['id'] = original_SD['id'] + f"_SCGen"
    # if "id" in original_SD['metadata']:
    #     original_SD['metadata']['id'] = original_SD['metadata']['id'] + f"_SCGen"
    # original_SD['metadata']['scenario_id'] = original_SD['metadata']['scenario_id'] + f"_SCGen"
    # original_SD['metadata']['dataset'] = 'SCGen_waymo'
    # original_SD['metadata']['source_file'] = original_SD['metadata']['source_file'] + f"_SCGen"

    return original_SD


def create_new_adv_deprecated(data_dict, col_heading=None, col_step=None):

    raise NotImplementedError("This function is deprecated, please use create_batched_new_adv instead.")

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

    # ===== Position =====
    adv_traj[last_valid_step] = ego_last_pos 
    # ====================

    # ===== Heading =====
    """
    On-going: can we use modified road heading as the new adv heading?
    """

    if col_heading is None:
        adv_heading[last_valid_step] = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    else:
        adv_heading[last_valid_step] = col_heading
    # ===================

    # ===== Velocity =====
    adv_speed = np.linalg.norm(ego_velocity[last_valid_step]) * 0.5 + np.random.normal(loc=0.0, scale=0.5)
    new_heading = adv_heading[last_valid_step]
    adv_velocity[last_valid_step] = adv_speed * np.array([np.cos(new_heading), np.sin(new_heading)])     # Project speed in heading direction
 
    # ego_speed = np.linalg.norm(ego_velocity[last_valid_step]) * 0.5 + np.random.normal(loc=0.0, scale=0.5)
    # ego_heading_last = ego_heading[last_valid_step]
    # ego_velocity[last_valid_step] = ego_speed * np.array([np.cos(ego_heading_last), np.sin(ego_heading_last)])
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

    return data_dict, len(data_dict["decoder/agent_id"]) - 1, last_valid_step



def batch_data(data_dict):
    """Add one additional dimension to all values in the dictionary."""
    ret = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            ret[k] = v[None]
        elif isinstance(v, torch.Tensor):
            ret[k] = v[None]
        else:
            ret[k] = v
    return ret



def sample_batched_col_step(last_valid_step, min_step=10):
    # last_valid_step: [B], each ≥ min_step
    import random
    col_step = [
        random.choice(list(range(((min_step // 5) + 1) * 5, step + 1, 5)))
        for step in last_valid_step.tolist()
    ]
    return torch.tensor(col_step, device=last_valid_step.device)


def create_new_adv_batched(batched_data_dict, diverse_col_step=False, collision_half_speed=False):
    B, T, N, _ = batched_data_dict["decoder/agent_position"].shape  # Batch, Time, Num agents

    ego_id = batched_data_dict["decoder/sdc_index"]  # shape: [B]
    device = ego_id.device

    batch_idx = torch.arange(B, device=device)

    # Gather ego info per batch
    ego_pos = batched_data_dict["decoder/agent_position"][batch_idx, :, ego_id]  # [B, T, 2]
    ego_heading = batched_data_dict["decoder/agent_heading"][batch_idx, :, ego_id]  # [B, T]
    ego_velocity = batched_data_dict["decoder/agent_velocity"][batch_idx, :, ego_id]  # [B, T, 2]
    ego_shape = batched_data_dict["decoder/agent_shape"][batch_idx, :, ego_id]  # [B, T, 2]
    ego_mask = batched_data_dict["decoder/agent_valid_mask"][batch_idx, :, ego_id]  # [B, T]


    ego_last_valid_step = ego_mask.float().cumsum(dim=1).argmax(dim=1)  # [B]
    if diverse_col_step:
        min_coll_step = 10
        adv_last_valid_step = sample_batched_col_step(ego_last_valid_step, min_step=min_coll_step)

    else:
        # adv_last_valid_step = torch.full((B,), ego_last_valid_step, dtype=torch.long, device=device) # always collid at the last valid step
        adv_last_valid_step = ego_last_valid_step.clone()  # copy the last valid step


    adv_mask = torch.zeros_like(ego_mask) # init
    adv_traj = torch.zeros_like(ego_pos)
    adv_heading = torch.zeros_like(ego_heading)
    adv_velocity = torch.zeros_like(ego_velocity)
    adv_shape = torch.zeros_like(ego_shape)

    for b in range(B):    # Set mask up to collision step
        adv_mask[b, :adv_last_valid_step[b] + 1] = 1

    ego_last_pos = ego_pos[batch_idx, ego_last_valid_step]  # [B, 2]
    adv_traj[batch_idx, ego_last_valid_step] = ego_last_pos

    adv_heading_value = torch.randn(B, device=device) * 2 * np.pi # randomly initialize ADV's collision headings
    # adv_heading_value = torch.remainder(adv_heading_value, 2 * np.pi) # do we need to wrap it to [0, 2pi]?

    adv_heading[batch_idx, adv_last_valid_step] = adv_heading_value

    if collision_half_speed:
        ego_speed = torch.norm(ego_velocity[batch_idx, adv_last_valid_step], dim=-1) * 0.5 + torch.randn(B, device=device) * 0.5
        cos_h = torch.cos(adv_heading_value)
        sin_h = torch.sin(adv_heading_value)
        adv_velocity_proj = torch.stack([ego_speed * cos_h, ego_speed * sin_h], dim=-1)  # [B, 2]
        adv_velocity[batch_idx, adv_last_valid_step] = adv_velocity_proj

    else:
        adv_speed = torch.norm(ego_velocity[batch_idx, adv_last_valid_step], dim=-1) + torch.randn(B, device=device) * 0.5
        cos_h = torch.cos(adv_heading_value)
        sin_h = torch.sin(adv_heading_value)
        adv_velocity_proj = torch.stack([adv_speed * cos_h, adv_speed * sin_h], dim=-1)

    # Shape: copy shape at collision step
    adv_shape[:, :, :] = ego_shape[batch_idx, adv_last_valid_step].unsqueeze(1).expand(-1, T, -1)

    def append_agent(tensor, new_agent):
        return torch.cat([tensor, new_agent.unsqueeze(2)], dim=2)  # append along agent dimension

    batched_data_dict["decoder/agent_position"] = append_agent(batched_data_dict["decoder/agent_position"], adv_traj)
    batched_data_dict["decoder/agent_heading"] = append_agent(batched_data_dict["decoder/agent_heading"], adv_heading)
    batched_data_dict["decoder/agent_velocity"] = append_agent(batched_data_dict["decoder/agent_velocity"], adv_velocity)
    batched_data_dict["decoder/agent_shape"] = append_agent(batched_data_dict["decoder/agent_shape"], adv_shape)
    batched_data_dict["decoder/agent_valid_mask"] = append_agent(batched_data_dict["decoder/agent_valid_mask"], adv_mask)

    # ===========
    adv_id = batched_data_dict["decoder/agent_position"].shape[2] - 1  # new index for ADV

    def append_field(field, value):
        # Ensure value shape is [B, 1, ...] if field is [B, N, ...]
        if value.ndim == field.ndim - 1:
            value = value.unsqueeze(1)
        return torch.cat([field, value], dim=1)

    # Correct shapes: all will now be [B, N+1, ...]
    batched_data_dict["decoder/current_agent_shape"] = append_field(
        batched_data_dict["decoder/current_agent_shape"],
        batched_data_dict["decoder/current_agent_shape"][batch_idx, ego_id]
    )

    batched_data_dict["decoder/current_agent_position"] = append_field(
        batched_data_dict["decoder/current_agent_position"],
        batched_data_dict["decoder/current_agent_position"][batch_idx, ego_id]
    )

    batched_data_dict["decoder/agent_type"] = append_field( # TODO: can vary for different agent types!!
        batched_data_dict["decoder/agent_type"],
        batched_data_dict["decoder/agent_type"][batch_idx, ego_id]
    )

    batched_data_dict["decoder/agent_id"] = append_field(
        batched_data_dict["decoder/agent_id"],
        torch.full((B, 1), adv_id, dtype=torch.long, device=device)
    )

    batched_data_dict["decoder/object_of_interest_id"] = append_field(
        batched_data_dict["decoder/object_of_interest_id"],
        torch.full((B, 1), adv_id, dtype=torch.long, device=device)
    )

    batched_data_dict["decoder/current_agent_valid_mask"] = append_field(
        batched_data_dict["decoder/current_agent_valid_mask"],
        torch.ones((B, 1), dtype=torch.long, device=device)
    )

    print(f"====================================")
    print(f"The new ADV is created at steps {adv_last_valid_step.tolist()}, ID: {adv_id}")
    print(f"====================================")

    return batched_data_dict, adv_id, adv_last_valid_step



# def transform_to_global_coordinate(data_dict):
#     map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
#     map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
#     assert (map_heading == 0).all()

#     expanded_mask = data_dict["decoder/reconstructed_valid_mask"][:, :, None]
#     data_dict["decoder/reconstructed_position"] += map_center[:, :, :2] * expanded_mask
#     return data_dict


def transform_to_global_coordinate(data_dict):
    new_data_dict = copy.deepcopy(data_dict)
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    assert "decoder/agent_position" in data_dict, "Have you set EVALUATION.PREDICT_ALL_AGENTS to False?"
    T, N, _ = data_dict["decoder/agent_position"].shape
    assert data_dict["decoder/agent_position"].ndim == 3

    expanded_mask = data_dict["decoder/reconstructed_valid_mask"][:, :, None]
    new_data_dict["decoder/reconstructed_position"] += map_center[:,:,:2] * expanded_mask

    return new_data_dict


def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


def _to_dict(d):
    if isinstance(d, easydict.EasyDict):
        return {k: _to_dict(v) for k, v in d.items()}
    return d


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



def _expand_batch(single_batch_data_dict, batch_size):
    return {
        k: v.repeat(batch_size, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor)
        else np.repeat(v, batch_size, axis=0) if isinstance(v, np.ndarray)
        else [copy.deepcopy(v[0]) for _ in range(batch_size)] if isinstance(v, list) and len(v) == 1
        else v
        for k, v in single_batch_data_dict.items()
    }


def is_sdc_parking(scenario_description):
    sdc_id = scenario_description["metadata"]["sdc_id"]
    ego_traj = scenario_description["tracks"][sdc_id]["state"]["position"][scenario_description["tracks"][sdc_id]["state"]["valid"]]
    ego_dist = np.linalg.norm(ego_traj[-1, :2] - ego_traj[0, :2])
    if ego_dist < 10: # skippping parking scene
        return False

class SCGEN_Generator:
    def __init__(self, model_name='0202_midgpt', TF_mode="all_TF_except_adv", ckpt_path="bmt/ckpt/last.ckpt"):

        from hydra import initialize_config_dir, compose
        from bmt.utils import REPO_ROOT

        # if not model_name.endswith(".yaml"):
        #     model_name += ".yaml"

        # config_path = REPO_ROOT / "cfgs"
        # with initialize_config_dir(config_dir=str(config_path), version_base=None):
        #     config = compose(config_name=model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert torch.cuda.is_available(), "CUDA is not available, please check your environment."

        pl_model = utils.get_model(checkpoint_path=ckpt_path).eval()

        config = pl_model.config
        config.PREPROCESSING.keep_all_data = True
        # Set the maximum number of agents, so we can avoid making prediction for those static agents, thus saving GPU.
        config.PREPROCESSING.MAX_AGENTS = 64

        from bmt.tokenization import get_tokenizer
        # tokenizer = get_tokenizer(config)
        tokenizer = pl_model.model.tokenizer

        self.config = config
        self.tokenizer = tokenizer
        self.pl_model = pl_model

        self.storage = {}
        self.cur_adv_agent = None

        
        self.storage = {}

        self.num_modes = 8 # for now one mode as CAT
        self.adv_id = None
        self.ego_traj = []
        self.ego_vel = []
        self.ego_heading = []
        self.current_scenario_id = None

        self.no_adaptive = False # FIXME: will be addeed

    def set_no_adaptive(self, no_adaptive):
        self.no_adaptive = no_adaptive
    


    def GPT_backwardAR(self, backward_input_dict, not_tf_ids=None):
        assert not_tf_ids is not None
        with torch.no_grad():
            ar_func = self.pl_model.model.autoregressive_rollout_backward_prediction_with_replay
            backward_output_dict = ar_func(
                backward_input_dict,
                # num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=self.config.SAMPLING.TEMPERATURE,
                topp=self.config.SAMPLING.TOPP,
                not_teacher_forcing_ids=not_tf_ids,
            )

        backward_output_dict = self.tokenizer.detokenize(
            backward_output_dict,
            detokenizing_gt=False,
            backward_prediction=True,
            flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
            teacher_forcing=True
        )

        return backward_output_dict



    def before_episode(self, env=None, scenario_data=None):

        if env is not None:
            self.env = env
            sid = self.env.engine.data_manager.current_scenario["id"]
        else:
            assert scenario_data is not None
            sid = scenario_data["id"]

        if sid not in self.storage:
            # from scenario_net data to infgen data_dict

            if scenario_data is None:
                assert env is not None
                scenario_data = self.env.engine.data_manager.current_scenario

            sdc_id = scenario_data['metadata']['sdc_id']
            ego_pos = scenario_data['tracks'][sdc_id]['state']['position'][:, :2]
            ego_heading = scenario_data['tracks'][sdc_id]['state']['heading']
            ego_vel = scenario_data['tracks'][sdc_id]['state']['velocity'][:, :2]

            sdc_gt_info = {"ego_traj": ego_pos, "ego_heading": ego_heading, "ego_vel": ego_vel}

            sdc_traj = sdc_gt_info["ego_traj"]
            sdc_heading = sdc_gt_info["ego_heading"]
            sdc_vel = sdc_gt_info["ego_vel"]

            self.storage[sid] = dict(
                SDC_traj=sdc_traj,
                SDC_heading=sdc_heading,
                SDC_vel=sdc_vel,
                sdc_initial_pos=sdc_traj[0].copy(),  # for later use
            )

            # self.ego_traj = [sdc_traj[0]] # future fix
            # self.ego_vel = [sdc_vel[0]]
            # self.ego_heading = [sdc_heading[0]]


    def after_episode(self):
        latest_ego_traj = np.array(self.ego_traj)  # now we have the whole new traj
        latest_ego_heading = np.array(self.ego_heading)
        latest_ego_vel = np.array(self.ego_vel)

        if len(latest_ego_traj) <= 10:
            print('Ignore traj less than 1s')  # abandon bad policy
            return

        sid = self.env.engine.data_manager.current_scenario["id"]
        # print("in after_episode, sid:", sid)

        self.storage[sid]['SDC_traj'] = latest_ego_traj
        self.storage[sid]['SDC_heading'] = latest_ego_heading
        self.storage[sid]['SDC_vel'] = latest_ego_vel


    def log_ego_history(self):
        obj = self.env.engine.current_track_agent

        self.ego_traj.append(obj.position)
        self.ego_vel.append(obj.velocity)
        self.ego_heading.append(obj.heading_theta)




    def reject_multi_batch(self, backward_output_dict, NUM_MODE):
        """
        return the first valid batch/traj
        """
        # vis_backward_output = {
        #     k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        #     for k, v in backward_output_dict.items()
        # }

        # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.
        curvature_threshold = 0.8

        for i in range(NUM_MODE):
            # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.

            adv_traj = backward_output_dict["decoder/reconstructed_position"][i,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][i,:,-1]] # (T_valid, 2)
            displacements = torch.norm(torch.diff(adv_traj, dim=0), dim=1) + 1e-6

            heading = backward_output_dict["decoder/reconstructed_heading"][i,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][i,:,-1]] 
            heading_diffs = torch.abs(torch.diff(heading))
            heading_diffs = torch.minimum(heading_diffs, 2*torch.pi - heading_diffs)
            curvatures = heading_diffs / displacements

            step_masks = backward_output_dict["decoder/reconstructed_valid_mask"][i,:,-1]
            last_valid_step = step_masks.nonzero(as_tuple=True)[0][-1].item()  # last valid step

            adv_dist = torch.linalg.norm(adv_traj[-1, :] - adv_traj[0, :], dim=-1)
            adv_dist = adv_dist.mean()  # Shape (1,)

            if adv_dist < last_valid_step/10 or torch.max(curvatures).item() >= curvature_threshold:
                # print("failed cases:", sid)
                # plot_pred(vis_backward_output, save_path=f"vis_scgen_diverse_backward/{sid}_SCGEN_col_{coll_step}_failed_{mode_count}_backward_out.png")
                # print("saving to...", f"{sid}_SCGEN_failed_{mode_count}_backward_out.png")
                continue

            else:
                return i, torch.max(curvatures).item()

        return NUM_MODE-1, torch.max(curvatures).item()


        
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
            


        from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
        original_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=overwritten_sd,
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            backward_prediction=True,
            tokenizer=self.pl_model.model.tokenizer
        )

        data_dict = copy.deepcopy(original_data_dict)
        batched_data_dict = batch_data(utils.numpy_to_torch(data_dict, device=self.pl_model.device))
        batched_data_dict = _expand_batch(batched_data_dict, batch_size=self.num_modes)


        batched_data_dict, adv_id, last_valid_step = create_new_adv_batched(batched_data_dict, diverse_col_step=False, collision_half_speed=True)
        batched_data_dict = {k: v.to(self.pl_model.device) if isinstance(v, torch.Tensor) else v for k, v in batched_data_dict.items()}


        # Force to run backward prediction first to make sure the data is tokenized correctly.
        tok_data_dict, _ = self.tokenizer.tokenize(batched_data_dict, backward_prediction=True)
        batched_data_dict.update(tok_data_dict)
        batched_data_dict["in_backward_prediction"] = torch.tensor([True] * self.num_modes, dtype=bool).to(self.pl_model.device)

        if batched_data_dict is None:
            print(f"Warning: No ADV is generated in {sid} !!!!!")
            return None

        self.adv_id = adv_id

        all_agents = batched_data_dict["decoder/agent_id"][0]
        not_tf_ids = torch.tensor([adv_id]).to(self.pl_model.device)
        output_data = self.GPT_backwardAR(batched_data_dict, not_tf_ids=not_tf_ids)
        
        # step 2: reject the batch if the adv is not moving
        best_mode_index, max_curvature = self.reject_multi_batch(output_data, self.num_modes)

        output_data_all_modes = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_data.items()
        }
        output_dict_mode = _get_mode(output_data_all_modes, best_mode_index, num_modes=self.num_modes)
        
        output_dict_mode_pred = transform_to_global_coordinate(output_dict_mode) # transform back to global coordinate, since we will overwrite the original SD

        # new_SD = overwrite_to_scenario_description(output_dict_mode_pred, copy.deepcopy(new_SD), adv_id=adv_id, ooi=[adv_id],add_offset=False) # overwrite the original SD with the new predicted agent positions
        new_SD = overwrite_to_scenario_description_new_agent(output_dict_mode=output_dict_mode_pred, original_SD=copy.deepcopy(scenario_data), adv_id=adv_id, from_GT=False, ooi=[adv_id])

        return new_SD # return modified scenario description


    def generate_from_raw_SD(self, scenario_data=None, track_length=91):

        assert scenario_data is not None
        sid = scenario_data["id"]
        
        from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm

        if is_sdc_parking(scenario_data):
            return None
        
        original_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=copy.deepcopy(scenario_data),
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            backward_prediction=True,
            tokenizer=self.pl_model.model.tokenizer
        )

        

        data_dict = copy.deepcopy(original_data_dict)
        batched_data_dict = batch_data(utils.numpy_to_torch(data_dict, device=self.pl_model.device))
        batched_data_dict = _expand_batch(batched_data_dict, batch_size=self.num_modes)


        batched_data_dict, adv_id, last_valid_step = create_new_adv_batched(batched_data_dict, diverse_col_step=False, collision_half_speed=True)
        batched_data_dict = {k: v.to(self.pl_model.device) if isinstance(v, torch.Tensor) else v for k, v in batched_data_dict.items()}


        # Force to run backward prediction first to make sure the data is tokenized correctly.
        tok_data_dict, _ = self.tokenizer.tokenize(batched_data_dict, backward_prediction=True)
        batched_data_dict.update(tok_data_dict)
        batched_data_dict["in_backward_prediction"] = torch.tensor([True] * self.num_modes, dtype=bool).to(self.pl_model.device)

        if batched_data_dict is None:
            print(f"Warning: No ADV is generated in {sid} !!!!!")
            return None

        all_agents = batched_data_dict["decoder/agent_id"][0]
        not_tf_ids = torch.tensor([adv_id]).to(self.pl_model.device)
        output_data = self.GPT_backwardAR(batched_data_dict, not_tf_ids=not_tf_ids)
        
        # step 2: reject the batch if the adv is not moving
        best_mode_index, max_curvature = self.reject_multi_batch(output_data, self.num_modes)

        output_data_all_modes = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_data.items()
        }
        output_dict_mode = _get_mode(output_data_all_modes, best_mode_index, num_modes=self.num_modes)
        
        output_dict_mode_pred = transform_to_global_coordinate(output_dict_mode) # transform back to global coordinate, since we will overwrite the original SD

        new_SD = overwrite_to_scenario_description_new_agent(output_dict_mode=output_dict_mode_pred, original_SD=copy.deepcopy(scenario_data), adv_id=adv_id, from_GT=False, ooi=[adv_id])

        return new_SD # return modified scenario description
        
        


    @property
    def adv_agent(self):
        return self.storage[self.current_scenario_id].get('adv_agent')
    @property
    def adv_traj(self):
        return self.storage[self.current_scenario_id].get('adv_traj')
    @property
    def adv_heading(self):
        return self.storage[self.current_scenario_id].get('adv_heading')
    @property
    def adv_vel(self):
        return self.storage[self.current_scenario_id].get('adv_velocity')
    


if __name__ == "__main__":
    from rl_train.train.ScenarioOnlineEnvWrapper import ScenarioOnlineEnvWrapper
	# for testing
    generator = SCGEN_Generator()
    
    from metadrive.policy.env_input_policy import EnvInputPolicy
    env = ScenarioOnlineEnvWrapper(
        generator,
        config=dict(
            use_render=False,
            manual_control=False,
            show_interface=False,
            data_directory="scenarionet_waymo_training_500/", 
            agent_policy=EnvInputPolicy,
            start_scenario_index=0,
            num_scenarios=400,
            sequential_seed=True,
            horizon=500,
        )
    )

    
    env.reset()

    generator.before_episode(env)

    for i in range(100): # just test for 10 steps
        o, r, tm, tc, info = env.step([0.0, 0.0])
        generator.log_AV_history()


    if tm or tc:
        generator.after_episode(update_AV_traj=True)


        












