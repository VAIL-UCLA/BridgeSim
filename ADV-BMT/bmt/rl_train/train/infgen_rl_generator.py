import copy

import numpy as np
import torch

from bmt import utils
from bmt.tokenization import get_tokenizer


def _convert_to_SD_types(data_dict_agent_type):
    type_map = {
        0: "UNSET",
        1: "VEHICLE",
        2: "PEDESTRIAN",
        3: "CYCLIST",
        4: "OTHER"
    }

    SD_type_name = type_map[data_dict_agent_type]

    return SD_type_name


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
    map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
    assert (map_heading == 0).all()

    expanded_mask = data_dict["decoder/reconstructed_valid_mask"][:, :, None]
    data_dict["decoder/reconstructed_position"] += map_center[:, :, :2] * expanded_mask
    return data_dict


def overwrite_to_scenario_description_new_agent(output_dict_mode, original_SD, ooi=None, type_convert_map=None,
                                                track_length=91):
    """
    Write all tracks in OOI to original_SD (discard all old tracks)
    """
    new_SD = copy.deepcopy(original_SD)
    if ooi is None:
        ooi = output_dict_mode['decoder/agent_id']  # overwrite all agents

    new_SD['metadata']['objects_of_interest'] = []
    new_SD['metadata']['tracks_to_predict'] = {}
    new_SD['tracks'] = {}

    sdc_track_name = str(output_dict_mode["decoder/track_name"][output_dict_mode["decoder/sdc_index"]])

    ego_traj = original_SD['tracks'][sdc_track_name]['state']['position'].copy()
    ego_traj[..., -1] = 0  # Reset Z axis to 0

    ego_avg_pos = ego_traj[original_SD['tracks'][sdc_track_name]['state']['valid']][..., :2].mean(0)
    all_avg_pos = output_dict_mode["decoder/reconstructed_position"][10, :].mean(axis=0)
    dist = np.linalg.norm(all_avg_pos - ego_avg_pos)
    assert dist < 500, f"Original SDC average position {ego_avg_pos} and new SDC average position {all_avg_pos} are not the same. Please check your code."
    print(f"Original SDC average position {ego_avg_pos} and all agents average position {all_avg_pos}")

    assert (output_dict_mode["decoder/sdc_index"] == 0)
    for id in ooi:

        if id == 0:
            new_SD['tracks'][sdc_track_name] = original_SD['tracks'][sdc_track_name]
            new_SD['tracks'][sdc_track_name]['state']['position'] = ego_traj
            new_SD['metadata']['objects_of_interest'].append(sdc_track_name)

            if sdc_track_name in original_SD['metadata']['tracks_to_predict']:
                sdc_tracks_to_predict = original_SD['metadata']['tracks_to_predict'][sdc_track_name]
                new_sdc_tracks_to_predict = {
                    'difficulty': sdc_tracks_to_predict['difficulty'],
                    'object_type': sdc_tracks_to_predict['object_type'],
                    'track_id': sdc_tracks_to_predict['track_id'],
                    'track_index': 0,
                }
                new_SD['metadata']['tracks_to_predict'][sdc_track_name] = new_sdc_tracks_to_predict

        else:
            new_agent_track_name = str(output_dict_mode["decoder/track_name"][id])

            agent_type = output_dict_mode['decoder/agent_type'][id]
            if type_convert_map is not None:
                new_agent_type = type_convert_map[agent_type]
            else:
                new_agent_type = _convert_to_SD_types(agent_type)

            new_SD['tracks'][new_agent_track_name] = {'state': {}, 'type': new_agent_type, 'metadata': {}}

            agent_traj = output_dict_mode["decoder/reconstructed_position"][:track_length, id, :2]
            agent_heading = output_dict_mode["decoder/reconstructed_heading"][:track_length, id]
            agent_vel = output_dict_mode["decoder/reconstructed_velocity"][:track_length, id]
            agent_traj_mask = output_dict_mode["decoder/reconstructed_valid_mask"][:track_length, id]

            if "decoder/reconstructed_shape" in output_dict_mode and output_dict_mode[
                "decoder/reconstructed_shape"] is not None:
                agent_length = output_dict_mode["decoder/reconstructed_shape"][:track_length, id, 0]
                agent_width = output_dict_mode["decoder/reconstructed_shape"][:track_length, id, 1]
                agent_height = output_dict_mode["decoder/reconstructed_shape"][:track_length, id, 2]

            else:
                length = float(output_dict_mode['decoder/agent_shape'][10, id, 0])
                width = float(output_dict_mode['decoder/agent_shape'][10, id, 1])
                height = float(output_dict_mode['decoder/agent_shape'][10, id, 2])

                agent_length = np.full((track_length,), length, dtype=float)
                agent_width = np.full((track_length,), width, dtype=float)
                agent_height = np.full((track_length,), height, dtype=float)

            agent_state = {
                'position': agent_traj,
                'velocity': agent_vel,
                'heading': agent_heading,
                'valid': agent_traj_mask,
                'length': agent_length,
                'width': agent_width,
                'height': agent_height,
            }

            new_track = {
                'state': agent_state,
                'metadata': {
                    'dataset': 'INFGEN',
                    'object_id': new_agent_track_name,
                    'track_length': track_length,
                    'type': new_agent_type
                },
                'type': new_agent_type
            }

            new_SD['tracks'][new_agent_track_name] = new_track
            new_SD['metadata']['objects_of_interest'].append(new_agent_track_name)

            new_SD['metadata']['tracks_to_predict'][new_agent_track_name] = {
                'difficulty': 0,
                'object_type': new_agent_type,
                'track_id': new_agent_track_name,
                'track_index': int(id),
            }

    # set new SID
    new_SD['metadata']['sdc_id'] = sdc_track_name
    new_SD['id'] = original_SD['id']
    if "id" in new_SD['metadata']:
        new_SD['metadata']['id'] = original_SD['metadata']['id']
    new_SD['metadata']['scenario_id'] = original_SD['metadata']['scenario_id']
    new_SD['metadata']['dataset'] = 'InfGen'

    return new_SD


def _recursive_check_type(obj, allow_types=(int, float, str, np.ndarray, dict, list, tuple, type(None), set), depth=0):
    # copy MD's sanity check here
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"checking key {k}")
            assert isinstance(k, str), "Must use string to be dict keys"
            _recursive_check_type(v, allow_types, depth=depth + 1)

    if isinstance(obj, list):
        for v in obj:
            _recursive_check_type(v, allow_types, depth=depth + 1)

    assert isinstance(obj, allow_types), "TypeError in key {}: Object type {} not allowed! ({})".format(obj, type(obj),
                                                                                                        allow_types)

    if depth > 1000:
        raise ValueError()


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


class InfgenRLScenarioGenerator:
    def __init__(self, model_name):

        from hydra import initialize_config_dir, compose
        from bmt.utils import REPO_ROOT

        if not model_name.endswith(".yaml"):
            model_name += ".yaml"
        # Load config with Hydra
        config_path = REPO_ROOT / "cfgs"
        with initialize_config_dir(config_dir=str(config_path), version_base=None):
            config = compose(config_name=model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert torch.cuda.is_available(), "CUDA is not available, please check your environment."
        pl_model = utils.get_model(config=config, device=device)

        config = pl_model.config
        config.PREPROCESSING.keep_all_data = True
        # Set the maximum number of agents, so we can avoid making prediction for those static agents, thus saving GPU.
        config.PREPROCESSING.MAX_AGENTS = 64

        model_name = model_name.replace(".yaml", "")
        assert model_name in ["infgen-full-large",
                              "infgen-base-large",
                              "infgen-full-large-nors"], "Model name not supported. Please use infgen-full-large or infgen-base-large."
        if model_name == "infgen-full-large":
            assert pl_model.config.EVALUATION.TG_REJECT_SAMPLING is True
            assert pl_model.config.EVALUATION.TG_SDC_DISTANCE_MASKING is False
        self.model_name = model_name

        tokenizer = get_tokenizer(config)

        self.config = config
        self.tokenizer = tokenizer
        self.pl_model = pl_model

        self.storage = {}
        self.cur_adv_agent = None

        self.num_modes = 8  # for now one mode as CAT
        self.adv_id = None
        self.ego_traj = []
        self.ego_vel = []
        self.ego_heading = []

        self.infgen_generator = None
        self.no_adaptive = False

    def set_no_adaptive(self, no_adaptive):
        self.no_adaptive = no_adaptive
        

    def GPT_AR(self, input_dict):
        if self.infgen_generator is None:
            from bmt.infer.infgen_generator import InfGenGenerator
            self.infgen_generator = InfGenGenerator(
                model=self.pl_model.model,
                device=self.pl_model.device,
            )
        with torch.no_grad():
            self.infgen_generator.reset(new_data_dict=input_dict)

            if self.model_name in ["infgen-full-large", "infgen-full-large-nors"]:
                output_dict = self.infgen_generator.generate_infgen_initial_state_and_motion(
                    progress_bar=False,
                    teacher_forcing_sdc=True,
                )
            elif self.model_name == "infgen-base-large":
                output_dict = self.infgen_generator.generate_infgen_motion(
                    progress_bar=False,
                    teacher_forcing_sdc=True,
                )
            else:
                raise ValueError("Model name not supported. Please use infgen-full-large or infgen-base-large.")

        return output_dict

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

    def log_ego_history(self):
        obj = self.env.engine.current_track_agent

        self.ego_traj.append(obj.position)
        # print("current pos:", obj.position)
        self.ego_vel.append(obj.velocity)
        self.ego_heading.append(obj.heading_theta)

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

        if self.no_adaptive: # for ablation
            overwritten_sd = copy.deepcopy(scenario_data)
        else:
            overwritten_sd = overwrite_new_sdc_traj_to_SD(copy.deepcopy(scenario_data), sdc_traj, sdc_heading, sdc_vel,
                                                        track_length=track_length)  # need to overwrite mask as well

        from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
        data_dict = preprocess_scenario_description_for_motionlm(
            scenario=overwritten_sd,
            config=self.config,
            in_evaluation=False,
            keep_all_data=True,
            tokenizer=self.pl_model.model.motion_tokenizer
        )

        batched_data_dict = utils.batch_data(utils.numpy_to_torch(data_dict, device=self.pl_model.device))
        output_data = self.GPT_AR(batched_data_dict)
        batched_data_dict.update(output_data)
        data_dict = utils.unbatch_data(utils.torch_to_numpy(batched_data_dict))

        global_output_dict = transform_to_global_coordinate(data_dict=data_dict)
        type_convert_map = {
            self.pl_model.model.veh_id: "VEHICLE",
            self.pl_model.model.ped_id: "PEDESTRIAN",
            self.pl_model.model.cyc_id: "CYCLIST",
        }
        new_SD = overwrite_to_scenario_description_new_agent(output_dict_mode=global_output_dict,
                                                             original_SD=scenario_data,
                                                             type_convert_map=type_convert_map)

        return new_SD  # return modified scenario description

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


if __name__ == '__main__':
    g = InfgenRLScenarioGenerator(model_name="infgen-base-large")
    from bmt.utils import REPO_ROOT
    import pathlib
    import pickle

    example_sd = pathlib.Path(
        REPO_ROOT) / "data" / "20scenarios" / "sd_training.tfrecord-00000-of-01000_1a7143a44e480ca6.pkl"
    with open(example_sd, "rb") as f:
        scenario_description = pickle.load(f)

    import tqdm
    for _ in tqdm.trange(20):
        g.before_episode(scenario_data=scenario_description)
        new_sd = g.generate(scenario_data=scenario_description)

    import pickle

    with open("sd_1a7143a44e480ca6_infgen_test.pkl", "wb") as f:
        pickle.dump(new_sd, f)
