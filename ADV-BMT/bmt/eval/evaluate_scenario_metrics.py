import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pickle
from bmt.dataset.scenarionet_utils import overwrite_gt_to_pred_field, merge_preds_along_mode_dim
import copy
import hydra
import numpy as np
import omegaconf

from bmt.utils import utils
from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch

from collections.abc import Iterable
from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from bmt.eval.scenario_evaluator import Evaluator
from bmt.utils.utils import numpy_to_torch
from metadrive.scenario.utils import read_dataset_summary

def _get_mode(output_dict, mode, num_modes):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


def _overwrite_datadict_all_agents(source_data_dict, dest_data_dict, ooi=None, T_end=91):
    import copy
    new_data_dict = copy.deepcopy(dest_data_dict)
    B, T, N, _ = source_data_dict["decoder/reconstructed_position"].shape

    if ooi is None:
        ooi = np.arange(N)
        
    is_tensor = isinstance(new_data_dict["decoder/agent_position"], torch.Tensor)
    
    if is_tensor:
        agent_position = new_data_dict["decoder/agent_position"].clone()
        agent_valid_mask = new_data_dict["decoder/agent_valid_mask"].clone()
        agent_heading = new_data_dict["decoder/agent_heading"].clone()
        agent_velocity = new_data_dict["decoder/agent_velocity"].clone()
    else:
        agent_position = new_data_dict["decoder/agent_position"]
        agent_valid_mask = new_data_dict["decoder/agent_valid_mask"]
        agent_heading = new_data_dict["decoder/agent_heading"]
        agent_velocity = new_data_dict["decoder/agent_velocity"]

    for id in ooi:  # overwrite all agents
        if is_tensor:
            traj = source_data_dict["decoder/reconstructed_position"][:, :91, id].clone()
            traj_mask = source_data_dict["decoder/reconstructed_valid_mask"][:, :91, id].clone()
            theta = source_data_dict['decoder/reconstructed_heading'][:, :91, id].clone()
            vel = source_data_dict['decoder/reconstructed_velocity'][:, :91, id].clone()
        else:
            traj = source_data_dict["decoder/reconstructed_position"][:, :91, id]
            traj_mask = source_data_dict["decoder/reconstructed_valid_mask"][:, :91, id]
            theta = source_data_dict['decoder/reconstructed_heading'][:, :91, id]
            vel = source_data_dict['decoder/reconstructed_velocity'][:, :91, id]

        agent_position[:, :, id, :2] = traj
        agent_position[:, :, id, 2] = 0.0
        agent_valid_mask[:, :, id] = traj_mask
        agent_heading[:, :, id] = theta
        agent_velocity[:, :, id] = vel

    new_data_dict["decoder/agent_position"] = agent_position
    new_data_dict["decoder/agent_valid_mask"] = agent_valid_mask
    new_data_dict["decoder/agent_heading"] = agent_heading
    new_data_dict["decoder/agent_velocity"] = agent_velocity

    return new_data_dict


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
        dist = np.min(distances)
        closest_index = np.argmin(distances)
        closest_step = valid_steps[closest_index]

        if dist < min_dist:
            adv_id = id
            min_dist = dist
            adv_closes_step = closest_step
    
    # adv_closes_step = round(adv_closes_step / 5) * 5
    if adv_closes_step < 5:
        adv_closes_step = 5


    # now get adv last valid step's information
    adv_pos = data_dict["decoder/agent_position"][adv_closes_step, adv_id, :2]
    # print("adv_pos:", adv_pos)
    # data_dict["decoder/agent_position"][adv_closes_step, adv_id, 0] -= 5
    # data_dict["decoder/agent_position"][adv_closes_step, adv_id, 0] -= 2

    adv_heading = data_dict["decoder/agent_heading"][adv_closes_step, adv_id]

    adv_vel = data_dict["decoder/agent_velocity"][adv_closes_step, adv_id]



    return adv_id, adv_pos, adv_heading, adv_vel, adv_closes_step

def set_adv(data_dict):
    """
    here is the current design: from existing agents, choose the one with its lastest step having nearest distance among all
    choose random heading; choose last step as collision point
    """
    
    ego_id = data_dict["decoder/sdc_index"]
    ego_traj = data_dict["decoder/agent_position"][:91, ego_id]
    ego_heading = data_dict["decoder/agent_heading"][:91, ego_id]
    ego_velocity = data_dict["decoder/agent_velocity"][:91, ego_id]
    ego_shape = data_dict["decoder/agent_shape"][:91, ego_id]
    ego_mask = data_dict["decoder/agent_valid_mask"][:91, ego_id]

    adv_id, adv_pos, adv_heading, adv_vel, last_valid_step = choose_nearest_adv(data_dict)
    last_valid_step = 90

    ego_last_pos = data_dict["decoder/agent_position"][last_valid_step, ego_id, :2]
    ego_last_heading = data_dict["decoder/agent_heading"][last_valid_step, ego_id]

    # begin to search
    # alphas = np.arange(0, 1.05, 0.05)
    collision_point = ego_last_pos #  - np.random.normal(loc=0.0, scale=1, size=ego_last_pos.shape[0])
    # print("collision point:", collision_point)
    # print("ego last point:", ego_last_pos)
    # for alpha in alphas:
    #     cand_adv_pos = (1 - alpha) * adv_pos + alpha * ego_last_pos

    #     if check_sdc_adv_collision(data_dict, ego_id, ego_last_pos, ego_last_heading, adv_id, cand_adv_pos,
    #                                          adv_heading):
    #         collision_point = cand_adv_pos
    #         break

    adv_mask = np.zeros_like(ego_mask)
    adv_mask[:last_valid_step + 1] = 1  # just like create_new_adv()
    
    data_dict["decoder/agent_valid_mask"][:, adv_id] = adv_mask

    # ===== Position =====
    data_dict["decoder/agent_position"][
        last_valid_step,
        adv_id, :2] = collision_point  # ego_traj[last_valid_step] - np.random.normal(loc=0.0, scale=2, size=3)
    # ====================

    # ===== Heading =====
    adv_heading = np.random.normal(loc=0.0, scale=np.deg2rad(360), size=1)
    data_dict["decoder/agent_heading"][last_valid_step,
                                       adv_id] = adv_heading #+ np.random.normal(loc=0.0, scale=0.1, size=1)
    # ===================

    # ===== Velocity =====
    # adv_vel = 0.5 * (adv_vel + np.random.normal(loc=0.0, scale=0.1, size=2))
    ego_vel = 0.5 * (ego_velocity[last_valid_step])
    adv_vel = 0.5 * (ego_vel)
    data_dict["decoder/agent_velocity"][last_valid_step, ego_id] = ego_vel
    data_dict["decoder/agent_velocity"][last_valid_step, adv_id] = adv_vel
    # ====================

    return data_dict, adv_id



def SCGEN_merge_filtered_preds_along_mode_dim(source_output_dicts, dest_output_dicts, valid_modes, start_merge_mode_index=0, end_merge_mode_index=None):
    # merge the corresponding mode indicies in valid_modes to dest_output_dicts in order from start_merge_mode_index to the end
    # merge only within keys of ["decoder/reconstructed_position", "decoder/reconstructed_valid_mask", "decoder/reconstructed_heading", "decoder/reconstructed_velocity"]:
    assert end_merge_mode_index is not None, "end_merge_mode_index should not be None."
    assert start_merge_mode_index <= end_merge_mode_index, "start_merge_mode_index should be less than end_merge_mode_index."
    
    merge_traj_len = end_merge_mode_index - start_merge_mode_index + 1
    assert merge_traj_len <= len(valid_modes), "merge_traj_len and len(valid_modes) must match."

    for k in ["decoder/reconstructed_position", "decoder/reconstructed_valid_mask", "decoder/reconstructed_heading", "decoder/reconstructed_velocity"]:
        for j in range(merge_traj_len):
            dest_output_dicts[k][start_merge_mode_index+j] = source_output_dicts[k][valid_modes[j]]

    return dest_output_dicts

def reject_SCGEN_batch(backward_output_dict, NUM_MODE, adv_id):
    """
    return the first valid batch/traj
    """
    # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.
    curvature_threshold = 0.8

    all_valid_mode_indicies = []

    for i in range(NUM_MODE):
        # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.

        adv_traj = backward_output_dict["decoder/reconstructed_position"][i,:,adv_id][backward_output_dict["decoder/reconstructed_valid_mask"][i,:,adv_id]] # (T_valid, 2)
        displacements = torch.norm(torch.diff(adv_traj, dim=0), dim=1) + 1e-6

        heading = backward_output_dict["decoder/reconstructed_heading"][i,:,adv_id][backward_output_dict["decoder/reconstructed_valid_mask"][i,:,adv_id]] 
        heading_diffs = torch.abs(torch.diff(heading))
        heading_diffs = torch.minimum(heading_diffs, 2*torch.pi - heading_diffs)
        curvatures = heading_diffs / displacements

        step_masks = backward_output_dict["decoder/reconstructed_valid_mask"][i,:,adv_id]
        last_valid_step = step_masks.nonzero(as_tuple=True)[0][-1].item()  # last valid step

        adv_dist = torch.linalg.norm(adv_traj[-1, :] - adv_traj[0, :], dim=-1)
        adv_dist = adv_dist.mean()  # Shape (1,)

        if adv_dist < last_valid_step/10 or torch.max(curvatures).item() >= curvature_threshold:
            # print("failed cases:", sid)
            # plot_pred(vis_backward_output, save_path=f"vis_scgen_diverse_backward/{sid}_SCGEN_col_{coll_step}_failed_{mode_count}_backward_out.png")
            # print("saving to...", f"{sid}_SCGEN_failed_{mode_count}_backward_out.png")
            continue

        else:
            all_valid_mode_indicies.append(i)

    # return NUM_MODE-1, torch.max(curvatures).item()
    return all_valid_mode_indicies


def convert_tensors_to_double(data_dict, double_keys):

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            if key in double_keys:
                data_dict[key] = value.astype(np.float32)

        elif isinstance(value, torch.Tensor):
            if key in double_keys:
                device = value.device
                data_dict[key] = value.to(device=device, dtype=torch.float32)

    return data_dict

from time import perf_counter
from contextlib import contextmanager
TIMER = False

@contextmanager
def timer(task_name: str):
    start = perf_counter()
    yield
    prof_t = perf_counter() - start
    if TIMER:
        print(f"{task_name}: {prof_t:.5f}")

class EvaluationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        evaluator: Evaluator,
        tokenizer,
        config,
        dataset,
        eval_mode="CAT",
        multi_mode=False,
        num_modes=1,
        backward_TF_mode="no_TF",
        save_path=None,
        overwrite_all_agent=False,
        reject_sampling=False
    ):
        """
            eval_mode: Mode of evaluation ("CAT", "SCGEN", "GPTmodel").
            num_modes: Number of modes to consider for evaluation.
        """
        super().__init__()
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.config = config
        self.dataset = dataset
        self.eval_mode = eval_mode
        self.num_modes = num_modes
        self.cat_summary = None
        self.baseline_summary = None
        self.multi_mode = multi_mode
        self.adv_index = None
        self.TF_mode = backward_TF_mode
        self.sid = None
        self.baseline_dir = "/scenarionet_waymo_training_500"
        self.overwrite_all_agent = overwrite_all_agent

        if self.eval_mode == "CAT":
            self.cat_dir = "CAT_scenarios/"
            summary_path = os.path.join(self.cat_dir, "dataset_summary.pkl")
            with open(summary_path, "rb") as f:
                self.cat_summary = pickle.load(f)
            f.close()

        elif self.eval_mode in ["STRIVE", "SEAL", "GOOSE"]:
            if self.eval_mode == "STRIVE":
                self.baseline_dir = "STRIVE_scenarios/"
            elif self.eval_mode == "SEAL":
                self.baseline_dir = "SEAL_scenarios/"
            elif self.eval_mode == "GOOSE":
                self.baseline_dir = "GOOSE_scenarios/"

            self.baseline_summary, _, _ = read_dataset_summary(self.baseline_dir)

        self.eval_mode = eval_mode
        self.save_path = save_path
        assert save_path is not None, "Please specify the save path for the evaluation results."

        self.reject_sample = reject_sampling #FIXME: Hardcoded for now, change later.



    def GPT_AR(self, input_data, backward_prediction=False, teacher_forcing=False):
        if not teacher_forcing:
            ar_func = self.model.model.autoregressive_rollout_GPT

            return ar_func(
                input_data,
                num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=self.config.SAMPLING.TEMPERATURE,
                backward_prediction=backward_prediction
            )
        else:
            ar_func = self.model.model.autoregressive_rollout_backward_prediction_with_replay # backward
            return ar_func(
                input_data,
                # num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=self.config.SAMPLING.TEMPERATURE,
                topp=self.config.SAMPLING.TOPP,
                not_teacher_forcing_ids=[],
            )
        
    def Backward_Forward_rollout(self, input_data, backward_prediction=True, teacher_forcing=False):
        ar_func = self.model.model.autoregressive_rollout
        with torch.no_grad():
            backward_output_dict = ar_func(
                input_data,
                num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=self.config.SAMPLING.TEMPERATURE,
                backward_prediction=backward_prediction
            )
        backward_output_dict = self.tokenizer.detokenize(
            backward_output_dict,
            detokenizing_gt=False,
            backward_prediction=backward_prediction,
            flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
            teacher_forcing=teacher_forcing
        )

        # processing forward.
        forward_input_dict = _overwrite_datadict_all_agents(source_data_dict=backward_output_dict, dest_data_dict=input_data, ooi=None)
        tok_data_dict, _ = self.tokenizer.tokenize(forward_input_dict, backward_prediction=False)
        forward_input_dict.update(tok_data_dict)

        # run forward with teacher forcing
        forward_input_dict["in_backward_prediction"] = torch.tensor([False] * self.num_modes, dtype=bool).to(self.model.device)

        with torch.no_grad():
            ar_func = self.model.model.autoregressive_rollout
            forward_output_dict = ar_func(
                forward_input_dict,
                # num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=self.config.SAMPLING.TEMPERATURE,
                topp=self.config.SAMPLING.TOPP,
            )

        forward_output_dict = self.tokenizer.detokenize( # forward detokenize
            forward_output_dict,
            detokenizing_gt=False,
            backward_prediction=False,
            flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
            teacher_forcing=teacher_forcing 
        )
        
        return forward_output_dict


    def SCGEN_rollout(self, backward_input_dict, not_tf_ids):
        with torch.no_grad():
            ar_func = self.model.model.autoregressive_rollout_backward_prediction_with_replay
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

        if self.eval_mode == "SCGEN":
            return backward_output_dict



    def SCGEN_rollout_with_filter(self, backward_input_dict, not_tf_ids):
        with torch.no_grad():
            ar_func = self.model.model.autoregressive_rollout_backward_prediction_with_replay
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

        # step 2: filter the backward output, and get all valid trajectories
        valid_modes = reject_SCGEN_batch(backward_output_dict=backward_output_dict, NUM_MODE=self.num_modes, adv_id=self.adv_index)

        merge = True
        if not merge:
            if self.multi_mode and len(valid_modes) >= 4:
                return backward_output_dict, True
            
            else:
                return backward_output_dict, False
            

        if merge:
            return valid_modes, backward_output_dict
        

        


        





    def preprocess_baseline(self, raw_data, mode_idx=None):
        sid = raw_data["metadata/scenario_id"]

        baseline_dir = self.baseline_dir

        baseline_file_name = next((file_name for file_name in self.baseline_summary.keys() if sid in file_name), None)
        if baseline_file_name == None:
            print("baseline file not found:", baseline_file_name)
            return None, None

        baseline_file_path = os.path.join(baseline_dir, baseline_file_name)

        if self.eval_mode in ["STRIVE", "SEAL", "GOOSE", "CAT"]:
            baseline_file_path = os.path.join(baseline_dir, str(mode_idx), baseline_file_name)

        if not os.path.exists(baseline_file_path):
            print("path not exists:", baseline_file_path)
            return None, None
        
        with open(baseline_file_path, "rb") as f:
            baseline_data = pickle.load(f)
        f.close()

        baseline_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=baseline_data,
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            backward_prediction=self.config.BACKWARD_PREDICTION,
            tokenizer=self.tokenizer
        )


        if self.eval_mode in ["STRIVE", "SEAL", "GOOSE", "CAT"]:
            adv_id = int(baseline_data_dict["decoder/adv_agent_id"])  # prepare parameters for differet CR metrics
            self.adv_index = adv_id
            assert self.adv_index is not None, "Adv index should not be None after setting adv."

        input_data = numpy_to_torch(raw_data, device=self.model.device)
        double_keys = ["decoder/agent_position", "decoder/agent_velocity", "decoder/reconstructed_position"]
        input_data = convert_tensors_to_double(input_data, double_keys)

        output_data = overwrite_gt_to_pred_field(baseline_data_dict)
        output_data = numpy_to_torch(output_data, device=self.model.device)
        output_data = convert_tensors_to_double(output_data, double_keys)
        output_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in output_data.items()}

        return input_data, output_data


    def preprocess_CAT(self, raw_data, mode_idx=None):
        sid = raw_data["metadata/scenario_id"]
        cat_file_name = f"sd_adv_reconstructed_v0_{sid}_CAT.pkl"

        if cat_file_name not in self.cat_summary:
            return None, None

        cat_file_path = os.path.join(self.cat_dir, cat_file_name)
        if mode_idx != None:
            cat_file_path = os.path.join(self.cat_dir, str(mode_idx), cat_file_name)
        else:
            cat_file_path = os.path.join(self.cat_dir, cat_file_name)
        with open(cat_file_path, "rb") as f:
            cat_data = pickle.load(f)
        f.close()

        cat_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=cat_data,
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            backward_prediction=self.config.BACKWARD_PREDICTION,
            tokenizer=self.tokenizer
        )

        adv_id = int(cat_data_dict["decoder/adv_agent_id"])  # prepare parameters for differet CR metrics
        self.adv_index = adv_id

        # ========= work around the CAT
        # masking
        adv_hist_mask = raw_data['decoder/agent_valid_mask'
                                 ][:11, self.adv_index
                                   ]  # the shape of adv_hist_mask should match with the adv hist mask in cat_data_dict
        cat_data_dict['decoder/agent_valid_mask'][:11, self.adv_index] = adv_hist_mask

        # velocity
        adv_hist_vel = raw_data['decoder/agent_velocity'][:11, self.adv_index]
        cat_data_dict['decoder/agent_velocity'][:11, self.adv_index] = adv_hist_vel

        # heading
        adv_hist_heading = raw_data['decoder/agent_heading'][:11, self.adv_index]
        cat_data_dict['decoder/agent_heading'][:11, self.adv_index] = adv_hist_heading
        # ======== end of working around

        input_data = numpy_to_torch(raw_data, device=self.model.device)
        double_keys = ["decoder/agent_position", "decoder/agent_velocity", "decoder/reconstructed_position"]
        input_data = convert_tensors_to_double(input_data, double_keys)

        output_data = overwrite_gt_to_pred_field(cat_data_dict)
        output_data = numpy_to_torch(output_data, device=self.model.device)
        output_data = convert_tensors_to_double(output_data, double_keys)
        output_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in output_data.items()}

        return input_data, output_data

    def preprocess_SCGEN(self, raw_data):

        from bmt.safety_critical_scenario_generation import create_new_adv
        data_dict = copy.deepcopy(raw_data)

        self.sid = raw_data["metadata/scenario_id"]

        # data_dict, adv_id = create_new_adv(data_dict) 
        data_dict, adv_id = set_adv(data_dict) 

        self.adv_index = adv_id
        assert self.adv_index is not None, "Adv index should not be None after setting adv."
        
        input_data = numpy_to_torch(data_dict, device=self.model.device)
        input_data["in_evaluation"] = torch.tensor([1], dtype=bool).to(self.model.device)
        original_data_dict_tensor = copy.deepcopy(input_data)

        input_data = {
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data.items()
        }

        # Force to run backward prediction first to make sure the data is tokenized correctly.
        tok_data_dict, _ = self.tokenizer.tokenize(input_data, backward_prediction=True)
        input_data.update(tok_data_dict)

        input_data["in_backward_prediction"] = torch.tensor([True] * self.num_modes, dtype=bool).to(self.model.device)


        all_agents = input_data["decoder/agent_id"][0]

        if self.TF_mode == "no_TF":
            not_tf_ids = all_agents

        elif self.TF_mode == "sdc_TF":
            not_tf_ids = all_agents[all_agents != 0]

        else:
            not_tf_ids = [adv_id]

        return input_data, adv_id, not_tf_ids, original_data_dict_tensor

    def preprocess_GPTmodel(self, raw_data, backward_prediction=False):
        # input_data = {
        #     k: torch.from_numpy(v).to(self.model.device) if isinstance(v, np.ndarray) and "track_name" not in k else v
        #     for k, v in raw_data.items()
        # }
        raw_data["in_evaluation"] = torch.tensor([1], dtype=bool).to(self.model.device)
        input_data = numpy_to_torch(raw_data, device=self.model.device)
        input_data = {
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data.items()
        }

        # Force to run backward prediction first to make sure the data is tokenized correctly!!!
        tok_data_dict, _ = self.tokenizer.tokenize(input_data, backward_prediction=backward_prediction)
        input_data.update(tok_data_dict)

        if not backward_prediction: # handle backward flag
            if self.config.BACKWARD_PREDICTION:
                input_data["in_backward_prediction"] = torch.tensor(
                    [False] * self.num_modes, dtype=bool
                ).to(self.model.device)
        else:
            input_data["in_backward_prediction"] = torch.tensor([True] * self.num_modes, dtype=bool).to(self.model.device)
                
        return input_data

    def test_step(self, batch, batch_idx):
        print("sid:", batch["metadata/scenario_id"]) # print scenario ID for debugging
        
        if self.eval_mode == "CAT":
            if self.multi_mode:
                output_datas = []
                for i in range(self.num_modes):
                    input_data, output_data = self.preprocess_CAT(batch, mode_idx=i)
                    if input_data is None:  # Skip if no valid CAT scenario
                        return
                    output_datas.append(output_data)
                output_data = merge_preds_along_mode_dim(output_datas)
            else:
                input_data, output_data = self.preprocess_CAT(batch)
                if input_data is None:  # Skip if no valid CAT scenario
                    return
        elif self.eval_mode in ["STRIVE", "SEAL", "GOOSE"]:
            if self.multi_mode:
                output_datas = []
                with timer("Preprocess"):
                    for i in range(self.num_modes):
                        input_data, output_data = self.preprocess_baseline(batch, mode_idx=i)
                        if input_data is None:  # Skip if no valid STRIVE scenario
                            return
                        output_datas.append(output_data)
                with timer("Merge"):
                    output_data = merge_preds_along_mode_dim(output_datas)
            else:
                input_data, output_data = self.preprocess_baseline(batch)
                if input_data is None:  # Skip if no valid STRIVE scenario
                    return

        elif self.eval_mode == "SCGEN":

            if not self.reject_sample:
                input_data, adv_id, not_tf_ids, original_data_dict_tensor = self.preprocess_SCGEN(
                    batch
                )
                self.adv_id = adv_id

                output_data = self.SCGEN_rollout(input_data, not_tf_ids)


            else:
                resample_count = 0

                """
                Alternatively, needs to merge all valid modes into one output_data with batch size of self.num_modes.
                If the resampling is successful, it will return a valid output_data.
                If not, it will resample up to 5 times and return the last output_data
                """

                len_valid_modes = 0

                while True:
                    input_data, adv_id, not_tf_ids, original_data_dict_tensor = self.preprocess_SCGEN(
                        batch
                    )
                    self.adv_id = adv_id

                    valid_modes, output_data = self.SCGEN_rollout_with_filter(input_data, not_tf_ids)

                    if resample_count == 0:
                        final_filtered_output_data = copy.deepcopy(output_data)

                    if len(valid_modes) > 0:
                        start_idx = len_valid_modes
                        end_idx = min(len_valid_modes + len(valid_modes) - 1, self.num_modes - 1)
                        if start_idx <= end_idx:
                            final_filtered_output_data = SCGEN_merge_filtered_preds_along_mode_dim(
                                source_output_dicts=output_data,
                                dest_output_dicts=final_filtered_output_data,
                                valid_modes=valid_modes[:end_idx - start_idx + 1],
                                start_merge_mode_index=start_idx,
                                end_merge_mode_index=end_idx
                            )

                    len_valid_modes += len(valid_modes)

                    if len_valid_modes >= self.num_modes or resample_count >= 5:
                        if len_valid_modes < self.num_modes:
                            print(f"Resample {resample_count} times, but still not enough valid SCGEN scenarios. Skipping this batch.")

                        output_data = final_filtered_output_data
                        break

                    resample_count += 1


            #  ======== overwrite the new adv's traj ===========
            # adv_traj = output_data["decoder/reconstructed_position"][:,:,-1]
            # original_data_dict_tensor["decoder/agent_position"][:,-1,:2] = adv_traj

            assert self.overwrite_all_agent == False # FIXME: change here.
            if not self.overwrite_all_agent:
                # ======= now, overwrite all other agents to keep consistent with CAT/STRIVE ==========
                original_agent_ids = original_data_dict_tensor["decoder/agent_id"]
                # ============ 
                for oid in original_agent_ids:
                    if oid == self.adv_id:
                        continue
                    output_data["decoder/reconstructed_position"][:,:,oid] = output_data["decoder/agent_position"][:,:,oid,:2]
                    output_data["decoder/reconstructed_heading"][:,:,oid] = output_data["decoder/agent_heading"][:,:,oid]
                    output_data["decoder/reconstructed_velocity"][:,:,oid] = output_data["decoder/agent_velocity"][:,:,oid]
                    output_data["decoder/reconstructed_valid_mask"][:,:,oid] = output_data["decoder/agent_valid_mask"][:,:,oid]


        elif self.eval_mode == "GPTmodel" or self.eval_mode == "Backward" or self.eval_mode == "Backward_TF" or self.eval_mode == "Backward_Forward":
            backward = (self.eval_mode == "Backward" or self.eval_mode == "Backward_TF" or self.eval_mode == "Backward_Forward")

            data_dict = copy.deepcopy(batch)
            input_data = numpy_to_torch(data_dict, device=self.model.device)

            original_data_dict_tensor = copy.deepcopy(input_data)

            input_data = self.preprocess_GPTmodel(batch, backward_prediction=backward)
            print("finish preprocessing GPTmodel")


            if self.eval_mode == "Backward_Forward":
                output_data = self.Backward_Forward_rollout(input_data, backward_prediction=backward, teacher_forcing=False)
            else:
                with torch.no_grad():
                    print("processing GPTmodel")
                    output_data = self.GPT_AR(input_data, backward_prediction=backward, teacher_forcing=False)
                output_data = self.tokenizer.detokenize(output_data, detokenizing_gt=False, backward_prediction=backward, teacher_forcing=False)

        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")
        
        gathered_output = output_data
        if not self.evaluator.start_metrics_only:
            if self.eval_mode == "GPTmodel" or self.eval_mode == "Backward" or self.eval_mode == "Backward_TF": 
                avg_sdc_adv_cr, avg_sdc_bv_cr, avg_adv_bv_cr, all_agent_cr = self.calculate_collision_statistics(
                    output_data,
                    is_CAT_data=False,
                )
                self.evaluator.add_customized_CR(
                    avg_sdc_adv_cr=avg_sdc_adv_cr, avg_adv_bv_cr=avg_adv_bv_cr,avg_sdc_bv_cr=avg_sdc_bv_cr, all_agent_cr=all_agent_cr
                )

            else: # safety-critical predictions
                avg_sdc_adv_cr, avg_sdc_bv_cr, avg_adv_bv_cr, all_agent_cr = self.calculate_collision_statistics(
                    output_data, is_CAT_data=(self.eval_mode in ["CAT", "STRIVE", "SEAL", "GOOSE"])
                )
                self.evaluator.add_customized_CR(
                    avg_sdc_adv_cr=avg_sdc_adv_cr, avg_sdc_bv_cr=avg_sdc_bv_cr, avg_adv_bv_cr=avg_adv_bv_cr, all_agent_cr=all_agent_cr
                )

        # Evaluate
        if self.eval_mode == "SCGEN":
            all_agents = input_data["decoder/agent_id"][0]  # prepare parameters for differet CR metrics
            sdc_id = int(input_data["decoder/sdc_index"][0])
            all_agents_except_sdc_adv = all_agents[all_agents != sdc_id]
            all_agents_except_sdc_adv = all_agents_except_sdc_adv[all_agents_except_sdc_adv != self.adv_id]

            device = self.model.device
            bv_list = all_agents_except_sdc_adv.to(device)
            self.evaluator.add(
                original_data_dict_tensor, gathered_output, adv_list=torch.tensor([self.adv_id], device=device), bv_list=bv_list, device=self.device
            )

        # Evaluate
        if self.eval_mode == "SCGEN":
            all_agents = input_data["decoder/agent_id"][0]  # prepare parameters for differet CR metrics
            sdc_id = int(input_data["decoder/sdc_index"][0])
            all_agents_except_sdc_adv = all_agents[all_agents != sdc_id]
            all_agents_except_sdc_adv = all_agents_except_sdc_adv[all_agents_except_sdc_adv != self.adv_id]

            device = self.model.device
            bv_list = all_agents_except_sdc_adv.to(device)
            self.evaluator.add(
                original_data_dict_tensor, gathered_output, adv_list=torch.tensor([self.adv_id], device=device), bv_list=bv_list, device=self.device
            )


        elif self.eval_mode in ["CAT", "STRIVE", "SEAL", "GOOSE"]:
            adv_id = int(output_data["decoder/adv_agent_id"][0])  # prepare parameters for differet CR metrics
            all_agents = output_data["decoder/agent_id"][0]

            sdc_id = int(output_data["decoder/sdc_index"][0])
            all_agents_except_sdc_adv = all_agents[all_agents != sdc_id]
            all_agents_except_sdc_adv = all_agents_except_sdc_adv[all_agents_except_sdc_adv != adv_id]

            device = self.model.device
            adv_list = torch.tensor([adv_id], device=device)
            bv_list = all_agents_except_sdc_adv.to(device)
            input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
            gathered_output = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in gathered_output.items()
            }

            self.evaluator.add(input_data, gathered_output, adv_list=adv_list, bv_list=bv_list, device=self.device)

        elif self.eval_mode == "GPTmodel" or self.eval_mode == "Backward" or self.eval_mode == "Backward_TF" or self.eval_mode == "Backward_Forward" :
            all_agents = batch["decoder/agent_id"]  # prepare parameters for differet CR metrics
            sdc_id = batch["decoder/sdc_index"]
            all_agents_except_sdc = all_agents[all_agents != sdc_id]
            self.evaluator.add(original_data_dict_tensor, gathered_output, adv_list=None, bv_list=all_agents_except_sdc, device=self.device)


        else:
            raise ValueError(f"Unexpected evaluation mode: {self.eval_mode}")

        return gathered_output

    def on_test_epoch_end(self):
        self.trainer.strategy.barrier() # ensure all processes are done with evaluation
        if self.trainer.is_global_zero:
            # self.evaluator.print()

            # Get aggregated metrics
            aggregated_metrics = self.evaluator.aggregate()

            self.log_dict(aggregated_metrics, sync_dist=True, rank_zero_only=True)
            self.evaluator.save(self.save_path)

    def configure_optimizers(self):
        # No optimizer required for evaluation
        return None

    def calculate_collision_statistics(self, output_data, cr_mode="avg", is_CAT_data=False):

        """
        For SCGEN/CAT/SEAL/STRIVE/GOOSE, we know that the adv_index is not None
        sdc_index is always 0, which is the SDC agent.
        """

        ooi_ind = output_data["decoder/agent_id"][0]  # ooi is all agent
        assert 0 in ooi_ind, "SDC agent should be in the output data."

        from bmt.dataset.preprocess_action_label import get_2D_collision_labels

        output_data_all_modes = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_data.items()
        }

        if not is_CAT_data:
            output_data_all_modes = _overwrite_datadict_all_agents(source_data_dict=output_data_all_modes, dest_data_dict=output_data_all_modes) # overwrite pred to GT

        if is_CAT_data:
            num_modes = 1
        else:
            num_modes = self.num_modes

        avg_sdc_adv_col = 0
        avg_sdc_bv_col = 0
        avg_adv_bv_col = 0
        avg_all_agent_cr = 0
        num_bv_agent = 0 


        sdc_index = 0
        adv_index = self.adv_index  # value is None for eval_mode = GPTmodel


        for i in range(num_modes):
            sdc_adv_col = 0
            sdc_bv_col = 0
            adv_bv_col = 0

            output_dict = utils.torch_to_numpy(output_data)
            output_dict = _get_mode(output_dict, i, self.num_modes)

            from bmt.gradio_ui.plot import create_animation_from_pred
            # video_path = f"{self.eval_mode}_pred_{output_dict['metadata/scenario_id']}_{i}.mp4"
            # video_path = create_animation_from_pred(output_dict, save_path=video_path, dpi=100)
            # print("predict gif path:", video_path)

            from bmt.gradio_ui.plot import plot_pred
            # plot_pred(output_dict, save_path=f"{self.eval_mode}_pred_{output_dict['metadata/scenario_id']}_{i}.png")

            output_dict_mode = _get_mode(output_data_all_modes, i, num_modes=num_modes)

            # we should overwrite the adv agent's trajectory to GT

            col_label = get_2D_collision_labels(data_dict=output_dict_mode, track_agent_indicies=ooi_ind)


            if self.eval_mode == "SCGEN":
                assert col_label[sdc_index][adv_index] # report cases for non-collision on SDC-ADV

            if adv_index is not None and col_label[sdc_index][adv_index]: # SDC-ADV collision
                sdc_adv_col += 1

            for agent_id in ooi_ind: # BV collision
                if agent_id == adv_index or agent_id == sdc_index: # skip SDC and ADV
                    continue

                if col_label[sdc_index][agent_id]: # SDC-BV collision
                    sdc_bv_col += 1

                if adv_index is not None and col_label[adv_index][agent_id]: # ADV-BV collision
                    adv_bv_col += 1

            avg_sdc_adv_col += sdc_adv_col
            avg_sdc_bv_col += sdc_bv_col
            avg_adv_bv_col += adv_bv_col
            avg_all_agent_cr += np.sum(np.triu(col_label, k=1)) / ooi_ind.shape[0] # average collision rate for all agents

            if self.eval_mode.startswith("SCGEN") or self.eval_mode.startswith("STRIVE") or self.eval_mode.startswith("SEAL") or self.eval_mode.startswith("GOOSE") or self.eval_mode.startswith("CAT"):
                num_bv_agent += ooi_ind.shape[0] - 2 # bv with no sdc and no adv
            else:
                num_bv_agent += ooi_ind.shape[0] - 1  # bv with no sdc

        assert num_bv_agent > 0
        avg_sdc_adv_cr = avg_sdc_adv_col / num_modes
        avg_sdc_bv_cr = avg_sdc_bv_col / (num_modes * num_bv_agent) 
        avg_adv_bv_cr = avg_adv_bv_col / (num_modes * num_bv_agent)
        avg_all_agent_cr = avg_all_agent_cr / num_modes
    
        return avg_sdc_adv_cr, avg_sdc_bv_cr, avg_adv_bv_cr, avg_all_agent_cr


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="0202_midgpt.yaml")
def run_combined_evaluation(config):
    from pytorch_lightning import Trainer

    path = "../ckpt/last.ckpt"  # change to your checkpoint path
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.BACKWARD_PREDICTION = True 

    # ===== FORMAL TEST =====
    config.DATA.TEST_DATA_DIR = "scenarionet_waymo_training_500" 
    
    model = utils.get_model(checkpoint_path=path)
    model = model.to("cuda")

    eval_mode = config.eval_mode
    tokenizer = model.model.tokenizer

    test_bs = 1
    limit_test_batches = 100

    assert config.multi_mode is True, "Please set config.multi_mode to True for multi-mode evaluation."

    # ===== LOCAL DEBUG =====
    # config.DATA.TRAINING_DATA_DIR = "data/20scenarios"
    # config.DATA.TEST_DATA_DIR = "data/20scenarios"
    # model = utils.get_model(config=config)
    # limit_test_batches = 1

    evaluator = Evaluator(key_metrics_only=False, start_metrics_only=False) # key_metrics_only contains metrics without waymo's evaluations

    if eval_mode == "GPTmodel":
        backward = False
    else:
        backward = True
        
    dataset = InfgenDataset(config, "test", backward_prediction=backward)
    dataloader = DataLoader(dataset, batch_size=test_bs, collate_fn=lambda x: x[0], num_workers=8) # collate_fn=dataset.collate_batch # collate_fn=sample_10_collate_fn) # lambda x: x[0]

    num_modes = 1 if not config.multi_mode else 6
    TF_mode = "all_TF_except_adv" # TF_mode: [no_TF, sdc_TF, all_TF_except_adv] 

    reject_sampling = True # FIXME: change here.
    overwrite_all_agent = False # FIXME: change here.

    # get the current date string in mmdd
    from datetime import datetime
    current_date = datetime.now().strftime("%m%d") # format as mmdd

    if eval_mode.startswith("SCGEN"):
        if overwrite_all_agent:
            assert not reject_sampling, "Reject sampling should be False when overwriting all agents."
            save_path = "{}_{}_{}_{}_open_loop_results".format(current_date, eval_mode, TF_mode, "overwrite_all_agent") if not config.multi_mode else "{}_{}_{}_{}_multi_mode_open_loop_results".format(current_date, eval_mode, TF_mode, "overwrite_all_agent")
        elif reject_sampling:
            assert not overwrite_all_agent, "Overwrite all agents should be False when rejecting sampling."
            save_path = "{}_{}_{}_{}_open_loop_results".format(current_date, eval_mode, TF_mode, "reject_sampling") if not config.multi_mode else "{}_{}_{}_{}_multi_mode_open_loop_results".format(current_date, eval_mode, TF_mode, "reject_sampling")
        else:
            save_path = "{}_{}_{}_open_loop_results".format(current_date,eval_mode, TF_mode) if not config.multi_mode else "{}_{}_{}_multi_mode_open_loop_results".format(current_date, eval_mode, TF_mode)
    else:
        save_path = "{}_{}_open_loop_results".format(current_date, eval_mode) if not config.multi_mode else "{}_{}_multi_mode_open_loop_results".format(current_date, eval_mode)
    
    evaluation_module = EvaluationLightningModule(
        model,
        evaluator,
        tokenizer,
        config,
        dataset,
        eval_mode=eval_mode,
        multi_mode=config.multi_mode,
        num_modes=num_modes,
        backward_TF_mode=TF_mode,
        save_path=save_path,
        overwrite_all_agent=overwrite_all_agent,
        reject_sampling=reject_sampling
    )

    trainer = Trainer(accelerator="gpu", devices=1, limit_test_batches=limit_test_batches)
    trainer.test(evaluation_module, dataloaders=dataloader)


if __name__ == '__main__':
    run_combined_evaluation()  # use command line to assign eval_mode: ["CAT", "SCGEN", "GPTmodel", "Backward", "Backward_Forward", "STRIVE"]