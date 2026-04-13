import pytorch_lightning as pl
from torch.utils.data import DataLoader
import copy
import pickle
from bmt.utils.safety_critical_generation_utils import _overwrite_data_given_agents_not_ooi, get_ego_edge_points, get_ego_edge_points_old, post_process_adv_traj, _overwrite_data_given_agents_ooi, _overwrite_data_given_agents, set_adv, run_backward_prediction_with_teacher_forcing, convert_tensors_to_double
from bmt.utils import utils
from bmt.dataset.scenarionet_utils import overwrite_gt_to_pred_field
import copy
import hydra
import numpy as np
import omegaconf
import tqdm

from bmt.utils import utils
from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch

import pathlib
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from google.protobuf import text_format
import tensorflow as tf
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features
import torch.nn.functional as F
import itertools
from waymo_open_dataset.protos import map_pb2
from collections.abc import Iterable
import pdb
from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from bmt.eval.debug_scenario_metrics import Evaluator
from bmt.utils.utils import numpy_to_torch


class EvaluationLightningModule(pl.LightningModule):
    def __init__(self, model, evaluator, tokenizer, config, dataset, eval_mode="CAT", num_modes=5):
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

        if self.eval_mode == "CAT":
            cat_dir = "CAT_waymo_validation_interactive_500_nearestADV"
            with open(f"{cat_dir}/dataset_summary.pkl", "rb") as f:
                self.cat_summary = pickle.load(f)

        self.eval_mode = eval_mode

    def sample_collate_fn(batch, num_scenario=5):
        return random.sample(batch, k=min(len(batch), num_scenario))

    def GPT_forwardAR(self, input_data):
        return self.model.model.autoregressive_rollout(
            input_data,
            num_decode_steps=None,
            sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
            temperature=self.config.SAMPLING.TEMPERATURE,
        )

    def GPT_backwardAR(self, backward_input_dict, teacher_forcing=False, not_tf_ids=None):
        if teacher_forcing:
            assert not_tf_ids is not None
            return run_backward_prediction_with_teacher_forcing(
                model=self.model,
                config=self.config,
                backward_input_dict=backward_input_dict,
                tokenizer=self.tokenizer,
                not_teacher_forcing_ids=not_tf_ids
            )

        else:
            return run_backward_prediction_with_teacher_forcing(
                model=self.model,
                config=self.config,
                backward_input_dict=backward_input_dict,
                tokenizer=self.tokenizer,
                not_teacher_forcing_ids=backward_input_dict["decoder/agent_id"]
            )

    def preprocess_CAT(self, raw_data):
        sid = raw_data["metadata/scenario_id"]
        cat_dir = "CAT_waymo_validation_interactive_500_nearestADV"
        cat_file_name = f"sd_adv_reconstructed_v0_{sid}_CAT.pkl"

        if cat_file_name not in self.cat_summary:
            return None, None

        with open(f"{cat_dir}/{cat_file_name}", "rb") as f:
            cat_data = pickle.load(f)

        cat_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=cat_data,
            config=self.config,
            in_evaluation=True,
            keep_all_data=True,
            cache=None,
            backward_prediction=self.config.BACKWARD_PREDICTION,
            tokenizer=self.tokenizer
        )

        input_data = numpy_to_torch(raw_data, device=self.model.device)
        double_keys = ["decoder/agent_position", "decoder/agent_velocity", "decoder/reconstructed_position"]
        input_data = convert_tensors_to_double(input_data, double_keys)

        output_data = overwrite_gt_to_pred_field(cat_data_dict)
        output_data = numpy_to_torch(output_data, device=self.model.device)
        output_data = convert_tensors_to_double(output_data, double_keys)
        output_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in output_data.items()}

        return input_data, output_data

    def preprocess_SCGEN(self, raw_data):

        from bmt.utils.safety_critical_generation_utils import set_adv
        data_dict = copy.deepcopy(raw_data)
        data_dict, adv_id = set_adv(data_dict)
        input_data = numpy_to_torch(data_dict, device=self.model.device)

        original_data_dict_tensor = copy.deepcopy(input_data)

        # Extend the batch dim:
        input_data = {
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data.items()
        }
        input_data["in_backward_prediction"] = torch.tensor([True] * self.num_modes, dtype=bool).to(self.model.device)

        all_agents = input_data["decoder/agent_id"][0]
        not_tf_ids = all_agents[all_agents != 0]

        return input_data, adv_id, not_tf_ids, original_data_dict_tensor

    def preprocess_GPTmodel(self, raw_data):
        input_data = {
            k: torch.from_numpy(v).to(self.model.device) if isinstance(v, np.ndarray) and "track_name" not in k else v
            for k, v in raw_data.items()
        }

        input_data = {
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=self.num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data.items()
        }
        input_data["in_evaluation"] = torch.tensor([1], dtype=bool).to(self.model.device)
        if self.config.BACKWARD_PREDICTION:
            input_data["in_backward_prediction"] = torch.tensor(
                [False] * self.num_modes, dtype=bool
            ).to(self.model.device)
        return input_data

    def test_step(self, batch, batch_idx):
        if self.eval_mode == "CAT":
            input_data, output_data = self.preprocess_CAT(batch)
            if input_data is None:  # Skip if no valid CAT scenario
                return

        elif self.eval_mode == "SCGEN":
            input_data, adv_id, not_tf_ids, original_data_dict_tensor = self.preprocess_SCGEN(batch)
            with torch.no_grad():
                teacher_forcing = True
                output_data = self.GPT_backwardAR(input_data, teacher_forcing=teacher_forcing, not_tf_ids=not_tf_ids)

        elif self.eval_mode == "GPTmodel":
            input_data = self.preprocess_GPTmodel(batch)
            with torch.no_grad():
                output_data = self.GPT_forwardAR(input_data)
            output_data = self.tokenizer.detokenize(output_data, detokenizing_gt=False, backward_prediction=False)
        else:
            raise ()

        # Evaluate
        if self.eval_mode == "SCGEN":
            all_agents = batch["decoder/agent_id"]  # prepare parameters for differet CR metrics
            sdc_id = batch["decoder/sdc_index"]
            all_agents_except_sdc = all_agents[all_agents != sdc_id]
            self.evaluator.add(original_data_dict_tensor, output_data, adv=[adv_id], bv=all_agents_except_sdc)

        elif self.eval_mode == "CAT":
            adv_id = int(output_data["decoder/adv_agent_id"][0])  # prepare parameters for differet CR metrics
            all_agents = output_data["decoder/agent_id"][0]
            sdc_id = int(output_data["decoder/sdc_index"][0])
            all_agents_except_sdc = all_agents[all_agents != sdc_id]
            self.evaluator.add(input_data, output_data, adv=[adv_id], bv=all_agents_except_sdc)

        elif self.eval_mode == "GPTmodel":
            all_agents = batch["decoder/agent_id"]  # prepare parameters for differet CR metrics
            sdc_id = batch["decoder/sdc_index"]
            all_agents_except_sdc = all_agents[all_agents != sdc_id]
            self.evaluator.add(batch, output_data)

        else:
            raise ()

    def on_test_epoch_end(self):
        # Aggregate and print evaluation metrics
        self.evaluator.aggregate()
        self.evaluator.print()

    def configure_optimizers(self):
        # No optimizer required for evaluation
        return None


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1031_midgpt.yaml")
def run_combined_evaluation(config):
    from pytorch_lightning import Trainer
    from bmt.utils import utils
    path = "../ckpt/last.ckpt"
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.pretrain = "../ckpt/last.ckpt"
    config.BACKWARD_PREDICTION = True  # <<<
    config.ADD_CONTOUR_RELATION = True
    config.DATA.TRAINING_DATA_DIR = "waymo_validation_interactive_500"  #"data/20scenarios"
    config.DATA.TEST_DATA_DIR = "waymo_validation_interactive_500"  #"data/20scenarios"
    omegaconf.OmegaConf.set_struct(config, True)
    model = utils.get_model(checkpoint_path=path)
    model = model.to("cuda")
    device = model.device

    eval_mode = config.eval_mode

    from bmt.tokenization.motion_tokenizers import get_tokenizer
    tokenizer = get_tokenizer(config)

    evaluator = Evaluator()
    dataset = InfgenDataset(config, "test")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])  # collate_fn=sample_10_collate_fn)

    evaluation_module = EvaluationLightningModule(model, evaluator, tokenizer, config, dataset, eval_mode=eval_mode)

    trainer = Trainer(accelerator="gpu", devices=1, limit_test_batches=10)  # limit_test_batches=10 precision=16
    trainer.test(evaluation_module, dataloaders=dataloader)


run_combined_evaluation()  # just use command line to assign eval_mode from ["CAT", "SCGEN", "GPTmodel"]
