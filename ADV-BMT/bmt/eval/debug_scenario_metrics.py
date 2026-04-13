import copy

import PIL
import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import seaborn as sns
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle, Rectangle
import tqdm
# Load model
from bmt.utils import utils
from bmt.dataset.dataset import InfgenDataset
from bmt.utils import REPO_ROOT
import torch

from bmt.gradio_ui.plot import plot_pred, create_animation_from_pred
from bmt.gradio_ui.plot import plot_pred
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
# @hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="motion_default.yaml")
# def debug(config):
#     omegaconf.OmegaConf.set_struct(config, False)
#     config.PREPROCESSING.keep_all_data = True
#     omegaconf.OmegaConf.set_struct(config, True)
#     test_dataset = InfgenDataset(config, "test")
#     ddd = iter(test_dataset)
#     while True:
#         try:
#             raw_data = data = next(ddd)
#
#             from infgen.tokenization import get_tokenizer
#             tokenizer = get_tokenizer(config)
#             data, _ = tokenizer.tokenize_numpy_array(data)
#             data["decoder/output_action"] = data["decoder/target_action"]
#             fill_zero = ~data["decoder/target_action_valid_mask"]
#             data["decoder/input_action_valid_mask"][fill_zero] = False
#
#             data = tokenizer.detokenize_numpy_array(data, detokenizing_gt=True)
#             raw_data.update(data)
#             # plot_pred(raw_data)
#             plot_pred(raw_data, show=True)
#
#             # break
#         except StopIteration:
#             break
#     print("End")
#


def print_type_and_dtype(name, tensor):
    print(f"{name} - Type: {type(tensor)}, Dtype: {getattr(tensor, 'dtype', 'N/A')}")


def conv(tensor, dtype=tf.float32):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    return tf.convert_to_tensor(tensor, dtype=dtype)


# conv = lambda tensor: tf.convert_to_tensor(tensor if type(tensor) == np.ndarray else tensor.cpu().numpy())
rconv = lambda tf_tensor: torch.from_numpy(tf_tensor if type(tf_tensor) == np.ndarray else tf_tensor.numpy()
                                           ).to(torch.device("cuda"))
from bmt.utils.utils import numpy_to_torch


def tf_to_torch(tf_tensor, device=None):
    # Convert TensorFlow tensor to NumPy array on CPU if necessary
    if tf_tensor.device.endswith("GPU:0"):  # If on GPU, move to CPU first
        tf_tensor = tf_tensor.cpu()
    np_array = tf_tensor.numpy()

    return torch.from_numpy(np_array).to(
        device if device else torch.device("cuda" if tf_tensor.device.endswith("GPU:0") else "cpu")
    )


def jsd(gt_hist, pred_hist, epsilon=1e-10):
    gt_prob = gt_hist / gt_hist.sum()
    pred_prob = pred_hist / pred_hist.sum()
    gt_prob += epsilon
    pred_prob += epsilon
    m = 0.5 * (gt_prob + pred_prob)
    jsd = 0.0
    jsd += F.kl_div(gt_prob.log(), m, reduction="sum")
    jsd += F.kl_div(pred_prob.log(), m, reduction="sum")
    return (0.5 * jsd)


# Timing
from time import perf_counter
from contextlib import contextmanager

TIMER = False
SCENE_IDX = 0


@contextmanager
def timer(task_name: str):
    start = perf_counter()
    yield
    prof_t = perf_counter() - start
    if TIMER:
        print(f"{task_name}: {prof_t:.5f}")


class Evaluator:
    # TP_safety_0 = 0
    # FP_safety_0 = 0
    # FN_safety_0 = 0
    #
    # TP_safety_1 = 0
    # FP_safety_1 = 0
    # FN_safety_1 = 0

    scenario_count = 0

    # Diversity
    minSFDE = 0  # (supervised) avg over scenarios: minimum over all modes: average of L2 error of final positions of all agents
    FDD = 0  # (unsupervised) avg over scenarios: average over all agents: maximum L2 distance in final position of that agent between generated modes
    # Xuanhao: In MixSim paper, they used squared norm of distance, but maybe they meant L2 norm not squared norm?
    # Unit given in AdvDiffuser for FDD is m not m^2, so I am using L2 norm here

    # Distribution Realism
    vel_jsd = 0  # avg over scenarios: build histogram across agents, modes, timestamps: velocity JS divergence
    acc_jsd = 0  # avg over scenarios: build histogram across agents, modes, timestamps: acceleration JS divergence
    ttc_jsd = 0  # avg over scenarios: build histogram across agents, modes, timestamps: time to collision JS divergence

    # Common Sense
    env_coll = 0  # offroad
    veh_coll = 0  # collision rate
    sdc_coll_adv = 0
    sdc_coll_adv_active = False
    sdc_coll_bv = 0
    sdc_coll_bv_active = False
    adv_coll_bv = 0
    adv_coll_bv_active = False
    coll_vel = 0  # collision velocity
    # no clue what collision JSD means so not calculating it for now

    # AV comfortable
    acc = 0  # Xuanhao: avg over scenarios: min over modes: max over time steps: acceleration of ego vehicle
    jerk = 0  # Xuanhao: avg over scenarios: min over modes: max over time steps: jerk of ego vehicle

    # Output Metrics
    metrics = {}
    metric_units = {}

    # Constants
    SECONDS_PER_STEP = 0.1

    def __init__(self):
        self.metric_units = {
            "# Scenarios": "",
            "minSFDE": "m",
            "FDD": "m",
            "Vel. JSD": "",
            "Acc. JSD": "",
            "TTC JSD": "",
            "Env CR": "",
            "Veh CR": "",
            "ADV+SDC CR": "",
            "ADV+BV CR": "",
            "SDC+BV CR": "",
            "Avg. Max Coll Vel.": "m/s",
            "Avg. Max Acc.": "m/s^2",
            "Avg. Max Jerk": "m/s^3",
        }
        self.display_keys = [
            "# Scenarios", "minSFDE", "FDD", "Vel. JSD", "Acc. JSD", "TTC JSD", "Env CR", "Veh CR", "ADV+SDC CR",
            "ADV+BV CR", "SDC+BV CR", "Avg. Max Coll Vel.", "Avg. Max Acc.", "Avg. Max Jerk"
        ]
        self.jsd_config = {
            "vel": {
                "min_val": 0.0,
                "max_val": 50.0,
                "num_bins": 100
            },
            "acc": {
                "min_val": -10.0,
                "max_val": 10.0,
                "num_bins": 200
            },
            "ttc": {
                "min_val": 0.0,
                "max_val": 5.0,
                "num_bins": 50
            }
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, gt_data_dict, pred_data_dict, **kwargs):
        global SCENE_IDX
        self.scenario_count += 1

        T_gt = gt_data_dict["decoder/agent_position"].shape[0]
        T_context = 0

        T_pred = pred_data_dict["decoder/reconstructed_position"].shape[1]
        K = pred_data_dict["decoder/reconstructed_position"].shape[0]
        N = gt_data_dict["decoder/agent_position"].shape[1]
        vehicle_mask = numpy_to_torch(gt_data_dict["decoder/agent_type"] == 1, device=self.device)  # (num agents)
        ooi_mask = torch.zeros_like(vehicle_mask, dtype=torch.bool, device=self.device)
        ooi_mask[(gt_data_dict["decoder/labeled_agent_id"])] = 1  # (num agents)
        gt_valid_mask = numpy_to_torch(
            gt_data_dict["decoder/agent_valid_mask"], device=self.device
        ).T  # (num agents, num steps)
        pred_valid_mask = pred_data_dict["decoder/reconstructed_valid_mask"].transpose(
            1, 2
        )  # (K, num agents, num steps)
        # joint_mask = vehicle_mask.unsqueeze(-1) & valid_mask # (num agents, num steps)
        gt_ooi_joint = ooi_mask.unsqueeze(-1) & gt_valid_mask[..., T_context:T_gt]  # (num agents, num steps)
        pred_ooi_joint = ooi_mask[None, ..., None] & pred_valid_mask[..., T_context:T_gt]  # (K, num agents, num steps)
        gt_shape = numpy_to_torch(
            gt_data_dict["decoder/current_agent_shape"][None], device=self.device
        ).expand(T_gt, -1, -1).transpose(0, 1)
        pred_shape = pred_data_dict["decoder/current_agent_shape"][:, None].expand(-1, T_pred, -1, -1).transpose(1, 2)

        # minSFDE
        with timer("minSFDE"):
            import pdb
            pdb.set_trace()
            self.minSFDE += torch.min(
                torch.sum(
                    torch.where(
                        gt_ooi_joint[None, ..., -1] & pred_ooi_joint[..., -1],
                        torch.linalg.norm(
                            numpy_to_torch(gt_data_dict["decoder/agent_position"],
                                           device=self.device)[None, T_gt - 1, :, :2] -
                            pred_data_dict["decoder/reconstructed_position"][:, T_gt - 1],
                            dim=-1
                        ), 0
                    ),
                    dim=-1
                ) / (gt_ooi_joint[None, ..., -1] & pred_ooi_joint[..., -1]).sum(dim=-1)
            )
        # FDD
        with timer("FDD"):
            # there doesn't appear to be an easy way to do this with cartesian product
            cur_FDD = None
            for i, j in itertools.product(range(K), range(K)):
                final_dist = torch.where(
                    pred_ooi_joint[..., -1],
                    torch.linalg.norm(
                        pred_data_dict["decoder/reconstructed_position"][i, T_gt - 1] -
                        pred_data_dict["decoder/reconstructed_position"][j, T_gt - 1],
                        dim=-1
                    ), 0
                )
                if cur_FDD == None:
                    cur_FDD = final_dist
                else:
                    cur_FDD = torch.maximum(cur_FDD, final_dist)
            self.FDD += cur_FDD.sum() / torch.pow(pred_ooi_joint[..., -1].sum(), 2)

        with timer("Kinematic Metrics"):
            gt_speed, gt_accel, gt_jerk = self._compute_kinematic_metrics(
                gt_data_dict["decoder/agent_velocity"].swapaxes(1, 0)
            )  # (N, T)
            pred_speed, pred_accel, pred_jerk = self._compute_kinematic_metrics(
                pred_data_dict["decoder/reconstructed_velocity"].transpose(1, 2)
            )  # (K, N, T)
            gt_speed = gt_speed[..., T_context:T_gt]
            gt_accel = gt_accel[..., T_context:T_gt]
            gt_jerk = gt_jerk[..., T_context:T_gt]
            pred_speed = pred_speed[..., T_context:T_gt]
            pred_accel = pred_accel[..., T_context:T_gt]
            pred_jerk = pred_jerk[..., T_context:T_gt]

        with timer("Collision Metrics"):
            # Following wosac_eval, fill in z with GT t = 10 data
            z_values = pred_data_dict["decoder/current_agent_position"][..., 2].unsqueeze(-1).expand(-1, -1, T_pred)

            def build_collision_data(candidate_agents, evaluate_agents):
                if isinstance(candidate_agents, torch.Tensor):
                    candidate_agents = candidate_agents.cpu().numpy()
                if isinstance(evaluate_agents, torch.Tensor):
                    evaluate_agents = evaluate_agents.cpu().numpy()
                if isinstance(candidate_agents, np.ndarray):
                    candidate_agents = candidate_agents.tolist()
                if isinstance(evaluate_agents, np.ndarray):
                    evaluate_agents = evaluate_agents.tolist()
                if not isinstance(candidate_agents, Iterable):
                    candidate_agents = [candidate_agents]
                if not isinstance(evaluate_agents, Iterable):
                    evaluate_agents = [evaluate_agents]
                candidate_agents = sorted(set(candidate_agents + evaluate_agents))
                candidate_agents_map = {v: k for k, v in enumerate(candidate_agents)}
                evaluate_agents_mask = np.zeros(len(candidate_agents), dtype=bool)
                evaluate_agents_mask[[candidate_agents_map[k] for k in evaluate_agents]] = 1
                return [
                    dict(
                        center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 0].T),
                        center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 1].T),
                        center_z=conv(z_values[k, candidate_agents]),
                        length=conv(pred_shape[k, candidate_agents, :, 0]),
                        width=conv(pred_shape[k, candidate_agents, :, 1]),
                        height=conv(pred_shape[k, candidate_agents, :, 2]),
                        heading=conv(pred_data_dict["decoder/reconstructed_heading"][k, :, candidate_agents].T),
                        valid=conv(pred_valid_mask[k, candidate_agents], dtype=tf.bool),
                        evaluated_object_mask=conv(evaluate_agents_mask, dtype=tf.bool)
                    ) for k in range(K)
                ]

            def get_dists(args_list):
                return torch.stack(
                    [
                        tf_to_torch(
                            interaction_features.compute_distance_to_nearest_object(**args_list[k]), device=self.device
                        ) for k in range(K)
                    ]
                )

            def calc_collision(dists, valid_masks):
                if type(valid_masks) == list:
                    valid_masks = torch.stack(valid_masks)
                collisions = torch.le(dists, interaction_features.COLLISION_DISTANCE_THRESHOLD)
                collisions = collisions[..., T_context:T_gt]
                valid_masks = valid_masks[..., T_context:T_gt]
                collision_rate = torch.min(
                    torch.any(collisions & valid_masks, dim=-1).double().sum(dim=-2) /
                    torch.any(valid_masks, dim=-1).double().sum(dim=-2)
                )
                return collisions, collision_rate

            def calc_collision_rate(candidate_agents, evaluate_agents):
                args = build_collision_data(candidate_agents, evaluate_agents)
                collisions, collision_rate = calc_collision(
                    get_dists(args), [
                        tf_to_torch(
                            tf.boolean_mask(args[k]["valid"], args[k]["evaluated_object_mask"], axis=0),
                            device=self.device
                        ) for k in range(K)
                    ]
                )
                return collisions, collision_rate

            pred_veh_collisions, veh_cr = calc_collision_rate(list(range(N)), gt_data_dict["decoder/labeled_agent_id"])
            self.veh_coll += veh_cr

            if "adv" in kwargs:  # assumes list of adv agents
                self.sdc_coll_adv_active = True
                self.sdc_coll_adv += calc_collision_rate(kwargs["adv"], pred_data_dict["decoder/sdc_index"])[-1]

            if "bv" in kwargs:  # assumes list of bv agents
                self.sdc_coll_bv_active = True
                self.sdc_coll_bv += calc_collision_rate(kwargs["bv"], pred_data_dict["decoder/sdc_index"])[-1]

            if "adv" in kwargs and "bv" in kwargs:
                self.adv_coll_bv_active = True
                self.adv_coll_bv += calc_collision_rate(kwargs["bv"], kwargs["adv"])[-1]

            map_feature = gt_data_dict["encoder/map_feature"]
            road_edges = [
                [map_pb2.MapPoint(x=map_feature[i, 0, 0], y=map_feature[i, 0, 1], z=map_feature[i, 0, 2])] + [
                    map_pb2.MapPoint(x=map_feature[i, j, 3], y=map_feature[i, j, 4], z=map_feature[i, j, 5])
                    for j in range(map_feature.shape[1]) if gt_data_dict['encoder/map_feature_valid_mask'][i, j]
                ]  # start point + end points
                for i in range(map_feature.shape[0]) if map_feature[i, 0, 15] == 1
            ]  # is boundary
            env_nearest_distances = torch.stack(
                [
                    tf_to_torch(
                        map_metric_features.compute_distance_to_road_edge(
                            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 0].T),
                            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 1].T),
                            center_z=conv(z_values[k]),
                            length=conv(pred_shape[k, ..., 0]),
                            width=conv(pred_shape[k, ..., 1]),
                            height=conv(pred_shape[k, ..., 2]),
                            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k].T),
                            valid=conv(pred_valid_mask[k], dtype=tf.bool),
                            evaluated_object_mask=conv(ooi_mask, dtype=tf.bool),
                            road_edge_polylines=road_edges,
                        ),
                        device=self.device
                    ) for k in range(K)
                ]
            )

            pred_env_collisions = torch.greater(env_nearest_distances, map_metric_features.OFFROAD_DISTANCE_THRESHOLD)
            pred_env_collisions = pred_env_collisions[..., T_context:T_gt]
            env_collision_rate = torch.min(
                pred_env_collisions.sum(dim=(-1, -2), dtype=torch.double) /
                pred_valid_mask[:, ooi_mask, T_context:T_gt].sum(dtype=torch.double, dim=(-1, -2))
            )

            self.env_coll += env_collision_rate

            # Debug: ground truth env collision rate
            # debug_env_nearest_distances = rconv(map_metric_features.compute_distance_to_road_edge(
            #     center_x=conv(gt_data_dict["decoder/agent_position"][..., 0].T),
            #     center_y=conv(gt_data_dict["decoder/agent_position"][..., 1].T),
            #     center_z=conv(gt_data_dict["decoder/agent_position"][..., 2].T),
            #     length=conv(gt_data_dict["decoder/agent_shape"][..., 0].T),
            #     width=conv(gt_data_dict["decoder/agent_shape"][..., 1].T),
            #     height=conv(gt_data_dict["decoder/agent_shape"][..., 2].T),
            #     heading=conv(gt_data_dict["decoder/agent_heading"].T),
            #     valid=conv(gt_data_dict["decoder/agent_valid_mask"].T),
            #     evaluated_object_mask=conv(vehicle_mask),
            #     road_edge_polylines=road_edges,
            # ))

            # debug_env_collisions = torch.le(debug_env_nearest_distances, map_metric_features.OFFROAD_DISTANCE_THRESHOLD)
            # debug_env_collisions = debug_env_collisions[..., 11:91]

            # debug_env_collision_rate = torch.min(debug_env_collisions.double().mean(dim=(-1, -2)))
            # print(debug_env_collision_rate)

            self.coll_vel += torch.min(
                torch.nan_to_num(torch.where(pred_env_collisions | pred_veh_collisions, pred_speed[:, ooi_mask],
                                             0)).amax(dim=(-1, -2))
            )
            self.acc += torch.min(torch.abs(torch.nan_to_num(pred_accel[:, 0])).amax(dim=-1))
            self.jerk += torch.min(torch.abs(torch.nan_to_num(pred_jerk[:, 0])).amax(dim=-1))

        with timer("Time to Collision"):
            gt_ttc = tf_to_torch(
                interaction_features.compute_time_to_collision_with_object_in_front(
                    center_x=conv(gt_data_dict["decoder/agent_position"][..., 0].T),
                    center_y=conv(gt_data_dict["decoder/agent_position"][..., 1].T),
                    length=conv(gt_shape[..., 0]),
                    width=conv(gt_shape[..., 1]),
                    heading=conv(gt_data_dict["decoder/agent_heading"].T),
                    valid=conv(gt_valid_mask, dtype=tf.bool),
                    evaluated_object_mask=conv(ooi_mask, dtype=tf.bool),
                    seconds_per_step=self.SECONDS_PER_STEP
                ),
                device=self.device
            )
            pred_ttc = torch.stack(
                [
                    tf_to_torch(
                        interaction_features.compute_time_to_collision_with_object_in_front(
                            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 0].T),
                            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 1].T),
                            length=conv(pred_shape[k, ..., 0]),
                            width=conv(pred_shape[k, ..., 1]),
                            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k].T),
                            valid=conv(pred_data_dict["decoder/reconstructed_valid_mask"][k].T, dtype=tf.bool),
                            evaluated_object_mask=conv(ooi_mask, dtype=tf.bool),
                            seconds_per_step=self.SECONDS_PER_STEP
                        ),
                        device=self.device
                    ) for k in range(K)
                ]
            )
            gt_ttc = gt_ttc[..., T_context:T_gt]
            pred_ttc = pred_ttc[..., T_context:T_gt]

        with timer("Histograms"):
            gt_speed_hist, gt_speed_bins = torch.histogram(
                torch.clip(
                    gt_speed[gt_ooi_joint & ~gt_speed.isnan()], self.jsd_config["vel"]["min_val"],
                    self.jsd_config["vel"]["max_val"]
                ).cpu(),
                self.jsd_config["vel"]["num_bins"],
                density=False
            )
            # .cpu() since histogram doesn't support cuda backend
            pred_speed_hist, pred_speed_bins = torch.histogram(
                torch.clip(
                    pred_speed[pred_ooi_joint & ~pred_speed.isnan()], self.jsd_config["vel"]["min_val"],
                    self.jsd_config["vel"]["max_val"]
                ).cpu(),
                self.jsd_config["vel"]["num_bins"],
                density=False
            )
            gt_accel_hist, gt_accel_bins = torch.histogram(
                torch.clip(
                    gt_accel[gt_ooi_joint & ~gt_accel.isnan()], self.jsd_config["acc"]["min_val"],
                    self.jsd_config["acc"]["max_val"]
                ).cpu(),
                self.jsd_config["acc"]["num_bins"],
                density=False
            )
            pred_accel_hist, pred_accel_bins = torch.histogram(
                torch.clip(
                    pred_accel[pred_ooi_joint & ~pred_accel.isnan()], self.jsd_config["acc"]["min_val"],
                    self.jsd_config["acc"]["max_val"]
                ).cpu(),
                self.jsd_config["acc"]["num_bins"],
                density=False
            )
            gt_ttc_hist, gt_ttc_bins = torch.histogram(
                torch.clip(
                    gt_ttc[gt_valid_mask[ooi_mask, T_context:T_gt] & ~gt_ttc.isnan()],
                    self.jsd_config["ttc"]["min_val"], self.jsd_config["ttc"]["max_val"]
                ).cpu(),
                self.jsd_config["ttc"]["num_bins"],
                density=False
            )
            pred_ttc_hist, pred_ttc_bins = torch.histogram(
                torch.clip(
                    pred_ttc[pred_valid_mask[:, ooi_mask, T_context:T_gt] & ~pred_ttc.isnan()],
                    self.jsd_config["ttc"]["min_val"], self.jsd_config["ttc"]["max_val"]
                ).cpu(),
                self.jsd_config["ttc"]["num_bins"],
                density=False
            )
            # visualize histograms for debug
            plt.clf()
            plt.hist(gt_speed_bins[:-1], bins=gt_speed_bins, weights=gt_speed_hist, density=False)
            plt.savefig(f"gt_speed{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()
            plt.hist(pred_speed_bins[:-1], bins=pred_speed_bins, weights=pred_speed_hist, density=False)
            plt.savefig(f"pred_speed{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()
            plt.hist(gt_accel_bins[:-1], bins=gt_accel_bins, weights=gt_accel_hist, density=False)
            plt.savefig(f"gt_accel{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()
            plt.hist(pred_accel_bins[:-1], bins=pred_accel_bins, weights=pred_accel_hist, density=False)
            plt.savefig(f"pred_accel{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()
            plt.hist(gt_ttc_bins[:-1], bins=gt_ttc_bins, weights=gt_ttc_hist, density=False)
            plt.savefig(f"gt_ttc{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()
            plt.hist(pred_ttc_bins[:-1], bins=pred_ttc_bins, weights=pred_ttc_hist, density=False)
            plt.savefig(f"pred_ttc{SCENE_IDX}.png", bbox_inches='tight')
            plt.clf()

        with timer("JSD"):
            speed_jsd = jsd(gt_speed_hist, pred_speed_hist)
            acc_jsd = jsd(gt_accel_hist, pred_accel_hist)
            ttc_jsd = jsd(gt_ttc_hist, pred_ttc_hist)
            self.vel_jsd += speed_jsd
            self.acc_jsd += acc_jsd
            self.ttc_jsd += ttc_jsd

    def _compute_kinematic_metrics(self, vel):
        if type(vel) == np.ndarray:
            vel = numpy_to_torch(vel, device=self.device)
        speed = torch.linalg.norm(vel, axis=-1)
        accel = self._central_diff(speed, pad_value=torch.nan) / self.SECONDS_PER_STEP
        jerk = self._central_diff(accel, pad_value=torch.nan) / self.SECONDS_PER_STEP
        return speed, accel, jerk

    def _central_diff(self, tensor, pad_value=torch.nan):
        pad_shape = (*tensor.shape[:-1], 1)
        pad_tensor = torch.ones(pad_shape, device=self.device) * pad_value
        diff_t = (tensor[..., 2:] - tensor[..., :-2]) / 2
        return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)

    def aggregate(self):
        # TODO: write some "aggregate" function to compute the metrics
        self.metrics["# Scenarios"] = self.scenario_count
        self.metrics["minSFDE"] = self.minSFDE / self.scenario_count
        self.metrics["FDD"] = self.FDD / self.scenario_count
        self.metrics["Vel. JSD"] = self.vel_jsd / self.scenario_count
        self.metrics["Acc. JSD"] = self.acc_jsd / self.scenario_count
        self.metrics["TTC JSD"] = self.ttc_jsd / self.scenario_count
        self.metrics["Env CR"] = self.env_coll / self.scenario_count
        self.metrics["Veh CR"] = self.veh_coll / self.scenario_count
        self.metrics["ADV+SDC CR"] = self.sdc_coll_adv / self.scenario_count
        self.metrics["ADV+BV CR"] = self.adv_coll_bv / self.scenario_count
        self.metrics["SDC+BV CR"] = self.sdc_coll_bv / self.scenario_count
        self.metrics["Avg. Max Coll Vel."] = self.coll_vel / self.scenario_count
        self.metrics["Avg. Max Acc."] = self.acc / self.scenario_count
        self.metrics["Avg. Max Jerk"] = self.jerk / self.scenario_count

    def print(self):
        # TODO(xuanhao): Maybe implement a handy function to print output
        pass
        # self.precision_safety_0 = self.TP_safety_0 / (self.TP_safety_0 + self.FP_safety_0)
        # self.recall_safety_0 = self.TP_safety_0 / (self.TP_safety_0 + self.FN_safety_0)
        #
        # print("=====================================")
        # print(
        #     "precision_safety_0: {:.5f} = {} / {}".format(
        #         self.precision_safety_0, self.TP_safety_0, self.TP_safety_0 + self.FP_safety_0
        #     )
        # )
        # print(
        #     "recall_safety_0: {:.5f} = {} / {}".format(
        #         self.recall_safety_0, self.TP_safety_0, self.TP_safety_0 + self.FN_safety_0
        #     )
        # )
        # print("=====================================")
        #
        # self.precision_safety_1 = self.TP_safety_1 / (self.TP_safety_1 + self.FP_safety_1)
        # self.recall_safety_1 = self.TP_safety_1 / (self.TP_safety_1 + self.FN_safety_1)
        # print(
        #     "precision_safety_1: {:.5f} = {} / {}".format(
        #         self.precision_safety_1, self.TP_safety_1, self.TP_safety_1 + self.FP_safety_1
        #     )
        # )
        # print(
        #     "recall_safety_1: {:.5f} = {} / {}".format(
        #         self.recall_safety_1, self.TP_safety_1, self.TP_safety_1 + self.FN_safety_1
        #     )
        # )
        #
        # print("=====================================")
        # self.precision_macro = (self.precision_safety_0 + self.precision_safety_1) / 2
        # self.recall_macro = (self.recall_safety_0 + self.recall_safety_1) / 2
        # self.f1_macro = 2 * self.precision_macro * self.recall_macro / (self.precision_macro + self.recall_macro)
        # print("precision_macro:", self.precision_macro)
        # print("recall_macro:", self.recall_macro)
        # print("f1_macro:", self.f1_macro)
        # print("=====================================")

        self.aggregate()
        for k in self.display_keys:
            if k == "ADV+SDC CR" and not self.sdc_coll_adv_active:
                continue
            if k == "ADV+BV CR" and not self.adv_coll_bv_active:
                continue
            if k == "SDC+BV CR" and not self.sdc_coll_bv_active:
                continue
            print(f"{k}: {self.metrics[k]:.5f} {self.metric_units[k]}")


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1031_midgpt.yaml")
def debug_run_model(config):
    import os
    global SCENE_IDX
    path = "../ckpt/last.ckpt"
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.pretrain = "../ckpt/last.ckpt"
    config.BACKWARD_PREDICTION = True  # <<<
    config.ADD_CONTOUR_RELATION = True
    config.DATA.TRAINING_DATA_DIR = "waymo_validation_interactive_500"  #"data/20scenarios"
    config.DATA.TEST_DATA_DIR = "waymo_validation_interactive_500"  #"data/20scenarios"

    omegaconf.OmegaConf.set_struct(config, True)

    model = utils.get_model(config, device="cuda")
    device = model.device

    test_dataset = InfgenDataset(config, "test")
    from bmt.tokenization import get_tokenizer
    tokenizer = get_tokenizer(config)

    evaluator = Evaluator()

    num_scenario = 100
    count = 0
    num_modes = 1
    for raw_data_dict in tqdm.tqdm(test_dataset):
        data_dict = copy.deepcopy(raw_data_dict)

        # Get the torch version of the data.
        input_data_dict = {
            k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) and "track_name" not in k else v
            for k, v in data_dict.items()
        }

        # Extend the batch dim:
        input_data_dict = {
            k: utils.expand_for_modes(v.unsqueeze(0), num_modes=num_modes) if isinstance(v, torch.Tensor) else v
            for k, v in input_data_dict.items()
        }
        input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
        if config.BACKWARD_PREDICTION:
            input_data_dict["in_backward_prediction"] = torch.tensor([False] * num_modes, dtype=bool).to(device)

        with torch.no_grad():
            ar_func = model.model.autoregressive_rollout
            output_dict = ar_func(
                input_data_dict,
                num_decode_steps=None,
                sampling_method=config.SAMPLING.SAMPLING_METHOD,
                temperature=config.SAMPLING.TEMPERATURE,
            )
        output_dict = tokenizer.detokenize(
            output_dict,
            detokenizing_gt=False,
            backward_prediction=False,
        )

        # Just for debug... Plot first mode.
        output_dict_numpy = {
            k: (v[0].cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }
        # plot_pred(output_dict_numpy, show=True, path=f"plot_output{SCENE_IDX}.png")
        SCENE_IDX += 1
        evaluator.add(raw_data_dict, output_dict)
        evaluator.print()

    # evaluator.aggregate()
    evaluator.print()
    print("End of evaluating ")


def _get_mode(data, mode):
    ret = {}
    for k, v in output_dict.items():
        if isinstance(v, np.ndarray) and len(v) == num_modes:
            ret[k] = v[mode]
        else:
            ret[k] = v
    return ret


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1031_midgpt.yaml")
def evaluate_scgen(config):
    from bmt.utils.safety_critical_generation_utils import _overwrite_data_given_agents_not_ooi, get_ego_edge_points, get_ego_edge_points_old, post_process_adv_traj, _overwrite_data_given_agents_ooi, _overwrite_data_given_agents, set_adv, run_backward_prediction_with_teacher_forcing
    from bmt.utils import utils
    import copy
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
    import torch
    model = model.to("cuda")
    device = model.device
    from bmt.tokenization import get_tokenizer
    tokenizer = get_tokenizer(config)

    evaluator = Evaluator()
    num_modes = 1
    count = 0
    num_scenario = 100

    pbar = tqdm.tqdm(total=500, desc="Scenario")
    # for count, raw_data in enumerate(datamodule.val_dataloader()):
    dataset = InfgenDataset(config, "test")
    for raw_data_dict in tqdm.tqdm(dataset):
        # if count >= num_scenario:
        #     break
        flip_heading_accordingly = True
        backward_prediction = True

        data_dict = raw_data_dict
        raw_data_dict = copy.deepcopy(data_dict)

        # Create a new ADV in the data so backward prediction will help us generate it.
        # TODO: If we also want to TF ego, then we should not overwrite ego data.
        sdc_id = data_dict["decoder/sdc_index"]

        data_dict, adv_id = set_adv(data_dict)
        # data_dict = create_new_adv(data_dict)
        # pdb.set_trace()

        input_data_dict = utils.numpy_to_torch(data_dict, device=device)
        original_data_dict_tensor = copy.deepcopy(input_data_dict)
        input_data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in input_data_dict.items()}

        all_agents = input_data_dict["decoder/agent_id"][0]
        not_tf_ids = all_agents[all_agents != 0]

        for iteration in range(1):
            print("====================================")
            print("Iteration: ", iteration)
            print("====================================")

            backward_input_dict = copy.deepcopy(input_data_dict)

            backward_output_dict = run_backward_prediction_with_teacher_forcing(
                model=model,
                config=config,
                backward_input_dict=backward_input_dict,
                tokenizer=tokenizer,

                # TODO: Which to TF?
                not_teacher_forcing_ids=not_tf_ids
            )

            # pdb.set_trace()

            # ===== Only used for vis =====
            # from infgen.utils.utils import numpy_to_torch

            # backward_output_dict_numpy = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in backward_output_dict.items()}

            # backward_output_dict_numpy = {
            #     k: (v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v)
            #     for k, v in backward_output_dict.items()
            # }

        # original_data_dict = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in original_data_dict.items()}
        all_agents = raw_data_dict["decoder/agent_id"]
        sdc_id = raw_data_dict["decoder/sdc_index"]
        all_agents_except_sdc = all_agents[all_agents != sdc_id]
        evaluator.add(original_data_dict_tensor, backward_output_dict, adv=[adv_id], bv=all_agents_except_sdc)
        evaluator.print()
    print("====================================")
    evaluator.print()
    print("End of evaluating SCGEN")


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


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "cfgs"), config_name="1031_midgpt.yaml")
def debug_eval_CAT(config):
    import os
    from bmt.dataset.scenarionet_utils import overwrite_gt_to_pred_field
    omegaconf.OmegaConf.set_struct(config, False)
    config.PREPROCESSING.keep_all_data = True
    config.pretrain = "../ckpt/last.ckpt"
    config.BACKWARD_PREDICTION = True 
    config.ADD_CONTOUR_RELATION = True
    config.DATA.TEST_DATA_DIR = "validation_interactive_500/"
    CAT_DIR = "cat_adv_validation_interactive/"
    omegaconf.OmegaConf.set_struct(config, True)

    test_dataset = InfgenDataset(config, "test")
    evaluator = Evaluator()

    model = utils.get_model(config, device="cuda")
    device = model.device

    num_scenario = 500
    num_modes = 1
    count = 0

    import pickle
    import os
    with open(os.path.join(CAT_DIR, 'dataset_summary.pkl'), "rb") as f:
        cat_summary = pickle.load(f)
    f.close()
    all_cat_scenarios = cat_summary.keys()

    for raw_data_dict in tqdm.tqdm(test_dataset):
        if count >= num_scenario:
            break
        input_dict = copy.deepcopy(raw_data_dict)
        sid = input_dict["metadata/scenario_id"]
        cat_file_name = f"sd_reconstructed_v0_{sid}.pkl"
        if cat_file_name not in all_cat_scenarios:
            continue

        input_data_dict = numpy_to_torch(input_dict, device=device)
        double_keys = [
            "decoder/agent_position", 'decoder/agent_heading', 'decoder/agent_velocity',
            "decoder/reconstructed_position", "decoder/reconstructed_heading", "decoder/reconstructed_velocity",
            "decoder/agent_shape", "decoder/current_agent_shape", "decoder/current_agent_position"
        ]
        input_data_dict = convert_tensors_to_double(input_data_dict, double_keys)

        with open(os.path.join(CAT_DIR, cat_file_name), 'rb') as f:
            cat_data = pickle.load(f)
        f.close()

        cat_data_dict = preprocess_scenario_description_for_motionlm(
            scenario=cat_data, config=config, in_evaluation=True, keep_all_data=True, cache=None
        )

        output_dict = overwrite_gt_to_pred_field(cat_data_dict)
        output_data_dict = numpy_to_torch(output_dict, device=device)
        output_data_dict = convert_tensors_to_double(output_data_dict, double_keys)
        output_data_dict = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in output_data_dict.items()
        }

        evaluator.add(input_data_dict, output_data_dict)
        evaluator.print()
        count += 1

    evaluator.print()
    print("End of evaluation of CAT generation")


if __name__ == '__main__':
    # debug_eval_CAT()
    evaluate_scgen()
    # debug_run_model()
