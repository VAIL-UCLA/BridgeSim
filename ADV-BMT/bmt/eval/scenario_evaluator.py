import dataclasses
import itertools
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import interaction_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import map_metric_features
from bmt.utils import wrap_to_pi, rotate
from bmt.utils import utils

def transform_to_global_coordinate(data_dict):
    assert "decoder/reconstructed_position" in data_dict

    if data_dict["decoder/reconstructed_position"].ndim == 3: # no batch dim
        map_center = data_dict["metadata/map_center"].reshape(-1, 1, 3)  # (1,1,3)
        T, N, _ = data_dict["decoder/reconstructed_position"].shape

        expanded_mask = data_dict["decoder/reconstructed_valid_mask"][:, :, None]
        data_dict["decoder/reconstructed_position"] += map_center * expanded_mask
        
    else: 
        assert data_dict["decoder/reconstructed_position"].ndim == 4
        B, T, N, _ = data_dict["decoder/reconstructed_position"].shape
        map_center = data_dict["metadata/map_center"].reshape(-1, 1, 1, 3)  # (1,1,3)
        map_center = map_center[...,:2]
        expanded_mask = data_dict["decoder/reconstructed_valid_mask"][..., None]
        data_dict["decoder/reconstructed_position"] += map_center * expanded_mask

    return data_dict


def detect_env_collision(contour_list1, mask1, lineString):
    collision_detected = []

    for i in range(len(contour_list1)):
        if mask1[i]:
            agent_poly = Polygon(contour_list1[i])

            if agent_poly.intersects(lineString):
                collision_detected.append(True)
            else:
                collision_detected.append(False)
        else:
            collision_detected.append(False)

    return collision_detected


from bmt.dataset.preprocess_action_label import cal_polygon_contour
from shapely.geometry import Polygon, LineString

def customized_env_collision_rate(data_dict, track_agent_indices=None):
    # step 1: create a line for all polyline
    map_feature = data_dict["vis/map_feature"]
    assert map_feature.ndim == 3  # This is unbatched.

    road_edges = []
    for i in range(map_feature.shape[0]):
        # For each map feature
        if map_feature[i, 0, 15] == 1:
            current_polyline = []
            for j in range(map_feature.shape[1]):
                if data_dict['encoder/map_feature_valid_mask'][i, j]:
                   current_polyline.append((map_feature[i, j, 0], map_feature[i, j, 1]))

            road_edges.append(current_polyline)

    # step 2: for each agent traj, create a contour 
    if not track_agent_indices:
        track_agent_indicies = data_dict["decoder/agent_id"]


    # step 3: detect collision
    agent_env_collisions = np.zeros_like(track_agent_indicies)
    for agent1_id in track_agent_indicies:
        traj = data_dict["decoder/reconstructed_position"][:91, agent1_id, :2]  # (91, 3)
        length = data_dict["decoder/agent_shape"][10, agent1_id, 0]
        width = data_dict["decoder/agent_shape"][10, agent1_id, 1]
        theta = data_dict['decoder/reconstructed_heading'][:91, agent1_id]  # (91, ) # in pi
        mask = data_dict['decoder/reconstructed_valid_mask'][:91, agent1_id]  # (91,)
        contour = cal_polygon_contour(traj[:, 0], traj[:, 1], theta, width, length)

        for j in range(len(road_edges)):
            line = LineString(road_edges[j])

            collision_detected = detect_env_collision(contour, mask, line)
            if any(collision_detected):
                    # print(f"Collision between {i} and {j} happen at step: {np.array(collision_detected).nonzero()}")
                    agent_env_collisions[agent1_id] = 1  # Label collisions for OOIs now. Later we will build a larger dict.
                    break
            

    print("agent_env_collisions", agent_env_collisions)
    return agent_env_collisions


def get_dists(args_list, device):
    return torch.stack(
        [
            tf_to_torch(interaction_features.compute_distance_to_nearest_object(**args_list[k]), device=device)
            for k in range(len(args_list))
        ]
    )


def build_collision_data(*, pred_data_dict, pred_shape, candidate_agents, evaluate_agents, z_values):
    candidate_agents = sorted(set(candidate_agents + evaluate_agents))
    candidate_agents_map = {int(v): k for k, v in enumerate(candidate_agents)}
    evaluate_agents_mask = np.zeros(len(candidate_agents), dtype=bool)

    assert evaluate_agents_mask.ndim == 1
    for k in evaluate_agents:
        evaluate_agents_mask[candidate_agents_map[int(k)]] = 1

    K = pred_shape.shape[0]

    return [
        dict(
            center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 0].T),
            center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, :, candidate_agents, 1].T),
            center_z=conv(z_values[k, candidate_agents]),
            length=conv(pred_shape[k, candidate_agents, :, 0]),
            width=conv(pred_shape[k, candidate_agents, :, 1]),
            height=conv(pred_shape[k, candidate_agents, :, 2]),
            heading=conv(pred_data_dict["decoder/reconstructed_heading"][k, :, candidate_agents].T),
            valid=conv(pred_data_dict["decoder/reconstructed_valid_mask"][k, :, candidate_agents].T, dtype=tf.bool),
            evaluated_object_mask=conv(evaluate_agents_mask, dtype=tf.bool)
        ) for k in range(K)
    ]


def calc_collision(*, dists, valid_masks, T_context, T_gt):
    if type(valid_masks) == list:
        valid_masks = torch.stack(valid_masks)
    

    threshold = interaction_features.COLLISION_DISTANCE_THRESHOLD
    threshold = 1e-4
    collisions = torch.le(dists, threshold)
    collisions = collisions[..., T_context:T_gt]
    valid_masks = valid_masks[..., T_context:T_gt]

    collisions = collisions & valid_masks  # Shape: (B, N, T)
    # Number of agents that has coll.
    collisions_count = torch.any(collisions, dim=-1).double().sum(dim=-1)  # Shape: (B,)
    valid_agent_for_collision = torch.any(valid_masks, dim=-1)  # Shape: (B, N)

    # Ratio of agents that has coll.
    mode_cr = collisions_count / valid_agent_for_collision.sum(dim=-1)

    assert mode_cr.ndim == 1
    return collisions, mode_cr


def calc_collision_rate(
    *, pred_data_dict, pred_shape, candidate_agents, evaluate_agents, device, T_gt, T_context, z_values
):

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

    args = build_collision_data(
        pred_data_dict=pred_data_dict,
        pred_shape=pred_shape,
        candidate_agents=candidate_agents,
        evaluate_agents=evaluate_agents,
        z_values=z_values
    )
    dist = get_dists(args, device=device)

    def _get_valid_masks(candidate_agents, evaluate_agents):

        candidate_agents = sorted(set(candidate_agents + evaluate_agents))
        candidate_agents_map = {int(v): k for k, v in enumerate(candidate_agents)}
        evaluate_agents_mask = np.zeros(len(candidate_agents), dtype=bool)
        assert evaluate_agents_mask.ndim == 1
        for k in evaluate_agents:
            evaluate_agents_mask[candidate_agents_map[int(k)]] = 1
        candidate_valid_mask = pred_data_dict["decoder/reconstructed_valid_mask"][:, T_context:T_gt, candidate_agents]
        evaluate_valid_mask = candidate_valid_mask[:, :, evaluate_agents_mask]
        return evaluate_valid_mask.swapaxes(1, 2)

    pred_veh_collisions, veh_cr_mode = calc_collision(
        dists=dist, valid_masks=_get_valid_masks(candidate_agents, evaluate_agents), T_context=T_context, T_gt=T_gt
    )
    return pred_veh_collisions, veh_cr_mode


def print_type_and_dtype(name, tensor):
    print(f"{name} - Type: {type(tensor)}, Dtype: {getattr(tensor, 'dtype', 'N/A')}")


def conv(tensor, dtype=tf.float32):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu()
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


@contextmanager
def timer(task_name: str):
    start = perf_counter()
    yield
    prof_t = perf_counter() - start
    if TIMER:
        print(f"{task_name}: {prof_t:.5f}")


@dataclasses.dataclass
class Metrics:

    scenario_count: int = 0
    sdc_coll_scenario_count: int = 0
    veh_coll_scenario_count: int = 0

    # Diversity
    sfde_avg: float = 0.0
    sade_avg: float = 0.0
    sfde_min: float = 0.0  # (supervised) avg over scenarios: minimum over all modes: average of L2 error of final positions of all agents
    sade_min: float = 0.0
    ssde_min: float = 0.0
    ssde_avg: float = 0.0


    skipped_sfde_avg: float = 0.0
    skipped_sade_avg: float = 0.0
    skipped_sfde_min: float = 0.0  # (supervised) avg over scenarios: minimum over all modes: average of L2 error of final positions of all agents
    skipped_sade_min: float = 0.0
    skipped_ssde_min: float = 0.0
    skipped_ssde_avg: float = 0.0

    sdd: float = 0.0  # (unsupervised) avg over scenarios: average over all agents: maximum L2 distance in first position of that agent between generated modes
    fdd: float = 0.0  # (unsupervised) avg over scenarios: average over all agents: maximum L2 distance in final position of that agent between generated modes
    add: float = 0.0  # (unsupervised) avg over scenarios: average over all agents: average across time: maximum L2 distance between generated modes of that agent at that time
    # Xuanhao: In MixSim paper, they used squared norm of distance, but maybe they meant L2 norm not squared norm?
    # Unit given in AdvDiffuser for FDD is m not m^2, so I am using L2 norm here

    # Distribution Realism
    vel_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: velocity JS divergence
    acc_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: acceleration JS divergence
    ttc_jsd: float = 0.0  # avg over scenarios: build histogram across agents, modes, timestamps: time to collision JS divergence

    # Common Sense
    # env_coll_max: float = 0.0  # offroad
    # env_coll_min: float = 0.0  # offroad
    # env_coll_avg: float = 0.0  # offroad

    veh_coll_max: float = 0.0  # collision rate
    veh_coll_min: float = 0.0  # collision rate
    veh_coll_avg: float = 0.0  # collision rate

    # SDC-ADV coll
    sdc_adv_coll_max: float = 0.0  # collision rate
    sdc_adv_coll_min: float = 0.0  # collision rate
    sdc_adv_coll_avg: float = 0.0  # collision rate

    sdc_bv_coll_max: float = 0.0  # collision rate
    sdc_bv_coll_min: float = 0.0  # collision rate
    sdc_bv_coll_avg: float = 0.0  # collision rate

    adv_bv_coll_max: float = 0.0  # collision rate
    adv_bv_coll_min: float = 0.0  # collision rate
    adv_bv_coll_avg: float = 0.0  # collision rate

    coll_vel_maxagent_avg: float = 0.0  # collision velocity max over agents, avg over modes
    coll_vel_maxagent_max: float = 0.0  # collision velocity max over agents, max over modes
    coll_vel_maxagent_min: float = 0.0  # collision velocity max over agents, min over modes
    coll_vel_sdc_avg: float = 0.0  # collision velocity only for SDC
    coll_vel_sdc_max: float = 0.0  # collision velocity only for SDC, max over modes
    coll_vel_sdc_min: float = 0.0  # collision velocity only for SDC, min over modes

    # no clue what collision JSD means so not calculating it for now

    # AV comfortable
    sdc_acc_maxtime_avg: float = 0.0
    sdc_acc_maxtime_min: float = 0.0
    sdc_acc_maxtime_max: float = 0.0
    sdc_acc_avgtime_avg: float = 0.0
    sdc_acc_avgtime_min: float = 0.0
    sdc_acc_avgtime_max: float = 0.0

    sdc_jerk_maxtime_avg: float = 0.0
    sdc_jerk_maxtime_min: float = 0.0
    sdc_jerk_maxtime_max: float = 0.0
    sdc_jerk_avgtime_avg: float = 0.0
    sdc_jerk_avgtime_min: float = 0.0
    sdc_jerk_avgtime_max: float = 0.0

    customized_max_sdc_adv_coll: float = 0.0
    customized_max_sdc_bv_coll: float = 0.0
    customized_max_adv_bv_coll: float = 0.0

    customized_min_sdc_adv_coll: float = 0.0
    customized_min_sdc_bv_coll: float = 0.0
    customized_min_adv_bv_coll: float = 0.0

    customized_avg_sdc_adv_coll: float = 0.0
    customized_avg_sdc_bv_coll: float = 0.0
    customized_avg_adv_bv_coll: float = 0.0

    customized_avg_overall_coll: float = 0.0

    customized_all_agent_coll: float = 0.0

    def clean(self):
        # If the entry is tensor, drop it to float.
        for k, v in dataclasses.asdict(self).items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.item())

    def aggregate(self):
        self.clean()

        # Get all metrics
        all_metrics = dataclasses.asdict(self)
        for k, v in all_metrics.items():
            if k.startswith("coll_vel_sdc"):
                if self.sdc_coll_scenario_count > 0:
                    all_metrics[k] = v / self.sdc_coll_scenario_count
                else:
                    all_metrics[k] = torch.nan

            elif k.startswith("coll_vel_maxagent"):
                if self.veh_coll_scenario_count > 0:
                    all_metrics[k] = v / self.veh_coll_scenario_count
                else:
                    all_metrics[k] = torch.nan

            elif k != "scenario_count":
                all_metrics[k] = v / self.scenario_count
        return all_metrics


class Evaluator:
    SECONDS_PER_STEP = 0.1

    def __init__(self, CR_mode="mean", key_metrics_only=False, start_metrics_only=False):
        assert CR_mode in ["min", "max", "mean"]
        self.CR_mode = CR_mode
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

            # From WOSAC: https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto#L80C1-L89C2
            "ttc": {
                "min_val": 0.0,
                "max_val": 5.0,
                "num_bins": 10
            }
        }

        self.metrics = Metrics()
        self.key_metrics_only = key_metrics_only
        self.start_metrics_only = start_metrics_only



    def filter_static_agents(self, gt_data_dict):
        # return a mask for all static agetns (GT traj in both x and y less than 5m)

        mask = torch.zeros_like(gt_data_dict["decoder/agent_id"], dtype=torch.bool)

        for id in gt_data_dict["decoder/agent_id"]:
            traj = gt_data_dict["decoder/agent_position"][:,id][gt_data_dict["decoder/agent_valid_mask"][:,id],:2]
            diffs = traj[0] - traj[-1] # calcualte the difference of start and end index

            dist = torch.norm(diffs, dim=-1)

            if dist < 5:
                mask[id] = 1

        return mask


    def add(self, gt_data_dict, pred_data_dict, adv_list, bv_list, device=None):

        self.metrics.scenario_count += 1

        T_gt = gt_data_dict["decoder/agent_position"].shape[0]
        T_context = 0

        T_pred = pred_data_dict["decoder/reconstructed_position"].shape[1]
        B = K = pred_data_dict["decoder/reconstructed_position"].shape[0]
        N = gt_data_dict["decoder/agent_position"].shape[1]

        vehicle_mask = numpy_to_torch(gt_data_dict["decoder/agent_type"] == 1, device=device)  # (num agents)
        static_agent_mask = self.filter_static_agents(gt_data_dict)

        ooi_mask = torch.zeros_like(vehicle_mask, dtype=torch.bool, device=device)
        # ooi_mask[(gt_data_dict["decoder/object_of_interest_id"])] = 1 
        # ooi_mask[(gt_data_dict["decoder/sdc_id"])] = 1 # now only predict OOI
        ooi_mask[(gt_data_dict["decoder/agent_id"])] = 1  # (num agents)

        gt_valid_mask = numpy_to_torch(
            gt_data_dict["decoder/agent_valid_mask"], device=device
        ).T  # (num agents, num steps)
        pred_valid_mask = pred_data_dict["decoder/reconstructed_valid_mask"].transpose(
            1, 2
        )  # (K, num agents, num steps)
        # joint_mask = vehicle_mask.unsqueeze(-1) & valid_mask # (num agents, num steps)
        # gt_ooi_joint = ooi_mask.unsqueeze(-1) & gt_valid_mask[..., T_context:T_gt]  # (num agents, num steps)
        # pred_ooi_joint = ooi_mask[None, ..., None] & pred_valid_mask[..., T_context:T_gt]  # (K, num agents, num steps)

        gt_ooi_joint = gt_valid_mask[..., T_context:T_gt]  # (num agents, num steps)
        pred_ooi_joint = pred_valid_mask[..., T_context:T_gt]  # (K, num agents, num steps)

        gt_shape = numpy_to_torch(
            gt_data_dict["decoder/current_agent_shape"][None], device=device
        ).expand(T_gt, -1, -1).transpose(0, 1)
        pred_shape = pred_data_dict["decoder/current_agent_shape"][:, None].expand(K, T_pred, -1, -1).transpose(1, 2)

        sdc_index = int(gt_data_dict["decoder/sdc_index"])
        sdc_index_in_ooi = list(gt_data_dict["decoder/agent_id"]).index(sdc_index)

        # minSFDE
        if not self.start_metrics_only:
            with timer("minSFDE"):
                gt_pos = numpy_to_torch(gt_data_dict["decoder/agent_position"], device=device)[None, ..., :2]
                pred_pos = pred_data_dict["decoder/reconstructed_position"][:, :T_gt]

                gt_valid = gt_ooi_joint[None]
                gt_valid_skipped = gt_valid[:,:,::5]

                last_valid_ind = gt_valid.cumsum(dim=-1).argmax(dim=-1)
                last_valid_ind_skipped = gt_valid_skipped.cumsum(dim=-1).argmax(dim=-1)

                error = torch.linalg.norm(gt_pos - pred_pos, dim=-1)
                assert error.ndim == 3
                assert error.shape[0] == B

                # --- minSFDE: minimum end-of-sequence displacement error ---
                last_valid_ind = last_valid_ind.unsqueeze(0).expand(B, 1, N)
                last_valid_ind_skipped = last_valid_ind_skipped.unsqueeze(0).expand(B, 1, N)

                assert last_valid_ind.shape == (B, 1, N)
                fde = torch.gather(error, 1, last_valid_ind).squeeze(1)  # shape: B, N
                fde_skipped = torch.gather(error, 1, last_valid_ind_skipped * 5).squeeze(1)  # shape: B, N

                assert fde.shape[0] == B
                agent_valid = gt_valid.any(-1).expand(B, N)  # shape: B, N
                agent_valid_skipped = gt_valid_skipped.any(-1).expand(B, N) 

                sfde = (fde * agent_valid).sum(-1) / agent_valid.sum(-1)
                sfde_skipped = (fde_skipped * agent_valid_skipped).sum(-1) / agent_valid_skipped.sum(-1)

                assert sfde.ndim == 1
                assert sfde.shape[0] == B
                self.metrics.sfde_min += sfde.min()
                self.metrics.sfde_avg += sfde.mean()
                self.metrics.skipped_sfde_min += sfde_skipped.min()
                self.metrics.skipped_sfde_avg += sfde_skipped.mean()

                # --- minSSDE: minimum start-of-sequence displacement error ---
                # Find the first valid index for each agent
                first_valid_ind = gt_valid.int().argmax(dim=-1)  # (1, N)
                first_valid_ind = first_valid_ind.unsqueeze(0).expand(B, 1, N)  # (B, 1, N)
                ssde = torch.gather(error, 1, first_valid_ind).squeeze(1)  # shape: (B, N)
                agent_valid_first = gt_valid.any(-1).expand(B, N)  # shape: (B, N)
                ssde_avg = (ssde * agent_valid_first).sum(-1) / agent_valid_first.sum(-1) # Average SSDE over valid agents for each mode

                assert ssde_avg.ndim == 1
                assert ssde_avg.shape[0] == B

                # Aggregate min and avg as you do for SFDE
                self.metrics.ssde_min += ssde_avg.min()
                self.metrics.ssde_avg += ssde_avg.mean()

                # --- skipped_ssde: start-of-sequence displacement error for skipped steps ---
                first_valid_ind_skipped = gt_valid_skipped.int().argmax(dim=-1)  # (1, N)
                first_valid_ind_skipped = first_valid_ind_skipped.unsqueeze(0).expand(B, 1, N)  # (B, 1, N)
                ssde_skipped = torch.gather(error[:,::5], 1, first_valid_ind_skipped).squeeze(1)  # shape: (B, N)
                agent_valid_first_skipped = gt_valid_skipped.any(-1).expand(B, N)  # shape: (B, N)
                ssde_skipped_avg = (ssde_skipped * agent_valid_first_skipped).sum(-1) / agent_valid_first_skipped.sum(-1)

                assert ssde_skipped_avg.ndim == 1
                assert ssde_skipped_avg.shape[0] == B

                self.metrics.skipped_ssde_min += ssde_skipped_avg.min()
                self.metrics.skipped_ssde_avg += ssde_skipped_avg.mean()

                # --- sfde and sade ---
                gt_valid = gt_valid.permute(0, 2, 1)
                gt_valid_skipped = gt_valid_skipped.permute(0, 2, 1)

                sade_per_agent = (error * gt_valid).sum(1) / gt_valid.sum(1).clamp(1)
                sade = sade_per_agent.sum(1) / gt_valid.any(1).sum(1)
                sade_skipped_per_agent = (error[:,::5] * gt_valid_skipped).sum(1) / gt_valid_skipped.sum(1).clamp(1)
                sade_skipped = sade_skipped_per_agent.sum(1) / gt_valid_skipped.any(1).sum(1)

                self.metrics.sade_min += sade.min()
                self.metrics.sade_avg += sade.mean()

                self.metrics.skipped_sade_min += sade_skipped.min()
                self.metrics.skipped_sade_avg += sade_skipped.mean()

        # Following wosac_eval, fill in z with GT t = 10 data
        z_values = pred_data_dict["decoder/current_agent_position"][..., 2].unsqueeze(-1).expand(K, -1, T_pred)

        # FDD
        if not self.start_metrics_only:
            with timer("FDD"):
                # there doesn't appear to be an easy way to do this with cartesian product
                cur_FDD = None
                pred_ooi_valid_mask = pred_valid_mask[:, ooi_mask]
                single_mode_ooi_valid_mask = pred_ooi_valid_mask[0]
                at_least_2_modes_ooi_valid_mask = torch.sum(pred_valid_mask.int(), dim=0).transpose(-1, -2) >= 2 # (T_pred, N)

                # assert torch.all(torch.any(pred_ooi_valid_mask, dim=-1))
                last_valid_ind = pred_ooi_valid_mask.cumsum(dim=-1).argmax(dim=-1)  # (K, N)
                ooi_reconstructed_pos = pred_data_dict["decoder/reconstructed_position"][:, :,
                                                                                        ooi_mask]  # (K, T_pred, N, 2)
                last_valid_ind_reshaped = last_valid_ind[:, None, :, None].expand(-1, -1, -1, 2)
                final_pos = torch.gather(ooi_reconstructed_pos, dim=1, index=last_valid_ind_reshaped).squeeze(1)
                for i, j in itertools.product(range(K), range(K)):
                    final_dist = torch.linalg.norm(final_pos[i] - final_pos[j], dim=-1)
                    assert final_dist.ndim == 1
                    if cur_FDD == None:
                        cur_FDD = final_dist
                    else:
                        cur_FDD = torch.maximum(cur_FDD, final_dist)
                self.metrics.fdd += utils.masked_average(cur_FDD, single_mode_ooi_valid_mask.any(-1), 0)
                
                # ADD, we define as maximum distance across modes averaged across time then across agents
                cur_ADD = None
                pairwise_dists = torch.linalg.norm(ooi_reconstructed_pos.unsqueeze(1) - ooi_reconstructed_pos.unsqueeze(0), dim=-1) # (K, K, T_pred, N)
                dists = torch.where(torch.logical_and(pred_ooi_valid_mask.unsqueeze(1), pred_ooi_valid_mask.unsqueeze(0)).transpose(-1, -2), pairwise_dists, 0)
                dists = torch.amax(dists, dim=(0, 1)) # (T_pred, N)
                cur_ADD = utils.masked_average(utils.masked_average(dists, at_least_2_modes_ooi_valid_mask, dim=0), at_least_2_modes_ooi_valid_mask.any(0), dim=0)
                self.metrics.add += cur_ADD
            
        with timer("SDD"):
            # there doesn't appear to be an easy way to do this with cartesian product
            cur_SDD = None
            pred_ooi_valid_mask = pred_valid_mask[:, ooi_mask]
            single_mode_ooi_valid_mask = pred_ooi_valid_mask[0]
            at_least_2_modes_ooi_valid_mask = torch.sum(pred_valid_mask.int(), dim=0).transpose(-1, -2) >= 2 # (T_pred, N)

            # assert torch.all(torch.any(pred_ooi_valid_mask, dim=-1))
            last_valid_ind = pred_ooi_valid_mask.int().argmax(dim=-1)  # (K, N)
            ooi_reconstructed_pos = pred_data_dict["decoder/reconstructed_position"][:, :,
                                                                                     ooi_mask]  # (K, T_pred, N, 2)
            last_valid_ind_reshaped = last_valid_ind[:, None, :, None].expand(-1, -1, -1, 2)
            final_pos = torch.gather(ooi_reconstructed_pos, dim=1, index=last_valid_ind_reshaped).squeeze(1)
            for i, j in itertools.product(range(K), range(K)):
                final_dist = torch.linalg.norm(final_pos[i] - final_pos[j], dim=-1)
                assert final_dist.ndim == 1
                if cur_SDD == None:
                    cur_SDD = final_dist
                else:
                    cur_SDD = torch.maximum(cur_SDD, final_dist)
            self.metrics.sdd += utils.masked_average(cur_SDD, single_mode_ooi_valid_mask.any(-1), 0)
        
        if not self.start_metrics_only:
            with timer("Kinematic Metrics"):
                gt_speed, gt_accel, gt_jerk = self._compute_kinematic_metrics(
                    gt_data_dict["decoder/agent_velocity"].swapaxes(1, 0),
                    device
                )  # (N, T)
                pred_speed, pred_accel, pred_jerk = self._compute_kinematic_metrics(
                    pred_data_dict["decoder/reconstructed_velocity"].transpose(1, 2),
                    device
                )  # (K, N, T)
                gt_speed = gt_speed[..., T_context:T_gt]
                gt_accel = gt_accel[..., T_context:T_gt]
                gt_jerk = gt_jerk[..., T_context:T_gt]
                pred_speed = pred_speed[..., T_context:T_gt]
                pred_accel = pred_accel[..., T_context:T_gt]
                pred_jerk = pred_jerk[..., T_context:T_gt]

            if not self.key_metrics_only:

                with (timer("Collision Metrics")):
                    candidate_agents = gt_data_dict["decoder/current_agent_valid_mask"]
                    if isinstance(candidate_agents, torch.Tensor):
                        candidate_agents = candidate_agents.cpu().numpy()
                    candidate_agents = candidate_agents.nonzero()[0]

                    pred_veh_collisions, veh_cr_mode = calc_collision_rate(
                        candidate_agents=candidate_agents,
                        evaluate_agents=gt_data_dict["decoder/agent_id"],
                        pred_data_dict=pred_data_dict,
                        pred_shape=pred_shape,
                        device=device,
                        T_gt=T_gt,
                        T_context=T_context,
                        z_values=z_values
                    )

                    assert veh_cr_mode.shape[0] == B
                    self.metrics.veh_coll_avg += veh_cr_mode.mean()
                    self.metrics.veh_coll_min += veh_cr_mode.min()
                    self.metrics.veh_coll_max += veh_cr_mode.max()

                    if adv_list is not None:
                        assert len(adv_list) == 1
                        assert adv_list[0] not in bv_list
                        assert sdc_index not in adv_list

                        # self.sdc_coll_adv_active = True
                        for kk in adv_list:
                            assert int(kk.item()) in candidate_agents

                        adv_sdc_coll, adv_sdc_coll_rate = calc_collision_rate(
                            candidate_agents=adv_list,
                            evaluate_agents=pred_data_dict["decoder/sdc_index"],
                            pred_data_dict=pred_data_dict,
                            pred_shape=pred_shape,
                            device=device,
                            T_gt=T_gt,
                            T_context=T_context,
                            z_values=z_values
                        )

                        assert adv_sdc_coll_rate.ndim == 1
                        self.metrics.sdc_adv_coll_avg += adv_sdc_coll_rate.mean()
                        self.metrics.sdc_adv_coll_min += adv_sdc_coll_rate.min()
                        self.metrics.sdc_adv_coll_max += adv_sdc_coll_rate.max()

                        adv_bv_coll, adv_bv_coll_rate = calc_collision_rate(
                            candidate_agents=bv_list,
                            evaluate_agents=adv_list,
                            pred_data_dict=pred_data_dict,
                            pred_shape=pred_shape,
                            device=device,
                            T_gt=T_gt,
                            T_context=T_context,
                            z_values=z_values
                        )
                        assert adv_bv_coll_rate.ndim == 1

                        assert adv_bv_coll_rate.shape[0] == B
                        self.metrics.adv_bv_coll_avg += adv_bv_coll_rate.mean()
                        self.metrics.adv_bv_coll_min += adv_bv_coll_rate.min()
                        self.metrics.adv_bv_coll_max += adv_bv_coll_rate.max()

                    assert sdc_index not in bv_list
                    assert bv_list is not None
                    for kk in bv_list:
                        assert int(kk.item()) in candidate_agents
                    sdc_bv_coll, sdc_bv_coll_rate = calc_collision_rate(
                        candidate_agents=bv_list,
                        evaluate_agents=pred_data_dict["decoder/sdc_index"],
                        pred_data_dict=pred_data_dict,
                        pred_shape=pred_shape,
                        device=device,
                        T_gt=T_gt,
                        T_context=T_context,
                        z_values=z_values
                    )
                    assert sdc_bv_coll_rate.ndim == 1

                    assert sdc_bv_coll_rate.shape[0] == B
                    self.metrics.sdc_bv_coll_avg += sdc_bv_coll_rate.mean()
                    self.metrics.sdc_bv_coll_min += sdc_bv_coll_rate.min()
                    self.metrics.sdc_bv_coll_max += sdc_bv_coll_rate.max()

                    # map_feature = gt_data_dict["encoder/map_feature"]
                    map_feature = gt_data_dict["vis/map_feature"]
                    assert map_feature.ndim == 3  # This is unbatched.

                    road_edges = []
                    for i in range(map_feature.shape[0]):
                        # For each map feature
                        if map_feature[i, 0, 15] == 1:
                            map_feat = []
                            for j in range(map_feature.shape[1]):
                                if gt_data_dict['encoder/map_feature_valid_mask'][i, j]:
                                    map_feat.append(
                                        map_pb2.MapPoint(
                                            x=map_feature[i, j, 0], y=map_feature[i, j, 1], z=0 #map_feature[i, j, 2] # let's say there is no z axis any more
                                        )
                                    )
                            road_edges.append(map_feat)

                    # ====================== debugging; customized env collision ==========================
                    num_mode = pred_data_dict["decoder/reconstructed_position"].shape[0]
                    pred_data_dict = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in pred_data_dict.items()}

                    def _get_mode(output_dict, mode, num_modes):
                        ret = {}
                        for k, v in output_dict.items():
                            if isinstance(v, np.ndarray) and len(v) == num_modes:
                                ret[k] = v[mode]
                            else:
                                ret[k] = v
                        return ret
                    
                    pred_env_collisions_traj_level = []
                    
                    # for mode in range(num_mode):
                    #     pred_data_dict_mode = _get_mode(pred_data_dict, mode, num_modes=num_mode)
                    #     sid = pred_data_dict_mode["metadata/scenario_id"]
                        # from infgen.gradio_ui.plot import plot_pred # for debugging 
                        # plot_pred(pred_data_dict_mode, save_path=f"pred_{sid}_mode{mode}.png")

                        # env_col = customized_env_collision_rate(pred_data_dict_mode)
                        # pred_env_collisions_traj_level.append(env_col)
                    # =======================================================================


                    # ======= old env coll detection has bugs =====================
                    # eval_mask = ooi_mask & (~static_agent_mask)
                    # env_nearest_distances = torch.stack(
                    #     [
                    #         tf_to_torch(
                    #             map_metric_features.compute_distance_to_road_edge(
                    #                 center_x=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 0].T),
                    #                 center_y=conv(pred_data_dict["decoder/reconstructed_position"][k, ..., 1].T),
                    #                 center_z=conv(z_values[k]),
                    #                 length=conv(pred_shape[k, ..., 0]),
                    #                 width=conv(pred_shape[k, ..., 1]),
                    #                 height=conv(pred_shape[k, ..., 2]),
                    #                 heading=conv(pred_data_dict["decoder/reconstructed_heading"][k].T),
                    #                 valid=conv(pred_valid_mask[k], dtype=tf.bool),
                    #                 evaluated_object_mask=conv(eval_mask, dtype=tf.bool),
                    #                 road_edge_polylines=road_edges,
                    #             ),
                    #             device=device
                    #         ) for k in range(K)
                    #     ]
                    # )

                    pred_valid_mask = pred_valid_mask[..., T_context:T_gt]

                    # pred_env_collisions = torch.greater(env_nearest_distances, map_metric_features.OFFROAD_DISTANCE_THRESHOLD)
                    # pred_env_collisions = pred_env_collisions[..., T_context:T_gt]
                    # pred_env_collisions_traj_level = pred_env_collisions.any(dim=-1)  # (B, num_ooi)
                    # # Avg over agent dim. Here we assume all evaluated agents are valid so don't do the masked_avg
                    # env_collision_rate = pred_env_collisions_traj_level.float().mean(-1)

                    # # ==================================== customized env collision rate ====================================
                    # # env_collision_rate = np.array(pred_env_collisions_traj_level).mean(-1)
                    # assert env_collision_rate.ndim == 1

                    # assert env_collision_rate.shape[0] == B
                    # self.metrics.env_coll_avg += env_collision_rate.mean()
                    # self.metrics.env_coll_min += env_collision_rate.min()
                    # self.metrics.env_coll_max += env_collision_rate.max()

                    step_wise_collision = pred_veh_collisions
                    scenario_has_collision = torch.any(step_wise_collision).item()
                    
                    if scenario_has_collision:
                        speed_when_collision = torch.where(step_wise_collision, pred_speed[:, ooi_mask], 0)

                        coll_vel_max_agent = speed_when_collision.amax(dim=(-1, -2))
                        coll_valid_mask = speed_when_collision.any(-1).any(-1)
                        coll_vel_max_agent = coll_vel_max_agent[coll_valid_mask]

                        assert coll_vel_max_agent.ndim == 1

                        if coll_vel_max_agent.numel() != 0:
                            self.metrics.coll_vel_maxagent_avg += (coll_vel_max_agent).sum() / coll_valid_mask.sum().clamp(1)
                            self.metrics.coll_vel_maxagent_min += coll_vel_max_agent.min()
                            self.metrics.coll_vel_maxagent_max += coll_vel_max_agent.max()

                            self.metrics.veh_coll_scenario_count += 1

                        
                    sdc_speed = pred_speed[:, sdc_index]
                    sdc_coll = step_wise_collision[:, sdc_index_in_ooi]

                    # sdc_speed_when_coll = torch.where(sdc_coll, sdc_speed, torch.nan).amax(-1)
                    sdc_speed_when_coll = torch.where(sdc_coll, sdc_speed, torch.nan)
                    valid_mask = ~torch.isnan(sdc_speed_when_coll)

                    if torch.any(sdc_coll).item(): # if there is valid collision
                        self.metrics.coll_vel_sdc_avg += (sdc_speed_when_coll[valid_mask]).sum() / valid_mask.sum()
                        self.metrics.coll_vel_sdc_max += (sdc_speed_when_coll[valid_mask]).max()
                        self.metrics.coll_vel_sdc_min += (sdc_speed_when_coll[valid_mask]).min()
                        self.metrics.sdc_coll_scenario_count += 1
                        assert sdc_speed_when_coll.shape[0] == B

                    sdc_acc = torch.abs(torch.nan_to_num(pred_accel[:, sdc_index]))  # Shape: (K, T)
                    sdc_mask = pred_valid_mask[:, sdc_index]
                    sdc_acc_avgt = (sdc_acc * sdc_mask).sum(-1) / sdc_mask.sum(-1).clamp(1)
                    assert sdc_acc_avgt.ndim == 1
                    assert sdc_acc_avgt.shape[0] == B
                    self.metrics.sdc_acc_avgtime_max += sdc_acc_avgt.max()
                    self.metrics.sdc_acc_avgtime_avg += sdc_acc_avgt.mean()
                    self.metrics.sdc_acc_avgtime_min += sdc_acc_avgt.min()

                    sdc_acc_maxt = sdc_acc.amax(-1)
                    assert sdc_acc_maxt.ndim == 1
                    assert sdc_acc_maxt.shape[0] == B
                    self.metrics.sdc_acc_maxtime_max += sdc_acc_maxt.max()
                    self.metrics.sdc_acc_maxtime_avg += sdc_acc_maxt.mean()
                    self.metrics.sdc_acc_maxtime_min += sdc_acc_maxt.min()

                    sdc_jerk = torch.abs(torch.nan_to_num(pred_jerk[:, sdc_index]))  # Shape: (K, T)
                    sdc_mask = pred_valid_mask[:, sdc_index]
                    sdc_jerk_avgt = (sdc_jerk * sdc_mask).sum(-1) / sdc_mask.sum(-1).clamp(1)
                    assert sdc_jerk_avgt.ndim == 1
                    assert sdc_jerk_avgt.shape[0] == B
                    self.metrics.sdc_jerk_avgtime_max += sdc_jerk_avgt.max()
                    self.metrics.sdc_jerk_avgtime_avg += sdc_jerk_avgt.mean()
                    self.metrics.sdc_jerk_avgtime_min += sdc_jerk_avgt.min()

                    sdc_jerk_maxt = sdc_jerk.amax(-1)
                    assert sdc_jerk_maxt.ndim == 1
                    assert sdc_jerk_maxt.shape[0] == B
                    self.metrics.sdc_jerk_maxtime_max += sdc_jerk_maxt.max()
                    self.metrics.sdc_jerk_maxtime_avg += sdc_jerk_maxt.mean()
                    self.metrics.sdc_jerk_maxtime_min += sdc_jerk_maxt.min()

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
                        device=device
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
                                device=device
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
                if not self.key_metrics_only:
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
                # plt.clf()
                # plt.hist(gt_speed_bins[:-1], bins=gt_speed_bins, weights=gt_speed_hist, density=False)
                # plt.savefig(f"gt_speed{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()
                # plt.hist(pred_speed_bins[:-1], bins=pred_speed_bins, weights=pred_speed_hist, density=False)
                # plt.savefig(f"pred_speed{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()
                # plt.hist(gt_accel_bins[:-1], bins=gt_accel_bins, weights=gt_accel_hist, density=False)
                # plt.savefig(f"gt_accel{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()
                # plt.hist(pred_accel_bins[:-1], bins=pred_accel_bins, weights=pred_accel_hist, density=False)
                # plt.savefig(f"pred_accel{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()
                # plt.hist(gt_ttc_bins[:-1], bins=gt_ttc_bins, weights=gt_ttc_hist, density=False)
                # plt.savefig(f"gt_ttc{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()
                # plt.hist(pred_ttc_bins[:-1], bins=pred_ttc_bins, weights=pred_ttc_hist, density=False)
                # plt.savefig(f"pred_ttc{SCENE_IDX}.png", bbox_inches='tight')
                # plt.clf()

            with timer("JSD"):
                speed_jsd = jsd(gt_speed_hist, pred_speed_hist)
                acc_jsd = jsd(gt_accel_hist, pred_accel_hist)
                if not self.key_metrics_only:
                    ttc_jsd = jsd(gt_ttc_hist, pred_ttc_hist)
                self.metrics.vel_jsd += speed_jsd
                self.metrics.acc_jsd += acc_jsd
                if not self.key_metrics_only:
                    self.metrics.ttc_jsd += ttc_jsd

    def _compute_kinematic_metrics(self, vel, device):
        if type(vel) == np.ndarray:
            vel = numpy_to_torch(vel, device=device)
        speed = torch.linalg.norm(vel, axis=-1)
        accel = self._central_diff(speed, device, pad_value=torch.nan) / self.SECONDS_PER_STEP
        jerk = self._central_diff(accel, device, pad_value=torch.nan) / self.SECONDS_PER_STEP
        return speed, accel, jerk

    def _central_diff(self, tensor, device, pad_value=torch.nan):
        pad_shape = (*tensor.shape[:-1], 1)
        pad_tensor = torch.ones(pad_shape, device=device) * pad_value
        diff_t = (tensor[..., 2:] - tensor[..., :-2]) / 2
        return torch.cat([pad_tensor, diff_t, pad_tensor], dim=-1)

    def add_customized_CR(
        self,
        max_sdc_adv_cr=None,
        max_sdc_bv_cr=None,
        max_adv_bv_cr=None,
        min_sdc_adv_cr=None,
        min_sdc_bv_cr=None,
        min_adv_bv_cr=None,
        avg_sdc_adv_cr=None,
        avg_sdc_bv_cr=None,
        avg_adv_bv_cr=None,
        all_agent_cr=None
        
    ):
        if max_sdc_adv_cr is not None:
            self.metrics.customized_max_sdc_adv_coll += max_sdc_adv_cr
        if max_sdc_bv_cr is not None:
            self.metrics.customized_max_sdc_bv_coll += max_sdc_bv_cr

        if max_adv_bv_cr is not None:
            self.metrics.customized_max_adv_bv_coll += max_adv_bv_cr

        if min_sdc_adv_cr is not None:
            self.metrics.customized_min_sdc_adv_coll += min_sdc_adv_cr
        if min_sdc_bv_cr is not None:
            self.metrics.customized_min_sdc_bv_coll += min_sdc_bv_cr
        if min_adv_bv_cr is not None:
            self.metrics.customized_min_adv_bv_coll += min_adv_bv_cr

        if avg_sdc_adv_cr is not None:
            self.metrics.customized_avg_sdc_adv_coll += avg_sdc_adv_cr
        if avg_sdc_bv_cr is not None:
            self.metrics.customized_avg_sdc_bv_coll += avg_sdc_bv_cr
        if avg_adv_bv_cr is not None:
            self.metrics.customized_avg_adv_bv_coll += avg_adv_bv_cr

        if all_agent_cr is not None:
            self.metrics.customized_all_agent_coll += all_agent_cr

    def aggregate(self):
        return self.metrics.aggregate()

    def print(self):
        metrics = self.metrics.aggregate()
        print("\n=====================================")
        print("Evaluation Metrics:")
        print(utils.pretty_print(metrics))
        print("=====================================")
        return metrics

    def save(self, save_path=None):
        if save_path is None:
            save_path = "evaluation_results"

        metrics = self.metrics.aggregate()
        metrics["save_path"] = save_path

        # Save a json:
        import json
        json_file = save_path + ".json"
        with open(json_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {json_file}")

        # Save a csv:
        import pandas as pd
        df = pd.DataFrame([metrics])
        csv_file = save_path + ".csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved metrics to {csv_file}")

        return metrics