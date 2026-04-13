import numpy as np
from shapely.geometry import Polygon

from bmt.utils import utils
import torch

INVALID_VALUE = -10000


class TurnAction:
    STOP = 0
    KEEP_STRAIGHT = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    U_TURN = 4

    num_actions = 5


class AccelerationAction:
    STOP = 0
    KEEP_SPEED = 1
    SPEED_UP = 2
    SLOW_DOWN = 3

    num_actions = 4


class SafetyAction:
    SAFE = 0
    COLLISION = 1
    num_actions = 2


def cal_polygon_contour(x, y, theta, width, length):

    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_front = np.column_stack((left_front_x, left_front_y))

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_front = np.column_stack((right_front_x, right_front_y))

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
    right_back = np.column_stack((right_back_x, right_back_y))

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
    left_back = np.column_stack((left_back_x, left_back_y))

    polygon_contour = np.concatenate(
        (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1
    )

    return polygon_contour


def detect_collision(contour_list1, mask1, contour_list2, mask2):
    collision_detected = []
    assert len(contour_list1) == len(contour_list2)

    for i in range(len(contour_list1)):
        if mask1[i] and mask2[i]:
            poly1 = Polygon(contour_list1[i])
            poly2 = Polygon(contour_list2[i])

            if poly1.intersects(poly2):
                collision_detected.append(True)
            else:
                collision_detected.append(False)
        else:
            collision_detected.append(False)

    return collision_detected


def get_direction_action_from_trajectory_batch(traj, mask, dt=0.1, ooi=None):
    U_TURN_DEG = 115
    LEFT_TURN_DEG = 25
    RIGHT_TURN_DEG = -25
    STOP_SPEED = 0.06

    assert traj.ndim == 3
    traj_diff = traj[1:] - traj[:-1]
    mask_diff = mask[1:] & mask[:-1]

    displacement = np.linalg.norm(traj_diff, axis=-1)

    mask_diff_stop = mask_diff & (displacement > 0.1)

    pred_angles = np.arctan2(traj_diff[..., 1], traj_diff[..., 0])
    pred_angles_diff = utils.wrap_to_pi(pred_angles[1:] - pred_angles[:-1])

    # It's meaning less to compute heading for a stopped vehicle. So mask them out!
    mask_diff_diff = mask_diff_stop[1:] & mask_diff_stop[:-1]
    # Note that we should not wrap to pi here because the sign is important.
    accumulated_heading_change_rad = (pred_angles_diff * mask_diff_diff).sum(axis=0)
    accumulated_heading_change_deg = np.degrees(accumulated_heading_change_rad)

    # print("accumulated_heading_change_deg: ", list(zip(ooi, accumulated_heading_change_deg)))

    speed = displacement / dt
    avg_speed = utils.masked_average_numpy(speed, mask_diff, dim=0)

    actions = np.zeros(accumulated_heading_change_deg.shape, dtype=int)
    actions.fill(TurnAction.KEEP_STRAIGHT)
    actions[accumulated_heading_change_deg > LEFT_TURN_DEG] = TurnAction.TURN_LEFT
    actions[accumulated_heading_change_deg < RIGHT_TURN_DEG] = TurnAction.TURN_RIGHT
    actions[accumulated_heading_change_deg > U_TURN_DEG] = TurnAction.U_TURN
    actions[accumulated_heading_change_deg < -U_TURN_DEG] = TurnAction.U_TURN
    actions[avg_speed < STOP_SPEED] = TurnAction.STOP
    return actions


def get_acce_action_from_trajectory_batch(batch_trajs, mask, ooi=None, dt=0.1):

    SPEEDUP_ACCEL = 0.3
    SPEEDDOWN_ACCEL = -0.3
    STOP_SPEED = 0.06

    traj_diff = batch_trajs[1:] - batch_trajs[:-1]  # (T, A, 2)
    mask_diff = mask[1:] & mask[:-1]  # (T, A)

    speed = np.linalg.norm(traj_diff, axis=-1) / dt  # (T, A)

    speed_change = speed[1:] - speed[:-1]
    mask_diff_diff = mask_diff[1:] & mask_diff[:-1]

    absolute_avg_speed = utils.masked_average_numpy(speed, mask_diff, dim=0)

    accumulated_speed_change = (speed_change * mask_diff_diff).sum(0)

    init_speed_ind = mask_diff.argmax(axis=0)
    init_speed = np.take_along_axis(speed, init_speed_ind[None, :], axis=0)[0]

    speed_change_ratio = accumulated_speed_change / np.maximum(init_speed, STOP_SPEED)

    # print("speed_change_ratio: ", list(zip(ooi, speed_change_ratio)))

    actions = np.zeros(speed_change_ratio.shape, dtype=int)

    actions.fill(AccelerationAction.KEEP_SPEED)
    actions[speed_change_ratio > SPEEDUP_ACCEL] = AccelerationAction.SPEED_UP
    actions[speed_change_ratio < SPEEDDOWN_ACCEL] = AccelerationAction.SLOW_DOWN
    actions[absolute_avg_speed <= STOP_SPEED] = AccelerationAction.STOP  # if stop

    return actions


def get_safety_action_from_sdc_adv(data_dict, adv_id):
    sdc_id = data_dict["decoder/sdc_index"]

    T = data_dict["decoder/reconstructed_position"].shape[0]
    agent_shape = np.tile(
        data_dict['decoder/current_agent_shape'][np.newaxis, :, :], (T, 1, 1)
    )

    contours = []
    for agent_id in [adv_id, sdc_id]:
        traj = data_dict["decoder/reconstructed_position"][:91, agent_id, :]  # (91, 3)
        length = agent_shape[:91, agent_id, 0]
        width = agent_shape[:91, agent_id, 1]
        theta = data_dict['decoder/reconstructed_heading'][:91, agent_id]  # (91, ) # in pi
        mask = data_dict['decoder/reconstructed_valid_mask'][:91, agent_id]  # (91,)

        poly = cal_polygon_contour(traj[:, 0], traj[:, 1], theta, width, length)
        contours.append(poly)

    sdc_mask = data_dict['decoder/agent_valid_mask'][:, sdc_id]  # (91,)
    adv_mask = data_dict['decoder/reconstructed_valid_mask'][:, adv_id]
    adv_contour = contours[0]
    sdc_contour = contours[1]

    collision_detected = detect_collision(adv_contour, adv_mask, sdc_contour, sdc_mask)

    # instead of loading a dict which saves all collision scenario, we could simply detect all agents' potential collision
    return collision_detected


def get_safety_action_from_trajectory_batch(data_dict, track_agent_indicies):

    safety_actions = np.zeros((track_agent_indicies.shape[0], ), dtype=int)  # plus sdc

    contours = []
    for agent1_id in track_agent_indicies:
        traj = data_dict["decoder/agent_position"][:, agent1_id, :]  # (91, 3)
        length = data_dict["decoder/agent_shape"][:, agent1_id, 0]
        width = data_dict["decoder/agent_shape"][:, agent1_id, 1]
        theta = data_dict['decoder/agent_heading'][:, agent1_id]  # (91, ) # in pi
        mask = data_dict['decoder/agent_valid_mask'][:, agent1_id]  # (91,)
        poly = cal_polygon_contour(traj[:, 0], traj[:, 1], theta, width, length)
        contours.append(poly)

    for i in range(track_agent_indicies.shape[0] - 1):
        for j in range(i + 1, track_agent_indicies.shape[0]):
            mask_1 = data_dict['decoder/agent_valid_mask'][:, track_agent_indicies[i]]  # (91,)
            mask_2 = data_dict['decoder/agent_valid_mask'][:, track_agent_indicies[j]]
            collision_detected = detect_collision(contours[i], mask_1, contours[j], mask_2)

            if any(collision_detected):
                # print(f"Collision between {i} and {j} happen at step: {np.array(collision_detected).nonzero()}")
                safety_actions[i] = 1  # Label collisions for OOIs now. Later we will build a larger dict.
                safety_actions[j] = 1

    # instead of loading a dict which saves all collision scenario, we could simply detect all agents' potential collision
    return safety_actions


def get_2D_collision_labels(data_dict, track_agent_indicies):
    # Now, instead of getting 1d-array of collision labels, let's do 2-d array to detect whether there is collision between given two agents.

    safety_actions = torch.zeros((track_agent_indicies.shape[0], track_agent_indicies.shape[0]), dtype=int)  # plus sdc

    contours = []
    for agent1_id in track_agent_indicies:
        traj = data_dict["decoder/agent_position"][:91, agent1_id, :]  # (91, 3)
        length = data_dict["decoder/agent_shape"][10, agent1_id, 0]
        width = data_dict["decoder/agent_shape"][10, agent1_id, 1]
        theta = data_dict['decoder/agent_heading'][:91, agent1_id]  # (91, ) # in pi
        mask = data_dict['decoder/agent_valid_mask'][:91, agent1_id]  # (91,)
        poly = cal_polygon_contour(traj[:, 0], traj[:, 1], theta, width, length)
        contours.append(poly)

    for i in range(track_agent_indicies.shape[0]):
        for j in range(track_agent_indicies.shape[0]):
            if i == j:
                continue  # leave as zero
            mask_1 = data_dict['decoder/agent_valid_mask'][:91, track_agent_indicies[i]]  # (91,)
            mask_2 = data_dict['decoder/agent_valid_mask'][:91, track_agent_indicies[j]]
            collision_detected = detect_collision(contours[i], mask_1, contours[j], mask_2)

            if any(collision_detected):
                # print(f"Collision between {i} and {j} happen at step: {np.array(collision_detected).nonzero()}")
                safety_actions[i][j] = 1  # Label collisions for OOIs now. Later we will build a larger dict.

    assert np.array_equal(safety_actions, safety_actions.T), "The 2D label is not symmetrical"
    return safety_actions


def prepare_action_label(*, data_dict, dt, mask_probability, config):
    """
    mask_probability: the probability of masking the label. Should be around 0.05 or 0.1. Can't be too high.
    """
    ooi_ind = data_dict["decoder/labeled_agent_id"]
    ooi_pos = utils.extract_data_by_agent_indices(data_dict["decoder/agent_position"], ooi_ind, agent_dim=1)[..., :2]
    ooi_valid = utils.extract_data_by_agent_indices(
        data_dict["decoder/agent_valid_mask"], ooi_ind, agent_dim=1
    )  # (T, A)

    # TODO: hardcoded here for now and we assume you can access GT trajectory. This won't work with test dataset.
    assert ooi_pos.shape[0] == 91
    assert ooi_valid.shape[0] == 91

    # get the degree, acceleration, speed
    turn_actions = get_direction_action_from_trajectory_batch(traj=ooi_pos, mask=ooi_valid, dt=dt, ooi=ooi_ind)
    acce_actions = get_acce_action_from_trajectory_batch(ooi_pos, ooi_valid, dt=dt, ooi=ooi_ind)

    # Rescatter labels to decoder-agent indices
    assert config.TRAINING.PREDICT_ALL_AGENTS
    B = data_dict["decoder/agent_valid_mask"].shape[1]

    full_turn_actions = np.full((B, ), -1, dtype=int)
    full_acce_actions = np.full((B, ), -1, dtype=int)

    label_mask = np.random.binomial(1, mask_probability, size=len(ooi_ind))
    label_invalid_mask = label_mask == 1

    turn_actions[label_invalid_mask] = -1
    acce_actions[label_invalid_mask] = -1

    full_turn_actions[ooi_ind] = turn_actions
    full_acce_actions[ooi_ind] = acce_actions

    data_dict["decoder/label_turning"] = full_turn_actions
    data_dict["decoder/label_acceleration"] = full_acce_actions

    return data_dict


def prepare_safety_label(*, data_dict, dt, mask_probability, config):
    ooi_ind = data_dict["decoder/labeled_agent_id"]

    ooi_pos = utils.extract_data_by_agent_indices(data_dict["decoder/agent_position"], ooi_ind, agent_dim=1)[..., :2]
    ooi_valid = utils.extract_data_by_agent_indices(
        data_dict["decoder/agent_valid_mask"], ooi_ind, agent_dim=1
    )  # (T, A)

    # TODO: hardcoded here for now and we assume you can access GT trajectory. This won't work with test dataset.
    assert ooi_pos.shape[0] == 91
    assert ooi_valid.shape[0] == 91

    safety_actions = get_safety_action_from_trajectory_batch(data_dict, ooi_ind)

    # Rescatter labels to decoder-agent indices
    assert config.TRAINING.PREDICT_ALL_AGENTS
    num_modeled_agents = data_dict["decoder/agent_valid_mask"].shape[1]

    full_safety_actions = np.full((num_modeled_agents, ), -1, dtype=int)

    label_mask = np.random.binomial(1, mask_probability, size=len(ooi_ind))
    label_invalid_mask = label_mask == 1

    label_invalid_mask[safety_actions == 1] = False  # We don't mask collision labels

    safety_actions[label_invalid_mask] = -1

    full_safety_actions[ooi_ind] = safety_actions

    data_dict["decoder/label_safety"] = full_safety_actions

    return data_dict


if __name__ == '__main__':
    scenario_dir = "/Users/claire_liu/validation_interactive_0/cat_reconstructed/sd_reconstructed_v0_ScenarioMap-21.pkl"
    cat_dir = "/Users/claire_liu/validation_interactive_0/save.pkl"

    import pickle

    with open(scenario_dir, 'rb') as f:
        scenario_data = pickle.load(f)
    f.close()

    with open(cat_dir, 'rb') as ff:
        cat_dict = pickle.load(ff)
    ff.close()

    batch_labels = get_3d_action_label(scenario_data, cat_dict)
    print(batch_labels)