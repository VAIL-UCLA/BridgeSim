"""
Translate a MetaDrive Scenario Description instance to a dict of tensors.
"""
import copy
import logging
import pickle

import numpy as np
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType

from bmt import utils
from bmt.dataset import constants
from bmt.dataset.preprocess_action_label import prepare_action_label, prepare_safety_label
from bmt.tokenization import get_tokenizer

logger = logging.getLogger(__file__)

extract_data_by_agent_indices = utils.extract_data_by_agent_indices


def centralize_to_map_center(position_array, map_center, map_heading):
    """
    Centralize the position array to the map center and rotate the position array to the map heading.
    Note that the map center and map heading do not change based on agent or timestep.
    """
    ndim = position_array.ndim
    # position_array = position_array.copy()
    if map_center is not None:
        assert map_center.shape == (3, )
        assert position_array.shape[-1] <= 3, position_array.shape
        map_center = map_center.reshape(*(1, ) * (ndim - 1), 3)
        position_array -= map_center[..., :position_array.shape[-1]]
    if map_heading == 0.0:
        return position_array

    if position_array.shape[-1] == 3:
        position_array = utils.rotate(
            position_array[..., 0], position_array[..., 1], -map_heading, z=position_array[..., 2]
        )
    elif position_array.shape[-1] == 2:
        position_array = utils.rotate(
            position_array[..., 0], position_array[..., 1], -map_heading, z=np.zeros_like(position_array[..., 0])
        )
    else:
        raise ValueError()
    return position_array


def extract_map_center_heading_locations(map_feature):
    assert isinstance(map_feature, dict)
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")
    min_x, min_y, min_z = float("+inf"), float("+inf"), float("+inf")
    for map_feat_id, map_feat in map_feature.items():
        if "polyline" in map_feat:
            locations = map_feat['polyline']
        elif "position" in map_feat:
            locations = map_feat['position']
        elif "polygon" in map_feat:
            locations = map_feat["polygon"]
        else:
            raise ValueError("Unknown map feature: {}, {}".format(map_feat_id, map_feat.keys()))
        locations = locations.reshape(-1, locations.shape[-1])
        map_feat["location"] = locations
        max_boundary = locations.max(axis=0)
        min_boundary = locations.min(axis=0)
        max_x = max_boundary[0]
        max_y = max_boundary[1]
        min_x = min_boundary[0]
        min_y = min_boundary[1]
        if locations.shape[-1] == 3:
            max_z = max_boundary[2]
            min_z = min_boundary[2]
    if max_z == float("-inf"):
        max_z = 0.0
    if min_z == float("+inf"):
        min_z = 0.0
    map_boundary_max = np.array([max_x, max_y, max_z])
    map_boundary_min = np.array([min_x, min_y, min_z])

    map_center = np.stack([map_boundary_max, map_boundary_min], axis=0).mean(axis=0)
    map_heading = 0.0

    return {
        "map_center": map_center,
        "map_heading": map_heading,
        "map_boundary_max": map_boundary_max,
        "map_boundary_min": map_boundary_min,
        "map_feature": map_feature
    }


def process_map_and_traffic_light(
    *, data_dict, scenario, map_feature, dynamic_map_states, track_length, max_vectors, max_map_features,
    max_length_per_map_feature, max_traffic_lights, remove_traffic_light_state, limit_map_range
):
    # ========== Find the boundary of the map first ==========
    map_center_info = extract_map_center_heading_locations(map_feature)
    map_center = map_center_info["map_center"]
    map_heading = map_center_info["map_heading"]
    map_feature_augmented = map_center_info["map_feature"]

    # ========== Process Map Features ==========
    # The output is a dict whose keys are the lane ID and key is a state array in shape [T, ???]

    # Get a compact representation of all points in the maps
    map_feature_list = []  # Key: map_feat_id, Value: A dict of processed values
    map_heading_list = []
    map_valid_mask_list = []
    map_position_list = []

    for map_index, (map_feat_id, map_feat) in enumerate(map_feature_augmented.items()):
        rotated_polyline = centralize_to_map_center(
            position_array=map_feat["location"],  # [num points, 2 or 3]
            map_center=map_center,  # [1, 1, 3]
            map_heading=map_heading
        )

        if "polygon" in map_feat:
            # For crosswalk, and other "polygon" based map features, we need to pad the last point to the first point.
            rotated_polyline = np.concatenate([rotated_polyline, rotated_polyline[:1]], axis=0)

        if rotated_polyline.shape[-1] == 2:
            rotated_polyline = np.concatenate([rotated_polyline, np.zeros((rotated_polyline.shape[0], 1))], axis=-1)

        start_points = rotated_polyline[:-1].copy()  # in shape [# map feats - 1, 2]
        end_points = rotated_polyline[1:].copy()  # in shape [# map feats - 1, 2]
        if start_points.shape[0] == 0:
            # A special case here is that the map feature contains only one points.
            # In this case, we suppose the vector has the same point as start point and end point (its len=0)
            start_points = rotated_polyline
            end_points = rotated_polyline
            num_vectors = 1

        else:
            num_vectors = start_points.shape[0]

        assert start_points.ndim == 2  # [num vectors, 3]
        assert start_points.shape[-1] == 3  # [num vectors, 3] # for CAT, start_points.shape[-1] = 2

        direction = end_points - start_points
        heading = np.arctan2(direction[..., 1], direction[..., 0])

        point_diff = np.linalg.norm(direction[..., :2], axis=-1)

        road_length = 0.0
        start_index = 0
        # Iterate over all "vectors" in a map feature.
        # We will produce a map features, containing a set of vectors, in these conditions:
        # (1) If the segment is a lane and has length > MAX_LENGTH_PER_MAP_FEATURE, or
        # (2) The segment has max_vectors vectors, or
        # (3) The segment contains the leftover vectors with less than max_vectors vectors.
        for i in range(num_vectors):
            road_length += point_diff[i]
            map_feat_too_long = (road_length >= max_length_per_map_feature)
            # map_feat_is_lane = MetaDriveType.is_lane(map_feat['type'])

            num_valid_vectors = i - start_index + 1

            too_many_vectors = num_valid_vectors >= max_vectors
            last_set_of_vectors = (i == num_vectors - 1) and ((i - start_index) > 0)
            if i - start_index == 0:
                continue
            if map_feat_too_long or too_many_vectors or last_set_of_vectors:
                # The map feature is a 2D array with shape [#vectors, 27].
                # map_feature = np.zeros([i - start_index, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)
                map_feature = np.zeros([max_vectors, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)

                end_index = i + 1
                map_feature[:num_valid_vectors, :3] = start_points[start_index:end_index]
                map_feature[:num_valid_vectors, 3:6] = end_points[start_index:end_index]
                map_feature[:num_valid_vectors, 6:9] = direction[start_index:end_index]
                map_feature[:num_valid_vectors, 9] = utils.wrap_to_pi(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 10] = np.sin(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 11] = np.cos(heading[start_index:end_index])
                map_feature[:num_valid_vectors, 12] = point_diff[start_index:end_index]

                map_feature[:num_valid_vectors, 13] = MetaDriveType.is_lane(map_feat['type'])
                map_feature[:num_valid_vectors, 14] = MetaDriveType.is_sidewalk(map_feat['type'])
                map_feature[:num_valid_vectors, 15] = MetaDriveType.is_road_boundary_line(map_feat['type'])
                map_feature[:num_valid_vectors, 16] = MetaDriveType.is_road_line(map_feat['type'])
                map_feature[:num_valid_vectors, 17] = MetaDriveType.is_broken_line(map_feat['type'])
                map_feature[:num_valid_vectors, 18] = MetaDriveType.is_solid_line(map_feat['type'])
                map_feature[:num_valid_vectors, 19] = MetaDriveType.is_yellow_line(map_feat['type'])
                map_feature[:num_valid_vectors, 20] = MetaDriveType.is_white_line(map_feat['type'])
                map_feature[:num_valid_vectors, 21] = MetaDriveType.is_driveway(map_feat['type'])
                map_feature[:num_valid_vectors, 22] = MetaDriveType.is_crosswalk(map_feat['type'])
                map_feature[:num_valid_vectors, 23] = MetaDriveType.is_speed_bump(map_feat['type'])
                map_feature[:num_valid_vectors, 24] = MetaDriveType.is_stop_sign(map_feat['type'])
                map_feature[:num_valid_vectors, 25] = road_length
                # valid_mask = np.ones_like(start_points[start_index:i, 0])
                map_feature[:num_valid_vectors, 26] = 1

                assert map_feature.shape[0] > 0
                avg_position = ((map_feature[:num_valid_vectors, 0:3] + map_feature[:num_valid_vectors, 3:6]) /
                                2).mean(axis=0)
                avg_heading = utils.wrap_to_pi(utils.average_angles(map_feature[:num_valid_vectors, 9]))

                # if i - start_index < max_vectors:
                #     map_feature = np.pad(map_feature, pad_width=((0, max_vectors - (i - start_index)), (0, 0)))
                #     valid_mask = np.pad(valid_mask, pad_width=(0, max_vectors - (i - start_index)))

                valid_mask = map_feature[:, 26].copy()

                map_feature_list.append(map_feature)
                map_valid_mask_list.append(valid_mask)

                map_heading_list.append(avg_heading)
                map_position_list.append(avg_position)

                start_index = i
                road_length = 0.0

        # if MetaDriveType.is_lane(map_feat['type']):
        #     map_id_of_lanes.append(map_feat_id)

    if len(map_feature_list) == 0:
        map_feature_position = np.zeros([0, 0, 3], dtype=np.float32)
        map_feature_heading = np.zeros([0, 0], dtype=np.float32)
    else:
        map_feature_position = np.stack(map_position_list, axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_heading = np.stack(map_heading_list, axis=0).astype(np.float32)  # [num map feat, 2]
    # print(f"# MAP FEATURES: {len(map_position_list)}, # Avg Vectors: {np.mean(np.sum(map_valid_mask_list, axis=1), axis=0)}, # Max Vectors: {np.max(np.sum(map_valid_mask_list, axis=1), axis=0)}" )

    # Filter out too many map features
    if limit_map_range:
        # crop map to 50m range within SDC's position
        sdc_id = scenario['metadata']['sdc_id']
        sdc_tracks = scenario['tracks'][sdc_id]['state']['position']
        current_step = scenario['metadata']['current_time_index']
        sdc_position = sdc_tracks[current_step] - map_center

        map_feature_position = np.stack(map_position_list)

        valid_map_feat = (
                (abs(map_feature_position[..., 0] - sdc_position[0]) < 50) &
                (abs(map_feature_position[..., 1] - sdc_position[1]) < 50)
        )
        indices = valid_map_feat.nonzero()[0]
        map_feature_position = map_feature_position[indices]
        map_feature_heading = np.stack([map_feature_heading[i] for i in indices],
                                       axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_list = [map_feature_list[i] for i in indices]
        map_valid_mask_list = [map_valid_mask_list[i] for i in indices]

    if len(map_feature_position) > max_map_features:
        # Sorted based on the distance to the SDC
        sdc_id = scenario['metadata']['sdc_id']
        sdc_tracks = scenario['tracks'][sdc_id]['state']['position']
        current_step = scenario['metadata']['current_time_index']
        sdc_position = sdc_tracks[current_step] - map_center

        dist = np.linalg.norm(map_feature_position[:, :2] - sdc_position[:2], axis=1)

        indices = np.argsort(dist)[:max_map_features]
        map_feature_position = map_feature_position[indices]
        map_feature_heading = np.stack([map_feature_heading[i] for i in indices],
                                       axis=0).astype(np.float32)  # [num map feat, 2]
        map_feature_list = [map_feature_list[i] for i in indices]
        map_valid_mask_list = [map_valid_mask_list[i] for i in indices]

    if len(map_valid_mask_list) > 0:
        map_feature = np.stack(map_feature_list, axis=0).astype(np.float32)  # [num map feat, max vectors, 27]
        assert map_feature.shape[-1] == constants.MAP_FEATURE_STATE_DIM
        map_feature_mask = np.stack(map_valid_mask_list, axis=0).astype(bool)  # [num map feat, max vectors]
        map_feature_heading = np.stack(map_feature_heading, axis=0).astype(np.float32)  # [num map feat, max vectors]

    else:
        map_feature = np.zeros([0, max_vectors, constants.MAP_FEATURE_STATE_DIM], dtype=np.float32)
        map_feature_mask = np.zeros([0, max_vectors], dtype=bool)
        map_feature_position = np.zeros([0, 3], dtype=np.float32)
        map_feature_heading = np.zeros([0], dtype=np.float32)

    num_map_feat = map_feature.shape[0]
    utils.assert_shape(map_feature, (num_map_feat, max_vectors, constants.MAP_FEATURE_STATE_DIM))
    utils.assert_shape(map_feature_mask, (
        num_map_feat,
        max_vectors,
    ))
    utils.assert_shape(map_feature_position, (num_map_feat, 3))
    utils.assert_shape(map_feature_heading, (num_map_feat, ))

    # num_lights = traffic_light_valid_mask.any(axis=0).sum()
    # print("num_lights: ", num_lights)

    data_dict.update(
        {
            "encoder/map_feature": map_feature,
            "encoder/map_position": map_feature_position,
            "encoder/map_heading": map_feature_heading,
            "encoder/map_valid_mask": map_feature_mask.sum(axis=-1) != 0,  # Token valid mask
            "encoder/map_feature_valid_mask": map_feature_mask,
            # "encoder/traffic_light_feature": traffic_light_feature,
            # "encoder/traffic_light_position": traffic_light_position,
            # "encoder/traffic_light_heading": traffic_light_heading,
            # "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
            "metadata/map_center": map_center,
            "metadata/map_heading": map_heading,
        }
    )

    data_dict = process_traffic_light(
        data_dict,
        map_feature,
        dynamic_map_states,
        track_length,
        max_vectors,
        max_map_features,
        max_length_per_map_feature,
        max_traffic_lights,
        map_center,
        map_heading,
        remove_traffic_light_state=remove_traffic_light_state
    )
    return data_dict


def process_traffic_light(
    data_dict, map_feature, dynamic_map_states, track_length, max_vectors, max_map_features, max_length_per_map_feature,
    max_traffic_lights, map_center, map_heading, remove_traffic_light_state
):

    # ===== Extract traffic light features =====
    traffic_light_position = np.zeros([max_traffic_lights, 3], dtype=np.float32)

    if remove_traffic_light_state:
        traffic_light_heading = np.zeros([max_traffic_lights], dtype=np.float32)
        traffic_light_feature = np.zeros([max_traffic_lights, constants.TRAFFIC_LIGHT_STATE_DIM], dtype=np.float32)
        traffic_light_valid_mask = np.zeros([max_traffic_lights], dtype=bool)
        for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
            traffic_light_state = [v for v in traffic_light["state"]["object_state"] if v is not None]
            tl_states, tl_counts = np.unique(traffic_light_state, return_counts=True)
            tl_state = str(tl_states[np.argmax(tl_counts)])
            stop_point = centralize_to_map_center(
                position_array=traffic_light["stop_point"], map_center=map_center, map_heading=map_heading
            )
            traffic_light_position[tl_count] = stop_point[..., :3]
            traffic_light_feature[tl_count, :3] = stop_point
            traffic_light_feature[tl_count, 3] = MetaDriveType.is_traffic_light_in_green(tl_state)
            traffic_light_feature[tl_count, 4] = MetaDriveType.is_traffic_light_in_yellow(tl_state)
            traffic_light_feature[tl_count, 5] = MetaDriveType.is_traffic_light_in_red(tl_state)
            traffic_light_feature[tl_count, 6] = MetaDriveType.is_traffic_light_unknown(tl_state)
            traffic_light_valid_mask[tl_count] = True
    else:
        traffic_light_heading = np.zeros([
            max_traffic_lights,
        ], dtype=np.float32) + constants.HEADING_PLACEHOLDER
        traffic_light_feature = np.zeros(
            [track_length, max_traffic_lights, constants.TRAFFIC_LIGHT_STATE_DIM], dtype=np.float32
        )
        traffic_light_valid_mask = np.zeros([track_length, max_traffic_lights], dtype=bool)

        for tl_count, (traffic_light_index, traffic_light) in enumerate(dynamic_map_states.items()):
            stop_point = centralize_to_map_center(
                position_array=traffic_light["stop_point"], map_center=map_center, map_heading=map_heading
            )

            traffic_light_position[tl_count] = stop_point[..., :3]
            for step in range(track_length):
                assert traffic_light['type'] == MetaDriveType.TRAFFIC_LIGHT
                traffic_light_state = {k: v[step] for k, v in traffic_light["state"].items()}
                traffic_light_feature[step, tl_count, :3] = stop_point
                traffic_light_feature[step, tl_count,
                                      3] = MetaDriveType.is_traffic_light_in_green(traffic_light_state["object_state"])
                traffic_light_feature[step, tl_count, 4] = MetaDriveType.is_traffic_light_in_yellow(
                    traffic_light_state["object_state"]
                )
                traffic_light_feature[step, tl_count,
                                      5] = MetaDriveType.is_traffic_light_in_red(traffic_light_state["object_state"])
                traffic_light_feature[step, tl_count,
                                      6] = MetaDriveType.is_traffic_light_unknown(traffic_light_state["object_state"])
                traffic_light_valid_mask[step, tl_count] = True
            if tl_count > max_traffic_lights:
                logger.debug(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")
                print(f"WARNING: {len(dynamic_map_states)} exceeds {max_traffic_lights} traffic lights!")
                break

    data_dict.update(
        {
            "encoder/traffic_light_feature": traffic_light_feature,
            "encoder/traffic_light_position": traffic_light_position,
            "encoder/traffic_light_heading": traffic_light_heading,
            "encoder/traffic_light_valid_mask": traffic_light_valid_mask,
        }
    )
    return data_dict


def filter_and_reorder_agent(data_dict, max_agents=None):
    """
    Put modeled agents and SDC to the first place.
    """
    num_agents = data_dict["encoder/agent_feature"].shape[1]
    agent_valid_mask = data_dict["encoder/agent_valid_mask"]
    modeled_agent_indices = data_dict["encoder/object_of_interest_id"]

    sdc_index = data_dict["encoder/sdc_index"]
    new_sdc_index = sdc_index

    # Sort agent based on validity. Put useless agent to the back.
    index_to_validity = []
    for agent_index in range(num_agents):
        index_to_validity.append((agent_index, agent_valid_mask[:, agent_index].sum()))
    sorted_indices = sorted(index_to_validity, key=lambda v: v[1], reverse=True)
    selected_agents = [key for key, _ in sorted_indices]

    if modeled_agent_indices is not None:
        for agent_index in modeled_agent_indices:
            selected_agents.remove(agent_index)
            selected_agents.insert(0, int(agent_index))

    # Put SDC to first place.
    assert sdc_index in selected_agents
    selected_agents.remove(sdc_index)
    selected_agents.insert(0, sdc_index)
    new_sdc_index = 0
    # new_sdc_index = selected_agents.index(sdc_index)

    if max_agents is not None:
        selected_agents = selected_agents[:max_agents]

    selected_agents = np.asarray(selected_agents, dtype=int)

    # ===== Reorder all data =====
    # Those data whose first dim is the agent dim:
    for key in [
            "encoder/agent_type",
            "encoder/current_agent_shape",
            "encoder/current_agent_valid_mask",
            "encoder/current_agent_position",
            "encoder/current_agent_heading",
            "encoder/current_agent_velocity",
            "encoder/track_name",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=selected_agents, agent_dim=0)
    # Those data whose second dim is the agent dim:
    for key in [
            "encoder/agent_feature",
            "encoder/agent_valid_mask",
            "encoder/agent_position",
            "encoder/agent_velocity",
            "encoder/agent_heading",
            # "encoder/future_agent_position",
            # "encoder/future_agent_heading",
            # "encoder/future_agent_valid_mask",
            "encoder/agent_shape",
    ]:
        data_dict[key] = extract_data_by_agent_indices(data_dict[key], agent_indices=selected_agents, agent_dim=1)

    # ===== Reorder modeled agents and SDC, change modeled_agent_indices if necessary =====
    if modeled_agent_indices is not None:
        # Need to translate track_index_to_predict
        new_modeled_agent_indices = []
        for old_agent_index in modeled_agent_indices:
            for new_ind, old_ind in enumerate(selected_agents):
                if old_agent_index == old_ind:
                    new_modeled_agent_indices.append(new_ind)
                    break
        assert len(new_modeled_agent_indices) == len(modeled_agent_indices)
        modeled_agent_indices = new_modeled_agent_indices
    new_sdc_index = 0
    if modeled_agent_indices is not None:
        # Also update SDC index
        if new_sdc_index in modeled_agent_indices:
            modeled_agent_indices.remove(new_sdc_index)
            modeled_agent_indices.insert(0, new_sdc_index)

    data_dict["encoder/sdc_index"] = new_sdc_index
    data_dict["encoder/object_of_interest_id"] = np.asarray(modeled_agent_indices)
    # Note that new ooi id doesn't change the order. So no need to change ooi name.

    return data_dict


def process_track(
    *,
    data_dict,
    tracks,
    track_length,
    sdc_name,  # We need to translate sdc_name to sdc_index
    max_agents,
    exempt_max_agent_filtering=False,
):
    map_center = data_dict["metadata/map_center"]
    map_heading = data_dict["metadata/map_heading"]
    current_t = data_dict["metadata/current_time_index"]

    agent_feature_dict = {}
    agent_valid_mask_dict = {}
    agent_velocity_dict = {}
    agent_position_dict = {}
    agent_heading_dict = {}
    agent_type_dict = {}
    agent_shape_dict = {}
    sdc_index = None
    sdc_name = str(sdc_name)

    valid_track_names = []
    track_count = 0

    for _, (track_name, cur_data) in enumerate(tracks.items()):  # number of objects

        # if not cur_data['type'] == 'VEHICLE': # CAT contains pedestrains which does not contain length, width, and height
        #     continue

        if not MetaDriveType.is_participant(cur_data["type"]):
            # TODO(pzh): TrafficCone is in tracks for some reason. Looks very weird. Might be some bug.
            continue
        track_name = str(track_name)

        if track_name == sdc_name:
            sdc_index = track_count

        cur_state = cur_data[SD.STATE]

        rotated_positions = centralize_to_map_center(
            position_array=cur_state["position"],  # [T, 3]
            map_center=map_center,
            map_heading=map_heading
        )  # [T, num agents, 3]

        rotated_heading = utils.wrap_to_pi(cur_state["heading"] - map_heading)  # [T, num agents]
        rotated_velocity = centralize_to_map_center(
            position_array=cur_state["velocity"], map_center=None, map_heading=map_heading
        )[..., :2]  # [T, num agents, 2]

        agent_shape_dict[track_name] = np.stack(
            [cur_state["length"].reshape(-1), cur_state["width"].reshape(-1), cur_state["height"].reshape(-1)], axis=1
        )  # (T, N, 3)

        speed = np.linalg.norm(cur_state["velocity"], axis=1)

        valid_mask = np.asarray(cur_state["valid"], dtype=bool)

        agent_state = np.zeros([track_length, constants.AGENT_STATE_DIM], dtype=np.float32)

        # print("shape of rotated_positions", rotated_positions.shape)
        # for CAT: we need to pad position dimension
        if rotated_positions.shape[1] != 3:
            rotated_positions = np.concatenate([rotated_positions, np.zeros((rotated_positions.shape[0], 1))], axis=-1)

        agent_state[:, :3] = rotated_positions
        agent_state[:, 3] = rotated_heading
        agent_state[:, 4] = np.sin(rotated_heading)
        agent_state[:, 5] = np.cos(rotated_heading)
        agent_state[:, 6:8] = rotated_velocity

        agent_state[:, 8] = speed

        agent_state[:, 9] = cur_state["length"].reshape(-1)
        agent_state[:, 10] = cur_state["width"].reshape(-1)
        agent_state[:, 11] = cur_state["height"].reshape(-1)

        agent_state[~valid_mask] = 0
        agent_state[:, 12] = MetaDriveType.is_vehicle(cur_data["type"])
        agent_state[:, 13] = MetaDriveType.is_pedestrian(cur_data["type"])
        agent_state[:, 14] = MetaDriveType.is_cyclist(cur_data["type"])
        agent_state[:, 15] = valid_mask

        # TODO(pzh): Remove mapping
        assert cur_data["type"] in constants.object_type_to_int

        agent_feature_dict[track_name] = agent_state
        agent_valid_mask_dict[track_name] = valid_mask

        agent_position_dict[track_name] = rotated_positions * valid_mask.reshape(-1, 1)
        agent_heading_dict[track_name] = rotated_heading * valid_mask
        agent_velocity_dict[track_name] = rotated_velocity * valid_mask.reshape(-1, 1)

        # TODO(pzh): Remove mapping
        agent_type_dict[track_name] = constants.object_type_to_int[cur_data["type"]]

        valid_track_names.append(str(track_name))

        track_count += 1

    assert sdc_index is not None

    # ===== Store all data into dict =====
    agent_feature = np.stack(list(agent_feature_dict.values()), axis=1)  # [T, ]
    num_agents = agent_feature.shape[1]
    utils.assert_shape(agent_feature, (track_length, num_agents, constants.AGENT_STATE_DIM))

    agent_valid_mask = np.stack(list(agent_valid_mask_dict.values()), axis=1).astype(bool)
    utils.assert_shape(agent_valid_mask, (
        track_length,
        num_agents,
    ))

    agent_position = np.stack(list(agent_position_dict.values()), axis=1)
    utils.assert_shape(agent_position, (track_length, num_agents, 3))

    agent_velocity = np.stack(list(agent_velocity_dict.values()), axis=1)
    utils.assert_shape(agent_velocity, (track_length, num_agents, 2))

    agent_heading = np.stack(list(agent_heading_dict.values()), axis=1)
    utils.assert_shape(agent_heading, (track_length, num_agents))

    agent_type = np.stack(list(agent_type_dict.values()), axis=0).astype(int)
    utils.assert_shape(agent_type, (num_agents, ))

    agent_shape = np.stack(list(agent_shape_dict.values()), axis=1)
    utils.assert_shape(agent_shape, (track_length, num_agents, 3))

    data_dict["encoder/agent_feature"] = agent_feature.astype(np.float32)  # [T, num agent, D_agent]
    data_dict["encoder/agent_valid_mask"] = agent_valid_mask.astype(bool)  # [T, num agent]
    data_dict["encoder/agent_position"] = agent_position.astype(np.float32)
    data_dict["encoder/agent_velocity"] = agent_velocity.astype(np.float32)
    data_dict["encoder/agent_heading"] = agent_heading.astype(np.float32)
    data_dict["encoder/agent_type"] = agent_type

    # data_dict["encoder/future_agent_position"] = agent_position.astype(np.float32)[current_t + 1:]
    # data_dict["encoder/future_agent_heading"] = agent_heading.astype(np.float32)[current_t + 1:]
    # data_dict["encoder/future_agent_valid_mask"] = agent_valid_mask.astype(bool)[current_t + 1:]
    # data_dict["encoder/future_agent_velocity"] = agent_velocity.astype(np.float32)[current_t + 1:]
    data_dict["encoder/current_agent_valid_mask"] = agent_valid_mask.astype(bool)[current_t]
    data_dict["encoder/current_agent_position"] = agent_position.astype(np.float32)[current_t]
    data_dict["encoder/current_agent_heading"] = agent_heading.astype(np.float32)[current_t]
    data_dict["encoder/current_agent_velocity"] = agent_velocity.astype(np.float32)[current_t]

    data_dict["encoder/track_name"] = np.array(valid_track_names, dtype=str)
    data_dict["encoder/agent_shape"] = agent_shape.astype(np.float32)
    data_dict["encoder/current_agent_shape"] = data_dict["encoder/agent_shape"][current_t]
    data_dict["encoder/sdc_index"] = sdc_index

    # ===== Process the case where the number of agents exceeds max_agents =====
    data_dict = filter_and_reorder_agent(data_dict, max_agents=max_agents if not exempt_max_agent_filtering else None)

    # Add agent ID:
    num_agents = data_dict["encoder/agent_feature"].shape[1]
    data_dict["encoder/agent_id"] = np.arange(num_agents)

    # assert (data_dict["decoder/current_agent_valid_mask"] == data_dict["encoder/agent_valid_mask"][current_t]).all()
    return data_dict


def prepare_modeled_agent_and_eval_data(
    data_dict, predict_all_agents, eval_all_agents, current_t, add_sdc_to_object_of_interest
):
    # ===== Need to extract only the modeled agents for decoder and GT =====
    object_of_interest = data_dict["encoder/object_of_interest_id"]

    if predict_all_agents:
        modeled_agent_indices = list(data_dict["encoder/current_agent_valid_mask"].nonzero()[0])

        # In the following code, we will select only the valid agents at this step as modeled agents.
        # After the selection, the order of agents will change (again ..). So the object_of_interests
        # should also be changed.

        new_object_of_interests = []
        for old_agent_index in object_of_interest:
            for new_ind, old_ind in enumerate(modeled_agent_indices):
                if old_agent_index == old_ind:
                    new_object_of_interests.append(new_ind)
                    break
        assert len(new_object_of_interests) == len(object_of_interest)

        data_dict["decoder/sdc_index"] = modeled_agent_indices.index(data_dict["encoder/sdc_index"])
        data_dict["decoder/object_of_interest_id"] = np.asarray(new_object_of_interests)
        # Note that new ooi id doesn't change the order. So no need to change ooi name

    else:
        raise ValueError("Not sure what will happen...")
        object_of_interest = data_dict["encoder/object_of_interest_id"]
        modeled_agent_indices = object_of_interest
        # object_of_interest don't change
        assert eval_all_agents is False
        data_dict["decoder/object_of_interest_id"] = np.arange(len(object_of_interest))

    data_dict["decoder/agent_id"] = np.arange(len(modeled_agent_indices))

    assert modeled_agent_indices is not None

    data_dict["decoder/agent_type"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_type"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/track_name"] = extract_data_by_agent_indices(
        data_dict["encoder/track_name"], modeled_agent_indices, agent_dim=0, fill=-1
    )
    data_dict["encoder/modeled_agent_id"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_id"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["encoder/modeled_agent_type"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_type"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_valid_mask"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_valid_mask"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_position"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_position"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_heading"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_heading"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_shape"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_shape"], agent_indices=modeled_agent_indices, agent_dim=0
    )
    data_dict["decoder/current_agent_velocity"] = extract_data_by_agent_indices(
        data_dict["encoder/current_agent_velocity"], agent_indices=modeled_agent_indices, agent_dim=0
    )

    # agent_dim = 1
    # data_dict["decoder/future_agent_position"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_position"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_heading"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_heading"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_velocity"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_velocity"], agent_indices=modeled_agent_indices, agent_dim=1
    # )
    # data_dict["decoder/future_agent_valid_mask"] = extract_data_by_agent_indices(
    #     data_dict["encoder/future_agent_valid_mask"], agent_indices=modeled_agent_indices, agent_dim=1
    # )

    data_dict["decoder/agent_position"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_position"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_velocity"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_velocity"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_heading"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_heading"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_valid_mask"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_valid_mask"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/agent_shape"] = extract_data_by_agent_indices(
        data_dict["encoder/agent_shape"], modeled_agent_indices, agent_dim=1
    )
    data_dict["decoder/object_of_interest_name"] = data_dict["encoder/object_of_interest_name"]

    if add_sdc_to_object_of_interest:

        if data_dict["metadata/sdc_name"] not in data_dict["encoder/object_of_interest_name"]:
            data_dict["encoder/object_of_interest_name"] = np.concatenate(
                [[data_dict["metadata/sdc_name"]], data_dict["encoder/object_of_interest_name"]]
            )
        else:
            assert data_dict["metadata/sdc_name"] == data_dict["encoder/object_of_interest_name"][0]

        if data_dict["metadata/sdc_name"] not in data_dict["decoder/object_of_interest_name"]:
            data_dict["decoder/object_of_interest_name"] = np.concatenate(
                [[data_dict["metadata/sdc_name"]], data_dict["encoder/object_of_interest_name"]]
            )
        else:
            assert data_dict["metadata/sdc_name"] == data_dict["decoder/object_of_interest_name"][0]

    # Evaluation data: all with leading dimensions: (num of interested objects, T, ...)
    # If not eval all agents, a new index system `eval/` is introduced.
    if eval_all_agents:
        pass
        # data_dict["eval/track_name"] = data_dict["decoder/track_name"]
        # data_dict["eval/agent_type"] = data_dict["decoder/agent_type"]
        # data_dict["eval/agent_position"] = data_dict["decoder/agent_position"]
        # data_dict["eval/agent_velocity"] = data_dict["decoder/agent_velocity"]
        # data_dict["eval/agent_heading"] = data_dict["decoder/agent_heading"]
        # data_dict["eval/agent_valid_mask"] = data_dict["decoder/agent_valid_mask"]
        # data_dict["eval/agent_shape"] = data_dict["decoder/agent_shape"]
    else:
        assert new_object_of_interests is not None
        decoder_ooi_id = new_object_of_interests
        data_dict["eval/track_name"] = extract_data_by_agent_indices(
            data_dict["decoder/track_name"], decoder_ooi_id, agent_dim=0
        )
        data_dict["eval/agent_type"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_type"], decoder_ooi_id, agent_dim=0
        )
        data_dict["eval/agent_position"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_position"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_velocity"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_velocity"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_heading"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_heading"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_valid_mask"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_valid_mask"], decoder_ooi_id, agent_dim=1
        )
        data_dict["eval/agent_shape"] = extract_data_by_agent_indices(
            data_dict["decoder/agent_shape"], decoder_ooi_id, agent_dim=1
        )
        assert data_dict["eval/agent_valid_mask"][current_t].all()  # not all object_of_interest is in CAT

    return data_dict


def preprocess_scenario_description(*args, config, **kwargs):
    # if scenario['length'] < 5: # TODO: filter out CAT data that is not valid
    #     return None
    # TODO: combine all cat info dictionary .pkl files into one and provide the paths in the config
    if config.MODEL.NAME in ["motionlm", "gpt"]:
        return preprocess_scenario_description_for_motionlm(*args, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {config.MODEL.NAME}")


def translate_abs_info_to_ego_centric(data_dict, current_t, retain_raw=False):

    if retain_raw:
        data_dict["vis/map_feature"] = data_dict["encoder/map_feature"].copy()
    data_dict["raw/map_feature"] = data_dict["encoder/map_feature"].copy()

    def _get_last_pos(pos, head, valid_mask):
        T, N = valid_mask.shape
        ind = np.arange(T).reshape(-1, 1).repeat(N, axis=1)  # T, N
        ind[~valid_mask] = 0
        ind = ind.max(axis=0)
        out = np.take_along_axis(pos, indices=ind.reshape(1, N, 1), axis=0)
        outh = np.take_along_axis(head, indices=ind.reshape(1, N), axis=0)
        out = np.squeeze(out, axis=0)
        outh = np.squeeze(outh, axis=0)
        return out, outh

    # === Agent features ===
    agent_p, agent_h = _get_last_pos(
        data_dict["encoder/agent_position"][:current_t + 1], data_dict["encoder/agent_heading"][:current_t + 1],
        data_dict["encoder/agent_valid_mask"][:current_t + 1]
    )

    pos = data_dict["encoder/agent_feature"][..., :3]
    pos = pos - agent_p[None]
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-agent_h.reshape(1, -1).repeat(pos.shape[0], axis=0), z=pos[..., 2]
    )
    data_dict["encoder/agent_feature"][..., :3] = pos

    head = data_dict["encoder/agent_feature"][..., 3]
    head = utils.wrap_to_pi(head - agent_h[None])
    data_dict["encoder/agent_feature"][..., 3] = head
    data_dict["encoder/agent_feature"][..., 4] = np.sin(head)
    data_dict["encoder/agent_feature"][..., 5] = np.cos(head)

    vel = data_dict["encoder/agent_feature"][..., 6:8]
    vel = utils.rotate(
        x=vel[..., 0],
        y=vel[..., 1],
        angle=-agent_h.reshape(1, -1).repeat(vel.shape[0], axis=0),
    )
    data_dict["encoder/agent_feature"][..., 6:8] = vel

    data_dict["encoder/agent_feature"][~data_dict["encoder/agent_valid_mask"]] = 0

    # === Map features ===
    map_pos = data_dict["encoder/map_position"][:, None]
    map_h = data_dict["encoder/map_heading"][:, None]

    pos = data_dict["encoder/map_feature"][..., :3] - map_pos
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., :3] = pos

    pos = data_dict["encoder/map_feature"][..., 3:6] - map_pos
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., 3:6] = pos

    pos = data_dict["encoder/map_feature"][..., 6:9]  # direction, no need to translate
    pos = utils.rotate(
        x=pos[..., 0], y=pos[..., 1], angle=-map_h.reshape(-1, 1).repeat(pos.shape[1], axis=1), z=pos[..., 2]
    )
    data_dict["encoder/map_feature"][..., 6:9] = pos

    head = data_dict["encoder/map_feature"][..., 9]
    head = utils.wrap_to_pi(head - map_h)
    data_dict["encoder/map_feature"][..., 9] = head
    data_dict["encoder/map_feature"][..., 10] = np.sin(head)
    data_dict["encoder/map_feature"][..., 11] = np.cos(head)

    # === Traffic light features ===
    # Note: We want to remove all absolute information so just remove traffic light position!
    data_dict["encoder/traffic_light_feature"][..., :3] = 0

    return data_dict


def limit_map_range(data_dict, limit_range=50):
    sdc_index = data_dict["decoder/sdc_index"]
    current_t = data_dict["metadata/current_time_index"]
    sdc_center = data_dict["decoder/agent_position"][current_t, sdc_index]  # (3,)

    # Limit the map range
    margin = 0
    valid_map_feat = (
        (abs(data_dict["encoder/map_position"][..., 0] - sdc_center[0]) < limit_range + margin) &
        (abs(data_dict["encoder/map_position"][..., 1] - sdc_center[1]) < limit_range + margin)
    )
    valid_map_feat = valid_map_feat & data_dict["encoder/map_valid_mask"]
    data_dict["encoder/map_feature_valid_mask"][~valid_map_feat] = False
    data_dict["encoder/map_valid_mask"][~valid_map_feat] = False

    # Delete agents that are out of the map range
    agent_pos = data_dict["encoder/agent_position"][current_t]
    distance_mask = (
            (abs(agent_pos[..., 0] - sdc_center[0]) < limit_range) &
            (abs(agent_pos[..., 1] - sdc_center[1]) < limit_range)
    )
    data_dict["encoder/agent_valid_mask"][current_t] = (
            data_dict["encoder/agent_valid_mask"][current_t] & distance_mask
    )
    data_dict["encoder/current_agent_valid_mask"] = data_dict["encoder/agent_valid_mask"][current_t].copy()
    agent_pos = data_dict["decoder/agent_position"][current_t]
    distance_mask = (
            (abs(agent_pos[..., 0] - sdc_center[0]) < limit_range) &
            (abs(agent_pos[..., 1] - sdc_center[1]) < limit_range)
    )
    data_dict["decoder/agent_valid_mask"][current_t] = (
            data_dict["decoder/agent_valid_mask"][current_t] & distance_mask
    )
    data_dict["decoder/current_agent_valid_mask"] = data_dict["decoder/agent_valid_mask"][current_t].copy()

    # TODO: eval/agent_valid_mask is not touched yet. But it's fine now...
    return data_dict


def preprocess_scenario_description_for_motionlm(
    scenario, config, in_evaluation, keep_all_data=False, backward_prediction=None, tokenizer=None
):
    original_SD = copy.deepcopy(scenario)

    backward_prediction = config.eval_backward_model or backward_prediction  # for eval backward model
    metadata = scenario[SD.METADATA]

    tracks_to_predict_dict = metadata.get('tracks_to_predict', {})
    track_index_to_predict = np.array([int(v['track_index']) for v in tracks_to_predict_dict.values()])
    track_name_to_predict = [int(k) for k in tracks_to_predict_dict.keys()]

    # Put SDC name to the first place.
    sdc_name = metadata["sdc_id"]
    try:
        sdc_name = int(sdc_name)
    except:
        pass
    if sdc_name in track_name_to_predict:
        track_name_to_predict.remove(sdc_name)
        track_name_to_predict.insert(0, sdc_name)
    track_name_to_predict = np.array(track_name_to_predict)

    data_dict = {
        "in_evaluation": in_evaluation,
        "metadata/sdc_name": sdc_name,
        "encoder/object_of_interest_name": track_name_to_predict,
        "encoder/object_of_interest_id": track_index_to_predict,
        "scenario_id": scenario[SD.ID]
    }
    if "current_time_index" in metadata:
        data_dict["metadata/current_time_index"] = metadata['current_time_index']
    else:
        # TODO: Not sure in nuscenes if there is no current_time_index. Might need to check.
        data_dict["metadata/current_time_index"] = 0
        metadata['current_time_index'] = 0

    # ===== Extract map and traffic light features =====
    data_dict = process_map_and_traffic_light(
        data_dict=data_dict,
        scenario=scenario,
        map_feature=scenario[SD.MAP_FEATURES],
        dynamic_map_states=scenario[SD.DYNAMIC_MAP_STATES],
        track_length=scenario[SD.LENGTH],
        max_vectors=config.PREPROCESSING.MAX_VECTORS,
        max_map_features=config.PREPROCESSING.MAX_MAP_FEATURES,
        limit_map_range=config.LIMIT_MAP_RANGE,
        max_length_per_map_feature=config.PREPROCESSING.MAX_LENGTH_PER_MAP_FEATURE,
        max_traffic_lights=config.PREPROCESSING.MAX_TRAFFIC_LIGHTS,
        remove_traffic_light_state=config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE,
    )

    # ===== Extract agent features =====
    data_dict = process_track(
        data_dict=data_dict,
        tracks=scenario[SD.TRACKS],
        track_length=scenario[SD.LENGTH],
        sdc_name=metadata["sdc_id"],
        max_agents=config.PREPROCESSING.MAX_AGENTS,
        exempt_max_agent_filtering=in_evaluation,
    )
    data_dict = prepare_modeled_agent_and_eval_data(
        data_dict=data_dict,
        predict_all_agents=config.TRAINING.PREDICT_ALL_AGENTS,
        eval_all_agents=config.EVALUATION.PREDICT_ALL_AGENTS,
        current_t=metadata['current_time_index'],
        add_sdc_to_object_of_interest=config.PREPROCESSING.ADD_SDC_TO_OBJECT_OF_INTEREST
    )

    if config.LIMIT_MAP_RANGE:
        data_dict = limit_map_range(data_dict)

    if config.MODEL.RELATIVE_PE:
        data_dict = translate_abs_info_to_ego_centric(
            data_dict, current_t=data_dict["metadata/current_time_index"], retain_raw=keep_all_data
        )

    # if use_action_label:
    sdc_ind = data_dict["decoder/sdc_index"]
    object_of_interest = data_dict["decoder/object_of_interest_id"]
    if sdc_ind not in object_of_interest:
        object_of_interest = np.concatenate([[sdc_ind], object_of_interest])
    data_dict["decoder/labeled_agent_id"] = np.asarray(object_of_interest).astype(int)

    # ===== Call the tokenizer and generate target discretized actions =====
    # Error stats is removed from here. It's used in independent test script.
    use_backward_prediction = config.BACKWARD_PREDICTION
    if in_evaluation:
        use_backward_prediction = False
    if use_backward_prediction:
        # Use 50% probability to set backward_prediction to True
        use_backward_prediction = np.random.rand() < 0.5
    if backward_prediction is not None:  # Overwrite the value
        use_backward_prediction = backward_prediction
    use_backward_prediction = use_backward_prediction or config.eval_backward_model

    if config.TOKENIZATION.TOKENIZATION_METHOD is not None:
        detok, error_stat = tokenizer.tokenize_numpy_array(data_dict, backward_prediction=use_backward_prediction)
        for k in ["decoder/target_action", "decoder/target_action_valid_mask", "decoder/input_action",
                    "decoder/input_action_valid_mask", "decoder/modeled_agent_position",
                    "decoder/modeled_agent_heading", "decoder/modeled_agent_velocity", "decoder/modeled_agent_delta",
                    "in_backward_prediction"]:
            if k in detok:
                data_dict[k] = detok[k]

    if config.ACTION_LABEL.USE_ACTION_LABEL:
        data_dict = prepare_action_label(
            data_dict=data_dict,
            dt=0.1,  # TODO(PZH): Hardcoded here.
            config=config,
            mask_probability=config.ACTION_LABEL.MASK_PROBABILITY_ACTION_LABEL if in_evaluation else 0.0
        )

    if config.get("ACTION_LABEL") and config.ACTION_LABEL.USE_SAFETY_LABEL:
        data_dict = prepare_safety_label(
            data_dict=data_dict,
            dt=0.1,  # TODO(PZH): Hardcoded here.
            config=config,
            mask_probability=config.ACTION_LABEL.MASK_PROBABILITY_SAFETY_LABEL if in_evaluation else 0.0
        )

    # TODO: A little hack here...
    if config.EVALUATION.NAME == "lmdb":
        keep_all_data = False
        in_evaluation = False

    if not keep_all_data:
        if in_evaluation:
            pass
            # data_dict = {k: v for k, v in data_dict.items() if not k.startswith("decoder/")}  # Remove decoder/ data
        else:

            for pattern in [
                    "eval/",
                    "encoder/current_",
                    "encoder/future_",
            ]:
                data_dict = {k: v for k, v in data_dict.items() if not k.startswith(pattern)}

            if config.GPT_STYLE and config.REMOVE_AGENT_FROM_SCENE_ENCODER:
                data_dict = {k: v for k, v in data_dict.items() if not k.startswith("encoder/agent_")}

            new_data_dict = {}
            for pattern in ["scenario_id", "decoder/label_", "decoder/agent_id", "decoder/agent_type",
                            "decoder/current_", "decoder/modeled_agent_", "decoder/input_", "decoder/target_",
                            "encoder/", "in_evaluation", "in_backward_prediction"]:
                new_data_dict.update({k: v for k, v in data_dict.items() if k.startswith(pattern)})
            data_dict = new_data_dict

    sorted_keys = sorted(data_dict.keys())
    data_dict = {k: data_dict[k] for k in sorted_keys}
    
    if "selected_adv_id" in metadata:  # For evaluating CAT
        data_dict["metadata/waymo_adv_agent_id"] = scenario["metadata"]["selected_adv_id"]
        adv_agent = scenario["metadata"]["selected_adv_id"]

        decoder_adv_mask = (data_dict["decoder/track_name"] == adv_agent)
        decoder_adv_id = data_dict["decoder/agent_id"][decoder_adv_mask].item()

        data_dict["decoder/adv_agent_id"] = decoder_adv_id

    data_dict["original_SD"] = original_SD
    
    return data_dict
