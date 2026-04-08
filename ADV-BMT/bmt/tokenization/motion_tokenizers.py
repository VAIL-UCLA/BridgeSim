"""
This file implements how we translate a trajectory to a sequence of discretized actions
and the reverse process.
"""
import logging
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from bmt.utils import rotate, wrap_to_pi
from bmt.utils import utils

from scipy.interpolate import CubicSpline
FOLDER = pathlib.Path(__file__).resolve().parent

logger = logging.getLogger(__file__)

STEPS_PER_SECOND = 10

# TODO:
beta = 1.0

# Define a special action
START_ACTION = 1_000_000
END_ACTION = 7_777_777


def nucleus_sampling(logits, p=None, epsilon=1e-8):
    # TODO: duplicate code.
    p = p or 0.9

    # logits = logits.clamp(-20, 20)

    # Replace NaN and Inf values in logits to avoid errors in entropy computation
    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits).fill_(-1e9), logits)
    logits = torch.where(torch.isinf(logits), torch.zeros_like(logits).fill_(-1e9), logits)

    # Adding a small epsilon to logits to avoid log(0)
    # logits = logits + epsilon

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Sort the probabilities to identify the top-p cutoff
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold p
    cutoff_index = cumulative_probs > p
    # Shift the mask to the right to keep the first token above the threshold
    cutoff_index[..., 1:] = cutoff_index[..., :-1].clone()
    cutoff_index[..., 0] = False

    # Zero out the probabilities for tokens not in the top-p set
    sorted_probs.masked_fill_(cutoff_index, 0)

    # Recover the original order of the probabilities
    original_probs = torch.zeros_like(probs)
    original_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    # original_probs += epsilon

    # Sample from the adjusted probability distribution
    # try:
    sampled_token_index = torch.distributions.Categorical(probs=original_probs).sample()
    # except ValueError:
    #     import ipdb; ipdb.set_trace()
    #     print(1111111)

    return sampled_token_index


def get_bin_centers(x_min, x_max, y_min, y_max, x_num_bins, y_num_bins):
    # Create linearly spaced bins for x and y
    x_bins = np.linspace(x_min, x_max, x_num_bins)
    y_bins = np.linspace(y_min, y_max, y_num_bins)

    # Create a meshgrid of x and y bin coordinates
    x_grid, y_grid = np.meshgrid(x_bins, y_bins, indexing='ij')

    # Stack the grid coordinates to create the 2D bins
    xy_bins = np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2)
    assert xy_bins.shape == (x_num_bins * y_num_bins, 2)

    return xy_bins.astype(np.float32)


def infer_heading(*, current_pos, last_pos, last_heading, min_displacement=-777, flip_heading=False, **kwargs):
    assert min_displacement != -777

    if flip_heading:
        current_pos, last_pos = last_pos, current_pos

    # Ensure the input shapes are correct
    assert current_pos.shape == last_pos.shape
    B, T, N, D = current_pos.shape
    last_heading = last_heading.reshape(B, T, N)
    # Calculate displacement
    displacement = current_pos - last_pos
    if isinstance(current_pos, np.ndarray):
        heading = np.arctan2(displacement[..., 1], displacement[..., 0])
    else:
        heading = torch.arctan2(displacement[..., 1], displacement[..., 0])

    # if flip_heading:
    #     # Note that you can't flip heading after masking. Should do it before masking.
    #     heading = heading + np.pi

    # Apply the previous heading for static or minimally moving objects
    if min_displacement is not None:
        movement_mask = displacement.norm(dim=-1) >= min_displacement
        heading[~movement_mask] = last_heading[~movement_mask]
    heading = utils.wrap_to_pi(heading)

    # mask = utils.wrap_to_pi(heading - last_heading).abs() > np.deg2rad(90)
    # heading[mask] = last_heading[mask]

    return heading


def rotate_bin_to_absolute_heading(bin_center, heading):
    B, num_actions, N, _ = bin_center.shape
    y_axis_in_relative_coord = heading
    x_axis_in_relative_coord = y_axis_in_relative_coord - np.pi / 2
    abs_pos = rotate(bin_center[..., 0], bin_center[..., 1], x_axis_in_relative_coord.expand(B, num_actions, N))
    return abs_pos


def _reconstruct_delta_pos_from_abs_vel(vel, heading, dt):
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    vel = utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)
    pos = vel * dt
    return pos


def get_relative_velocity(vel, heading):
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    # TODO: WHAT"S WRONG HERE???????????
    return utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)


def interpolate_reconstructed_valid_mask(input_valid_mask, fine_factor=5):
    """
    It's quite tricky, for input mask: 0=T, 5=T, 10=T, 15=F
    We need to interpolate it to: 0=T, ..., 10=T, 11=T, ..., 14=T, 15=T, 16=F, ...
    This is because our model predicts future 5 steps so it's actually cover the last+1 macro step.
    """
    valid = input_valid_mask
    # Offset 1 step in the time dimension.

    B, T, N = input_valid_mask.shape

    valid = valid.reshape(B, -1, 1, N).expand(-1, -1, fine_factor, -1).reshape(B, -1, N)
    valid = torch.cat([valid, input_valid_mask[:, -1:]], dim=1)
    reconstructed_valid_mask = valid

    def find_last_valid(mask):
        B, T, N = mask.shape
        indices = mask * torch.arange(T, device=mask.device).reshape(1, T, 1).expand(*mask.shape)
        indices = indices.argmax(1, keepdims=True)
        return indices

    last_valid = find_last_valid(input_valid_mask)
    last_valid_plus_one = (last_valid + 1) * fine_factor
    last_valid_plus_one = torch.minimum(
        last_valid_plus_one, torch.tensor(reconstructed_valid_mask.shape[1] - 1, device=valid.device)
    )

    # Set last_valid to 1 in valid.
    reconstructed_valid_mask.scatter_(1, last_valid_plus_one, 1)
    reconstructed_valid_mask[~input_valid_mask.any(1, keepdims=True).expand_as(reconstructed_valid_mask)] = 0

    return reconstructed_valid_mask


def interpolate_trajectory_spline(pos, heading, vel, mask, fine_factor=5):
    """
    Interpolate pos, heading, vel from coarse steps (e.g., every 5 frames) to finer steps (every frame).

    Args:
        pos: (B, T, N, 2) positions at macro steps.
        heading: (B, T, N) headings at macro steps.
        vel: (B, T, N, 2) velocities at macro steps.
        original_times: array/list of shape (T,) with the original macro step times. For example, [0, 5, 10, 15, ...]
        fine_factor: How many subdivisions per macro step. If original spacing is 5 frames, fine_factor=5 means
                     you'll get one sample per frame.

    Returns:
        fine_times: (T_fine,) new time array with fine steps.
        pos_fine: (B, T_fine, N, 2)
        heading_fine: (B, T_fine, N)
        vel_fine: (B, T_fine, N, 2)
    """

    # Convert to numpy
    pos_np = pos.cpu().numpy()  # (B, T, N, 2)
    heading_np = heading.cpu().numpy()  # (B, T, N)
    vel_np = vel.cpu().numpy()  # (B, T, N, 2)
    mask_np = mask.cpu().numpy()  # (B, T, N)

    B, T, N, _ = pos.shape

    # The original interval might be something like every 5 frames
    # We know original_times: e.g. [0, 5, 10, ...]
    # Let's construct new times with finer resolution
    # last_time = T * fine_factor
    dt_coarse = fine_factor
    dt_fine = 1
    last_time = (T - 1) * fine_factor
    T_fine = (T - 1) * fine_factor + 1
    fine_times = np.linspace(0, last_time, T_fine)

    original_times = np.linspace(0, last_time, T)

    # Flatten B and N into one dimension: BN = B*N
    BN = B * N

    # Reorder axes so T is first for spline fitting:
    # pos: (T, B, N, 2) -> (T, BN, 2)
    # pos_reshaped = pos_np.transpose(1, 0, 2, 3).reshape(T, BN, 2)
    # vel_reshaped = vel_np.transpose(1, 0, 2, 3).reshape(T, BN, 2)

    # heading: (B, T, N) -> (T, B, N) -> (T, BN)
    # heading_reshaped = heading_np.transpose(1, 0, 2).reshape(T, BN)

    # Unwrap heading along time axis to avoid discontinuities
    # np.unwrap operates along axis=0 (time), this works vectorized for all BN trajectories
    # heading_unwrapped = np.unwrap(heading_reshaped, axis=0)

    reconstructed_valid_mask = interpolate_reconstructed_valid_mask(mask[:, :-1])
    reconstructed_valid_mask_np = reconstructed_valid_mask.cpu().numpy()

    pos_fine = np.zeros((B, T_fine, N, 2), dtype=pos_np.dtype)
    heading_fine = np.zeros((B, T_fine, N), dtype=heading_np.dtype)
    vel_fine = np.zeros((B, T_fine, N, 2), dtype=vel_np.dtype)

    # Loop over batch and agents
    for b in range(B):
        for n in range(N):
            # Extract the series for this agent
            pos_series = pos_np[b, :, n, :]  # Shape: (T, 2)
            heading_series = heading_np[b, :, n]  # Shape: (T,)
            vel_series = vel_np[b, :, n, :]  # Shape: (T, 2)
            m = reconstructed_valid_mask_np[b, :, n][::5]

            if not m.any():
                # If all positions are masked, skip this agent
                continue

            if m.sum() == 1:
                ind = m.nonzero()[0].item()
                pos_fine[b, ind * fine_factor, n] = pos_series[ind]
                heading_fine[b, ind * fine_factor, n] = heading_series[ind]
                vel_fine[b, ind * fine_factor, n] = vel_series[ind]
                continue

            # Unwrap heading to avoid discontinuities
            unwrapped_heading = np.unwrap(heading_series)

            # Interpolate each dimension of position and velocity with a spline
            for dim in range(2):
                # Position
                cs_pos = CubicSpline(original_times[m], pos_series[:, dim][m])
                pos_fine[b, :, n, dim] = cs_pos(fine_times)

                # Velocity
                cs_vel = CubicSpline(original_times[m], vel_series[:, dim][m])
                vel_fine[b, :, n, dim] = cs_vel(fine_times)

            # Interpolate heading (after unwrap)
            cs_heading = CubicSpline(original_times[m], unwrapped_heading[m], extrapolate=False)
            heading_fine_unwrapped = cs_heading(fine_times)
            # Wrap back to [-pi, pi]
            heading_fine[b, :, n] = utils.wrap_to_pi(heading_fine_unwrapped)

    pos_fine[~reconstructed_valid_mask_np] = 0
    vel_fine[~reconstructed_valid_mask_np] = 0
    heading_fine[~reconstructed_valid_mask_np] = 0

    # Convert back to PyTorch tensors
    pos_fine = torch.from_numpy(pos_fine).to(pos.device)
    heading_fine = torch.from_numpy(heading_fine).to(heading.device)
    vel_fine = torch.from_numpy(vel_fine).to(vel.device)
    # valid_mask = torch.from_numpy(reconstructed_valid_mask).to(mask.device)

    return {"position": pos_fine, "velocity": vel_fine, "heading": heading_fine, "valid_mask": reconstructed_valid_mask}


def interpolate(input_tensor, num_skipped_steps, remove_first_step=True):
    """
    TODO: This is linear interpolation on position, which might be incorrect as we need to consider heading.
    """
    is_4d = False
    if input_tensor.ndim == 4:
        is_4d = True
        _, _, N, D = input_tensor.shape
        tensor = input_tensor.flatten(2, 3)
    else:
        tensor = input_tensor
    B, T_before_plus_1, _ = tensor.shape
    T_before = T_before_plus_1 - 1
    tensor = tensor.permute(0, 2, 1)  # Reshape tensor to put the time dimension last
    T_after = num_skipped_steps * T_before
    interpolated = F.interpolate(tensor, size=T_after + 1, mode="linear", align_corners=True)
    if remove_first_step:
        interpolated = interpolated[:, :, 1:]
    else:
        T_after = T_after + 1
    interpolated = interpolated.permute(0, 2, 1)  # Reshape back
    if is_4d:
        interpolated = interpolated.reshape(B, T_after, N, D)
    assert interpolated.shape[:2] == (B, T_after)
    # assert interpolated[:, 4::5] == input_tensor[:, 1:]
    return interpolated


def interpolate_heading(input_tensor, num_skipped_steps, remove_first_step=True):
    is_4d = False
    if input_tensor.ndim == 4:
        is_4d = True
        _, _, N, D = input_tensor.shape
        tensor = input_tensor.flatten(2, 3)
    else:
        tensor = input_tensor
    B, T_before_plus_1, _ = tensor.shape
    T_before = T_before_plus_1 - 1
    tensor = tensor.permute(0, 2, 1)  # Reshape tensor to put the time dimension last
    T_after = num_skipped_steps * T_before

    # Circular interpolation for headings
    headings_cos = torch.cos(tensor)
    headings_sin = torch.sin(tensor)
    headings_cos_interp = F.interpolate(headings_cos, size=T_after + 1, mode="linear", align_corners=True)
    headings_sin_interp = F.interpolate(headings_sin, size=T_after + 1, mode="linear", align_corners=True)

    # Recompose interpolated headings
    interpolated = torch.atan2(headings_sin_interp, headings_cos_interp)

    #
    # interpolated = F.interpolate(tensor, size=T_after + 1, mode="linear", align_corners=True)
    if remove_first_step:
        interpolated = interpolated[:, :, 1:]
    else:
        T_after = T_after + 1
    interpolated = interpolated.permute(0, 2, 1)  # Reshape back
    if is_4d:
        interpolated = interpolated.reshape(B, T_after, N, D)
    assert interpolated.shape[:2] == (B, T_after)
    # assert interpolated[:, 4::5] == input_tensor[:, 1:]

    interpolated = utils.wrap_to_pi(interpolated)
    return interpolated


class BaseTokenizer:
    get_relative_velocity = get_relative_velocity

    def __init__(self, config):
        self.num_skipped_steps = config.TOKENIZATION.NUM_SKIPPED_STEPS
        self.predict_all_agents = config.TRAINING.PREDICT_ALL_AGENTS
        self.dt = (1 / STEPS_PER_SECOND) * self.num_skipped_steps

    def detokenize_numpy_array(
        self, data_dict, interpolation=True, detokenizing_gt=False, backward_prediction=False, **kwargs
    ):
        with torch.no_grad():
            new_data_dict = self.detokenize(
                self._numpy_to_tensor(data_dict),
                interpolation=interpolation,
                detokenizing_gt=detokenizing_gt,
                backward_prediction=backward_prediction,
                **kwargs
            )
        data_dict = self._tensor_to_numpy(new_data_dict)
        return data_dict

    def _numpy_to_tensor(self, data_dict):
        # Translate to tensors
        new_data_dict = {"in_evaluation": torch.from_numpy(np.array([data_dict["in_evaluation"]]))}

        for k, v in data_dict.items():
            if k.startswith("decoder/") and isinstance(v, np.ndarray):
                if np.issubdtype(v.dtype, np.number) or v.dtype == bool:
                    new_data_dict[k] = torch.from_numpy(v).unsqueeze(dim=0)
                else:
                    pass

        # TODO: The device is default to CPU for now. Might be set from config.
        return new_data_dict

    def _tensor_to_numpy(self, data_dict):
        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                d = data_dict[k].cpu().numpy()
            elif isinstance(data_dict[k], bool):
                d = [data_dict[k]]
            else:
                raise ValueError("Unknown type: {}".format(type(data_dict[k])))
            assert len(d) == 1
            data_dict[k] = d[0]
        data_dict["in_evaluation"] = data_dict["in_evaluation"].all().item()
        return data_dict

    def tokenize_numpy_array(self, data_dict, **kwargs):
        with torch.no_grad():
            new_data_dict, stat = self.tokenize(self._numpy_to_tensor(data_dict), **kwargs)
        data_dict = self._tensor_to_numpy(new_data_dict)
        return data_dict, stat

    def tokenize(self, data_dict, **kwargs):
        raise NotImplementedError

    def detokenize(self, data_dict, interpolation=True, detokenizing_gt=False, backward_prediction=False, **kwargs):
        raise NotImplementedError

    def detokenize_step(self, *args, **kwargs):
        ret = self._detokenize_a_step(*args, **kwargs)
        # assert "delta_pos" in ret
        return ret

    def get_motion_feature(self):
        m = torch.from_numpy(self.bin_centers_flat)
        dist = m.norm(p=2, dim=-1).unsqueeze(-1)
        heading = torch.atan2(m[..., 1], m[..., 0]).unsqueeze(-1)
        return torch.cat([m, dist, heading], dim=-1)

    def get_bin_centers(self, agent_type):
        B, N = agent_type.shape
        if self.bin_centers is not None:
            if self.use_type_specific_bins:
                bin_centers = self.bin_centers.to(agent_type.device).expand(B, self.num_actions, N, 3, 2)
                agent_type = agent_type - 1  # Veh: 0, Ped: 1, Cyc: 2
                agent_type[agent_type < 0] = 0
                agent_type = agent_type.reshape(B, 1, N, 1, 1).expand(B, bin_centers.shape[1], N, 1, 2)
                bin_centers = torch.gather(bin_centers, dim=-2, index=agent_type).squeeze(-2)
            else:
                bin_centers = self.bin_centers.to(agent_type.device).expand(B, self.num_actions, N, 2)
        else:
            bin_centers = None
        return bin_centers

    def hole_filling(self, data_dict):
        # ===== Get initial data =====
        # If we don't clone here, the following hole-filling code will overwrite raw data.
        agent_pos = data_dict["decoder/agent_position"]
        agent_heading = data_dict["decoder/agent_heading"]
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]
        agent_velocity = data_dict["decoder/agent_velocity"]
        B, T_full, N, _ = agent_pos.shape
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::self.num_skipped_steps]
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        T_chunks = agent_pos.shape[1]

        # ===== Hole filling =====
        for i in range(T_chunks):
            current_pos = agent_pos[:, i:i + 1]
            current_heading = agent_heading[:, i:i + 1]
            current_vel = agent_velocity[:, i:i + 1]
            current_valid_mask = agent_valid_mask[:, i:i + 1]
            # === There exists a very rare case, that the agent validity is True, False, True ===
            # When it happens in the beginning first 3 steps, the case become very complex.
            # A solution is to assume a default action for the newly added agents.
            if 0 < i < T_chunks - 1:
                step0_valid_mask = agent_valid_mask[:, i - 1:i]
                step1_valid_mask = agent_valid_mask[:, i:i + 1]
                step2_valid_mask = agent_valid_mask[:, i + 1:i + 2]
                is_rare_case = step2_valid_mask & step0_valid_mask & ~step1_valid_mask
                if is_rare_case.any():
                    # Interpolate position, heading and velocity
                    int_pos = (agent_pos[:, i - 1:i] + agent_pos[:, i + 1:i + 2]) / 2
                    int_vel = (agent_velocity[:, i - 1:i] + agent_velocity[:, i + 1:i + 2]) / 2

                    # Circular interpolation for headings
                    head_s = agent_heading[:, i - 1:i]
                    head_e = agent_heading[:, i + 1:i + 2]
                    tensor = torch.atan2(torch.sin(head_s) + torch.sin(head_e), torch.cos(head_s) + torch.cos(head_e))
                    int_heading = tensor

                    agent_pos[:, i:i +
                              1] = torch.where(is_rare_case[..., None].expand(-1, -1, -1, 3), int_pos, current_pos)
                    agent_heading[:, i:i + 1] = torch.where(is_rare_case, int_heading, current_heading)
                    agent_velocity[:, i:i + 1] = torch.where(
                        is_rare_case[..., None].expand(-1, -1, -1, 2), int_vel, current_vel
                    )
                    agent_valid_mask[:, i:i + 1] = torch.logical_or(current_valid_mask, is_rare_case)
        # Write back:
        data_dict["decoder/agent_position"][:, ::self.num_skipped_steps] = agent_pos
        data_dict["decoder/agent_heading"][:, ::self.num_skipped_steps] = agent_heading
        data_dict["decoder/agent_velocity"][:, ::self.num_skipped_steps] = agent_velocity
        data_dict["decoder/agent_valid_mask"][:, ::self.num_skipped_steps] = agent_valid_mask
        return data_dict


class DeltaDeltaTokenizer(BaseTokenizer):
    def __init__(self, config):
        super().__init__(config)

        # We reuse x_max and y_max to refer to the maximal acceleration in 1s in the x and y dimensions respectively.
        # Note that this isn't the maximal change in velocity between two consecutive timesteps.
        assert "X_LIMIT" not in config.TOKENIZATION, "Please use X_MAX/MIN, Y_MAX/MIN instead!"
        assert "Y_LIMIT" not in config.TOKENIZATION, "Please use X_MAX/MIN, Y_MAX/MIN instead!"
        # x_max = config.TOKENIZATION.X_MAX / STEPS_PER_SECOND * self.num_skipped_steps
        # x_min = config.TOKENIZATION.X_MIN / STEPS_PER_SECOND * self.num_skipped_steps
        # y_max = config.TOKENIZATION.Y_MAX / STEPS_PER_SECOND * self.num_skipped_steps
        # y_min = config.TOKENIZATION.Y_MIN / STEPS_PER_SECOND * self.num_skipped_steps
        assert config.TOKENIZATION.X_MAX == 3.5, "X_MAX is deprecated!"

        x_num_bins = y_num_bins = config.TOKENIZATION.NUM_BINS
        self.num_bins = config.TOKENIZATION.NUM_BINS
        self.num_actions = x_num_bins * y_num_bins
        self.config = config

        x_limit_veh = config.TOKENIZATION.VEH_LIMIT / STEPS_PER_SECOND * self.num_skipped_steps
        x_limit_ped = config.TOKENIZATION.PED_LIMIT / STEPS_PER_SECOND * self.num_skipped_steps
        x_limit_cyc = config.TOKENIZATION.CYC_LIMIT / STEPS_PER_SECOND * self.num_skipped_steps
        # assert x_num_bins == y_num_bins == 33

        # Precompute the bin positions. In the future, we can load them from dataset.
        bin_veh = get_bin_centers(
            x_min=-x_limit_veh,
            x_max=x_limit_veh,
            y_min=-x_limit_veh,
            y_max=x_limit_veh,
            x_num_bins=x_num_bins,
            y_num_bins=y_num_bins
        )
        if x_limit_veh == x_limit_ped and x_limit_veh == x_limit_cyc:
            self.bin_centers_flat = bin_veh
            # Assert if (dx=0, dy=0) are in the bin centers.
            assert self.bin_centers_flat.shape == (self.num_actions, 2)
            self.default_action = int(np.argmin(np.linalg.norm(self.bin_centers_flat, axis=-1)))
            self.bin_centers = torch.from_numpy(self.bin_centers_flat.reshape(1, self.num_actions, 1, 2))
            self.use_type_specific_bins = False
        else:
            bin_ped = get_bin_centers(
                x_min=-x_limit_ped,
                x_max=x_limit_ped,
                y_min=-x_limit_ped,
                y_max=x_limit_ped,
                x_num_bins=x_num_bins,
                y_num_bins=y_num_bins
            )
            bin_cyc = get_bin_centers(
                x_min=-x_limit_cyc,
                x_max=x_limit_cyc,
                y_min=-x_limit_cyc,
                y_max=x_limit_cyc,
                x_num_bins=x_num_bins,
                y_num_bins=y_num_bins
            )
            self.bin_centers_flat = np.stack([bin_veh, bin_ped, bin_cyc], axis=1)
            # Assert if (dx=0, dy=0) are in the bin centers.
            assert self.bin_centers_flat.shape == (self.num_actions, 3, 2)
            self.default_action = int(np.argmin(np.linalg.norm(self.bin_centers_flat, axis=-1).mean(1)))
            self.bin_centers = torch.from_numpy(self.bin_centers_flat.reshape(1, self.num_actions, 1, 3, 2))
            self.use_type_specific_bins = True

        self.add_noise = config.TOKENIZATION.ADD_NOISE
        # assert self.add_noise is False

        num_bins = self.num_bins
        # Create coordinate grid centered at (0,0)
        y, x = np.ogrid[-(num_bins // 2):(num_bins + 1) // 2, -(num_bins // 2):(num_bins + 1) // 2]
        # Calculate the distance from the center
        dist_from_center = np.sqrt(x**2 + y**2)
        # Normalize distances so that the center is -1 and edges are 0
        max_distance = dist_from_center.max()
        min_val = 1e-5
        normalized_dist = ((dist_from_center / max_distance) - 1) * min_val

        # Flatten to get a (num_bins^2,) vector
        self.noise = torch.from_numpy(normalized_dist.ravel()).reshape(1, num_bins * num_bins, 1)

    def get_bin_centers(self, agent_type):
        agent_type = agent_type.clone()
        B, N = agent_type.shape
        if self.bin_centers is not None:
            if self.use_type_specific_bins:
                bin_centers = self.bin_centers.to(agent_type.device).expand(B, self.num_actions, N, 3, 2)
                agent_type = agent_type - 1  # Veh: 0, Ped: 1, Cyc: 2
                agent_type[agent_type < 0] = 0
                agent_type = agent_type.reshape(B, 1, N, 1, 1).expand(B, bin_centers.shape[1], N, 1, 2)
                bin_centers = torch.gather(bin_centers, dim=-2, index=agent_type).squeeze(-2)
            else:
                bin_centers = self.bin_centers.to(agent_type.device).expand(B, self.num_actions, N, 2)
        else:
            bin_centers = None
        return bin_centers


    def tokenize(self, data_dict, backward_prediction=False, **kwargs):
        """

        Args:
            data_dict: Input data

        Returns:
            Discretized action in an int array with shape (num time steps for actions, num agents).
        """

        if backward_prediction:
            return self._tokenize_backward_prediction(data_dict, **kwargs)

        # TODO: Hardcoded here...
        if self.config.GPT_STYLE:
            start_step = 0
        else:
            start_step = 2

        # ===== Hole Filling =====
        data_dict = self.hole_filling(data_dict)

        # ===== Get initial data =====
        # If we don't clone here, the following hole-filling code will overwrite raw data.
        agent_pos = data_dict["decoder/agent_position"]  # .clone()
        agent_heading = data_dict["decoder/agent_heading"]  # .clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  # .clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  # .clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  # .clone()
        agent_type = data_dict["decoder/agent_type"]  # .clone()
        B, T_full, N, _ = agent_pos.shape
        # assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos_full = agent_pos.clone()
        agent_heading_full = agent_heading.clone()
        agent_velocity_full = agent_velocity.clone()
        agent_valid_mask_full = agent_valid_mask.clone()
        agent_pos = agent_pos[:, ::self.num_skipped_steps]
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        T_chunks = agent_pos.shape[1]
        # assert T_chunks == 19

        # ===== Build up some variables =====
        current_pos = agent_pos[:, start_step:start_step + 1, ..., :2]
        current_heading = agent_heading[:, start_step:start_step + 1]
        current_vel = agent_velocity[:, start_step:start_step + 1, ..., :2]
        current_valid_mask = agent_valid_mask[:, start_step:start_step + 1]

        init_pos = current_pos.clone()
        init_heading = current_heading.clone()
        init_vel = current_vel.clone()
        init_valid_mask = current_valid_mask.clone()


        init_delta = _reconstruct_delta_pos_from_abs_vel(current_vel, current_heading, dt=self.dt)

        # if self.config.DELTA_POS_IS_VELOCITY:
        #     init_delta = get_relative_velocity(current_vel, current_heading)
        # else:
        #     init_delta = _reconstruct_delta_pos_from_abs_vel(current_vel, current_heading, dt=self.dt)


        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        target_action = []
        target_action_valid_mask = []
        reconstruction_list = []
        relative_delta_pos_list = []
        pos = []
        heading = []
        vel = []

        # ===== Loop to reconstruct the scenario =====
        tokenization_state = None
        for next_step in range(start_step + 1, T_chunks):
            res = self._tokenize_a_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_vel=current_vel,
                current_valid_mask=current_valid_mask,
                next_pos=agent_pos[:, next_step:next_step + 1, ..., :2],  # (B, 1, N, 2)
                next_heading=agent_heading[:, next_step:next_step + 1],  # (B, 1, N)
                next_valid_mask=agent_valid_mask[:, next_step:next_step + 1],  # (B, 1, N)
                next_velocity=agent_velocity[:, next_step:next_step + 1, ..., :2],  # (B, 1, N, 2)
                bin_centers=bin_centers,
                add_noise=self.add_noise,
                topk=self.config.TOKENIZATION.NOISE_TOPK,
                agent_shape=agent_shape,
                agent_type=agent_type,
                dt=self.dt,
                tokenization_state=tokenization_state,
                agent_pos_full=agent_pos_full[:, (next_step - 1) *
                                              self.num_skipped_steps:next_step * self.num_skipped_steps + 1],
                agent_heading_full=agent_heading_full[:, (next_step - 1) *
                                                      self.num_skipped_steps:next_step * self.num_skipped_steps + 1],
                agent_velocity_full=agent_velocity_full[:, (next_step - 1) *
                                                        self.num_skipped_steps:next_step * self.num_skipped_steps + 1],
                agent_valid_mask_full=agent_valid_mask_full[:, (next_step - 1) *
                                                            self.num_skipped_steps:next_step * self.num_skipped_steps +
                                                            1],
            )
            tokenization_state = res

            best_action = res["action"]
            recon_next_pos = res["pos"]
            recon_next_heading = res["heading"]
            recon_next_vel = res["vel"]
            recon_next_valid_mask = res["mask"]
            recon_next_delta_pos = res["delta_pos"]  # The input delta for next step.

            best_action = best_action.reshape(B, 1, N)

            # ===== Process the target action/valid mask =====
            target_action_valid_mask.append(recon_next_valid_mask.clone())
            target_action.append(best_action)

            # Some debug asserts
            assert (best_action[recon_next_valid_mask] >= 0).all()
            assert (best_action[~recon_next_valid_mask] == -1).all()

            # ===== Process the "current_xxx" for next step =====
            if self.config.GPT_STYLE:
                assert self.config.TOKENIZATION.ALLOW_SKIP_STEP
            if self.config.TOKENIZATION.ALLOW_SKIP_STEP:
                # Use the next valid mask as the valid mask for next step.
                # In contrast, if this flag is False, then we will use "next valid mask & if it's not removed" for next
                # step.
                next_valid_mask = agent_valid_mask[:, next_step:next_step + 1]
                newly_added = torch.logical_and(~recon_next_valid_mask, next_valid_mask)
                if newly_added.any():
                    recon_next_pos[newly_added] = agent_pos[:, next_step:next_step + 1, ..., :2][newly_added]
                    recon_next_heading[newly_added] = agent_heading[:, next_step:next_step + 1][newly_added]
                    recon_next_vel[newly_added] = agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added]

                    recon_next_delta_pos[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                        vel=agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added],
                        heading=agent_heading[:, next_step:next_step + 1][newly_added],
                        dt=self.dt
                    )

                    # if self.config.DELTA_POS_IS_VELOCITY:
                    #     recon_next_delta_pos[newly_added] = get_relative_velocity(
                    #         vel=agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added],
                    #         heading=agent_heading[:, next_step:next_step + 1][newly_added],
                    #     )
                    # else:
                    #     recon_next_delta_pos[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                    #         vel=agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added],
                    #         heading=agent_heading[:, next_step:next_step + 1][newly_added],
                    #         dt=self.dt
                    #     )


                    recon_next_valid_mask[newly_added] = next_valid_mask[newly_added]

            relative_delta_pos_list.append(recon_next_delta_pos)
            current_vel = recon_next_vel
            current_heading = recon_next_heading
            current_pos = recon_next_pos
            current_valid_mask = recon_next_valid_mask
            pos.append(current_pos.clone())
            heading.append(current_heading.clone())
            vel.append(current_vel.clone())

        # ===== Postprocess and prepare the "start action" =====
        # In GPT style, some agents will be added in the middle of the scene.
        # So we need to find out when they are in and add a start action before that step.
        # In non-GPT style, we only need to prepare the start action for the first step.
        target_actions = torch.cat(target_action, dim=1)  # (B, T_skipped, N)
        target_action_valid_mask = torch.cat(target_action_valid_mask, dim=1)  # (B, T_skipped, N)
        relative_delta_pos_list = torch.cat(relative_delta_pos_list, dim=1)  # (B, T_skipped, N)
        pos = torch.cat(pos, dim=1)
        heading = torch.cat(heading, dim=1)
        vel = torch.cat(vel, dim=1)

        pos = torch.cat([init_pos, pos], dim=1)
        heading = torch.cat([init_heading, heading], dim=1)
        vel = torch.cat([init_vel, vel], dim=1)
        relative_delta_pos_list = torch.cat([init_delta, relative_delta_pos_list], dim=1)

        # If not in back prediction, what will be:
        # 1. The first tokens in input_actions? START_ACTION
        # 2. The last tokens in input_actions? Just the tokens at t=18 (t=85~90)
        # 3. The first tokens in target_actions? The tokens at t=0 (t=0~5) for GPT and t=2 otherwise.
        # 4. The last tokens in target_actions? All -1 because there is no GT for t=19 (t=90~95)
        if self.config.GPT_STYLE:
            # Search for the first step that has newly added agents
            assert start_step == 0
            already_tokenized = init_valid_mask.clone()
            start_action = torch.full_like(target_actions[:, :1], -1)
            start_action[init_valid_mask] = START_ACTION
            assert target_actions.shape[1] == T_chunks - 1
            input_action = torch.cat([start_action, target_actions], dim=1)
            input_action_valid_mask = torch.cat([init_valid_mask, target_action_valid_mask], dim=1)
            for next_step in range(start_step + 1, T_chunks):
                next_valid_mask = agent_valid_mask[:, next_step:next_step + 1]
                is_newly_added = torch.logical_and(~already_tokenized, next_valid_mask)
                if is_newly_added.any():
                    input_action[:, next_step:next_step + 1][is_newly_added] = START_ACTION
                    input_action_valid_mask[:, next_step:next_step + 1][is_newly_added] = \
                        next_valid_mask[is_newly_added]
                already_tokenized = torch.logical_or(already_tokenized, is_newly_added)

        else:
            start_action = torch.full_like(target_actions[:, :1], -1)
            start_action[init_valid_mask] = START_ACTION
            input_action = torch.cat([start_action, target_actions], dim=1)
            input_action_valid_mask = torch.cat([init_valid_mask.reshape(B, 1, N), target_action_valid_mask], dim=1)

        target_actions = torch.cat([target_actions, target_actions.new_full((B, 1, N), -1)], dim=1)
        target_action_valid_mask = torch.cat(
            [target_action_valid_mask, target_action_valid_mask.new_zeros((B, 1, N))], dim=1
        )
        data_dict["in_backward_prediction"] = False
        assert (agent_valid_mask[:, start_step:] >= target_action_valid_mask).all()
        assert (agent_valid_mask[:, start_step + 1:] >= target_action_valid_mask[:, :-1]).all()
        assert (agent_valid_mask[:, start_step:] >= input_action_valid_mask).all()

        # # Some debug asserts for backward prediction:
        # assert (target_actions[:, :-1] == flipped_target_actions[:, :-1].flip(dims=[1])).all()
        # minp = (input_action * (input_action != START_ACTION))
        # minp = minp * (input_action != -1)
        # mfinp = (flipped_input_action * (flipped_input_action != END_ACTION))
        # mfinp = mfinp * (flipped_input_action != -1)
        # assert (minp[:, 1:] == mfinp[:, 1:].flip(dims=[1])).all()
        # assert (pos == flipped_pos.flip(dims=[1])).all()
        # assert (heading == flipped_heading.flip(dims=[1])).all()
        # assert (vel == flipped_vel.flip(dims=[1])).all()

        data_dict["decoder/target_action"] = target_actions
        data_dict["decoder/target_action_valid_mask"] = target_action_valid_mask
        data_dict["decoder/input_action"] = input_action
        data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
        data_dict["decoder/modeled_agent_delta"] = relative_delta_pos_list
        data_dict["decoder/modeled_agent_position"] = pos
        data_dict["decoder/modeled_agent_heading"] = heading
        data_dict["decoder/modeled_agent_velocity"] = vel

        # Debug:
        # pos_diff = (pos - agent_pos[..., :2]).norm(dim=-1).numpy()
        # heading_diff = utils.wrap_to_pi(heading - agent_heading).abs().numpy()
        # vel_diff = (vel - agent_velocity[..., :2]).norm(dim=-1).numpy()

        # All input actions should be >0
        assert (input_action[input_action_valid_mask] >= 0).all()
        assert (target_actions[target_action_valid_mask] >= 0).all()
        assert (input_action[~input_action_valid_mask] == -1).all()
        assert (target_actions[~target_action_valid_mask] == -1).all()

        return data_dict, {"reconstruction_list": reconstruction_list}

    def _tokenize_backward_prediction(self, data_dict, **kwargs):
        # TODO: Hardcoded here...
        if self.config.GPT_STYLE:
            start_step = 0
        else:
            raise ValueError()
            start_step = 2

        # ===== Hole Filling =====
        data_dict = self.hole_filling(data_dict)

        # ===== Get initial data =====
        # If we don't clone here, the following hole-filling code will overwrite raw data.
        agent_pos = data_dict["decoder/agent_position"]  # .clone()
        agent_heading = data_dict["decoder/agent_heading"]  # .clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"]  # .clone()
        agent_velocity = data_dict["decoder/agent_velocity"]  # .clone()
        agent_shape = data_dict["decoder/current_agent_shape"]  # .clone()
        agent_type = data_dict["decoder/agent_type"]  # .clone()
        B, T_full, N, _ = agent_pos.shape
        # assert T_full == 91
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::self.num_skipped_steps]
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        T_chunks = agent_pos.shape[1]
        # assert T_chunks == 19

        # ===== Build up some variables =====
        current_pos = agent_pos[:, -1:, ..., :2]
        current_heading = agent_heading[:, -1:]
        current_vel = agent_velocity[:, -1:, ..., :2]
        current_valid_mask = agent_valid_mask[:, -1:]

        init_pos = current_pos.clone()
        init_heading = current_heading.clone()
        init_vel = current_vel.clone()
        init_valid_mask = current_valid_mask.clone()

        init_delta = _reconstruct_delta_pos_from_abs_vel(current_vel, current_heading + np.pi, dt=self.dt)

        # # NOTE: +180deg here.
        # if self.config.DELTA_POS_IS_VELOCITY:
        #     # TODO: Is it correct here even for bicycle model????
        #     init_delta = get_relative_velocity(current_vel, current_heading)
        # else:
        #     init_delta = _reconstruct_delta_pos_from_abs_vel(current_vel, current_heading, dt=self.dt)

        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        target_action = []
        target_action_valid_mask = []
        reconstruction_list = []
        relative_delta_pos_list = []
        pos = []
        heading = []
        vel = []
        previously_added = torch.zeros_like(current_valid_mask, dtype=torch.bool)

        # ===== Loop to reconstruct the scenario =====
        for backward_next_step in range(1, T_chunks):
            # backward_next_step = 1, ..., 18

            forward_next_step = T_chunks - backward_next_step - 1
            # forward_next_step = 17, ..., 0

            res = self._tokenize_a_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_vel=current_vel,
                current_valid_mask=current_valid_mask,
                next_pos=agent_pos[:, forward_next_step:forward_next_step + 1, ..., :2],  # (B, 1, N, 2)
                next_heading=agent_heading[:, forward_next_step:forward_next_step + 1],  # (B, 1, N)
                next_valid_mask=agent_valid_mask[:, forward_next_step:forward_next_step + 1],  # (B, 1, N)
                next_velocity=agent_velocity[:, forward_next_step:forward_next_step + 1, ..., :2],  # (B, 1, N, 2)
                bin_centers=bin_centers,
                add_noise=self.add_noise,
                topk=self.config.TOKENIZATION.NOISE_TOPK,
                agent_shape=agent_shape,
                agent_type=agent_type,
                dt=-self.dt,
                **kwargs
            )

            best_action = res["action"]
            recon_next_pos = res["pos"]
            recon_next_heading = res["heading"]
            recon_next_vel = res["vel"]
            recon_next_valid_mask = res["mask"]
            recon_next_delta_pos = res["delta_pos"]  # The input delta for next step.

            best_action = best_action.reshape(B, 1, N)

            # ===== Process the target action/valid mask =====
            target_action_valid_mask.append(recon_next_valid_mask.clone())
            target_action.append(best_action)

            # Some debug asserts
            assert (best_action[recon_next_valid_mask] >= 0).all()
            assert (best_action[~recon_next_valid_mask] == -1).all()

            # ===== Process the "current_xxx" for next step =====
            # Use the next valid mask as the valid mask for next step.
            # In contrast, if this flag is False, then we will use "next valid mask & if it's not removed" for next
            # step.
            next_valid_mask = agent_valid_mask[:, forward_next_step:forward_next_step + 1]
            newly_added = torch.logical_and(~recon_next_valid_mask, next_valid_mask)
            # newly_added = newly_added & (~previously_added)

            if newly_added.any():
                # previously_added[newly_added] = True
                recon_next_pos[newly_added] = agent_pos[:, forward_next_step:forward_next_step + 1,
                                                        ..., :2][newly_added]
                recon_next_heading[newly_added] = agent_heading[:, forward_next_step:forward_next_step + 1][newly_added]
                recon_next_vel[newly_added] = agent_velocity[:, forward_next_step:forward_next_step + 1,
                                                             ..., :2][newly_added]
                

                recon_next_delta_pos[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                    vel=agent_velocity[:, forward_next_step:forward_next_step + 1, ..., :2][newly_added],
                    # heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                    heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added] + np.pi,
                    dt=self.dt
                )
                
                # if self.config.DELTA_POS_IS_VELOCITY:
                #     recon_next_delta_pos[newly_added] = get_relative_velocity(
                #         vel=agent_velocity[:, forward_next_step:forward_next_step + 1, ..., :2][newly_added],
                #         # heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                #         heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                #     )
                # else:
                #     recon_next_delta_pos[newly_added] = _reconstruct_delta_pos_from_abs_vel(
                #         vel=agent_velocity[:, forward_next_step:forward_next_step + 1, ..., :2][newly_added],
                #         # heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                #         heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                #         dt=self.dt
                #     )


                recon_next_valid_mask[newly_added] = next_valid_mask[newly_added]

            relative_delta_pos_list.append(recon_next_delta_pos)
            current_vel = recon_next_vel
            current_heading = recon_next_heading
            current_pos = recon_next_pos
            current_valid_mask = recon_next_valid_mask
            pos.append(current_pos.clone())
            heading.append(current_heading.clone())
            vel.append(current_vel.clone())

        # ===== Postprocess and prepare the "start action" =====
        # In GPT style, some agents will be added in the middle of the scene.
        # So we need to find out when they are in and add a start action before that step.
        # In non-GPT style, we only need to prepare the start action for the first step.
        target_actions = torch.cat(target_action, dim=1)  # (B, T_skipped, N)
        target_action_valid_mask = torch.cat(target_action_valid_mask, dim=1)  # (B, T_skipped, N)
        relative_delta_pos_list = torch.cat(relative_delta_pos_list, dim=1)  # (B, T_skipped, N)
        pos = torch.cat(pos, dim=1)
        heading = torch.cat(heading, dim=1)
        vel = torch.cat(vel, dim=1)

        pos = torch.cat([init_pos, pos], dim=1)
        heading = torch.cat([init_heading, heading], dim=1)
        vel = torch.cat([init_vel, vel], dim=1)
        relative_delta_pos_list = torch.cat([init_delta, relative_delta_pos_list], dim=1)

        # Search for the first step that has newly added agents
        assert start_step == 0
        already_tokenized = init_valid_mask.clone()
        start_action = torch.full_like(target_actions[:, :1], -1)
        start_action[init_valid_mask] = END_ACTION
        assert target_actions.shape[1] == T_chunks - 1
        input_action = torch.cat([start_action, target_actions], dim=1)
        input_action_valid_mask = torch.cat([init_valid_mask, target_action_valid_mask], dim=1)

        previously_added = torch.zeros_like(current_valid_mask, dtype=torch.bool)
        for backward_next_step in range(1, T_chunks):
            forward_next_step = T_chunks - backward_next_step - 1
            next_valid_mask = agent_valid_mask[:, forward_next_step:forward_next_step + 1]
            is_newly_added = torch.logical_and(~already_tokenized, next_valid_mask)
            # newly_added = newly_added & (~previously_added)

            if is_newly_added.any():
                # previously_added[newly_added] = True
                input_action[:, backward_next_step:backward_next_step + 1][is_newly_added] = END_ACTION
                input_action_valid_mask[:, backward_next_step:backward_next_step + 1][is_newly_added] = \
                    next_valid_mask[is_newly_added]
            already_tokenized = torch.logical_or(already_tokenized, is_newly_added)

        target_actions = torch.cat([target_actions, target_actions.new_full((B, 1, N), -1)], dim=1)
        target_action_valid_mask = torch.cat(
            [target_action_valid_mask, target_action_valid_mask.new_zeros((B, 1, N))], dim=1
        )
        data_dict["in_backward_prediction"] = True

        flipped_agent_valid_mask = agent_valid_mask.flip(dims=[1])
        assert (flipped_agent_valid_mask[:, start_step:] >= target_action_valid_mask).all()
        assert (flipped_agent_valid_mask[:, start_step + 1:] >= target_action_valid_mask[:, :-1]).all()
        assert (flipped_agent_valid_mask[:, start_step:] >= input_action_valid_mask).all()

        data_dict["decoder/target_action"] = target_actions
        data_dict["decoder/target_action_valid_mask"] = target_action_valid_mask
        data_dict["decoder/input_action"] = input_action
        data_dict["decoder/input_action_valid_mask"] = input_action_valid_mask
        data_dict["decoder/modeled_agent_delta"] = relative_delta_pos_list
        data_dict["decoder/modeled_agent_position"] = pos
        data_dict["decoder/modeled_agent_heading"] = heading
        data_dict["decoder/modeled_agent_velocity"] = vel

        # All input actions should be >0
        assert (input_action[input_action_valid_mask] >= 0).all()
        assert (target_actions[target_action_valid_mask] >= 0).all()
        assert (input_action[~input_action_valid_mask] == -1).all()
        assert (target_actions[~target_action_valid_mask] == -1).all()

        return data_dict, {"reconstruction_list": reconstruction_list}

    def detokenize(
        self,
        data_dict,
        interpolation=True,
        detokenizing_gt=False,
        backward_prediction=False,
        flip_wrong_heading=False,
        teacher_forcing=True,
        autoregressive_start_step=2,
        **kwargs,
    ):  # actions, current_pos, current_vel, current_heading):
        """
        Compared to the non-gpt style, this function dynamically adds new agents into the scene.
        A very interesting point here is we can't start with 'current position' in the data.
        Because the model is predicting according to the first few tokens, which already have some errors.
        """

        if backward_prediction:
            return self._detokenize_backward_prediction(
                data_dict, interpolation=interpolation, detokenizing_gt=detokenizing_gt, teacher_forcing=teacher_forcing, **kwargs
            )

        # TODO: Hardcoded here...
        if self.config.GPT_STYLE:
            start_step = 0
        else:
            start_step = 2

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"].clone()
        agent_heading = data_dict["decoder/agent_heading"].clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"].clone()
        agent_velocity = data_dict["decoder/agent_velocity"].clone()
        agent_shape = data_dict["decoder/current_agent_shape"].clone()
        agent_type = data_dict["decoder/agent_type"].clone()
        if detokenizing_gt:
            target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
        input_mask = data_dict["decoder/input_action_valid_mask"]
        B, T_full, N, _ = agent_pos.shape
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::self.num_skipped_steps].clone()
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        # T_chunks = agent_pos.shape[1]

        # ===== Prepare some variables =====
        action = data_dict["decoder/output_action"]
        T_actions = action.shape[1]
        # if T_actions + start_step != T_chunks:
        #     print(
        #         "WARNING: The number of actions is not consistent with the number of raw data chunks! You have {} actions, start step is {} and the number of chunks is {}."
        #         .format(T_actions, start_step, T_chunks)
        #     )
        T_generated_chunks = T_actions + start_step

        current_pos = agent_pos[:, start_step:start_step + 1, ..., :2].clone()
        current_heading = agent_heading[:, start_step:start_step + 1].clone()
        current_vel = agent_velocity[:, start_step:start_step + 1, ..., :2].clone()
        current_valid_mask = agent_valid_mask[:, start_step:start_step + 1].clone()

        if detokenizing_gt:
            # Merge input mask with target mask
            input_mask = input_mask & target_action_valid_mask

        reconstructed_pos_list = [current_pos.clone()]
        reconstructed_heading_list = [current_heading.clone()]
        reconstructed_vel_list = [current_vel.clone()]

        already_interpolated = False
        reconstructed_pos_full_list = [current_pos.clone()]
        reconstructed_heading_full_list = [current_heading.clone()]
        reconstructed_vel_full_list = [current_vel.clone()]

        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        kwargs["detokenization_state"] = None

        for curr_step in range(T_generated_chunks):
            if curr_step < start_step:
                next_pos = agent_pos[:, curr_step + 1:curr_step + 2, ..., :2]
                next_heading = agent_heading[:, curr_step + 1:curr_step + 2]
                next_vel = agent_velocity[:, curr_step + 1:curr_step + 2, ..., :2]
                next_valid_mask = agent_valid_mask[:, curr_step + 1:curr_step + 2]

            else:
                # We assume that starting from start_step, the agent valid mask will not change.
                action_step = curr_step - start_step
                action_valid_mask_step = input_mask[:, action_step:action_step + 1]

                act = action[:, action_step:action_step + 1]
                assert (act[action_valid_mask_step] != -1).all()
                res = self._detokenize_a_step(
                    current_pos=current_pos,
                    current_heading=current_heading,
                    current_valid_mask=action_valid_mask_step,
                    current_vel=current_vel,
                    action=act,
                    agent_shape=agent_shape,
                    agent_type=agent_type,
                    bin_centers=bin_centers,
                    dt=self.dt,
                    flip_wrong_heading=flip_wrong_heading,
                    **kwargs
                )
                kwargs["detokenization_state"] = res

                next_pos, next_heading, next_vel = res["pos"], res["heading"], res["vel"]
                assert "delta_pos" in res
                next_pos = next_pos.reshape(B, 1, N, 2)
                next_heading = next_heading.reshape(B, 1, N)
                next_vel = next_vel.reshape(B, 1, N, 2)
                next_valid_mask = current_valid_mask

                # ===== A special case: fill in the info for the agents added in next step =====
                # ===== Another special case: if you are detokenizing the raw tokenized data, you need to fill in
                # the info for the agents added in the next step. =====
                if (curr_step < autoregressive_start_step) or (detokenizing_gt and curr_step < T_generated_chunks - 1):
                    # Fill in the initial states of newly added agents
                    action_valid_mask_next_step = input_mask[:, action_step + 1:action_step + 2]
                    newly_added = torch.logical_and(~action_valid_mask_step, action_valid_mask_next_step)
                    next_pos[newly_added] = agent_pos[:, curr_step + 1:curr_step + 2, ..., :2][newly_added]
                    next_heading[newly_added] = agent_heading[:, curr_step + 1:curr_step + 2][newly_added]
                    next_vel[newly_added] = agent_velocity[:, curr_step + 1:curr_step + 2, ..., :2][newly_added]
                    next_valid_mask[newly_added] = action_valid_mask_next_step[newly_added]
                    if "reconstructed_position" in res:
                        # If some agents are added in the next step, the "last step" in reconstructed chunk
                        # aka the 5-th step in the chunk should be replaced by the GT states.
                        res["reconstructed_position"][-1][newly_added] = agent_pos[:, curr_step + 1:curr_step + 2,
                                                                                   ..., :2][newly_added]
                        res["reconstructed_heading"][-1][newly_added] = agent_heading[:, curr_step + 1:curr_step +
                                                                                      2][newly_added]
                        res["reconstructed_velocity"][-1][newly_added] = agent_velocity[:, curr_step + 1:curr_step + 2,
                                                                                        ..., :2][newly_added]

                if "reconstructed_position" in res:
                    already_interpolated = True
                    reconstructed_pos_full_list.extend(res["reconstructed_position"])
                    reconstructed_heading_full_list.extend(res["reconstructed_heading"])
                    reconstructed_vel_full_list.extend(res["reconstructed_velocity"])

            current_pos = next_pos
            current_heading = next_heading
            current_vel = next_vel
            current_valid_mask = next_valid_mask

            reconstructed_pos_list.append(current_pos.clone())
            reconstructed_heading_list.append(current_heading.clone())
            reconstructed_vel_list.append(current_vel.clone())

        reconstructed_pos = torch.cat(reconstructed_pos_list, dim=1)
        reconstructed_heading = torch.cat(reconstructed_heading_list, dim=1)
        reconstructed_vel = torch.cat(reconstructed_vel_list, dim=1)

        # Every input token has it's own position (before the action).
        # As we have 19 tokens, and the last one token will lead us to a new place,
        # So it's totally 20 positions.
        assert reconstructed_pos.shape[1] == T_generated_chunks + 1
        assert input_mask.shape[1] == T_generated_chunks - start_step

        # Interpolation
        if interpolation:

            if already_interpolated:
                reconstructed_pos = torch.cat(reconstructed_pos_full_list, dim=1)
                reconstructed_heading = torch.cat(reconstructed_heading_full_list, dim=1)
                reconstructed_vel = torch.cat(reconstructed_vel_full_list, dim=1)

            else:

                # spline_res = interpolate_trajectory_spline(
                #     pos=reconstructed_pos,
                #     heading=reconstructed_heading,
                #     vel=reconstructed_vel,
                #     mask=torch.cat([input_mask, input_mask[:, -1:]], dim=1),
                # )
                # reconstructed_pos = spline_res["position"]
                # reconstructed_heading = spline_res["heading"]
                # reconstructed_vel = spline_res["velocity"]
                # reconstructed_valid_mask = spline_res["valid_mask"]

                new_reconstructed_pos = interpolate(reconstructed_pos, self.num_skipped_steps, remove_first_step=False)
                assert (new_reconstructed_pos[:, ::5] == reconstructed_pos).all()
                reconstructed_pos = new_reconstructed_pos

                reconstructed_heading = interpolate_heading(
                    reconstructed_heading, self.num_skipped_steps, remove_first_step=False
                )
                reconstructed_vel = interpolate(reconstructed_vel, self.num_skipped_steps, remove_first_step=False)

            input_mask_augmented = torch.cat([agent_valid_mask[:, :start_step], input_mask], dim=1)
            assert input_mask_augmented.shape[1] == T_generated_chunks
            valid = input_mask_augmented
            valid = valid.reshape(B, -1, 1, N).expand(-1, -1, self.num_skipped_steps, -1).reshape(B, -1, N)

            if teacher_forcing:

                # ====== insert True at newly_added token ======
                B, T, N = valid.shape
                first_true_index = (input_mask.cumsum(dim=1) == 1).max(dim=1).indices  # (B, N)  # Find the first True index for each agent in each batch            

                new_valid = torch.zeros((B, T + 1, N), dtype=valid.dtype, device=valid.device)  # Create a new valid tensor with expanded size (B, T+1, N)

                new_valid[:, :T, :] = valid  # Copy old values first # Copy everything before the insertion point

                # Create shift mask: Identify locations after first_true_index
                shift_mask = torch.arange(T, device=input_mask.device).view(1, -1, 1).expand(B, T, N) >= first_true_index.unsqueeze(1)

                # Apply shift in a fully vectorized way
                new_valid[:, 1:, :] = torch.where(shift_mask, valid, new_valid[:, 1:, :])

                # Insert True at first valid index without overwriting other values
                batch_indices = torch.arange(B, device=input_mask.device).view(-1, 1).expand(-1, N)
                agent_indices = torch.arange(N, device=input_mask.device).view(1, -1).expand(B, -1)

                new_valid[batch_indices, first_true_index * self.num_skipped_steps, agent_indices] = True  # Insert True

                # =============================

            else:
                new_valid = torch.cat([valid, input_mask[:, -1:]], dim=1)

            reconstructed_valid_mask = new_valid

            # Mask out:
            reconstructed_pos = reconstructed_pos * reconstructed_valid_mask.unsqueeze(-1)
            reconstructed_vel = reconstructed_vel * reconstructed_valid_mask.unsqueeze(-1)
            reconstructed_heading = reconstructed_heading * reconstructed_valid_mask

            # We ensure that the output must be 5*T_chunks+1
            assert reconstructed_pos.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_valid_mask.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_vel.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_heading.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
        else:
            reconstructed_valid_mask = input_mask

        data_dict["decoder/reconstructed_position"] = reconstructed_pos
        data_dict["decoder/reconstructed_heading"] = reconstructed_heading
        data_dict["decoder/reconstructed_velocity"] = reconstructed_vel
        data_dict["decoder/reconstructed_valid_mask"] = reconstructed_valid_mask

        return data_dict

    def _detokenize_backward_prediction(
        self,
        data_dict,
        interpolation=True,
        detokenizing_gt=False,
        flip_wrong_heading=False,
        teacher_forcing=False
    ):  # actions, current_pos, current_vel, current_heading):
        
        """
        Compared to the non-gpt style, this function dynamically adds new agents into the scene.
        A very interesting point here is we can't start with 'current position' in the data.
        Because the model is predicting according to the first few tokens, which already have some errors.
        """
        # TODO: Hardcoded here...
        assert self.config.GPT_STYLE
        start_step = 0
        # autoregressive_start_step = 2

        # ===== Get initial data =====
        agent_pos = data_dict["decoder/agent_position"].clone()
        agent_heading = data_dict["decoder/agent_heading"].clone()
        agent_valid_mask = data_dict["decoder/agent_valid_mask"].clone()
        agent_velocity = data_dict["decoder/agent_velocity"].clone()
        agent_shape = data_dict["decoder/current_agent_shape"].clone()
        agent_type = data_dict["decoder/agent_type"].clone()
        target_action_valid_mask = data_dict["decoder/target_action_valid_mask"]
        input_mask = data_dict["decoder/input_action_valid_mask"]
        B, T_full, N, _ = agent_pos.shape
        assert T_full == 91  # TODO: hardcoded
        assert agent_pos.ndim == 4

        # ===== Skip some steps =====
        agent_pos = agent_pos[:, ::self.num_skipped_steps]
        agent_heading = agent_heading[:, ::self.num_skipped_steps]
        agent_valid_mask = agent_valid_mask[:, ::self.num_skipped_steps]
        agent_velocity = agent_velocity[:, ::self.num_skipped_steps]
        T_chunks = agent_pos.shape[1]
        assert T_chunks == 19  # TODO: hardcoded

        # ===== Prepare some variables =====
        action = data_dict["decoder/output_action"]
        T_actions = action.shape[1]
        if T_actions + start_step != T_chunks:
            print(
                "WARNING: The number of actions is not consistent with the number of raw data chunks! You have {} actions, start step is {} and the number of chunks is {}."
                .format(T_actions, start_step, T_chunks)
            )
        T_generated_chunks = T_actions + start_step

        current_pos = agent_pos[:, -1:, ..., :2]
        current_heading = agent_heading[:, -1:]
        current_vel = agent_velocity[:, -1:, ..., :2]
        current_valid_mask = agent_valid_mask[:, -1:]

        if detokenizing_gt:
            # Merge input mask with target mask
            input_mask = input_mask & target_action_valid_mask

        reconstructed_pos_list = [current_pos.clone()]
        reconstructed_heading_list = [current_heading.clone()]
        reconstructed_vel_list = [current_vel.clone()]

        # Select correct bins:
        bin_centers = self.get_bin_centers(agent_type)

        previously_added = torch.zeros_like(current_valid_mask, dtype=torch.bool)

        for curr_backward_step in range(T_generated_chunks):
            # curr_backward_step = 0, 1, ..., 18

            curr_forward_step = T_chunks - curr_backward_step - 1
            # curr_forward_step = 18, 17, ..., 0

            next_forward_step = curr_forward_step - 1
            # next_forward_step = 17, 16, ..., -1

            action_valid_mask_step = input_mask[:, curr_backward_step:curr_backward_step + 1]
            act = action[:, curr_backward_step:curr_backward_step + 1]
            assert (act[action_valid_mask_step] != -1).all()
            res = self._detokenize_a_step(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=action_valid_mask_step,
                current_vel=current_vel,
                action=act,
                agent_shape=agent_shape,
                agent_type=agent_type,
                bin_centers=bin_centers,
                dt=-self.dt,
                flip_wrong_heading=flip_wrong_heading,
            )
            next_pos, next_heading, next_vel = res["pos"], res["heading"], res["vel"]
            next_pos = next_pos.reshape(B, 1, N, 2)
            next_heading = next_heading.reshape(B, 1, N)
            next_vel = next_vel.reshape(B, 1, N, 2)
            next_valid_mask = current_valid_mask

            # if detokenizing_gt and curr_backward_step < T_generated_chunks - 1:
            # TODO: Here the detokenizing_gt is ignored and we always add new agents in.
            if curr_backward_step < T_generated_chunks - 1:
                # Fill in the initial states of newly added agents
                action_valid_mask_next_step = input_mask[:, curr_backward_step + 1:curr_backward_step + 2]
                newly_added = torch.logical_and(~action_valid_mask_step, action_valid_mask_next_step)
                # newly_added = newly_added & (~previously_added)

                if newly_added.any():
                    # previously_added[newly_added] = True
                    next_pos[newly_added] = agent_pos[:, next_forward_step:next_forward_step + 1, ..., :2][newly_added]
                    next_heading[newly_added] = agent_heading[:, next_forward_step:next_forward_step + 1][newly_added]
                    next_vel[newly_added] = agent_velocity[:, next_forward_step:next_forward_step + 1, ..., :2][newly_added]
                    next_valid_mask[newly_added] = action_valid_mask_next_step[newly_added]

            current_pos = next_pos
            current_heading = next_heading
            current_vel = next_vel
            current_valid_mask = next_valid_mask

            reconstructed_pos_list.append(current_pos.clone())
            reconstructed_heading_list.append(current_heading.clone())
            reconstructed_vel_list.append(current_vel.clone())

        reconstructed_pos = torch.cat(reconstructed_pos_list, dim=1)
        reconstructed_heading = torch.cat(reconstructed_heading_list, dim=1)
        reconstructed_vel = torch.cat(reconstructed_vel_list, dim=1)

        # Every input token has it's own position (before the action).
        # As we have 19 tokens, and the last one token will lead us to a new place,
        # So it's totally 20 positions.
        assert reconstructed_pos.shape[1] == T_generated_chunks + 1
        assert input_mask.shape[1] == T_generated_chunks - start_step

        # TODO: Not sure if we should return flipped data or not.
        reconstructed_pos = reconstructed_pos.flip(dims=[1])
        reconstructed_heading = reconstructed_heading.flip(dims=[1])
        reconstructed_vel = reconstructed_vel.flip(dims=[1])
        input_mask = input_mask.flip(dims=[1])

        # Interpolation
        if interpolation:
            new_reconstructed_pos = interpolate(reconstructed_pos, self.num_skipped_steps, remove_first_step=False)
            assert (new_reconstructed_pos[:, ::5] == reconstructed_pos).all()
            reconstructed_pos = new_reconstructed_pos

            reconstructed_heading = interpolate_heading(
                reconstructed_heading, self.num_skipped_steps, remove_first_step=False
            )
            reconstructed_vel = interpolate(reconstructed_vel, self.num_skipped_steps, remove_first_step=False)

            # input_mask_augmented = torch.cat([agent_valid_mask[:, :start_step], input_mask], dim=1)
            input_mask_augmented = input_mask
            assert input_mask_augmented.shape[1] == T_generated_chunks
            valid = input_mask_augmented
            
            true_tensor = torch.ones((B, 1, N), dtype=torch.bool, device=input_mask.device)  # Shape: (B, 1, N)
            valid = valid.reshape(B, -1, 1, N).expand(-1, -1, self.num_skipped_steps, -1).reshape(B, -1, N)
            # valid = torch.cat([valid, input_mask[:, -1:]], dim=1)
            # valid = torch.cat([input_mask[:, 0:1], valid], dim=1)
            # insert corresponding newly added index mask

            if teacher_forcing:

                B, T, N = valid.shape
                # ====== insert True at newly_added token ======

                # Find the last True index for each agent in each batch
                last_true_index = (input_mask.cumsum(dim=1) == input_mask.sum(dim=1, keepdim=True)).max(dim=1).indices  # (B, N)

                # Mask to check which agents have at least one True
                valid_entries = input_mask.any(dim=1)  # (B, N)

                # Create a new valid tensor with expanded size (B, T+1, N)
                new_valid = torch.zeros((B, T + 1, N), dtype=valid.dtype, device=valid.device)

                # Copy everything before the insertion point
                new_valid[:, :T, :] = valid  # Copy old values first

                # Create shift mask: Identify locations after last_true_index
                shift_mask = torch.arange(T, device=input_mask.device).view(1, -1, 1).expand(B, T, N) >= last_true_index.unsqueeze(1)

                # Apply shift in a fully vectorized way
                new_valid[:, 1:, :] = torch.where(shift_mask, valid, new_valid[:, 1:, :])

                # Insert True at first valid index without overwriting other values
                batch_indices = torch.arange(B, device=input_mask.device).view(-1, 1).expand(-1, N)
                agent_indices = torch.arange(N, device=input_mask.device).view(1, -1).expand(B, -1)

                new_valid[batch_indices, last_true_index * self.num_skipped_steps, agent_indices] = True  # Insert True

            # =============================

            else:
                new_valid = torch.cat([input_mask[:, 0:1], valid], dim=1)

            reconstructed_valid_mask = new_valid

            # Mask out:
            reconstructed_pos = reconstructed_pos * reconstructed_valid_mask.unsqueeze(-1)
            reconstructed_vel = reconstructed_vel * reconstructed_valid_mask.unsqueeze(-1)
            reconstructed_heading = reconstructed_heading * reconstructed_valid_mask

            # We ensure that the output must be 5*T_chunks+1
            assert reconstructed_pos.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_valid_mask.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_vel.shape[1] == self.num_skipped_steps * T_generated_chunks + 1
            assert reconstructed_heading.shape[1] == self.num_skipped_steps * T_generated_chunks + 1

        else:
            reconstructed_valid_mask = input_mask

        if not teacher_forcing:
            T_start = 5
        else:
            T_start = 0

        data_dict["decoder/reconstructed_position"] = reconstructed_pos[:,T_start:] # return the corresponding 91-step trajectories
        data_dict["decoder/reconstructed_heading"] = reconstructed_heading[:,T_start:]
        data_dict["decoder/reconstructed_velocity"] = reconstructed_vel[:, T_start:]
        data_dict["decoder/reconstructed_valid_mask"] = reconstructed_valid_mask[:, T_start:]

        assert reconstructed_pos.shape[1] == 91, f"be careful that reconstructed backward trajectory has T={reconstructed_pos.shape[1]}"

        return data_dict

    def _tokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading, next_valid_mask,
        next_velocity, bin_centers, add_noise, topk, agent_shape, agent_type, dt, **kwargs
    ):
        B, _, N, _ = current_pos.shape

        valid_mask = torch.logical_and(current_valid_mask, next_valid_mask)

        delta_vel = rotate_bin_to_absolute_heading(bin_centers, current_heading)

        candidate_vel = delta_vel + current_vel

        candidate_pos = candidate_vel * dt + current_pos

        flip_heading_accordingly = kwargs.get("flip_heading_accordingly", True)

        candidate_heading = infer_heading(
            current_pos=candidate_pos,
            last_pos=current_pos.expand(-1, self.num_actions, -1, -1),
            last_heading=current_heading.expand(-1, self.num_actions, -1),
            min_displacement=self.config.TOKENIZATION.MIN_DISPLACEMENT,
            flip_heading=(flip_heading_accordingly and dt < 0)
        )

        contour = utils.cal_polygon_contour_torch(
            x=candidate_pos[..., 0],
            y=candidate_pos[..., 1],
            theta=candidate_heading,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )

        gt_contour = utils.cal_polygon_contour_torch(
            x=next_pos[..., 0],
            y=next_pos[..., 1],
            theta=next_heading,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )

        error_ade = torch.norm(candidate_pos - next_pos, dim=-1)
        error_ade = error_ade * valid_mask

        error_pos = torch.norm(contour - gt_contour, dim=-1).mean(-1)
        error_pos = error_pos * valid_mask  # masking
        assert error_pos.ndim == 3

        # error_heading = utils.wrap_to_pi(candidate_vel_heading - next_heading.expand(-1, self.num_actions, -1))
        # error_heading = error_heading.abs() * valid_mask

        if self.config.TOKENIZATION.USE_CONTOUR_ERROR:
            error = error_pos  # + error_heading

        else:
            error = error_ade  # + error_heading

        if add_noise:
            # raise ValueError()
            print("Noise is not supported in the current version.")
            # sampled_action = nucleus_sampling(logits=1 / (error.permute(0, 2, 1) + 1e-6), p=0.95)
            # sampled_error = torch.gather(error, 1, sampled_action.unsqueeze(1)).squeeze(1)
            # best_action = sampled_action
            min_result = error.min(dim=1)
            best_action = min_result.indices

        else:
            # Pick the best bin with the least error:
            min_result = error.min(dim=1)
            best_action = min_result.indices

        best_action[~valid_mask.squeeze(1)] = -1

        # Update reconstructed position and velocity according to the best action:
        ind = best_action.reshape(B, 1, N, 1).expand(B, 1, N, 2).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_pos = torch.gather(candidate_pos, index=ind, dim=1)
        reconstructed_vel = torch.gather(candidate_vel, index=ind, dim=1)
        reconstructed_heading = torch.gather(candidate_heading, index=ind[..., 0], dim=1)

        reconstructed_vel[mask] = 0
        reconstructed_pos[mask] = 0
        reconstructed_heading[~valid_mask] = 0
        assert current_pos.shape == reconstructed_pos.shape

        # FIXME: This is actually wrong in backward prediction. It's the flipped version of "current velocity".
        #  But that's OK if Tokenization/Autoregressive share the same code.
        relative_delta_pos = reconstructed_pos - current_pos
        relative_delta_pos = utils.rotate(
            relative_delta_pos[..., 0], relative_delta_pos[..., 1], angle=-reconstructed_heading
        )
        relative_delta_pos[mask] = 0

        # AID = 0
        # print("CUR {}, Recon Pos: {}, GT Pos {}, Cur Vel: {}, Vel: {}, CUR Head: {}, RECON Head: {}".format(
        #     current_pos[0,0,AID],
        #     reconstructed_pos[0,0,AID],
        #     next_pos[0,0,AID],
        #     current_vel[0,0,AID],
        #     reconstructed_vel[0,0,AID],
        #     current_heading[0,0,AID],
        #     reconstructed_heading[0,0,AID]
        # ))

        return dict(
            action=best_action,
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            mask=valid_mask,
            delta_pos=relative_delta_pos,
        )

    # def detokenize_for_step(self, data_dict, action):
    #     # get reconstructed heading:
    #     return self._detokenize_a_step(
    #         current_pos=data_dict["decoder/current_agent_position"],
    #         current_heading=data_dict["decoder/current_agent_heading"],
    #         current_valid_mask=data_dict["decoder/current_agent_valid_mask"],
    #         current_vel=data_dict["decoder/current_agent_velocity"],
    #         action=action
    #     )

    def _detokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, action, bin_centers, dt,
        flip_wrong_heading, **kwargs
    ):
        assert action.ndim == 3
        B, T_action, N = action.shape

        assert T_action == 1

        # TODO: delta_pos computing is updated.
        if self.config.DELTA_POS_IS_VELOCITY:
            raise ValueError

        # if self.bin_centers.device != action.device:
        #     self.bin_centers = self.bin_centers.to(action.device)
        # bin_centers = self.bin_centers
        # bin_centers = bin_centers.reshape(1, 1, 1, self.num_actions, 2).expand(B, T_action, N, self.num_actions, 2)

        action_expanded = action.reshape(B, T_action, N, 1).expand(B, T_action, N, 2).clone()
        mask = (action_expanded == -1) | (action_expanded == START_ACTION) | (action_expanded == END_ACTION)
        action_expanded[mask] = 0
        delta_vel_candidates = torch.gather(bin_centers, index=action_expanded, axis=1)  # .squeeze(1)
        # delta_vel_candidates[mask.squeeze(-2)] = 0
        # assert (current_valid_mask ==  (action!=-1)).all()

        unrotated_delta_vel = delta_vel_candidates

        reconstructed_pos = torch.clone(current_pos[..., :2]).reshape(B, 1, N, 2)
        reconstructed_heading = torch.clone(current_heading).reshape(B, 1, N)
        reconstructed_vel = torch.clone(current_vel).reshape(B, 1, N, 2)

        # Reconstruct position and heading:
        delta_vel = rotate_bin_to_absolute_heading(unrotated_delta_vel, reconstructed_heading)
        new_reconstructed_vel = delta_vel + reconstructed_vel
        new_reconstructed_pos = new_reconstructed_vel * dt + reconstructed_pos

        flip_heading_accordingly = kwargs.get("flip_heading_accordingly", True)
        new_reconstructed_heading = infer_heading(
            current_pos=new_reconstructed_pos,
            last_pos=reconstructed_pos,
            last_heading=reconstructed_heading,
            current_velocity=reconstructed_vel,
            # init_pos=init_pos,
            min_displacement=self.config.TOKENIZATION.MIN_DISPLACEMENT,
            min_displacement_init=self.config.TOKENIZATION.MIN_DISPLACEMENT_INIT,
            min_speed=self.config.TOKENIZATION.MIN_SPEED,
            smooth_factor=self.config.TOKENIZATION.SMOOTH_FACTOR,
            max_heading_diff=self.config.TOKENIZATION.MAX_HEADING_DIFF,
            flip_heading=flip_heading_accordingly and dt < 0,
            # ema_heading=ema_heading
        )

        # PZH: This is a dirty workaround!
        if flip_wrong_heading:
            wrong_heading_mask = utils.wrap_to_pi(new_reconstructed_heading -
                                                  reconstructed_heading).abs() > np.deg2rad(90)
            wrong_heading_mask = wrong_heading_mask & current_valid_mask
            # Flipped??
            new_reconstructed_heading[wrong_heading_mask] = utils.wrap_to_pi(
                new_reconstructed_heading[wrong_heading_mask] + np.pi
            )

        new_reconstructed_heading = new_reconstructed_heading.reshape(B, 1, N)

        # Update reconstructed pos and vel
        reconstructed_pos = new_reconstructed_pos
        assert reconstructed_pos.shape == (B, 1, N, 2)
        reconstructed_vel = new_reconstructed_vel

        reconstructed_pos = reconstructed_pos.reshape(B, N, 2)
        new_reconstructed_heading = new_reconstructed_heading.reshape(B, N)
        reconstructed_vel = reconstructed_vel.reshape(B, N, 2)

        # Masking
        reconstructed_pos = (current_valid_mask.reshape(B, N, 1).expand(B, N, 2) * reconstructed_pos)
        new_reconstructed_heading = (current_valid_mask.reshape(B, N) * new_reconstructed_heading)

        reconstructed_pos = reconstructed_pos.reshape(B, 1, N, 2)
        new_reconstructed_heading = new_reconstructed_heading.reshape(B, 1, N)
        reconstructed_vel = reconstructed_vel.reshape(B, 1, N, 2)

        relative_delta_pos = reconstructed_pos.reshape(B, 1, N, 2) - current_pos
        relative_delta_pos = utils.rotate(
            relative_delta_pos[..., 0],
            relative_delta_pos[..., 1],
            angle=-new_reconstructed_heading.reshape(B, 1, N) + np.pi
        )
        # AID = 14
        # print(
        #     "POS: {}, HEAD: {}, VEL: {}, SPEED: {}, unrotated_delta_vel: {}, cur vel {}".format(
        #         reconstructed_pos[0, 0, AID].cpu().numpy(),
        #         reconstructed_heading[0, 0, AID],
        #         reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy(),
        #         reconstructed_vel.norm(dim=-1)[0, 0, AID],
        #         unrotated_delta_vel[0, 0, AID].cpu().numpy(),
        #         current_vel[0, 0, AID].norm(dim=-1)
        #     )
        # )

        return dict(
            pos=reconstructed_pos,
            heading=new_reconstructed_heading,
            vel=reconstructed_vel,
            delta_pos=relative_delta_pos,
            # trajectory_pos=rotated_selected_trajs_pos,
            # trajectory_heading=rotated_selected_trajs_head
        )


class DeltaTokenizer(DeltaDeltaTokenizer):
    def __init__(self, config):
        BaseTokenizer.__init__(self, config)

        from bmt.utils import REPO_ROOT
        # import numpy as np
        import pickle

        self.use_type_specific_bins = False

        with open(REPO_ROOT / config.DELTA_TOKENIZER_FILE_NAME, 'rb') as f:
            veh = pickle.load(f)
        all_trajs = veh["trajs"]
        all_head = veh["heading"]

        self.num_actions = len(all_trajs)

        self.all_trajs = torch.from_numpy(all_trajs).float()
        self.bin_centers = self.all_trajs[:, -1].reshape(1, self.num_actions, 1, 2)

        self.config = config
        self.all_heading = torch.from_numpy(all_head).float()

        self.default_action = 0  # We set action 0 to be all zeros.
        self.add_noise = config.TOKENIZATION.ADD_NOISE

    def get_motion_feature(self):
        # m = torch.from_numpy(self.bin_centers_flat)
        m = self.all_trajs[:, -1]  # (1025, 2)
        dist = m.norm(p=2, dim=-1).unsqueeze(-1)
        heading = self.all_heading[:, -1]
        return torch.cat([m, dist, heading], dim=-1)

    def _tokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading, next_valid_mask,
        next_velocity, bin_centers, add_noise, agent_shape, **kwargs
    ):
        B, _, N, _ = current_pos.shape
        valid_mask = torch.logical_and(current_valid_mask, next_valid_mask)
        delta_pos = rotate_bin_to_absolute_heading(bin_centers, current_heading)
        candidate_pos = delta_pos + current_pos
        head = self.all_heading[:, -1].reshape(1, -1, 1).expand(B, -1, N)
        candidate_heading = current_heading.reshape(B, 1, N) + head
        candidate_pos = candidate_pos.reshape(B, -1, N, 2)
        contour = utils.cal_polygon_contour_torch(
            x=candidate_pos[..., 0],
            y=candidate_pos[..., 1],
            theta=candidate_heading,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )
        gt_contour = utils.cal_polygon_contour_torch(
            x=next_pos[..., 0],
            y=next_pos[..., 1],
            theta=next_heading,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )
        error_pos = torch.norm(contour - gt_contour, dim=-1).mean(-1)
        error = error_pos * valid_mask

        if add_noise:
            raise ValueError()
            sampled_action = nucleus_sampling(logits=1 / (error.permute(0, 2, 1) + 1e-6), p=0.95)
            sampled_error = torch.gather(error, 1, sampled_action.unsqueeze(1)).squeeze(1)
            best_action = sampled_action

        else:
            # Pick the best bin with the least error:
            min_result = error.min(dim=1)
            best_action = min_result.indices

        best_action[~valid_mask.squeeze(1)] = -1

        # Update reconstructed position and velocity according to the best action:
        ind = best_action.reshape(B, 1, N, 1).expand(B, 1, N, 2).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_pos = torch.gather(candidate_pos, index=ind, dim=1)
        reconstructed_pos[mask] = 0
        assert current_pos.shape == reconstructed_pos.shape

        if self.all_heading.device != reconstructed_pos.device:
            self.all_heading = self.all_heading.to(reconstructed_pos.device)
        all_heading = self.all_heading[:, -1].reshape(1, self.num_actions, 1).expand(B, -1, N)
        ind = best_action.reshape(B, 1, N).clone()
        ind[ind == -1] = self.default_action
        reconstructed_heading = torch.gather(all_heading, index=ind, dim=1)
        reconstructed_heading = reconstructed_heading + current_heading
        reconstructed_heading[~valid_mask] = 0

        reconstructed_vel = (reconstructed_pos - current_pos) / self.dt
        reconstructed_vel[~valid_mask] = 0

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[~valid_mask] = 0

        best_action[~valid_mask.squeeze(1)] = -1
        assert (best_action[valid_mask.squeeze(1)] >= 0).all()
        assert (best_action[~valid_mask.squeeze(1)] == -1).all()
        # AID = 26
        # print("CUR {}, Recon Pos: {}, GT Pos {}, Cur Vel: {}, Vel: {}, CUR Head: {}, RECON Head: {}".format(
        #     current_pos[0,0,AID],
        #     reconstructed_pos[0,0,AID],
        #     next_pos[0,0,AID],
        #     current_vel[0,0,AID],
        #     reconstructed_vel[0,0,AID],
        #     current_heading[0,0,AID],
        #     reconstructed_heading[0,0,AID]
        # ))
        return dict(
            action=best_action,
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            mask=valid_mask,
            delta_pos=relative_delta_pos,
        )

    def _detokenize_a_step(self, *, current_pos, current_heading, current_valid_mask, current_vel, action, **kwargs):
        assert action.ndim == 3
        B, T_action, N = action.shape

        assert T_action == 1

        bin_centers = self.bin_centers.to(action.device)
        bin_centers = bin_centers.reshape(1, 1, 1, self.num_actions, 2).expand(B, T_action, N, self.num_actions, 2)
        action_expanded = action.reshape(B, T_action, N, 1, 1).expand(B, T_action, N, 1, 2).clone()
        mask = (action_expanded == -1) | (action_expanded == START_ACTION)
        action_expanded[mask] = 0
        delta_pos_candidates = torch.gather(bin_centers, index=action_expanded, axis=3).squeeze(-2)

        reconstructed_pos = torch.clone(current_pos[..., :2]).reshape(B, 1, N, 2)
        reconstructed_heading = torch.clone(current_heading).reshape(B, 1, N)

        # Reconstruct position and heading:
        delta_pos = rotate_bin_to_absolute_heading(delta_pos_candidates, current_heading.reshape(B, 1, N))
        new_reconstructed_pos = delta_pos + reconstructed_pos

        if self.all_trajs.device != new_reconstructed_pos.device:
            self.all_trajs = self.all_trajs.to(new_reconstructed_pos.device)
        if self.all_heading.device != reconstructed_heading.device:
            self.all_heading = self.all_heading.to(reconstructed_heading.device)

        all_trajs = self.all_trajs.reshape(1, self.num_actions, 1, 5, 2).expand(B, -1, N, -1, -1)
        action_expanded_for_traj = action.reshape(B, T_action, N, 1, 1).expand(B, T_action, N, 5, 2).clone()
        mask = (action_expanded_for_traj == -1) | (action_expanded_for_traj == START_ACTION)
        action_expanded_for_traj[mask] = 0
        selected_trajs = torch.gather(all_trajs, index=action_expanded_for_traj, axis=1).squeeze(1)

        all_heading = self.all_heading.reshape(1, self.num_actions, 1, 5).expand(B, -1, N, -1)
        action_expanded_for_traj = action.reshape(B, T_action, N, 1).expand(B, T_action, N, 5).clone()
        mask = (action_expanded_for_traj == -1) | (action_expanded_for_traj == START_ACTION)
        action_expanded_for_traj[mask] = 0
        selected_heading = torch.gather(all_heading, index=action_expanded_for_traj, axis=1).squeeze(1)

        reconstructed_heading = selected_heading[..., -1] + current_heading.reshape(B, N)
        reconstructed_heading = reconstructed_heading.reshape(B, 1, N)

        new_reconstructed_vel = (new_reconstructed_pos - reconstructed_pos) / self.dt
        reconstructed_vel = new_reconstructed_vel

        # Update reconstructed pos and vel
        reconstructed_pos = new_reconstructed_pos
        assert reconstructed_pos.shape == (B, 1, N, 2)

        reconstructed_pos = reconstructed_pos.reshape(B, N, 2)
        reconstructed_heading = reconstructed_heading.reshape(B, N)
        reconstructed_vel = reconstructed_vel.reshape(B, N, 2)

        # Masking
        reconstructed_pos = (current_valid_mask.reshape(B, N, 1).expand(B, N, 2) * reconstructed_pos)
        reconstructed_heading = (current_valid_mask.reshape(B, N) * reconstructed_heading)

        rotated_selected_trajs_pos = rotate_bin_to_absolute_heading(
            selected_trajs[..., :2],
            current_heading.reshape(B, N, 1).expand(B, N, 5)
        )
        rotated_selected_trajs_pos = rotated_selected_trajs_pos + current_pos.reshape(B, N, 1, 2).expand(B, N, 5, 2)
        rotated_selected_trajs_head = selected_heading + current_heading.reshape(B, N, 1).expand(B, N, 5)

        full_pos = torch.cat([current_pos.swapaxes(1, 2), rotated_selected_trajs_pos], dim=2)
        rotated_selected_trajs_vel = (full_pos[:, :, 1:] - full_pos[:, :, :-1]) / (self.dt / self.num_skipped_steps)

        # Masking
        rotated_selected_trajs_pos = (
            current_valid_mask.reshape(B, N, 1, 1).expand(B, N, 5, 2) * rotated_selected_trajs_pos
        )
        rotated_selected_trajs_head = (
            current_valid_mask.reshape(B, N, 1).expand(B, N, 5) * rotated_selected_trajs_head
        )
        rotated_selected_trajs_vel = (
            current_valid_mask.reshape(B, N, 1, 1).expand(B, N, 5, 2) * rotated_selected_trajs_vel
        )
        # AID = 26
        # print(
        #     "ACTION: {}, Cur Pos: {}, POS: {}, HEAD: {}, VEL: {}, SPEED: {}, cur vel {}, cur mask {}".format(
        #         action[0, 0, AID].cpu().numpy(),
        #         current_pos[0, 0, AID].cpu().numpy(),
        #         reconstructed_pos[0, AID].cpu().numpy(),
        #         reconstructed_heading[0, AID],
        #         reconstructed_vel[0, AID].norm(dim=-1).cpu().numpy(),
        #         reconstructed_vel.norm(dim=-1)[0, AID],
        #         # unrotated_delta_vel[0, 0, AID].cpu().numpy(),
        #         current_vel[0, 0, AID].norm(dim=-1),
        #         current_valid_mask[0, 0, AID]
        #     )
        # )

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[~current_valid_mask.reshape(B, N)] = 0

        return dict(
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            delta_pos=relative_delta_pos,
            # trajectory_pos=rotated_selected_trajs_pos,
            reconstructed_position=[
                rotated_selected_trajs_pos[:, :, t].unsqueeze(1) for t in range(self.num_skipped_steps)
            ],
            reconstructed_heading=[
                rotated_selected_trajs_head[:, :, t].unsqueeze(1) for t in range(self.num_skipped_steps)
            ],
            reconstructed_velocity=[
                rotated_selected_trajs_vel[:, :, t].unsqueeze(1) for t in range(self.num_skipped_steps)
            ],
        )
