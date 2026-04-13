import numpy as np
import torch

from bmt.tokenization.motion_tokenizers import DeltaDeltaTokenizer, get_relative_velocity, START_ACTION, END_ACTION, BaseTokenizer, interpolate, interpolate_heading
from bmt.utils import rotate
from bmt.utils import utils


def get_relative_velocity(vel, heading):
    return utils.rotate(vel[..., 0], vel[..., 1], angle=-heading)


# class BicycleModelTokenizerFixed0124(DeltaDeltaTokenizer):
class BicycleModelTokenizerFixed0124(BaseTokenizer):
    ACC_MAX = 10  # m/s2
    YAW_RATE_MAX = np.pi / 2  # Just set to < 90 deg otherwise the tan() function will be too large.

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bin_centers = None
        assert self.config.DELTA_POS_IS_VELOCITY

        ACC_MAX = self.ACC_MAX
        YAW_RATE_MAX = self.YAW_RATE_MAX
        print("BicycleModelTokenizer: ACC_MAX: ", ACC_MAX, "YAW_RATE_MAX: ", YAW_RATE_MAX)

        self.x_max = ACC_MAX
        self.x_min = -self.x_max
        self.y_max = YAW_RATE_MAX
        self.y_min = -self.y_max
        # assert self.y_max < np.pi / 2

        self.num_bins = config.TOKENIZATION.NUM_BINS
        # assert self.num_bins == 33
        self.num_actions = self.num_bins**2

        self.acceleration_bins = torch.linspace(self.x_min, self.x_max, self.num_bins)
        self.steering_bins = torch.linspace(self.y_min, self.y_max, self.num_bins)

        self.default_action = self.num_bins**2 // 2

        a_grid, delta_grid = torch.meshgrid(self.acceleration_bins, self.steering_bins, indexing='ij')
        a_grid = a_grid.flatten()  # .to(device)  # Shape: (num_bins^2,)
        delta_grid = delta_grid.flatten()  # .to(device)  # Shape: (num_bins^2,)

        self.a_grid_flat = a_grid
        self.delta_grid_flat = delta_grid
        self.bin_centers_flat = torch.stack([a_grid, delta_grid], dim=-1).cpu().numpy()  # Shape: (num_bins^2, 2)

        self.use_type_specific_bins = False


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

        assert self.config.DELTA_POS_IS_VELOCITY
        init_delta = get_relative_velocity(current_vel, current_heading)

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
                add_noise=False,
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

                    recon_next_delta_pos[newly_added] = get_relative_velocity(
                        vel=agent_velocity[:, next_step:next_step + 1, ..., :2][newly_added],
                        heading=agent_heading[:, next_step:next_step + 1][newly_added],
                    )
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
        assert self.config.GPT_STYLE
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

    def detokenize(
        self,
        data_dict,
        interpolation=True,
        detokenizing_gt=False,
        backward_prediction=False,
        flip_wrong_heading=False,
        autoregressive_start_step=2,
        **kwargs,
    ):  # actions, current_pos, current_vel, current_heading):

        if backward_prediction:
            return self._detokenize_backward_prediction(
                data_dict, interpolation=interpolation, detokenizing_gt=detokenizing_gt, **kwargs
            )

        # TODO: Hardcoded here...
        assert self.config.GPT_STYLE
        start_step = 0

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
            valid = torch.cat([valid, input_mask[:, -1:]], dim=1)
            reconstructed_valid_mask = valid

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

    def _tokenize_backward_prediction(self, data_dict, **kwargs):
        start_step = 0

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

        init_delta = get_relative_velocity(current_vel, current_heading)

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
                add_noise=False,
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
            if newly_added.any():
                recon_next_pos[newly_added] = agent_pos[:, forward_next_step:forward_next_step + 1,
                                                        ..., :2][newly_added]
                recon_next_heading[newly_added] = agent_heading[:, forward_next_step:forward_next_step + 1][newly_added]
                recon_next_vel[newly_added] = agent_velocity[:, forward_next_step:forward_next_step + 1,
                                                             ..., :2][newly_added]
                recon_next_delta_pos[newly_added] = get_relative_velocity(
                    vel=agent_velocity[:, forward_next_step:forward_next_step + 1, ..., :2][newly_added],
                    heading=agent_heading[:, forward_next_step:forward_next_step + 1][newly_added],
                )
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
        for backward_next_step in range(1, T_chunks):
            forward_next_step = T_chunks - backward_next_step - 1
            next_valid_mask = agent_valid_mask[:, forward_next_step:forward_next_step + 1]
            is_newly_added = torch.logical_and(~already_tokenized, next_valid_mask)
            if is_newly_added.any():
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


    def _tokenize_a_step_backward(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading, next_valid_mask,
        next_vel, add_noise, agent_shape, dt, **kwargs
    ):
        assert dt > 0

        device = current_pos.device
        B, T, N, _ = current_pos.shape
        assert T == 1

        # Prepare grid of accelerations and steering angles for batch processing
        a_grid = self.a_grid_flat.to(device)  # Shape: (num_bins^2,)
        delta_grid = self.delta_grid_flat.to(device)  # Shape: (num_bins^2,)

        num_candidates = a_grid.shape[0]
        a_grid_exp = a_grid.view(1, -1, 1).expand(B, num_candidates, N)  # (B, num_candidates, N, 1)
        yaw_rate = delta_grid.view(1, -1, 1).expand(B, num_candidates, N)  # (B, num_candidates, N, 1)

        # Repeat current states for batch computation
        current_pos_exp = current_pos.expand(B, num_candidates, N, 2)  # (B, num_candidates, N, 2)
        current_heading_exp = current_heading.expand(B, num_candidates, N)  # (B, num_candidates, N)
        next_heading_exp = next_heading.expand(B, num_candidates, N)  # (B, num_candidates, N)

        # Next speed:
        next_speed = next_vel.norm(dim=-1).expand(B, num_candidates, N)
        current_speed_candidate = next_speed - a_grid_exp * dt
        average_speed = (current_speed_candidate + next_speed) / 2

        # wheel_base = agent_shape[..., 0].reshape(B, 1, N)
        # yaw_rate = (average_speed / wheel_base) * torch.tan(delta_grid_exp)
        delta_theta = yaw_rate * dt
        current_heading_candidate = utils.wrap_to_pi(next_heading_exp - delta_theta)
        average_heading = utils.wrap_to_pi(utils.average_heading(current_heading_candidate, next_heading_exp))

        average_velocity_candidate = rotate(average_speed, torch.zeros_like(average_speed), angle=average_heading)
        current_velocity_candidate = rotate(
            current_speed_candidate, torch.zeros_like(current_speed_candidate), angle=current_heading_candidate
        )

        current_pos_reconstructed = next_pos - average_velocity_candidate * dt
        current_pos_reconstructed = current_pos_reconstructed.expand(B, num_candidates, N, 2)

        contour = utils.cal_polygon_contour_torch(
            x=current_pos_reconstructed[..., 0],
            y=current_pos_reconstructed[..., 1],
            theta=current_heading_candidate,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )

        gt_contour = utils.cal_polygon_contour_torch(
            x=current_pos_exp[..., 0],
            y=current_pos_exp[..., 1],
            theta=current_heading_exp,
            width=agent_shape[..., 1].reshape(B, 1, N),
            length=agent_shape[..., 0].reshape(B, 1, N)
        )

        error_pos = torch.norm(contour - gt_contour, dim=-1).mean(-1)
        error = error_pos  # + error_heading

        if self.noise.device != error.device:
            self.noise = self.noise.to(error.device)
        error = error + self.noise

        if add_noise:
            # Get top-k actions based on the error
            candidates = error.topk(5, largest=False, dim=1).indices
            best_action = torch.gather(
                candidates, index=torch.randint(0, 5, size=(B, 1, N)).to(candidates.device), dim=1
            ).squeeze(1)
            raise ValueError()

        else:
            # Pick the best bin with the least error:
            min_result = error.min(dim=1)
            best_action = min_result.indices

        # Update reconstructed position and velocity according to the best action:
        ind = best_action.reshape(B, 1, N, 1).expand(B, 1, N, 2).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_pos = torch.gather(current_pos_reconstructed, index=ind, dim=1)
        reconstructed_vel = torch.gather(current_velocity_candidate, index=ind, dim=1)

        ind = best_action.reshape(B, 1, N).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_heading = torch.gather(current_heading_candidate, index=ind, dim=1)

        valid_mask = current_valid_mask & next_valid_mask
        assert current_pos.shape == reconstructed_pos.shape
        best_action = best_action.reshape(B, 1, N)
        best_action[~valid_mask] = -1
        reconstructed_pos[~valid_mask] = 0
        reconstructed_vel[~valid_mask] = 0
        reconstructed_heading[~valid_mask] = 0
        assert (best_action[valid_mask] >= 0).all()
        assert (best_action[~valid_mask] == -1).all()
        assert self.num_bins == 33

        # Just return the relative velocity.
        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[mask] = 0

        # AID = 0
        # masked_best_action = best_action * valid_mask
        # delta = torch.gather(delta_grid_exp, index=masked_best_action.reshape(B, 1, N), dim=1)
        # acc = torch.gather(a_grid_exp, index=masked_best_action.reshape(B, 1, N), dim=1)
        # action = best_action[0,0,AID]
        # average_velocity_candidate = average_velocity_candidate[0,action,AID]
        # current_velocity_candidate = current_velocity_candidate[0,action,AID]
        # print(
        #     f"[TOK] AID{AID} CUR POS: {current_pos[0, 0, AID].cpu().numpy()}, "
        #     f"RECON POS: {reconstructed_pos[0, 0, AID].cpu().numpy()}, "
        #     f"NEXT POS: {next_pos[0, 0, AID].cpu().numpy()}, "
        #     f"Action: {best_action[0, 0, AID].cpu().numpy()}, "
        #     f"ACC: {acc[0, 0, AID].cpu().numpy()}, "
        #     f"STEER: {delta[0, 0, AID].cpu().numpy()}, "
        #     # f"CUR VEL: {current_vel[0, 0, AID].norm(dim=-1).cpu().numpy()}, "
        #     f"RECON VEL: {reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy()}, "
        #     f"VALID: {valid_mask[0, 0, AID].cpu().numpy()}",
        #     f"CUR VEL: {current_velocity_candidate.cpu().numpy()}",
        #     f"AVG VEL: {average_velocity_candidate.cpu().numpy()}",
        # )

        return dict(
            action=best_action,
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            mask=valid_mask,
            delta_pos=relative_delta_pos
        )

    def _tokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading, next_valid_mask,
        add_noise, agent_shape, dt, **kwargs
    ):

        if dt < 0:
            # TODO: This is a trick to handle the backward prediction. We flip current/next states and dt.
            #  Might cause confusion to other users.
            return self._tokenize_a_step_backward(
                current_pos=next_pos,
                current_heading=next_heading,
                current_valid_mask=next_valid_mask,
                current_vel=None,
                next_vel=current_vel,
                next_pos=current_pos,
                next_heading=current_heading,
                next_valid_mask=current_valid_mask,
                add_noise=add_noise,
                agent_shape=agent_shape,
                dt=-dt,
                **kwargs
                # current_pos, current_heading, current_valid_mask, current_vel, next_pos, next_heading,
                #     next_valid_mask,
                #     add_noise, agent_shape, -dt, **kwargs
            )

        device = current_pos.device
        B, T, N, _ = current_pos.shape
        assert T == 1

        # Prepare grid of accelerations and steering angles for batch processing
        a_grid = self.a_grid_flat.to(device)  # Shape: (num_bins^2,)
        delta_grid = self.delta_grid_flat.to(device)  # Shape: (num_bins^2,)

        num_candidates = a_grid.shape[0]
        a_grid_exp = a_grid.view(1, -1, 1).expand(B, num_candidates, N)  # (B, num_candidates, N, 1)
        delta_grid_exp = delta_grid.view(1, -1, 1).expand(B, num_candidates, N)  # (B, num_candidates, N, 1)

        # Repeat current states for batch computation
        current_pos_exp = current_pos.expand(B, num_candidates, N, 2)  # (B, num_candidates, N, 2)
        current_heading_exp = current_heading.expand(B, num_candidates, N)  # (B, num_candidates, N)

        # Current speed in local frame:
        current_speed = current_vel.norm(dim=-1)  # (B, num_candidates, N)
        current_speed = current_speed.expand(B, num_candidates, N)  # (B, num_candidates, N, 2)
        next_speed_candidate = current_speed + a_grid_exp * self.dt
        average_speed = (current_speed + next_speed_candidate) / 2

        # wheel_base = agent_shape[..., 0].reshape(B, 1, N)
        # yaw_rate = (average_speed / wheel_base) * torch.tan(delta_grid_exp)
        yaw_rate = delta_grid_exp
        delta_theta = yaw_rate * self.dt
        next_heading_candidate = utils.wrap_to_pi(current_heading_exp + delta_theta)
        average_heading = utils.wrap_to_pi(utils.average_heading(next_heading_candidate, current_heading_exp))

        # Rotate velocity vector to update both v_x and v_y
        next_velocity_candidate = rotate(
            next_speed_candidate, torch.zeros_like(next_speed_candidate), angle=next_heading_candidate
        )  # (B, num_candidates, N, 2)

        average_next_velocity = rotate(average_speed, torch.zeros_like(average_speed), angle=average_heading)
        next_pos_candidate = current_pos_exp + average_next_velocity * self.dt

        no_displacement_mask = None
        invalid_next_pos = None

        contour = utils.cal_polygon_contour_torch(
            x=next_pos_candidate[..., 0],
            y=next_pos_candidate[..., 1],
            theta=next_heading_candidate,
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
        error = error_pos  # + error_heading

        # Add the very small noise to break the tie!
        if self.noise.device != error.device:
            self.noise = self.noise.to(error.device)
        error = error + self.noise

        if add_noise:
            # Get top-k actions based on the error
            TOPK = 5
            candidates = error.topk(TOPK, largest=False, dim=1).indices
            best_action = torch.gather(
                candidates, index=torch.randint(0, TOPK, size=(B, 1, N)).to(candidates.device), dim=1
            ).squeeze(1)
            raise ValueError()
        else:
            # Pick the best bin with the least error:
            min_result = error.min(dim=1)
            best_action = min_result.indices

        if no_displacement_mask is not None:
            best_action[no_displacement_mask.squeeze(1)] = self.default_action

        if invalid_next_pos is not None:
            best_action[invalid_next_pos.squeeze(1)] = self.default_action

        # Update reconstructed position and velocity according to the best action:
        ind = best_action.reshape(B, 1, N, 1).expand(B, 1, N, 2).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_pos = torch.gather(next_pos_candidate, index=ind, dim=1)
        reconstructed_vel = torch.gather(next_velocity_candidate, index=ind, dim=1)

        ind = best_action.reshape(B, 1, N).clone()
        mask = ind == -1
        ind[mask] = self.default_action  # Workaround the gather can't handle -1
        reconstructed_heading = torch.gather(next_heading_candidate, index=ind, dim=1)

        valid_mask = current_valid_mask & next_valid_mask
        reconstructed_vel[~valid_mask] = 0
        reconstructed_pos[~valid_mask] = 0
        reconstructed_heading[~valid_mask] = 0
        assert current_pos.shape == reconstructed_pos.shape

        if invalid_next_pos is not None:
            reconstructed_heading[invalid_next_pos] = current_heading[invalid_next_pos]

        if no_displacement_mask is not None:
            reconstructed_heading[no_displacement_mask] = current_heading[no_displacement_mask]

        best_action = best_action.reshape(B, 1, N)
        best_action[~valid_mask] = -1
        reconstructed_pos[~valid_mask] = 0
        reconstructed_vel[~valid_mask] = 0
        reconstructed_heading[~valid_mask] = 0

        assert (best_action[valid_mask] >= 0).all()
        assert (best_action[~valid_mask] == -1).all()

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[mask] = 0

        # AID = 0
        # masked_best_action = best_action * valid_mask
        # delta = torch.gather(delta_grid_exp, index=masked_best_action.reshape(B, 1, N), dim=1)
        # acc = torch.gather(a_grid_exp, index=masked_best_action.reshape(B, 1, N), dim=1)
        # print(
        #     f"[TOK] AID{AID} CUR POS: {current_pos[0, 0, AID].cpu().numpy()}, "
        #     f"RECON POS: {reconstructed_pos[0, 0, AID].cpu().numpy()}, "
        #     f"GT POS: {next_pos[0, 0, AID].cpu().numpy()}, "
        #     f"Action: {best_action[0, 0, AID].cpu().numpy()}, "
        #     f"ACC: {acc[0, 0, AID].cpu().numpy()}, "
        #     f"STEER: {delta[0, 0, AID].cpu().numpy()}, "
        #     f"CUR VEL: {current_vel[0, 0, AID].norm(dim=-1).cpu().numpy()}, "
        #     f"RECON VEL: {reconstructed_vel[0, 0, AID].norm(dim=-1).cpu().numpy()}, "
        #     f"VALID: {valid_mask[0, 0, AID].cpu().numpy()}"
        # )

        return dict(
            action=best_action,
            pos=reconstructed_pos,
            heading=reconstructed_heading,
            vel=reconstructed_vel,
            mask=valid_mask,
            delta_pos=relative_delta_pos
        )

    def _detokenize_backward_prediction(
        self,
        data_dict,
        interpolation=True,
        detokenizing_gt=False,
        flip_wrong_heading=False,
        teacher_forcing=False,
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
            valid = valid.reshape(B, -1, 1, N).expand(-1, -1, self.num_skipped_steps, -1).reshape(B, -1, N)
            # valid = torch.cat([valid, input_mask[:, -1:]], dim=1)

            if teacher_forcing:

                B, T, N = valid.shape
                # ====== insert True at newly_added token ======
                
                # Find the first True index for each agent in each batch
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

            else:
                new_valid = torch.cat([input_mask[:, 0:1], valid], dim=1)

            # =============================
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

        if teacher_forcing:
            T_start = 0
        else:
            T_start = 5

        data_dict["decoder/reconstructed_position"] = reconstructed_pos[:,T_start:]
        data_dict["decoder/reconstructed_heading"] = reconstructed_heading[:,T_start:]
        data_dict["decoder/reconstructed_velocity"] = reconstructed_vel[:,T_start:]
        data_dict["decoder/reconstructed_valid_mask"] = reconstructed_valid_mask[:,T_start:]

        assert data_dict["decoder/reconstructed_position"].shape[1] == 91, f"be careful that reconstructed backward trajectory has T={data_dict['decoder/reconstructed_position'].shape[1]}"

        return data_dict

    def _detokenize_a_step_backward(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, action, agent_shape, dt, **kwargs
    ):
        assert dt > 0

        assert action.ndim == 3
        B, T_action, N = action.shape
        assert T_action == 1
        # Retrieve acceleration and steering angle based on decoded bins
        if self.acceleration_bins.device != action.device:
            self.acceleration_bins = self.acceleration_bins.to(action.device)
            self.steering_bins = self.steering_bins.to(action.device)

        action_expanded = action.reshape(B, T_action, N, 1).expand(B, T_action, N, 1).clone()
        mask = (action_expanded == -1) | (action_expanded == START_ACTION) | (action_expanded == END_ACTION)
        action_expanded[mask] = 0

        acceleration_bins = self.acceleration_bins.reshape(1, 1, 1, -1).expand(B, T_action, N, -1)
        steering_bins = self.steering_bins.reshape(1, 1, 1, -1).expand(B, T_action, N, -1)

        # Decode the action into acceleration and steering angle bins
        best_a_idx = action_expanded // self.num_bins
        best_delta_idx = action_expanded % self.num_bins

        best_acceleration = torch.gather(acceleration_bins, index=best_a_idx, axis=3).squeeze(-1)
        best_steering = torch.gather(steering_bins, index=best_delta_idx, axis=3).squeeze(-1)

        # Next speed:
        next_heading_exp = current_heading
        next_pos = current_pos
        next_vel = current_vel
        next_speed = next_vel.norm(dim=-1)
        current_speed_candidate = next_speed - best_acceleration * dt
        average_speed = (current_speed_candidate + next_speed) / 2

        # wheel_base = agent_shape[..., 0].reshape(B, 1, N)
        # yaw_rate = (average_speed / wheel_base) * torch.tan(best_steering)
        yaw_rate = best_steering
        delta_theta = yaw_rate * dt
        current_heading_candidate = utils.wrap_to_pi(next_heading_exp - delta_theta)
        average_heading = utils.wrap_to_pi(utils.average_heading(current_heading_candidate, next_heading_exp))

        reconstructed_vel = rotate(
            current_speed_candidate, torch.zeros_like(current_speed_candidate), angle=current_heading_candidate
        )
        average_velocity = rotate(average_speed, torch.zeros_like(average_speed), angle=average_heading)

        reconstructed_pos = next_pos - average_velocity * dt
        reconstructed_heading = current_heading_candidate

        # Masking
        valid_mask = current_valid_mask.reshape(B, 1, N, 1).expand(B, 1, N, 2)
        reconstructed_pos[~valid_mask] = 0
        reconstructed_vel[~valid_mask] = 0
        reconstructed_heading[~valid_mask[..., 0]] = 0

        # AID = 0
        # print(
        #     f"[DETOK] AID{AID} PRED POS: {reconstructed_pos[0,0,AID].cpu().numpy()}, "
        #     f"PRED HEAD: {reconstructed_heading[0,0,AID]}, "
        #     f"PRED VEL: {reconstructed_vel[0,0,AID].norm(dim=-1).cpu().numpy()}, "
        #     f"SPEED: {next_speed[0,0,AID]:.4f}, "
        #     f"NEXT POS: {current_pos[0,0,AID].cpu().numpy()}, "
        #     f"NEXT HEAD: {current_heading[0,0,AID]:.4f}, "
        #     f"VALID: {valid_mask[0,0,AID].cpu().numpy()}"
        #     f"ACTION: {action[0,0,AID].cpu().numpy()}, "
        # )

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[~valid_mask] = 0

        return dict(
            pos=reconstructed_pos, heading=reconstructed_heading, vel=reconstructed_vel, delta_pos=relative_delta_pos
        )

    def _detokenize_a_step(
        self, *, current_pos, current_heading, current_valid_mask, current_vel, action, agent_shape, dt, **kwargs
    ):
        if dt < 0:
            # TODO: This is a trick to handle the backward prediction. We flip current/next states and dt.
            #  Might cause confusion to other users.
            return self._detokenize_a_step_backward(
                current_pos=current_pos,
                current_heading=current_heading,
                current_valid_mask=current_valid_mask,
                current_vel=current_vel,
                action=action,
                agent_shape=agent_shape,
                dt=-dt,
                **kwargs
            )

        assert action.ndim == 3
        B, T_action, N = action.shape
        assert T_action == 1
        # Retrieve acceleration and steering angle based on decoded bins
        if self.acceleration_bins.device != action.device:
            self.acceleration_bins = self.acceleration_bins.to(action.device)
            self.steering_bins = self.steering_bins.to(action.device)

        action_expanded = action.reshape(B, T_action, N, 1).expand(B, T_action, N, 1).clone()
        mask = (action_expanded == -1) | (action_expanded == START_ACTION)
        action_expanded[mask] = 0

        acceleration_bins = self.acceleration_bins.reshape(1, 1, 1, -1).expand(B, T_action, N, -1)
        steering_bins = self.steering_bins.reshape(1, 1, 1, -1).expand(B, T_action, N, -1)

        # Decode the action into acceleration and steering angle bins
        best_a_idx = action_expanded // self.num_bins
        best_delta_idx = action_expanded % self.num_bins

        best_acceleration = torch.gather(acceleration_bins, index=best_a_idx, axis=3).squeeze(-1)
        best_steering = torch.gather(steering_bins, index=best_delta_idx, axis=3).squeeze(-1)

        # Update velocity components
        current_speed = current_vel.norm(dim=-1)  # (B, N)
        next_speed = current_speed + best_acceleration.reshape_as(current_speed) * self.dt
        average_speed = (current_speed + next_speed) / 2

        # Compute yaw rate and resulting change in heading
        # wheelbase = agent_shape[..., 0].reshape(B, 1, N)  # shape = Length, Width, Height
        # yaw_rate = (average_speed / wheelbase) * torch.tan(best_steering.squeeze(-1))
        yaw_rate = best_steering.reshape_as(current_heading)
        delta_theta = yaw_rate * self.dt
        next_heading = utils.wrap_to_pi(current_heading + delta_theta)
        reconstructed_heading = next_heading.reshape(B, 1, N)
        average_heading = utils.wrap_to_pi(utils.average_heading(current_heading, next_heading))

        average_velocity = rotate(average_speed, torch.zeros_like(average_speed), angle=average_heading)
        next_velocity = rotate(next_speed, torch.zeros_like(next_speed), angle=next_heading)
        reconstructed_vel = next_velocity.reshape(B, 1, N, 2)

        next_pos = current_pos + average_velocity * self.dt
        reconstructed_pos = next_pos.reshape(B, 1, N, 2)

        # Masking
        valid_mask = current_valid_mask.reshape(B, 1, N, 1).expand(B, 1, N, 2)
        reconstructed_pos[~valid_mask] = 0
        reconstructed_vel[~valid_mask] = 0
        reconstructed_heading[~valid_mask[..., 0]] = 0

        relative_delta_pos = get_relative_velocity(reconstructed_vel, reconstructed_heading)
        relative_delta_pos[~valid_mask] = 0

        return dict(
            pos=reconstructed_pos, heading=reconstructed_heading, vel=reconstructed_vel, delta_pos=relative_delta_pos
        )
