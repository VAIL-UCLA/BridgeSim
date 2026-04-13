import torch

from bmt.models.motionlm import nucleus_sampling
from bmt.tokenization.gen_tokenizers import Tokens, InfgenTokenizer


def sample_action(logits, config):
    sampling_method = config.SAMPLING.SAMPLING_METHOD
    temperature = config.SAMPLING.TEMPERATURE
    topp = config.SAMPLING.TOPP

    if sampling_method == "argmax":
        selected_action = logits.argmax(-1)
    elif sampling_method == "softmax":
        selected_action = torch.distributions.Categorical(logits=logits / temperature).sample()
    elif sampling_method == "topp":
        selected_action = nucleus_sampling(logits=logits / temperature, p=topp)
    else:
        raise ValueError("Unknown sampling method: {}".format(sampling_method))

    return selected_action


def mask_out_invalid_actions(logits, valid_min=None, valid_max=None, valids=None):
    mask = logits.new_ones(logits.shape).bool()  # 1: to be filled, 0: good to go

    if valids is not None:
        for v in valids:
            mask[..., v].fill_(0)

    if valid_min is not None:
        assert valid_max is not None
        mask[..., valid_min:valid_max].fill_(0)

    logits = logits.masked_fill(mask, -1e9)
    return logits


class StateMachine:
    def __init__(
        self, *, state, init_tokens, init_valid_mask, causal_mask_offset, map_ids, agent_ids, config, batch_id, step=0
    ):
        self.state = state

        end_index = int(init_valid_mask.sum(-1))
        assert init_valid_mask[:end_index].all().item() is True
        assert init_valid_mask[end_index:].any().item() is False
        self.tokens = Tokens.create(
            ids=init_tokens[:end_index],
            mask=init_valid_mask[:end_index],
            causal_mask_offset=causal_mask_offset[:end_index],
            length=end_index,
            use_numpy=False,
        )

        self.start_index = 0
        self.end_index = end_index
        self.config = config
        self.map_ids = set(map_ids)
        self.agent_ids = set(agent_ids)
        self.batch_id = batch_id
        self.step = step
        self.intra_step_start = 0
        self.intra_step_end = end_index

    def update(self, model_output):
        """
        According to current state, read the model's output (do some indexing / slicing etc.), and update the state.
        Return the parsed new tokens.
        """
        batch_id = self.batch_id
        assert model_output.ndim == 2

        if self.state == InfgenTokenizer.UPDATE_START:
            # Read N actions, set state to UPDATE_END
            return self.process_UPDATE_START(model_output)

        if self.state == InfgenTokenizer.UPDATE_END:
            # Read 1 action, set state to REMOVE_START or STEP_END according to the input
            return self.process_UPDATE_END(model_output)

        elif self.state == InfgenTokenizer.REMOVE_START:
            # Read 1 agent_id or REMOVE_END
            return self.process_REMOVE_START(model_output)

        elif self.state == InfgenTokenizer.STEP_START:
            # Read 1 action, set state to UPDATE_START or ADD_START according to the input
            return self.process_STEP_START(model_output)

        else:
            raise ValueError(f"Invalid state: {self.state}")

    def process_UPDATE_START(self, model_output):
        """Read N actions, set state to UPDATE_END."""
        out = model_output[self.start_index:self.end_index]

        action_id_min, action_id_max = InfgenTokenizer.get_action_id_range(self.config)
        out = mask_out_invalid_actions(out, valid_min=action_id_min, valid_max=action_id_max)

        action = sample_action(out, self.config)

        length = action.shape[0]
        action_tokens = Tokens.create(
            ids=action,
            mask=self.tokens.mask[self.start_index:self.end_index],
            causal_mask_offset=self.tokens.causal_mask_offset.new_ones(length) * length,
            length=length,
            use_numpy=False
        )
        new_tokens = Tokens.concatenate(
            [action_tokens, InfgenTokenizer.get_update_end_tokens(use_numpy=False, device=out.device)]
        )
        self.tokens = Tokens.concatenate([self.tokens, new_tokens])

        self.state = InfgenTokenizer.UPDATE_END
        print(f"Scenario {self.batch_id} change state from UPDATE_START to UPDATE_END")

        self.start_index += new_tokens.length
        self.end_index = self.start_index + 1  # Looking for REMOVE_START or STEP_END

        self.intra_step_start = self.intra_step_end
        self.intra_step_end += new_tokens.length

        return {
            "tokens": new_tokens,
            "step": self.tokens.causal_mask_offset.new_ones(new_tokens.length) * self.step,
            "intra_step": torch.arange(self.intra_step_start, self.intra_step_end).to(out.device)
        }

    def process_UPDATE_END(self, model_output):
        """Read 1 action, set state to REMOVE_START or STEP_END according to the input."""
        out = model_output[self.start_index:self.end_index]

        out = mask_out_invalid_actions(out, valids=[InfgenTokenizer.STEP_END, InfgenTokenizer.REMOVE_START])

        action = sample_action(out, self.config)

        a = action.item()

        # TODO ===== Fix in future
        if a == InfgenTokenizer.REMOVE_START:
            a = InfgenTokenizer.STEP_END
        # TODO ===== Fix in future

        if a == InfgenTokenizer.REMOVE_START:
            # TODO double check
            self.state = InfgenTokenizer.REMOVE_START
            print(f"Scenario {self.batch_id} change state from UPDATE_END to REMOVE_START")
            self.start_index += 1
            self.end_index = self.start_index + 1  # Looking for AGENT_ID to be removed
            out = action

        elif a == InfgenTokenizer.STEP_END:
            out = torch.cat([action, torch.as_tensor([InfgenTokenizer.STEP_START], dtype=out.dtype, device=out.device)])

            self.state = InfgenTokenizer.STEP_START
            print(f"Scenario {self.batch_id} change state from UPDATE_END to STEP_START")

            self.start_index += out.shape[0]
            self.end_index = self.start_index + 1  # Looking for REMOVE_START or STEP_END
            out = action

        else:
            raise ValueError(f"Invalid action: {a}")

        # TODO: Make tokens here
        return out

    def process_REMOVE_START(self, model_output):
        # Read 1 agent_id or REMOVE_END
        out = model_output[self.start_index:self.end_index]

        # Options: REMOVE_END, AGENT_ID (that are valid now)
        out = mask_out_invalid_actions(out, valids=[InfgenTokenizer.REMOVE_END] + list(self.agent_ids))

        action = sample_action(out, self.config)

        if action == InfgenTokenizer.REMOVE_END:
            # TODO
            # Automatically add REMOVE_END, STEP_END, STEP_START
            self.start_index += 1
            self.end_index = self.start_index + 3
            pass
        else:
            assert action in self.agent_ids
            self.agent_ids.remove(action)

            # no need to change state.
            self.start_index += 1
            self.end_index = self.start_index + 1

        # TODO: Make tokens
        return None

    def process_STEP_START(self, model_output):
        out = model_output[self.start_index:self.end_index]

        out = mask_out_invalid_actions(out, valids=[InfgenTokenizer.UPDATE_START, InfgenTokenizer.ADD_START])

        action = sample_action(out, self.config)

        a = action.item()
        if a == InfgenTokenizer.UPDATE_START:
            # Add all existing agent_id to the tokens.

            self.state = InfgenTokenizer.UPDATE_START
            print(f"Scenario {self.batch_id} change state from STEP_START to UPDATE_START")

            out = torch.cat([action, self.agent_ids])

            self.start_index += 1
            self.end_index = self.start_index + 1  # Looking for AGENT_ID to be removed
            out = action

        elif a == InfgenTokenizer.STEP_END:
            out = torch.cat([action, torch.as_tensor([InfgenTokenizer.STEP_START], out.dtype, out.device)])

            self.state = InfgenTokenizer.STEP_START
            print(f"Scenario {self.batch_id} change state from STEP_START to STEP_START")
            self.start_index += out.shape[0]
            self.end_index = self.start_index + 1  # Looking for REMOVE_START or STEP_END
            out = action


class ARRollout:
    """
    This class helps organize the rollout of the autoregressive model.
    """
    def __init__(self, init_tokens, init_valid_mask, causal_mask_offset, map_ids, agent_ids, config):
        self.B = init_tokens.shape[0]
        self.config = config
        self.states = [
            StateMachine(
                state=InfgenTokenizer.UPDATE_START,
                init_tokens=init_tokens[i],
                init_valid_mask=init_valid_mask[i],
                causal_mask_offset=causal_mask_offset[i],
                map_ids=map_ids[i],
                agent_ids=agent_ids[i],
                config=config,
                batch_id=i
            ) for i in range(self.B)
        ]

    def get_tokens(self):
        """
        We truncate the tokens for each scenario i to the range start_indices[i] to end_indices[i],
        stack them to for a batched tokens, and apply padding.
        """
        tokens = [s.tokens for s in self.states]
        max_len = max([t.length for t in tokens])

        # padding all masks
        for t in tokens:
            t.mask = torch.nn.functional.pad(t.mask, (0, max_len - t.length), value=0)

        # padding all causal_mask_offset
        for t in tokens:
            t.causal_mask_offset = torch.nn.functional.pad(t.causal_mask_offset, (0, max_len - t.length), value=-1)

        # padding all ids
        for t in tokens:
            t.ids = torch.nn.functional.pad(t.ids, (0, max_len - t.length), value=-1)

        # Stack
        out = Tokens.create(
            ids=torch.stack([t.ids for t in tokens], dim=0),
            mask=torch.stack([t.mask for t in tokens], dim=0),
            causal_mask_offset=torch.stack([t.causal_mask_offset for t in tokens], dim=0),
            length=max_len,
            use_numpy=False
        )

        return out

    def update(self, logits):
        """
        Let's say in the get_tokens function, you get a batch of tokens each scenario has valid tokens:
        [
            end_indices[0] - start_indices[0]
            ...
            end_indices[i] - start_indices[i]
            ...
            end_indices[B-1] - start_indices[B-1]
        ]. The maximum number of tokens is L.
        You call the model, which will return you something in shape (B, L, D) - before sampling or
        (B, L) -after sampling.
        Now, how we append the new tokens to the old tokens?
        There is a significant challenge that the number of new tokens to be added might vary in different scenarios.
        Scenario A might be updating the states so there are N new tokens.
        Scenario B might be adding new object so there should only has 1 new token.
        We will handover the right to determine how many tokens should be appended to the external function.
        And faithfully append the tokens to the old tokens.
        """
        assert logits.shape[0] == self.B

        output = []
        step = []
        intra_step = []
        for b, logits_per_scenario in enumerate(logits):

            out = self.states[b].update(logits_per_scenario)
            output.append(out["tokens"])
            step.append(out["step"])
            intra_step.append(out["intra_step"])

        max_len = max([t.length for t in output])

        # print(1111)
        return output
