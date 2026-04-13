import copy
from dataclasses import dataclass
from typing import Union, Dict

import numpy as np
import torch


@dataclass
class Tokens:
    ids: Union[torch.Tensor, np.ndarray]
    mask: Union[torch.Tensor, np.ndarray]

    # Say now we are creating the causal mask for the n-th object: cm[n-1]. What should be the correct value?
    # In most cases the causal mask offset should be n.
    # That is the n-th rows of the causal mask looks like: causal_mask[n-1] = [1,...,1, 0,...,0] where
    # causal_mask[n-1][:n] = 1
    # This is the very classic causal mask in the decoder of the transformer.
    # However, in some cases, we might have some objects that are updated at the same time. In this case, we
    # don't want to create the dependency between the objects, aka the later object are dependent to the state
    # of the object that is updated earlier. So we need to set the causal mask offset to be the length of the
    # sequence. In this cause the casual_mask_offset[:n] = n, where n is the number of objects.
    causal_mask_offset: Union[torch.Tensor, np.ndarray]
    length: int

    @classmethod
    def create(cls, ids, mask, causal_mask_offset, length=None, use_numpy=None, device=None):
        if isinstance(ids, np.ndarray):
            use_numpy = True
        elif isinstance(ids, torch.Tensor):
            use_numpy = False
        elif use_numpy is None:
            raise ValueError("use_numpy must be specified when the ids is not a numpy array or a torch tensor.")

        if use_numpy:
            ids: np.ndarray = np.asarray(ids, dtype=int)
            mask: np.ndarray = np.asarray(mask, dtype=bool)
            causal_mask_offset: np.ndarray = np.asarray(causal_mask_offset, dtype=int)
        else:
            ids: torch.Tensor = torch.as_tensor(ids, device=device).int()
            mask: torch.Tensor = torch.as_tensor(mask, device=device).bool()
            causal_mask_offset: torch.Tensor = torch.as_tensor(causal_mask_offset, device=device).int()
        if length is None:
            assert ids.ndim == 1, ids.shape
            length = len(ids)
        causal_mask_offset[causal_mask_offset == -1] = length
        assert causal_mask_offset.min() > 0, causal_mask_offset.min()
        assert ids.shape == mask.shape == causal_mask_offset.shape, (ids.shape, mask.shape, causal_mask_offset.shape)
        assert causal_mask_offset.max() <= length, (causal_mask_offset.max(), length)

        return cls(ids=ids, mask=mask, causal_mask_offset=causal_mask_offset, length=length)

    @classmethod
    def concatenate(cls, group_list, axis=-1):
        if isinstance(group_list[0].ids, torch.Tensor):
            cat = torch.cat
        else:
            cat = np.concatenate
        data = cat([t.ids for t in group_list], axis=axis)
        mask = cat([t.mask for t in group_list], axis=axis)

        current_causal_mask_offset = 0
        causal_mask_offsets = []
        for t in group_list:
            causal_mask_offsets.append(t.causal_mask_offset + current_causal_mask_offset)
            current_causal_mask_offset += len(t)
        causal_mask_offset = cat(causal_mask_offsets, axis=axis)

        length = sum(len(t) for t in group_list)

        return cls.create(data, mask, causal_mask_offset, length=length)

    def __len__(self):
        return self.length

    def to_tensor(self, batch_size, device):
        assert self.ids.ndim == 1
        if isinstance(self.ids, np.ndarray):
            self.ids = torch.as_tensor(self.ids, device=device).unsqueeze(0)
            self.mask = torch.as_tensor(self.mask, device=device).unsqueeze(0)
            self.causal_mask_offset = torch.as_tensor(self.causal_mask_offset, device=device).unsqueeze(0)
        return Tokens.create(
            self.ids.repeat(batch_size, 1),
            self.mask.repeat(batch_size, 1),
            self.causal_mask_offset.repeat(batch_size, 1),
            self.length,
            use_numpy=False
        )

    def unbatch(self):
        assert self.ids.ndim == 2
        return [
            Tokens.create(self.ids[i], self.mask[i], self.causal_mask_offset[i], self.length, use_numpy=False)
            for i in range(len(self.ids))
        ]

    @staticmethod
    def block_causal_mask_offset(number_of_objects, batch_size, device):
        return torch.empty((batch_size, number_of_objects), dtype=int, device=device).fill_(number_of_objects)

    # def add_step(self, step: int):
    #     if isinstance(self.ids, torch.Tensor):
    #         self.step = torch.zeros_like(self.ids).fill_(step)
    #     else:
    #         self.step = step
    #     return self


def translate_id(ids, min, max, allow_invalid=False, reverse=False):
    if reverse:
        ids = ids - min
        return ids

    assert isinstance(min, int)
    assert isinstance(max, int)

    if isinstance(ids, int):
        if allow_invalid:
            if ids == -1:
                return -1
            else:
                pass
        else:
            ids = -min - 1
        ids = ids + min
        assert ids < max
        assert ids >= min
        return ids

    else:
        assert isinstance(ids, (np.ndarray, torch.Tensor))
        if allow_invalid:
            ids = copy.deepcopy(ids)
            ids[ids == -1] = -min - 1
        else:
            ids[ids == -1] = max - min - 1  # Set it to the maximum value
        ids = ids + min
        if allow_invalid:
            max_val = ids[ids != -1].max()
            min_val = ids[ids != -1].min()
        else:
            max_val = ids.max()
            min_val = ids.min()
        assert max_val < max
        assert min_val >= min
        return ids


def in_range(ids, min, max):
    if isinstance(ids, torch.Tensor):
        ret1 = ids >= min
        ret2 = ids < max
        return torch.logical_and(ret1, ret2)
    elif isinstance(ids, np.ndarray):
        ret1 = ids >= min
        ret2 = ids < max
        return np.logical_and(ret1, ret2)
    else:
        return min <= ids < max


class InfgenTokenizer(GenTokenizer):
    @classmethod
    def get_add_start_tokens(cls):
        """
        [ADD_START, ]
        """
        return Tokens.create([cls.ADD_START], [True], [1], use_numpy=True)

    @classmethod
    def get_add_end_tokens(cls):
        """
        [ADD_END, ]
        """
        return Tokens.create([cls.ADD_END], [True], [1], use_numpy=True)

    @classmethod
    def get_remove_start_tokens(cls):
        """
        [REMOVE_START, ]
        """
        return Tokens.create([cls.REMOVE_START], [True], [1], use_numpy=True)

    @classmethod
    def get_remove_end_tokens(cls):
        """
        [REMOVE_END, ]
        """
        return Tokens.create([cls.REMOVE_END], [True], [1], use_numpy=True)

    @classmethod
    def get_add_operation(cls, agent_id, valid_mask, last_valid_mask, config):
        if last_valid_mask is None:
            # All valid objects should be added
            new_agent_id = agent_id[valid_mask]
            num_new_agents = len(new_agent_id)
        else:
            # Only the newly added objects should be added
            new_agent_id = agent_id[valid_mask & ~last_valid_mask]
            num_new_agents = len(new_agent_id)
        if num_new_agents == 0:
            return None, None, None
        agent_tokens = Tokens.create(
            cls.get_agent_id(new_agent_id, config), [True] * num_new_agents, list(range(1, 1 + num_new_agents))
        )
        # TODO: Each agent token should follow a map token.
        # TODO: They should be interleaved.
        tokens = Tokens.concatenate([cls.get_add_start_tokens(), agent_tokens, cls.get_add_end_tokens()])

        # when see ADD_START, you should start making prediction. You should also predict whether ADD_END.
        # when you saw ADD_END, you must predict UPDATE_START.
        should_predict = [True] + [True] * num_new_agents + [True]
        is_gt = [True] + [True] * num_new_agents + [True]

        return tokens, should_predict, is_gt

    @classmethod
    def get_remove_operation(cls, agent_id, valid_mask, last_valid_mask, next_valid_mask, config):
        if last_valid_mask is None:
            return None, None, None
        removed_agent_id = agent_id[last_valid_mask & ~valid_mask]
        num_removed_agents = len(removed_agent_id)
        if num_removed_agents == 0:
            return None, None, None
        agent_tokens = Tokens.create(
            cls.get_agent_id(removed_agent_id, config), [True] * num_removed_agents,
            list(range(1, 1 + num_removed_agents))
        )
        tokens = Tokens.concatenate([cls.get_remove_start_tokens(), agent_tokens, cls.get_remove_end_tokens()])

        # when you see REMOVE_START, you should start making prediction. You should also predict whether REMOVE_END.
        should_predict = [True] + [True] * num_removed_agents + [True]
        is_gt = [True] + [True] * num_removed_agents + [True]
        return tokens, should_predict, is_gt

    @classmethod
    def get_token_names(cls, tokens: Tokens, config: Dict):
        def _get_names_for_a_sequence(tokens):
            out = []
            if isinstance(tokens, Tokens):
                data = tokens.ids
                mask = tokens.mask
            else:
                data = tokens
                mask = None
            for i in range(len(tokens)):
                if mask is not None and bool(mask[i]) is False:
                    out.append("INVALID")
                    continue
                if data[i] == cls.STEP_START:
                    out.append("STEP_START")
                elif data[i] == cls.STEP_END:
                    out.append("STEP_END")
                elif data[i] == cls.UPDATE_START:
                    out.append("UPDATE_START")
                elif data[i] == cls.UPDATE_END:
                    out.append("UPDATE_END")
                elif data[i] == cls.ADD_START:
                    out.append("ADD_START")
                elif data[i] == cls.ADD_END:
                    out.append("ADD_END")
                elif data[i] == cls.REMOVE_START:
                    out.append("REMOVE_START")
                elif data[i] == cls.REMOVE_END:
                    out.append("REMOVE_END")
                elif cls.is_agent_tokens(data[i], config):
                    out.append("AGENT_{}".format(cls.get_agent_id(data[i], config, reverse=True)))
                elif cls.is_action_tokens(data[i], config):
                    out.append("ACTION_{}".format(cls.get_action_id(data[i], config, reverse=True)))
                elif data[i] == -1:
                    out.append("INVALID")
                else:
                    raise ValueError()
            return out

        if isinstance(tokens, Tokens):
            data = tokens.ids
        else:
            data = tokens

        if data.ndim == 1:
            return _get_names_for_a_sequence(tokens)

        elif data.ndim == 2:
            if isinstance(tokens, Tokens):
                tokens = tokens.unbatch()
            else:
                tokens = list(tokens)
            return [_get_names_for_a_sequence(t) for t in tokens]

        else:
            raise ValueError()
