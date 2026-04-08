import numpy as np
import torch

# from torch.nn.modules.transformer import TransformerEncoderLayer as NativeTransformerEncoderLayer
from bmt.dataset import constants
from bmt.models.layers import position_encoding_utils
from bmt.utils import rotate, utils

# def pairwise_mask(mask):
#     """
#     input mask is in shape (B, N), we need to prepare a pairwise mask in shape (B, N, N).
#     It's not correct to naively expand the mask. We need to maintain the symmetry of the mask.
#     """
#     B, N = mask.shape
#     mask = mask.unsqueeze(1).expand(B, N, N)
#     mask = mask & mask.transpose(1, 2)
#     return mask


def pairwise_mask(mask_a, mask_b):
    assert mask_a.ndim == mask_b.ndim == 2
    mask_a = mask_a.unsqueeze(-1)
    mask_b = mask_b.unsqueeze(-2)
    mask = torch.logical_and(mask_a, mask_b)
    return mask


def pairwise_relative_diff(positions_a, positions_b):
    """
    Compute pairwise relative diffs for a batch of objects.
    For the ouput [b, i, j, :], it means the relative differences of [b, j] - [b, i],
    which is the pos of j in i's coordinate system.

    Parameters:
    - positions: A PyTorch tensor of shape (B, N, 2)

    Returns:
    - A PyTorch tensor of shape (B, N, N, 2) containing pairwise relative positions.
    """
    assert positions_a.ndim == positions_b.ndim
    # Expand dimensions to get tensors of shapes (B, N, 1, ...) and (B, 1, N, ...)
    positions_expanded_a = positions_a.unsqueeze(2)  # Shape: (B, N, 1, ...)
    positions_expanded_b = positions_b.unsqueeze(1)  # Shape: (B, 1, N, ...)

    # Compute the pairwise relative positions by subtraction
    relative_positions = positions_expanded_b - positions_expanded_a  # Shape: (B, N, N, ...)

    return relative_positions


def compute_relation(
    query_pos,
    query_heading,
    query_valid_mask,
    key_pos,
    key_heading,
    key_valid_mask,
    hidden_dim,
    causal_valid_mask,
    knn=128,
    max_distance=None,
    gather=True,
    return_pe=True,
    query_step=None,
    query_vel=None,
    key_step=None,
    key_vel=None,
    include_contour=False,
    query_width=None,
    query_length=None,
    key_width=None,
    key_length=None,
    non_agent_relation=False,
):
    """
    Compute the relation encoding for the transformer encoder.
    """
    assert max_distance is None, "Not implemented"
    assert query_pos.ndim == key_pos.ndim == 3
    assert query_heading.ndim == key_heading.ndim == 2
    assert query_valid_mask.ndim == key_valid_mask.ndim == 2

    pairwise_heading = pairwise_relative_diff(query_heading, key_heading)

    heading_fill_0_mask = pairwise_mask(
        query_heading == constants.HEADING_PLACEHOLDER, key_heading == constants.HEADING_PLACEHOLDER
    )
    pairwise_heading[heading_fill_0_mask] = 0

    rel_pos = pairwise_relative_diff(query_pos[..., :2], key_pos[..., :2])

    rel_vel = None
    if query_vel is not None:
        assert key_vel is not None
        rel_vel = pairwise_relative_diff(query_vel, key_vel)

    B, Q = query_heading.shape
    K = key_heading.shape[1]

    # i's local coordinate's y-axis (the heading) in the global coordinate
    i_local_y_wrt_global = query_heading.reshape(B, Q, 1).expand(B, Q, K)
    i_local_x_wrt_global = i_local_y_wrt_global - np.pi / 2
    rotated_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x_wrt_global)

    if rel_vel is not None:
        rotated_vel = rotate(rel_vel[..., 0], rel_vel[..., 1], angle=-i_local_x_wrt_global)

    valid_mask = pairwise_mask(query_valid_mask, key_valid_mask)

    if include_contour:
        contour_q = utils.cal_polygon_contour_torch(
            x=query_pos[..., 0], y=query_pos[..., 1], theta=query_heading, width=query_width, length=query_length
        )
        contour_k = utils.cal_polygon_contour_torch(
            x=key_pos[..., 0], y=key_pos[..., 1], theta=key_heading, width=key_width, length=key_length
        )
        contour_diff = pairwise_relative_diff(contour_q, contour_k)
        contour_diff = rotate(
            contour_diff[..., 0], contour_diff[..., 1], angle=-i_local_x_wrt_global.unsqueeze(-1).expand(-1, -1, -1, 4)
        )

    # THRESHOLD = 100
    dist = rel_pos.norm(dim=-1)
    if causal_valid_mask is not None:
        if causal_valid_mask.ndim == 2:
            # the causal mask is not batched
            causal_valid_mask = causal_valid_mask.unsqueeze(0).expand(B, -1, -1)
            dist = dist.masked_fill_(~causal_valid_mask, float("+inf"))
            valid_mask = valid_mask & causal_valid_mask

        else:
            raise ValueError()  # TODO

    # dist_mask = dist < THRESHOLD
    # rel_mask = torch.logical_and(mask, dist_mask)
    rel_mask = valid_mask

    if query_step is not None:
        step_diff = pairwise_relative_diff(query_step, key_step)

    indices = None
    if knn:
        dist = dist.masked_fill_(~valid_mask, float("+inf"))
        indices = dist.argsort(dim=-1)[..., :knn]

        if gather:
            rotated_pos = torch.gather(rotated_pos, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            if rel_vel is not None:
                rotated_vel = torch.gather(rotated_vel, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            pairwise_heading = torch.gather(pairwise_heading, dim=-1, index=indices)
            rel_mask = torch.gather(rel_mask, dim=-1, index=indices)
            if query_step is not None:
                step_diff = torch.gather(step_diff, dim=-1, index=indices)
            if include_contour:
                contour_diff = torch.gather(
                    contour_diff, dim=-3, index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 4, 2)
                )

        else:

            # Create a new mask with the same shape as rel_mask, initially set to False
            original_valid_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

            # Use advanced indexing to set True for indices selected by KNN
            batch_indices = torch.arange(B).view(B, 1, 1)
            query_indices = torch.arange(Q).view(1, Q, 1)
            assert original_valid_mask.shape[0] == B, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, B
            )
            assert original_valid_mask.shape[1] == Q, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, Q
            )
            assert original_valid_mask.shape[2] == K, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, K
            )
            original_valid_mask[batch_indices, query_indices, indices] = True

            # Update rel_mask to only include indices selected by KNN
            rel_mask = rel_mask & original_valid_mask

            # Just pass them:
            # rotated_pos
            # pairwise_heading
    if return_pe:
        pos_pe = position_encoding_utils.gen_sineembed_for_relation(
            rotated_pos[rel_mask], pairwise_heading[rel_mask], hidden_dim=hidden_dim
        )
        pos_pe = utils.unwrap(pos_pe, rel_mask)
        assert query_step is None
        assert query_vel is None
        assert not include_contour
        return pos_pe, rel_mask, indices
    else:
        distance = torch.norm(rotated_pos, p=2, dim=-1)
        ret = [rotated_pos, distance[..., None], pairwise_heading[..., None]]
        if query_step is not None:
            assert key_step is not None
            ret.append(step_diff[..., None])
        if query_vel is not None:
            ret.append(rotated_vel)
        if include_contour:
            ret.append(contour_diff.flatten(-2, -1))
        ret = torch.cat(ret, dim=-1)
        ret[~rel_mask] = 0
        return ret, rel_mask, indices


def compute_relation_simple_relation(
    *,
    query_pos,
    query_heading,
    query_valid_mask,
    key_pos,
    key_heading,
    key_valid_mask,
    hidden_dim,
    causal_valid_mask,
    knn=128,
    max_distance=None,
    gather=True,
    return_pe=True,
    query_step=None,
    query_vel=None,
    key_step=None,
    key_vel=None,
    include_contour=False,
    query_width=None,
    query_length=None,
    key_width=None,
    key_length=None,
    non_agent_relation=False,
    per_contour_point_relation=None,
):
    """
    Compute the relation encoding for the transformer encoder.
    """
    assert per_contour_point_relation is not None, "Not implemented"
    assert query_pos.ndim == key_pos.ndim == 3
    assert query_heading.ndim == key_heading.ndim == 2
    assert query_valid_mask.ndim == key_valid_mask.ndim == 2

    pairwise_heading = pairwise_relative_diff(query_heading, key_heading)

    heading_fill_0_mask = pairwise_mask(
        query_heading == constants.HEADING_PLACEHOLDER, key_heading == constants.HEADING_PLACEHOLDER
    )
    pairwise_heading[heading_fill_0_mask] = 0

    rel_pos = pairwise_relative_diff(query_pos[..., :2], key_pos[..., :2])

    rel_vel = None
    if query_vel is not None:
        raise ValueError()
        # assert key_vel is not None
        # rel_vel = pairwise_relative_diff(query_vel, key_vel)

    B, Q = query_heading.shape
    K = key_heading.shape[1]

    # i's local coordinate's y-axis (the heading) in the global coordinate
    i_local_y_wrt_global = query_heading.reshape(B, Q, 1).expand(B, Q, K)
    i_local_x_wrt_global = i_local_y_wrt_global - np.pi / 2

    rotated_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x_wrt_global)

    # if rel_vel is not None:
    #     rotated_vel = rotate(rel_vel[..., 0], rel_vel[..., 1], angle=-i_local_x_wrt_global)

    valid_mask = pairwise_mask(query_valid_mask, key_valid_mask)

    if not non_agent_relation:

        if per_contour_point_relation:
            contour_q_center = utils.cal_polygon_contour_torch(
                x=query_pos[..., 0],
                y=query_pos[..., 1],
                theta=query_heading,

                # Note that set width and length to zeros so that the contour is a point.
                # There is no need to compute per-contour-point relation.
                width=torch.zeros_like(query_pos[..., 0]),
                length=torch.zeros_like(query_pos[..., 0])
            )
            contour_k = utils.cal_polygon_contour_torch(
                x=key_pos[..., 0],
                y=key_pos[..., 1],
                theta=key_heading,
                width=key_width if key_width is not None else torch.zeros_like(key_pos[..., 0]),
                length=key_length if key_length is not None else torch.zeros_like(key_pos[..., 0])
            )
            contour_q = utils.cal_polygon_contour_torch(
                x=query_pos[..., 0],
                y=query_pos[..., 1],
                theta=query_heading,
                width=query_width if query_width is not None else torch.zeros_like(query_pos[..., 0]),
                length=query_length if query_length is not None else torch.zeros_like(query_pos[..., 0])
            )
            contour_k_center = utils.cal_polygon_contour_torch(
                x=key_pos[..., 0],
                y=key_pos[..., 1],
                theta=key_heading,
                width=torch.zeros_like(key_pos[..., 0]),
                length=torch.zeros_like(key_pos[..., 0])
            )
            contour_diff_in_q = pairwise_relative_diff(contour_q_center, contour_k)
            # contour_diff_in_q = rotate(
            #     contour_diff_in_q[..., 0],
            #     contour_diff_in_q[..., 1],
            #     angle=-i_local_x_wrt_global.unsqueeze(-1).expand(-1, -1, -1, 4)
            # )
            # contour_info = contour_diff_in_q.reshape(B, Q, K, 8)
            contour_diff_in_q_min = contour_diff_in_q.min(dim=-2).values
            contour_diff_in_q_max = contour_diff_in_q.max(dim=-2).values
            contour_diff_in_k = pairwise_relative_diff(contour_k_center, contour_q)

            # i's local coordinate's y-axis (the heading) in the global coordinate
            i_local_y_wrt_global_key = key_heading.reshape(B, K, 1).expand(B, K, Q)
            i_local_x_wrt_global_key = i_local_y_wrt_global_key - np.pi / 2
            contour_diff_in_k = rotate(
                contour_diff_in_k[..., 0],
                contour_diff_in_k[..., 1],
                angle=-i_local_x_wrt_global_key.unsqueeze(-1).expand(-1, -1, -1, 4)
            )
            contour_diff_in_k_min = contour_diff_in_k.min(dim=-2).values
            contour_diff_in_k_max = contour_diff_in_k.max(dim=-2).values
            contour_diff_in_k_min = contour_diff_in_k_min.permute(0, 2, 1, 3)
            contour_diff_in_k_max = contour_diff_in_k_max.permute(0, 2, 1, 3)

            contour_info = torch.cat(
                [contour_diff_in_q_min, contour_diff_in_q_max, contour_diff_in_k_min, contour_diff_in_k_max], dim=-1
            )

        else:
            contour_q_center = utils.cal_polygon_contour_torch(
                x=query_pos[..., 0],
                y=query_pos[..., 1],
                theta=query_heading,

                # Note that set width and length to zeros so that the contour is a point.
                # There is no need to compute per-contour-point relation.
                width=torch.zeros_like(query_pos[..., 0]),
                length=torch.zeros_like(query_pos[..., 0])
            )
            contour_k = utils.cal_polygon_contour_torch(
                x=key_pos[..., 0],
                y=key_pos[..., 1],
                theta=key_heading,
                width=key_width if key_width is not None else torch.zeros_like(key_pos[..., 0]),
                length=key_length if key_length is not None else torch.zeros_like(key_pos[..., 0])
            )
            contour_diff_in_q = pairwise_relative_diff(contour_q_center, contour_k)
            contour_diff_in_q = rotate(
                contour_diff_in_q[..., 0],
                contour_diff_in_q[..., 1],
                angle=-i_local_x_wrt_global.unsqueeze(-1).expand(-1, -1, -1, 4)
            )
            contour_info = contour_diff_in_q.reshape(B, Q, K, 8)

    # THRESHOLD = 100
    dist = rel_pos.norm(dim=-1)
    if causal_valid_mask is not None:
        if causal_valid_mask.ndim == 2:
            # the causal mask is not batched
            causal_valid_mask = causal_valid_mask.unsqueeze(0).expand(B, -1, -1)
            dist = dist.masked_fill_(~causal_valid_mask, float("+inf"))
            valid_mask = valid_mask & causal_valid_mask

        else:
            raise ValueError()  # TODO

    # dist_mask = dist < THRESHOLD
    # rel_mask = torch.logical_and(mask, dist_mask)

    if max_distance is not None:
        within_dist = dist < max_distance  # Shape (B, Q, K)
        # Allow at least 8 neighbors...
        closest = dist.argsort(dim=-1)[..., :8]
        # fill in True for these 8 neighbors
        within_dist[torch.arange(B).view(B, 1, 1), torch.arange(Q).view(1, Q, 1), closest] = True

        valid_mask = valid_mask & within_dist

    # rel_mask = valid_mask

    if not non_agent_relation and query_step is not None:
        step_diff = pairwise_relative_diff(query_step, key_step)
    else:
        step_diff = None

    indices = None
    if knn:
        dist = dist.masked_fill_(~valid_mask, float("+inf"))
        indices = dist.argsort(dim=-1)[..., :knn]

        if gather:
            rotated_pos = torch.gather(rotated_pos, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            # if rel_vel is not None:
            #     rotated_vel = torch.gather(rotated_vel, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 2))
            pairwise_heading = torch.gather(pairwise_heading, dim=-1, index=indices)
            valid_mask = torch.gather(valid_mask, dim=-1, index=indices)
            if query_step is not None:
                step_diff = torch.gather(step_diff, dim=-1, index=indices)
            # if include_contour:

            if not non_agent_relation:
                contour_info = torch.gather(contour_info, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 8))

        else:

            # Create a new mask with the same shape as rel_mask, initially set to False
            original_valid_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

            # Use advanced indexing to set True for indices selected by KNN
            batch_indices = torch.arange(B).view(B, 1, 1)
            query_indices = torch.arange(Q).view(1, Q, 1)
            assert original_valid_mask.shape[0] == B, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, B
            )
            assert original_valid_mask.shape[1] == Q, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, Q
            )
            assert original_valid_mask.shape[2] == K, (
                original_valid_mask.shape, indices.shape, valid_mask.shape, dist.shape, K
            )
            original_valid_mask[batch_indices, query_indices, indices] = True

            # Update rel_mask to only include indices selected by KNN
            valid_mask = valid_mask & original_valid_mask

            # Just pass them:
            # rotated_pos
            # pairwise_heading
    # if return_pe:
    #     pos_pe = position_encoding_utils.gen_sineembed_for_relation(
    #         rotated_pos[rel_mask], pairwise_heading[rel_mask], hidden_dim=hidden_dim
    #     )
    #     pos_pe = utils.unwrap(pos_pe, rel_mask)
    #     assert query_step is None
    #     assert query_vel is None
    #     assert not include_contour
    #     return pos_pe, rel_mask, indices
    # else:
    distance = torch.norm(rotated_pos, p=2, dim=-1)
    relative_direction = torch.atan2(rotated_pos[..., 1], rotated_pos[..., 0])
    ret = [relative_direction[..., None], distance[..., None], pairwise_heading[..., None]]
    # if query_step is not None:
    # assert key_step is not None

    if non_agent_relation:
        pass
    else:
        if step_diff is not None:
            ret.append(step_diff[..., None])

    # if query_vel is not None:
    #     ret.append(rotated_vel)
    # if include_contour:
    # ret.append(contour_diff.flatten(-2, -1))
    if non_agent_relation:
        pass
    else:
        ret.append(contour_info)

    ret = torch.cat(ret, dim=-1)
    ret[~valid_mask] = 0
    return ret, valid_mask, indices
