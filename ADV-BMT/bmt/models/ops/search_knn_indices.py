import torch
import torch.nn.functional as F

from bmt.models.ops.knn import knn_utils


def search_k_nearest_object_indices(
    ego_position_full,  # The "ego object" position
    ego_valid_mask,
    neighbor_position_full,  # The "other object" position
    neighbor_valid_mask,
    num_neighbors
):
    """
    ego_position_full: [B, T, max_ego_objects, 2]
    ego_valid_mask: [B, T, max_ego_objects]
    neighbor_position_full: [B, T, max_neighbor_objects, 2]
    neighbor_valid_mask: [B, T, max_neighbor_objects]
    num_neighbors: int
    """
    assert ego_position_full.ndim == 4
    B, T, max_ego_objects, pos_dim = ego_position_full.shape
    # assert pos_dim == 3
    assert ego_valid_mask.shape == (B, T, max_ego_objects)
    assert neighbor_position_full.shape[:2] == (B, T)
    _, _, max_neighbor_objects, pos_dim = neighbor_position_full.shape
    assert neighbor_valid_mask.shape == (B, T, max_neighbor_objects)

    ego_position_full = ego_position_full.clone()
    neighbor_position_full = neighbor_position_full.clone()

    effective_batch_size = B * T

    valid_ego_position = ego_position_full[ego_valid_mask]  # [num valid ego, 2]
    if valid_ego_position.shape[-1] == 2:
        valid_ego_position = F.pad(valid_ego_position, (0, 1))  # [num valid ego, 3]

    # Build a lookup table that translate the index of a valid ego object to the "batch index".
    # The batch index should in range [0, B*T]. See below for more discussion.
    batch_index = torch.arange(0, effective_batch_size, device=ego_position_full.device, dtype=torch.int)  # [B*T,]
    batch_index = batch_index.reshape(B, T, 1)  # [B, T, 1]
    batch_index = batch_index.repeat(1, 1, max_ego_objects)  # [B, T, max_ego_objects]
    valid_batch_index = batch_index[ego_valid_mask]  # [num valid ego,]

    neighbor_position_full[~neighbor_valid_mask] = 100000
    neighbor_position_flat = neighbor_position_full.flatten(start_dim=0, end_dim=2)  # [B*T*max_neighbors, 2]
    # neighbor_position_flat = neighbor_position_full[neighbor_valid_mask]  # [num valid neighbor, 2]
    if neighbor_position_flat.shape[-1] == 2:
        neighbor_position_flat = F.pad(neighbor_position_flat, (0, 1))  # [num valid ego, 3]

    # neighbor_valid_mask is in [B, T, N]
    # neighbor_batch_index = neighbor_valid_mask.sum(-1)  # [B, T]
    # neighbor_batch_index = neighbor_batch_index.reshape(-1)  # [B * T]
    # neighbor_batch_index = neighbor_batch_index.cumsum(-1).int()  # [B * T]
    # neighbor_batch_index = F.pad(neighbor_batch_index, (1, 0))  # [1 + B*T]

    # traffic_light_offsets is in shape []
    neighbor_offsets = \
        max_neighbor_objects * torch.arange(0, effective_batch_size + 1, device=ego_position_full.device,
                                            dtype=torch.int)

    assert len(neighbor_offsets) - 1 == valid_batch_index.max() + 1
    assert neighbor_position_flat.shape[0] == neighbor_offsets.max()
    assert neighbor_position_flat.shape[-1] == valid_ego_position.shape[-1] == 3

    # Output is in range: [0, max_neighbors]
    # (an alternative is to fall into [0, num valids neighbor], which is deprecated)
    # print(f"Searching near {num_neighbors}, {neighbor_offsets}, {valid_batch_index}")
    k_nearest_neighbor_index = knn_utils.knn_batch_mlogk(
        valid_ego_position,  # position of "ego" object
        neighbor_position_flat,  # position of "other objects"
        valid_batch_index,  # the batch index of each ego object, telling ego belongs to which batch
        neighbor_offsets,  # the index offset. For ego objects in (b, t), the offset will be (b*t-1)*max_neighbors
        # neighbor_batch_index,
        num_neighbors
    )
    # print("Finish searching.")

    # It is possible that at some (batch: b, time: t), there are no valid neighbor objects at all!
    # We will do postprocessing to tell that for those ego objects in (b, t), they have no neighbors since
    # all neighbors at that (b, t) are invalid.
    this_batch_has_no_neighbor = (~neighbor_valid_mask).all(dim=-1).flatten(0, 1)  # after flatten the shape is [B*T]

    i_have_no_neighbor = this_batch_has_no_neighbor[valid_batch_index]  # [num_valid,]

    k_nearest_neighbor_index[i_have_no_neighbor] = -1

    # return k_nearest_neighbor_index

    ret = torch.empty((B, T, max_ego_objects, num_neighbors)).to(batch_index)
    ret.fill_(-1)
    ret[ego_valid_mask] = k_nearest_neighbor_index

    return ret


def search_k_nearest_map_feature_indicies(
    ego_position_full,  # The "ego object" position
    ego_valid_mask,
    neighbor_position_full,  # The "other object" position
    neighbor_valid_mask,
    num_neighbors
):
    assert ego_position_full.ndim == 4
    B, T, max_ego_objects, pos_dim = ego_position_full.shape
    assert ego_valid_mask.shape == (B, T, max_ego_objects)
    # assert neighbor_position_full.shape[:2] == (B, T)
    _, max_map_feats, pos_dim = neighbor_position_full.shape
    assert neighbor_valid_mask.shape == (B, max_map_feats)

    # effective_batch_size = B * T

    valid_ego_position = ego_position_full[ego_valid_mask]  # [num valid ego, 2]
    if valid_ego_position.shape[-1] == 2:
        valid_ego_position = F.pad(valid_ego_position, (0, 1))  # [num valid ego, 3]

    # Build a lookup table that translate the index of a valid ego object to the "batch index".
    # The batch index should in range [0, B].
    batch_index = torch.arange(0, B, device=ego_position_full.device, dtype=torch.int)  # [B,]
    batch_index = batch_index.reshape(B, 1, 1)  # [B, 1, 1]
    batch_index = batch_index.repeat(1, T, max_ego_objects)  # [B, T, max_ego_objects]
    valid_batch_index = batch_index[ego_valid_mask]  # [num valid ego,]

    # neighbor_position_full[~neighbor_valid_mask] = float("+inf")
    # neighbor_position_flat = neighbor_position_full.flatten(start_dim=0, end_dim=1)  # [B*T*max_neighbors, 2]

    # traffic_light_offsets is in shape []
    # neighbor_offsets = \
    #     max_map_feats * torch.arange(0, B + 1, device=ego_position_full.device, dtype=torch.int)

    neighbor_position_flat = neighbor_position_full[neighbor_valid_mask]  # [num valid neighbor, 2]
    if neighbor_position_flat.shape[-1] == 2:
        neighbor_position_flat = F.pad(neighbor_position_flat, (0, 1))  # [num valid ego, 3]

    # neighbor_valid_mask is in [B, M]
    neighbor_batch_index = neighbor_valid_mask.sum(-1)  # [B,]
    neighbor_batch_index = neighbor_batch_index.cumsum(-1).int()  # [B,]
    neighbor_batch_index = F.pad(neighbor_batch_index, (1, 0))  # [1+B]

    assert len(neighbor_batch_index) - 1 == valid_batch_index.max() + 1
    assert neighbor_position_flat.shape[0] == neighbor_batch_index.max()
    assert neighbor_position_flat.shape[-1] == valid_ego_position.shape[-1] == 3

    # Output will be in shape [num valid ego objects, K]
    k_nearest_neighbor_index = knn_utils.knn_batch_mlogk(
        valid_ego_position,  # position of "ego" object
        neighbor_position_flat,  # position of "other objects"
        valid_batch_index,  # the batch index of each ego object, telling ego belongs to which batch
        # neighbor_offsets,  # the index offset. For ego objects in (b, t), the offset will be (b*t-1)*max_neighbors
        neighbor_batch_index,
        num_neighbors
    )

    # return k_nearest_neighbor_index

    ret = torch.empty((B, T, max_ego_objects, num_neighbors)).to(batch_index)
    ret.fill_(-1)
    ret[ego_valid_mask] = k_nearest_neighbor_index

    return ret


def search_k_nearest_map_feature_indicies_for_map(
    ego_position_full,  # The "ego object" position
    ego_valid_mask,
    num_neighbors
):
    B, max_map_feats, pos_dim = ego_position_full.shape
    assert ego_valid_mask.shape == (B, max_map_feats)

    valid_ego_position = ego_position_full[ego_valid_mask]  # [num valid ego, 2]
    if valid_ego_position.shape[-1] == 2:
        valid_ego_position = F.pad(valid_ego_position, (0, 1))  # [num valid ego, 3]

    # Build a lookup table that translate the index of a valid map feat to the "batch index".
    # The batch index should in range [0, B].
    batch_index = torch.arange(0, B, device=ego_position_full.device, dtype=torch.int)  # [B,]
    batch_index = batch_index.reshape(B, 1)  # [B, 1, 1]
    batch_index = batch_index.repeat(1, max_map_feats)  # [B, max_ego_objects]
    valid_batch_index = batch_index[ego_valid_mask]  # [num valid map feat,]

    # ego_position_full[~ego_valid_mask] = float("+inf")
    # neighbor_position_flat = ego_position_full.flatten(start_dim=0, end_dim=1)  # [B*max_neighbors, 2]
    neighbor_position_flat = ego_position_full[ego_valid_mask]
    if neighbor_position_flat.shape[-1] == 2:
        neighbor_position_flat = F.pad(neighbor_position_flat, (0, 1))  # [num valid ego, 3]

    # traffic_light_offsets is in shape []
    # neighbor_offsets = max_map_feats * torch.arange(0, B + 1, device=ego_position_full.device, dtype=torch.int)

    # neighbor_valid_mask is in [B, M]
    neighbor_batch_index = ego_valid_mask.sum(-1)  # [B,]
    neighbor_batch_index = neighbor_batch_index.cumsum(-1).int()  # [B,]
    neighbor_batch_index = F.pad(neighbor_batch_index, (1, 0))  # [1+B]

    assert len(neighbor_batch_index) - 1 == valid_batch_index.max() + 1
    assert neighbor_position_flat.shape[0] == neighbor_batch_index.max()
    assert neighbor_position_flat.shape[-1] == valid_ego_position.shape[-1] == 3

    # Output will be in shape [num valid ego objects, K]
    k_nearest_neighbor_index = knn_utils.knn_batch_mlogk(
        valid_ego_position,  # position of "ego" object
        neighbor_position_flat,  # position of "other objects"
        valid_batch_index,  # the batch index of each ego object, telling ego belongs to which batch
        # neighbor_offsets,  # the index offset. For ego objects in (b, t), the offset will be (b*t-1)*max_neighbors
        neighbor_batch_index,  # the index offset. For ego objects in (b, t), the offset will be (b*t-1)*max_neighbors
        num_neighbors
    )

    # return k_nearest_neighbor_index

    ret = torch.empty((B, max_map_feats, num_neighbors)).to(batch_index)
    ret.fill_(-1)
    ret[ego_valid_mask] = k_nearest_neighbor_index
    return ret
