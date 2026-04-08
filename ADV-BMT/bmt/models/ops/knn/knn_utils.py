# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi
# All Rights Reserved

import torch
from torch.autograd import Function

from . import knn_cuda


class KNNBatch(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda, (xyz.is_contiguous(), xyz.is_cuda)
        assert query_xyz.is_contiguous() and query_xyz.is_cuda, (query_xyz.is_contiguous(), query_xyz.is_cuda)
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda, (batch_idxs.is_contiguous(), batch_idxs.is_cuda)
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda, \
            (query_batch_offsets.is_contiguous(), query_batch_offsets.is_cuda)

        # idx = torch.cuda.IntTensor(n, k).zero_()
        idx = torch.zeros([n, k], device=xyz.device, dtype=torch.int)

        knn_cuda.knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


knn_batch = KNNBatch.apply


class KNNBatchMlogK(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''
        assert xyz.shape[-1] == 3
        assert query_xyz.shape[-1] == 3

        n = xyz.size(0)
        m = query_xyz.size(0)
        # assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda, (xyz.is_contiguous(), xyz.is_cuda)
        assert query_xyz.is_contiguous() and query_xyz.is_cuda, (query_xyz.is_contiguous(), query_xyz.is_cuda)
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda, (batch_idxs.is_contiguous(), batch_idxs.is_cuda)
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda, \
            (query_batch_offsets.is_contiguous(), query_batch_offsets.is_cuda)
        assert k <= 128

        assert query_batch_offsets.max() == query_batch_offsets[-1] == query_xyz.shape[0]
        assert query_batch_offsets[0] == 0
        assert query_batch_offsets.shape[0] == batch_idxs.max() + 1 + 1
        assert batch_idxs.shape[0] == n

        # idx = torch.cuda.IntTensor(n, k).zero_()
        idx = torch.zeros([n, k], device=xyz.device, dtype=torch.int)

        # half_precision = int(query_xyz.dtype == torch.bfloat16)

        if query_xyz.dtype == torch.bfloat16:
            knn_cuda.knn_batch_mlogk(
                xyz.type(torch.float32), query_xyz.type(torch.float32), batch_idxs, query_batch_offsets, idx, n, m, k
            )
        elif query_xyz.dtype == torch.float16:
            knn_cuda.knn_batch_mlogk(
                xyz.type(torch.float32), query_xyz.type(torch.float32), batch_idxs, query_batch_offsets, idx, n, m, k
            )
        else:
            knn_cuda.knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


knn_batch_mlogk = KNNBatchMlogK.apply
